#!/usr/bin/env python3
"""Top-team Kaggriculture replay fetcher: resumable, budgeted, verified.

Why: the official daily datasets (kaggle/kaggriculture-episodes-YYYY-MM-DD) hold only the ~600 games/day with the
highest average rating (20 GiB cap). Top teams also play many games against weaker opponents that never appear
there. This tool fetches those games one replay at a time under a per-UTC-day call budget.

Candidates (no per-episode calls):
  --xsf-index  episodes_index.parquet from the public dataset xishengfeng/kaggriculture-replay-db
               (episode id, create_time, type, team/submission ids, rewards); refresh it with
               kaggle datasets download xishengfeng/kaggriculture-replay-db -f replay_db/index/episodes_index.parquet
  --list-top N official API listing: team-submissions for each top-N team, then episodes for every active
               submission (listing calls, budgeted separately by --max-list-calls-per-day).
Selection: public, completed, create day in [--since, --until], winner seat is a current top-N team, our team in
neither seat, not in the local official archives or this tool's manifest; ordered by loser strength tier
(rank <= 100, <= top 20%, rest), then newest day, round-robin over winner teams with at most --cap-sub-day games
per (winner submission, day) in each round. --require-official-day (default) skips days whose official daily zip
is not local yet, so no call is spent on a game the official dump is about to include.
Fetch: official authenticated API (kaggle SDK competition_episode_replay = CompetitionApiService/GetEpisodeReplay),
at most --max-replays-per-day per UTC day and --min-interval seconds apart. HTTP 429 stops the run and blocks
further calls for --cooldown-hours. Community guidance for this endpoint: <= 3600 calls/day and <= 1/s
(robga "Simulations Episode Scraper" notebook); the default budget stays far below it.
Verify: JSON parses; info.EpisodeId == id; len(steps) == configuration.episodeSteps == 720; every final status
DONE; numeric rewards; configuration without seed == REFERENCE-CONFIG.json.
Output (--out): archives/kaggriculture-topteams-<day>-<stamp>.zip with members <episode_id>.json (same member
format as the official daily zips), MANIFEST.jsonl (one line per archived episode: ids, source, sha256, ...),
state.json (per-day call counts, failures, cooldown). Verified replays wait in staging/ and are packed into a new
zip (written to .tmp, then renamed) every --pack-every episodes and at exit; a restart packs leftovers first.
--ingest-dir verifies and packs replay JSONs already on disk (zero calls). --dry-run prints the plan only.
"""
import argparse
import collections
import csv
import datetime as dt
import glob
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile

REF_DEFAULT = '/mnt/ai/scratch/farmNEAT/primitive-bc-data/REFERENCE-CONFIG.json'
LOCAL_DEFAULT = ['/mnt/ai/scratch/farmNEAT/primitive-bc-data/kaggriculture-episodes-*.zip',
                 '/mnt/steam/farm-data/kaggle/kaggriculture-episodes-*/*.zip']
COMP = 'kaggriculture'


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def today():
    return utc_now().strftime('%Y-%m-%d')


def log(msg):
    print(f'{utc_now():%Y-%m-%dT%H:%M:%SZ} {msg}', flush=True)


# ------------------------------------------------------------------ state / manifest
class State:
    def __init__(self, out):
        self.path = os.path.join(out, 'state.json')
        self.d = {'calls': {}, 'failed': {}, 'attempts': {}, 'blocked_until': None}
        if os.path.exists(self.path):
            self.d.update(json.load(open(self.path)))

    def save(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.d, f, indent=0, sort_keys=True)
        os.replace(tmp, self.path)

    def used(self, kind):
        return self.d['calls'].get(today(), {}).get(kind, 0)

    def count(self, kind):
        day = self.d['calls'].setdefault(today(), {})
        day[kind] = day.get(kind, 0) + 1
        self.save()


def manifest_ids(out):
    p = os.path.join(out, 'MANIFEST.jsonl')
    if not os.path.exists(p):
        return set()
    return {json.loads(l)['episode_id'] for l in open(p) if l.strip()}


def local_ids(patterns):
    ids, days = set(), set()
    for pat in patterns:
        for p in glob.glob(pat):
            m = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(p))
            try:
                names = zipfile.ZipFile(p).namelist()
            except zipfile.BadZipFile:
                continue
            if m:
                days.add(m.group(1))
            ids.update(int(n[:-5]) for n in names if n.endswith('.json') and n[:-5].isdigit())
    return ids, days


# ------------------------------------------------------------------ verification
def verify(raw, episode_id, ref):
    """Return (meta, None) or (None, reason)."""
    try:
        d = json.loads(raw)
    except Exception as e:
        return None, f'parse: {str(e)[:80]}'
    info = d.get('info') or {}
    if int(info.get('EpisodeId') or -1) != int(episode_id):
        return None, f'episode id mismatch: {info.get("EpisodeId")}'
    cfg = dict(d.get('configuration') or {})
    cfg.pop('seed', None)
    diff = sorted(k for k in set(cfg) | set(ref) if cfg.get(k) != ref.get(k))
    if diff:
        return None, 'config differs: ' + ','.join(diff)
    steps = d.get('steps') or []
    if len(steps) != cfg.get('episodeSteps') or len(steps) != 720:
        return None, f'steps {len(steps)}'
    final = [s.get('status') for s in steps[-1]]
    if len(final) != 2 or any(s != 'DONE' for s in final):
        return None, f'final statuses {final}'
    rewards = [s.get('reward') for s in steps[-1]]
    if any(not isinstance(r, (int, float)) for r in rewards):
        return None, f'rewards {rewards}'
    return {'team_names': info.get('TeamNames'), 'rewards': rewards, 'seed': info.get('seed'),
            'module_version': d.get('module_version')}, None


# ------------------------------------------------------------------ candidates
def load_leaderboard(path):
    if not path:
        tmpd = tempfile.mkdtemp()
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        api.competition_leaderboard_download(COMP, tmpd)
        zf = glob.glob(os.path.join(tmpd, '*.zip'))[0]
        zipfile.ZipFile(zf).extractall(tmpd)
        path = glob.glob(os.path.join(tmpd, '*.csv'))[0]
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    return {int(r['TeamId']): (int(r['Rank']), r['TeamName']) for r in rows}, len(rows), path


def cands_from_xsf(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path).to_pylist()
    for r in t:
        yield {'episode_id': int(r['episode_id']), 'day': str(r['create_time'])[:10], 'type': r['type'], 'state': 'COMPLETED',
               'team_ids': [r['team_id_0'], r['team_id_1']], 'sub_ids': [r['submission_id_0'], r['submission_id_1']],
               'rewards': [r['reward_0'], r['reward_1']], 'scores': [r.get('score_0'), r.get('score_1')], 'cand_source': 'xsf-index'}


def cands_from_api(lb, top_n, st, max_list):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    out = []
    for tid, (rank, name) in sorted(lb.items(), key=lambda kv: kv[1][0]):
        if rank > top_n:
            continue
        if st.used('list') >= max_list:
            log('listing budget reached'); break
        subs = api.competition_team_submissions(tid)
        st.count('list')
        for s in subs:
            if st.used('list') >= max_list:
                break
            sid = getattr(s, 'id', None) or getattr(s, 'ref', None)
            eps = api.competition_list_episodes(int(sid)); st.count('list'); time.sleep(1.0)
            for e in eps:
                ag = sorted(e.agents, key=lambda a: a.index)
                if len(ag) != 2:
                    continue
                out.append({'episode_id': int(e.id), 'day': str(e.create_time)[:10], 'type': str(e.type).split('.')[-1],
                            'state': str(e.state).split('.')[-1], 'team_ids': [a.team_id for a in ag],
                            'sub_ids': [a.submission_id for a in ag], 'rewards': [a.reward for a in ag],
                            'scores': [getattr(a, 'initial_score', None) for a in ag],
                            'scores_after': [getattr(a, 'updated_score', None) for a in ag], 'cand_source': 'api-list'})
        time.sleep(1.0)
    return out


def plan(cands, lb, n_teams, a, exclude, official_days):
    top20p = int(round(0.2 * n_teams))
    rank = {tid: r for tid, (r, _) in lb.items()}
    ours = {tid for tid, (_, nm) in lb.items() if nm in a.exclude_team}
    seen, pool = set(), []
    for c in cands:
        e = c['episode_id']
        if e in seen or e in exclude:
            continue
        seen.add(e)
        if 'PUBLIC' not in str(c['type']) or 'COMPLETED' not in str(c['state']):
            continue
        if not (a.since <= c['day'] <= a.until):
            continue
        if a.require_official_day and c['day'] not in official_days:
            continue
        if any(t in ours for t in c['team_ids']):
            continue
        r0, r1 = (x if isinstance(x, (int, float)) else float('-inf') for x in c['rewards'])
        if r0 == r1:
            continue
        w = 0 if r0 > r1 else 1
        wt, lt = c['team_ids'][w], c['team_ids'][1 - w]
        wr, lr = rank.get(wt, 10 ** 9), rank.get(lt, 10 ** 9)
        if wr > a.top_n:
            continue
        tier = 0 if lr <= 100 else (1 if lr <= top20p else 2)
        pool.append(dict(c, winner_seat=w, winner_rank=wr, loser_rank=lr, tier=tier, winner_sub=c['sub_ids'][w], winner_team=wt))
    # Order: tier, newest day, then rounds: within a (winner submission, day) group the k-th newest game gets
    # round k // cap_sub_day; inside a round teams alternate (best rank first), one game per team per turn.
    k_sub = collections.Counter()
    for p in sorted(pool, key=lambda p: -p['episode_id']):
        g = (p['winner_sub'], p['day'])
        p['round'] = k_sub[g] // a.cap_sub_day
        k_sub[g] += 1
    k_team = collections.Counter()
    for p in sorted(pool, key=lambda p: (p['tier'], p['day'], p['round'], -p['episode_id'])):
        g = (p['tier'], p['day'], p['round'], p['winner_team'])
        p['turn'] = k_team[g]
        k_team[g] += 1
    return sorted(pool, key=lambda p: (p['tier'], [-int(x) for x in p['day'].split('-')], p['round'], p['turn'],
                                       p['winner_rank'], -p['episode_id']))


# ------------------------------------------------------------------ fetch / pack
class Fetcher:
    def __init__(self):
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.api = KaggleApi(); self.api.authenticate()
        self.tmp = tempfile.mkdtemp(prefix='replay-')

    def get(self, episode_id):
        self.api.competition_episode_replay(int(episode_id), path=self.tmp, quiet=True)
        p = os.path.join(self.tmp, f'episode-{episode_id}-replay.json')
        raw = open(p, 'rb').read(); os.remove(p)
        return raw


def stage(out, episode_id, raw, meta):
    sd = os.path.join(out, 'staging'); os.makedirs(sd, exist_ok=True)
    tmp = os.path.join(sd, f'{episode_id}.json.tmp')
    with open(tmp, 'wb') as f:
        f.write(raw)
    with open(os.path.join(sd, f'{episode_id}.meta.tmp'), 'w') as f:
        json.dump(meta, f)
    os.replace(os.path.join(sd, f'{episode_id}.meta.tmp'), os.path.join(sd, f'{episode_id}.meta'))
    os.replace(tmp, os.path.join(sd, f'{episode_id}.json'))


def pack(out):
    sd = os.path.join(out, 'staging')
    if not os.path.isdir(sd):
        return 0
    for f in os.listdir(sd):  # partial writes of an interrupted run (single-process tool)
        if f.endswith('.tmp'):
            os.remove(os.path.join(sd, f))
    ids = sorted(int(f[:-5]) for f in os.listdir(sd) if f.endswith('.json') and os.path.exists(os.path.join(sd, f[:-5] + '.meta')))
    done = manifest_ids(out)  # crash after the manifest append but before staging cleanup: already archived
    for e in [e for e in ids if e in done]:
        os.remove(os.path.join(sd, f'{e}.json')); os.remove(os.path.join(sd, f'{e}.meta'))
    ids = [e for e in ids if e not in done]
    if not ids:
        return 0
    by_day = collections.defaultdict(list)
    metas = {}
    for e in ids:
        metas[e] = json.load(open(os.path.join(sd, f'{e}.meta')))
        by_day[metas[e]['day']].append(e)
    ad = os.path.join(out, 'archives'); os.makedirs(ad, exist_ok=True)
    stamp = utc_now().strftime('%Y%m%dT%H%M%S')
    lines = []
    for day, es in sorted(by_day.items()):
        name = f'kaggriculture-topteams-{day}-{stamp}.zip'
        tmp = os.path.join(ad, name + '.tmp')
        with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for e in es:
                z.write(os.path.join(sd, f'{e}.json'), arcname=f'{e}.json')
        with zipfile.ZipFile(tmp) as z:  # re-read every member and check its hash before committing
            for e in es:
                if hashlib.sha256(z.read(f'{e}.json')).hexdigest() != metas[e]['sha256']:
                    raise RuntimeError(f'zip readback hash mismatch {e}')
        os.replace(tmp, os.path.join(ad, name))
        for e in es:
            lines.append(json.dumps(dict(metas[e], archive=f'archives/{name}', member=f'{e}.json'), sort_keys=True))
    with open(os.path.join(out, 'MANIFEST.jsonl'), 'a') as f:
        f.write('\n'.join(lines) + '\n'); f.flush(); os.fsync(f.fileno())
    for e in ids:
        os.remove(os.path.join(sd, f'{e}.json')); os.remove(os.path.join(sd, f'{e}.meta'))
    log(f'packed {len(ids)} episodes into {len(by_day)} archive(s)')
    return len(ids)


def make_meta(c, raw, vm, source, lb):
    rank = {tid: r for tid, (r, _) in lb.items()} if lb else {}
    rw = vm['rewards']; w = 0 if rw[0] > rw[1] else (1 if rw[1] > rw[0] else None)
    return {'episode_id': int(c['episode_id']), 'day': c.get('day'), 'source': source, 'cand_source': c.get('cand_source'),
            'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'team_names': vm['team_names'],
            'team_ids': c.get('team_ids'), 'submission_ids': c.get('sub_ids'), 'rewards': rw, 'winner_seat': w,
            'team_ranks_at_fetch': [rank.get(t) for t in (c.get('team_ids') or [])], 'seed': vm['seed'],
            'scores': c.get('scores'), 'scores_after': c.get('scores_after'),
            'module_version': vm['module_version'], 'fetched_utc': utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference-config', default=REF_DEFAULT)
    ap.add_argument('--local-archives', nargs='*', default=LOCAL_DEFAULT)
    ap.add_argument('--leaderboard', help='leaderboard CSV (default: download current one, 1 leaderboard call)')
    ap.add_argument('--xsf-index', nargs='*', default=[])
    ap.add_argument('--list-top', type=int, default=0, help='also list episodes of the top-N teams via the official API')
    ap.add_argument('--top-n', type=int, default=20)
    ap.add_argument('--exclude-team', nargs='*', default=['Russell Kirk', '有辣条有权'])
    ap.add_argument('--since', default='2026-09-03')
    ap.add_argument('--until', default='9999-12-31')
    ap.add_argument('--no-require-official-day', dest='require_official_day', action='store_false')
    ap.add_argument('--cap-sub-day', type=int, default=20)
    ap.add_argument('--max-replays-per-day', type=int, default=500)
    ap.add_argument('--max-list-calls-per-day', type=int, default=60)
    ap.add_argument('--max-new', type=int, default=0, help='stop after this many fetches in this run (0 = budget only)')
    ap.add_argument('--min-interval', type=float, default=2.0)
    ap.add_argument('--cooldown-hours', type=float, default=6.0)
    ap.add_argument('--pack-every', type=int, default=50)
    ap.add_argument('--ingest-dir', help='verify + pack replay JSONs already on disk (no calls)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ref = json.load(open(a.reference_config)); ref.pop('seed', None)
    st = State(a.out)
    pack(a.out)  # resume: commit anything verified by an interrupted run
    lb, n_teams, lb_path = load_leaderboard(a.leaderboard)
    have = manifest_ids(a.out)

    if a.ingest_dir:
        known = {c['episode_id']: c for p in a.xsf_index for c in cands_from_xsf(p)}
        ids = {nm: tid for tid, (_, nm) in lb.items()}
        n = 0
        for p in sorted(glob.glob(os.path.join(a.ingest_dir, '*.json'))):
            raw = open(p, 'rb').read()
            try:
                eid = int(json.loads(raw)['info']['EpisodeId'])
            except Exception as ex:
                log(f'REJECT {p}: no episode id ({str(ex)[:60]})'); continue
            if eid in have:
                log(f'skip {eid}: already archived'); continue
            vm, err = verify(raw, eid, ref)
            if err:
                log(f'REJECT {p}: {err}'); continue
            c = known.get(eid) or {'episode_id': eid, 'day': 'unknown-day', 'sub_ids': None,
                                  'team_ids': [ids.get(x) for x in (vm['team_names'] or [])], 'cand_source': 'ingest'}
            stage(a.out, eid, raw, make_meta(c, raw, vm, f'ingest:{os.path.basename(p)}', lb)); n += 1
        pack(a.out)
        log(f'ingested {n}')
        return

    loc, official_days = local_ids(a.local_archives)
    exclude = loc | have | {int(k) for k, v in st.d['failed'].items() if v.get('permanent')}
    cands = []
    for p in a.xsf_index:
        cands.extend(cands_from_xsf(p))
    if a.list_top:
        cands.extend(cands_from_api(lb, a.list_top, st, a.max_list_calls_per_day))
    order = plan(cands, lb, n_teams, a, exclude, official_days)
    by = collections.Counter((p['tier'], p['day']) for p in order)
    log(f'candidates {len(cands)}; local ids {len(loc)}; archived {len(have)}; plan {len(order)}; '
        f'by tier/day (top 12): {sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1]), reverse=False)[:12]}')
    if a.dry_run:
        teams = collections.Counter((p['winner_rank'], lb.get(p['winner_team'], (0, '?'))[1]) for p in order)
        for (r, nm), n in sorted(teams.items()):
            log(f'  rank {r:>3} {nm}: {n}')
        return
    bu = st.d.get('blocked_until')
    if bu and utc_now().timestamp() < bu:
        log(f'cooling down until {dt.datetime.fromtimestamp(bu, dt.timezone.utc)}; no calls'); return
    fetcher = Fetcher()
    got = 0; last = 0.0; staged = 0
    for p in order:
        if st.used('replay') >= a.max_replays_per_day:
            log(f'daily replay budget reached ({a.max_replays_per_day})'); break
        if a.max_new and got >= a.max_new:
            break
        e = p['episode_id']
        wait = a.min_interval - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        last = time.time(); st.count('replay')
        try:
            raw = fetcher.get(e)
        except Exception as ex:
            msg = str(ex)[:160]
            att = st.d['attempts'].get(str(e), 0) + 1; st.d['attempts'][str(e)] = att
            if '429' in msg or 'Too Many' in msg:
                st.d['blocked_until'] = utc_now().timestamp() + a.cooldown_hours * 3600; st.save()
                log(f'429 on {e}: stopping, cooldown {a.cooldown_hours} h'); break
            perm = '404' in msg or '403' in msg or att >= 3
            st.d['failed'][str(e)] = {'reason': msg, 'permanent': perm}; st.save()
            log(f'fetch failed {e}: {msg}'); continue
        vm, err = verify(raw, e, ref)
        if err:
            st.d['failed'][str(e)] = {'reason': err, 'permanent': True}; st.save()
            log(f'REJECT {e}: {err}'); continue
        stage(a.out, e, raw, make_meta(p, raw, vm, 'kaggle-api:GetEpisodeReplay', lb))
        got += 1; staged += 1
        log(f'ok {e} day {p["day"]} tier {p["tier"]} winner rank {p["winner_rank"]} loser rank {p["loser_rank"]} ({len(raw)} B)')
        if staged >= a.pack_every:
            pack(a.out); staged = 0
    pack(a.out)
    log(f'fetched {got}; replay calls today {st.used("replay")}/{a.max_replays_per_day}; list calls today {st.used("list")}')


if __name__ == '__main__':
    main()
