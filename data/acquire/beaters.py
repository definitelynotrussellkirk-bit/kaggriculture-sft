"""Track every opponent that beats us (owner 2026-09-26 ~00:00 UTC: "we want to track every one that beats us"), and what beats
THEIR policy ("and see what beats their policy", "and save a special folder for that!").

Per run: list the episodes of our submissions (SDK listing: agents with team, submission id and reward; 1 call per submission),
record every public loss in BEATERS.jsonl (append-only, one row per episode) and rewrite BEATERS.md (per-team table), then put
every loss replay into beat archives for training (winner = the opponent): copies from our hosted folders when on disk, else one
replay download each. Calls are added to the shared per-day counter; the run stops at --cap.

  work/venv/python -B beaters.py --subs 56558205 56561674 [--cap 2900] [--max-calls N --pause S]
"""
import argparse, collections, csv, datetime as dt, glob, gzip, json, os, sys, time
from pathlib import Path
sys.path.insert(0, '/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra')
import fetch_topteam_replays as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = '/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/beat-us'
OUT2 = '/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/beats-our-beaters'   # games our beaters' policies lost
RELIST_HOURS = 6
US = 16624026


def counter():
    return HERE / f'KAGGLE-REQUESTS-{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}.txt'


class RunBudget(Exception):
    """This run's call allowance is spent (owner 2026-09-27: keep pulling past 3k "at the slow rate", "do it SLOWLY")."""


RUN = {'calls': 0, 'max': None, 'pause': 0.0}


def take(cap):
    if RUN['max'] is not None and RUN['calls'] >= RUN['max']:
        raise RunBudget()
    p = counter()
    v = int(p.read_text().strip() or 0) if p.exists() else 0
    if v >= cap:
        raise SystemExit(f'budget reached {v}/{cap}')
    p.write_text(f'{v + 1}\n')
    RUN['calls'] += 1
    if RUN['pause']:
        time.sleep(RUN['pause'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subs', nargs='+', type=int, required=True)
    ap.add_argument('--cap', type=int, default=2900)
    ap.add_argument('--max-calls', type=int, default=None, help='Kaggle calls allowed in this run (listings + downloads)')
    ap.add_argument('--pause', type=float, default=0.0, help='seconds to wait after every Kaggle call')
    a = ap.parse_args()
    RUN.update(max=a.max_calls, pause=a.pause)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    names = {}
    for p in glob.glob(str(ROOT / 'work/runs/deployment/*/SUBMISSION.json')):
        try:
            names[int(json.load(open(p))['submission_id'])] = Path(p).parent.name
        except Exception:  # noqa: BLE001
            pass
    lbs = sorted(glob.glob('/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/top30-losses/LEADERBOARD-*.csv'))
    rank = {int(r['TeamId']): (int(r['Rank']), float(r['Score'])) for r in csv.DictReader(open(lbs[-1], encoding='utf-8-sig'))} if lbs else {}
    log = HERE / 'BEATERS.jsonl'
    seen = {json.loads(l)['episode_id'] for l in open(log)} if log.exists() else set()
    new = []
    for sid in a.subs:
        take(a.cap)
        for e in api.competition_list_episodes(sid):
            ag = sorted(e.agents, key=lambda x: x.index)
            if len(ag) != 2 or 'PUBLIC' not in str(e.type) or 'COMPLETED' not in str(e.state):
                continue
            rw = [x.reward for x in ag]
            if None in rw or ag[0].team_id == ag[1].team_id:
                continue
            s = [x.submission_id for x in ag].index(sid)
            if rw[s] >= rw[1 - s] or int(e.id) in seen:
                continue
            o = ag[1 - s]
            row = {'episode_id': int(e.id), 'create_time': str(e.create_time), 'our_sub': sid, 'ours': names.get(sid),
                   'our_seat': s, 'opponent': o.team_name, 'opp_team_id': o.team_id, 'opp_sub': o.submission_id,
                   'our_cash': rw[s], 'opp_cash': rw[1 - s], 'margin': rw[s] - rw[1 - s],
                   'opp_rank_at_record': rank.get(o.team_id, (None,))[0], 'leaderboard': Path(lbs[-1]).name if lbs else None,
                   'team_ids': [x.team_id for x in ag], 'sub_ids': [x.submission_id for x in ag]}
            new.append(row); seen.add(row['episode_id'])
        time.sleep(1.0)
    with open(log, 'a') as f:
        f.write(''.join(json.dumps(r) + '\n' for r in new))
    rows = [json.loads(l) for l in open(log)]
    # replays into beat archives (official member format)
    packed = {json.loads(l)['episode_id'] for l in open(f'{OUT}/MANIFEST.jsonl')} if os.path.exists(f'{OUT}/MANIFEST.jsonl') else set()
    ref = json.load(open(F.REF_DEFAULT)); fx = None; got = 0
    have_local, _ = F.local_ids(F.LOCAL_DEFAULT)   # coordinator 14:4x: 99 of today's beat-us downloads were already local (hosted-ours); skip them
    try:
        for r in rows:
            e = r['episode_id']
            if e in packed:
                continue
            disk = glob.glob(str(ROOT / f'work/runs/deployment/*/hosted/episode-{e}-replay.json.gz'))
            if disk:
                raw = gzip.decompress(open(disk[0], 'rb').read()); src = 'our-hosted-copy'
            elif e in have_local:
                print('skip-local', e); continue
            else:
                fx = fx or F.Fetcher(); take(a.cap)
                raw = fx.get(e); src = 'official-api'; time.sleep(1.2)
            vm, why = F.verify(raw, e, ref)
            if vm is None:
                print('reject', e, why); continue
            c = {'episode_id': e, 'day': r['create_time'][:10], 'cand_source': 'beat-us', 'team_ids': r['team_ids'], 'sub_ids': r['sub_ids']}
            meta = F.make_meta(c, raw, vm, src, None)
            meta.update({'reason': 'beat-us', 'our_sub': r['our_sub'], 'opponent': r['opponent'], 'opp_sub': r['opp_sub']})
            F.stage(OUT, e, raw, meta); got += 1
    except RunBudget:
        pass
    finally:
        n = F.pack(OUT)
    second = beaters_of_beaters(api, rows, a.cap, ref)
    by = collections.defaultdict(list)
    for r in rows:
        by[r['opponent']].append(r)
    lines = ['# Teams that beat us (hosted public games of our tracked submissions)', '',
             f'Updated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC; {len(rows)} losses; ranks from {Path(lbs[-1]).name if lbs else "-"}.', '',
             '| team | rank | losses to them | our submissions | their submissions | mean margin | last |', '|---|---|---|---|---|---|---|']
    for t, g in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rk = rank.get(g[0]['opp_team_id'], (None,))[0]
        lines.append(f"| {t} | {rk} | {len(g)} | {', '.join(sorted({str(x['ours'] or x['our_sub']) for x in g}))} | "
                     f"{', '.join(sorted({str(x['opp_sub']) for x in g}))} | {sum(x['margin'] for x in g) / len(g):+.0f} | {max(x['create_time'] for x in g)[:16]} |")
    (HERE / 'BEATERS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'new_losses': len(new), 'total_losses': len(rows), 'replays_packed': n, 'downloaded': got, 'second_level': second}))


def local_map():
    """episode_id -> (zip, member) over every local archive (official dailies, reconstructed, extra/*/archives)."""
    E = '/mnt/ai/scratch/farmNEAT/primitive-bc-data'
    out = {}
    for z in sorted(glob.glob(f'{E}/kaggriculture-episodes-*.zip')) + sorted(glob.glob(f'{E}/extra/reconstructed/*.zip')) + \
            sorted(glob.glob(f'{E}/extra/*/archives/*.zip')):
        try:
            for n in __import__('zipfile').ZipFile(z).namelist():
                if n.endswith('.json') and n[:-5].isdigit():
                    out.setdefault(int(n[:-5]), (z, n))
        except Exception:  # noqa: BLE001
            pass
    return out


def beaters_of_beaters(api, rows, cap, ref):
    """For every submission that beat us, list its episodes (at most every RELIST_HOURS) and save the games it LOST into OUT2
    (training target = whoever beat our beater). Replays come from local archives when present, else one download each."""
    import zipfile
    os.makedirs(f'{OUT2}/listings', exist_ok=True)
    now = time.time()
    lost = {}
    for sub in sorted({r['opp_sub'] for r in rows}):
        lp = Path(f'{OUT2}/listings/{sub}.json')
        if not lp.exists() or now - lp.stat().st_mtime > RELIST_HOURS * 3600:
            try:
                take(cap)
            except RunBudget:
                if not lp.exists():
                    continue
            else:
                eps = []
                for e in api.competition_list_episodes(sub):
                    ag = sorted(e.agents, key=lambda x: x.index)
                    if len(ag) == 2:
                        eps.append({'episode_id': int(e.id), 'create_time': str(e.create_time), 'type': str(e.type).split('.')[-1],
                                    'state': str(e.state).split('.')[-1], 'team_names': [x.team_name for x in ag],
                                    'team_ids': [x.team_id for x in ag], 'sub_ids': [x.submission_id for x in ag],
                                    'rewards': [x.reward for x in ag]})
                lp.write_text(json.dumps(eps)); time.sleep(1.0)
        for e in json.loads(lp.read_text()):
            if e['type'] != 'EPISODE_TYPE_PUBLIC' or e['state'] != 'COMPLETED' or None in e['rewards'] or e['sub_ids'][0] == e['sub_ids'][1]:
                continue
            i = e['sub_ids'].index(sub)
            if e['rewards'][i] < e['rewards'][1 - i]:
                lost.setdefault(e['episode_id'], dict(e, beater_sub=sub, beater_seat=i))
    packed = {json.loads(l)['episode_id'] for l in open(f'{OUT2}/MANIFEST.jsonl')} if os.path.exists(f'{OUT2}/MANIFEST.jsonl') else set()
    todo = [e for e in sorted(lost.values(), key=lambda e: e['create_time'], reverse=True) if e['episode_id'] not in packed]
    where = local_map() if todo else {}
    fx = None; got = copied = 0
    try:
        for e in todo:
            eid = e['episode_id']
            disk = glob.glob(str(ROOT / f'work/runs/deployment/*/hosted/episode-{eid}-replay.json.gz'))
            if eid in where:
                raw = zipfile.ZipFile(where[eid][0]).read(where[eid][1]); src = 'local-copy'; copied += 1
            elif disk:
                raw = gzip.decompress(open(disk[0], 'rb').read()); src = 'our-hosted-copy'; copied += 1
            else:
                fx = fx or F.Fetcher(); take(cap)
                raw = fx.get(eid); src = 'official-api'; got += 1; time.sleep(1.2)
            vm, why = F.verify(raw, eid, ref)
            if vm is None:
                print('reject', eid, why); continue
            c = {'episode_id': eid, 'day': e['create_time'][:10], 'cand_source': 'beats-our-beater', 'team_ids': e['team_ids'],
                 'sub_ids': e['sub_ids']}
            meta = F.make_meta(c, raw, vm, src, None)
            meta.update({'reason': 'beats-our-beater', 'beater_sub': e['beater_sub'], 'beater_team': e['team_names'][e['beater_seat']],
                         'beater_seat': e['beater_seat']})
            F.stage(OUT2, eid, raw, meta)
    except RunBudget:
        pass
    finally:
        n = F.pack(OUT2)
    return {'beater_subs': len({r['opp_sub'] for r in rows}), 'their_losses': len(lost), 'new_packed': n, 'downloaded': got, 'copied': copied}


if __name__ == '__main__':
    main()
