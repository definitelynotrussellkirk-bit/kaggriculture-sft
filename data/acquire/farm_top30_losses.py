#!/usr/bin/env python3
"""Farm every public game that a CURRENT top-30 policy lost (owner 2026-09-25 ~23:00 UTC: "whenever someone in the top 30 loses
... we want to find it and train that for sure"; include losses on their way up). Training target = the winner of each game.

Per run: leaderboard (1 call) -> for each top-N team, team-submissions (1 call) -> each submission whose public score is at least
the rank-N cutoff (their top-N policy) -> list its episodes (1 call) -> losses (reward below the opponent's) -> replay download
(1 call each) for games not already local (official dailies, reconstructed, any extra/*/MANIFEST.jsonl), newest first,
round-robin over losing teams. Every call increments --counter (one file per UTC day, shared with the other farmers) and the run
stops at --cap. Output: --out/archives/*.zip (official member format) + MANIFEST.jsonl (seats' team/submission ids, rewards,
winner seat, loser team rank at fetch) + LISTINGS-<stamp>.json (all listed episodes of those policies).

  python3 farm_top30_losses.py --out top30-losses --counter .../KAGGLE-REQUESTS-<day>.txt --cap 910 [--top 30]
"""
import argparse, collections, csv, datetime as dt, glob, io, json, os, sys, tempfile, time, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_topteam_replays as F
import requests

EXTRA = os.path.dirname(os.path.abspath(__file__))


class Budget:
    def __init__(self, path, cap):
        self.path, self.cap = path, cap

    def take(self):   # same lock as the shell bump(): flock on <counter>.lock; atomic replace (no truncated reads)
        import fcntl
        with open(self.path + '.lock', 'a') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            v = int(open(self.path).read().strip() or 0) if os.path.exists(self.path) else 0
            if v >= self.cap:
                raise SystemExit(f'budget reached: {v}/{self.cap}')
            tmp = f'{self.path}.tmp{os.getpid()}'; open(tmp, 'w').write(f'{v + 1}\n'); os.replace(tmp, self.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--counter', required=True)
    ap.add_argument('--cap', type=int, required=True)
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--max-replays', type=int, default=10 ** 9)
    ap.add_argument('--listings', help='reuse a LISTINGS json instead of listing again')
    ap.add_argument('--extra-episodes', help='JSON {"episodes": [...]} of more losses (same fields as LISTINGS episodes, e.g. '
                                             'older submissions of the same teams from the xsf index); fetched after the listed ones')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    b = Budget(a.counter, a.cap)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M')
    if a.listings:
        L = json.load(open(a.listings))
    else:
        b.take()
        tmp = tempfile.mkdtemp()
        api.competition_leaderboard_download('kaggriculture', tmp)
        f = os.path.join(tmp, os.listdir(tmp)[0])
        raw = zipfile.ZipFile(f).read(zipfile.ZipFile(f).namelist()[0]).decode('utf-8-sig') if f.endswith('.zip') else open(f, encoding='utf-8-sig').read()
        lbrows = list(csv.DictReader(io.StringIO(raw)))
        open(os.path.join(a.out, f'LEADERBOARD-{stamp}.csv'), 'w').write(raw)
        top = sorted(lbrows, key=lambda r: int(r['Rank']))[:a.top]
        cut = min(float(r['Score']) for r in top)
        L = {'stamp': stamp, 'cutoff': cut, 'teams': {}, 'episodes': []}
        for r in top:
            tid = int(r['TeamId'])
            b.take()
            subs = api.competition_team_submissions(tid)
            pol = []
            for s in subs:
                sid = int(getattr(s, 'id', None) or getattr(s, 'ref'))
                sc = getattr(s, 'public_score', None)
                try:
                    sc = float(sc)
                except (TypeError, ValueError):
                    sc = None
                if sc is not None and sc >= cut:
                    pol.append((sid, sc))
            L['teams'][tid] = {'rank': int(r['Rank']), 'name': r['TeamName'], 'score': float(r['Score']), 'policies': pol}
            for sid, sc in pol:
                b.take()
                for e in api.competition_list_episodes(sid):
                    ag = sorted(e.agents, key=lambda x: x.index)
                    if len(ag) != 2:
                        continue
                    L['episodes'].append({'episode_id': int(e.id), 'create_time': str(e.create_time),
                                          'type': str(e.type).split('.')[-1], 'state': str(e.state).split('.')[-1],
                                          'team_ids': [x.team_id for x in ag], 'team_names': [x.team_name for x in ag],
                                          'sub_ids': [x.submission_id for x in ag], 'rewards': [x.reward for x in ag],
                                          'policy_sub': sid, 'policy_team': tid})
                time.sleep(1.0)
        json.dump(L, open(os.path.join(a.out, f'LISTINGS-{stamp}.json'), 'w'))
    losses, seen = [], set()
    extra = json.load(open(a.extra_episodes))['episodes'] if a.extra_episodes else []
    for e in L['episodes'] + [dict(x, extra=True) for x in extra]:
        if e['type'] != 'EPISODE_TYPE_PUBLIC' or e['state'] != 'COMPLETED' or None in e['rewards']:
            continue
        i = e['sub_ids'].index(e['policy_sub'])
        if e['sub_ids'][0] == e['sub_ids'][1] or e['rewards'][i] >= e['rewards'][1 - i] or e['episode_id'] in seen:
            continue
        seen.add(e['episode_id'])
        losses.append(dict(e, day=e['create_time'][:10], loser_seat=i, winner_seat=1 - i))
    have, _ = F.local_ids(F.LOCAL_DEFAULT)
    for m in glob.glob(f'{EXTRA}/reconstructed/MANIFEST-*.jsonl') + glob.glob(f'{EXTRA}/*/MANIFEST.jsonl'):
        have |= {json.loads(l)['episode_id'] for l in open(m) if l.strip()}
    todo = [x for x in losses if x['episode_id'] not in have]
    print(f"top-{a.top} policies: {sum(len(t['policies']) for t in L['teams'].values())} of {len(L['teams'])} teams; "
          f"listed episodes {len(L['episodes'])}; losses {len(losses)}; already local {len(losses) - len(todo)}; to fetch {len(todo)}", flush=True)
    order = []
    for part in ([x for x in todo if not x.get('extra')], [x for x in todo if x.get('extra')]):
        by = collections.defaultdict(list)
        for x in sorted(part, key=lambda x: x['create_time'], reverse=True):
            by[x['policy_team']].append(x)
        while any(by.values()):
            for k in list(by):
                if by[k]:
                    order.append(by[k].pop(0))
    print(f"order: {sum(not x.get('extra') for x in order)} listed, {sum(bool(x.get('extra')) for x in order)} extra", flush=True)
    ref = json.load(open(F.REF_DEFAULT))
    fx = F.Fetcher()
    got = 0
    try:
        for x in order[:a.max_replays]:
            b.take()
            try:
                raw = fx.get(x['episode_id'])
            except requests.exceptions.HTTPError as err:
                code = err.response.status_code if err.response is not None else None
                print('http', code, x['episode_id'], flush=True)
                if code == 429:
                    break
                time.sleep(30)
                continue
            vm, why = F.verify(raw, x['episode_id'], ref)
            if vm is None:
                print('reject', x['episode_id'], why, flush=True)
                continue
            c = {'episode_id': x['episode_id'], 'day': x['day'], 'cand_source': 'top30-loss', 'team_ids': x['team_ids'],
                 'sub_ids': x['sub_ids']}
            meta = F.make_meta(c, raw, vm, 'official-api', None)
            team = L['teams'].get(x['policy_team']) or L['teams'].get(str(x['policy_team'])) or {}
            meta.update({'loser_seat': x['loser_seat'], 'loser_policy_sub': x['policy_sub'], 'loser_team_id': x['policy_team'],
                         'loser_team_rank_at_listing': team.get('rank'), 'reason': 'top30-loss-extra' if x.get('extra') else 'top30-loss',
                         'source_index': x.get('source_index')})
            F.stage(a.out, x['episode_id'], raw, meta)
            got += 1
            if got % 25 == 0:
                print('fetched', got, flush=True)
            time.sleep(1.2)
    finally:
        print('packed', F.pack(a.out), 'fetched', got, flush=True)


if __name__ == '__main__':
    main()
