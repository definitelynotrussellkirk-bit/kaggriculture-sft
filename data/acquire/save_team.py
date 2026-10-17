#!/usr/bin/env python3
"""Save every public game of one team, including submissions it has since replaced (owner 2026-09-26: Dipam Chakraborty climbed
to the top, then replaced his submission).

team submissions (1 call) -> SUBMISSIONS-<stamp>.json; each submission's episodes (1 call each) -> LISTINGS-<stamp>.json;
replay download (1 call each) for public completed games not already local, highest-scoring submission first, newest first.
Every call increments --counter; the run stops at --cap.

  python3 save_team.py --team-id 16811917 --out save-dipam-chakraborty --counter .../KAGGLE-REQUESTS-<day>.txt --cap 2700 [--list-only]
"""
import argparse, datetime as dt, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_topteam_replays as F
from farm_top30_losses import Budget, EXTRA
import requests


def sub_record(s):
    d = s.to_dict() if hasattr(s, 'to_dict') else {k.lstrip('_'): v for k, v in vars(s).items()}
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--team-id', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--counter', required=True)
    ap.add_argument('--cap', type=int, required=True)
    ap.add_argument('--listings', help='reuse a LISTINGS json instead of listing again')
    ap.add_argument('--list-only', action='store_true')
    ap.add_argument('--subs-only', action='store_true', help='list the team submissions (1 call) and stop')
    ap.add_argument('--subs', help='comma-separated submission ids: list and fetch only these')
    ap.add_argument('--max-replays', type=int, default=10 ** 9)
    ap.add_argument('--newest-sub-first', action='store_true', help='owner 09-30 08:4x: newest submission first (then newest game), not highest score')
    ap.add_argument('--reuse-listing-min', type=float, default=0, help='reuse the newest LISTINGS json in --out if younger than this many minutes (saves auth + list calls)')
    ap.add_argument('--active-only', type=int, default=0, help='list episodes only for the N newest submissions (0 = all); older ones come from earlier LISTINGS')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    try:   # R-FETCH-EFFICIENCY (owner 16:1x one pool: 09-30 WINS of each top-20 team's HIGHEST-RATED live submission): no API call
        _pol = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FETCH-POLICY.json')))   # for teams outside it
    except (OSError, ValueError):
        _pol = {}
    if _pol.get('pool_only') and not a.list_only and not a.subs_only:
        import csv
        _lb = sorted(glob.glob(os.path.join(EXTRA, 'top30-losses', 'LEADERBOARD-*.csv')))
        _rk = {int(r['TeamId']): int(r['Rank']) for r in csv.DictReader(open(_lb[-1], encoding='utf-8-sig'))} if _lb else {}
        if _rk.get(a.team_id, 10 ** 6) > int(_pol.get('pool_rank_max', 20)) and str(a.team_id) not in set(map(str, _pol.get('allsubs_teams', []))):
            print(f'policy pool_only: team rank {_rk.get(a.team_id)} > {_pol.get("pool_rank_max", 20)}: not in the pool, no calls', flush=True)
            print('public games 0; already local 0; to fetch 0', flush=True)
            print('packed 0 fetched 0', flush=True)
            return
    b = Budget(a.counter, a.cap)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M')
    pick = {int(x) for x in a.subs.split(',')} if a.subs else None
    if _pol.get('pool_only') and a.reuse_listing_min > 0:   # R-FETCH-TOP5-ALLSUBS: ranks 1..allsubs_rank_max re-list at least every
        import csv                                          # allsubs_reuse_min (11) minutes so a new upload's games land within a cycle
        _lb2 = sorted(glob.glob(os.path.join(EXTRA, 'top30-losses', 'LEADERBOARD-*.csv')))
        _rk2 = {int(r['TeamId']): int(r['Rank']) for r in csv.DictReader(open(_lb2[-1], encoding='utf-8-sig'))} if _lb2 else {}
        if _rk2.get(a.team_id, 10 ** 6) <= int(_pol.get('allsubs_rank_max', 0)) or str(a.team_id) in set(map(str, _pol.get('allsubs_teams', []))):
            a.reuse_listing_min = min(a.reuse_listing_min, float(_pol.get('allsubs_reuse_min', 11)))
    if not a.listings and a.reuse_listing_min > 0:
        cands = sorted(glob.glob(os.path.join(a.out, 'LISTINGS-*.json')), key=os.path.getmtime)
        if cands and time.time() - os.path.getmtime(cands[-1]) < a.reuse_listing_min * 60:
            a.listings = cands[-1]
            print('reusing listing', os.path.basename(cands[-1]), flush=True)
    if a.listings:
        L = json.load(open(a.listings))
        if pick is not None:
            L["episodes"] = [e for e in L["episodes"] if e["policy_sub"] in pick]
    else:
        b.take()   # `import kaggle` authenticates at import (one IntrospectToken request)
        from kaggle import api   # already authenticated; no second IntrospectToken request
        b.take()
        subs = [sub_record(s) for s in api.competition_team_submissions(a.team_id)]
        json.dump(subs, open(os.path.join(a.out, f'SUBMISSIONS-{stamp}.json'), 'w'), indent=1)
        if a.subs_only:
            for s in subs:
                print({k: s.get(k) for k in ('ref', 'id', 'date', 'status', 'public_score', 'publicScore', 'description', 'file_name', 'fileName')
                       if s.get(k) not in (None, '')}, flush=True)
            return
        L = {'stamp': stamp, 'team_id': a.team_id, 'submissions': subs, 'episodes': []}
        listed = subs
        if a.active_only:
            listed = sorted(subs, key=lambda s: str(s.get('dateSubmitted') or s.get('date') or ''), reverse=True)[:a.active_only]
        for s in listed:
            sid = int(s.get('ref') or s.get('id'))
            if pick is not None and sid not in pick:
                continue
            b.take()
            for e in api.competition_list_episodes(sid):
                ag = sorted(e.agents, key=lambda x: x.index)
                if len(ag) != 2:
                    continue
                L['episodes'].append({'episode_id': int(e.id), 'create_time': str(e.create_time),
                                      'type': str(e.type).split('.')[-1], 'state': str(e.state).split('.')[-1],
                                      'team_ids': [x.team_id for x in ag], 'team_names': [x.team_name for x in ag],
                                      'sub_ids': [x.submission_id for x in ag], 'rewards': [x.reward for x in ag],
                                      'policy_sub': sid, 'policy_team': a.team_id})
            time.sleep(1.0)
        if a.active_only:   # PRO CHECK-GRABBER-CHAIN: keep unfetched games of older submissions from the previous listing
            relisted = {int(s.get('ref') or s.get('id')) for s in listed}
            prev = sorted(glob.glob(os.path.join(a.out, 'LISTINGS-*.json')), key=os.path.getmtime)
            if prev:
                try:
                    P = json.load(open(prev[-1]))
                    carry = [e for e in P.get('episodes', []) if e.get('policy_sub') not in relisted]
                    L['episodes'].extend(carry)
                    L['carried_from'], L['carried_episodes'] = os.path.basename(prev[-1]), len(carry)
                    print(f'carried {len(carry)} episodes of older submissions from {os.path.basename(prev[-1])}', flush=True)
                except (OSError, ValueError) as ex:
                    print(f'carry-over skipped: {ex}', flush=True)
        json.dump(L, open(os.path.join(a.out, f'LISTINGS-{stamp}.json'), 'w'))
    score = {}
    for s in L['submissions']:
        try:
            score[int(s.get('ref') or s.get('id'))] = float(s.get('public_score') or s.get('publicScore'))
        except (TypeError, ValueError):
            score[int(s.get('ref') or s.get('id'))] = -1.0
    eps, seen = [], set()
    for e in L['episodes']:
        if e['type'] not in ('EPISODE_TYPE_PUBLIC', 'EPISODE_TYPE_VALIDATION') or e['state'] != 'COMPLETED' or None in e['rewards'] or e['episode_id'] in seen:
            continue
        seen.add(e['episode_id'])
        eps.append(e)
    have, _ = F.local_ids(F.LOCAL_DEFAULT)
    for m in glob.glob(f'{EXTRA}/reconstructed/MANIFEST-*.jsonl') + glob.glob(f'{EXTRA}/*/MANIFEST.jsonl'):
        have |= {json.loads(l)['episode_id'] for l in open(m) if l.strip()}
    todo = [e for e in eps if e['episode_id'] not in have]
    if a.newest_sub_first:
        todo.sort(key=lambda e: (-int(e['policy_sub']), [-ord(c) for c in e['create_time']]))
    else:
        todo.sort(key=lambda e: (e.get('type') != 'EPISODE_TYPE_VALIDATION', -score.get(e['policy_sub'], -1.0), [-ord(c) for c in e['create_time']]))  # owner 09-29 22:00: validation (mirror) games first
    pol = {}
    try:   # owner 09-30 12:4x: bigger share of the budget to TODAY's replays; older ones scarce and selective (FETCH-POLICY.json, read every call)
        pol = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FETCH-POLICY.json')))
    except (OSError, ValueError):
        pass
    if pol.get('pool_only'):   # R-FETCH-EFFICIENCY: only 09-30 (UTC) WINS of the highest-rated of the 2 live submissions, newest first
        today = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
        won = lambda e: e['rewards'][e['sub_ids'].index(e['policy_sub'])] > e['rewards'][1 - e['sub_ids'].index(e['policy_sub'])]
        live = sorted(L['submissions'], key=lambda s: str(s.get('dateSubmitted') or s.get('date') or ''), reverse=True)[:2]
        def _sc(s):
            try:
                return float(s.get('publicScore') or s.get('public_score'))
            except (TypeError, ValueError):
                return -1.0
        best = int(max(live, key=_sc).get('ref') or max(live, key=_sc).get('id')) if live else None
        pool = sorted((e for e in todo if e['create_time'][:10] >= today and e['policy_sub'] == best and won(e)),
                      key=lambda e: [-ord(c) for c in e['create_time']])
        if str(a.team_id) in set(map(str, pol.get('losses_all_teams', []))):   # R-FETCH-MMPQ-LOSSES: every 09-30 game this team LOST
            mirror = lambda e: len(set(e.get('team_ids') or [])) == 1            # or played against itself, any listed submission, first,
            lost = sorted((e for e in todo if e['create_time'][:10] >= today and e not in pool and (mirror(e) or not won(e))),
                          key=lambda e: [-ord(c) for c in e['create_time']])     # no cap (BEAT1 bucket)
            print(f'policy losses_all_teams: {len(lost)} lost/mirror 09-30 games of subs {sorted({e["policy_sub"] for e in lost})} first', flush=True)
            pool = lost + pool
        rest = [e for e in todo if e not in pool and e['create_time'][:10] >= today]
        rmax = int(pol.get('rest_max', 0))
        _rk = {}
        try:
            import csv
            _lbf = sorted(glob.glob(os.path.join(EXTRA, 'top30-losses', 'LEADERBOARD-*.csv')))[-1]
            _rk = {int(r['TeamId']): int(r['Rank']) for r in csv.DictReader(open(_lbf, encoding='utf-8-sig'))}
        except (IndexError, OSError, ValueError):
            pass
        if _rk.get(a.team_id, 10 ** 6) <= int(pol.get('allsubs_rank_max', 0)) or str(a.team_id) in set(map(str, pol.get('allsubs_teams', []))):   # R-FETCH-TOP3/5-ALLSUBS + allsubs_teams (owner 17:5x Marvin)
            _gap = float(pol.get('allsubs_max_gap', 1e9)); _top = max(_sc(x) for x in live)   # coordinator 17:5x: skip a live sub rated
            _now = dt.datetime.now(dt.timezone.utc)
            def _young(x):   # coordinator 17:5x: a submission uploaded < allsubs_young_h (3) hours ago is exempt (rating still climbing)
                try:
                    t = dt.datetime.fromisoformat(str(x.get('dateSubmitted') or x.get('date')).replace('Z', '+00:00'))
                    return (_now - (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc))).total_seconds() < 3600 * float(pol.get('allsubs_young_h', 3))
                except (TypeError, ValueError):
                    return False
            livesubs = {int(x.get('ref') or x.get('id')) for x in live if _sc(x) < 0 or _young(x) or _sc(x) >= _top - _gap}   # > gap below the best
            print(f'policy allsubs gap guard: live {[(int(x.get("ref") or x.get("id")), _sc(x)) for x in live]}, kept {sorted(livesubs)} (max gap {_gap:g})', flush=True)
            first = ([e for e in pool if len(set(e.get('team_ids') or [])) == 1 or not won(e)]            # MMPQ losses/mirrors stay first
                     if str(a.team_id) in set(map(str, pol.get('losses_all_teams', []))) else [])
            allg = sorted((e for e in todo if e['create_time'][:10] >= today and e['policy_sub'] in livesubs and e not in first),
                          key=lambda e: (-int(e['policy_sub']), not won(e), [-ord(c) for c in e['create_time']]))   # newest sub, wins, newest
            pool, rest, rmax = first + allg, [], 0
            print(f'policy allsubs (rank {_rk.get(a.team_id)}): live subs {sorted(livesubs)}; losses/mirrors first {len(first)}; then {len(allg)} 09-30 games, newest submission first, wins before losses', flush=True)
        print(f'policy pool_only: best live sub {best}; pool wins today {len(pool)}; other today {len(rest)} (cap {rmax}); older skipped', flush=True)
        todo = pool + rest[:rmax]
    elif pol.get('today_first'):
        today = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
        won = lambda e: e['rewards'][e['sub_ids'].index(e['policy_sub'])] > e['rewards'][1 - e['sub_ids'].index(e['policy_sub'])]
        lf = str(a.team_id) in set(map(str, pol.get('losses_first_teams', [])))   # their losses = winners over them, first
        new_ = sorted((e for e in todo if e['create_time'][:10] >= today), key=lambda e: (lf and won(e), -int(e['policy_sub']), [-ord(c) for c in e['create_time']]))   # owner 13:1x 'from their new policies': newest submission first
        old_ = [e for e in todo if e['create_time'][:10] < today and (won(e) or not pol.get('old_wins_only', True) or str(a.team_id) in set(map(str, pol.get('all_games_teams', []))))]   # owner 14:1x 'we want all that one'
        cap = int(pol.get('old_max_by_team', {}).get(str(a.team_id), pol.get('old_max', 0)))
        print(f'policy today_first: today {len(new_)}, older eligible {len(old_)} (cap {cap}), older skipped {len(todo) - len(new_) - len(old_)}', flush=True)
        todo = new_ + old_[:cap]
    mx = pol.get('max_by_team', {}).get(str(a.team_id))
    if mx is not None:   # owner 12:4x: sample lower top-20 teams a little, for diversity
        print(f'policy max_by_team: {len(todo)} -> {min(len(todo), int(mx))}', flush=True)
        todo = todo[:int(mx)]
    per = {}
    for e in eps:
        k = e['policy_sub']
        i = e['sub_ids'].index(k)
        w = e['rewards'][i] > e['rewards'][1 - i]
        n, nw, nl = per.get(k, (0, 0, 0))
        per[k] = (n + 1, nw + w, nl + (e['episode_id'] not in have))
    for k, (n, nw, nl) in sorted(per.items(), key=lambda kv: -score.get(kv[0], -1.0)):
        print(f'submission {k} score {score.get(k)}: public games {n}, wins {nw}, not local {nl}', flush=True)
    print(f'public games {len(eps)}; already local {len(eps) - len(todo)}; to fetch {len(todo)}', flush=True)
    if a.list_only or not todo or a.max_replays <= 0:
        return
    ref = json.load(open(F.REF_DEFAULT))
    for _ in range(1 if "kaggle" in sys.modules else 2):   # Fetcher(): KaggleApi().authenticate() (+ the import-time one
        b.take()                                        # when kaggle is not imported yet), one IntrospectToken request each
    fx = F.Fetcher()
    got = 0
    try:
        for x in todo[:a.max_replays]:
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
            c = {'episode_id': x['episode_id'], 'day': x['create_time'][:10], 'cand_source': 'saved-team',
                 'team_ids': x['team_ids'], 'sub_ids': x['sub_ids']}
            meta = F.make_meta(c, raw, vm, 'official-api', None)
            meta.update({'reason': 'saved-team', 'saved_team_id': a.team_id, 'policy_sub': x['policy_sub'],
                         'policy_seat': x['sub_ids'].index(x['policy_sub']), 'policy_sub_score': score.get(x['policy_sub'])})
            F.stage(a.out, x['episode_id'], raw, meta)
            got += 1
            if got % 25 == 0:
                print('fetched', got, flush=True)
            time.sleep(1.2)
    finally:
        print('packed', F.pack(a.out), 'fetched', got, flush=True)


if __name__ == '__main__':
    main()
