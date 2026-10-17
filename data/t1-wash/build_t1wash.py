#!/usr/bin/env python3
"""T1-ONLY wash rosters (owner 14:2x "imitating his WIN"): MMPQ's (#1) own seats only, all dates, VAL-MMPQ excluded.
  T1W-BEST (4x per seat): MMPQ's winning seat in his mirror games that involve his best submission (highest-rated in the newest listing).
  T1W (1x): every other MMPQ winning seat (incl. other mirrors).
  WDL-only rows (N flow only): MMPQ's losing seats (incl. the losing side of mirrors), act_loss 0.
Only seats with a non-empty v11 .stp are written; the rest go to broad/t1/wash/UNCACHED.tsv (queued for caching).
Outputs broad/t1/wash/: rows-t1wash-nflow.<sha12>.tsv (+ .receipt.json with bucket_probs_arg), rows-t1wash-b.<sha12>.tsv
(integer weights 4/1), LATEST-NFLOW / LATEST-B symlinks."""
import glob, hashlib, json, os, sys, datetime as dt
import pandas as pd
FS = '/mnt/ai/scratch/farmNEAT'; OUT = f'{FS}/broad/t1/wash'
C = f'{FS}/base-v1-stepcache-v11'; CAT = f'{FS}/primitive-bc-data/catalog'; E = f'{FS}/primitive-bc-data/extra'
TEAM = 16681125; MULT = {'T1W': 1.0, 'BEAT1': 4.0}   # T1W-BEST folded into BEAT1
sys.path.insert(0, f'{FS}/broad'); import build_t1 as B1
ok = lambda e, z: os.path.exists(f'{C}/{e}-{z}.stp') and os.path.getsize(f'{C}/{e}-{z}.stp') > 0


def pinned_subs(tid):
    """R-POOL-PIN (owner 16:3x): every submission seen in the team's listings with its latest known score (not only the newest
    listing's live pair): an evicted best stays pinned; a newer submission takes over only by out-scoring it."""
    ss, sb, nf, live_sc = B1.listing_subs(tid)
    d = f"{E}/{B1.SLUG.get(tid, f'copy-team-{tid}')}"; sc = {}; games = {}
    for f in sorted(glob.glob(f'{d}/LISTINGS-*.json'), key=os.path.getmtime):
        L = json.load(open(f))
        for x in L.get('submissions') or []:
            try: sc[int(x.get('ref') or x.get('id'))] = float(x.get('publicScore') or x.get('public_score'))
            except (TypeError, ValueError): pass
        for e in L.get('episodes') or []:
            if e.get('state') == 'COMPLETED' and e.get('type') == 'EPISODE_TYPE_PUBLIC': games.setdefault(int(e['policy_sub']), set()).add(int(e['episode_id']))
    PINS = {16681125: 56701470, 16621497: 56704120, 16623559: 56705174}   # R-POOL-PIN (owner 16:3x)
    if tid in PINS and PINS[tid] in sc:
        p = PINS[tid]; cand = {p: sc[p], **{k: v for k, v in live_sc.items() if k > p and v > sc[p]}}   # a NEWER sub overtakes only by out-scoring
    else:
        cand = live_sc or sc                                                  # other teams: best of the newest listing's live pair
    return ss, sb, nf, cand, {k: len(v) for k, v in games.items()}, live_sc


def main():
    os.makedirs(OUT, exist_ok=True)
    seat_sub, subs, nf, sc, ngames, live_mm_sc = pinned_subs(TEAM)
    best = max(sc, key=lambda k: (sc[k], k)) if sc else max(subs)
    # owner 17:3x (ranks 1-5): the bin = wins of the pinned/best submission AND of any live submission NEWER than it
    KEEP_NEW = {56716576}   # owner 17:3x named DECEM's new submission: kept regardless of the 40-game rule
    newer_ok = lambda q, bq, live, ng: q > bq and (q in KEEP_NEW or not (ng.get(q, 0) >= 40 and live.get(q, float('inf')) < live.get(bq, sc.get(bq, float('-inf')))))
    acc_mm = {best} | {k for k in live_mm_sc if newer_ok(k, best, {**sc, **live_mm_sc}, ngames)}   # 17:40: settled-below newer subs stay out
    newest = max(subs) if subs else best
    new_ok = False   # superseded by acc_mm (17:3x): newer live MMPQ submissions join T1W at once
    pin_flag = [f'MMPQ new submission {newest} ({ngames.get(newest, 0)} rated games; joins T1W: {new_ok}) vs pinned {best} ({sc.get(best)})'] if newest > 56711476 else []
    import exclusion_union as XU
    held = XU.held_out_ids()
    FINAL = os.environ.get('T1_FINAL') == '1'   # owner: held-out MMPQ games enter only the final 18:40 wash (T1-ONLY-FINAL)
    held_out = {k: v for k, v in held.items() if k != 'VAL-MMPQ10-0930.json'}   # D-1544ho: VAL-MMPQ10 trains in the wash
    vm = set() if FINAL else {e for v in held_out.values() for e in v}
    s = pd.read_parquet(f'{CAT}/SEATS.parquet', columns=['episode_id', 'seat', 'day', 'kaggle', 'result', 'team_id', 'opp_team_id', 'archive', 'member',
                                                        'final_cash', 'opp_cash', 'submission_id', 'opp_submission_id', 'static', 'tsv_ready'])
    s['day'] = s.day.astype(str)
    m = s[(s.kaggle == True) & (s.team_id == TEAM) & ~s.episode_id.isin(vm) & (s.static != True) & (s.tsv_ready == True)].drop_duplicates(['episode_id', 'seat']).copy()
    m['sub'] = [seat_sub.get((int(e), int(z)), x) for e, z, x in zip(m.episode_id, m.seat, m.submission_id)]
    m['osub'] = [seat_sub.get((int(e), 1 - int(z)), x) for e, z, x in zip(m.episode_id, m.seat, m.opp_submission_id)]
    mirror = m.opp_team_id == TEAM
    win = m.result == 'W'
    m['bucket'] = None
    m.loc[win & m.bucket.isna() & m['sub'].isin(list(acc_mm)), 'bucket'] = 'T1W'   # pinned best + newer live subs (17:3x)
    WDAYS = json.loads(os.environ.get('WASH_DAYS', '["2026-09-30"]'))   # D-1609mx: only 09-30 games
    m.loc[~m.day.isin(WDAYS), 'bucket'] = None
    m['cached'] = [ok(int(e), int(z)) for e, z in zip(m.episode_id, m.seat)]
    qd = f'{OUT}/QA-DROP.json'   # POOL-QA drop list (failed checks / outliers): excluded from every rebuild
    QA = {int(k) for k in json.load(open(qd))} if os.path.exists(qd) else set()
    m = m[~m.episode_id.isin(QA)]
    # owner 16:0x D-1604wm: 80% #1 (MMPQ; T1W-BEST 4x inside) + 10% #2 + 10% #3 winning seats (09-29/30), exclusion union minus
    # nothing (split, leak, sealed, TOP4 VAL etc. apply to #2/#3), every held-out set applied
    import csv as _csv, glob as _glob
    lbf = sorted(_glob.glob(f'{E}/top30-losses/LEADERBOARD-*.csv'))[-1]
    lbr = sorted((r for r in _csv.DictReader(open(lbf, encoding='utf-8-sig')) if int(r['TeamId']) != 16624026), key=lambda r: int(r['Rank']))
    top = [(int(r['TeamId']), r['TeamName'], float(r['Score'])) for r in lbr[:20]]
    mix_flag0 = [] if (top[0][0] == TEAM and top[0][2] - top[1][2] >= 50) else [f"#1 {top[0][1]} {top[0][2]} vs #2 {top[1][1]} {top[1][2]} (gap {top[0][2] - top[1][2]:.1f})"]
    mix_flag = mix_flag0 + pin_flag
    others = pd.DataFrame()
    newsub_rule = {top[0][1]: {str(q): {'rated_games': ngames.get(q, 0), 'score': live_mm_sc.get(q), 'pin_score': sc.get(best), 'in': q in acc_mm} for q in live_mm_sc if q > best}}
    bests = {top[0][1]: {'team_id': TEAM, 'submission': best, 'score': sc.get(best), 'listing': nf, 'included_subs': sorted(acc_mm)}}
    if os.environ.get('WASH_MIX', '1') == '1':
        import exclusion_union as XU2
        xm = XU2.build(); xu = pd.read_csv(xm['path'], sep='\t', dtype={'seat': str})
        keepfam = {'leak', 'sealed', 'fixed', 'quarantine_sub'}   # D-1611pool: data-integrity exclusions only; no eval holdouts
        xe = set(xu[(xu.seat == '*') & xu.family.isin(keepfam)].eid.astype(int)) | {e for v in held.values() for e in v}
        vt = '/mnt/ai/scratch/farmNEAT/broad/VAL-TRAIN-0930.json'
        if os.path.exists(vt) and not os.path.exists('/mnt/ai/scratch/farmNEAT/broad/NO-EVALS'): xe |= {int(g['episode_id']) for g in json.load(open(vt))['games']}
        for k, (tid, nm, scr) in [(2, top[1]), (3, top[2])] + [(f'4X{j + 4}', t) for j, t in enumerate(top[3:int(os.environ.get('OTHERS_MAX_RANK', '12'))])]:   # owner 17:0x: ranks 4-12 only
            ss2, sb2, nf2, sc2, _ng2, live2 = pinned_subs(tid)
            b2 = max(sc2, key=lambda q: (sc2[q], q)) if sc2 else None
            rank = next((i + 1 for i, t in enumerate(top) if t[0] == tid), 99)
            pin2 = {16681125: 56701470, 16621497: 56704120, 16623559: 56705174}.get(tid)
            acc = ({b2} | {q for q in live2 if b2 is not None and newer_ok(q, b2, {**sc2, **live2}, _ng2)}) if rank <= 5 else {b2}
            if pin2 and rank <= 5: acc |= {pin2} | {q for q in live2 if q > pin2 and newer_ok(q, pin2, {**sc2, **live2}, _ng2)}   # union with the pin
            if rank <= 5: newsub_rule[nm] = {str(q): {'rated_games': _ng2.get(q, 0), 'score': live2.get(q), 'pin_score': sc2.get(b2), 'in': q in acc} for q in live2 if b2 is not None and q > b2}
            bests[nm] = {'team_id': tid, 'submission': b2, 'score': sc2.get(b2) if b2 else None, 'listing': nf2, 'included_subs': sorted(q for q in acc if q)}
            o = s[(s.kaggle == True) & (s.team_id == tid) & (s.result == 'W') & s.day.isin(WDAYS)   # mirrors count if the winner is the best sub
                  & ~s.episode_id.isin(xe) & (s.static != True) & (s.tsv_ready == True)].drop_duplicates(['episode_id', 'seat']).copy()
            o['sub'] = [ss2.get((int(e), int(z)), x) for e, z, x in zip(o.episode_id, o.seat, o.submission_id)]
            o['osub'] = [ss2.get((int(e), 1 - int(z)), x) for e, z, x in zip(o.episode_id, o.seat, o.opp_submission_id)]
            o = o[o['sub'].isin(list(acc))] if b2 is not None else o.iloc[0:0]
            o = o[~o.episode_id.isin(QA)]
            # not yet in the catalog (fetched since the last :20 pass): wins from the team's listings + local manifests, cached only
            have_k = set(zip(o.episode_id.astype(int), o.seat.astype(int)))
            dteam = f"{E}/{B1.SLUG.get(tid, f'copy-team-{tid}')}"
            Lt = {}
            for f in sorted(_glob.glob(f'{dteam}/LISTINGS-*.json'), key=os.path.getmtime):
                for e in json.load(open(f)).get('episodes', []): Lt[int(e['episode_id'])] = e
            mant = {int(json.loads(l)['episode_id']): json.loads(l) for l in open(f'{dteam}/MANIFEST.jsonl') if l.strip()} if os.path.exists(f'{dteam}/MANIFEST.jsonl') else {}
            add = []
            for eid, e in Lt.items():
                sid_, tid_, rw_ = e.get('sub_ids') or [], e.get('team_ids') or [], e.get('rewards') or []
                if eid in xe or eid in QA or e.get('state') != 'COMPLETED' or e.get('create_time', '')[:10] not in WDAYS or len(sid_) != 2 or None in rw_: continue
                for zz in (0, 1):
                    if tid_[zz] == tid and sid_[zz] in acc and rw_[zz] > rw_[1 - zz] and (eid, zz) not in have_k and eid in mant and ok(eid, zz):
                        mm = mant[eid]
                        add.append({**{c: None for c in o.columns}, 'episode_id': eid, 'seat': zz, 'day': e['create_time'][:10], 'kaggle': True, 'result': 'W',
                                    'team_id': tid, 'opp_team_id': tid_[1 - zz], 'archive': f"{dteam}/{mm['archive']}", 'member': mm['member'],
                                    'final_cash': float(rw_[zz]), 'opp_cash': float(rw_[1 - zz]), 'submission_id': sid_[zz], 'opp_submission_id': sid_[1 - zz],
                                    'static': False, 'tsv_ready': True, 'sub': sid_[zz], 'osub': sid_[1 - zz], 'cached': True})
            if add: o = pd.concat([o, pd.DataFrame(add, columns=o.columns)])
            o['bucket'] = 'T4X' if str(k).startswith('4X') else f'T{k}W'   # D-1611pool: ranks 4-20 one bin, uniform over seats
            if os.environ.get('POOL_MIX') == '8020':   # owner 17:5x top mode: ranks 2-5 one OTHER bucket (uniform over seats), ranks 6+ out
                PROMOTE_T25 = set()   # one-line promotion, e.g. {16736435} puts CDE's (pin/best + newer live subs) wins into T25
                o['bucket'] = 'T25' if (rank <= 5 or tid in PROMOTE_T25) else None
                o = o[o.bucket.notna()]
            o['cached'] = [ok(int(e), int(z)) for e, z in zip(o.episode_id, o.seat)]
            if len(o): others = pd.concat([others, o])
    if len(others): others['cached'] = others['cached'].astype(bool)
    MIX = {'T1': 0.60, 'T2W': 0.15, 'T3W': 0.15, 'T4X': 0.10}   # D-1609mx
    if os.environ.get('POOL_MIX') == '8020':
        MIX = {'T1': 0.80, 'T25': 0.20}
    elif os.environ.get('POOL_MIX') in ('801010', '801406'):   # owner 16:5x last hour; 17:4x "80-14-6": DECEM .14 / Victor .06 by TEAM
        MIX = {'T1': 0.80, 'T2W': 0.10, 'T3W': 0.10, 'T4X': 0.0}
        if os.environ.get('POOL_MIX') == '801406':
            by_team = {16623559: 0.14, 16621497: 0.06}
            MIX['T2W'] = by_team.get(top[1][0], 0.10); MIX['T3W'] = by_team.get(top[2][0], 0.10)
        others = others[others.bucket != 'T4X'] if len(others) else others
    inc_p = f'{OUT}/SUBS-INCLUDED.json'
    prev_inc = set(json.load(open(inc_p))) if os.path.exists(inc_p) else None
    now_inc = {q for v in bests.values() for q in v.get('included_subs', [])}
    new_subs = sorted(now_inc - prev_inc) if prev_inc is not None else []
    if os.environ.get('NO_LINK') != '1': json.dump(sorted(now_inc), open(inc_p, 'w'))
    bp_ = f'{OUT}/BESTS.json'
    prev = json.load(open(bp_)) if os.path.exists(bp_) else {}
    changed = [f"{k}: {prev[k]['submission']} -> {v['submission']}" for k, v in bests.items() if k in prev and prev[k].get('submission') != v['submission']]
    if os.environ.get('NO_LINK') != '1': json.dump(bests, open(bp_, 'w'), indent=1)
    act = m[m.bucket.notna() & m.cached].sort_values(['day', 'episode_id', 'seat'])
    # BEAT1 (owner 16:1x "GOLD", mandatory): the winning seat of every game MMPQ's best submission LOST to another team, 4x a T1W
    # seat inside the #1 bin; known losses come from the MMPQ listings (not the catalog), archive/member from the copy-mmpq manifest
    Lb = {}
    for f in sorted(_glob.glob(f'{E}/copy-mmpq/LISTINGS-*.json'), key=os.path.getmtime):
        for e in json.load(open(f))['episodes']: Lb[int(e['episode_id'])] = e
    manb = {int(json.loads(l)['episode_id']): json.loads(l) for l in open(f'{E}/copy-mmpq/MANIFEST.jsonl') if l.strip()}
    brows, bmiss = [], []
    live_mm = set(B1.listing_subs(TEAM)[3]) | {best}   # MMPQ's live submissions (newest listing's pair) + the pinned best
    for eid, e in Lb.items():
        sid, tid, rw = e.get('sub_ids') or [], e.get('team_ids') or [], e.get('rewards') or []
        # BEAT1 (coordinator 16:3x correction): the winning seat of EVERY game an MMPQ live submission lost, mirrors included
        if e.get('state') != 'COMPLETED' or len(tid) != 2 or len(sid) != 2 or None in rw or rw[0] == rw[1]: continue
        ws = 0 if rw[0] > rw[1] else 1; ms = 1 - ws
        if tid[ms] != TEAM or sid[ms] not in live_mm: continue
        mm = manb.get(eid)
        if eid in QA: continue
        if mm is None or not ok(eid, ws): bmiss.append((eid, ws, mm)); continue
        brows.append({**{c: None for c in act.columns}, 'episode_id': eid, 'seat': ws, 'day': e['create_time'][:10], 'kaggle': True, 'result': 'W',
                      'team_id': tid[ws], 'opp_team_id': tid[ms], 'archive': f"{E}/copy-mmpq/{mm['archive']}", 'member': mm['member'],
                      'final_cash': float(rw[ws]), 'opp_cash': float(rw[ms]), 'submission_id': sid[ws], 'opp_submission_id': sid[ms],
                      'static': False, 'tsv_ready': True, 'sub': sid[ws], 'osub': sid[ms], 'bucket': 'BEAT1', 'cached': True})
    need = {115669607, 115838526, 115935708}   # the 3 known MMPQ live-sub losses (coordinator 16:3x): build fails if one is missing
    have_b = {r['episode_id'] for r in brows}
    if need - have_b: raise SystemExit(f'REFUSED: known BEAT1 games missing: {sorted(need - have_b)}')
    if bmiss:
        open(f'{OUT}/BEAT1-UNCACHED.tsv', 'w').write('archive\tmember\tday\teid\tseat\tweight\treason\n' + ''.join(
            f"{E}/copy-mmpq/{mm['archive']}\t{mm['member']}\t2026-09-30\t{eid}\t{ws}\t1\tbeat1\n" for eid, ws, mm in bmiss if mm))
        raise SystemExit(f'REFUSED: known BEAT1 seats missing from the pool (not local/cached yet): {[(e, w) for e, w, _ in bmiss]}')
    if brows:
        act = act[~act.set_index(['episode_id', 'seat']).index.isin([(r['episode_id'], r['seat']) for r in brows])]
        act = pd.concat([act, pd.DataFrame(brows, columns=act.columns)])
    if len(others): act = pd.concat([act, others[others.cached][act.columns]])
    act = act.drop_duplicates(['episode_id', 'seat'], keep='first').sort_values(['day', 'episode_id', 'seat'])   # BEAT1 wins over T2W/T3W/T4X
    wdl = m.iloc[0:0]   # set below: only the losing seats of this pool's own games (opponents of the act rows)
    unc = m[~m.cached & m.bucket.notna()]
    if len(others): unc = pd.concat([unc, others[~others.cached][unc.columns]])
    xb = ['T4X'] if (act.bucket == 'T4X').any() else []
    # WDL-only rows (coordinator 16:2x): exactly the losing seat of each act game (09-30 by construction), cached ones only
    opp = act[['episode_id', 'seat', 'day', 'archive', 'member', 'final_cash', 'opp_cash']].copy()
    opp['seat'] = 1 - opp.seat.astype(int); opp['final_cash'], opp['opp_cash'] = act['opp_cash'].values, act['final_cash'].values
    opp = opp.drop_duplicates(['episode_id', 'seat'])
    opp = opp[~opp.set_index(['episode_id', 'seat']).index.isin(list(zip(act.episode_id.astype(int), act.seat.astype(int))))]
    opp['cached'] = [ok(int(e), int(z)) for e, z in zip(opp.episode_id, opp.seat)]
    wdl = opp[opp.cached].sort_values(['day', 'episode_id', 'seat'])
    unc_opp = opp[~opp.cached]
    n = {b: int((act.bucket == b).sum()) for b in list(MULT) + ['T2W', 'T3W', 'T25'] + xb}
    tot = sum(n[b] * MULT[b] for b in MULT)
    t1 = {b: (n[b] * MULT[b] / tot if tot else 0.0) for b in MULT if n[b] > 0}      # never a 0.0 bucket
    live = {'T1': MIX['T1'] if t1 else 0.0, **{b: MIX[b] for b in ('T2W', 'T3W', 'T25') if b in MIX and n[b] > 0}, **({'T4X': MIX['T4X']} if xb and MIX.get('T4X') else {})}
    z = sum(live.values())
    probs = {**{b: v * live['T1'] / z for b, v in t1.items()}, **{b: live[b] / z for b in ('T2W', 'T3W', 'T25') if b in live},
             **{b: live['T4X'] / z / len(xb) for b in xb}}
    # N flow roster
    L = ['path\teid\tseat\tweight\ttier\twdl\tact_loss\tfinal_cash\topp_cash\tbucket']
    for r in act.itertuples(index=False):
        L.append(f"{C}/{int(r.episode_id)}-{int(r.seat)}.stp\t{int(r.episode_id)}\t{int(r.seat)}\t1.0\t1\t2\t1\t{float(r.final_cash)}\t{float(r.opp_cash)}\t{r.bucket}")
    for r in wdl.itertuples(index=False):
        L.append(f"{C}/{int(r.episode_id)}-{int(r.seat)}.stp\t{int(r.episode_id)}\t{int(r.seat)}\t1.0\t1\t0\t0\t{float(r.final_cash)}\t{float(r.opp_cash)}\t")
    body = '\n'.join(L) + '\n'; sha = hashlib.sha256(body.encode()).hexdigest()
    nfp = f'{OUT}/rows-t1wash-nflow.{sha[:12]}.tsv'; open(nfp + '.tmp', 'w').write(body); os.replace(nfp + '.tmp', nfp)
    present = sorted(set(int(e) for e in act.episode_id) & {e for v in held_out.values() for e in v})
    val10_in = sorted(set(int(e) for e in act.episode_id) & set(held.get('VAL-MMPQ10-0930.json', [])))
    if present and not FINAL: raise SystemExit(f'REFUSED: held-out episodes in the wash roster: {present[:10]}')
    rec = {'schema': 't1wash-nflow/v3', 'final': FINAL, 'mix': MIX, 'best_submissions': bests, 'best_changed': changed,
           'rule_d1604': 'winning seats only from each team\'s highest-rated live submission (newest listing scores); mirror 4x only for #1 best', 'leaderboard': os.path.basename(lbf), 'top3': top, 'mix_flag': mix_flag,
           'beat1': sorted(int(e) for e in act[act.bucket == 'BEAT1'].episode_id),
           'seats_by_submission': {f'{b}:{int(k)}': int(v) for (b, k), v in act.groupby(['bucket', 'sub']).size().items() if k == k},
           'new_subs': new_subs, 'newsub_rule': newsub_rule,
           'team_seats': {'#1 ' + top[0][1]: n.get('T1W', 0) + n.get('T1W-BEST', 0), 'BEAT1 (winners over MMPQ live subs, mirrors incl.)': n.get('BEAT1', 0), '#2 ' + top[1][1]: n.get('T2W', 0), '#3 ' + top[2][1]: n.get('T3W', 0),
                          'ranks 4-12': n.get('T4X', 0)}, 'days': WDAYS,
           'ranks_4_12_by_team': {nm: int(((act.bucket == 'T4X') & (act.team_id == tid)).sum()) for tid, nm, _ in top[3:12]}, 'held_out_sets': {k: len(v) for k, v in held_out.items()}, 'held_out_included': present,
           'val_mmpq10_included': val10_in, 'rule': 'owner 14:2x: MMPQ own winning seats only (act), his losing seats wdl-only; T1W-BEST 4x = mirror wins involving best sub',
           'team_id': TEAM, 'best_submission': best, 'best_score': sc.get(best), 'listing': nf, 'rows': len(L) - 1, 'act_rows': len(act), 'wdl_rows': len(wdl), 'wdl_rule': 'losing seats of the act games only (09-30)', 'wdl_uncached': int(len(unc_opp)),
           'bucket_rows': n, 'multipliers': MULT, 'bucket_probs': {b: round(p, 6) for b, p in probs.items()},
           'bucket_probs_arg': ','.join(f'{b}={probs[b]:.6f}' for b in probs), 'uncached_seats': int(len(unc)),
           'excluded_val_mmpq': sorted(vm), 'days': sorted(act.day.unique().tolist()), 'built_utc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           't1w_best_episodes': sorted(int(e) for e in act[act.bucket == 'T1W-BEST'].episode_id), 'out': nfp, 'out_sha256': sha}
    json.dump(rec, open(nfp + '.receipt.json', 'w'), indent=1)
    # B roster (rows-buckets format, integer weights 4:1)
    # B: integer repeat weights so each bucket's share of sampled rows = probs[b] (per-seat weight = probs/n, scaled, rounded)
    per = {b: probs[b] / n[b] for b in probs}; base = min(per.values()); scale = 20.0
    wint = {b: max(1, int(round(scale * per[b] / base))) for b in per}
    massb = {b: wint[b] * n[b] for b in wint}; tm = sum(massb.values())
    real_b = {b: round(massb[b] / tm, 4) for b in massb}
    LB = ['archive\tmember\tday\teid\tseat\tweight\treason'] + [
        f"{r.archive}\t{r.member}\t{r.day}\t{int(r.episode_id)}\t{int(r.seat)}\t{wint[r.bucket]}\tbucket:{r.bucket}" for r in act.itertuples(index=False)]
    bb = '\n'.join(LB) + '\n'; bsha = hashlib.sha256(bb.encode()).hexdigest()
    bp = f'{OUT}/rows-t1wash-b.{bsha[:12]}.tsv'; open(bp + '.tmp', 'w').write(bb); os.replace(bp + '.tmp', bp)
    json.dump({'schema': 't1wash-b/v2', 'roster': bp, 'sha256': bsha, 'rows': len(LB) - 1, 'bucket_rows': n, 'integer_weight': wint,
               'target_share': {b: round(p, 4) for b, p in probs.items()}, 'realised_share': real_b, 'mix_flag': mix_flag, 'top3': top,
               'best_submission': best}, open(bp + '.receipt.json', 'w'), indent=1)
    open(f'{OUT}/UNCACHED.tsv', 'w').write('archive\tmember\tday\teid\tseat\tweight\treason\n' + ''.join(
        f"{r.archive}\t{r.member}\t{r.day}\t{int(r.episode_id)}\t{int(r.seat)}\t1\tt1wash\n" for r in unc.itertuples(index=False)) + ''.join(
        f"{r.archive}\t{r.member}\t{r.day}\t{int(r.episode_id)}\t{int(r.seat)}\t1\tt1wash-wdl\n" for r in unc_opp.itertuples(index=False)))
    LP = os.environ.get('LINK_PREFIX', 'LATEST')
    for link, tgt in (((f'{LP}-NFLOW', nfp), (f'{LP}-B', bp)) if os.environ.get('NO_LINK') != '1' else ()):
        t = f'{OUT}/{link}.tmp'
        if os.path.lexists(t): os.remove(t)
        os.symlink(tgt, t); os.replace(t, f'{OUT}/{link}')
    print(json.dumps({k: rec[k] for k in ('best_submission', 'best_score', 'act_rows', 'wdl_rows', 'bucket_rows', 'bucket_probs_arg', 'uncached_seats', 't1w_best_episodes', 'team_seats', 'mix_flag', 'best_submissions', 'best_changed')}))
    print(json.dumps({'b_integer_weight': wint, 'b_realised_share': real_b}))
    print(json.dumps({'nflow': nfp, 'b': bp}))


if __name__ == '__main__':
    main()
