#!/usr/bin/env python3
"""Self-play buckets (owner 13:3x: "one self-play with random tests (only winners, 5% of total) ... and 15% of total - selfplay games
among our BEST current ones, we keep adding our newest best"). Source: league/records/**/<prefix>-NNNN[__node]/<k>/ game dirs
(game.json + replay.json.gz) of the SP-RAND / SP-BEST streams. Winner seat only (draws skipped), DONE/DONE, no failed step.
Gate: a game whose world_seed is in any non-SP eval bank (league/records/banks/*.json, tournament/T-plan/F banks) is refused.
Keep the newest `keep` winner seats; then trim any one agent to at most agent_cap of the bucket (its oldest seats go first).
New games are packed once into broad/selfplay/archives/sp-<stamp>.zip as <eid>.json (the league episode id; info.EpisodeId).
  select(cfg) -> {bucket: [row dicts]}, report     (called by build_buckets.py)"""
import glob, gzip, json, os, re, zipfile, datetime as dt
LR = '/mnt/ai/scratch/farmNEAT/league/records'
SPD = '/mnt/ai/scratch/farmNEAT/broad/selfplay'


def eval_seeds():
    out = set()
    for f in glob.glob(f'{LR}/banks/*.json'):
        if os.path.basename(f).upper().startswith('SP-'): continue
        try: d = json.load(open(f))
        except Exception: continue
        s = d.get('seeds') if isinstance(d, dict) else None
        if isinstance(s, list): out |= {int(x) for x in s if isinstance(x, int) or str(x).isdigit()}
    return out


def games(tags):
    """Game dirs of stream directories whose name carries one of the tags (e.g. zsp-best, zsp-nfbe): league/records/<lane>/<stream>/<k>/"""
    for sdir in glob.glob(f'{LR}/*/*/'):
        name = os.path.basename(sdir.rstrip('/'))
        if not any(t in name for t in tags): continue
        for gdir in glob.glob(f'{sdir}*/'):
            gj = os.path.join(gdir, 'game.json'); rp = os.path.join(gdir, 'replay.json.gz')
            if os.path.exists(gj) and os.path.exists(rp): yield gdir.rstrip('/'), gj, rp


def select(cfg):
    os.makedirs(f'{SPD}/archives', exist_ok=True)
    idx_p = f'{SPD}/INDEX.json'
    idx = json.load(open(idx_p)) if os.path.exists(idx_p) else {}
    bank = eval_seeds()
    out, rep, new = {}, {}, []
    for b, c in cfg.items():
        cand, refused = [], {'bank_world': 0, 'not_done': 0, 'draw': 0, 'bad': 0}
        for gdir, gj, rp in games(c['tags']):
            try: g = json.load(open(gj)); gm = g['game']
            except Exception: refused['bad'] += 1; continue
            if gm.get('status') != ['DONE', 'DONE'] or gm.get('failed_step') is not None: refused['not_done'] += 1; continue
            if int(gm.get('world_seed', g.get('world_seed', -1))) in bank: refused['bank_world'] += 1; continue
            sc = gm.get('score') or []
            if len(sc) != 2 or sc[0] == sc[1]: refused['draw'] += 1; continue
            ws = 0 if sc[0] > sc[1] else 1
            eid = int(g['episode_id'])
            cand.append({'gdir': gdir, 'rp': rp, 'eid': eid, 'seat': ws, 'agent': gm['seats'][ws], 'opp': gm['seats'][1 - ws],
                         'world_seed': int(gm['world_seed']), 't': os.path.getmtime(gj)})
        cand.sort(key=lambda r: (r['t'], r['eid']), reverse=True)
        keep = cand[:int(c['keep'])]
        while keep:                                   # per-agent cap: trim the most over-represented agent's oldest seat
            cnt = {}
            for r in keep: cnt[r['agent']] = cnt.get(r['agent'], 0) + 1
            top = max(cnt, key=cnt.get)
            if cnt[top] <= float(c['agent_cap']) * len(keep): break
            for i in range(len(keep) - 1, -1, -1):
                if keep[i]['agent'] == top: keep.pop(i); break
        for r in keep:
            if r['gdir'] not in idx: new.append(r)
        out[b] = keep
        cnt = {}
        for r in keep: cnt[r['agent']] = cnt.get(r['agent'], 0) + 1
        rep[b] = {'candidates': len(cand), 'kept': len(keep), 'refused': refused, 'by_agent': dict(sorted(cnt.items(), key=lambda kv: -kv[1])),
                  'tags': c['tags'], 'keep': c['keep'], 'agent_cap': c['agent_cap']}
    if new:
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')
        zp = f'{SPD}/archives/sp-{stamp}.zip'
        with zipfile.ZipFile(zp + '.tmp', 'w', zipfile.ZIP_DEFLATED) as z:
            for r in new:
                raw = gzip.decompress(open(r['rp'], 'rb').read())
                rj = json.loads(raw)
                if int((rj.get('info') or {}).get('EpisodeId', -1)) != r['eid']: raise SystemExit(f"REFUSED: {r['gdir']} replay EpisodeId != game.json episode_id")
                z.writestr(f"{r['eid']}.json", raw)
        os.replace(zp + '.tmp', zp)
        for r in new: idx[r['gdir']] = {'archive': zp, 'member': f"{r['eid']}.json", 'eid': r['eid']}
        json.dump(idx, open(idx_p + '.tmp', 'w'), indent=1); os.replace(idx_p + '.tmp', idx_p)
    rows = {b: [{'archive': idx[r['gdir']]['archive'], 'member': idx[r['gdir']]['member'], 'day': '2026-09-30', 'episode_id': r['eid'],
                 'seat': r['seat'], 'team': r['agent'], 'world_seed': r['world_seed']} for r in keep] for b, keep in out.items()}
    return rows, rep


if __name__ == '__main__':
    cfg = json.load(open('/mnt/ai/scratch/farmNEAT/broad/BUCKET-RECIPE.json')).get('selfplay_buckets') or {
        'S_RAND': {'tags': ['zsp-rand', 'zsp-pool'], 'keep': 1000, 'agent_cap': .30}, 'S_BEST': {'tags': ['zsp-best', 'zsp-nfbe'], 'keep': 3000, 'agent_cap': .50}}
    rows, rep = select({k: v for k, v in cfg.items() if isinstance(v, dict) and 'tags' in v})
    print(json.dumps(rep, indent=1))
