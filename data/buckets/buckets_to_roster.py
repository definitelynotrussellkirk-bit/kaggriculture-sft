"""Exact emulation of the bucket sampler for step-cache trainers (owner 02:5x: "each batch just randomly samples from a bucket"; coordinator:
emulate a weighted roster with weight = bucket_share / bucket_size). The step cache loads a row w times per round (integer w), so each seat
gets w_i = round(share_b / size_b / min_b(share_b / size_b)) (>= 1), i.e. uniform within its bucket and bucket mass ~ share_b; empty buckets
are dropped and the rest renormalised. Receipt = requested vs effective shares (rounding error), consumed-file hashes, generation receipt
and EFFECTIVE-SHARES hashes, recipe hash.
  python buckets_to_roster.py BUCKET_DIR OUT_DIR      (BUCKET_DIR = broad/buckets/<sha12>: RECEIPT.json, EFFECTIVE-SHARES.json, <bucket>.tsv)
Fail closed (EASY-CHECK-26): the generation proof (receipt, EFFECTIVE-SHARES bound to the receipt, hash of exactly the consumed files,
12-hex directory = hash prefix) is required even when B2R_SHARES overrides the shares; missing/blocked/zero-byte cache seats and duplicate
seats refuse with counts; nothing is written unless every check passes."""
import hashlib, json, math, os, re, sys
bd, out = sys.argv[1].rstrip('/'), sys.argv[2]
C = '/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11'
RECIPE = '/mnt/ai/scratch/farmNEAT/broad/BUCKET-RECIPE.json'
BLOCKED_SUBS = {56680736}
h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
name = os.path.basename(bd)
if not re.fullmatch(r'[0-9a-f]{12}', name): sys.exit(f'REFUSED: {bd} is not a 12-hex generation directory')
rc = [f for f in ('RECEIPT.json', f'RECEIPT-{name}.json') if os.path.exists(os.path.join(bd, f))]
if len(rc) != 1: sys.exit(f'REFUSED: need exactly one generation receipt in {bd}, found {rc}')
rc_p = os.path.join(bd, rc[0]); man = json.load(open(rc_p))
es_p = os.path.join(bd, 'EFFECTIVE-SHARES.json')
if not os.path.exists(es_p): sys.exit(f'REFUSED: {bd} has no EFFECTIVE-SHARES.json (capped shares required, owner 04:39)')
es = json.load(open(es_p))
eff_gen = es.get('effective_shares')
if not isinstance(eff_gen, dict) or not eff_gen: sys.exit('REFUSED: EFFECTIVE-SHARES.json has no effective_shares')
LIN_VIEW = None
LIN = os.environ.get('B2R_LINEAGE') or None   # W_<lineage> buckets are consumed only by their own lineage's roster
if LIN:
    lv = (es.get('lineage_views') or {}).get(LIN); lr = (man.get('lineage_views') or {}).get(LIN)
    if not lv and not lr and not (json.loads(man.get('identity_json') or '{}').get('own_win_buckets')):
        LIN_VIEW = 'neutral'   # D-1613own: no own-win buckets in this generation -> every lineage uses the neutral shares
    else:
        if not lv or not lr: sys.exit(f'REFUSED: generation {name} has no lineage view for {LIN} (views: {sorted(es.get("lineage_views") or {})})')
        if lv.get('effective_shares') != lr.get('effective_shares'): sys.exit(f'REFUSED: lineage view {LIN} differs between EFFECTIVE-SHARES and the receipt')
        eff_gen = lv['effective_shares']
rs = man.get('effective_shares') if (not LIN or LIN_VIEW == 'neutral') else man['lineage_views'][LIN]['effective_shares']
if not isinstance(rs, dict) or set(rs) != set(eff_gen) or any(abs(float(rs[k]) - float(eff_gen[k])) > 1e-9 for k in rs):
    sys.exit(f'REFUSED: EFFECTIVE-SHARES.json {eff_gen} differs from the generation receipt {rs}')
listed = es.get('files') or [f'{k}.tsv' for k in eff_gen]   # hashed files, in generation order
if not isinstance(listed, list) or len(set(listed)) != len(listed): sys.exit(f'REFUSED: EFFECTIVE-SHARES files invalid: {listed}')
unlisted = [f'{k}.tsv' for k in eff_gen if f'{k}.tsv' not in listed]
if unlisted: sys.exit(f'REFUSED: consumed bucket files {unlisted} are not among the hashed files {listed}')
miss = [f for f in listed if not os.path.exists(os.path.join(bd, f))]
if miss: sys.exit(f'REFUSED: bucket files missing in {bd}: {miss}')
rawf = {f: open(os.path.join(bd, f), 'rb').read() for f in listed}
extra = [f for f in listed if f[:-4] not in eff_gen and not f.startswith(('W_', 'N_')) and len([l for l in rawf[f].splitlines() if l.strip()]) > 1]
if extra: sys.exit(f'REFUSED: hashed files {extra} carry seats but have no effective share')
raw = {k: rawf[f'{k}.tsv'] for k in eff_gen}
body_b = b''.join(rawf[f] for f in listed)
gh = hashlib.sha256((man['identity_json'].encode() + b'\n' + body_b) if isinstance(man.get('identity_json'), str) else body_b).hexdigest()
if gh != man.get('sha256') or gh[:12] != name:
    sys.exit(f'REFUSED: generation hash mismatch in {bd}: files {gh[:12]} receipt {str(man.get("sha256"))[:12]} dir {name}')
# ASTRA-5 #5: a generation built under an older recipe policy is refused (rebuild, do not relabel)
if not os.path.exists(RECIPE): sys.exit(f'REFUSED: current recipe missing: {RECIPE}')
if not isinstance(man.get('identity_json'), str): sys.exit(f'REFUSED: {bd} has no identity_json (pre-policy generation)')
_pol = json.loads(man['identity_json']); _cur = json.load(open(RECIPE))
_om = set(man.get('omitted_zero_share_buckets') or [])   # buckets the build omitted (no publishable seats) are not a policy change
for _k in ('shares', 'dynamic_shares'):
    if isinstance(_cur.get(_k), dict): _cur[_k] = {a: b for a, b in _cur[_k].items() if a not in _om or a in (_pol.get(_k) or {})}
_POLICY_KEYS = ('schema', 'version', 'shares', 'merge', 'small_bucket_cap', 'beat_us_days', 'beat_us_bucket', 'dynamic_shares', 'reserved', 'sampling', 'a_top')
_chg = [k for k in dict.fromkeys(list(_POLICY_KEYS) + list(_pol)) if _pol.get(k) != _cur.get(k)]
if _chg: sys.exit(f'REFUSED: stale generation policy in {bd}: recipe changed {_chg} since the build; use a newer generation')
shares = {k: float(v) for k, v in eff_gen.items()}; shares_source = es_p
if os.environ.get('B2R_SHARES'):   # override replaces shares only; the generation proof above still applies
    try: ov = json.loads(os.environ['B2R_SHARES'])
    except ValueError: sys.exit('REFUSED: B2R_SHARES is not JSON')
    if not isinstance(ov, dict) or not ov: sys.exit('REFUSED: B2R_SHARES must be a non-empty mapping')
    try: ov = {k: float(v) for k, v in ov.items()}
    except (TypeError, ValueError): sys.exit(f'REFUSED: B2R_SHARES has a non-numeric share: {ov}')
    if set(ov) - set(eff_gen): sys.exit(f'REFUSED: B2R_SHARES names buckets outside the generation: {sorted(set(ov) - set(eff_gen))}')
    if any(not math.isfinite(v) or v < 0 for v in ov.values()) or sum(ov.values()) <= 0: sys.exit(f'REFUSED: B2R_SHARES must be finite, >= 0, positive total: {ov}')
    shares = ov; shares_source = os.environ.get('B2R_SHARES_SOURCE', 'B2R_SHARES')
import pyarrow.parquet as _pq   # quarantined submissions + static tapes are not admissible BC seats
_t = _pq.read_table('/mnt/ai/scratch/farmNEAT/primitive-bc-data/catalog/SEATS.parquet', columns=['episode_id', 'seat', 'submission_id', 'static'])
_blk = {(int(e), int(z)) for e, z, u, st in zip(*(_t[c].to_pylist() for c in ['episode_id', 'seat', 'submission_id', 'static']))
        if (u is not None and u == u and int(u) in BLOCKED_SUBS) or bool(st)}
seats, fmeta, bad, seen_all, dup = {}, {}, {'missing_cache': [], 'blocked_or_static': [], 'duplicate': []}, set(), 0
for k in shares:
    L = raw[k].decode().splitlines(); hd = L[0].split('\t') if L else []
    if 'seat' not in hd or not ({'eid', 'episode_id'} & set(hd)): sys.exit(f'REFUSED: {k}.tsv header lacks eid/seat')
    ie = hd.index('eid') if 'eid' in hd else hd.index('episode_id'); iz = hd.index('seat')
    ia = {c: hd.index(c) for c in ('archive', 'member', 'day') if c in hd}
    rows = []
    for l in L[1:]:
        if not l.strip(): continue
        f = l.split('\t'); e, s = int(f[ie]), int(f[iz]); p = f'{C}/{e}-{s}.stp'
        if (e, s) in seen_all: bad['duplicate'].append(f'{k}:{e}-{s}'); continue
        seen_all.add((e, s))
        if not os.path.exists(p) or os.path.getsize(p) == 0: bad['missing_cache'].append(f'{k}:{e}-{s}'); continue
        if (e, s) in _blk: bad['blocked_or_static'].append(f'{k}:{e}-{s}'); continue
        rows.append((f[ia['archive']] if 'archive' in ia else '-', f[ia['member']] if 'member' in ia else f'{e}.json', f[ia['day']] if 'day' in ia else '-', e, s))
    seats[k] = rows; fmeta[k] = {'path': os.path.join(bd, f'{k}.tsv'), 'sha256': hashlib.sha256(raw[k]).hexdigest(), 'cached_seats': len(rows)}
# ASTRA-5 #6: reservations added after the build are rechecked now (fresh TOP4 VAL / MMPQ-VAL / practice VAL / leak + the last union)
def _ids(p, allow_empty=False):
    if not os.path.exists(p): sys.exit(f'REFUSED: reservation list missing: {p}')
    ls = [l for l in open(p).read().splitlines() if l.strip() and not l.startswith('#')]
    hd = ls[0].split('\t') if ls else []; key = 'eid' if 'eid' in hd else ('episode_id' if 'episode_id' in hd else None)
    out = set()
    for l in (ls[1:] if key else ls):
        f = l.split('\t') if key else l.split(); x = (f[hd.index(key)] if key and len(f) > hd.index(key) else (f[0] if f else '')).strip()
        if not x.isdigit(): sys.exit(f'REFUSED: malformed reservation row in {p}: {l[:80]!r}')
        out.add(int(x))
    if not out and not allow_empty: sys.exit(f'REFUSED: reservation list parsed to 0 ids: {p}')
    return out
_FS = '/mnt/ai/scratch/farmNEAT'
_res = _ids(f'{_FS}/corpus-final/special/TOP4-BEATERS-VAL.eids', True) | _ids(f'{_FS}/mmpq-val/VAL-GAMES.tsv') | \
       _ids(f'{_FS}/practice-v1/val/VAL.tsv') | _ids(f'{_FS}/touchup-top8-0929/LEAK-HOLDOUT-IDS.txt')
_xl = f'{_FS}/broad/exclusions/EXCLUSIONS-LATEST'
if not os.path.exists(_xl): sys.exit(f'REFUSED: exclusion union missing: {_xl}')
_xq = set()
for l in open(_xl).read().splitlines()[1:]:
    e_, z_, _f = l.split('\t'); (_res.add(int(e_)) if z_ == '*' else _xq.add((int(e_), int(z_))))
for k in shares:
    for a_, m_, d_, e, s in seats[k]:
        if e in _res or (e, s) in _xq: bad.setdefault('reserved_after_build', []).append(f'{k}:{e}-{s}')
if any(bad.values()):
    sys.exit('REFUSED: listed seats would vanish: ' + json.dumps({k: {'n': len(v), 'ids': v[:20]} for k, v in bad.items()}))
live = {b: shares[b] for b in shares if seats[b] and shares[b] > 0}
if not live: sys.exit('REFUSED: every bucket is empty')
tot = sum(live.values()); req = {b: live[b] / tot for b in live}
per = {b: req[b] / len(seats[b]) for b in live}; unit = min(per.values())
ok = False
for k in range(1, 101):   # smallest base weight k such that every effective share is within .005 of the request (integer repeats)
    w = {b: max(1, round(k * per[b] / unit)) for b in live}
    eff_mass = {b: w[b] * len(seats[b]) for b in live}; T = sum(eff_mass.values()); eff = {b: round(eff_mass[b] / T, 4) for b in live}
    if max(abs(eff[b] - req[b]) for b in live) <= 0.005: ok = True; break
if not ok: sys.exit(f'REFUSED: integer weights up to k=100 miss the requested shares by > .005: requested {req}, got {eff}')
body = ['archive\tmember\tday\teid\tseat\tweight\treason']
for b in live:
    for a, m, d, e, s in seats[b]: body.append(f'{a}\t{m}\t{d}\t{e}\t{s}\t{w[b]}\tbucket:{b}')
text = '\n'.join(body) + '\n'; sha = hashlib.sha256(text.encode()).hexdigest()
os.makedirs(out, exist_ok=True)
path = f'{out}/rows-buckets.{sha[:12]}.tsv'; open(path + '.tmp', 'w').write(text); os.replace(path + '.tmp', path)
rec = {'roster': path, 'sha256': sha, 'bucket_dir': bd, 'lineage': LIN, 'generation_sha256': man.get('sha256'), 'generation_receipt': {'path': rc_p, 'sha256': h(rc_p)},
       'effective_shares_file': {'path': es_p, 'sha256': h(es_p), 'effective_shares': eff_gen}, 'bucket_files': fmeta,
       'recipe_sha256': h(RECIPE) if os.path.exists(RECIPE) else None,
       'requested_share_renormalised': {b: round(v, 4) for b, v in req.items()}, 'empty_buckets_dropped': [b for b in shares if b not in live],
       'seats': {b: len(seats[b]) for b in shares}, 'dropped': {k: len(v) for k, v in bad.items()}, 'integer_weight': w, 'effective_share': eff,
       'max_abs_share_error': round(max(abs(eff[b] - req[b]) for b in live), 4), 'shares_source': shares_source, 'shares_in': shares,
       'per_seat_weights': 'none (owner: uniform within bucket)'}
rp = f'{out}/BUCKETS-ROSTER-{sha[:12]}.json'; json.dump(rec, open(rp + '.tmp', 'w'), indent=1); os.replace(rp + '.tmp', rp); print(json.dumps(rec))
