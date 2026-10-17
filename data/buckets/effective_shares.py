"""Bucket probabilities a bucket-mode sampler must use (owner 04:39): the generation's EFFECTIVE-SHARES.json (data-ops) when present;
otherwise the same cap computed here from the recipe and the bucket sizes: no seat may be drawn more than CAP (4) x as often as an A seat,
the excess goes to A (iterated to a fixed point). Prints {"shares": {...}, "source": ...}.
  python effective_shares.py BUCKET_DIR [ROSTER_TSV]   (ROSTER: count seats per bucket from its reason column instead of the bucket TSVs)"""
import csv, json, os, sys
bd = sys.argv[1]; CAP = 4.0
p = os.path.join(bd, 'EFFECTIVE-SHARES.json')
if os.path.exists(p):
    d = json.load(open(p)); s = d.get('shares') or d.get('effective_shares') or d   # bucket names are the generation's (recipe v4: A, BC, D, E)
    s = {k: float(v) for k, v in s.items() if isinstance(v, (int, float)) and os.path.exists(os.path.join(bd, f'{k}.tsv'))}
    print(json.dumps({'shares': s, 'source': p})); sys.exit(0)
rc = json.load(open(os.path.join(bd, 'RECEIPT.json')))
r = rc.get('recipe') or {}; r = r.get('shares') or r
rec = {k: float(v) for k, v in r.items() if isinstance(v, (int, float)) and os.path.exists(os.path.join(bd, f'{k}.tsv'))}
if len(sys.argv) > 2:
    n = {}
    for row in csv.DictReader(open(sys.argv[2]), delimiter='\t'):
        b = row['reason'].split(':')[-1]; n[b] = n.get(b, 0) + 1
else:
    n = {b: sum(1 for _ in open(os.path.join(bd, f'{b}.tsv'))) - 1 for b in rec if os.path.exists(os.path.join(bd, f'{b}.tsv'))}
live = {b: rec[b] for b in rec if n.get(b, 0) > 0}; t = sum(live.values()); s = {b: v / t for b, v in live.items()}
for _ in range(100):
    ra = s['A'] / n['A']; moved = 0.0
    for b in s:
        if b != 'A' and s[b] / n[b] > CAP * ra * (1 + 1e-9):
            new = CAP * ra * n[b]; moved += s[b] - new; s[b] = new
    s['A'] += moved
    if moved < 1e-12:
        break
print(json.dumps({'shares': {b: round(v, 6) for b, v in s.items()}, 'source': f'computed cap {CAP}x A per-seat rate from recipe + sizes {n}'}))
