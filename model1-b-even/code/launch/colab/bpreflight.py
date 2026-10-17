"""Colab B relaunch preflight (PRO COLAB-B-TRAINER #2/#1/#4), run ON the VM before the trainer is stopped. Exit 0 only if all pass.
  python3 bpreflight.py CKPT ROWS CACHE_DIR CODE_DIR OUT_DIR
Checks: (a) checkpoint: model, main EMA and every extra EMA have the identical key set, matching shapes, all finite; optimizer groups
cover exactly the model's parameters named in optimizer_param_names (no missing/extra/duplicate names, state count <= params);
(b) a {name, size, sha256} manifest of every .stp the rows file selects, rehashed from the bytes on the VM (FETCHED-v11.txt not trusted);
(c) one launch receipt tying checkpoint sha, rows sha, cache-manifest sha and code sha; it records that a resume does not restore the
step-cache position (PRO #1), so this is not an exact resume."""
import csv, hashlib, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import torch

ck, rows, cache, code, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
fails = []
def h(p, n=1 << 22):
    d = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(n), b''): d.update(c)
    return d.hexdigest()

# (a) checkpoint
b = torch.load(ck, map_location='cpu', weights_only=False)
model = {k: v for k, v in b['model'].items() if torch.is_tensor(v)}
sets = {'ema': b.get('ema') or {}}
for k, v in (b.get('ema_extra') or {}).items(): sets[f'ema_extra[{k}]'] = v or {}
for name, e in sets.items():
    e = {k: v for k, v in e.items() if torch.is_tensor(v)}
    if set(e) != set(model): fails.append(f'{name}: key set differs from model (missing {len(set(model) - set(e))}, extra {len(set(e) - set(model))})')
    bad_shape = [k for k in set(e) & set(model) if tuple(e[k].shape) != tuple(model[k].shape)]
    if bad_shape: fails.append(f'{name}: {len(bad_shape)} shape mismatches, e.g. {bad_shape[:3]}')
    nonfinite = [k for k, v in e.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
    if nonfinite: fails.append(f'{name}: non-finite tensors {nonfinite[:3]}')
if not sets['ema']: fails.append('main EMA missing')
nf = [k for k, v in model.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
if nf: fails.append(f'model: non-finite tensors {nf[:3]}')
pn = b.get('optimizer_param_names'); opt = (b.get('optimizer') or {}).get('multi')
names = [n for grp in (pn or []) for n in grp]
params = {k for k, v in b['model'].items() if torch.is_tensor(v) and v.is_floating_point()}
if not pn or not opt: fails.append('optimizer or optimizer_param_names missing')
else:
    if len(names) != len(set(names)): fails.append('optimizer: duplicate parameter names across groups')
    miss = set(names) - set(model)
    if miss: fails.append(f'optimizer: {len(miss)} group names not in the model, e.g. {sorted(miss)[:3]}')
    groups = [len(g['params']) for o in opt for g in o['param_groups']]
    if sum(groups) != len(names) or groups != [len(x) for x in pn]: fails.append(f'optimizer: group sizes {groups} != names {[len(x) for x in pn]}')
    states = sum(len(o['state']) for o in opt)
    if states > len(names): fails.append(f'optimizer: {states} states for {len(names)} params')
ck_sha = h(ck)

# (b) manifest of the selected .stp, rehashed from bytes
with open(rows) as f:
    sel = sorted({f"{r['eid']}-{r['seat']}.stp" for r in csv.DictReader(f, delimiter='\t') if int(r.get('weight') or 1) > 0})
def one(n):
    p = os.path.join(cache, n)
    if not os.path.isfile(p) or os.path.getsize(p) == 0: return {'name': n, 'size': None, 'sha256': None}
    return {'name': n, 'size': os.path.getsize(p), 'sha256': h(p)}
t0 = time.time()
with ThreadPoolExecutor(int(os.environ.get("BPF_THREADS", "8"))) as ex: man = list(ex.map(one, sel))
missing = [m['name'] for m in man if m['sha256'] is None]
if missing: fails.append(f'cache: {len(missing)} selected seat files missing/empty, e.g. {missing[:3]}')
mb = ''.join(f"{m['sha256']}  {m['size']}  {m['name']}\n" for m in man).encode()
open(os.path.join(out, 'CACHE-MANIFEST.txt'), 'wb').write(mb)
man_sha = hashlib.sha256(mb).hexdigest()

# (c) receipt
code_files = {}
for root, _, fs in os.walk(code):
    for fn in fs:
        if fn.endswith('.py'):
            p = os.path.join(root, fn); code_files[os.path.relpath(p, code)] = h(p)
code_sha = hashlib.sha256(json.dumps(code_files, sort_keys=True).encode()).hexdigest()
rec = {'ok': not fails, 'fails': fails, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
       'checkpoint': ck, 'checkpoint_sha256': ck_sha, 'update': b.get('update'),
       'ema_sets': sorted(sets), 'model_tensors': len(model), 'optimizer_groups': [len(x) for x in (pn or [])],
       'rows': rows, 'rows_sha256': h(rows), 'selected_seats': len(sel), 'cache_manifest_sha256': man_sha, 'manifest_seconds': round(time.time() - t0, 1),
       'code_dir': code, 'code_sha256': code_sha, 'train_py_sha256': code_files.get('work/runs/maintainer/primitive-bc/train.py'),
       'resume_note': 'resume restores model/EMA/optimizer/RNG but NOT the step-cache position (PRO COLAB-B-TRAINER #1): not an exact resume',
       'fetched_log_trusted': False}
json.dump(rec, open(os.path.join(out, 'LAUNCH-RECEIPT.json'), 'w'), indent=1)
print(json.dumps({k: rec[k] for k in ('ok', 'fails', 'update', 'checkpoint_sha256', 'rows_sha256', 'selected_seats', 'cache_manifest_sha256', 'code_sha256', 'manifest_seconds')}))
sys.exit(0 if not fails else 1)
