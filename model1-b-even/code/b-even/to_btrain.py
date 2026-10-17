"""Phase-1 (oa3 runner, KD) last checkpoint -> B trainer checkpoint: the B-even init blob with model := runner model, ema and every
ema_extra := runner EMA, update := 31000 (so N u = trainer u - 30000). python to_btrain.py RUNNER_CKPT INIT OUT"""
import json, sys, torch
r = torch.load(sys.argv[1], map_location='cpu', weights_only=False); b = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
assert set(r['model']) == set(b['model']), 'state keys differ'
for k, v in r['model'].items(): assert v.shape == b['model'][k].shape, k
if 'config' in r: assert r['config']['heads'] == b['config']['heads'], 'heads differ'
b['model'] = {k: v.detach().clone().to(b['model'][k].dtype) for k, v in r['model'].items()}
b['ema'] = {k: v.detach().clone().to(b['ema'][k].dtype) for k, v in r['ema'].items()}
b['ema_extra'] = {d: {k: v.clone() for k, v in b['ema'].items()} for d in (b.get('ema_extra') or {})}
b['update'] = 31000; b['beven_phase1'] = {'runner_checkpoint': sys.argv[1], 'runner_update': r.get('update')}
torch.save(b, sys.argv[3]); print(json.dumps(b['beven_phase1']))
