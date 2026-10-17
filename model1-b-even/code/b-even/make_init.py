"""B-even init: B u31047 training checkpoint with the encoder's 36 heads regrouped to H (same tensors), model := EMA, every
ema_extra := EMA. The optimizer/RNG/stream state stay B31's. python make_init.py H OUT"""
import hashlib, json, sys, torch
H, out = int(sys.argv[1]), sys.argv[2]
src = '/mnt/ai/scratch/farmNEAT/base-v1/runs/b/ckpt-0031047.pt'
b = torch.load(src, map_location='cpu', weights_only=False)
assert b['config']['width'] % H == 0 and b['config']['heads'] == 36
b['config'] = dict(b['config'], heads=H)
b['model'] = {k: v.clone() for k, v in b['ema'].items()}
b['ema_extra'] = {d: {k: v.clone() for k, v in b['ema'].items()} for d in (b.get('ema_extra') or {})}
b['beven'] = {'source': src, 'source_sha256': hashlib.sha256(open(src, 'rb').read()).hexdigest(), 'heads': H, 'model': 'B31 EMA'}
torch.save(b, out); print(json.dumps({k: v for k, v in b['beven'].items()}), list(b['ema_extra']) if b.get('ema_extra') else None)
