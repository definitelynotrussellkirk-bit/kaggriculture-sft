"""B-even (owner 09-29 22:20): B31 EMA with its 36 encoder heads (dim 12) regrouped to 12 (dim 36) or 18 (dim 24): same weights,
different head split. Agreement with unmodified B31 on held-out frozen-final validation steps (fp32, CPU, eval, B's PBC env).
  python beven_check.py OUT.json"""
import json, os, sys, time
TR = os.environ['OA3_TRAINER']; sys.path[:0] = [TR, '/mnt/ai/scratch/farmNEAT/oa3-astra/tools-cont']
import torch
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
import step_cache as SC, model as M
from oa3.build import load_checkpoint
from oa3.training import forward
from oa3.cache_train import device_batch
from oa3.losses import masked_kl
B31 = '/mnt/ai/scratch/farmNEAT/base-v1/runs/b/chunks/chunk9-ckpt-0031047.pt'
t0 = time.time(); NEV = int(os.environ.get('NEVAL', 10))
base, rec = load_checkpoint(B31, 'ema', TR); base.eval()
it = iter(SC.StepChunks('/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11', 16, 12, pool_seats=8, rows='/mnt/ai/scratch/farmNEAT/oa3-astra/frozen-final/validation.tsv'))
ev = []
while len(ev) < NEV:
    ch = next(it)
    if ch['step_valid'].all() and ch['loss_mask'].all(): ev.append(device_batch(ch, torch.device('cpu'), require_step_w=False))
HEADS = ('unit_logits', 'unit_qty_logits', 'order_logits', 'order_qty_logits')
def outs(net):
    with torch.no_grad(): return [{k: forward(net, b)[k].float() for k in HEADS} for b in ev]
ref = outs(base)
rep = {'b31': B31, 'encoder_heads': base.config.heads, 'dec_heads': base.config.dec_heads, 'width': base.config.width,
       'pbc_env': {k: v for k, v in os.environ.items() if k.startswith('PBC_')}, 'steps': 16 * NEV, 'variants': {}}
for H in (12, 18, 36):
    cfg = base.config.to_dict(); cfg['heads'] = H
    net = M.PolicyNet(M.ModelConfig.from_dict(cfg)); net.load_state_dict(base.state_dict(), strict=True); net.eval()
    res = outs(net); a, n, kl, chg, st = {}, {}, 0.0, 0, 0
    for b, o0, o1 in zip(ev, ref, res):
        ut, ot = b['unit_tok'].long(), b['order_tok'].long()
        m = {'unit_logits': ut >= 0, 'unit_qty_logits': (ut >= 0) & base.unit_qty_mask[ut.clamp_min(0)] & (b['unit_qty'] >= 0),
             'order_logits': ot >= 0, 'order_qty_logits': (ot >= 0) & base.order_qty_mask[ot.clamp_min(0)] & (b['order_qty'] >= 0)}
        d = torch.zeros(ut.shape[0], dtype=torch.bool)
        for k in HEADS:
            same = (o0[k].argmax(-1) == o1[k].argmax(-1)) & m[k]; a[k] = a.get(k, 0) + int(same.sum()); n[k] = n.get(k, 0) + int(m[k].sum())
            d |= ((o0[k].argmax(-1) != o1[k].argmax(-1)) & m[k]).any(-1); kl += float(masked_kl(o1[k], o0[k], m[k], 1.0).sum())
        chg += int(d.sum()); st += d.numel()
    rep['variants'][str(H)] = {'head_dim': base.config.width // H, 'whole_step_change': chg / st, 'kl_b31_variant_per_step': kl / st,
                               **{k: a[k] / max(1, n[k]) for k in a}}
    print(H, json.dumps(rep['variants'][str(H)]), flush=True)
rep['seconds'] = time.time() - t0
json.dump(rep, open(sys.argv[1], 'w'), indent=1)
