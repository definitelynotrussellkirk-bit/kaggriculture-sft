"""B-even layouts vs B31 EMA on the same 160 held-out steps as BEVEN-CHECK (frozen-final validation, StepChunks seed 12, 16 x 10),
teacher-forced (decode_teacher on recorded labels) and free-decoded (greedy decode_free; no same-turn prefix callback, identical for
every net). Variants: as-built 18x24 (adjacent heads, no rescale), QK-clustered 18x24 and 12x36 (PRO N-ARCH R2 head_surgery: balanced
QK-operator-similarity groups, function-preserving permutation, then regroup; qk 'mean-logit' factor g^-1/4), clustered 18x24 without
rescale, and job 4308's phase-1 end EMA (u1000).   python beven_check2.py OUT.json"""
import json, math, os, sys, time
TR = os.environ['OA3_TRAINER']; sys.path[:0] = [TR, '/mnt/ai/scratch/farmNEAT/oa3-astra/tools-cont',
    '/mnt/ai/scratch/farmNEAT/b-even']
import torch
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
import step_cache as SC, model as M
from oa3.build import load_checkpoint
from oa3.training import forward, GROW
from oa3.cache_train import device_batch
from head_surgery import balanced_similarity_groups, grouping_objective, groups_to_permutation
B31 = '/mnt/ai/scratch/farmNEAT/base-v1/runs/b/chunks/chunk9-ckpt-0031047.pt'; U1000 = '/mnt/ai/scratch/farmNEAT/b-even/p2-seed.pt'
t0 = time.time(); NEV = int(os.environ.get('NEVAL', 10))
base, _ = load_checkpoint(B31, 'ema', TR); base.eval()
it = iter(SC.StepChunks('/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11', 16, 12, pool_seats=8, rows='/mnt/ai/scratch/farmNEAT/oa3-astra/frozen-final/validation.tsv'))
ev = []
while len(ev) < NEV:
    ch = next(it)
    if ch['step_valid'].all() and ch['loss_mask'].all(): ev.append(device_batch(ch, torch.device('cpu'), require_step_w=False))
H = ('unit_logits', 'unit_qty_logits', 'order_logits', 'order_qty_logits'); A = ('unit_tok', 'unit_qty', 'order_tok', 'order_qty')

def qk_sim(blk):
    h, dh, w = blk.h, blk.dh, blk.qkv.weight.shape[1]; W = blk.qkv.weight.detach().float()
    q, k = W[:w].reshape(h, dh, w), W[w:2 * w].reshape(h, dh, w)
    v = (torch.einsum('hdi,hdj->hij', q, k) / math.sqrt(dh)).flatten(1); u = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    s = u @ u.T; s.fill_diagonal_(1.0); return s

def permute_(blk, perm):
    h, dh, w = blk.h, blk.dh, blk.qkv.weight.shape[1]
    with torch.no_grad():
        blk.qkv.weight.copy_(blk.qkv.weight.reshape(3, h, dh, w)[:, perm].reshape(3 * w, w))
        blk.qkv.bias.copy_(blk.qkv.bias.reshape(3, h, dh)[:, perm].reshape(3 * w))
        blk.proj.weight.copy_(blk.proj.weight.reshape(w, h, dh)[:, perm].reshape(w, w))

def variant(heads, cluster, scale):
    cfg = base.config.to_dict(); src = base.config.heads
    net = M.PolicyNet(M.ModelConfig.from_dict(dict(cfg, heads=heads))); st = {k: v.clone() for k, v in base.state_dict().items()}
    tmp = M.PolicyNet(M.ModelConfig.from_dict(cfg)); tmp.load_state_dict(st, strict=True); meta = []
    g = src // heads
    for blk in tmp.blocks:
        if cluster:
            s = qk_sim(blk); groups = balanced_similarity_groups(s, g); adj = [list(range(i, i + g)) for i in range(0, src, g)]
            permute_(blk, groups_to_permutation(groups)); meta.append((round(grouping_objective(s, adj), 4), round(grouping_objective(s, groups), 4)))
        if scale:
            with torch.no_grad():
                w = blk.qkv.weight.shape[1]; f = g ** -0.25
                blk.qkv.weight[:2 * w].mul_(f); blk.qkv.bias[:2 * w].mul_(f)
    net.load_state_dict(tmp.state_dict(), strict=True); net.eval(); return net, meta

def outs(net):
    tf, fr = [], []
    with torch.no_grad():
        for b in ev:
            o = forward(net, b); tf.append({k: o[k].float() for k in H})
            ctx = net.encode(b['rows'], b['token_mask'], b['knownness'], **{k: b[k] for k in GROW if k in b})
            d = net.decode_free(ctx, b['unit_pos'], temperature=0.0); fr.append({k: d[k] for k in A})
    return tf, fr

def compare(ref, res):
    (rt, rf), (ct, cf) = ref, res; a, n, chg, st = {}, {}, 0, 0; fa, fn, fchg = {}, {}, 0
    for b, o0, o1, f0, f1 in zip(ev, rt, ct, rf, cf):
        ut, ot = b['unit_tok'].long(), b['order_tok'].long()
        m = {'unit_logits': ut >= 0, 'unit_qty_logits': (ut >= 0) & base.unit_qty_mask[ut.clamp_min(0)] & (b['unit_qty'] >= 0),
             'order_logits': ot >= 0, 'order_qty_logits': (ot >= 0) & base.order_qty_mask[ot.clamp_min(0)] & (b['order_qty'] >= 0)}
        d = torch.zeros(ut.shape[0], dtype=torch.bool); fd = torch.zeros(ut.shape[0], dtype=torch.bool)
        for k in H:
            same = (o0[k].argmax(-1) == o1[k].argmax(-1)) & m[k]; a[k] = a.get(k, 0) + int(same.sum()); n[k] = n.get(k, 0) + int(m[k].sum())
            d |= ((o0[k].argmax(-1) != o1[k].argmax(-1)) & m[k]).any(-1)
        for k in A:
            x, y = f0[k].long(), f1[k].long(); L = min(x.shape[1], y.shape[1])
            if x.shape[1] != y.shape[1]:   # pad the shorter with -1
                pad = lambda t, L2: torch.cat([t, torch.full((t.shape[0], L2 - t.shape[1]), -1, dtype=t.dtype)], 1)
                L = max(x.shape[1], y.shape[1]); x, y = pad(x, L), pad(y, L)
            valid = (x >= 0) | (y >= 0); eq = (x == y) & valid
            fa[k] = fa.get(k, 0) + int(eq.sum()); fn[k] = fn.get(k, 0) + int(valid.sum()); fd |= ((x != y) & valid).any(-1)
        chg += int(d.sum()); fchg += int(fd.sum()); st += d.numel()
    return {'tf_whole_step_change': round(chg / st, 4), **{f'tf_{k.replace("_logits", "")}': round(a[k] / max(1, n[k]), 4) for k in H},
            'free_whole_step_change': round(fchg / st, 4), **{f'free_{k}': round(fa[k] / max(1, fn[k]), 4) for k in A}}

ref = outs(base)
rep = {'steps': 16 * NEV, 'parent': B31, 'free_decode': 'decode_free greedy, no prefix callback (same for every net)', 'variants': {}}
for name, (h, c, s) in {'asbuilt_18x24': (18, False, False), 'cluster_18x24_meanlogit': (18, True, True), 'cluster_12x36_meanlogit': (12, True, True),
                        'cluster_18x24_noscale': (18, True, False), 'asbuilt_18x24_meanlogit': (18, False, True)}.items():
    net, meta = variant(h, c, s); r = compare(ref, outs(net)); r['group_similarity_adjacent_vs_selected'] = meta[:3] if meta else None
    rep['variants'][name] = r; print(name, json.dumps(r), flush=True)
u, _ = load_checkpoint(U1000, 'ema', TR); u.eval(); r = compare(ref, outs(u)); rep['variants']['job4308_u1000_ema'] = r; print('job4308_u1000_ema', json.dumps(r), flush=True)
rep['seconds'] = round(time.time() - t0, 1); json.dump(rep, open(sys.argv[1], 'w'), indent=1)
