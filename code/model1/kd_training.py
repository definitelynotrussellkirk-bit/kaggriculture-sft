"""One-step, same-recorded-prefix BC/KD. No simulator targets or online RL."""
from __future__ import annotations
from contextlib import nullcontext
import datetime as dt
import json
import math
import os
from pathlib import Path
import time
import torch
from .losses import masked_kl, set_nll
from .source import model_module

GROW = ('extra','unit_extra','reach_eta','unit_prefix','order_prefix','order_cash',
        'prev_unit','prev_order','v6_token','v6_resource_eta','v6_resource_route',
        'v6_job_deadline','v6_job_deadline_present','x8_token','x8_unit','lattice')
LABELS = ('unit_tok','unit_qty','order_tok','order_qty')
INPUTS = ('rows','token_mask','knownness','unit_pos')

def forward(net, batch: dict) -> dict:
    """A strict actor-input whitelist: future/opponent target arrays never enter encode."""
    ctx = net.encode(batch['rows'], batch['token_mask'], batch['knownness'],
                     **{k:batch[k] for k in GROW if k in batch})
    up = torch.stack((batch['unit_tok'],batch['unit_qty']),-1)
    op = torch.stack((batch['order_tok'],batch['order_qty']),-1)
    return net.decode_teacher(ctx,batch['unit_pos'],up,op)

def _nll(z, y, valid):
    # The one-hot is small for these typed vocabularies, and uses the SAME finite
    # support validation as the optional class loss. No -inf NaNs in padding.
    allowed = torch.zeros_like(z,dtype=torch.bool).scatter_(-1,y.clamp(0,z.shape[-1]-1).unsqueeze(-1),True)
    return set_nll(z,y,allowed,valid)

def objective(out: dict, teacher: dict | None, b: dict, net, *, alpha=.5, temperature=1.5,
              movement_weight=.5, order_weight=2., order_denominator=2.35,
              use_unit_mask=False) -> tuple[torch.Tensor,dict]:
    if not 0 <= alpha <= 1 or any(not math.isfinite(v) or v <= 0 for v in (temperature,movement_weight,order_weight,order_denominator)):
        raise ValueError('Invalid loss coefficients')
    if alpha and teacher is None:
        raise ValueError('KD coefficient is nonzero but no teacher output was supplied')
    ut,uq,ot,oq=(b[k].long() for k in LABELS)
    present=ut>=0
    for y,n in ((ut,net.config.n_unit_tokens),(ot,net.config.n_order_tokens)):
        if ((y < -1) | (y >= n)).any():
            raise ValueError('Recorded command outside vocabulary')
    if use_unit_mask:
        keep=b.get('unit_loss_keep')
        if keep is None or keep.dtype != torch.bool or keep.shape != ut.shape:
            raise ValueError('Explicit noop masking needs a matching, schema-checked boolean mask')
        present=present & keep
    masks=(present,present & net.unit_qty_mask[ut.clamp_min(0)] & (uq>=0),
           ot>=0,(ot>=0) & net.order_qty_mask[ot.clamp_min(0)] & (oq>=0))
    keys=('unit_logits','unit_qty_logits','order_logits','order_qty_logits')
    bc=[_nll(out[k],y,mask) for k,y,mask in zip(keys,(ut,uq,ot,oq),masks)]
    kd=[masked_kl(out[k],teacher[k],mask,temperature) if teacher is not None else torch.zeros_like(x)
        for k,mask,x in zip(keys,masks,bc)]
    unit_den=present.sum(-1).clamp_min(1)
    move=torch.where((ut>=1)&(ut<=4),movement_weight,1.)
    def weight(x):
        return ((x[0]*move).sum(-1)+x[1].sum(-1))/unit_den + order_weight*(x[2].sum(-1)+x[3].sum(-1))/order_denominator
    bcw,kdw=weight(bc),weight(kd)
    sw=b.get('step_w')
    if sw is not None:
        sw=sw.to(bcw.dtype); bcw,kdw=bcw*sw,kdw*sw
    loss=((1-alpha)*bcw+alpha*kdw).mean()
    # All metrics are synchronized scalars intentionally. This conservative path
    # prioritizes audited correctness; use measured end-to-end throughput.
    metrics={'bc':float(bcw.detach().mean()),'kd':float(kdw.detach().mean()),
        'objective':float(loss.detach()),
        'canonical_bits_per_step':float(sum(x.detach().sum(-1) for x in bc).mean()/math.log(2)),
        'steps':len(ut)}
    return loss,metrics

class Deadline:
    """Cooperative stop checked before batches/updates, not an OS process kill.

    A running kernel, cache read, or checkpoint write may finish after the bound.
    Use an external job supervisor for a strict allocation deadline.
    """
    def __init__(self, stop_at: str, max_seconds: float):
        parsed=dt.datetime.fromisoformat(stop_at.replace('Z','+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError('stop_at must include an explicit UTC offset')
        remaining=(parsed-dt.datetime.now(dt.timezone.utc)).total_seconds()
        if remaining <= 0 or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('Deadline is past or max_seconds is invalid')
        self.started=time.monotonic();self.limit=min(remaining,max_seconds)
        self.stop_at=parsed.isoformat()
    def remaining(self):
        return max(0.,self.limit-(time.monotonic()-self.started))
    def expired(self):
        return self.remaining() <= 0

def save_training_checkpoint(path, net, optimizer, ema, update: int, receipt: dict):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'schema':net.config.schema,'config':net.config.to_dict(),
        'model':{k:v.detach().cpu() for k,v in net.state_dict().items()},
        'ema':{k:v.detach().cpu() for k,v in ema.items()},
        'optimizer':optimizer.state_dict(),'update':update,'receipt':receipt,
        'torch_rng':torch.get_rng_state(),
        'resume_note':'Warm restart possible; exact data-iterator replay is not provided.'}
    temp=path.with_name(path.name+'.tmp')
    torch.save(payload,temp);os.replace(temp,path)

def synthetic_batch(net, *, batch=2, n_units=3, seed=0, device='cpu') -> dict:
    """Shapes exercise real model code. Feature VALUES ARE SYNTHETIC, not game observations."""
    m=model_module();c=net.config
    rows,mask,known,pos=m.synthetic_inputs(n_tokens=250,n_units=n_units,max_units=net.U,batch=batch,seed=seed)
    B,N=rows.shape[:2];U,O=net.U,net.O
    # Upstream synthetic_inputs leaves all carrier ordinals at zero. Repair the
    # fixture (not production data) so multi-worker carrier_read is meaningful.
    for i in range(B):
        indices=((rows[i,:,3]>.5)&(rows[i,:,12]>.5)).nonzero().flatten()
        rows[i,indices,16]=torch.arange(len(indices),dtype=rows.dtype)/100
    b={'rows':rows,'token_mask':mask,'knownness':known,'unit_pos':pos}
    b['unit_tok']=torch.full((B,U),-1,dtype=torch.long);b['unit_tok'][:,:n_units]=0
    b['unit_qty']=torch.full((B,U),-1,dtype=torch.long)
    # Include a quantity-bearing worker and an active market order plus STOP.
    import codec
    b['unit_tok'][:,0]=codec.UNIT_VOCAB.index('PICKUP:WHEAT');b['unit_qty'][:,0]=6
    b['order_tok']=torch.full((B,O),-1,dtype=torch.long)
    b['order_tok'][:,0]=codec.ORDER_VOCAB.index('SELL:WHEAT');b['order_tok'][:,1]=c.stop_index
    b['order_qty']=torch.full((B,O),-1,dtype=torch.long);b['order_qty'][:,0]=6
    b['unit_loss_keep']=pos[...,0]>=0
    if c.extra_width:
        b['extra']=torch.zeros(B,N,c.extra_width+2+c.calc_width)
    if c.unit_extra_width:b['unit_extra']=torch.zeros(B,U,c.unit_extra_width)
    if c.reach_job:b['reach_eta']=torch.zeros(B,U,100,dtype=torch.long)
    for k,w,L in [('unit_prefix',c.prefix_unit_width,U),('order_prefix',c.prefix_order_width,O),('order_cash',c.prefix_cash_width,O)]:
        if w:b[k]=torch.zeros(B,L,w)
    if c.v6_prev:
        b['prev_unit']=torch.full((B,2,U,2),-1,dtype=torch.long)
        b['prev_order']=torch.full((B,2,O,2),-1,dtype=torch.long)
    if c.job_eta_v6:
        b['v6_token']=torch.zeros(B,N,8)
        b['v6_resource_eta']=torch.zeros(B,U,100,2,dtype=torch.long)
        b['v6_resource_route']=torch.zeros(B,U,100,2,dtype=torch.bool)
        b['v6_job_deadline']=torch.zeros(B,100,9,dtype=torch.long)
        b['v6_job_deadline_present']=torch.zeros(B,100,9,dtype=torch.bool)
        b['extra'][:,:,96:105]=1. # exercise nonempty job-bias arithmetic
    if c.x8_token_width:b['x8_token']=torch.zeros(B,N,c.x8_token_width)
    if c.x8_unit_width:b['x8_unit']=torch.zeros(B,U,c.x8_unit_width)
    if c.lattice_channels:b['lattice']=torch.zeros(B,2,10,10,c.lattice_channels)
    return {k:v.to(device) for k,v in b.items()}
