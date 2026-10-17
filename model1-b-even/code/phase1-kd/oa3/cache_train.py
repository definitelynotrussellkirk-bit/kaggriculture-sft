"""Bounded cache-backed experiment runner against the unchanged upstream reader.

Only locally trusted .stp cache files are supported: the upstream format is
pickle, not a safe interchange format. This module never builds or downloads data.
"""
from __future__ import annotations
from contextlib import nullcontext
import copy
import importlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
import torch
from .build import load_checkpoint
from .data_guard import verify_frozen, read_roster, validate_feature_subset
from .source import activate, sha256, source_receipt
from .training import INPUTS, GROW, LABELS, Deadline, forward, objective, save_training_checkpoint


def _step_w_chunks(sc, table_path):
    """StepChunks whose steps carry step_w from a TSV (episode_id seat first_step last_step step_multiplier); 1.0 elsewhere.
    Same sampling as upstream StepChunks (step_cache.py stays byte-identical); gpu-ops 09-29 copy of the code-arms scheme."""
    import numpy as np, random
    table = {}
    with open(table_path) as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(head, line.rstrip("\n").split("\t")))
            e = table.setdefault((int(r["episode_id"]), int(r["seat"])), np.ones(720, dtype=np.float32))
            e[int(r["first_step"]):int(r["last_step"]) + 1] *= float(r["step_multiplier"])

    class StepWChunks(sc.StepChunks):
        def __iter__(self):
            wi = torch.utils.data.get_worker_info()
            wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
            rng = random.Random(self.seed * 9_999_991 + wid)
            seats = self._seats(wid, nw)

            def load():
                path = next(seats)
                arrs, dt = sc.decode(path.read_bytes())
                idx = list(range(arrs["step"].shape[0]))
                if not idx:
                    raise ValueError("empty cached seat has zero steps")
                eid, seat = (int(v) for v in path.stem.split("-"))
                steps = np.asarray(arrs["step"]).reshape(len(idx)).astype(np.int64)
                e = table.get((eid, seat))
                arrs = dict(arrs)
                arrs["step_w"] = e[steps] if e is not None else np.ones(len(idx), dtype=np.float32)
                rng.shuffle(idx)
                return [arrs, idx], dt

            live, dtypes = [], None
            while len(live) < self.pool_seats:
                e, dtypes = load()
                live.append(e)
            while True:
                take = []
                for _ in range(self.chunk):
                    j = rng.randrange(len(live))
                    arrs, idx = live[j]
                    take.append((arrs, idx.pop()))
                    if not idx:
                        live[j], dtypes = load()
                yield sc.collate_steps(take, dtypes, self.noop_schema)
    return StepWChunks


def make_loader(cache_dir, rows, *, microbatch, seed, workers, pool_seats=2):
    sc=importlib.import_module('step_cache')
    tsv=os.environ.get('OA3_STEP_W_TSV')
    if tsv:
        from .stepw import StepWChunks   # module-level class: picklable for spawn workers
        dataset=StepWChunks(cache_dir,microbatch,seed,pool_seats=pool_seats,rows=rows,table_path=tsv)
    else:
        dataset=sc.StepChunks(cache_dir,microbatch,seed,pool_seats=pool_seats,rows=rows)
    if len(dataset.files)!=len(dataset.all_files):
        raise ValueError('Cache reader dropped missing seats; refusing changed mixture')
    return torch.utils.data.DataLoader(dataset,batch_size=None,num_workers=workers,
        pin_memory=torch.cuda.is_available(),persistent_workers=workers>0,
        **({'prefetch_factor':2,'multiprocessing_context':'spawn'} if workers>0 else {}))

def device_batch(chunk, device, *, mask_schema=None, require_step_w=True):
    sc=importlib.import_module('step_cache')
    if not chunk['step_valid'].all() or not chunk['loss_mask'].all():
        raise ValueError('Only real one-step cache chunks are supported, not padded/BPTT windows')
    if mask_schema is not None and chunk.get('unit_noop_mask_schema')!=mask_schema:
        raise ValueError('Unit noop mask schema mismatch')
    allowed=set(INPUTS+GROW+LABELS+('unit_loss_keep','_dtypes','step_w'))
    narrow={k:v for k,v in chunk.items() if k in allowed}
    if 'step_w' in narrow:
        # step_cache.to_device casts keys missing from _dtypes to bool (float weights -> 1.0); declare the weight dtype
        narrow['_dtypes']=dict(narrow['_dtypes'],step_w='float32')
    wide=sc.to_device(narrow,device)
    out={}
    for k,v in wide.items():
        if not torch.is_tensor(v) or v.ndim<2 or v.shape[1]!=1:
            raise ValueError(f'Expected one-step [B,1,...] cache tensor: {k}')
        out[k]=v[:,0]
    if require_step_w and os.environ.get('OA3_STEP_W_TSV') and (out.get('step_w') is None or not out['step_w'].is_floating_point()):
        raise ValueError('step weights configured but missing or non-float at loss ingress')
    return out

def _autocast(device, bf16):
    return torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device.type=='cuda' and bf16 else nullcontext()

def _matching_grammar(a,b):
    for name in ('n_unit_tokens','n_order_tokens','max_units','max_orders','n_qty','stop_index',
                 'unit_has_qty','order_has_qty','feature_schema'):
        if getattr(a.config,name)!=getattr(b.config,name):
            raise ValueError(f'Teacher/student contract mismatch: {name}')
    if a.config.think!='off' or b.config.think!='off':
        raise ValueError('Stateless think=off models required')

def _check_cache_inventory(cache, frozen, rec):
    for name in ('train.tsv','validation.tsv'):
        for row in read_roster(frozen/name):
            f=cache/f'{row[3]}-{row[4]}.stp'
            if not f.is_file():
                raise ValueError(f'Frozen roster cache seat missing: {f}')
            old_size=rec.get('cache_seat_sizes',{}).get(f.name)
            if old_size is not None and f.stat().st_size!=old_size:
                raise ValueError(f'Cache seat changed size after freeze: {f}')

@torch.no_grad()
def evaluate(net, cache, rows, *, microbatch, seed, val_batches, device, bf16, deadline):
    net.eval(); measurements=[]
    # Reconstruct with the same seed and single worker at each evaluation so the
    # same validation steps are used. Validation uses unmasked canonical CE.
    loader=make_loader(cache,rows,microbatch=microbatch,seed=seed,workers=0)
    it=iter(loader)
    try:
        for _ in range(val_batches):
            if deadline.expired():break
            batch=device_batch(next(it),device)
            with _autocast(device,bf16):
                _,metrics=objective(forward(net,batch),None,batch,net,alpha=0.)
            measurements.append(metrics['canonical_bits_per_step'])
    finally:
        del it,loader
    return {'batches':len(measurements),'steps':len(measurements)*microbatch,
            'canonical_bits_per_step':statistics.mean(measurements) if measurements else None,
            'seed':seed,'interpretation':'Fixed cached validation diagnostic, not playing strength'}

def ez_schedule(total, cap=20000):
    """EZ-OA3-SCHEDULE-0930 (by TOTAL update, continuation starts at u1200): 200-update warm restart 1e-6 -> 3e-5, flat 3e-5
    through u16000, cosine to 1e-5 at u20000; KD weight 0.50 through u4000, 0.30 through u12000, 0.15 after. -> (lr, kd_weight)"""
    # cap-scaled (coordinator 09-29 20:3x): flat to 0.8*cap, cosine to 1e-5 at cap; KD stages at 0.2*cap and 0.6*cap
    c = total - 1200
    flat_end, k1, k2 = 0.8 * cap, 0.2 * cap, 0.6 * cap
    if c <= 200:
        lr = 1e-6 + (3e-5 - 1e-6) * max(0, c) / 200
    elif total <= flat_end:
        lr = 3e-5
    else:
        lr = 1e-5 + 0.5 * (3e-5 - 1e-5) * (1 + math.cos(math.pi * min(1.0, (total - flat_end) / (cap - flat_end))))
    kd = 0.5 if total <= k1 else 0.3 if total <= k2 else 0.15
    return lr, kd


def run(args) -> dict:
    # Start the budget BEFORE expensive checkpoints/cache preflight.
    deadline=Deadline(args.stop_at,args.max_seconds)
    if args.updates<1 or args.microbatch<1 or args.effective_batch<args.microbatch or args.effective_batch%args.microbatch:
        raise ValueError('Positive updates; effective batch must be an integer multiple of microbatch')
    if args.workers<0 or args.val_batches<1 or args.save_every<1 or args.warmup<0:
        raise ValueError('Invalid worker/evaluation/checkpoint settings')
    if not 0<=args.alpha<=1 or not 0<=args.ema<1:
        raise ValueError('alpha and ema outside supported range')
    for name in ('lr','min_lr','clip','temperature'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<=0:
            raise ValueError(f'Positive finite {name} required')
    out=Path(args.out).resolve()
    if out.exists():raise FileExistsError(f'Use a new run directory: {out}')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. This is not a GPU benchmark; use --device cpu only for smoke tests.')
    if device.type=='cuda' and not args.fp32 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unavailable; explicitly select --fp32')
    activate(args.trainer)
    sc=importlib.import_module('step_cache')
    cache=Path(args.cache).resolve();frozen=Path(args.frozen).resolve()
    fr=verify_frozen(frozen);_check_cache_inventory(cache,frozen,fr)
    meta=sc.validate_metadata(cache/'CACHE.json') # no bypass or metadata rewriting
    student,srec=load_checkpoint(args.student,args.student_component,args.trainer)
    teacher,trec=load_checkpoint(args.teacher,args.teacher_component,args.trainer)
    _matching_grammar(student,teacher)
    validate_feature_subset(meta,student.config);validate_feature_subset(meta,teacher.config)
    if args.unit_mask:
        noop=importlib.import_module('noop_mask')
        if meta.get('noop_mask_schema')!=noop.SCHEMA:
            raise ValueError('Cache does not certify the current noop-mask schema')
    torch.manual_seed(args.seed)
    student=student.to(device);teacher=teacher.to(device).eval().requires_grad_(False)
    # Fresh optimizer by design. We do NOT transplant OA1 moments across layers.
    fast=os.environ.get('OA3_FAST_PREFIX','')   # N (09-29): e.g. grow_recur_gate at OA3_FAST_MULT x LR, no weight decay
    if fast:
        fmult=float(os.environ.get('OA3_FAST_MULT','10'))
        fp=[p for n,p in student.named_parameters() if n.startswith(tuple(fast.split(',')))]
        op=[p for n,p in student.named_parameters() if not n.startswith(tuple(fast.split(',')))]
        if not fp: raise ValueError(f'OA3_FAST_PREFIX {fast} matches no parameter')
        optimizer=torch.optim.AdamW([{'params':op,'lr_mult':1.0},{'params':fp,'lr_mult':fmult,'weight_decay':0.0}],lr=args.lr,betas=(.9,.95),weight_decay=.05)
    else:
        optimizer=torch.optim.AdamW(student.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=.05)
    ema={k:v.detach().clone() for k,v in student.state_dict().items()}
    start=int(getattr(args,'start_update',0) or 0)
    if getattr(args,'state',None):
        # Continuation (gpu-ops 09-29): carry AdamW moments and EMA from a saved oa3 checkpoint; --student must be the
        # same file with component=model. The update counter restarts at --start-update of the new schedule.
        sb=torch.load(args.state,map_location='cpu',weights_only=True)
        if sha256(args.state)!=sha256(args.student) or args.student_component!='model':
            raise ValueError('--state must be the --student checkpoint loaded with component=model')
        optimizer.load_state_dict(sb['optimizer'])
        if set(sb['ema'])!=set(ema):raise ValueError('EMA keys differ from the student state')
        ema={k:sb['ema'][k].to(device=ema[k].device,dtype=ema[k].dtype).clone() for k in ema}
        del sb
    receipt={'student':srec,'teacher':trec,'source':source_receipt(args.trainer),
        'args':vars(args),'frozen':fr,'cache_metadata_sha256':sha256(cache/'CACHE.json'),
        'python':platform.python_version(),'torch':str(torch.__version__),'device':str(device),
        'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'pbc_environment':{k:v for k,v in os.environ.items() if k.startswith('PBC_')},
        'step_w_table':({'path':os.environ['OA3_STEP_W_TSV'],'sha256':sha256(os.environ['OA3_STEP_W_TSV'])} if os.environ.get('OA3_STEP_W_TSV') else None),
        'optimizer':'fresh AdamW beta=(.9,.95) wd=.05; explicit clip; no upstream env clip override',
        'kd':'Same recorded full prefix and command-conditioned quantity heads, not online/on-policy distillation',
        'budget':'Cooperative stop; one in-flight batch or checkpoint may exceed wall bound',
        'game_strength_verified':False,'continuation':{'state':getattr(args,'state',None),'start_update':int(getattr(args,'start_update',0) or 0)}}
    out.mkdir(parents=True)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    accumulation=args.effective_batch//args.microbatch
    loader=make_loader(cache,frozen/'train.tsv',microbatch=args.microbatch,seed=args.seed,
                       workers=args.workers)
    # PRO CRAZY OA3 DAgger (gpu-ops 09-29): OA3_DAGGER_CACHE + OA3_DAGGER_ROWS -> half of each update's microbatches come from
    # OA3-visited states (league replays), KL-only (alpha 1); expert microbatches keep the scheduled BC/KD mix.
    dagger_it=None
    if os.environ.get('OA3_DAGGER_CACHE') and os.environ.get('OA3_DAGGER_ROWS'):
        dsc=importlib.import_module('step_cache')
        dds=dsc.StepChunks(os.environ['OA3_DAGGER_CACHE'],args.microbatch,args.seed+77,pool_seats=2,rows=os.environ['OA3_DAGGER_ROWS'])
        dagger_it=iter(torch.utils.data.DataLoader(dds,batch_size=None,num_workers=1,pin_memory=torch.cuda.is_available(),
            persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn'))
        receipt['dagger']={'cache':os.environ['OA3_DAGGER_CACHE'],'rows':os.environ['OA3_DAGGER_ROWS'],
            'rows_sha256':sha256(os.environ['OA3_DAGGER_ROWS']),'mix':'50% expert (scheduled BC/KD) / 50% DAgger (KL only)'}
    iterator=iter(loader);completed=start;durations=[];termination='update_cap';pending_interrupt=False
    bf16=not args.fp32
    mask_schema=meta.get('noop_mask_schema') if args.unit_mask else None
    try:
        # Both baselines use the same fixed validation steps. No checkpoint is
        # selected automatically by validation; only a later playing panel can promote.
        initial={}
        for name,net in [('student',student),('teacher',teacher)]:
            initial[name]=evaluate(net,cache,frozen/'validation.tsv',microbatch=args.microbatch,
                seed=args.seed_val,val_batches=args.val_batches,device=device,bf16=bf16,deadline=deadline)
        (out/'validation-initial.json').write_text(json.dumps(initial,indent=2)+'\n')
        for update in range(start+1,args.updates+1):
            if deadline.expired():termination='deadline';break
            if getattr(args,'pause_file',None) and Path(args.pause_file).exists():termination='paused';break
            tick=time.monotonic();student.train();optimizer.zero_grad(set_to_none=True)
            ratio=min(1.,update/max(1,args.warmup))
            alpha=args.alpha
            if getattr(args,'schedule','cosine')=='ez':
                lr,alpha=ez_schedule(update+args.total_offset,getattr(args,'cap',20000))
            elif update>args.warmup:
                fraction=min(1.,(update-args.warmup)/max(1,args.updates-args.warmup))
                lr=args.min_lr+.5*(args.lr-args.min_lr)*(1+math.cos(math.pi*fraction))
            else:lr=args.lr*ratio
            for group in optimizer.param_groups:group['lr']=lr*group.get('lr_mult',1.0)
            hg_s=None
            if os.environ.get('OA3_HG_ANNEAL') and hasattr(student,'grow_hg_s'):
                # N option A': hourglass skip s(t) = 1 -> 0 linearly over OA3_HG_ANNEAL updates, floored by OA3_HG_FLOOR_FILE
                floor=0.0; ff=os.environ.get('OA3_HG_FLOOR_FILE')
                if ff and Path(ff).exists():
                    try: floor=float(Path(ff).read_text().strip())
                    except ValueError: floor=0.0
                hg_s=max(floor,1.0-update/float(os.environ['OA3_HG_ANNEAL']))
                student.grow_hg_s.fill_(hg_s)
            metrics=[];partial=False
            for mi in range(accumulation):
                if deadline.expired():partial=True;break
                use_dagger=dagger_it is not None and mi%2==1
                batch=device_batch(next(dagger_it if use_dagger else iterator),device,mask_schema=mask_schema,require_step_w=not use_dagger)
                mb_alpha=1.0 if use_dagger else alpha
                # no_grad (rather than inference_mode) keeps detached outputs safe
                # when their probabilities participate in the student backward graph.
                with torch.no_grad(),_autocast(device,bf16):
                    tout=forward(teacher,batch) if mb_alpha else None
                with _autocast(device,bf16):
                    loss,met=objective(forward(student,batch),tout,batch,student,
                        alpha=mb_alpha,temperature=args.temperature,use_unit_mask=args.unit_mask)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective; no optimizer step taken')
                (loss/accumulation).backward();metrics.append(met)
            if partial:
                optimizer.zero_grad(set_to_none=True);termination='deadline_partial_accumulation_discarded';break
            norm=torch.nn.utils.clip_grad_norm_(student.parameters(),args.clip,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for k,v in student.state_dict().items():
                    if v.is_floating_point() and k!='grow_hg_s':ema[k].lerp_(v,1-args.ema)
                    else:ema[k].copy_(v)
            completed=update;elapsed=time.monotonic()-tick;durations.append(elapsed)
            record={'hg_s':hg_s,'update':update,'total_update':update+getattr(args,'total_offset',0),'lr':lr,'bc_weight':1-alpha,'kd_weight':alpha,'grad_norm_before_clip':float(norm),'seconds':elapsed,
                'remaining_seconds':deadline.remaining(),
                **{k:statistics.mean(m[k] for m in metrics) for k in ('objective','bc','kd','canonical_bits_per_step')}}
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if update==1 or update%25==0:print(json.dumps(record),flush=True)
            if update==100:
                recent=durations[-50:]
                (out/'pilot.json').write_text(json.dumps({'updates':100,'median_update_seconds':statistics.median(recent),
                    'p90_update_seconds':sorted(recent)[int(.9*(len(recent)-1))],
                    'projected_remaining_compute_seconds':statistics.mean(recent)*max(0,args.updates-update),
                    'remaining_budget_seconds':deadline.remaining(),
                    'warning':'Projection excludes checkpoint/evaluation/queue variability; not promised completion.'},indent=2)+'\n')
            if (update+getattr(args,'total_offset',0))%args.save_every==0:
                save_training_checkpoint(out/f'u{update:06d}.pt',student,optimizer,ema,update,receipt)
                if getattr(args,'val_on_save',False):
                    raw={k:v.detach().clone() for k,v in student.state_dict().items()}
                    student.load_state_dict(ema,strict=True)
                    try:
                        v=evaluate(student,cache,frozen/'validation.tsv',microbatch=args.microbatch,seed=args.seed_val,
                            val_batches=args.val_batches,device=device,bf16=bf16,deadline=deadline)
                    finally:
                        student.load_state_dict(raw,strict=True)
                    (out/f'validation-u{update:06d}-ema.json').write_text(json.dumps(v,indent=2)+'\n')
                    print(json.dumps({'update':update,'validation_ema_bits_per_step':v['canonical_bits_per_step']}),flush=True)
        if not deadline.expired() and termination!='paused':
            # Evaluate EMA, not raw model, and restore raw state before checkpointing.
            raw={k:v.detach().clone() for k,v in student.state_dict().items()}
            student.load_state_dict(ema,strict=True)
            try:
                final_val=evaluate(student,cache,frozen/'validation.tsv',microbatch=args.microbatch,
                    seed=args.seed_val,val_batches=args.val_batches,device=device,bf16=bf16,deadline=deadline)
            finally:
                student.load_state_dict(raw,strict=True)
            (out/'validation-final-ema.json').write_text(json.dumps(final_val,indent=2)+'\n')
    except KeyboardInterrupt:
        termination='keyboard_interrupt';pending_interrupt=True
    except Exception as exc:
        termination=f'error:{type(exc).__name__}'
        (out/'ERROR.txt').write_text(repr(exc)+'\n')
        raise
    finally:
        # Final save is explicit, even on errors; it is NEVER auto-promoted.
        receipt['termination']=termination
        save_training_checkpoint(out/'last.pt',student,optimizer,ema,completed,receipt)
        if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
        del iterator,loader
        summary={'completed_updates':completed,'termination':termination,
            'wall_seconds':time.monotonic()-deadline.started,
            'last_sha256':sha256(out/'last.pt'),'game_strength_verified':False}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if pending_interrupt:raise KeyboardInterrupt
    return summary
