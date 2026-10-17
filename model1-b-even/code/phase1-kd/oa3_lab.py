#!/usr/bin/env python3
"""OA2 replacement lab. Every write is local; no scheduler or upload action."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trainer',help='Full upstream trainer directory; defaults to bundled reference')
    sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build',help='Lossy depth reduction with explicit checkpoint component')
    b.add_argument('--parent',required=True);b.add_argument('--component',choices=['model','ema'],required=True)
    b.add_argument('--enc-layers',type=int,default=6);b.add_argument('--dec-layers',type=int,default=2)
    b.add_argument('--features',choices=['inherit','v8'],default='v8');b.add_argument('--seed',type=int,default=930)
    b.add_argument('--out',required=True)
    f=sub.add_parser('freeze',help='Immutable weighted rows; owner exclusion registry required')
    for name in ('train','validation','exclusions','out'):f.add_argument('--'+name,required=True)
    f.add_argument('--cache',help='Validate all frozen cache seats exist; recommended')
    s=sub.add_parser('patch-serving',help='Write reviewed v8 serving patch into a NEW source copy')
    s.add_argument('--out',required=True)
    e=sub.add_parser('export',help='Explicitly export selected component into standard PolicyNet file')
    e.add_argument('--checkpoint',required=True);e.add_argument('--component',choices=['model','ema'],required=True)
    e.add_argument('--dtype',choices=['fp32','fp16'],default='fp16');e.add_argument('--out',required=True)
    g=sub.add_parser('plan-panel',help='Freeze artifacts and disjoint two-seat screen/confirm worlds')
    for name in ('candidate','b','oa1','rules-file','registry','out'):g.add_argument('--'+name,required=True)
    g.add_argument('--phase',choices=['screen','confirm'],required=True);g.add_argument('--screen-manifest')
    g.add_argument('--seed',type=int,default=930)
    r=sub.add_parser('reduce-panel',help='Reduce complete two-seat world pairs; provenance manifest required')
    r.add_argument('--results',required=True);r.add_argument('--manifest',required=True);r.add_argument('--out',required=True)
    t=sub.add_parser('train',help='Bounded cache-backed BC/KD; never extracts data or changes live jobs')
    for name in ('student','teacher','cache','frozen','out','stop-at'):t.add_argument('--'+name,required=True)
    t.add_argument('--student-component',choices=['model','ema'],required=True)
    t.add_argument('--teacher-component',choices=['model','ema'],required=True)
    t.add_argument('--device',default='cuda');t.add_argument('--max-seconds',type=float,default=3600)
    t.add_argument('--updates',type=int,default=1200);t.add_argument('--microbatch',type=int,default=16)
    t.add_argument('--effective-batch',type=int,default=512);t.add_argument('--workers',type=int,default=2)
    t.add_argument('--val-batches',type=int,default=8);t.add_argument('--save-every',type=int,default=100)
    t.add_argument('--warmup',type=int,default=100);t.add_argument('--lr',type=float,default=3e-5)
    t.add_argument('--min-lr',type=float,default=1e-6);t.add_argument('--clip',type=float,default=1.)
    t.add_argument('--ema',type=float,default=.999);t.add_argument('--alpha',type=float,default=.5)
    t.add_argument('--temperature',type=float,default=1.5);t.add_argument('--seed',type=int,default=930)
    t.add_argument('--fp32',action='store_true');t.add_argument('--unit-mask',action='store_true')
    t.add_argument('--state',help='continuation: carry optimizer+EMA from this oa3 checkpoint (same file as --student, component model)')
    t.add_argument('--start-update',type=int,default=0);t.add_argument('--pause-file',help='cooperative pause: stop before the next update, save last.pt')
    t.add_argument('--val-on-save',action='store_true');t.add_argument('--seed-val',type=int,default=930+1000003)
    t.add_argument('--schedule',choices=['cosine','ez'],default='cosine');t.add_argument('--total-offset',type=int,default=0)
    t.add_argument('--cap',type=int,default=20000)
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    if args.trainer:os.environ['OA3_TRAINER']=str(Path(args.trainer).resolve())
    from oa3.source import activate,trainer_dir,sha256
    activate()
    if args.command=='build':
        import torch
        from oa3.build import load_checkpoint,build_student
        path=Path(args.out)
        if path.exists():raise FileExistsError(path)
        parent,pr=load_checkpoint(args.parent,args.component)
        torch.manual_seed(args.seed)
        net,rec=build_student(parent,enc_layers=args.enc_layers,dec_layers=args.dec_layers,feature_mode=args.features)
        path.parent.mkdir(parents=True,exist_ok=True)
        net.save(path,extra={'parent':pr,'build':rec,'seed':args.seed,'trained':False})
        rec.update(parent=pr,seed=args.seed,checkpoint_sha256=sha256(path),trained=False)
        path.with_suffix(path.suffix+'.json').write_text(json.dumps(rec,indent=2)+'\n')
        result={k:rec[k] for k in ('parent_parameters','student_parameters','encoder_indices','decoder_indices','function_preserving','trained')}
    elif args.command=='freeze':
        from oa3.data_guard import freeze_rosters
        ex=json.loads(Path(args.exclusions).read_text())
        if not isinstance(ex,dict) or not isinstance(ex.get('episode_ids'),list) or not ex.get('provenance'):
            raise ValueError('Exclusions JSON must contain episode_ids list and provenance description')
        result=freeze_rosters(args.train,args.validation,args.out,ex['episode_ids'],args.cache)
        dest=Path(args.out);(dest/'exclusions.json').write_text(json.dumps(ex,indent=2)+'\n')
        result['exclusions_sha256']=sha256(dest/'exclusions.json')
        result['frozen_hashes']['exclusions.json']=result['exclusions_sha256']
        (dest/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    elif args.command=='patch-serving':
        from oa3.serving import write_isolated
        result=write_isolated(trainer_dir(),args.out)
    elif args.command=='export':
        import torch
        from oa3.build import load_checkpoint
        path=Path(args.out)
        if path.exists():raise FileExistsError(path)
        net,rec=load_checkpoint(args.checkpoint,args.component)
        path.parent.mkdir(parents=True,exist_ok=True)
        net.save(path,dtype=torch.float16 if args.dtype=='fp16' else torch.float32,extra={'export':rec,'dtype':args.dtype})
        result={'input':rec,'output_sha256':sha256(path),'dtype':args.dtype,'cold_archive_verified':False}
        path.with_suffix(path.suffix+'.json').write_text(json.dumps(result,indent=2)+'\n')
    elif args.command=='plan-panel':
        from oa3.panel import make_plan,BASELINES
        result=make_plan(candidate=args.candidate,baselines=dict(zip(BASELINES,[args.b,args.oa1])),
            rules_file=args.rules_file,registry=json.loads(Path(args.registry).read_text()),phase=args.phase,seed=args.seed,
            screen=json.loads(Path(args.screen_manifest).read_text()) if args.screen_manifest else None)
        path=Path(args.out)
        if path.exists():raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    elif args.command=='reduce-panel':
        from oa3.panel import reduce_pairs
        manifest=json.loads(Path(args.manifest).read_text())
        rows=[json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
        result=reduce_pairs(rows,expected_worlds=len(manifest['worlds']),manifest=manifest)
        path=Path(args.out)
        if path.exists():raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    else:
        from oa3.cache_train import run
        result=run(args)
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(2)
