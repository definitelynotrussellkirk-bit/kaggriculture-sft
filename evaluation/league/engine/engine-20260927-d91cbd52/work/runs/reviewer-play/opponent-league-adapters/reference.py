"""Independent oracle: run unmodified main.py via runpy and exact env.run.

No imports of league.py, bridges.py, panel.py, or the repository's cached tape
loader. One fresh process per reference game. Only exact own-seat observations
and the engine's post-initialization public configuration go to main.py.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment, pass_agent


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main',required=True,type=Path)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--seat',required=True,type=int,choices=(0,1))
    parser.add_argument('--world-seed',required=True,type=int)
    parser.add_argument('--sampler-seed',required=True,type=int)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    raw=args.main.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.sha256:
        raise ValueError('raw reference main.py hash differs')
    started=time.monotonic();cpu=time.process_time()
    random.seed(args.sampler_seed)
    namespace=runpy.run_path(str(args.main.resolve()))
    agent=namespace['agent']
    load_cpu=time.process_time()-cpu
    calls=0;actor_cpu=0.0
    def direct(observation,configuration):
        nonlocal calls,actor_cpu
        assert observation['player']==args.seat and configuration['seed'] is None
        begun=time.process_time()
        try:
            return agent(observation,configuration)
        finally:
            calls+=1;actor_cpu+=time.process_time()-begun
    actors=[pass_agent,pass_agent];actors[args.seat]=direct
    env=ExactKaggricultureEnvironment(seed=args.world_seed)
    game=env.run(actors,record_actions=True)
    if game.steps!=719 or game.statuses!=('DONE','DONE'):
        raise ValueError('raw reference did not complete a full game')
    row={'schema':'slot98-independent-main-reference-v1','mode':'NATIVE',
         'main_sha256':args.sha256,'seat':args.seat,'world_seed':args.world_seed,
         'sampler_seed':args.sampler_seed,'steps':game.steps,'calls':calls,
         'terminal_cash':list(game.rewards),'terminal_statuses':list(game.statuses),
         'actions_by_seat':game.actions,'shop_history':game.shop_history,
         'cpu_seconds':time.process_time()-cpu,'load_cpu_seconds':load_cpu,
         'actor_cpu_seconds':actor_cpu,'wall_seconds':time.monotonic()-started,
         'loader':'fresh subprocess runpy.run_path; exact env.run two-agent loop',
         'reference_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.output,'xt') as stream:
        json.dump(row,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
    print(json.dumps({k:row[k] for k in ('seat','world_seed','steps','terminal_cash','cpu_seconds')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
