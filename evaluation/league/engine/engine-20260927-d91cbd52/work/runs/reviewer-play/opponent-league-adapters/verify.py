"""Complete-game parity, same-seat interleaving independence and CPU measurements.

Each scripted opponent gets three independent main.py reference subprocesses and
three simultaneous in-process adapter instances. Two lanes deliberately share the
SAME player seat but have different world seeds; a third lane covers seat 1.
30 physical games total for all five scripts. Frozen source bytes are not edited.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from league import ROOT, HOME, SCRIPTED, League, canonical, digest, seal, sha_file, freerun
from panel import atomic_json, charge_native_start
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment, pass_agent

CASES=({'seat':0,'world_seed':98001,'sampler_seed':11},
       {'seat':0,'world_seed':98002,'sampler_seed':13},
       {'seat':1,'world_seed':98001,'sampler_seed':17})


def verify_one(name, output):
    league=League();identity=league.identity(name)
    references=[]
    for index,case in enumerate(CASES):
        path=output/f'{name}-reference-{index}.json.gz'
        charge_native_start()
        command=[sys.executable,'-B',str(HOME/'reference.py'),
                 '--main',str(ROOT/identity['main_path']),'--sha256',identity['main_sha256'],
                 '--seat',str(case['seat']),'--world-seed',str(case['world_seed']),
                 '--sampler-seed',str(case['sampler_seed']),'--output',str(path)]
        run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=240)
        (output/f'{name}-reference-{index}.log').write_text(run.stdout+run.stderr)
        if run.returncode:
            raise RuntimeError(f'independent reference failed for {name} lane {index}: {run.stderr[-1500:]}')
        with gzip.open(path,'rt') as stream:references.append(json.load(stream))
    # All instances coexist. No global RNG changes survive construction/calls.
    ambient=random.getstate();actors=[];envs=[];load_costs=[]
    for case in CASES:
        begun=time.process_time()
        actors.append(league.new_actor(name,seat=case['seat'],sampler_seed=case['sampler_seed']))
        load_costs.append(time.process_time()-begun)
        charge_native_start()
        envs.append(ExactKaggricultureEnvironment(seed=case['world_seed']))
    if random.getstate()!=ambient:
        raise AssertionError('adapter construction/engine changed ambient Python RNG')
    if len({id(a.module) for a in actors})!=len(actors):
        raise AssertionError('shared agent source-module namespace')
    tapes=[[[],[]] for _ in CASES];lane_cpu=[0.0]*len(CASES)
    started=time.monotonic()
    for step in range(719):
        # Rotate lane order: unlike a same-input replay, this catches source-global
        # state accidentally borrowed from an actor that has advanced farther.
        order=list(range(len(CASES)));order=order[step%3:]+order[:step%3]
        for i in order:
            begun=time.process_time();case=CASES[i];env=envs[i]
            pair=[pass_agent,pass_agent];pair[case['seat']]=actors[i]
            actions=[agent(env.agent_observation(seat),env.configuration) for seat,agent in enumerate(pair)]
            for seat in (0,1):tapes[i][seat].append(copy.deepcopy(actions[seat]))
            env.step(actions)
            lane_cpu[i]+=time.process_time()-begun
        if random.getstate()!=ambient:
            raise AssertionError('adapter call changed ambient Python RNG')
    rows=[]
    for i,case in enumerate(CASES):
        reference=references[i];cash,statuses=freerun._terminal(envs[i],719)
        expected=reference['actions_by_seat']
        mismatch=next(({'step':step,'seat':seat,'raw':expected[seat][step],'adapter':tapes[i][seat][step]}
                       for step in range(719) for seat in (0,1)
                       if tapes[i][seat][step]!=expected[seat][step]),None)
        passed=mismatch is None and cash==reference['terminal_cash'] and statuses==reference['terminal_statuses']
        row={'case':case,'passed':passed,'first_action_mismatch':mismatch,
             'all_1438_joint_actions_equal':mismatch is None,
             'raw_actions_sha256':digest(expected),'adapter_actions_sha256':digest(tapes[i]),
             'terminal_cash':cash,'reference_terminal_cash':reference['terminal_cash'],
             'steps':719,'adapter_calls':actors[i].calls,
             'adapter_cpu_seconds':lane_cpu[i]+load_costs[i],
             'adapter_loop_cpu_seconds':lane_cpu[i], 'adapter_load_cpu_seconds':load_costs[i],
             'adapter_actor_cpu_seconds':actors[i].cpu_seconds,
             'reference_cpu_seconds':reference['cpu_seconds'],
             'reference_actor_cpu_seconds':reference['actor_cpu_seconds'],
             'reference_file_sha256':sha_file(output/f'{name}-reference-{i}.json.gz')}
        rows.append(row)
    if league.identity(name)!=identity:
        raise ValueError('opponent source/asset drift during parity test')
    result=seal({'schema':'slot98-complete-game-parity-v1','opponent':name,'identity':identity,
        'passed':all(r['passed'] for r in rows),'games':rows,
        'independence':'three interleaved instances, two on seat0 with different worlds; third on seat1',
        'ambient_rng_unchanged':True,'physical_games_started':6,'physical_games_completed':6,
        'scripted_actor_decisions_compared':719*3,
        'adapter_mean_cpu_seconds_per_game':sum(r['adapter_cpu_seconds'] for r in rows)/len(rows),
        'raw_mean_cpu_seconds_per_game':sum(r['reference_cpu_seconds'] for r in rows)/len(rows),
        'interleaved_wall_seconds':time.monotonic()-started,
        'measurement_note':'process CPU, n=3 per implementation/opponent, includes agent load; excludes reference subprocess startup and charge bookkeeping. Adapter engine initialization excluded (<loop timing); reference includes it. Not a throughput campaign.'})
    atomic_json(output/f'{name}-PARITY.json',result)
    if not result['passed']:
        raise AssertionError(f'{name} full-game parity failed; first mismatch retained')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opponents',nargs='+',choices=SCRIPTED,default=list(SCRIPTED))
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    out=args.output.resolve()
    if not out.is_relative_to(HOME):parser.error('output must stay inside task directory')
    out.mkdir(parents=True,exist_ok=False)
    results=[]
    registration={'schema':'slot98-parity-plan-v1','cases':CASES,'opponents':args.opponents,
                  'physical_games_planned':6*len(args.opponents),
                  'python':sys.version,'platform':platform.platform(),
                  'source_pins':{p.name:sha_file(p) for p in (HOME/'verify.py',HOME/'reference.py',HOME/'league.py')}}
    atomic_json(out/'PLAN.json',seal(registration))
    for name in args.opponents:
        result=verify_one(name,out);results.append(result)
        atomic_json(out/'RESULT.json',seal({'schema':'slot98-parity-summary-v1',
            'results':results,'physical_games_started':sum(r['physical_games_started'] for r in results),
            'physical_games_completed':sum(r['physical_games_completed'] for r in results),
            'passed':len(results)==len(args.opponents) and all(r['passed'] for r in results)}))
        print(f"{name}: PASS; three complete action-stream matches; "
              f"adapter CPU/game={result['adapter_mean_cpu_seconds_per_game']:.4f}s",flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
