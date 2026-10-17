"""Reusable fixed-shop curriculum and regression cases, separate from Kaggle play.

The native engine, economy and opponent assets stay unchanged. Only the declared
shop sequence and random-weed setting differ. Ordinary shop visibility is kept.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import gzip
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
from farm_neat.evaluation.common import bound,canonical,digest,equal,file_sha,read,write_new
from farm_neat.evaluation.shop_router import ShopRouterAdapter
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment,PASS_ACTION,DEFAULT_CONFIGURATION,load_engine,assert_bundled_engine_identity
from farm_neat.economy.action_space import validate_program
from farm_neat.economy.orders import FullSeasonPlan

CATALOG=ROOT/'project/configs/scenarios.json'
SCHEMA='farmneat-fixed-shop-scenario-v1'
TEACHERS={name:ROOT/f'work/datasets/corpus/teachers-{version}-hall.json'
    for name,version in (('teacher-v3-00','v3'),('teacher-v5-00','v5'))}


# These are zero-based positions in the retained file, NOT a live cash ranking.
# The hall-v3 namespace preserves both legacy teacher-v3-00/v5-00 identities.
# Changing the file requires a new reviewed pin; never silently reidentify it.
HALL_INDICES = (0, 10, 21, 31, 42, 52)
HALL_SHA256 = '820e79a041b974d9bc2cd13498021d018549f49ff3f14d80223cf75f427a9419'


@dataclass(frozen=True)
class HallRung:
    name: str
    index: int
    hall_sha256: str = HALL_SHA256


HALL_RUNGS = tuple(HallRung(f'hall-v3-{index:02d}', index) for index in HALL_INDICES)


def _hall_registry():
    """Fail closed on ambiguous declarations, including aliases of one index."""
    names, identities = {}, set()
    reserved = {'pass', 'tape0', 'router', *TEACHERS}
    for rung in HALL_RUNGS:
        if type(rung.index) is not int or rung.index < 0:
            raise ValueError('hall index must be a nonnegative integer')
        if (not isinstance(rung.name, str) or not rung.name or
                not isinstance(rung.hall_sha256, str) or len(rung.hall_sha256) != 64 or
                any(c not in '0123456789abcdef' for c in rung.hall_sha256)):
            raise ValueError('explicit hall name and SHA256 required')
        key = (rung.hall_sha256, rung.index)
        if rung.name in names or rung.name in reserved or key in identities:
            raise ValueError('hall opponent identity collision')
        names[rung.name] = rung
        identities.add(key)
    return names


def _hall_opponent(rung):
    """Resolve a pinned source/index; defensive runtime copies never edit the hall.

    Hash and parse the SAME byte snapshot. Program's existing terminal-STOP
    normalization is retained; no handicap, sorting or outcome-based selection.
    Names and source provenance stay evaluator-only, as in the other opponents.
    """
    path = TEACHERS['teacher-v3-00']
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular hall source file required')
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != rung.hall_sha256:
        raise ValueError(f'hall source drift: expected {rung.hall_sha256}, found {actual}')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def bad(value):
        raise ValueError('nonfinite JSON value')

    try:
        hall = json.loads(payload, object_pairs_hook=pairs, parse_constant=bad)
        if not isinstance(hall, list):
            raise ValueError('hall must be a list')
    except (ValueError, UnicodeError) as error:
        raise ValueError(f'malformed hall: {error}') from error
    if rung.index >= len(hall):
        raise ValueError(f'absent index {rung.index} in frozen hall')
    try:
        record = hall[rung.index]
        if not isinstance(record, dict):
            raise ValueError('hall entry must be an object')
        program = record['program']
        if not isinstance(program, list) or not program or any(not isinstance(c, dict) for c in program):
            raise ValueError('nonempty ordered program required')
        actor = Program(program)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(f'malformed hall program at index {rung.index}: {error}') from error
    identity = bound({**copy.deepcopy(actor.identity), 'kind': 'FROZEN_ECONOMIC_PROGRAM',
                      'name': rung.name, 'hall_sha256': actual, 'hall_index': rung.index,
                      'program_sha256': digest(program)})
    actor.identity = copy.deepcopy(identity)
    return actor, identity


def validate(spec):
    if set(spec)!={'id','purpose','seed','shops','opponent','random_weeds'}:
        raise ValueError('complete explicit scenario fields required')
    if not isinstance(spec['id'],str) or not spec['id'] or not isinstance(spec['purpose'],str):
        raise ValueError('scenario identity and purpose required')
    if type(spec['seed']) is not int or not 0<=spec['seed']<2**31 or type(spec['random_weeds']) is not bool:
        raise ValueError('explicit seed and random-weed choice required')
    if spec['opponent'] not in ('pass','tape0','router',*TEACHERS,*_hall_registry()):
        raise ValueError('frozen supported opponent required')
    if not isinstance(spec['shops'],list) or len(spec['shops'])!=8 or any(s not in load_engine().SHOPS for s in spec['shops']):
        raise ValueError('exactly eight legal scheduled shops required')
    return copy.deepcopy(spec)


def catalog():
    value=read(CATALOG,binding=False)
    if value['schema']!=SCHEMA:raise ValueError('known scenario catalog required')
    rows=[validate(s) for s in value['scenarios']]
    if len({s['id'] for s in rows})!=len(rows):raise ValueError('distinct scenario names required')
    return {s['id']:s for s in rows}


def sources():
    from farm_neat.evaluation.shop_router import sources as router_sources
    files=[*sorted((ROOT/'src/farm_neat/economy').rglob('*.py')),
        ROOT/'src/farm_neat/game/constants.py',ROOT/'src/farm_neat/game/exact_engine.py',
        ROOT/'src/farm_neat/game/kaggriculture_engine_reference.py',
        ROOT/'src/farm_neat/game/kaggriculture.json',Path(__file__),CATALOG,*TEACHERS.values()]
    return {**router_sources(),**{str(p.relative_to(ROOT)):file_sha(p) for p in files}}


class ScenarioEnvironment(ExactKaggricultureEnvironment):
    """Per-world schedule injection after unlock, before any agent can observe it.

    Shops are drawn last in the native end-of-day phase; replacing the newly
    unlocked name here preserves that turn's preceding market/weed transitions.
    No shared engine function or RNG is patched. Economic losses still occur.
    """
    def __init__(self,spec):
        self.scenario=validate(spec)
        super().__init__(seed=spec['seed'],configuration={
            'weedSpawnChance':DEFAULT_CONFIGURATION['weedSpawnChance'] if spec['random_weeds'] else 0.})
        self.scenario_identity=bound({'schema':SCHEMA,'scenario':self.scenario,
            'rules':'FIXED_SHOPS; DECLARED_RANDOM_WEEDS; ALL_OTHER_NATIVE_RULES',
            'visibility':'NORMAL_OBSERVATION; FUTURE_SHOPS_AND_SEED_NOT_ACTOR_INPUTS',
            'engine':assert_bundled_engine_identity(),'adapter_sha256':file_sha(__file__)})

    def step(self,joint_action):
        before=len(self.state[0].observation.town['unlocked_shops'])
        equal(self.state[0].observation.town['unlocked_shops'],self.scenario['shops'][:before],'unchanged scheduled prefix')
        result=super().step(joint_action)
        shops=self.state[0].observation.town['unlocked_shops']
        expected=min(8,self.step_index//72)
        if len(shops)!=expected or len(shops)-before not in (0,1):raise ValueError('native unlock clock changed')
        if len(shops)>before:shops[-1]=self.scenario['shops'][before]
        return result


class FixedTape:
    def __init__(self):
        p=ROOT/'src/farm_neat/evaluation/shop_router_0909/actions.json'
        self.actions=read(p,binding=False)[0]
        self.identity={'kind':'FROZEN_EXTERNAL_TAPE0','file_sha256':file_sha(p),'actions_sha256':digest(self.actions)}

    def __call__(self,observation,configuration=None):
        return copy.deepcopy(self.actions[observation['step']])


class Program:
    def __init__(self,program,command_budget=64,*,revisions=None,capture=False):
        values=copy.deepcopy(program)
        if not values or values[-1].get('op')!='STOP':values.append({'op':'STOP'})
        if type(command_budget) is not int or not 1<=command_budget<=256 or len(values)>command_budget:
            raise ValueError('program exceeds declared command budget including STOP')
        self.program=validate_program(values);self.controller=None;self.opening_refusals=[]
        if revisions is None:revisions={}
        if (not isinstance(revisions,dict) or any(type(step) is not int or step not in FullSeasonPlan.STEPS[1:] for step in revisions)
                or type(capture) is not bool):raise ValueError('explicit morning revisions and capture flag required')
        self.revisions={step:validate_program(copy.deepcopy(value)) for step,value in revisions.items()}
        if any(len(value)>command_budget for value in self.revisions.values()):raise ValueError('revision exceeds command budget')
        self.capture=capture;self.records=[];self.publications=[]
        self.identity={'kind':'EXPLICIT_ECONOMIC_PROGRAM','program':self.program,'command_budget':command_budget}
        if self.revisions:self.identity['revisions']={str(step):value for step,value in self.revisions.items()}

    def __call__(self,observation,configuration=None):
        if self.controller is None:
            if observation['step']!=0:raise ValueError('program agent starts at opening')
            self.controller=FullSeasonPlan('scenario-program')
            self.controller.observe(observation)
        else:self.controller.observe(observation)
        step=observation['step']
        if step==0 or step in self.revisions:
            program=self.program if step==0 else self.revisions[step]
            if self.capture:self.records.append({'step':step,'observation':copy.deepcopy(observation),
                'snapshot':self.controller.snapshot(),'configuration':copy.deepcopy(configuration),'opponent_snapshot':None})
            proposal=self.controller.propose(observation,program)
            if step==0:self.opening_refusals=copy.deepcopy(proposal['rejected'])
            self.controller=self.controller.publish(observation,proposal)
            if self.capture:self.publications.append({'step':step,'program':copy.deepcopy(program),
                'proposal_sha256':proposal['content_sha256'],'rejected':copy.deepcopy(proposal['rejected'])})
        return self.controller(observation,configuration)


def opponent(kind):
    hall = _hall_registry()
    if kind in hall:
        return _hall_opponent(hall[kind])
    if kind=='pass':return lambda o,c:copy.deepcopy(PASS_ACTION),{'kind':'PASS'}
    if kind=='tape0':
        actor=FixedTape();return actor,actor.identity
    if kind=='router':
        actor=ShopRouterAdapter();return actor.act,actor.identity
    if kind in TEACHERS:
        path=TEACHERS[kind];actor=Program(read(path,binding=False)[0]['program'])
        return actor,{**actor.identity,'kind':'FROZEN_ECONOMIC_PROGRAM','name':kind,'hall_sha256':file_sha(path)}
    raise ValueError('unknown scenario opponent')


def play(spec,agent,seat,timeout=120.,*,capture_state=False):
    """One full case; a caller can supply a neural policy without another engine."""
    if type(seat) is not int or seat not in (0,1) or not 0<timeout<=300:raise ValueError('seat and bounded timeout required')
    env=ScenarioEnvironment(spec);other,identity=opponent(spec['opponent']);started=time.monotonic()
    def guarded(actor):
        def call(obs,cfg):
            if time.monotonic()-started>timeout:raise TimeoutError('scenario game deadline')
            return actor(obs,cfg)
        return call
    actors=[guarded(other),guarded(other)];actors[seat]=guarded(agent)
    try:
        result=env.run(actors,record_actions=True)
        cash=list(result.rewards);margin=cash[seat]-cash[1-seat]
        row={'status':'COMPLETE','scenario_identity':env.scenario_identity,'seat':seat,
            'opponent_identity':identity,'physical_games_completed':1,'physical_steps':result.steps,
            'terminal_cash_by_seat':cash,'wdl':'WIN' if margin>0 else 'LOSS' if margin<0 else 'DRAW',
            'margin':margin,'actions_by_seat':result.actions,'actions_sha256':digest(result.actions),
            'shop_history':result.shop_history,'cash_history':result.cash_history,
            'opening_refusals':getattr(agent,'opening_refusals',[]),
            'seconds':time.monotonic()-started,'scope':'FIXED_SCENARIO_PRACTICE_OR_REGRESSION; NOT_OFFICIAL_RNG_STRENGTH'}
        if capture_state:row['terminal_observation']=env.agent_observation(seat)
        return row
    except Exception as error:
        return {'status':'FAILED','scenario_identity':env.scenario_identity,'seat':seat,
            'opponent_identity':identity,'physical_games_completed':0,'physical_steps':env.step_index,
            'error':repr(error),'seconds':time.monotonic()-started,'no_automatic_retry':True}


def journal(path,row):
    import os
    with path.open('a') as stream:
        stream.write(canonical(row)+'\n');stream.flush();os.fsync(stream.fileno())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('list','run'))
    parser.add_argument('--scenario',action='append',help='Repeat for several cases; default all')
    parser.add_argument('--agent',choices=('program','tape0'),default='program')
    parser.add_argument('--program',type=Path);parser.add_argument('--teacher-index',type=int)
    parser.add_argument('--command-budget',type=int,default=64)
    parser.add_argument('--seat',choices=('both','0','1'),default='both')
    parser.add_argument('--repeat',type=int,default=1);parser.add_argument('--seconds',type=float,default=300)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();cases=catalog()
    selected=[cases[name] for name in (args.scenario or cases)]
    if args.command=='list':print(json.dumps(selected,indent=2));return
    if not 1<=args.repeat<=1000 or not 1<=args.seconds<=7200 or args.output is None:parser.error('bounded repeats, seconds and new output directory required')
    out=args.output.absolute()
    if not out.is_relative_to(ROOT/'work/runs') or out!=out.resolve() or out.exists():parser.error('new non-symlink directory under work/runs required')
    if args.agent=='program':
        if args.program is None:parser.error('--program required')
        values=read(args.program,binding=False)
        if args.teacher_index is not None:values=values[args.teacher_index]['program']
        make_agent=lambda:Program(values,args.command_budget)
    else:make_agent=FixedTape
    agent_identity=make_agent().identity
    seats=[0,1] if args.seat=='both' else [int(args.seat)]
    requests=[{'scenario':s,'seat':seat,'repeat':r} for r in range(args.repeat) for s in selected for seat in seats]
    out.mkdir(parents=True)
    plan=write_new(out/'PLAN.json',{'schema':SCHEMA,'cases':requests,'agent':agent_identity,
        'catalog_sha256':file_sha(CATALOG),'source_sha256':file_sha(__file__),'source_pins':sources(),'seconds':args.seconds,
        'repeated_attempts_are_not_distinct_scenarios':True,'future_shop_visibility':'ORDINARY_GAME_VISIBILITY',
        'retention':{'purpose':'Reusable scenario training/evaluation evidence','dependencies':[str(CATALOG.relative_to(ROOT))],
            'remove_when':'No active learner, regression or scenario comparison depends on this run'}})
    rows=[];started=time.monotonic()
    for index,case in enumerate(requests):
        remaining=args.seconds-(time.monotonic()-started)
        if remaining<=0:break
        journal(out/'attempts.jsonl',{'event':'STARTED','index':index,**case})
        row=play(case['scenario'],make_agent(),case['seat'],min(120.,remaining))
        row.update(index=index,repeat=case['repeat'],plan_sha256=plan['content_sha256'])
        path=out/f'case-{index:04d}.json.gz'
        with path.open('xb') as f:f.write(gzip.compress(canonical(bound(row)).encode(),mtime=0))
        saved={'index':index,'scenario':case['scenario']['id'],'seat':case['seat'],'repeat':case['repeat'],
            'file':str(path.relative_to(ROOT)),'sha256':file_sha(path),**{k:row[k] for k in ('status','physical_games_completed','physical_steps')}}
        if row['status']=='COMPLETE':saved.update(wdl=row['wdl'],margin=row['margin'],actions_sha256=row['actions_sha256'])
        rows.append(saved);journal(out/'attempts.jsonl',{'event':row['status'],**saved});print(canonical(saved),flush=True)
    complete=[r for r in rows if r['status']=='COMPLETE']
    result=write_new(out/'RESULT.json',{'schema':SCHEMA,'plan_sha256':plan['content_sha256'],'requested_games':len(requests),
        'attempted_games':len(rows),'complete_games':len(complete),'failed_games':len(rows)-len(complete),
        'unstarted_games':len(requests)-len(rows),'distinct_scenarios':len({s['id'] for s in selected}),
        'distinct_scenario_seats':len({(r['scenario'],r['seat']) for r in rows}),
        'physical_steps':sum(r['physical_steps'] for r in rows),'rows':rows,
        'wdl':{k:sum(r['wdl']==k for r in complete) for k in ('WIN','DRAW','LOSS')},
        'mean_margin':sum(r['margin'] for r in complete)/len(complete) if complete else None,
        'seconds':time.monotonic()-started,'fresh_learning_games':0,'optimizer_updates':0,
        'scope':'FIXED_SCENARIO_SCRIPTED_BASELINE; NOT_NEURAL_IMPROVEMENT_OR_OFFICIAL_EVALUATION'})
    print(canonical({k:v for k,v in result.items() if k!='rows'}),flush=True)
    if result['failed_games'] or result['unstarted_games']:raise SystemExit(1)


if __name__=='__main__':main()
