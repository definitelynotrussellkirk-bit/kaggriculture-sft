"""Prospective seven-opponent W/D/L panel, fixed-shop CRN plus original rules.

Defaults use the four DISCOVERY worlds only, not protected confirmation/monitor
seeds. Original rules on discovery seeds are explicitly diagnostic. No seed
rerolls, silent missing rivals, training, promotion, GPU, or opponent pooling.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from league import (HOME, ROOT, League, MissingOpponent, OPPONENTS, DEFAULT_CONTROLS,
                    legacy, freerun, digest, seal, sha_file, canonical, runtime_pins)
from bridges import play_evaluation_case, bridge_identity

SCHEDULE = ROOT / 'project/configs/v0-search-crn-schedules.json'
SCHEDULE_SHA256 = '9301fd90d2fde7577d84f33bd6e680548ae160d21649e950fce9ef582ee978a9'


def validate_schedule(config):
    if config.get('schema') != 'farm-v0-search-crn-schedules-v1' or config.get('seats') != [0, 1]:
        raise ValueError('known both-seat CRN schedule required')
    protected = {w['seed'] for split in ('confirmation', 'monitor') for w in config['worlds'][split]}
    seen = set()
    for world in config['worlds']['discovery']:
        shops = world['shops']
        raw = (json.dumps(shops, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode()
        if hashlib.sha256(raw).hexdigest() != world['schedule_sha256']:
            raise ValueError('fixed-shop schedule digest mismatch')
        if len(shops) != 8 or any(s not in config['sampling']['shops'] for s in shops):
            raise ValueError('eight registered shops required')
        if world['seed'] in protected or world['seed'] in seen:
            raise ValueError('discovery seed collision with protected/duplicate world')
        seen.add(world['seed'])
    return copy.deepcopy(config)


def load_schedule():
    if sha_file(SCHEDULE) != SCHEDULE_SHA256:
        raise ValueError('assigned schedule file SHA-256 changed; do not silently rebind')
    return validate_schedule(legacy.read_json(SCHEDULE))


def make_plan(league, *, candidate, smoke=False, sampler_seed=9800001, controls=None):
    if type(sampler_seed) is not int or not 0 <= sampler_seed < 2**63:
        raise ValueError('bounded sampler seed required')
    config = load_schedule()
    controls = copy.deepcopy(controls or DEFAULT_CONTROLS)
    if controls.get('device') != 'cpu':
        raise ValueError('panel is CPU-only')
    # Reuse the existing controls validation, including all explicit dawns.
    if controls.get('publication_steps') != list(range(0, 719, 24)):
        raise ValueError('all 30 dawns required')
    freerun.validate_spec({
        'schema':'public-winner-bc-comparison-spec-v1', 'world_seeds':[0],
        'sampler_seed':sampler_seed, 'opponents':list(freerun.OPPONENTS),
        'rules':[{'id':'control-validation-original','kind':'ORIGINAL'},
                 {'id':'control-validation-fixed','kind':'FIXED_SHOP',
                  'shops':config['worlds']['discovery'][0]['shops'],'random_weeds':True}],
        'controls':controls})
    candidate_id = league.describe(candidate)
    opponents = {name: league.describe(name) for name in OPPONENTS}
    worlds = config['worlds']['discovery'][:1] if smoke else config['worlds']['discovery']
    cells = []
    for world in worlds:
        rules = [
            {'id': world['world_id']+'-fixed', 'kind':'FIXED_SHOP',
             'shops':world['shops'], 'random_weeds':True},
            {'id': world['world_id']+'-original', 'kind':'ORIGINAL'},
        ]
        for rule in rules:
            for opponent in OPPONENTS:
                for seat in (0, 1):
                    pair = {'base':sampler_seed, 'world_id':world['world_id'],
                            'rule':rule['kind'], 'seat':seat}
                    stream = lambda role: int(digest({**pair, 'role':role})[:16],16) % 2**63
                    cell = {'candidate':candidate, 'opponent':opponent,
                        'seat':seat, 'world_seed':world['seed'], 'world_id':world['world_id'],
                        'rule':copy.deepcopy(rule), 'sampler_seed':stream('candidate'),
                        'opponent_sampler_seed':stream('opponent/'+opponent),
                        'schedule_sha256':world['schedule_sha256'] if rule['kind']=='FIXED_SHOP' else None,
                        'exposure':'DISCOVERY_DIAGNOSTIC_NOT_HELDOUT_CONFIRMATION'}
                    cell['case_id'] = digest({'cell':cell,'candidate':candidate_id,
                                              'opponent':opponents[opponent]})[:24]
                    cells.append(cell)
    from farm_neat.game.exact_engine import assert_bundled_engine_identity
    return seal({'schema':'slot98-win-mass-panel-plan-v1', 'candidate':candidate_id,
        'opponents':opponents, 'opponent_order':list(OPPONENTS), 'cases':cells,
        'controls':controls, 'smoke':smoke, 'sampler_seed':sampler_seed,
        'schedule_path':str(SCHEDULE.relative_to(ROOT)), 'schedule_file_sha256':SCHEDULE_SHA256,
        'engine':assert_bundled_engine_identity(), 'runtime_pins':{
            **runtime_pins(), str((HOME/'bridges.py').relative_to(ROOT)):sha_file(HOME/'bridges.py'),
            str((HOME/'panel.py').relative_to(ROOT)):sha_file(HOME/'panel.py')},
        'bridge':bridge_identity('slot81'),
        'score':'W=1; D=0.5; L=0; win_mass=(W+0.5*D)/complete_games',
        'pairing':'same world/seat/candidate stream across opponent arms; fixed shops identical across arms',
        'original_rules':'Native shop draws may differ across actions; no CRN shop-equality claim',
        'selection':'all registered discovery worlds, or first world for explicit smoke; no outcome filtering',
        'freshness':'exposed discovery diagnostic; protected confirmation/monitor worlds not opened',
        'training_admission':False, 'promotion':False, 'automatic_retry':False})


def summarize(plan, rows):
    legacy.check_bound(plan, 'panel plan')
    by_id = {}
    planned = {c['case_id']:c for c in plan['cases']}
    for row in rows:
        key = row['case']['case_id']
        if key in by_id:
            raise ValueError('duplicate result / implicit retry refused')
        if key not in planned or row['case'] != planned[key]:
            raise ValueError('row differs from prospective case')
        if row['status'] not in ('COMPLETE','FAILED','BLOCKED'):
            raise ValueError('unknown row status')
        if row['status']=='COMPLETE':
            if (row.get('mode')!='NATIVE' or row.get('physical_steps')!=719
                    or row.get('physical_games_started')!=1 or row.get('physical_games_completed')!=1
                    or row.get('terminal_statuses')!=['DONE','DONE']):
                raise ValueError('complete native 719-transition terminal game required')
            cash=row.get('terminal_cash_by_seat')
            if (not isinstance(cash,list) or len(cash)!=2 or
                    any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in cash)):
                raise ValueError('two finite nonnegative terminal cash values required')
            seat=row['case']['seat'];margin=cash[seat]-cash[1-seat]
            expected='W' if margin>0 else 'L' if margin<0 else 'D'
            if (row.get('wdl')!=expected or row.get('margin')!=margin
                    or row.get('match_score')!={'W':1.0,'D':0.5,'L':0.0}[expected]):
                raise ValueError('terminal cash, WDL and win-mass score disagree')
        elif any(k in row for k in ('wdl','match_score','terminal_cash_by_seat')):
            raise ValueError('censored result cannot carry a terminal outcome')
        by_id[key] = row
    groups = {}
    for case in plan['cases']:
        key = (case['rule']['id'],case['opponent'],case['seat'])
        group = groups.setdefault(key, {'rule':case['rule'], 'world_id':case['world_id'],
            'opponent':case['opponent'], 'candidate_seat':case['seat'],
            'planned_games':0, 'complete_games':0, 'failed_games':0,
            'blocked_games':0, 'not_started_games':0, 'wdl':{'W':0,'D':0,'L':0}})
        group['planned_games'] += 1
        row = by_id.get(case['case_id'])
        if row is None:
            group['not_started_games'] += 1
        elif row['status'] == 'COMPLETE':
            group['complete_games'] += 1
            group['wdl'][row['wdl']] += 1
        elif row['status'] == 'BLOCKED':
            group['blocked_games'] += 1
        else:
            group['failed_games'] += 1
    strata = list(groups.values())
    for group in strata:
        mass = group['wdl']['W'] + 0.5*group['wdl']['D']
        group['win_mass_sum'] = mass if group['complete_games'] else None
        group['win_mass'] = mass/group['complete_games'] if group['complete_games'] else None
        group['all_planned_complete'] = group['complete_games']==group['planned_games']
    complete = sum(r['status']=='COMPLETE' for r in rows)
    return seal({'schema':'slot98-win-mass-panel-result-v1',
        'plan_sha256':plan['content_sha256'],
        'status':'COMPLETE' if complete==len(plan['cases']) else 'PARTIAL',
        'counts':{'planned':len(plan['cases']), 'attempted_rows':len(rows),
            'physical_games_started':sum(r['physical_games_started'] for r in rows),
            'physical_games_completed':sum(r['physical_games_completed'] for r in rows),
            'complete':complete, 'failed':sum(r['status']=='FAILED' for r in rows),
            'blocked':sum(r['status']=='BLOCKED' for r in rows),
            'not_started':len(plan['cases'])-len(rows)},
        'strata':strata, 'pooled_across_opponents':False,
        'limits':['Discovery diagnostics, not held-out strength or promotion evidence.',
                  'Incomplete cells are not assigned wins, draws or losses.',
                  'Native original-rule shops need not match across arms.']})


def charge_native_start():
    result = subprocess.run([sys.executable,'-B',str(ROOT/'src/farm_neat/tooling/review_package.py'),
                             '--charge-native-start'],cwd=ROOT,
                            capture_output=True,text=True,check=False)
    if result.returncode:
        raise RuntimeError('native-start accounting failed: '+result.stderr[-2000:]+result.stdout[-2000:])


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def run_panel(plan, league, output):
    legacy.check_bound(plan,'panel plan')
    for rel,sha in plan['runtime_pins'].items():
        if sha_file(ROOT/rel)!=sha:
            raise ValueError('panel source changed after registration: '+rel)
    output = Path(output).resolve()
    if not output.is_relative_to(HOME):
        raise ValueError('result output must stay inside opponent-league-adapters')
    output.mkdir(parents=True,exist_ok=False)  # No overwrite, resume/reroll, or stale result reuse.
    legacy.write_new(output/'PLAN.json',plan)
    rows = []
    atomic_json(output/'RESULT.json',summarize(plan,rows))
    with (output/'games.jsonl').open('x') as journal, (output/'starts.jsonl').open('x') as starts:
        for index,case in enumerate(plan['cases']):
            starts.write(canonical({'type':'CASE_ATTEMPT','case_id':case['case_id']})+'\n')
            starts.flush();os.fsync(starts.fileno())
            unavailable = [desc for desc in (plan['candidate'],plan['opponents'][case['opponent']])
                           if desc['status']!='READY']
            if unavailable:
                row={'case':copy.deepcopy(case),'status':'BLOCKED','mode':'NATIVE',
                     'physical_games_started':0,'physical_games_completed':0,'physical_steps':0,
                     'error':'; '.join(d['error'] for d in unavailable),'no_automatic_retry':True}
            else:
                # Identity is frozen before outcomes and verified again before every arm.
                for name,expected in ((case['candidate'],plan['candidate']['identity']),
                                      (case['opponent'],plan['opponents'][case['opponent']]['identity'])):
                    if league.identity(name)!=expected:
                        raise ValueError('registered actor identity drift: '+name)
                trace=None
                row={'case':copy.deepcopy(case),'status':'FAILED','mode':'NATIVE',
                     'physical_games_started':0,'physical_games_completed':0,'physical_steps':0,
                     'no_automatic_retry':True}
                def invalidate(error):
                    row.update(status='FAILED',physical_games_completed=0,
                               error={'type':type(error).__name__,'message':str(error)})
                    for key in ('terminal_cash_by_seat','terminal_statuses','margin','wdl','match_score'):
                        row.pop(key,None)
                try:
                    trace=freerun.TraceStore(output/(case['case_id']+'.trace.jsonl.gz'),
                                            plan['controls']['trace_byte_limit'])
                    row=play_evaluation_case(case,league=league,controls=plan['controls'],trace=trace,
                                             on_native_start=charge_native_start)
                    trace.flush()
                except (Exception,KeyboardInterrupt) as error:
                    invalidate(error)
                finally:
                    if trace is not None:
                        try:
                            trace.close()
                        except (Exception,KeyboardInterrupt) as error:
                            invalidate(error)
            journal.write(canonical(row)+'\n');journal.flush();os.fsync(journal.fileno())
            rows.append(row)
            atomic_json(output/'RESULT.json',summarize(plan,rows))
            print(f"{index+1}/{len(plan['cases'])} {case['rule']['id']} {case['opponent']} "
                  f"seat={case['seat']} {row['status']} {row.get('wdl','-')}",flush=True)
    for rel,sha in plan['runtime_pins'].items():
        if sha_file(ROOT/rel)!=sha:
            raise ValueError('panel source changed during execution: '+rel)
    return summarize(plan,rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',choices=(*OPPONENTS,'CANDIDATE'),default='SHOP_ROUTER')
    parser.add_argument('--first-bc-binding',type=Path,
                        help='Exact checkpoint_adapter binding JSON; no weights means blocked FIRST_BC cells')
    parser.add_argument('--candidate-binding',type=Path,help='Frozen learned candidate binding; use --candidate CANDIDATE')
    parser.add_argument('--smoke',action='store_true',help='First discovery schedule + native original rules, all rivals/both seats')
    parser.add_argument('--sampler-seed',type=int,default=9800001)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--prepare-only',action='store_true',help='Write plan only; no game or weight load')
    args=parser.parse_args(argv)
    if args.candidate_binding is not None and args.candidate!='CANDIDATE':
        parser.error('--candidate-binding requires --candidate CANDIDATE; no ignored candidate weights')
    league=League(first_bc_binding=legacy.read_json(args.first_bc_binding) if args.first_bc_binding else None,
                  candidate_binding=legacy.read_json(args.candidate_binding) if args.candidate_binding else None)
    plan=make_plan(league,candidate=args.candidate,smoke=args.smoke,sampler_seed=args.sampler_seed)
    if args.prepare_only:
        if not args.output.resolve().is_relative_to(HOME):
            parser.error('output must stay inside this task directory')
        legacy.write_new(args.output,plan)
        print(f"planned {len(plan['cases'])} cells; no games executed")
        return 0
    result=run_panel(plan,league,args.output)
    print(json.dumps({'status':result['status'],'counts':result['counts']},indent=2))
    return 0 if result['status']=='COMPLETE' else 2


if __name__=='__main__':
    raise SystemExit(main())
