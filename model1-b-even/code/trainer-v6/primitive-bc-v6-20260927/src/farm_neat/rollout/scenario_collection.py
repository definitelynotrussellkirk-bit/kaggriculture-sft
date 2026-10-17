"""Collect complete fixed-scenario KEEP/edit groups for the existing outcome learner.

Explicit authored alternatives have no behavior likelihood. All outcomes, including
losses and no-ops, remain in the dataset. This is exposed curriculum experience.
"""
from pathlib import Path
import argparse
import gzip
import json
from multiprocessing import get_context
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
from farm_neat.evaluation.common import canonical,digest,equal,file_sha,read,write_new
from farm_neat.evaluation import scenarios
from farm_neat.economy.action_space import validate_program
from farm_neat.economy.orders import FullSeasonPlan

SCHEMA='farmneat-fixed-scenario-edit-collection-v1'


def sources():
    helper=ROOT/'src/farm_neat/evaluation/common.py'
    return {**scenarios.sources(),str(Path(__file__).relative_to(ROOT)):file_sha(__file__),str(helper.relative_to(ROOT)):file_sha(helper)}


def check(path,sha):
    path=Path(path).resolve();equal(file_sha(path),sha,'registered collection')
    if not path.is_relative_to(ROOT/'work/runs'):raise ValueError('run plan belongs under work/runs')
    plan=read(path)
    if plan['schema']!=SCHEMA or not 1<=plan['workers']<=16 or not 1<=plan['seconds']<=7200:
        raise ValueError('known bounded collection required')
    equal(plan['source_pins'],sources(),'exact collection sources')
    if not 1<=len(plan['groups'])<=512:raise ValueError('bounded complete group allocation')
    cases=scenarios.catalog();seen=set()
    for group in plan['groups']:
        key=digest(group)
        if key in seen or group['index']!=len(seen):raise ValueError('unique ordered groups required')
        seen.add(key);case=plan['scenarios'][group['scenario']]
        equal(case,cases[group['scenario']],'registered case')
        equal([group['seed'],group['opponent']],[case['seed'],case['opponent']],'case identities')
        if type(group['seat']) is not int or group['seat'] not in (0,1) or group['step'] not in FullSeasonPlan.STEPS[1:]:
            raise ValueError('eligible seat and morning revision required')
        scenarios.Program(plan['programs'][group['base']],64)
        menu=plan['edits'][group['base']]
        order=plan['edit_order']
        if (not isinstance(order,list) or len(order)!=len(menu) or set(order)!=set(menu)
                or order[0]!='keep' or menu['keep']!=[{'op':'STOP'}] or 'hands_same' not in menu or not 2<=len(menu)<=32):
            raise ValueError('KEEP-first menu with explicit no-op control required')
        for program in menu.values():
            validate_program(program)
            if len(program)>64:raise ValueError('explicit64-slot budget required')
    return plan


def one_group(job):
    folder,plan,group,deadline=job;folder=Path(folder);base=group['base']
    raw=folder/f"g{group['index']:04d}.jsonl";journal=folder/f"g{group['index']:04d}.attempts.jsonl"
    if raw.exists() or journal.exists() or raw.with_suffix('.jsonl.gz').exists():raise ValueError('existing attempt is not a new collection')
    rows=[];reference=None;failed=None;started=time.monotonic();attempted=0;complete_games=0;steps=0
    for name in plan['edit_order']:
        revision=plan['edits'][base][name]
        remaining=deadline-time.monotonic()
        if remaining<=0:failed='collection deadline before next game';break
        scenarios.journal(journal,{'event':'STARTED','group':group,'edit':name})
        attempted+=1
        actor=scenarios.Program(plan['programs'][base],64,revisions={group['step']:revision},capture=True)
        game=scenarios.play(plan['scenarios'][group['scenario']],actor,group['seat'],min(120.,remaining),capture_state=True)
        steps+=game['physical_steps'];complete_games+=int(game['status']=='COMPLETE')
        if game['status']!='COMPLETE':
            failed=game['error']
            scenarios.journal(journal,{'event':'FAILED','edit':name,'result':game});break
        try:
            if game['physical_steps']!=719 or any(p['rejected'] for p in actor.publications):raise ValueError('complete unrefused game required')
            given=next(r for r in actor.records if r['step']==group['step']);given={k:v for k,v in given.items() if k!='step'}
            actions=[list(pair) for pair in zip(*game['actions_by_seat'],strict=True)]
            pair=(digest(given),digest(actions[:group['step']]))
            if reference is None:reference=pair
            equal(pair,reference,'same observed state and joint prefix')
            seat=group['seat'];cash=game['terminal_cash_by_seat'];own,other=cash[seat],cash[1-seat]
            row={'schema':SCHEMA,'group':group,'edit':name,'program':revision,'input_sha256':pair[0],
                'prefix_actions_sha256':pair[1],'input':given if name=='keep' else None,
                'own_cash':own,'opponent_cash':other,'cash_margin':own-other,
                'wdl_score':1. if own>other else .5 if own==other else 0.,
                'physical_steps':719,'actions':actions,'actions_sha256':digest(actions),
                'publications':actor.publications,'terminal_observation':game['terminal_observation'],
                'scenario_identity':game['scenario_identity'],'opponent_identity':game['opponent_identity'],
                'shop_history':game['shop_history'],'authored_not_sampled':True,'seconds':game['seconds']}
            scenarios.journal(raw,row)
            rows.append({k:v for k,v in row.items() if k not in ('input','actions','terminal_observation','publications')})
            scenarios.journal(journal,{'event':'COMPLETE','edit':name,'cash':cash,'physical_steps':719})
        except Exception as error:
            failed=repr(error)
            scenarios.journal(journal,{'event':'REJECTED_COMPLETE_GAME','edit':name,'result':game,'error':failed});break
    if failed is not None:
        return {'group':group,'complete':False,'rows':rows,'error':failed,'physical_steps':steps,
            'attempted_games':attempted,'complete_physical_games':complete_games,
            'raw_file':str(raw.relative_to(ROOT)),'journal':str(journal.relative_to(ROOT)),
            'unstarted_alternatives':len(plan['edits'][base])-attempted}
    alias=next(r for r in rows if r['edit']=='hands_same');keep=rows[0]
    same=all(alias[k]==keep[k] for k in ('actions_sha256','own_cash','opponent_cash'))
    target=raw.with_suffix('.jsonl.gz');payload=raw.read_bytes()
    with target.open('xb') as stream:stream.write(gzip.compress(payload,mtime=0))
    equal(gzip.decompress(target.read_bytes()).hex(),payload.hex(),'lossless completed records');raw.unlink()
    return {'group':group,'complete':True,'no_op_alias_verified':same,'rows':rows,
        'attempted_games':attempted,'complete_physical_games':complete_games,'physical_steps':steps,
        'file':str(target.relative_to(ROOT)),'sha256':file_sha(target),'journal':str(journal.relative_to(ROOT)),
        'journal_sha256':file_sha(journal),'seconds':time.monotonic()-started}


def reduction(groups):
    result={}
    for group in groups:
        if not group['complete']:continue
        rows=group['rows'];keep=rows[0];key=group['group']['scenario']
        section=result.setdefault(key,{'groups':0,'games':0,'wins':0,'draws':0,'losses':0,
            'groups_with_win_and_loss':0,'groups_with_wdl_preference':0,'edits':{}})
        section['groups']+=1;section['games']+=len(rows)
        values={r['wdl_score'] for r in rows}
        section['groups_with_win_and_loss']+=int(0. in values and 1. in values)
        section['groups_with_wdl_preference']+=int(len(values)>1)
        for row in rows:
            section['wins' if row['wdl_score']==1 else 'draws' if row['wdl_score']==.5 else 'losses']+=1
            stats=section['edits'].setdefault(row['edit'],{'games':0,'wins':0,'draws':0,'losses':0,
                'margin_helps':0,'margin_harms':0,'margin_ties':0,'delta_margin_sum':0.})
            stats['games']+=1;stats['wins' if row['wdl_score']==1 else 'draws' if row['wdl_score']==.5 else 'losses']+=1
            delta=row['cash_margin']-keep['cash_margin'];stats['delta_margin_sum']+=delta
            stats['margin_helps' if delta>0 else 'margin_harms' if delta<0 else 'margin_ties']+=1
    return result


def run(path,sha):
    plan=check(path,sha);out=Path(path).resolve().parent
    write_new(out/'STARTED.json',{'plan_sha256':sha,'authored_not_policy_samples':True})
    folder=out/'groups';folder.mkdir(exist_ok=False);started=time.monotonic();groups=[]
    jobs=[(folder,plan,g,started+plan['seconds']) for g in plan['groups']]
    with get_context('spawn').Pool(plan['workers']) as pool:
        for group in pool.imap_unordered(one_group,jobs,chunksize=1):
            groups.append(group);scenarios.journal(out/'completed.jsonl',group)
            print(canonical({'groups':len(groups),'complete':sum(g['complete'] for g in groups),'seconds':time.monotonic()-started}),flush=True)
    groups.sort(key=lambda g:g['group']['index']);check(path,sha)
    complete=all(g['complete'] and g['no_op_alias_verified'] for g in groups)
    attempted=sum(json.loads(line)['event']=='STARTED' for p in folder.glob('*.attempts.jsonl') for line in p.read_text().splitlines())
    games=sum(len(g['rows']) for g in groups)
    result=write_new(out/'RESULT.json',{'schema':SCHEMA,'plan_sha256':sha,'complete':complete,'groups':groups,
        'summary':reduction(groups),'requested_games':sum(len(plan['edits'][g['base']]) for g in plan['groups']),
        'attempted_games':attempted,'complete_physical_games':sum(g['complete_physical_games'] for g in groups),
        'admitted_game_rows':games,'physical_steps':sum(g['physical_steps'] for g in groups),
        'distinct_scenarios':len({g['scenario'] for g in plan['groups']}),
        'distinct_scenario_seats':len({(g['scenario'],g['seat']) for g in plan['groups']}),
        'authored_edit_decisions':games,'sampled_policy_decisions':0,'optimizer_updates':0,
        'seconds':time.monotonic()-started,'scope':'EXPOSED_FIXED_SCENARIO_CURRICULUM; NOT_INDEPENDENT_WORLDS_OR_ORIGINAL_RULE_STRENGTH'})
    print(canonical({k:v for k,v in result.items() if k not in ('groups','summary')}),flush=True)
    if not complete:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('check','run'))
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args()
    if args.command=='check':check(args.plan,args.plan_sha256);print('REGISTERED; NO GAMES OR UPDATES')
    else:run(args.plan,args.plan_sha256)


if __name__=='__main__':main()
