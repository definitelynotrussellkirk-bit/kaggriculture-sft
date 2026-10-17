"""CPU ranker fit/calibration and exact audit; a consumer, not an actor trainer.

fit_ranker(plan: dict, *, output) -> dict writes a new directory containing
MODEL.json and RESULT.json. The model is a small diagnostic linear ranker, not a
V0 actor/checkpoint. Input is a SHA-bound JSONL corpus of closed envelopes:
{pool_id, candidate_id, selected_by_search: bool, adaptive_search: bool, record}.
record is the EXISTING v0_regret_records wire format, including rejected/tied/
losing comparisons. No predicted, censored or unreplayed outcome is a target.
All observed exclusions and exact duplicates remain in the receipt. No outcome is
imputed for unexecuted candidates. Authentic execution/chronology is an external
attestation: record validation does not itself rerun the simulator.

Splits are whole connected components of source families, parent checkpoint/draw,
candidate families, rounds, world hashes and pools. Explicit partitions cannot
cut a component; automatic partitions depend on metadata ONLY. Ridge selection
uses dev, residual calibration uses calibration, test never changes the model.
Calibration uses one MAX absolute residual per connected calibration component;
the finite-sample quantile needs enough components. Adaptive distribution shift
invalidates exchangeability: intervals are diagnostics, not admission bounds.

CLI: python -m farm_neat.training.v0_regret_rank {fit,order,audit} --plan PLAN
     --output NEW_PATH. Paths inside plans resolve relative to the plan file.
All commands start zero native games. Real fitting/auditing is maintainer-only
under the pinned runtime/storage/SLURM contract. Fixture work requires explicit
fixture_only and can never be reported as native policy strength.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from farm_neat.policy import v0_regret_comparator as C
from farm_neat.training import v0_regret_records as R

FIT_SCHEMA='farm-v0-regret-rank-fit-plan-v1'
MC_FIT_SCHEMA='farm-v0-regret-mc-value-fit-plan-v1'
ORDER_PLAN_SCHEMA='farm-v0-regret-rank-order-plan-v1'
AUDIT_PLAN_SCHEMA='farm-v0-regret-rank-audit-plan-v1'
SPLITS=('train','dev','calibration','test')
MAX_FILE_BYTES=128*1024*1024
MAX_RECORDS=10000


def _read_bound(ref: dict) -> bytes:
    C.need(type(ref) is dict and set(ref)=={'path','sha256'}, 'FILE_REFERENCE_FIELDS')
    path=Path(ref['path'])
    C.need(path.stat().st_size<=MAX_FILE_BYTES,'INPUT_BYTE_BUDGET')
    data=path.read_bytes()
    C.need(hashlib.sha256(data).hexdigest()==ref['sha256'],'INPUT_HASH_MISMATCH')
    return data


def _load_corpus(ref, maximum):
    C.need(type(maximum) is int and 0<=maximum<=MAX_RECORDS,'RECORD_BUDGET')
    lines=_read_bound(ref).decode('utf-8').splitlines()
    C.need(len(lines)<=maximum,'RECORD_BUDGET')
    return [R._loads(line) for line in lines if line.strip()]


def _delta(record):
    # The record validator recomputes this mean from complete per-world WDL rows.
    return float(record['comparison']['mean_wdl_delta'])


def validate_rows(rows: list[dict], *, fixture_only: bool, discovery_only: bool=False) -> dict:
    """Read-only validation of existing paired facts; retain all nonwinner signs."""
    C.need(type(fixture_only) is bool and type(discovery_only) is bool and
           type(rows) is list and len(rows)<=MAX_RECORDS,
           'CORPUS_OPTIONS_OR_BUDGET')
    usable,excluded,seen=[],[],set()
    pair_ids, attempts, candidate_bindings = {},{},{}
    for index,item in enumerate(rows):
        required = {'pool_id','candidate_id','selected_by_search','adaptive_search','record'}
        C.need(type(item) is dict and required <= set(item) and
               not set(item)-required-({'source_family_id'} if discovery_only else set()), 'CORPUS_ROW_FIELDS')
        if 'source_family_id' in item:
            C.need(type(item['source_family_id']) is str and bool(item['source_family_id']), 'SOURCE_FAMILY_ID')
        C.need(all(type(item[k]) is str and item[k] for k in ('pool_id','candidate_id')) and
               all(type(item[k]) is bool for k in ('selected_by_search','adaptive_search')),
               'CORPUS_ROW_TYPES')
        if discovery_only:
            raw = item['record']
            C.need(type(raw) is dict and raw.get('identity',{}).get('split') == 'discovery' and
                   raw.get('evaluation',{}).get('mode') != 'CONFIRMED_EXPECTATION' and
                   raw.get('evaluation',{}).get('confirmation_world_groups') == [] and
                   raw.get('evaluation',{}).get('confirmation_registration') is None and
                   raw.get('challenger',{}).get('selection_receipt',{}).get('confirmation_registration_sha256') is None,
                   'DISCOVERY_ONLY_CORPUS')
        record=R.validate_pair(item['record'])
        ident,ev,comp=record['identity'],record['evaluation'],record['comparison']
        C.need(ident['split'] not in ('monitor','final'),'PROTECTED_EVALUATION_SPLIT')
        synthetic='SYNTHETIC_DIAGNOSTIC' in comp['reason_codes']
        C.need(fixture_only or not synthetic,'SYNTHETIC_EVIDENCE')
        h=record['content_sha256']
        if h in seen:
            excluded.append({'index':index,'reason':'EXACT_DUPLICATE','record_sha256':h})
            continue
        seen.add(h)
        identity=ident['pair_id']
        C.need(identity not in pair_ids or pair_ids[identity]==h,'CONFLICTING_PAIR_ID')
        pair_ids[identity]=h
        # Reused attempt IDs across distinct records are not new observations.
        for attempt in ev['attempt_ids']:
            C.need(attempt not in attempts or attempts[attempt]==h,'REUSED_EXECUTION_ATTEMPT')
            attempts[attempt]=h
        binding=(ident['actor_context_sha256'],ident['parent_checkpoint_sha256'],
                 record['learner']['path_sha256'],
                 R.canonical_sha256(C.public_proposal(record['challenger'])))
        key=(item['pool_id'],item['candidate_id'])
        C.need(key not in candidate_bindings or candidate_bindings[key]==binding,'CANDIDATE_ID_REBOUND')
        candidate_bindings[key]=binding
        reason=None
        if ev['mode'] not in ('EXACT_REALIZATION','CONFIRMED_EXPECTATION'):
            reason='NOT_EXECUTED'
        elif not ev['complete']:
            reason='INCOMPLETE_EXECUTION'
        elif not ev['replay_verified']:
            reason='UNVERIFIED_EXECUTION'
        elif comp['mean_wdl_delta'] is None:
            reason='NO_OBSERVED_WDL'
        if reason:
            excluded.append({'index':index,'reason':reason,'record_sha256':h})
        else:
            usable.append(copy.deepcopy(item))
    return {'usable':usable,'excluded':excluded}


def _tokens(item):
    r=item['record'];i=r['identity'];p=r['challenger'];ev=r['evaluation']
    worlds=ev['discovery_world_groups']+ev['confirmation_world_groups']
    parent=i['parent_checkpoint_sha256']
    return {'source_family':{i['source_group_id'],item.get('source_family_id',i['source_group_id']),
                              *(w['source_group_id'] for w in worlds)},
        'parent':{parent,*('draw:'+parent+':'+x for x in p['parent_proposal_ids']),
                  'draw:'+parent+':'+r['learner']['direct_draw_id']},
        'candidate_family':{p['proposal_family']},'round':{i['round_id']},
        'world':{w['world_sha256'] for w in worlds},'pool':{item['pool_id']}}


def split_families(rows, *, partitions=None, seed=0) -> dict:
    """No label reads. A transitive A-parent-B-family-C chain stays in one split."""
    C.need(type(seed) is int and seed>=0,'SPLIT_SEED')
    parents=list(range(len(rows)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    owners={}; tokens=[_tokens(item) for item in rows]
    for index,tok in enumerate(tokens):
        for dim,values in tok.items():
            for value in values:
                key=(dim,value)
                if key in owners: parents[root(index)]=root(owners[key])
                else: owners[key]=index
    components={}
    for i in range(len(rows)): components.setdefault(root(i),[]).append(i)
    if partitions is not None:
        C.need(type(partitions) is dict and set(partitions)=={x['pool_id'] for x in rows} and
               all(s in SPLITS for s in partitions.values()),'PARTITION_MEMBERSHIP')
    groups=[]
    for members in components.values():
        signature=sorted({(dim,value) for index in members for dim,values in tokens[index].items() for value in values})
        group_id=R.canonical_sha256(signature)
        labels={partitions[rows[i]['pool_id']] for i in members} if partitions is not None else set()
        C.need(len(labels)<=1,'SPLIT_LEAKAGE')
        groups.append((group_id,members,next(iter(labels)) if labels else None))
    groups.sort(key=lambda g:R.canonical_sha256([seed,g[0]]))
    indices={s:[] for s in SPLITS}; group_for_index={}; group_splits={}
    manifest={s:{d:set() for d in ('source_family','parent','candidate_family','round','world','pool')} for s in SPLITS}
    for offset,(gid,members,label) in enumerate(groups):
        label=label or SPLITS[offset%4]
        indices[label].extend(members);group_splits[gid]=label
        for index in members:
            group_for_index[index]=gid
            for dim,values in tokens[index].items(): manifest[label][dim].update(values)
    return {'indices':{s:sorted(v) for s,v in indices.items()},'group_for_index':group_for_index,
        'group_splits':group_splits,'component_count':len(groups),
        'tokens':{s:{d:sorted(v) for d,v in dims.items()} for s,dims in manifest.items()},
        'method':'CONNECTED_SOURCE_PARENT_CANDIDATE_ROUND_WORLD_POOL','seed':seed}


def _features(rows):
    values=[]
    for start in range(0,len(rows),C.MAX_BATCH):
        batch=rows[start:start+C.MAX_BATCH]
        values.extend(C.encode_pairs([x['record']['actor_context'] for x in batch],
            [{'learner':x['record']['learner'],'challenger':x['record']['challenger']} for x in batch])['features'])
    return np.asarray(values,dtype=np.float64).reshape((-1,C.FEATURE_WIDTH))


def _weighted_mse(y,pred,groups):
    errors={}
    for target,guess,group in zip(y,pred,groups): errors.setdefault(group,[]).append(float((target-guess)**2))
    return math.fsum(math.fsum(v)/len(v) for v in errors.values())/len(errors) if errors else None


def _calibrate(y,pred,groups,coverage):
    residual={}
    for target,guess,group in zip(y,pred,groups):
        residual[group]=max(residual.get(group,0.0),abs(float(target-guess)))
    ordered=sorted(residual.values());k=math.ceil((len(ordered)+1)*coverage)
    enough=bool(ordered) and k<=len(ordered)
    return {'status':'CALIBRATED' if enough else 'UNCALIBRATED',
        'radius':ordered[k-1] if enough else 4.0,'coverage':coverage,'group_count':len(ordered),
        'order_statistic':k,'unit':'MAX_ABSOLUTE_RESIDUAL_PER_CONNECTED_FAMILY',
        'guarantee':'NONE_UNDER_ADAPTIVE_DISTRIBUTION_SHIFT',
        'residuals_sha256':R.canonical_sha256(residual)}


def _metrics(y,pred,radius,adaptive):
    if not len(y): return {'records':0,'interval_coverage':None,'adaptive_records':0,'reliability':[]}
    y=np.asarray(y);pred=np.asarray(pred)
    edges=(-2.,-1.,-.25,.25,1.,2.0000000001)
    bins=[]
    for low,high in zip(edges,edges[1:]):
        mask=(pred>=low)&(pred<high)
        if not mask.any():continue
        bins.append({'lower':low,'upper':min(2.,high),'n':int(mask.sum()),
            'mean_prediction':float(pred[mask].mean()),'mean_observed_delta':float(y[mask].mean()),
            'mae':float(np.abs(pred[mask]-y[mask]).mean())})
    mask=np.asarray(adaptive,dtype=bool)
    return {'records':len(y),'mse':float(np.mean((pred-y)**2)),
        'interval_coverage':float(np.mean(np.abs(pred-y)<=radius+1e-12)),
        'adaptive_records':int(mask.sum()),
        'adaptive_interval_coverage':float(np.mean(np.abs(pred[mask]-y[mask])<=radius+1e-12)) if mask.any() else None,
        'reliability':bins}


def audit_selection(selection: dict, rows: list[dict], *, fixture_only: bool=False) -> dict:
    """Join a frozen random audit to replay-attested outcomes; missing stays pending.

    The positive-regret estimate is m/a * sum(max(0, observed_delta)) on the exact
    uniform complement sample. It estimates finite-pool missed realized yield,
    not expected superiority and not a certificate about unexecuted candidates.
    """
    C.check_seal(selection,'ORDER_HASH_MISMATCH')
    C.need(selection.get('schema')==C.ORDER_SCHEMA,'ORDER_SCHEMA')
    pool=selection['pool'];C.need(R.canonical_sha256(pool)==selection['pool_sha256'],'POOL_HASH_MISMATCH')
    ids=[x['candidate_id'] for x in pool['candidates']]
    selected=selection['selected_ids'];nonselected=selection['nonselected_ids'];audit=selection['audit']
    C.need(0<len(ids)<=C.MAX_CANDIDATES and
           type(selection['shortlist_size']) is int and 1<=selection['shortlist_size']<=C.MAX_CANDIDATES and
           type(selection['audit_size']) is int and 1<=selection['audit_size']<=C.MAX_CANDIDATES and
           type(audit['seed']) is int and audit['seed']>=0 and
           type(selection['fixture_only']) is bool, 'AUDIT_OPTIONS')
    C.need(len(ids)==len(set(ids)) and len(selected)==len(set(selected)) and
           set(selected).isdisjoint(nonselected) and set(selected)|set(nonselected)==set(ids) and
           len(nonselected)==len(set(nonselected)), 'SELECTION_PARTITION')
    C.need(audit['algorithm']==C.AUDIT_ALGORITHM and audit['candidate_ids']==C.audit_draw(
        selection['pool_sha256'],nonselected,selection['audit_size'],audit['seed']) and
        audit['denominator']==len(nonselected) and audit['numerator']==len(audit['candidate_ids']),
        'AUDIT_DRAW_MISMATCH')
    C.need(selection['ordered_ids'][:selection['shortlist_size']]==selected and
           sorted(selection['ordered_ids'])==sorted(ids),'SELECTION_ORDER_MISMATCH')
    C.need(selection['paired_confirmation_required'] is True and type(selection['confirmation_n']) is int and
           selection['confirmation_n']>=1 and selection['executed_pairs']==0,'ADMISSION_BOUNDARY')
    C.need(fixture_only or not selection['fixture_only'],'SYNTHETIC_EVIDENCE')
    clean=validate_rows(rows,fixture_only=fixture_only)
    bindings=selection['evidence_binding']; candidates={x['candidate_id']:x['proposal'] for x in pool['candidates']}
    grouped={};hashes=[]
    C.need(len({x['pool_id'] for x in clean['usable']})<=1,'AUDIT_MULTIPLE_POOLS')
    for item in clean['usable']:
        candidate=item['candidate_id'];r=item['record'];i=r['identity']
        C.need(candidate in candidates and r['actor_context']==pool['actor_context'] and
               C.public_proposal(r['learner'])==pool['learner'] and
               C.public_proposal(r['challenger'])==candidates[candidate],'AUDIT_PATH_CONTEXT_MISMATCH')
        C.need(bindings=={'learner_path_sha256':r['learner']['path_sha256'],
            'direct_draw_id':r['learner']['direct_draw_id'],
            'parent_checkpoint_sha256':i['parent_checkpoint_sha256']},'AUDIT_LEARNER_BINDING')
        grouped.setdefault(candidate,{}).setdefault(i['source_group_id'],[]).append(_delta(r))
        hashes.append(r['content_sha256'])
    values={c:math.fsum(math.fsum(v)/len(v) for v in groups.values())/len(groups) for c,groups in grouped.items()}
    missing_selected=[c for c in selected if c not in values]
    missing_audit=[c for c in audit['candidate_ids'] if c not in values]
    chosen=[values[c] for c in selected if c in values]
    audit_values=[values[c] for c in audit['candidate_ids'] if c in values]
    complete=not missing_selected and not missing_audit
    m,a=len(nonselected),len(audit['candidate_ids'])
    selected_mean=math.fsum(chosen)/len(chosen) if chosen else None
    return C.seal({'schema':'farm-v0-regret-rank-audit-v1','status':'COMPLETE' if complete else 'PENDING_EXECUTION',
        'scope':'FIXTURE_DIAGNOSTIC' if fixture_only else 'REPLAY_ATTESTED_REALIZED_WDL',
        'selection_sha256':selection['content_sha256'],'record_sha256s':hashes,
        'missing_selected_ids':missing_selected,'missing_audit_ids':missing_audit,
        'selected_records':len(chosen),'random_audit_records':len(audit_values),
        'top_k_verified_mean_delta':selected_mean if not missing_selected else None,
        'top_k_verified_positive_regret':math.fsum(max(0,d) for d in chosen)/len(chosen) if chosen and not missing_selected else None,
        'oracle_gap':(max(0.,max(values.values()))-max(0.,max(chosen))) if len(values)==len(ids) and chosen else None,
        'missed_positive_total_estimate':(m/a*math.fsum(max(0,d) for d in audit_values) if a else 0.0)
            if not missing_audit else None,
        'verified_deltas':values,'excluded':clean['excluded'],
        'paired_confirmation_required':True,'confirmation_n':selection['confirmation_n'],
        'native_games_started':0,
        'limitation':'Executed corpus audit; no proof of pre-exposure chronology or independent expected improvement.'})


def _pool_audits(model,rows,plan):
    pools={}
    for item in rows: pools.setdefault(item['pool_id'],[]).append(item)
    results=[]
    for pool_id,items in sorted(pools.items()):
        first=items[0]['record'];candidates={}
        for item in items:
            candidates.setdefault(item['candidate_id'],{'candidate_id':item['candidate_id'],
                                                        'proposal':item['record']['challenger']})
        ordering=C.order_proposals(model,first['actor_context'],first['learner'],list(candidates.values()),
            shortlist_size=plan['shortlist_size'],audit_size=plan['audit_size'],seed=plan['audit_seed'],
            confirmation_n=plan['confirmation_n'],allow_fixture=plan['fixture_only'])
        result=audit_selection(ordering,[{k:v for k,v in item.items() if k != 'source_family_id'}
                                          for item in items],fixture_only=plan['fixture_only'])
        results.append({'pool_id':pool_id,'selection':ordering,'result':result})
    return results


def fit_ranker(plan: dict, *, output) -> dict:
    """Fit/calibrate on disjoint families, then audit held-out searched proposals.

    The output directory must be new. MODEL/RESULT commit together by rename;
    errors leave no half model. No hidden checkpoint/GPU/native-game calls.
    """
    output=Path(output)
    if output.exists(): raise FileExistsError(str(output))
    R._safe(plan)
    required={'schema','corpus','fixture_only','split_seed','ridge_grid','coverage','shortlist_size',
              'audit_size','audit_seed','confirmation_n','max_records'}
    C.need(type(plan) is dict and required<=set(plan) and not set(plan)-required-{'partitions'} and
           plan['schema'] in (FIT_SCHEMA,MC_FIT_SCHEMA),'FIT_PLAN_FIELDS')
    C.need(type(plan['coverage']) in (int,float) and .5<plan['coverage']<1 and
           type(plan['ridge_grid']) is list and 0<len(plan['ridge_grid'])<=8 and
           all(type(x) in (int,float) and math.isfinite(x) and 0<x<=1e6 for x in plan['ridge_grid']),
           'FIT_NUMERICS')
    raw=_load_corpus(plan['corpus'],plan['max_records'])
    discovery_only = plan['schema'] == MC_FIT_SCHEMA
    clean=validate_rows(raw,fixture_only=plan['fixture_only'],discovery_only=discovery_only);rows=clean['usable']
    # Validate requested partitions against the ENTIRE corpus, including exclusions.
    whole=split_families(raw,partitions=plan.get('partitions'),seed=plan['split_seed'])
    by_hash={x['record']['content_sha256']:i for i,x in enumerate(raw)}
    indices={s:[] for s in SPLITS};groups=[]
    split_for_index={i:s for s,ix in whole['indices'].items() for i in ix}
    for index,item in enumerate(rows):
        original=by_hash[item['record']['content_sha256']]
        indices[split_for_index[original]].append(index)
        groups.append(whole['group_for_index'][original])
    X=_features(rows);y=np.asarray([_delta(x['record']) for x in rows],dtype=np.float64)
    model={'schema':C.MODEL_SCHEMA,'feature_schema':C.FEATURE_SCHEMA,'source_pins':C.source_pins(),
        'fixture_only':plan['fixture_only'],'status':'NO_DATA' if not rows else 'INSUFFICIENT_SPLITS',
        'coefficients':[0.0]*C.FEATURE_WIDTH,'ridge':None,'calibration':{'status':'UNCALIBRATED',
        'radius':4.0,'coverage':plan['coverage'],'group_count':0},
        'corpus_sha256':plan['corpus']['sha256'],'plan_sha256':R.canonical_sha256(plan)}
    model.update(training_scope='DISCOVERY_ONLY' if discovery_only else 'LEGACY_REPLAY_CORPUS',
                 target_kind='PAIRED_DISCOVERY_WDL_DELTA' if discovery_only else 'PAIRED_REPLAY_WDL_DELTA')
    trials=[];reason_codes=[]
    tr,dev,cal=indices['train'],indices['dev'],indices['calibration']
    # Selection-biased winners-only data are not a rank corpus. This gate reads
    # TRAIN labels only: flipping test outcomes must not choose another model.
    train_coverage=bool(tr) and all(any(y[i]*sign>0 for i in tr) for sign in (-1,1)) and \
        any(y[i]==0 for i in tr) and any(not rows[i]['selected_by_search'] for i in tr)
    if tr and not train_coverage:
        model['status']='INCOMPLETE_OUTCOME_COVERAGE'
    if rows and train_coverage and all(indices[s] for s in SPLITS):
        tr_groups=[groups[i] for i in tr];dev_groups=[groups[i] for i in dev]
        counts={g:tr_groups.count(g) for g in set(tr_groups)}
        weights=np.asarray([1/counts[g] for g in tr_groups]);weights*=len(tr)/weights.sum()
        xtx=X[tr].T@(X[tr]*weights[:,None]);xty=X[tr].T@(y[tr]*weights)
        candidates=[]
        for ridge in plan['ridge_grid']:
            coef=np.linalg.solve(xtx+float(ridge)*np.eye(C.FEATURE_WIDTH),xty)
            prediction=np.clip(X[dev]@coef,-2,2)
            mse=_weighted_mse(y[dev],prediction,dev_groups)
            trials.append({'ridge':ridge,'dev_family_mse':mse})
            candidates.append((mse,float(ridge),coef))
        best=min(candidates,key=lambda c:(c[0],c[1]))
        coef=best[2].tolist();C.need(all(math.isfinite(x) for x in coef),'NONFINITE_FIT')
        prediction=C._predict(coef,X[cal].tolist())
        calibration=_calibrate(y[cal],prediction,[groups[i] for i in cal],plan['coverage'])
        neutral=_weighted_mse(y[dev],np.zeros(len(dev)),dev_groups)
        model.update(coefficients=coef,ridge=best[1],calibration=calibration,
                     status='FITTED' if best[0]<neutral-1e-12 else 'NO_RANK_SIGNAL')
        if calibration['status']!='CALIBRATED':model['status']='UNCALIBRATED'
    if model['status']!='FITTED':reason_codes.append(model['status'])
    C.seal(model)
    test_indices=indices['test'];test_rows=[rows[i] for i in test_indices]
    predictions=C._predict(model['coefficients'],X[test_indices].tolist()) if model['status']=='FITTED' else [0.0]*len(test_indices)
    test=_metrics(y[test_indices],predictions,model['calibration']['radius'],[x['adaptive_search'] for x in test_rows])
    audits=_pool_audits(model,test_rows,plan) if test_rows else []
    means=[a['result']['top_k_verified_mean_delta'] for a in audits if a['result']['top_k_verified_mean_delta'] is not None]
    test.update(top_k_verified_mean_delta=math.fsum(means)/len(means) if means else None,
        random_audit_records=sum(a['result']['random_audit_records'] for a in audits),
        audit_status='COMPLETE_RETROSPECTIVE' if audits and all(a['result']['status']=='COMPLETE' for a in audits) else 'NO_AUDIT_DATA',
        prospective_audit_status='NOT_RUN',
        audit_population='ONLY_CANDIDATES_PRESENT_IN_EXECUTED_CORPUS; NOT_THE_FULL_UNEXECUTED_SEARCH_POOL')
    outcomes={'positive':int((y>0).sum()),'zero':int((y==0).sum()),'negative':int((y<0).sum())}
    report=C.seal({'schema':'farm-v0-regret-rank-fit-result-v1',
        'status':('FITTED_FIXTURE' if plan['fixture_only'] else 'FITTED_REPLAY_ATTESTED') if model['status']=='FITTED' else 'FALLBACK',
        'reason_codes':reason_codes,'model_sha256':model['content_sha256'],
        'plan_sha256':R.canonical_sha256(plan),'corpus_sha256':plan['corpus']['sha256'],
        'source_pins':model['source_pins'],'fixture_only':plan['fixture_only'],
        'training_scope':model['training_scope'],'target_kind':model['target_kind'],
        'counts':{'input_records':len(raw),'records':len(rows),'outcomes':outcomes,
                  'not_selected':sum(not x['selected_by_search'] for x in rows),
                  'train_outcome_coverage':train_coverage,
                  'actor_admitted_records':sum(x['record']['comparison']['actor_training_admissible'] for x in rows)},
        'excluded':clean['excluded'],
        # JSON object keys become strings on disk. Normalize index keys BEFORE
        # sealing so numeric-vs-lexicographic sorting cannot invalidate receipts.
        'splits':{**whole,'group_for_index':{str(i):g for i,g in whole['group_for_index'].items()}},
        'ridge_trials':trials,
        'calibration':model['calibration'],'test':test,'test_pool_audits':audits,
        'native_games_started':0,'actor_updates':0,
        'limitations':['Fixture outcomes are fabricated, not playing strength.',
            'Source/family identity and replay chronology rely on authenticated producer attestations.',
            'Calibration is empirical on family holdouts; adaptive distribution shift has no coverage guarantee.',
            'Prospective audits require freezing the entire pool before requesting missing executions.',
            'No predicted-only labels; paired execution and fixed-N admission remain mandatory.']})
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.rank-fit-',dir=output.parent))
    try:
        for name,value in (('MODEL.json',model),('RESULT.json',report)):
            (temp/name).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
        if output.exists():raise FileExistsError(str(output))
        os.rename(temp,output)
    finally:
        if temp.exists():shutil.rmtree(temp)
    return report


def fit_rollout_value(plan: dict, *, output) -> dict:
    """Discovery-only CPU fit using the accepted ranker, family splits and controls.

    Targets are observed, replay-verified paired WDL samples, including negative
    and zero deltas. Squared-error regression estimates a conditional mean; the
    empirical family-residual interval is NOT a confidence bound on that mean.
    Confirmation/training/monitor/final records are refused, not relabeled. All
    representable censored rows remain in the corpus/split manifest with masked
    targets. Nonrepresentable refusals stay in rollout_trace_rows' sidecar.
    """
    C.need(type(plan) is dict and plan.get('schema') == MC_FIT_SCHEMA, 'MC_FIT_SCHEMA')
    return fit_ranker(plan, output=output)


def rollout_trace_rows(trace: dict, *, fixture_only: bool=False) -> dict:
    """Read the accepted MC trace into existing discovery record envelopes.

    Revalidate stored responses and replay signatures using the paired evaluator;
    never call a game runner. Reconcile attempts (including failures/replays) to
    jobs before exposing any target. Hashes establish consistency, not producer
    authenticity. Cache hits/refusals/not-started jobs are retained, never imputed.
    This does not certify pre-exposure chronology or physical execution of a tape.
    """
    from farm_neat.evaluation import v0_regret_pairs as E
    from farm_neat.training.teachers import v0_regret_searcher as S
    C.need(type(fixture_only) is bool and type(trace) is dict, 'MC_TRACE_OPTIONS')
    C.check_seal(trace, 'MC_TRACE_HASH_MISMATCH')
    C.need(trace.get('schema') == S.MC_SCHEMA and trace.get('status') == 'DISCOVERY_ONLY' and
           trace.get('actor_training_admissible') is False and trace.get('teacher_weight') == 0 and
           trace.get('confirmation_required') is True, 'MC_TRACE_SCOPE')
    sources = E.source_pins() | {p:hashlib.sha256((R.ROOT/p).read_bytes()).hexdigest() for p in S.SOURCE_PATHS}
    C.need(trace['source_pins'] == sources, 'MC_TRACE_SOURCE_MISMATCH')
    execution = trace['execution']
    C.need(E.panel.bound({k:v for k,v in execution.items() if k != 'content_sha256'}) == execution,
           'MC_EXECUTION_HASH')
    C.need(execution['kind'] in ('NATIVE_GAME','SYNTHETIC_DIAGNOSTIC'), 'MC_EXECUTION_KIND')
    native = execution['kind'] == 'NATIVE_GAME'
    C.need(fixture_only or native, 'SYNTHETIC_EVIDENCE')
    jobs, attempts = trace['jobs'], trace['attempts']
    C.need(type(jobs) is list and len(jobs) <= MAX_RECORDS and type(attempts) is list and
           len(attempts) <= 4*MAX_RECORDS, 'MC_TRACE_BUDGET')
    C.need([j['job_id'] for j in jobs] == list(range(len(jobs))), 'MC_JOB_IDENTITIES')
    candidates = {c['candidate_id']:c for c in trace['candidates']}
    C.need(len(candidates) == len(trace['candidates']) and 0 < len(candidates) <= C.MAX_CANDIDATES,
           'MC_CANDIDATE_IDENTITIES')
    groups = {g['source_group_id']:g for g in trace['registration']['discovery_world_groups']}
    families = trace['registration']['source_families']
    C.need(set(groups) <= set(families), 'MC_SOURCE_FAMILIES')
    observed, by_job = set(), {j['job_id']:[] for j in jobs}
    for event in attempts:
        jid = event['job_id']; C.need(jid in by_job, 'MC_ATTEMPT_JOB')
        job = jobs[jid]; branch = event['branch']; outcome = event['outcome']
        E._bound(branch, 'MC_BRANCH_HASH'); E._bound(branch['request'], 'MC_REQUEST_HASH')
        identity = branch['attempt_id']
        C.need(identity not in observed, 'MC_REUSED_ATTEMPT'); observed.add(identity)
        C.need(event['candidate_id'] == job['candidate_id'] and event['group_id'] == job['group_id'] and
               branch['execution'] == execution and branch['request'] == job.get('request') and
               branch['arm'] == event['arm'] and branch['phase'] == event['phase'] and
               event['arm'] in E.ARMS and event['phase'] in ('EVALUATE','REPLAY'), 'MC_ATTEMPT_JOIN')
        C.need(type(event['simulator_starts']) is int and event['simulator_starts'] == 1 and
               type(event['physical_starts']) is int and event['physical_starts'] == int(native), 'MC_START_ACCOUNTING')
        C.need(outcome['attempt_id'] == identity and outcome['branch_sha256'] == branch['content_sha256'] and
               outcome['arm'] == event['arm'] and outcome['phase'] == event['phase'], 'MC_OUTCOME_JOIN')
        if outcome['status'] == 'COMPLETE':
            terminal = E._validate_response(branch, outcome['response'])
            C.need(terminal is not None and terminal == outcome['terminal'], 'MC_TERMINAL_MISMATCH')
        else:
            C.need(outcome['status'] in ('INCOMPLETE','INVALID') and outcome['terminal'] is None,
                   'MC_CENSORED_TERMINAL')
        by_job[jid].append(event)
    counts = trace['counts']
    actual = dict(simulator_starts=len(attempts), physical_starts=int(native)*len(attempts),
                  replay_checks=sum(e['phase']=='REPLAY' for e in attempts))
    C.need(all(type(counts[k]) is int and counts[k] == v for k,v in actual.items()), 'MC_START_ACCOUNTING')
    rows, details, strata, target_hashes = [], [], {}, set()
    for job in jobs:
        cid,gid = job['candidate_id'],job['group_id']
        C.need(cid in candidates and gid in groups, 'MC_JOB_POOL')
        status=job['status']; own=by_job[job['job_id']]
        C.need(status in ('COMPLETE','CENSORED','INVALID','NOT_STARTED','CACHE_HIT'), 'MC_JOB_STATUS')
        strata[status]=strata.get(status,0)+1
        pair=job.get('paired_result');target=False;rh=None
        if status in ('CACHE_HIT','NOT_STARTED'):
            C.need(not own and pair is None, 'MC_UNEXECUTED_JOB_COST')
        if pair is not None:
            E._bound(pair,'MC_PAIR_HASH')
            record=R.validate_pair(pair['record']); rh=record['content_sha256']
            C.need(pair['execution'] == execution and pair['request_sha256']==job['request']['content_sha256'],
                   'MC_PAIR_REQUEST_JOIN')
            C.need(C.public_proposal(record['challenger']) == C.public_proposal(candidates[cid]['geometry']) and
                   record['identity']['source_group_id']==gid and
                   record['evaluation']['discovery_world_groups']==[groups[gid]], 'MC_RECORD_JOIN')
            evidence=[*pair['arms'],*pair['replay']['arms']]
            C.need(evidence == [e['outcome'] for e in own], 'MC_PAIR_ATTEMPT_JOIN')
            rebuilt=E._assemble(job['request'],execution,pair['arms'],pair['replay'])
            C.need(rebuilt==pair, 'MC_PAIR_RECOMPUTE')
            if status=='COMPLETE':
                C.need([(e['phase'],e['arm']) for e in own] ==
                       [(p,a) for p in ('EVALUATE','REPLAY') for a in E.ARMS] and
                       pair['replay']['status']=='VERIFIED' and pair['replay']['mismatches']==[], 'MC_COMPLETE_REPLAY')
                for original,repeated in zip(pair['arms'],pair['replay']['arms']):
                    C.need(original['status']==repeated['status']=='COMPLETE' and
                           E._replay_signature(original)==E._replay_signature(repeated), 'MC_REPLAY_MISMATCH')
                target=True;target_hashes.add(rh)
            # An INVALID control is not rehabilitated by a structurally complete
            # individual pair. Such nonrepresentable statuses remain in details.
            if status != 'INVALID':
                rows.append(dict(pool_id=trace['binding'],candidate_id=cid,selected_by_search=False,
                    adaptive_search=True,source_family_id=families[gid],record=copy.deepcopy(record)))
        else:
            C.need(status != 'COMPLETE', 'MC_COMPLETE_WITHOUT_RECORD')
        details.append(dict(job_id=job['job_id'],candidate_id=cid,group_id=gid,status=status,
            reason=job.get('reason'),record_sha256=rh,target_available=target,
            simulator_starts=len(own),physical_starts=int(native)*len(own),
            replay_checks=sum(e['phase']=='REPLAY' for e in own)))
    clean=validate_rows(rows,fixture_only=fixture_only,discovery_only=True)
    C.need({r['record']['content_sha256'] for r in clean['usable']} == target_hashes, 'MC_TARGET_ACCOUNTING')
    return C.seal({'schema':'farm-v0-regret-mc-value-corpus-v1','scope':'DISCOVERY_ONLY',
        'trace_sha256':trace['content_sha256'],'fixture_only':not native,'rows':rows,'jobs':details,
        'usable_records':len(clean['usable']),'excluded':clean['excluded'],'strata':strata,
        'rejected_candidates':copy.deepcopy(trace['rejected_candidates']),
        'duplicate_candidates':copy.deepcopy(trace['duplicate_candidates']),**actual,
        'fresh_native_starts':0,'limitation':'Revalidates stored evidence, not authentic chronology or fresh execution.'})


def audit_rollout_budget(allocation: dict, trace: dict, *, fixture_only: bool=False) -> dict:
    """Cost-aware retrospective audit of a frozen ordering on a fixed MC pool.

    Only candidates complete on EVERY registered discovery world contribute a
    value; source families are equally weighted. Missing/invalid/censored pairs
    stay pending. A zero physical-start denominator yields null, never infinity
    or a synthetic efficiency claim. Confirmation and actor admission stay owned
    by their existing gates. Reordering previously observed outcomes is NOT a
    prospective experiment or a claim that the historical schedule changed.
    """
    C.check_seal(allocation,'MC_ALLOCATION_HASH')
    C.need(allocation.get('schema')==C.MC_ALLOCATION_SCHEMA, 'MC_ALLOCATION_SCHEMA')
    sel=allocation['selection']; C.check_seal(sel,'ORDER_HASH_MISMATCH')
    maximum=allocation['max_simulator_starts'];ng=allocation['group_count']
    C.need(type(ng) is int and 1<=ng<=256 and type(maximum) is int and
           0<=maximum<=4*C.MAX_CANDIDATES*ng, 'MC_ALLOCATION_BUDGET')
    expected=C._rollout_schedule(sel,maximum,ng)
    reserved=sum(x['reserved_simulator_starts'] for x in expected)
    C.need(allocation['schedule']==expected and allocation['reserved_simulator_starts']==reserved and
           allocation['unused_simulator_start_budget']==maximum-reserved and
           allocation['simulator_starts']==allocation['physical_starts']==0 and
           allocation['requires_execution'] is True and allocation['confirmation_budget_included'] is False and
           allocation['paired_confirmation_required'] is True and
           allocation['confirmation_n']==sel['confirmation_n'], 'MC_ALLOCATION_CONTRACT')
    imported=rollout_trace_rows(trace,fixture_only=fixture_only)
    registry=trace['registration']; groups={g['source_group_id'] for g in registry['discovery_world_groups']}
    C.need(ng==len(groups), 'MC_ALLOCATION_GROUP_COUNT')
    C.need({c['candidate_id'] for c in sel['pool']['candidates']} ==
           {c['candidate_id'] for c in trace['candidates']}, 'MC_ALLOCATION_POOL')
    # Reuse the accepted actor-context, learner, candidate and random-draw joins.
    joined=audit_selection(sel,[{k:v for k,v in row.items() if k!='source_family_id'}
                               for row in imported['rows']],fixture_only=fixture_only)
    scheduled=[r['candidate_id'] for r in expected]
    known={row['record']['content_sha256']:row['record'] for row in imported['rows']}
    observed={cid:{} for cid in scheduled}
    costs={k:0 for k in ('simulator_starts','physical_starts','replay_checks')}
    for job in imported['jobs']:
        if job['candidate_id'] not in observed:continue
        for k in costs:costs[k]+=job[k]
        if job['target_available']:
            observed[job['candidate_id']][job['group_id']]=_delta(known[job['record_sha256']])
    C.need(costs['simulator_starts']<=reserved, 'MC_ALLOCATION_COST_OVERRUN')
    values={}
    for cid, outcomes in observed.items():
        if set(outcomes)!=groups:continue
        families={}
        for gid,value in outcomes.items():families.setdefault(registry['source_families'][gid],[]).append(value)
        values[cid]=math.fsum(math.fsum(v)/len(v) for v in families.values())/len(families)
    pending=[cid for cid in scheduled if cid not in values]
    best=max(values.values()) if values else None
    selected=[cid for cid in sel['selected_ids'] if cid in scheduled]
    audited=[cid for cid in sel['audit']['candidate_ids'] if cid in scheduled]
    audited_complete=len(audited)==len(sel['audit']['candidate_ids']) and all(cid in values for cid in audited)
    m,a=len(sel['nonselected_ids']),len(audited)
    top_k = (math.fsum(values[c] for c in selected)/len(selected)
             if selected and all(c in values for c in selected) else None)
    return C.seal({'schema':'farm-v0-regret-mc-value-audit-v1','scope':'RETROSPECTIVE_FIXED_POOL_DIAGNOSTIC',
        'status':'COMPLETE' if scheduled and not pending else 'PENDING_EXECUTION',
        'allocation_sha256':allocation['content_sha256'],'trace_sha256':trace['content_sha256'],
        'fixture_only':imported['fixture_only'],'scheduled_candidate_ids':scheduled,'pending_candidate_ids':pending,
        'verified_candidate_deltas':values,'best_verified_wdl_delta':best,
        'top_k_verified_mean_delta':top_k,
        'top_k_mean_delta_per_simulator_start':top_k/costs['simulator_starts']
            if top_k is not None and costs['simulator_starts'] else None,
        'top_k_mean_delta_per_physical_start':top_k/costs['physical_starts']
            if top_k is not None and costs['physical_starts'] else None,
        'best_verified_delta_per_simulator_start':best/costs['simulator_starts']
            if best is not None and costs['simulator_starts'] else None,
        'best_verified_delta_per_physical_start':best/costs['physical_starts']
            if best is not None and costs['physical_starts'] else None,
        'random_audit':{'status':'COMPLETE' if audited_complete else 'PENDING_EXECUTION',
            'candidate_ids':audited,'verified_deltas':{c:values[c] for c in audited if c in values},
            'missed_positive_total_estimate':m/a*math.fsum(max(0,values[c]) for c in audited)
                if a and audited_complete else (0.0 if not m else None)},
        **costs,'historical_trace_starts':imported['simulator_starts'],
        'historical_trace_physical_starts':imported['physical_starts'],'fresh_native_starts':0,
        'confirmation_n':allocation['confirmation_n'],'paired_confirmation_required':True,
        'evidence_join_sha256':joined['content_sha256'],
        'limitations':['Retrospective reordering; no prospective speedup or expected-improvement certificate.',
                       'No target for an incomplete candidate; selection/censoring can bias complete-case results.',
                       'Synthetic starts are not physical simulator starts.']})


def _write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('fit','fit-mc','order','audit'):
        p=sub.add_parser(name);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:
        path=Path(args.plan).resolve();plan=R._loads(path.read_text())
        for name in ('corpus','model','selection'):
            if isinstance(plan.get(name),dict) and 'path' in plan[name]:
                plan[name]['path']=str((path.parent/plan[name]['path']).resolve())
        if args.command=='fit': result=fit_ranker(plan,output=args.output)
        elif args.command=='fit-mc': result=fit_rollout_value(plan,output=args.output)
        elif args.command=='order':
            C.need(plan.pop('schema',None)==ORDER_PLAN_SCHEMA,'ORDER_PLAN_SCHEMA')
            ref=plan.pop('model');model=R._loads(_read_bound(ref).decode()) if ref is not None else None
            result=C.order_proposals(model,**plan);_write_new(args.output,result)
        else:
            C.need(set(plan)=={'schema','selection','corpus','fixture_only','max_records'} and
                   plan['schema']==AUDIT_PLAN_SCHEMA,'AUDIT_PLAN_FIELDS')
            result=audit_selection(R._loads(_read_bound(plan['selection']).decode()),
                _load_corpus(plan['corpus'],plan['max_records']),fixture_only=plan['fixture_only'])
            _write_new(args.output,result)
        print(json.dumps({'status':result.get('status',result.get('ordering_mode')),
            'output':args.output,'content_sha256':result['content_sha256'],'native_games_started':0},sort_keys=True))
        return 0
    except (R.RecordError,C.RankError,OSError,ValueError,TypeError,KeyError,np.linalg.LinAlgError) as exc:
        print(json.dumps({'status':'INVALID','reason':getattr(exc,'code',type(exc).__name__),
                          'native_games_started':0},sort_keys=True))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
