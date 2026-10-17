"""Actor-visible V0 pair ranking, never actor supervision or an admission gate.

Public API: score_pairs(model, actor_contexts, proposal_pairs) -> dict.
A pair is {learner: proposal, challenger: proposal}. Proposals contain raw_actions
and canonical_plan; complete existing regret proposals are also accepted. ONLY
those two fields are encoded. Paths are rebuilt on their OWN canonical prefixes
with the native menu, not with stored teacher prefixes, raw alias histories,
likelihoods, selected-winner flags, private simulator fields or evaluation data.
Actor contexts use the existing CLOSED v0_regret_records allowlist.

This deliberately small accelerator is a shared linear utility difference over
128 bounded hashed public features, clipped to [-2,2]. Swapping is antisymmetric;
equal final Plans have a neutral mean, even if paths differ. This guard is NOT an
execution-equivalence certificate. Ordered compiler memory and reconstructed
successor paths are retained as input, without claiming a sufficient state.

order_proposals is the safe consuming interface: bounded shortlist, independent
uniform audit of its complement, and unchanged caller-supplied confirmation N.
No function creates an actor label, likelihood, teacher weight, game execution,
or confirmation waiver. Failed/stale/unfitted rankers use fixed/random ordering.
Fixture rankers cannot steer this consumer without explicit allow_fixture=True.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import random
from typing import Any

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.policy import v0_menu_fast
from farm_neat.training import v0_regret_records as R

FEATURE_SCHEMA = 'farm-v0-regret-rank-features-v1'
MODEL_SCHEMA = 'farm-v0-regret-ranker-v1'
ORDER_SCHEMA = 'farm-v0-regret-order-v1'
FEATURE_WIDTH = 128
MAX_CANDIDATES = 32
MAX_BATCH = 512
AUDIT_ALGORITHM = 'python-random-v3/sample-without-replacement-v1'


class RankError(ValueError):
    """Stable code; no private input dumps in failures."""
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def need(ok: bool, code: str) -> None:
    if not ok:
        raise RankError(code)


def seal(value: dict) -> dict:
    value['content_sha256'] = R.canonical_sha256({k:v for k,v in value.items() if k != 'content_sha256'})
    return value


def check_seal(value: dict, code='HASH_MISMATCH') -> None:
    need(type(value) is dict and value.get('content_sha256') ==
         R.canonical_sha256({k:v for k,v in value.items() if k != 'content_sha256'}), code)


def source_pins() -> dict:
    paths = ('src/farm_neat/policy/v0_regret_comparator.py',
             'src/farm_neat/training/v0_regret_rank.py')
    return {**R.source_pins(), **{p:hashlib.sha256((R.ROOT/p).read_bytes()).hexdigest() for p in paths}}


def public_proposal(proposal: dict) -> dict:
    """Strip ALL proposal provenance. Validate geometry separately with encode_pairs."""
    need(type(proposal) is dict and 'raw_actions' in proposal and 'canonical_plan' in proposal,
         'PROPOSAL_FIELDS')
    actions = proposal['raw_actions']
    need(type(actions) is list and 0 < len(actions) <= V.MAX_EDITS_PER_EPOCH+1, 'INCOMPLETE_PATH')
    need(all(type(a) is int and 0 <= a < V.N_ACTIONS for a in actions), 'ILLEGAL_ACTION')
    R._plan(proposal['canonical_plan'], 'comparator.canonical_plan')
    return copy.deepcopy({k:proposal[k] for k in ('raw_actions','canonical_plan')})


def _geometry(state, initial, options, proposal, menus):
    public = public_proposal(proposal)
    prefix = initial
    successors = []
    for action in public['raw_actions']:
        need(not prefix.terminated, 'ACTION_AFTER_STOP')
        key = R.canonical_sha256(prefix.payload())
        if key not in menus:
            menus[key] = v0_menu_fast.build_menu(state, prefix, projection_kwargs=options, backend='fast')
        menu = menus[key]
        need(action in menu.legal_ids, 'ILLEGAL_ACTION')
        index = menu.legal_ids.index(action)
        successor = next(c.successor for c in menu.classes if index in c.indices)
        successors.append({'prefix':successor.payload(),
                           'forced_stop':prefix.edit_count == V.MAX_EDITS_PER_EPOCH})
        prefix = successor
    need(prefix.terminated, 'INCOMPLETE_PATH')
    need(prefix.plan.canonical() == public['canonical_plan'], 'PLAN_MISMATCH')
    # A caller's optional trace/continuation/provenance is never used as a prefix.
    return public, {'final_plan':prefix.plan.canonical(), 'successors':successors}


def _hashed(value: Any, width: int) -> list[float]:
    """Bounded signed feature hashing. Schema/hash constants are not observations."""
    values = [0.0]*width
    def add(key, number):
        digest=hashlib.sha256(key.encode('utf-8')).digest()
        index=int.from_bytes(digest[:4], 'big') % width
        values[index] += number * (1 if digest[4]&1 else -1)
    def walk(obj, path):
        if isinstance(obj, dict):
            for key in sorted(obj):
                if key not in ('schema','action_schema_sha256'):
                    walk(obj[key], path+'/'+key)
        elif isinstance(obj, list):
            add(path+'/length', math.log1p(len(obj)))
            for i,item in enumerate(obj):
                walk(item, path+'/'+str(i))
        elif type(obj) in (int,float):
            # Record validation ensures finite JSON and bounded integer sizes.
            number=math.copysign(math.log1p(abs(obj)), obj)
            add(path, number)
        elif type(obj) is bool:
            add(path+'='+str(obj), 1.0)
        else:
            add(path+'='+str(obj), 1.0)
    walk(value,'')
    norm=math.sqrt(1+math.fsum(x*x for x in values))
    return [x/norm for x in values]


def encode_pairs(actor_contexts: list[dict], proposal_pairs: list[dict]) -> dict:
    """Validate actual native prefixes and encode ONLY public facts; no model calls.

    Per-call caches are bounded by MAX_BATCH and the V0 17-step cap. Hash collisions
    are approximation, never state equivalences. Unsupported contexts fail closed.
    """
    need(type(actor_contexts) is list and type(proposal_pairs) is list and
         len(actor_contexts)==len(proposal_pairs) <= MAX_BATCH, 'BATCH_SHAPE_OR_BUDGET')
    contexts, paths = {}, {}
    features, identities, identical = [], [], []
    for ctx,pair in zip(actor_contexts, proposal_pairs):
        need(type(pair) is dict and 'learner' in pair and 'challenger' in pair, 'PAIR_FIELDS')
        key = R.actor_context_key(ctx)  # strict actor allowlist, including ordered memory
        if key not in contexts:
            state,prefix,options = R._actor(ctx)
            contexts[key] = state,prefix,options,{},_hashed(ctx,32)
        state,prefix,options,menus,context_features = contexts[key]
        encoded, public = [], []
        for name in ('learner','challenger'):
            proposal=public_proposal(pair[name])
            pathkey=(key,R.canonical_sha256(proposal))
            if pathkey not in paths:
                view,geometry = _geometry(state,prefix,options,proposal,menus)
                paths[pathkey] = view,_hashed(geometry,64)
            view,vector=paths[pathkey]
            public.append(view); encoded.append(vector)
        same = public[0]['canonical_plan']==public[1]['canonical_plan']
        delta=[0.0]*64 if same else [b-a for a,b in zip(*encoded)]
        # Linear in plan difference; context interactions cannot break antisymmetry.
        interaction=[0.0]*64
        for i,c in enumerate(context_features):
            for j,d in enumerate(delta):
                interaction[(i*17+j*13)%64] += c*d*(1 if (i+j)%2 else -1)/8.0
        features.append(delta+interaction)
        identical.append(same)
        identities.append({'actor_context_sha256':key,
            'pair_sha256':R.canonical_sha256(public)})
    return {'feature_schema':FEATURE_SCHEMA, 'features':features,
            'identities':identities, 'identical_plans':identical}


def _model_reasons(model) -> list[str]:
    if model is None:
        return ['NO_MODEL']
    try:
        R._safe(model)
        check_seal(model, 'MODEL_HASH_MISMATCH')
        need(model.get('schema')==MODEL_SCHEMA and model.get('feature_schema')==FEATURE_SCHEMA,
             'MODEL_SCHEMA_MISMATCH')
        need(model.get('source_pins')==source_pins(), 'SOURCE_MISMATCH')
        need(model.get('status')=='FITTED', str(model.get('status','FAILED_MODEL')))
        coef=model['coefficients']
        need(type(coef) is list and len(coef)==FEATURE_WIDTH and
             all(type(x) in (int,float) and math.isfinite(x) and abs(x) <= 1e8 for x in coef),
             'INVALID_COEFFICIENTS')
        need(type(model.get('fixture_only')) is bool, 'MODEL_SCOPE_MISSING')
        cal=model['calibration']
        need(cal['status']=='CALIBRATED', 'UNCALIBRATED')
        need(type(cal['radius']) in (int,float) and 0<=cal['radius']<=4 and
             type(cal['coverage']) in (int,float) and 0<cal['coverage']<1 and
             type(cal['group_count']) is int and cal['group_count']>0, 'INVALID_CALIBRATION')
        return []
    except (R.RecordError, RankError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return [exc.code if isinstance(exc,(R.RecordError,RankError)) else 'INVALID_MODEL']


def _predict(coefficients, features):
    return [max(-2.0,min(2.0,math.fsum(w*x for w,x in zip(coefficients,row)))) for row in features]


def score_pairs(model, actor_contexts, proposal_pairs) -> dict:
    """Signed challenger-minus-learner delta and calibrated empirical uncertainty.

    Intervals predict replay-attested paired WDL, not expected causal improvement.
    Calibration requires exchangeable future families for any coverage interpretation;
    adaptive distribution shift voids that interpretation. Execution remains required.
    """
    enc=encode_pairs(actor_contexts,proposal_pairs)
    reasons=_model_reasons(model)
    predictions=[0.0]*len(proposal_pairs) if reasons else _predict(model['coefficients'],enc['features'])
    radius=4.0 if reasons else model['calibration']['radius']
    rows=[]
    for identity,mean,same in zip(enc['identities'],predictions,enc['identical_plans']):
        if same: mean=0.0
        low,high=max(-2.0,mean-radius),min(2.0,mean+radius)
        rows.append({**identity,'signed_delta':mean, 'interval':[low,high],
            'rank_sign_resolved':low>0 or high<0, 'identical_final_plans':same})
    return seal({'schema':'farm-v0-regret-pair-scores-v1','status':'FALLBACK' if reasons else 'SCORED',
        'model_sha256':model.get('content_sha256') if isinstance(model,dict) and not reasons else None,
        'scope':'RANK_ONLY', 'fixture_only':model.get('fixture_only',True) if isinstance(model,dict) else True,
        'reason_codes':reasons,'scores':rows})


def audit_draw(pool_sha256: str, nonselected: list[str], size: int, seed: int) -> list[str]:
    """Uniform subset of the frozen complement; outcomes/scores are not inputs."""
    material=R.canonical_sha256([AUDIT_ALGORITHM,seed,pool_sha256])
    return random.Random(int(material,16)).sample(sorted(nonselected), min(size,len(nonselected)))


def order_proposals(model, actor_context, learner, candidates, *, shortlist_size=8,
                    audit_size=2, seed=0, fallback='fixed', confirmation_n,
                    allow_fixture=False) -> dict:
    """Search-facing bounded consumer. Carries N unchanged; does not admit anything."""
    need(type(candidates) is list and 0 < len(candidates)<=MAX_CANDIDATES, 'POOL_BUDGET')
    need(type(shortlist_size) is int and 1<=shortlist_size<=MAX_CANDIDATES and
         type(audit_size) is int and 1<=audit_size<=MAX_CANDIDATES, 'SHORTLIST_AUDIT_BUDGET')
    need(type(seed) is int and seed>=0 and type(confirmation_n) is int and confirmation_n>=1,
         'SEED_OR_CONFIRMATION_N')
    need(fallback in ('fixed','random') and type(allow_fixture) is bool, 'ORDER_OPTIONS')
    need(all(type(c) is dict and type(c.get('candidate_id')) is str and c['candidate_id']
             and 'proposal' in c for c in candidates), 'CANDIDATE_FIELDS')
    ids=[c['candidate_id'] for c in candidates]
    need(len(set(ids))==len(ids), 'DUPLICATE_CANDIDATE_ID')
    scores=score_pairs(model,[actor_context]*len(candidates),
        [{'learner':learner,'challenger':c['proposal']} for c in candidates])
    reasons=list(scores['reason_codes'])
    if not reasons and model['fixture_only'] and not allow_fixture:
        reasons.append('FIXTURE_ONLY')
    if reasons:
        order=list(range(len(ids)))
        if fallback=='random': random.Random(seed).shuffle(order)
        mode='FALLBACK_'+fallback.upper()
    else:
        order=sorted(range(len(ids)), key=lambda i:(-scores['scores'][i]['signed_delta'],i))
        mode='RANKED'
    selected=[ids[i] for i in order[:shortlist_size]]
    nonselected=[i for i in ids if i not in selected]
    pool={'actor_context':copy.deepcopy(actor_context), 'learner':public_proposal(learner),
          'candidates':[{'candidate_id':c['candidate_id'],'proposal':public_proposal(c['proposal'])}
                        for c in candidates]}
    pool_hash=R.canonical_sha256(pool)
    audited=audit_draw(pool_hash,nonselected,audit_size,seed)
    return seal({'schema':ORDER_SCHEMA,'scope':'RANK_ONLY','pool':pool,'pool_sha256':pool_hash,
        'ordering_mode':mode,'fallback_reasons':reasons,'model_sha256':scores['model_sha256'],
        'fixture_only':scores['fixture_only'] if mode=='RANKED' else False,
        'evidence_binding':{'learner_path_sha256':learner.get('path_sha256'),
            'direct_draw_id':learner.get('direct_draw_id'),
            'parent_checkpoint_sha256':learner.get('sampling_provenance',{}).get('checkpoint_sha256')},
        'ordered_ids':[ids[i] for i in order],
        'selected_ids':selected,'nonselected_ids':nonselected,
        'scores':scores['scores'],'shortlist_size':shortlist_size,'audit_size':audit_size,
        'audit':{'algorithm':AUDIT_ALGORITHM,'seed':seed,'candidate_ids':audited,
                 'numerator':len(audited),'denominator':len(nonselected)},
        'confirmation_n':confirmation_n,'paired_confirmation_required':True,'executed_pairs':0})


MC_ALLOCATION_SCHEMA = 'farm-v0-regret-mc-value-allocation-v1'


def _rollout_schedule(selection: dict, maximum: int, group_count: int) -> list[dict]:
    """Reserve complete paired execution + fresh replay; never spend confirmation N."""
    selected = selection['selected_ids']
    audited = selection['audit']['candidate_ids']
    mandatory = selected + audited
    cost = 4 * group_count  # learner, challenger, replay learner, replay challenger
    if maximum < cost * len(mandatory):
        return []
    priority = mandatory + [i for i in selection['ordered_ids'] if i not in mandatory]
    return [{'candidate_id':i, 'role':'TOP_K' if i in selected else
             'RANDOM_AUDIT' if i in audited else 'REMAINDER',
             'reserved_simulator_starts':cost} for i in priority[:maximum // cost]]


def allocate_rollout_budget(model, actor_context, learner, candidates, *,
                            max_simulator_starts: int, shortlist_size: int, audit_size: int,
                            seed: int, confirmation_n: int, group_count: int = 1,
                            fallback: str = 'fixed', allow_fixture: bool = False) -> dict:
    """Order-only seam for the MC allocator: no callbacks, execution or actor labels.

    Inputs are the existing actor context and complete reachable candidate paths.
    The budget reserves FOUR starts per candidate/world, including both replays;
    confirmation is separate and its caller-supplied N is carried unchanged. A
    mandatory uniform complement audit cannot be silently dropped to fit the cap.
    This pessimistic reservation does not presume another worker's cache savings.
    Group identities, actual starts, full execution and confirmation remain owned
    by the searcher/evaluator. An uncalibrated, absent, stale, fixture-disallowed or
    non-discovery model deterministically uses the existing fixed/random fallback.
    """
    need(type(group_count) is int and 1 <= group_count <= 256 and
         type(max_simulator_starts) is int and
         0 <= max_simulator_starts <= 4 * MAX_CANDIDATES * group_count, 'MC_START_BUDGET')
    reasons = _model_reasons(model)
    if not reasons and (model.get('training_scope') != 'DISCOVERY_ONLY' or
                        model.get('target_kind') != 'PAIRED_DISCOVERY_WDL_DELTA'):
        reasons.append('NOT_DISCOVERY_VALUE_MODEL')
    selection = order_proposals(None if reasons else model, actor_context, learner, candidates,
        shortlist_size=shortlist_size, audit_size=audit_size, seed=seed,
        confirmation_n=confirmation_n, fallback=fallback, allow_fixture=allow_fixture)
    if reasons:
        selection['fallback_reasons'] = reasons
        seal(selection)
    schedule = _rollout_schedule(selection, max_simulator_starts, group_count)
    reserved = sum(row['reserved_simulator_starts'] for row in schedule)
    return seal({'schema':MC_ALLOCATION_SCHEMA, 'scope':'ORDER_AND_RESERVE_ONLY',
        'status':'READY_FOR_EXECUTION' if schedule else 'INSUFFICIENT_AUDIT_BUDGET',
        'selection':selection, 'max_simulator_starts':max_simulator_starts,
        'group_count':group_count, 'schedule':schedule,
        'reserved_simulator_starts':reserved, 'unused_simulator_start_budget':max_simulator_starts-reserved,
        'simulator_starts':0, 'physical_starts':0, 'requires_execution':True,
        'confirmation_n':confirmation_n, 'paired_confirmation_required':True,
        'confirmation_budget_included':False,
        'uncertainty':'EMPIRICAL_FAMILY_RESIDUAL_INTERVAL; NOT_AN_EXPECTED_VALUE_CONFIDENCE_BOUND',
        'limitation':'Reservation only; downstream execution and fixed-N admission are mandatory.'})
