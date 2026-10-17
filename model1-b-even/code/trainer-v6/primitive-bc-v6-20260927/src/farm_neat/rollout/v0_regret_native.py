"""Source-bound genuine parent draws for the existing regret collector and RL fit.

``factory(config)`` returns exactly sampler/proposer/evaluator. Put the result of
``plan`` in the collector spec's callback_config and use
``--callbacks farm_neat.rollout.v0_regret_native:factory``. The sampler returns the
UNCHANGED regret learner record; full native decoder traces, seed/RNG state and
ordered compiler continuation remain in this producer's journal. The collector
still owns challenger selection, launch custody, paired replay and admission.

``direct`` executes ONLY an existing v0_rl plan's collection_jobs. It uses the
committed-scorer bridge, OpeningScorerPolicy/PolicyAgent and v0_policy.play_game.
No confirmation clone, selected proposal or collector control becomes direct RL.
The output is v0_regret_train.DIRECT_SCHEMA, not the paired opportunities corpus.
Every registered game must complete and pass both unchanged direct validators and
independent parent likelihood replay before DIRECT.json is published. Failures,
unknown launches and missing draws stay in the inherited append-only journal.
Resume is at-most-once, not retry-until-success; after an uncertain launch the
round stops. Replay executes no games and never redraws a sample.

All execution is default-off: FARM_V0_REGRET_NATIVE_EXECUTE=1 is required by the
factory; CLI execution additionally requires --execute. Import, plan and check
never load a model or step an environment. Parent/runtime authentication occurs
only at explicit execution through load_committed_scorer, never a second loader.
CUDA additionally requires the maintainer's existing SLURM allocation. This
module never constructs an optimizer, updates weights or selects a checkpoint.

World hashes: fixed_shop_registry uses the existing router_panel World.sha256;
original_rules hashes OriginalRuleCase's initialization without its opponent.
Opponent hashes are wire_hash(scenarios.opponent(id)[1]). Use world_binding() and
opponent_sha256() when registering a future RL plan. Fixed-shop evidence is NOT
original-rule transfer. The opening RL actor codec uses a neutral seat; direct
sampling uses BC.actor_state(T.actor_observation(actual_state)), records the real
seat separately, and validates every emitted prefix against the actual state.
No sampled trace header or likelihood is rewritten after decoding.

Custody assumes a single protected output root per producer registration. Source
hashes and filesystem journals detect mistakes, not a malicious operator who
copies journals or fabricates a completely new registration. Cross-run allocation
and production pilot decisions remain the maintainer's responsibility.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
import re

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.evaluation import router_panel as Panel, v0_policy as P, v0_regret_pairs as E
from farm_neat.evaluation.v0_continuation import V0Continuation
from farm_neat.policy import v0_scorer as A
from farm_neat.rollout import v0_regret_collect as C
from farm_neat.training import v0_bc as BC, v0_rl as RL, v0_regret_records as R, v0_regret_train as RT
from farm_neat.training import v0_targets as T
from farm_neat.training.teachers import v0_regret_proposals as Proposals

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'farm-v0-regret-native-producer-v1'
SPEC_SCHEMA = SCHEMA + '-spec'
CONFIG_SCHEMA = SCHEMA + '-config'
DRAW_SCHEMA = SCHEMA + '-draw-receipt'
AUTH_ENV = 'FARM_V0_REGRET_NATIVE_EXECUTE'
_NATIVE_PLAY_GAME = P.play_game
_GEOMETRY_FIELDS = ('prefix', 'support', 'phi_sha256', 'raw_id', 'class_id',
                    'class_members', 'continuation', 'forced_stop', 'termination')
need = C.need


def source_pins():
    return {**C.source_pins(), **P.source_pins(), **RT.source_pins(),
            'src/farm_neat/rollout/v0_regret_native.py': RL.file_sha(__file__)}


def _relative(value, *, output=False):
    need(isinstance(value, str) and bool(value.strip()), 'REPOSITORY_RELATIVE_PATH_REQUIRED')
    path = Path(value)
    need(not path.is_absolute() and '..' not in path.parts and str(path) == value,
         'REPOSITORY_RELATIVE_PATH_REQUIRED')
    if output:
        need(value.startswith('work/runs/') and len(path.parts) >= 4, 'RUN_OUTPUT_REQUIRED')
    full = ROOT / path
    need(not any(p.is_symlink() for p in (full, *full.parents)), 'SYMLINKED_PATH_REFUSED')
    return full


def _ref(value):
    R._fields(value, ('path', 'sha256'), '$.reference')
    _relative(value['path']); R._hash(value['sha256'], '$.reference.sha256')
    return value


def opponent_sha256(opponent):
    """Identity used by the existing policy-game runner, not a new opponent."""
    need(opponent in E.OPPONENTS, 'UNSUPPORTED_OPPONENT')
    return P.wire_hash(P.scenarios.opponent(opponent)[1])


def world_binding(kind, world_id, *, seed=None):
    """Data-only world identity: no environment initialization or transition."""
    R._text(world_id, '$.world_id')
    if kind == 'fixed_shop_registry':
        need(seed is None, 'FIXED_SHOP_SEED_OVERRIDE_REFUSED')
        world = Panel.world_registry()[world_id]
        return dict(kind=kind, world_id=world_id, world_sha256=world.sha256)
    need(kind == 'original_rules', 'EXPLICIT_WORLD_RULES_REQUIRED')
    R._int(seed, '$.world.seed', 0, 2**63-1)
    spec = P.OriginalRuleCase(seed, 0, 'pass').document()['scenario']
    identity = {k: v for k, v in spec.items() if k != 'opponent'}
    return dict(kind=kind, world_id=world_id, seed=seed, world_sha256=P.wire_hash(identity))


def _direct_plan(spec):
    ref = _ref(spec['direct_plan'])
    return RL.load_plan(_relative(ref['path']), ref['sha256'])


def _validate_spec(spec):
    R._safe(spec)
    R._fields(spec, ('schema', 'producer_id', 'parent', 'device', 'production', 'output',
                     'timeout_seconds', 'collector_draws', 'direct_plan', 'worlds'), '$.spec')
    need(spec['schema'] == SPEC_SCHEMA, 'PRODUCER_SPEC_SCHEMA')
    R._text(spec['producer_id'], '$.producer_id')
    R._fields(spec['parent'], ('path', 'commit_sha256'), '$.parent')
    _relative(spec['parent']['path']); R._hash(spec['parent']['commit_sha256'], '$.parent')
    _relative(spec['output'], output=True)
    R._bool(spec['production'], '$.production')
    need(type(spec['device']) is str and re.fullmatch(r'cpu|cuda(?::\d+)?', spec['device']), 'CPU_OR_CUDA_REQUIRED')
    need(spec['production'] or spec['device'] == 'cpu', 'DIAGNOSTIC_CPU_ONLY')
    R._number(spec['timeout_seconds'], '$.timeout_seconds', 0., 300.)
    need(spec['timeout_seconds'] > 0., 'BOUNDED_GAME_TIMEOUT_REQUIRED')
    draws = R._list(spec['collector_draws'], '$.collector_draws')
    worlds = R._list(spec['worlds'], '$.worlds')
    ids, seeds = set(), set()
    if spec['direct_plan'] is None:
        need(0 < len(draws) <= 64 and not worlds, 'COLLECTOR_DRAW_REGISTRATION_REQUIRED')
        for row in draws:
            R._fields(row, ('draw_id', 'seed', 'actor_context_sha256', 'continuation'), '$.draw')
            R._text(row['draw_id'], '$.draw_id'); R._hash(row['actor_context_sha256'], '$.context')
            R._int(row['seed'], '$.seed', 0, 2**63-1)
            # This registration contains only an actor hash, not its observation.
            # Do not fabricate quadrant ownership to restore carried LAND here.
            need(isinstance(row['continuation'], dict)
                 and row['continuation'].get('land_commitment') is None,
                 'COLLECTOR_LAND_REQUIRES_OBSERVATION_CONTEXT')
            R._memory(row['continuation'], '$.continuation')
            need(row['draw_id'] not in ids, 'DRAW_ID_REUSE')
            need(row['seed'] not in seeds, 'SAMPLER_SEED_REUSE')
            ids.add(row['draw_id']); seeds.add(row['seed'])
    else:
        need(not draws, 'COLLECTOR_CONTROLS_ARE_NOT_DIRECT_RL')
        plan = _direct_plan(spec)
        need(plan['behavior_checkpoint'] == spec['parent'], 'DIRECT_PARENT_MISMATCH')
        expected = []
        for group in plan['train_groups']:
            RL._world(group, group=True)
            for seed in group['sample_seeds']:
                need(seed not in seeds, 'SAMPLER_SEED_REUSE')
                seeds.add(seed)
                expected.append(dict(job_id=f"collect:{group['group_id']}:{seed}",
                    group_id=group['group_id'], world_id=group['world_id'], world_sha256=group['world_sha256'],
                    seat=group['seat'], opponent_id=group['opponent_id'], sampler_seed=seed,
                    arm=RL.BASELINE_ARM, mode='sample_raw', complete_game_required=True))
        need(0 < len(expected) <= RL.MAX_TRAIN_GAMES and plan['collection_jobs'] == expected,
             'REGISTERED_COLLECTION_JOBS_REQUIRED')
        need(len({j['job_id'] for j in expected}) == len(expected), 'DUPLICATE_GAME_JOB')
        bindings = {}
        for world in worlds:
            need(isinstance(world, dict), 'WORLD_BINDING_REQUIRED')
            wanted = world_binding(world.get('kind'), world.get('world_id'), seed=world.get('seed'))
            need(world == wanted, 'WORLD_RULES_OR_SOURCE_MISMATCH')
            need(world['world_id'] not in bindings, 'DUPLICATE_WORLD')
            bindings[world['world_id']] = world
        need(set(bindings) == {j['world_id'] for j in expected}, 'EXACT_TRAINING_WORLD_COVERAGE_REQUIRED')
        opponents = {o['id']: o['sha256'] for o in plan['opponents']}
        for job in expected:
            need(bindings[job['world_id']]['world_sha256'] == job['world_sha256'], 'REGISTERED_WORLD_HASH_MISMATCH')
            need(opponents.get(job['opponent_id']) == opponent_sha256(job['opponent_id']), 'OPPONENT_IDENTITY_MISMATCH')
            if bindings[job['world_id']]['kind'] == 'original_rules':
                need(job['opponent_id'] in ('pass', 'router'), 'ORIGINAL_RULE_OPPONENT_UNSUPPORTED')
        # No evaluation job is ever collected by this producer.
        train = {(j['world_sha256'], j['seat'], j['opponent_id']) for j in expected}
        held = {(j['world_sha256'], j['seat'], j['opponent_id']) for j in plan['evaluation_cases']}
        need(train.isdisjoint(held), 'PROTECTED_WORLD_OVERLAP')
    return copy.deepcopy(spec)


def build_config(spec):
    return C.bind(dict(schema=CONFIG_SCHEMA, spec=_validate_spec(spec), source_pins=source_pins(),
                       strength_claim=False))


def _config(config):
    C.bound(config, 'CONFIG_HASH_MISMATCH')
    R._fields(config, ('schema', 'spec', 'source_pins', 'strength_claim', 'content_sha256'), '$.config')
    need(config['schema'] == CONFIG_SCHEMA and config['strength_claim'] is False, 'PRODUCER_CONFIG_SCHEMA')
    need(config['source_pins'] == source_pins(), 'PRODUCER_SOURCE_MISMATCH')
    _validate_spec(config['spec'])
    return copy.deepcopy(config)


def check(config):
    config = _config(config)
    return dict(status='CONFIG_VALID', config_sha256=config['content_sha256'],
        mode='direct' if config['spec']['direct_plan'] else 'collector',
        parent_authentication='NOT_YET_RUN', model_loads=0, physical_games_started=0,
        runtime='UNBOUND_UNTIL_COMMITTED_SCORER_AUTHENTICATION',
        rules=[w['kind'] for w in config['spec']['worlds']], strength_claim=False)


def _authorize(config, *, execute=True):
    need(execute is True, 'EXPLICIT_EXECUTE_REQUIRED')
    need(os.environ.get(AUTH_ENV) == '1', 'EXECUTION_NOT_AUTHORIZED')
    if config['spec']['device'].startswith('cuda'):
        need(bool(os.environ.get('SLURM_JOB_ID')) and bool(os.environ.get('CUDA_VISIBLE_DEVICES')),
             'EXISTING_SLURM_ALLOCATION_REQUIRED')


def _freeze(journal, config, *, existing=False):
    frozen = journal.by_key.get('producer', {}).get('CONFIG')
    need(frozen is not None or (not existing and not journal.events), 'PRODUCER_CUSTODY_REQUIRED')
    journal.once('CONFIG', 'producer', config)


def _load(config, journal):
    _authorize(config)
    spec = config['spec']; parent = spec['parent']
    loaded = P.load_committed_scorer(_relative(parent['path']), parent['commit_sha256'],
                                    device=spec['device'], production=spec['production'])
    need(loaded.identity['commit_sha256'] == parent['commit_sha256'], 'LOADED_PARENT_MISMATCH')
    need(loaded.identity['diagnostic'] is (not spec['production']), 'PARENT_DIAGNOSTIC_SCOPE_MISMATCH')
    if spec['direct_plan'] is not None:
        plan = _direct_plan(spec)
        need(loaded.identity['state_sha256'] == plan['behavior_checkpoint_state_sha256'] and
             loaded.identity['model_identity'] == plan['behavior_model_identity'], 'DIRECT_PARENT_STATE_OR_MODEL_MISMATCH')
    journal.once('PARENT', 'parent', loaded.identity)
    return loaded


def _job_ids(config, job):
    key = C.digest([config['content_sha256'], job['job_id']])
    return 'direct-game-'+key, 'direct-draw-'+key


def _draw_key(draw_id):
    return 'draw-'+C.digest(draw_id)


def _empty_memory(game_id):
    agent = P.PolicyAgent(None, game_id=game_id, labor_factor=4., hands_mode='daily',
                          belief=copy.deepcopy(T.SETTINGS['belief']))
    return V0Continuation.capture(agent).to_dict()


def _context(state, prefix, continuation):
    facts = {k: copy.deepcopy(getattr(state, k)) for k in R.OBS_FIELDS}
    facts['quadrant_names'] = list(state.quadrant_names)
    facts['seeds'] = {crop: state.seeds.get(crop, 0) for crop in A.CROPS}
    # Preserve list order, not a sort of allocation/started/HOLD entries.
    return dict(schema=R.ACTOR_SCHEMA, observation=facts, prefix=prefix.payload(),
                compiler_memory={k: copy.deepcopy(v) for k, v in continuation.items() if k != 'game_id'})


def _entry(config, draw_id, seed, context):
    spec = config['spec']
    if spec['direct_plan'] is None:
        matches = [r for r in spec['collector_draws'] if r['draw_id'] == draw_id]
        need(len(matches) == 1, 'UNREGISTERED_DRAW')
        row = matches[0]
        need(row['seed'] == seed and row['actor_context_sha256'] == C.digest(context), 'DRAW_CONTEXT_OR_SEED_MISMATCH')
        memory = row['continuation']
    else:
        matches = [j for j in _direct_plan(spec)['collection_jobs'] if _job_ids(config, j)[1] == draw_id]
        need(len(matches) == 1, 'UNREGISTERED_DIRECT_DRAW')
        job = matches[0]
        need(seed == job['sampler_seed'] and context['observation']['player'] == job['seat'] and
             context['observation']['step'] == 0 and context['prefix'] == A.Prefix(Plan()).payload(),
             'DIRECT_OPENING_CONTEXT_OR_SEED_MISMATCH')
        memory = _empty_memory(_job_ids(config, job)[0])
    need(context['compiler_memory'] == {k: v for k, v in memory.items() if k != 'game_id'},
         'DRAW_COMPILER_MEMORY_MISMATCH')
    R._actor(context)
    return copy.deepcopy(memory)


def _record(trace, draw_id, parent, before, after):
    rows = [dict(schema=R.PREFIX_SCHEMA, **{k: copy.deepcopy(s[k]) for k in _GEOMETRY_FIELDS})
            for s in trace['steps']]
    final = A.Prefix(Plan.from_canonical(rows[-1]['continuation']['plan']),
        rows[-1]['continuation']['last_key'], rows[-1]['continuation']['edit_count'],
        rows[-1]['continuation']['terminated'])
    return C.bind(dict(raw_actions=copy.deepcopy(trace['actions']), prefix_class_trace=rows,
        canonical_plan=copy.deepcopy(trace['final_plan']),
        continuation_signature=T.continuation_signature(final.plan, final.last_key, final.edit_count, final.terminated),
        direct_draw_id=draw_id, raw_behavior_logp=trace['raw_path_logp'],
        sampling_provenance=dict(type='direct_policy', mode='sample_raw', checkpoint_sha256=parent,
            temperature=1., raw_step_logps=[s['raw_logp'] for s in trace['steps']],
            rng_before_sha256=C.digest(before), rng_after_sha256=C.digest(after))), 'path_sha256')


def verify_draw(config, audit, loaded):
    """Rebuild actual-prefix geometry and parent likelihood; never sample or play."""
    import torch
    config = _config(config)
    C.bound(audit, 'DRAW_RECEIPT_HASH_MISMATCH')
    R._fields(audit, ('schema', 'config_sha256', 'parent_identity', 'actor_context',
        'actor_context_sha256', 'continuation', 'continuation_wire_sha256', 'sample_seed',
        'rng_before', 'rng_after', 'decode_trace', 'decode_state_projection', 'draw',
        'evidence_kind', 'physical_games', 'content_sha256'), '$.draw_receipt')
    expected_kind = ('SYNTHETIC_DIAGNOSTIC' if loaded.identity['diagnostic']
                     else 'AUTHENTICATED_POLICY_DRAW')
    need(audit['evidence_kind'] == expected_kind and type(audit['physical_games']) is int
         and audit['physical_games'] == 0, 'DRAW_EVIDENCE_SCOPE_MISMATCH')
    need(audit['schema'] == DRAW_SCHEMA and audit['config_sha256'] == config['content_sha256'], 'DRAW_CONFIG_MISMATCH')
    need(audit['parent_identity'] == loaded.identity and
         loaded.identity['commit_sha256'] == config['spec']['parent']['commit_sha256'], 'DRAW_PARENT_MISMATCH')
    need(audit['actor_context_sha256'] == C.digest(audit['actor_context']), 'DRAW_CONTEXT_MISMATCH')
    draw, context = audit['draw'], audit['actor_context']
    memory = _entry(config, draw['direct_draw_id'], audit['sample_seed'], context)
    need(audit['continuation'] == memory and audit['continuation_wire_sha256'] == P.wire_hash(memory),
         'DRAW_CONTINUATION_MISMATCH')
    for state in (audit['rng_before'], audit['rng_after']):
        need(type(state) is list and state and all(type(v) is int and 0 <= v <= 255 for v in state), 'RNG_STATE_REQUIRED')
    parameters = list(loaded.scorer.parameters())
    device = parameters[0].device if parameters else torch.device('cpu')
    initial = torch.Generator(device=device).manual_seed(audit['sample_seed']).get_state().tolist()
    need(initial == audit['rng_before'], 'SAMPLER_STATE_REUSE_OR_SEED_MISMATCH')
    need(len(audit['rng_after']) == len(initial), 'RNG_STATE_LENGTH_MISMATCH')
    need(draw == _record(audit['decode_trace'], draw['direct_draw_id'],
                        loaded.identity['commit_sha256'], audit['rng_before'], audit['rng_after']), 'DRAW_TRACE_JOIN_MISMATCH')
    state, prefix, options = R._actor(context)
    R._proposal(draw, {'parent_checkpoint_sha256': loaded.identity['commit_sha256']},
                prefix, state, options, {}, direct=True)
    direct = config['spec']['direct_plan'] is not None
    expected_projection = 'V0_RL_ACTOR_STATE' if direct else 'REGRET_ACTOR_CONTEXT'
    need(audit['decode_state_projection'] == expected_projection, 'ACTOR_PROJECTION_MISMATCH')
    decode_state = BC.actor_state(T.actor_observation(state)) if direct else state
    need(loaded.scorer.training is False, 'EVAL_PARENT_REQUIRED')
    with torch.inference_mode():
        result = A.replay_epoch(loaded.scorer, decode_state, prefix, copy.deepcopy(audit['decode_trace']),
                                projection_kwargs=options, verify_logps=True, backend=RL.PROTOCOL['menu_backend'])
    need(list(result.actions) == draw['raw_actions'], 'REPLAY_ACTION_MISMATCH')
    return dict(status='VERIFIED', direct_draw_id=draw['direct_draw_id'], raw_path_logp=float(result.raw_logp),
                physical_games_started=0, class_path_not_final_plan_marginal=True)


def verify_persisted_draw(config, audit, loaded):
    """Admission for externally persisted draws; one seeded resample only."""
    import math
    import torch
    config = _config(config)
    verified = verify_draw(config, audit, loaded)
    state, prefix, options = R._actor(audit['actor_context'])
    if config['spec']['direct_plan'] is not None:
        state = BC.actor_state(T.actor_observation(state))
    parameters = list(loaded.scorer.parameters())
    device = parameters[0].device if parameters else torch.device('cpu')
    generator = torch.Generator(device=device).manual_seed(audit['sample_seed'])
    need(generator.get_state().tolist() == audit['rng_before'], 'INITIAL_RNG_MISMATCH')
    with torch.inference_mode():
        resampled = A.decode_epoch(loaded.scorer, state, prefix, generator=generator,
            mode='sample_raw', temperature=1., projection_kwargs=copy.deepcopy(options),
            backend=RL.PROTOCOL['menu_backend'])
    need(list(resampled.actions) == audit['draw']['raw_actions'], 'SAMPLED_ACTION_PATH_MISMATCH')
    final_rng = generator.get_state().tolist()
    need(final_rng == audit['rng_after'], 'POST_DRAW_RNG_MISMATCH')
    need(math.isclose(float(resampled.raw_logp), audit['draw']['raw_behavior_logp'],
                      rel_tol=2e-6, abs_tol=3e-5), 'SAMPLED_DRAW_LOGLIK_MISMATCH')
    return {**verified, 'status': 'VERIFIED_PERSISTED_DRAW',
            'raw_actions': list(resampled.actions), 'rng_after_sha256': R.canonical_sha256(final_rng),
            'resamples': 1, 'admission_scope': 'EXTERNALLY_PERSISTED_DRAW_ONLY'}


def _sample(config, journal, loaded, policy, context, draw_id, seed):
    import torch
    _authorize(config)
    need(config['source_pins'] == source_pins(), 'PRODUCER_SOURCE_MISMATCH')
    memory = _entry(config, draw_id, seed, context)
    journal.once('PARENT', 'parent', loaded.identity)
    need(loaded.scorer.training is False, 'EVAL_PARENT_REQUIRED')
    key = _draw_key(draw_id)
    need('INTENT' not in journal.by_key.get(key, {}), 'DRAW_ALREADY_LAUNCHED_NO_REROLL')
    state, prefix, options = R._actor(context)
    direct = config['spec']['direct_plan'] is not None
    decode_state = BC.actor_state(T.actor_observation(state)) if direct else state
    argument = dict(config_sha256=config['content_sha256'], context=context, continuation=memory,
                    draw_id=draw_id, seed=seed, parent_checkpoint_sha256=loaded.identity['commit_sha256'])
    def generate(_):
        need(policy.mode == 'sample_raw' and policy.seed == seed and policy.temperature == 1., 'RAW_POLICY_REQUIRED')
        before = policy.generator.get_state().tolist()
        initial = torch.Generator(device=policy.generator.device).manual_seed(seed).get_state().tolist()
        need(before == initial, 'SAMPLER_STATE_REUSE_OR_SEED_MISMATCH')
        with torch.inference_mode():
            result = A.decode_epoch(loaded.scorer, decode_state, prefix, generator=policy.generator,
                mode='sample_raw', temperature=1., projection_kwargs=options, backend=RL.PROTOCOL['menu_backend'])
        after = policy.generator.get_state().tolist()
        policy.last_trace = copy.deepcopy(result.trace)
        audit = C.bind(dict(schema=DRAW_SCHEMA, config_sha256=config['content_sha256'],
            parent_identity=copy.deepcopy(loaded.identity), actor_context=copy.deepcopy(context),
            actor_context_sha256=C.digest(context), continuation=memory, continuation_wire_sha256=P.wire_hash(memory),
            sample_seed=seed, rng_before=before, rng_after=after, decode_trace=copy.deepcopy(result.trace),
            decode_state_projection='V0_RL_ACTOR_STATE' if direct else 'REGRET_ACTOR_CONTEXT',
            draw=_record(result.trace, draw_id, loaded.identity['commit_sha256'], before, after),
            evidence_kind='SYNTHETIC_DIAGNOSTIC' if loaded.identity['diagnostic'] else 'AUTHENTICATED_POLICY_DRAW',
            physical_games=0))
        verify_draw(config, audit, loaded)
        return audit
    result = journal.call(key, kind='sample', argument=argument, callback=generate, lane='draw')
    need(result['status'] == 'RETURNED', 'DRAW_FAILED_NO_REROLL:'+str(result['error']))
    return result['value']


class NativeSampler:
    """One source-bound registration, fresh sampler per draw; no model on init."""
    def __init__(self, config):
        self.config = _config(config)
        need(self.config['spec']['direct_plan'] is None, 'COLLECTOR_FACTORY_REQUIRES_COLLECTOR_CONFIG')
        self.loaded = None

    def __call__(self, context, *, draw_id, seed, parent_checkpoint_sha256):
        config = _config(self.config); _authorize(config)
        need(parent_checkpoint_sha256 == config['spec']['parent']['commit_sha256'], 'DRAW_PARENT_MISMATCH')
        _entry(config, draw_id, seed, context)
        with C._lock(_relative(config['spec']['output'], output=True)):
            journal = C.Journal(_relative(config['spec']['output'], output=True)); _freeze(journal, config)
            need('INTENT' not in journal.by_key.get(_draw_key(draw_id), {}), 'DRAW_ALREADY_LAUNCHED_NO_REROLL')
            if self.loaded is None:
                self.loaded = _load(config, journal)
            policy = self.loaded.policy('sample_raw', seed, temperature=1.)
            return _sample(config, journal, self.loaded, policy, context, draw_id, seed)['draw']


def factory(config):
    """Trusted collector callback factory. Config is the sealed plan document."""
    config = _config(config); _authorize(config)
    need(config['spec']['production'] or E.native_runner is not E._NATIVE_RUNNER,
         'DIAGNOSTIC_NATIVE_CALLBACKS_FORBIDDEN')
    return dict(sampler=NativeSampler(config), proposer=Proposals.propose, evaluator=E.native_runner)


def draw_receipt(config, draw_id):
    config = _config(config)
    journal = C.Journal(_relative(config['spec']['output'], output=True))
    frozen = journal.by_key.get('producer', {}).get('CONFIG')
    need(frozen is not None and journal.get(frozen['object']) == config, 'DRAW_CUSTODY_CONFIG_MISMATCH')
    saved = journal.by_key.get(_draw_key(draw_id), {}).get('RESULT')
    need(saved is not None, 'DRAW_RESULT_MISSING')
    result = journal.get(saved['object'])
    need(result['status'] == 'RETURNED', 'DRAW_NOT_COMPLETE')
    return result['value']


class _Opening(P.OpeningScorerPolicy):
    """Recording shim; shared native scorer, actor codec, decoder and executor."""
    def __init__(self, loaded, config, journal, job):
        super().__init__(loaded.scorer, identity=loaded.identity, mode='sample_raw', seed=job['sampler_seed'])
        self.loaded, self.config, self.journal, self.job = loaded, config, journal, job

    def __call__(self, state, plan, masks):
        if state.step != 0:
            return super().__call__(state, plan, masks)
        need(plan == Plan() and A._projection_options(masks.kw) == A._projection_options(T.SETTINGS),
             'DIRECT_OPENING_ONLY_EMPTY_PLAN_REQUIRED')
        game_id, draw_id = _job_ids(self.config, self.job)
        context = _context(state, A.Prefix(plan), _empty_memory(game_id))
        audit = _sample(self.config, self.journal, self.loaded, self, context, draw_id, self.seed)
        self.last_trace = copy.deepcopy(audit['decode_trace'])
        return tuple(audit['draw']['raw_actions'])


class _Case:
    def __init__(self, document): self._document = document
    def document(self): return copy.deepcopy(self._document)


def _case(config, job):
    world = next(w for w in config['spec']['worlds'] if w['world_id'] == job['world_id'])
    game_id, _ = _job_ids(config, job)
    if world['kind'] == 'original_rules':
        return P.OriginalRuleCase(world['seed'], job['seat'], job['opponent_id'], game_id=game_id).document()
    spec = Panel.world_registry()[world['world_id']].scenario()
    spec['opponent'] = job['opponent_id']; spec = P.scenarios.validate(spec)
    body = dict(scenario=spec, seat=job['seat'], opponent_identity=P.scenarios.opponent(job['opponent_id'])[1])
    return dict(id=P.wire_hash(body), **body, game_id=game_id)


def _runner_binding(config):
    native = P.play_game is _NATIVE_PLAY_GAME
    need(config['spec']['production'] or not native, 'DIAGNOSTIC_NATIVE_EXECUTION_FORBIDDEN')
    return dict(kind='NATIVE_GAME' if native else 'SYNTHETIC_DIAGNOSTIC',
                runner=Panel._fingerprint(P.play_game), strength_claim=False)


def _play(config, journal, loaded, job):
    case = _case(config, job)
    used = []
    def policy_factory(mode, seed):
        need(mode == 'sample_raw' and seed == job['sampler_seed'] and not used, 'SAMPLER_FACTORY_REUSE')
        actor = _Opening(loaded, config, journal, job); used.append(actor)
        return actor
    arm = P.Arm('direct-parent', policy_factory, loaded.identity)
    registration = P.registration([_Case(case)], [arm], modes=('sample_raw',),
        sample_seeds=(job['sampler_seed'],), max_games=1, labor_factor=4., hands_mode='daily',
        timeout_seconds=config['spec']['timeout_seconds'])
    # Explicit zero-flow memory, not None; this is the actual executor setting.
    registration['compiler']['belief'] = copy.deepcopy(T.SETTINGS['belief'])
    registration = P.bind(registration)
    journal.once('GAME_PLAN', 'game-'+job['job_id'], registration)
    raw = P.play_game(case, arm, registration['jobs'][0], registration)
    # play_game returns its in-memory receipt, where projection inventories may
    # still use integer keys even though their JSON representation is canonical.
    # The custody journal accepts only the exact finite JSON data model.
    raw = R._loads(P.wire(raw))
    C.need(isinstance(raw, dict) and raw == P.bind(raw), 'GAME_RECEIPT_HASH_MISMATCH')
    return raw


def _row(config, job, raw, audit):
    need(raw.get('status') == 'COMPLETE' and raw.get('checks_passed') is True and
         raw.get('physical_games_completed') == 1 and raw.get('physical_steps') == V.LAST_STEP+1,
         'INCOMPLETE_OR_CENSORED_GAME')
    epochs = raw.get('epochs', [])
    need([r.get('step') for r in epochs] == list(range(0, V.LAST_STEP+1, 24)), 'COMPLETE_DAWN_SEQUENCE_REQUIRED')
    need(epochs[0].get('decode_trace') == audit['decode_trace'] and
         epochs[0].get('raw_edits') == audit['draw']['raw_actions'], 'LIVE_DRAW_GAME_TRACE_MISMATCH')
    need(all(e.get('raw_edits') == [V.STOP_ID] and 'decode_trace' not in e for e in epochs[1:]),
         'DIRECT_LATER_NEURAL_PUBLICATION_REFUSED')
    need(raw.get('job', {}).get('mode') == 'sample_raw' and
         raw['job']['sampler_seed'] == job['sampler_seed'] and raw['case_id'] == _case(config, job)['id'],
         'GAME_JOB_CONTEXT_MISMATCH')
    wdl, _ = RL.terminal_wdl(raw['own_final_cash'], raw['rival_final_cash'])
    need(raw['wdl'] == wdl, 'GAME_WDL_MISMATCH')
    state, _, _ = R._actor(audit['actor_context'])
    game_id, draw_id = _job_ids(config, job)
    row = dict(game_id=game_id, group_id=job['group_id'], world_id=job['world_id'],
        world_sha256=job['world_sha256'], seat=job['seat'], opponent_id=job['opponent_id'],
        sampler_seed=job['sampler_seed'], status='COMPLETE', checks_passed=True, censored=False,
        physical_steps=raw['physical_steps'], own_final_cash=raw['own_final_cash'], rival_final_cash=raw['rival_final_cash'],
        wdl=wdl, actor=T.actor_observation(state), initial_plan=Plan().canonical(),
        decode_trace=copy.deepcopy(audit['decode_trace']), behavior_checkpoint_sha256=config['spec']['parent']['commit_sha256'])
    origin = dict(game_id=game_id, direct_draw_id=draw_id, type='direct_policy', mode='sample_raw', selection='none',
        confirmation_clone=False, retry_count=0, checkpoint_sha256=config['spec']['parent']['commit_sha256'],
        sampler_seed=job['sampler_seed'], trace_sha256=RL.digest(row['decode_trace']))
    return row, origin


def _write_same(path, value):
    data = C.wire(value)+b'\n'
    if path.exists():
        need(path.read_bytes() == data, 'IMMUTABLE_EXPORT_MISMATCH')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        RL._fsync_dir(path.parent)
    return dict(path=str(path.relative_to(ROOT)), sha256=RL.file_sha(path))


def _export_direct(config, journal, loaded, execution):
    plan = _direct_plan(config['spec']); rows, origins, failed = [], [], []
    counts = dict(registered_games=len(plan['collection_jobs']), attempted_jobs=0,
        draw_attempts=0, direct_policy_draws=0, completed_rows=0, native_start_charges=0,
        native_starts=0, native_games_completed=0, unknown_launches=0, unknown_native_launches=0,
        physical_steps_recorded=0, replay_independent_worlds_added=0, optimizer_updates=0)
    for job in plan['collection_jobs']:
        key = 'game-'+job['job_id']; events = journal.by_key.get(key, {})
        if 'LAUNCH' in events:
            counts['attempted_jobs'] += 1
            counts['native_start_charges'] += int(execution['kind'] == 'NATIVE_GAME')
        draw_events = journal.by_key.get(_draw_key(_job_ids(config, job)[1]), {})
        counts['draw_attempts'] += int('LAUNCH' in draw_events)
        if 'RESULT' in draw_events:
            draw_result = journal.get(draw_events['RESULT']['object'])
            counts['direct_policy_draws'] += int(draw_result['status'] == 'RETURNED')
        try:
            need('RESULT' in events, 'MISSING_OR_UNKNOWN_REGISTERED_GAME')
            result = journal.get(events['RESULT']['object'])
            if result['status'] == 'UNKNOWN':
                counts['unknown_launches'] += 1
                counts['unknown_native_launches'] += int(execution['kind'] == 'NATIVE_GAME')
            need(result['status'] == 'RETURNED', str(result.get('error')))
            raw = result['value']
            if execution['kind'] == 'NATIVE_GAME':
                counts['native_starts'] += raw.get('physical_games_started', 0)
                counts['physical_steps_recorded'] += raw.get('physical_steps', 0)
                counts['native_games_completed'] += raw.get('physical_games_completed', 0)
            audit = draw_receipt(config, _job_ids(config, job)[1])
            verify_persisted_draw(config, audit, loaded)
            row, origin = _row(config, job, raw, audit)
            rows.append(row); origins.append(origin)
        except (ValueError, KeyError, TypeError) as exc:
            failed.append(dict(job_id=job['job_id'], reason=str(exc)))
    counts['completed_rows'] = len(rows)
    document = dict(schema=RL.ROLLOUT_SCHEMA, plan_sha256=config['spec']['direct_plan']['sha256'],
        behavior_checkpoint_sha256=config['spec']['parent']['commit_sha256'],
        scorer_source_pins=plan['scorer_source_pins'], games=rows)
    corpus = dict(schema=RT.DIRECT_SCHEMA, document=document, provenance=origins)
    candidate = _write_same(journal.root/'snapshots'/('DIRECT-'+C.digest(corpus)+'.json'), corpus)
    ready = not failed and len(rows) == counts['registered_games']
    try:
        data = RL.validate_rollout_document(document, plan=plan,
            plan_sha256=config['spec']['direct_plan']['sha256'], rollout_sha256=candidate['sha256'])
        diagnostics = data.diagnostics
    except ValueError as exc:
        # The owner raises, rather than returns diagnostics, when no complete
        # group survives. Retain its rejection, not invented admission counts.
        ready = False
        diagnostics = dict(status='REJECTED', rejection=str(exc))
        failed.append(dict(job_id='ROLLOUT_BUFFER', reason=str(exc)))
    try:
        # The exact unchanged trainer gate owns direct provenance and denominators.
        RT._direct_data(dict(direct_plan=config['spec']['direct_plan'], direct_corpus=candidate), plan)
    except ValueError as exc:
        ready = False
        failed.append(dict(job_id='FIT_BUFFER', reason=str(exc)))
    direct_ref = _write_same(journal.root/'DIRECT.json', corpus) if ready else None
    report = C.bind(dict(schema=SCHEMA+'-result', status='COMPLETE' if ready else 'INCOMPLETE',
        fit_ready=ready, evidence_kind=execution['kind'], config_sha256=config['content_sha256'],
        direct_corpus=direct_ref, retained_snapshot=candidate, counts=counts, failed_jobs=failed,
        count_scope='NATIVE_START_CHARGES_INCLUDE_UNKNOWN_LAUNCHES;NATIVE_STARTS_COMPLETIONS_STEPS_ARE_RUNNER_REPORTED',
        diagnostics=diagnostics, execution=execution,
        rules=[w['kind'] for w in config['spec']['worlds']],
        scope='DIRECT_POLICY_OPENING_ONLY;FIXED_SHOP_IS_NOT_ORIGINAL_RULE_TRANSFER;NO_STRENGTH_CLAIM',
        journal_sha256=RL.file_sha(journal.path), parent_identity=loaded.identity))
    _write_same(journal.root/'snapshots'/('RESULT-'+report['content_sha256']+'.json'), report)
    return report


def collect_direct(config, *, execute=False):
    config = _config(config); _authorize(config, execute=execute)
    need(config['spec']['direct_plan'] is not None, 'DIRECT_PLAN_REQUIRED')
    execution = _runner_binding(config)
    with C._lock(_relative(config['spec']['output'], output=True)):
        journal = C.Journal(_relative(config['spec']['output'], output=True)); _freeze(journal, config)
        journal.once('EXECUTION', 'execution', execution)
        loaded = _load(config, journal)
        for job in _direct_plan(config['spec'])['collection_jobs']:
            key = 'game-'+job['job_id']
            result = journal.call(key, kind='direct_game', lane='direct',
                argument=dict(config_sha256=config['content_sha256'], job=job, execution=execution),
                callback=lambda _, job=job: _play(config, journal, loaded, job))
            if result['status'] != 'RETURNED':
                break  # Unknown side effects are never retried or followed by more games.
        return _export_direct(config, journal, loaded, execution)


def replay(config, *, execute=False):
    config = _config(config); _authorize(config, execute=execute)
    output = _relative(config['spec']['output'], output=True)
    need((output/'JOURNAL.jsonl').is_file(), 'EXISTING_PRODUCER_JOURNAL_REQUIRED')
    with C._lock(output):
        journal = C.Journal(output); _freeze(journal, config, existing=True)
        loaded = _load(config, journal)
        draws = (config['spec']['collector_draws'] if config['spec']['direct_plan'] is None else
                 [dict(draw_id=_job_ids(config, j)[1]) for j in _direct_plan(config['spec'])['collection_jobs']])
        verified, failed = [], []
        for row in draws:
            try:
                verified.append(verify_persisted_draw(config, draw_receipt(config, row['draw_id']), loaded))
            except (ValueError, KeyError, TypeError) as exc:
                failed.append(dict(draw_id=row['draw_id'], reason=str(exc)))
        return dict(status='VERIFIED' if not failed else 'INCOMPLETE', verified_draws=len(verified),
            registered_draws=len(draws), failures=failed, config_sha256=config['content_sha256'],
            physical_games_started=0, new_direct_draws=0, optimizer_updates=0,
            likelihood_scope='RAW_ID_PATH_REPLAY;NOT_FINAL_PLAN_MASS;NOT_GAME_REPLAY')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan'); p.add_argument('--spec', required=True); p.add_argument('--output', required=True)
    for name in ('check', 'callback', 'direct', 'replay'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True); p.add_argument('--config-sha256', required=True)
        p.add_argument('--execute', action='store_true')
        if name == 'check': p.add_argument('--authenticate-parent', action='store_true')
        if name == 'callback':
            p.add_argument('--context', required=True); p.add_argument('--context-sha256', required=True)
            p.add_argument('--draw-id', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'plan':
            result = build_config(R._loads(Path(args.spec).read_text()))
            RL._write_json_new(args.output, result)
        else:
            config = RL._pinned_json(args.config, args.config_sha256)
            if args.command == 'check':
                result = check(config)
                if args.authenticate_parent:
                    _authorize(config, execute=args.execute)
                    spec = config['spec']
                    info = P.inspect_committed_scorer(_relative(spec['parent']['path']), spec['parent']['commit_sha256'],
                                                      device=spec['device'], production=spec['production'])
                    result.update(parent_authentication='VERIFIED', parent_identity=info['identity'])
            elif args.command == 'callback':
                _authorize(config, execute=args.execute)
                row = next(r for r in config['spec']['collector_draws'] if r['draw_id'] == args.draw_id)
                ctx = RL._pinned_json(args.context, args.context_sha256)
                result = factory(config)['sampler'](ctx, draw_id=args.draw_id, seed=row['seed'],
                    parent_checkpoint_sha256=config['spec']['parent']['commit_sha256'])
            elif args.command == 'direct':
                result = collect_direct(config, execute=args.execute)
            else:
                result = replay(config, execute=args.execute)
        print(P.wire(result))
        return 0 if result.get('status') not in ('INCOMPLETE', 'INVALID') else 2
    except (ValueError, OSError, KeyError, TypeError, StopIteration) as exc:
        print(P.wire(dict(status='INVALID', error_type=type(exc).__name__, reason=str(exc))))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
