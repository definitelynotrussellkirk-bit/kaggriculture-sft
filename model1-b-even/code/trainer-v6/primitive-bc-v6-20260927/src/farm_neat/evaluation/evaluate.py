"""Plan-first full-economic MoE versus frozen ShopRouter or opening NN, both seats.

Purpose/owner: maintainer's fixed-opponent checkpoint comparison, never training.
Status: additive evaluation binding; actual game evidence is a closed RESULT.
Reuses the exact engine, EconomicOrders, entity packer, Graph.sample and frozen
ShopRouterAdapter. SAMPLED_T1/B1 is explicit, not historical B2 training parity
or an invented greedy decoder. Both seats share each declared world seed; they
are two fresh games, not two independent worlds or fresh reference witnesses.
Intended for the existing evaluation component; no optimizer, upload,
automatic retry, competing broker or copied model. Use the storage guard for run.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from farm_neat.training import sustain as common

SCHEMA = 'farm-full-economic-fixed-opponent-evaluation-v2'
MODE = 'SAMPLED_T1'
BATCH = 'ONE_ACTOR_B1_DYNAMIC_ENTITIES; NOT_B2_TRAINING_REPLAY'
MAX_RAW_BYTES = 32_000_000  # Total uncompressed JSONL, before writing/compression.
MAX_LINE_BYTES = 1_000_000
MAX_WORLDS = 64
NEURAL_OPPONENT = 'FROZEN_FULL_ECONOMIC_OPENING_POLICY'
STEPS = tuple(range(0, 719, 24))


def _bound(value):
    value = {k: v for k, v in value.items() if k != 'content_sha256'}
    return {**value, 'content_sha256': common.digest(value)}


def make_plan(root, checkpoint, model_sha256, *, seeds, actor_seed, owner, command_budget=8, mode=MODE, opening_only=False,
              opponent_checkpoint=None, opponent_model_sha256=None, opponent_seed=None, shop_revision_every=0, full_plan=False):
    """Small metadata/source reads only; no Torch import, tensors or world creation."""
    root = Path(root).resolve()
    neural_args = (opponent_checkpoint, opponent_model_sha256, opponent_seed)
    neural = any(x is not None for x in neural_args)
    if (type(shop_revision_every) is not int or shop_revision_every not in (0,2)
            or (opening_only and shop_revision_every)):
        raise ValueError('opening-only and shop-revision clocks must be explicit and mutually exclusive')
    fast_mode = opening_only or bool(shop_revision_every)
    if type(full_plan) is not bool or (full_plan and not fast_mode):
        raise ValueError('full-plan evaluation requires an explicit opening/morning clock')
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= MAX_WORLDS
            or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds) or len(set(seeds)) != len(seeds)
            or type(actor_seed) is not int or not 0 <= actor_seed <= 2**31-2*len(seeds)
            or type(opening_only) is not bool
            or type(command_budget) is not int or not 2 <= command_budget <= (256 if fast_mode else 8)
            or mode != MODE or not isinstance(owner, str) or not owner.strip()):
        raise ValueError('explicit 1..64 distinct owner-evaluation seeds, bounded sampler stream and supported SAMPLED_T1 command budget required')
    binding = common.checkpoint(root, checkpoint, tensors=False)
    base = root / binding['directory']
    manifest, parent = common.read(base/'model.pt.json'), common.read(base/'PLAN.json')
    if (binding['model_sha256'] != model_sha256
            or manifest.get('schema') != 'farm-full-economic-moe-policy-v1'
            or manifest.get('numerics') != 'TORCH_FP32_FULL_ECONOMIC_MOE_V1'
            or manifest.get('parameter_report', {}).get('total_trainable_parameters') != 60_403_676
            or manifest['parameter_report'].get('layers') != 16):
        raise ValueError('externally pinned intended full60M/16-layer checkpoint required, not old29/44M')
    if fast_mode:
        from farm_neat.training import campaign as campaign
        # A deliberate current-executor comparison of frozen checkpoints, NOT old-run reproduction.
        if manifest['source_pins'].get('src/farm_neat/policy/network.py') == campaign.policy.LEGACY_SOURCE_SHA256:
            campaign.policy.check_model_sources(manifest, allow_remaining_window_transition=True)
        else:
            common.attest(root, manifest['source_pins'])
        execution_pins = campaign.sources()
    else:
        common.attest(root, parent['source_pins'])  # Never overwrite a stale parent pin with current bytes.
        execution_pins = parent['source_pins']
    pins = dict(execution_pins)
    extra = ()
    if not neural:
        from farm_neat.evaluation import shop_router as router
        opponent = router.identity()
        # The adapter's own closure; the old semantic-benchmark pins went with the
        # modules that were not migrated.
        pins.update({**router.sources(), **opponent['assets']})
        extra = ('src/farm_neat/evaluation/shop_router.py', 'src/farm_neat/evaluation/opponents.py',
                 'src/farm_neat/evaluation/shop_router_0909/PROVENANCE.json')
    for name in ('src/farm_neat/evaluation/evaluate.py', 'tests/contracts/test_evaluation.py',
                 'src/farm_neat/training/sustain.py', *extra):
        pins[name] = common.sha(common.path(root, name))
    common.attest(root, pins)
    profile = {k: v for k, v in parent['environment'].items() if k != 'bitsandbytes'}
    if (profile.get('dtype') != 'FLOAT32' or profile.get('tf32') is not False
            or profile.get('autocast') is not False):
        raise ValueError('recorded FP32 no-TF32/no-autocast profile required')
    if neural:
        if (not fast_mode or any(x is None for x in neural_args)
                or type(opponent_seed) is not int or not 0 <= opponent_seed <= 2**31-2*len(seeds)):
            raise ValueError('frozen neural opponent requires opening-only, checkpoint/hash and bounded separate sampler seed')
        if set(range(actor_seed, actor_seed+2*len(seeds))) & set(range(opponent_seed, opponent_seed+2*len(seeds))):
            raise ValueError('learner/opponent sampler streams must be disjoint')
        peer = common.checkpoint(root, opponent_checkpoint, tensors=False)
        peer_dir = root/peer['directory']
        peer_manifest, peer_plan = common.read(peer_dir/'model.pt.json'), common.read(peer_dir/'PLAN.json')
        if (peer['model_sha256'] != opponent_model_sha256
                or peer_manifest.get('schema') != manifest['schema']
                or peer_manifest.get('numerics') != manifest['numerics']
                or peer_manifest.get('architecture') != manifest.get('architecture')
                or peer_manifest.get('parameter_report') != manifest['parameter_report']
                or {k: v for k, v in peer_plan['environment'].items() if k != 'bitsandbytes'} != profile):
            raise ValueError('same full-economic architecture/profile and externally pinned opponent required')
        if peer_manifest['source_pins'].get('src/farm_neat/policy/network.py') == campaign.policy.LEGACY_SOURCE_SHA256:
            campaign.policy.check_model_sources(peer_manifest, allow_remaining_window_transition=True)
        else:
            common.attest(root, peer_manifest['source_pins'])
        opponent = _bound({'kind': NEURAL_OPPONENT, 'checkpoint': peer,
            'model_manifest_file_sha256': common.sha(peer_dir/'model.pt.json'),
            'checkpoint_source_pins': peer_plan['source_pins'],
            'economic_clock': campaign.execution_for(True)['economic_clock'],
            'decision_steps': [0], 'command_budget': command_budget, 'temperature': 1.0,
            'behavior': 'FROZEN_MODEL_CURRENT_OPENING_EXECUTOR; ONE_SAMPLED_PLAN_THEN_OBSERVATION_RESPONSIVE_LOGISTICS',
            'training_admission': False})
    return _bound({'schema': SCHEMA+'-plan', 'owner': owner, 'checkpoint': binding,
        'model_manifest_file_sha256': common.sha(base/'model.pt.json'), 'source_pins': pins,
        'opponent': opponent, 'mode': mode, 'neural_batch': 'FAST_FP32_B1_BUCKET32_PCG64_NOT_TRAINING_BATCH_PARITY' if fast_mode else BATCH, 'temperature': 1.0,
        'seeds': seeds, 'actor_seed': actor_seed, 'command_budget': command_budget,
        'opponent_seed': opponent_seed,
        'environment': profile, 'physical_steps': 719, 'decision_steps': [0,144,288,432,576] if shop_revision_every else [0] if opening_only else list(STEPS),
        'opening_only': opening_only, 'checkpoint_source_pins': parent['source_pins'],
        **({'shop_revision_every':shop_revision_every} if shop_revision_every else {}),
        **({'full_plan':True, 'plan_adapter':campaign.plan_features.identity(),
            'sampling_support': 'REMAINING_WINDOW_V2; UNTIL_GE_ACTUAL_STEP; PAST_STARTS_ALLOWED'} if full_plan else {}),
        'execution_boundary': 'CURRENT_SHOP_REVISION_EXECUTOR_WITH_FROZEN_MODEL_NOT_HISTORICAL_REPRODUCTION' if shop_revision_every else
                             'CURRENT_OPENING_EXECUTOR_WITH_FROZEN_MODEL_NOT_HISTORICAL_REPRODUCTION' if opening_only else 'EXACT_PARENT_SOURCES',
        'conditions': [{'id': f'world-{seed}-seat-{seat}', 'seed': seed, 'seat': seat,
                        'actor_rng_seed': actor_seed+2*i+seat,
                        **({'opponent_rng_seed': opponent_seed+2*i+seat} if opponent_seed is not None else {})}
                       for i, seed in enumerate(seeds) for seat in (0, 1)],
        'seed_scope': 'OWNER_ASSERTED_OPENED_EVALUATION; NOT_CROSS_RUN_LEDGER_AUTHENTICATION_OR_HELDOUT',
        'max_raw_jsonl_bytes': MAX_RAW_BYTES*max(1, (len(seeds)+7)//8), 'max_line_bytes': MAX_LINE_BYTES,
        'failure_policy': 'FAIL_FAST_RETAIN_PARTIAL; NO_REROLL_OR_LOSS_IMPUTATION',
        'trace_scope': 'ACTUAL_PROGRAMS_ACTIONS_OBSERVATION_AND_INPUT_HASHES; NO_DUPLICATE_FEATURE_TENSORS',
        'training_admission': False, 'optimizer_updates': 0, 'fresh_reference_games': 0,
        'population_strength_claim': False})


def check_plan(root, plan):
    peer = plan['opponent'].get('checkpoint', {}) if plan['opponent'].get('kind') == NEURAL_OPPONENT else {}
    expected = make_plan(root, plan['checkpoint']['directory'], plan['checkpoint']['model_sha256'],
        seeds=plan['seeds'], actor_seed=plan['actor_seed'], owner=plan['owner'],
        command_budget=plan['command_budget'], mode=plan['mode'], opening_only=plan['opening_only'],
        opponent_checkpoint=peer.get('directory'), opponent_model_sha256=peer.get('model_sha256'), opponent_seed=plan['opponent_seed'],
        shop_revision_every=plan.get('shop_revision_every',0), full_plan=plan.get('full_plan',False))
    if plan != expected:
        raise ValueError('evaluation plan/source/model/opponent/profile changed')


class Runtime:
    """One read-only graph; fresh world, economic controller, opponent and RNG per case."""
    def __init__(self, root, plan, *, with_opponent=True):
        import torch
        from farm_neat.training import moe_lab as lab
        from farm_neat.evaluation.shop_router import ShopRouterAdapter
        from farm_neat.policy import network as policy, full_economic_rollout as rollout
        from farm_neat.economy.orders import EconomicOrders
        from farm_neat.policy.observation import pack
        from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
        if not sys.dont_write_bytecode or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('explicit-B/:4096:8 CUDA evaluation profile required')
        from farm_neat.training import campaign as campaign
        self.fast_mode = plan['opening_only'] or bool(plan.get('shop_revision_every'))
        expected_profile = campaign.environment_profile() if self.fast_mode else lab.device_profile()
        if expected_profile != plan['environment']:
            raise ValueError('recorded current CUDA profile required; no CPU fallback')
        self.graph, manifest = policy.load_model(root/plan['checkpoint']['directory']/'model.pt',
            device='cuda', expected_file_sha256=plan['checkpoint']['model_sha256'],
            allow_remaining_window_transition=self.fast_mode)
        if manifest['content_sha256'] != plan['checkpoint']['manifest_sha256']:
            raise ValueError('loaded model differs from the registered manifest')
        self.graph.eval()
        self.architecture_sha256 = manifest['architecture']['content_sha256']
        self.torch, self.lab, self.rollout, self.pack = torch, lab, rollout, pack
        self.environment, self.controller, self.opponent = ExactKaggricultureEnvironment, EconomicOrders, ShopRouterAdapter
        self.plan = plan
        if self.fast_mode:
            from farm_neat.policy import fast_profile as fast
            self.fast, self.campaign = fast, campaign
            self.fast_identity = fast.identity()
            self.encoder = fast.FastEncoder(self.graph, expected_identity=self.fast_identity)
            self.sampler = fast.GraphedSampler(self.graph, 1, 1.0)
        if with_opponent and plan['opponent'].get('kind') == NEURAL_OPPONENT:
            # Read-only weights may be shared, never controller/RNG/observations.
            peer = plan['opponent']['checkpoint']
            self.peer_runtime = self if peer['model_sha256'] == plan['checkpoint']['model_sha256'] else Runtime(
                root, {**plan, 'checkpoint': peer}, with_opponent=False)

    def neural_opponent(self, condition):
        return OpeningOpponent(self.peer_runtime, self.plan['opponent']['content_sha256'], condition['opponent_rng_seed'])

    def generator(self, seed):
        if self.fast_mode: return self.fast.LaneStream(seed)
        return self.torch.Generator(device='cuda').manual_seed(seed)

    def decide(self, observation, orders, configuration, generator):
        from farm_neat.economy.orders import FullSeasonPlan
        packed = (self.campaign.pack_plan(self.plan, observation, orders, configuration)
                  if isinstance(orders, FullSeasonPlan) else self.pack(observation, orders, configuration))
        with self.torch.no_grad():
            if self.fast_mode:
                batch = self.campaign.collate_bucket([packed])
                tensors = tuple(self.torch.as_tensor(batch[k], device='cuda') for k in ('rows', 'token_mask', 'knownness'))
                context, values, _ = self.encoder(*tensors)
                records, _, _ = self.sampler.sample(context, [generator], max_commands=self.plan['command_budget'],
                    **({'as_of_step': observation['step']} if isinstance(orders, FullSeasonPlan) else {}))
                _, entropy = self.fast.replay_padded(self.graph, context, records, expected_identity=self.fast_identity)
            else:
                context, values, _ = self.graph.encode(*self.lab.tensor_inputs(self.graph, [packed]))
                records, _, entropy = self.graph.sample(context, generator=generator,
                    max_commands=self.plan['command_budget'], temperature=1.0)
        behavior = records[0]
        self.rollout._trace(behavior, {'architecture_sha256': self.architecture_sha256,
            'temperature': 1.0, 'command_budget': self.plan['command_budget'],
            'full_plan': isinstance(orders, FullSeasonPlan)}, step=observation['step'])
        return {'behavior': behavior, 'value': float(values[0]), 'conditional_entropy': float(entropy[0]),
                'input_sha256': packed['content_sha256'], 'entity_tokens': len(packed['rows'])}

    def observation_sha(self, observation):
        return self.rollout.observation_sha(observation)


class OpeningOpponent:
    """Fresh per-game state/RNG, frozen neural weights and observation-based executor."""
    def __init__(self, runtime, identity, seed):
        self.runtime, self.identity = runtime, identity
        self.controller, self.generator = runtime.controller(), runtime.generator(seed)
        self.decision, self.rejected, self.steps, self.inference_seconds = None, [], 0, 0.0

    def act(self, observation, configuration):
        from farm_neat.training.campaign import apply_opening
        if observation['step'] != self.steps:
            raise ValueError('frozen opponent requires contiguous own observations')
        self.controller.observe(observation)
        if self.steps == 0:
            started = time.perf_counter()
            self.decision = self.runtime.decide(observation, self.controller, configuration, self.generator)
            self.inference_seconds = time.perf_counter()-started
            self.controller, self.rejected = apply_opening(self.controller, self.decision['behavior']['program'], step=0)
        action = self.controller(observation, configuration)
        self.steps += 1
        return action

    def finish(self, observation):
        if self.steps != 719 or observation['step'] != 719:
            raise ValueError('terminal opponent settlement requires719 completed turns')
        self.controller.observe(observation)

    def snapshot(self):
        return {'identity': self.identity, 'physical_steps': self.steps, 'neural_calls': int(self.decision is not None),
                'controller_sha256': common.digest(self.controller.snapshot()),
                'opening_decision_sha256': common.digest(self.decision), 'rejected_edits': self.rejected}


def play(plan, condition, runtime, emit, progress):
    """Maintained observe/daily-pack/apply/act order, one responsive joint engine step."""
    env = runtime.environment(seed=condition['seed'])
    progress.update(environment_created=True, physical_steps=0, decisions=0)
    controller = runtime.controller()
    if plan.get('full_plan'):
        from farm_neat.economy.orders import FullSeasonPlan
        controller = FullSeasonPlan(condition['id'], publication_steps=plan['decision_steps'])
    neural_opponent = plan['opponent'].get('kind') == NEURAL_OPPONENT
    opponent = runtime.neural_opponent(condition) if neural_opponent else runtime.opponent()
    generator = runtime.generator(condition['actor_rng_seed'])
    seat, other = condition['seat'], 1-condition['seat']
    rejected, inference_seconds, engine_seconds = 0, 0.0, 0.0
    while not env.done:
        step = env.step_index
        if step != progress['physical_steps'] or step >= 719:
            raise ValueError('exact contiguous actionable steps0..718 required')
        obs = [env.agent_observation(s) for s in (0, 1)]
        controller.observe(obs[seat])
        decision, errors = None, []
        if step in plan['decision_steps']:
            if plan.get('shop_revision_every'):
                from farm_neat.training.campaign import check_decision_observation
                check_decision_observation(plan,step,obs[seat],env.configuration)
            started = time.perf_counter()
            decision = runtime.decide(obs[seat], controller, env.configuration, generator)
            inference_seconds += time.perf_counter()-started
            if any(type(decision[k]) not in (float, int) or not math.isfinite(decision[k])
                   for k in ('value', 'conditional_entropy')):
                raise ValueError('finite actor/critic diagnostic required')
            progress['decisions'] += 1
            if plan['opening_only'] or plan.get('shop_revision_every'):
                from farm_neat.training.campaign import publish_program
                controller, errors = publish_program(plan, controller, decision['behavior']['program'], step=step, observation=obs[seat])
            else:
                for command in decision['behavior']['program']:
                    try:
                        controller.apply(command)
                    except ValueError as exc:
                        errors.append({'command': command, 'error': str(exc)[:1000]})
        actions = [None, None]
        actions[seat] = controller(obs[seat], env.configuration)
        # Always invoke the frozen opponent on THIS fresh world's observations.
        actions[other] = opponent.act(obs[other], env.configuration)
        before = [runtime.observation_sha(o) for o in obs]
        started = time.perf_counter(); env.step(actions); engine_seconds += time.perf_counter()-started
        progress['physical_steps'] = env.step_index
        rejected += len(errors)
        emit({'schema': SCHEMA+'-turn', 'step': step, 'learner_seat': seat,
              'observation_sha256': before, 'decision': decision, 'rejected_edits': errors,
              'opponent_decision': opponent.decision if neural_opponent and step == 0 else None,
              'opponent_rejected_edits': opponent.rejected if neural_opponent and step == 0 else [],
              'actions': actions, 'successor_observation_sha256': [runtime.observation_sha(env.agent_observation(s)) for s in (0, 1)],
              'learner_orders_sha256': common.digest(controller.snapshot())})
    if env.step_index != 719 or progress['decisions'] != len(plan['decision_steps']) or [s.status for s in env.state] != ['DONE', 'DONE']:
        raise ValueError('censored or failed game cannot supply a score')
    terminal = [env.agent_observation(s) for s in (0, 1)]
    controller.observe(terminal[seat])
    if neural_opponent: opponent.finish(terminal[other])
    cash = [terminal[0]['farms'][s]['money'] for s in (0, 1)]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in cash):
        raise ValueError('finite nonnegative terminal cash required')
    opponent_state = opponent.snapshot()
    if opponent_state.get('identity') != plan['opponent']['content_sha256']:
        raise ValueError('actual responsive opponent instance differs from plan')
    if neural_opponent and (opponent_state['neural_calls'] != 1 or opponent_state['physical_steps'] != 719):
        raise ValueError('frozen opening opponent must execute all719 turns with one neural call')
    return {'condition': condition, 'status': 'COMPLETE', 'physical_steps': 719, 'model_decisions': len(plan['decision_steps']),
        'cash': cash, 'wdl': int(cash[seat] > cash[other])-int(cash[seat] < cash[other]),
        'rejected_edits': rejected, 'inference_seconds': inference_seconds, 'engine_seconds': engine_seconds,
        'frozen_opponent_neural_calls': 1 if neural_opponent else 0,
        'frozen_opponent_inference_seconds': opponent.inference_seconds if neural_opponent else 0.0,
        'terminal_observation_sha256': [runtime.observation_sha(o) for o in terminal],
        'terminal_orders_sha256': common.digest(controller.snapshot()),
        'opponent_state_sha256': common.digest(opponent_state)}


def score_slices(rows):
    """Preserve whole-world clusters and seats; margin is diagnostic, not score."""
    def summarize(selected):
        return {'games': len(selected), 'score': sum((r['wdl']+1)/2 for r in selected)/len(selected),
            'mean_cash_margin': sum(r['cash'][r['condition']['seat']]-r['cash'][1-r['condition']['seat']] for r in selected)/len(selected)}
    return {dimension: {str(v): summarize([r for r in rows if r['condition'][field] == v])
        for v in sorted({r['condition'][field] for r in rows})}
        for dimension, field in (('by_whole_world_seed', 'seed'), ('by_seat', 'seat'))}


def run(root, output, *, plan_file_sha256, owner, runtime_factory=Runtime):
    root, output = Path(root).resolve(), common.path(root, output, output=True)
    plan = common.read(output/'PLAN.json')
    if common.sha(output/'PLAN.json') != plan_file_sha256 or owner != plan['owner']:
        raise ValueError('exact predeclared plan hash and maintainer owner required')
    if set(p.name for p in output.iterdir()) != {'PLAN.json'}:
        raise ValueError('execution requires an unattempted plan; no overwrite or retry')
    check_plan(root, plan)
    common.write(output/'EXECUTION-START.json', {'plan_file_sha256': plan_file_sha256, 'owner': owner})
    started, rows, failures, raw_bytes, progress = time.perf_counter(), [], [], 0, {}
    try:
        runtime = runtime_factory(root, plan)
        cold_seconds = time.perf_counter()-started
        for condition in plan['conditions']:
            progress = {'condition': condition, 'environment_created': False, 'physical_steps': 0, 'decisions': 0}
            check_plan(root, plan)
            trace = output/(condition['id']+'.jsonl.gz')
            with trace.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
                def emit(row):
                    nonlocal raw_bytes
                    data = (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
                    if len(data) > plan['max_line_bytes'] or raw_bytes+len(data) > plan['max_raw_jsonl_bytes']:
                        raise ValueError('declared evaluation raw-output cap reached; no completed score')
                    stream.write(data); raw_bytes += len(data)
                result = play(plan, condition, runtime, emit, progress)
            check_plan(root, plan)
            if common.sha(root/plan['checkpoint']['directory']/'model.pt') != plan['checkpoint']['model_sha256']:
                raise ValueError('behavior checkpoint changed during evaluation')
            if plan['opponent'].get('kind') == NEURAL_OPPONENT:
                peer = plan['opponent']['checkpoint']
                if common.sha(root/peer['directory']/'model.pt') != peer['model_sha256']:
                    raise ValueError('frozen opponent checkpoint changed during evaluation')
            result.update(plan_sha256=plan['content_sha256'], model_sha256=plan['checkpoint']['model_sha256'],
                          opponent_sha256=plan['opponent']['content_sha256'], trace=trace.name,
                          trace_sha256=common.sha(trace), training_admission=False, fresh_reference_games=0)
            result = _bound(result)
            common.write(output/(condition['id']+'.json'), result)
            rows.append(result)
    except Exception as exc:
        failures.append({'error_type': type(exc).__name__, 'error': str(exc)[:2000], 'progress': progress})
        common.write(output/'FAILURE.json', failures[-1])
        cold_seconds = locals().get('cold_seconds')
    wins, draws, losses = (sum(r['wdl'] == v for r in rows) for v in (1, 0, -1))
    failed_started = bool(failures and progress.get('environment_created'))
    failed_terminal = failed_started and progress.get('physical_steps') == 719
    result = _bound({'schema': SCHEMA+'-result', 'status': 'PARTIAL_FAILED' if failures else 'COMPLETE',
        'plan_sha256': plan['content_sha256'], 'plan_file_sha256': plan_file_sha256,
        'completed_cases': rows, 'failures': failures, 'scheduled_games': len(plan['conditions']),
        'score_slices': score_slices(rows),
        'physical_game_attempts': len(rows)+int(failed_started),
        'completed_physical_games': len(rows)+int(failed_terminal), 'scored_games': len(rows),
        'scored_unique_worlds': len({r['condition']['seed'] for r in rows}),
        'wins': wins, 'draws': draws, 'losses': losses,
        'completed_only_score': (wins+0.5*draws)/len(rows) if rows else None,
        'scheduled_score_lower_bound': (wins+0.5*draws)/len(plan['conditions']),
        'failure_is_not_an_imputed_loss': True, 'cold_seconds': cold_seconds,
        'elapsed_seconds': time.perf_counter()-started, 'raw_jsonl_bytes': raw_bytes,
        'training_admission': False, 'optimizer_updates': 0, 'fresh_reference_games': 0,
        'population_strength_claim': False, 'mode': plan['mode'], 'neural_batch': plan['neural_batch']})
    common.write(output/'RESULT.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_mutually_exclusive_group()
    actions.add_argument('--prepare', action='store_true'); actions.add_argument('--execute', action='store_true')
    p.add_argument('--checkpoint'); p.add_argument('--model-sha256'); p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--actor-seed', type=int); p.add_argument('--owner', required=True)
    p.add_argument('--opponent-checkpoint'); p.add_argument('--opponent-model-sha256'); p.add_argument('--opponent-seed', type=int)
    p.add_argument('--command-budget', type=int, default=8); p.add_argument('--mode', default=MODE, choices=[MODE])
    p.add_argument('--opening-only', action='store_true', help='one atomic opening program; current fast executor, full responsive opponent game')
    p.add_argument('--full-plan',action='store_true',help='evaluate immutable-past/all-stage full plans; frozen opponents retain their own executor')
    p.add_argument('--shop-revision-every',type=int,choices=(2,),default=0,help='opening then visible shops2/4/6/8; frozen opponents keep their own clock')
    p.add_argument('--output'); p.add_argument('--plan-sha256')
    a = p.parse_args(argv)
    if a.execute:
        if not a.output or not a.plan_sha256:
            p.error('--execute requires --output and --plan-sha256')
        result = run(ROOT, a.output, plan_file_sha256=a.plan_sha256, owner=a.owner)
    else:
        if not a.checkpoint or not a.model_sha256 or a.seeds is None or a.actor_seed is None:
            p.error('plan requires --checkpoint, --model-sha256, --seeds and --actor-seed')
        result = make_plan(ROOT, a.checkpoint, a.model_sha256, seeds=a.seeds, actor_seed=a.actor_seed,
                           owner=a.owner, command_budget=a.command_budget, mode=a.mode, opening_only=a.opening_only,
                           opponent_checkpoint=a.opponent_checkpoint, opponent_model_sha256=a.opponent_model_sha256, opponent_seed=a.opponent_seed,
                           shop_revision_every=a.shop_revision_every, full_plan=a.full_plan)
        if a.prepare:
            if not a.output:
                p.error('--prepare requires a new --output')
            output = common.path(ROOT, a.output, output=True)
            if output.exists():
                raise ValueError('new output required; no plan overwrite')
            common.write(output/'PLAN.json', result)
            result = {'status': 'PREPARED_NO_GAMES', 'output': str(output),
                      'plan_file_sha256': common.sha(output/'PLAN.json'), 'conditions': result['conditions']}
    print(json.dumps(result, sort_keys=True, indent=2))
    return 1 if result.get('status') == 'PARTIAL_FAILED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
