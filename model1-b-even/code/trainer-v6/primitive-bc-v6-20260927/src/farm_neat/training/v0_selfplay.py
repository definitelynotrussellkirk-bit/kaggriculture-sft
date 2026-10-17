"""Fresh V0 mirror games -> one on-policy terminal-WDL update -> fresh games.

Separate protocol from the fixed-corpus RL pilot. Both seats sample independent
raw paths from the frozen cohort policy. One physical game supplies two correlated
episodes. Opening-only publication, exact original rules, no reward shaping.
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
import time

import torch

from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import State
from farm_neat.evaluation import v0_policy as E
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
from farm_neat.policy import v0_scorer as A
from farm_neat.policy.qwen_hybrid import Config
from farm_neat.training import v0_bc as BC, v0_rl as RL, v0_targets as T

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'farm-v0-fresh-mirror-pg-v1'
PROTOCOL = {'reward': 'TERMINAL_WDL_MINUS1_0_PLUS1', 'baseline': 'ZERO',
            'decision_scope': 'OPENING_ONLY_LATER_KEEP',
            'likelihood': 'SAMPLED_RAW_PATH_AT_REGISTERED_TEMPERATURE',
            'reduction': 'MEAN_OVER_BOTH_SEATS;TWO_CORRELATED_EPISODES_PER_GAME',
            'updates_per_fresh_cohort': 1, 'learning_rate': 1e-5,
            'max_grad_norm': 1.0, 'promotion': 'NONE'}


def sources():
    return {**E.source_pins(), **RL.source_pins(),
            str(Path(__file__).relative_to(ROOT)): RL.file_sha(__file__)}


def checkpoint_sources_compatible(recorded, *, transition=None):
    """Strict by default after the execution change; explicit provenance for old parents."""
    if recorded == sources():
        return transition is None
    try:
        return transition is not None and transition == execution_source_transition(recorded)
    except (ValueError, TypeError):
        return False


def validate_plan(plan):
    if plan['schema'] != SCHEMA or plan['protocol'] != PROTOCOL or plan['sources'] != sources():
        raise ValueError('preregistered self-play protocol/source identity mismatch')
    collection_options(plan)
    transition = plan.get('execution_source_transition')
    if transition is not None:
        if (not isinstance(transition, dict) or 'from_sources' not in transition
                or transition != execution_source_transition(transition['from_sources'])):
            raise ValueError('explicit execution source transition mismatch')
    rounds, evaluation = plan['train_worlds'], plan['evaluation_worlds']
    if not isinstance(rounds, list) or not 1 <= len(rounds) <= 16:
        raise ValueError('1..16 bounded cohorts required')
    if any(not isinstance(r, list) or not 2 <= len(r) <= 32 for r in rounds):
        raise ValueError('2..32 fresh games per cohort required')
    training = [s for row in rounds for s in row]
    if not isinstance(evaluation, list) or not 2 <= len(evaluation) <= 32:
        raise ValueError('2..32 independent evaluation worlds required')
    seeds = training + evaluation
    if any(type(s) is not int or not 0 <= s < 2**63 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('distinct train/evaluation world seeds required')
    if not math.isfinite(plan['temperature']) or not 0 < plan['temperature'] <= 5:
        raise ValueError('bounded positive temperature required')
    if not 0 < plan['game_timeout'] <= 600:
        raise ValueError('bounded per-game timeout required')
    train_rng = {plan['sampler_base'] + i * 10000 + j
                 for i, row in enumerate(rounds) for j in range(2 * len(row))}
    eval_rng = {plan['eval_sampler_base'] + j for j in range(2 * len(evaluation))}
    if train_rng & eval_rng or any(type(s) is not int or not 0 <= s < 2**63 for s in train_rng | eval_rng):
        raise ValueError('distinct private sampler allocations required')


class Opening(E.OpeningScorerPolicy):
    def __call__(self, state, plan, masks):
        if state.step:
            return super().__call__(state, plan, masks)
        self.actor = T.actor_observation(state)
        with torch.inference_mode():
            result = A.decode_epoch(self.scorer, BC.actor_state(self.actor), plan, generator=self.generator,
                mode=self.mode, temperature=self.temperature,
                projection_kwargs=copy.deepcopy(masks.kw), backend='fast')
        self.last_trace = copy.deepcopy(result.trace)
        self.opening_trace = copy.deepcopy(result.trace)
        return result.actions


def collect_game(left, right, *, seed, sampler_seeds, identity, temperature, timeout,
                 failure_path=None):
    if len(sampler_seeds) != 2 or sampler_seeds[0] == sampler_seeds[1]:
        raise ValueError('two independent sampler seeds required')
    started = time.monotonic()
    env = ExactKaggricultureEnvironment(seed=seed)
    policies = [Opening(model, identity=identity, mode='sample_raw', seed=s,
                        temperature=temperature)
                for model, s in zip((left, right), sampler_seeds)]
    agents = [E.PolicyAgent(p, game_id=f'selfplay-{seed}-{seat}',
                            labor_factor=4., belief=copy.deepcopy(T.SETTINGS['belief']))
              for seat, p in enumerate(policies)]
    actions_hash = hashlib.sha256()
    try:
        while not env.done:
            if time.monotonic() - started > timeout:
                raise TimeoutError(f'self-play game {seed} exceeded its registered timeout')
            joint = [agent(copy.deepcopy(env.agent_observation(seat)), copy.deepcopy(env.configuration))
                     for seat, agent in enumerate(agents)]
            actions_hash.update(E.wire(joint).encode())
            env.step(joint)
        if env.step_index != 719:
            raise ValueError('incomplete game cannot produce terminal learning labels')
        cash = [float(env.agent_observation(0)['farms'][i]['money']) for i in (0, 1)]
        reward = (cash[0] > cash[1]) - (cash[0] < cash[1])
        episodes = []
        for seat, policy in enumerate(policies):
            trace = policy.opening_trace
            if any(r.get('refusals') for r in agents[seat].records):
                raise ValueError('publication refusal; preserve failed game, do not fit it')
            episodes.append({'seat': seat, 'sampler_seed': sampler_seeds[seat],
                'actor': policy.actor, 'trace': trace, 'old_logp': trace['raw_path_logp'],
                'reward': reward if seat == 0 else -reward})
        return {'seed': seed, 'steps': env.step_index, 'physical_games': 1,
                'episodes': episodes, 'cash': cash, 'joint_actions_sha256': actions_hash.hexdigest(),
                'seconds': time.monotonic() - started}
    except BaseException as exc:
        if failure_path is not None:
            RL._write_json_new(Path(failure_path), {
                'status': 'FAILED_NOT_FOR_TRAINING', 'error': repr(exc),
                'seed': seed, 'sampler_seeds': sampler_seeds, 'identity': identity,
                'temperature': temperature, 'steps': env.step_index,
                'joint_actions_sha256': actions_hash.hexdigest(),
                'policies': [{'seat': seat, 'actor': getattr(policy, 'actor', None),
                    'trace': getattr(policy, 'opening_trace', None),
                    'publications': agents[seat].records,
                    'last_observation': env.agent_observation(seat)}
                    for seat, policy in enumerate(policies)]})
        raise


# Execution mechanics are registered separately from the unchanged learning protocol.
COLLECTION_SCHEMA = 'farm-v0-spawn-collection-v1'
SELFPLAY_SOURCE = 'src/farm_neat/training/v0_selfplay.py'
# Exact integrated spawn-collector preimage shipped with slot 64, not a wildcard.
PRE_OPENING_BATCH_SOURCE = '66d728f05ccca50e97d0045cf88eaed87011a36e886afcbb99e7b2755e1e7264'
OPENING_BATCH_SCHEMA = 'farm-v0-selfplay-opening-batch-v1'
OPENING_BATCH_NUMERICS = 'FP32_RAGGED_SCORE_BATCH_RAW_CATEGORICAL_V1'


def opening_batch_settings(max_live_rows=8, max_wait_seconds=.01):
    # An opening has two seats. Odd bounds are valid; the scheduler admits
    # floor(bound / 2) complete openings and never rounds the row cap upward.
    if type(max_live_rows) is not int or not 2 <= max_live_rows <= 64:
        raise ValueError('2..64 integer opening batch live rows required')
    if (type(max_wait_seconds) not in (int, float) or not math.isfinite(max_wait_seconds)
            or not 0 <= max_wait_seconds <= 1):
        raise ValueError('opening batch wait must be finite and within 0..1 seconds')
    return {'schema': OPENING_BATCH_SCHEMA, 'max_live_rows': max_live_rows,
            'max_wait_seconds': float(max_wait_seconds), 'numerics': OPENING_BATCH_NUMERICS,
            'cross_device_bitwise_parity': False}


def _registered_opening_batch(given):
    try:
        expected = opening_batch_settings(given['max_live_rows'], given['max_wait_seconds'])
    except (KeyError, TypeError) as exc:
        raise ValueError('complete registered opening batch settings required') from exc
    if given != expected:
        raise ValueError('registered opening batch settings mismatch')
    return expected

PRE_PARALLEL_SOURCES = frozenset({
    '37a80bad4f06d81a24347873175f42be648924199514f03257e0e249fe6a0344',
    'a96930417527fcf6db58dd3d6d501d593b749bacbd55a2957a3aa97d23b07489',
})


def collection_settings(workers=1, *, startup_timeout=60., cleanup_timeout=5., opening_batch=None):
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('1..32 integer collection workers required')
    for name, value, bound in (('startup', startup_timeout, 120), ('cleanup', cleanup_timeout, 30)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= bound:
            raise ValueError(f'bounded positive {name} timeout required')
    result = {'schema': COLLECTION_SCHEMA, 'workers': workers, 'start_method': 'spawn',
              'startup_timeout': startup_timeout, 'cleanup_timeout': cleanup_timeout}
    if opening_batch is not None:
        result['opening_batch'] = _registered_opening_batch(opening_batch)
    return result


def collection_options(plan, *, allocation=False):
    given = plan.get('collection', collection_settings())
    try:
        expected = collection_settings(given['workers'], startup_timeout=given['startup_timeout'],
                                       cleanup_timeout=given['cleanup_timeout'],
                                       opening_batch=given.get('opening_batch'))
    except (KeyError, TypeError) as exc:
        raise ValueError('complete registered collection settings required') from exc
    if given != expected:
        raise ValueError('registered spawn collection settings mismatch')
    if allocation and given['workers'] > 1:
        available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            try:
                available = min(available, int(os.environ['SLURM_CPUS_PER_TASK']))
            except ValueError as exc:
                raise ValueError('invalid SLURM CPU allocation') from exc
        if given['workers'] > available:
            raise ValueError('collection workers exceed CPU allocation')
    return dict(given)


def execution_source_transition(recorded):
    """Prospective opt-in only; never change a historical COMMIT or state file."""
    old, new = dict(recorded), sources()
    if (old.get(SELFPLAY_SOURCE) not in PRE_PARALLEL_SOURCES | {PRE_OPENING_BATCH_SOURCE}
            or {k: v for k, v in old.items() if k != SELFPLAY_SOURCE}
            != {k: v for k, v in new.items() if k != SELFPLAY_SOURCE}):
        raise ValueError('only the supplied enumerated self-play sources may transition')
    return {'schema': 'farm-v0-selfplay-execution-transition-v2',
            'from_sources': old, 'to_sources': new,
            'reason': 'Opt-in parent opening/CPU collection execution change; ragged FP32 numerics versioned; historical checkpoint remains unchanged'}


def _send(connection, message):
    # No pickled model/tensor objects cross this boundary; send immutable wire bytes.
    connection.send_bytes(E.wire(message).encode())


def _receive(connection):
    return json.loads(connection.recv_bytes())


def _without_labels(value):
    if isinstance(value, dict):
        return {k: _without_labels(v) for k, v in value.items() if k != 'reward'}
    if isinstance(value, (list, tuple)):
        return [_without_labels(v) for v in value]
    return value


def _check_openings(openings, observations):
    if len(openings) != 2 or len(observations) != 2:
        raise ValueError('two seat-bound opening paths required')
    streams = []
    for seat, (row, observation) in enumerate(zip(openings, observations)):
        if (row['seat'] != seat or observation['step'] != 0
                or row['actor'] != T.actor_observation(State(observation))):
            raise ValueError('worker initial actor observation differs from sampled observation')
        actions = row['trace']['actions']
        if (not actions or actions[-1] != 0 or 0 in actions[:-1]
                or any(type(a) is not int for a in actions)):
            raise ValueError('unaltered STOP-terminated opening required')
        if (not math.isfinite(row['old_logp'])
                or row['old_logp'] != row['trace']['raw_path_logp'] or 'reward' in row):
            raise ValueError('unlabelled sampled raw likelihood required')
        streams.append(row['sampler_seed'])
    if (len(set(streams)) != 2
            or any(type(s) is not int or not 0 <= s < 2**63 for s in streams)):
        raise ValueError('two independent sampler seeds required')


def sample_openings(left, right, observations, configuration, *, sampler_seeds,
                    identity, temperature, seed):
    """Parent-only decode using precisely the worker's seed/seat observations."""
    policies = [Opening(model, identity=identity, mode='sample_raw', seed=stream,
                        temperature=temperature)
                for model, stream in zip((left, right), sampler_seeds)]
    agents = [E.PolicyAgent(policy, game_id=f'selfplay-{seed}-{seat}',
                            labor_factor=4., belief=copy.deepcopy(T.SETTINGS['belief']))
              for seat, policy in enumerate(policies)]
    try:
        rows = []
        for seat, (agent, policy) in enumerate(zip(agents, policies)):
            agent(copy.deepcopy(observations[seat]), copy.deepcopy(configuration))
            if any(r.get('refusals') for r in agent.records):
                raise ValueError('opening publication refusal; do not retry or fit it')
            trace = copy.deepcopy(policy.opening_trace)
            rows.append({'seat': seat, 'sampler_seed': sampler_seeds[seat],
                         'actor': copy.deepcopy(policy.actor), 'trace': trace,
                         'old_logp': trace['raw_path_logp']})
        _check_openings(rows, observations)
        return rows
    except BaseException as exc:
        # Preserve successful seat-0 sampling even if seat 1 fails; never label it.
        exc.selfplay_diagnostic = {'seed': seed, 'sampler_seeds': list(sampler_seeds),
            'identity': identity, 'temperature': temperature,
            'policies': [{'seat': seat, 'actor': getattr(policy, 'actor', None),
                          'trace': getattr(policy, 'opening_trace', None),
                          'publications': agents[seat].records}
                         for seat, policy in enumerate(policies)]}
        raise



class _OpeningPending(Exception):
    """Internal suspension before PolicyAgent validates/publishes an opening."""


class _BatchedOpening(Opening):
    """Capture the real actor call, then publish only that call's sampled path.

    A fresh agent is used for publication: the capture agent is discarded before
    any compiler/executor action is emitted, so its partially observed controller
    state cannot affect the real opening. Neither phase starts an environment.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs, self.result = None, None

    def __call__(self, state, plan, masks):
        if state.step != 0:
            raise ValueError('batched sampling accepts actual opening actors only')
        actor = T.actor_observation(state)
        prefix = A._prefix(plan)
        options = A._projection_options(copy.deepcopy(masks.kw))
        if self.result is None:
            if self.inputs is not None:
                raise ValueError('an opening actor must be captured exactly once')
            self.actor = copy.deepcopy(actor)
            self.inputs = (BC.actor_state(self.actor), prefix, options)
            raise _OpeningPending()
        if (actor != self.actor or prefix.payload() != self.inputs[1].payload()
                or options != self.inputs[2]):
            raise ValueError('publication actor/prefix/masks differ from sampled inputs')
        self.last_trace = copy.deepcopy(self.result.trace)
        self.opening_trace = copy.deepcopy(self.result.trace)
        return self.result.actions


@torch.inference_mode()
def _decode_opening_batch(scorer, states, prefixes, *, generators,
                          projection_kwargs, temperature=1., diagnostics=None):
    """One score_batch per live prefix wave; reuse the native RAW chooser.

    The epoch envelope mirrors A._epoch, using its menu/choice/digest primitives.
    Focused scalar replay tests guard this envelope; scorer.py is unchanged.
    Finished rows leave the live wave without consuming another RNG draw. Even
    a single live row and forced administrative STOP use score_batch, not score.
    """
    from farm_neat.policy import v0_menu_fast
    if (not 0 < len(states) <= 64 or not len(states) == len(prefixes) == len(generators)
            == len(projection_kwargs) or any(not isinstance(g, torch.Generator) for g in generators)
            or len({id(g) for g in generators}) != len(generators)):
        raise ValueError('aligned bounded rows with independent generators required')
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('finite positive opening temperature required')
    if scorer.training:
        raise ValueError('opening cohort scorer must remain in eval mode')
    prefixes = [A._prefix(prefix) for prefix in prefixes]
    options = [A._projection_options(value) for value in projection_kwargs]
    identity = scorer.get_extra_state()
    feature_schema = identity.get('feature_schema_sha256')
    if feature_schema == A.FEATURE_SCHEMA_SHA256:
        # Frozen V2 scorers expose only the one-argument digest API.
        state_digest = A._state_digest
    elif (getattr(A, 'FEATURE_SCHEMA_SHA256_V3', None) is not None
          and feature_schema == A.FEATURE_SCHEMA_SHA256_V3):
        state_digest = lambda state: A._state_digest(state, 'v3')
    else:
        raise ValueError('unsupported opening feature schema')
    headers = [{'schema': A.TRACE_SCHEMA, 'model_identity': copy.deepcopy(identity),
        'state_sha256': state_digest(state),
        'projection_sha256': A._digest(option), 'initial_prefix': prefix.payload(),
        'mode': 'sample_raw', 'temperature': float(temperature)}
        for state, prefix, option in zip(states, prefixes, options)]
    choices = [[] for _ in states]
    live = list(range(len(states)))
    if diagnostics is not None:
        diagnostics.extend([{**copy.deepcopy(header), 'steps': [], 'actions': []} for header in headers])
    while live:
        menus = [v0_menu_fast.build_menu(states[i], prefixes[i], projection_kwargs=options[i], backend='fast')
                 for i in live]
        batch = scorer.score_batch([states[i] for i in live], [prefixes[i] for i in live],
            [menu.legal_ids for menu in menus], [menu.phi for menu in menus],
            memories=[{'hold_releases': options[i]['hold_releases']} for i in live])
        next_live = []
        for row, (i, menu) in enumerate(zip(live, menus)):
            n = len(menu.legal_ids)
            # Reject misaligned padded outputs; never treat padding as a raw ID.
            if (batch.logits.shape[0] != len(live) or batch.valid.shape != batch.logits.shape
                    or batch.legal_ids.shape != batch.logits.shape
                    or not bool(batch.valid[row, :n].all()) or bool(batch.valid[row, n:].any())
                    or batch.legal_ids[row, :n].tolist() != list(menu.legal_ids)):
                raise ValueError('opening score_batch changed the actual legal support')
            choice = A.choose(menu, batch.logits[row, :n], generator=generators[i],
                              mode='sample_raw', temperature=temperature)
            choices[i].append(choice)
            prefixes[i] = choice.prefix
            if diagnostics is not None:
                diagnostics[i]['steps'].append(copy.deepcopy(choice.record))
                diagnostics[i]['actions'].append(choice.action)
            if not choice.prefix.terminated:
                next_live.append(i)
        live = next_live
    results = []
    for header, prefix, row in zip(headers, prefixes, choices):
        raw = torch.stack([choice.raw_logp.double() for choice in row]).sum()
        classes = torch.stack([choice.class_logp.double() for choice in row]).sum()
        actions = tuple(choice.action for choice in row)
        trace = {**header, 'steps': [choice.record for choice in row], 'actions': list(actions),
            'final_plan': prefix.plan.canonical(), 'raw_path_logp': float(raw),
            'class_path_logp': float(classes), 'class_path_not_final_plan_marginal': True}
        results.append(A.EpochResult(prefix.plan.copy(), actions, raw, classes, trace))
    return results


def sample_openings_batch(scorer, requests, *, identity, temperature, max_live_rows=8):
    """Parent-only bounded openings, keyed by registered game index (not arrival).

    Requests contain index, seed, sampler_seeds, actual observations and
    configuration. One resident scorer serves every seat, independently seeded.
    This is the native raw-action law, never search-driver class sampling.
    """
    settings = opening_batch_settings(max_live_rows, 0.)
    if not requests or 2 * len(requests) > settings['max_live_rows']:
        raise ValueError('ready opening batch exceeds its registered live-row bound')
    indices = [request['index'] for request in requests]
    seeds = [request['seed'] for request in requests]
    streams = [stream for request in requests for stream in request['sampler_seeds']]
    if (len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 for i in indices)
            or len(set(seeds)) != len(seeds) or len(set(streams)) != len(streams)
            or any(len(r['observations']) != 2 or len(r['sampler_seeds']) != 2 for r in requests)):
        raise ValueError('distinct indexed openings and two independent seat streams required')
    policies, agents, partial = [], [], []
    def agent_for(policy, request, seat):
        return E.PolicyAgent(policy, game_id=f"selfplay-{request['seed']}-{seat}",
            labor_factor=4., belief=copy.deepcopy(T.SETTINGS['belief']))
    try:
        for request in requests:
            for seat, stream in enumerate(request['sampler_seeds']):
                policy = _BatchedOpening(scorer, identity=identity, mode='sample_raw',
                                         seed=stream, temperature=temperature)
                policies.append(policy)
                agent = agent_for(policy, request, seat)
                agents.append(agent)
                try:
                    agent(copy.deepcopy(request['observations'][seat]), copy.deepcopy(request['configuration']))
                except _OpeningPending:
                    pass
                else:
                    raise ValueError('PolicyAgent did not expose its actual opening inputs')
        results = _decode_opening_batch(scorer, [p.inputs[0] for p in policies],
            [p.inputs[1] for p in policies], generators=[p.generator for p in policies],
            projection_kwargs=[p.inputs[2] for p in policies], temperature=temperature, diagnostics=partial)
        # Retain all sampled paths even when publication later refuses any seat.
        for policy, result in zip(policies, results):
            policy.result = result
            policy.opening_trace = copy.deepcopy(result.trace)
        outputs = {}
        for ordinal, request in enumerate(requests):
            rows = []
            for seat in (0, 1):
                i = ordinal * 2 + seat
                policy = policies[i]
                agents[i] = agent_for(policy, request, seat)
                agents[i](copy.deepcopy(request['observations'][seat]), copy.deepcopy(request['configuration']))
                if any(record.get('refusals') for record in agents[i].records):
                    raise ValueError('opening publication refusal; do not retry or fit it')
                if (len(agents[i].records) != 1
                        or agents[i].records[0].get('after_plan') != policy.result.plan.canonical()):
                    raise ValueError('decoder/agent successor Plan disagreement')
                trace = copy.deepcopy(policy.opening_trace)
                rows.append({'seat': seat, 'sampler_seed': request['sampler_seeds'][seat],
                    'actor': copy.deepcopy(policy.actor), 'trace': trace, 'old_logp': trace['raw_path_logp']})
            _check_openings(rows, request['observations'])
            outputs[request['index']] = rows
        return outputs
    except BaseException as exc:
        exc.selfplay_diagnostic = {'schema': OPENING_BATCH_SCHEMA, 'settings': settings,
            'requests': copy.deepcopy(requests), 'identity': identity, 'temperature': temperature,
            'policies': [{'index': requests[i // 2]['index'], 'seat': i % 2,
                'actor': getattr(policy, 'actor', None), 'trace': getattr(policy, 'opening_trace', None),
                'partial_trace': partial[i] if i < len(partial) else None,
                'publications': agents[i].records if i < len(agents) else []}
                for i, policy in enumerate(policies)]}
        raise

def _play_openings(env, openings, *, seed, timeout, started, diagnostic):
    """Run immutable openings in an already-created world; never sample or restart."""
    observations = [env.agent_observation(seat) for seat in (0, 1)]
    _check_openings(openings, observations)  # Before the first physical step.
    agents = [E.PolicyAgent(E.TrajectoryPolicy({0: row['trace']['actions'][:-1]}),
        game_id=f'selfplay-{seed}-{seat}', labor_factor=4.,
        belief=copy.deepcopy(T.SETTINGS['belief'])) for seat, row in enumerate(openings)]
    actions_hash = hashlib.sha256()
    try:
        while not env.done:
            if time.monotonic() - started > timeout:
                raise TimeoutError(f'self-play game {seed} exceeded its registered timeout')
            joint = [agent(copy.deepcopy(env.agent_observation(seat)), copy.deepcopy(env.configuration))
                     for seat, agent in enumerate(agents)]
            actions_hash.update(E.wire(joint).encode())
            env.step(joint)
        if env.step_index != 719:
            raise ValueError('incomplete game cannot produce terminal learning labels')
        if any(r.get('refusals') for agent in agents for r in agent.records):
            raise ValueError('publication refusal; preserve failed game, do not fit it')
        cash = [float(env.agent_observation(0)['farms'][seat]['money']) for seat in (0, 1)]
        reward = (cash[0] > cash[1]) - (cash[0] < cash[1])
        return {'seed': seed, 'steps': env.step_index, 'physical_games': 1, 'cash': cash,
                'episodes': [{**copy.deepcopy(row), 'reward': reward if seat == 0 else -reward}
                             for seat, row in enumerate(openings)],
                'joint_actions_sha256': actions_hash.hexdigest(),
                'seconds': time.monotonic() - started}
    except BaseException:
        diagnostic.update({'steps': env.step_index, 'joint_actions_sha256': actions_hash.hexdigest(),
            'policies': [{**copy.deepcopy(row), 'publications': agents[seat].records,
                          'last_observation': env.agent_observation(seat)}
                         for seat, row in enumerate(openings)]})
        raise


def _rollout_worker(connection):
    """Persistent spawn worker. CPU engine only; the parent supplies all samples."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    diagnostic, index = {}, None
    try:
        _send(connection, {'kind': 'ready'})
        while True:
            request = _receive(connection)
            if request['kind'] == 'stop':
                return
            if request['kind'] != 'start':
                raise ValueError('expected one explicit native start')
            index, seed, timeout = request['index'], request['seed'], request['timeout']
            diagnostic = {'seed': seed, 'steps': 0}
            started = time.monotonic()
            env = ExactKaggricultureEnvironment(seed=seed)  # Exactly once per START intent.
            _send(connection, {'kind': 'observation', 'index': index,
                'observations': [env.agent_observation(seat) for seat in (0, 1)],
                'configuration': env.configuration})
            if not connection.poll(timeout):
                raise TimeoutError('parent opening response timeout')
            response = _receive(connection)
            if response.get('kind') != 'openings' or response.get('index') != index:
                raise ValueError('opening response index mismatch')
            diagnostic['openings'] = response['openings']
            game = _play_openings(env, response['openings'], seed=seed, timeout=timeout,
                                  started=started, diagnostic=diagnostic)
            _send(connection, {'kind': 'result', 'index': index, 'game': game})
            del env, game
    except BaseException as exc:
        try:
            _send(connection, {'kind': 'error', 'index': index, 'error': repr(exc),
                                'detail': _without_labels(diagnostic)})
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


def _shutdown_workers(lanes, *, timeout, abort):
    """One shared cleanup deadline, not N per-worker timeouts; no feeder threads."""
    deadline = time.monotonic() + timeout
    processes = [lane['process'] for lane in lanes]
    if not abort:
        for lane in lanes:
            try:
                _send(lane['connection'], {'kind': 'stop'})
            except (BrokenPipeError, EOFError, OSError):
                pass
    else:
        for process in processes:
            if process.is_alive():
                process.terminate()
    gentle = time.monotonic() + timeout * .5
    for process in processes:
        process.join(max(0., gentle - time.monotonic()))
    forced = []
    for process in processes:
        if process.is_alive():
            forced.append(process.pid)
            process.kill()
    for process in processes:
        process.join(max(0., deadline - time.monotonic()))
    alive = [p.pid for p in processes if p.is_alive()]
    receipt = {'alive_pids': alive, 'killed_pids': forced,
               'exitcodes': {str(p.pid): p.exitcode for p in processes}, 'timeout': timeout}
    for lane in lanes:
        lane['connection'].close()
    for process in processes:
        if not process.is_alive():
            process.close()
    return receipt


def _validate_collected_game(game, seed, openings):
    if game['seed'] != seed or game['steps'] != 719 or game['physical_games'] != 1:
        raise ValueError('worker returned wrong/incomplete physical game')
    if len(game['cash']) != 2 or not all(math.isfinite(c) for c in game['cash']):
        raise ValueError('finite terminal cash required')
    reward = (game['cash'][0] > game['cash'][1]) - (game['cash'][0] < game['cash'][1])
    expected = [{**row, 'reward': reward if seat == 0 else -reward}
                for seat, row in enumerate(openings)]
    if game['episodes'] != expected:
        raise ValueError('worker altered sampled path, actor, likelihood or WDL')
    digest = game['joint_actions_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('full joint-action digest required')


def _collect_processes(seeds, opening_callback, *, workers, timeout, startup_timeout=60.,
                       cleanup_timeout=5., directory=None, before_start=None, opening_batch=None,
                       opening_batch_callback=None, opening_metrics=None, before_commit=None):
    """Bounded spawn pool with a parent sampling handshake and cohort-wide commit.

    callback(index, observations, configuration) runs only in the parent. START
    intents are exclusive, never retried, and failed cohorts return no episodes.
    A worker owns at most one native environment, and reuses only its process.
    """
    import multiprocessing
    from multiprocessing.connection import wait
    settings = collection_settings(workers, startup_timeout=startup_timeout,
                                   cleanup_timeout=cleanup_timeout, opening_batch=opening_batch)
    collection_options({'collection': settings}, allocation=True)
    opening_batch = settings.get('opening_batch')
    if (opening_batch is None) != (opening_batch_callback is None):
        raise ValueError('registered opening batch and callback must be supplied together')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('nonempty distinct world seeds required')
    if not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError('bounded per-game timeout required')
    if directory is not None:
        directory = Path(directory)
        RL._write_json_new(directory / 'COLLECTION-START.json', {'seeds': list(seeds), 'settings': settings})
    started = time.monotonic()
    lanes, games, openings, completion, starts = [], [None] * len(seeds), {}, [], []
    ctx = multiprocessing.get_context('spawn')  # Never fork a CUDA-owning parent.
    error, failure_detail, next_index = None, None, 0
    initial_observations = {}
    pending, opening_batches = [], []
    cleanup = {'alive_pids': []}
    try:
        for ordinal in range(min(workers, len(seeds))):
            parent, child = ctx.Pipe(duplex=True)
            process = ctx.Process(target=_rollout_worker, args=(child,),
                                  name=f'v0-rollout-{ordinal}', daemon=True)
            try:
                process.start()
            except BaseException:
                parent.close(); child.close()
                raise
            child.close()
            lanes.append({'process': process, 'connection': parent, 'phase': 'boot',
                          'index': None, 'deadline': time.monotonic() + startup_timeout})

        def dispatch(lane):
            nonlocal next_index
            if next_index == len(seeds):
                lane['phase'], lane['deadline'] = 'idle', math.inf
                return
            index = next_index
            metadata = before_start(index, seeds[index]) if before_start is not None else None
            if directory is not None:
                RL._write_json_new(directory / f'START-{index:04d}.json',
                    {**(metadata or {}), 'seed': seeds[index], 'index': index,
                     'worker_pid': lane['process'].pid})
            starts.append(index)
            next_index += 1
            lane.update(index=index, phase='observation', deadline=time.monotonic() + timeout)
            _send(lane['connection'], {'kind': 'start', 'index': index, 'seed': seeds[index], 'timeout': timeout})

        while len(completion) != len(seeds):
            now = time.monotonic()
            if any(lane['deadline'] <= now for lane in lanes):
                expired = next(lane for lane in lanes if lane['deadline'] <= now)
                raise TimeoutError(f"worker {expired['phase']} timeout; game index {expired['index']}")
            handles = [h for lane in lanes for h in (lane['connection'], lane['process'].sentinel)]
            pause = min(.05, min(lane['deadline'] - now for lane in lanes))
            if pending:
                pause = min(pause, max(0., pending[0]['queued_at'] + opening_batch['max_wait_seconds'] - now))
                if len(pending) >= opening_batch['max_live_rows'] // 2:
                    pause = 0.
            ready = wait(handles, timeout=pause)
            for lane in lanes:
                connection, process = lane['connection'], lane['process']
                if connection in ready:
                    try:
                        message = _receive(connection)
                    except (EOFError, OSError) as exc:
                        raise RuntimeError(f'worker {process.pid} closed its result pipe') from exc
                    kind, index = message['kind'], lane['index']
                    if kind == 'error':
                        failure_detail = message
                        raise RuntimeError(f"worker failed: {message['error']}")
                    if lane['phase'] == 'boot' and kind == 'ready':
                        dispatch(lane)
                    elif (lane['phase'] == 'observation' and kind == 'observation'
                          and message.get('index') == index):
                        initial_observations[index] = copy.deepcopy(message['observations'])
                        if opening_batch is not None:
                            lane['phase'] = 'opening'
                            pending.append({'lane': lane, 'queued_at': time.monotonic(),
                                'request': {'index': index, 'observations': copy.deepcopy(message['observations']),
                                            'configuration': copy.deepcopy(message['configuration'])}})
                            continue
                        rows = opening_callback(index, message['observations'], message['configuration'])
                        # Round-trip freezes caller-owned mutable dictionaries, retaining exact floats.
                        openings[index] = json.loads(E.wire(rows))
                        if directory is not None:
                            RL._write_json_new(directory / f'OPENING-{index:04d}.json',
                                {'seed': seeds[index], 'openings': openings[index]})
                        if time.monotonic() >= lane['deadline']:
                            raise TimeoutError(f'parent sampling timeout; game index {index}')
                        _send(connection, {'kind': 'openings', 'index': index, 'openings': openings[index]})
                        lane['phase'] = 'play'
                    elif lane['phase'] == 'play' and kind == 'result' and message.get('index') == index:
                        game = message['game']
                        _validate_collected_game(game, seeds[index], openings[index])
                        games[index] = game
                        completion.append(index)
                        dispatch(lane)
                    else:
                        raise ValueError(f"unexpected worker message {kind} in {lane['phase']}")
                elif process.sentinel in ready:
                    raise RuntimeError(f'worker {process.pid} exited with code {process.exitcode}')
            if pending:
                # Do not deliberately wait for a full batch after the cap/deadline
                # or when every remaining lane is already playing/idle/pending.
                capacity = opening_batch['max_live_rows'] // 2
                now = time.monotonic()
                full = len(pending) >= capacity
                due = now >= pending[0]['queued_at'] + opening_batch['max_wait_seconds']
                tail = not any(lane['phase'] in ('boot', 'observation') for lane in lanes)
                if full or due or tail:
                    for lane in lanes:
                        if lane['process'].exitcode is not None:
                            raise RuntimeError(f"worker {lane['process'].pid} exited with code {lane['process'].exitcode}")
                        if now >= lane['deadline']:
                            raise TimeoutError(f"worker {lane['phase']} timeout; game index {lane['index']}")
                    group, pending = pending[:capacity], pending[capacity:]
                    requests = [entry['request'] for entry in group]
                    indices = [request['index'] for request in requests]
                    record = {'indices': indices, 'rows': 2 * len(group),
                        'wait_seconds': [now - entry['queued_at'] for entry in group],
                        'trigger': 'FULL' if full else 'WAIT' if due else 'TAIL'}
                    opening_batches.append(record)
                    sampled = opening_batch_callback(requests)
                    record['sampling_seconds'] = time.monotonic() - now
                    if not isinstance(sampled, dict) or set(sampled) != set(indices):
                        raise ValueError('opening batch callback must return exactly its registered indices')
                    for index in indices:
                        openings[index] = json.loads(E.wire(sampled[index]))
                        if directory is not None:
                            RL._write_json_new(directory / f'OPENING-{index:04d}.json',
                                {'seed': seeds[index], 'openings': openings[index],
                                 'opening_batch': opening_batch})
                    # A synchronous model call cannot be preempted by this loop.
                    # It must finish within every live game/boot deadline before
                    # any sampled response is released; outer SLURM guards hard hangs.
                    now = time.monotonic()
                    for lane in lanes:
                        if now >= lane['deadline']:
                            raise TimeoutError(f"parent sampling timeout; game index {lane['index']}")
                        if lane['process'].exitcode is not None:
                            raise RuntimeError(f"worker {lane['process'].pid} exited with code {lane['process'].exitcode}")
                    for entry in group:
                        lane = entry['lane']
                        index = entry['request']['index']
                        _send(lane['connection'], {'kind': 'openings', 'index': index, 'openings': openings[index]})
                        lane['phase'] = 'play'
        if before_commit is not None:
            before_commit()
    except BaseException as exc:
        error = exc
        failure_detail = getattr(exc, 'selfplay_diagnostic', failure_detail)
    finally:
        worker_pids = [lane['process'].pid for lane in lanes]
        cleanup = _shutdown_workers(lanes, timeout=cleanup_timeout, abort=error is not None)
    if cleanup['alive_pids']:
        error = RuntimeError(f"worker cleanup incomplete: {cleanup['alive_pids']}; original={error!r}")
    receipt = {'schema': COLLECTION_SCHEMA, 'settings': settings, 'worker_pids': worker_pids,
               'started_indices': starts, 'completion_order': completion, 'cleanup': cleanup,
               'seconds': time.monotonic() - started}
    if opening_batch is not None:
        receipt['opening_batches'] = opening_batches
    if opening_metrics is not None:
        receipt['opening_sampling'] = copy.deepcopy(opening_metrics)
    if error is not None:
        if directory is not None:
            RL._write_json_new(directory / 'FAILED-COHORT.json', {**receipt,
                'status': 'FAILED_NOT_FOR_TRAINING', 'error': repr(error), 'optimizer_updates': 0,
                'openings': _without_labels(openings), 'worker_failure': _without_labels(failure_detail),
                'initial_observations': initial_observations,
                'completed_diagnostics': _without_labels([g for g in games if g is not None])})
        raise error
    if directory is not None:
        RL._write_json_new(directory / 'COLLECTION.json', {**receipt, 'status': 'COMPLETE',
                                                          'physical_games': len(games)})
    return games



def _opening_metrics(batched):
    return {'schema': OPENING_BATCH_SCHEMA, 'numerics': OPENING_BATCH_NUMERICS if batched
            else 'FP32_SCALAR_RAW_CATEGORICAL_V1', 'cross_device_bitwise_parity': False,
            'model_calls': 0, 'scalar_model_calls': 0, 'score_batch_model_calls': 0,
            'failed_model_calls': 0, 'batch_widths': [], 'scoring_seconds_synchronized': 0.,
            'torch_version': str(torch.__version__)}


class _ScoringMeter:
    """Non-owning parent-local view; no model copies or worker serialization.

    Time includes host score input construction and a device synchronization at
    both boundaries on CUDA. Scalar model.score's internal score_batch is counted
    once, not twice. Successful and failed attempted calls are retained.
    """
    def __init__(self, scorer, metrics):
        self.scorer, self.metrics = scorer, metrics
        self.device = next(scorer.parameters(), torch.empty(0)).device
        metrics.setdefault('devices', [])
        if str(self.device) not in metrics['devices']:
            metrics['devices'].append(str(self.device))

    def __getattr__(self, name):
        return getattr(self.scorer, name)

    def eval(self):
        self.scorer.eval()
        return self

    def _call(self, function, width, scalar, *args, **kwargs):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.monotonic()
        self.metrics['model_calls'] += 1
        self.metrics['scalar_model_calls' if scalar else 'score_batch_model_calls'] += 1
        self.metrics['batch_widths'].append(width)
        try:
            return function(*args, **kwargs)
        except BaseException:
            self.metrics['failed_model_calls'] += 1
            raise
        finally:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.metrics['scoring_seconds_synchronized'] += time.monotonic() - started

    def score(self, *args, **kwargs):
        return self._call(self.scorer.score, 1, True, *args, **kwargs)

    def score_batch(self, states, *args, **kwargs):
        return self._call(self.scorer.score_batch, len(states), False, states, *args, **kwargs)


def _policy_stamp(model):
    # No expensive tensor copies/hashes or GPU synchronization. Normal optimizer
    # and in-place writes/rebindings change this stamp; illicit .data writes are
    # outside this single-owner collector's contract, not an adversarial boundary.
    return (model.training, E.wire(model.get_extra_state()),
            tuple((name, id(value), value._version)
                  for name, value in list(model.named_parameters()) + list(model.named_buffers())))


def collect_cohort(left, right, *, seeds, sampler_pairs, identity, temperature, timeout,
                   collection=None, directory=None):
    """Exactly one frozen cohort; serial remains the default and original collector."""
    settings = collection_options({} if collection is None else {'collection': collection}, allocation=True)
    batch = settings.get('opening_batch')
    streams = [s for pair in sampler_pairs for s in pair]
    if (len(seeds) != len(sampler_pairs) or len(set(seeds)) != len(seeds)
            or any(len(pair) != 2 for pair in sampler_pairs) or len(set(streams)) != len(streams)):
        raise ValueError('distinct cohort worlds and private sampler streams required')
    if batch is not None and left is not right:
        raise ValueError('opening batches require one resident frozen mirror scorer')
    if batch is not None and left.training:
        raise ValueError('opening batch policy must be frozen in eval mode')
    stamp = _policy_stamp(left) if batch is not None else None
    def frozen():
        if batch is not None and _policy_stamp(left) != stamp:
            raise ValueError('opening batch policy changed during its frozen cohort')
    if directory is not None:
        directory = Path(directory)
    metrics = _opening_metrics(batch is not None)
    measured_left = _ScoringMeter(left, metrics)
    measured_right = measured_left if right is left else _ScoringMeter(right, metrics)
    started = time.monotonic()
    if settings['workers'] == 1 and batch is None:
        if directory is not None:
            RL._write_json_new(directory / 'COLLECTION-START.json', {'seeds': list(seeds), 'settings': settings})
        games = []
        try:
            for index, (seed, pair) in enumerate(zip(seeds, sampler_pairs)):
                if directory is not None:
                    RL._write_json_new(directory / f'START-{index:04d}.json', {'seed': seed, 'samplers': pair})
                games.append(collect_game(measured_left, measured_right, seed=seed,
                    sampler_seeds=pair, identity=identity, temperature=temperature, timeout=timeout,
                    failure_path=None if directory is None else directory / f'FAILED-GAME-{index:04d}.json'))
        except BaseException as exc:
            if directory is not None:
                RL._write_json_new(directory / 'FAILED-COHORT.json', {
                    'status': 'FAILED_NOT_FOR_TRAINING', 'error': repr(exc), 'optimizer_updates': 0,
                    'completed_diagnostics': _without_labels(games), 'opening_sampling': metrics})
            raise
        if directory is not None:
            RL._write_json_new(directory / 'COLLECTION.json', {'schema': COLLECTION_SCHEMA,
                'status': 'COMPLETE', 'physical_games': len(games), 'settings': settings,
                'seconds': time.monotonic() - started, 'opening_sampling': metrics})
    else:
        def sample(index, observations, configuration):
            return sample_openings(measured_left, measured_right, observations, configuration,
                sampler_seeds=sampler_pairs[index], identity=identity, temperature=temperature, seed=seeds[index])
        def sample_batch(requests):
            frozen()
            requests = [{**request, 'seed': seeds[request['index']],
                         'sampler_seeds': sampler_pairs[request['index']]} for request in requests]
            rows = sample_openings_batch(measured_left, requests, identity=identity,
                temperature=temperature, max_live_rows=batch['max_live_rows'])
            try:
                frozen()
            except BaseException as exc:
                exc.selfplay_diagnostic = {'openings': rows, 'requests': requests}
                raise
            return rows
        games = _collect_processes(seeds, sample, workers=settings['workers'], timeout=timeout,
            startup_timeout=settings['startup_timeout'], cleanup_timeout=settings['cleanup_timeout'], directory=directory,
            before_start=lambda index, seed: {'samplers': sampler_pairs[index],
                                               'identity': identity, 'temperature': temperature},
            opening_batch=batch, opening_batch_callback=sample_batch if batch is not None else None,
            opening_metrics=metrics, before_commit=frozen if batch is not None else None)
    if directory is not None:
        for index, game in enumerate(games):
            RL._write_json_new(directory / f'GAME-{index:04d}.json', game)
    return games


def replay_saved_games(saved_games, *, workers=1, timeout=300., directory=None, before_start=None):
    """CPU diagnostic ONLY: replay supplied raw paths without any neural sampling."""
    seeds = [game['seed'] for game in saved_games]
    rows = [[{k: copy.deepcopy(row[k]) for k in ('seat', 'sampler_seed', 'actor', 'trace', 'old_logp')}
             for row in game['episodes']] for game in saved_games]
    def fixed(index, observations, configuration):
        _check_openings(rows[index], observations)
        return rows[index]
    if workers == 1:
        if directory is not None:
            RL._write_json_new(directory / 'COLLECTION-START.json', {'seeds': seeds, 'workers': 1, 'diagnostic': True})
        started = time.monotonic()
        games = []
        for index, seed in enumerate(seeds):
            if before_start is not None:
                before_start(index, seed)
            if directory is not None:
                RL._write_json_new(directory / f'START-{index:04d}.json', {'seed': seed, 'diagnostic': True})
            t0 = time.monotonic()
            env = ExactKaggricultureEnvironment(seed=seed)
            games.append(_play_openings(env, rows[index], seed=seed, timeout=timeout,
                                        started=t0, diagnostic={}))
        if directory is not None:
            RL._write_json_new(directory / 'COLLECTION.json', {'status': 'COMPLETE', 'workers': 1,
                'physical_games': len(games), 'seconds': time.monotonic() - started})
    else:
        games = _collect_processes(seeds, fixed, workers=workers, timeout=timeout,
                                  directory=directory, before_start=before_start)
    for reference, actual in zip(saved_games, games):
        for key in ('seed', 'steps', 'physical_games', 'cash', 'joint_actions_sha256', 'episodes'):
            if reference[key] != actual[key]:
                raise ValueError(f'saved replay parity mismatch: seed {actual["seed"]}, {key}')
    if directory is not None:
        for index, game in enumerate(games):
            RL._write_json_new(directory / f'GAME-{index:04d}.json', game)
    return games


def replay(model, row):
    result = A.replay_epoch(model, BC.actor_state(copy.deepcopy(row['actor'])), Plan(),
        copy.deepcopy(row['trace']), projection_kwargs=copy.deepcopy(T.SETTINGS),
        verify_logps=True, backend='fast')
    if not math.isclose(float(result.raw_logp.detach()), row['old_logp'], rel_tol=2e-6, abs_tol=3e-5):
        raise ValueError('behavior likelihood mismatch')
    return result.raw_logp


def update(model, optimizer, episodes):
    if not episodes or any(row['reward'] not in (-1, 0, 1) for row in episodes):
        raise ValueError('complete terminal WDL episodes required')
    model.eval()  # Same dropout/numerics as collection, with gradients enabled.
    optimizer.zero_grad(set_to_none=True)
    # Authenticate every path before even considering a zero-signal return.
    with torch.no_grad():
        for row in episodes:
            replay(model, row)
    if not any(row['reward'] for row in episodes):
        return {'status': 'NO_WDL_SIGNAL', 'optimizer_updates': 0}
    loss_sum = 0.
    for row in episodes:
        loss = -row['reward'] * replay(model, row) / len(episodes)
        loss.backward()
        loss_sum += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL['max_grad_norm'],
                                        error_if_nonfinite=True, foreach=False)
    if float(norm) <= 1e-12:
        return {'status': 'ZERO_GRADIENT', 'optimizer_updates': 0, 'grad_norm': float(norm)}
    optimizer.step()
    if not RL._finite_tree(model.state_dict()) or not RL._finite_tree(optimizer.state_dict()):
        raise ValueError('nonfinite update; resume only a committed checkpoint')
    return {'status': 'UPDATED', 'optimizer_updates': 1, 'loss': loss_sum,
            'grad_norm': float(norm), 'episodes': len(episodes)}


def save_checkpoint(directory, model, optimizer, *, run_identity, round_index):
    directory.mkdir(exist_ok=False)
    state = RL._cpu_tree({'schema': SCHEMA, 'identity': run_identity,
        'round': round_index, 'model_identity': model.get_extra_state(),
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'parameter_order': [n for n, _ in model.named_parameters()],
        'rng': RL._rng_state(next(model.parameters()).device)})
    path = directory / 'state.pt'
    with path.open('xb') as stream:
        torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
    restored = torch.load(path, map_location='cpu', weights_only=True)
    if not RL._same_tree(state, restored):
        raise ValueError('checkpoint round-trip failed')
    commit = {'schema': SCHEMA, 'status': 'COMMITTED', 'round': round_index,
              'identity': run_identity, 'state_sha256': RL.file_sha(path),
              'state_bytes': path.stat().st_size}
    RL._write_json_new(directory / 'COMMIT.json', commit)
    return commit


def load_checkpoint(directory, *, device, production=True, source_transition=None):
    commit = json.loads((directory / 'COMMIT.json').read_text())
    if commit['schema'] != SCHEMA or commit['status'] != 'COMMITTED':
        raise ValueError('committed self-play checkpoint required')
    if not checkpoint_sources_compatible(commit['identity']['sources'], transition=source_transition):
        raise ValueError('self-play source identity changed')
    if commit['identity']['runtime'] != BC.runtime_identity(torch.device(device)):
        raise ValueError('self-play runtime identity changed')
    path = directory / 'state.pt'
    if RL.file_sha(path) != commit['state_sha256'] or path.stat().st_size != commit['state_bytes']:
        raise ValueError('self-play checkpoint bytes changed')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['identity'] != commit['identity'] or state['round'] != commit['round']:
        raise ValueError('checkpoint/commit disagreement')
    BC.validate_model_manifest(state['model_identity'], production=production)
    if state['model_identity']['diagnostic'] and torch.device(device).type != 'cpu':
        raise ValueError('diagnostic continuation is CPU-only')
    model = A.V0Scorer(Config(**state['model_identity']['config']),
                      diagnostic=state['model_identity']['diagnostic']).to(device)
    model.load_state_dict(state['model'], strict=True)
    if [n for n, _ in model.named_parameters()] != state['parameter_order']:
        raise ValueError('optimizer parameter order changed')
    optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL['learning_rate'], foreach=False)
    optimizer.load_state_dict(state['optimizer'])
    RL._restore_rng(state['rng'], torch.device(device))
    return model.eval(), optimizer, commit


def run(plan, output):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('SLURM GPU allocation required')
    validate_plan(plan)
    collection = collection_options(plan, allocation=True)
    parent = ROOT / plan['parent']['path']
    parent_meta = json.loads((parent / 'COMMIT.json').read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(parent_meta['identity']['runtime']['interop_threads'])
    device = E.inference_runtime('cuda:0')
    BC.require_production_runtime()
    if RL.file_sha(parent / 'COMMIT.json') != plan['parent']['commit_sha256']:
        raise ValueError('parent commit identity changed')
    if parent_meta['schema'] == SCHEMA:
        model, optimizer, _ = load_checkpoint(parent, device=device,
            source_transition=plan.get('execution_source_transition'))
    else:
        if plan.get('execution_source_transition') is not None:
            raise ValueError('execution transition applies only to a self-play checkpoint')
        loaded = E.load_committed_scorer(parent, plan['parent']['commit_sha256'], device=device)
        model = loaded.scorer.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL['learning_rate'], foreach=False)
    frozen = copy.deepcopy(model).eval().requires_grad_(False)
    output.mkdir(parents=True, exist_ok=False)
    identity = {'plan_sha256': RL.digest(plan), 'sources': sources(),
                'runtime': BC.runtime_identity(device), 'protocol': PROTOCOL,
                'parent': plan['parent'], 'collection': collection,
                'execution_source_transition': plan.get('execution_source_transition')}
    RL._write_json_new(output / 'PLAN.json', plan)
    if plan.get('execution_source_transition') is not None:
        RL._write_json_new(output / 'SOURCE-TRANSITION.json', {
            **plan['execution_source_transition'], 'parent': plan['parent'],
            'historical_checkpoint_written': False})
    summaries = []
    started = time.monotonic()
    try:
        for ordinal, seeds in enumerate(plan['train_worlds']):
            cohort = output / f'round-{ordinal:04d}'
            cohort.mkdir()
            model.eval()
            cohort_started = time.monotonic()
            sampler_pairs = [(plan['sampler_base'] + ordinal * 10000 + index * 2,
                              plan['sampler_base'] + ordinal * 10000 + index * 2 + 1)
                             for index in range(len(seeds))]
            games = collect_cohort(model, model, seeds=seeds, sampler_pairs=sampler_pairs,
                identity={'behavior_round': ordinal, 'run': RL.digest(identity)},
                temperature=plan['temperature'], timeout=plan['game_timeout'],
                collection=collection, directory=cohort)
            collection_seconds = time.monotonic() - cohort_started
            for index, game in enumerate(games):
                print(json.dumps({'round': ordinal, 'game': index, 'cash': game['cash'],
                                  'seconds': game['seconds']}), flush=True)
            step = update(model, optimizer, [r for g in games for r in g['episodes']])
            summary = {'round': ordinal, 'physical_games': len(games), **step,
                       'collection_seconds': collection_seconds,
                       'collection_games_per_second': len(games) / collection_seconds,
                       'collection_workers': collection['workers']}
            if step['optimizer_updates']:
                saved = save_checkpoint(cohort / 'checkpoint', model, optimizer,
                                        run_identity=identity, round_index=ordinal)
                summary['checkpoint'] = {'path': str(cohort / 'checkpoint'),
                                         'sha256': saved['state_sha256']}
                verified = subprocess.run([sys.executable, '-B', '-m', 'farm_neat.training.v0_selfplay',
                    '--verify-checkpoint', str(cohort / 'checkpoint'),
                    '--commit-sha256', RL.file_sha(cohort / 'checkpoint/COMMIT.json')],
                    check=True, capture_output=True, text=True, timeout=180)
                summary['fresh_process_reload'] = json.loads(verified.stdout)
                # Reload committed tensors/moments before the next fresh cohort.
                del model, optimizer
                model, optimizer, _ = load_checkpoint(cohort / 'checkpoint', device=device)
            summaries.append(summary)
            RL._write_json_new(cohort / 'RESULT.json', summary)
            print(json.dumps(summary), flush=True)
            if not step['optimizer_updates']:
                break
        evaluations = []
        if summaries and all(s['optimizer_updates'] for s in summaries):
            for index, seed in enumerate(plan['evaluation_worlds']):
                left, right = (model, frozen) if index % 2 == 0 else (frozen, model)
                game = collect_game(left, right, seed=seed,
                    sampler_seeds=(plan['eval_sampler_base'] + index * 2, plan['eval_sampler_base'] + index * 2 + 1),
                    identity={'evaluation': True}, temperature=plan['temperature'], timeout=plan['game_timeout'],
                    failure_path=output / f'FAILED-EVAL-{index:04d}.json')
                RL._write_json_new(output / f'EVAL-{index:04d}.json', game)
                evaluations.append({'seed': seed, 'child_seat': index % 2,
                                    'reward': game['episodes'][index % 2]['reward'], 'cash': game['cash']})
        result = {'schema': SCHEMA, 'status': 'COMPLETE' if len(summaries) == len(plan['train_worlds'])
                  and all(s['optimizer_updates'] for s in summaries) else 'STOPPED_NO_SIGNAL',
                  'rounds': summaries, 'evaluation': evaluations,
                  'fresh_training_games': sum(s['physical_games'] for s in summaries),
                  'optimizer_updates': sum(s['optimizer_updates'] for s in summaries),
                  'seconds': time.monotonic() - started, 'strength_claim': False}
        RL._write_json_new(output / 'RESULT.json', result)
        return result
    except BaseException as exc:
        RL._write_json_new(output / 'FAILURE.json', {'error': repr(exc), 'completed_rounds': summaries})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Metadata-only preregistration; no checkpoint tensors or CUDA')
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--seed-base', type=int)
    parser.add_argument('--collection-workers', type=int, default=1,
                        help='Prepare only: bounded spawn workers; 1 retains original serial mode')
    parser.add_argument('--opening-batch-rows', type=int, default=0,
                        help='Prepare only: opt in to 2..64 live parent-scored seat rows (0 disables)')
    parser.add_argument('--opening-batch-wait-ms', type=float, default=0.,
                        help='Prepare only: 0..1000 ms maximum additional ready-opening coalescing wait')
    parser.add_argument('--cohorts', type=int, default=2)
    parser.add_argument('--games-per-cohort', type=int, default=4)
    parser.add_argument('--evaluation-games', type=int, default=4)
    parser.add_argument('--allow-execution-source-transition', action='store_true',
                        help='Prepare only: explicitly pin the known old self-play source transition')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--verify-checkpoint', type=Path)
    parser.add_argument('--commit-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.verify_checkpoint:
        if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('SLURM GPU allocation required')
        commit = RL._pinned_json(args.verify_checkpoint / 'COMMIT.json', args.commit_sha256)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(commit['identity']['runtime']['interop_threads'])
        device = E.inference_runtime('cuda:0')
        BC.require_production_runtime()
        model, optimizer, _ = load_checkpoint(args.verify_checkpoint, device=device)
        print(json.dumps({'status': 'RELOADED', 'state_sha256': commit['state_sha256'],
                          'parameters': sum(p.numel() for p in model.parameters()),
                          'optimizer_state_entries': len(optimizer.state)}))
        return
    if args.output is None:
        parser.error('--output is required')
    if args.prepare:
        if args.parent is None or args.seed_base is None:
            parser.error('--prepare requires --parent and --seed-base')
        if not (1 <= args.cohorts <= 16 and 2 <= args.games_per_cohort <= 32
                and 2 <= args.evaluation_games <= 32):
            parser.error('prepare requires 1..16 cohorts and 2..32 train/evaluation games')
        parent = args.parent.resolve().relative_to(ROOT)
        parent_meta = json.loads((ROOT / parent / 'COMMIT.json').read_text())
        transition = None
        if args.allow_execution_source_transition:
            if parent_meta['schema'] != SCHEMA:
                raise ValueError('execution transition applies only to a self-play checkpoint')
            transition = execution_source_transition(parent_meta['identity']['sources'])
        elif (parent_meta['schema'] == SCHEMA
              and not checkpoint_sources_compatible(parent_meta['identity']['sources'])):
            raise ValueError('old self-play source requires --allow-execution-source-transition')
        if not args.opening_batch_rows and args.opening_batch_wait_ms != 0:
            parser.error('--opening-batch-wait-ms requires --opening-batch-rows')
        batch = (opening_batch_settings(args.opening_batch_rows, args.opening_batch_wait_ms / 1000.)
                 if args.opening_batch_rows else None)
        plan = {'schema': SCHEMA, 'protocol': PROTOCOL, 'sources': sources(),
                'parent': {'path': str(parent), 'commit_sha256': RL.file_sha(ROOT / parent / 'COMMIT.json')},
                'train_worlds': [[args.seed_base + i * args.games_per_cohort + j
                                  for j in range(args.games_per_cohort)] for i in range(args.cohorts)],
                'evaluation_worlds': [args.seed_base + max(100, args.cohorts * args.games_per_cohort) + j
                                      for j in range(args.evaluation_games)],
                'sampler_base': args.seed_base + 1000000, 'eval_sampler_base': args.seed_base + 2000000,
                'temperature': 1.5, 'game_timeout': 300,
                'collection': collection_settings(args.collection_workers, opening_batch=batch),
                'purpose': 'Fresh collect/update cohorts, cold tensor/moment reload, frozen-parent evaluation; no promotion'}
        if transition is not None:
            plan['execution_source_transition'] = transition
        validate_plan(plan)
        RL._write_json_new(args.output, plan)
        print(json.dumps({'plan': str(args.output), 'sha256': RL.file_sha(args.output)}))
        return
    if (args.collection_workers != 1 or args.allow_execution_source_transition
            or args.opening_batch_rows != 0 or args.opening_batch_wait_ms != 0
            or args.cohorts != 2 or args.games_per_cohort != 4 or args.evaluation_games != 4):
        parser.error('collection/source/size options are --prepare only; execution uses the pinned plan')
    if args.plan is None or args.plan_sha256 is None:
        parser.error('execution requires --plan and --plan-sha256')
    plan = RL._pinned_json(args.plan, args.plan_sha256)
    print(json.dumps(run(plan, args.output)), flush=True)


if __name__ == '__main__':
    main()
