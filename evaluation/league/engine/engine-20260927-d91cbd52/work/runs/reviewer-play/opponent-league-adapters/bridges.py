"""Task-local bridges into unchanged Slot81 and Slot89 two-agent game loops.

Both shipped loops expose one scripted dispatch branch named SHOP_ROUTER. We bind
that branch to a fresh registered actor in a per-call backend, never alter shared
module attributes, and retain the real opponent name/identity in rows and traces.
The dispatch tag is implementation plumbing, NOT an opponent identity or fallback.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import sys
import time
import types

from league import (ROOT, HOME, SLOT81, SLOT89, DEFAULT_CONTROLS, SCRIPTED,
                    exact_import, freerun, legacy, sha_file, seal)

FREERUN_SHA256 = '00ea4ba10d903b89be4d34228bba9e7af548da949a016098722f18a4f1e7ca86'
COLLECTION_SHA256 = '6a94738d5bf8b03ee4b9d63234fc5bbb8f170ee2642ab99de349f3a6d699e7ee'


def _check_loop(path, expected):
    if sha_file(path) != expected:
        raise ValueError(f'upstream game-loop source changed: {path}; review bridge before rebinding')


def private_function(function, overlay):
    """Same code/defaults/closure; a PRIVATE globals dictionary, no monkey patch."""
    namespace = {**function.__globals__, **overlay}
    result = types.FunctionType(function.__code__, namespace, function.__name__,
                                function.__defaults__, function.__closure__)
    result.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    return result


class _Proxy:
    def __init__(self, target, **overrides):
        self._target, self._overrides = target, overrides
    def __getattr__(self, key):
        if key in self._overrides:
            return self._overrides[key]
        return getattr(self._target, key)


class _ObservedEnvironment:
    """Read-only metadata recording around the same engine's step implementation."""
    def __init__(self, environment):
        self._environment = environment
        self.shop_history = []
        self._shops = list(environment.agent_observation(0)['town']['unlocked_shops'])
    def __getattr__(self, name):
        return getattr(self._environment, name)
    def step(self, actions):
        result = self._environment.step(actions)
        obs = self._environment.agent_observation(0)
        shops = list(obs['town']['unlocked_shops'])
        if shops != self._shops:
            self.shop_history.append({'step': self.step_index, 'day': obs['day'],
                                      'unlocked_shops': shops})
            self._shops = shops
        return result


@dataclass(frozen=True)
class _Candidate:
    name: str


def bridge_identity(consumer: str) -> dict:
    return seal({'schema': 'slot98-consumer-bridge-v1', 'consumer': consumer,
                 'dispatch_tag': 'SHOP_ROUTER',
                 'meaning': 'Only the upstream scripted callsite tag; actual name and SHA remain explicit',
                 'adapter_sha256': sha_file(HOME / 'bridges.py'),
                 'upstream_sha256': FREERUN_SHA256 if consumer == 'slot81' else COLLECTION_SHA256,
                 'shared_module_mutation': False, 'economics_changed': False})


class _Trace:
    def __init__(self, delegate, case):
        self.delegate, self.case = delegate, case
    @property
    def exhausted(self):
        return self.delegate.exhausted
    def emit(self, row):
        row = copy.deepcopy(row)
        if row['type'] == 'CASE_STARTED':
            row['case'] = copy.deepcopy(self.case)
            row['bridge'] = bridge_identity('slot81')
        self.delegate.emit(row)


def play_evaluation_case(case, *, league, controls, trace,
                         on_native_start=None, environment_factory=None, mode='NATIVE'):
    """Free-running evaluator seam. Only normal own-seat observations reach actors.

    Synthetic environments are explicit software fixtures; their native counters
    are reset to zero. Production-like learned inference here remains CPU-only.
    """
    _check_loop(SLOT81 / 'freerun.py', FREERUN_SHA256)
    if mode not in ('NATIVE', 'SYNTHETIC_TERMINAL_FIXTURE'):
        raise ValueError('explicit NATIVE or synthetic fixture mode required')
    if environment_factory is not None and mode != 'SYNTHETIC_TERMINAL_FIXTURE':
        raise ValueError('injected environment must be explicitly synthetic fixture evidence')
    if mode == 'SYNTHETIC_TERMINAL_FIXTURE' and environment_factory is None:
        raise ValueError('synthetic fixture requires an injected environment')
    if controls.get('device', 'cpu') != 'cpu':
        raise ValueError('league bridge is CPU-only')
    case = copy.deepcopy(case)
    candidate_id = league.identity(case['candidate'])
    opponent_id = league.identity(case['opponent'])
    begun = time.process_time()

    class Backend(freerun.NativeBackend):
        kind = mode
        def __init__(self):
            super().__init__(root=ROOT, device='cpu')
            self.environment, self.actors = None, []
        def new_actor(self, loaded, **kwargs):
            actor = league.new_actor(loaded.name, **kwargs)
            self.actors.append(actor)
            return actor
        def new_router(self):
            actor = league.new_actor(case['opponent'], seat=1-case['seat'],
                sampler_seed=case['opponent_sampler_seed'], controls=controls,
                emit=lambda event: trace.emit(event))
            self.actors.append(actor)
            return actor
        def new_environment(self, projected, rule):
            if environment_factory is None:
                if on_native_start is not None:
                    on_native_start()
                env = super().new_environment(projected, rule)
            else:
                env = environment_factory(copy.deepcopy(case), copy.deepcopy(rule))
            self.environment = _ObservedEnvironment(env)
            return self.environment

    backend = Backend()
    projected = {**case, 'model': 'candidate', 'opponent': 'SHOP_ROUTER',
                 'opponent_identity': opponent_id}
    plan = {'spec': {'controls': copy.deepcopy(controls)}, 'models': {'candidate': candidate_id}}
    row = freerun._play_case(projected, case['rule'], plan,
                            {'candidate': _Candidate(case['candidate'])}, backend, _Trace(trace, case))
    row['case'] = case
    row['bridge'] = bridge_identity('slot81')
    row['cpu_seconds'] = time.process_time() - begun
    row['actor_cpu_seconds'] = [getattr(a, 'cpu_seconds', None) for a in backend.actors]
    row['shop_history'] = copy.deepcopy(backend.environment.shop_history) if backend.environment else []
    if mode != 'NATIVE':
        row['physical_games_started'] = row['physical_games_completed'] = 0
        row['environment_start_attempted'] = False
        row['synthetic_games_completed'] = int(row['status'] == 'COMPLETE')
    if row['status'] == 'COMPLETE' and case['rule']['kind'] == 'FIXED_SHOP':
        observed = row['shop_history'][-1]['unlocked_shops'] if row['shop_history'] else []
        if observed != case['rule']['shops']:
            row.update(status='FAILED', physical_games_completed=0,
                       error={'type':'ValueError','message':'observed shop history differs from CRN schedule'})
            for key in ('terminal_cash_by_seat','terminal_statuses','margin','wdl','match_score'):
                row.pop(key, None)
    return row


def load_collection():
    _check_loop(SLOT89 / 'fe_collection.py', COLLECTION_SHA256)
    if str(SLOT89) not in sys.path:
        sys.path.insert(0, str(SLOT89))
    return exact_import('fe_collection', SLOT89)


def play_collection_game(behavior, case, *, league, controls, opponent=None,
                         environment_factory=None, mode='NATIVE', on_native_start=None):
    """Slot89 collector bridge; sampler events/likelihoods/rewards are untouched.

    Existing SELF_PLAY/PASS/FIRST_BC cases delegate directly. For the five scripts,
    use the same play_game bytecode with a per-call backend/opponent-id binding.
    No field of the original fe_collection, fe_bindings or freerun is patched.
    """
    module = load_collection()
    if case['opponent'] not in SCRIPTED:
        return module.play_game(behavior, case, opponent=opponent, controls=controls,
            environment_factory=environment_factory, mode=mode, on_native_start=on_native_start)
    identity = league.identity(case['opponent'])
    base = module.b.freerun.NativeBackend

    class Backend(base):
        def new_router(self):
            return league.new_actor(case['opponent'], seat=1-case['seat'],
                sampler_seed=case['opponent_sampler_seed'], controls=controls)

    original_identity = module.opponent_identity
    def opponent_identity(kind, loaded=None):
        return copy.deepcopy(identity) if kind == 'SHOP_ROUTER' else original_identity(kind, loaded)
    private_b = _Proxy(module.b, freerun=_Proxy(module.b.freerun, NativeBackend=Backend))
    play = private_function(module.play_game, {'b': private_b, 'opponent_identity': opponent_identity})
    projected = {**copy.deepcopy(case), 'opponent': 'SHOP_ROUTER'}
    row = play(behavior, projected, opponent=None, controls=controls,
        environment_factory=environment_factory, mode=mode, on_native_start=on_native_start)
    row['case'] = copy.deepcopy(case)
    row['bridge'] = bridge_identity('slot89')
    return row


def collect_cohort(behavior, cases, *, league, controls, opponent=None,
                   environment_factory=None, mode='NATIVE', on_native_start=None):
    """Collect complete trajectories with the original fail-fast/terminal policy.

    Return an explicitly versioned league envelope containing the original cohort
    format. `upstream_cohort` retains the unmodified behavior and decision records;
    this does not grant automatic training/teacher admission.
    """
    module = load_collection()
    rows = []
    for case in cases:
        row = play_collection_game(behavior, case, league=league, controls=controls,
            opponent=opponent, environment_factory=environment_factory,
            mode=mode, on_native_start=on_native_start)
        rows.append(row)
        if row['status'] != 'COMPLETE':
            break
    if module.b.tensor_tree_digest(behavior.graph.state_dict()) != behavior.binding['weights_sha256']:
        raise ValueError('behavior Graph mutated during collection')
    cohort = module.bound_cohort(rows, behavior, mode)
    return seal({'schema':'slot98-league-full-economic-cohort-v1',
                 'bridge':bridge_identity('slot89'), 'upstream_cohort':cohort,
                 'training_admission':False})
