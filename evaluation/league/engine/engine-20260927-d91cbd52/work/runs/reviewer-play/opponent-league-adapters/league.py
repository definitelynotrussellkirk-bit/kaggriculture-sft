"""SHA-pinned, fresh-instance scripted league. Imports and source bytes are read-only.

The five registered scripts use only standard-library imports. Each execution gets
its own module (including embedded exec namespaces), not the cached tape loader.
Python's module RNG is saved/restored under a cooperative lock. Named agents use
no NumPy RNG; arbitrary third-party native/global state is NOT sandboxed here.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import threading
import types
from typing import Any

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[3]
SLOT81 = ROOT / 'work/runs/reviewer-play/public-winner-bc-freerun'
SLOT89 = ROOT / 'work/runs/reviewer-play/full-economic-selfplay-train'
for directory in (ROOT / 'src', SLOT81):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def exact_import(name: str, directory: Path):
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != (directory / (name + '.py')).resolve():
        raise ValueError(f'wrong-source module collision: {name}')
    return module


legacy = exact_import('checkpoint_adapter', SLOT81)
freerun = exact_import('freerun', SLOT81)
from farm_neat.game.exact_engine import AttrDict, DEFAULT_CONFIGURATION, PASS_ACTION

canonical = legacy.canonical
digest = legacy.digest
seal = legacy.bound
sha_file = legacy.sha256_file
OPPONENTS = ('PASS', 'FIRST_BC', 'SHOP_ROUTER', 'K0006', 'XMAN', 'MELON', 'A5362D13')
SCRIPTED = OPPONENTS[2:]
DEFAULT_CONTROLS = {
    'publication_steps': list(range(0, 719, 24)), 'command_budget': 32,
    'temperature': 1.0, 'seconds_per_game': 180.0,
    'trace_byte_limit': 100_000_000, 'device': 'cpu',
}
_LOCK = threading.RLock()
_NAMES = itertools.count()
# Cache only immutable compiled code, never a loaded module, namespace or agent.
_CODE: dict[tuple[str, str], types.CodeType] = {}


class MissingOpponent(ValueError):
    """A required frozen rival is absent. This never means PASS."""


def _seat_seed(seat: int, seed: int) -> None:
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError('explicit seat 0 or 1 required')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('explicit sampler seed in [0, 2**63) required')


def kaggle_configuration(configuration: dict | None) -> AttrDict:
    """Public engine vocabulary, with hidden seed redacted rather than inferred.

    Keep seed=None (the exact engine's post-initialization value): omitting the
    key can change scripts that distinguish .get('seed', 0) from explicit None.
    Evaluator case metadata, schedules and arbitrary config extras are excluded.
    """
    given = configuration or {}
    public = {key: copy.deepcopy(given.get(key, default))
              for key, default in DEFAULT_CONFIGURATION.items()}
    return AttrDict({**public, 'seed': None})


class ScriptedActor:
    """One trusted, pinned main.py instance for exactly one seat and one game."""
    def __init__(self, path: Path, *, expected_sha256: str, seat: int,
                 sampler_seed: int, identity: dict | None = None):
        _seat_seed(seat, sampler_seed)
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        if observed != expected_sha256:
            raise ValueError(f'main.py SHA-256 mismatch: {self.path}')
        self.seat, self.sampler_seed = seat, sampler_seed
        self.last_step, self.calls, self.cpu_seconds = -1, 0, 0.0
        self.identity = copy.deepcopy(identity or {'main_sha256': observed})
        self._rng_state = random.Random(sampler_seed).getstate()
        name = f'_slot98_actor_{observed[:12]}_{next(_NAMES)}'
        self.module = types.ModuleType(name)
        self.module.__file__ = str(self.path)
        self.module.__package__ = ''
        key = (str(self.path), observed)
        with _LOCK:
            code = _CODE.get(key)
            if code is None:
                code = _CODE[key] = compile(raw, str(self.path), 'exec')
        with self._runtime():
            exec(code, self.module.__dict__)
        self.agent = getattr(self.module, 'agent', None)
        if not callable(self.agent):
            raise ValueError(f'pinned source exports no agent(obs, cfg): {self.path}')

    @contextmanager
    def _runtime(self):
        # Scope also covers imports/from-random and embedded exec calls. No
        # mutation survives the call, including when the frozen agent raises.
        with _LOCK:
            caller_rng = random.getstate()
            previous_module = sys.modules.get(self.module.__name__)
            sys.modules[self.module.__name__] = self.module
            random.setstate(self._rng_state)
            try:
                yield
            finally:
                self._rng_state = random.getstate()
                random.setstate(caller_rng)
                if previous_module is None:
                    sys.modules.pop(self.module.__name__, None)
                else:
                    sys.modules[self.module.__name__] = previous_module

    def __call__(self, observation: dict, configuration: dict | None = None) -> dict:
        import time
        begun = time.process_time()
        clean = legacy.actor_view(observation, self.seat)
        step = clean.get('step')
        if type(step) is not int or step != self.last_step + 1 or not 0 <= step < 719:
            raise ValueError('fresh actor required; contiguous steps 0..718 only, no game reuse/retry')
        cfg = kaggle_configuration(configuration)
        # A failed decision consumes the step; never retry an altered RNG stream.
        self.last_step = step
        try:
            with self._runtime():
                action = self.agent(AttrDict(clean), cfg)
            if not isinstance(action, dict):
                raise ValueError('scripted agent returned a non-dictionary action; no fallback')
            canonical(action)  # Reject nonfinite/unserializable values; do not repair economics.
            self.calls += 1
            return copy.deepcopy(action)
        finally:
            self.cpu_seconds += time.process_time() - begun

    act = __call__


class PassActor:
    def __init__(self, seat: int, seed: int, identity: dict):
        _seat_seed(seat, seed)
        self.seat, self.identity = seat, copy.deepcopy(identity)
        self.calls, self.cpu_seconds = 0, 0.0

    def __call__(self, observation: dict, configuration=None):
        if observation.get('player') != self.seat:
            raise ValueError('PASS player/seat mismatch')
        self.calls += 1
        return copy.deepcopy(PASS_ACTION)

    act = __call__


def runtime_pins() -> dict:
    paths = [HOME / 'league.py', HOME / 'registry.json', SLOT81 / 'checkpoint_adapter.py',
             SLOT81 / 'freerun.py', ROOT / 'src/farm_neat/game/exact_engine.py',
             ROOT / 'src/farm_neat/evaluation/scenarios.py',
             ROOT / 'src/farm_neat/training/bc.py']
    return {str(p.relative_to(ROOT)): sha_file(p) for p in paths}


class League:
    opponents = OPPONENTS

    def __init__(self, *, root: Path = ROOT, first_bc_binding: dict | None = None,
                 candidate_binding: dict | None = None):
        self.root = Path(root).resolve()
        if self.root != ROOT:
            raise ValueError('league must use this source extraction, not a substituted root')
        self.registry = legacy.read_json(HOME / 'registry.json')
        legacy.check_bound(self.registry, 'league registry')
        if (self.registry['schema'] != 'farm-opponent-league-registry-v1'
                or tuple(self.registry['order']) != OPPONENTS):
            raise ValueError('closed seven-opponent registry required')
        self._bindings = {}
        for name, binding in (('FIRST_BC', first_bc_binding), ('CANDIDATE', candidate_binding)):
            if binding is not None:
                legacy.validate_binding(binding)
                self._bindings[name] = copy.deepcopy(binding)
        self._loaded = {}

    def identity(self, name: str) -> dict:
        if name not in OPPONENTS and name != 'CANDIDATE':
            raise ValueError(f'unknown opponent/candidate: {name}; no fallback')
        if name in ('FIRST_BC', 'CANDIDATE'):
            if name not in self._bindings:
                raise MissingOpponent(f'{name}: exact frozen checkpoint binding and weights required; no PASS fallback')
            binding = self._bindings[name]
            try:
                legacy.verify_artifacts(binding, self.root)
            except (ValueError, FileNotFoundError) as error:
                raise MissingOpponent(f'{name}: {error}') from error
            return seal({'schema': 'slot98-league-identity-v1', 'id': name,
                         'kind': 'FROZEN_FULL_ECONOMIC', 'binding': binding,
                         'registry_sha256': sha_file(HOME / 'registry.json'),
                         'runtime_pins': runtime_pins(),
                         'provenance': 'Explicit caller-supplied frozen checkpoint binding',
                         'license': 'Not asserted; retain supplied checkpoint provenance'})
        entry = copy.deepcopy(self.registry['entries'][name])
        for relative, expected in entry.get('assets', {}).items():
            p = (self.root / relative).resolve()
            if not p.is_relative_to(self.root) or not p.is_file() or sha_file(p) != expected:
                raise ValueError(f'frozen league asset SHA-256 mismatch or missing: {relative}')
        for key in entry.get('forbidden_environment', []):
            if os.environ.get(key):
                raise ValueError(f'{name}: unpinned external runtime input {key} is set')
        return seal({'schema': 'slot98-league-identity-v1', 'id': name, **entry,
                     'registry_sha256': sha_file(HOME / 'registry.json'),
                     'runtime_pins': runtime_pins(),
                     'rng': 'private Python random stream; world seed never an actor feature'})

    def describe(self, name: str) -> dict:
        try:
            return {'status': 'READY', 'identity': self.identity(name)}
        except MissingOpponent as error:
            return {'status': 'MISSING_INPUT', 'id': name, 'error': str(error)}

    def new_actor(self, name: str, *, seat: int, sampler_seed: int,
                  controls: dict | None = None, emit=None):
        _seat_seed(seat, sampler_seed)
        identity = self.identity(name)
        if name == 'PASS':
            return PassActor(seat, sampler_seed, identity)
        if name in ('FIRST_BC', 'CANDIDATE'):
            controls = copy.deepcopy(controls or DEFAULT_CONTROLS)
            if controls.get('device', 'cpu') != 'cpu':
                raise ValueError('this league entrypoint is CPU-only')
            if name not in self._loaded:
                self._loaded[name] = legacy.load_frozen(self._bindings[name], root=self.root, device='cpu')
            return legacy.LiveBCActor(self._loaded[name], seat=seat,
                sampler_seed=sampler_seed, controls=controls, emit=emit or (lambda event: None))
        return ScriptedActor(self.root / identity['main_path'],
            expected_sha256=identity['main_sha256'], seat=seat,
            sampler_seed=sampler_seed, identity=identity)
