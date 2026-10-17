"""CPU Kaggle runtime for the JOINT EntityGraph policy and its unchanged executor.

Owner: evaluation maintainer. Archive side only; the builder, cold preflight and
overage simulation live in joint_deployment.py. The actor is the archived,
byte-identical work/runs/reviewer-play/public-winner-bc-freerun/checkpoint_adapter.py
LiveBCActor driving EntityGraph(arm='joint') from the archived entity_model.py:
actor_view -> observation.pack(clean, None, cfg) -> encode -> sample(torch.Generator)
-> FullSeasonPlan propose/publish at dawns, controller(clean, cfg) every step.

Routed experts are stored as group64 int8 (deployment_codec) and decoded once to
FP32. This is a deployment numerical profile, not a training checkpoint; no
CPU/CUDA or original/quantized parity is claimed.

Overage guard (owner-approved): before each non-opening dawn publication, read the
raw observation['remainingOverageTime']. If that bank minus a conservative
predicted charge (safety factor x this episode's slowest measured publication
call, less actTimeout) is below the reserve, the dawn is skipped and the executor
keeps the currently published plan. Every skip is logged. Step 0 is never skipped.

Failure policy: any exception is logged and the step executes the last published
plan (PASS if none). The agent never raises into the host.

Stderr lines are compact ASCII JSON, one per call plus episode start/end, each
tagged "fj":1; see LOG_EVENTS.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

SCHEMA = 'farmneat-joint-cpu-deployment-v1'
OPTIONS_SCHEMA = 'farmneat-joint-deployment-options-v1'
GUARD_SCHEMA = 'farmneat-joint-overage-guard-v1'
LOG_SCHEMA = 'farmneat-joint-deployment-log-v1'
BINDING_SCHEMA = 'farmneat-joint-deployment-binding-v1'
ACTOR_KIND = 'JOINT_ENTITY_GRAPH_LIVE_BC'
LOADER_KIND = 'DEPLOYMENT_DECODED_GROUP64_INT8_EXPERT_STORAGE'
SAMPLING_RNG = 'TORCH_CPU_GENERATOR_SEEDED_FROM_OS_ENTROPY_PER_EPISODE'
FALLBACK = 'LOG_EXCEPTION_THEN_LAST_PUBLISHED_PLAN_ELSE_PASS'
ENTITY_MODEL = 'work/runs/reviewer-play/public-bc-entity-readout/entity_model.py'
ADAPTER = 'work/runs/reviewer-play/public-winner-bc-freerun/checkpoint_adapter.py'
ORIGINAL_MANIFEST = 'ORIGINAL-MODEL.json'
JOINT_SCHEMA = 'farm-full-economic-joint-state-option-policy-v1'
DAWNS = tuple(range(0, 720, 24))  # 30 dawns: 0, 24, ..., 696
THREAD_CAP = 4
LAST_STEP = 718
PASS = {'farmer': ['PASS'], 'hands': [], 'market': []}
LOG_TAG = 'fj'
LOG_EVENTS = {'S': 'episode start', 'P': 'call with published dawn plan', 'K': 'call with guard-skipped dawn',
              'T': 'ordinary call', 'X': 'call that raised and used the fallback', 'Z': 'episode end (after step 718)',
              'F': 'main.py agent construction failure (PASS thereafter)'}
MAX_ERROR_DETAILS = 20
DEFAULT_LOG_BYTES_PER_CALL = 10000


def default_profile():
    return {'publication_steps': list(DAWNS), 'command_budget': 8, 'temperature': 1.0}


def default_options():
    return {'schema': OPTIONS_SCHEMA, 'command_budget': None, 'publication_steps': None}


def default_guard(reserve_seconds=20.0):
    return {'schema': GUARD_SCHEMA, 'reserve_seconds': float(reserve_seconds), 'safety_factor': 1.5,
            'prior_publication_seconds': 6.0, 'never_skip_steps': [0],
            'bank_field': 'remainingOverageTime', 'act_timeout_field': 'actTimeout',
            'default_act_timeout_seconds': 1.0,
            'prediction': 'SAFETY_FACTOR_X_MAX_MEASURED_PUBLICATION_CALL_SECONDS_THIS_EPISODE_ELSE_PRIOR',
            'charge_rule': 'MAX(0, CALL_SECONDS - ACT_TIMEOUT)',
            'skip_rule': 'SKIP_NON_OPENING_DAWN_IF_BANK_MINUS_PREDICTED_CHARGE_BELOW_RESERVE; NO_BANK_MEANS_PUBLISH'}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def effective_profile(profile, options):
    """Default profile with explicit, versioned, default-OFF (None) options applied."""
    if type(options) is not dict or set(options) != {'schema', 'command_budget', 'publication_steps'} \
            or options['schema'] != OPTIONS_SCHEMA:
        raise ValueError('explicit versioned deployment options required')
    result = copy.deepcopy(profile)
    budget, steps = options['command_budget'], options['publication_steps']
    if budget is not None:
        result['command_budget'] = budget
    if steps is not None:
        if (type(steps) is not list or not steps or steps[0] != 0 or steps != sorted(set(steps))
                or any(type(s) is not int or s not in DAWNS for s in steps)):
            raise ValueError('publication-step option must be an ordered subset of the 30 dawns starting at 0')
        result['publication_steps'] = list(steps)
    return result


def validate_settings(settings):
    """Return the effective profile; refuse anything but the current joint schema."""
    from farm_neat.economy import action_space
    if type(settings) is not dict or settings.get('schema') != SCHEMA or settings.get('arm') != 'joint':
        raise ValueError('explicit current joint deployment settings required')
    if settings.get('profile') != default_profile():
        raise ValueError('versioned default joint profile required; cheaper profiles are options')
    effective = effective_profile(settings['profile'], settings.get('options'))
    if settings.get('effective_profile') != effective:
        raise ValueError('declared effective profile differs from profile+options')
    budget = effective['command_budget']
    if type(budget) is not int or not 2 <= budget <= action_space.MAX_COMMANDS:
        raise ValueError('command budget including forced final STOP required')
    if not _finite(effective['temperature']) or effective['temperature'] <= 0:
        raise ValueError('finite positive temperature required')
    guard = settings.get('overage_guard')
    expected = default_guard(guard.get('reserve_seconds', -1) if type(guard) is dict else -1)
    if type(guard) is not dict or guard != expected or not _finite(guard['reserve_seconds']) \
            or guard['reserve_seconds'] < 0:
        raise ValueError('explicit current overage guard with a finite nonnegative reserve required')
    for key, value in (('sampling_rng', SAMPLING_RNG), ('fallback', FALLBACK), ('log_schema', LOG_SCHEMA),
                       ('actor_kind', ACTOR_KIND)):
        if settings.get(key) != value:
            raise ValueError('joint deployment contract field changed: ' + key)
    if type(settings.get('fixture')) is not bool or type(settings.get('member_sha256')) is not dict:
        raise ValueError('explicit fixture flag and member pins required')
    return effective


def guard_decision(*, step, bank, publication_seconds, act_timeout, guard):
    """Pure skip rule. Returns (publish, detail); never skips a never-skip step."""
    if step in guard['never_skip_steps']:
        return True, {'reason': 'NEVER_SKIP'}
    if bank is None:
        return True, {'reason': 'NO_BANK'}
    basis = max(publication_seconds) if publication_seconds else guard['prior_publication_seconds']
    charge = max(0.0, guard['safety_factor']*basis - act_timeout)
    publish = bank - charge >= guard['reserve_seconds']
    return publish, {'reason': 'BANK_OK' if publish else 'BANK_BELOW_RESERVE', 'bank': bank,
                     'predicted_charge': charge, 'basis_seconds': basis, 'reserve': guard['reserve_seconds']}


def read_bank(observation, guard):
    try:
        value = observation.get(guard['bank_field'])
    except Exception:
        return None
    return float(value) if _finite(value) else None


def read_act_timeout(configuration, guard):
    try:
        value = (configuration or {}).get(guard['act_timeout_field'])
    except Exception:
        value = None
    return float(value) if _finite(value) and value > 0 else guard['default_act_timeout_seconds']


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_private(name, path):
    """Import a pinned file by path under a private module name (dataclasses need sys.modules)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def verify_members(root, pins):
    root = Path(root).resolve()
    for name, expected in sorted(pins.items()):
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve() != path or not path.is_relative_to(root):
            raise ValueError('regular in-archive member required: ' + name)
        if file_sha(path) != expected:
            raise ValueError('deployment member changed: ' + name)


def codec_view(original):
    """Codec-shaped copy of a joint manifest: adds only parameter_tensors, rebound.

    deployment_codec requires parameter_report.parameter_tensors, which joint
    manifests do not carry. The original manifest ships unchanged beside it and
    both runtime and builder require this exact derivation.
    """
    from farm_neat.evaluation import deployment_codec as codec
    report = original.get('parameter_report')
    if type(report) is not dict or 'parameter_tensors' in report or type(original.get('tensors')) is not dict:
        raise ValueError('joint manifest without codec tensor count required')
    body = {k: copy.deepcopy(v) for k, v in original.items() if k != 'content_sha256'}
    body['parameter_report'] = {**copy.deepcopy(report), 'parameter_tensors': len(original['tensors'])}
    body['codec_view_of_manifest_sha256'] = original['content_sha256']
    return {**body, 'content_sha256': codec.digest(body)}


def check_original(original, entity):
    """Exact joint manifest bound to the archived entity_model sources; returns Config."""
    base = entity.base
    if type(original) is not dict:
        raise ValueError('joint model manifest required')
    body = {k: v for k, v in original.items() if k != 'content_sha256'}
    if original.get('content_sha256') != base.actions.digest(body):
        raise ValueError('model manifest content SHA mismatch')
    if (original.get('arm') != 'joint' or original.get('schema') != entity.SCHEMAS['joint']
            or original.get('schema') != JOINT_SCHEMA or original.get('numerics') != entity.NUMERICS):
        raise ValueError('JOINT EntityGraph manifest required; pooled/cached manifests refused')
    if original.get('source_pins') != entity.sources():
        raise ValueError('entity_model source pins differ from the model manifest')
    config = base.hybrid.Config(**original['architecture']['config'])
    if original['architecture'] != entity.architecture('joint', config):
        raise ValueError('joint architecture mismatch')
    return config


def load_modules(root):
    root = Path(root).resolve()
    paths = {}
    for name in (ENTITY_MODEL, ADAPTER):
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise ValueError('regular archived actor source required: ' + name)
        paths[name] = path
    entity = load_private('farmneat_joint_entity_model', paths[ENTITY_MODEL])
    adapter = load_private('farmneat_joint_checkpoint_adapter', paths[ADAPTER])
    return entity, adapter


def load_graph(root, settings, entity):
    """Decode every tensor and strictly load EntityGraph(arm='joint') on CPU."""
    import torch
    from farm_neat.evaluation import deployment_codec as codec
    root = Path(root).resolve()
    raw = (root / ORIGINAL_MANIFEST).read_bytes()
    if hashlib.sha256(raw).hexdigest() != settings['original_manifest_file_sha256']:
        raise ValueError('original manifest file identity changed')
    original = json.loads(raw)
    config = check_original(original, entity)
    if original['content_sha256'] != settings['original_manifest_content_sha256']:
        raise ValueError('original manifest content identity changed')
    deployment = json.loads((root / 'MODEL.bin.json').read_bytes())
    if deployment.get('original_manifest') != codec_view(original):
        raise ValueError('deployment manifest does not encode this joint model')
    if deployment.get('original_checkpoint_sha256') != settings['original_model_sha256']:
        raise ValueError('deployment checkpoint identity mismatch')
    arrays = codec.decode_state((root / 'MODEL.bin').read_bytes(), deployment,
        expected_manifest_sha256=settings['deployment_sha256'], expected_identity=deployment['codec_identity'])
    with torch.device('meta'):
        graph = entity.EntityGraph(config, arm='joint')
    # assign=True adopts the decoded FP32 tensors directly; no random init/copy.
    graph.load_state_dict({n: torch.from_numpy(a) for n, a in arrays.items()}, strict=True, assign=True)
    del arrays
    if any(t.device.type != 'cpu' or t.dtype != torch.float32 for t in (*graph.parameters(), *graph.buffers())):
        raise ValueError('complete CPU FP32 joint graph required')
    graph.eval()
    graph.requires_grad_(False)
    if entity.parameter_report(graph) != original['parameter_report']:
        raise ValueError('decoded joint parameter accounting differs')
    return graph, original


def configure_threads(torch):
    count = os.cpu_count() or 1
    torch.set_num_threads(max(1, min(count, THREAD_CAP)))
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {'cpu_count': count, 'affinity': affinity, 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads()}


def rss_mb():
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1)
    except Exception:
        return None


class DawnGate:
    """LiveBCActor asks `step in controls['publication_steps']`; open only for an admitted dawn."""
    __slots__ = ('steps', 'open_step')

    def __init__(self, steps):
        self.steps, self.open_step = tuple(steps), None

    def __contains__(self, step):
        return self.open_step is not None and step == self.open_step

    def __iter__(self):
        return iter(self.steps)


class Episode:
    def __init__(self, index, seat, seed, profile):
        self.index, self.seat, self.seed = index, seat, seed
        self.steps = tuple(profile['publication_steps'])
        self.gate = DawnGate(self.steps)
        self.actor = None
        self.publication_seconds = []   # every attempted publication call (guard basis)
        self.publications, self.skips, self.errors, self.calls = [], [], [], []
        self.executor_failed = False
        self.decision = self.admission = None
        self.log_failures = 0
        self.ended = False


class Agent:
    """Per-episode LiveBCActor over one decoded joint graph; never raises to the host."""

    def __init__(self, root, *, expected_settings_sha256, started=None, entropy=os.urandom, clock=time.perf_counter):
        self.clock, self.entropy = clock, entropy
        started = clock() if started is None else started
        self.root = Path(root).resolve()
        raw = (self.root / 'DEPLOYMENT.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_settings_sha256:
            raise ValueError('deployment settings identity changed')
        self.settings = json.loads(raw)
        self.settings_sha256 = expected_settings_sha256
        self.profile = validate_settings(self.settings)
        self.guard = self.settings['overage_guard']
        verify_members(self.root, self.settings['member_sha256'])
        import numpy
        import torch
        if torch.cuda.is_initialized() or torch.is_autocast_enabled('cpu'):
            raise ValueError('CPU FP32 deployment required')
        self.hardware = configure_threads(torch)
        self.entity, self.adapter = load_modules(self.root)
        self.graph, self.original = load_graph(self.root, self.settings, self.entity)
        self._timing = {}
        self._instrument(self.graph)
        binding = self.adapter.bound({'schema': BINDING_SCHEMA, 'kind': LOADER_KIND,
            'original_model_sha256': self.settings['original_model_sha256'],
            'deployment_sha256': self.settings['deployment_sha256'], 'settings_sha256': expected_settings_sha256,
            'original_manifest_content_sha256': self.original['content_sha256'],
            'architecture_sha256': self.original['architecture']['content_sha256']})
        self.loaded = self.adapter.LoadedCheckpoint(binding, self.graph, self.original, LOADER_KIND)
        self.runtime = {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                        'numpy': str(numpy.__version__)}
        self.cold_seconds = clock() - started
        self.cold_reported = False
        self.episode_index = -1
        self.ep = None
        self.orphan_errors = []
        self._budget = DEFAULT_LOG_BYTES_PER_CALL
        self._attempt = 'T'

    # ----- instrumentation and logging -------------------------------------------------
    def _instrument(self, graph):
        encode, sample, timing, clock = graph.encode, graph.sample, self._timing, self.clock

        def timed_encode(*args, **kwargs):
            started = clock()
            result = encode(*args, **kwargs)
            timing['encode'] = clock() - started
            try:
                timing['entity_tokens'] = int(args[1].sum())
            except Exception:
                timing['entity_tokens'] = None
            return result

        def timed_sample(*args, **kwargs):
            started = clock()
            result = sample(*args, **kwargs)
            timing['sample'] = clock() - started
            return result
        graph.encode, graph.sample = timed_encode, timed_sample

    def _write(self, value, *, optional=()):
        """One bounded ASCII JSON line; optional keys drop first; failures never propagate."""
        try:
            value = {LOG_TAG: 1, **value}
            line = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n'
            for key in optional:
                if len(line) <= self._budget:
                    break
                if key in value:
                    size = len(json.dumps(value[key], separators=(',', ':'), ensure_ascii=True, allow_nan=False))
                    value = {**{k: v for k, v in value.items() if k != key}, key + '_omitted_bytes': size}
                    line = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                                      allow_nan=False) + '\n'
            if len(line) > self._budget:
                raise ValueError('log line exceeds per-call budget')
            stream = sys.stderr  # the host may replace streams between calls
            stream.write(line)
            stream.flush()
            self._budget -= len(line)
            return True
        except Exception:
            if self.ep is not None:
                self.ep.log_failures += 1
            return False

    def _emit(self, event):
        ep = self.ep
        if ep is None:
            return
        if event.get('type') == 'DECISION':
            behavior = event['behavior']
            ep.decision = {'program': copy.deepcopy(behavior['program']),
                'joint_logp': float(behavior['joint_logp']), 'factors': len(behavior['selected_logps']),
                'value': float(event['value_prediction']), 'input_sha256': event['input_sha256']}
        elif event.get('type') == 'PUBLICATION':
            ep.admission = {'proposal_sha256': event['proposal_sha256'],
                'refused': [[r['command_index'], str(r['reason'])[:80]] for r in event['refused_commands']]}

    # ----- episode control ---------------------------------------------------------------
    def _start_episode(self, seat):
        self.episode_index += 1
        self.ep = None
        seed = int.from_bytes(self.entropy(8), 'little') & ((1 << 63) - 1)
        ep = Episode(self.episode_index, seat, seed, self.profile)
        self.ep = ep
        controls = {'publication_steps': list(self.profile['publication_steps']),
                    'command_budget': self.profile['command_budget'], 'temperature': self.profile['temperature']}
        ep.actor = self.adapter.LiveBCActor(self.loaded, seat=seat, sampler_seed=seed, controls=controls,
                                            emit=self._emit)
        # The controller keeps the full dawn tuple; the actor publishes only when the gate opens.
        ep.actor.controls['publication_steps'] = ep.gate
        self._write({'e': 'S', 's': 0, 'schema': LOG_SCHEMA, 'ep': ep.index, 'seat': seat, 'seed': seed,
            'seed_src': 'OS_ENTROPY', 'cold_s': None if self.cold_reported else round(self.cold_seconds, 3),
            'settings': self.settings_sha256, 'deployment': self.settings['deployment_sha256'],
            'model': self.settings['original_model_sha256'], 'fixture': self.settings['fixture'],
            **self.runtime, **self.hardware, 'dawns': list(self.profile['publication_steps']),
            'budget': self.profile['command_budget'], 'temp': self.profile['temperature'],
            'options': {k: v for k, v in self.settings['options'].items() if k != 'schema'},
            'reserve': self.guard['reserve_seconds'], 'safety': self.guard['safety_factor'], 'rss_mb': rss_mb()})
        self.cold_reported = True

    def _act(self, observation, configuration):
        step, seat = observation.get('step'), observation.get('player')
        if type(step) is not int or type(seat) is not int:
            raise ValueError('integer step and player required')
        if step == 0:
            self._start_episode(seat)
        ep = self.ep
        if ep is None or ep.seat != seat:
            raise ValueError('no active episode for this seat')
        kind, detail = 'T', {}
        ep.decision = ep.admission = None
        self._timing.clear()
        self._attempt = 'T'
        if step in ep.steps:
            bank = read_bank(observation, self.guard)
            publish, detail = guard_decision(step=step, bank=bank, publication_seconds=ep.publication_seconds,
                act_timeout=read_act_timeout(configuration, self.guard), guard=self.guard)
            ep.gate.open_step = step if publish else None
            kind = self._attempt = 'P' if publish else 'K'
        action = ep.actor(observation, configuration)
        return action, kind, detail

    def _fallback(self, observation, configuration):
        ep = self.ep
        step = observation.get('step') if isinstance(observation, dict) else None
        if ep is None or ep.executor_failed or ep.actor is None:
            return copy.deepcopy(PASS), 'PASS_NO_EPISODE' if ep is None or ep.actor is None else 'PASS_EXECUTOR_FAILED'
        actor = ep.actor
        try:
            clean = self.adapter.actor_view(observation, ep.seat)
            if actor.controller.committed_step is None:
                actor.controller.observe(clean)
                action, mode = copy.deepcopy(PASS), 'PASS_NO_PLAN'
            else:
                action = actor.controller(clean, self.adapter.public_configuration(configuration))
                mode = 'LAST_PUBLISHED_PLAN'
            actor.last_step = step
            return action, mode
        except Exception as second:
            ep.executor_failed = True
            self._record_error(step, second, 'EXECUTOR')
            return copy.deepcopy(PASS), 'PASS_EXECUTOR_FAILED'

    def _record_error(self, step, error, stage):
        target = self.ep.errors if self.ep is not None else self.orphan_errors
        record = {'step': step, 'stage': stage, 'type': type(error).__name__}
        if len(target) < MAX_ERROR_DETAILS:
            record['message'] = str(error)[:300]
        target.append(record)
        return record

    def __call__(self, observation, configuration=None):
        started = self.clock()
        try:
            self._budget = int((configuration or {}).get('maxLogLength', DEFAULT_LOG_BYTES_PER_CALL))
        except Exception:
            self._budget = DEFAULT_LOG_BYTES_PER_CALL
        error = mode = attempted = None
        kind, detail = 'T', {}
        self._attempt = 'T'
        try:
            action, kind, detail = self._act(observation, configuration)
        except Exception as exc:
            error, attempted, kind = exc, self._attempt, 'X'
            try:
                action, mode = self._fallback(observation, configuration)
            except Exception:
                action, mode = copy.deepcopy(PASS), 'PASS_FALLBACK_FAILED'
        finally:
            if self.ep is not None:
                self.ep.gate.open_step = None
        seconds = self.clock() - started
        try:
            self._account(observation, kind, detail, seconds, error, mode, attempted)
        except Exception:
            pass
        return action

    def _account(self, observation, kind, detail, seconds, error, mode, attempted):
        ep = self.ep
        step = observation.get('step') if isinstance(observation, dict) else None
        bank = read_bank(observation, self.guard) if isinstance(observation, dict) else None
        line = {'e': kind, 's': step, 'ms': round(seconds*1000, 1), 'b': None if bank is None else round(bank, 3)}
        optional = ()
        if ep is not None:
            ep.calls.append([step, kind, seconds])
        if kind == 'P' or (kind == 'X' and attempted == 'P'):
            ep.publication_seconds.append(seconds)
        if kind == 'P':
            decision, admission, timing = ep.decision or {}, ep.admission or {}, dict(self._timing)
            program = decision.get('program')
            record = {'step': step, 'seconds': seconds, 'bank': bank, 'encode_seconds': timing.get('encode'),
                'sample_seconds': timing.get('sample'), 'entity_tokens': timing.get('entity_tokens'),
                'factors': decision.get('factors'), 'commands': None if program is None else len(program),
                'program': program, 'program_sha256': None if program is None else self.adapter.digest(program),
                'joint_logp': decision.get('joint_logp'), 'value': decision.get('value'),
                'input_sha256': decision.get('input_sha256'), 'proposal_sha256': admission.get('proposal_sha256'),
                'refused': admission.get('refused')}
            ep.publications.append(record)
            line.update(enc=None if record['encode_seconds'] is None else round(record['encode_seconds']*1000, 1),
                smp=None if record['sample_seconds'] is None else round(record['sample_seconds']*1000, 1),
                tok=record['entity_tokens'], fac=record['factors'], cmd=record['commands'],
                lp=None if record['joint_logp'] is None else round(record['joint_logp'], 4),
                v=None if record['value'] is None else round(record['value'], 4),
                rf=record['refused'], psha=record['program_sha256'], prog=program)
            optional = ('prog', 'rf')
        elif kind == 'K':
            ep.skips.append({'step': step, 'seconds': seconds, **detail})
            line.update(pc=round(detail['predicted_charge'], 3), basis=round(detail['basis_seconds'], 3),
                        rs=detail['reserve'])
        elif kind == 'X':
            record = self._record_error(step, error, 'ACT')
            record.update(mode=mode, attempted=attempted)
            line.update(err=record['type'], fb=mode, at=attempted)
            if 'message' in record:
                line['msg'] = record['message'][:200]
        self._write(line, optional=optional)
        if ep is not None and step == LAST_STEP and not ep.ended:
            ep.ended = True
            times = [c[2] for c in ep.calls]
            self._write({'e': 'Z', 's': step, 'ep': ep.index, 'calls': len(ep.calls), 'pubs': len(ep.publications),
                'skips': len(ep.skips), 'errs': len(ep.errors), 'exec_failed': ep.executor_failed,
                'tot_s': round(sum(times), 3), 'max_ms': round(max(times)*1000, 1),
                'pub_s': round(sum(p['seconds'] for p in ep.publications), 3),
                'log_failures': ep.log_failures, 'rss_mb': rss_mb()})

    def summary(self):
        ep = self.ep
        return {'episode': None if ep is None else ep.index, 'seat': None if ep is None else ep.seat,
            'sampler_seed': None if ep is None else ep.seed, 'sampler_seed_source': 'OS_ENTROPY',
            'cold_seconds': self.cold_seconds, 'runtime': self.runtime, 'hardware': self.hardware,
            'profile': self.profile, 'guard': self.guard, 'fixture': self.settings['fixture'],
            'publications': [] if ep is None else ep.publications, 'skips': [] if ep is None else ep.skips,
            'errors': (self.orphan_errors + ([] if ep is None else ep.errors)),
            'calls': [] if ep is None else ep.calls, 'executor_failed': None if ep is None else ep.executor_failed,
            'log_failures': None if ep is None else ep.log_failures}


def main_source(settings_sha256):
    """Archive-root main.py: lazy Agent at the first call; PASS forever if construction fails."""
    return ('''"""farmNEAT JOINT EntityGraph actor; CPU FP32 decoded from grouped-int8 expert storage."""
import os as _os
import sys as _sys
import time as _time
from pathlib import Path as _Path
_os.environ['CUDA_VISIBLE_DEVICES'] = ''
_root = _Path('/kaggle_simulations/agent')
if not (_root / 'DEPLOYMENT.json').is_file():
    _root = _Path(__file__).resolve().parent if '__file__' in globals() else _Path.cwd()
if str(_root / 'src') not in _sys.path:
    _sys.path.insert(0, str(_root / 'src'))
''' + f'_SETTINGS_SHA256 = {settings_sha256!r}\n' + '''_resident = None
_load_error = None


def _report(error):
    try:
        import json as _json
        import traceback as _traceback
        tail = ''.join(_traceback.format_exception(type(error), error, error.__traceback__)[-3:])[-1200:]
        _sys.stderr.write(_json.dumps({'fj': 1, 'e': 'F', 'err': type(error).__name__, 'msg': str(error)[:300],
            'tb': tail}, ensure_ascii=True, separators=(',', ':'), sort_keys=True) + '\\n')
        _sys.stderr.flush()
    except Exception:
        pass


def agent(observation, configuration=None):
    global _resident, _load_error
    if _resident is None and _load_error is None:
        _started = _time.perf_counter()
        try:
            from farm_neat.evaluation.joint_deployment_runtime import Agent as _Agent
            _resident = _Agent(_root, expected_settings_sha256=_SETTINGS_SHA256, started=_started)
        except Exception as _error:
            _load_error = _error
            _report(_error)
    if _resident is None:
        return {'farmer': ['PASS'], 'hands': [], 'market': []}
    return _resident(observation, configuration)
''').encode()
