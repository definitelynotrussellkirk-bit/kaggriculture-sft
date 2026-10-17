"""CPU submission runtime for the complete policy and its unchanged executor.

Only expert storage is quantized; all parameters run in FP32 after one decode.
This is a distinct deployment numerical profile, never a training checkpoint.
"""
from __future__ import annotations

import hashlib
import base64
import json
import math
import os
from pathlib import Path
import sys
import time
import zlib

import numpy as np
import torch

from farm_neat.economy.orders import FullSeasonPlan
from farm_neat.economy import action_space
from farm_neat.policy import network, fast_profile, plan_observation
from farm_neat.evaluation import deployment_codec as codec

ROOT = Path(__file__).resolve().parents[3]
STEPS = (0, 144, 288, 432, 576)
SCHEMA = 'farmneat-full-plan-cpu-deployment-v1'
DIAGNOSTIC_SCHEMA = 'farmneat-deployment-diagnostic-v1'
START_LOG_BYTES = 1600
PUBLICATION_LOG_BYTES = 8000
EPISODE_LOG_BYTES = START_LOG_BYTES + len(STEPS)*PUBLICATION_LOG_BYTES
MAX_DIAGNOSTIC_PAYLOAD_BYTES = 262144


def diagnostic_profile():
    # Kaggle's public core/agent default is 10,000 characters per stream/call.
    # ASCII JSON keeps byte/character accounting identical; step0 is <=9,600.
    return {'schema': DIAGNOSTIC_SCHEMA, 'stream': 'CURRENT_SYS_STDERR_PER_EVENT',
        'format': 'ASCII_JSONL_OR_LOSSLESS_ZLIB_BASE64_JSON_ENVELOPE',
        'startup_bytes': START_LOG_BYTES, 'publication_bytes': PUBLICATION_LOG_BYTES,
        'episode_bytes': EPISODE_LOG_BYTES, 'events_per_episode': 1+len(STEPS),
        'decoded_payload_bytes': MAX_DIAGNOSTIC_PAYLOAD_BYTES,
        'overflow': 'EXPLICIT_LOGGING_FAILURE; COMPLETE_DECISION_RETAINED_IN_MEMORY; NO_TRUNCATION',
        'failure_policy': 'COUNT_AND_REPORT_NEXT_EVENT; GAMEPLAY_CONTINUES'}


def _diagnostic_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def decode_diagnostic(line):
    """Bounded offline decoder; never execute a returned diagnostic payload."""
    if len(line.encode('ascii')) > PUBLICATION_LOG_BYTES:
        raise ValueError('bounded ASCII diagnostic line required')
    value = json.loads(line)
    if value.get('encoding') == 'zlib-base64':
        if value.get('decoded_bytes', MAX_DIAGNOSTIC_PAYLOAD_BYTES+1) > MAX_DIAGNOSTIC_PAYLOAD_BYTES:
            raise ValueError('bounded decoded diagnostic required')
        packed = base64.b64decode(value['payload'], validate=True)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(packed, MAX_DIAGNOSTIC_PAYLOAD_BYTES+1)
        if (len(raw) > MAX_DIAGNOSTIC_PAYLOAD_BYTES or not decoder.eof or decoder.unused_data
                or decoder.unconsumed_tail or len(raw) != value['decoded_bytes']
                or hashlib.sha256(raw).hexdigest() != value['payload_sha256']):
            raise ValueError('complete bound diagnostic payload required')
        decoded = json.loads(raw)
        if any(decoded.get(k) != value.get(k) for k in ('schema', 'event', 'sequence', 'step')):
            raise ValueError('diagnostic envelope identity mismatch')
        value = decoded
    if value.get('schema') != DIAGNOSTIC_SCHEMA:
        raise ValueError('deployment diagnostic schema required')
    return value


class DiagnosticLog:
    """At most six bounded events; logging failures never become game failures.

    Reserve an entire write before touching a stream, because a raising stream
    may already have consumed bytes. Never cache a stream or retry a partial line.
    Complete failed publication payloads already live in Agent.decisions.
    """
    def __init__(self, common, *, max_call_bytes=10000):
        self.common = common
        self.valid_call_limit = max_call_bytes is None or (type(max_call_bytes) is int and max_call_bytes >= 0)
        self.max_call_bytes = (10000 if max_call_bytes is None else max_call_bytes) if self.valid_call_limit else 0
        self.attempted_events = self.emitted_events = self.failed_events = 0
        self.reserved_bytes = self.written_bytes = self.raw_payload_bytes = 0
        self.failures = []
        self._step_reserved = {}

    def summary(self):
        return {'attempted_events': self.attempted_events, 'emitted_events': self.emitted_events,
            'failed_events': self.failed_events, 'reserved_bytes': self.reserved_bytes,
            'known_written_bytes': self.written_bytes, 'raw_payload_bytes': self.raw_payload_bytes,
            'complete': self.failed_events == 0 and self.attempted_events == self.emitted_events == 1+len(STEPS),
            'failures': list(self.failures)}

    def _failure(self, event, step, reason, **evidence):
        self.failed_events += 1
        if len(self.failures) < 1+len(STEPS):
            self.failures.append({'event': event, 'step': step, 'reason': reason, **evidence})

    def emit(self, event, *, step, **payload):
        sequence = self.attempted_events
        before = self.summary()
        self.attempted_events += 1
        cap = START_LOG_BYTES if event == 'EPISODE_START' else PUBLICATION_LOG_BYTES
        available = min(cap, EPISODE_LOG_BYTES-self.reserved_bytes,
                        self.max_call_bytes-self._step_reserved.get(step, 0))
        if sequence >= 1+len(STEPS) or available <= 0:
            self._failure(event, step, 'OUTPUT_BUDGET_EXHAUSTED' if self.valid_call_limit else 'INVALID_CONFIGURED_LOG_LIMIT')
            return False
        try:
            value = {'schema': DIAGNOSTIC_SCHEMA, **self.common, 'event': event, 'step': step,
                     'sequence': sequence, 'logging_before': before, **payload}
            raw = _diagnostic_json(value).encode('ascii')
            self.raw_payload_bytes += len(raw)
            line = raw.decode('ascii')+'\n'
            if len(raw) > MAX_DIAGNOSTIC_PAYLOAD_BYTES:
                raise ValueError('diagnostic payload limit')
            if len(line) > available:
                line = _diagnostic_json({'schema': DIAGNOSTIC_SCHEMA, 'event': event, 'step': step,
                    'sequence': sequence, 'encoding': 'zlib-base64', 'decoded_bytes': len(raw),
                    'payload_sha256': hashlib.sha256(raw).hexdigest(),
                    'payload': base64.b64encode(zlib.compress(raw, level=9)).decode('ascii')})+'\n'
            if len(line) > available:
                self._failure(event, step, 'COMPLETE_PAYLOAD_EXCEEDS_OUTPUT_BUDGET',
                              payload_sha256=hashlib.sha256(raw).hexdigest(), payload_bytes=len(raw))
                # Emit an explicit failure marker if it fits. No program prefix
                # masquerades as a complete program; the full decision is kept.
                line = _diagnostic_json({'schema': DIAGNOSTIC_SCHEMA, 'event': 'LOGGING_FAILURE',
                    'step': step, 'sequence': sequence, 'failed_event': event,
                    'reason': 'COMPLETE_PAYLOAD_EXCEEDS_OUTPUT_BUDGET', 'program_truncated': False,
                    'payload_sha256': hashlib.sha256(raw).hexdigest(), 'payload_bytes': len(raw),
                    'complete_program_retained_by_caller': True})+'\n'
                complete_event = False
            else:
                complete_event = True
            if len(line) > available:
                return False
        except Exception as error:
            self._failure(event, step, 'SERIALIZATION_FAILURE', error_type=type(error).__name__)
            return False
        self.reserved_bytes += len(line)
        self._step_reserved[step] = self._step_reserved.get(step, 0)+len(line)
        try:
            stream = sys.stderr  # Kaggle replaces streams between agent calls.
            count = stream.write(line)
            if type(count) is int and 0 <= count <= len(line):
                self.written_bytes += count
            if count != len(line):
                self._failure(event, step, 'SHORT_OR_UNKNOWN_WRITE')
                return False
            stream.flush()
        except Exception as error:
            self._failure(event, step, 'STREAM_FAILURE', error_type=type(error).__name__)
            return False
        if complete_event:
            self.emitted_events += 1
        return complete_event


def publication_evidence(proposal, controller):
    """Admission from the existing proposal; not a claim of economic realization."""
    rejected = [{k: r[k] for k in ('command_index', 'reason', 'transaction')} for r in proposal['rejected']]
    refused = {r['command_index'] for r in rejected}
    return {'proposal_sha256': proposal['content_sha256'],
        'published_sha256': controller.last_publication_sha256,
        'base_sha256': proposal['base_sha256'], 'base_revision': proposal['base_revision'],
        'observation_sha256': proposal['observation_sha256'],
        'executor_source_sha256': proposal['executor_source_sha256'],
        'accepted_command_indices': [i for i,c in enumerate(proposal['program']) if i not in refused and c['op'] != 'STOP'],
        'stop_command_indices': [i for i,c in enumerate(proposal['program']) if c['op'] == 'STOP'],
        'refused_commands': rejected, 'scope': 'COMMAND_ADMISSION_ONLY; NO_FULFILLMENT_OR_PROFIT_CLAIM'}


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_settings(settings):
    if settings.get('schema') != SCHEMA or settings.get('decision_steps') != list(STEPS):
        raise ValueError('explicit current full-plan deployment profile required')
    budget = settings.get('command_budget')
    if type(budget) is not int or not 2 <= budget <= action_space.MAX_COMMANDS:
        raise ValueError('explicit command slots including forced final STOP required')
    if type(settings.get('temperature')) not in (int, float) or not math.isfinite(settings['temperature']) or settings['temperature'] <= 0:
        raise ValueError('finite positive deployment temperature required')
    if settings.get('sampling_rng') != 'PCG64_FRESH_OS_ENTROPY_PER_EPISODE':
        raise ValueError('fresh private episode RNG profile required')
    seeds = settings.get('diagnostic_lane_seeds')
    if not isinstance(seeds, list) or len(seeds) != 2 or len(set(seeds)) != 2 or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds):
        raise ValueError('two distinct explicit sampler seeds required')


def episode_seed(fixed=None):
    if fixed is None:
        return int.from_bytes(os.urandom(8), 'little') & ((1 << 63)-1)
    if type(fixed) is not int or not 0 <= fixed < 2**63:
        raise ValueError('explicit diagnostic sampler seed required')
    return fixed


def collate_one(packed):
    """Reuse the canonical collation and bucket arithmetic for a CPU B1 cohort."""
    batch = plan_observation.collate([packed])
    n = batch['rows'].shape[1]
    extra = fast_profile.pad_bucket(n) - n
    if extra:
        for key, trailing in (('rows', (80,)), ('knownness', (80,)), ('token_mask', ())):
            a = batch[key]
            batch[key] = np.concatenate((a, np.zeros((1, extra, *trailing), dtype=a.dtype)), axis=1)
    return tuple(torch.as_tensor(batch[k]) for k in ('rows', 'token_mask', 'knownness'))


def load_graph(root, settings):
    """Authenticate the archive closure, decode every tensor and build full graph."""
    validate_settings(settings)
    for name, expected in settings['member_sha256'].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve() != path or not path.is_relative_to(root):
            raise ValueError('regular in-archive source required')
        if file_sha(path) != expected:
            raise ValueError('deployment member changed: ' + name)
    deployment = json.loads((root / 'MODEL.bin.json').read_text())
    original = deployment['original_manifest']
    if original['source_pins'] != network.sources() or original['architecture'] != json.loads(json.dumps(network.architecture())):
        raise ValueError('original policy source or architecture changed')
    if original['parameter_report']['total_trainable_parameters'] != 60_403_676 or original['parameter_report']['layers'] != 16:
        raise ValueError('complete retained 60M/16-layer actor required')
    arrays = codec.decode_state((root / 'MODEL.bin').read_bytes(), deployment,
        expected_manifest_sha256=settings['deployment_sha256'], expected_identity=deployment['codec_identity'])
    with torch.random.fork_rng(devices=[]):
        graph = network.torch_graph(device='cpu')
    graph.load_state_dict({n: torch.from_numpy(a) for n, a in arrays.items()}, strict=True)
    graph.eval()
    if network.parameter_report(graph) != original['parameter_report']:
        raise ValueError('decoded actor parameter accounting differs')
    return graph


class Agent:
    """Five whole-season publications with policy-owned economics and CPU inference."""
    def __init__(self, root=ROOT, *, expected_settings_sha256):
        self.root = Path(root).resolve()
        started = time.perf_counter()
        if file_sha(self.root / 'DEPLOYMENT.json') != expected_settings_sha256:
            raise ValueError('deployment settings identity changed')
        self.settings = json.loads((self.root / 'DEPLOYMENT.json').read_text())
        validate_settings(self.settings)
        if self.settings.get('diagnostics') != diagnostic_profile():
            raise ValueError('explicit current deployment diagnostic profile required')
        self.settings_sha256 = expected_settings_sha256
        if torch.cuda.is_initialized() or torch.is_autocast_enabled('cpu'):
            raise ValueError('CPU FP32 deployment required')
        torch.set_num_threads(1)
        self.graph = load_graph(self.root, self.settings)
        self.encoder = fast_profile.FastEncoder(self.graph, expected_identity=self.settings['fast_profile'])
        self.cold_seconds = time.perf_counter() - started
        self.controller = None
        self.seat = None
        self.last_step = -1
        self.decisions = []
        self.failed = False
        self._diagnostic_seed = None
        self.diagnostics = None

    def use_diagnostic_sampler_seed(self, seed):
        """Explicit preflight override before any game; main.py never calls this."""
        if self.last_step != -1 or self.controller is not None:
            raise ValueError('diagnostic seed must be selected before first call')
        self._diagnostic_seed = episode_seed(seed)

    def __call__(self, observation, configuration=None):
        if self.failed:
            raise RuntimeError('previous deployment failure; no fallback policy')
        try:
            return self.act(observation, configuration)
        except Exception:
            self.failed = True
            raise

    def act(self, observation, configuration=None):
        step, seat = observation['step'], observation['player']
        if type(step) is not int or not 0 <= step < 719 or type(seat) is not int or seat not in (0, 1):
            raise ValueError('supported physical step and seat required')
        if step == 0 and self.last_step in (-1, 718):
            self.seat, self.last_step, self.decisions = seat, -1, []
            self.controller = FullSeasonPlan('deployment-seat-' + str(seat), publication_steps=STEPS)
            self.episode_sampler_seed = episode_seed(self._diagnostic_seed)
            self.stream = fast_profile.LaneStream(self.episode_sampler_seed)
            common = {'model_sha256': self.settings['original_model_sha256'],
                'deployment_sha256': self.settings['deployment_sha256'], 'settings_sha256': self.settings_sha256,
                'runtime_source_sha256': self.settings['member_sha256']['src/farm_neat/evaluation/deployment_runtime.py'],
                'runtime': {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'numpy': str(np.__version__)},
                'seat': seat, 'sampler_seed': self.episode_sampler_seed,
                'sampler_seed_source': 'OS_ENTROPY' if self._diagnostic_seed is None else 'EXPLICIT_DIAGNOSTIC_OVERRIDE',
                'temperature': self.settings['temperature'], 'command_budget': self.settings['command_budget']}
            self.diagnostics = DiagnosticLog(common, max_call_bytes=(configuration or {}).get('maxLogLength', 10000))
            self.diagnostics.emit('EPISODE_START', step=step, cold_load_seconds=self.cold_seconds,
                                  diagnostic_profile=diagnostic_profile())
        if self.controller is None or self.seat != seat or step != self.last_step + 1:
            raise ValueError('consecutive complete-game calls with stable seat required')
        self.controller.observe(observation)
        if step in STEPS:
            started = time.perf_counter()
            packed = plan_observation.pack(observation, self.controller, configuration,
                                          expected_identity=self.settings['plan_adapter'])
            tensors = collate_one(packed)
            with torch.inference_mode():
                context, values, _ = self.encoder(*tensors)
                records, likelihood, entropy = fast_profile.sample_batched(self.graph, context, [self.stream],
                    max_commands=self.settings['command_budget'], temperature=self.settings['temperature'], as_of_step=step)
            if not all(bool(torch.isfinite(x).all()) for x in (context, values, likelihood, entropy)):
                raise ValueError('nonfinite neural inference')
            record = records[0]
            proposal = self.controller.propose(observation, record['program'])
            self.controller = self.controller.publish(observation, proposal)
            decision = {'step': step, 'input_sha256': packed['content_sha256'],
                'sampler_seed': self.episode_sampler_seed,
                'program': record['program'], 'joint_logp': record['joint_logp'],
                'factor_count': len(record['selected_logps']), 'rejected': proposal['rejected'],
                'seconds': time.perf_counter() - started}
            self.decisions.append(decision)
            try:
                decision['admission'] = publication_evidence(proposal, self.controller)
                self.diagnostics.emit('PUBLICATION', step=step, input_sha256=decision['input_sha256'],
                    program=decision['program'], program_sha256=action_space.digest(decision['program']),
                    behavior_joint_logp=decision['joint_logp'], factor_count=decision['factor_count'],
                    admission=decision['admission'], publication_seconds=decision['seconds'])
            except Exception as error:
                self.diagnostics._failure('PUBLICATION', step, 'DIAGNOSTIC_CONSTRUCTION_FAILURE',
                                          error_type=type(error).__name__)
        result = self.controller(observation, configuration)
        self.last_step = step
        return result
