"""Source-bound fixed-buffer V0 RL + admitted class-path teaching.

The RL objective, group advantages and raw replay are delegated to ``v0_rl``.
The atomic, round-trip-verified checkpoint transaction is inherited from its
native-BC-parent trainer; the distinct adapter identity prevents a combined child
from masquerading as a plain RL/BC continuation. BC supplies RNG/tree validation
and the cursor. One cursor item is ONE fixed, complete RL+paired buffer pass.

Inputs are pinned JSON files. DIRECT_SCHEMA wraps an unmodified v0_rl rollout
``document`` and one closed ``provenance`` row per game. PAIRED_SCHEMA contains
``protocol`` and ``opportunities=[{pair, admission}]``; each admission is recomputed.
The pair denominator counts ALL registered seat/context locks, not accepted rows,
worlds, tokens, or only the supplied bundles. Missing opportunities have zero weight.
Two-seat admission is a joint gate; its two class-path losses are averaged through
that seat/context denominator, not claimed as separate statistical discoveries.

``run_fit(plan, *, output)`` is the public consumer. ``plan``, ``train`` and
``replay`` are the production CLI; diagnostic models are Python/CPU fixtures only.
No collection, native games, promotion, or prediction-only labels occur here.
The bounded fit keeps the accepted two-update RL cap and resets Adam at the
objective boundary. Reference likelihoods are evaluated at the frozen parent,
then detached and identity-bound; a second in-memory reference model is unnecessary.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys
from typing import Any

import torch

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.policy import v0_scorer as A, v0_menu_fast
from farm_neat.training import v0_bc as BC, v0_rl as RL, v0_targets as T
from farm_neat.training import v0_regret_records as R, v0_regret_admission as AD
from farm_neat.training.v0_regret_loss import teacher_loss

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'farm-v0-regret-fit-v1'
PLAN_SCHEMA = SCHEMA + '-plan'
DIRECT_SCHEMA = SCHEMA + '-direct-corpus'
PAIRED_SCHEMA = SCHEMA + '-paired-corpus'
MODES = ('direct-rl-only', 'direct-rl+gated-bc', 'direct-rl+gated-pairwise')
REDUCTIONS = {
    'direct': RL.PROTOCOL['gradient_denominator'],
    'teacher': 'ALL_REGISTERED_SEAT_CONTEXT_PAIR_OPPORTUNITIES_IN_FROZEN_PROTOCOL;INCLUDING_UNADMITTED_AND_MISSING',
    'paths': 'SUM_OWN_PREFIX_CLASS_LOGPS_INCLUDING_SAMPLED_STOP;CAP_STOP_ZERO;NOT_FINAL_PLAN_MARGINAL',
    'passes': 'ONE_FIXED_COMPLETE_BUFFER_PER_UPDATE;NO_MINIBATCH_RENORMALIZATION',
    'gradient_recipients': 'ACTOR_ONLY;REFERENCE_ADMISSION_WEIGHT_DETACHED;NO_AUXILIARIES',
}
_need = RL._require


class CapabilityError(ValueError):
    """A required maintainer capability is absent; no substitute is constructed."""
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(code + ': ' + detail)


def source_pins() -> dict[str, str]:
    names = ('src/farm_neat/training/v0_regret_train.py',
             'src/farm_neat/training/v0_regret_loss.py',
             'src/farm_neat/training/v0_regret_admission.py',
             'project/runtime/interpreter.toml', 'docs/guides/compute.md',
             'src/farm_neat/rollout/v0_regret_collect.py',
             'src/farm_neat/rollout/v0_regret_dawn.py',
             'src/farm_neat/training/teachers/v0_regret_searcher.py',
             'src/farm_neat/training/teachers/v0_regret_budget.py')
    return {**RL.source_pins(), **BC.source_pins(), **R.source_pins(),
            **{name: RL.file_sha(ROOT / name) for name in names}}


def seal_plan(plan: dict) -> dict:
    result = copy.deepcopy(plan)
    result.pop('content_sha256', None)
    result['content_sha256'] = RL.digest(result)
    return result


def _ref(value: dict) -> dict:
    _need(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'closed file reference required')
    RL._nonempty(value['path'], 'file path'); RL._sha256(value['sha256'], label='file')
    return copy.deepcopy(value)


def _read(ref: dict) -> dict:
    ref = _ref(ref)
    return RL._pinned_json(ROOT / ref['path'], ref['sha256'])


def _experiment(plan: dict) -> dict:
    # Segment length, device and resume location are operational choices. Device
    # numerics are separately joined by the inherited checkpoint runtime identity.
    return {k: copy.deepcopy(v) for k, v in plan.items() if k not in ('execution', 'content_sha256')}


def build_plan(*, direct_plan: dict, direct_corpus: dict, paired_corpus: dict | None,
               mode: str, lambda_teacher: float, beta: float, device: str = 'cpu',
               updates: int = 2, recovery_corpus: dict | None = None,
               recovery_mix: dict | None = None, recovery_teacher_enabled: bool = True,
               teacher_objective: str = 'TERMINAL_WDL') -> dict:
    direct_plan = _ref(direct_plan)
    rl_plan = RL.load_plan(ROOT / direct_plan['path'], direct_plan['sha256'])
    result = seal_plan(dict(schema=PLAN_SCHEMA, direct_plan=direct_plan,
        direct_corpus=_ref(direct_corpus), paired_corpus=None if paired_corpus is None else _ref(paired_corpus),
        reference_parent=copy.deepcopy(rl_plan['behavior_checkpoint']), mode=mode,
        lambda_teacher=lambda_teacher, beta=beta, source_pins=source_pins(),
        reductions=copy.deepcopy(REDUCTIONS), rl_protocol=copy.deepcopy(RL.PROTOCOL),
        execution=dict(device=device, updates=updates, resume=None), strength_claim=False))
    if recovery_corpus is not None:
        result.update(recovery_corpus=_ref(recovery_corpus),
            recovery_mix=copy.deepcopy(recovery_mix) if recovery_mix is not None else
                dict(max_fraction=.25, max_degraded_fraction=.1, seed=0),
            recovery_teacher_enabled=recovery_teacher_enabled)
        result = seal_plan(result)
    else:
        _need(recovery_mix is None, 'recovery mix requires a pinned corpus')
    if teacher_objective != 'TERMINAL_WDL':
        from farm_neat.evaluation import v0_regret_pairs as E
        _need(teacher_objective == E.MARGIN_OBJECTIVE, 'MARGIN_OBJECTIVE_OPT_IN')
        result['teacher_objective'] = teacher_objective
        result = seal_plan(result)
    _validate_plan(result)
    return result


def with_execution(plan: dict, *, updates: int | None = None, device: str | None = None,
                   resume: dict | None = None) -> dict:
    """Create a new sealed segment request without changing its fit identity."""
    result = copy.deepcopy(plan)
    if updates is not None: result['execution']['updates'] = updates
    if device is not None: result['execution']['device'] = device
    result['execution']['resume'] = None if resume is None else _ref(resume)
    return seal_plan(result)


def _validate_plan(plan: dict) -> dict:
    keys = {'schema', 'direct_plan', 'direct_corpus', 'paired_corpus', 'reference_parent', 'mode',
            'lambda_teacher', 'beta', 'source_pins', 'reductions', 'rl_protocol', 'execution',
            'strength_claim', 'content_sha256'}
    if isinstance(plan, dict) and 'teacher_objective' in plan:
        from farm_neat.evaluation import v0_regret_pairs as E
        keys.add('teacher_objective')
        _need(plan['teacher_objective'] == E.MARGIN_OBJECTIVE, 'MARGIN_OBJECTIVE_OPT_IN')
        _need(plan.get('mode') == MODES[1] and plan.get('paired_corpus') is not None
              and not any(k in plan for k in ('recovery_corpus', 'belief_production')),
              'MARGIN_SUPERVISED_BC_ONLY')
    extension = {'recovery_corpus', 'recovery_mix', 'recovery_teacher_enabled'}
    if isinstance(plan, dict) and set(plan) & extension:
        keys |= extension
    if isinstance(plan,dict) and 'belief_production' in plan:
        keys.add('belief_production')
        ext=plan['belief_production']
        _need(type(ext) is dict and set(ext)=={'schema','population','options'}
              and ext['schema']==BELIEF_ARM_SCHEMA,'BELIEF_ARM_FIELDS')
        _ref(ext['population']); _ref(ext['options'])
    _need(isinstance(plan, dict) and set(plan) == keys and plan.get('schema') == PLAN_SCHEMA,
          'closed regret fit plan required')
    _need(plan == seal_plan(plan), 'fit plan content hash mismatch')
    _need(plan['source_pins'] == source_pins(), 'REGRET_SOURCE_MISMATCH')
    _need(plan['reductions'] == REDUCTIONS and plan['rl_protocol'] == RL.PROTOCOL,
          'objective/reduction/protocol drift')
    _need(plan['mode'] in MODES and plan['strength_claim'] is False, 'explicit supported mode required')
    for key, positive in (('lambda_teacher', False), ('beta', True)):
        x = plan[key]
        _need(type(x) in (int, float) and math.isfinite(x) and (x > 0 if positive else x >= 0),
              'finite valid ' + key + ' required')
    _need(plan['mode'] != MODES[0] or plan['lambda_teacher'] == 0, 'direct-rl-only requires lambda_teacher=0')
    execution = plan['execution']
    _need(isinstance(execution, dict) and set(execution) == {'device', 'updates', 'resume'},
          'closed execution request required')
    _need(type(execution['updates']) is int and 1 <= execution['updates'] <= RL.PROTOCOL['optimizer_updates'],
          'accepted RL update cap exceeded')
    _need(isinstance(execution['device'], str) and torch.device(execution['device']).type in ('cpu', 'cuda'),
          'CPU or CUDA device required')
    if execution['resume'] is not None: _ref(execution['resume'])
    _ref(plan['direct_corpus'])
    if plan['paired_corpus'] is not None: _ref(plan['paired_corpus'])
    if 'recovery_corpus' in plan:
        _ref(plan['recovery_corpus'])
        _need(type(plan['recovery_teacher_enabled']) is bool, 'explicit recovery teaching flag required')
        mix = plan['recovery_mix']
        _need(type(mix) is dict and set(mix) == {'max_fraction','max_degraded_fraction','seed'}, 'closed recovery mix required')
        for key in ('max_fraction','max_degraded_fraction'):
            _need(type(mix[key]) in (int,float) and math.isfinite(mix[key]) and 0 <= mix[key] <= .5,
                  'RECOVERY_CAP: finite fractions in [0,.5] required')
        _need(mix['max_degraded_fraction'] <= mix['max_fraction'] and type(mix['seed']) is int
              and 0 <= mix['seed'] < 2**63, 'RECOVERY_MIX_INVALID')
    rl_plan = RL.load_plan(ROOT / _ref(plan['direct_plan'])['path'], plan['direct_plan']['sha256'])
    _need(plan['reference_parent'] == rl_plan['behavior_checkpoint'], 'REFERENCE_PARENT_MISMATCH')
    return rl_plan


def _capability(rl_plan: dict, device: str, *, production_cli: bool = False) -> None:
    identity = rl_plan['behavior_model_identity']
    BC.validate_model_manifest(identity)
    if identity['diagnostic']:
        if production_cli:
            raise CapabilityError('DIAGNOSTIC_PLAN_REFUSED', 'production CLI never constructs reduced fixtures')
        if torch.device(device).type != 'cpu':
            raise CapabilityError('DIAGNOSTIC_CPU_ONLY', 'diagnostic fixture requires CPU')
    else:
        try:
            BC.require_production_runtime()
        except ValueError as exc:
            raise CapabilityError('PINNED_RUNTIME_REQUIRED', str(exc) + '; see project/runtime/interpreter.toml') from exc
        if torch.device(device).type != 'cuda':
            raise CapabilityError('PRODUCTION_SLURM_CUDA_REQUIRED', 'production fits require the registered GPU allocation')
        import os
        if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise CapabilityError('PRODUCTION_SLURM_CUDA_REQUIRED', 'SLURM_JOB_ID and CUDA_VISIBLE_DEVICES required')
    directory = ROOT / rl_plan['behavior_checkpoint']['path']
    missing = [str(directory / name) for name in ('COMMIT.json', 'state.pt') if not (directory / name).is_file()]
    if missing:
        raise CapabilityError('PARENT_WEIGHTS_UNAVAILABLE', '; '.join(missing))


def _direct_data(plan: dict, rl_plan: dict) -> tuple[RL.RolloutDataset, dict]:
    corpus = _read(plan['direct_corpus'])
    _need(set(corpus) == {'schema', 'document', 'provenance'} and corpus['schema'] == DIRECT_SCHEMA,
          'closed direct corpus with explicit provenance required')
    document, provenance = corpus['document'], corpus['provenance']
    _need(isinstance(document, dict) and isinstance(document.get('games'), list)
          and isinstance(provenance, list), 'DIRECT_PROVENANCE: games/provenance lists required')
    fields = {'game_id', 'direct_draw_id', 'type', 'mode', 'selection', 'confirmation_clone',
              'retry_count', 'checkpoint_sha256', 'sampler_seed', 'trace_sha256'}
    by_id, draws = {}, set()
    for row in provenance:
        _need(isinstance(row, dict) and set(row) == fields, 'DIRECT_PROVENANCE: closed origin row required')
        _need(row['type'] == 'direct_policy' and row['mode'] == 'sample_raw' and row['selection'] == 'none'
              and row['confirmation_clone'] is False and type(row['retry_count']) is int and row['retry_count'] == 0,
              'DIRECT_PROVENANCE: not an unselected original direct-policy draw')
        gid, draw = row['game_id'], row['direct_draw_id']
        _need(isinstance(gid, str) and gid and isinstance(draw, str) and draw
              and gid not in by_id and draw not in draws, 'DIRECT_PROVENANCE: duplicate/missing draw or game id')
        by_id[gid] = row; draws.add(draw)
    games = document['games']
    _need(len(games) == len(by_id) and {r.get('game_id') for r in games} == set(by_id),
          'DIRECT_PROVENANCE: exact game coverage required')
    for row in games:
        origin = by_id[row['game_id']]
        _need(origin['checkpoint_sha256'] == rl_plan['behavior_checkpoint']['commit_sha256']
              and origin['sampler_seed'] == row['sampler_seed']
              and origin['trace_sha256'] == RL.digest(row['decode_trace']),
              'DIRECT_PROVENANCE: parent/seed/trace join mismatch')
    try:
        data = RL.validate_rollout_document(document, plan=rl_plan, plan_sha256=plan['direct_plan']['sha256'],
                                            rollout_sha256=plan['direct_corpus']['sha256'])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('DIRECT_ROLLOUT_REJECTED: ' + str(exc)) from exc
    _need(data.diagnostics['excluded_groups'] == 0,
          'DIRECT_ROLLOUT_REJECTED: invalid/censored/missing group; do not silently change this fixed fit buffer')
    return data, by_id


def _paired_data(plan: dict) -> tuple[list[dict], dict]:
    if plan['paired_corpus'] is None:
        return [], dict(corpus=None, opportunity_count=0, admissions=[], active_pair_ids=[])
    corpus = _read(plan['paired_corpus'])
    if corpus.get('schema') == AD.MARGIN_CORPUS_SCHEMA:
        return _margin_paired_data(plan, corpus)
    _need(plan.get('teacher_objective', 'TERMINAL_WDL') == 'TERMINAL_WDL',
          'MARGIN_OBJECTIVE_REQUIRES_MARGIN_CORPUS')
    _need(set(corpus) == {'schema', 'protocol', 'opportunities'} and corpus['schema'] == PAIRED_SCHEMA,
          'closed paired corpus required')
    protocol = corpus['protocol']
    # Reuse the frozen-family validator, including all missing/unadmitted locks.
    try:
        tests = AD._protocol(protocol)
    except (AD._Refusal, R.RecordError) as exc:
        raise ValueError('PAIRED_PROTOCOL_REJECTED: ' + str(exc)) from exc
    _need(isinstance(corpus['opportunities'], list), 'opportunities list required')
    parent = plan['reference_parent']['commit_sha256']
    for test in tests.values():
        for lock in test['seat_locks']:
            _need(lock['parent_checkpoint_sha256'] == parent, 'PAIRED_REFERENCE_PARENT_MISMATCH')
    denominator = sum(len(test['seat_locks']) for test in tests.values())
    active, admissions, seen = [], [], set()
    for opportunity in corpus['opportunities']:
        _need(isinstance(opportunity, dict) and set(opportunity) == {'pair', 'admission'},
              'closed opportunity and admission receipt required')
        result = AD.admit_pair(opportunity['pair'], protocol=protocol)
        _need(result == opportunity['admission'], 'PAIRED_ADMISSION_RECEIPT_MISMATCH')
        _need(result['status'] not in ('INVALID', 'CENSORED', 'UNSUPPORTED'),
              'PAIRED_ADMISSION_REJECTED: ' + ','.join(result['reason_codes']))
        test_id = result['test_id']
        _need(test_id in tests and test_id not in seen, 'PAIRED_DUPLICATE_OR_UNREGISTERED_OPPORTUNITY')
        seen.add(test_id); admissions.append(result)
        pair = opportunity['pair']
        rows = pair['records'] if pair.get('schema') == AD.BUNDLE_SCHEMA else [pair]
        unique = {}
        for row in rows:
            clean = R.validate_pair(row)  # validate rejected geometry too; never re-tokenize it
            _need(clean['identity']['parent_checkpoint_sha256'] == parent, 'PAIRED_REFERENCE_PARENT_MISMATCH')
            unique[clean['identity']['pair_id']] = clean
        if result['actor_training_admissible'] and result['teacher_weight'] > 0:
            for row in unique.values():
                active.append(dict(record=row, lower_bound=result['lower_bound'], test_id=test_id))
    binding = dict(corpus=copy.deepcopy(plan['paired_corpus']), protocol_sha256=RL.digest(protocol),
        opportunity_count=denominator, admissions=admissions, missing_test_ids=sorted(set(tests) - seen),
        active_pair_ids=[row['record']['identity']['pair_id'] for row in active])
    return active, binding


def _margin_paired_data(plan: dict, corpus: dict) -> tuple[list[dict], dict]:
    from farm_neat.evaluation import v0_regret_pairs as E
    _need(plan.get('teacher_objective') == E.MARGIN_OBJECTIVE, 'MARGIN_OBJECTIVE_OPT_IN')
    _need(plan['mode'] == MODES[1], 'MARGIN_SUPERVISED_BC_ONLY')
    _need(set(corpus) == {'schema', 'request', 'result', 'admission'}
          and corpus['schema'] == AD.MARGIN_CORPUS_SCHEMA, 'MARGIN_CORPUS_FIELDS')
    request = corpus['request']
    E.validate_margin_request(request)
    _need(request['checkpoints']['old']['identity']['commit_sha256'] ==
          plan['reference_parent']['commit_sha256'], 'MARGIN_REFERENCE_PARENT_MISMATCH')
    admission = AD.admit_margin_experiment(request, corpus['result'])
    _need(admission == corpus['admission'], 'MARGIN_ADMISSION_RECEIPT_MISMATCH')
    rows = {r['cell_sha256']: r for r in admission['training_records']}
    active = []
    for cell in request['cells']:
        # The request validator already rebuilt every prefix/menu, even rejected
        # targets. Only complete, joined activation receipts can turn on weights.
        row = rows.get(cell['content_sha256'])
        if row is not None and row['teacher_weight'] > 0:
            _need(admission['activated'] is True and row['objective'] == E.MARGIN_OBJECTIVE
                  and row['loss'] == 'OFF_POLICY_SUPERVISED_CLASS_PATH' and row['on_policy'] is False
                  and row['target_behavior_logp'] is None and row['signed_margin_direction'] == 1
                  and 0 < row['teacher_weight'] <= 1., 'MARGIN_TEACHER_SEMANTICS')
            # The existing loss maps gate_value/2 to a bounded weight. This is
            # only an adapter coefficient, NOT a WDL confidence lower bound.
            active.append(dict(margin_cell=copy.deepcopy(cell), training_record=copy.deepcopy(row),
                teacher_gate=2.*row['teacher_weight'], test_id=cell['content_sha256'],
                source_kind='EXPERIMENTAL_OFF_POLICY_MARGIN'))
    binding = dict(corpus=copy.deepcopy(plan['paired_corpus']),
        protocol_sha256=request['content_sha256'], objective=E.MARGIN_OBJECTIVE,
        opportunity_count=32, admissions=[admission], missing_test_ids=[],
        active_pair_ids=[r['margin_cell']['content_sha256'] for r in active])
    return active, binding


def score_margin_path(model, cell: dict, side: str, *, check_behavior=False) -> dict:
    """Same entry-prefix and menu scorer, distinct evidence and likelihood roles."""
    _need(side in ('learner', 'challenger'), 'margin learner or challenger path required')
    _need(not check_behavior or side == 'learner', 'margin targets are off-policy, not behavior draws')
    state, prefix, options = R.validate_pair_v1_entry_prefix(cell['actor_context'], cell['continuation'])
    sample = cell['old' if side == 'learner' else 'new']['sample']
    return _score_geometry(model, state, prefix, options, sample, check_behavior=check_behavior)


def score_path(model: A.V0Scorer, record: dict, side: str, *, check_behavior: bool = False) -> dict:
    """Rebuild a class path on its OWN successors; never synthesize search logps.

    Only public actor context reaches the scorer. Trainer-only WDL, registration,
    provenance, parent hashes and opponent/world IDs are not feature inputs.
    ``raw`` is used solely to verify an existing learner draw, never for a search
    winner's PPO ratio. The search path's supervised likelihood is ``class``.
    """
    _need(side in ('learner', 'challenger'), 'learner or challenger path required')
    _need(not check_behavior or side == 'learner', 'search winners have no behavior likelihood')
    state, prefix, options = R.validate_pair_v1_entry_prefix(record['actor_context'], record['continuation'])
    proposal = record[side]
    return _score_geometry(model, state, prefix, options, proposal, check_behavior=check_behavior)


def _score_geometry(model, state, prefix, options, proposal, *, check_behavior=False):
    classes, raw, raw_steps = [], [], []
    for row in proposal['prefix_class_trace']:
        menu = v0_menu_fast.build_menu(state, prefix, projection_kwargs=options, backend='fast')
        _need(row['prefix'] == prefix.payload() and row['support'] == list(menu.legal_ids)
              and row['phi_sha256'] == RL.digest(menu.phi), 'PAIRED_STALE_PREFIX_OR_SUPPORT')
        action = row['raw_id']
        _need(action in menu.legal_ids, 'PAIRED_UNREACHABLE_ACTION')
        index = menu.legal_ids.index(action)
        cls = next(c for c in menu.classes if index in c.indices)
        forced = prefix.edit_count == V.MAX_EDITS_PER_EPOCH
        termination = ('CAP' if forced else 'STOP') if action == V.STOP_ID else 'CONTINUE'
        _need(row['class_id'] == cls.class_id and row['class_members'] == [menu.legal_ids[j] for j in cls.indices]
              and row['continuation'] == cls.successor.payload() and row['forced_stop'] is forced
              and row['termination'] == termination, 'PAIRED_STALE_CLASS_OR_STOP')
        logits = model.score(state, prefix, menu.legal_ids, menu.phi, memory={'hold_releases': options['hold_releases']})
        logps = A._raw_log_probs(logits, menu)
        raw.append(logps[index].double())
        classes.append(torch.logsumexp(logps[list(cls.indices)], 0).double())
        raw_steps.append(float(raw[-1].detach()))
        prefix = cls.successor  # saved successor/prefix is evidence, never decoder state
    _need(classes and prefix.terminated and proposal['raw_actions'] == [r['raw_id'] for r in proposal['prefix_class_trace']]
          and prefix.plan.canonical() == proposal['canonical_plan']
          and T.continuation_signature(prefix.plan, prefix.last_key, prefix.edit_count, prefix.terminated)
              == proposal['continuation_signature'], 'PAIRED_INCOMPLETE_OR_UNREACHABLE_PATH')
    raw_sum, class_sum = torch.stack(raw).sum(), torch.stack(classes).sum()
    if check_behavior:
        saved = proposal['sampling_provenance']['raw_step_logps']
        _need(proposal['sampling_provenance']['temperature'] == 1.0 and len(saved) == len(raw_steps)
              and all(math.isclose(a, b, rel_tol=2e-6, abs_tol=3e-5) for a, b in zip(saved, raw_steps))
              and math.isclose(float(raw_sum), proposal['raw_behavior_logp'], rel_tol=2e-6, abs_tol=3e-5),
              'PAIRED_COPIED_OR_STALE_BEHAVIOR_LIKELIHOOD')
    return {'raw': raw_sum, 'class': class_sum, 'raw_steps': raw_steps}


def _safe_output(path: str | Path) -> Path:
    path = Path(path)
    _need(not any(p.is_symlink() for p in (path, *path.parents)), 'symlinked output refused')
    _need(not path.exists(), 'output must be a new directory')
    return path


class Trainer(RL.Trainer):
    """Thin combined objective over the accepted native BC-parent RL transaction."""
    def __init__(self, fit_plan: dict):
        fit_plan = copy.deepcopy(fit_plan)
        rl_plan = _validate_plan(fit_plan)
        device = fit_plan['execution']['device']
        belief_data = _belief_data(fit_plan,rl_plan)
        _capability(rl_plan, device)
        direct, origins = _direct_data(fit_plan, rl_plan)
        active, paired_binding = _paired_data(fit_plan)
        recovery, paired_binding = _recovery_data(fit_plan, paired_binding,
            diagnostic=rl_plan['behavior_model_identity']['diagnostic'])
        active += recovery
        super().__init__(rl_plan, fit_plan['direct_plan']['sha256'], direct, RL.ARM_PG, device=device)
        self.fit_plan, self.active_pairs, self.paired_binding = fit_plan, active, paired_binding
        self.reference_logps = []
        self.model.eval()
        with torch.no_grad():
            # Reproduce the registered sampler draw, not merely a plausible logp.
            # This refuses traces copied from other seeds or a selected best-of-N.
            for game in direct.games:
                generator = torch.Generator(device=self.device).manual_seed(game.sampler_seed)
                result = A.decode_epoch(self.model, BC.actor_state(game.actor), Plan(), generator=generator,
                    mode='sample_raw', temperature=1.0, projection_kwargs=T.SETTINGS, backend='fast')
                _need(list(result.actions) == game.trace['actions'], 'DIRECT_DRAW_REPLAY: registered seed chose another trace')
            for pair in active:
                winner = _teacher_path(self.model, pair, 'challenger')['class']
                learner = _teacher_path(self.model, pair, 'learner', check_behavior=True)['class']
                self.reference_logps.append([float(winner), float(learner)])
        self.model.train()
        self.sampler = BC.Cursor(1, rl_plan['seed'])  # one work item = full fixed buffer
        self.teacher_active = (fit_plan['mode'] != MODES[0] and fit_plan['lambda_teacher'] > 0 and bool(active))
        if self.teacher_active:
            self.signal = {**self.signal, 'direct_signal': self.signal['status'], 'status': 'READY',
                           'action': 'RUN_REGISTERED_DIRECT_RL_PLUS_ADMITTED_TEACHER'}
        self.arm = {'id': fit_plan['mode'], 'bc_anchor': None}
        self.identity = {**self.identity, 'schema': SCHEMA, 'arm': copy.deepcopy(self.arm),
            'adapter': {'experiment': _experiment(fit_plan), 'paired': paired_binding,
                'direct_provenance_sha256': RL.digest(origins), 'reference_logps': self.reference_logps,
                'reference_parent': copy.deepcopy(fit_plan['reference_parent']),
                'reference_state_sha256': rl_plan['behavior_checkpoint_state_sha256'],
                'cursor': 'BC_CURSOR_ONE_FIXED_FULL_BUFFER_ITEM', 'no_production_strength_claim': True}}

        _belief_initialize(self,belief_data)

    def _live_join(self) -> None:
        _need(self.fit_plan['source_pins'] == source_pins(), 'REGRET_SOURCE_MISMATCH during fit')
        _need(self.identity['adapter']['experiment'] == _experiment(self.fit_plan), 'fit identity mutated')
        for key in ('direct_plan', 'direct_corpus', 'paired_corpus', 'recovery_corpus'):
            ref = self.fit_plan.get(key)
            if ref is not None:
                _need((ROOT / ref['path']).is_file() and RL.file_sha(ROOT / ref['path']) == ref['sha256'],
                      'live data hash mismatch: ' + key)
        if self.belief is not None:
            for key in ('population','options'):
                ref=self.fit_plan['belief_production'][key]
                _need(RL.file_sha(ROOT/ref['path'])==ref['sha256'],'BELIEF_LIVE_INPUT_DRIFT: '+key)
            if self.belief['enabled']:
                _need(_belief_cache_key(self.belief['rows'])==self.belief['cache_sha256'],
                      'BELIEF_DERIVED_CACHE_DRIFT')
            pop=self.belief['population']
            from farm_neat.rollout import v0_regret_collect as C
            _need(pop['source_pins']==C.belief_production_source_pins(),'BELIEF_LIVE_SOURCE_DRIFT')
            if 'native_join_inputs' in pop:
                # Embedded frozen labels cannot substitute for originals that
                # disappeared/changed after construction or preflight.
                for key in C.BELIEF_NATIVE_JOIN_REFS:
                    ref = pop['native_join_inputs'][key]
                    path = ROOT / ref['path']
                    _need(path.is_file() and RL.file_sha(path) == ref['sha256'],
                          'BELIEF_LIVE_PRODUCER_INPUT_DRIFT: ' + key)
            source=pop['callback_provenance']['producer_source']
            _need(RL.file_sha(ROOT/source['path'])==source['sha256'],'BELIEF_PRODUCER_SOURCE_DRIFT')
            _need(RL.digest(dict(population=pop,options=self.belief['options']))==self.belief['frozen_sha256'],
                  'BELIEF_IN_MEMORY_LABEL_DRIFT')
        parent = self.fit_plan['reference_parent']
        current = RL.inspect_bc_parent(ROOT / parent['path'], parent['commit_sha256'])
        _need(current['state_sha256'] == self.identity['adapter']['reference_state_sha256'], 'REFERENCE_PARENT_STATE_MISMATCH')

    def step(self) -> dict:
        _need(not self._failed, 'failed trainer must resume its last committed parent')
        _need(self.signal['status'] == 'READY', 'no RL or admitted teaching signal; no optimizer step')
        _need(self.global_update < RL.PROTOCOL['optimizer_updates'], 'pilot optimizer update cap reached')
        self._live_join()
        try:
            self.model.train(); self.optimizer.zero_grad(set_to_none=True)
            _need(self.sampler.take(1) == [0], 'fixed-buffer cursor mismatch')
            n, b = len(self.data.games), self.paired_binding['opportunity_count']
            pg_total, teacher_total, ratios, groups = 0.0, 0.0, [], {}
            # Literal accepted direct-RL calculation and accumulation order.
            for rollout in self.data.games:
                current = RL.replay_raw_logp(self.model, rollout, verify_behavior=False)
                loss, ratio = RL.clipped_pg_loss(current, rollout.old_logp, rollout.advantage)
                (loss / n).backward()
                contribution = float(loss.detach()) / n
                pg_total += contribution; ratios.append(float(ratio.detach()))
                group = groups.setdefault(rollout.group_id, {'games': 0, 'contribution': 0.0, 'advantages': []})
                group['games'] += 1; group['contribution'] += contribution; group['advantages'].append(rollout.advantage)
            teacher_rows = []
            if self.teacher_active:
                loss_mode = 'gated_bc' if self.fit_plan['mode'] == MODES[1] else 'gated_pair'
                for pair, reference in zip(self.active_pairs, self.reference_logps):
                    winner = _teacher_path(self.model, pair, 'challenger')['class'].reshape(1)
                    if loss_mode == 'gated_pair':
                        learner = _teacher_path(self.model, pair, 'learner')['class'].reshape(1)
                    else:
                        with torch.no_grad():
                            learner = _teacher_path(self.model, pair, 'learner')['class'].reshape(1)
                    ref_w, ref_l = (winner.new_tensor([value]) for value in reference)
                    credit_term = _belief_credit_term(self, pair, reference, winner, learner)
                    credit_report = None
                    if credit_term is not None:
                        term=credit_term.loss; credit_report=credit_term.report()
                    else:
                        term = teacher_loss(winner, learner, ref_w, ref_l,
                        torch.ones(1, dtype=torch.bool, device=self.device), winner.new_tensor([pair['teacher_gate'] if 'margin_cell' in pair else pair['lower_bound']]),
                        mode=loss_mode, beta=self.fit_plan['beta'], opportunity_count=b)
                    (self.fit_plan['lambda_teacher'] * term).backward()
                    teacher_total += float(term.detach())
                    teacher_rows.append({'pair_id': _teacher_id(pair),
                        'source_kind': pair.get('source_kind', 'ORDINARY'),
                        'test_id': pair['test_id'], 'winner_nll': -float(winner.detach()[0]),
                        'learner_nll': -float(learner.detach()[0]), 'reference_winner_nll': -reference[0],
                        'reference_learner_nll': -reference[1], 'contribution': float(term.detach())})
                    if credit_report is not None:
                        teacher_rows[-1]['decision_credit']=credit_report
            belief_metric = _belief_backward(self)
            gradient_groups = {}
            for name, param in self.model.named_parameters():
                group = gradient_groups.setdefault(name.split('.')[0], {'parameters': 0, 'with_grad': 0, 'squared': []})
                group['parameters'] += 1
                if param.grad is not None:
                    group['with_grad'] += 1; group['squared'].append(param.grad.detach().double().square().sum())
            for group in gradient_groups.values():
                values = group.pop('squared')
                group['norm'] = float(torch.stack(values).sum().sqrt()) if values else 0.0
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), RL.PROTOCOL['max_grad_norm'],
                                                  error_if_nonfinite=True, foreach=False)
            self.optimizer.step()
            _need(BC._finite_tree(self.model.state_dict()) and BC._finite_tree(self.optimizer.state_dict()),
                  'nonfinite combined optimizer update')
            self.global_update += 1
            metric = dict(global_update=self.global_update, complete_game_denominator=n, pair_opportunity_denominator=b,
                pg_loss=pg_total, teacher_loss=teacher_total, lambda_teacher=self.fit_plan['lambda_teacher'],
                total_loss=pg_total + self.fit_plan['lambda_teacher'] * teacher_total,
                teacher_optimizer_updates=int(self.teacher_active), grad_norm=float(norm),
                rl_groups=groups, gradient_groups=gradient_groups, teacher_rows=teacher_rows,
                pre_step_ratio_min=min(ratios), pre_step_ratio_max=max(ratios),
                post_step_ratios=RL.ratio_diagnostics(self.model, self.data),
                return_variance=self.data.diagnostics['return_variance'],
                nonzero_advantages=self.data.diagnostics['nonzero_advantages'])
            if belief_metric is not None:
                metric['belief']=belief_metric
                metric['total_loss']+=belief_metric['weighted_loss']
            return metric
        except BaseException:
            self._failed = True
            raise

    def snapshot(self) -> dict:
        result = {**super().snapshot(), 'cursor': BC._cpu_tree(self.sampler.state_dict()),
                'direct_games_seen': self.global_update * len(self.data.games),
                'pair_opportunities_seen': self.global_update * self.paired_binding['opportunity_count']}
        if self.belief is not None:
            result['belief_records_seen']=self.global_update*self.belief['population']['record_count']
        return result

    def _validate_snapshot(self, value: dict) -> None:
        added = {'cursor', 'direct_games_seen', 'pair_opportunities_seen'}
        if self.belief is not None:
            added.add('belief_records_seen')
            _need(type(value.get('belief_records_seen')) is int and value['belief_records_seen']==
                  value['global_update']*self.belief['population']['record_count'],'BELIEF_CURSOR_EXPOSURE_MISMATCH')
        _need(added <= set(value), 'complete regret cursor/exposure fields required')
        super()._validate_snapshot({k: v for k, v in value.items() if k not in added})
        probe = BC.Cursor(1, self.plan['seed']); probe.load_state_dict(value['cursor'])
        age = value['global_update']
        expected = BC.Cursor(1, self.plan['seed'])
        for _ in range(age): expected.take(1)
        _need(BC._same_tree(probe.state_dict(), expected.state_dict()), 'regret cursor/RNG/update mismatch')
        _need(type(value['direct_games_seen']) is int and value['direct_games_seen'] == age * len(self.data.games)
              and type(value['pair_opportunities_seen']) is int
              and value['pair_opportunities_seen'] == age * self.paired_binding['opportunity_count'],
              'regret cursor/exposure mismatch')

    def save(self, directory: str | Path, *, metrics: dict | None = None) -> dict:
        self._live_join(); _safe_output(directory)
        return super().save(directory, metrics=metrics)

    @classmethod
    def resume(cls, fit_plan: dict, directory: str | Path, expected_sha256: str) -> Trainer:
        directory = Path(directory)
        commit = RL._pinned_json(directory / 'COMMIT.json', expected_sha256)
        _need(commit.get('schema') == RL.COMMIT_SCHEMA and commit.get('status') == 'COMMITTED'
              and commit.get('identity', {}).get('schema') == SCHEMA, 'committed regret adapter checkpoint required')
        path = directory / 'state.pt'
        _need(path.is_file() and path.stat().st_size == commit['state_bytes']
              and RL.file_sha(path) == commit['state_sha256'], 'regret checkpoint payload hash/size mismatch')
        trainer = cls(fit_plan)
        _need(commit['identity'] == trainer.identity, 'regret parent/source/data/coefficient/reference/runtime mismatch')
        value = torch.load(path, map_location='cpu', weights_only=True)
        trainer._validate_snapshot(value)
        _need(value['global_update'] == commit['global_update'], 'regret commit age mismatch')
        trainer.model.load_state_dict(value['model'], strict=True)
        trainer.optimizer.load_state_dict(value['optimizer']); trainer.sampler.load_state_dict(value['cursor'])
        trainer.global_update = value['global_update']
        trainer.parent_sha256 = expected_sha256; trainer.parent_directory = directory.resolve()
        trainer.model.train(value['model_training'])
        BC._restore_rng(value['rng'], trainer.device)  # LAST: no init/replay may consume resumed RNG
        return trainer


def run_fit(plan: dict, *, output: str | Path) -> dict:
    """Perform only the explicitly pinned segment; never substitute missing capability."""
    _safe_output(output); _validate_plan(plan)
    resume = plan['execution']['resume']
    trainer = (Trainer(plan) if resume is None else
               Trainer.resume(plan, ROOT / resume['path'], resume['sha256']))
    return RL.run_segment(trainer, output, updates=plan['execution']['updates'])


def replay(plan: dict, *, checkpoint: dict) -> dict:
    """Verify an exact continuation and score current raw paths; zero optimizer steps."""
    checkpoint = _ref(checkpoint)
    trainer = Trainer.resume(plan, ROOT / checkpoint['path'], checkpoint['sha256'])
    trainer._validate_snapshot(trainer.snapshot())
    return dict(schema=SCHEMA + '-replay', status='EXACT_CONTINUATION_VERIFIED_NOT_STRENGTH',
        global_update=trainer.global_update, checkpoint=checkpoint,
        raw_path_ratios=RL.ratio_diagnostics(trainer.model, trainer.data),
        reference_parent=plan['reference_parent'], reductions=REDUCTIONS,
        optimizer_updates=0, native_games=0, strength_claim=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    planning = subs.add_parser('plan', help='freeze source/data/reference/coefficient identities; no training')
    for name in ('direct-plan', 'direct-corpus', 'paired-corpus', 'recovery-corpus', 'belief-population', 'belief-options'):
        planning.add_argument('--' + name, required=name in ('direct-plan','direct-corpus'))
        planning.add_argument('--' + name + '-sha256', required=name in ('direct-plan','direct-corpus'))
    planning.add_argument('--recovery-max-fraction', type=float, default=.25)
    planning.add_argument('--recovery-max-degraded-fraction', type=float, default=.1)
    planning.add_argument('--recovery-seed', type=int, default=0)
    planning.add_argument('--disable-recovery-teacher', action='store_true')
    from farm_neat.evaluation import v0_regret_pairs as E
    planning.add_argument('--teacher-objective', choices=('TERMINAL_WDL', E.MARGIN_OBJECTIVE), default='TERMINAL_WDL')
    planning.add_argument('--mode', choices=MODES, required=True)
    planning.add_argument('--lambda-teacher', type=float, required=True)
    planning.add_argument('--beta', type=float, required=True)
    planning.add_argument('--device', default='cuda'); planning.add_argument('--updates', type=int, default=2)
    planning.add_argument('--output', required=True)
    for command in ('train', 'replay', 'belief-preflight'):
        p = subs.add_parser(command)
        p.add_argument('--plan', required=True); p.add_argument('--plan-sha256', required=True)
        if command == 'train':
            p.add_argument('--output', required=True); p.add_argument('--updates', type=int)
            p.add_argument('--device'); p.add_argument('--resume'); p.add_argument('--resume-sha256')
        elif command == 'replay':
            p.add_argument('--checkpoint', required=True); p.add_argument('--checkpoint-sha256', required=True)
    p = subs.add_parser('recovery-evaluation-plan', help='write heldout/ablation recipe; no models or games run')
    p.add_argument('--corpus', required=True); p.add_argument('--corpus-sha256', required=True)
    p.add_argument('--ordinary-opportunities', type=int, required=True)
    p.add_argument('--updates', type=int, default=2); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-fraction', type=float, default=.25)
    p.add_argument('--max-degraded-fraction', type=float, default=.1)
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    started_fit = False
    try:
        if args.command == 'recovery-evaluation-plan':
            corpus = RL._pinned_json(args.corpus, args.corpus_sha256)
            result = recovery_evaluation_recipe(corpus, optimizer_updates=args.updates,
                ordinary_opportunities=args.ordinary_opportunities, max_fraction=args.max_fraction,
                max_degraded_fraction=args.max_degraded_fraction, seed=args.seed)
            RL._write_json_new(args.output, result)
        elif args.command == 'plan':
            _need(bool(args.paired_corpus) == bool(args.paired_corpus_sha256), 'paired corpus path/hash required together')
            _need(bool(args.recovery_corpus) == bool(args.recovery_corpus_sha256), 'recovery corpus path/hash required together')
            result = build_plan(direct_plan={'path': args.direct_plan, 'sha256': args.direct_plan_sha256},
                direct_corpus={'path': args.direct_corpus, 'sha256': args.direct_corpus_sha256},
                paired_corpus=None if args.paired_corpus is None else
                    {'path': args.paired_corpus, 'sha256': args.paired_corpus_sha256},
                mode=args.mode, lambda_teacher=args.lambda_teacher, beta=args.beta,
                teacher_objective=args.teacher_objective,
                device=args.device, updates=args.updates,
                recovery_corpus=None if args.recovery_corpus is None else {'path':args.recovery_corpus,'sha256':args.recovery_corpus_sha256},
                recovery_mix=None if args.recovery_corpus is None else dict(max_fraction=args.recovery_max_fraction,
                    max_degraded_fraction=args.recovery_max_degraded_fraction, seed=args.recovery_seed),
                recovery_teacher_enabled=not args.disable_recovery_teacher)
            _need(bool(args.belief_population)==bool(args.belief_population_sha256)==
                  bool(args.belief_options)==bool(args.belief_options_sha256),'BELIEF_POPULATION_OPTIONS_PATHS_AND_HASHES_REQUIRED')
            if args.belief_population:
                result=with_belief_production(result,
                    population=dict(path=args.belief_population,sha256=args.belief_population_sha256),
                    options=dict(path=args.belief_options,sha256=args.belief_options_sha256))
            RL._write_json_new(args.output, result)
            result = {'status': 'PLANNED_NOT_RUN', 'path': args.output, 'sha256': RL.file_sha(args.output)}
        else:
            plan = RL._pinned_json(args.plan, args.plan_sha256)
            if args.command == 'train':
                _need(bool(args.resume) == bool(args.resume_sha256), 'resume path/hash required together')
                plan = with_execution(plan, updates=args.updates, device=args.device,
                    resume=plan['execution']['resume'] if args.resume is None else
                        {'path': args.resume, 'sha256': args.resume_sha256})
            if args.command == 'belief-preflight':
                print(RL.canonical(belief_preflight(plan))); return 0
            rl_plan = _validate_plan(plan)
            _belief_data(plan,rl_plan)
            _capability(rl_plan, plan['execution']['device'], production_cli=True)
            started_fit = args.command == 'train'
            result = (run_fit(plan, output=args.output) if args.command == 'train' else
                      replay(plan, checkpoint={'path': args.checkpoint, 'sha256': args.checkpoint_sha256}))
        print(RL.canonical(result)); return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        failure = Path(args.output) / 'FAILURE.json' if started_fit else None
        print(RL.canonical({'schema': SCHEMA + '-error',
            'status': ('CAPABILITY_ERROR' if isinstance(exc, CapabilityError) else
                       'FAILED' if started_fit else 'REFUSED'),
            'code': getattr(exc, 'code', type(exc).__name__), 'error': str(exc),
            # A late exception may follow a committed update. Do not report zero
            # without evidence; the immutable last COMMIT is authoritative.
            'optimizer_updates': None if started_fit else 0,
            'continuation_receipt': str(failure) if failure is not None and failure.is_file() else None,
            'strength_claim': False}))
        return 2



def mix_recovery_curriculum(corpus: dict, *, ordinary_opportunities: int,
                            max_fraction: float = .25, max_degraded_fraction: float = .1,
                            seed: int = 0) -> dict:
    """Deterministic stratified fixed-buffer selection; no replacement or top-LCB.

    Fractions bound actual opportunity counts, including zero-weight negatives.
    At most one source per lineage enters a buffer; held-out sources never do.
    At least half the buffer remains ordinary, and a separate degraded-source
    cap stops constructed recoveries from replacing the natural distribution.
    """
    from collections import defaultdict, deque
    from fractions import Fraction
    import random
    from farm_neat.rollout import v0_regret_collect as C
    _need(type(ordinary_opportunities) is int and ordinary_opportunities >= 0,
          'nonnegative ordinary opportunity count required')
    for key, value in (('max_fraction', max_fraction), ('max_degraded_fraction', max_degraded_fraction)):
        _need(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= .5,
              'RECOVERY_CAP: '+key+' must be in [0,.5]')
    _need(max_degraded_fraction <= max_fraction, 'DEGRADED_CAP_EXCEEDS_RECOVERY_CAP')
    _need(type(seed) is int and 0 <= seed < 2**63, 'curriculum seed required')
    rows = C.validate_recovery_corpus(corpus)
    rng = random.Random(seed)
    fraction, degraded_fraction = Fraction(str(max_fraction)), Fraction(str(max_degraded_fraction))
    capacity = int(ordinary_opportunities*fraction/(1-fraction))
    buckets = defaultdict(list)
    for row in sorted(rows, key=lambda r:r['source_id']):
        if row['split'] != 'training': continue
        stratum = row['stratum']
        # Lineages are separately de-duplicated. Every remaining requested axis
        # participates; the measured weight does not order candidates.
        key = RL.digest({k:v for k,v in stratum.items() if k != 'source_family'})
        buckets[key].append(row)
    queues = []
    for key in sorted(buckets):
        rng.shuffle(buckets[key]); queues.append(deque(buckets[key]))
    rng.shuffle(queues)
    selected, used_families, degraded = [], set(), 0
    while len(selected) < capacity:
        progressed = False
        for queue in queues:
            while queue and queue[0]['source_family_id'] in used_families: queue.popleft()
            if not queue or len(selected) >= capacity: continue
            row = queue[0]; is_degraded = row['kind'] == 'LEGALLY_DEGRADED'
            if is_degraded and degraded+1 > int(degraded_fraction*(ordinary_opportunities+len(selected)+1)):
                continue
            queue.popleft(); selected.append(row); used_families.add(row['source_family_id'])
            degraded += int(is_degraded); progressed = True
        if not progressed: break
    return dict(schema='farm-v0-recovery-mixture-v1', corpus_sha256=corpus['content_sha256'],
        ordinary_opportunities=ordinary_opportunities, opportunity_count=ordinary_opportunities+len(selected),
        selected_source_ids=[r['source_id'] for r in selected], selected=selected,
        degraded_count=degraded, zero_weight_selected=sum(r['gate']['teacher_weight']==0 for r in selected),
        max_fraction=max_fraction, max_degraded_fraction=max_degraded_fraction, seed=seed,
        heldout_exposure=0, sampling='FIXED_STRATIFIED_WITHOUT_REPLACEMENT_ONE_PER_SOURCE_FAMILY')


def score_recovery_path(model: A.V0Scorer, row: dict, side: str) -> dict:
    """Class-path score from the real dawn and exact HOLD/started memory.

    This is supervised/reference scoring, NEVER a recovered behavior logp.
    It cannot be passed into the direct RL buffer. Only projection State, native
    Prefix and hold_releases reach model.score; no provenance or hidden future.
    """
    from farm_neat.rollout import v0_regret_dawn as D, v0_regret_collect as C
    from farm_neat.training.teachers import v0_regret_searcher as S
    _need(side in ('learner', 'challenger'), 'recovery side required')
    source = C._check_recovery_source(row['source']); snapshot = source['snapshot']
    _need(row['verification'] == C._source_verification(source), 'RECOVERY_REPLAY_VERIFICATION_JOIN')
    result = row['recovery']
    _need(result is not None and result['actor_context'] == D.recovery_actor_context(snapshot), 'RECOVERY_ACTOR_JOIN')
    expected = result['candidates'][0] if side == 'learner' else result['best']
    geometry = S._recovery_candidate(snapshot, expected['raw_actions'], result['actor_context'], {}, edit_limit=16)
    _need(all(geometry[k] == expected[k] for k in ('raw_actions','canonical_plan',
        'prefix_class_trace','continuation_signature')), 'RECOVERY_STALE_PREFIX_OR_MEMORY')
    env, agent = D.validate_snapshot(snapshot, continuation=snapshot['continuation'])
    state = D.P.State(env.agent_observation(snapshot['seat']))
    proposal = {**geometry, 'continuation_signature':geometry['continuation_signature']['decoder']}
    return _score_geometry(model, state, A.Prefix(agent.plan), agent.projection_kw(), proposal,
                           check_behavior=False)


def _teacher_path(model, pair, side, *, check_behavior=False):
    if 'margin_cell' in pair:
        return score_margin_path(model, pair['margin_cell'], side, check_behavior=check_behavior)
    if 'recovery_row' in pair:
        return score_recovery_path(model, pair['recovery_row'], side)
    return score_path(model, pair['record'], side, check_behavior=check_behavior)


def _teacher_id(pair):
    if 'margin_cell' in pair:
        return pair['margin_cell']['content_sha256']
    return (pair['recovery_row']['source']['recipe']['source_id'] if 'recovery_row' in pair
            else pair['record']['identity']['pair_id'])


def _recovery_data(plan, ordinary_binding, *, diagnostic):
    from farm_neat.rollout import v0_regret_collect as C
    ref = plan.get('recovery_corpus')
    if ref is None: return [], ordinary_binding
    corpus = _read(ref)
    mixture = mix_recovery_curriculum(corpus, ordinary_opportunities=ordinary_binding['opportunity_count'],
                                     **plan['recovery_mix'])
    rows = {r['source']['recipe']['source_id']:r for r in corpus['rows']}
    parent = plan['reference_parent']['commit_sha256']
    for row in corpus['rows']:
        recipe = row['source']['recipe']
        _need(recipe['checkpoint_sha256'] == parent, 'RECOVERY_REFERENCE_PARENT_MISMATCH')
        _need(diagnostic or recipe['evidence_scope'] == 'NATIVE_GAME', 'FIXTURE_RECOVERY_CANNOT_TRAIN_PRODUCTION')
    active = []
    for selected in mixture['selected']:
        if selected['gate']['teacher_weight'] > 0 and plan['recovery_teacher_enabled']:
            active.append(dict(recovery_row=rows[selected['source_id']],
                lower_bound=selected['gate']['lower_bound'], test_id=selected['source_id'],
                source_kind=selected['kind']))
    binding = {**ordinary_binding, 'ordinary_opportunity_count':ordinary_binding['opportunity_count'],
        'opportunity_count':mixture['opportunity_count'], 'recovery_corpus':copy.deepcopy(ref),
        'recovery_mixture':mixture, 'recovery_teacher_enabled':plan['recovery_teacher_enabled']}
    return active, binding


def recovery_evaluation_recipe(corpus: dict, *, optimizer_updates: int,
                               ordinary_opportunities: int, max_fraction: float = .25,
                               max_degraded_fraction: float = .1, seed: int = 0) -> dict:
    """Prospective direct-policy test/ablation contract; no fit or game is run.

    Both arms read the SAME immutable sources and use the SAME denominator,
    parent, ordinary RL buffer and number of updates. Base masks recovery teacher
    terms off; treatment enables only admitted terms. This matches registered
    data exposure, not the number of nonzero correction gradients (the treatment).
    """
    from farm_neat.rollout import v0_regret_collect as C
    _need(type(optimizer_updates) is int and 1 <= optimizer_updates <= RL.PROTOCOL['optimizer_updates'],
          'use bounded accepted trainer update cap')
    rows = C.validate_recovery_corpus(corpus)
    held = [r for r in rows if r['split'] == 'heldout']
    _need(held, 'HELDOUT_NATURAL_BAD_SOURCES_REQUIRED')
    mixture = mix_recovery_curriculum(corpus, ordinary_opportunities=ordinary_opportunities,
        max_fraction=max_fraction, max_degraded_fraction=max_degraded_fraction, seed=seed)
    exposure = dict(ordinary_pair_opportunities=optimizer_updates*ordinary_opportunities,
        recovery_source_opportunities=optimizer_updates*len(mixture['selected_source_ids']),
        recovery_source_ids=mixture['selected_source_ids'], direct_rl='IDENTICAL_PINNED_DIRECT_CORPUS_EACH_UPDATE')
    arms = [dict(name=name, optimizer_updates=optimizer_updates, registered_source_exposures=exposure,
                 recovery_teacher_enabled=flag, teacher_denominator=mixture['opportunity_count'])
            for name, flag in (('base-regret',False),('base-regret-plus-recovery',True))]
    return dict(schema='farm-v0-recovery-evaluation-recipe-v1', status='RECIPE_NOT_EXECUTED',
        corpus_sha256=corpus['content_sha256'], mixture=mixture,
        evaluation=dict(source_ids=[r['source_id'] for r in held], search_enabled=False,
            teacher_prefixes=False, source_kind='NATURAL_BAD', frozen_before_fit=True,
            current_dawn='SAMPLE_STUDENT_ON_OWN_PREFIX_FROM_RESTORED_SNAPSHOT_THEN_REAL_COMPILER',
            future_dawns='SAME_SEARCH_DISABLED_STUDENT_WITH_FRESH_REGISTERED_RNG',
            state_restore='EXACT_WORLD_CONTROLLER_STARTED_HOLD_OPPONENT_STATE;REPLACE_ONLY_FROZEN_ACTOR_LAW',
            pairing='SAME_HELDOUT_SOURCES_SEATS_RIVALS_AND_FROZEN_SAMPLER_SEEDS_FOR_BOTH_ARMS',
            metrics=['terminal_win_draw_loss_counts','mean_terminal_wdl_utility',
                     'paired_student_minus_base_terminal_wdl_delta',
                     'recovery_rate=mean(student_terminal_wdl>base_terminal_wdl)',
                     'source_family_clustered_uncertainty','censored_and_failed_counts'],
            ordinary_state_regression=dict(required=True, population='SEPARATE_HELDOUT_ORDINARY_LEARNER_DAWNS_AND_COMPLETE_GAMES',
                report='SAME_PAIRED_WDL_METRICS;DO_NOT_POOL_WITH_BAD_STATES'),
            exclusions=['NO_CONSTRUCTED_DEGRADATION_AS_TRANSFER_GATE','NO_SEARCH_ASSISTANCE',
                        'NO_HELDOUT_CHECKPOINT_SELECTION','NO_CASH_OR_LOSS_AS_RECOVERY_LABEL']),
        ablation=dict(arms=arms, controls=['SAME_PARENT_AND_OPTIMIZER_INITIALIZATION',
            'SAME_DIRECT_RL_CORPUS_AND_ORDINARY_PAIR_CORPUS','SAME_CURRICULUM_SOURCE_EXPOSURE_AND_DENOMINATOR',
            'SAME_UPDATE_COUNT_AND_FROZEN_SOURCE_FAMILY_SPLITS'],
            instruction='Build both plans with identical recovery_corpus and recovery_mix; only recovery_teacher_enabled differs.'),
        resources=dict(production='PINNED_RUNTIME_SLURM_STORAGE_RUN_REQUIRED', reviewer='CPU_FIXTURES_ONLY'),
        optimizer_updates_run=0, evaluation_games_run=0, strength_claim=False)


def _belief_curriculum_arm(registration: dict, run: dict) -> tuple[dict, dict]:
    """Recompute all gate receipts, including failed/no-confirmation opportunities."""
    from farm_neat.rollout import v0_regret_collect as C
    reg, report = C.read_belief_curriculum_run(registration, run)
    arm = reg['arms'][run['mode']]
    journal = C.Journal(Path(run['output']))
    archive = {}
    for line in Path(report['artifacts']['archive']).read_text().splitlines():
        row = R.validate_pair(R._loads(line))
        archive[row['content_sha256']] = row
    admissions, pairs = [], []
    if report['admissions']:
        frozen = journal.by_key.get('family', {}).get('PROTOCOL')
        _need(frozen is not None, 'CURRICULUM_MISSING_PROTOCOL')
        protocol = journal.get(frozen['object'])
        for receipt in report['admissions']:
            test_id = receipt['test_id']
            _need(test_id in arm['selected_state_ids'], 'CURRICULUM_UNSELECTED_ADMISSION')
            rows = [archive[h] for h in receipt['pair_content_sha256s']]
            bundle = dict(schema=AD.BUNDLE_SCHEMA, test_id=test_id, records=rows)
            result = AD.admit_pair(bundle, protocol=protocol)
            saved = journal.by_key.get(test_id, {}).get('ADMISSION')
            _need(saved is not None and journal.get(saved['object']) == result == receipt,
                  'CURRICULUM_ADMISSION_MISMATCH')
            admissions.append(result)
            pairs.extend((test_id, row) for row in rows)
        _need(len({r['test_id'] for r in admissions}) == len(admissions), 'CURRICULUM_DUPLICATE_ADMISSION')
    if report['paired_export']['status'] == 'READY':
        corpus, proof = C._paired_corpus(journal, arm['collection_plan'], pairs, admissions, report['reason_codes'])
        _need(corpus == R._loads(Path(report['artifacts']['paired_corpus']).read_text()) and
              proof == report['paired_export'], 'CURRICULUM_PAIRED_EXPORT_MISMATCH')
    counts = report['counts']
    simulator = counts['base_starts']+counts['replay_starts']+counts['search_discovery_starts']
    physical = counts['native_starts']
    _need(0 <= physical <= simulator <= reg['budget']['simulator_start_slots_per_arm'], 'CURRICULUM_SPEND_BOUNDS')
    selected = arm['selected_state_ids']
    positive = [a for a in admissions if a['actor_training_admissible'] and a['teacher_weight'] > 0]
    # Each admission is one selected single-seat opportunity, not one outcome,
    # replay, epoch, or independent world. Correlated families are not pooled
    # into a curriculum-level confidence interval.
    gain = math.fsum(a['mean_wdl_delta'] for a in positive)
    lcb = math.fsum(a['lower_bound'] for a in positive)
    strata = {}
    for stratum in C.BELIEF_CURRICULUM_STRATA:
        keys = [k for k in selected if reg['features'][k]['stratum'] == stratum]
        admitted = [a for a in positive if a['test_id'] in keys]
        strata[stratum] = dict(selected_opportunities=len(keys), admitted_corrections=len(admitted),
            correction_yield=len(admitted)/len(keys) if keys else None,
            confirmed_expected_wdl_delta_sum=math.fsum(a['mean_wdl_delta'] for a in admitted))
    summary = dict(selected_opportunities=len(selected), reported_admissions=len(admissions),
        missing_admissions=len(selected)-len(admissions), admitted_corrections=len(positive),
        correction_yield=len(positive)/len(selected), confirmed_expected_wdl_delta_sum=gain,
        confirmed_expected_wdl_lower_bound_sum=lcb,
        confirmed_expected_wdl_delta_per_physical_start=gain/physical if physical else None,
        confirmed_expected_wdl_delta_per_simulator_start=gain/simulator if simulator else None,
        confirmed_expected_wdl_lcb_per_physical_start=lcb/physical if physical else None,
        simulator_starts=simulator, physical_starts=physical,
        optimizer_updates_run=counts['optimizer_updates'], unused_simulator_start_slots=(
            reg['budget']['simulator_start_slots_per_arm']-simulator),
        coverage=copy.deepcopy(arm['coverage']), per_stratum=strata,
        disagreement_without_calibration=arm['coverage']['strata_selected']['DISAGREEMENT_UNCALIBRATED'],
        gate_status_counts={s: sum(a['status'] == s for a in admissions) for s in sorted({a['status'] for a in admissions})},
        collection_status=report['status'], paired_export_status=report['paired_export']['status'],
        interpretation='SELECTED_OPPORTUNITY_YIELD_NOT_POPULATION_UNBIASED_ESTIMATE_OR_POLICY_GAIN')
    return summary, report


def compare_belief_curricula(registration: dict, runs: dict) -> dict:
    """Compare actual costs/yields, never equate a budget cap with spent games.

    Zero native starts gives a NULL per-physical-start rate, not a simulated
    denominator. Equal zero optimizer work is not a training comparison. All
    negatives/missing opportunities stay in the yield/coverage denominator.
    This read-only collection comparison neither selects labels nor fits models.
    """
    from farm_neat.rollout import v0_regret_collect as C
    reg = C.validate_belief_curriculum(registration)
    _need(type(runs) is dict and set(runs) == set(C.BELIEF_CURRICULUM_MODES), 'CURRICULUM_BOTH_ARMS_REQUIRED')
    arms = {}
    for mode, run in runs.items():
        _need(run['mode'] == mode, 'CURRICULUM_RUN_MODE_MISMATCH')
        arms[mode], _ = _belief_curriculum_arm(reg, run)
    _need(Path(runs['uniform']['output']).resolve() != Path(runs['focused']['output']).resolve(),
          'CURRICULUM_REUSED_PHYSICAL_RUN')
    equal_sim = len({a['simulator_starts'] for a in arms.values()}) == 1
    equal_physical = len({a['physical_starts'] for a in arms.values()}) == 1
    complete = all(a['collection_status'] == 'COMPLETE' for a in arms.values())
    fixture = all(a['physical_starts'] == 0 for a in arms.values())
    status = ('INCOMPLETE_COLLECTION' if not complete else 'UNMATCHED_COLLECTION_SPEND'
        if not (equal_sim and equal_physical) else 'MATCHED_COLLECTION_FIXTURE' if fixture else
        'MATCHED_COLLECTION_NATIVE')
    return C.bind(dict(schema=C.BELIEF_CURRICULUM_SCHEMA+'-comparison', status=status,
        registration_sha256=reg['content_sha256'], run_sha256s={m:r['content_sha256'] for m,r in runs.items()},
        equal_simulator_spend=equal_sim, equal_physical_spend=equal_physical,
        registered_budget=copy.deepcopy(reg['budget']), arms=arms,
        optimizer_updates_run={m:a['optimizer_updates_run'] for m,a in arms.items()},
        training_comparison_status='NOT_RUN', strength_claim=False,
        ordinary_regression_status='NOT_MEASURED;COVERAGE_IS_NOT_PLAYING_STRENGTH',
        gain_semantics='SUM_OF_ADMITTED_PAIRED_TERMINAL_WDL_MEANS;NOT_FROZEN_FORECASTS_OR_POPULATION_GENERALIZATION',
        objective='TERMINAL_WDL', teacher_weight_rule='UNCHANGED_REGRET_ADMISSION_ONLY'))


def build_belief_curriculum_fit_plans(registration: dict, runs: dict, *, direct_plan: dict,
        direct_corpus: dict, reference_parent_sha256: str, mode: str,
        lambda_teacher: float, beta: float, device: str = 'cuda') -> dict:
    """Use unchanged fit plans/gates with shared direct data and equal update caps.

    This is a recipe, not training. Both arms share parent, RL buffer, coefficients,
    optimizer initialization and registered opportunity count; only the already
    prospectively selected teacher corpus differs. Failed/short-spend comparisons
    cannot silently proceed as matched experiments. Genuine direct draws are not
    replaced by focused teacher states. The ordinary trainer remains the executor.
    """
    from farm_neat.rollout import v0_regret_collect as C
    reg = C.validate_belief_curriculum(registration)
    _need(reference_parent_sha256 == reg['collection_plan']['spec']['parent_checkpoint_sha256'],
          'CURRICULUM_DIRECT_PARENT_MISMATCH')
    comparison = compare_belief_curricula(reg, runs)
    _need(comparison['status'] in ('MATCHED_COLLECTION_NATIVE', 'MATCHED_COLLECTION_FIXTURE'),
          'CURRICULUM_MATCHED_COLLECTION_REQUIRED')
    plans = {}
    for arm in C.BELIEF_CURRICULUM_MODES:
        _, report = _belief_curriculum_arm(reg, runs[arm])
        _need(report['paired_export']['status'] == 'READY', 'CURRICULUM_FIT_CORPUS_UNAVAILABLE')
        plans[arm] = build_plan(direct_plan=direct_plan, direct_corpus=direct_corpus,
            paired_corpus=dict(path=report['artifacts']['paired_corpus'], sha256=report['artifact_sha256s']['paired_corpus']),
            mode=mode, lambda_teacher=lambda_teacher, beta=beta, device=device,
            updates=reg['budget']['optimizer_updates_per_arm'])
        _need(plans[arm]['reference_parent']['commit_sha256'] == reference_parent_sha256,
              'CURRICULUM_DIRECT_PARENT_MISMATCH')
        _, binding = _paired_data(plans[arm])
        _need(binding['opportunity_count'] == reg['budget']['opportunity_slots_per_arm'],
              'CURRICULUM_FIT_DENOMINATOR_MISMATCH')
    return C.bind(dict(schema=C.BELIEF_CURRICULUM_SCHEMA+'-fit-recipe', status='RECIPE_NOT_EXECUTED',
        registration_sha256=reg['content_sha256'], comparison_sha256=comparison['content_sha256'],
        plans=plans, optimizer_updates_run=0, native_games_run=0, strength_claim=False,
        controls=['SAME_DIRECT_RL_CORPUS_PARENT_COEFFICIENTS_AND_OPTIMIZER_INITIALIZATION',
                  'SAME_REGISTERED_OPPORTUNITY_DENOMINATOR_INCLUDING_NEGATIVES',
                  'SAME_OPTIMIZER_UPDATE_CAP;RECORD_ACTUAL_EXECUTED_UPDATES_SEPARATELY'],
        evaluation_gate='FRESH_SEARCH_DISABLED_OWN_PREFIX_WDL_AND_ORDINARY_STATE_REGRESSION;NOT_YET_RUN'))


# Opt-in accepted-loss join on the existing transaction, not another trainer.
BELIEF_ARM_SCHEMA = 'farm-v0-belief-production-join-v1'
BELIEF_FEATURE_SCHEMA = 'PUBLIC_PRIOR_EVIDENCE_AND_AVAILABLE_FORECAST_SUMMARY_7_V1'


def with_belief_production(plan: dict, *, population: dict, options: dict) -> dict:
    """Pin a full fixed population and explicit accepted-loss options; zero fitting."""
    result = copy.deepcopy(plan)
    result['belief_production'] = dict(schema=BELIEF_ARM_SCHEMA,
        population=_ref(population), options=_ref(options))
    result = seal_plan(result)
    rl_plan = _validate_plan(result)
    _belief_data(result, rl_plan)  # Refuse bad provenance/labels before model construction.
    return result


def belief_actor_features(actor: dict) -> dict:
    """Closed PUBLIC subtree only. Labels/identities are never numeric features.

    Seven descriptors: raw prior, initial mass, bounded relative public log
    evidence, mean available LOSS/DRAW/WIN forecast, and forecast coverage.
    Forecasts are public actor-time predictions, NOT the outcome tensor.
    Missing forecasts use zeros plus separate coverage. Hashes align outputs
    only. Identical public descriptors deliberately cannot distinguish models.
    """
    from farm_neat.training import v0_belief_records as B
    B.actor_payload_key(actor)
    hypotheses = actor['belief']['hypotheses']
    peak = max(h['log_evidence'] for h in hypotheses)
    features = []
    for h in hypotheses:
        forecasts = [v for v in h['conditional_wdl'].values() if v is not None]
        mean = [math.fsum(v[i] for v in forecasts)/len(forecasts) if forecasts else 0. for i in range(3)]
        relative = h['log_evidence'] - peak
        relative = -1. if not math.isfinite(relative) else relative / (1. + abs(relative))
        features.append([h['raw_prior'], h['initial_mass'], relative, *mean,
                         len(forecasts)/len(actor['candidates'])])
    return dict(schema=BELIEF_FEATURE_SCHEMA,
        hypothesis_ids=[h['hypothesis_id'] for h in hypotheses],
        kinds=[h['kind'] for h in hypotheses], hypothesis_features=features)


def _belief_options(ref: dict, plan: dict, population: dict) -> dict:
    import json
    options = _read(ref)
    baseline = json.loads((ROOT/'project/configs/v0-belief.json').read_text())
    _need(type(options) is dict and set(options) == set(baseline), 'BELIEF_OPTIONS_FIELDS')
    mutable = {'minimum_mass', 'decision_credit', 'proper_scores'}
    _need(all(options[k] == baseline[k] for k in baseline if k not in mutable),
          'BELIEF_OBJECTIVE_OR_OPTIONS_DRIFT')
    allowed = {'minimum_mass': {'method','k','y','coefficient'},
               'decision_credit': {'weighting','teacher_mode','teacher_beta'},
               'proper_scores': {'belief_mass_coefficient','conditional_outcome_coefficient'}}
    for group, fields in allowed.items():
        _need(type(options[group]) is dict and set(options[group]) == set(baseline[group]),
              'BELIEF_OPTIONS_FIELDS: ' + group)
        _need(all(options[group][k] == baseline[group][k] for k in baseline[group] if k not in fields),
              'BELIEF_UNACCEPTED_OBJECTIVE: ' + group)
    proper, floor, credit = (options[k] for k in ('proper_scores','minimum_mass','decision_credit'))
    for value in (proper['belief_mass_coefficient'],proper['conditional_outcome_coefficient'],floor['coefficient']):
        _need(type(value) in (int,float) and math.isfinite(value) and value >= 0, 'BELIEF_FINITE_COEFFICIENT')
    _need(credit['weighting'] in ('none','binary_admitted','confidence','distribution'), 'BELIEF_CREDIT_WEIGHTING')
    loss_mode = 'gated_bc' if plan['mode'] == MODES[1] else 'gated_pair'
    _need(credit['teacher_mode'] == loss_mode and credit['teacher_beta'] == plan['beta'],
          'BELIEF_TEACHER_OBJECTIVE_JOIN')
    _need(floor['method'] in ('none','soft_floor') and type(floor['y']) in (int,float)
          and math.isfinite(floor['y']) and floor['y'] >= 0, 'BELIEF_MINIMUM_MASS_OPTIONS')
    if floor['method'] == 'none':
        _need(floor['y'] == 0 and floor['coefficient'] == 0 and floor['k'] is None,
              'BELIEF_DISABLED_FLOOR_MUST_BE_ZERO')
    else:
        k, y = floor['k'], float(floor['y'])
        _need(type(k) is int and all(len(r['record']['actor']['belief']['hypotheses']) == k
              for r in population['rows']), 'BELIEF_FULL_CATALOG_K')
        num, den = y.as_integer_ratio()
        _need(den-k*num > 0 and (den-k*num)/den >= 1.e-6, 'BELIEF_MINIMUM_MASS_SLACK')
    return options


def _belief_data(plan: dict, rl_plan: dict) -> tuple[dict, dict] | None:
    if 'belief_production' not in plan:
        return None
    from farm_neat.rollout import v0_regret_collect as C
    extension = plan['belief_production']
    population = C.validate_belief_production_population(_read(extension['population']),
        production=not rl_plan['behavior_model_identity']['diagnostic'])
    _need(population['parent_checkpoint_sha256'] == plan['reference_parent']['commit_sha256'],
          'BELIEF_REFERENCE_PARENT_JOIN')
    heldout = {c['world_sha256'] for c in rl_plan['evaluation_cases']}
    _need(heldout <= set(population['protected_groups']['world_sha256s']),
          'BELIEF_RL_EVALUATION_PROTECTION_REQUIRED')
    return population, _belief_options(extension['options'], plan, population)


def _belief_credit_map(population: dict, options: dict, active_pairs: list) -> dict:
    """Join exact admitted paths; a candidate digest alone is not enough."""
    if options['decision_credit']['weighting'] == 'none':
        return {}
    from farm_neat.training.v0_belief_credit import DecisionCreditRecord
    available = {}
    for row in population['rows']:
        private = row['record']['trainer_only']
        credit = private['decision_credit']
        if credit is None:
            continue
        pair = R._loads(credit['inputs_wire'])['pair']
        key = R.canonical_sha256(pair)
        _need(key not in available, 'BELIEF_DUPLICATE_CREDIT_PAIR')
        available[key] = dict(record=DecisionCreditRecord(credit,
            expected_actor_sha256=row['actor_payload_sha256']), actor=row['actor_payload_sha256'],
            candidates=(credit['learner_candidate_sha256'],credit['challenger_candidate_sha256']),
            confidence=private['confidence'], confidence_source=private['confidence_source_sha256'],
            base_weight=credit['admission']['teacher_weight'])
    joined = {}
    for pair in active_pairs:
        _need(pair.get('source_kind','ORDINARY') == 'ORDINARY', 'BELIEF_RECOVERY_CREDIT_NOT_BOUND')
        key = R.canonical_sha256(pair['record'])
        _need(key in available, 'BELIEF_ADMITTED_PAIR_CREDIT_MISSING')
        value = available[key]
        _need(math.isclose(value['base_weight'],max(0.,pair['lower_bound'])/2.,rel_tol=0.,abs_tol=1.e-12),
              'BELIEF_ADMISSION_WEIGHT_JOIN')
        joined[_teacher_id(pair)] = value
    return joined


def _belief_credit_term(trainer, pair, reference, winner, learner):
    """One accepted credit-loss call for both readiness and actual teaching."""
    value = trainer.belief['credit'].get(_teacher_id(pair)) if trainer.belief else None
    if value is None:
        return None
    from farm_neat.training.v0_belief_loss import decision_credit_loss
    ref_w, ref_l = (winner.new_tensor([v]) for v in reference)
    return decision_credit_loss(winner, learner, ref_w, ref_l,
        records=[value['record']], expected_actor_sha256s=[value['actor']],
        candidate_pairs=[value['candidates']], confidence=winner.new_tensor([value['confidence']]),
        confidence_source_sha256=value['confidence_source'],
        weighting=trainer.belief['options']['decision_credit']['weighting'],
        mode='gated_bc' if trainer.fit_plan['mode'] == MODES[1] else 'gated_pair',
        beta=trainer.fit_plan['beta'], opportunity_count=trainer.paired_binding['opportunity_count'])


def _belief_primary_signal(trainer) -> None:
    """Credit can remove all admitted teaching; auxiliaries cannot replace it.

    Use the accepted loss owner's detached weights, not a second eligibility
    formula. The cached reference class-path logps are float64 (as in actual
    teaching); this dry calculation performs no actor forward, RNG draw or
    gradient. Positive weights establish support, not a nonzero gradient or
    playing strength. The admission-only control is left untouched.
    """
    if not trainer.teacher_active or not trainer.belief['credit']:
        return
    weights, base_weights = [], []
    with torch.no_grad():
        for pair, reference in zip(trainer.active_pairs, trainer.reference_logps):
            probe = torch.tensor(reference, dtype=torch.float64, device=trainer.device)
            term = _belief_credit_term(trainer, pair, reference, probe[:1], probe[1:])
            _need(term is not None, 'BELIEF_ADMITTED_PAIR_CREDIT_MISSING')
            weights.extend(term.weights.cpu().tolist())
            base_weights.extend(term.base_weights.cpu().tolist())
    positive = sum(w > 0 for w in weights)
    if not positive:
        trainer.teacher_active = False
        trainer.signal = RL._signal_decision(trainer.data)
        trainer.signal['direct_signal'] = trainer.signal['status']
        if trainer.signal['status'] != 'READY':
            trainer.signal['status'] = 'STOP_ZERO_EFFECTIVE_TEACHER_WEIGHT'
    trainer.signal['belief_credit'] = dict(
        weighting=trainer.belief['options']['decision_credit']['weighting'],
        admitted_rows=len(weights), positive_weight_rows=positive,
        effective_weight_sum=math.fsum(weights), base_weight_sum=math.fsum(base_weights),
        weight_semantics='ACCEPTED_CREDIT_WEIGHTS_BEFORE_LAMBDA_AND_OPPORTUNITY_REDUCTION')


def _belief_initialize(trainer, data) -> None:
    trainer.belief = None
    if data is None:
        return
    from farm_neat.policy import v0_belief as H
    population, options = data
    proper, floor = options['proper_scores'], options['minimum_mass']
    mass_weight, conditional_weight = proper['belief_mass_coefficient'], proper['conditional_outcome_coefficient']
    enabled = bool(mass_weight or conditional_weight or floor['coefficient'])
    trainer.belief = dict(population=population, options=options, enabled=enabled,
        credit=_belief_credit_map(population, options, trainer.active_pairs), rows=[],
        frozen_sha256=RL.digest(dict(population=population,options=options)))
    _belief_primary_signal(trainer)
    if not enabled:  # No new parameters, optimizer slots or RNG draws in the control.
        return
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(trainer.plan['seed'])
        head = H.V0BeliefHead(trainer.model.config.width,7,candidate_dim=128)
    trainer.model.add_module('belief_aux',head.to(trainer.device))
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(),lr=RL.PROTOCOL['learning_rate'],
        betas=(.9,.999),eps=1.e-8,weight_decay=0.,foreach=False)
    trainer.parameter_order = [name for name,_ in trainer.model.named_parameters()]
    for row in population['rows']:
        actor, private = row['record']['actor'], row['record']['trainer_only']
        features = belief_actor_features(actor)
        candidates = H.encode_conditional_candidates(actor['context'],actor['candidates'])
        target = H.conditional_targets_from_tensor(private['outcome_tensor'],
            expected_actor_sha256=row['actor_payload_sha256'],hypothesis_ids=features['hypothesis_ids'],
            candidate_ids=candidates['candidate_ids'],device=trainer.device)['targets']
        trainer.belief['rows'].append(dict(actor=actor,private=private,features=features,
            candidates=candidates,targets=target))

    trainer.belief['cache_sha256']=_belief_cache_key(trainer.belief['rows'])


def _belief_cache_key(rows: list) -> str:
    # Missing labels retain their mask; -1 is hash-only, never a training label.
    return RL.digest([dict(features=r['features'],candidates=r['candidates'],
        hypothesis_ids=list(r['targets'].hypothesis_ids),candidate_ids=list(r['targets'].candidate_ids),
        probabilities=r['targets'].probabilities.detach().cpu().nan_to_num(nan=-1.).tolist(),
        observed=r['targets'].observed_mask.detach().cpu().tolist()) for r in rows])


def _belief_backward(trainer) -> dict | None:
    if trainer.belief is None:
        return None
    from farm_neat.training import v0_belief_loss as L
    b = trainer.belief
    proper, floor = b['options']['proper_scores'], b['options']['minimum_mass']
    weights = dict(mass=proper['belief_mass_coefficient'],conditional=proper['conditional_outcome_coefficient'],
                   minimum_mass=floor['coefficient'])
    metric = dict(enabled=b['enabled'],records_seen=b['population']['record_count'],
        distinct_worlds=b['population']['independent_world_count'],
        reduction='MEAN_OVER_ALL_FROZEN_RECORDS;WITHIN_RECORD_MEAN_OBSERVED_CONDITIONALS',
        raw_calibration_mass='posterior_mass',weights=weights,mass_loss=0.,conditional_loss=0.,
        minimum_mass_loss=0.,weighted_loss=0.,observed_mass_rows=0,observed_conditionals=0,
        unknown_missing_conditionals=0,missing_conditionals=0,credit_weighting=b['options']['decision_credit']['weighting'])
    if not b['enabled']:
        return metric
    n = len(b['rows']); head=trainer.model.belief_aux
    for row in b['rows']:
        actor, private, features = row['actor'],row['private'],row['features']
        state, prefix, actor_options = R._actor(actor['context'])
        pooled,_ = trainer.model._encode([state],[prefix],[{'hold_releases':actor_options['hold_releases']}])
        h = pooled.new_tensor(features['hypothesis_features'])
        kwargs = dict(hypothesis_ids=features['hypothesis_ids'],kinds=features['kinds'])
        mass = head(pooled[0],h,**kwargs)
        conditional = head.conditional_value(pooled[0],h,pooled.new_tensor(row['candidates']['features']),
            candidate_ids=row['candidates']['candidate_ids'],**kwargs)
        target = private['mass_target']
        probabilities = (target['probabilities'] if target['observed'] else
                         [float('nan')]*len(target['hypothesis_ids']))
        ml = L.belief_mass_loss(mass,torch.tensor(probabilities,dtype=torch.float64,device=trainer.device),
            torch.tensor(target['observed'],dtype=torch.bool,device=trainer.device),
            hypothesis_ids=target['hypothesis_ids'],mass_used='posterior_mass')
        cl = L.conditional_wdl_loss(conditional,row['targets'])
        fl = L.minimum_mass_loss(mass,k=len(features['hypothesis_ids']),y=floor['y'],method=floor['method'])
        total = weights['mass']*ml.loss + weights['conditional']*cl.loss + weights['minimum_mass']*fl.loss
        (total/n).backward()
        for name,term in (('mass_loss',ml),('conditional_loss',cl),('minimum_mass_loss',fl)):
            metric[name] += float(term.loss.detach())/n
        metric['weighted_loss'] += float(total.detach())/n
        metric['observed_mass_rows'] += ml.observed_rows
        metric['observed_conditionals'] += cl.observed_rows
        missing = ~row['targets'].observed_mask
        unknown = torch.tensor([kind=='UNKNOWN' for kind in features['kinds']],device=trainer.device)
        metric['missing_conditionals'] += int(missing.sum())
        metric['unknown_missing_conditionals'] += int((missing & unknown[:,None]).sum())
    return metric


def belief_preflight(plan: dict, *, production: bool = True) -> dict:
    """Data-first refusal; READY needs the pinned production runtime and inputs.

    Construct the SAME trainer transaction to check strict parent/reference/path
    joins and live hashes. No update, collection, game, submission or promotion.
    CPU fixture preflight explicitly cannot authorize the production CLI.
    """
    rl_plan = _validate_plan(plan)
    _need('belief_production' in plan,'BELIEF_ARM_REQUIRED')
    _belief_data(plan,rl_plan)
    _capability(rl_plan,plan['execution']['device'],production_cli=production)
    # Use the identical fresh/resume transaction as run_fit. A valid parent is
    # not evidence that a requested child exists, has intact optimizer/RNG state,
    # or has enough of the registered update budget left for this segment.
    execution = plan['execution']
    resume = execution['resume']
    trainer = (Trainer(plan) if resume is None else
               Trainer.resume(plan, ROOT / resume['path'], resume['sha256']))
    trainer._live_join()
    _need(trainer.signal['status'] == 'READY',
          'BELIEF_TRAINING_SIGNAL_REQUIRED: ' + trainer.signal['status'])
    requested = execution['updates']
    remaining = RL.PROTOCOL['optimizer_updates'] - trainer.global_update
    _need(1 <= requested <= remaining,
          f'BELIEF_SEGMENT_EXCEEDS_REMAINING_UPDATE_BUDGET: requested={requested} remaining={remaining}')
    return dict(schema=BELIEF_ARM_SCHEMA+'-preflight',
        status='READY_TO_LAUNCH' if production else 'DIAGNOSTIC_READY_NOT_PRODUCTION',
        plan_sha256=RL.digest(plan),inputs=copy.deepcopy(plan['belief_production']),
        parent=copy.deepcopy(plan['reference_parent']),model_identity=rl_plan['behavior_model_identity'],
        parameter_count=sum(p.numel() for p in trainer.model.parameters()),
        record_count=trainer.belief['population']['record_count'],
        protected_groups=trainer.belief['population']['protected_groups'],
        execution=copy.deepcopy(execution), starting_global_update=trainer.global_update,
        remaining_update_budget=remaining, signal=copy.deepcopy(trainer.signal),
        expected_exposure=dict(direct_games=requested*len(trainer.data.games),
            pair_opportunities=requested*trainer.paired_binding['opportunity_count'],
            belief_records=requested*trainer.belief['population']['record_count']),
        exposure_semantics='FIXED_BUFFER_REUSES_NOT_FRESH_GAMES',
        optimizer_updates=0,native_games=0,strength_claim=False)


if __name__ == '__main__':
    sys.exit(main())
