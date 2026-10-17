"""Prospective V0 regret comparison registration and terminal-WDL reduction.

register_comparison(plan: dict) -> detached, hash-bound registration.
summarize_comparison(plan: dict, records: list[dict]) -> hash-bound assessment.
jobs_for(registered_plan) -> deterministic, trainer-only execution slots.

The default register/jobs/summarize paths run NO games and load NO models.
The explicit plan-run/check-run/run bridge cold-loads committed scorers and uses
v0_policy for original-rule games only with --allow-native. It never trains,
selects or promotes a model.
It consumes complete-game records from an original-rule maintainer runner using
v0_policy.PolicyAgent/ScorerPolicy own-prefix execution and router_panel's frozen
actor, fresh-instance, both-seat and terminal cash/WDL contracts. Existing fixed
scenario/Router panels are explicitly blacklisted, not silently made held-out.

Primary estimand: change in P(win)+.5 P(draw). Average samplers, seats and worlds
within source group, then average training replications WITHIN the same group.
Only distinct source groups enter n. CIs use a two-sided bounded-mean Hoeffding
bound with Bonferroni allocation across every reported primary/opponent/WDL CI.
No bootstrap degeneracy can turn two identical wins into certainty. Independence
between declared groups remains an experimental assumption, not a hash property.

A registration freezes sample size, opponents/weights/tolerances, sources,
checkpoints, costs and final candidate. A final registration joins one previously
monitored candidate and compute-matched control from >=3 development training
seeds, with a disjoint new panel. Records must cite that exact registration.
The exclusive CLI prevents accidental overwrites, not dishonest backdating or
forging a new plan AND all records. Preserve registration externally before play;
provide a complete exposure ledger and authenticated runtime/rollout receipts.

Synthetic records exercise rule_decision but ALWAYS yield decision=HOLD. GO is
an evidence assessment for maintainers, never promotion/submission authority.
Incomplete panels have descriptive observed counts/means but NO inferential CI.
Cash only validates terminal WDL; it cannot improve a tied/lost comparison.

CLI (no native execution):
  python -m farm_neat.evaluation.v0_regret_evaluate register --plan PLAN.json --out NEW.json
  python -m farm_neat.evaluation.v0_regret_evaluate jobs --plan NEW.json --out JOBS.json
  python -m farm_neat.evaluation.v0_regret_evaluate summarize --plan NEW.json --records ROWS.json --out SUMMARY.json

Wire examples and maintainer integration are in the task's owned return evidence.
Checkpoint_files is an optional checkpoint SHA -> file path map; supplied bytes
are checked at registration AND reduction, without deserialization. A native GO
requires all checkpoint bytes and runtime/restart receipts. Use the pinned
maintainer interpreter/storage/SLURM contract for such real-asset work.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
from typing import Any

from farm_neat.evaluation import router_panel, v0_policy
from farm_neat.evaluation.common import bound, digest, file_sha, read, write_new
from farm_neat.training.v0_regret_records import source_pins as record_source_pins

ROOT = Path(__file__).resolve().parents[3]
PLAN_SCHEMA = 'farm-v0-regret-comparison-plan-v1'
RECORD_SCHEMA = 'farm-v0-regret-comparison-game-v1'
SUMMARY_SCHEMA = 'farm-v0-regret-comparison-summary-v1'
EXECUTION = {
    'rules': 'original', 'actor_inputs': 'OWN_VISIBLE_State, OWN_CURRENT_Plan, OWN_PREFIX_Masks',
    'teacher_prefixes': False, 'publication': 'opening_only_later_keep',
    'max_nonstop_edits': 16, 'labor_factor': 4.0, 'hands_mode': 'daily',
    'flow_belief': 'explicit_zero_flow', 'decode_mode': 'greedy_class',
    'stop': 'sampled_STOP_or_administrative_edit_cap',
}
SPLITS = ('discovery', 'confirmation', 'training', 'monitor', 'final')
COST_FIELDS = ('cpu_seconds', 'gpu_seconds', 'end_to_end_seconds', 'search_cpu_seconds',
               'search_gpu_seconds', 'extra_rollout_cpu_seconds', 'extra_rollout_gpu_seconds')
KINDS = {'frozen_parent', 'direct_rl', 'gated_bc', 'gated_pairwise', 'direct_rl_compute', 'random_challenger'}
TEACHERS = {'gated_bc', 'gated_pairwise', 'random_challenger'}
SCORE = {'WIN': 1.0, 'DRAW': .5, 'LOSS': 0.0}
STATUSES = ('COMPLETE', 'FAILED', 'CENSORED', 'UNSUPPORTED')
JOB_FIELDS = ('arm', 'training_seed', 'world', 'seat', 'opponent', 'sampler_seed')
MAX_JOBS = 1_000_000


class ComparisonError(ValueError):
    def __init__(self, code: str, location: str):
        self.code, self.location = code, location
        super().__init__(f'{code}: {location}')


def _need(ok, code, at):
    if not ok:
        raise ComparisonError(code, at)


def _safe(value: Any, depth=0):
    _need(depth <= 64, 'JSON_LIMIT', '$')
    if value is None or type(value) in (str, bool):
        return
    if type(value) is int:
        _need(value.bit_length() <= 4096, 'JSON_LIMIT', '$')
    elif type(value) is float:
        _need(math.isfinite(value), 'NONFINITE', '$')
    elif type(value) is list:
        for item in value:
            _safe(item, depth+1)
    elif type(value) is dict:
        _need(all(type(k) is str for k in value), 'INVALID_JSON', '$')
        for item in value.values():
            _safe(item, depth+1)
    else:
        raise ComparisonError('INVALID_JSON', '$')


def _fields(value, names, at, optional=()):
    _need(type(value) is dict and set(names) <= set(value)
          and set(value) <= set(names) | set(optional), 'FIELDS', at)


def _text(value, at):
    _need(type(value) is str and bool(value.strip()), 'STRING', at)


def _hash(value, at, nullable=False):
    if nullable and value is None:
        return
    _need(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'HASH', at)


def _integer(value, at, lo=0):
    _need(type(value) is int and lo <= value < 2**63, 'INTEGER', at)


def _number(value, at, lo=0, hi=math.inf):
    _need(type(value) in (int, float), 'NUMBER', at)
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    _need(finite and lo <= value <= hi, 'NUMBER', at)


def _list(value, at, nonempty=False):
    _need(type(value) is list and len(value) <= MAX_JOBS and (value or not nonempty), 'LIST', at)


def _unique(values, at):
    _need(len(values) == len(set(values)), 'DUPLICATE', at)


def _same(left, right, code, at):
    _need(digest(left) == digest(right), code, at)


def _costs(value, at):
    _fields(value, COST_FIELDS, at)
    for key in COST_FIELDS:
        _number(value[key], at+'.'+key)
    for unit in ('cpu', 'gpu'):
        _need(value[f'search_{unit}_seconds'] + value[f'extra_rollout_{unit}_seconds']
              <= value[f'{unit}_seconds'], 'COMPUTE_ACCOUNTING', at)


def source_pins() -> dict:
    """Reuse the existing V0/record closure and freeze this reducer/Router protocol."""
    paths = ('src/farm_neat/evaluation/router_panel.py',
             'src/farm_neat/evaluation/v0_regret_evaluate.py', 'docs/contracts/evaluation.md',
             'project/runtime/interpreter.toml', 'src/farm_neat/training/v0_rl.py')
    return {**v0_policy.source_pins(), **record_source_pins(), **{p: file_sha(ROOT/p) for p in paths}}


def _replications(arm):
    return {r['training_seed']: r for r in arm['replications']}


def _compute(costs, model, prefix=''):
    return costs[prefix+'cpu_seconds'] + model['gpu_second_weight']*costs[prefix+'gpu_seconds']


def _validate_rep(rep, arm, p):
    _fields(rep, ('training_seed', 'checkpoint_sha256', 'parent_sha256', 'direct_data_order_sha256',
                  'optimizer_sha256', 'updates', 'direct_rollout_ids', 'non_direct_execution_ids',
                  'rollout_receipt_sha256', 'runtime_receipt_sha256', 'restart_receipt_sha256',
                  'admitted_pairs', 'costs'), arm['name'])
    for key in ('training_seed', 'updates', 'admitted_pairs'):
        _integer(rep[key], key)
    for key in ('checkpoint_sha256', 'parent_sha256', 'direct_data_order_sha256', 'optimizer_sha256',
                'rollout_receipt_sha256', 'runtime_receipt_sha256', 'restart_receipt_sha256'):
        _hash(rep[key], key, nullable=key in ('runtime_receipt_sha256', 'restart_receipt_sha256'))
    _same(rep['parent_sha256'], p['parent_sha256'], 'PARENT_DRIFT', arm['name'])
    for key in ('direct_rollout_ids', 'non_direct_execution_ids'):
        _list(rep[key], key)
        for item in rep[key]:
            _text(item, key)
        _unique(rep[key], key)
    _need(not set(rep['direct_rollout_ids']) & set(rep['non_direct_execution_ids']),
          'NON_GENUINE_ROLLOUT', arm['name'])
    _costs(rep['costs'], arm['name'])
    if arm['kind'] == 'frozen_parent':
        _need(rep['checkpoint_sha256'] == p['parent_sha256'] and rep['updates'] == 0
              and not rep['direct_rollout_ids'] and not rep['non_direct_execution_ids']
              and rep['admitted_pairs'] == 0 and not any(rep['costs'].values()), 'FROZEN_PARENT', arm['name'])
    else:
        _need(rep['updates'] > 0 and bool(rep['direct_rollout_ids']), 'TRAINING_REPLICATION', arm['name'])
    if arm['kind'] in ('direct_rl', 'direct_rl_compute'):
        _need(rep['admitted_pairs'] == 0 and _compute(rep['costs'], p['compute_match'], 'search_') == 0,
              'DIRECT_RL_CONTROL', arm['name'])


def _match_compute(candidate, control, p):
    model = p['compute_match']; tolerance = model['relative_tolerance']
    for seed, cand in _replications(candidate).items():
        ctrl = _replications(control)[seed]
        _same(cand['optimizer_sha256'], ctrl['optimizer_sha256'], 'COMPUTE_OPTIMIZER', str(seed))
        _need(set(cand['direct_rollout_ids']) <= set(ctrl['direct_rollout_ids']), 'COMPUTE_BASE_ROLLOUTS', str(seed))
        extra = set(ctrl['direct_rollout_ids']) - set(cand['direct_rollout_ids'])
        search = _compute(cand['costs'], model, 'search_')
        required = math.ceil(search/model['rollout_compute_seconds'])
        _need(len(extra) >= required, 'COMPUTE_EXTRA_ROLLOUTS', str(seed))
        _need(_compute(ctrl['costs'], model, 'extra_rollout_') >= search*(1-tolerance),
              'COMPUTE_EXTRA_ROLLOUT_COST', str(seed))
        total, baseline = _compute(cand['costs'], model), _compute(ctrl['costs'], model)
        _need(abs(total-baseline) <= tolerance*max(total, baseline), 'COMPUTE_TOTAL_MISMATCH', str(seed))


def _validate_plan(p):
    _safe(p)
    _fields(p, ('schema', 'id', 'stage', 'evidence_kind', 'parent_sha256', 'source_pins',
                'exposure_receipt_sha256', 'exposures', 'worlds', 'sample_size_groups', 'seats',
                'sampler_seeds', 'execution', 'arms', 'opponents', 'decision_rule', 'compute_match',
                'checkpoint_files', 'development_summary'), '$', ('content_sha256',))
    _need(p['schema'] == PLAN_SCHEMA, 'SCHEMA', '$')
    _text(p['id'], 'id')
    _need(p['stage'] in ('development', 'final'), 'STAGE', '$')
    _need(p['evidence_kind'] in ('SYNTHETIC_DIAGNOSTIC', 'NATIVE_GAME'), 'EVIDENCE_KIND', '$')
    _hash(p['parent_sha256'], 'parent_sha256'); _hash(p['exposure_receipt_sha256'], 'exposure_receipt_sha256')
    _same(p['source_pins'], source_pins(), 'SOURCE_DRIFT', 'source_pins')
    execution = copy.deepcopy(EXECUTION)
    _need(type(p['execution']) is dict and p['execution'].get('decode_mode') in v0_policy.MODES,
          'EXECUTION_CONTRACT', 'decode_mode')
    execution['decode_mode'] = p['execution']['decode_mode']
    _same(p['execution'], execution, 'EXECUTION_CONTRACT', 'execution')
    _same(p['seats'], [0, 1], 'BOTH_SEATS_REQUIRED', 'seats')
    _list(p['sampler_seeds'], 'sampler_seeds', True)
    for seed in p['sampler_seeds']:
        _integer(seed, 'sampler_seed')
    _unique(p['sampler_seeds'], 'sampler_seeds')
    if execution['decode_mode'] == 'greedy_class':
        _same(p['sampler_seeds'], [0], 'GREEDY_DUPLICATION', 'sampler_seeds')
    model = p['compute_match']
    _fields(model, ('gpu_second_weight', 'rollout_compute_seconds', 'profile_receipt_sha256', 'relative_tolerance'), 'compute_match')
    _number(model['gpu_second_weight'], 'gpu_second_weight', 1e-12)
    _number(model['rollout_compute_seconds'], 'rollout_compute_seconds', 1e-12)
    _number(model['relative_tolerance'], 'relative_tolerance', 0, .2)
    _hash(model['profile_receipt_sha256'], 'profile_receipt_sha256')
    rule = p['decision_rule']
    _fields(rule, ('family_alpha', 'min_effect_score', 'min_world_groups', 'max_training_compute',
                  'max_eval_seconds_per_game'), 'decision_rule')
    _number(rule['family_alpha'], 'family_alpha', 1e-12, .1)
    _number(rule['min_effect_score'], 'min_effect_score', 0, 1)
    _integer(rule['min_world_groups'], 'min_world_groups', 2)
    for key in ('max_training_compute', 'max_eval_seconds_per_game'):
        _number(rule[key], key, 1e-12)
    _list(p['opponents'], 'opponents', True)
    for opp in p['opponents']:
        _fields(opp, ('id', 'identity_sha256', 'weight', 'regression_tolerance'), 'opponent')
        _text(opp['id'], 'opponent.id'); _hash(opp['identity_sha256'], 'opponent.identity_sha256')
        _number(opp['weight'], 'opponent.weight', 1e-12, 1)
        _number(opp['regression_tolerance'], 'regression_tolerance', 0, 1)
    _unique([o['id'] for o in p['opponents']], 'opponents')
    _need(abs(math.fsum(o['weight'] for o in p['opponents'])-1) < 1e-12, 'OPPONENT_WEIGHTS', '$')
    _list(p['arms'], 'arms', True)
    for arm in p['arms']:
        _fields(arm, ('name', 'kind', 'actor_mode', 'assisted_from', 'replications'), 'arm')
        _text(arm['name'], 'arm.name')
        _text(arm['kind'], 'arm.kind'); _text(arm['actor_mode'], 'arm.actor_mode')
        _need(arm['kind'] in KINDS and arm['actor_mode'] in ('direct', 'search_assisted'), 'ARM_KIND', arm['name'])
        _list(arm['replications'], 'replications', True)
        for rep in arm['replications']:
            _validate_rep(rep, arm, p)
        _unique([r['training_seed'] for r in arm['replications']], 'training_seeds')
        if arm['kind'] != 'frozen_parent':
            _need(len({r['checkpoint_sha256'] for r in arm['replications']}) == len(arm['replications']),
                  'TRAINING_REPLICATION_CLONE', arm['name'])
    _unique([a['name'] for a in p['arms']], 'arms')
    direct_ids = {i for a in p['arms'] for r in a['replications'] for i in r['direct_rollout_ids']}
    nondirect_ids = {i for a in p['arms'] for r in a['replications'] for i in r['non_direct_execution_ids']}
    _need(not direct_ids & nondirect_ids, 'NON_GENUINE_ROLLOUT', 'cross_arm_execution_ids')
    seeds = sorted(_replications(p['arms'][0]))
    for arm in p['arms']:
        _same(sorted(_replications(arm)), seeds, 'TRAINING_SEED_PAIRING', arm['name'])
    direct = [a for a in p['arms'] if a['actor_mode'] == 'direct']
    by_kind = {a['kind']: a for a in direct}
    _unique([a['kind'] for a in direct], 'direct_arm_kinds')
    by_name = {a['name']: a for a in direct}
    for arm in p['arms']:
        if arm['actor_mode'] == 'direct':
            _need(arm['assisted_from'] is None, 'SEARCH_ASSISTED_BINDING', arm['name'])
        else:
            _text(arm['assisted_from'], 'assisted_from')
            _need(arm['assisted_from'] in by_name, 'SEARCH_ASSISTED_BINDING', arm['name'])
            base = by_name[arm['assisted_from']]
            _same(arm['replications'], base['replications'], 'SEARCH_ASSISTED_BINDING', arm['name'])
            _same(arm['kind'], base['kind'], 'SEARCH_ASSISTED_BINDING', arm['name'])
    if p['stage'] == 'development':
        _need({'frozen_parent', 'direct_rl', 'gated_bc', 'gated_pairwise'} <= set(by_kind), 'DEVELOPMENT_ARMS', '$')
        _need(p['development_summary'] is None, 'DEVELOPMENT_LINEAGE', '$')
        for arm in direct:
            if arm['kind'] in TEACHERS:
                for seed, rep in _replications(arm).items():
                    base = _replications(by_kind['direct_rl'])[seed]
                    for key in ('updates', 'optimizer_sha256', 'direct_data_order_sha256', 'direct_rollout_ids'):
                        _same(rep[key], base[key], 'MATCHED_UPDATE', arm['name']+'.'+key)
                if 'direct_rl_compute' in by_kind:
                    _match_compute(arm, by_kind['direct_rl_compute'], p)
    else:
        _need(len(p['arms']) == len(direct) == 2 and len(seeds) == 1 and 'direct_rl_compute' in by_kind
              and len(set(by_kind) & {'gated_bc', 'gated_pairwise'}) == 1, 'FINAL_SELECTION', '$')
        dev = p['development_summary']
        _need(type(dev) is dict and dev.get('schema') == SUMMARY_SCHEMA and dev.get('stage') == 'development',
              'DEVELOPMENT_REQUIRED', '$')
        _same(dev, bound(dev), 'DEVELOPMENT_BINDING', '$')
        _need(dev['complete'] is True and dev['training_seed_replications'] >= 3,
              'DEVELOPMENT_REPLICATIONS', '$')
        _same(dev['source_pins'], p['source_pins'], 'SOURCE_DRIFT', 'development')
        _same(dev['execution'], p['execution'], 'FINAL_SELECTION', 'execution')
        _same(dev['evidence_kind'], p['evidence_kind'], 'EVIDENCE_KIND', 'development')
        dev_arms = {a['name']: a for a in dev['registered_arms']}
        _need(bool(dev_arms) and all(len(_replications(a)) == dev['training_seed_replications']
                                    for a in dev_arms.values()), 'DEVELOPMENT_REPLICATIONS', '$')
        for arm in direct:
            _need(arm['name'] in dev_arms, 'FINAL_SELECTION', arm['name'])
            old = dev_arms[arm['name']]
            _need(arm['kind'] == old['kind'] and old['actor_mode'] == 'direct', 'FINAL_SELECTION', arm['name'])
            _same(arm['replications'][0], _replications(old).get(seeds[0]), 'FINAL_SELECTION', arm['name'])
        candidate = next(a for a in direct if a['kind'] != 'direct_rl_compute')
        _match_compute(candidate, by_kind['direct_rl_compute'], p)
    # A renamed historical family or seed is still exposed. The fixed panel is
    # ALWAYS included even if the supplied ledger omits it.
    fixed = router_panel.world_registry().worlds
    exposed_groups = {w.id for w in fixed}; exposed_seeds = {w.seed for w in fixed}
    _fields(p['exposures'], SPLITS, 'exposures')
    for split in SPLITS:
        _list(p['exposures'][split], 'exposures.'+split)
        for row in p['exposures'][split]:
            _fields(row, ('source_group', 'seed'), 'exposure')
            _text(row['source_group'], 'source_group'); _integer(row['seed'], 'seed')
            exposed_groups.add(row['source_group']); exposed_seeds.add(row['seed'])
    if p['stage'] == 'final':
        dev = p['development_summary']
        exposed_groups.update(dev['exposed_source_groups'])
        exposed_seeds.update(dev['exposed_world_seeds'])
    _list(p['worlds'], 'worlds', True)
    for world in p['worlds']:
        _fields(world, ('id', 'source_group', 'seed'), 'world')
        _text(world['id'], 'world.id'); _text(world['source_group'], 'source_group'); _integer(world['seed'], 'world.seed')
        _need(world['source_group'] not in exposed_groups and world['id'] not in exposed_groups
              and world['seed'] not in exposed_seeds, 'EXPOSED_WORLD', world['id'])
    _unique([w['id'] for w in p['worlds']], 'world_ids')
    _unique([w['seed'] for w in p['worlds']], 'world_seeds')
    _integer(p['sample_size_groups'], 'sample_size_groups', 1)
    _need(p['sample_size_groups'] == len({w['source_group'] for w in p['worlds']}), 'SAMPLE_SIZE', '$')
    planned = len(p['worlds'])*2*len(seeds)*len(p['sampler_seeds'])*len(p['opponents'])*len(p['arms'])
    _need(planned <= MAX_JOBS, 'JOB_BUDGET', '$')
    _need(type(p['checkpoint_files']) is dict, 'CHECKPOINT_FILES', '$')
    checkpoints = {r['checkpoint_sha256'] for a in p['arms'] for r in a['replications']}
    for sha, path in p['checkpoint_files'].items():
        _hash(sha, 'checkpoint_sha256'); _text(path, 'checkpoint_path')
        _need(sha in checkpoints, 'CHECKPOINT_DRIFT', path)
        _same(file_sha(ROOT/path), sha, 'CHECKPOINT_DRIFT', path)


def register_comparison(plan: dict) -> dict:
    """Validate before outcomes. No inferred seed freshness or automatic selection."""
    if type(plan) is dict and 'content_sha256' in plan:
        _same(plan, bound(plan), 'REGISTRATION_DRIFT', '$')
    _validate_plan(plan)
    return bound(copy.deepcopy(plan))


def jobs_for(plan: dict) -> list[dict]:
    """Trainer-only metadata; never pass this dict or world seeds to an actor."""
    return [dict(zip(JOB_FIELDS, (a['name'], r['training_seed'], w['id'], seat, o['id'], sampler)))
            for w in plan['worlds'] for o in plan['opponents'] for seat in (0, 1)
            for sampler in plan['sampler_seeds'] for a in plan['arms'] for r in a['replications']]


def _job_key(row):
    return tuple(row[k] for k in JOB_FIELDS)


def _validate_records(p, records):
    _list(records, 'records'); _safe(records)
    wanted = {_job_key(j) for j in jobs_for(p)}
    arms = {a['name']: a for a in p['arms']}; worlds = {w['id']: w for w in p['worlds']}
    opps = {o['id']: o for o in p['opponents']}
    sources, execution = digest(p['source_pins']), digest(p['execution'])
    by_job = {}
    for row in records:
        _fields(row, (*JOB_FIELDS, 'schema', 'registration_sha256', 'source_group', 'checkpoint_sha256',
                      'opponent_sha256', 'sources_sha256', 'execution_sha256', 'actor_mode', 'evidence_kind',
                      'status', 'steps', 'cash', 'wdl', 'error', 'actions_sha256', 'runner_receipt_sha256',
                      'replay_verified', 'restart_verified', 'semantic_valid', 'costs'), 'record')
        for key in ('seat', 'training_seed', 'sampler_seed', 'steps'):
            _integer(row[key], 'record.'+key)
        for key in ('arm', 'world', 'opponent'):
            _text(row[key], 'record.'+key)
        key = _job_key(row)
        _need(key in wanted, 'UNREGISTERED_JOB', 'record')
        _need(key not in by_job, 'DUPLICATE_JOB', 'record')
        arm = arms[row['arm']]; rep = _replications(arm)[row['training_seed']]
        expected = {'schema': RECORD_SCHEMA, 'registration_sha256': p['content_sha256'],
                    'source_group': worlds[row['world']]['source_group'], 'checkpoint_sha256': rep['checkpoint_sha256'],
                    'opponent_sha256': opps[row['opponent']]['identity_sha256'], 'sources_sha256': sources,
                    'execution_sha256': execution, 'actor_mode': arm['actor_mode'], 'evidence_kind': p['evidence_kind']}
        for field, value in expected.items():
            _same(row[field], value, 'RECORD_DRIFT', field)
        for field in ('replay_verified', 'restart_verified', 'semantic_valid'):
            _need(type(row[field]) is bool or row[field] is None, 'RECORD_BOOLEAN', field)
        _hash(row['runner_receipt_sha256'], 'runner_receipt_sha256')
        _costs(row['costs'], 'record.costs')
        if arm['actor_mode'] == 'direct':
            _need(row['costs']['search_cpu_seconds'] == row['costs']['search_gpu_seconds'] == 0,
                  'SEARCH_IN_PRIMARY_ACTOR', 'record')
        _need(row['status'] in STATUSES, 'RECORD_STATUS', 'record')
        _need(row['steps'] <= 719, 'GAME_STEPS', 'record')
        if row['status'] == 'COMPLETE':
            _need(row['steps'] == 719 and row['error'] is None, 'TERMINAL_REQUIRED', 'record')
            _need(type(row['cash']) is list and len(row['cash']) == 2, 'TERMINAL_CASH', 'record')
            for cash in row['cash']:
                _number(cash, 'cash', -math.inf)
            own, rival = row['cash'][row['seat']], row['cash'][1-row['seat']]
            wdl = 'WIN' if own > rival else 'LOSS' if own < rival else 'DRAW'
            _same(row['wdl'], wdl, 'TERMINAL_WDL', 'record')
            _hash(row['actions_sha256'], 'actions_sha256')
        else:
            _need(row['cash'] is None and row['wdl'] is None, 'CENSORED_TERMINAL_LABEL', 'record')
            _text(row['error'], 'record.error')
            _hash(row['actions_sha256'], 'actions_sha256', nullable=True)
        by_job[key] = row
    return by_job, wanted


def _mean(values):
    return math.fsum(values)/len(values) if values else None


def _interval(values, alpha, family, low, high, complete):
    if not complete or not values:
        return None
    radius = (high-low)*math.sqrt(math.log(2*family/alpha)/(2*len(values)))
    mean = _mean(values)
    return [max(low, mean-radius), min(high, mean+radius)]


def _sum_costs(rows):
    return {k: math.fsum(r['costs'][k] for r in rows) for k in COST_FIELDS}


def _contrasts(p):
    direct = {a['kind']: a['name'] for a in p['arms'] if a['actor_mode'] == 'direct'}
    if p['stage'] == 'final':
        cand = next(v for k, v in direct.items() if k != 'direct_rl_compute')
        return [(cand, direct['direct_rl_compute'])]
    result = [(direct['direct_rl'], direct['frozen_parent']),
              (direct['gated_bc'], direct['direct_rl']), (direct['gated_pairwise'], direct['direct_rl'])]
    if 'direct_rl_compute' in direct:
        result += [(direct[k], direct['direct_rl_compute']) for k in ('gated_bc', 'gated_pairwise')]
    return result


def _rule(p, comparison, rows, arms):
    """Fixed single-look rule; an assessment, not checkpoint promotion."""
    rule = p['decision_rule']; reasons = []
    if any(r['semantic_valid'] is False for r in rows):
        return 'CLOSED_NEGATIVE', ['SEMANTIC_DEFECT']
    if arms[0]['kind'] in TEACHERS and not any(r['admitted_pairs'] for r in arms[0]['replications']):
        reasons.append('NO_ADMITTED_TEACHERS')
    if comparison['ci'] is None:
        reasons.append('INCOMPLETE_PANEL')
    if any(any(r[k] is not True for k in ('replay_verified', 'restart_verified', 'semantic_valid')) for r in rows):
        reasons.append('REPLAY_RESTART_UNVERIFIED')
    if comparison['world_groups'] < rule['min_world_groups']:
        reasons.append('LOW_POWER')
    if comparison['mean_score_delta'] == 0 and comparison['all_pairs_equal']:
        reasons.append('NO_SIGNAL')
    if any(_compute(r['costs'], p['compute_match']) > rule['max_training_compute']
           for a in arms for r in a['replications']):
        reasons.append('TRAINING_COST_LIMIT')
    for arm in arms:
        times = [r['costs']['end_to_end_seconds'] for r in rows if r['arm'] == arm['name']]
        if times and _mean(times) > rule['max_eval_seconds_per_game']:
            reasons.append('EVALUATION_COST_LIMIT')
            break
    if reasons:
        return 'HOLD', reasons
    low, high = comparison['ci']
    if high <= rule['min_effect_score']:
        return 'CLOSED_NEGATIVE', ['MEASURED_NON_IMPROVEMENT']
    opps = comparison['per_opponent']
    if any(o['regression'] == 'REGRESSION' for o in opps.values()):
        return 'CLOSED_NEGATIVE', ['OPPONENT_REGRESSION']
    if low > rule['min_effect_score'] and all(o['regression'] == 'CLEAR' for o in opps.values()):
        return 'GO', ['POSITIVE_WDL_AND_REGRESSION_GATE']
    return 'HOLD', ['INSUFFICIENT_PRECISION_OR_REGRESSION_CLEARANCE']


def summarize_comparison(plan: dict, records: list[dict]) -> dict:
    """Reduce frozen jobs; missing/failing attempts remain explicit, never losses."""
    _need(type(plan) is dict and 'content_sha256' in plan, 'REGISTRATION_REQUIRED', '$')
    p = register_comparison(plan)
    by_job, wanted = _validate_records(p, records)
    groups = sorted({w['source_group'] for w in p['worlds']})
    group_worlds = {g: [w['id'] for w in p['worlds'] if w['source_group'] == g] for g in groups}
    arms = {a['name']: a for a in p['arms']}
    seeds = sorted(_replications(p['arms'][0])); opponents = p['opponents']
    contrasts = _contrasts(p)
    # Includes all W/D/L/score intervals and each mixture/opponent contrast;
    # training-seed means have no additional inferential interval.
    family = len(contrasts)*(len(opponents)+1) + 4*len(arms)*len(opponents)
    alpha = p['decision_rule']['family_alpha']
    blocks = {}; wdl_reports = {}
    for name, arm in arms.items():
        for opp in opponents:
            all_rows = []
            for group in groups:
                for seed in seeds:
                    keys = [(name, seed, world, seat, opp['id'], sampler)
                            for world in group_worlds[group] for seat in (0, 1) for sampler in p['sampler_seeds']]
                    rows = [by_job[k] for k in keys if k in by_job and by_job[k]['status'] == 'COMPLETE']
                    all_rows.extend(rows)
                    blocks[name, opp['id'], group, seed] = rows if len(rows) == len(keys) else None
            proportions = {}
            for label in ('WIN', 'DRAW', 'LOSS', 'SCORE'):
                vals = []
                for group in groups:
                    rows = [blocks[name, opp['id'], group, s] for s in seeds]
                    if all(rs is not None for rs in rows):
                        vals.append(_mean([_mean([SCORE[r['wdl']] if label == 'SCORE' else float(r['wdl'] == label)
                                                 for r in rs]) for rs in rows]))
                proportions[label] = {'mean': _mean(vals), 'ci': _interval(vals, alpha, family, 0, 1, len(vals) == len(groups))}
            counts = Counter(r['wdl'] for r in all_rows)
            wdl_reports[name, opp['id']] = {
                'wins': counts['WIN'], 'draws': counts['DRAW'], 'losses': counts['LOSS'],
                'completed': len(all_rows), 'planned': len(p['worlds'])*2*len(seeds)*len(p['sampler_seeds']),
                'world_cluster_proportions': proportions,
            }
    comparisons = {}
    for candidate, control in contrasts:
        opponent_results = {}; deltas = {}; all_equal = True
        for opp in opponents:
            vals = []; gained = lost = 0
            for group in groups:
                seed_values = []
                for seed in seeds:
                    a, b = blocks[candidate, opp['id'], group, seed], blocks[control, opp['id'], group, seed]
                    if a is None or b is None:
                        continue
                    differences = [SCORE[x['wdl']]-SCORE[y['wdl']] for x, y in zip(a, b)]
                    all_equal &= all(d == 0 for d in differences)
                    gained += sum(x['wdl'] == 'WIN' and y['wdl'] != 'WIN' for x, y in zip(a, b))
                    lost += sum(x['wdl'] != 'WIN' and y['wdl'] == 'WIN' for x, y in zip(a, b))
                    deltas[opp['id'], group, seed] = _mean(differences)
                    seed_values.append(_mean(differences))
                if len(seed_values) == len(seeds):
                    vals.append(_mean(seed_values))
            ci = _interval(vals, alpha, family, -1, 1, len(vals) == len(groups))
            tolerance = opp['regression_tolerance']
            opponent_results[opp['id']] = {
                'candidate_wdl': wdl_reports[candidate, opp['id']], 'control_wdl': wdl_reports[control, opp['id']],
                'mean_score_delta': _mean(vals), 'ci': ci, 'complete_world_groups': len(vals),
                'regression_tolerance': tolerance, 'gained_wins': gained, 'lost_wins': lost,
                'regression': 'UNRESOLVED' if ci is None else 'REGRESSION' if ci[1] < -tolerance
                              else 'CLEAR' if ci[0] >= -tolerance else 'UNRESOLVED',
            }
        vals = []; by_seed = []
        for group in groups:
            if all((o['id'], group, s) in deltas for o in opponents for s in seeds):
                vals.append(math.fsum(o['weight']*_mean([deltas[o['id'], group, s] for s in seeds]) for o in opponents))
        for seed in seeds:
            seed_vals = [math.fsum(o['weight']*deltas[o['id'], g, seed] for o in opponents)
                         for g in groups if all((o['id'], g, seed) in deltas for o in opponents)]
            by_seed.append({'training_seed': seed, 'mean_score_delta': _mean(seed_vals), 'complete_world_groups': len(seed_vals)})
        c = {'candidate': candidate, 'control': control, 'world_groups': len(groups),
             'complete_world_groups': len(vals), 'mean_score_delta': _mean(vals),
             'ci': _interval(vals, alpha, family, -1, 1, len(vals) == len(groups)),
             'all_pairs_equal': all_equal, 'per_opponent': opponent_results, 'by_training_seed': by_seed}
        rows = [r for r in records if r['arm'] in (candidate, control)]
        c['rule_decision'], c['reasons'] = _rule(p, c, rows, [arms[candidate], arms[control]])
        comparisons[candidate+'_vs_'+control] = c
    counts = Counter(r['status'] for r in records)
    counts['MISSING'] = len(wanted)-len(by_job)
    complete = len(wanted) == counts['COMPLETE'] and all(r['semantic_valid'] is True for r in records)
    rule_decision, reasons = ('HOLD', ['DEVELOPMENT_ONLY']) if p['stage'] == 'development' else (
        next(iter(comparisons.values()))['rule_decision'], list(next(iter(comparisons.values()))['reasons']))
    decision = rule_decision
    if p['evidence_kind'] != 'NATIVE_GAME':
        decision = 'HOLD'; reasons.append('SYNTHETIC_DIAGNOSTIC_NOT_PLAYING_EVIDENCE')
    elif p['stage'] == 'final':
        cps = {r['checkpoint_sha256'] for a in p['arms'] for r in a['replications']}
        missing = not cps <= set(p['checkpoint_files']) or any(
            r['runtime_receipt_sha256'] is None or r['restart_receipt_sha256'] is None
            for a in p['arms'] for r in a['replications'])
        if missing:
            decision = 'HOLD'; reasons.append('MISSING_PRODUCTION_ARTIFACTS')
    evaluation_costs = {name: _sum_costs([r for r in records if r['arm'] == name]) for name in arms}
    training_costs = {name: _sum_costs(a['replications']) for name, a in arms.items()}
    optional = {name: {'kind': a['kind'], 'actor_mode': a['actor_mode'], 'training_costs': training_costs[name],
                       'evaluation_costs': evaluation_costs[name],
                       'per_opponent': {o['id']: wdl_reports[name, o['id']] for o in opponents},
                       'primary_decision_input': False}
                for name, a in arms.items() if a['kind'] == 'random_challenger' or a['actor_mode'] == 'search_assisted'}
    exposed_groups = {w['source_group'] for w in p['worlds']}
    exposed_seeds = {w['seed'] for w in p['worlds']}
    for split in SPLITS:
        exposed_groups.update(r['source_group'] for r in p['exposures'][split])
        exposed_seeds.update(r['seed'] for r in p['exposures'][split])
    if p['development_summary']:
        exposed_groups.update(p['development_summary']['exposed_source_groups'])
        exposed_seeds.update(p['development_summary']['exposed_world_seeds'])
    return bound({
        'schema': SUMMARY_SCHEMA, 'registration_sha256': p['content_sha256'], 'stage': p['stage'],
        'records_sha256': digest(records),
        'source_pins': p['source_pins'], 'execution': p['execution'], 'evidence_kind': p['evidence_kind'],
        'registered_arms': p['arms'], 'complete': complete, 'decision': decision,
        'rule_decision': rule_decision, 'reasons': reasons, 'comparisons': comparisons,
        'independent_world_groups': len(groups), 'training_seed_replications': len(seeds),
        'worlds': len(p['worlds']), 'record_counts': {s: counts[s] for s in (*STATUSES, 'MISSING')},
        'physical_game_executions': len(records) if p['evidence_kind'] == 'NATIVE_GAME' else 0,
        'synthetic_records': len(records) if p['evidence_kind'] == 'SYNTHETIC_DIAGNOSTIC' else 0,
        'native_starts_including_failures': len(records) if p['evidence_kind'] == 'NATIVE_GAME' else 0,
        'failures': [{k: r[k] for k in (*JOB_FIELDS, 'status', 'steps', 'error', 'runner_receipt_sha256')}
                     for r in records if r['status'] != 'COMPLETE'],
        'statistical_method': {'name': 'two_sided_Hoeffding_Bonferroni_source_group_means', 'family_alpha': alpha,
                               'intervals_in_family': family, 'score': 'P(win)+0.5*P(draw)',
                               'assumption': 'Independent predeclared source groups; seats/samplers/training seeds are not worlds.',
                               'training_seed_scope': 'Conditional on these frozen checkpoints; seed means are descriptive replications.'},
        'costs': {'training_by_arm': training_costs, 'evaluation_by_arm': evaluation_costs,
                  'compute_match': p['compute_match'], 'training_totals_are_not_unique_collection_counts': True},
        'optional_controls': optional, 'exposed_source_groups': sorted(exposed_groups),
        'exposed_world_seeds': sorted(exposed_seeds),
        'original_rule_transfer': {'rules': 'original', 'search_disabled_in_primary': True,
                                   'measured': p['evidence_kind'] == 'NATIVE_GAME' and complete,
                                   'scope': 'Registered opponent mixture and fresh groups, not ladder strength.'},
        'no_automatic_promotion': True, 'no_automatic_submission': True,
        'limitations': ['Hashes identify bytes, not honest chronology, genuine rollout collection or independent groups.',
                       'The caller supplies complete historical exposures; fixed Router/catalog seeds are additionally excluded.',
                       'No actions, model probabilities, replay or runtime execution are performed by this reducer.',
                       'Missing/censored results have no inferred terminal labels and disable comparison intervals.',
                       'Native runner, rollout, replay and runtime receipts need maintainer authentication.'],
    })


RUN_SCHEMA = 'farm-v0-regret-comparison-execution-v1'


def plan_execution(plan, *, output, max_physical_starts, selected_jobs=None,
                   device='cpu', timeout_seconds=180., production=True):
    """Freeze a default-off producer request; no model allocation or game starts.

    A development smoke may select one registered arm/world/opponent/training
    seed and BOTH seats: two primaries plus two fresh full replays = four starts.
    Omitted comparison jobs stay missing, not replaced by fabricated outcomes.
    """
    _need('content_sha256' in plan, 'REGISTRATION_REQUIRED', '$')
    p = register_comparison(plan)
    _need(type(production) is bool, 'SCOPE', 'production')
    _need(not production or p['evidence_kind'] == 'NATIVE_GAME', 'SCOPE', 'production needs NATIVE_GAME')
    _need(production or p['evidence_kind'] == 'SYNTHETIC_DIAGNOSTIC', 'SCOPE', 'diagnostic cannot become native evidence')
    selected = copy.deepcopy(jobs_for(p) if selected_jobs is None else selected_jobs)
    _list(selected, 'selected_jobs', True)
    wanted = {_job_key(j) for j in jobs_for(p)}
    groups = defaultdict(set)
    arms = {a['name']: a for a in p['arms']}
    for job in selected:
        _fields(job, JOB_FIELDS, 'job')
        _need(_job_key(job) in wanted, 'UNREGISTERED_JOB', 'job')
        _need(arms[job['arm']]['actor_mode'] == 'direct', 'CAPABILITY', 'search-assisted actor not implemented here')
        groups[tuple(job[k] for k in JOB_FIELDS if k != 'seat')].add(job['seat'])
    _unique([_job_key(j) for j in selected], 'selected_jobs')
    _need(all(seats == {0, 1} for seats in groups.values()), 'BOTH_SEATS_REQUIRED', 'selected_jobs')
    _integer(max_physical_starts, 'max_physical_starts', 1)
    _need(2*len(selected) <= max_physical_starts <= 2*MAX_JOBS, 'START_BUDGET', 'primary + fresh replay')
    _number(timeout_seconds, 'timeout_seconds', 1e-12, 1800)
    destination = Path(output).absolute()
    _need(not any(p.is_symlink() for p in (destination, *destination.parents)), 'OUTPUT', 'symlink')
    checkpoints = {}
    for job in selected:
        rep = _replications(arms[job['arm']])[job['training_seed']]
        sha = rep['checkpoint_sha256']
        _need(sha in p['checkpoint_files'], 'CAPABILITY', 'committed checkpoint file missing: '+sha)
        marker = ROOT / p['checkpoint_files'][sha]
        _need(marker.name == 'COMMIT.json', 'CAPABILITY', 'checkpoint_files must name native COMMIT.json')
        if sha not in checkpoints:
            info = v0_policy.inspect_committed_scorer(marker.parent, sha, device=device, production=production)
            checkpoints[sha] = {'directory': str(marker.parent.resolve()), 'identity': info['identity']}
        kind = checkpoints[sha]['identity']['kind']
        arm_kind = arms[job['arm']]['kind']
        _need((arm_kind == 'frozen_parent' and kind == 'BC') or
              (arm_kind in ('direct_rl', 'direct_rl_compute') and kind in ('RL', 'REGRET')) or
              (arm_kind in TEACHERS and kind == 'REGRET'), 'CHECKPOINT_KIND', job['arm'])
        # Preserve the embedded regret loss identity; never rename one teacher
        # mode into another comparison arm merely because tensor shapes match.
        if kind == 'REGRET':
            info = v0_policy.inspect_committed_scorer(marker.parent, sha, device=device, production=production)
            mode = info['training_identity']['arm']['id']
            expected = {'direct_rl': 'direct-rl-only', 'direct_rl_compute': 'direct-rl-only',
                        'gated_bc': 'direct-rl+gated-bc', 'gated_pairwise': 'direct-rl+gated-pairwise'}
            _need(arm_kind in expected and mode == expected[arm_kind], 'CHECKPOINT_KIND', 'regret objective')
    cases = {}
    worlds = {w['id']: w for w in p['worlds']}
    opps = {o['id']: o for o in p['opponents']}
    for job in selected:
        case = v0_policy.OriginalRuleCase(worlds[job['world']]['seed'], job['seat'], job['opponent']).document()
        _same(digest(case['opponent_identity']), opps[job['opponent']]['identity_sha256'], 'OPPONENT_DRIFT', job['opponent'])
        cases[digest(job)] = case
    return bound({'schema': RUN_SCHEMA, 'registration_sha256': p['content_sha256'],
                  'source_pins': source_pins(), 'selected_jobs': selected, 'cases': cases,
                  'checkpoints': checkpoints, 'output': str(destination), 'device': str(device),
                  'production': production, 'evidence_kind': p['evidence_kind'],
                  'max_physical_starts': max_physical_starts, 'reserved_physical_starts': 2*len(selected),
                  'timeout_seconds': float(timeout_seconds), 'restart_steps': [528],
                  'temperature': 1., 'execution': copy.deepcopy(p['execution']),
                  'replay': 'FRESH_CHECKPOINT_LOAD_AND_FULL_GAME; NO_REROLL',
                  'cost_scope': 'LOAD_PRIMARY_REPLAY_AND_VALIDATION; CUDA_ALLOCATION_WALL_NOT_UTILIZATION',
                  'scope': 'PRODUCER_ONLY; NO_SELECTION_PROMOTION_DEPLOYMENT'})


def check_execution(plan, request):
    """Rejoin all frozen inputs without creating an actor or initializing a game."""
    _same(request, bound(request), 'RUN_PLAN_BINDING', '$')
    expected = plan_execution(plan, output=request['output'], max_physical_starts=request['max_physical_starts'],
                              selected_jobs=request['selected_jobs'], device=request['device'],
                              timeout_seconds=request['timeout_seconds'], production=request['production'])
    _same(request, expected, 'RUN_PLAN_DRIFT', '$')
    return expected


def _restart_verified(row, steps):
    if row.get('status') != 'COMPLETE' or not steps:
        return False
    shadows = row.get('shadows', [])
    for step in steps:
        hits = [s for s in shadows if s.get('name') == f'restart-{step}']
        cuts = [c for c in row.get('checkpoints', []) if c.get('actor', {}).get('step') == step]
        if len(hits) != 1 or len(cuts) != 1:
            return False
        s = hits[0]
        if not (s.get('from_step') == step and s.get('steps_checked') == 719-step
                and s.get('complete_action_equivalence') is True and s.get('actions_equal') is True
                and s.get('exact_plans_equal') is True and s.get('first_divergence') is None):
            return False
    return True


def _verify_game_receipt(row, job, case, policy_registration):
    """Check producer joins and stream integrity, not caller-supplied truth flags."""
    _need(type(row) is dict and row == v0_policy.bind(row), 'RUNNER_RECEIPT_BINDING', '$')
    expected = {'schema': v0_policy.SCHEMA+'-game', 'job': policy_registration['jobs'][0],
                'case_id': case['id'], 'scenario': case['scenario'], 'seat': job['seat'],
                'world_seed': case['scenario']['seed'], 'opponent_identity': case['opponent_identity'],
                'mode': policy_registration['jobs'][0]['mode'], 'sampler_seed': job['sampler_seed'],
                'compiler': policy_registration['compiler'],
                'registration_sha256': policy_registration['content_sha256'],
                'arm_identity_sha256': v0_policy.wire_hash(policy_registration['arms'][0]['identity'])}
    for key, value in expected.items():
        _same(row.get(key), value, 'RUNNER_RECEIPT_DRIFT', key)
    initial = row.get('policy_initial_state')
    if initial is not None:
        actor_identity = copy.deepcopy(policy_registration['arms'][0]['identity'])
        actor_identity.pop('publication', None)
        expected_state = {'identity': actor_identity, 'mode': expected['mode'],
                          'seed': job['sampler_seed'], 'temperature': 1.,
                          'publication': 'opening_only_later_keep'}
        _need(initial.get('kind') == 'SCORER', 'ACTOR_IDENTITY', '$')
        for key, value in expected_state.items():
            _same(initial.get('state', {}).get(key), value, 'ACTOR_IDENTITY', key)
    elif row.get('status') == 'COMPLETE':
        raise ComparisonError('ACTOR_IDENTITY', 'complete game without initial scorer state')
    for field in ('own_actions', 'joint_actions'):
        _same(row.get(field+'_sha256'), v0_policy.wire_hash(row.get(field)), 'ACTION_STREAM_DRIFT', field)
    _need(row['physical_steps'] == len(row['joint_actions']) == len(row['own_actions']), 'STEP_STREAM_DRIFT', '$')
    if row['status'] == 'COMPLETE':
        _need(row['physical_steps'] == 719 and row['physical_games_completed'] == 1
              and len(row['epochs']) == 30, 'TERMINAL_REQUIRED', '$')


def _comparison_record(p, request, job, policy_registration, primary, replay, receipt_sha, costs):
    """One existing-schema row; all replay/restart flags derive from kept evidence."""
    case = request['cases'][digest(job)]
    for row in (primary, replay):
        if row is not None:
            _verify_game_receipt(row, job, case, policy_registration)
    restart = all(r is not None and _restart_verified(r, request['restart_steps']) for r in (primary, replay))
    complete = all(r is not None and r.get('status') == 'COMPLETE' for r in (primary, replay))
    replay_ok = complete and all(primary.get(k) == replay.get(k) for k in (
        'joint_actions_sha256', 'own_actions_sha256', 'terminal_cash_by_seat',
        'terminal_environment_sha256', 'terminal_agent_signature', 'terminal_controller_sha256',
        'terminal_plan', 'epochs', 'policy_initial_state'))
    semantic = complete and all(r.get('checks_passed') is True for r in (primary, replay))
    status = 'COMPLETE' if replay_ok and restart and semantic else 'CENSORED'
    rep = _replications(next(a for a in p['arms'] if a['name'] == job['arm']))[job['training_seed']]
    world = next(w for w in p['worlds'] if w['id'] == job['world'])
    opp = next(o for o in p['opponents'] if o['id'] == job['opponent'])
    error = None if status == 'COMPLETE' else 'primary/replay/restart incomplete or divergent; no retry'
    if primary and primary.get('error'):
        error += ': '+str(primary['error'])
    return {**job, 'schema': RECORD_SCHEMA, 'registration_sha256': p['content_sha256'],
            'source_group': world['source_group'], 'checkpoint_sha256': rep['checkpoint_sha256'],
            'opponent_sha256': opp['identity_sha256'], 'sources_sha256': digest(p['source_pins']),
            'execution_sha256': digest(p['execution']), 'actor_mode': 'direct', 'evidence_kind': p['evidence_kind'],
            'status': status, 'steps': 0 if primary is None else primary['physical_steps'],
            'cash': primary['terminal_cash_by_seat'] if status == 'COMPLETE' else None,
            'wdl': primary['wdl'] if status == 'COMPLETE' else None, 'error': error,
            'actions_sha256': None if primary is None else primary['joint_actions_sha256'],
            'runner_receipt_sha256': receipt_sha, 'replay_verified': bool(replay_ok),
            'restart_verified': bool(restart), 'semantic_valid': bool(semantic), 'costs': costs}


def run_execution(plan, request, *, allow_native=False):
    """Explicitly authorized serial producer; exclusive frozen output, no retries.

    Interrupted runs retain their fsynced START journal and partial receipts. Do
    not delete/reuse that directory: missing jobs remain missing/censored at the
    unchanged comparison reducer. Deadlines are cooperative, as in v0_policy.
    """
    import os
    import time
    _need(allow_native is True, 'NATIVE_OPT_IN_REQUIRED', 'run')
    request = check_execution(plan, request)
    p = register_comparison(plan)
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/'EXECUTION.json', request)
    write_new(output/'COMPARISON.json', p)
    rows, charges = [], 0
    arms = {a['name']: a for a in p['arms']}
    with (output/'JOURNAL.jsonl').open('x') as journal:
        def emit(event):
            journal.write(json.dumps(bound(event), sort_keys=True)+'\n')
            journal.flush(); os.fsync(journal.fileno())
        for index, job in enumerate(request['selected_jobs']):
            start, cpu = time.monotonic(), time.process_time()
            rep = _replications(arms[job['arm']])[job['training_seed']]
            checkpoint = request['checkpoints'][rep['checkpoint_sha256']]
            case = request['cases'][digest(job)]
            case_obj = v0_policy.OriginalRuleCase(case['scenario']['seed'], job['seat'], job['opponent'])
            game_rows, refs, policy_registration, failure = [], [], None, None
            # Both primary and replay are prospectively reserved. Replay is not
            # another draw: same world, same sampler seed, independently loaded model.
            for role in ('primary', 'replay'):
                if role == 'replay' and (not game_rows or game_rows[0]['status'] != 'COMPLETE'):
                    break
                emit({'event': 'ATTEMPT', 'job': job, 'role': role,
                      'execution_sha256': request['content_sha256']})
                try:
                    loaded = v0_policy.load_committed_scorer(checkpoint['directory'], rep['checkpoint_sha256'],
                                      device=request['device'], production=request['production'])
                    _same(loaded.identity, checkpoint['identity'], 'CHECKPOINT_DRIFT', 'loaded')
                    def factory(mode, seed):
                        return loaded.policy(mode, seed, temperature=request['temperature'], opening_only=True)
                    arm = v0_policy.Arm(job['arm'], factory, {**loaded.identity, 'publication': p['execution']['publication']})
                    policy_registration = v0_policy.registration([case_obj], [arm], modes=(p['execution']['decode_mode'],),
                        sample_seeds=(job['sampler_seed'],), max_games=1, timeout_seconds=request['timeout_seconds'],
                        restart_steps=tuple(request['restart_steps']))
                    reg_path = output/f'job-{index:06d}-{role}-registration.json'
                    v0_policy.write_new(reg_path, policy_registration)
                    _same(source_pins(), request['source_pins'], 'SOURCE_DRIFT', 'before physical start')
                    _need(charges < request['max_physical_starts'], 'START_BUDGET', role)
                    charges += 1
                    emit({'event': 'START', 'job': job, 'role': role, 'charged_starts': charges,
                          'policy_registration_sha256': policy_registration['content_sha256']})
                    row = v0_policy.play_game(case, arm, policy_registration['jobs'][0], policy_registration)
                    path = output/f'job-{index:06d}-{role}.json'
                    v0_policy.write_new(path, row)
                    refs.append({'role': role, 'path': path.name, 'sha256': file_sha(path),
                                 'registration_path': reg_path.name, 'registration_sha256': file_sha(reg_path)})
                    _verify_game_receipt(row, job, case, policy_registration)
                    game_rows.append(row)
                    emit({'event': 'RESULT', 'job': job, 'role': role, 'status': row['status'], 'receipt': refs[-1]})
                except Exception as exc:
                    failure = {'type': type(exc).__name__, 'message': str(exc), 'role': role}
                    emit({'event': 'FAILED', 'job': job, 'failure': failure})
                    break
            elapsed = time.monotonic()-start
            costs = {k: 0. for k in COST_FIELDS}
            costs.update(cpu_seconds=time.process_time()-cpu, end_to_end_seconds=elapsed,
                         gpu_seconds=elapsed if request['device'].startswith('cuda') else 0.)
            receipt_path = output/f'job-{index:06d}-receipt.json'
            write_new(receipt_path, {'schema': RUN_SCHEMA+'-receipt', 'job': job,
                'registration_sha256': p['content_sha256'], 'execution_sha256': request['content_sha256'],
                'arm': arms[job['arm']], 'checkpoint': checkpoint, 'games': refs, 'failure': failure,
                'costs': costs, 'cost_scope': request['cost_scope'], 'no_automatic_retry': True})
            primary = game_rows[0] if game_rows else None
            replay = game_rows[1] if len(game_rows) == 2 else None
            row = _comparison_record(p, request, job, policy_registration, primary, replay,
                                     file_sha(receipt_path), costs)
            if failure is not None:
                row.update(status='CENSORED', cash=None, wdl=None, error=str(failure), semantic_valid=False)
            rows.append(row)
            emit({'event': 'RECORD', 'record': row})
            # Avoid retaining many production models across the panel.
            if 'loaded' in locals():
                del loaded
    _validate_records(p, rows)
    with (output/'RECORDS.json').open('x') as stream:
        json.dump(rows, stream, sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    _same(source_pins(), request['source_pins'], 'SOURCE_DRIFT', 'after run')
    summary = summarize_comparison(p, rows)
    write_new(output/'SUMMARY.json', summary)
    return write_new(output/'RESULT.json', {'schema': RUN_SCHEMA+'-result',
        'execution_sha256': request['content_sha256'], 'status': 'RECORDED', 'charged_physical_starts': charges,
        'records': len(rows), 'censored': sum(r['status'] != 'COMPLETE' for r in rows),
        'records_sha256': file_sha(output/'RECORDS.json'), 'summary_sha256': file_sha(output/'SUMMARY.json'),
        'scope': request['scope']})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('register', 'jobs', 'summarize'):
        p = sub.add_parser(name); p.add_argument('--plan', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        if name == 'summarize':
            p.add_argument('--records', type=Path, required=True)
    for name in ('plan-run', 'check-run', 'run'):
        q = sub.add_parser(name)
        q.add_argument('--plan', type=Path, required=True)
        if name == 'plan-run':
            q.add_argument('--out', type=Path, required=True)
            q.add_argument('--run-directory', type=Path, required=True)
            q.add_argument('--max-physical-starts', type=int, required=True)
            q.add_argument('--selected-jobs', type=Path)
            q.add_argument('--device', default='cpu')
            q.add_argument('--timeout-seconds', type=float, default=180.)
            q.add_argument('--diagnostic', action='store_true')
        else:
            q.add_argument('--request', type=Path, required=True)
            if name == 'run':
                q.add_argument('--allow-native', action='store_true')
    args = parser.parse_args(argv)
    try:
        plan = read(args.plan, binding=False)
        if args.command == 'plan-run':
            result = plan_execution(plan, output=args.run_directory, max_physical_starts=args.max_physical_starts,
                selected_jobs=None if args.selected_jobs is None else read(args.selected_jobs, binding=False),
                device=args.device, timeout_seconds=args.timeout_seconds, production=not args.diagnostic)
            write_new(args.out, result)
            print(json.dumps({'status': 'PLANNED_NO_STARTS', 'content_sha256': result['content_sha256']}))
            return 0
        if args.command in ('check-run', 'run'):
            request = read(args.request)
            result = (check_execution(plan, request) if args.command == 'check-run' else
                      run_execution(plan, request, allow_native=args.allow_native))
            print(json.dumps({'status': 'CHECKED_NO_STARTS' if args.command == 'check-run' else result['status'],
                              'content_sha256': result['content_sha256']}))
            return 0
        if args.command == 'register':
            result = register_comparison(plan)
        elif args.command == 'jobs':
            _need('content_sha256' in plan, 'REGISTRATION_REQUIRED', '$')
            plan = register_comparison(plan)
            result = bound({'schema': PLAN_SCHEMA+'-jobs', 'registration_sha256': plan['content_sha256'],
                            'jobs': jobs_for(plan), 'actor_visible': False})
        else:
            result = summarize_comparison(plan, read(args.records, binding=False))
        write_new(args.out, result)
        print(json.dumps({'status': 'WRITTEN', 'path': str(args.out), 'content_sha256': result['content_sha256']}))
        return 0
    except (ValueError, OSError, TypeError, KeyError, OverflowError, RecursionError) as error:
        print(json.dumps({'status': 'REFUSED', 'error': str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
