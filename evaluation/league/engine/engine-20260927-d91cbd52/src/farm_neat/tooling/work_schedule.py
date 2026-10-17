"""Read-only dependency view over the existing project/review.toml assignments.

No second backlog, job launcher or automatic acceptance. A maintainer accepts a task
by attaching a reviewed, byte-bound receipt of the completed review. Its source
pins record historical inputs, not perpetual dependencies on the current tree.
project/dispatch.toml only names which READY reviewer tasks this wave hands out under
numbers and which it holds back; it never changes a task's state or dependencies.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
import tomllib

from farm_neat.tooling import layout

FIELDS = {'wave', 'lane', 'owner', 'depends_on', 'writes', 'state', 'evidence'}
LANES = {'BUILD', 'THEOREM', 'MAINTAINER'}
STATES = {'OPEN', 'PARKED', 'ACCEPTED'}
# Owner allocation of reviewer draws (2026-09-17): about 30% to the strongest non-neural policy
# (programmatic/search/heuristic solver, teacher and oracle), 70% to the V0 learner, its data,
# evaluation and training-directed theory. A task without `track` is LEARNER. Each draw still
# selects exactly one READY task; the shares hold over repeated draws, not per worker.
TRACK_SHARES = {'LEARNER': 70, 'SOLVER': 30}


def _relative(name: str) -> str:
    if (not isinstance(name, str) or not name or '\\' in name
            or PurePosixPath(name).is_absolute()
            or any(part in ('..', '.') for part in name.split('/'))):
        raise ValueError('repository-relative path required')
    return name


def _bound_bytes(root: Path, record: dict) -> bytes:
    if not isinstance(record, dict) or set(record) != {'path', 'sha256'}:
        raise ValueError('receipt identity requires path and sha256')
    data = layout.checked_path(root, _relative(record['path'])).read_bytes()
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('receipt identity mismatch: ' + record['path'])
    return data


def _acceptance(root: Path, item: dict, evidence: list) -> None:
    if not evidence:
        raise ValueError('ACCEPTED work requires a reviewed receipt')
    for record in evidence:
        receipt = json.loads(_bound_bytes(root, record))
        if receipt.get('task_id') != item['id'] or receipt.get('decision') != 'ACCEPT':
            raise ValueError('receipt must accept the exact task')
        checks = receipt.get('checks')
        if (not isinstance(checks, list) or not checks
                or any(not isinstance(c, dict) or not c.get('name')
                       or c.get('passed') is not True for c in checks)):
            raise ValueError('acceptance requires named passed checks')
        pins = receipt.get('source_pins')
        if not isinstance(pins, dict) or not pins:
            raise ValueError('acceptance requires source pins')
        for name, expected in pins.items():
            _relative(name)
            if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
                raise ValueError('acceptance source pins require full SHA256 identities')


def inspect(root: Path, assignments: list[dict]) -> dict:
    """Validate the DAG and recorded acceptance; report, never execute, work.

    Receipt checks establish identities and recorded acceptance, not independent
    scientific validation. Existing assignments without metadata remain compatible.
    """
    rows = {}
    for item in assignments:
        key = item['id']
        if key in rows:
            raise ValueError('duplicate scheduled task: ' + key)
        s = item.get('schedule', {'wave': 0, 'lane': 'THEOREM', 'owner': 'reviewer',
                                 'depends_on': [], 'writes': [], 'state': 'OPEN', 'evidence': []})
        if not isinstance(s, dict) or set(s) - {'track'} != FIELDS:
            raise ValueError('closed schedule fields required')
        s = {'track': 'LEARNER', **s}
        if s['track'] not in TRACK_SHARES:
            raise ValueError('known schedule track required: ' + ', '.join(sorted(TRACK_SHARES)))
        if type(s['wave']) is not int or not 0 <= s['wave'] <= 99:
            raise ValueError('wave must be an integer 0..99')
        if s['lane'] not in LANES or s['state'] not in STATES:
            raise ValueError('known schedule lane/state required')
        if not isinstance(s['owner'], str) or not s['owner'].strip():
            raise ValueError('scheduled owner role required')
        for field in ('depends_on', 'writes'):
            value = s[field]
            if (not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value)
                    or len(value) != len(set(value))):
                raise ValueError('distinct schedule ' + field + ' required')
        for name in s['writes']:
            _relative(name)
        if not isinstance(s['evidence'], list):
            raise ValueError('evidence must be a list of receipt identities')
        if s['state'] == 'ACCEPTED':
            _acceptance(root, item, s['evidence'])
        elif s['evidence']:
            raise ValueError('unaccepted work cannot attach acceptance evidence')
        rows[key] = {'id': key, 'title': item.get('title', key), **s,
                     'weight': item.get('weight', 1),
                     'acceptance': item.get('acceptance', [])}
    visiting, visited = set(), set()

    def visit(key: str) -> None:
        if key in visiting:
            raise ValueError('schedule dependency cycle: ' + key)
        if key in visited:
            return
        visiting.add(key)
        row = rows[key]
        for dep in row['depends_on']:
            if dep not in rows:
                raise ValueError('unknown dependency: ' + dep)
            if rows[dep]['wave'] > row['wave']:
                raise ValueError('dependency must be in the same or an earlier wave')
            visit(dep)
        visiting.remove(key); visited.add(key)
        waiting = [d for d in row['depends_on'] if rows[d]['state'] != 'ACCEPTED']
        if row['state'] == 'ACCEPTED' and waiting:
            raise ValueError('accepted task has unaccepted dependency: ' + key)
        row['waiting_for'] = waiting
        row['status'] = (row['state'] if row['state'] != 'OPEN'
                         else 'WAITING' if waiting else 'READY')

    for key in rows:
        visit(key)
    ordered = sorted(rows.values(), key=lambda r: (r['wave'], r['id']))
    ready = [r for r in ordered if r['status'] == 'READY']
    review = [r for r in ready if r['lane'] != 'MAINTAINER']
    total = sum(r['weight'] for r in review)
    draw = draw_weights(review)
    return {'schema': 'farm-review-work-schedule-v1',
            'authority': 'project/review.toml [[assignments]]',
            'tasks': ordered,
            'ready_for_review': [r['id'] for r in review],
            'ready_for_maintainer': [r['id'] for r in ready if r['lane'] == 'MAINTAINER'],
            'review_lottery_weight': total,
            'build_lottery_weight': sum(r['weight'] for r in review if r['lane'] == 'BUILD'),
            'track_shares': dict(TRACK_SHARES),
            'draw_probability': {r['id']: round(w / sum(draw), 4) for r, w in zip(review, draw)} if review else {},
            'jobs_started': 0, 'optimizer_updates': 0,
            'limitations': ['A wave is dependency order, not a completion-time estimate.',
                            'Receipt validation checks historical acceptance, not current source compatibility; run affected tests before integration.',
                            'Owner is a role, not a claim that a worker or GPU job is running.']}



def _source_module(path: str) -> str | None:
    """Translate one src/*.py write target to its import name."""
    p = PurePosixPath(path)
    if len(p.parts) < 3 or p.parts[0] != "src" or p.suffix != ".py":
        return None
    parts = list(p.with_suffix("").parts[1:])
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or None


def _paths_overlap(a: str, b: str) -> bool:
    """Ownership overlaps when either repository-relative path contains the other."""
    aa, bb = PurePosixPath(a), PurePosixPath(b)
    return aa == bb or aa in bb.parents or bb in aa.parents


def lint_dispatch(root: Path, assignments: list[dict], plan: dict) -> dict:
    """Preflight structural mistakes that otherwise surface only after worker returns.

    This intentionally checks only machine-provable dispatch structure. Scientific
    sufficiency (for example whether four native games can establish strength) remains
    an acceptance-design decision, not a string heuristic.
    """
    view = inspect(root, assignments)
    rows = {row["id"]: row for row in view["tasks"]}
    entrypoints = tomllib.loads((root / "project/entrypoints.toml").read_text())
    declared = {item.get("module") for kind in ("step", "tool") for item in entrypoints.get(kind, [])
                if isinstance(item, dict) and isinstance(item.get("module"), str)}
    planned: dict[str, str] = {}
    for item in entrypoints.get("planned", []):
        if (not isinstance(item, dict) or set(item) != {"module", "task", "purpose"}
                or not all(isinstance(item.get(k), str) and item[k] for k in item)):
            raise ValueError("planned entry needs module, task and purpose")
        module, owner = item["module"], item["task"]
        if module in planned:
            raise ValueError("duplicate planned module: " + module)
        planned[module] = owner

    errors: list[dict] = []
    slots = plan.get("slots", []) if isinstance(plan, dict) else []
    dispatched = {item.get("task") for item in slots if isinstance(item, dict)}
    for row in view["tasks"]:
        if row["id"] not in dispatched:
            continue
        for path in row["writes"]:
            module = _source_module(path)
            if module is None or (root / path).exists():
                continue
            if module in declared:
                continue
            owner = planned.get(module)
            if owner is None:
                errors.append({
                    "code": "UNREGISTERED_PLANNED_MODULE",
                    "task": row["id"],
                    "path": path,
                    "module": module,
                    "message": "absent source module has no [[planned]] entrypoint owned by its task",
                })
            elif owner != row["id"]:
                errors.append({
                    "code": "PLANNED_MODULE_OWNER_MISMATCH",
                    "task": row["id"],
                    "path": path,
                    "module": module,
                    "planned_task": owner,
                    "message": "planned source module is owned by a different task",
                })

    for left_index, left in enumerate(slots):
        left_row = rows.get(left.get("task")) if isinstance(left, dict) else None
        if left_row is None:
            raise ValueError("dispatch lint needs known slot tasks")
        for right in slots[left_index + 1:]:
            right_row = rows.get(right.get("task")) if isinstance(right, dict) else None
            if right_row is None:
                raise ValueError("dispatch lint needs known slot tasks")
            for a in left_row["writes"]:
                for b in right_row["writes"]:
                    if _paths_overlap(a, b):
                        errors.append({
                            "code": "DISPATCH_WRITE_OVERLAP",
                            "task_a": left_row["id"],
                            "task_b": right_row["id"],
                            "path_a": a,
                            "path_b": b,
                            "message": "simultaneously dispatched tasks have overlapping write ownership",
                        })

    return {
        "schema": "farm-review-dispatch-lint-v1",
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "checked_tasks": len(view["tasks"]),
        "checked_slots": len(slots),
        "limitations": [
            "Checks structural dispatch hazards only; it does not judge scientific sufficiency.",
            "Existing source modules are covered by the separate reachability check.",
        ],
    }

def draw_weights(ready: list[dict]) -> list[int]:
    """Exact integer weights: track share times the task's weight within its track.

    A track with no ready task gives its share to the others. w_i = share(t) * weight_i * P / W(t),
    with P the product of the ready tracks' total weights, so every w_i is an integer.
    """
    totals = {}
    for r in ready:
        totals[r['track']] = totals.get(r['track'], 0) + r['weight']
    product = 1
    for value in totals.values():
        product *= value
    factor = {t: TRACK_SHARES[t] * product // total for t, total in totals.items()}
    common = math.gcd(*factor.values()) if factor else 1  # a single-track pool keeps its declared weights
    return [r['weight'] * factor[r['track']] // common for r in ready]


def lottery_weights(root: Path, assignments: list[dict]) -> list[int]:
    view = inspect(root, assignments)
    ready = {r['id']: r for r in view['tasks'] if r['id'] in set(view['ready_for_review'])}
    rows = [ready[a['id']] for a in assignments if a['id'] in ready]
    by_id = dict(zip([r['id'] for r in rows], draw_weights(rows))) if rows else {}
    weights = [by_id.get(a['id'], 0) for a in assignments]
    if not sum(weights):
        raise ValueError('no ready reviewer work; do not reroll or reopen parked theory')
    return weights


def dispatch(view: dict, spec: dict) -> dict:
    """Join numbered slots and holds to the validated schedule; refuse anything stale.

    A slot names a READY reviewer task; a hold names an open (READY or WAITING) reviewer
    task kept out of this wave. Every READY reviewer task is exactly one of the two.
    """
    if not isinstance(spec, dict) or set(spec) - {'slot', 'hold'} != {'version'} or spec['version'] != 1:
        raise ValueError('dispatch needs version = 1 with only [[slot]] and [[hold]] tables')
    slots, holds = spec.get('slot', []), spec.get('hold', [])
    if not isinstance(slots, list) or not isinstance(holds, list):
        raise ValueError('dispatch slots and holds must be arrays of tables')
    rows = {r['id']: r for r in view['tasks']}

    def reviewer_task(key, kind):
        if not isinstance(key, str) or key not in rows:
            raise ValueError(f'dispatch {kind} names an unknown task: {key}')
        row = rows[key]
        if row['lane'] == 'MAINTAINER':
            raise ValueError(f'dispatch {kind} names maintainer-only work: {key}')
        return row

    numbered, seen = [], set()
    for item in slots:
        if not isinstance(item, dict) or set(item) != {'number', 'task'}:
            raise ValueError('dispatch slot needs exactly number and task')
        if type(item['number']) is not int or item['number'] != len(numbered) + 1:
            raise ValueError('dispatch slot numbers must run 1, 2, 3, ... in order')
        if item['task'] in seen:
            raise ValueError('task dispatched twice: ' + str(item['task']))
        row = reviewer_task(item['task'], 'slot')
        if row['status'] != 'READY':
            raise ValueError(f"slot {item['number']} names {row['status']} work: {row['id']}")
        seen.add(row['id'])
        numbered.append({'number': item['number'], 'task': row['id'], 'title': row['title'],
                         'track': row['track'], 'lane': row['lane'], 'wave': row['wave']})
    held = []
    for item in holds:
        if not isinstance(item, dict) or set(item) != {'task', 'reason'}:
            raise ValueError('dispatch hold needs exactly task and reason')
        if not isinstance(item['reason'], str) or not item['reason'].strip():
            raise ValueError('dispatch hold needs a reason')
        if item['task'] in seen:
            raise ValueError('task both dispatched and held, or held twice: ' + str(item['task']))
        row = reviewer_task(item['task'], 'hold')
        if row['status'] not in ('READY', 'WAITING'):
            raise ValueError(f"hold names {row['status']} work: {row['id']}")
        seen.add(row['id'])
        held.append({'task': row['id'], 'status': row['status'], 'reason': item['reason']})
    loose = [key for key in view['ready_for_review'] if key not in seen]
    if loose:
        raise ValueError('READY reviewer work neither dispatched nor held: ' + ', '.join(loose))
    return {'schema': 'farm-review-dispatch-v1', 'authority': 'project/review.toml [[assignments]]',
            'slots': numbered, 'holds': held}


def resolve(plan: dict, view: dict, *, slot: int | None = None, task: str | None = None) -> dict:
    """One dispatched slot by number or by task id; held, blocked and finished work refuse."""
    if (slot is None) == (task is None):
        raise ValueError('give exactly one of a slot number or a task id')
    if slot is not None:
        matches = [s for s in plan['slots'] if s['number'] == slot]
        if not matches:
            raise ValueError(f"unknown slot {slot}; this ZIP dispatches 1..{len(plan['slots'])}")
        return matches[0]
    matches = [s for s in plan['slots'] if s['task'] == task]
    if matches:
        return matches[0]
    for hold in plan['holds']:
        if hold['task'] == task:
            raise ValueError(f"{task} is on hold: {hold['reason']}")
    rows = {r['id']: r for r in view['tasks']}
    if task not in rows:
        raise ValueError('unknown task: ' + str(task))
    row = rows[task]
    if row['lane'] == 'MAINTAINER':
        raise ValueError(f'{task} is maintainer-only work')
    raise ValueError(f"{task} is {row['status']}, not dispatched")


def summary(view: dict, plan: dict | None = None) -> str:
    marks = {}
    if plan is not None:
        marks.update({s['task']: f"slot {s['number']}" for s in plan['slots']})
        marks.update({h['task']: 'HOLD' for h in plan['holds']})
    lines = ['# V0 training work schedule', '',
             'Dependency waves, not dates. Acceptance records completed review, not current-source certification; this command starts no jobs.',
             'Tracks: LEARNER (V0 learner, data, evaluation) and SOLVER (non-neural policy, teachers, oracle).', '',
             '| Wave | Task | Track | Lane / owner role | Status | Dispatch | Waiting for |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for r in view['tasks']:
        lines.append(f"| {r['wave']} | {r['id']} | {r['track']} | {r['lane']} / {r['owner']} | {r['status']} | {marks.get(r['id'], '-')} | {', '.join(r['waiting_for']) or '-'} |")
    lines += ['', 'Ready for reviewers: ' + (', '.join(view['ready_for_review']) or 'none'),
              'Ready for maintainer execution: ' + (', '.join(view['ready_for_maintainer']) or 'none'), '']
    if plan is not None:
        lines += [dispatch_summary(plan), '']
    lines += ['Take your number: review_package.py --assign-slot N (or --assign TASK-ID)', '']
    return '\n'.join(lines)


def dispatch_summary(plan: dict) -> str:
    lines = [f"Slot {s['number']}: {s['task']} [{s['track']} {s['lane']}] {s['title']}" for s in plan['slots']]
    lines += [f"HOLD: {h['task']} ({h['status']}): {h['reason']}" for h in plan['holds']]
    return '\n'.join(lines)
