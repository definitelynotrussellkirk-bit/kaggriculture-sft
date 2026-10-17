"""Exact own-unit execution masks, schema v1 (Python standard library only).

Training: masks supervision, NEVER modifies the teacher action or its prefix.
Inference: token feasibility from the evolving PrefixExecutor, with a monotone
PLANT request budget. All codec quantity bins are positive, so feasibility is
constant across the quantity bins of a token (partial transfers still count).

This is an execution predicate, NOT a value/policy improvement oracle. DROP
can destroy resources and repeated FERTILIZE can waste them; both have effects.
Market orders are deliberately not masked: opponent lockstep prices are hidden.
"""
from __future__ import annotations

import copy
from collections import Counter
from functools import lru_cache

SCHEMA = 'unit-noop-mask-v1'


def unit_reason(farm, private, k, action, *, engine, day, board_size, shed_capacity):
    """Return 'effect', 'pass', a proven no-effect reason, or 'unknown'.

    Reads but never mutates engine-format state. Requires valid own private
    inventories; callers must not supply zero-filled stand-ins for hidden data.
    Checks operation ordering exactly as _apply_unit_action, notably MOVE and
    shed paths before LOCKED, and animal PLACE before quantity validation.
    """
    pos = engine._farmer_position(farm, k)
    if pos is None:
        return 'absent_unit'
    if not isinstance(action, list) or not action:
        return 'malformed'
    op = action[0]
    if not isinstance(op, str):
        return 'unknown'  # malformed engine-crashing input is not a proven no-op
    x, y = pos
    if op in engine.FARMER_MOVES:
        dx, dy = engine.FARMER_MOVES[op]
        return 'effect' if 0 <= x + dx < board_size and 0 <= y + dy < board_size else 'board_edge'
    if op == 'PASS':
        return 'pass'
    inventories = private['inventories']
    inv = inventories[k] if k < len(inventories) else {}
    shed = private['shed']
    tile = farm['tiles'][y][x]
    half = board_size // 2
    adjacent = x in (half - 1, half) and y in (half - 1, half)
    if op == 'DROP':
        if not adjacent:
            return 'not_shed_access'
        # Full-shed DROP destroys positive carried goods; it is not a no-op.
        return 'effect' if any(v > 0 for v in inv.values()) else 'empty_carrier'
    if op in ('PICKUP', 'PLACE'):
        if len(action) < 2:
            return 'missing_item'
        item = action[1]
        if not isinstance(item, str):
            return 'unknown'
        if (op == 'PLACE' and item in engine.ANIMALS and isinstance(tile, dict)
                and tile.get('kind') == engine.ANIMALS[item]['structure'] and 'animal' not in tile):
            return 'effect' if inv.get(item, 0) >= 1 else 'no_carried_item'
        if not adjacent:
            return 'not_shed_access'
        try:
            n = int(action[2]) if len(action) > 2 else 1
        except (TypeError, ValueError, OverflowError):
            return 'unknown'
        if n <= 0:
            return 'nonpositive_quantity'
        if op == 'PICKUP':
            return 'effect' if shed.get(item, 0) > 0 else 'no_shed_item'
        if inv.get(item, 0) <= 0:
            return 'no_carried_item'
        return 'effect' if sum(shed.values()) < shed_capacity else 'shed_full'
    if tile == 'LOCKED':
        return 'locked_tile'
    plant = isinstance(tile, dict) and tile.get('kind') == 'PLANT'
    animal = isinstance(tile, dict) and 'animal' in tile
    if op == 'PLANT':
        if len(action) < 2 or not isinstance(action[1], str) or action[1] not in engine.CROPS:
            return 'invalid_crop'
        if tile is not None:
            return 'occupied_tile'
        return 'effect' if private['seeds'].get(action[1], 0) > 0 else 'no_seed'
    if op == 'WATER':
        if not plant:
            return 'not_plant'
        return 'already_watered' if tile['watered_today'] else 'effect'
    if op == 'HARVEST':
        if not (plant or animal):
            return 'not_producer'
        if tile.get('yield_units', 0) <= 0:
            return 'no_yield'
        if plant and day - tile['planted_day'] < engine.CROPS[tile['crop']]['first_yield_day']:
            return 'immature'
        return 'effect'
    if op == 'FERTILIZE':
        if not plant:
            return 'not_plant'
        return 'effect' if inv.get('FERTILIZER', 0) >= 1 else 'no_fertilizer'
    if op == 'DIG':
        if animal:
            return 'placed_animal'
        return 'empty_tile' if tile is None else 'effect'
    if op in ('BUILD_COOP', 'BUILD_PASTURE'):
        return 'effect' if tile is None else 'occupied_tile'
    if op in ('FEED', 'CARE', 'COLLECT_FERTILIZER'):
        if not animal:
            return 'not_animal'
        if op == 'CARE':
            return 'already_cared' if tile['cared_today'] else 'effect'
        if op == 'COLLECT_FERTILIZER':
            return 'effect' if tile['fertilizer_available'] else 'no_fertilizer_yield'
        if tile['fed_today']:
            return 'already_fed'
        return 'effect' if inv.get('WHEAT', 0) >= 1 else 'no_carried_wheat'
    return 'unknown'  # future engine extensions must be explicitly supported


def training_unit_mask(obs, action, *, max_units, engine, configuration=None):
    """(keep, reasons) for canonical/decoded teacher wire actions.

    Applies final full-turn atomic validation before replaying units. Thus an
    earlier PLANT can be masked even while its teacher-forced PASS-suffix prefix
    showed success. The suffix is used ONLY for labels, never input features.
    Extra nonexistent hand requests still count toward atomic demand.
    """
    cfg = configuration or {}
    farm = copy.deepcopy(obs['farms'][int(obs['player'])])
    private = copy.deepcopy(obs['private'])
    for key in ('shed', 'seeds', 'inventories'):
        if key not in private:
            raise ValueError(f'{SCHEMA}: exact own private.{key} required')
    board = int(cfg.get('boardSize', len(farm['tiles'])))
    tpd = max(1, int(cfg.get('turnsPerDay', 24)))
    day = int(obs['step']) // tpd
    cap = int(cfg.get('shedCapacity', 100))
    action = action if isinstance(action, dict) else {}
    hands = action.get('hands', [])
    units = [action.get('farmer', ['PASS']), *(hands if isinstance(hands, list) else [])]
    demand = Counter(a[1] for a in units if isinstance(a, list) and len(a) >= 2
                     and a[0] == 'PLANT' and isinstance(a[1], str))
    blocked = {c for c, n in demand.items() if n > private['seeds'].get(c, 0)}
    keep, reasons = [False] * max_units, ['padding'] * max_units
    for k in range(1 + len(farm['hands'])):
        a = units[k] if k < len(units) else ['PASS']
        atomic = isinstance(a, list) and len(a) >= 2 and a[0] == 'PLANT' and a[1] in blocked
        reason = 'atomic_plant_block' if atomic else unit_reason(
            farm, private, k, a, engine=engine, day=day, board_size=board, shed_capacity=cap)
        if k < max_units:
            keep[k], reasons[k] = reason in ('effect', 'pass', 'unknown'), reason
        # Unknown input is not silently ignored: if the real engine rejects it,
        # the audit aborts rather than manufacturing exact masks downstream.
        engine._apply_unit_action(farm, private, k, ['PASS'] if atomic else a,
                                  board, day, tpd, cap)
    return keep, reasons


@lru_cache(maxsize=8)
def _candidate_commands(vocab):
    # Positive quantities are equivalent for *any effect*, not effect magnitude.
    out = []
    for name in vocab:
        a = name.split(':')
        if a[0] in ('PICKUP', 'PLACE'):
            a.append(1)
        out.append(a)
    return tuple(out)


def feasible_unit_tokens(pe, codec, k):
    """Cheap 44-token mask from current PrefixExecutor. PASS is always allowed.

    Stock alone is NOT the PLANT constraint. The engine counts requests even on
    occupied/locked tiles, so use initial seed stock minus prior requests too.
    The method is read-only: no candidate engine stepping and no whole-farm copy.
    """
    # PrefixExecutor exposes the same engine through its defining module. Avoid
    # a second bundled engine copy, and allow the caller to use tested revisions.
    import sys
    engine = sys.modules[type(pe).__module__].E
    mask = []
    for a in _candidate_commands(tuple(codec.UNIT_VOCAB)):
        reason = unit_reason(pe.farm, pe.private, k, a, engine=engine, day=pe.day,
                             board_size=pe.board_size, shed_capacity=pe.cap)
        ok = reason in ('effect', 'pass', 'unknown')
        if a[0] == 'PLANT' and (a[1] in pe.blocked or pe.requests[a[1]] >= pe.seeds0.get(a[1], 0)):
            ok = False
        mask.append(ok)
    mask[codec.PASS] = True
    return mask


def teacher_supervision_mask(obs, raw_action, wire_action, *, max_units, engine, configuration=None):
    """Remove only labels that are no-ops in BOTH raw and encoded executions.

    Quantity quantization or lenient codec scrubbing can change a later worker's
    state. Do not falsely drop a successful teacher command just because the
    encoded prefix consumed a different quantity. This guard does not claim to
    repair the underlying lossy codec or its already-existing input/history drift.
    Engine-crashing raw commands are conservatively left unmasked and reported.
    """
    kw = dict(max_units=max_units, engine=engine, configuration=configuration)
    wire_keep, wire_reasons = training_unit_mask(obs, wire_action, **kw)
    if raw_action == wire_action:
        return wire_keep, {'disagreed_units': 0, 'raw_error': None, 'reasons': wire_reasons}
    try:
        raw_keep, _ = training_unit_mask(obs, raw_action, **kw)
    except (TypeError, ValueError, KeyError, IndexError, OverflowError) as exc:
        present = min(max_units, 1 + len(obs['farms'][int(obs['player'])]['hands']))
        return [k < present for k in range(max_units)], {
            'disagreed_units': None, 'raw_error': f'{type(exc).__name__}: {exc}', 'reasons': wire_reasons}
    return [w or r for w, r in zip(wire_keep, raw_keep)], {
        'disagreed_units': sum(w != r for w, r in zip(wire_keep, raw_keep)),
        'raw_error': None, 'reasons': wire_reasons}
