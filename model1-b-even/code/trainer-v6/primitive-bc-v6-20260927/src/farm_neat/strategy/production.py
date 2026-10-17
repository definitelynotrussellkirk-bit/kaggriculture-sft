"""Bounded production replanning in dated cohort space; no routes or actions.

Input contract (no imports of parallel workers):
  context.memory['production']: ProductionBook of own cohorts and requested bundles.
  context.derived['production_value']: optional ProductionValuator. It values the
    ENTIRE remaining portfolio, not independently priced new units. Its receipts
    include pooled stock/co-products, its input and labor costs use the same dated
    continuation, and its rival_cash is separate. Missing evidence stays UNKNOWN.
  context.derived['capacity_planner']: optional {Proposal.key: Feasibility} answers.

Output: mutually exclusive KEEP, ADD and RETIRE proposals against the SAME book.
Requirements describe the FULL post-change production remainder, not incremental
resource reservations. The coordinator must replace, not add, these obligations.
KEEP's deltas are identically zero; its remaining obligations are not zero.

The fixed 13 full-service policies reuse unit_economics.simulate, including late
fertilizer-only animal columns. Daily prefix differences expose dated inputs,
worker-held output, acquisition and attempted service. These are conditional
module tapes, NOT successful native routes, delivery, receipts or feasibility.
Actor-built live cohorts retain one receipt-backed RealizedService per site.
Their policy names declare FUTURE service; past events are never rewritten into
that policy. Replay certifies the exact current tile, including neglect, pending
care, unharvested yield and manure. Custody-only PLACE receipts remain in the
history but cannot reconstruct past inventory/transfer quantities. Current stock
comes exclusively from the current own observation. Legacy manually supplied
cohorts still require verified_step == this dawn for their ideal-policy tapes.
Intraday continuation abstains. Live crop retirement stops at its next replant,
retains that harvest, and never refunds sunk acquisition. Live animal exits are
UNKNOWN; unstarted animals/crops may be cancelled. There is no animal resale here.

A valuation provider owns its guards/provenance and full-path price interaction.
It must account for input consumption versus sales, sale timing, shared inventory,
and all induced costs, not sum single-module profits. No zero-rival, zero-wage,
base-price or future-shop forecast is supplied by this module. It enumerates the
requested finite frontier only; omissions and unsupported exits remain explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Protocol

from farm_neat.evaluation.unit_economics import simulate
from farm_neat.game.exact_engine import load_engine
from farm_neat.game.constants import ANIMALS, CROPS, LAND_PRICES, PRODUCTS, SHOPS
from farm_neat.strategy.types import (
    Estimate, Evidence, Feasibility, Proposal, Requirements, StrategyContext,
    UNKNOWN, Verdict, difference,
)

DAYS = 30
LAST_STEP = 718
MEMORY_BOOK_SCHEMA = 'farm-strategy-pipeline-production-book-v1'
BOOK_OWNER_KEY = 'production_book_owner'
BOOK_OWNER_SCHEMA = 'farm-production-book-owner-v1'
SERVICE_RECEIPT_KEY = 'production_service_receipts'
SERVICE_RECEIPT_SCHEMA = 'farm-production-service-receipts-v1'
_SERVICE_OPS = frozenset({
    'PLANT', 'WATER', 'HARVEST', 'FERTILIZE', 'DIG', 'BUILD_COOP', 'BUILD_PASTURE',
    'PLACE', 'FEED', 'COLLECT_FERTILIZER', 'CARE',
})
PARENTS = (
    '91b24bcb7845b98b112d99227b87ec0f25ed6fadbaf4c506183dfe89e858295f',
    '530f64f901790c8f720ce440204faec81514359d512dc1c7acf3d326a442fe9f',
    '4a74a8c92963b7f11d10005c4d6a0d6b58193e92471cb61e224d8817f124a1a6',
    'ac88b236ebfc299cafd98ec976e6363b7ad357f368e4438846419086ccc5201b',
    'a0a40fc0d2447a5b365df0a6d8d1b96f4260144b165f973ed00cb032a6653b8b',
    '6b78c2ce34907ed23f66da8ca58edd912d28cf003026ddf072885cd9e9bb3baf',
)


def _integer(value, lo, hi, name):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return value


def _key(value):
    if not isinstance(value, str) or not value or len(value) > 120 or any(c.isspace() for c in value):
        raise ValueError('key must be a nonempty, whitespace-free string of at most 120 characters')
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()



def _unit_position(farm, unit):
    if unit == 0:
        return farm.get('farmer')
    hands = farm.get('hands')
    return hands[unit - 1] if isinstance(hands, (list, tuple)) and unit - 1 < len(hands) else None


def _receipt_site_key(x, y):
    return f'{x},{y}'


def _receipt_ledger(value, *, player=None):
    """Validate the closed actor-owned service-receipt codec.

    ``through_step`` means every own submitted unit word through that step was
    inspected.  Site rows contain only successful production effects reproduced
    from the actor's own pre-action farm/private state plus the action actually
    submitted by that actor.  They deliberately contain no rival or evaluator
    state and no inferred biological history.
    """
    if (not isinstance(value, Mapping)
            or set(value) != {'schema', 'player', 'through_step', 'segment_start_step', 'sites'}
            or value.get('schema') != SERVICE_RECEIPT_SCHEMA):
        raise ValueError('invalid production service-receipt schema')
    owner = value['player']
    _integer(owner, 0, 1, 'service receipt player')
    if player is not None and owner != player:
        raise ValueError('production service receipts belong to another player')
    through = value['through_step']
    if type(through) is not int or not -1 <= through <= LAST_STEP:
        raise ValueError('service receipt through_step must be -1..718')
    segment = value['segment_start_step']
    if type(segment) is not int or not 0 <= segment <= LAST_STEP + 1:
        raise ValueError('service receipt segment_start_step must be 0..719')
    sites = value['sites']
    if not isinstance(sites, Mapping) or len(sites) > 100:
        raise ValueError('service receipt sites must be a bounded mapping')
    clean = {}
    for key, row in sites.items():
        if (not isinstance(key, str) or not isinstance(row, Mapping)
                or set(row) != {'site', 'producer', 'birth_day', 'acquired_step', 'events'}):
            raise ValueError('invalid service receipt site row')
        site = row['site']
        if (not isinstance(site, (list, tuple)) or len(site) != 2
                or any(type(v) is not int or v < 0 or v > 99 for v in site)
                or key != _receipt_site_key(site[0], site[1])):
            raise ValueError('invalid service receipt site coordinate')
        producer = row['producer']
        if producer not in CROPS and producer not in ANIMALS:
            raise ValueError('invalid service receipt producer')
        birth = _integer(row['birth_day'], 0, DAYS - 1, 'service receipt birth_day')
        acquired = row['acquired_step']
        if type(acquired) is not int or not 0 <= acquired <= through:
            raise ValueError('invalid service receipt acquisition step')
        events = row['events']
        if not isinstance(events, (list, tuple)) or not events:
            raise ValueError('service receipt site needs acquisition evidence')
        cleaned_events, previous = [], None
        for event in events:
            if (not isinstance(event, (list, tuple)) or len(event) != 3
                    or type(event[0]) is not int or not 0 <= event[0] <= through
                    or type(event[1]) is not int or not 0 <= event[1] <= 100
                    or event[2] not in _SERVICE_OPS):
                raise ValueError('invalid production service receipt event')
            item = [event[0], event[1], event[2]]
            order = (item[0], item[1])
            if previous is not None and order < previous:
                raise ValueError('service receipt events must be chronological')
            previous = order
            cleaned_events.append(item)
        acquisition = 'PLANT' if producer in CROPS else 'PLACE'
        if (cleaned_events[0][0] != acquired or cleaned_events[0][2] != acquisition
                or acquired // 24 != birth):
            raise ValueError('service receipt acquisition event is inconsistent')
        clean[key] = dict(site=list(site), producer=producer, birth_day=birth,
                          acquired_step=acquired, events=cleaned_events)
    return dict(schema=SERVICE_RECEIPT_SCHEMA, player=owner, through_step=through,
                segment_start_step=segment, sites=clean)


def _fresh_receipt_ledger(player, *, through_step=-1, segment_start_step=0):
    return dict(schema=SERVICE_RECEIPT_SCHEMA, player=player, through_step=through_step,
                segment_start_step=segment_start_step, sites={})


def update_service_receipts(context: StrategyContext, action: Mapping) -> dict:
    """Advance actor-owned production receipts from the action actually submitted.

    The native unit phase is replayed only against a JSON copy of this actor's
    visible pre-action own farm/private state.  A receipt is appended only when
    that exact submitted unit action causes a production effect in that replay.
    Failed commands therefore cannot certify service.  A step gap or an
    un-replayable production command drops all live site history rather than
    guessing across the missing edge.
    """
    if not isinstance(context, StrategyContext) or not isinstance(action, Mapping):
        raise ValueError('StrategyContext and submitted action mapping required')
    if context.step < 0 or context.step > LAST_STEP:
        raise ValueError('service receipts require an in-game step')
    prior = context.memory.get(SERVICE_RECEIPT_KEY)
    ledger = (_fresh_receipt_ledger(context.player, through_step=context.step - 1,
                                    segment_start_step=context.step)
              if prior is None else _receipt_ledger(prior, player=context.player))
    if ledger['through_step'] != context.step - 1:
        if ledger['through_step'] >= context.step:
            raise ValueError('service receipt time must advance exactly once per submitted step')
        ledger = _fresh_receipt_ledger(context.player, through_step=context.step - 1,
                                       segment_start_step=context.step)

    farmer = action.get('farmer', ['PASS'])
    hands = action.get('hands', [])
    if not isinstance(farmer, list) or not farmer or not isinstance(hands, list):
        raise ValueError('submitted action has invalid unit-word shape')
    unit_actions = [farmer, *hands]
    if any(not isinstance(word, list) or not word for word in unit_actions):
        raise ValueError('submitted unit words must be nonempty lists')
    relevant = any(word[0] in _SERVICE_OPS for word in unit_actions)
    if not relevant:
        ledger['through_step'] = context.step
        return ledger

    farms = context.observation.get('farms')
    private = context.observation.get('private')
    own = farms[context.player] if isinstance(farms, (list, tuple)) and len(farms) > context.player else None
    if (not isinstance(own, Mapping) or not isinstance(private, Mapping)
            or not isinstance(own.get('tiles'), (list, tuple))
            or not isinstance(own.get('farmer'), (list, tuple))
            or not isinstance(own.get('hands'), (list, tuple))
            or not isinstance(private.get('inventories'), (list, tuple))
            or not isinstance(private.get('seeds'), Mapping)
            or not isinstance(private.get('shed'), Mapping)):
        return _fresh_receipt_ledger(context.player, through_step=context.step,
                                     segment_start_step=context.step + 1)
    try:
        farm = json.loads(json.dumps(own, allow_nan=False))
        own_private = json.loads(json.dumps(private, allow_nan=False))
    except (TypeError, ValueError):
        return _fresh_receipt_ledger(context.player, through_step=context.step,
                                     segment_start_step=context.step + 1)
    tiles = farm.get('tiles')
    size = len(tiles)
    if (size < 1 or any(not isinstance(row, list) or len(row) != size for row in tiles)
            or len(own_private['inventories']) < 1):
        return _fresh_receipt_ledger(context.player, through_step=context.step,
                                     segment_start_step=context.step + 1)

    plant_demand = {}
    for word in unit_actions:
        if len(word) >= 2 and word[0] == 'PLANT' and isinstance(word[1], str):
            plant_demand[word[1]] = plant_demand.get(word[1], 0) + 1
    blocked = {crop for crop, count in plant_demand.items()
               if count > own_private['seeds'].get(crop, 0)}
    engine = load_engine()
    for unit, submitted in enumerate(unit_actions):
        pos = _unit_position(farm, unit)
        if (not isinstance(pos, (list, tuple)) or len(pos) != 2
                or any(type(v) is not int for v in pos)):
            if submitted[0] in _SERVICE_OPS:
                return _fresh_receipt_ledger(context.player, through_step=context.step,
                                             segment_start_step=context.step + 1)
            continue
        x, y = pos
        if not (0 <= x < size and 0 <= y < size):
            if submitted[0] in _SERVICE_OPS:
                return _fresh_receipt_ledger(context.player, through_step=context.step,
                                             segment_start_step=context.step + 1)
            continue
        command = (['PASS'] if len(submitted) >= 2 and submitted[0] == 'PLANT'
                   and submitted[1] in blocked else submitted)
        before_tile = json.loads(json.dumps(farm['tiles'][y][x], allow_nan=False))
        before_private = json.dumps(own_private, sort_keys=True, separators=(',', ':'), allow_nan=False)
        try:
            engine._apply_unit_action(farm, own_private, unit, command, size,
                                      context.step // 24, 24, 100)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            if submitted[0] in _SERVICE_OPS:
                return _fresh_receipt_ledger(context.player, through_step=context.step,
                                             segment_start_step=context.step + 1)
            continue
        op = submitted[0]
        if op not in _SERVICE_OPS or command[0] == 'PASS':
            continue
        after_tile = farm['tiles'][y][x]
        after_private = json.dumps(own_private, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if before_tile == after_tile and before_private == after_private:
            continue
        site_key = _receipt_site_key(x, y)
        event = [context.step, unit, op]
        if (op == 'PLANT' and isinstance(after_tile, Mapping)
                and after_tile.get('kind') == 'PLANT'):
            ledger['sites'][site_key] = dict(site=[x, y], producer=after_tile['crop'],
                                             birth_day=after_tile['planted_day'],
                                             acquired_step=context.step, events=[event])
        elif (op == 'PLACE' and isinstance(after_tile, Mapping) and 'animal' in after_tile
              and (not isinstance(before_tile, Mapping) or before_tile.get('animal') != after_tile['animal'])):
            ledger['sites'][site_key] = dict(site=[x, y], producer=after_tile['animal'],
                                             birth_day=after_tile['placed_day'],
                                             acquired_step=context.step, events=[event])
        elif op in ('DIG', 'BUILD_COOP', 'BUILD_PASTURE'):
            ledger['sites'].pop(site_key, None)
        elif op == 'HARVEST' and isinstance(before_tile, Mapping):
            if before_tile.get('kind') == 'PLANT' and not CROPS[before_tile['crop']]['ongoing']:
                ledger['sites'].pop(site_key, None)
            elif site_key in ledger['sites']:
                ledger['sites'][site_key]['events'].append(event)
        elif site_key in ledger['sites']:
            producer = ledger['sites'][site_key]['producer']
            same = ((isinstance(after_tile, Mapping) and after_tile.get('kind') == 'PLANT'
                     and producer == after_tile.get('crop'))
                    or (isinstance(after_tile, Mapping) and producer == after_tile.get('animal')))
            if same:
                ledger['sites'][site_key]['events'].append(event)
            else:
                ledger['sites'].pop(site_key, None)
    ledger['through_step'] = context.step
    return _receipt_ledger(ledger, player=context.player)


def _event_counts(events):
    counts = {}
    for step, _unit, op in events:
        key = (step // 24, op)
        counts[key] = counts.get(key, 0) + 1
    return counts


@lru_cache(maxsize=2048)
def _canonical_service_state(producer, policy, birth_day, current_day):
    """Full-service dawn state/event multiset for one current biological generation.

    This is the same native unit/day machinery used by ``unit_economics``.  It is
    a static game-rule calculation, not a replay of evaluator truth.  Events are
    successful state changes; no-op attempts (for example day-0 manure collection)
    are irrelevant to biological state equivalence.
    """
    if producer not in CROPS and producer not in ANIMALS:
        raise ValueError('unknown producer')
    if not birth_day <= current_day <= DAYS:
        raise ValueError('invalid service-state interval')
    engine = load_engine()
    private = {'inventories': [{'WHEAT': 10_000, 'FERTILIZER': 10_000}],
               'seeds': {crop: 10_000 for crop in CROPS}, 'shed': {}}
    events = []

    def changed(farm, op, day, arg=None):
        before = json.dumps({'farm': farm, 'private': private}, sort_keys=True,
                            separators=(',', ':'), allow_nan=False)
        command = [op] if arg is None else [op, arg]
        engine._apply_unit_action(farm, private, 0, command, 1, day, 24, 100)
        after = json.dumps({'farm': farm, 'private': private}, sort_keys=True,
                           separators=(',', ':'), allow_nan=False)
        if before != after:
            events.append((day, op))

    if producer in CROPS:
        farm = {'tiles': [[None]], 'farmer': [0, 0], 'hands': [], 'money': 0,
                'unlocked_quadrants': ['NW'], 'hires_today': 0}
        crop = CROPS[producer]
        fertilize = policy == 'watered_fertilized'
        if policy not in ('watered', 'watered_fertilized'):
            raise ValueError('unknown crop policy')
        per_water = 2 if fertilize else 1
        start = (crop['max_yield_day'] + 1) // 2
        harvest_age = crop['max_yield_day']
        for age in range(crop['first_yield_day'], crop['max_yield_day'] + 1):
            if 1 + per_water * max(0, age - start + 1) >= crop['max_yield']:
                harvest_age = age
                break
        planted = None
        for day in range(birth_day, current_day):
            tile = farm['tiles'][0][0]
            spent = (isinstance(tile, Mapping) and tile.get('kind') == 'PLANT'
                     and 0 <= tile.get('max_lifespan_step', -1) <= day * 24)
            if isinstance(tile, Mapping) and (tile.get('kind') == 'WEED' or spent):
                changed(farm, 'DIG', day)
                tile = None
            if tile is None:
                changed(farm, 'PLANT', day, producer)
                planted = day
                tile = farm['tiles'][0][0]
            age = day - planted
            if fertilize and tile.get('fertilized_until_day', -1) < day:
                productive = ((start <= age <= crop['max_yield_day']) if not crop['ongoing']
                              else age >= crop['first_yield_day'] - 1)
                if productive:
                    changed(farm, 'FERTILIZE', day)
            changed(farm, 'WATER', day)
            tile = farm['tiles'][0][0]
            if crop['ongoing']:
                if age >= crop['first_yield_day'] and tile.get('yield_units', 0) > 0:
                    changed(farm, 'HARVEST', day)
            elif age >= harvest_age:
                changed(farm, 'HARVEST', day)
                changed(farm, 'PLANT', day, producer)
                planted = day
                changed(farm, 'WATER', day)
            for turn in range(24):
                engine._decay_plants(farm, day * 24 + turn)
            engine._daily_refresh_plants(farm, day, 24)
    else:
        if policy != 'fed_cared':
            raise ValueError('unknown animal policy')
        farm = {'tiles': [[engine._new_animal(producer, birth_day)]], 'farmer': [0, 0],
                'hands': [], 'money': 0, 'unlocked_quadrants': ['NW'], 'hires_today': 0}
        events.append((birth_day, 'PLACE'))
        product = ANIMALS[producer]['product']
        for day in range(birth_day, current_day):
            for op in ('FEED', 'CARE', 'COLLECT_FERTILIZER'):
                changed(farm, op, day)
            tile = farm['tiles'][0][0]
            if isinstance(tile, Mapping) and tile.get('yield_units', 0) > 0:
                before = private['inventories'][0].get(product, 0)
                changed(farm, 'HARVEST', day)
                if private['inventories'][0].get(product, 0) == before:
                    raise RuntimeError('canonical animal harvest did not change custody')
            engine._daily_refresh_animals(farm, day)
    tile = farm['tiles'][0][0]
    signature = None if tile is None else tuple(sorted(tile.items()))
    counts = {}
    for day, op in events:
        counts[(day, op)] = counts.get((day, op), 0) + 1
    return tuple(sorted(counts.items())), signature


def _policy_for_receipts(row, visible_tile, current_day):
    actual = tuple(sorted(_event_counts(row['events']).items()))
    policies = ('watered', 'watered_fertilized') if row['producer'] in CROPS else ('fed_cared',)
    visible_signature = tuple(sorted(visible_tile.items()))
    matches = []
    for policy in policies:
        expected, signature = _canonical_service_state(row['producer'], policy,
                                                       row['birth_day'], current_day)
        if actual == expected and visible_signature == signature:
            matches.append(policy)
    # Before the first fertilizer-relevant day, both crop policies have identical
    # history.  Choosing watered is an explicit low-input continuation choice from
    # this dawn; it is not a claim that past intent was observed.
    return matches[0] if matches else None


@dataclass(frozen=True)
class RealizedService:
    """One live generation, not a full-service-policy label or inventory ledger.

    ``events`` are the unchanged successful own receipts through the prior step;
    ``tile`` is their exact biological replay result, checked against observation.
    A site is never pooled with a causally different site. Future service belongs
    to Cohort.policy. Routes, past custody quantities/cash and future feasibility
    are excluded. Only produce_actor_book establishes this actor-owned binding;
    project_cohorts checks it again before calling any valuation provider.
    """
    site: tuple[int, int]
    acquired_step: int
    through_step: int
    events: tuple[tuple[int, int, str], ...]
    tile: Mapping

    def __post_init__(self):
        site = tuple(self.site)
        if len(site) != 2:
            raise ValueError('realized service requires one site coordinate')
        for v in site:
            _integer(v, 0, 99, 'realized site')
        _integer(self.acquired_step, 0, LAST_STEP, 'realized acquired_step')
        _integer(self.through_step, self.acquired_step, LAST_STEP - 1, 'realized through_step')
        if (self.through_step + 1) % 24:
            raise ValueError('realized service must end immediately before a dawn')
        if not isinstance(self.tile, Mapping):
            raise ValueError('realized tile must be a mapping')
        tile = dict(self.tile)
        producer = tile.get('crop') if tile.get('kind') == 'PLANT' else tile.get('animal')
        birth = tile.get('planted_day') if producer in CROPS else tile.get('placed_day')
        # Reuse the existing receipt codec, not a second permissive event parser.
        row = dict(site=list(site), producer=producer, birth_day=birth,
                   acquired_step=self.acquired_step, events=self.events)
        ledger = _receipt_ledger(dict(schema=SERVICE_RECEIPT_SCHEMA, player=0,
                                      through_step=self.through_step, segment_start_step=0,
                                      sites={_receipt_site_key(*site): row}))
        events = tuple(tuple(e) for e in ledger['sites'][_receipt_site_key(*site)]['events'])
        if len({e[:2] for e in events}) != len(events):
            raise ValueError('duplicate successful service event for one unit step')
        if _hash(tile) != _hash(dict(_replayed_service_tile(producer, birth, self.acquired_step,
                                                            self.through_step, events) or ())):
            raise ValueError('successful service receipts contradict the current tile')
        object.__setattr__(self, 'site', site)
        object.__setattr__(self, 'events', events)
        object.__setattr__(self, 'tile', MappingProxyType(tile))

    def to_dict(self):
        return dict(site=list(self.site), acquired_step=self.acquired_step,
                    through_step=self.through_step, events=[list(e) for e in self.events],
                    tile=dict(self.tile))


def _isolated_farm(tile):
    return dict(tiles=[[tile]], farmer=[0, 0], hands=[], money=0,
                unlocked_quadrants=['NW'], hires_today=0)


@lru_cache(maxsize=4096)
def _replayed_service_tile(producer, birth, acquired, through, events):
    """Exact single-generation biology from successful receipts, no world replay.

    Inputs to each recorded successful FEED/FERTILIZE are supplied conditionally,
    NOT inferred as historical stock. Acquisition is certified by the first event.
    A later PLACE on an occupied live tile can only transfer custody (native rule),
    so its unavailable argument/quantity is not invented. Its event stays retained.
    """
    engine = load_engine()
    tile = (engine._new_plant(producer, birth, 24) if producer in CROPS
            else engine._new_animal(producer, birth))
    farm = _isolated_farm(tile)
    private = dict(inventories=[{}], seeds={}, shed={})
    cursor = 1  # the successful acquisition made the native newborn above
    allowed = ({'WATER', 'FERTILIZE', 'HARVEST', 'PLACE'} if producer in CROPS
               else {'FEED', 'CARE', 'COLLECT_FERTILIZER', 'HARVEST', 'PLACE'})
    for step in range(acquired, through + 1):
        while cursor < len(events) and events[cursor][0] == step:
            op = events[cursor][2]
            current = farm['tiles'][0][0]
            if op not in allowed or not isinstance(current, dict) or (
                    current.get('crop') != producer and current.get('animal') != producer):
                raise ValueError('receipt leaves the supported live biological generation')
            if op != 'PLACE':
                private['inventories'][0] = {'WHEAT': 1, 'FERTILIZER': 1}
                before = _hash(dict(tile=current, private=private))
                engine._apply_unit_action(farm, private, 0, [op], 1, step // 24, 24, 100)
                if before == _hash(dict(tile=farm['tiles'][0][0], private=private)):
                    raise ValueError('successful service receipt replays as a no-op')
            cursor += 1
        engine._decay_plants(farm, step)
        if step % 24 == 23:
            engine._daily_refresh_plants(farm, step // 24, 24)
            engine._daily_refresh_animals(farm, step // 24)
    if cursor != len(events):
        raise ValueError('service events fall outside the acquired generation')
    result = farm['tiles'][0][0]
    # Cache immutable values; no caller can alter a future certification result.
    return None if result is None else tuple(sorted(result.items()))


@dataclass(frozen=True)
class Cohort:
    key: str
    producer: str
    policy: str
    birth_day: int
    count: int
    verified_step: int | None = None
    realized: RealizedService | None = None

    def __post_init__(self):
        _key(self.key)
        _integer(self.birth_day, 0, DAYS - 1, 'birth_day')
        _integer(self.count, 1, 100, 'count')
        policies = ('watered', 'watered_fertilized') if self.producer in CROPS else (
            ('fed_cared',) if self.producer in ANIMALS else ())
        if self.policy not in policies:
            raise ValueError('unknown producer/policy pair')
        if self.verified_step is not None:
            _integer(self.verified_step, self.birth_day * 24, LAST_STEP, 'verified_step')
        if self.realized is not None:
            if not isinstance(self.realized, RealizedService) or self.count != 1:
                raise ValueError('a realized cohort must name exactly one receipt-certified site')
            tile = self.realized.tile
            producer = tile.get('crop') if tile.get('kind') == 'PLANT' else tile.get('animal')
            birth = tile.get('planted_day') if tile.get('kind') == 'PLANT' else tile.get('placed_day')
            if (producer != self.producer or birth != self.birth_day
                    or self.verified_step != self.realized.through_step + 1):
                raise ValueError('cohort and realized service identity differ')

    def to_dict(self):
        return dict(key=self.key, producer=self.producer, policy=self.policy,
                    birth_day=self.birth_day, count=self.count, verified_step=self.verified_step,
                    **({} if self.realized is None else {'realized': self.realized.to_dict()}))


def _cohorts(rows):
    rows = tuple(rows)
    if any(not isinstance(c, Cohort) for c in rows) or len({c.key for c in rows}) != len(rows):
        raise ValueError('cohorts need distinct keys and Cohort values')
    sites = [c.realized.site for c in rows if c.realized is not None]
    if len(set(sites)) != len(sites):
        raise ValueError('duplicate realized production site')
    if len(rows) > 100 or sum(c.count for c in rows) > 100:
        raise ValueError('cohort frontier is bounded to 100 simultaneous module sites')
    return rows


@dataclass(frozen=True)
class Expansion:
    key: str
    cohorts: tuple[Cohort, ...]
    land_quadrants: int = 0  # requested additional land, not a minimal-land certificate

    def __post_init__(self):
        _key(self.key)
        object.__setattr__(self, 'cohorts', _cohorts(self.cohorts))
        if not self.cohorts:
            raise ValueError('an expansion needs at least one cohort')
        if any(c.verified_step is not None for c in self.cohorts):
            raise ValueError('expansion cohorts must be unstarted')
        _integer(self.land_quadrants, 0, 3, 'land_quadrants')

    def to_dict(self):
        return dict(key=self.key, cohorts=[c.to_dict() for c in self.cohorts],
                    land_quadrants=self.land_quadrants)


@dataclass(frozen=True)
class ProductionBook:
    cohorts: tuple[Cohort, ...] = ()
    expansions: tuple[Expansion, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'cohorts', _cohorts(self.cohorts))
        items = tuple(self.expansions)
        if len(items) > 256 or any(not isinstance(e, Expansion) for e in items):
            raise ValueError('at most 256 typed expansion requests')
        if len({e.key for e in items}) != len(items):
            raise ValueError('duplicate expansion key')
        for e in items:
            _cohorts(self.cohorts + e.cohorts)
        object.__setattr__(self, 'expansions', items)

    def to_dict(self):
        return dict(cohorts=[c.to_dict() for c in self.cohorts],
                    expansions=[e.to_dict() for e in self.expansions])


def book_from_memory(value) -> 'ProductionBook':
    """Rehydrate the closed actor-memory codec used by StrategyPipeline.

    This accepts only ProductionBook itself or the one named JSON envelope.  It is
    intentionally not a permissive mapping-to-object constructor.
    """
    if isinstance(value, ProductionBook):
        return value
    if (not isinstance(value, Mapping) or set(value) != {'schema', 'cohorts', 'expansions'}
            or value.get('schema') != MEMORY_BOOK_SCHEMA):
        raise ValueError('production memory requires a ProductionBook or its pipeline JSON codec')

    def cohort(row):
        fields = {'key', 'producer', 'policy', 'birth_day', 'count', 'verified_step'}
        if not isinstance(row, Mapping) or set(row) not in (fields, fields | {'realized'}):
            raise ValueError('invalid serialized production cohort')
        data = dict(row)
        if 'realized' in data:
            realized = data['realized']
            if (not isinstance(realized, Mapping)
                    or set(realized) != {'site', 'acquired_step', 'through_step', 'events', 'tile'}):
                raise ValueError('invalid serialized realized service')
            data['realized'] = RealizedService(**realized)
        return Cohort(**data)

    expansions = []
    for row in value['expansions']:
        if not isinstance(row, Mapping) or set(row) != {'key', 'cohorts', 'land_quadrants'}:
            raise ValueError('invalid serialized expansion')
        expansions.append(Expansion(row['key'], tuple(cohort(c) for c in row['cohorts']),
                                    row['land_quadrants']))
    return ProductionBook(tuple(cohort(c) for c in value['cohorts']), tuple(expansions))


def actor_snapshot_sha256(context: StrategyContext, book: ProductionBook) -> str:
    """Actor-derived proposal binding; never an evaluator Snapshot identity."""
    if not isinstance(book, ProductionBook):
        raise ValueError('typed ProductionBook required for actor snapshot binding')
    return _hash(dict(step=context.step, player=context.player,
                      observation=dict(context.observation), shops=published_shops(context),
                      book=book.to_dict()))


@dataclass(frozen=True)
class ProductionBookBuild:
    """Conservative actor-side bootstrap result.  A refusal is explicit."""
    book: ProductionBook | None
    reason: str | None = None

    def __post_init__(self):
        if (self.book is None) == (self.reason is None):
            raise ValueError('book build returns exactly one of book or refusal reason')
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason):
            raise ValueError('book refusal reason must be nonempty')


def produce_actor_book(context: StrategyContext) -> ProductionBookBuild:
    """Build the finite dawn frontier from actor-visible state and own receipts.

    A clean farm still needs no history.  A live site is admitted only when the
    actor-owned receipt ledger is continuous through the preceding step, names the
    same acquisition, contains the successful own production effects for that
    generation, and those effects reproduce the exact current native biological
    state. Past realized service is separate from the declared future policy:
    watered (crops) or fed_cared (animals). No past full-service intent is inferred.
    Current tile state is only a consistency check; it can never create or advance
    ``verified_step`` by itself.
    """
    if not isinstance(context, StrategyContext):
        raise ValueError('StrategyContext required')
    if context.step % 24:
        return ProductionBookBuild(None, 'BOOK_DAWN_REQUIRED')
    for field, value in (('step', context.step), ('player', context.player)):
        if field in context.observation and context.observation[field] != value:
            raise ValueError('context and observation identity differ')
    farms = context.observation.get('farms')
    if (not isinstance(farms, (tuple, list)) or len(farms) <= context.player
            or not isinstance(farms[context.player], Mapping)):
        return ProductionBookBuild(None, 'BOOK_MISSING_OWN_FARM')
    farm = farms[context.player]
    if 'tiles' not in farm or 'unlocked_quadrants' not in farm:
        return ProductionBookBuild(None, 'BOOK_MISSING_OWN_FARM_GEOMETRY')
    tiles = farm['tiles']
    if (not isinstance(tiles, (tuple, list)) or not tiles
            or any(not isinstance(row, (tuple, list)) for row in tiles)):
        return ProductionBookBuild(None, 'BOOK_MISSING_OWN_TILE_GRID')
    stock, _, _ = _own_inventory(context)
    if stock is None:
        return ProductionBookBuild(None, 'BOOK_MISSING_OWN_INVENTORY')
    live = []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if isinstance(tile, Mapping) and (tile.get('kind') == 'PLANT' or 'animal' in tile):
                live.append((x, y, tile))

    cohorts = []
    if live:
        raw = context.memory.get(SERVICE_RECEIPT_KEY)
        if raw is None:
            return ProductionBookBuild(None, 'BOOK_MISSING_SERVICE_RECEIPTS')
        try:
            receipts = _receipt_ledger(raw, player=context.player)
        except ValueError:
            return ProductionBookBuild(None, 'BOOK_INCONSISTENT_SERVICE_RECEIPTS')
        if receipts['through_step'] != context.step - 1:
            return ProductionBookBuild(None, 'BOOK_INCOMPLETE_SERVICE_RECEIPTS')
        day = context.step // 24
        for x, y, tile in live:
            row = receipts['sites'].get(_receipt_site_key(x, y))
            if row is None:
                return ProductionBookBuild(None, 'BOOK_MISSING_SERVICE_RECEIPTS')
            producer = tile.get('crop') if tile.get('kind') == 'PLANT' else tile.get('animal')
            birth = tile.get('planted_day') if tile.get('kind') == 'PLANT' else tile.get('placed_day')
            if (producer != row['producer'] or birth != row['birth_day']
                    or row['acquired_step'] < receipts['segment_start_step']):
                return ProductionBookBuild(None, 'BOOK_INCONSISTENT_SERVICE_RECEIPTS')
            # Bind every site separately; equal counts or equal visible states do
            # not identify the same service history/custody/route obligations.
            try:
                realized = RealizedService((x, y), row['acquired_step'], receipts['through_step'],
                                           tuple(tuple(e) for e in row['events']), tile)
            except ValueError:
                return ProductionBookBuild(None, 'BOOK_INCONSISTENT_SERVICE_RECEIPTS')
            policy = 'watered' if producer in CROPS else 'fed_cared'
            key = f'actor-live-v1:{producer.lower()}:{x},{y}:' + _hash(realized.to_dict())[:16]
            cohorts.append(Cohort(key, producer, policy, birth, 1, context.step, realized))

    # A request on already-owned land needs some actor-visible empty site.  Locked
    # cells and bare structures are not silently treated as free production sites.
    expansions = []
    if any(tile is None for row in tiles for tile in row) and sum(c.count for c in cohorts) < 100:
        day = context.step // 24
        for producer in sorted(CROPS):
            for policy in ('watered', 'watered_fertilized'):
                key = f'actor-frontier-v1:{producer.lower()}:{policy}:d{day:02d}'
                expansions.append(Expansion(key, (Cohort(key, producer, policy, day, 1),), 0))
        for producer in sorted(ANIMALS):
            key = f'actor-frontier-v1:{producer.lower()}:fed_cared:d{day:02d}'
            expansions.append(Expansion(key, (Cohort(key, producer, 'fed_cared', day, 1),), 0))
    return ProductionBookBuild(ProductionBook(tuple(cohorts), tuple(expansions)))


@dataclass(frozen=True)
class ProductionValue:
    """Total REMAINING scenario flows; acquisition/land are subtracted separately.

    A provider may report EXACT only under its explicit guards. UNKNOWN is not
    zero. rival_cash is the scenario's rival continuation value, not own receipts.
    Fixed producer/land acquisition must not also be charged in input/labor_cash.
    """
    receipts: Estimate = UNKNOWN
    input_cash: Estimate = UNKNOWN
    labor_cash: Estimate = UNKNOWN
    rival_cash: Estimate = UNKNOWN
    guards: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ('receipts', 'input_cash', 'labor_cash', 'rival_cash'):
            if not isinstance(getattr(self, name), Estimate):
                raise ValueError(f'{name} must be an Estimate')
        for name in ('guards', 'parents'):
            items = tuple(getattr(self, name))
            if any(not isinstance(s, str) or not s.strip() for s in items):
                raise ValueError(f'{name} must contain nonempty strings')
            object.__setattr__(self, name, items)
        if any(getattr(self, n).known for n in ('receipts', 'input_cash', 'labor_cash', 'rival_cash')):
            if not self.guards or not self.parents:
                raise ValueError('known valuation needs guards and source parents')

    def own(self, tape):
        value = difference(difference(self.receipts, self.input_cash), self.labor_cash)
        return difference(difference(value, Estimate.exact(sum(tape.fixed_cash))), tape.land_cash)

    def to_dict(self):
        return {**{n: getattr(self, n).to_dict() for n in ('receipts', 'input_cash', 'labor_cash', 'rival_cash')},
                'guards': list(self.guards), 'parents': list(self.parents)}


class ProductionValuator(Protocol):
    def quote(self, context: StrategyContext, tape: 'PortfolioTape') -> ProductionValue: ...


@dataclass(frozen=True)
class PortfolioTape:
    from_day: int
    outputs: Mapping[str, tuple[int, ...]]
    inputs: Mapping[str, tuple[int, ...]]
    fixed_cash: tuple[int, ...]
    actions: tuple[int, ...]
    sites: tuple[int, ...]
    stock: Mapping[str, int] | None
    history_certified: bool
    land_cash: Estimate = field(default_factory=lambda: Estimate.exact(0))

    def __post_init__(self):
        for name in ('outputs', 'inputs'):
            object.__setattr__(self, name, MappingProxyType({k: tuple(v) for k, v in getattr(self, name).items()}))
        if self.stock is not None:
            object.__setattr__(self, 'stock', MappingProxyType(dict(self.stock)))
        for name in ('fixed_cash', 'actions', 'sites'):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def prefix_purchases(self):
        """Necessary pooled-input lower bound, output-before-use within each day.

        Not a native funding/shed/custody certificate; selling internal inputs
        can require MORE purchases. Sum cohorts BEFORE taking prefix maxima.
        """
        if self.stock is None or not self.history_certified:
            return None
        result = {}
        for p, consumed in self.inputs.items():
            deficit, peak = -self.stock.get(p, 0), 0
            for day in range(self.from_day, DAYS):
                deficit += consumed[day] - self.outputs[p][day]
                peak = max(peak, deficit)
            result[p] = peak
        return result

    def to_dict(self):
        return dict(from_day=self.from_day, outputs={p: list(v) for p, v in self.outputs.items()},
                    inputs={p: list(v) for p, v in self.inputs.items()}, fixed_cash=list(self.fixed_cash),
                    actions=list(self.actions), sites=list(self.sites),
                    stock=None if self.stock is None else dict(self.stock),
                    history_certified=self.history_certified, land_cash=self.land_cash.to_dict())


def published_shops(context):
    """Use the ordered prefix, never a shop-count vector.

    types.py currently rejects duplicate strings. For that source version ONLY,
    an empty context.shops may use the actor's native ordered unlocked_shops.
    Nonempty inconsistent prefixes are invalid, not silently merged/truncated.
    """
    native = context.observation.get('town', {}).get('unlocked_shops')
    if native is not None:
        if not isinstance(native, (list, tuple)):
            raise ValueError('native shop prefix must be ordered')
        native = tuple(native)
        if context.shops and context.shops != native:
            raise ValueError('conflicting ordered shop prefix')
    shops = context.shops or native or ()
    if any(p not in SHOPS for p in shops):
        raise ValueError('unknown shop in published prefix')
    return tuple(shops)


@lru_cache(maxsize=13)
def _relative_tape(producer, policy):
    """Prefix differences of the existing native one-tile policy, not new economics."""
    keys = ('primary', 'manure', 'wheat', 'fertilizer', 'fixed_cash', 'actions', 'seeds')
    rows = {k: [] for k in keys}
    previous = dict(units=0, fertilizer_collected=0, wheat_fed=0, fertilizer_used=0,
                    seeds_used=0, actions={})
    for days in range(1, DAYS + 1):
        now = simulate(producer, policy, days)
        for dest, source in (('primary', 'units'), ('manure', 'fertilizer_collected'),
                             ('wheat', 'wheat_fed'), ('fertilizer', 'fertilizer_used'), ('seeds', 'seeds_used')):
            rows[dest].append(now[source] - previous[source])
        rows['actions'].append(sum(now['actions'].values()) - sum(previous['actions'].values()))
        rows['fixed_cash'].append(CROPS[producer]['seed'] * rows['seeds'][-1] if producer in CROPS
                                  else ANIMALS[producer]['cost'] if days == 1 else 0)
        previous = now
    return MappingProxyType({k: tuple(v) for k, v in rows.items()})


def _own_inventory(context):
    private = context.observation.get('private')
    if not isinstance(private, Mapping) or not all(k in private for k in ('shed', 'inventories', 'seeds')):
        return None, {}, {}
    if not isinstance(private['inventories'], (list, tuple)):
        raise ValueError('own carried inventories must be a sequence')
    stock = dict.fromkeys((*PRODUCTS, *ANIMALS), 0)
    for container in (private['shed'], *private['inventories']):
        if not isinstance(container, Mapping):
            raise ValueError('own inventory must be a mapping')
        for p in stock:
            n = container.get(p, 0)
            if type(n) is not int or n < 0:
                raise ValueError('own inventory units must be nonnegative integers')
            stock[p] += n
    seeds = private['seeds']
    if not isinstance(seeds, Mapping) or any(type(n) is not int or n < 0 for n in seeds.values()):
        raise ValueError('own seeds must be nonnegative integer counts')
    return {p: stock[p] for p in PRODUCTS}, dict(seeds), {a: stock[a] for a in ANIMALS}


def _realized_binding(context, cohort, receipts):
    """Recheck current actor state/receipt identity; stale book is not a certificate."""
    state = cohort.realized
    if state is None or cohort.verified_step != context.step or receipts is None:
        return False
    if receipts['through_step'] != context.step - 1:
        return False
    row = receipts['sites'].get(_receipt_site_key(*state.site))
    if (row is None or row['acquired_step'] < receipts['segment_start_step']
            or row['producer'] != cohort.producer or row['birth_day'] != cohort.birth_day
            or row['acquired_step'] != state.acquired_step
            or tuple(tuple(e) for e in row['events']) != state.events):
        return False
    farms = context.observation.get('farms', ())
    if not isinstance(farms, (tuple, list)) or len(farms) <= context.player:
        return False
    farm = farms[context.player]
    tiles = farm.get('tiles', ()) if isinstance(farm, Mapping) else ()
    x, y = state.site
    if (not isinstance(tiles, (tuple, list)) or len(tiles) <= y
            or not isinstance(tiles[y], (tuple, list)) or len(tiles[y]) <= x):
        return False
    return _hash(tiles[y][x]) == _hash(dict(state.tile))


@lru_cache(maxsize=4096)
def _live_tape(producer, policy, from_day, tile_signature, stop_day=None):
    """Conditional native one-tile remainder from realized biology, in absolute days.

    Same service policy as unit_economics.simulate, now starting at this dawn's
    age/neglect/held yield/care state. Inputs are supplied conditionally and counted,
    not treated as available stock. This is neither a route nor terminal valuation.
    """
    from farm_neat.evaluation.unit_economics import _one_shot_harvest_age

    engine = load_engine()
    farm = _isolated_farm(dict(tile_signature))
    private = dict(inventories=[{'WHEAT': 10_000, 'FERTILIZER': 10_000}],
                   seeds={crop: 10_000 for crop in CROPS}, shed={})
    rows = {k: [0] * DAYS for k in ('primary', 'manure', 'wheat', 'fertilizer',
                                   'fixed_cash', 'actions', 'seeds')}
    crop = CROPS.get(producer)
    product = producer if crop else ANIMALS[producer]['product']
    fertilize = policy == 'watered_fertilized'
    harvest_age = _one_shot_harvest_age(crop, 2 if fertilize else 1) if crop else None

    def do(op, day):
        inv = private['inventories'][0]
        before, seeds = dict(inv), private['seeds'].get(producer, 0)
        engine._apply_unit_action(farm, private, 0,
                                  [op, producer] if op == 'PLANT' else [op], 1, day, 24, 100)
        rows['actions'][day] += 1
        if op == 'HARVEST':
            rows['primary'][day] += inv.get(product, 0) - before.get(product, 0)
        elif op == 'COLLECT_FERTILIZER':
            rows['manure'][day] += inv.get('FERTILIZER', 0) - before.get('FERTILIZER', 0)
        elif op == 'FEED':
            rows['wheat'][day] += before.get('WHEAT', 0) - inv.get('WHEAT', 0)
        elif op == 'FERTILIZE':
            rows['fertilizer'][day] += before.get('FERTILIZER', 0) - inv.get('FERTILIZER', 0)
        elif op == 'PLANT':
            used = seeds - private['seeds'].get(producer, 0)
            rows['seeds'][day] += used
            rows['fixed_cash'][day] += used * crop['seed']

    for day in range(from_day, DAYS):
        if crop:
            tile = farm['tiles'][0][0]
            spent = (isinstance(tile, dict) and tile.get('kind') == 'PLANT'
                     and 0 <= tile.get('max_lifespan_step', -1) <= day * 24)
            if isinstance(tile, dict) and (tile.get('kind') == 'WEED' or spent):
                do('DIG', day)
                tile = None
            if tile is None:
                if day == stop_day:
                    break
                do('PLANT', day)
                tile = farm['tiles'][0][0]
            age = day - tile['planted_day']
            start = (crop['max_yield_day'] + 1) // 2
            productive = (start <= age <= crop['max_yield_day'] if not crop['ongoing']
                          else age >= crop['first_yield_day'] - 1)
            if fertilize and tile['fertilized_until_day'] < day and productive:
                do('FERTILIZE', day)
            do('WATER', day)
            if crop['ongoing']:
                if age >= crop['first_yield_day'] and tile['yield_units'] > 0:
                    do('HARVEST', day)
            elif age >= harvest_age:
                do('HARVEST', day)
                if day == stop_day:
                    break
                do('PLANT', day)
                do('WATER', day)
            for turn in range(24):
                engine._decay_plants(farm, day * 24 + turn)
            engine._daily_refresh_plants(farm, day, 24)
        else:
            for op in ('FEED', 'CARE', 'COLLECT_FERTILIZER'):
                do(op, day)
            if farm['tiles'][0][0]['yield_units'] > 0:
                do('HARVEST', day)
            engine._daily_refresh_animals(farm, day)
    return MappingProxyType({key: tuple(values) for key, values in rows.items()})


def _cohort_tape(cohort, day, stop=None):
    if cohort.realized is not None:
        return _live_tape(cohort.producer, cohort.policy, day,
                          tuple(sorted(cohort.realized.tile.items())), stop)
    relative = _relative_tape(cohort.producer, cohort.policy)
    return {key: (0,) * cohort.birth_day + values[:DAYS - cohort.birth_day]
            for key, values in relative.items()}


def project_cohorts(context: StrategyContext, cohorts: tuple[Cohort, ...], *, stop_replacement=None,
                    land_quadrants=0) -> PortfolioTape:
    """Project one full conditional remainder; stop_replacement is (cohort key, day)."""
    if context.step % 24:
        raise ValueError('daily production projection requires a dawn')
    day = context.step // 24
    _integer(land_quadrants, 0, 3, 'land_quadrants')
    cohorts = _cohorts(cohorts)
    if stop_replacement is not None:
        if not isinstance(stop_replacement, tuple) or len(stop_replacement) != 2:
            raise ValueError('stop_replacement must be a (cohort key, replacement day) tuple')
        key, stop = stop_replacement
        cohort = next((c for c in cohorts if c.key == key), None)
        if (cohort is None or cohort.producer not in CROPS or type(stop) is not int
                or not max(day, cohort.birth_day + 1) <= stop < DAYS
                or not _cohort_tape(cohort, day)['seeds'][stop]):
            raise ValueError('stop_replacement must name a future native crop replacement')
    outputs = {p: [0] * DAYS for p in PRODUCTS}
    inputs = {p: [0] * DAYS for p in ('WHEAT', 'FERTILIZER')}
    fixed, actions, sites = ([0] * DAYS for _ in range(3))
    seed_use = {p: [0] * DAYS for p in CROPS}
    animal_use = {p: [0] * DAYS for p in ANIMALS}
    history = True
    receipts = None
    if any(c.realized is not None for c in cohorts):
        try:
            receipts = _receipt_ledger(context.memory.get(SERVICE_RECEIPT_KEY), player=context.player)
        except ValueError:
            pass  # stale/missing evidence remains uncertified; never call a valuator
    for c in cohorts:
        history &= (c.birth_day >= day or c.verified_step == context.step)
        # A preplaced newborn needs a partial-day acquisition/service trace that
        # this daily interface does not claim to restore.
        history &= not (c.birth_day == day and c.verified_step is not None)
        if c.realized is not None:
            history &= _realized_binding(context, c, receipts)
        product = ANIMALS[c.producer]['product'] if c.producer in ANIMALS else c.producer
        stop = stop_replacement[1] if stop_replacement and stop_replacement[0] == c.key else None
        row = _cohort_tape(c, day, stop)
        for d in range(max(day, c.birth_day), DAYS):
            if stop is not None and d > stop:
                continue
            vals = {k: row[k][d] for k in row}
            if d == stop and c.realized is None:
                if CROPS[c.producer]['ongoing']:
                    vals = dict.fromkeys(vals, 0)
                    vals['actions'] = 1  # cleanup of the spent plant, NOT a replacement
                else:
                    vals['actions'] -= 2  # omit replant and its extra watering; retain harvest/care
                    vals['fixed_cash'] -= CROPS[c.producer]['seed']
                    vals['seeds'] -= 1
            outputs[product][d] += c.count * vals['primary']
            outputs['FERTILIZER'][d] += c.count * vals['manure']
            inputs['WHEAT'][d] += c.count * vals['wheat']
            inputs['FERTILIZER'][d] += c.count * vals['fertilizer']
            fixed[d] += c.count * vals['fixed_cash']
            actions[d] += c.count * vals['actions']
            sites[d] += c.count  # occupied during this day; exact release timing is the planner's job
            if c.producer in CROPS:
                seed_use[c.producer][d] += c.count * vals['seeds']
            elif d == c.birth_day:
                animal_use[c.producer][d] += c.count
    stock, seeds, animals = _own_inventory(context)
    for p, used in seed_use.items():
        held = seeds.get(p, 0)
        for d in range(day, DAYS):
            supplied = min(held, used[d]); held -= supplied
            fixed[d] -= supplied * CROPS[p]['seed']
    for animal, used in animal_use.items():
        held = animals.get(animal, 0)
        for d in range(day, DAYS):
            supplied = min(held, used[d]); held -= supplied
            fixed[d] -= supplied * ANIMALS[animal]['cost']
    land = Estimate.exact(0)
    if land_quadrants:
        farms = context.observation.get('farms', [])
        own = farms[context.player] if len(farms) > context.player else {}
        owned = own.get('unlocked_quadrants')
        n = len(owned) if isinstance(owned, (tuple, list)) else 0
        land = (Estimate.exact(sum(LAND_PRICES[n - 1:n - 1 + land_quadrants]))
                if 1 <= n <= 4 and n + land_quadrants <= 4 else UNKNOWN)
    return PortfolioTape(day, outputs, inputs, fixed, actions, sites, stock, bool(history), land)


def _requirements(tape, land_quadrants):
    buys = tape.prefix_purchases()
    valid = tape.history_certified and tape.stock is not None
    # $1 is the public input-price floor. No future receipts fund this lower bound.
    cash = UNKNOWN
    if valid:
        lower = sum(tape.fixed_cash) + sum(buys.values())
        if tape.land_cash.kind is Evidence.EXACT:
            lower += tape.land_cash.value
        cash = Estimate.bound(low=lower)
    return Requirements(cash=cash, reserve=UNKNOWN,
                        inventory={p: Estimate.exact(sum(q)) if valid else UNKNOWN
                                   for p, q in tape.inputs.items() if any(q)},
                        market_slots=UNKNOWN,
                        worker_hours=Estimate.bound(low=sum(tape.actions)) if valid else UNKNOWN,
                        land_quadrants=Estimate.exact(land_quadrants),
                        window=(24 * tape.from_day, LAST_STEP))


@dataclass(frozen=True)
class ProductionAnalysis:
    proposals: tuple[Proposal, ...]
    diagnostics: tuple[str, ...]

    def to_dict(self):
        return dict(proposals=[p.to_dict() for p in self.proposals], diagnostics=list(self.diagnostics))


class ProductionReplanner:
    name = 'production'

    def __init__(self, max_changes=32):
        self.max_changes = _integer(max_changes, 0, 256, 'max_changes')

    def propose(self, context: StrategyContext):
        return self.analyze(context).proposals

    def analyze(self, context: StrategyContext) -> ProductionAnalysis:
        book = context.memory.get('production')
        if book is None:
            return ProductionAnalysis((), ('NO_PRODUCTION_BOOK',))
        if not isinstance(book, ProductionBook):
            raise ValueError('memory.production must be a ProductionBook')
        shops = published_shops(context)
        if context.step % 24:
            return ProductionAnalysis((), ('DAWN_REQUIRED',))
        for field, value in (('step', context.step), ('player', context.player)):
            if field in context.observation and context.observation[field] != value:
                raise ValueError('context and observation identity differ')
        diagnostics = []
        for c in book.cohorts:
            if c.birth_day < context.step // 24 and c.verified_step != context.step:
                diagnostics.append('UNVERIFIED_HISTORY:' + c.key)
        baseline = project_cohorts(context, book.cohorts)
        provider = context.derived.get('production_value')
        if provider is not None and not callable(getattr(provider, 'quote', None)):
            raise ValueError('derived.production_value needs quote(context, tape)')

        def quote(tape):
            if provider is None or not tape.history_certified or tape.stock is None:
                return ProductionValue()
            value = provider.quote(context, tape)
            if not isinstance(value, ProductionValue):
                raise ValueError('quote must return ProductionValue')
            return value

        base_value = quote(baseline)
        # Include ALL actor observation data, including routes/custody, in the key
        # so a capacity witness cannot survive an otherwise value-neutral move.
        snapshot = actor_snapshot_sha256(context, book)
        proposals = []

        def emit(kind, key, tape, changes, land=0):
            value = base_value if kind == 'KEEP' else quote(tape)
            own = Estimate.exact(0) if kind == 'KEEP' else difference(value.own(tape), base_value.own(baseline))
            rival = Estimate.exact(0) if kind == 'KEEP' else difference(value.rival_cash, base_value.rival_cash)
            margin = difference(own, rival)
            low = margin.value if margin.kind is Evidence.EXACT else margin.low if margin.kind is Evidence.BOUND else None
            high = margin.value if margin.kind is Evidence.EXACT else margin.high if margin.kind is Evidence.BOUND else None
            assessment = ('KEEP' if kind == 'KEEP' else 'POSITIVE_MARGIN' if low is not None and low > 0
                          else 'NEGATIVE_MARGIN' if high is not None and high < 0 else 'UNKNOWN')
            decision = f'{kind.lower()}:{key}:' + _hash(dict(snapshot=snapshot, tape=tape.to_dict(), changes=changes,
                                                          control=base_value.to_dict(), value=value.to_dict()))
            answer = Feasibility((self.name, decision), Verdict.UNKNOWN)
            answers = context.derived.get('capacity_planner', {})
            if not isinstance(answers, Mapping):
                raise ValueError('capacity_planner must map proposal keys to Feasibility')
            if (self.name, decision) in answers:
                answer = answers[self.name, decision]
                if not isinstance(answer, Feasibility) or answer.proposal != (self.name, decision):
                    raise ValueError('capacity answer must bind this exact proposal key')
            resources = {'production', 'cash', 'labor', 'land'}
            resources.update('inventory:' + p for p, q in tape.inputs.items() if any(q))
            resources.update('market:' + p for p, q in tape.outputs.items() if any(q))
            guards = ('same actor snapshot and ordered published-shop prefix',
                      ('receipt-backed live state; declared FUTURE service is conditional'
                       if any(c.realized is not None for c in book.cohorts)
                       else 'declared birth dates and full-service module tapes are realized'),
                      'requirements are the full production remainder, not additive reservations',
                      'valuation includes dated delivery, input allocation and induced labor costs',
                      'other common cash flows cancel only under the valuation provider guards')
            proposals.append(Proposal(
                self.name, decision,
                dict(schema='farm-production-proposal-v1', kind=kind, request_key=key, changes=changes,
                     baseline_sha256=_hash(baseline.to_dict()), snapshot_sha256=snapshot,
                     published_shops=list(shops), requirements_scope='FULL_PRODUCTION_REMAINDER',
                     tape=tape.to_dict(), minimum_purchases=tape.prefix_purchases(),
                     assessment=assessment, valuation=value.to_dict(),
                     feasibility=dict(verdict=answer.verdict.value, witness=answer.witness,
                                      certificate=answer.certificate, marginal=answer.marginal.to_dict(),
                                      parents=list(answer.parents))),
                tuple(resources), _requirements(tape, land), own, rival, margin,
                tuple(dict.fromkeys((*guards, *base_value.guards, *value.guards))),
                ('new dawn or shop', 'birth/service/custody history changes', 'input or receipt evidence changes',
                 'cash, labor, land, inventory or route changes'),
                tuple(sorted(set((*PARENTS, *base_value.parents, *value.parents, *answer.parents)))),
            ))

        emit('KEEP', 'baseline', baseline, {})
        changes = []
        for e in sorted(book.expansions, key=lambda e: e.key):
            if any(c.birth_day < baseline.from_day for c in e.cohorts):
                diagnostics.append('PAST_BIRTH:' + e.key)
                continue
            changes.append(('ADD', e.key, book.cohorts + e.cohorts,
                            {'add': [c.to_dict() for c in e.cohorts]}, e.land_quadrants, None))
        for c in sorted(book.cohorts, key=lambda c: c.key):
            if c.birth_day >= baseline.from_day and c.verified_step is None:
                changes.append(('RETIRE', c.key, tuple(x for x in book.cohorts if x.key != c.key),
                                {'retire': dict(cohort=c.key, count=c.count, mode='CANCEL_UNSTARTED')}, 0, None))
            elif c.verified_step != context.step:
                diagnostics.append('LIVE_EXIT_UNKNOWN:' + c.key)
            elif c.producer in ANIMALS:
                diagnostics.append('LIVE_EXIT_UNKNOWN:' + c.key)
            else:
                row = _cohort_tape(c, baseline.from_day)
                stop = next((d for d in range(max(baseline.from_day, c.birth_day + 1), DAYS)
                             if row['seeds'][d]), None)
                if stop is None:
                    diagnostics.append('NO_REPLACEMENT_IN_HORIZON:' + c.key)
                else:
                    changes.append(('RETIRE', c.key, book.cohorts,
                                    {'retire': dict(cohort=c.key, count=c.count, mode='STOP_REPLACEMENT', day=stop)},
                                    0, (c.key, stop)))
        for i, (kind, key, cohorts, delta, land, stop) in enumerate(changes):
            if i >= self.max_changes:
                diagnostics.append(f'UNKNOWN_BUDGET:{kind}:{key}')
                continue
            target = project_cohorts(context, cohorts, stop_replacement=stop, land_quadrants=land)
            emit(kind, key, target, delta, land)
        return ProductionAnalysis(tuple(proposals), tuple(diagnostics))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', action='store_true', help='Print an unvalued, unexecuted one-cohort request')
    args = parser.parse_args(argv)
    if args.example:
        book = ProductionBook(expansions=(Expansion('example', (Cohort('example', 'COW', 'fed_cared', 0, 1),)),))
        ctx = StrategyContext(0, 0, {}, memory={'production': book})
        print(json.dumps(ProductionReplanner().analyze(ctx).to_dict(), sort_keys=True, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
