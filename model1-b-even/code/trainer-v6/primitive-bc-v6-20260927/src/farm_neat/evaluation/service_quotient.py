"""Exact animal-day consequence compression; read-only, not a routing policy.

The native animal DP remains the sole transition implementation used here. This
adapter joins successful unit effects to it, checks every saved day boundary,
and exports separate biological and cash targets. A biological alias is NOT a
whole-game/value alias. Runnable commands: docs/guides/service-quotient.md.

No world initialization, gameplay, model forward, fitting, or network access.

## The animal-day service quotient theorem

Fix one native animal's species, placement day, actionable horizon and biological
state **before the first service of a day**. Conditional on successful services
at that animal, its next biological state and total harvested product, collected
fertilizer and consumed feed depend only on four daily Boolean flags:

`(FEED, CARE, HARVEST, COLLECT_FERTILIZER)`.

They do not depend on worker identities, intraday service order, movement paths or
service hours within that day. Therefore two complete physical traces with the
same birth calendar, starting biological state and successful daily service flags
have identical day-boundary biology and identical daily transfers from the animal.
This is a **component-output equivalence**, not a whole-farm state/value equivalence.

A sufficient state is `(alive, consecutive_unfed, pending_care, held_product,
fertilizer_available)`, together with species, current day, placement day and final
actionable day. Do not omit the care bank, production phase or terminal horizon.
Do not apply this day-boundary state to arbitrary midday observations: consumed
`fed_today`/`cared_today` flags and remaining opportunities would also be required.
Births after escapes start new lives; a policy's replacement purchase is not part
of the old animal's transition.

### Proof from native mechanics

1. FEED consumes one delivered WHEAT and sets a once-per-day flag; CARE sets a
   distinct once-per-day flag. Their order cannot alter either flag. Animal
   products and fertilizer are not WHEAT, so their collections cannot supply the
   feed precondition. Feed availability is a condition, not a predicted success.
2. No animal product or fertilizer is produced intraday. A positive HARVEST takes
   all held product and leaves zero; a positive fertilizer collection takes one
   available unit and clears availability. Thus there is at most one positive
   event of each kind per day. Duplicates are no-ops. The four operations commute
   on this projected state and total transfers, although their carrier inventory
   destinations and route consequences need not commute.
3. The native midnight transition reads only that projected state and those flags.
   Unfed streak becomes zero if fed, otherwise increments; at two, the animal
   escapes before production. On a production boundary, held product becomes
   `min(capacity, held_after_harvest + 1 + (pending_care if fed else 0))` and the
   old care bank resets **even when not fed**. Today's fed-and-cared bonus is
   banked **after** production. Fertilizer refreshes and daily flags reset.
4. On the last actionable day there is no midnight: no escape check, production,
   fertilizer refresh or care banking. Only intraday transfers matter. Induction
   over successive days proves the claimed equivalence.

Authority: `_apply_unit_action`, `_daily_refresh_animals` and `interpreter` in
`game/kaggriculture_engine_reference.py`. The model calls the existing
`economy/heuristic/animal_dp.step`; it does not introduce a second implementation
of these rules. Tests independently generate reachable states using the pinned
native functions and compare the DP, service order, duplicate requests and every
possible terminal age. They are local fixtures, not exploratory games.

### What this actually removes from search

For an animal's biological consequences, replace a 719-turn route history with at
most 30 four-bit service records plus births. For optimization conditional on
service availability, reuse `animal_dp.solve`: memoize `(day, biological_state)`
rather than enumerate every route history or every complete calendar. Preserve
calendar phase and holding/care capacity; route feasibility and funding remain
outside this local DP.

There are at most 16 requested daily masks. The existing priced DP considers at
most 12: care without feed earns no biological reward and costs an action, while
empty harvest/collection requests are no-ops. That pruning is appropriate only
for its stated nonnegative action-cost, local linear objective. The trace adapter
retains all successful CARE actions, including care without feed: observed work
must not disappear just because a local optimizer would omit it.

## A smaller lemma: the first cow yield depends on a count

For a newborn cow on day `p`, suppose every feed through day `p+7` succeeds. The
first milk appears at midnight after day `p+7` (morning `p+8`). Let `C` count
successful CAREs on days `p` through `p+6`. Then first held milk is

`min(6, 1 + C)`.

Care on day `p+7` is too late for that first burst; it banks for a later burst.
This follows directly from the production-before-new-care ordering above. The
seven early care bits have 128 assignments but only six first-boundary states
when all other conditions and the boundary-day care flag are fixed. Class sizes
for first milk 1,2,3,4,5,6 are `1,7,21,35,35,29`. Daily action costs remain distinct.
`prefix_care_certificate` enumerates every assignment and retains each member.

From seven early CAREs, removing any one or two preserves first milk; removing
any three loses one. Zero individual sensitivity is not joint redundancy. Removing
the feed before first production discards the bank; feed dates are not interchangeable.

## Call the functions

```python
from farm_neat.economy.heuristic.animal_dp import State
from farm_neat.evaluation.service_quotient import day_effect

result = day_effect(
    "COW", day=7, placed_day=0,
    state=State(alive=True, consecutive_unfed=0, pending_care=7,
                held=0, fertilizer_available=True),
    requested=(True, True, False, False), strict=True,
)
assert result["state_after"]["held"] == 6
assert result["state_after"]["pending_care"] == 1
assert result["product_units"] == 0  # milk is held, not yet harvested
assert result["feed_units"] == 1
```

`day_effect` evaluates a conditional local transition. `replay_calendar` composes
it across a complete remaining season. `calendar_sensitivity` withholds each
successful service once while keeping all other abstract requests fixed; requests
made impossible by an escape or unavailable output become explicit no-ops.
`prefix_care_certificate` checks the first-cycle count lemma exhaustively.
`compress_turns` joins native unit effects and independently checks every daily
boundary and transferred quantity against the DP. `audit_bundle` binds original
sources, inputs, attempts, joint tapes and both physical-seat terminal balances.

Ablations do **not** rerun routing, replacement, displaced work, losses, deposits
or markets. Service signatures are observed targets, never pre-choice model inputs.

## Rank alternatives without inventing equivalent value

Use `rank_visible_targets` on a complete native menu with KEEP, grouped by exact
actor-visible encoder tensor **and menu**, not hidden seed/opponent identity. It
retains matched members for every alternative and reports their joint outcomes.
Its reference ordering is empirical WDL first, mean terminal margin second, with
KEEP on an exact tie. Non-KEEP ties remain explicit instead of using menu order.
This is an exposed finite-cohort reference, not a learned selection or an estimate
of population probabilities. Uniform recorded-member weights are explicit; they
must not be mistaken for the deployment distribution.

The stored cash decomposition obeys, in exact cents:

`delta_margin = delta_own_cash - delta_opponent_cash`.

`compare_outcomes` independently checks that actual market-fill deltas sum to the
actor's cash delta. Margin differences never become WDL labels. A known losing
menu can have useful economic rankings, but better margin still is not a win.
Cash effects require costs, actual sale fills, ordering, shared-market changes and
displaced work. Equal biology is no justification for equal cash predictions.

Important counterexamples are retained in the tests: successful feed differs
from requested feed without WHEAT; holding and selling can have equal biology and
different cash; an extra hand can leave biology unchanged yet affect both players'
cash; and a late animal can receive perfect service but have no yield before the
terminal horizon. No rule here authorizes implicit sales or forces a hire.

## Prediction schema for the existing full learner

Use the existing `plan_observation.pack` input and complete native program/KEEP
representation in `training.plan_effects.pack_group`. The report contains all
raw-file joins and exact targets; a full-learner loader/auxiliary head is **not**
implemented here. Do not claim it has trained. A maintainer-authorized fit can
separate three outputs while retaining the existing backbone and grammar:

* Forecast placement and **joint** daily service calendars, including feed/input
  failures. Decode calendar hypotheses through the exact component model to get
  survival, dated held/harvested quantities and consumed resources.
* Forecast transport/deposit losses, actual market fills, purchases, labor and
  displaced work separately. Harvested, deposited and sold are different labels.
* Forecast both terminal balances and WDL, with the original same-visible-input
  outcome distribution. Keep the empirical reference's ties and losses visible.

Do not multiply independently predicted feed and care probabilities unless their
independence is justified. `E[feed*care]` need not equal `E[feed]*E[care]`, and
saturation/escape introduce additional nonlinearity. A few weighted joint calendar
hypotheses preserve those dependencies; passing mean bits into an exact nonlinear
transition does not. This is a proposed modeling route, not measured improvement.

The first competence test should be one cow: placement/first-yield day, survival,
care-bank/capacity effects, sale versus hold, no-feed escape/replacement and the
late-horizon KEEP contrast. Include aliases with identical biology but different
market fills. Compare one bounded complete-group fit against its frozen parent,
then test on reserved **distinct visible-state cohorts**, not shuffled rows or new
seeds that encode identically. Measure consequence errors, menu ranks/ties,
emission validity and actually sampled playing outcomes separately. This guide
neither registers nor launches that maintainer experiment.

## Related primary research

Abel et al., *Value Preserving State-Action Abstractions* (AISTATS 2020),
https://proceedings.mlr.press/v108/abel20a.html. Barreto et al., *Successor Features
for Transfer in Reinforcement Learning*, https://arxiv.org/abs/1606.05312.
These motivate separating dynamics and value; neither proves our component lemma
or learned improvement. Shared markets are not assumed to have fixed linear prices.

"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
from itertools import product
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from farm_neat.economy.heuristic import animal_dp
from farm_neat.evaluation.common import bound, canonical, digest, equal, file_sha, read
from farm_neat.evaluation.service_timeline import cents, successful_effect
from farm_neat.game.constants import ANIMALS

ROOT = Path(__file__).resolve().parents[3]
BUNDLES = Path('work/runs/learning-start/edits/scenarios/bundles')
SCHEMA = 'farm-animal-day-service-quotient-v1'
OPS = ('FEED', 'CARE', 'HARVEST', 'COLLECT_FERTILIZER')
ZERO_PRICES = animal_dp.Prices(0., 0., 0., 0.)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def animal_state(tile: Any) -> animal_dp.State:
    """Project only biology. Terminal fed/cared flags are not continuation state."""
    if not isinstance(tile, dict) or not tile.get('animal'):
        return animal_dp.State(alive=False)
    require(tile['animal'] in ANIMALS, 'unknown animal')
    values = [tile['consecutive_unfed'], tile.get('pending_care_bonus', 0), tile['yield_units']]
    require(all(type(v) is int and v >= 0 for v in values), 'nonnegative integer animal state')
    require(values[0] < 2 and values[2] <= ANIMALS[tile['animal']]['max_held'], 'invalid living animal state')
    require(type(tile['fertilizer_available']) is bool, 'boolean fertilizer state')
    return animal_dp.State(True, *values, tile['fertilizer_available'])


def day_effect(kind: str, day: int, placed_day: int, state: animal_dp.State,
               requested: Iterable[bool], *, total_days: int = 30,
               strict: bool = False) -> dict:
    """Apply an ABSTRACT service calendar using the existing exact animal DP.

    Inputs mean service at the animal with feed already delivered, NOT routes.
    strict=True requires positive witnessed services. Otherwise impossible
    harvest/collect/dead-animal requests become recorded no-ops, as in the engine.
    No automatic replacement after escape; a new placement is a separate life.
    """
    action = tuple(requested)
    require(kind in ANIMALS, 'known animal required')
    require(type(total_days) is int and 1 <= total_days <= 30, '1..30 actionable days required')
    require(all(type(x) is int for x in (day, placed_day)) and 0 <= placed_day <= day < total_days,
            'valid placement and service days required')
    require(len(action) == 4 and all(type(x) is bool for x in action), 'four boolean service requests required')
    require(isinstance(state, animal_dp.State), 'native animal DP State required')
    require(type(state.alive) is bool and type(state.fertilizer_available) is bool, 'boolean state required')
    require(all(type(x) is int and x >= 0 for x in
                (state.consecutive_unfed, state.pending_care, state.held)), 'integer state required')
    require(not state.alive or (state.consecutive_unfed <= 1 and state.held <= ANIMALS[kind]['max_held']),
            'valid living state required')
    require(state.alive or state == animal_dp.State(alive=False), 'canonical absent animal required')
    effective = tuple(a and ok for a, ok in zip(action, (state.alive, state.alive,
                      state.alive and state.held > 0, state.alive and state.fertilizer_available), strict=True))
    if strict:
        equal(action, effective, 'witness claimed a service that cannot succeed')
    nxt, _value, product, fertilizer = animal_dp.step(
        kind, day, placed_day, state, effective, ZERO_PRICES, total_days - 1)
    return dict(day=day, requested=list(action), effective=list(effective),
                unavailable_requests=[op for op, a, b in zip(OPS, action, effective, strict=True) if a and not b],
                state_before=asdict(state), state_after=asdict(nxt),
                product_units=product, fertilizer_units=fertilizer,
                feed_units=int(effective[0]), successful_service_actions=sum(effective),
                escaped=state.alive and not nxt.alive,
                has_end_of_day=day < total_days - 1)


def replay_calendar(kind: str, placed_day: int, calendar: Iterable[Iterable[bool]], *,
                    start: int | None = None, initial: animal_dp.State | None = None,
                    total_days: int = 30, strict: bool = False) -> dict:
    """Complete remaining calendar, conditional on service/input availability."""
    first = placed_day if start is None else start
    require(type(first) is int and placed_day <= first < total_days, 'valid calendar start required')
    actions = [tuple(a) for a in calendar]
    require(len(actions) == total_days - first, 'complete remaining calendar required')
    state = animal_dp.State() if initial is None else initial
    days = []
    for d, a in enumerate(actions, first):
        row = day_effect(kind, d, placed_day, state, a, total_days=total_days, strict=strict)
        days.append(row)
        state = animal_dp.State(**row['state_after'])
    return dict(kind=kind, placed_day=placed_day, start=first, total_days=total_days, days=days,
                features={field: sum(row[field] for row in days) for field in
                          ('product_units', 'fertilizer_units', 'feed_units', 'successful_service_actions')},
                terminal_state=asdict(state),
                first_escape_day=next((r['day'] for r in days if r['escaped']), None),
                scope='CONDITIONAL_SINGLE_ANIMAL_BIOLOGY_NOT_ROUTE_FEASIBILITY_OR_CASH')


def calendar_sensitivity(kind: str, placed_day: int, calendar: Iterable[Iterable[bool]], *,
                         start: int | None = None, initial: animal_dp.State | None = None,
                         total_days: int = 30) -> dict:
    """Exact one-day service ablations; all other abstract requests are held fixed.

    These are mechanistic queries, NOT played interventions or cash predictions.
    In particular an escaped animal is not replaced as an executor might do.
    """
    actions = [list(a) for a in calendar]
    base = replay_calendar(kind, placed_day, actions, start=start, initial=initial,
                           total_days=total_days, strict=True)
    rows = []
    for i, action in enumerate(actions):
        for j, op in enumerate(OPS):
            if not action[j]:
                continue
            changed = copy.deepcopy(actions); changed[i][j] = False
            result = replay_calendar(kind, placed_day, changed, start=start, initial=initial,
                                     total_days=total_days)
            rows.append(dict(day=base['start'] + i, withheld=op,
                             delta={k: result['features'][k] - v for k, v in base['features'].items()},
                             first_escape_day=result['first_escape_day'],
                             terminal_state=result['terminal_state']))
    return dict(baseline=base, interventions=rows,
                scope='ABSTRACT_ONE_SERVICE_ABLATIONS; ROUTES_MARKETS_REPLACEMENT_NOT_REPLAYED')


def prefix_care_certificate(kind: str, placed_day: int, calendar: Iterable[Iterable[bool]], *,
                            total_days: int = 30) -> dict:
    """Enumerate ONLY pre-first-yield CARE bits, with all other services fixed.

    This small exact conditional certificate demonstrates saturation and joint
    effects. It is neither route search nor a forecast of feasible/cash effects.
    CARE on the day immediately before first yield is NOT in this prefix: the
    engine banks that care after the first production event.
    """
    actions = [list(a) for a in calendar]
    base = replay_calendar(kind, placed_day, actions, total_days=total_days, strict=True)
    n = ANIMALS[kind]['first_yield_day'] - 1
    boundary_day = placed_day + n
    require(boundary_day < total_days - 1, 'first production must precede terminal')
    require(all(a[0] for a in actions[:n + 1]), 'certificate conditions require every prefix feed')
    require(all(not a[2] for a in actions[:n + 1]), 'newborn prefix cannot harvest before production')
    classes = {}
    for bits in product((False, True), repeat=n):
        changed = copy.deepcopy(actions)
        for i, care in enumerate(bits):
            changed[i][1] = care
        result = replay_calendar(kind, placed_day, changed, total_days=total_days)
        boundary = result['days'][n]['state_after']
        expected = min(ANIMALS[kind]['max_held'], 1 + sum(bits))
        equal(boundary['held'], expected, 'first-production care-count lemma')
        key = digest(boundary)
        item = classes.setdefault(key, dict(boundary_state=boundary, members=[]))
        item['members'].append(dict(care_bits=list(bits), early_care_count=sum(bits),
            season_product_units=result['features']['product_units'],
            successful_service_actions=result['features']['successful_service_actions']))
        equal(item['members'][0]['season_product_units'], result['features']['product_units'],
              'same boundary and suffix imply same later product')
    return dict(kind=kind, placed_day=placed_day, boundary_after_day=boundary_day,
        varied_days=list(range(placed_day, boundary_day)), varied_service='CARE',
        abstract_calendars=2 ** n, boundary_classes=len(classes), classes=classes,
        baseline_product_units=base['features']['product_units'],
        exact_first_yield_formula='min(capacity, 1 + prefix_feed_and_care_count)',
        scope='CONDITIONAL_BIOLOGY; DIFFERENT_SERVICE_COSTS_REMAIN; ZERO_SINGLE_DELETION_EFFECT_IS_NOT_JOINT_REDUNDANCY')


def compress_turns(turns: list[dict], *, tile: tuple[int, int] = (3, 4)) -> dict:
    """Check one tile across a COMPLETE saved game, then discard route timing.

    Native placement is kept at day resolution. Species/placement-day changes
    make separate lives. Each observed unit effect is joined in engine order;
    the old DP independently predicts the end-of-day state and output quantities.
    """
    require(len(turns) == 719 and [r['step'] for r in turns] == list(range(719)),
            'complete ordered 719-turn trace required')
    daily, service_events = [], []
    prior = turns[0]['animal_before']
    for day in range(30):
        chunk = turns[24 * day:min(24 * (day + 1), 719)]
        initial_tile = copy.deepcopy(chunk[0]['animal_before'])
        alive_at_start = animal_state(initial_tile).alive
        current = copy.deepcopy(initial_tile)
        starting = copy.deepcopy(initial_tile) if alive_at_start else None
        born_at = None
        flags = [False] * 4
        observed = Counter()
        if starting is not None:
            require(not starting['fed_today'] and not starting['cared_today'], 'day must start before service')
        for turn in chunk:
            equal(turn['animal_before'], prior, 'consecutive local animal observations')
            local = copy.deepcopy(turn['animal_before'])
            for effect in turn['unit_effects']:
                if effect['position'] != list(tile):
                    continue
                equal(effect['tile_before'], local, 'unit effect joins local state in engine order')
                local = copy.deepcopy(effect['tile_after'])
                if not successful_effect(effect):
                    continue
                op = effect['engine_received_action'][0]
                if op == 'PLACE' and animal_state(local).alive:
                    require(starting is None and born_at is None, 'multiple/same-day replacement unsupported')
                    require(local['placed_day'] == day, 'birth day mismatch')
                    starting = copy.deepcopy(local); born_at = turn['step']
                    equal(asdict(animal_state(starting)), asdict(animal_dp.State()), 'native newborn state')
                elif op in OPS and animal_state(effect['tile_before']).alive:
                    require(starting is not None, 'service without a placed animal')
                    index = OPS.index(op)
                    require(not flags[index], 'duplicate successful same-day animal service')
                    flags[index] = True
                    carried = effect['delta']['carried']
                    if op == 'FEED':
                        equal(carried, {'WHEAT': -1}, 'successful feed consumes one wheat')
                        observed['feed_units'] += 1
                    elif op == 'HARVEST':
                        product = ANIMALS[starting['animal']]['product']
                        require(set(carried) == {product} and carried[product] > 0, 'actual animal harvest units')
                        observed['product_units'] += carried[product]
                    elif op == 'COLLECT_FERTILIZER':
                        equal(carried, {'FERTILIZER': 1}, 'successful fertilizer collection')
                        observed['fertilizer_units'] += 1
                    elif op == 'CARE':
                        equal(carried, {}, 'care has no inventory transfer')
                    service_events.append(dict(step=turn['step'], day=day, op=op,
                                               unit=effect['unit'], carried=carried))
            if (turn['step'] + 1) % 24 != 0:
                equal(local, turn['animal_after'], 'no intra-day animal change outside captured units')
            prior = turn['animal_after']; current = copy.deepcopy(prior)
        if starting is None:
            require(not animal_state(current).alive and not any(flags), 'unobserved animal appearance')
            daily.append(dict(day=day, kind=None, placed_day=None, born=False, service_mask=0,
                              product_units=0, fertilizer_units=0, feed_units=0,
                              successful_service_actions=0, escaped=False))
            continue
        kind, placed = starting['animal'], starting['placed_day']
        projected = day_effect(kind, day, placed, animal_state(starting), flags, strict=True)
        if animal_state(current).alive:
            require(current['animal'] == kind and current['placed_day'] == placed, 'unobserved animal replacement')
            if day < 29:
                require(not current['fed_today'] and not current['cared_today'], 'daily flags did not reset')
            else:
                equal([current['fed_today'], current['cared_today']], flags[:2], 'terminal intraday flags')
        equal(projected['state_after'], asdict(animal_state(current)), 'native DP vs actual day boundary')
        for field in ('feed_units', 'product_units', 'fertilizer_units'):
            equal(projected[field], observed[field], 'native DP vs actual ' + field)
        daily.append(dict(day=day, kind=kind, placed_day=placed, born=born_at is not None,
                          # Exact placement hour is a witness, NOT part of biology signature.
                          birth_step=born_at, service_mask=sum(int(a) << i for i, a in enumerate(flags)),
                          **{k: projected[k] for k in ('state_before', 'state_after', 'product_units',
                             'fertilizer_units', 'feed_units', 'successful_service_actions', 'escaped')}))
    # Component signature: excludes unit identity, route, service hour and site.
    # Keep calendar, birth generation and native sufficient state. It is not a
    # pre-choice input or permission to equate cash forecasts.
    reduced = [{k: v for k, v in r.items() if k != 'birth_step'} for r in daily]
    return dict(schema=SCHEMA, tile=list(tile), days=daily, biology_sha256=digest(reduced),
                service_events=service_events,
                features={k: sum(r[k] for r in daily) for k in
                          ('product_units', 'fertilizer_units', 'feed_units', 'successful_service_actions')},
                births=sum(r['born'] for r in daily), escapes=sum(r['escaped'] for r in daily),
                primitive_turns=719, animal_day_records=len(daily),
                scope='EXACT_COMPONENT_REPLAY; IDENTICAL_BIOLOGY_DOES_NOT_IMPLY_IDENTICAL_CASH')


def _score(own: float, other: float) -> float:
    return 1. if own > other else .5 if own == other else 0.


def compare_outcomes(before: dict, after: dict) -> dict:
    """Actor-oriented, matched-case deltas; margin is NEVER used as a WDL label."""
    own = cents(after['own_cash']) - cents(before['own_cash'])
    other = cents(after['opponent_cash']) - cents(before['opponent_cash'])
    market = {}
    for key in sorted(set(before['actual_market']) | set(after['actual_market'])):
        a = before['actual_market'].get(key, {}); b = after['actual_market'].get(key, {})
        row = {field: b.get(field, 0) - a.get(field, 0) for field in ('units', 'cash_cents')}
        if any(row.values()):
            market[key] = row
    equal(sum(r['cash_cents'] for r in market.values()), own, 'paired actual cash decomposition')
    return dict(delta_own_cash_cents=own, delta_opponent_cash_cents=other,
                delta_margin_cents=own - other,
                delta_wdl=_score(after['own_cash'], after['opponent_cash']) - _score(before['own_cash'], before['opponent_cash']),
                actual_market_deltas=market)


def rank_visible_targets(cohort: dict) -> dict:
    """Uniform exposed-cohort reference, not a fitted/population cash forecast.

    Rank empirical WDL first, then empirical margin, on identical visible inputs
    AND identical native menus. Retain every paired outcome and all exact ties.
    KEEP wins an exact tie; otherwise no arbitrary menu-order choice is made.
    """
    n = len(cohort['members']); require(n > 0, 'nonempty visible cohort required')
    targets = cohort['candidate_targets']
    require('keep' in targets and set(targets) == set(cohort['menu']), 'complete native menu with KEEP required')
    stats, keys = {}, {}
    for arm, rows in targets.items():
        equal([r['paired_member_index'] for r in rows], list(range(n)), 'matched cohort member order')
        counts = Counter(); own = other = 0
        deltas = []
        for row, keep in zip(rows, targets['keep'], strict=True):
            a, b = cents(row['own_cash']), cents(row['opponent_cash'])
            score = _score(a, b); equal(score, row['wdl'], 'WDL is the joint cash outcome')
            counts['wins' if score == 1 else 'draws' if score == .5 else 'losses'] += 1
            own += a; other += b
            deltas.append(dict(paired_member_index=row['paired_member_index'],
                delta_own_cash_cents=a-cents(keep['own_cash']),
                delta_opponent_cash_cents=b-cents(keep['opponent_cash']),
                delta_margin_cents=a-b-cents(keep['own_cash'])+cents(keep['opponent_cash']),
                delta_wdl=score-keep['wdl']))
        points = 2*counts['wins'] + counts['draws']
        keys[arm] = (points, own-other)
        stats[arm] = dict(members=n, **{k: counts[k] for k in ('wins', 'draws', 'losses')},
            empirical_wdl=points/(2*n), mean_own_cash_cents=own/n,
            mean_opponent_cash_cents=other/n, mean_margin_cents=(own-other)/n,
            matched_keep_deltas=deltas)
    best_key = max(keys.values())
    tied = sorted(arm for arm, key in keys.items() if key == best_key)
    for arm, key in keys.items():
        stats[arm]['empirical_wdl_gap_to_best'] = (best_key[0]-key[0])/(2*n)
        stats[arm]['margin_gap_cents_at_equal_wdl'] = ((best_key[1]-key[1])/n if key[0] == best_key[0] else None)
    return dict(candidates=stats, equally_best=tied,
                reference_choice='keep' if 'keep' in tied else tied[0] if len(tied) == 1 else None,
                criterion='LEXICOGRAPHIC_EMPIRICAL_WDL_THEN_MARGIN_KEEP_ON_EXACT_TIE',
                weighting='UNIFORM_OVER_RECORDED_MATCHED_MEMBERS_NOT_ESTIMATED_POPULATION_FREQUENCIES',
                scope='EXPOSED_FINITE_COHORT_REFERENCE_NOT_A_LEARNED_POLICY_OR_GENERALIZATION_RESULT')


def _tensor_identity(given: dict) -> str:
    # No model construction/import or weights: this is the existing NumPy input encoder.
    from farm_neat.economy.orders import FullSeasonPlan
    from farm_neat.policy import plan_observation
    pack = plan_observation.pack(given['observation'], FullSeasonPlan.restore(given['snapshot']),
                                given['configuration'], expected_identity=plan_observation.identity())
    h = hashlib.sha256()
    for name in ('rows', 'knownness', 'token_mask'):
        a = pack[name]; h.update(str((name, a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def audit_bundle(bundle_directory: Path = BUNDLES) -> dict:
    """All registered dairy arms/attempts, source-bound. Never chooses a subset."""
    directory = ROOT / bundle_directory
    plan = read(directory / 'PLAN.json'); result = read(directory / 'RESULT.json')
    equal(result['plan_sha256'], file_sha(directory / 'PLAN.json'), 'original bundle registration')
    equal(len(result['rows']), len(plan['cases']), 'all registered cases')
    equal(result['complete_games'], len(plan['cases']), 'all completed games')
    equal(result['physical_steps'], 719 * len(plan['cases']), 'all physical steps')
    pins = dict(plan['sources'])
    copies = None
    for name, expected in pins.items():
        current = file_sha(ROOT / name)
        if current != expected:
            # A reviewed migration receipt may certify the original bytes through a retained
            # byte-identical copy under work/ (review-release contract); this proves the old
            # source identity, never compatibility of the current source with old outcomes.
            if copies is None:
                from farm_neat.tooling import evidence_compaction
                copies = evidence_compaction.historical_sources(ROOT)
            copy = copies.get((name, expected))
            current = file_sha(ROOT / copy) if copy else current
        equal(current, expected, 'original execution source: ' + name)
    # Pin all new analysis dependencies, without rewriting historical producer pins.
    for name in ('src/farm_neat/evaluation/service_quotient.py',
                 'src/farm_neat/economy/heuristic/animal_dp.py',
                 'src/farm_neat/evaluation/service_timeline.py',
                 'src/farm_neat/evaluation/common.py'):
        pins[name] = file_sha(ROOT / name)
    inputs = {str((directory / n).relative_to(ROOT)): file_sha(directory / n)
              for n in ('PLAN.json', 'RESULT.json', 'attempts.jsonl')}
    attempts = [json.loads(line) for line in (directory / 'attempts.jsonl').read_text().splitlines()]
    require(len(attempts) == len(plan['cases']), 'every registered attempt must remain')
    cases, groups, tensor_cache = [], defaultdict(dict), {}
    for expected, row, attempt in zip(plan['cases'], result['rows'], attempts, strict=True):
        equal(row['case'], expected, 'registered ordered case')
        require(row['complete'] is True and row['physical_steps'] == 719, 'complete terminal branch required')
        equal(attempt['case_index'], expected['index'], 'attempt index')
        equal(attempt['sha256'], row['sha256'], 'attempt receipt')
        path = ROOT / row['file']; equal(file_sha(path), row['sha256'], 'full raw game bytes')
        inputs[row['file']] = row['sha256']
        raw = json.loads(gzip.decompress(path.read_bytes()))
        equal(raw['outcome'], {k: v for k, v in row.items() if k not in ('file', 'sha256')}, 'raw outcome join')
        require(not row['publication_refusals'], 'refused branch cannot supply an accepted target')
        equal(digest(raw['input']), row['input_sha256'], 'original recorded input')
        equal([len(x) for x in raw['actions']], [719, 719], 'complete joint action tapes')
        equal(digest(raw['actions']), row['actions_sha256'], 'actual actions')
        equal(digest([a[:expected['start']] for a in raw['actions']]), row['prefix_actions_sha256'], 'immutable action prefix')
        seat = expected['seat']; require(type(seat) is int and seat in (0, 1), 'physical seat')
        equal([raw['terminal']['farms'][seat]['money'], raw['terminal']['farms'][1-seat]['money']],
              [row['own_cash'], row['opponent_cash']], 'both actor-oriented terminal balances')
        cash = 300000; totals = {}
        for turn in raw['turns']:
            for commit in turn['commits']:
                cash += cents(commit['cash_delta'])
                key = commit['op'] + (':' + commit['item'] if commit['item'] else '')
                acc = totals.setdefault(key, dict(units=0, cash_cents=0, failed_unit_attempts=0))
                acc['units'] += commit['units']; acc['cash_cents'] += cents(commit['cash_delta'])
                acc['failed_unit_attempts'] += commit['failed_unit_attempts']
            equal(cash, cents(turn['cash']), 'every actual cash transition')
        equal(totals, row['actual_market'], 'actual lifetime fills')
        equal(cash, cents(row['own_cash']), 'final actor cash')
        compressed = compress_turns(raw['turns'])
        initial = row['input_sha256']
        if initial not in tensor_cache:
            tensor_cache[initial] = _tensor_identity(raw['input'])
        case = dict(index=expected['index'], opponent=expected['scenario']['opponent'],
                    seed=expected['scenario']['seed'], seat=seat, start=expected['start'], arm=expected['arm'],
                    raw_file=row['file'], raw_sha256=row['sha256'], input_sha256=initial,
                    actor_tensor_sha256=tensor_cache[initial], prefix_sha256=row['prefix_actions_sha256'],
                    program=expected['program'], own_cash=row['own_cash'], opponent_cash=row['opponent_cash'],
                    wdl=_score(row['own_cash'], row['opponent_cash']), actual_market=totals,
                    first_milk_sale_step=row['first_milk_sale_step'], biology=compressed)
        cases.append(case)
        group = groups[(case['opponent'], seat, case['start'])]
        require(case['arm'] not in group, 'unique arm in matched group'); group[case['arm']] = case
    comparisons, rankings = [], []
    for identity, group in groups.items():
        equal(list(group), ['keep', 'dairy', 'hold_milk', 'no_feed', 'no_extra_hand'], 'all alternatives, KEEP first')
        require(len({r['input_sha256'] for r in group.values()}) == 1, 'same raw actor input within menu')
        require(len({r['prefix_sha256'] for r in group.values()}) == 1, 'same executed prefix within menu')
        b, a = group['dairy'], group['no_extra_hand']
        comparisons.append(dict(opponent=identity[0], seat=identity[1], start=identity[2],
            baseline_index=b['index'], alternative_index=a['index'],
            same_biology=b['biology']['biology_sha256'] == a['biology']['biology_sha256'],
            same_service_hours=[(x['step'], x['op']) for x in b['biology']['service_events']] ==
                               [(x['step'], x['op']) for x in a['biology']['service_events']],
            delta_harvested_product=a['biology']['features']['product_units'] - b['biology']['features']['product_units'],
            **compare_outcomes(b, a)))
        choices = list(group.values())
        # KEEP is retained on an exact primary/secondary tie, not menu-first dairy.
        best = max(choices, key=lambda r: (r['wdl'], r['own_cash'] - r['opponent_cash'], r['arm'] == 'keep'))
        rankings.append(dict(opponent=identity[0], seat=identity[1], start=identity[2],
                             hindsight_choice=best['arm'],
                             choices={r['arm']: compare_outcomes(group['keep'], r) for r in choices}))
    # Tensor aliases, not hidden seed/opponent labels, define learnable cohorts.
    # Keep the complete menu in the key: different menus are different decisions.
    visible = {}
    for identity, group in groups.items():
        keep = group['keep']; menu = {name: row['program'] for name, row in group.items()}
        key = digest([keep['actor_tensor_sha256'], menu])
        cohort = visible.setdefault(key, dict(actor_tensor_sha256=keep['actor_tensor_sha256'],
            menu=menu, members=[], candidate_targets={name: [] for name in menu}))
        cohort['members'].append(dict(opponent=identity[0], seat=identity[1], start=identity[2]))
        for arm, row in group.items():
            cohort['candidate_targets'][arm].append(dict(case_index=row['index'],
                paired_member_index=len(cohort['members'])-1,
                product_harvested=row['biology']['features']['product_units'],
                feed_consumed=row['biology']['features']['feed_units'],
                successful_services=row['biology']['features']['successful_service_actions'],
                births=row['biology']['births'], escapes=row['biology']['escapes'],
                product_sold=row['actual_market'].get('SELL:MILK', {}).get('units', 0),
                own_cash=row['own_cash'], opponent_cash=row['opponent_cash'],
                margin=row['own_cash'] - row['opponent_cash'], wdl=row['wdl']))
    for cohort in visible.values():
        cohort['ranking_reference'] = rank_visible_targets(cohort)
    sensitivities = {}
    for start in (0, 288, 576):
        case = groups[('pass', 0, start)]['dairy']
        days = [r for r in case['biology']['days'] if r['kind']]
        require(case['biology']['births'] == 1 and case['biology']['escapes'] == 0, 'one surviving life required for local sensitivity')
        calendar = [[bool(r['service_mask'] & (1 << i)) for i in range(4)] for r in days]
        sensitivities[str(start)] = dict(source_case_index=case['index'],
            **calendar_sensitivity('COW', days[0]['placed_day'], calendar,
                start=days[0]['day'], initial=animal_dp.State(**days[0]['state_before'])))
    opening = sensitivities['0']['baseline']
    care_certificate = prefix_care_certificate('COW', 0, [d['requested'] for d in opening['days']])
    from farm_neat.policy import plan_observation
    encoder_identity = plan_observation.identity()
    for name, expected in encoder_identity['source_pins'].items():
        equal(file_sha(ROOT / name), expected, 'live encoder source ' + name)
        if name in pins:
            equal(pins[name], expected, 'encoder vs original execution source ' + name)
        pins[name] = expected
    return dict(schema=SCHEMA, sources=pins, inputs=inputs, encoder_identity=encoder_identity,
        verified_saved_games=len(cases), verified_saved_joint_steps=719 * len(cases),
        verified_animal_day_records=sum(len(r['biology']['days']) for r in cases),
        distinct_biology_signatures=len({r['biology']['biology_sha256'] for r in cases}),
        actor_visible_input_menu_cohorts=len(visible), cases=cases, extra_hand_comparisons=comparisons,
        hindsight_rankings=rankings, visible_cohorts=visible, conditional_sensitivities=sensitivities,
        prefix_care_certificate=dict(source_case_index=sensitivities['0']['source_case_index'], **care_certificate),
        new_games=0, model_calls=0, optimizer_updates=0,
        prediction_schema=dict(
            inputs='Existing actor-visible plan_observation.pack plus native candidate via plan_effects.pack_group; no future/service labels as input',
            exact_component='Animal-day next state and harvested units conditional on a realized service calendar',
            learned_unknowns='Placement/service dates, input delivery, shed deposit/loss and market-fill timing under the existing executor',
            outcome_targets='Joint actor/opponent cash and empirical WDL; preserve complete same-visible-input outcome distributions',
            invariant_loss_scope='BIOLOGICAL OUTPUTS ONLY for service aliases, never a forced equal cash/WDL score',
            curriculum='First forecast placement day, survival and dated milk output; retain explicit sale/hold and KEEP/horizon cash contrasts',
            training_status='LABELS_AND_ADAPTER_PROTOCOL_ONLY_NO_FIT_OR_NEW_HEAD_IMPLEMENTED'),
        scope='EXPOSED_DIAGNOSTIC_COHORT_NOT_GENERALIZATION; BIOLOGY_QUOTIENT_NOT_FULL_FARM_MDP_OR_VALUE_EQUIVALENCE',
        retention=dict(purpose='Exact service compression, counterexamples and full-policy group target schema',
                       dependencies=['source tests', 'Q1-Q4 decisions', '60 original raw dairy games'],
                       remove_when='A verified successor preserves the theorem tests, monetary non-equivalence and target joins'))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Exact animal-day effects and matched saved-game rankings; no games or training.')
    p.add_argument('command', choices=('audit', 'verify'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bundle-directory', type=Path, default=BUNDLES)
    args = p.parse_args(argv)
    report = bound(audit_bundle(args.bundle_directory))
    if args.command == 'verify':
        require(args.out.is_file() and not args.out.is_symlink(), 'existing regular report required')
        saved = json.loads(gzip.decompress(args.out.read_bytes()))
        equal(saved, bound(saved), 'report content binding')
        equal(saved, report, 'entire reproduced service quotient')
    else:
        require(not any(x.is_symlink() for x in (args.out, *args.out.parents)), 'no symlinked output')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('xb') as stream:
            stream.write(gzip.compress((canonical(report) + '\n').encode(), mtime=0))
    print(canonical({k: report[k] for k in ('schema', 'verified_saved_games', 'verified_saved_joint_steps',
          'verified_animal_day_records', 'distinct_biology_signatures', 'actor_visible_input_menu_cohorts',
          'new_games', 'model_calls', 'optimizer_updates')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
