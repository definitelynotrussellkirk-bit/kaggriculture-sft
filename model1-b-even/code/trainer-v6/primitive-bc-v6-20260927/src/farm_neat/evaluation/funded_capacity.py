"""Exact pre-receipt hire-cost frontier; a relaxation, never a crop scheduler.

For D full 24-turn days, no income, fixed up-front non-wage debits, and at most
H<=10 hands/day, daily wage C(h)=sum(fib(0..h-1)). Daily hands vanish overnight.
The least wage for K hand-days is the sum of the K cheapest prerequisite-closed
marginals. Nondecreasing native marginals make a heap merge exact. Optional
per-day hire floors remain separate; no producer/service feasibility is inferred.

Each hand supplies AT MOST 23 unit-turns (hiring follows the unit phase); the ten
native market slots permit that maximum only with room to hire at hour zero.
The executor's five-hire batch, other orders, travel, prerequisites, birth times,
custody, and deadlines can only reduce this bound. No private rival or future
shop data is accepted. This module neither samples actors nor changes orders.py.
"""
from __future__ import annotations

from bisect import bisect_right
from heapq import heapify, heappop, heappush
from typing import Sequence

from farm_neat.evaluation.capacity_law import day_unit_turns
from farm_neat.evaluation.service_timeline import cash_flow_segment
from farm_neat.game.constants import hire_cost


def _integer(value: int, name: str, lo: int, hi: int | None = None) -> int:
    if type(value) is not int or value < lo or (hi is not None and value > hi):
        raise ValueError(f'{name}: integer in [{lo}, {hi}] required')
    return value


def hire_bill(hands: int) -> int:
    """Native dollars, NOT cents; default cost multiplier one."""
    _integer(hands, 'hands', 0, 10)
    return sum(hire_cost(k) for k in range(hands))


def hire_frontier(full_days: int, hand_cap: int = 8,
                  minimum_hands: Sequence[int] | None = None) -> dict:
    """Minimum wage dollars for every total hand-day count, under daily floors.

    Returns O(D*H) integers and one deterministic heap-merge ordering, rather
    than enumerating (H+1)^D schedules. A day is a 24-turn window before the
    native truncated last day; a partial day must NOT be passed as a full day.
    The scope excludes all other spending/receipts and day-specific hire prices.
    """
    _integer(full_days, 'full_days', 1, 29)
    _integer(hand_cap, 'hand_cap', 0, 10)
    floors = [0] * full_days if minimum_hands is None else list(minimum_hands)
    if len(floors) != full_days:
        raise ValueError('one hire floor per full day required')
    for h in floors:
        _integer(h, 'minimum_hands', 0, hand_cap)
    counts = floors.copy()
    costs = [sum(hire_bill(h) for h in counts)]
    heap = [(hire_cost(h), d) for d, h in enumerate(counts) if h < hand_cap]
    heapify(heap)
    order = []
    while heap:
        price, day = heappop(heap)
        costs.append(costs[-1] + price)
        order.append(day)
        counts[day] += 1
        if counts[day] < hand_cap:
            heappush(heap, (hire_cost(counts[day]), day))
    return {'full_days': full_days, 'hand_cap': hand_cap, 'minimum_hands': floors,
            'base_hand_days': sum(floors), 'minimum_wage_dollars': costs,
            'increment_days': order,
            'uncompressed_schedule_count': (hand_cap + 1) ** full_days,
            'floor_conditioned_schedule_count': _product(hand_cap + 1 - h for h in floors),
            'frontier_states': len(costs)}


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


def funded_action_bound(start_cents: int, committed_cents: int, full_days: int,
                        hand_cap: int = 8, *, protected_cents: int = 0,
                        minimum_hands: Sequence[int] | None = None,
                        fixed_actions: int = 0, travel_lower_bound: int = 0) -> dict:
    """Necessary funded unit-turn upper bound for a declared no-income window.

    All non-wage debits are fixed and paid/reserved up front. Available cash must
    cover the supplied floors; otherwise only THAT conditional problem is empty.
    fixed_actions and travel_lower_bound must be necessary counts for the SAME
    window, disjoint from one another. The result is not a production-unit bound:
    compose it with capacity_law.quantity_bound using necessary per-batch costs,
    and count partial-batch/base output separately.
    """
    for name, value in [('start_cents', start_cents), ('committed_cents', committed_cents),
                        ('protected_cents', protected_cents), ('fixed_actions', fixed_actions),
                        ('travel_lower_bound', travel_lower_bound)]:
        _integer(value, name, 0)
    frontier = hire_frontier(full_days, hand_cap, minimum_hands)
    _, required = cash_flow_segment([-committed_cents])
    budget_cents = start_cents - protected_cents - required
    index = bisect_right(frontier['minimum_wage_dollars'], budget_cents // 100) - 1
    if budget_cents < 0 or index < 0:
        return {'status': 'INFEASIBLE_UNDER_CONDITIONS', 'binding': 'cash_prefix',
                'available_wage_cents': budget_cents, 'frontier': frontier,
                'native_feasibility': 'UNKNOWN'}
    schedule = frontier['minimum_hands'].copy()
    for day in frontier['increment_days'][:index]:
        schedule[day] += 1
    hand_days = frontier['base_hand_days'] + index
    slots = sum(day_unit_turns(0, [0] * h) for h in schedule)
    assert slots == 24 * full_days + 23 * hand_days
    remaining = slots - fixed_actions - travel_lower_bound
    return {'status': 'UPPER_BOUND_ONLY' if remaining >= 0 else 'INFEASIBLE_UNDER_CONDITIONS',
            'binding': 'cash_and_hire_cap' if remaining >= 0 else 'funded_unit_turns',
            'available_wage_cents': budget_cents,
            'hand_days_upper': hand_days, 'unit_turns_upper': slots,
            'remaining_action_slots_upper': max(0, remaining),
            'minimum_wage_dollars': frontier['minimum_wage_dollars'][index],
            'next_hand_day_minimum_wage_dollars': (frontier['minimum_wage_dollars'][index + 1]
                if index + 1 < len(frontier['minimum_wage_dollars']) else None),
            'cost_minimizing_hire_counts': schedule, 'frontier': frontier,
            'native_feasibility': 'UNKNOWN',
            'boundary': 'Wage/slot relaxation only; no receipts, up-front non-wage costs, full days, H<=10; not a birth, service, custody, sale or WDL certificate.'}
