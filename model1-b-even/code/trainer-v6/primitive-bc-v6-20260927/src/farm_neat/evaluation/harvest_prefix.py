"""Current-turn HARVEST upper bound for the pinned EconomicOrders dispatcher.

A pure, actor-visible diagnostic, not a new executor rule or a biological bound.
Before its first HARVEST job, the dispatcher must consume one worker for each
DISTINCT tile having an unconditional job (need=None, op != PLANT). A prior
assignment at that tile also consumes a worker. Thus m such tiles leave at most
max(0, n-m) HARVEST assignments. These tiles cannot be harvested this turn under
its one-duty-per-tile rule. Ignore travel and all other competition to upper-bound
harvested units by the largest remaining ripe stocks. Per-product maxima share
workers and MUST NOT be added. This conditional cut is not native infeasibility.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from farm_neat.economy.heuristic.state import View
from farm_neat.game.constants import ANIMALS, PRODUCTS

SCHEMA = 'farm-harvest-prefix-v1'


def harvest_prefix_bound(view: View, jobs: Sequence[Mapping]) -> dict:
    """Bound worker-custody gains from HARVEST in ONE current unit phase.

    `jobs` must be the exact list generated for `view` by the pinned executor.
    Stable priority order and one-duty-per-tile are essential conditions. No
    opponent private state, future demand, fills or inherited trajectory is read.
    COLLECT_FERTILIZER, purchases, output generated after this phase, and sales
    are not counted. Bound all current harvests, not the game's optimal actions.
    """
    if not isinstance(view, View) or view.n_units < 1:
        raise ValueError('one own View with at least the farmer is required')
    if not isinstance(jobs, Sequence) or isinstance(jobs, (str, bytes)):
        raise ValueError('exact executor job sequence required')
    for j in jobs:
        if not isinstance(j, Mapping) or not all(k in j for k in ('priority','tile','op','need')):
            raise ValueError('executor job fields missing')
        if len(j['tile']) != 2 or any(type(x) is not int or not 0 <= x < 10 for x in j['tile']):
            raise ValueError('native tile required')
        if not j['op'] or not isinstance(j['op'][0], str):
            raise ValueError('nonempty job action required')
        if not isinstance(j['priority'], (tuple, list)) or any(type(x) is not int for x in j['priority']):
            raise ValueError('integer priority tuple required')
    ordered = sorted(jobs, key=lambda j: tuple(j['priority']))
    first = next((i for i,j in enumerate(ordered) if j['op'][0] == 'HARVEST'), len(ordered))
    forced = {tuple(j['tile']) for j in ordered[:first]
              if j['need'] is None and j['op'][0] != 'PLANT'}
    remaining = max(0, view.n_units - len(forced))
    stocks, excluded, seen = {}, 0, set()
    for j in ordered:
        if j['op'][0] != 'HARVEST':
            continue
        pos = tuple(j['tile'])
        if pos in seen:
            raise ValueError('at most one native HARVEST job per tile required')
        seen.add(pos)
        if j['need'] is not None:
            raise ValueError('native HARVEST has no input prerequisite')
        t = view.tile(*pos)
        if not isinstance(t, dict):
            raise ValueError('HARVEST job must refer to its current productive tile')
        product = t.get('crop') or ANIMALS.get(t.get('animal'), {}).get('product')
        amount = t.get('yield_units', 0)
        if product not in PRODUCTS or type(amount) is not int or amount < 0:
            raise ValueError('known product and integer held stock required')
        if pos in forced:
            excluded += 1
        else:
            stocks.setdefault(product, []).append(amount)
    upper = {p: sum(sorted(q, reverse=True)[:remaining]) for p,q in stocks.items()}
    # Include a zero for every product whose only harvestable tiles were claimed.
    for j in ordered:
        if j['op'][0] == 'HARVEST':
            t = view.tile(*j['tile'])
            p = t.get('crop') or ANIMALS.get(t.get('animal'), {}).get('product')
            upper.setdefault(p, 0)
    return {'schema': SCHEMA, 'step': view.step, 'available_workers': view.n_units,
            'harvest_jobs': len(seen), 'unconditional_prefix_tiles': len(forced),
            'remaining_worker_upper': remaining, 'prefix_claimed_harvest_tiles': excluded,
            'harvested_units_upper_by_product': dict(sorted(upper.items())),
            'zero_harvest_certificate': bool(seen) and all(n == 0 for n in upper.values()),
            'stage': 'HARVEST gain into own worker custody in this unit phase only',
            'native_feasibility': 'UNKNOWN', 'policy': 'PINNED_EXECUTOR_ONLY'}
