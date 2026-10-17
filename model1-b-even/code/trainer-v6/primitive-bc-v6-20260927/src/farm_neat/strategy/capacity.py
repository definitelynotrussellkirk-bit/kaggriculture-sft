"""Proof-carrying *conditional* capacity checks for economic Proposals.

Consumer contract
-----------------
``check_feasibility(context, proposal) -> types.Feasibility`` reads a
``CapacityCase`` from ``context.derived['capacity_inputs'][proposal.key]``.
A producer binds its case with ``CapacityCase.bind`` BEFORE publishing that map.
Cases bind the whole proposal, actor observation, own memory, ordered published
shops, and other derived outputs. They are invalid after any of those changes.
No engine, opponent, tape, future shop, hidden state or policy is consulted.

The producer owns extraction/certification of necessary marginal Requirements,
existing commitments, material/ownership prerequisites and the proposal's allowed
routing family. This module owns resource inequalities, temporal scheduling and
certificates, NOT biology, economics, speculative receipts or proposal selection.
Missing certification is UNKNOWN. A provenance string is a caller attestation,
not a proof merely because it has a hash. Bind only trusted, checked producer
outputs; never set all certificates just to get FEASIBLE.

``available`` reuses Requirements, in the proposal's SAME inclusive joint-step
window. Its cash is spendable cash AFTER existing commitments; the proposal's
reserve is an ADDITIONAL hold, counted once. Its other capacities/inventories are
net of existing commitments. available.reserve is unused and must be zero or
UNKNOWN. Dollars are dollars, worker_hours are worker-turns (one game hour).
Every positive answer needs upper requirements <= lower available quantities.
Only EXACT/BOUND estimates certify; a lower requirement > upper availability is
a cut. A scalar pass, pooled resources or a labor/funding screen is not a schedule.

Required named certificates (values: nonempty producer evidence references):
- all models: requirements, resources. They certify necessary costs and the net
  same-window envelope, respectively. Bounds used for cuts must be OPTIMISTIC,
  not artificially protected cash or one conservative candidate's budget.
- chains: fixed_chains, anchored_durations, occupancy_windows,
  continuous_workers, independent_resources, non_temporal_prerequisites,
  complete_coverage. Durations include invariant routed setup and service; the
  exact worker/order is part of the proposal, not one failed routing attempt.
- gaps: fixed_gaps, independent_services, service_windows,
  non_temporal_prerequisites, complete_coverage. The PROPOSAL restricts itself to
  the supplied complete nonoverlapping same-day gaps and 1 or 2 services/gap.
  Services occupy distinct sites, with no material/custody/biological coupling.
- unit_work: necessary_windows, capacity_upper_bound,
  quantity_invariant_funding. A screen pass ALWAYS remains UNKNOWN.
- funded: no_receipts, fixed_upfront_costs, full_days, native_hire_prices,
  necessary_disjoint_work. The existing funded_action_bound is reused unchanged.
  Surviving wage/turn bounds ALWAYS remain UNKNOWN.

For a positive chain/gap answer, non_temporal_prerequisites and complete_coverage
must certify ALL remaining market order timing/cash prefixes, land availability,
materials/custody, readiness and proposed obligations. In particular, a route
witness by itself must not certify an arbitrary economic candidate. Requirements
and available are not automatically extracted from Proposal.candidate.

Model JSON (closed fields; times are absolute half-open [release, deadline),
except unit_work group windows which are local to context.step):
  chains: {kind, chains:[{worker, entry, until, duties:[
      {id, duration, release, deadline, marginal:bool}]}]}
  gaps: {kind, max_per_gap:1|2, services:[{id,site:[x,y],duration,release,deadline}],
      gaps:[{id,worker,a:[x,y],b:[x,y],ta,tb}]}
    ta/tb pin original one-turn anchor actions; available turns are [ta+1,tb).
  unit_work: {kind, capacity:[int], base:[[r,d,count]], bundle:[[r,d,count]],
      copies:int, direction:[int] (optional, zero by default)}
  funded: {kind,full_days,hand_cap,start_cents,committed_cents,protected_cents,
      required_turns,fixed_actions,travel_lower_bound,minimum_hands:[int] (optional)}

The gap frontier applies live-site forgetting and slack dominance to ALL gaps
(the residual-flow family is empty). This is exact for the admitted hereditary
size<=2 menus, bounded rather than an unrestricted placement search. At most 32
sites, 128 gaps, 512 chain duties, 24 unit-work hours; frontier state_budget counts
candidate transitions, including discarded ones. Exceeding limits is UNKNOWN.
Rank-optimal gap witnesses need not minimize travel. A route above a certified
resource limit gives UNKNOWN, never an INFEASIBLE claim about cheaper alternatives.
Its exact marginal work is returned only with an adequately funded witness.

``explain_feasibility`` adds an UNKNOWN reason without changing types.py.
``verify_feasibility`` rechecks the canonical certificate against fresh inputs,
within an explicit caller budget; it is not an independent proof of producer
attestations. Tests additionally use independent brute-force oracles.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
import hashlib
from itertools import combinations, permutations
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from farm_neat.evaluation.funded_capacity import funded_action_bound
from farm_neat.strategy.types import (EVALUATOR_ONLY, Estimate, Evidence, Feasibility,
                                      Proposal, Requirements, StrategyContext, UNKNOWN, Verdict)

# Admitted reductions. D10/D11 and policy constants deliberately do not appear here.
GAP_SIGNATURE = '273568edddb59399a35eb8c66d61db7ee79cb2bba558e4bcf6d1923e1760947b'
GAP_MATCHING = '40738f0fca045a6d9657b65501639688bb0ade57e2189f8b0fda93a81c25c21a'
GAP_FRONTIER = '902d365a3dc33cad71f55b54814ba930aca2375be8d8ce01cd0148327077301f'
CHAIN_CALENDAR = '04909a23f45c7193b26ef768a6a38c2998b497eb065ad348b5f0055acbd199da'
LABOR_BUNDLE = '61f648f0702d5ed00390ba38765dc119ee7b935c4fa1a79531e8b0f0d8071d9e'
LABOR_RAY = '022320075e091cbe2ad9e643a05753b54d1f1dd446fd53d3c4ab4aee6e8de53d'
FUNDED_HANDS = 'aff1d19527368e4fe386a1727dd12949f235ecdf12e9854dd5ace0ded81bf6c6'
COMMON = frozenset(('requirements', 'resources'))
CONDITIONS = {
    'chains': COMMON | {'fixed_chains', 'anchored_durations', 'occupancy_windows',
        'continuous_workers', 'independent_resources', 'non_temporal_prerequisites', 'complete_coverage'},
    'gaps': COMMON | {'fixed_gaps', 'independent_services', 'service_windows',
        'non_temporal_prerequisites', 'complete_coverage'},
    'unit_work': COMMON | {'necessary_windows', 'capacity_upper_bound', 'quantity_invariant_funding'},
    'funded': COMMON | {'no_receipts', 'fixed_upfront_costs', 'full_days',
        'native_hire_prices', 'necessary_disjoint_work'},
}
PARENTS = {'chains': (CHAIN_CALENDAR,), 'gaps': (GAP_SIGNATURE, GAP_MATCHING, GAP_FRONTIER),
           'unit_work': (LABOR_BUNDLE, LABOR_RAY), 'funded': (FUNDED_HANDS,)}


class _Outside(Exception):
    """A well-formed request outside this reduction, never an infeasibility cut."""


def _plain(value):
    """Stable finite data; also detects nested evaluator-only keys in inputs."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        if EVALUATOR_ONLY.intersection(value):
            raise ValueError('evaluator-only keys are not capacity inputs')
        if all(isinstance(k, str) for k in value):
            return {k: _plain(v) for k, v in sorted(value.items())}
        pairs = [(_plain(k), _plain(v)) for k, v in value.items()]
        return {'__keyed_pairs__': sorted(pairs, key=lambda p: json.dumps(p[0], sort_keys=True))}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('capacity inputs must be finite data, not executable objects')


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _binding(context: StrategyContext, proposal: Proposal) -> str:
    return _digest({'step': context.step, 'player': context.player, 'observation': context.observation,
        'memory': context.memory, 'shops': context.shops, 'proposal': proposal.to_dict(),
        'derived': {k: v for k, v in context.derived.items() if k != 'capacity_inputs'}})


def _bindings(context: StrategyContext, proposals: Sequence[Proposal]) -> tuple[str, ...]:
    """Legacy binding bytes for one local, callback-free validation batch.

    Only freshly validated finite DATA is shared: _plain detaches all context
    containers, then JSON bytes and a local SHA256 prefix state are immutable.
    Proposal bytes are captured before hashing. Nothing is cached on a context,
    returned as a reusable prepared object, or trusted from a caller. Publication
    and every direct public check continue to use the unchanged _binding.
    """
    shared = _plain({'step': context.step, 'player': context.player,
        'observation': context.observation, 'memory': context.memory,
        'shops': context.shops,
        'derived': {k: v for k, v in context.derived.items() if k != 'capacity_inputs'}})

    def encode(plain):
        return json.dumps(plain, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, allow_nan=False).encode()

    # These are the sorted top-level fields of _binding, split at "proposal".
    # Removing only the outer braces preserves escaping, floats, keyed-pair
    # ordering, repeated shops, and every byte of the established wire format.
    prefix = encode({k: shared[k] for k in ('derived', 'memory', 'observation', 'player')})[:-1]
    suffix = b',' + encode({k: shared[k] for k in ('shops', 'step')})[1:]
    words = tuple(encode(_plain(p.to_dict())) for p in proposals)
    initial = hashlib.sha256(prefix + b',"proposal":')
    result = []
    for word in words:
        digest = initial.copy()
        digest.update(word)
        digest.update(suffix)
        result.append(digest.hexdigest())
    return tuple(result)


@dataclass(frozen=True)
class CapacityCase:
    """Immutable producer-certified reduction; see the module's input contract."""
    binding: str
    available: Requirements
    model: Mapping[str, Any]
    certificates: Mapping[str, str]
    parents: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.binding, str) or len(self.binding) != 64 or any(c not in '0123456789abcdef' for c in self.binding):
            raise ValueError('binding must be a SHA256')
        if not isinstance(self.available, Requirements):
            raise ValueError('available must be Requirements')
        if self.available.reserve not in (UNKNOWN, Estimate.exact(0)):
            raise ValueError('available.reserve is unused: use zero; available.cash is already net')
        if not isinstance(self.model, Mapping) or not isinstance(self.certificates, Mapping):
            raise ValueError('model and certificates must be mappings')
        if not all(isinstance(k, str) and k and isinstance(v, str) and v.strip()
                   for k, v in self.certificates.items()):
            raise ValueError('certificates map names to nonempty evidence references')
        object.__setattr__(self, 'model', _freeze(_plain(self.model)))
        object.__setattr__(self, 'certificates', _freeze(_plain(self.certificates)))
        object.__setattr__(self, 'parents', tuple(self.parents))
        # Use the existing parent's validation, not a parallel reference contract.
        Feasibility(('capacity', 'validate'), Verdict.UNKNOWN, parents=self.parents)

    @classmethod
    def bind(cls, context: StrategyContext, proposal: Proposal, *, available: Requirements,
             model: Mapping[str, Any], certificates: Mapping[str, str], parents=()):
        return cls(_binding(context, proposal), available, model, certificates, tuple(parents))

    def to_dict(self):
        return {'binding': self.binding, 'available': self.available.to_dict(),
                'model': _plain(self.model), 'certificates': dict(self.certificates),
                'parents': list(self.parents)}


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name}: integer in [{minimum}, {maximum}] required')
    return value


def _fields(data, required, optional=()):
    if not isinstance(data, Mapping) or set(data) - set(required) - set(optional) or set(required) - set(data):
        raise ValueError('closed model fields required: ' + ', '.join(sorted(required)))


def _identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('nonempty worker/duty/gap/service id required')
    return value


def _bounds(value: Estimate, *, requirement=False):
    if value.kind is Evidence.EXACT:
        low = high = Fraction(str(value.value))
    elif value.kind is Evidence.BOUND:
        low = None if value.low is None else Fraction(str(value.low))
        high = None if value.high is None else Fraction(str(value.high))
    else:
        return None, None
    if requirement and any(x is not None and x < 0 for x in (low, high)):
        raise ValueError('negative Requirements are releases/deltas, not required resources')
    return low, high


def _resource_check(required: Requirements, available: Requirements):
    lo_cash, hi_cash = _bounds(required.cash, requirement=True)
    lo_hold, hi_hold = _bounds(required.reserve, requirement=True)
    cash = (None if lo_cash is None or lo_hold is None else lo_cash + lo_hold,
            None if hi_cash is None or hi_hold is None else hi_cash + hi_hold)
    rows = [('cash', cash, available.cash)]
    rows += [(n, _bounds(getattr(required, n), requirement=True), getattr(available, n))
             for n in ('market_slots', 'worker_hours', 'land_quadrants')]
    rows += [('inventory:' + k, _bounds(v, requirement=True), available.inventory.get(k, UNKNOWN))
             for k, v in required.inventory.items()]
    unknown = []
    for name, (need_low, need_high), have in rows:
        have_low, have_high = _bounds(have)
        if need_low is not None and have_high is not None and need_low > have_high:
            return {'kind': 'resource', 'resource': name, 'required_lower': float(need_low),
                    'available_upper': float(have_high)}, unknown
        if need_high is None or have_low is None or need_high > have_low:
            unknown.append(name)
    return None, unknown


def _marginal_hours(required, hours):
    """A certified full schedule supplies exact additional work, not a labor multiplier."""
    lo, hi = _bounds(required.worker_hours, requirement=True)
    if (lo is not None and hours < lo) or (hi is not None and hours > hi):
        raise ValueError('marginal worker_hours disagrees with the supplied schedule')
    return replace(required, worker_hours=Estimate.exact(hours))


def _chains(model, window):
    _fields(model, ('kind', 'chains'))
    workers, ids, schedule, signatures = set(), set(), [], []
    total, marginal = 0, 0
    # Validate the WHOLE family before emitting a cut from any one member.
    for c in model['chains']:
        _fields(c, ('worker', 'entry', 'until', 'duties'))
        worker = _identifier(c['worker'])
        if worker in workers:
            raise _Outside('worker reused across chains: joint calendar not certified')
        workers.add(worker)
        entry = _integer(c['entry'], 'entry', 0, 718)
        until = _integer(c['until'], 'until', 1, 719)
        if not window[0] <= entry < until <= window[1] + 1:
            raise _Outside('chain availability is outside the proposal window')
        if entry // 24 != (until - 1) // 24:
            raise _Outside('chain crosses a worker-generation/day boundary')
        for j in c['duties']:
            _fields(j, ('id', 'duration', 'release', 'deadline', 'marginal'))
            ident = _identifier(j['id'])
            if ident in ids:
                raise ValueError('duplicate duty id')
            ids.add(ident); total += 1
            _integer(j['duration'], 'duration', 1)
            r = _integer(j['release'], 'release', 0, 718)
            d = _integer(j['deadline'], 'deadline', 1, 719)
            if r >= d or type(j['marginal']) is not bool:
                raise ValueError('duty needs release < deadline and boolean marginal')
            if j['marginal']:
                marginal += j['duration']
    if total > 512 or len(workers) > 128:
        raise _Outside('chain input budget exceeded')
    for c in model['chains']:
        time = c['entry']; P, R, U = 0, None, None
        for j in c['duties']:
            p, r, d = j['duration'], j['release'], min(j['deadline'], c['until'])
            start = max(time, r); finish = start + p
            if finish > d:
                return {'cut': {'kind': 'chain_deadline', 'worker': c['worker'], 'duty': j['id'],
                    'entry': c['entry'], 'earliest_start': start, 'earliest_finish': finish,
                    'deadline': d, 'prefix': schedule}}, marginal
            # Exact three-scalar composition; the earliest-start witness is separate.
            singleton_U = d - p
            U = singleton_U if U is None else min(U, singleton_U - P)
            R = r + p if R is None else max(R + p, r + p)
            P += p
            schedule.append({'worker': c['worker'], 'duty': j['id'], 'start': start,
                             'finish': finish, 'duration': p, 'marginal': j['marginal']})
            time = finish
        signatures.append({'worker': c['worker'], 'P': P, 'R': R, 'U': U})
        if P:
            assert c['entry'] <= U and time == max(c['entry'] + P, R)
    return {'schedule': schedule, 'signatures': signatures}, marginal


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _site(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('site must be [x, y]')
    return tuple(_integer(x, 'coordinate', 0, 9) for x in value)


def _visit(gap, order, services):
    """Earliest Manhattan route through a fixed order, including return to the anchor."""
    time, pos, traveled, rows = gap['ta'] + 1, gap['a'], 0, []
    for i in order:
        s = services[i]; move = _distance(pos, s['site']); traveled += move
        start = max(time + move, s['release']); finish = start + s['duration']
        if finish > s['deadline']:
            return None
        rows.append({'gap': gap['id'], 'worker': gap['worker'], 'service': s['id'],
                     'site': list(s['site']), 'start': start, 'finish': finish})
        time, pos = finish, s['site']
    traveled += _distance(pos, gap['b']); time += _distance(pos, gap['b'])
    if time > gap['tb']:
        return None
    extra = traveled - _distance(gap['a'], gap['b']) + sum(services[i]['duration'] for i in order)
    return {'schedule': rows, 'route': {'gap': gap['id'], 'worker': gap['worker'],
            'a': list(gap['a']), 'b': list(gap['b']), 'ta': gap['ta'], 'tb': gap['tb'],
            'services': [services[i]['id'] for i in order], 'return_at': time, 'extra_turns': extra}}


def _gap_menus(model, window):
    _fields(model, ('kind', 'max_per_gap', 'services', 'gaps'))
    limit = _integer(model['max_per_gap'], 'max_per_gap', 1)
    if limit > 2:
        raise _Outside('more than two services/gap is outside the admitted family')
    services, gaps = model['services'], model['gaps']
    if len(services) > 32 or len(gaps) > 128:
        raise _Outside('gap input budget exceeded')
    ids, sites = set(), set()
    for s in services:
        _fields(s, ('id', 'site', 'duration', 'release', 'deadline'))
        ident = _identifier(s['id']); site = _site(s['site'])
        if ident in ids:
            raise ValueError('duplicate service id')
        if site in sites:
            raise _Outside('multiple services at one site need a coupled-service model')
        ids.add(ident); sites.add(site)
        _integer(s['duration'], 'duration', 1)
        r = _integer(s['release'], 'release', 0, 718)
        d = _integer(s['deadline'], 'deadline', 1, 719)
        if r >= d:
            raise ValueError('service needs release < deadline')
    ids, by_worker = set(), {}
    for g in gaps:
        _fields(g, ('id', 'worker', 'ta', 'tb', 'a', 'b'))
        ident = _identifier(g['id']); worker = _identifier(g['worker'])
        if ident in ids:
            raise ValueError('duplicate gap id')
        ids.add(ident); _site(g['a']); _site(g['b'])
        a = _integer(g['ta'], 'ta', 0, 718); b = _integer(g['tb'], 'tb', 0, 718)
        if a >= b:
            raise ValueError('gap needs ta < tb')
        if a // 24 != b // 24 or a < window[0] or b > window[1]:
            raise _Outside('gap outside proposal window or crossing worker reset')
        if _distance(g['a'], g['b']) > b - a - 1:
            raise _Outside('baseline gap endpoints are themselves unreachable')
        by_worker.setdefault(worker, []).append(g)
    for gs in by_worker.values():
        for a, b in zip(sorted(gs, key=lambda g: g['ta']), sorted(gs, key=lambda g: g['ta'])[1:]):
            if a['tb'] > b['ta'] or (a['tb'] == b['ta'] and a['b'] != b['a']):
                raise _Outside('overlapping or inconsistent same-worker gaps')
            # Distinct closed gaps also need a feasible bridge, not teleportation.
            if a['tb'] < b['ta'] and _distance(a['b'], b['a']) > b['ta'] - a['tb'] - 1:
                raise _Outside('inconsistent anchors between same-worker gaps')
    menus = []
    for g in gaps:
        options = {0: ()}
        singles = [i for i in range(len(services)) if _visit(g, (i,), services) is not None]
        options.update({1 << i: (i,) for i in singles})
        if limit == 2:
            for pair in combinations(singles, 2):
                valid = []
                for order in permutations(pair):
                    visit = _visit(g, order, services)
                    if visit is not None:
                        valid.append((visit['route']['extra_turns'], order))
                if valid:
                    _, order = min(valid)
                    options[sum(1 << i for i in pair)] = order
        menus.append(options)
    return services, gaps, menus


def _matching(menus, services, gaps):
    adjacency = [[g for g, menu in enumerate(menus) if 1 << s in menu] for s in range(len(services))]
    owner = {}
    def augment(s, visited):
        for g in adjacency[s]:
            if g in visited:
                continue
            visited.add(g)
            if g not in owner or augment(owner[g], visited):
                owner[g] = s; return True
        return False
    for s in range(len(services)):
        augment(s, set())
    if len(owner) == len(services):
        return [(g, (s,)) for g, s in sorted(owner.items())], None
    reachable_s = set(range(len(services))) - set(owner.values()); reachable_g = set()
    queue = list(sorted(reachable_s))
    for s in queue:
        for g in adjacency[s]:
            reachable_g.add(g)
            if g in owner and owner[g] not in reachable_s:
                reachable_s.add(owner[g]); queue.append(owner[g])
    assert len(reachable_s) > len(reachable_g)
    return None, {'kind': 'hall', 'services': [services[s]['id'] for s in sorted(reachable_s)],
                  'neighbor_gaps': [gaps[g]['id'] for g in sorted(reachable_g)]}


def gap_frontier(menus: Sequence[Sequence[int]], n_sites: int, *, state_budget=4096):
    """Exact rank of hereditary empty/singleton/pair bitmask menus, or UNKNOWN.

    Public for producers already owning exact menus. Geometry must be certified
    separately. Every retained state is [live_used_mask, best_served_count]; this
    applies the admitted slack dominance, NOT plain inclusion on forgotten keys.
    A finite rank cut retains all frontier layers. No worker/cash facts inferred.
    """
    _integer(state_budget, 'state_budget', 1, 100000)
    _integer(n_sites, 'n_sites', 0, 32)
    if len(menus) > 128:
        return {'status': 'UNKNOWN', 'reason': 'gap input budget exceeded'}
    canonical = []
    for menu in menus:
        opts = sorted(set(menu))
        if not opts or 0 not in opts:
            raise ValueError('hereditary gap menu must contain empty option')
        for mask in opts:
            _integer(mask, 'service mask', 0, (1 << n_sites) - 1)
            if mask.bit_count() > 2 or any(1 << i not in opts for i in range(n_sites) if mask & (1 << i)):
                raise ValueError('menus must be hereditary and size <= 2')
        canonical.append(opts)
    future = [0] * (len(canonical) + 1)
    for g in range(len(canonical) - 1, -1, -1):
        future[g] = future[g + 1]
        for mask in canonical[g]:
            future[g] |= mask
    frontier = {0: (0, ())}; layers = [[[0, 0]]]; transitions = comparisons = 0
    for g, opts in enumerate(canonical):
        new = {}
        for used, (score, history) in sorted(frontier.items()):
            for mask in opts:
                transitions += 1
                if transitions > state_budget:
                    return {'status': 'UNKNOWN', 'reason': 'frontier transition budget exceeded',
                            'transitions': transitions, 'comparisons': comparisons}
                if mask & used:
                    continue
                live = (used | mask) & future[g + 1]
                candidate = (score + mask.bit_count(), history + (mask,))
                old = new.get(live)
                if old is None or candidate[0] > old[0] or (candidate[0] == old[0] and candidate[1] < old[1]):
                    new[live] = candidate
        keep = {}
        for u, record in sorted(new.items()):
            dominated = False
            for v, other in sorted(new.items()):
                comparisons += 1
                if comparisons > 32 * state_budget:
                    return {'status': 'UNKNOWN', 'reason': 'frontier dominance budget exceeded',
                            'transitions': transitions, 'comparisons': comparisons}
                if u != v and u & v == u and other[0] - v.bit_count() >= record[0] - u.bit_count():
                    dominated = True; break
            if not dominated:
                keep[u] = record
        frontier = keep
        layers.append([[u, score] for u, (score, _) in sorted(frontier.items())])
    score, history = max(frontier.values(), key=lambda row: (row[0], tuple(-m for m in row[1])))
    return {'status': 'EXACT', 'rank': score, 'history': list(history), 'layers': layers,
            'transitions': transitions, 'comparisons': comparisons}


def _gaps(model, window, state_budget):
    services, gaps, menus = _gap_menus(model, window)
    stats = {'services': len(services), 'gaps': len(gaps), 'menu_options': sum(map(len, menus))}
    if model['max_per_gap'] == 1:
        assignment, cut = _matching(menus, services, gaps)
        if cut:
            return {'cut': cut}, None, stats
    else:
        rank = gap_frontier([list(m) for m in menus], len(services), state_budget=state_budget)
        stats.update({k: rank[k] for k in ('transitions', 'comparisons') if k in rank})
        if rank['status'] != 'EXACT':
            raise _Outside(rank['reason'])
        if rank['rank'] < len(services):
            return {'cut': {'kind': 'gap_frontier', 'required': len(services), 'rank': rank['rank'],
                            'layers': rank['layers']}}, None, stats
        assignment = [(g, menus[g][mask]) for g, mask in enumerate(rank['history']) if mask]
    rows, routes, hours = [], [], 0
    for g, order in assignment:
        visit = _visit(gaps[g], order, services)
        assert visit is not None
        rows.extend(visit['schedule']); routes.append(visit['route']); hours += visit['route']['extra_turns']
    return {'schedule': rows, 'routes': routes}, hours, stats


def _labor_inputs(capacity, base, bundle, direction):
    cap = tuple(_integer(c, 'capacity', 0) for c in capacity)
    if not 1 <= len(cap) <= 24:
        raise ValueError('unit-work horizon must be 1..24')
    v = (0,) * len(cap) if direction is None else tuple(direction)
    if len(v) != len(cap) or any(type(x) is not int for x in v):
        raise ValueError('one integer direction per capacity hour required')
    groups = []
    for values in (base, bundle):
        if len(values) > 512:
            raise ValueError('unit-work group limit is 512')
        normalized = []
        for row in values:
            if len(row) != 3:
                raise ValueError('unit workload group is [release, deadline, count]')
            r, d, n = row
            _integer(r, 'release', 0, len(cap) - 1); _integer(d, 'deadline', 1, len(cap))
            _integer(n, 'count', 0)
            if r >= d:
                raise ValueError('unit job needs release < deadline')
            normalized.append((r, d, n))
        groups.append(normalized)
    return cap, groups[0], groups[1], v


def _intervals(cap, base, bundle, direction):
    for a in range(len(cap)):
        for b in range(a + 1, len(cap) + 1):
            C = sum(cap[a:b]); V = sum(direction[a:b])
            B = sum(n for r, d, n in base if a <= r and d <= b)
            M = sum(n for r, d, n in bundle if a <= r and d <= b)
            yield a, b, C, V, B, M


def labor_ray(capacity, base, bundle, direction=None):
    """Invert B+nM <= c+nv once. Half-open local unit-job windows, not routed work.

    With an infeasible base the prefix theorem does not apply: return
    BASE_INFEASIBLE, never claim every positive n fails. None endpoint on an
    admissible base means unbounded. A zero direction is the bundle endpoint;
    a zero bundle is the capacity-ray endpoint. No quantity enumeration.
    """
    cap, base, bundle, v = _labor_inputs(capacity, base, bundle, direction)
    endpoint, cut = None, None
    for a, b, C, V, B, M in _intervals(cap, base, bundle, v):
        if B > C:
            return {'status': 'BASE_INFEASIBLE', 'endpoint': None,
                    'cut': {'interval': [a, b], 'demand': B, 'capacity': C}}
        net = M - V
        if net > 0 and (endpoint is None or (C - B) // net < endpoint):
            endpoint = (C - B) // net
            cut = {'interval': [a, b], 'slack': C - B, 'net': net,
                   'rejected_copies': endpoint + 1}
    return {'status': 'EXACT_UNIT_RAY', 'endpoint': endpoint, 'cut': cut}


def _unit_work(model, window, now):
    _fields(model, ('kind', 'capacity', 'base', 'bundle', 'copies'), ('direction',))
    if len(model['capacity']) > 24 or len(model['base']) > 512 or len(model['bundle']) > 512:
        raise _Outside('unit-work input budget exceeded')
    cap, base, bundle, v = _labor_inputs(model['capacity'], model['base'], model['bundle'], model.get('direction'))
    copies = _integer(model['copies'], 'copies', 0)
    if now + len(cap) > window[1] + 1 or now // 24 != (now + len(cap) - 1) // 24:
        raise _Outside('unit-work row crosses its certified window/day')
    for a, b, C, V, B, M in _intervals(cap, base, bundle, v):
        demand, capacity = B + copies * M, C + copies * V
        if demand > capacity:
            return {'cut': {'kind': 'interval', 'origin': now, 'interval': [a, b],
                            'copies': copies, 'demand': demand, 'capacity': capacity}}
    return {'reason': 'unit-job screen passes; routing/material/worker feasibility remains UNKNOWN',
            'ray': labor_ray(cap, base, bundle, v)}


def _funded(model, window, now):
    names = ('kind', 'full_days', 'hand_cap', 'start_cents', 'committed_cents', 'protected_cents',
             'required_turns', 'fixed_actions', 'travel_lower_bound')
    _fields(model, names, ('minimum_hands',))
    for n in names[1:]:
        _integer(model[n], n, 0)
    days = model['full_days']
    if (not 1 <= days <= 29 or model['hand_cap'] > 10 or now % 24 or
            now + 24 * days > 696 or window != (now, now + 24 * days - 1)):
        raise _Outside('funded theorem needs matching complete pre-final days and hand_cap <= 10')
    row = funded_action_bound(**{k: model[k] for k in names[1:] if k != 'required_turns'},
                              minimum_hands=model.get('minimum_hands'))
    if row['binding'] == 'cash_prefix':
        return {'cut': {'kind': 'funded_cash', 'available_wage_cents': row['available_wage_cents'],
                        'minimum_floor_wage_cents': 100 * row['frontier']['minimum_wage_dollars'][0]}}
    # Do not clamp a negative residual to zero; existing work may itself overload.
    remaining = row['unit_turns_upper'] - model['fixed_actions'] - model['travel_lower_bound']
    if model['required_turns'] > remaining:
        return {'cut': {'kind': 'funded_turns', 'required_lower': model['required_turns'],
                'capacity_upper': remaining, 'hand_days_upper': row['hand_days_upper'],
                'unit_turns_upper': row['unit_turns_upper'], 'minimum_wage_dollars': row['minimum_wage_dollars'],
                'cost_minimizing_hire_counts': row['cost_minimizing_hire_counts']}}
    return {'reason': 'funding/turn upper bound survives; no native schedule is certified'}


def _unbound(proposal_key, marginal, parents, reason):
    return Feasibility(proposal_key, Verdict.UNKNOWN, marginal=marginal, parents=parents), reason, {}


def _analyze(context, proposal, state_budget):
    _integer(state_budget, 'state_budget', 1, 100000)
    cases = context.derived.get('capacity_inputs', {})
    case = cases.get(proposal.key) if isinstance(cases, Mapping) else None
    if not isinstance(case, CapacityCase):
        return _unbound(proposal.key, proposal.requirements, tuple(proposal.parents),
                        'no producer-certified capacity case')
    if case.binding != _binding(context, proposal):
        return _unbound(proposal.key, proposal.requirements, tuple(proposal.parents),
                        'stale context/proposal/derived binding')
    return _analyze_case(context.step, proposal.key, proposal.requirements,
                         tuple(proposal.parents), case, state_budget)


def _analyze_case(step, proposal_key, marginal, parents, case, state_budget):
    """Shared capacity logic AFTER a fresh binding check; no serialization owner."""
    parents = tuple(sorted(set(parents) | set(case.parents)))
    stats = {}
    def unknown(reason):
        return Feasibility(proposal_key, Verdict.UNKNOWN, marginal=marginal, parents=parents), reason, stats
    if marginal.window is None or case.available.window != marginal.window or marginal.window[0] < step:
        return unknown('missing, stale or mismatched same-window resource envelope')
    if COMMON - set(case.certificates):
        return unknown('missing necessary-requirement/net-resource certificates')
    header = {'schema': 'farm-capacity-certificate-v1', 'binding': case.binding,
              'case_sha256': _digest(case.to_dict()), 'conditions': dict(case.certificates),
              'scope': 'conditional capacity of this proposal under the supplied producer certificates'}
    def rejected(cut):
        return Feasibility(proposal_key, Verdict.INFEASIBLE, certificate=header | {'cut': cut},
                           marginal=marginal, parents=parents), 'checkable conditional cut', stats
    cut, missing_resources = _resource_check(marginal, case.available)
    if cut:
        return rejected(cut)
    kind = case.model.get('kind')
    if kind not in CONDITIONS:
        return unknown('unmapped/unsupported capacity model')
    missing = CONDITIONS[kind] - set(case.certificates)
    if missing:
        return unknown('missing model certificates: ' + ', '.join(sorted(missing)))
    parents = tuple(sorted(set(parents) | set(PARENTS[kind])))
    try:
        if kind == 'chains':
            result, hours = _chains(case.model, marginal.window)
            marginal = _marginal_hours(marginal, hours)
        elif kind == 'gaps':
            result, hours, stats = _gaps(case.model, marginal.window, state_budget)
            if hours is not None:
                # Rank maximization does not minimize detour cost. This particular
                # route's cost is not a necessary lower bound over all assignments.
                lo, hi = _bounds(marginal.worker_hours, requirement=True)
                if (lo is not None and hours < lo) or (hi is not None and hours > hi):
                    return unknown('gap witness disagrees with marginal bounds; alternatives not optimized')
                candidate = _marginal_hours(marginal, hours)
                candidate_cut, candidate_missing = _resource_check(candidate, case.available)
                if candidate_cut or candidate_missing:
                    return unknown('gap witness lacks sufficient resources; alternatives not optimized')
                marginal = candidate
        elif kind == 'unit_work':
            result = _unit_work(case.model, marginal.window, step)
        else:
            result = _funded(case.model, marginal.window, step)
    except _Outside as error:
        return unknown(str(error))
    if 'cut' in result:
        return rejected(result['cut'])
    if kind in ('unit_work', 'funded'):
        return unknown(result['reason'])
    cut, missing_resources = _resource_check(marginal, case.available)
    if cut:
        return rejected(cut)
    if missing_resources:
        return unknown('no sufficient resource bounds: ' + ', '.join(missing_resources))
    return Feasibility(proposal_key, Verdict.FEASIBLE, witness=header | result,
                       marginal=marginal, parents=parents), 'certified conditional schedule', stats


def check_feasibility(context: StrategyContext, proposal: Proposal, *, state_budget=4096) -> Feasibility:
    """Return a witness, a cut, or UNKNOWN. Malformed data raises ValueError.

    No proposal is emitted or executed and no objective delta is altered. Other
    modules may remain unimplemented; only their certified derived data is needed.
    """
    if not isinstance(context, StrategyContext) or not isinstance(proposal, Proposal):
        raise ValueError('StrategyContext and Proposal required')
    return _analyze(context, proposal, state_budget)[0]


def explain_feasibility(context: StrategyContext, proposal: Proposal, *, state_budget=4096) -> dict:
    """Companion diagnostics because the frozen Feasibility type has no UNKNOWN reason."""
    result, reason, stats = _analyze(context, proposal, state_budget)
    return {'result': _plain(result), 'reason': reason, 'stats': stats}


def _explain_feasibilities(context: StrategyContext, proposals: Sequence[Proposal], *, state_budget=4096) -> tuple[dict, ...]:
    """Pipeline-only, bounded validation cut; never spans a publisher or a yield.

    Snapshot only immutable analysis inputs (step/key/Requirements/parents and
    CapacityCases). All mutable observation, memory, derived and candidate data
    participates in freshly validated binding bytes; none is consulted after
    that capture. No prepared state escapes this call, so publishing, mutation,
    later steps and direct public calls cannot reuse it. Cases without a producer
    retain the legacy early UNKNOWN, without unnecessarily validating their data.
    """
    proposals = tuple(proposals)
    if not proposals:
        return ()
    _integer(state_budget, 'state_budget', 1, 100000)
    step = context.step
    cases = context.derived.get('capacity_inputs', {})
    entries = tuple((p.key, p.requirements, tuple(p.parents),
                     cases.get(p.key) if isinstance(cases, Mapping) else None) for p in proposals)
    bound_proposals = tuple(p for p, entry in zip(proposals, entries)
                            if isinstance(entry[3], CapacityCase))
    bindings = iter(_bindings(context, bound_proposals) if bound_proposals else ())
    details = []
    for key, marginal, parents, case in entries:
        if not isinstance(case, CapacityCase):
            analysis = _unbound(key, marginal, parents, 'no producer-certified capacity case')
        elif case.binding != next(bindings):
            analysis = _unbound(key, marginal, parents, 'stale context/proposal/derived binding')
        else:
            analysis = _analyze_case(step, key, marginal, parents, case, state_budget)
        result, reason, stats = analysis
        details.append({'result': _plain(result), 'reason': reason, 'stats': stats})
    return tuple(details)


def verify_feasibility(context: StrategyContext, proposal: Proposal, result: Feasibility, *, state_budget=4096) -> bool:
    """Recompute canonical certificate; never trust claimed rank, timings or cash.

    Uses only the caller's budget. An input/certificate tamper or different
    canonical witness returns False; a genuinely valid alternative schedule is
    not accepted by this *canonical* verifier. Does not authenticate attestations.
    """
    if not isinstance(result, Feasibility):
        return False
    try:
        return result == check_feasibility(context, proposal, state_budget=state_budget)
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description='Print a synthetic conditional capacity witness; no games.')
    parser.add_argument('--example', action='store_true')
    args = parser.parse_args(argv)
    if args.example:
        zero = Estimate.exact(0)
        needed = Requirements(cash=zero, reserve=zero, market_slots=zero, land_quadrants=zero, window=(0, 23))
        p = Proposal('production', 'service:example', {'family': 'service'}, ('labor',), needed)
        ctx = StrategyContext(0, 0, {'player': 0, 'step': 0})
        case = CapacityCase.bind(ctx, p, available=replace(needed, worker_hours=Estimate.exact(24)),
            model={'kind': 'chains', 'chains': [{'worker': 'farmer', 'entry': 0, 'until': 24,
                'duties': [{'id': 'service', 'duration': 2, 'release': 0, 'deadline': 24, 'marginal': True}]}]},
            certificates={k: 'synthetic-example:explicit-precondition' for k in CONDITIONS['chains']})
        ctx = ctx.with_derived('capacity_inputs', {p.key: case})
        print(json.dumps(_plain(check_feasibility(ctx, p)), indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
