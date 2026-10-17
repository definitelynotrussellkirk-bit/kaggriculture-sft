"""Conditional, actor-bound minimum wages; not a production-profit certificate.

``certify_labor(context, tape, problem, support=...) -> LaborCertificate`` accepts
ONLY explicit, independently certified remaining-day workloads and ordered cash
commits. It never infers windows, routes, hands or future cash from tape.actions.
A bare aggregate/prefix-capacity pass, a realized evaluator trace, and an empirical
turns/service ratio are not support. Missing conditions yield UNKNOWN, not zero.

Problem v1 is finite JSON: {schema, days}. EVERY remaining day (including empty
ones) has {day, mode, hand_cap, protected_cents, events, jobs, chains}. Events are
{hour, delta_cents, hire_slots}, split and ordered within each hour. UNIT_WINDOWS
jobs are {release, deadline, count}, with INCLUSIVE unit-action windows; chains
must be empty. FIXED_CHAINS jobs must be empty; each chain is {worker, entry,
anchor, jobs}, with jobs {duration, release, deadline, before, after}. Chain
windows are occupancy [release, deadline), with EXCLUSIVE deadlines. Anchors are
caller-certified names/digests of the complete relevant boundary state (position,
cargo, materials), not merely tile names. Adjacent states must match exactly.
Worker 0 is the farmer; hand j is the j-th hire and first acts at hire_hour+1.

LaborSupport binds cost_key(context, WHOLE tape), problem_key(problem), and the
order-sensitive own custody key. Its ten prerequisite entries must EACH supply
EXACT(1) and nonempty actor-legal provenance. Caller evidence establishes the
premises; this module computes the schedule/wage consequence, not their scientific
authenticity. It neither loads receipts nor accepts callbacks/evaluator snapshots.
A supported quote is the least wage in this fixed-law problem, NOT a forecast of
what the executor will spend. PortfolioValuator consumes the quote conditionally.

K-LABOR-INTERVAL-22A8C6E5 supplies interval feasibility over earliest safe hire
rows. K-ANCHORED-CHAIN-CALENDAR-F2A9949C supplies the (P,R,U) fixed-chain fold.
The small kernels below productionize those admitted algorithms, without runtime
imports from work/ or modifications to their immutable evidence. Native hire_bill
is reused. Daily minimum wages compose only under explicit day_independence:
non-wage cash, windows, anchors and outputs are invariant to the permitted timing
choices; native overnight resets remove hands; no unmodelled coupled obligation.
Spending fewer wages then leaves weakly more cash for every later fixed day.

Costs are integral cents internally; Estimate and PortfolioValuator use dollars.
Output retains rejected rows, overloaded intervals or worker witnesses, canonical
hire events, constructive schedules and cash minima. UNKNOWN/infeasible results
never return a partial wage. No route search, native game, I/O or policy change.
"""
from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

from farm_neat.evaluation.funded_capacity import hire_bill
from farm_neat.strategy.types import Estimate, Evidence, UNKNOWN, _parents
from farm_neat.strategy.valuation import (
    _custody_order_key, _hash, _json, _public, _validate_tape, _wire,
    actor_key, cost_key, tape_key,
)

SCHEMA = 'farm-labor-problem-v1'
PARENTS = (
    '38dd573f312c72a31db2ceb4ab4d421edba6dc5f2f9b599585de9ddcf350fa84',
    '04909a23f45c7193b26ef768a6a38c2998b497eb065ad348b5f0055acbd199da',
    'dd8643b1d848039aa3f46fc06348cdfa99742e4b2e5b989b7403c76640c9913f',
)
REQUIRED_PREREQUISITES = (
    'whole_tape_workload', 'release_windows', 'worker_assignment', 'route',
    'custody', 'materials', 'cash', 'hire_slots', 'day_independence', 'default_rules',
)


def _integer(value, name, low=0, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f'{name}: integer in [{low}, {high}] required')
    return value


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _fields(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise ValueError(name + ': explicit closed-schema fields required')


def _name(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError('explicit bounded anchor-state name/digest required')


def problem_key(problem) -> str:
    """Hash all explicit scheduling/cash inputs; does not certify any input."""
    return _hash(_wire(problem))


def custody_key(context) -> str:
    """Native overnight overflow respects own bag insertion order; retain it."""
    return _custody_order_key(context)


@dataclass(frozen=True)
class Prerequisite:
    """An actor-legal conditional premise; observations/heuristics are not proof."""
    evidence: Estimate = UNKNOWN
    parents: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.evidence, Estimate):
            raise ValueError('prerequisite evidence must be an Estimate')
        object.__setattr__(self, 'parents', _parents(self.parents))

    def to_dict(self):
        return dict(evidence=self.evidence.to_dict(), parents=list(self.parents))


@dataclass(frozen=True)
class LaborSupport:
    cost_key: str
    problem_key: str
    custody_key: str
    prerequisites: Mapping[str, Prerequisite]
    scope: str = 'ACTOR_CONDITIONAL'

    def __post_init__(self):
        if not all(_digest(x) for x in (self.cost_key, self.problem_key, self.custody_key)):
            raise ValueError('support needs full cost/problem/custody SHA256 bindings')
        if not isinstance(self.prerequisites, Mapping):
            raise ValueError('explicit prerequisite mapping required')
        facts = dict(self.prerequisites)
        if set(facts) - set(REQUIRED_PREREQUISITES) or any(not isinstance(v, Prerequisite) for v in facts.values()):
            raise ValueError('unknown prerequisite name or untyped evidence')
        object.__setattr__(self, 'prerequisites', MappingProxyType(facts))

    def to_dict(self):
        return dict(cost_key=self.cost_key, problem_key=self.problem_key, custody_key=self.custody_key,
                    scope=self.scope, prerequisites={k: v.to_dict() for k, v in self.prerequisites.items()})


@dataclass(frozen=True)
class LaborBudget:
    max_events: int = 3000
    max_job_groups: int = 3000
    max_interval_checks: int = 100_000

    def __post_init__(self):
        for name, cap in (('max_events', 10_000), ('max_job_groups', 10_000), ('max_interval_checks', 100_000)):
            _integer(getattr(self, name), name, 0, cap)


@dataclass(frozen=True)
class LaborCertificate:
    """Immutable computed receipt. ``to_dict`` returns a defensive JSON copy.

    ``matches`` checks current binding, not the truth of caller-supplied premises.
    No executable-profit or routed-realization label is implied by CERTIFIED.
    """
    cost_key: str
    actor_key: str
    tape_key: str
    custody_key: str | None
    problem_key: str
    status: str
    labor_cash: Estimate
    parents: tuple[str, ...]
    diagnostics: tuple[str, ...]
    _details_json: str

    def __post_init__(self):
        if not all(_digest(x) for x in (self.cost_key, self.actor_key, self.tape_key, self.problem_key)):
            raise ValueError('certificate binding hashes required')
        if self.custody_key is not None and not _digest(self.custody_key):
            raise ValueError('invalid custody binding')
        if self.status not in ('CERTIFIED', 'UNKNOWN', 'INFEASIBLE_UNDER_CONDITIONS'):
            raise ValueError('invalid labor certificate status')
        if (not isinstance(self.labor_cash, Estimate)
                or (self.status == 'CERTIFIED' and (self.labor_cash.kind is not Evidence.EXACT
                    or self.labor_cash.value < 0 or self.custody_key is None))
                or (self.status != 'CERTIFIED' and self.labor_cash != UNKNOWN)):
            raise ValueError('only a complete conditional certificate carries an exact nonnegative wage')
        object.__setattr__(self, 'parents', _parents(self.parents))
        object.__setattr__(self, 'diagnostics', tuple(self.diagnostics))
        if self.status == 'CERTIFIED' and not self.parents:
            raise ValueError('known wages require parents')
        details = _wire(json.loads(self._details_json))
        _fields(details, 'problem support days interval_checks budget', 'certificate details')
        if problem_key(details['problem']) != self.problem_key:
            raise ValueError('certificate problem binding differs from the retained request')
        if _hash({'actor': self.actor_key, 'tape': self.tape_key}) != self.cost_key:
            raise ValueError('certificate context/tape binding is inconsistent')
        if self.status == 'CERTIFIED':
            proof = details['support']
            if (not isinstance(proof, dict) or proof.get('scope') != 'ACTOR_CONDITIONAL'
                    or (proof.get('cost_key'), proof.get('problem_key'), proof.get('custody_key'))
                    != (self.cost_key, self.problem_key, self.custody_key)):
                raise ValueError('certificate support binding differs from the retained premises')
            premises = proof.get('prerequisites', {})
            for name in REQUIRED_PREREQUISITES:
                fact = premises.get(name, {})
                if (fact.get('evidence') != Estimate.exact(1).to_dict() or not fact.get('parents')
                        or not set(fact['parents']).issubset(self.parents)):
                    raise ValueError('certificate prerequisite missing or uncertified')
            if len(details['days']) != len(details['problem']['days']) or not details['days']:
                raise ValueError('certificate needs a whole-tape wage witness')
            cents = 0
            for day in details['days']:
                wage = 100 * hire_bill(day['minimum_hands'])
                if (day['minimum_wage_cents'] != wage or day['selected']['wage_cents'] != wage
                        or day['selected']['hands'] != day['minimum_hands'] or not day['selected']['feasible']):
                    raise ValueError('inconsistent conditional wage witness')
                cents += wage
            if self.labor_cash != Estimate.exact(cents / 100):
                raise ValueError('certificate wage witness and quoted dollars differ')

    def matches(self, context, tape) -> bool:
        # actor leakage remains a programming error, not an unavailable forecast.
        key = actor_key(context)
        try:
            return (self.status == 'CERTIFIED' and self.actor_key == key
                    and self.tape_key == tape_key(tape) and self.cost_key == cost_key(context, tape)
                    and self.custody_key == custody_key(context))
        except (ValueError, KeyError, TypeError):
            return False

    def summary(self):
        return dict(status=self.status, cost_key=self.cost_key, actor_key=self.actor_key,
                    tape_key=self.tape_key, custody_key=self.custody_key, problem_key=self.problem_key,
                    labor_cash=self.labor_cash.to_dict(), parents=list(self.parents),
                    diagnostics=list(self.diagnostics), scope='CONDITIONAL_MINIMUM_WAGE',
                    native_realization='UNKNOWN', executable_profit='UNKNOWN')

    def to_dict(self):
        return dict(schema='farm-labor-certificate-v1', **self.summary(), **json.loads(self._details_json))


def _parse(problem, from_day, budget):
    _fields(problem, 'schema days', 'problem')
    if problem['schema'] != SCHEMA or not isinstance(problem['days'], list):
        raise ValueError('unsupported labor problem schema')
    days = problem['days']
    if len(days) != 30 - from_day:
        raise ValueError('WHOLE_TAPE_DAYS_REQUIRED: explicitly include every remaining day, even empty days')
    event_count = job_count = 0
    for expected, day in enumerate(days, from_day):
        _fields(day, 'day mode hand_cap protected_cents events jobs chains', 'day')
        if type(day['day']) is not int or day['day'] != expected:
            raise ValueError('WHOLE_TAPE_DAYS_REQUIRED: chronological consecutive remaining days')
        hours = 23 if expected == 29 else 24
        _integer(day['hand_cap'], 'hand_cap', 0, 10)
        _integer(day['protected_cents'], 'protected_cents')
        if any(not isinstance(day[k], list) for k in ('events', 'jobs', 'chains')):
            raise ValueError('explicit event/job/chain arrays required')
        event_count += len(day['events'])
        job_count += len(day['jobs'])
        if event_count > budget.max_events or job_count > budget.max_job_groups:
            raise ValueError('LABOR_BUDGET_EXHAUSTED')
        previous = -1
        slots = [0] * hours
        for e in day['events']:
            _fields(e, 'hour delta_cents hire_slots', 'event')
            h = _integer(e['hour'], 'hour', 0, hours - 1)
            _integer(e['delta_cents'], 'delta_cents', -(10**15), 10**15)
            _integer(e['hire_slots'], 'hire_slots', 0, 10)
            slots[h] += e['hire_slots']
            if h < previous or slots[h] > 10 or (e['delta_cents'] and e['hire_slots']):
                raise ValueError('ordered split cash commits and reserved HIRE slots required')
            previous = h
        if day['mode'] == 'UNIT_WINDOWS':
            if day['chains']:
                raise ValueError('UNIT_WINDOWS cannot erase precedence or worker-specific chains')
            for j in day['jobs']:
                _fields(j, 'release deadline count', 'unit window')
                r = _integer(j['release'], 'release', 0, hours - 1)
                _integer(j['deadline'], 'deadline', r, hours - 1)
                _integer(j['count'], 'count')
        elif day['mode'] == 'FIXED_CHAINS':
            if day['jobs'] or len(day['chains']) > day['hand_cap'] + 1:
                raise ValueError('fixed chains require separate explicit worker ownership')
            workers = set()
            for chain in day['chains']:
                _fields(chain, 'worker entry anchor jobs', 'chain')
                worker = _integer(chain['worker'], 'worker', 0, day['hand_cap'])
                if worker in workers:
                    raise ValueError('one continuous chain per fixed worker required')
                workers.add(worker)
                _integer(chain['entry'], 'entry', 0, hours)
                _name(chain['anchor'])
                if not isinstance(chain['jobs'], list) or not chain['jobs']:
                    raise ValueError('nonempty fixed-worker chain required; omit idle workers')
                job_count += len(chain['jobs'])
                if job_count > budget.max_job_groups:
                    raise ValueError('LABOR_BUDGET_EXHAUSTED')
                anchor = chain['anchor']
                for j in chain['jobs']:
                    _fields(j, 'duration release deadline before after', 'anchored job')
                    _integer(j['duration'], 'duration', 1, 24)
                    r = _integer(j['release'], 'release', 0, hours - 1)
                    _integer(j['deadline'], 'exclusive deadline', r + 1, hours)
                    _name(j['before']); _name(j['after'])
                    if j['before'] != anchor:
                        raise ValueError('ANCHOR_DISCONTINUITY: omitted route/custody transition')
                    anchor = j['after']
        else:
            raise ValueError('unsupported labor scheduling mode')
    return days


def _frontier(day, start):
    """Admitted earliest safe rows; wages never use a later cash receipt early."""
    events, protected = day['events'], day['protected_cents']
    hours = 23 if day['day'] == 29 else 24
    balances = [start]
    for event in events:
        balances.append(balances[-1] + event['delta_cents'])
    if min(balances) < protected:
        return [], dict(reason='FIXED_CASH_PREFIX', first_breach=next(
            i for i, cash in enumerate(balances) if cash < protected), minimum_cash_cents=min(balances))
    suffix = [0] * len(balances)
    suffix[-1] = balances[-1] - protected
    for i in range(len(balances) - 2, -1, -1):
        suffix[i] = min(balances[i] - protected, suffix[i + 1])
    costs = [100 * hire_bill(k) for k in range(day['hand_cap'] + 1)]
    chosen = []
    for i, event in enumerate(events):
        for _ in range(event['hire_slots']):
            if len(chosen) == day['hand_cap'] or costs[len(chosen) + 1] > suffix[i]:
                break
            chosen.append(i)
    rows = []
    for k in range(len(chosen) + 1):
        selected = chosen[:k]
        hire_hours = [events[i]['hour'] for i in selected]
        cash, minimum, hired = start, start, 0
        for i, event in enumerate(events):
            cash += event['delta_cents']; minimum = min(minimum, cash)
            n = selected.count(i)
            cash -= costs[hired + n] - costs[hired]; hired += n
            minimum = min(minimum, cash)
        rows.append(dict(hands=k, wage_cents=costs[k], hire_event_indices=selected,
                         hire_hours=hire_hours, capacity_by_hour=[1 + sum(h < t for h in hire_hours) for t in range(hours)],
                         minimum_cash_cents=minimum, terminal_cash_cents=cash))
    return rows, None


def _interval_tables(jobs, hours):
    histogram = [[0] * hours for _ in range(hours)]
    for j in jobs:
        histogram[j['release']][j['deadline']] += j['count']
    demand = [[0] * hours for _ in range(hours)]
    for a in range(hours - 1, -1, -1):
        for b in range(a, hours):
            demand[a][b] = histogram[a][b]
            if a + 1 < hours:
                demand[a][b] += demand[a + 1][b]
            if b:
                demand[a][b] += demand[a][b - 1]
                if a + 1 < hours:
                    demand[a][b] -= demand[a + 1][b - 1]
    return histogram, demand


def _interval_check(demand, capacity):
    prefix = [0]
    for c in capacity:
        prefix.append(prefix[-1] + c)
    hours = len(capacity)
    cut = None
    for length in range(1, hours + 1):
        for a in range(hours - length + 1):
            b = a + length - 1
            available = prefix[b + 1] - prefix[a]
            if cut is None and demand[a][b] > available:
                cut = dict(start=a, end=b, demand=demand[a][b], capacity=available,
                           shortfall=demand[a][b] - available)
    return dict(feasible=cut is None, cut=cut,
                prefix_only_passes=all(demand[0][t] <= prefix[t + 1] for t in range(hours)))


def _edf(histogram, capacity):
    heap, schedule = [], []
    for t, slots in enumerate(capacity):
        for d, n in enumerate(histogram[t]):
            if n:
                heapq.heappush(heap, (d, t, n))
        placed = []
        while slots and heap:
            deadline, release, count = heapq.heappop(heap)
            if deadline < t:
                raise AssertionError('interval certificate and EDF disagree')
            take = min(slots, count)
            placed.append(dict(release=release, deadline=deadline, count=take))
            slots -= take
            if count > take:
                heapq.heappush(heap, (deadline, release, count - take))
        schedule.append(placed)
    if heap:
        raise AssertionError('unplaced jobs after interval certificate')
    return schedule


def _chain_signature(jobs):
    # Accepted (P,R,U) composition, not duration-weighted pooled capacity.
    work, floor, latest = 0, None, None
    for j in jobs:
        p, r, d = j['duration'], j['release'], j['deadline']
        if r + p > d or (floor is not None and floor > d - p):
            return None
        work, floor, latest = (work + p, r + p if floor is None else max(floor + p, r + p),
                               d - p if latest is None else min(latest, d - p - work))
    return dict(work=work, floor=floor, latest_entry=latest)


def _chains_check(chains, row):
    receipts = []
    for chain in chains:
        worker = chain['worker']
        signature = _chain_signature(chain['jobs'])
        if worker > row['hands']:
            return dict(feasible=False, cut=dict(worker=worker, reason='WORKER_NOT_HIRED'), chains=receipts)
        entry = max(chain['entry'], 0 if worker == 0 else row['hire_hours'][worker - 1] + 1)
        if signature is None or entry > signature['latest_entry']:
            return dict(feasible=False, cut=dict(worker=worker, reason='CHAIN_WINDOW', entry=entry,
                                                 signature=signature), chains=receipts)
        finish = max(entry + signature['work'], signature['floor'])
        t, schedule = entry, []
        for j in chain['jobs']:
            start = max(t, j['release']); t = start + j['duration']
            schedule.append(dict(start=start, finish=t, before=j['before'], after=j['after']))
        receipts.append(dict(worker=worker, entry=entry, finish=finish, signature=signature, schedule=schedule))
    return dict(feasible=True, cut=None, chains=receipts)


def certify_labor(context, tape, problem, *, support=None, budget=None) -> LaborCertificate:
    """Compute a bound conditional wage, or fail closed with explicit diagnostics.

    Invalid scheduling/support domains return UNKNOWN; proved failure inside the
    supported fixed-law domain is INFEASIBLE_UNDER_CONDITIONS. Actor leakage or
    non-JSON callbacks raise ValueError before either result can be constructed.
    """
    actor = actor_key(context)
    clean = _wire(problem)  # reject hidden input/callbacks before UNKNOWN handling
    binding, tkey, pkey = cost_key(context, tape), tape_key(tape), problem_key(clean)
    custody, facts, receipts, checks = None, None, [], 0
    parents = PARENTS
    budget = budget or LaborBudget()
    if not isinstance(budget, LaborBudget):
        raise ValueError('LaborBudget required')

    def result(status, diagnostics=(), cents=None):
        return LaborCertificate(binding, actor, tkey, custody, pkey, status,
                                UNKNOWN if cents is None else Estimate.exact(cents / 100), parents,
                                tuple(diagnostics), _json(dict(problem=clean, support=facts, days=receipts,
                                                             interval_checks=checks, budget=vars(budget))))

    try:
        if tape.history_certified is not True or tape.stock is None:
            raise ValueError('OWN_PROJECTION_OR_CUSTODY_UNCERTIFIED')
        _public(context); _validate_tape(context, tape)
        farm = context.observation['farms'][context.player]
        if farm.get('hands') not in ([], ()) or type(farm.get('hires_today')) is not int or farm['hires_today'] != 0:
            raise ValueError('DEFAULT_DAWN_WORKER_RESET_REQUIRED')
        if len(context.observation['private']['inventories']) != 1:
            raise ValueError('DEFAULT_DAWN_WORKER_CUSTODY_REQUIRED')
        custody = custody_key(context)
        if not isinstance(support, LaborSupport):
            raise ValueError('PREREQUISITES_UNCERTIFIED')
        facts = support.to_dict()
        if support.scope != 'ACTOR_CONDITIONAL':
            raise ValueError('ACTOR_LEGAL_SUPPORT_REQUIRED')
        if (support.cost_key, support.problem_key, support.custody_key) != (binding, pkey, custody):
            raise ValueError('SUPPORT_BINDING_MISMATCH')
        missing = [name for name in REQUIRED_PREREQUISITES
                   if name not in support.prerequisites
                   or support.prerequisites[name].evidence != Estimate.exact(1)
                   or not support.prerequisites[name].parents]
        if missing:
            raise ValueError('PREREQUISITES_UNCERTIFIED: ' + ','.join(missing))
        parents = tuple(sorted(set(PARENTS).union(*(v.parents for v in support.prerequisites.values()))))
        days = _parse(clean, tape.from_day, budget)
        money = farm.get('money')
        if type(money) not in (int, float) or money < 0:
            raise ValueError('EXACT_OWN_CENT_CASH_REQUIRED')
        decimal = Decimal(str(money)) * 100
        if not decimal.is_finite() or decimal != decimal.to_integral_value() or decimal > 10**15:
            raise ValueError('EXACT_OWN_CENT_CASH_REQUIRED')
        cash, total_wage = int(decimal), 0
        for day in days:
            # Only a consistency check: this never certifies completeness or
            # converts an aggregate attempted-service count into job windows.
            declared = (sum(j['count'] for j in day['jobs']) if day['mode'] == 'UNIT_WINDOWS'
                        else sum(len(c['jobs']) for c in day['chains']))
            if declared < tape.actions[day['day']]:
                raise ValueError('WORKLOAD_UNDERCOVERS_TAPE_ACTIONS')
            rows, failure = _frontier(day, cash)
            day_receipt = dict(day=day['day'], mode=day['mode'], start_cash_cents=cash,
                               rows=[], selected=None, minimum_hands=None, minimum_wage_cents=None)
            receipts.append(day_receipt)
            if failure:
                day_receipt['failure'] = failure
                return result('INFEASIBLE_UNDER_CONDITIONS', ('FIXED_CASH_PREFIX',))
            hours = 23 if day['day'] == 29 else 24
            if day['mode'] == 'UNIT_WINDOWS':
                needed = len(rows) * hours * (hours + 1) // 2
                if checks + needed > budget.max_interval_checks:
                    raise ValueError('LABOR_BUDGET_EXHAUSTED')
                histogram, demand = _interval_tables(day['jobs'], hours)
            selected = None
            for row in rows:
                if day['mode'] == 'UNIT_WINDOWS':
                    check = _interval_check(demand, row['capacity_by_hour'])
                    checks += hours * (hours + 1) // 2
                else:
                    check = _chains_check(day['chains'], row)
                witness = dict(row, **check)
                day_receipt['rows'].append(witness)
                if selected is None and check['feasible']:
                    selected = dict(witness)
            if selected is None:
                return result('INFEASIBLE_UNDER_CONDITIONS', ('NO_FEASIBLE_FUNDED_SCHEDULE',))
            if day['mode'] == 'UNIT_WINDOWS':
                selected['schedule'] = _edf(histogram, selected['capacity_by_hour'])
            day_receipt.update(selected=selected, minimum_hands=selected['hands'],
                               minimum_wage_cents=selected['wage_cents'])
            total_wage += selected['wage_cents']
            cash = selected['terminal_cash_cents']
        return result('CERTIFIED', cents=total_wage)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        return result('UNKNOWN', (str(error),))
