"""Certified, bounded market proposals; no game stepping or executor mutation.

``MarketLoop.propose(StrategyContext)`` consumes only the actor observation, own
memory and two publications in ``derived``. This is a market-component comparison,
NOT a terminal value function. PASS means keep the coordinator's current queue.

Input adapters (local, pending integration with the parallel module producers):
* memory['market_loop']: phase='POST_WORKER', through (inclusive market step),
  resource_closed=True, orders={step: native queue}. Closure certifies no omitted
  cash/custody/worker/feed/deposit changes for either trader through that window. Future shop
  publication is never predicted. Maximum 24 market phases, ten slots, 100 units
  per order, two flow hypotheses. A missing certificate causes abstention.
  Per scan: <=4096 comparisons and <=2,000,000 unit quotations. Lower constructor
  limits are supported; exhaustion is explicit and never certifies a size from
  an unfinished family. Budget counters exclude no component quotations.
* derived['liquidity']: step, reserve (types.Reserve), capital_shadow/slot_shadow
  (types.Estimate), protected_inventory={product: Estimate}. Protected cash and
  stock remain untouched even by the baseline queue; the coordinator must remove
  already-authorized spending from this *external* protection to avoid counting
  the same commitment twice. Missing shadow prices are UNKNOWN, never zero.
* derived['opponent_state']: step, through, kind (EXACT/BOUND/HEURISTIC), basis
  (public provenance), scenarios=[{probability, orders, stock_lower, headroom_lower}].
  Stock/headroom are certified public LOWER bounds (Estimate), not private truth.
  Rival cash is read from public farms. EXACT requires one certified request word;
  BOUND means a certified exhaustive support, HEURISTIC a declared forecast. A tape
  match alone does not certify requests or resources in a counterfactual.

Orders are empty rows or SELL/BUY_PRODUCT/BUY_SEED/BUY_ANIMAL triples. Unsupported
HIRE/land/worker side effects abstain, not approximate. Full queues preserve ranks,
including empty rows. Floor-backed immediate sales/liquidations remain legal;
nonfloor cycle/offset/hull families abstain on any floor sale in either branch.
quote_window computes CONDITIONAL full-fill arithmetic;
compare_schedule additionally certifies both branches' resources and disclosure.
Resource certificates are prefix minima, including product custody, not aggregate
occupancy. Both seats quote before either commits, using the engine's price law.

Cash, rival cash, margin and shadow-adjusted score remain distinct. Requirements
refer to the FULL replacement window (not additive to the baseline): cash is net
spending; reserve is additional temporary capital; their sum is peak cash need.
Headroom and item stock requirements are also published in candidate metadata.
Same-product buy/sell queues are native-legal, but are NOT representable by current
V0 SALE/BUY rules. This module does not circumvent that executor limitation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from typing import Mapping, Sequence

from farm_neat.game.exact_engine import load_engine
from farm_neat.strategy.types import (
    Evidence, Estimate, LAST_STEP, Proposal, Requirements, Reserve, StrategyContext, UNKNOWN,
)

NAME = 'market_loop'
SCHEMA = 'farm-market-loop-v1'
MAX_PHASES = 24
MAX_UNITS = 100
MAX_SCENARIOS = 2
MAX_COMPARISONS = 4096
MAX_UNIT_QUOTES = 2_000_000
PARENTS = (
    '41f072a9688d54930f36d7c9a9c77531d6325aa67e76476ebe5cb06b357f2b05',
    '62711c42a6b930bcacfc43130078719039e62cf6955f9c71053df299a3d71f4e',
    'a6531beff5081e04ffd9b6edb56208910c99d1921a39a925f7048fb4c2e468c2',
    '0b82c8df5959b1ab65c3478bf79dcc181c400d8ffa7bda7a6f3d43c0a8f3402f',
    '3c0b08f2d534daf4f64486720f39f49264926929533754f1b9155ebeec323507',
    'e7e5ed627d1444c4df41cb5dc14603e67b876b762b86ac9594528b5abab8476c',
    '8edd0b76829c0a107abf5074cb8b8e085e5b8904670b6b4e151c0d07efa27044',
    'e36789f2efb18ce199f746439b489b197641b1b51e406c74c8dfa8c7e5442dbc',
)


@lru_cache(maxsize=65536)
def _price(product: str, inventory: int) -> int:
    return load_engine().market_price(product, inventory)


def _integer(value, label, low=None, high=None):
    if type(value) is not int or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f'{label}: integer outside supported range')
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label}: nonnegative finite number required')
    return value


def _estimate(value) -> Estimate:
    if isinstance(value, Estimate):
        return value
    return Estimate.from_dict(value) if isinstance(value, Mapping) else UNKNOWN


def _lower(value):
    e = _estimate(value)
    return e.value if e.kind is Evidence.EXACT else e.low if e.kind is Evidence.BOUND else None


def _upper(value):
    e = _estimate(value)
    return e.value if e.kind is Evidence.EXACT else e.high if e.kind is Evidence.BOUND else None


def _schedule(raw, start: int, end: int) -> dict[int, tuple[tuple, ...]]:
    if not isinstance(raw, Mapping):
        raise ValueError('orders: explicit step-to-queue mapping required')
    e = load_engine()
    domains = {'SELL': e.PRODUCTS, 'BUY_PRODUCT': ('WHEAT', 'FERTILIZER'),
               'BUY_SEED': e.CROPS, 'BUY_ANIMAL': e.ANIMALS}
    out = {}
    for key, queue in raw.items():
        t = int(key) if isinstance(key, str) and key.isdecimal() else key
        _integer(t, 'order step', start, end)
        if t in out or not isinstance(queue, (list, tuple)) or len(queue) > 10:
            raise ValueError('orders: duplicate step or more than ten slots')
        rows = []
        for row in queue:
            if not isinstance(row, (list, tuple)):
                raise ValueError('order must be a native list/tuple')
            if not row:
                rows.append(())
                continue
            if len(row) != 3 or row[0] not in domains or row[1] not in domains[row[0]]:
                raise ValueError('UNSUPPORTED order/side effects; no atomic HIRE or land approximation')
            _integer(row[2], 'order quantity', 1, MAX_UNITS)
            rows.append(tuple(row))
        out[t] = tuple(rows)
    return out


def _wire(schedule):
    return {str(t): [list(row) for row in rows] for t, rows in sorted(schedule.items())}


def known_demand(shops: Sequence[str], step: int) -> dict[str, int]:
    """Published instances only; duplicates consume independently, as in the engine."""
    e = load_engine()
    result = Counter()
    if any(shop not in e.SHOPS for shop in shops):
        raise ValueError('unknown published shop')
    if step % 4 == 0:
        for shop in shops:
            products = e.SHOPS[shop]
            for product in products:
                result[product] += 2 if len(products) == 1 else 1
    if step % 24 == 0:
        result.update(e.TOWN_CENTER_PRODUCTS)
    return dict(result)


@dataclass
class ResourcePath:
    """Conditional full-fill path; required opening resources are nonnegative."""
    cash: int = 0
    peak_cash: int = 0
    stock_delta: Counter = field(default_factory=Counter)
    stock_required: Counter = field(default_factory=Counter)
    headroom_required: int = 0
    slots: int = 0
    floor_sales: int = 0

    def commit(self, op, item, price):
        change = price if op == 'SELL' else -price
        self.cash += change
        self.peak_cash = max(self.peak_cash, -self.cash)
        if op != 'BUY_SEED':
            self.stock_delta[item] += -1 if op == 'SELL' else 1
            self.stock_required[item] = max(self.stock_required[item], -self.stock_delta[item])
            self.headroom_required = max(self.headroom_required, sum(self.stock_delta.values()))
        if op == 'SELL' and price == 1:
            self.floor_sales += 1


@dataclass
class WindowQuote:
    own: ResourcePath
    rival: ResourcePath
    inventory: dict[str, int]
    unit_quotes: int


def _view(context: StrategyContext):
    e = load_engine()
    memory = context.memory.get(NAME, {})
    if not isinstance(memory, Mapping) or memory.get('phase') != 'POST_WORKER':
        raise ValueError('UNKNOWN post-worker custody; phase certificate missing')
    end = _integer(memory.get('through'), 'through', context.step, min(LAST_STEP, context.step + MAX_PHASES - 1))
    if memory.get('resource_closed') is not True:
        raise ValueError('UNKNOWN omitted resource changes inside window')
    if set(memory) - {'phase', 'through', 'resource_closed', 'orders'}:
        raise ValueError('unsupported market-window memory fields')
    obs = context.observation
    if 'step' in obs and (type(obs['step']) is not int or obs['step'] != context.step):
        raise ValueError('UNKNOWN stale observation/context step join')
    market = obs.get('market', {})
    town = obs.get('town', {})
    private = obs.get('private', {})
    if not all(isinstance(part, Mapping) for part in (market,town,private)):
        raise ValueError('UNKNOWN malformed market/town/own-private publication')
    if market.get('params') not in (None, e.MARKET_PARAMS):
        raise ValueError('UNSUPPORTED custom market parameters')
    inv = market.get('inventory', {})
    if not isinstance(inv, Mapping) or set(inv) != set(e.PRODUCTS):
        raise ValueError('complete public product inventories required')
    inv = {p: _integer(i, 'market inventory') for p,i in inv.items()}
    shops = town.get('unlocked_shops', ())
    if not isinstance(shops,(tuple,list)):
        raise ValueError('UNKNOWN malformed published-shop prefix')
    shops = tuple(shops)
    if shops != context.shops or any(s not in e.SHOPS for s in shops):
        raise ValueError('UNKNOWN ordered published-shop prefix mismatch')
    # Refuse a window crossing even one unknown shop; never sample a future one.
    if len(shops) < 8 and end >= (context.step // 72 + 1) * 72:
        raise ValueError('UNKNOWN future shop publication inside window')
    stock = private.get('shed')
    if not isinstance(stock, Mapping) or set(stock) - set((*e.PRODUCTS, *e.ANIMALS)):
        raise ValueError('own per-product shed custody required')
    stock = {p: _integer(n, 'shed stock', 0, 100) for p,n in stock.items()}
    if sum(stock.values()) > 100:
        raise ValueError('own shed capacity exceeded')
    farms = obs.get('farms')
    if (not isinstance(farms, (list, tuple)) or len(farms) != 2
            or not all(isinstance(farm,Mapping) for farm in farms)):
        raise ValueError('two public farms required')
    cash = [_number(farms[s].get('money'), 'public money') for s in (context.player,1-context.player)]
    own = _schedule(memory.get('orders', {}), context.step, end)
    return end, inv, shops, stock, cash, own


class _BudgetLimit(ValueError):
    """Internal control flow; never a claim that an unsearched option loses."""


def _charge(budget, key: str, amount: int) -> None:
    if budget is not None:
        if budget[key] + amount > budget['max_' + key]:
            raise _BudgetLimit('scan budget exhausted: ' + key)
        budget[key] += amount


def _quote(start, end, inv, shops, own, rival, budget=None):
    """Full-fill induction, using native two-seat precommit quotation timing.

    Computing this alone NEVER certifies that anyone can actually fill the word.
    """
    _charge(budget,'unit_quotes',sum(row[2] for schedule in (own,rival)
                                      for queue in schedule.values() for row in queue if row))
    e = load_engine()
    market = dict(inv)
    paths = [ResourcePath(), ResourcePath()]
    count = 0
    for t in range(start, end + 1):
        queues = [own.get(t, ()), rival.get(t, ())]
        for slot in range(max(map(len, queues), default=0)):
            rows = [q[slot] if slot < len(q) else () for q in queues]
            for path,row in zip(paths,rows):
                path.slots += bool(row)
            for j in range(max((row[2] if row else 0 for row in rows), default=0)):
                quotes = []
                for row in rows:
                    if not row or j >= row[2]:
                        quotes.append(None)
                        continue
                    op,p,_ = row
                    price = (_price(p,market[p] - (op == 'BUY_PRODUCT'))
                             if op in ('SELL','BUY_PRODUCT') else
                             e.CROPS[p]['seed'] if op == 'BUY_SEED' else e.ANIMALS[p]['cost'])
                    quotes.append((op,p,price))
                    count += 1
                # Both quotes were made before these commits; seat permutation
                # cannot affect successful full-fill deltas under this contract.
                for path,quote in zip(paths, quotes):
                    if quote is None:
                        continue
                    op,p,price = quote
                    path.commit(op,p,price)
                    if op == 'BUY_PRODUCT':
                        market[p] -= 1
                    elif op == 'SELL' and price > 1:
                        market[p] += 1
        if t < end:  # Window ends AFTER its final market, BEFORE town consumption.
            for p,n in known_demand(shops,t).items():
                market[p] -= n
    return WindowQuote(*paths, market, count)


def quote_window(context: StrategyContext, own_orders: Mapping, rival_orders: Mapping) -> WindowQuote:
    """Conditional arithmetic for a declared word, not a rival-state inference."""
    end, inv, shops, _, _, _ = _view(context)
    own = _schedule(own_orders, context.step, end)
    rival = _schedule(rival_orders, context.step, end)
    return _quote(context.step,end,inv,shops,own,rival)


def _input(context):
    view = _view(context)
    end, _, _, _, cash, _ = view
    liq = context.derived.get('liquidity')
    flow = context.derived.get('opponent_state')
    if not isinstance(liq, Mapping) or liq.get('step') != context.step:
        raise ValueError('UNKNOWN or stale LIQUIDITY')
    inventory_protection = liq.get('protected_inventory', {})
    if (not isinstance(inventory_protection,Mapping) or
            set(inventory_protection)-set((*load_engine().PRODUCTS,*load_engine().ANIMALS))):
        raise ValueError('UNKNOWN malformed inventory protection publication')
    for value in inventory_protection.values():
        bound = _upper(value)
        if bound is not None and not 0 <= bound <= 100:
            raise ValueError('invalid inventory protection bound; cannot create product custody')
    for key in ('capital_shadow','slot_shadow'):
        bound = _upper(liq.get(key))
        if bound is not None and bound < 0:
            raise ValueError('nonnegative shadow costs required')
    reserve = liq.get('reserve')
    if not isinstance(reserve, Reserve) or reserve.step != context.step or reserve.cash != cash[0]:
        raise ValueError('UNKNOWN or stale LIQUIDITY reserve/public-cash join')
    protected, bank = _upper(reserve.protected), _lower(reserve.discretionary)
    if protected is None or bank is None:
        raise ValueError('UNKNOWN protected/discretionary cash bound')
    if min(protected,bank) < 0 or protected + bank > cash[0] + 1e-9:
        raise ValueError('inconsistent liquidity split')
    bank = min(bank, cash[0]-protected)
    if not isinstance(flow, Mapping) or flow.get('step') != context.step or flow.get('through') != end:
        raise ValueError('UNKNOWN or stale OPPONENT-STATE window')
    if not isinstance(flow.get('basis'), str) or not flow['basis'].strip():
        raise ValueError('UNKNOWN public flow provenance')
    try:
        kind = Evidence(flow.get('kind'))
    except (ValueError,TypeError) as exc:
        raise ValueError('UNKNOWN flow evidence') from exc
    scenarios = flow.get('scenarios')
    if kind not in (Evidence.EXACT,Evidence.BOUND,Evidence.HEURISTIC) or not isinstance(scenarios,(tuple,list)) or not 1 <= len(scenarios) <= MAX_SCENARIOS:
        raise ValueError('UNKNOWN flow support (one or two hypotheses required)')
    if kind is Evidence.EXACT and len(scenarios) != 1:
        raise ValueError('multiple flow hypotheses are BOUND or HEURISTIC, not EXACT')
    parsed = []
    for sc in scenarios:
        if not isinstance(sc,Mapping) or set(sc) != {'probability','orders','stock_lower','headroom_lower'}:
            raise ValueError('closed public flow scenario fields required; no private state')
        probability = _number(sc['probability'], 'probability')
        if not isinstance(sc['stock_lower'], Mapping) or set(sc['stock_lower']) - set((*load_engine().PRODUCTS,*load_engine().ANIMALS)):
            raise ValueError('public custody lower-bound product keys required')
        stock = {p:_lower(v) for p,v in sc['stock_lower'].items()}
        headroom = _lower(sc['headroom_lower'])
        values = [x for x in (*stock.values(),headroom) if x is not None]
        if any(x < 0 or x > 100 for x in values) or sum(x or 0 for x in stock.values()) + (headroom or 0) > 100:
            raise ValueError('inconsistent public resource bounds')
        parsed.append((probability, _schedule(sc['orders'],context.step,end),stock,headroom))
    if not math.isclose(sum(row[0] for row in parsed),1,rel_tol=0,abs_tol=1e-12):
        raise ValueError('flow probabilities must sum to one')
    return view,liq,kind,parsed,bank


def _aggregate(values, kind, weights):
    if kind is Evidence.EXACT:
        return Estimate.exact(values[0])
    if kind is Evidence.BOUND:
        return Estimate.bound(min(values),max(values))
    return Estimate.heuristic(sum(w*v for w,v in zip(weights,values)))


def _point(estimate):
    return estimate.low if estimate.kind is Evidence.BOUND else estimate.value


def _base_proposal(context, product, family, schedule, quantity=0):
    orders = _wire(schedule)
    identity = json.dumps({'family':family,'product':product,'orders':orders},sort_keys=True,separators=(',',':'))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    candidate = {'schema':SCHEMA, 'family':family, 'product':product,'quantity':quantity,
                 'orders':orders,'eligible':False,'refusals':[], 'score':UNKNOWN.to_dict(),
                 'scope':'CONDITIONAL_MARKET_WINDOW_NOT_TERMINAL_VALUE',
                 'execution':'NATIVE_MARKET_QUEUE_NOT_V0_RULES'}
    return digest,candidate


def _unknown(context,product,family,reason,schedule=None,quantity=0):
    digest,c = _base_proposal(context,product,family,schedule or {},quantity)
    # A refused plan is diagnostic, never an execution payload.
    c.update(status='UNKNOWN', refusals=[reason], orders={})
    return Proposal(NAME,f'{product}:{family}:{context.step}:{digest}',c,
                    ('cash','shed','market_slots','market:'+product),
                    guards=('all resource and flow certificates required before execution',),
                    invalidated_by=('new market event or refreshed certificate',),parents=PARENTS)


def _pass(context,product):
    digest,c = _base_proposal(context,product,'PASS',{})
    c.update(status='KEEP_BASELINE',score=Estimate.exact(0).to_dict(), peak_cash=0,
             headroom=Estimate.exact(0).to_dict())
    zero = Estimate.exact(0)
    return Proposal(NAME,f'{product}:PASS:{context.step}:{digest}',c,('market:'+product,),
                    Requirements(cash=zero,reserve=zero,market_slots=zero,worker_hours=zero,land_quadrants=zero,
                                 window=(context.step,context.step)),zero,zero,zero,
                    invalidated_by=('baseline queue or market event changed',),parents=PARENTS)


def _compare(context, parsed, schedule, product, family, quantity=0, budget=None):
    _charge(budget,'comparisons',1)
    view,liq,kind,scenarios,bank = parsed
    end,inv,shops,stock,cash,control = view
    quotes = []
    refusals = []
    protected_stock = liq.get('protected_inventory',{})
    for index,(probability,rival,rival_stock,headroom) in enumerate(scenarios):
        pair = []
        for label,own in (('control',control),('candidate',schedule)):
            result = _quote(context.step,end,inv,shops,own,rival,budget)
            if (family in ('aligned_cycle','sandwich','sale_shift','hold','carry','bridge')
                    and (result.own.floor_sales or result.rival.floor_sales)):
                refusals.append(f'{label}/{index}:floor sales invalidate nonfloor family certificate')
            for role,path in (('own',result.own),('rival',result.rival)):
                available_cash = bank if role == 'own' else cash[1]
                if path.peak_cash > available_cash:
                    refusals.append(f'{label}/{index}/{role}:cash needs {path.peak_cash}, certified {available_cash}')
                for p,n in path.stock_required.items():
                    if n <= 0:
                        continue
                    if role == 'own':
                        keep = _upper(protected_stock.get(p))
                        available = None if keep is None else stock.get(p,0)-keep
                    else:
                        available = rival_stock.get(p)
                    if available is None or available < n:
                        refusals.append(f'{label}/{index}/{role}:stock:{p} needs {n}, certified {available}')
                available = 100-sum(stock.values()) if role=='own' else headroom
                if path.headroom_required and (available is None or available < path.headroom_required):
                    refusals.append(f'{label}/{index}/{role}:headroom needs {path.headroom_required}, certified {available}')
            pair.append(result)
        quotes.append(pair)
    if refusals:
        return _unknown(context,product,family,'; '.join(refusals),schedule,quantity)
    weights = [row[0] for row in scenarios]
    own_d = [b.own.cash-a.own.cash for a,b in quotes]
    rival_d = [b.rival.cash-a.rival.cash for a,b in quotes]
    margin_d = [a-b for a,b in zip(own_d,rival_d)]
    own_est = _aggregate(own_d,kind,weights)
    rival_est = _aggregate(rival_d,kind,weights)
    margin_est = _aggregate(margin_d,kind,weights)
    peak = max(q[1].own.peak_cash for q in quotes)
    spent = max(0,max(-q[1].own.cash for q in quotes))
    slots = max(q[1].own.slots for q in quotes)
    required_stock = {}
    for _,q in quotes:
        for p,n in q.own.stock_required.items():
            if n:
                required_stock[p] = Estimate.exact(max(n,required_stock.get(p,Estimate.exact(0)).value))
    capital,slot_cost = _upper(liq.get('capital_shadow')),_upper(liq.get('slot_shadow'))
    if capital is not None and capital < 0 or slot_cost is not None and slot_cost < 0:
        raise ValueError('nonnegative shadow costs required')
    score = UNKNOWN if capital is None or slot_cost is None else _aggregate(
        [v-capital*peak-slot_cost*slots for v in margin_d],kind,weights)
    digest,c = _base_proposal(context,product,family,schedule,quantity)
    c.update(status='CERTIFIED' if kind is not Evidence.HEURISTIC else 'FORECAST_ONLY',
             control_orders=_wire(control), peak_cash=peak,
             headroom=Estimate.exact(max(q[1].own.headroom_required for q in quotes)).to_dict(),
             score=score.to_dict(), capital_shadow=capital, slot_shadow=slot_cost,
             scenario_deltas=[{'probability':w,'own_cash':a,'rival_cash':b,'margin':a-b}
                              for w,a,b in zip(weights,own_d,rival_d)],
             endpoint_market_equal=all(a.inventory==b.inventory for a,b in quotes),
             terminal_value=UNKNOWN.to_dict(),
             public_flow_basis=context.derived['opponent_state']['basis'],
             eligible=kind in (Evidence.EXACT,Evidence.BOUND) and _point(score) is not None and _point(score)>0)
    if score.kind is Evidence.UNKNOWN:
        c['status'] = 'UNKNOWN_SHADOW_COST'
        c['refusals'] = ['UNKNOWN liquidity capital or slot shadow cost']
    return Proposal(NAME,f'{product}:{family}:{context.step}:{digest}',c,
                    ('cash','shed','market_slots','market:'+product),
                    Requirements(cash=Estimate.exact(spent),reserve=Estimate.exact(max(0,peak-spent)),
                                 inventory=required_stock,market_slots=Estimate.exact(slots),
                                 worker_hours=Estimate.exact(0),land_quadrants=Estimate.exact(0),window=(context.step,end)),
                    own_est,rival_est,margin_est,
                    guards=('both control and candidate fully fill under certified cash, product stock and headroom',
                            'all native queue ranks preserved; no omitted resource changes',
                            'protected cash and protected product stocks untouched',
                            'published-shop prefix and public flow support unchanged',
                            'compare only this market window; continuation value remains UNKNOWN'),
                    invalidated_by=('step or ordered published-shop prefix changed','market inventory changed',
                                    'either public cash account changed','own custody, feed or worker deposit changed',
                                    'baseline queue changed','liquidity or opponent-state certificate changed'),parents=PARENTS)


def compare_schedule(context: StrategyContext, replacement: Mapping, *, product: str,
                     family: str, quantity: int=0) -> Proposal:
    """Compare a full native queue replacement with the context's own baseline.

    UNKNOWN is an abstention, never a claim of infeasibility or terminal inferiority.
    This is not the separate counterfactual module: no farm/world restoration occurs.
    """
    if product not in load_engine().PRODUCTS:
        raise ValueError('unknown product')
    try:
        parsed = _input(context)
        schedule = _schedule(replacement,context.step,parsed[0][0])
        return _compare(context,parsed,schedule,product,family,quantity)
    except ValueError as exc:
        return _unknown(context,product,family,str(exc),quantity=quantity)


def capital_hull(rows: Sequence[Proposal]) -> tuple[Proposal,...]:
    """Hard-feasible rows FIRST, then the upper concave reserve/margin hull.

    Constant slot charge is required within a call. No belief or budget grid is
    used: exact conditional rows are constructed once and reused for shadow prices.
    """
    known = [p for p in rows if _point(p.margin_delta) is not None and 'peak_cash' in p.candidate]
    if not known:
        return ()
    charges = {p.requirements.market_slots.value for p in known if p.candidate['quantity']>0}
    if len(charges)>1:
        raise ValueError('capital hull requires constant slot charge for non-PASS rows')
    points = {}
    for p in known:
        r = p.candidate['peak_cash']
        m = _point(p.margin_delta)
        old = points.get(r)
        if old is None or (m,-p.candidate['quantity']) > (_point(old.margin_delta),-old.candidate['quantity']):
            points[r] = p
    hull = []
    best = -math.inf
    for r,p in sorted(points.items()):
        y = _point(p.margin_delta)
        if y<=best:
            continue
        best = y
        while len(hull)>=2:
            a,b = hull[-2:]
            x0,x1 = a.candidate['peak_cash'],b.candidate['peak_cash']
            y0,y1 = _point(a.margin_delta),_point(b.margin_delta)
            if (y1-y0)*(r-x1) > (y-y1)*(x1-x0):
                break
            hull.pop()
        hull.append(p)
    return tuple(hull)


def belief_envelope(rows: Sequence[Proposal]) -> dict:
    """Exact two-hypothesis upper envelope, including lower-reserve ties.

    Input rows must already be feasible under BOTH hypotheses and the hard cash
    cap. Probability is not gridded. Slopes/intersections use rational arithmetic;
    isolated breakpoint winners are retained even if they own no open interval.
    """
    lines = []
    for row in rows:
        c = row.candidate
        if c['family']=='PASS':
            lines.append((Fraction(0),Fraction(0),0,0))
            continue
        if row.margin_delta.kind is Evidence.UNKNOWN:
            continue
        ds = c.get('scenario_deltas', [])
        if len(ds)!=2 or c.get('capital_shadow') is None or c.get('slot_shadow') is None:
            raise ValueError('common-valid two-hypothesis rows and exact shadow upper bounds required')
        cost = (Fraction(str(c['capital_shadow']))*c['peak_cash']
                + Fraction(str(c['slot_shadow']))*int(row.requirements.market_slots.value))
        a,b = (Fraction(d['margin'])-cost for d in ds)
        lines.append((a,b-a,c['peak_cash'],c['quantity']))
    if not lines:
        return {'intervals':[], 'points':[]}
    # Same slope: keep largest intercept, then least reserve/quantity.
    by_slope = {}
    for line in lines:
        a,b,r,q = line
        old = by_slope.get(b)
        if old is None or (a,-r,-q)>(old[0],-old[2],-old[3]):
            by_slope[b] = line
    hull, starts = [], []
    for line in (by_slope[b] for b in sorted(by_slope)):
        cut = None
        while hull:
            prev = hull[-1]
            cut = (prev[0]-line[0])/(line[1]-prev[1])
            if len(hull)==1 or cut>starts[-1]:
                break
            hull.pop(); starts.pop()
        hull.append(line)
        starts.append(cut if len(hull)>1 else None)
    cuts = sorted({Fraction(0),Fraction(1)} | {x for x in starts if x is not None and 0<x<1})
    def winner(at):
        return min(lines,key=lambda v:(-(v[0]+at*v[1]),v[2],v[3]))[3]
    def fraction(at):
        return [at.numerator,at.denominator]
    return {'intervals':[{'low':fraction(a),'high':fraction(b),'quantity':winner((a+b)/2)}
                         for a,b in zip(cuts,cuts[1:])],
            'points':[{'at':fraction(p),'quantity':winner(p)} for p in cuts]}


def _best(rows):
    # Mix scenario CASH first. Projecting certainty hulls before mixing loses q5
    # in the admitted WHEAT {13,1} witness.
    two_point = any(len(p.candidate.get('scenario_deltas',()))==2 for p in rows)
    envelope = None
    if two_point and all(p.candidate.get('capital_shadow') is not None and
                         p.candidate.get('slot_shadow') is not None
                         for p in rows if p.candidate['family']!='PASS'):
        # Robust BOUND maximizes the worst support point, not an expectation.
        if all(p.margin_delta.kind is not Evidence.BOUND for p in rows):
            envelope = belief_envelope(rows)
            retained = {r['quantity'] for r in envelope['intervals']+envelope['points']}
            front = [p for p in rows if p.candidate['quantity'] in retained]
        else:
            front = capital_hull(rows)
    else:
        front = capital_hull(rows)
    choices = [p for p in front if _point(_estimate(p.candidate['score'])) is not None
               and _point(_estimate(p.candidate['score']))>0]
    best = min(choices,key=lambda p:(-_point(_estimate(p.candidate['score'])),
                                    p.candidate['peak_cash'],p.candidate['quantity'],p.decision_key),default=None)
    if best is not None:
        # Payload is JSON-owned by Proposal, not the input context.
        best.candidate['sizing'] = {'method':'common_valid_belief_envelope' if envelope else 'hard_budget_then_capital_hull',
                                    'feasible_rows':len(rows), 'retained_rows':len(front)}
        if envelope is not None:
            best.candidate['belief_envelope'] = envelope
    return best


def _put(schedule,t,slot,row):
    result = dict(schedule)
    queue = list(result.get(t,()))
    while len(queue)<=slot:
        queue.append(())
    queue[slot] = tuple(row)
    result[t] = tuple(queue)
    return result


def _empty(schedule,t,slot):
    return slot<10 and (slot>=len(schedule.get(t,())) or not schedule[t][slot])


def _positions(schedule, t, rival_schedules):
    """Only empty quote-boundary ranks; never compact a queue or move a barrier."""
    relevant = {0}
    for q in (schedule, *rival_schedules):
        for i,row in enumerate(q.get(t,())):
            if row:
                relevant.update((i,i+1))
    return sorted(i for i in relevant if _empty(schedule,t,i))


def event_windows(context: StrategyContext) -> tuple[int,...]:
    """Event/cut boundaries, not a dense time grid; no new shop speculation."""
    view,_,_,scenarios,_ = _input(context)
    end,_,shops,_,_,own = view
    points = {context.step,end}
    events = {t for schedule in (own,*(s[1] for s in scenarios)) for t,q in schedule.items() if any(q)}
    events.update(t for t in range(context.step,end) if known_demand(shops,t))
    for t in events:
        points.update((t,t+1))
    return tuple(sorted(t for t in points if context.step<=t<=end))


def _sale_variants(context, parsed, budget=None):
    view,liq,_,scenarios,_ = parsed
    end,inv,shops,stock,_,baseline = view
    rivals = [s[1] for s in scenarios]
    guards = [_quote(context.step,end,inv,shops,baseline,r,budget) for r in rivals]
    for p in load_engine().PRODUCTS:
        keep = _upper(liq.get('protected_inventory',{}).get(p))
        if keep is None or keep<0:
            continue
        occupied = max(g.own.stock_required[p] for g in guards)
        available = max(0,math.floor(stock.get(p,0)-keep-occupied))
        if not available:
            continue
        family = 'liquidation' if context.step==LAST_STEP else 'immediate_sale'
        variants = []
        for slot in _positions(baseline,context.step,rivals):
            for q in range(1,available+1):
                variants.append((_put(baseline,context.step,slot,('SELL',p,q)),q))
        yield family,p,variants


def _timing_variants(context, parsed, budget=None):
    baseline = parsed[0][-1]
    rivals = [s[1] for s in parsed[3]]
    windows = event_windows(context)
    groups = {}
    for original_t,queue in sorted(baseline.items()):
        for original_slot,row in enumerate(queue):
            if not row or row[0]!='SELL':
                continue
            _,p,q = row
            removed = _put(baseline,original_t,original_slot,())
            for t in windows:
                for slot in _positions(removed,t,rivals):
                    if (t,slot)==(original_t,original_slot):
                        continue
                    candidate = _put(removed,t,slot,row)
                    family = 'hold' if t>original_t else 'sale_shift'
                    groups.setdefault((family,p),[]).append((candidate,q))
    for (family,p),variants in groups.items():
        yield family,p,variants


def _funded_variants(context, parsed, budget=None):
    end,_,_,_,_,baseline = parsed[0]
    rivals = [s[1] for s in parsed[3]]
    t = context.step
    for p in ('WHEAT','FERTILIZER'):
        # This tranche is isolated from other own same-product flow. The exact
        # comparison still includes other-product cash/custody commitments.
        if any(row and row[1]==p for queue in baseline.values() for row in queue):
            continue
        # Strict same-phase sandwiches: preserve an empty rank before and after
        # a rival BUY rank common to every hypothesis.
        for slot in range(1,9):
            if any(len(r.get(t,()))<=slot or not r[t][slot] or r[t][slot][:2]!=('BUY_PRODUCT',p)
                   for r in rivals):
                continue
            before = [s for s in _positions(baseline,t,rivals) if s<slot]
            after = [s for s in _positions(baseline,t,rivals) if s>slot]
            if not before or not after:
                continue
            # Only an isolated rival block uses the strict capital-hull theorem.
            if any(any(row and row[1]==p and i!=slot for i,row in enumerate(r.get(t,()))) for r in rivals):
                continue
            variants = []
            for q in range(1,101):
                candidate = _put(_put(baseline,t,before[-1],('BUY_PRODUCT',p,q)),t,after[0],('SELL',p,q))
                variants.append((candidate,q))
            yield 'sandwich',p,variants
        if end==t:
            continue
        active = [i for schedule in (baseline,*rivals) for i,row in enumerate(schedule.get(t,())) if row]
        buy_slot = max(active,default=-1)+1
        if not _empty(baseline,t,buy_slot):
            continue
        # Strict entry after current market activity, exits at known event/cut
        # boundaries. At each exit, retain before/after rival rank possibilities.
        groups = {}
        for exit_t in event_windows(context):
            if exit_t<=t:
                continue
            mixed = any(row and row[1]==p for r in rivals for when,queue in r.items()
                        if t<when<=exit_t for row in queue)
            family = 'bridge' if mixed else 'carry'
            for exit_slot in _positions(baseline,exit_t,rivals):
                for q in range(1,101):
                    candidate = _put(_put(baseline,t,buy_slot,('BUY_PRODUCT',p,q)),
                                     exit_t,exit_slot,('SELL',p,q))
                    groups.setdefault(family,[]).append((candidate,q))
        for family,variants in groups.items():
            yield family,p,variants


class MarketLoop:
    """Pure proposal scan; invoke after each relevant event, never just an opening."""
    name = NAME

    def __init__(self, *, max_comparisons: int=MAX_COMPARISONS,
                 max_unit_quotes: int=MAX_UNIT_QUOTES):
        self.max_comparisons = _integer(max_comparisons,'max comparisons',1,MAX_COMPARISONS)
        self.max_unit_quotes = _integer(max_unit_quotes,'max unit quotes',1,MAX_UNIT_QUOTES)

    def propose(self, context: StrategyContext) -> tuple[Proposal,...]:
        products = load_engine().PRODUCTS
        proposals = [_pass(context,p) for p in products]
        budget = {'comparisons':0,'unit_quotes':0,
                  'max_comparisons':self.max_comparisons,'max_unit_quotes':self.max_unit_quotes}
        try:
            proposals = list(self._scan(context,proposals,budget))
        except _BudgetLimit as exc:
            # Only completed groups have been emitted. An interrupted group has
            # no chosen size; completed positive rows remain conditional options,
            # never a claim of optimality over the unsearched remainder.
            proposals.extend(_unknown(context,p,'scan_limit',str(exc)) for p in products)
        complete = not any(p.candidate['family'] in ('scan','scan_limit') for p in proposals)
        for proposal in proposals:
            proposal.candidate['scan_complete'] = complete
            proposal.candidate['scan_budget'] = dict(budget)
        return tuple(proposals)

    def _scan(self, context, proposals, budget):
        products = load_engine().PRODUCTS
        try:
            parsed = _input(context)
        except ValueError as exc:
            return tuple(proposals+[_unknown(context,p,'scan',str(exc)) for p in products])
        view,_,_,scenarios,_ = parsed
        end,_,_,_,_,baseline = view
        t = context.step
        for p in ('WHEAT','FERTILIZER'):
            for slot in range(9):
                if not _empty(baseline,t,slot) or not _empty(baseline,t,slot+1):
                    continue
                sizes = []
                for _,schedule,_,_ in scenarios:
                    q = schedule.get(t,())
                    if len(q)<=slot+1 or not q[slot] or not q[slot+1]:
                        break
                    a,b = q[slot:slot+2]
                    if a[:2]!=('BUY_PRODUCT',p) or b[:2]!=('SELL',p) or a[2]!=b[2]:
                        break
                    sizes.append(a[2])
                if len(sizes)!=len(scenarios):
                    continue
                # Admitted aligned-cycle dominance: q>=max(r) cannot beat PASS.
                # Funds are checked BEFORE projection, never after an unbudgeted hull.
                rows = [_pass(context,p)]
                for q in range(1,max(sizes)):
                    candidate = _put(_put(baseline,t,slot,('BUY_PRODUCT',p,q)),t,slot+1,('SELL',p,q))
                    values = [_quote(t,end,view[1],view[2],candidate,s[1],budget) for s in scenarios]
                    if any(v.own.floor_sales or v.rival.floor_sales for v in values):
                        continue  # The aligned nonfloor dominance theorem no longer applies.
                    row = _compare(context,parsed,candidate,p,'aligned_cycle',q,budget)
                    if row.margin_delta.kind is not Evidence.UNKNOWN:
                        rows.append(row)
                best = _best(rows)
                if best is not None:
                    proposals.append(best)
        for producer in (_sale_variants,_timing_variants,_funded_variants):
            for family,p,variants in producer(context,parsed,budget):
                rows = [_pass(context,p)]
                seen = set()
                for schedule,q in variants:
                    key = json.dumps(_wire(schedule),sort_keys=True)
                    if key in seen:
                        continue
                    seen.add(key)
                    row = _compare(context,parsed,schedule,p,family,q,budget)
                    if row.margin_delta.kind is not Evidence.UNKNOWN:
                        rows.append(row)
                best = _best(rows)
                if best is not None:
                    proposals.append(best)
        # Independent generators can find the same full replacement; one key only.
        return tuple({p.decision_key:p for p in proposals}.values())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',action='store_true',help='Print local adapter and resource limits')
    args = parser.parse_args(argv)
    if args.contract:
        print(json.dumps({'schema':SCHEMA,'module':NAME,'derived':['liquidity','opponent_state'],
                          'max_phases':MAX_PHASES,'max_order_units':MAX_UNITS,'max_scenarios':MAX_SCENARIOS,
                          'max_comparisons':MAX_COMPARISONS,'max_unit_quotes':MAX_UNIT_QUOTES,
                          'execution':'native queue proposals; V0 conversion unsupported'},sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
