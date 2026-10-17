"""Exact finite-state service DP for one animal.

Replaces the closed-form `econ.animal_npv`, which review returns R1-R5 independently found
to use the wrong event calendar: it started costs *and* fertilizer at the first product yield,
so pre-yield feed, care, labour and fertilizer were all omitted; it missed the first accumulated
care burst; it ignored `max_held`; and it applied one service-action rate to every species.

This module owns only the animal's biology and service timing, which the engine defines exactly:

* escape after two *consecutive* unfed end-of-days (`agent/constants.py`, gotcha 22);
* base production is NOT gated on feeding -- only the CARE bonus is (gotcha 6);
* CARE banks +1, paid out on the next *fed* production day, then reset; an unfed production
  day zeroes the bank (gotcha 7);
* `fertilizer_available` refreshes for every surviving animal every end-of-day (gotcha 8);
* held product is capped at `max_held` and further production is lost;
* the final day has no end-of-day at all, so feeding and caring on it are dead actions
  (gotcha 20).

Everything else -- walking, depot routing, shed capacity, market impact, the opponent -- belongs
to the route and market layers and is deliberately excluded. The caller supplies prices; the DP
supplies the exact service schedule and the resulting dated event stream.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from farm_neat.game.constants import ANIMALS

__all__ = ["Prices", "State", "Plan", "solve", "value_of", "events_of"]


@dataclass(frozen=True)
class Prices:
    """Flat per-unit quotes used inside the DP horizon.

    `product` and `fertilizer` are what one unit fetches, `wheat` what one feed unit costs,
    and `action` the shadow price of one unit-action (see `econ.ACTION_COST`; R3/R5 both argue
    this should become a state-dependent dual, so it is an explicit input, not a constant).
    """
    product: float
    fertilizer: float
    wheat: float
    action: float

    @staticmethod
    def quantized(product, fertilizer, wheat, action, buckets: int = 10) -> "Prices":
        """Snap every quote to a coarse multiplicative grid so `solve`'s memo hits.

        A cold solve costs about 21 ms. Priced from a live forecast the quotes move every
        turn, so an exact key misses on essentially every call and a season of twenty animals
        costs minutes. Selection is snapped to `buckets` steps per decade; the resulting
        schedule is then valued exactly by `Plan.value_at`.
        """
        def snap(v):
            v = float(v)
            if v <= 0:
                return 0.0
            import math
            lg = math.log10(v)
            return round(10.0 ** (round(lg * buckets) / buckets), 4)
        return Prices(snap(product), snap(fertilizer), snap(wheat), snap(action))


@dataclass(frozen=True)
class State:
    alive: bool = True
    consecutive_unfed: int = 0
    pending_care: int = 0
    held: int = 0
    fertilizer_available: bool = False


@dataclass(frozen=True)
class Day:
    """One day of the solved schedule."""
    day: int
    feed: bool
    care: bool
    harvest: bool
    collect: bool
    product_units: int = 0      # units taken off the animal by this day's HARVEST
    fertilizer_units: int = 0   # units taken by this day's COLLECT_FERTILIZER

    @property
    def actions(self) -> int:
        return self.feed + self.care + self.harvest + self.collect

    @property
    def label(self) -> str:
        return ("F" if self.feed else "") + ("C" if self.care else "") \
             + ("H" if self.harvest else "") + ("K" if self.collect else "") or "-"


@dataclass(frozen=True)
class Plan:
    """The exact optimal service schedule for one animal over a horizon."""
    kind: str
    value: float                # cash-equivalent, purchase price NOT deducted
    days: tuple[Day, ...]
    survives: bool

    @property
    def actions(self) -> int:
        return sum(d.actions for d in self.days)

    @property
    def feed_units(self) -> int:
        return sum(1 for d in self.days if d.feed)

    @property
    def product_units(self) -> int:
        return sum(d.product_units for d in self.days)

    @property
    def fertilizer_units(self) -> int:
        return sum(d.fertilizer_units for d in self.days)

    def actions_on(self, day: int) -> int:
        for d in self.days:
            if d.day == day:
                return d.actions
        return 0

    def value_at(self, prices: "Prices") -> float:
        """Exact cash value of THIS schedule at `prices`.

        For a fixed action schedule the DP's objective is linear in every price, so the
        schedule can be selected on a coarse price grid -- which is what makes `solve`'s memo
        hit -- and then priced exactly here. The only approximation left is schedule choice,
        and a schedule is only ever re-chosen at a genuine breakpoint.
        """
        return (self.product_units * prices.product
                + self.fertilizer_units * prices.fertilizer
                - self.feed_units * prices.wheat
                - self.actions * prices.action)

    def profile(self, field: str) -> dict[int, int]:
        """`{day: n}` for one of `actions`, `feed_units`, `product_units`, `fertilizer_units`."""
        out = {}
        for d in self.days:
            n = d.actions if field == "actions" else \
                int(d.feed) if field == "feed_units" else getattr(d, field)
            if n:
                out[d.day] = n
        return out


def _choices(state: State):
    """(feed, care, harvest, collect) tuples worth considering.

    CARE without FEED is never banked (`fed_today` gates the bonus) and costs an action, so it
    is strictly dominated and excluded. HARVEST with nothing held and COLLECT with no refreshed
    fertilizer are engine no-ops and are excluded the same way.
    """
    if not state.alive:
        yield (False, False, False, False)
        return
    for feed, care in ((False, False), (True, False), (True, True)):
        for harvest in ((False, True) if state.held > 0 else (False,)):
            for collect in ((False, True) if state.fertilizer_available else (False,)):
                yield feed, care, harvest, collect


def step(kind: str, day: int, placed_day: int, state: State, action, prices: Prices,
         last_day: int | None = None):
    """One day: intraday unit actions, then the engine's end-of-day for this animal.

    Returns `(next_state, cash, product_units, fertilizer_units)`. `last_day` is the final
    actionable day; its end-of-day never runs (gotcha 20).
    """
    if not state.alive:
        return state, 0.0, 0, 0
    a = ANIMALS[kind]
    feed, care, harvest, collect = action
    held, fert = state.held, state.fertilizer_available
    cash = 0.0
    got_product = got_fert = 0

    # --- intraday: unit actions resolve before the end-of-day phase --------------
    if harvest:
        cash += held * prices.product - prices.action
        got_product, held = held, 0
    if collect:
        cash += prices.fertilizer - prices.action
        got_fert, fert = 1, False
    if feed:
        cash -= prices.wheat + prices.action
    if care:
        cash -= prices.action

    if last_day is not None and day >= last_day:
        # No end-of-day: no escape check, no production, no fertilizer refresh.
        return replace(state, held=held, fertilizer_available=fert), cash, got_product, got_fert

    # --- end of day --------------------------------------------------------------
    unfed = 0 if feed else state.consecutive_unfed + 1
    if unfed >= 2:
        return State(alive=False), cash, got_product, got_fert

    pending, nxt = state.pending_care, day + 1
    since = nxt - placed_day - a["first_yield_day"]
    if since >= 0 and since % a["interval"] == 0:
        held = min(a["max_held"], held + 1 + (pending if feed else 0))
        pending = 0
    if care and feed:          # today's care banks *after* today's production
        pending += 1
    return State(True, unfed, pending, held, True), cash, got_product, got_fert


@lru_cache(maxsize=1 << 17)
def solve(kind: str, placed_day: int, total_days: int, prices: Prices,
          initial: State = State(), start: int | None = None) -> Plan:
    """Exact optimal service schedule for one animal placed on `placed_day`.

    A cold solve is about 21 ms, which is a fifth of a turn budget, so this is memoized on its
    exact arguments; callers must therefore quantize forecast prices (see `Prices.quantized`)
    rather than passing raw floats, or the cache never hits.

    `total_days` is the season length, so the last actionable day is `total_days - 1` and its
    end-of-day never runs. `start` is the first day we get to choose actions on -- it defaults
    to `placed_day`, and differs when resuming an animal that is already on the board.
    The purchase price is NOT deducted; see `value_of`.
    """
    last = total_days - 1
    if start is None:
        start = placed_day

    @lru_cache(None)
    def dp(day: int, state: State):
        if day > last:
            return 0.0, (), 0
        best = None
        for action in _choices(state):
            nxt, cash, prod, fert = step(kind, day, placed_day, state, action, prices, last)
            fcash, ftail, facts = dp(day + 1, nxt)
            rec = Day(day, *action, product_units=prod, fertilizer_units=fert)
            total, acts = cash + fcash, rec.actions + facts
            # Deterministic tie-break: same money, fewer actions.
            key = (total, -acts)
            if best is None or key > best[0]:
                best = (key, total, (rec,) + ftail, acts)
        return best[1], best[2], best[3]

    value, days, _ = dp(start, initial)
    alive = initial
    for d in days:
        alive, *_ = step(kind, d.day, placed_day, alive, (d.feed, d.care, d.harvest, d.collect),
                         prices, last)
    return Plan(kind, value, days, alive.alive)


def value_of(kind: str, placed_day: int, total_days: int, prices: Prices,
             exact: Prices | None = None) -> float:
    """NPV of buying and placing one animal on `placed_day`, purchase and setup included.

    Setup is three unit-actions: PICKUP at the shed, walk-free PLACE, and the BUILD of the
    structure (free in cash, gotcha 21). Walking is priced by the route layer, not here.
    """
    plan = solve(kind, placed_day, total_days, prices)
    exact = exact or prices
    return plan.value_at(exact) - ANIMALS[kind]["cost"] - 3 * exact.action


def events_of(plan: Plan) -> tuple[dict, ...]:
    """The plan as dated events for the shared ledger (`neural.events`)."""
    out = []
    for d in plan.days:
        if d.feed:
            out.append({"day": d.day, "op": "FEED", "requires": {"WHEAT": 1}, "actions": 1})
        if d.care:
            out.append({"day": d.day, "op": "CARE", "actions": 1})
        if d.harvest:
            out.append({"day": d.day, "op": "HARVEST", "actions": 1,
                        "produces": {ANIMALS[plan.kind]["product"]: d.product_units}})
        if d.collect:
            out.append({"day": d.day, "op": "COLLECT_FERTILIZER", "actions": 1,
                        "produces": {"FERTILIZER": d.fertilizer_units}})
    return tuple(out)
