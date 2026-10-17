"""Deterministic public-state projections of a plan: tapes, land, hands, cash, market, masks.

Everything here is computed from one observation, the plan and a declared flow belief
(per product, expected rival displacement per day; zero by default). Nothing reads the
rival's private state or the hidden shop schedule. Projections are the compiler's
commitments (what it will schedule) and the policy's features (phi); they are not
realization certificates: the executor realizes, blockers are its receipts.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from farm_neat.economy.heuristic.market import price_at
from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.game.constants import (ANIMALS, BUYABLE_PRODUCTS, MARKET_I0, MARKET_PARAMS,
                                      PRICE_FLOOR, SHOPS, TOWN_CENTER_PRODUCTS, hire_cost)

DAYS = V.SEASON_DAYS
LAST_STEP = V.LAST_STEP
BUYABLE = tuple(V.PRODUCTS.index(p) for p in BUYABLE_PRODUCTS)
MAX_QUANTITY = 100
DEFAULT_LABOR_FACTOR = 4.0  # executor turns per service action; measured, see ROUNDTRIP.json


def hire_bill(hands: int) -> int:
    return sum(hire_cost(k) for k in range(hands))


def day_hours(day: int) -> int:
    return 23 if day == DAYS - 1 else 24


def hands_for(actions: int, day: int, labor_factor: float = DEFAULT_LABOR_FACTOR) -> int:
    """Smallest hand count whose unit-turn bound covers the day's service actions (None if > cap)."""
    need = -(-int(actions * labor_factor * 1000) // 1000)  # ceil without float drift
    hours = day_hours(day)
    if need <= hours:
        return 0
    k = -(-(need - hours) // (hours - 1))
    return k if k <= V.HAND_CAP else None


# Fixed V0/reference calendar, not a world-specific schedule. Observations at t
# precede consumption at t; LAST_STEP is the last executed transition.
DEMAND_STOP = LAST_STEP + 1
DEMAND_SPEC = {
    "schema": "farm-v0-public-demand-path-v1",
    "observation_boundary": "before transition t; created events are [0,t)",
    "shop_interval": 4, "center_interval": 24,
    "reveal_steps": tuple(72 * j for j in range(1, 9)),
    "phase_edges": (0, 240, 480, DEMAND_STOP),
    "remaining_horizons": ("near:[t,t+72)", "mid:[t+72,t+240)", "late:[t+240,719)"),
    "prior": "uniform over sorted SHOPS, with replacement; unknown draws only",
    "shop_products": {s: tuple(SHOPS[s]) for s in sorted(SHOPS)},
    "center_products": tuple(TOWN_CENTER_PRODUCTS),
    "unsatisfied_demand": "max(0,-public_inventory); signed-inventory debt, not unfilled player orders",
    "target_inventory_gap": "max(0,public_I0-public_inventory); net replenishment gap, not cumulative demand",
}


def _event_count(start: int, end: int, interval: int) -> int:
    """Number of multiples of interval in the half-open interval [start,end)."""
    return max(0, (end + interval - 1) // interval - (start + interval - 1) // interval)


def _shop_rates(shops) -> dict[str, int]:
    return {p: sum((2 if len(SHOPS[s]) == 1 else 1) for s in shops if p in SHOPS[s])
            for p in V.PRODUCTS}


def _public_shop_prefix(step, shops) -> tuple[str, ...]:
    if (type(step) is not int or not 0 <= step <= DEMAND_STOP
            or not isinstance(shops, (list, tuple))
            or len(shops) != min(step // 72, 8)
            or any(type(s) is not str or s not in SHOPS for s in shops)):
        raise ValueError("V3 requires the default-calendar public reveal prefix")
    return tuple(shops)


class PublicDemandHistory:
    """Own public-memory cursor, one per physical game, initialized at the origin.

    Call observe (or State(..., demand_history=cursor)) at every reveal boundary;
    other observations may be sparse. It increments scheduled withdrawals over
    each observed interval, never reads inventory, supply, sales, private state or
    a future schedule. Repeated observations are idempotent. Skipping an unseen
    reveal, rewinding or changing an already observed reveal fails atomically.
    Persist payload() for exact continuation, not just the endpoint observation.
    """
    def __init__(self):
        self._step = 0
        self._shops = ()
        self._created = {p: [0, 0, 0] for p in V.PRODUCTS}

    def observe(self, observation: dict) -> dict:
        step = observation["step"]
        shops = _public_shop_prefix(step, (observation.get("town") or {}).get("unlocked_shops", []))
        if step < self._step or shops[:len(self._shops)] != self._shops:
            raise ValueError("public demand history cannot rewind or change its observed prefix")
        if len(self._shops) < 8 and 72 * (len(self._shops) + 1) < step:
            raise ValueError("unobserved reveal boundary: supply the complete public prefix history")
        # New shops become available AT the new observation, not before it.
        rates = _shop_rates(self._shops)
        created = copy.deepcopy(self._created)
        edges = DEMAND_SPEC["phase_edges"]
        for i, (left, right) in enumerate(zip(edges, edges[1:])):
            lo, hi = max(self._step, left), min(step, right)
            if lo >= hi:
                continue
            for p in V.PRODUCTS:
                created[p][i] += rates[p] * _event_count(lo, hi, 4)
                if p in TOWN_CENTER_PRODUCTS:
                    created[p][i] += _event_count(lo, hi, 24)
        self._step, self._shops, self._created = step, shops, created
        return self.payload()

    def payload(self) -> dict:
        return {"schema": DEMAND_SPEC["schema"], "step": self._step,
                "shops": list(self._shops), "created_by_phase": copy.deepcopy(self._created)}

    @classmethod
    def from_payload(cls, payload: dict) -> "PublicDemandHistory":
        if (not isinstance(payload, dict)
                or set(payload) != {"schema", "step", "shops", "created_by_phase"}
                or payload["schema"] != DEMAND_SPEC["schema"]):
            raise ValueError("exact public demand history schema required; no provenance extras")
        shops = _public_shop_prefix(payload["step"], payload["shops"])
        created = payload["created_by_phase"]
        if not isinstance(created, dict) or set(created) != set(V.PRODUCTS):
            raise ValueError("complete public demand history counters required")
        for p, row in created.items():
            if (not isinstance(row, (list, tuple)) or len(row) != 3
                    or any(type(x) is not int or x < 0 for x in row)
                    or any(row[i] and start >= payload["step"]
                           for i, start in enumerate(DEMAND_SPEC["phase_edges"][:-1]))
                    or (p == "FERTILIZER" and any(row))):
                raise ValueError("invalid public demand history counters")
        expected = cls()
        for index, reveal in enumerate(DEMAND_SPEC["reveal_steps"][:len(shops)]):
            expected.observe({"step": reveal,
                              "town": {"unlocked_shops": list(shops[:index + 1])}})
        if expected._step != payload["step"]:
            expected.observe({"step": payload["step"],
                              "town": {"unlocked_shops": list(shops)}})
        if any(list(created[p]) != expected._created[p] for p in V.PRODUCTS):
            raise ValueError("invalid public demand history counters")
        return expected


def _public_rival(observation: dict, player: int) -> dict:
    """Allowlisted other-farm summary. Never traverse any private/custody field."""
    farms = observation["farms"]
    farm = farms[1 - player] if player in (0, 1) and len(farms) == 2 else {}
    tiles = farm.get("tiles")
    present = None if tiles is None else [t for row in tiles for t in row if isinstance(t, dict)]
    summary = {"cash": farm.get("money"),
               "quadrants": None if "unlocked_quadrants" not in farm else len(farm["unlocked_quadrants"]),
               "hands_today": None if "hands" not in farm else len(farm["hands"]),
               "hires_today": farm.get("hires_today"),
               "occupied_sites": None if present is None else len(present)}
    products = {}
    for p in V.PRODUCTS:
        producers = None if present is None else [t for t in present
            if (t.get("kind") == "PLANT" and t.get("crop") == p)
            or (t.get("animal") in ANIMALS and ANIMALS[t["animal"]]["product"] == p)]
        products[p] = {
            "producer_sites": None if producers is None else len(producers),
            "visible_yield": None if producers is None or any("yield_units" not in t for t in producers)
                else sum(t["yield_units"] for t in producers),
            "serviced_today": None if producers is None or any(
                ("watered_today" if t.get("kind") == "PLANT" else "fed_today") not in t for t in producers)
                else sum(bool(t.get("watered_today", t.get("fed_today"))) for t in producers),
            "fertilizer_available": None if present is None else sum(
                bool(t.get("fertilizer_available")) for t in present if t.get("animal") in ANIMALS)
                if p == "FERTILIZER" else 0,
        }
    return {"summary": summary, "products": products}


class State:
    """The public facts of one observation the projections use."""

    # Older actor-only adapters construct State without __init__. Absence does
    # not invent ownership or a LAND authorization; explicit LAND binding still
    # requires validated ordered ownership in LandCommitment._check_state.
    quadrant_names = None
    land_commitment = None

    def __init__(self, observation: dict, *, demand_history: PublicDemandHistory | None = None,
                 land_commitment: LandCommitment | None = None):
        self.player = int(observation["player"])
        self.step = int(observation["step"])
        self.day, self.hour = divmod(self.step, 24)
        farm = observation["farms"][self.player]
        private = observation.get("private", {}) or {}
        self.cash = float(farm["money"])
        self.quadrants = len(farm.get("unlocked_quadrants", ["NW"]))
        self.quadrant_names = (None if "unlocked_quadrants" not in farm
                               else tuple(farm["unlocked_quadrants"]))
        # Legitimate own compiler memory, never a field taken from the observation.
        # Store its closed JSON payload so actor-input mutation guards can hash State.
        if land_commitment is not None:
            if not isinstance(land_commitment, LandCommitment):
                raise ValueError("typed own LAND commitment required")
            land_commitment.validate_observation(self)
        self.land_commitment = None if land_commitment is None else land_commitment.payload()
        self.hands_today = len(farm.get("hands", []))
        self.hires_today = int(farm.get("hires_today", 0))
        shed = private.get("shed", {}) or {}
        carried = private.get("inventories", []) or []
        self.stock = {p: int(shed.get(p, 0)) + sum(int(i.get(p, 0)) for i in carried) for p in V.PRODUCTS}
        self.seeds = {c: int(n) for c, n in (private.get("seeds", {}) or {}).items()}
        market = observation.get("market", {}) or {}
        self.inventory = {p: int((market.get("inventory") or {}).get(p, MARKET_I0)) for p in V.PRODUCTS}
        self.params = market.get("params")
        self.shops = list((observation.get("town", {}) or {}).get("unlocked_shops", []))
        tiles = farm.get("tiles", [])
        self.occupied = sum(1 for row in tiles for t in row if isinstance(t, dict))
        self.rival_public = _public_rival(observation, self.player)
        if demand_history is not None and not isinstance(demand_history, PublicDemandHistory):
            raise ValueError("explicit PublicDemandHistory cursor required")
        # The origin is the only endpoint with a known empty history. Non-origin
        # observations do not silently claim that cumulative demand is zero.
        if demand_history is None and self.step == 0 and not self.shops:
            demand_history = PublicDemandHistory()
        self.demand_history = None if demand_history is None else demand_history.observe(observation)

    def market_features(self, *, require_history: bool = True) -> dict:
        """Public, seed-independent V3 market features, in product/near-mid-late order.

        Remaining buckets include the current not-yet-executed transition. Known
        demand includes center + observed shops only. The prior adds only future
        shop instances (uniform with replacement), never realized future demand.
        Unsatisfied demand is signed-inventory debt; the separate I0 gap is net
        replenishment pressure. Neither is a player-sale/fulfillment statistic.
        """
        shops = _public_shop_prefix(self.step, self.shops)
        snapshot = self.demand_history
        if snapshot is not None and (snapshot["step"] != self.step or tuple(snapshot["shops"]) != shops):
            raise ValueError("public demand history does not match current observation")
        if require_history and snapshot is None:
            raise ValueError("V3 requires complete public demand history; endpoint reconstruction is prohibited")
        rates = _shop_rates(shops)
        prior = _shop_rates(tuple(sorted(SHOPS)))
        edges = (self.step, min(DEMAND_STOP, self.step + 72),
                 min(DEMAND_STOP, self.step + 240), DEMAND_STOP)
        out = {}
        for p in V.PRODUCTS:
            known, expected = [], []
            for lo, hi in zip(edges, edges[1:]):
                known.append(rates[p] * _event_count(lo, hi, 4)
                             + int(p in TOWN_CENTER_PRODUCTS) * _event_count(lo, hi, 24))
                events = sum(_event_count(max(lo, reveal), hi, 4)
                             for reveal in DEMAND_SPEC["reveal_steps"][len(shops):])
                expected.append(events * prior[p] / len(SHOPS))
            phase = None if snapshot is None else tuple(snapshot["created_by_phase"][p])
            out[p] = {"market_inventory": self.inventory[p],
                      "target_inventory_gap": max(0, (self.params or MARKET_PARAMS)[p]["I0"] - self.inventory[p]),
                      "unsatisfied_demand": max(0, -self.inventory[p]),
                      "total_demand_created": None if phase is None else sum(phase),
                      "created_by_phase": phase, "known_remaining": tuple(known),
                      "expected_unrevealed": tuple(expected), "remaining_reveals": 8 - len(shops)}
        return out

    def demand(self, product: str, step: int) -> int:
        """Calendar demand of the known shops and the town centre at one step (units)."""
        d = 0
        if step % 4 == 0:
            for shop in self.shops:
                items = SHOPS[shop]
                if product in items:
                    d += 2 if len(items) == 1 else 1
        if step % 24 == 0 and product in TOWN_CENTER_PRODUCTS:
            d += 1
        return d

    def next_demand(self, product: str, after: int) -> int | None:
        for t in range(after + 1, LAST_STEP + 1):
            if self.demand(product, t):
                return t
        return None

    def next_reveal(self, after: int) -> int | None:
        for j in range(1, 9):
            if 72 * j > after:
                return 72 * j if 72 * j <= LAST_STEP else None
        return None

    def horizon_step(self, product: int, horizon: int) -> int | None:
        name = V.HORIZONS[horizon]
        if name == "NOW":
            return self.step
        if name == "NEXT_DEMAND":
            return self.next_demand(V.PRODUCTS[product], self.step)
        if name == "NEXT_REVEAL":
            return self.next_reveal(self.step)
        if name == "NEXT_DAWN":
            t = 24 * (self.day + 1)
            return t if t <= LAST_STEP else None
        return LAST_STEP


class Tapes:
    """Daily aggregates of a plan's cohorts: units, inputs, acquisition cash, actions, sites."""

    __slots__ = ("prod", "wheat", "fert", "fixed", "acts", "occ")

    def __init__(self, plan: Plan):
        self.prod = {p: [0] * DAYS for p in V.PRODUCTS}
        self.wheat, self.fert, self.fixed, self.acts, self.occ = ([0] * DAYS for _ in range(5))
        for module, n in plan.cohorts.items():
            row = V.tape(module, plan.regime(module))
            primary, manure = self.prod[row["product"]], self.prod["FERTILIZER"]
            for t in range(DAYS):
                primary[t] += n * row["primary"][t]
                manure[t] += n * row["manure"][t]
                self.wheat[t] += n * row["wheat"][t]
                self.fert[t] += n * row["fertilizer"][t]
                self.fixed[t] += n * row["fixed_cash"][t]
                self.acts[t] += n * (row["service_actions"][t] + row["manure_actions"][t])
                self.occ[t] += n * row["occupancy"][t]


def land_days(occupancy: list[int], owned: int) -> dict[int, int]:
    """Quadrant index (1..3) -> first day its sites are needed, for quadrants not yet owned."""
    out = {}
    for t, sites in enumerate(occupancy):
        k = 0
        while sites > V.SITES_PER_QUADRANT * (k + 1) and k < 3:
            k += 1
        for j in range(owned, k + 1):
            out.setdefault(j, t)
    return out



LAND_COMMITMENT_SCHEMA = "farm-v0-land-commitment-v1"
LAND_COMMITMENT_FIELDS = frozenset({"schema", "timing", "target", "authorized_step",
                                    "purchase_step", "status", "fulfilled_step"})


@dataclass(frozen=True)
class LandCommitment:
    """One explicit LAND authorization, with a permanent fulfilled tombstone.

    Target is the compiler/engine quadrant index (1=NE, 2=SW, 3=SE), not a
    moving count of additional purchases. Dates come only from the public
    calendar and the Plan at authorization. No history is reconstructed on load.
    """

    timing: str
    target: int
    authorized_step: int
    purchase_step: int
    status: str = "PENDING"
    fulfilled_step: int | None = None

    def __post_init__(self):
        if not isinstance(self.timing, str) or self.timing not in V.LAND_TIMINGS[:3]:
            raise ValueError("invalid LAND timing")
        if type(self.target) is not int or not 1 <= self.target <= 3:
            raise ValueError("invalid LAND target")
        if (type(self.authorized_step) is not int or not 0 <= self.authorized_step <= LAST_STEP
                or self.authorized_step % 24):
            raise ValueError("invalid LAND authorization step")
        if (type(self.purchase_step) is not int
                or not self.authorized_step <= self.purchase_step <= LAST_STEP):
            raise ValueError("invalid LAND purchase step")
        if self.timing == "NEXT_DAWN":
            if self.purchase_step not in (self.authorized_step, min(LAST_STEP, self.authorized_step + 24)):
                raise ValueError("LAND date does not match authorized timing")
        elif self.purchase_step % 24:
            raise ValueError("LAND binding date must be a dawn")
        if not isinstance(self.status, str) or self.status not in ("PENDING", "FULFILLED"):
            raise ValueError("invalid LAND status")
        if self.status == "PENDING":
            if self.fulfilled_step is not None:
                raise ValueError("pending LAND cannot carry fulfillment")
        elif (type(self.fulfilled_step) is not int
              or not self.purchase_step < self.fulfilled_step <= LAST_STEP + 1):
            raise ValueError("invalid LAND fulfillment step")

    def payload(self) -> dict:
        return {"schema": LAND_COMMITMENT_SCHEMA, "timing": self.timing, "target": self.target,
                "authorized_step": self.authorized_step, "purchase_step": self.purchase_step,
                "status": self.status, "fulfilled_step": self.fulfilled_step}

    @classmethod
    def from_payload(cls, value: dict) -> LandCommitment:
        if (not isinstance(value, dict) or set(value) != LAND_COMMITMENT_FIELDS
                or value["schema"] != LAND_COMMITMENT_SCHEMA):
            raise ValueError("closed versioned LAND commitment required")
        return cls(**{k: v for k, v in value.items() if k != "schema"})

    def validate_plan(self, plan: Plan):
        if plan.land is None or V.LAND_TIMINGS[plan.land] != self.timing:
            raise ValueError("LAND commitment does not match Plan")

    @staticmethod
    def _check_state(state: State):
        if (type(state.step) is not int or not 0 <= state.step <= LAST_STEP + 1
                or type(state.quadrants) is not int or not 1 <= state.quadrants <= 4
                or state.quadrant_names != tuple(V.QUADRANTS[:state.quadrants])):
            raise ValueError("LAND requires observed ordered own quadrants and a valid step")

    def validate_observation(self, state: State):
        """Strict restoration/consumption check, with no inference or mutation."""
        self._check_state(state)
        expected = self.target + int(self.status == "FULFILLED")
        if state.quadrants != expected:
            raise ValueError("stale LAND target/status versus observed quadrants")
        if state.step < (self.fulfilled_step if self.status == "FULFILLED" else self.authorized_step):
            raise ValueError("LAND memory is ahead of the observation")

    def fulfill(self, state: State) -> LandCommitment:
        """A transition, not an intent receipt; only actual own ownership proves it."""
        self._check_state(state)
        if self.status != "PENDING":
            raise ValueError("duplicate LAND fulfillment")
        if state.quadrants != self.target + 1 or state.step <= self.purchase_step:
            raise ValueError("LAND fulfillment requires observed target acquisition")
        return replace(self, status="FULFILLED", fulfilled_step=state.step)

    def observe(self, state: State) -> LandCommitment:
        """Reconcile live observations once; repeated observations are idempotent."""
        self._check_state(state)
        if self.status == "PENDING" and state.quadrants == self.target + 1:
            return self.fulfill(state)
        self.validate_observation(state)
        return self

    @classmethod
    def bind(cls, state: State, plan: Plan, *, needed=None) -> LandCommitment:
        cls._check_state(state)
        if state.hour != 0 or state.step > LAST_STEP or state.quadrants >= 4 or plan.land is None:
            raise ValueError("LAND authorization requires a dawn and a then-next quadrant")
        timing = V.LAND_TIMINGS[plan.land]
        if timing not in V.LAND_TIMINGS[:3]:
            raise ValueError("reserved LAND timing is not an authorization")
        needed = land_days(Tapes(plan).occ, state.quadrants) if needed is None else needed
        day = needed.get(state.quadrants)
        if day is None:
            if timing == "NEXT_BINDING":
                raise ValueError("LAND NEXT_BINDING requires a binding day at authorization")
            step = state.step if timing == "NOW" else min(LAST_STEP, 24 * (state.day + 1))
        else:
            # Keep the existing timing semantics, but resolve only this target and
            # only at authorization/revision, never at a later KEEP epoch.
            binding = 24 * max(day, state.day)
            if timing == "NOW":
                step = state.step if day <= state.day + 1 else binding
            elif timing == "NEXT_DAWN":
                step = min(24 * (state.day + 1), binding) if day > state.day else binding
            else:
                step = binding
        return cls(timing, state.quadrants, state.step, step)


def hold_release(state: State, plan: Plan, hold_releases, product: int) -> int | None:
    """The step a HOLD releases at: the event bound when it was set while the plan keeps that exact
    overlay (never a horizon re-resolved at a later dawn), else the horizon resolved now."""
    primitive, horizon, _ = plan.overlay(product)
    if V.PRIMITIVES[primitive] != "HOLD":
        return None
    bound = (hold_releases or {}).get(product)
    if bound is not None and tuple(bound[0]) == (primitive, horizon):
        return max(state.step, bound[1])
    return state.horizon_step(product, horizon)


class Projection:
    """One plan projected from one state under one belief: the compiler's commitments."""

    def __init__(self, state: State, plan: Plan, *, belief=None, labor_factor: float = DEFAULT_LABOR_FACTOR,
                 hold_releases=None):
        self.state, self.plan, self.labor_factor = state, plan, labor_factor
        self.belief = {p: 0 for p in V.PRODUCTS} | dict(belief or {})
        # product index -> ((primitive, horizon), release step) bound when the HOLD was set;
        # a binding applies only while the plan still carries that exact overlay.
        self.hold_releases = dict(hold_releases or {})
        self.tapes = Tapes(plan)
        self.land = land_days(self.tapes.occ, state.quadrants)
        self.land_steps = self._land_steps()
        self.hands = [hands_for(self.tapes.acts[d], d, labor_factor) for d in range(DAYS)]
        self.cash_rows = self._cash_rows()
        # An overdue, still-unfulfilled purchase remains payable. Keep its actual
        # bound date in land_steps/commands; book only the outstanding cost today.
        # This shared postpass also covers the fast-menu Projection subclass.
        if (self.land_commitment is not None and self.land_commitment.status == "PENDING"
                and self.land_commitment.purchase_step // 24 < state.day and self.cash_rows):
            cost = V.LAND_PRICES[self.land_commitment.target - 1]
            self.cash_rows[0]["land"] += cost
            self.cash_rows[0]["net"] -= cost
        prefix, worst = 0.0, 0.0
        for row in self.cash_rows:
            prefix += row["net"]
            worst = min(worst, prefix)
        self.reserve = -worst
        self.binding = self._binding()

    # -- land -----------------------------------------------------------------------
    def _land_steps(self) -> dict[int, int]:
        """Explicit LAND consumes one commitment; absent LAND keeps capacity scheduling.

        A standalone/candidate Plan can resolve a new authorization here. Live
        agents supply their original memory in State; a changed LAND timing is
        the only candidate edit that can replace it. Projection never mutates it.
        """
        payload = self.state.land_commitment
        kept = None if payload is None else LandCommitment.from_payload(payload)
        if kept is not None:
            kept.validate_observation(self.state)
        if self.plan.land is None:
            if kept is not None:
                raise ValueError("LAND commitment cannot be implicitly cancelled")
            self.land_commitment = None
            return {j: 24 * max(day, self.state.day) for j, day in sorted(self.land.items())}
        if kept is None or V.LAND_TIMINGS[self.plan.land] != kept.timing:
            kept = LandCommitment.bind(self.state, self.plan, needed=self.land)
        self.land_commitment = kept
        return {} if kept.status == "FULFILLED" else {kept.target: kept.purchase_step}

    # -- cash -----------------------------------------------------------------------
    def _price(self, product: str, inventory: int) -> int:
        return price_at(product, inventory, self.state.params)

    def sell_value(self, product: str, inventory: int, quantity: int) -> tuple[int, int]:
        """(revenue, inventory after) of selling `quantity` solo at `inventory`; floor sales move nothing."""
        total = 0
        for _ in range(quantity):
            p = self._price(product, inventory)
            total += p
            if p > PRICE_FLOOR:
                inventory += 1
        return total, inventory

    def buy_cost(self, product: str, inventory: int, quantity: int) -> tuple[int, int]:
        total = 0
        for _ in range(quantity):
            inventory -= 1
            total += self._price(product, inventory)
        return total, inventory

    def _cash_rows(self) -> list[dict]:
        """Per remaining day: wages, land, acquisitions, inputs, receipts and the day's net.

        Convention (the knapsack relaxation): a day's production is sold at the next dawn
        under the product's overlay; inputs short of stock are bought at that dawn; demand
        of the known shops and the belief drift the inventory between dawns.
        """
        s, t = self.state, self.tapes
        inv = dict(s.inventory)
        stock = dict(s.stock)
        rows = []
        held = {p: 0 for p in V.PRODUCTS}
        for d in range(s.day, DAYS):
            wage = hire_bill(self.hands[d]) if self.hands[d] is not None else hire_bill(V.HAND_CAP)
            land = sum(V.LAND_PRICES[j - 1] for j, step in self.land_steps.items() if step // 24 == d)
            fixed = t.fixed[d]
            if d > s.day:
                for p in V.PRODUCTS:
                    stock[p] += t.prod[p][d - 1]
            needs = {"WHEAT": t.wheat[d], "FERTILIZER": t.fert[d]}
            inputs = 0
            for p, need in needs.items():
                short = need - stock[p]
                if short > 0:
                    cost, inv[p] = self.buy_cost(p, inv[p], short)
                    inputs += cost
                    stock[p] += short
                stock[p] -= need
            receipts = 0
            for p in V.PRODUCTS:
                primitive, horizon, _rank = self.plan.overlay(V.PRODUCTS.index(p))
                sell_day = self._overlay_sell_day(p, primitive, horizon)
                if stock[p] > 0 and (sell_day is None or d >= sell_day):
                    value, inv[p] = self.sell_value(p, inv[p], stock[p])
                    receipts += value
                    stock[p] = 0
                elif stock[p] > 0:
                    held[p] = stock[p]
            net = receipts - wage - land - fixed - inputs
            rows.append({"day": d, "wage": wage, "land": land, "fixed": fixed, "inputs": inputs,
                         "receipts": receipts, "net": net, "hands": self.hands[d]})
            for p in V.PRODUCTS:
                drift = sum(s.demand(p, step) for step in range(24 * d, min(24 * d + 24, LAST_STEP + 1))) - self.belief[p]
                inv[p] = max(0, inv[p] - drift)
        self.held = held
        return rows

    def hold_release(self, product: int) -> int | None:
        return hold_release(self.state, self.plan, self.hold_releases, product)

    def _overlay_sell_day(self, product: str, primitive: int, horizon: int) -> int | None:
        name = V.PRIMITIVES[primitive]
        if name in ("CLEAR", "CARRY", "FRONTRUN", "PIGGYBACK"):
            return None
        index = V.PRODUCTS.index(product)
        step = self.hold_release(index) if name == "HOLD" else self.state.horizon_step(index, horizon)
        return None if step is None else step // 24

    # -- feasibility --------------------------------------------------------------------
    def _binding(self) -> str | None:
        if any(h is None for h in self.hands[self.state.day:]):
            return "hands"
        if max(self.tapes.occ) > V.SITES_PER_QUADRANT * 4:
            return "sites"
        if self.reserve > self.state.cash + 1e-9:
            return "cash"
        return None

    @property
    def feasible(self) -> bool:
        return self.binding is None

    def peak_hands(self) -> int:
        return max((h for h in self.hands[self.state.day:] if h is not None), default=0)

    def wages(self) -> int:
        return sum(r["wage"] for r in self.cash_rows)


def project(state: State, plan: Plan, **kw) -> Projection:
    return Projection(state, plan, **kw)


class Resolver:
    """NEXT_BINDING and MAX_FEASIBLE counts for a module, from projections of the plan."""

    def __init__(self, state: State, **kw):
        self.state, self.kw = state, kw

    def feasible(self, plan: Plan) -> bool:
        return Projection(self.state, plan, **self.kw).feasible

    def __call__(self, plan: Plan, module: int, delta: str) -> int:
        current = plan.cohorts.get(module, 0)
        if V.MODULES[module]["birth"] < self.state.day:
            return current
        base = Projection(self.state, plan, **self.kw)
        signature = (tuple(base.hands), tuple(sorted(base.land.items())))
        best = current
        for n in range(current + 1, current + 1 + 4 * V.SITES_PER_QUADRANT):
            trial = plan.copy()
            trial.cohorts[module] = n
            p = Projection(self.state, trial, **self.kw)
            if not p.feasible:
                break
            if delta == "NEXT_BINDING" and (tuple(p.hands), tuple(sorted(p.land.items()))) != signature:
                if best == current:
                    best = n  # the first count that moves a kink is itself the kink
                break
            best = n
        return best


# -- market frontier ----------------------------------------------------------------------
def sell_frontier(state: State, proj: Projection, product: int, horizon: int) -> list[dict]:
    """Nondominated (reserve, slots, margin, q) points for SELL q at the horizon, rest at the next demand.

    Margin is own revenue under the belief and the known calendar; reserve is 0. Rank 0 is
    the empty point. The remainder sells at the next demand event after the horizon (or the
    horizon itself when none follows).
    """
    name = V.PRODUCTS[product]
    h = state.horizon_step(product, horizon)
    if h is None:
        return []
    stock = state.stock[name] + sum(proj.tapes.prod[name][state.day:max(state.day, h // 24)])
    stock = min(stock, MAX_QUANTITY)
    inv_h = _drifted(state, proj, name, state.step, h)
    later = state.next_demand(name, h) or h
    points = []
    for q in range(stock + 1):
        value, inv_after = proj.sell_value(name, inv_h, q)
        rest_inv = _drifted(state, proj, name, h, later, start_inventory=inv_after)
        rest_value, _ = proj.sell_value(name, rest_inv, stock - q)
        points.append({"reserve": 0, "slots": 1 if q else 0, "margin": value + rest_value, "q": q})
    return pareto(points)


def carry_frontier(state: State, proj: Projection, product: int, horizon: int, bank: float) -> list[dict]:
    """BUY q at a demand step, SELL q the step after: reserve O(q), two slots, margin the own gain.

    The bank cap enters first: only q with O(q) <= bank are candidates (K-CYCLE-HARD-BUDGET).
    """
    name = V.PRODUCTS[product]
    if product not in BUYABLE:
        return []
    h = state.horizon_step(product, horizon)
    if h is None or h >= LAST_STEP or not state.demand(name, h):
        return []
    inv_h = _drifted(state, proj, name, state.step, h)
    points = [{"reserve": 0, "slots": 0, "margin": 0, "q": 0}]
    for q in range(1, MAX_QUANTITY + 1):
        cost, inv_after = proj.buy_cost(name, inv_h, q)
        if cost > bank:
            break
        drift = state.demand(name, h) - proj.belief[name]
        revenue, _ = proj.sell_value(name, max(0, inv_after - drift), q)
        points.append({"reserve": cost, "slots": 2, "margin": revenue - cost, "q": q})
    return pareto(points)


def pareto(points: list[dict]) -> list[dict]:
    """Keep points no other point beats on (lower reserve, fewer slots, higher margin); sort by reserve, slots."""
    kept = []
    for a in points:
        dominated = any((b["reserve"] <= a["reserve"] and b["slots"] <= a["slots"] and b["margin"] >= a["margin"]
                         and (b["reserve"], b["slots"], b["margin"]) != (a["reserve"], a["slots"], a["margin"]))
                        for b in points)
        if not dominated and not any(k["reserve"] == a["reserve"] and k["slots"] == a["slots"] and k["margin"] == a["margin"] for k in kept):
            kept.append(a)
    kept.sort(key=lambda p: (p["reserve"], p["slots"], -p["margin"]))
    return kept[:V.RANKS]


def _drifted(state: State, proj: Projection, name: str, start: int, end: int, start_inventory: int | None = None) -> int:
    inv = state.inventory[name] if start_inventory is None else start_inventory
    for t in range(start, end):
        inv -= state.demand(name, t)
    days = max(0, (end - start) // 24)
    return max(0, inv - days * proj.belief[name])


def frontier(state: State, proj: Projection, product: int, primitive: int, horizon: int) -> list[dict]:
    name = V.PRIMITIVES[primitive]
    if name == "SELL":
        return sell_frontier(state, proj, product, horizon)
    if name == "CARRY":
        return carry_frontier(state, proj, product, horizon, state.cash - proj.reserve)
    if name in ("CLEAR", "HOLD"):
        return [{"reserve": 0, "slots": 0 if name == "HOLD" else 1, "margin": 0, "q": None}]
    return []  # FRONTRUN and PIGGYBACK: the executor has no same-turn BUY/SELL word yet


# -- masks and features ---------------------------------------------------------------------
class Masks:
    """Legal actions at one epoch state; every rule is a row of the factor table."""

    def __init__(self, state: State, plan: Plan, resolver: Resolver | None = None, **kw):
        self.state, self.plan, self.kw = state, plan, kw
        self.resolver = resolver or Resolver(state, **kw)
        self.base = Projection(state, plan, **kw)
        self._plus_one = {}
        self._frontiers = {}

    def plus_one_feasible(self, module: int) -> bool:
        if module not in self._plus_one:
            trial = self.plan.copy()
            trial.cohorts[module] = trial.cohorts.get(module, 0) + 1
            self._plus_one[module] = Projection(self.state, trial, **self.kw).feasible
        return self._plus_one[module]

    def frontier(self, product: int, primitive: int, horizon: int) -> list[dict]:
        key = (product, primitive, horizon)
        if key not in self._frontiers:
            self._frontiers[key] = frontier(self.state, self.base, product, primitive, horizon)
        return self._frontiers[key]

    def legal(self, action: int) -> bool:
        fields = V.decode(action)
        kind = fields[0]
        if kind == "STOP":
            return True
        if kind == "PROD":
            module, delta = fields[1], V.DELTAS[fields[2]]
            count = self.plan.cohorts.get(module, 0)
            born = V.MODULES[module]["birth"] < self.state.day
            if delta in ("DROP_ALL", "MINUS_ONE"):
                # A born crop cohort may exit (its tiles are freed for later cohorts); a born
                # animal cannot leave its structure, and no cohort can be added after its birth.
                return count > 0 and not (born and V.MODULES[module]["animal"])
            return not born and self.plus_one_feasible(module)
        if kind == "SERVICE":
            module, regime = fields[1], fields[2]
            return module in self.plan.cohorts and V.regime_legal(module, regime) and self.plan.regime(module) != regime
        if kind == "LAND":
            timing = V.LAND_TIMINGS[fields[1]]
            if self.state.quadrants >= 4 or timing == "LAST_PROFITABLE" or self.plan.land == fields[1]:
                return False
            if timing == "NEXT_BINDING" and not self.base.land:
                return False
            trial = self.plan.copy()
            trial.land = fields[1]
            return Projection(self.state, trial, **self.kw).feasible
        product, primitive, horizon, rank = fields[1:]
        name = V.PRIMITIVES[primitive]
        if (primitive, horizon, rank) == self.plan.overlay(product):
            return False
        if name == "CLEAR":
            return horizon == 0 and rank == 0
        if name == "HOLD":
            return rank == 0 and horizon != 0 and self.state.horizon_step(product, horizon) is not None \
                and (self.state.stock[V.PRODUCTS[product]] > 0 or any(self.base.tapes.prod[V.PRODUCTS[product]][self.state.day:]))
        if name in ("FRONTRUN", "PIGGYBACK"):
            return False
        points = self.frontier(product, primitive, horizon)
        return rank < len(points) and (rank > 0 or name != "SELL")

    def vector(self) -> list[bool]:
        return [self.legal(a) for a in range(V.N_ACTIONS)]

    def legal_actions(self) -> list[int]:
        out = [V.STOP_ID]
        for module in range(V.N_MODULES):
            for delta in range(len(V.DELTAS)):
                a = V.encode("PROD", module, delta)
                if self.legal(a):
                    out.append(a)
            for regime in range(len(V.REGIMES)):
                a = V.encode("SERVICE", module, regime)
                if self.legal(a):
                    out.append(a)
        for timing in range(len(V.LAND_TIMINGS)):
            a = V.encode("LAND", timing)
            if self.legal(a):
                out.append(a)
        for product in range(len(V.PRODUCTS)):
            for primitive in range(len(V.PRIMITIVES)):
                for horizon in range(len(V.HORIZONS)):
                    if V.PRIMITIVES[primitive] in ("FRONTRUN", "PIGGYBACK"):
                        continue
                    for rank in range(len(self.frontier(product, primitive, horizon)) or 1):
                        a = V.encode("MARKET", product, primitive, horizon, rank)
                        if self.legal(a):
                            out.append(a)
        return out


PHI_NAMES = ("reserve_delta", "reserve_after", "cash_slack", "peak_hands_delta", "wages_delta", "sites_delta",
             "sites_after", "wheat_delta", "fertilizer_delta", "output_delta", "land_price", "land_step_delta",
             "market_reserve", "market_slots", "market_margin", "binding_none", "binding_hands", "binding_sites",
             "binding_cash", "edits_days_to_birth")


def phi(state: State, plan: Plan, action: int, masks: Masks | None = None, **kw) -> list[float]:
    """The deterministic consequences of one edit, known before execution (section 5 of the guide)."""
    masks = masks or Masks(state, plan, **kw)
    base = masks.base
    fields = V.decode(action)
    new = plan.apply(action, masks.resolver) if fields[0] != "STOP" else plan
    after = Projection(state, new, **masks.kw) if fields[0] != "STOP" else base
    out = {name: 0.0 for name in PHI_NAMES}
    out["reserve_delta"] = after.reserve - base.reserve
    out["reserve_after"] = after.reserve
    out["cash_slack"] = state.cash - after.reserve
    out["peak_hands_delta"] = after.peak_hands() - base.peak_hands()
    out["wages_delta"] = after.wages() - base.wages()
    out["sites_delta"] = max(after.tapes.occ) - max(base.tapes.occ)
    out["sites_after"] = max(after.tapes.occ)
    out["wheat_delta"] = sum(after.tapes.wheat) - sum(base.tapes.wheat)
    out["fertilizer_delta"] = sum(after.tapes.fert) - sum(base.tapes.fert)
    out["output_delta"] = sum(sum(v) for v in after.tapes.prod.values()) - sum(sum(v) for v in base.tapes.prod.values())
    if fields[0] == "LAND" and after.land_steps:
        j = min(after.land_steps)
        out["land_price"] = V.LAND_PRICES[j - 1]
        out["land_step_delta"] = after.land_steps[j] - state.step
    if fields[0] == "MARKET":
        product, primitive, horizon, rank = fields[1:]
        points = masks.frontier(product, primitive, horizon)
        if rank < len(points):
            out["market_reserve"], out["market_slots"], out["market_margin"] = (points[rank]["reserve"], points[rank]["slots"], points[rank]["margin"])
    if fields[0] == "PROD":
        out["edits_days_to_birth"] = V.MODULES[fields[1]]["birth"] - state.day
    out["binding_" + (after.binding or "none")] = 1.0
    return [float(out[name]) for name in PHI_NAMES]
