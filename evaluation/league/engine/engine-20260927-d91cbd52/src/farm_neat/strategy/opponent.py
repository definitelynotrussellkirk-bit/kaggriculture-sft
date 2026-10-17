"""Public-only, bounded online opponent-flow state; no economic proposals.

Coordinator interface (one model may serve either seat; it has no mutable state)::

    model = OpponentModel()
    published = model.publish(context)         # derived['opponent']: OpponentState
    action = coordinator(published)            # this module chooses no action
    memory = model.remember(published, action) # commit the ACTUAL submitted action
    # Next pre-market observation: StrategyContext(..., memory=memory).

Only current market.inventory, published town.unlocked_shops (or context.shops),
step/player and this module's own memory are read. No farms, private inventories,
other derived values, tapes, replay labels, future observations or evaluator facts
are read. Missing own-action records are NOT PASS. Clear memory['opponent'] at a
new game; it is seat/time bound, never keyed by a hidden seed/world identifier.

For an observed t -> t+1 transition, and only when our own same-product queue was
passive, E_eff = I(t+1)-I(t)+D(t). This is historical effective displacement, not
rival requested/gross fills or cash. Zero does not identify PASS (floor sales and
closed cycles can both be invisible). No rank, cycle size or policy family can be
certified from these endpoints. The ladder's replay action matches are not an
actor-visible policy identity.

The entire empirical marginal law (integer counts, exact rational probabilities)
is retained, conditioned on ordered published shops and the deterministic demand
vector of the market event. Its empirical support is MEASURED, its reuse as a
next-event forecast is HEURISTIC, and its exhaustive future support is UNKNOWN.
It is NOT a joint law across products/time or a counterfactual active-actor law.
There is no calibrated forecast, full-fill certificate or terminal value claim.

Domain: the bundled default town clocks and market-order cap, as declared by the
caller; StrategyContext has no configuration certificate. Set default_calendar=False
when the default rules are not established: all flow labels then remain UNKNOWN.
The model uses O(history_limit * len(PRODUCTS)**2) time and
O(history_limit * len(PRODUCTS)) space, history_limit <= 718; the product set
is fixed at nine. No native game, search, file/network read or GPU is performed. Only the versioned
whitelisted observation/action projection is persisted, not an engine snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from farm_neat.evaluation.market_flow_labels import demand_at_step
from farm_neat.game.constants import DEFAULTS, MAX_MARKET_ORDERS, MAX_SHOP_INSTANCES, PRODUCTS, SHOPS
from farm_neat.strategy.types import Evidence, Estimate, LAST_STEP, StrategyContext, UNKNOWN

MEMORY_SCHEMA = "strategy-opponent-memory-v1"
STATE_SCHEMA = "strategy-opponent-state-v1"
PARENTS = (
    "1c940884402c5260d22d02e9714d2096ea8987d33908f232e1a980e8564e4abc",
    "c696f8a84b94d72dc54a7570db3f9eef757a98598a1a9157032057ff975977a1",
    "9dcaddabe881ff3098bc0cb256aa9c14dc293491d50fcd2b141d3fe7d8b10df5",
    "4c1d9de9c0c7d10ccf28d381b4fbb79118a8ed39ea9bc912beda6cfd8b5f6fc1",
)
_PRODUCT_INDEX = {p: i for i, p in enumerate(PRODUCTS)}
_UNKNOWN_VECTOR = (None,) * len(PRODUCTS)


def _wire(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class FlowLaw:
    """Empirical E_eff law, not an all-support or full-fill certificate."""
    counts: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        try:
            counts = tuple(tuple(row) for row in self.counts)
        except TypeError as error:
            raise ValueError("counts need (integer effect, positive integer count) rows") from error
        if (any(len(row) != 2 or type(row[0]) is not int or type(row[1]) is not int
                or row[1] <= 0 for row in counts)
                or len({row[0] for row in counts}) != len(counts)):
            raise ValueError("counts need unique integer effects and positive integer counts")
        object.__setattr__(self, "counts", tuple(sorted(counts)))

    @classmethod
    def from_samples(cls, samples: Iterable[int]) -> FlowLaw:
        values = tuple(samples)
        if any(type(x) is not int for x in values):
            raise ValueError("flow samples must be integers")
        return cls(tuple(Counter(values).items()))

    @property
    def sample_count(self) -> int:
        return sum(n for _, n in self.counts)

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(e for e, _ in self.counts)

    @property
    def atoms(self) -> tuple[tuple[int, Fraction], ...]:
        n = self.sample_count
        return tuple((e, Fraction(count, n)) for e, count in self.counts)

    @property
    def evidence(self) -> Evidence:
        return Evidence.MEASURED if self.counts else Evidence.UNKNOWN

    def moments(self) -> tuple[Fraction, Fraction] | None:
        """Diagnostics only: these are deliberately not the law or its cache key."""
        if not self.counts:
            return None
        mean = sum((e * p for e, p in self.atoms), Fraction(0))
        return mean, sum((p * (e - mean) ** 2 for e, p in self.atoms), Fraction(0))

    def to_dict(self) -> dict:
        return {"target": "E_eff", "evidence": self.evidence.value,
                "counts": [list(row) for row in self.counts], "sample_count": self.sample_count,
                "support": list(self.support), "support_scope": "OBSERVED_NOT_EXHAUSTIVE_FUTURE",
                "atoms": [{"effect": e, "numerator": p.numerator, "denominator": p.denominator}
                          for e, p in self.atoms]}


@dataclass(frozen=True)
class MarketEvent:
    step: int
    player: int
    shops: tuple[str, ...] | None
    demand: Mapping[str, Estimate]
    shop_event_index: Estimate
    # Joint-turn index and public shop-event count are NOT a rival market rank.
    rival_order_rank: Estimate = UNKNOWN
    alignment: str = "UNKNOWN"

    def __post_init__(self):
        object.__setattr__(self, "demand", MappingProxyType(dict(self.demand)))

    def to_dict(self) -> dict:
        return {"step": self.step, "player": self.player, "phase": "PRE_MARKET",
                "shops": list(self.shops) if self.shops is not None else None,
                "demand": {p: v.to_dict() for p, v in self.demand.items()},
                "shop_event_index": self.shop_event_index.to_dict(),
                "rival_order_rank": self.rival_order_rank.to_dict(), "alignment": self.alignment}


@dataclass(frozen=True)
class FlowSample:
    product: str
    start_step: int
    end_step: int
    effect: Estimate
    demand: Estimate
    own_order_ranks: tuple[int, ...] | None
    reason: str | None
    shops: tuple[str, ...] | None
    demand_vector: tuple[int | None, ...]
    rival_order_rank: Estimate = UNKNOWN
    alignment: str = "UNKNOWN"

    @property
    def regime(self):
        return None if self.shops is None or None in self.demand_vector else (self.shops, self.demand_vector)

    def to_dict(self) -> dict:
        return {"product": self.product, "start_step": self.start_step, "end_step": self.end_step,
                "effect": self.effect.to_dict(), "demand": self.demand.to_dict(),
                "own_order_ranks": list(self.own_order_ranks) if self.own_order_ranks is not None else None,
                "reason": self.reason, "shops": list(self.shops) if self.shops is not None else None,
                "conditioning_demand": list(self.demand_vector),
                "rival_order_rank": self.rival_order_rank.to_dict(), "alignment": self.alignment}


@dataclass(frozen=True)
class FlowBelief:
    product: str
    law: FlowLaw
    sample_steps: tuple[int, ...]  # start steps; every admitted sample has arrived by as_of_step
    as_of_step: int
    unknown_history_count: int
    support_bound: Estimate = UNKNOWN

    @property
    def forecast_evidence(self) -> Evidence:
        return Evidence.HEURISTIC if self.law.counts else Evidence.UNKNOWN

    def to_dict(self) -> dict:
        return {"product": self.product, "law": self.law.to_dict(),
                "sample_steps": list(self.sample_steps), "as_of_step": self.as_of_step,
                "last_observed_step": self.sample_steps[-1] + 1 if self.sample_steps else None,
                "unknown_history_count": self.unknown_history_count,
                "forecast_evidence": self.forecast_evidence.value,
                "support_bound": self.support_bound.to_dict(),
                "forecast_horizon": "NEXT_MARKET_PHASE",
                "scope": "SAME_PRODUCT_PASSIVE_ACTOR_MARGINAL_NOT_ALIGNED_CYCLE",
                "cross_product_and_temporal_dependence": "UNKNOWN"}


@dataclass(frozen=True)
class OpponentState:
    event: MarketEvent
    history: tuple[FlowSample, ...]
    beliefs: Mapping[str, FlowBelief]
    public_fingerprint: str
    history_limit: int
    default_calendar: bool
    policy_family: str = "UNKNOWN"
    policy_family_evidence: Evidence = Evidence.UNKNOWN
    rival_order_rank: Estimate = UNKNOWN
    alignment: str = "UNKNOWN"

    def __post_init__(self):
        object.__setattr__(self, "beliefs", MappingProxyType(dict(self.beliefs)))

    def latest(self, product: str) -> FlowSample | None:
        if product not in _PRODUCT_INDEX:
            raise KeyError(product)
        return next((row for row in reversed(self.history) if row.product == product), None)

    def to_dict(self) -> dict:
        return {"schema": STATE_SCHEMA, "event": self.event.to_dict(),
                "history": [row.to_dict() for row in self.history],
                "beliefs": {p: b.to_dict() for p, b in self.beliefs.items()},
                "public_fingerprint": self.public_fingerprint,
                "fingerprint_scope": "CURRENT_PUBLIC_PROJECTION_NOT_POLICY_IDENTITY",
                "history_limit": self.history_limit, "default_calendar": self.default_calendar,
                "policy_family": self.policy_family, "policy_family_evidence": self.policy_family_evidence.value,
                "rival_order_rank": self.rival_order_rank.to_dict(), "alignment": self.alignment,
                "parents": list(PARENTS),
                "excludes": ["rival custody", "future orders or shops", "gross fills", "cycle sizes",
                             "cash/margin deltas", "counterfactual flow", "terminal value"]}


@dataclass(frozen=True)
class _Snapshot:
    step: int
    player: int
    inventory: tuple[int | None, ...]
    shops: tuple[str, ...] | None

    def to_dict(self) -> dict:
        return {"step": self.step, "player": self.player, "inventory": list(self.inventory),
                "shops": list(self.shops) if self.shops is not None else None}


@dataclass(frozen=True)
class _Record:
    snapshot: _Snapshot
    ranks: tuple[tuple[int, ...], ...] | None

    def to_dict(self) -> dict:
        return {"snapshot": self.snapshot.to_dict(),
                "ranks": [list(r) for r in self.ranks] if self.ranks is not None else None}


def _shop_prefix(value, step: int, default_calendar: bool) -> tuple[str, ...] | None:
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or len(value) > MAX_SHOP_INSTANCES
            or any(not isinstance(s, str) or s not in SHOPS for s in value)):
        raise ValueError("invalid published shop prefix")
    if default_calendar:
        interval = DEFAULTS["turnsPerDay"] * DEFAULTS["townShopUnlockInterval"]
        expected = min(MAX_SHOP_INSTANCES, step // interval)
        if len(value) > expected:
            raise ValueError("shop prefix contains not-yet-published instances")
        if len(value) != expected:
            return None  # incomplete prefix cannot certify town demand
    return tuple(value)  # repetitions and order are essential


class OpponentModel:
    """Stateless typed-answer module. Other modules consume derived['opponent'].

    It emits no Proposal family and owns no economic action, cash delta or
    coordinator decision. remember() must be called with the actual action after
    selection; proposing then discarding an action does not create a valid record.
    """
    name = "opponent"

    def __init__(self, *, history_limit: int = 96, default_calendar: bool = True):
        if type(history_limit) is not int or not 1 <= history_limit <= LAST_STEP:
            raise ValueError("history_limit must be an integer in 1..718")
        if type(default_calendar) is not bool:
            raise ValueError("default_calendar must be a bool")
        self.history_limit = history_limit
        self.default_calendar = default_calendar

    def _capture(self, context: StrategyContext) -> _Snapshot:
        if not isinstance(context, StrategyContext):
            raise TypeError("StrategyContext required")
        obs = context.observation
        for key, expected in (("step", context.step), ("player", context.player)):
            if key in obs and (type(obs[key]) is not int or obs[key] != expected):
                raise ValueError("observation/context " + key + " mismatch")
        market = obs.get("market", {})
        inventory = market.get("inventory", {}) if isinstance(market, Mapping) else {}
        if not isinstance(inventory, Mapping):
            inventory = {}
        values = tuple(inventory.get(p) for p in PRODUCTS)
        if any(v is not None and type(v) is not int for v in values):
            raise ValueError("public inventory must contain integers or missing values")
        town = obs.get("town", {})
        prefix = town.get("unlocked_shops", context.shops) if isinstance(town, Mapping) else context.shops
        if context.shops and (not isinstance(prefix, (list, tuple)) or tuple(prefix) != context.shops):
            raise ValueError("observation/context published prefix mismatch")
        return _Snapshot(context.step, context.player, values,
                         _shop_prefix(prefix, context.step, self.default_calendar))

    def _demand(self, snapshot: _Snapshot) -> tuple[int | None, ...]:
        if not self.default_calendar or snapshot.shops is None:
            return _UNKNOWN_VECTOR
        values = demand_at_step(snapshot.shops, snapshot.step)
        return tuple(values[p] for p in PRODUCTS)

    def _records(self, context: StrategyContext, now: _Snapshot) -> list[_Record]:
        data = context.memory.get(self.name)
        if data is None:
            return []
        if (not isinstance(data, Mapping) or set(data) != {"schema", "player", "default_calendar", "records"}
                or data["schema"] != MEMORY_SCHEMA or type(data["player"]) is not int
                or data["player"] != now.player or type(data["default_calendar"]) is not bool
                or data["default_calendar"] != self.default_calendar
                or not isinstance(data["records"], list) or len(data["records"]) > LAST_STEP + 1):
            raise ValueError("incompatible or malformed opponent memory")
        records = []
        for row in data["records"]:
            if not isinstance(row, Mapping) or set(row) != {"snapshot", "ranks"}:
                raise ValueError("invalid opponent memory record")
            s = row["snapshot"]
            if (not isinstance(s, Mapping) or set(s) != {"step", "player", "inventory", "shops"}
                    or type(s["step"]) is not int or not 0 <= s["step"] <= now.step
                    or type(s["player"]) is not int or s["player"] != now.player
                    or not isinstance(s["inventory"], list) or len(s["inventory"]) != len(PRODUCTS)
                    or any(v is not None and type(v) is not int for v in s["inventory"])):
                raise ValueError("invalid opponent memory snapshot or time/seat mismatch")
            shops = _shop_prefix(s["shops"], s["step"], self.default_calendar)
            if s["shops"] is not None and shops is None:
                raise ValueError("invalid opponent memory published prefix")
            snapshot = _Snapshot(s["step"], s["player"], tuple(s["inventory"]), shops)
            ranks = row["ranks"]
            if ranks is not None:
                if (not isinstance(ranks, list) or len(ranks) != len(PRODUCTS)
                        or any(not isinstance(r, list) or any(type(i) is not int or not 0 <= i < MAX_MARKET_ORDERS for i in r)
                               or r != sorted(set(r)) for r in ranks)):
                    raise ValueError("invalid opponent memory own order ranks")
                flat = [i for r in ranks for i in r]
                if len(flat) != len(set(flat)):
                    raise ValueError("one own order rank cannot touch two products")
                ranks = tuple(tuple(r) for r in ranks)
            if records and records[-1].snapshot.step >= snapshot.step:
                raise ValueError("opponent memory requires strictly increasing steps")
            records.append(_Record(snapshot, ranks))
        sequence = [r.snapshot for r in records] + [now]
        for before, after in zip(sequence, sequence[1:]):
            if before.step == after.step and before != after:
                raise ValueError("conflicting public observation at the same step")
            if before.shops is not None and after.shops is not None and after.shops[:len(before.shops)] != before.shops:
                raise ValueError("published shop prefix was rewritten; clear memory at a new game")
        return records[-(self.history_limit + 1):]

    def _ranks(self, action: Mapping[str, Any]) -> tuple[tuple[int, ...], ...] | None:
        if not isinstance(action, Mapping):
            raise ValueError("actual submitted action must be a mapping")
        orders = action.get("market", [])
        if not isinstance(orders, list):
            return None
        ranks: list[list[int]] = [[] for _ in PRODUCTS]
        for i, order in enumerate(orders[:MAX_MARKET_ORDERS]):
            if not isinstance(order, list) or not order:
                continue  # native parser ignores non-list orders and placeholders
            if isinstance(order[0], str) and order[0] in ("SELL", "BUY_PRODUCT"):
                if len(order) < 2 or not isinstance(order[1], str):
                    return None
                if order[1] in _PRODUCT_INDEX:
                    # Even zero/malformed/failed requests are conservatively active;
                    # requests never become fills and BUY/SELL never net to PASS.
                    ranks[_PRODUCT_INDEX[order[1]]].append(i)
        return tuple(tuple(r) for r in ranks)

    def observe(self, context: StrategyContext) -> OpponentState:
        """Derive a fresh answer; never mutate memory or consume future input."""
        now = self._capture(context)
        records = self._records(context, now)
        if records and records[-1].snapshot.step == now.step:
            records = records[:-1]  # idempotent observation after same-step remember
        records = records[-self.history_limit:]
        demand_now = self._demand(now)
        event = MarketEvent(now.step, now.player, now.shops,
                            {p: UNKNOWN if d is None else Estimate.exact(d) for p, d in zip(PRODUCTS, demand_now)},
                            UNKNOWN if now.shops is None else Estimate.exact(len(now.shops)))
        history = []
        afters = [r.snapshot for r in records[1:]] + [now]
        for record, after in zip(records, afters):
            before = record.snapshot
            demands = self._demand(before)
            for j, p in enumerate(PRODUCTS):
                ranks = record.ranks[j] if record.ranks is not None else None
                reason = ("NONCONTIGUOUS_OBSERVATIONS" if after.step != before.step + 1 else
                          "CALENDAR_UNCERTIFIED" if not self.default_calendar else
                          "PUBLISHED_PREFIX_UNCERTIFIED" if demands[j] is None else
                          "MISSING_PUBLIC_INVENTORY" if before.inventory[j] is None or after.inventory[j] is None else
                          "OWN_ACTION_UNCERTIFIED" if ranks is None else
                          "OWN_SAME_PRODUCT_REQUEST" if ranks else None)
                effect = UNKNOWN
                if reason is None:
                    units = after.inventory[j] - before.inventory[j] + demands[j]
                    # Shared Estimate stores floats. Never certify an integer that
                    # this wire type rounds, even for synthetic extreme inputs.
                    try:
                        effect = Estimate.exact(units)
                        if int(effect.value) != units:
                            reason, effect = "EFFECT_NOT_EXACTLY_REPRESENTABLE", UNKNOWN
                    except (ValueError, OverflowError):
                        reason, effect = "EFFECT_NOT_EXACTLY_REPRESENTABLE", UNKNOWN
                history.append(FlowSample(p, before.step, after.step, effect,
                                          UNKNOWN if demands[j] is None else Estimate.exact(demands[j]),
                                          ranks, reason, before.shops, demands))
        regime = None if now.shops is None or None in demand_now else (now.shops, demand_now)
        beliefs = {}
        for p in PRODUCTS:
            rows = [row for row in history if row.product == p]
            selected = [r for r in rows if r.effect.kind is Evidence.EXACT and regime is not None and r.regime == regime]
            beliefs[p] = FlowBelief(p, FlowLaw.from_samples(int(r.effect.value) for r in selected),
                                    tuple(r.start_step for r in selected), now.step,
                                    sum(r.effect.kind is Evidence.UNKNOWN for r in rows))
        return OpponentState(event, tuple(history), beliefs,
                             hashlib.sha256(_wire(now.to_dict()).encode()).hexdigest(),
                             self.history_limit, self.default_calendar)

    def publish(self, context: StrategyContext) -> StrategyContext:
        """Copy context with one typed answer; no in-place publication or execution."""
        return context.with_derived(self.name, self.observe(context))

    def remember(self, context: StrategyContext, action: Mapping[str, Any]) -> dict:
        """Return own memory after actual action selection, preserving other keys.

        Only product-touch ranks are stored, not requests/fills/private custody.
        At most history_limit+1 snapshots preserve the same history both before
        and after saving the current action (including a full-window restart).
        Same-step saves must agree on this projection; caller must not share the
        returned per-seat memory across games or record a discarded proposal.
        """
        now = self._capture(context)
        records = self._records(context, now)
        ranks = self._ranks(action)
        if records and records[-1].snapshot.step == now.step:
            if records[-1].ranks != ranks:
                raise ValueError("different action projection already recorded at the same step")
            records = records[:-1]
        records.append(_Record(now, ranks))
        return {**context.memory, self.name: {"schema": MEMORY_SCHEMA, "player": now.player,
                "default_calendar": self.default_calendar,
                "records": [row.to_dict() for row in records[-(self.history_limit + 1):]]}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Public-only opponent flow model; no games.")
    parser.add_argument("--example", action="store_true", help="Print a deterministic two-observation example")
    args = parser.parse_args(argv)
    if args.example:
        model = OpponentModel()
        before = StrategyContext(1, 0, {"market": {"inventory": dict.fromkeys(PRODUCTS, 10000)}})
        memory = model.remember(before, {"market": []})
        inventory = {**dict.fromkeys(PRODUCTS, 10000), "WHEAT": 9997}
        after = StrategyContext(2, 0, {"market": {"inventory": inventory}}, memory)
        print(json.dumps(model.publish(after).derived["opponent"].to_dict(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
