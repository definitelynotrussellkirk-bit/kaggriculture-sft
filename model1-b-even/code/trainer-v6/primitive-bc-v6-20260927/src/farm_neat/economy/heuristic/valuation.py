"""One place that decides HOW things are valued, so call sites only say WHAT they value.

Every review return asked for the same thing in different words: R1 wanted "one authoritative
data object" that economics, allocation, rendering and training features all derive from; R3
wanted "one event-calendar source of truth"; R5 wanted the fertilizer baselines named apart.
Before this module, `heuristic/policy.py` called seven different `econ` functions with
hand-threaded arguments, and a valuation choice could only be changed by editing every caller.

`Valuation` is built once per turn from the forecast and the config. It owns:

* which fertilizer baseline a caller means (sale, use-advantage, or gross marginal);
* whether an animal is priced by the exact service DP or the legacy closed form;
* the shadow price of a unit-action;
* price quantization, so the DP's memo actually hits inside the 1-second turn budget.

It deliberately does NOT own routing, capacity or the market queue. Those are separate layers.
"""
from __future__ import annotations

from dataclasses import astuple

from farm_neat.game.constants import ANIMALS
from . import animal_dp, econ

__all__ = ["Valuation"]


class Valuation:
    def __init__(self, fc, day: int, total_days: int, cfg=None):
        self.fc = fc
        self.day = day
        self.total_days = total_days
        self.action_cost = float(_cfg(cfg, "ACTION_COST", econ.ACTION_COST))
        self.dynamic_labour = bool(_cfg(cfg, "DYNAMIC_ACTION_COST", False))
        self.gross_fertilizer = bool(_cfg(cfg, "FERTILIZER_GROSS_MARGINAL", True))
        self.exact_animals = bool(_cfg(cfg, "ANIMAL_EXACT_DP", True))
        self._use_advantage = 0.0
        self._memo: dict = {}

    # --- fertilizer: three named quantities with three different baselines ------------

    def note_labour(self, job_values, capacity) -> float:
        """Reprice a unit-action from the current job surface (`econ.action_shadow_price`).

        Called with the PREVIOUS turn's surface, because job values themselves depend on the
        action cost -- an animal's daily net subtracts its service actions -- and pricing the
        surface with the price derived from that same surface is circular. A one-turn lag is
        exact enough for a quantity that moves on the scale of a day's labour supply.
        """
        if not self.dynamic_labour:
            return self.action_cost
        self.action_cost = econ.action_shadow_price(
            job_values, capacity, ceiling=4.0 * econ.ACTION_COST, default=self.action_cost)
        return self.action_cost

    def note_fertilizer_use(self, advantage: float) -> None:
        """Record the best available USE advantage; animal values include it."""
        self._use_advantage = max(self._use_advantage, float(advantage))

    def fertilizer_sale(self, horizon: float = 0.0) -> float:
        """Cash from selling one unit. Baseline: keeping it."""
        return econ.fertilizer_sale_value(self.fc, horizon)

    def fertilizer_use_advantage(self, tile, day=None) -> float:
        """How much better using one on `tile` is than selling it. Baseline: sold."""
        return econ.fertilizer_use_advantage(tile, self.day if day is None else day, self.fc)

    def fertilizer_marginal(self, horizon: float = 0.0) -> float:
        """What owning one more is worth. Baseline: not owning it. This is the one an
        animal's daily production line should be credited with."""
        return econ.fertilizer_gross_marginal_value(
            self.fc, self._use_advantage, horizon, self.gross_fertilizer)

    # --- animals ---------------------------------------------------------------------

    def _prices(self, kind: str, units: int, horizon: float):
        """`(coarse, exact)` -- coarse selects the schedule so the DP memo hits, exact prices it."""
        a = ANIMALS[kind]
        exact = animal_dp.Prices(
            float(self.fc.avg_sell_price(a["product"], max(1, units), horizon)),
            float(self.fertilizer_marginal(horizon)),
            float(self.fc.buy_price("WHEAT")),
            self.action_cost,
        )
        return animal_dp.Prices.quantized(*astuple(exact)), exact

    def animal_npv(self, kind: str, day=None) -> float:
        """NPV of buying and placing one more animal of `kind` today, purchase included."""
        day = self.day if day is None else day
        if not self.exact_animals:
            return econ.animal_npv(kind, day, self.total_days, self.fc, self._use_advantage,
                                   action_cost=self.action_cost)
        left = max(0, self.total_days - day)
        coarse, exact = self._prices(kind, max(1, ANIMALS[kind]["max_held"]), left / 2)
        return animal_dp.value_of(kind, day, self.total_days, coarse, exact)

    def animal_plan(self, kind: str, day=None) -> animal_dp.Plan:
        """The exact service schedule -- per-day feed/care/harvest/collect and the units they
        yield. This is what `neural.allocate` needs to charge an animal's real action stream
        instead of one flat rate, and what the doer needs for its task calendar."""
        day = self.day if day is None else day
        left = max(0, self.total_days - day)
        coarse, _ = self._prices(kind, ANIMALS[kind]["max_held"], left / 2)
        return animal_dp.solve(kind, day, self.total_days, coarse)

    def animal_remaining(self, tile, day=None) -> float:
        """Remaining value of a LIVING animal -- what an urgent FEED tonight protects.

        The exact route resumes the DP from the animal's actual engine state (held product,
        care bank, unfed streak, fertilizer flag), which the closed form ignored entirely.
        """
        day = self.day if day is None else day
        kind = tile.get("animal", "GOOSE")
        if not self.exact_animals:
            return econ.animal_remaining_value(tile, day, self.total_days, self.fc,
                                               self._use_advantage,
                                               action_cost=self.action_cost)
        left = max(0, self.total_days - day)
        if left <= 0:
            return 0.0
        state = animal_dp.State(
            alive=True,
            consecutive_unfed=int(tile.get("consecutive_unfed", 0)),
            pending_care=int(tile.get("pending_care_bonus", 0)),
            held=int(tile.get("yield_units", 0)),
            fertilizer_available=bool(tile.get("fertilizer_available", False)),
        )
        key = (kind, int(tile.get("placed_day", 0)), state, day)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        units = max(1, state.held or ANIMALS[kind]["max_held"])
        coarse, exact = self._prices(kind, units, left / 2)
        plan = animal_dp.solve(kind, int(tile.get("placed_day", 0)), self.total_days,
                               coarse, initial=state, start=day)
        self._memo[key] = value = plan.value_at(exact)
        return value

    def feed_value(self, tile, urgent: bool, day=None) -> float:
        """What tonight's FEED is worth on `tile`.

        Urgent means the animal escapes without it, so the whole remaining value is at risk.
        A routine feed only protects the care bank and the next production, which the DP
        prices exactly as the difference between feeding and not feeding tonight.
        """
        if urgent:
            return self.animal_remaining(tile, day)
        if not self.exact_animals:
            return econ.routine_feed_value(tile, self.fc, self._use_advantage,
                                           action_cost=self.action_cost)
        day = self.day if day is None else day
        with_feed = self.animal_remaining(tile, day)
        skipped = dict(tile)
        skipped["consecutive_unfed"] = int(tile.get("consecutive_unfed", 0)) + 1
        skipped["pending_care_bonus"] = 0
        return max(0.0, with_feed - self.animal_remaining(skipped, day + 1))


def _cfg(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
