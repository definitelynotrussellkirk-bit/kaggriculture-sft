"""Executor-neutral strategic directive shared by learning, solvers, and telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from farm_neat.game.constants import CROPS


MACRO_DIRECTIVE_VERSION = 3


@dataclass(frozen=True)
class MacroDirective:
    """One day's strategic request, before an executor applies feasibility and clipping.

    It deliberately describes desired outcomes rather than implementation knobs. The production
    executor owns the mapping to routing, market, and farm-management configuration.
    """

    cash_crop: str | None
    target_hands: int
    minimum_sell_ratio: float
    allow_land_purchase: bool
    target_goose_population: int
    feed_cover_days: int
    target_cow_population: int = 0
    target_sheep_population: int = 0
    early_herd_target: int = 0
    version: int = MACRO_DIRECTIVE_VERSION

    def __post_init__(self):
        if self.version != MACRO_DIRECTIVE_VERSION:
            raise ValueError(f"unsupported macro directive version {self.version}")
        if self.cash_crop is not None and self.cash_crop != "AUTO" and self.cash_crop not in CROPS:
            raise ValueError(f"unknown cash crop {self.cash_crop!r}")
        if (
            self.target_hands < 0
            or self.target_goose_population < 0
            or self.target_cow_population < 0
            or self.target_sheep_population < 0
            or self.early_herd_target < 0
        ):
            raise ValueError("labor and livestock targets must be nonnegative")
        if self.minimum_sell_ratio < 0:
            raise ValueError("minimum sell ratio must be nonnegative")
        if self.feed_cover_days < 0:
            raise ValueError("feed cover must be nonnegative")

    @classmethod
    def from_head_values(cls, values: Mapping[str, Any]):
        crop = values["crop"]
        return cls(
            cash_crop=None if crop == "NONE" else str(crop),
            target_hands=int(values["hands"]),
            minimum_sell_ratio=float(values["sell"]),
            allow_land_purchase=bool(values["land"]),
            target_goose_population=int(values["geese"]),
            # Compatibility for frozen v1/flat-policy rows. New structured records always carry
            # the explicit feed head.
            feed_cover_days=int(values.get("feed", 0)),
            # Herd heads appended 2026-08-30; absent in frozen v1/flat rows (no pasture herd).
            target_cow_population=int(values.get("cows", 0)),
            target_sheep_population=int(values.get("sheep", 0)),
            early_herd_target=int(values.get("early", 0)),
        )

    def to_head_values(self) -> dict[str, Any]:
        return {
            "crop": "NONE" if self.cash_crop is None else self.cash_crop,
            "hands": self.target_hands,
            "sell": self.minimum_sell_ratio,
            "land": self.allow_land_purchase,
            "geese": self.target_goose_population,
            "feed": self.feed_cover_days,
            "cows": self.target_cow_population,
            "sheep": self.target_sheep_population,
            "early": self.early_herd_target,
        }

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def macro_directive_contract_payload() -> dict[str, Any]:
    return {
        "version": MACRO_DIRECTIVE_VERSION,
        "executor_neutral": True,
        "fields": [
            {
                "name": "cash_crop",
                "type": "optional crop identity",
                "meaning": "preferred cash crop; feed wheat remains an orthogonal allocation",
            },
            {
                "name": "target_hands",
                "type": "nonnegative integer",
                "meaning": "desired total daily hands, not immediate HIRE count",
            },
            {
                "name": "minimum_sell_ratio",
                "type": "nonnegative base-price ratio",
                "meaning": "sale floor subject to terminal liquidation",
            },
            {
                "name": "allow_land_purchase",
                "type": "boolean",
                "meaning": "permission for the executor's value/affordability gate",
            },
            {
                "name": "target_goose_population",
                "type": "nonnegative integer",
                "meaning": "irreversible livestock ceiling",
            },
            {
                "name": "feed_cover_days",
                "type": "nonnegative integer",
                "meaning": "desired wheat/feed reserve horizon independent of cash crop",
            },
            {
                "name": "target_cow_population",
                "type": "nonnegative integer",
                "meaning": "irreversible pasture cow ceiling",
            },
            {
                "name": "target_sheep_population",
                "type": "nonnegative integer",
                "meaning": "irreversible pasture sheep ceiling",
            },
            {
                "name": "early_herd_target",
                "type": "nonnegative integer",
                "meaning": "animals funded ahead of the seed budget through day 3",
            },
        ],
        "telemetry": {
            "requested": "the exact MacroDirective",
            "adapted_config": "production-executor tunables after semantic adaptation",
            "emitted": "typed wire actions accumulated during the directive day",
            "realized": "observed state effects relative to the day-start snapshot",
        },
    }
