"""Portable exact adapter for the bundled authoritative Kaggriculture interpreter.

The development packet intentionally carries the engine source and configuration.  This module
executes those exact bytes without requiring the optional ``kaggle_environments`` package.  It is
for reproducible development evaluation only; agents still receive only their normal observation.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[3]
# Focused review packets expose review-facing aliases; compact source packets may retain only the
# repository authority paths. Resolve either layout, then verify the same pinned bytes.
HERE = Path(__file__).resolve().parent
ENGINE_CANDIDATES = (HERE / "kaggriculture_engine_reference.py",)
ENGINE_CONFIG_CANDIDATES = (HERE / "kaggriculture.json",)
ENGINE_PATH = next((path for path in ENGINE_CANDIDATES if path.is_file()), ENGINE_CANDIDATES[0])
ENGINE_CONFIG_PATH = next(
    (path for path in ENGINE_CONFIG_CANDIDATES if path.is_file()),
    ENGINE_CONFIG_CANDIDATES[0],
)
EXPECTED_ENGINE_SHA256 = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"
EXPECTED_ENGINE_CONFIG_SHA256 = "a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867"

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}
DEFAULT_CONFIGURATION = {
    "episodeSteps": 720,
    "boardSize": 10,
    "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "farmHandCostMult": 1,
    "marketParams": {},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_bundled_engine_identity() -> dict[str, str]:
    if not ENGINE_PATH.is_file():
        checked = ", ".join(path.relative_to(ROOT).as_posix() for path in ENGINE_CANDIDATES)
        raise RuntimeError(f"bundled engine is missing; checked: {checked}")
    if not ENGINE_CONFIG_PATH.is_file():
        checked = ", ".join(
            path.relative_to(ROOT).as_posix() for path in ENGINE_CONFIG_CANDIDATES
        )
        raise RuntimeError(f"bundled engine configuration is missing; checked: {checked}")
    observed_engine = sha256_file(ENGINE_PATH)
    observed_config = sha256_file(ENGINE_CONFIG_PATH)
    if observed_engine != EXPECTED_ENGINE_SHA256:
        raise RuntimeError(
            f"bundled engine hash mismatch: expected {EXPECTED_ENGINE_SHA256}, got {observed_engine}"
        )
    if observed_config != EXPECTED_ENGINE_CONFIG_SHA256:
        raise RuntimeError(
            "bundled engine configuration hash mismatch: "
            f"expected {EXPECTED_ENGINE_CONFIG_SHA256}, got {observed_config}"
        )
    return {
        "engine_path": ENGINE_PATH.relative_to(ROOT).as_posix(),
        "engine_sha256": observed_engine,
        "configuration_path": ENGINE_CONFIG_PATH.relative_to(ROOT).as_posix(),
        "configuration_sha256": observed_config,
    }


class AttrDict(dict):
    """Attribute/mapping bridge matching Kaggle's interpreter structs."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _copy_wire(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_wire(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_wire(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported Kaggriculture wire value {type(value).__name__}")


def _install_resolve_episode_seed_stub() -> tuple[bool, bool]:
    """Install the one missing utility temporarily when Kaggle is absent.

    The returned flags let ``load_engine`` remove only modules created here. This avoids leaving a
    synthetic ``kaggle_environments`` package in ``sys.modules`` after the interpreter captures
    the helper function.
    """
    try:
        from kaggle_environments.utils import resolve_episode_seed as _unused  # noqa: F401
        return False, False
    except ImportError:
        pass

    package = sys.modules.get("kaggle_environments")
    created_package = package is None
    if package is None:
        package = types.ModuleType("kaggle_environments")
        package.__path__ = []  # type: ignore[attr-defined]
        sys.modules["kaggle_environments"] = package
    utils = sys.modules.get("kaggle_environments.utils")
    created_utils = utils is None
    if utils is None:
        utils = types.ModuleType("kaggle_environments.utils")
        sys.modules["kaggle_environments.utils"] = utils

    def resolve_episode_seed(env: Any) -> int:
        raw = env.configuration.get("seed", None)
        seed = 0 if raw is None else int(raw)
        env.info["seed"] = seed
        # Match the installed utility's privacy contract: the hidden seed is not retained in the
        # configuration object supplied to agents after initialization.
        env.configuration["seed"] = None
        return seed

    utils.resolve_episode_seed = resolve_episode_seed
    package.utils = utils
    return created_package, created_utils


_ENGINE = None


def load_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    assert_bundled_engine_identity()
    created_package, created_utils = _install_resolve_episode_seed_stub()
    spec = importlib.util.spec_from_file_location(
        "farm_default_plan_bundled_kaggriculture", ENGINE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load bundled engine from {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if created_utils:
            sys.modules.pop("kaggle_environments.utils", None)
        if created_package:
            sys.modules.pop("kaggle_environments", None)
    _ENGINE = module
    return module


@dataclass
class GameResult:
    rewards: tuple[float, float]
    statuses: tuple[str, str]
    steps: int
    final_state: list[AttrDict]
    actions: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    shop_history: list[dict[str, Any]]
    cash_history: tuple[list[float], list[float]]


class ExactKaggricultureEnvironment:
    """Minimal exact environment around the bundled interpreter."""

    def __init__(self, *, seed: int, episode_steps: int = 720, configuration: dict | None = None):
        self.engine = load_engine()
        supplied = dict(configuration or {})
        self.configuration = AttrDict(
            {**DEFAULT_CONFIGURATION, **supplied, "episodeSteps": int(episode_steps), "seed": int(seed)}
        )
        self.info = {"seed": int(seed)}
        self.done = False
        self.step_index = 0
        self.state = [
            AttrDict(
                observation=AttrDict({"player": player, **({"step": 0} if player == 0 else {})}),
                action=copy.deepcopy(PASS_ACTION),
                reward=0.0,
                status="ACTIVE",
                info={},
            )
            for player in range(2)
        ]
        self.engine.interpreter(self.state, self)

    def agent_observation(self, player: int) -> AttrDict:
        observation = AttrDict(_copy_wire(self.state[player].observation))
        observation["step"] = self.step_index
        return observation

    def step(self, joint_action: Iterable[dict[str, Any]]) -> list[AttrDict]:
        if self.done:
            raise ValueError("cannot step a completed environment")
        actions = list(joint_action)
        if len(actions) != 2:
            raise ValueError("joint action must contain exactly two seats")
        for player, action in enumerate(actions):
            self.state[player].action = copy.deepcopy(action)
        self.state[0].observation.step = self.step_index
        self.engine.interpreter(self.state, self)
        self.step_index += 1
        self.state[0].observation.step = self.step_index
        self.done = all(row.status == "DONE" for row in self.state)
        return self.state

    def run(self, agents: Iterable[Callable], *, record_actions: bool = True) -> GameResult:
        agent_list = list(agents)
        if len(agent_list) != 2:
            raise ValueError("exactly two agents are required")
        action_tapes: tuple[list[dict[str, Any]], list[dict[str, Any]]] = ([], [])
        shops_previous: list[str] = []
        shop_history: list[dict[str, Any]] = []
        cash_history: tuple[list[float], list[float]] = ([3000.0], [3000.0])
        while not self.done:
            actions = [
                agent(self.agent_observation(player), self.configuration)
                for player, agent in enumerate(agent_list)
            ]
            if record_actions:
                for player in range(2):
                    action_tapes[player].append(_copy_wire(actions[player]))
            self.step(actions)
            public = self.state[0].observation
            for player in range(2):
                cash_history[player].append(float(public.farms[player]["money"]))
            shops_now = list(public.town.get("unlocked_shops", []))
            if shops_now != shops_previous:
                shop_history.append(
                    {
                        "step": self.step_index,
                        "day": int(public.day),
                        "unlocked_shops": shops_now,
                        "new_shops": shops_now[len(shops_previous) :],
                    }
                )
                shops_previous = shops_now
        return GameResult(
            rewards=(float(self.state[0].reward), float(self.state[1].reward)),
            statuses=(str(self.state[0].status), str(self.state[1].status)),
            steps=self.step_index,
            final_state=copy.deepcopy(self.state),
            actions=action_tapes,
            shop_history=shop_history,
            cash_history=cash_history,
        )


def pass_agent(_observation: Any, _configuration: Any = None) -> dict[str, Any]:
    return {"farmer": ["PASS"], "hands": [], "market": []}
