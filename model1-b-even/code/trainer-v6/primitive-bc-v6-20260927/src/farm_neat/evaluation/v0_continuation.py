"""Exact data-only continuation for mutable state owned by a V0Agent.

Policy/RNG, observation/source binding and diagnostic records are deliberately
outside this object.  Evaluators and search tools wrap this one memory codec
with their own provenance instead of independently defining V0Agent state.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from farm_neat.economy.orders import FullSeasonPlan
from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.epochs import V0Agent
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import LandCommitment, State


SCHEMA = "farm-v0-agent-continuation-v2"
FIELDS = frozenset({"schema", "land_commitment", "plan", "allocation", "started", "hold_releases", "controller",
                    "game_id", "labor_factor", "hands_mode", "belief"})


@dataclass(frozen=True)
class V0Continuation:
    """Closed JSON-safe V0Agent memory, excluding policy and diagnostics.

    Allocation, started and HOLD rows preserve the live dict insertion order.
    Identity is an exact ordered continuation, not a canonical quotient of equal
    mappings: equal mappings in different orders may have different wire hashes.
    Restore retains the supplied order and still rejects lossy/duplicate state.
    Policy/RNG and source/observation binding remain the caller's responsibility.
    """

    _memory: dict

    @classmethod
    def capture(cls, agent: V0Agent) -> "V0Continuation":
        if not isinstance(agent, V0Agent):
            raise TypeError("V0Agent continuation required")
        return cls.from_dict({
            "schema": SCHEMA,
            "land_commitment": None if agent.land_commitment is None else agent.land_commitment.payload(),
            "plan": agent.plan.canonical(),
            "allocation": [[list(key), None if tile is None else list(tile)]
                           for key, tile in agent.allocation.items()],
            "started": [[list(tile), list(key)] for tile, key in agent.started.items()],
            "hold_releases": [[product, list(overlay), release]
                              for product, (overlay, release) in agent.hold_releases.items()],
            "controller": None if agent.controller is None else agent.controller.snapshot(),
            "game_id": agent.game_id,
            "labor_factor": agent.labor_factor,
            "hands_mode": agent.hands_mode,
            "belief": copy.deepcopy(agent.belief),
        })

    @classmethod
    def from_dict(cls, memory: dict) -> "V0Continuation":
        if not isinstance(memory, dict) or set(memory) != FIELDS:
            raise ValueError("closed V0Agent continuation required")
        if memory["schema"] != SCHEMA:
            raise ValueError("explicit versioned V0Agent continuation required; legacy state is not inferred")
        plan = Plan.from_canonical(memory["plan"])
        if memory["land_commitment"] is None:
            if plan.land is not None:
                raise ValueError("retained LAND requires its original commitment; no silent rebind")
        else:
            LandCommitment.from_payload(memory["land_commitment"]).validate_plan(plan)
        return cls(copy.deepcopy(memory))

    def to_dict(self) -> dict:
        return copy.deepcopy(self._memory)

    def restore(self, policy, *, agent_cls=V0Agent, observation: dict | None = None) -> V0Agent:
        if not isinstance(agent_cls, type) or not issubclass(agent_cls, V0Agent):
            raise TypeError("V0Agent subclass required")
        memory = type(self).from_dict(self.to_dict()).to_dict()
        commitment = None
        if memory["land_commitment"] is not None:
            if observation is None:
                raise ValueError("LAND restoration requires the actual observation; no silent inference")
            commitment = LandCommitment.from_payload(memory["land_commitment"])
            commitment.validate_observation(State(observation))
        factor = memory["labor_factor"]
        if type(factor) not in (int, float) or not math.isfinite(factor) or factor <= 0:
            raise ValueError("positive finite labor factor required")
        if not isinstance(memory["game_id"], str) or not memory["game_id"].strip():
            raise ValueError("nonempty game identity required")
        if not isinstance(memory["hands_mode"], str) or not memory["hands_mode"]:
            raise ValueError("explicit hands mode required")

        plan = Plan.from_canonical(memory["plan"])
        if plan.canonical() != memory["plan"]:
            raise ValueError("noncanonical Plan continuation")
        agent = agent_cls(policy, plan=plan, labor_factor=factor, hands_mode=memory["hands_mode"],
                          belief=copy.deepcopy(memory["belief"]), game_id=memory["game_id"])
        agent.land_commitment = commitment

        def pair(value, *, instance=False):
            if not isinstance(value, list) or len(value) != 2 or any(type(v) is not int for v in value):
                raise ValueError("explicit continuation pair required")
            if instance:
                if not (0 <= value[0] < V.N_MODULES and 0 <= value[1] < 10000):
                    raise ValueError("invalid continuation instance")
            elif any(not 0 <= v < 10 for v in value):
                raise ValueError("invalid continuation tile")
            return tuple(value)

        allocation = {}
        if not isinstance(memory["allocation"], list):
            raise ValueError("allocation rows required")
        for row in memory["allocation"]:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("allocation row required")
            key = pair(row[0], instance=True)
            if key in allocation:
                raise ValueError("duplicate allocation continuation row")
            allocation[key] = None if row[1] is None else pair(row[1])
        if any(key not in plan.instances() for key in allocation):
            raise ValueError("allocation continuation is not in Plan")
        agent.allocation = allocation

        started = {}
        if not isinstance(memory["started"], list):
            raise ValueError("started rows required")
        for row in memory["started"]:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("started row required")
            tile, key = pair(row[0]), pair(row[1], instance=True)
            if tile in started:
                raise ValueError("duplicate started continuation row")
            started[tile] = key
        agent.started = started

        holds = {}
        if not isinstance(memory["hold_releases"], list):
            raise ValueError("HOLD rows required")
        for row in memory["hold_releases"]:
            if (not isinstance(row, list) or len(row) != 3 or type(row[0]) is not int
                    or not 0 <= row[0] < len(V.PRODUCTS) or not isinstance(row[1], list)
                    or len(row[1]) != 2 or any(type(v) is not int for v in row[1])
                    or type(row[2]) is not int or not 0 <= row[2] <= V.LAST_STEP):
                raise ValueError("invalid HOLD continuation row")
            product, overlay, release = row
            primitive, horizon = overlay
            if (product in holds or not 0 <= primitive < len(V.PRIMITIVES)
                    or not 0 <= horizon < len(V.HORIZONS)):
                if product in holds:
                    raise ValueError("duplicate HOLD continuation row")
                raise ValueError("invalid HOLD continuation row")
            overlay_tuple = (primitive, horizon)
            if V.PRIMITIVES[primitive] != "HOLD" or tuple(plan.overlay(product)[:2]) != overlay_tuple:
                raise ValueError("HOLD continuation does not match Plan")
            holds[product] = (overlay_tuple, release)
        agent.hold_releases = holds

        if memory["controller"] is not None:
            agent.controller = FullSeasonPlan.restore(copy.deepcopy(memory["controller"]))
            if agent.controller.game_id != agent.game_id:
                raise ValueError("controller/game continuation identity mismatch")
        if V0Continuation.capture(agent).to_dict() != memory:
            raise ValueError("V0Agent continuation did not round trip exactly")
        return agent
