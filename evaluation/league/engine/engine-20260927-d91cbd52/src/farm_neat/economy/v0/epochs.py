"""Epoch controller: a policy edits the plan at every dawn; the compiler and executor play.

The controller is a game agent for one seat. At each dawn it builds the public state, the
masks and the projections, asks the policy for an edit set (canonical, at most 16 keys,
STOP-terminated), applies it, compiles the plan and publishes the program through the
executor's morning publication clock; between dawns the executor dispatches. Policies here
are the reference ones: keep, random legal, a fixed trajectory and a target plan.
"""
from __future__ import annotations

import copy
import random
import time
from collections import Counter

from farm_neat.economy.orders import FullSeasonPlan
from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.compiler import allocate_sites, compile_epoch, unplaced, window
from farm_neat.economy.v0.plan import Plan, canonical_edit_set, edit_trajectory
from farm_neat.economy.v0.projection import LandCommitment, Masks, Projection, State

DAWNS = tuple(range(0, V.LAST_STEP + 1, 24))


class KeepPolicy:
    def __call__(self, state, plan, masks):
        return [V.STOP_ID]


class RandomPolicy:
    """Uniform over legal actions, one key per edit, stops with probability `stop` per edit."""

    def __init__(self, seed: int = 0, stop: float = 0.25, max_edits: int = 8):
        self.rng, self.stop, self.max_edits = random.Random(seed), stop, max_edits

    def __call__(self, state, plan, masks):
        edits, used = [], set()
        legal = masks.legal_actions()
        for _ in range(self.max_edits):
            if self.rng.random() < self.stop:
                break
            choices = [a for a in legal if a != V.STOP_ID and V.plan_key(a) not in used]
            if not choices:
                break
            a = self.rng.choice(choices)
            edits.append(a)
            used.add(V.plan_key(a))
        return edits + [V.STOP_ID]


class TrajectoryPolicy:
    """A fixed schedule {dawn step: [edits]}; every other epoch keeps the plan."""

    def __init__(self, schedule: dict[int, list[int]]):
        self.schedule = {int(k): list(v) for k, v in schedule.items()}

    def __call__(self, state, plan, masks):
        return self.schedule.get(state.step, []) + [V.STOP_ID]


class TargetPolicy:
    """Each epoch, the canonical edits that move the plan toward a target plan (or `target(state)`)."""

    def __init__(self, target):
        self.target = target
        self.residuals = {}

    def __call__(self, state, plan, masks):
        target = self.target(state) if callable(self.target) else self.target
        edits, _reached, residual = edit_trajectory(target, plan, masks.resolver)
        masked = [V.describe(a) for a in edits if a != V.STOP_ID and not masks.legal(a)]
        if masked:
            residual["__masked__"] = masked
        if residual:
            self.residuals[state.step] = residual
        return [a for a in edits if a == V.STOP_ID or masks.legal(a)]


class V0Agent:
    """One seat: plan, allocation, publication and per-epoch records."""

    def __init__(self, policy, *, labor_factor: float = 1.0, hands_mode: str = "daily", belief=None,
                 plan: Plan | None = None, game_id: str = "v0"):
        self.policy, self.labor_factor, self.hands_mode, self.belief = policy, labor_factor, hands_mode, belief
        self.plan = plan.copy() if plan is not None else Plan()
        self.allocation = {}
        self.started = {}  # tile -> key of the instance whose stage has started there
        # product index -> ((primitive, horizon), release step): the event a HOLD was set for.
        # Compiler memory, not plan state and not an action: STOP and unrelated edits keep it,
        # a replaced overlay rebinds, CLEAR or any other primitive drops it.
        self.hold_releases = {}
        # One explicit authorization, retained after fulfillment until LAND changes.
        self.land_commitment: LandCommitment | None = None
        self.controller = None
        self.game_id = game_id
        self.records = []

    def projection_kw(self):
        return {"labor_factor": self.labor_factor, "belief": self.belief, "hold_releases": dict(self.hold_releases)}

    def actor_state(self, observation, *, initialize: bool = True):
        """Observe own LAND ownership and expose only its closed commitment to phi.

        Initial Plan authorization is bound at the first observation, not during
        restoration. Snapshot callers pass initialize=False to forbid missing
        history reconstruction; they may reconcile a newly observed acquisition.
        """
        state = State(observation)
        kept = self.land_commitment
        if kept is None and self.plan.land is not None:
            if not initialize or (self.controller is not None and self.controller.committed_step is not None):
                raise ValueError("retained LAND requires its original commitment; no silent rebind")
            kept = LandCommitment.bind(state, self.plan)
        elif kept is not None:
            kept.validate_plan(self.plan)
            kept = kept.observe(state)
        self.land_commitment = kept
        state.land_commitment = None if kept is None else kept.payload()
        return state

    def bind_holds(self, state, before: Plan, after: Plan):
        for product in range(len(V.PRODUCTS)):
            primitive, horizon, rank = after.overlay(product)
            if V.PRIMITIVES[primitive] != "HOLD":
                self.hold_releases.pop(product, None)
                continue
            kept = self.hold_releases.get(product)
            if kept is not None and kept[0] == (primitive, horizon) and before.overlay(product) == (primitive, horizon, rank):
                continue
            h = state.horizon_step(product, horizon)
            self.hold_releases[product] = ((primitive, horizon), state.step if h is None else h)

    def epoch(self, observation):
        state = self.actor_state(observation)
        masks = Masks(state, self.plan, **self.projection_kw())
        requested = list(self.policy(state, self.plan, masks))
        edits = canonical_edit_set(requested)
        before = self.plan
        prefix = before.copy()
        # Support and resolver belong to the preceding canonical prefix, not dawn
        # entry. Validate locally so a refused later edit cannot commit a partial plan.
        for action in edits:
            if action == V.STOP_ID:
                break
            masks = Masks(state, prefix, **self.projection_kw())
            if not masks.legal(action):
                raise ValueError("policy emitted masked actions: " + V.describe(action))
            prefix = prefix.apply(action, masks.resolver)
        commitment = self.land_commitment
        if before.land != prefix.land:
            commitment = LandCommitment.bind(state, prefix)
        self.plan, self.land_commitment = prefix, commitment
        state.land_commitment = None if commitment is None else commitment.payload()
        self.bind_holds(state, before, self.plan)
        proj = Projection(state, self.plan, **self.projection_kw())
        previous = dict(self.allocation)
        self.allocation = allocate_sites(self.plan, proj.land_steps, previous, owned=state.quadrants, now=state.step)
        # Every tile whose published stage no longer belongs to the same instance is retired
        # before the new stages; started instances never move, so no asset is destroyed.
        dropped = sorted({tile for key, tile in previous.items() if self.allocation.get(key) != tile})
        for key, tile in self.allocation.items():
            if window(key[0], self.plan.regime(key[0]))[0] <= state.step:
                self.started[tile] = key
        replace_tiles = {tile for key, tile in self.allocation.items() if self.started.get(tile, key) != key}
        program = compile_epoch(state, proj, self.allocation, hands_mode=self.hands_mode, fresh=state.step == 0,
                                dropped=dropped, replace_tiles=replace_tiles)
        proposal = self.controller.propose(observation, program)
        self.controller = self.controller.publish(observation, proposal)
        self.records.append({"step": state.step, "edits": [V.describe(a) for a in edits], "plan": self.plan.digest(),
                             "changed": before != self.plan, "reserve": proj.reserve, "binding": proj.binding,
                             "hands": proj.hands[state.day:], "land_steps": dict(proj.land_steps),
                             "land_commitment": None if commitment is None else commitment.payload(),
                             "commands": len(program), "rejected": len(proposal["rejected"]),
                             "unplaced": len(unplaced(self.allocation)),
                             "hold_releases": {V.PRODUCTS[p]: step for p, (_, step) in sorted(self.hold_releases.items())}})

    def __call__(self, observation, configuration=None):
        if self.controller is None:
            if observation["step"] != 0:
                raise ValueError("the V0 agent starts at the opening")
            self.controller = FullSeasonPlan(self.game_id, publication_steps=DAWNS)
        self.actor_state(observation)
        self.controller.observe(observation, return_snapshot=False)
        if observation["step"] % 24 == 0:
            self.epoch(observation)
        return self.controller(observation, configuration)


def play(spec: dict, agent, seat: int, *, opponent_kind: str | None = None, capture_events=None) -> dict:
    """One native game of `agent` in `seat` against the scenario's frozen opponent; receipts by product."""
    from farm_neat.evaluation import realization as r
    from farm_neat.evaluation.scenarios import ScenarioEnvironment, opponent
    env = ScenarioEnvironment(spec)
    other, _ = opponent(opponent_kind or spec["opponent"])
    sold, receipts, spend, acts = Counter(), Counter(), Counter(), Counter()
    cash_min = [1e9, 1e9]
    t0 = time.monotonic()
    with r.market_tap(env, seat) as tap:
        for step in range(V.LAST_STEP + 1):
            obs = env.agent_observation(seat)
            joint = [None, None]
            joint[seat] = agent(obs, env.configuration)
            joint[1 - seat] = other(env.agent_observation(1 - seat), env.configuration)
            tap["commits"], tap["pre_market"], tap["unit_effects"] = {}, None, []
            env.step(joint)
            after = env.agent_observation(seat)
            if capture_events is not None:
                capture_events(step, obs, joint[seat], after, tap)
            for e in tap["unit_effects"]:
                acts[e["engine_received_action"][0]] += 1
            for s_ in (0, 1):
                cash_min[s_] = min(cash_min[s_], after["farms"][s_]["money"])
            for (op, item), row in tap["commits"].items():
                if op == "SELL":
                    sold[item] += row["units"]; receipts[item] += round(row["cash_delta"])
                else:
                    spend[f"{op}:{item}"] += -round(row["cash_delta"])
    final = env.agent_observation(seat)
    cash = [f["money"] for f in final["farms"]]
    margin = cash[seat] - cash[1 - seat]
    return {"seat": seat, "cash": cash, "margin": margin, "wdl": "WIN" if margin > 0 else "LOSS" if margin < 0 else "DRAW",
            "sold": dict(sold), "receipts": dict(receipts), "receipts_total": sum(receipts.values()),
            "spend": dict(spend), "spend_total": sum(spend.values()), "actions": dict(acts),
            "unlocked": final["farms"][seat].get("unlocked_quadrants"), "cash_min": cash_min,
            "seconds": round(time.monotonic() - t0, 1)}


def run(spec: dict, seat: int, policy, **agent_kw) -> tuple[dict, V0Agent]:
    agent = V0Agent(policy, **agent_kw)
    result = play(spec, agent, seat)
    result["epochs"] = copy.deepcopy(agent.records)
    result["plan"] = agent.plan.canonical()
    return result, agent
