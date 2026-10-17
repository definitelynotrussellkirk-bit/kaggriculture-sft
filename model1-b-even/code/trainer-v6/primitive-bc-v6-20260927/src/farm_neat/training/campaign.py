"""Fast-profile full-economic self-play campaign: parallel collectors, lean witness, resident learner.

Owner: full-economic runtime/throughput maintainer. Execution profile:
``prototype/full_economic_fast_profile.py`` (NEW numerics, declared there). This
command reuses the exact engine, EconomicOrders executor, entity observation,
action grammar, trace records, terminal-WDL PPO objective, AdamW8bit state,
CPU/CUDA RNG and inherited global schedule of the maintained learner. It is not
the historical B2/B32/streaming numerical profile and never relabels their receipts.

Round r: K collector subprocesses (16 fresh worlds / 32 private lanes each) sample
with the current child; a spawned CPU pool reconstructs every world from seed and
recorded programs (fresh engine, exact executor inputs, actions, terminal cash);
the resident learner gates every cohort on within-profile collect/replay parity,
runs a cross-profile canary against the frozen reference on registered cohorts,
commits accumulated PPO updates, publishes the child, and the next round collects
with it. Storage is checked before each round through RUN.py storage check.

Censored or failed worlds never receive terminal labels; a failed round leaves a
RUN-FAILURE receipt and the last committed child. No promotion, upload or strength
claim follows from any receipt here.

The opt-in opening-only experiment samples one whole-season economic program per
seat at step0. Later turns execute its persistent rules with no neural revisions.
Collection, fresh reconstruction, PPO and receipts bind that changed decision
clock; old daily data are never relabeled as opening-only experience.

--shop-revision-every 2 is a separate clock: opening, then fresh observed shops
2/4/6/8. Persistent-plan edits receive full-game GRPO credit; future auxiliary
queries are strictly after each decision, conditioned on its current state.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gzip
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""): sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from farm_neat.policy import fast_profile as fast
from farm_neat.rollout import rollout as base
from farm_neat.rollout import batched_rollout as b32
from farm_neat.training import batched_training as b32fit
from farm_neat.training import ppo as legacy
from farm_neat.rollout import batched_replay as heads
from farm_neat.policy import observation as features
from farm_neat.policy import network as policy
from farm_neat.economy import action_space as actions
from farm_neat.economy.orders import EconomicOrders, FullSeasonPlan
from farm_neat.policy import plan_observation as plan_features
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
from farm_neat.training import moe_lab as lab
from farm_neat.training import sustain as io
from farm_neat.training import streaming_campaign as streaming
from farm_neat.training.schedule import learning_rate_at, validate_settings

SCHEMA = "farm-full-economic-fast-campaign-v1"
COHORT_SCHEMA = SCHEMA + "-cohort-plan"
WORLD_SCHEMA = SCHEMA + "-world-plan"
GAME_SCHEMA = SCHEMA + "-game"
EPISODE_SCHEMA = SCHEMA + "-episode"
OPTIMIZER_SCHEMA = SCHEMA + "-optimizer"
SUPERVISED_STAGE = SCHEMA + "-observed-prefix-bc-v1"
EFFECT_STAGE = SCHEMA + "-observed-plan-effects-v1"
BATCH_MODE = "WORLD_MAJOR_SEAT_MINOR_BUCKET32_PADDING_LOCKSTEP_PCG64"
PAD_MULTIPLE = 32
STEPS, ATOL = base.STEPS, base.ATOL
bound, digest, file_sha, _sealed, _sha, _finite = base.bound, base.digest, base.file_sha, base._sealed, base._sha, base._finite
EXECUTION = {"schema": SCHEMA+"-execution", "numerics": fast.NUMERICS, "profile": fast.SCHEMA,
    "objective": "UNCHANGED_TERMINAL_MC_FULL_PROGRAM_PPO", "advantage_scope": "ROUND_COMPLETE_ROLLOUT",
    "gradient_weighting": "CALL_COUNT_OVER_ACCUMULATION_GROUP", "shuffle": "ONE_CPU_TORCH_RANDPERM_PER_ROUND",
    "admission": "FRESH_ENGINE_LEAN_WITNESS_BINARY_INPUT_SIDECAR_PLUS_WITHIN_PROFILE_PARITY_GATE_PLUS_CROSS_PROFILE_CANARY",
    "metrics": "FULL_ROUND_INITIAL_FINAL_AND_EVERY_N_UPDATES", "sampling": BATCH_MODE + "; CUDA_GRAPH_SAMPLER",
    "encoder": "FAST_PROFILE_STACKED_COHORTS_OPTIONAL_BLOCK_RECOMPUTATION", "factor_replay": "FAST_PROFILE_PADDED_POSITION_REPLAY",
    "overlap": "NEXT_ROUND_COLLECTS_WITH_LATEST_PUBLISHED_CHILD_WHILE_THIS_ROUND_IS_WITNESSED_AND_FITTED; PARITY_SCORED_BY_FROZEN_BEHAVIOR_COPY",
    "tolerances": fast.TOLERANCES, "training_admission": False}
OWN = ("src/farm_neat/training/campaign.py", "tests/contracts/test_campaign.py",
       "src/farm_neat/policy/plan_observation.py", "tests/unit/test_observation.py",
       "src/farm_neat/economy/linter.py", "tests/unit/test_linter.py",
       "src/farm_neat/policy/plan_learning.py", "tests/unit/test_plan_learning.py",
       "src/farm_neat/evaluation/shop_router.py", "src/farm_neat/evaluation/opponents.py", "src/farm_neat/training/sustain.py",
       "tests/unit/test_opponent_weighting.py", "src/farm_neat/training/plan_effects.py",
       "tests/contracts/test_plan_effects.py")
_IMPORTED_SHA = file_sha(__file__)


def sources():
    if file_sha(__file__) != _IMPORTED_SHA: raise ValueError("loaded campaign source changed")
    from farm_neat.evaluation import shop_router as router
    return {**base.sources(), **fast.identity()["source_pins"], **heads.identity()["source_pins"],
            **router.identity()["assets"],
            **{n: file_sha(ROOT/n) for n in OWN}}


def attest(pins):
    if pins != sources(): raise ValueError("fast campaign source closure changed")
    base.attest(pins)


def _equal(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(label + " mismatch")


MAX_WORLDS = 64
PLAN_LEARNING_SCHEMA = SCHEMA + "-grouped-plan-learning-v1"
PLAN_MARGIN_SCHEMA = SCHEMA + "-grouped-plan-learning-v2"
PLAN_POOL_SCHEMA = SCHEMA + "-grouped-plan-learning-v3"
PLAN_WEIGHTED_SCHEMA = SCHEMA + "-grouped-plan-learning-v4"
NEURAL_PEER = "FROZEN_ROUND_START_NEURAL_POLICY"


def validate_plan_learning(config):
    """Explicit new objective/opponent contract; no reinterpretation of PPO data.

    Frozen economic programs still use the responsive deterministic executor.
    They are not counterfactual labels or neural sampling calls. Their complete
    commands are embedded in the registration; optional provenance is descriptive.
    V3 also permits the round-start neural policy, frozen for every full game.
    Its independent opponent calls are reconstructed and resampled, never labels.
    """
    if config is None:
        return None
    required = {"schema", "group_size", "opponents", "critic_coefficient",
                "future_coefficient", "head_seed", "samples_per_stratum"}
    if isinstance(config, dict) and config.get("schema") in (PLAN_MARGIN_SCHEMA, PLAN_POOL_SCHEMA, PLAN_WEIGHTED_SCHEMA):
        required.add("reward")
    if isinstance(config, dict) and config.get("schema") == PLAN_WEIGHTED_SCHEMA:
        required.add("router_actor_weight")
    if not isinstance(config, dict) or set(config) != required or config["schema"] not in (PLAN_LEARNING_SCHEMA, PLAN_MARGIN_SCHEMA, PLAN_POOL_SCHEMA, PLAN_WEIGHTED_SCHEMA):
        raise ValueError("explicit versioned plan-learning configuration required")
    if config["schema"] == PLAN_WEIGHTED_SCHEMA:
        weight = config["router_actor_weight"]
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("explicit finite positive Router actor weight required")
    if "reward" in config:
        from farm_neat.policy.plan_learning import validate_margin_reward
        validate_margin_reward(config["reward"])
    actions._integer(config["group_size"], 2, 16, "GRPO group size")
    actions._integer(config["samples_per_stratum"], 1, 8, "future samples per stratum")
    actions._integer(config["head_seed"], 0, actions.MAX_INTEGER, "auxiliary initialization seed")
    for name in ("critic_coefficient", "future_coefficient"):
        if type(config[name]) not in (int, float) or not 0 < config[name] <= 1:
            raise ValueError("explicit positive bounded auxiliary loss coefficients required")
    if not isinstance(config["opponents"], list) or not 1 <= len(config["opponents"]) <= 32:
        raise ValueError("bounded frozen opponent pool required")
    for opponent in config["opponents"]:
        if not isinstance(opponent, dict): raise ValueError("typed frozen opponent required")
        if opponent.get("kind") == "FROZEN_ECONOMIC_PROGRAM":
            if set(opponent) != {"kind", "program", "name"} or not isinstance(opponent["name"], str) or not opponent["name"]:
                raise ValueError("named complete frozen economic program required")
            actions.validate_program(opponent["program"])
        elif opponent.get("kind") == "SHOP_ROUTER":
            from farm_neat.evaluation import shop_router as router
            _equal(opponent, {"kind": "SHOP_ROUTER", "identity": router.identity()}, "frozen ShopRouter")
        elif opponent.get("kind") == NEURAL_PEER:
            if (config["schema"] not in (PLAN_POOL_SCHEMA, PLAN_WEIGHTED_SCHEMA) or set(opponent) != {"kind", "name", "sampler_seed"}
                    or not isinstance(opponent["name"], str) or not opponent["name"]):
                raise ValueError("explicit v3 frozen round-start neural opponent required")
            actions._integer(opponent["sampler_seed"], 0, actions.MAX_INTEGER, "opponent sampler namespace")
        else:
            raise ValueError("unsupported frozen opponent kind")
    if config["schema"] == PLAN_WEIGHTED_SCHEMA:
        kinds = {p["kind"] for p in config["opponents"]}
        if "SHOP_ROUTER" not in kinds or len(kinds) < 2:
            raise ValueError("Router weighting requires Router and another frozen opponent kind")
    return copy.deepcopy(config)


def auxiliary_resume(aux, saved_execution, learning, age):
    """Join auxiliary state to ITS contract across one explicit versioned change.

    Head shapes/targets, loss coefficients, opponent pool and sampling stay fixed.
    Loading weights and Adam is mandatory across v1 -> v2 and v2 reward changes;
    this never reinitializes a critic at a new reward boundary.
    V2/V3 -> V3 may instead change only the opponent pool, keeping reward fixed.
    """
    old = validate_plan_learning(saved_execution.get("plan_learning"))
    new = validate_plan_learning(learning)
    if old is None:
        if aux is not None: raise ValueError("undeclared auxiliary checkpoint state")
        return {"kind":"INITIALIZE_AUXILIARY_HEADS_ONLY", "global_age":age}
    if (not isinstance(aux,dict) or aux.get("configuration") != old
            or type(aux.get("global_age")) is not int or aux["global_age"] != age
            or not isinstance(aux.get("heads"),dict) or not isinstance(aux.get("optimizer"),dict)):
        raise ValueError("exact continued auxiliary heads/Adam/configuration/age required")
    if old == new: return {"kind":"EXACT_AUXILIARY_CONTINUATION", "global_age":age}
    if new is not None and PLAN_WEIGHTED_SCHEMA in (old["schema"], new["schema"]):
        comparable_weight = lambda c: {k:v for k,v in c.items() if k not in ("schema", "router_actor_weight")}
        if (old["schema"] not in (PLAN_POOL_SCHEMA, PLAN_WEIGHTED_SCHEMA)
                or new["schema"] not in (PLAN_POOL_SCHEMA, PLAN_WEIGHTED_SCHEMA)
                or comparable_weight(old) != comparable_weight(new)):
            raise ValueError("actor-weight-only transition must preserve opponents, reward, auxiliaries and group size")
        return {"kind":"ROUTER_ACTOR_WEIGHT_ONLY_TRANSITION_EXACT_AUXILIARY_WEIGHTS_ADAM_AGE",
            "old_configuration_sha256":digest(old), "new_configuration_sha256":digest(new),
            "old_router_actor_weight":old.get("router_actor_weight", 1.),
            "new_router_actor_weight":new.get("router_actor_weight", 1.), "global_age":age}
    if new is not None and new["schema"] == PLAN_POOL_SCHEMA:
        comparable_pool = lambda c: {k:v for k,v in c.items() if k not in ("schema", "opponents")}
        if old["schema"] not in (PLAN_MARGIN_SCHEMA, PLAN_POOL_SCHEMA) or comparable_pool(old) != comparable_pool(new):
            raise ValueError("pool-only transition must preserve reward, auxiliary targets/coefficients and group size")
        return {"kind":"OPPONENT_POOL_ONLY_TRANSITION_EXACT_AUXILIARY_WEIGHTS_ADAM_AGE",
            "old_configuration_sha256":digest(old), "new_configuration_sha256":digest(new),
            "old_opponents":old["opponents"], "new_opponents":new["opponents"], "global_age":age}
    comparable = lambda c: {k:v for k,v in c.items() if k not in ("schema","reward")}
    if new is None or new["schema"] != PLAN_MARGIN_SCHEMA or comparable(old) != comparable(new):
        raise ValueError("only explicitly versioned reward changes may reuse auxiliary continuation")
    return {"kind":"REWARD_ONLY_TRANSITION_EXACT_AUXILIARY_WEIGHTS_ADAM_AGE",
            "old_configuration_sha256":digest(old), "new_configuration_sha256":digest(new),
            "old_reward":old.get("reward", "TERMINAL_WDL_GROUP_STD"),
            "new_reward":new["reward"], "global_age":age}


def witnessed_episode_cash(episode):
    cash = episode.get("terminal_cash")
    if not isinstance(cash,list) or len(cash) != 2:
        raise ValueError("witnessed own/opponent terminal cash required for margin credit")
    for n in cash:
        _finite(n)
        if n < 0: raise ValueError("nonnegative terminal bank cash required")
    delta = cash[0]-cash[1]
    if episode["terminal_reward"] != int(delta > 0)-int(delta < 0):
        raise ValueError("witnessed terminal cash must agree with W/D/L")
    return list(cash)


def lane_order_for(worlds, learning=None, group_offset=0):
    if learning is None: return [[w, s] for w in range(worlds) for s in (0, 1)]
    size = learning["group_size"]
    if worlds % size: raise ValueError("whole GRPO groups must stay within a collection cohort")
    return [[w, (group_offset+w//size) % 2] for w in range(worlds)]


def opponent_for(plan, index):
    learning = plan.get("plan_learning")
    if learning is None: return None
    group = plan["group_offset"]+index//learning["group_size"]
    # Both seat perspectives get each opponent before moving to the next one.
    opponent = learning["opponents"][(group//2) % len(learning["opponents"])]
    if opponent["kind"] == NEURAL_PEER:
        # Common independent random stream within K trials; future decisions
        # still respond to that trial's own observations, not a fixed action tape.
        seed = int(digest([NEURAL_PEER, opponent["sampler_seed"], group,
                           plan["seeds"][index], 1-plan["lane_order"][index][1]]), 16) % (2**63)
        if seed in plan["actor_rng_seeds"]: raise ValueError("opponent/learner sampler collision")
        return {**opponent, "model_sha256":plan["model_sha256"],
                "model_manifest_sha256":plan["model_manifest_sha256"], "rng_seed":seed,
                "decision_steps":list(decision_steps(plan)), "full_plan":True,
                "weights":"EXACT_ROUND_START_BEHAVIOR_FROZEN_UNTIL_ALL_GAMES_FINISH"}
    return opponent


def opponent_lanes(plan):
    if plan.get("plan_learning") is None: return []
    return [[w, 1-s] for w,s in plan["lane_order"] if opponent_for(plan,w)["kind"] == NEURAL_PEER]


def initial_controllers(plan, index):
    controllers = [EconomicOrders(), EconomicOrders()]
    if plan.get("full_plan"):
        seats = (plan["lane_order"][index][1],) if plan.get("plan_learning") else (0, 1)
        for seat in seats:
            controllers[seat] = FullSeasonPlan(f"{plan['cohort_id']}-world{index:03d}-seat{seat}",
                publication_steps=decision_steps(plan))
    peer = opponent_for(plan, index)
    if peer is not None and peer["kind"] == NEURAL_PEER:
        seat = 1-plan["lane_order"][index][1]
        controllers[seat] = FullSeasonPlan(f"{plan['cohort_id']}-world{index:03d}-seat{seat}",
                                          publication_steps=decision_steps(plan))
    if peer is not None and peer["kind"] == "SHOP_ROUTER":
        from farm_neat.evaluation.shop_router import ShopRouterAdapter
        controllers[1-plan["lane_order"][index][1]] = ShopRouterAdapter()
    return controllers


def pack_plan(plan, observation, controller, configuration):
    if plan.get("full_plan"):
        if not isinstance(controller, FullSeasonPlan):
            raise ValueError("full-plan learner cannot silently use the legacy executor")
        return plan_features.pack(observation, controller, configuration,
            expected_identity=plan["plan_adapter"])
    return features.pack(observation, controller, configuration)


def joint_action(controllers, observations, configuration):
    return [(c.act(o, configuration) if hasattr(c, "act") else c(o, configuration))
            for c, o in zip(controllers, observations, strict=True)]


def publish_peer(plan, index, controllers):
    peer = opponent_for(plan, index)
    if peer is None or peer["kind"] != "FROZEN_ECONOMIC_PROGRAM": return []
    seat = 1-plan["lane_order"][index][1]
    controllers[seat], failures = apply_opening(controllers[seat], peer["program"], step=0)
    return [{"step": 0, "seat": seat, **f} for f in failures]


def future_sampler(plan, index):
    if plan.get("plan_learning") is None: return None
    from farm_neat.policy.plan_learning import BoundarySampler
    if plan.get("shop_revision_every"):
        return {step: BoundarySampler(seed=int(digest([plan["actor_rng_seeds"][index], step]), 16),
            per_stratum=plan["plan_learning"]["samples_per_stratum"],
            tolerances={"shed_free_units": 1.0}) for step in decision_steps(plan)}
    return BoundarySampler(seed=plan["actor_rng_seeds"][index],
        per_stratum=plan["plan_learning"]["samples_per_stratum"],
        tolerances={"shed_free_units": 1.0})


def observe_future(sampler, env, controller, seat):
    if sampler is None: return
    from farm_neat.policy.plan_learning import snapshot_targets
    values = snapshot_targets(env.agent_observation(seat), controller.last_report, env.configuration)
    if isinstance(sampler, dict):
        for anchor, reservoir in sampler.items():
            if env.step_index > anchor:
                reservoir.observe(step=env.step_index-anchor, values=values,
                                  slack={"shed_free_units": values["shed_free_units"]})
    else:
        sampler.observe(step=env.step_index, values=values, slack={"shed_free_units": values["shed_free_units"]})


def finish_future(sampler):
    if isinstance(sampler, dict):
        return {"schema": SCHEMA+"-decision-futures-v1",
                "by_step": {str(step): reservoir.finish(horizon=719-step) for step, reservoir in sampler.items()}}
    return sampler.finish() if sampler is not None else None


def shop_clock(opening_only, every):
    if type(opening_only) is not bool or type(every) is not int or every not in (0, 2):
        raise ValueError("explicit boolean opening_only and shop_revision_every 0 or2 required")
    if opening_only and every:
        raise ValueError("opening-only and shop revisions are mutually exclusive")
    return every


def decision_steps(plan):
    """The registered economic clock; never infer it from missing tape records."""
    opening = plan["opening_only"]
    if type(opening) is not bool:
        raise ValueError("explicit boolean opening_only required")
    every = shop_clock(opening, plan.get("shop_revision_every", 0))
    return (0, 144, 288, 432, 576) if every else (0,) if opening else STEPS


def check_decision_observation(plan, step, observation, configuration):
    """Gate on actually visible shop instances, not on future shop identities.

    The exact default engine has a synchronous spawn clock. Reject a different
    clock/missing shop instead of silently training a scheduled fictitious event.
    Duplicate shop types count as separate arrivals. No hidden seed is inspected.
    """
    if not plan.get("shop_revision_every"):
        return
    if (step not in decision_steps(plan) or observation.get("step") != step
            or observation.get("day") != step//24 or observation.get("hour") != 0
            or configuration.get("townShopUnlockInterval", 3) != 3
            or configuration.get("turnsPerDay", 24) != 24):
        raise ValueError("registered shop-revision clock and fresh observation required")
    from farm_neat.game.constants import SHOPS
    shops = observation.get("town", {}).get("unlocked_shops")
    if (not isinstance(shops, list) or len(shops) != step//72
            or any(shop not in SHOPS for shop in shops)):
        raise ValueError("actual visible shop arrival required before plan revision")


def execution_for(opening_only=False, plan_learning=None, shop_revision_every=0, full_plan=False):
    if type(opening_only) is not bool:
        raise ValueError("explicit boolean opening_only required")
    every = shop_clock(opening_only, shop_revision_every)
    if type(full_plan) is not bool or (full_plan and not (opening_only or every)):
        raise ValueError("explicit full-plan mode requires a registered opening/morning clock")
    result = copy.deepcopy(EXECUTION)
    if opening_only:
        result.update(economic_clock={"schema": SCHEMA+"-opening-plan-v2", "decision_steps": [0],
            "publication": "ATOMIC_OPENING_ONLY_NO_LATER_ECONOMIC_REVISION", "physical_steps": 719,
            "conflicts": "REFUSE_ONLY_CONFLICTING_COMMAND_KEEP_VALID_PREFIX_AND_SUFFIX; NO_REPLACEMENT_OR_REROLL",
            "credit": "ONE_SAMPLED_PROGRAM_PER_SEAT_TERMINAL_RETURN", "draft_search": False},
            objective="TERMINAL_MC_OPENING_PROGRAM_PPO", overlap="DISABLED_FRESH_CHILD_EACH_ROUND")
    if plan_learning is not None:
        if not (opening_only or every): raise ValueError("grouped plan learning requires opening-only or shop-revision execution")
        result.update(plan_learning=validate_plan_learning(plan_learning),
            objective="GROUPED_TERMINAL_WDL_GRPO_PLUS_SUPERVISED_PLAN_CRITIC_AND_FUTURE_HUBER",
            advantage_scope="SAME_INITIAL_WORLD_SEAT_FROZEN_OPPONENT_K_INDEPENDENT_PROGRAMS",
            gradient_weighting="EQUAL_LEARNER_PROGRAMS; OPPONENTS_NEVER_TRAINING_ROWS",
            critic="PLAN_CONDITIONED_SUPERVISED_WDL; NOT_GRPO_BASELINE_OR_REWARD",
            reward_shaping="NONE; CAPITAL_SHAPING_REMAINS_A_SEPARATE_EXPERIMENT")
        if "reward" in plan_learning:
            result.update(objective="GROUPED_ASYMMETRIC_WDL_MARGIN_GRPO_AND_SUPERVISED_AUXILIARIES",
                reward_shaping=copy.deepcopy(plan_learning["reward"]),
                advantage_normalization="GROUP_MEAN_ONLY; BONUS_WEIGHT_NOT_CANCELLED_BY_GROUP_STD")
    if every:
        if plan_learning is None:
            raise ValueError("shop revisions require explicit grouped plan learning")
        result.update(economic_clock={"schema": SCHEMA+"-shop-revisions-v1",
            "decision_steps": [0,144,288,432,576], "shop_revision_every": every,
            "publication": "OPENING_THEN_VISIBLE_SHOP_INSTANCES_2_4_6_8",
            "physical_steps": 719, "persistent_edits": True, "draft_search": False,
            "conflicts": "REFUSE_CONFLICTING_COMMAND_KEEP_VALID_PREFIX_SUFFIX_AND_EXISTING_COMMITMENTS"},
            overlap="DISABLED_FRESH_CHILD_EACH_ROUND",
            advantage_scope="SAME_INITIAL_WORLD_SEAT_FROZEN_OPPONENT_K_INDEPENDENT_FIVE_DECISION_TRAJECTORIES",
            gradient_weighting="EQUAL_DECISIONS_PER_COMPLETE_GAME; TERMINAL_GROUP_ADVANTAGE_TO_ALL_FIVE_DECISIONS",
            future_queries="STRICTLY_AFTER_EACH_DECISION; RELATIVE_PHYSICAL_HORIZON; ACTUAL_RESPONSIVE_CONTINUATION")
    if full_plan:
        result.update(full_plan={"schema": FullSeasonPlan.SCHEMA,
            "sampling_support": "REMAINING_WINDOW_V2; UNTIL_GE_ACTUAL_STEP; PAST_STARTS_ALLOWED",
            "publication": "FULL_FUTURE_PLUS_IMMUTABLE_ELAPSED_TERMS_AND_REALIZATION",
            "adapter": plan_features.identity(), "seed_plan": None,
            "refusals": "CONFLICTING_COMMAND_OR_PAST_ONLY_WINDOW; RETAIN_OTHER_REQUESTS",
            "log_probability": "ORIGINAL_SAMPLED_EDIT_PROGRAM; COPIED_HISTORY_HAS_NO_NEW_CHOICE_LOSS"})
    if plan_learning is not None and any(p["kind"] == NEURAL_PEER for p in plan_learning["opponents"]):
        if not full_plan or every != 2:
            raise ValueError("neural opponent requires full-plan five-decision shop clock")
        result["neural_opponent"] = {"kind":NEURAL_PEER, "model":"EXACT_ROUND_START_BEHAVIOR",
            "sampling":"OWN_OBSERVATIONS; GROUP_COMMON_DISJOINT_PCG64_STREAM",
            "admission":"FRESH_ENGINE_INPUTS_AND_ALL_FIVE_NEURAL_RESAMPLES_BEFORE_OPTIMIZER",
            "gradient":"LEARNER_ONLY; OPPONENT_DECISIONS_NEVER_TRAINING_ROWS"}
    if plan_learning is not None and plan_learning["schema"] == PLAN_WEIGHTED_SCHEMA:
        result.update(objective="GROUPED_ASYMMETRIC_WDL_MARGIN_GRPO_WITH_ROUTER_ACTOR_WEIGHTING_AND_SUPERVISED_AUXILIARIES",
            gradient_weighting="ACTOR_WEIGHTED_MEAN_OVER_COMPLETE_ACCUMULATION_GROUP; ENTROPY_AND_AUXILIARIES_UNWEIGHTED_DECISION_MEANS",
            actor_weighting={"schema":"farm-router-actor-weighting-v1",
                "router_weight":plan_learning["router_actor_weight"], "other_weight":1.,
                "identity":"OPPONENT_KIND_FROM_WITNESSED_REGISTERED_WORLD; NOT_AN_ACTOR_INPUT",
                "all_ones":"EXACT_LEGACY_BACKWARD_PATH", "opponent_rows":"NEVER_TRAINING_ROWS"})
    return result


def apply_opening(controller, program, *, step):
    """Publish once: retain valid edits and explicitly refuse conflicting commands.

    The original sampled program/logp remains the learning action. No reroll,
    inferred profitable substitute, or later-day repair of economic intent.
    Syntax is checked for the whole program first. Each EconomicOrders.apply is
    atomic; only its documented tile-conflict refusal is recoverable. Unexpected
    execution errors fail loudly, without publishing the staged controller.
    """
    if type(step) is not int or step != 0:
        raise ValueError("opening-plan publication requires step0")
    return _apply_plan_edits(controller, program)


def publish_program(plan, controller, program, *, step, observation=None):
    if plan.get("full_plan"):
        if not isinstance(controller, FullSeasonPlan) or observation is None or observation.get("step") != step:
            raise ValueError("full-plan publication requires actual observed state and controller")
        proposal = controller.propose(observation, program)
        return controller.publish(observation, proposal), proposal["rejected"]
    if plan.get("shop_revision_every"):
        if type(step) is not int or step not in decision_steps(plan):
            raise ValueError("only opening and registered shop-revision steps can publish")
        return _apply_plan_edits(controller, program)
    return apply_opening(controller, program, step=step)


def _apply_plan_edits(controller, program):
    """Atomic publication over existing commitments; STOP never clears a plan."""
    commands = actions.validate_program(program)
    staged, rejected = copy.deepcopy(controller), []
    for index, command in enumerate(commands):
        try:
            staged.apply(command)
        except ValueError as exc:
            if not str(exc).startswith("CONFLICTING_TILE_ORDER:"):
                raise
            rejected.append({"command": copy.deepcopy(command), "command_index": index,
                "reason": str(exc), "transaction": "COMMAND_REFUSED_REMAINDER_PRESERVED"})
    return staged, rejected


def _registration(seeds, rng, budget, cohort_id, environment, learning=None):
    if (not isinstance(seeds, list) or not 2 <= len(seeds) <= MAX_WORLDS
            or any(type(v) is not int or not 0 <= v <= actions.MAX_INTEGER for v in seeds)
            or (learning is None and len(set(seeds)) != len(seeds))):
        raise ValueError(f"2..{MAX_WORLDS} distinct explicit physical world seeds required")
    if learning is not None:
        size = learning["group_size"]
        if (len(seeds) % size or len(set(seeds)) != len(seeds)//size
                or any(len(set(seeds[i:i+size])) != 1 for i in range(0, len(seeds), size))):
            raise ValueError("distinct complete groups of repeated initial world seeds required")
    if (not isinstance(rng, list) or len(rng) != (2 if learning is None else 1)*len(seeds) or any(type(v) is not int or not 0 <= v < 2**63 for v in rng)
            or len(set(rng)) != len(rng)):
        raise ValueError("one distinct explicit sampler seed per ordered lane required")
    actions._integer(budget, 2, actions.MAX_COMMANDS, "command budget")
    if not isinstance(cohort_id, str) or not cohort_id or len(cohort_id) > 200: raise ValueError("explicit bounded cohort identity required")
    if (not isinstance(environment, dict) or environment.get("dtype") != "FLOAT32" or environment.get("tf32") is not False
            or environment.get("autocast") is not False): raise ValueError("explicit FP32 no-TF32/no-autocast profile required")


def environment_profile():
    profile = lab.device_profile()
    return {**profile, "pad_multiple": PAD_MULTIPLE, "numerics": fast.NUMERICS}


# ---------------------------------------------------------------- registration
def make_cohort_plan(manifest, model_sha256, *, seeds, actor_rng_seeds, command_budget, cohort_id, environment, source_pins=None, opening_only=False,
                     plan_learning=None, group_offset=0, shop_revision_every=0, full_plan=False):
    _sealed(manifest)
    _equal(manifest["schema"], policy.SCHEMA, "model schema")
    _equal(manifest["architecture"], json.loads(json.dumps(policy.architecture())), "full model architecture")
    model_transition = policy.check_model_sources(manifest, allow_remaining_window_transition=True)
    learning = validate_plan_learning(plan_learning)
    actions._integer(group_offset, 0, actions.MAX_INTEGER, "group offset")
    _registration(seeds, actor_rng_seeds, command_budget, cohort_id, {**environment, "dtype": "FLOAT32", "tf32": False, "autocast": False}, learning)
    pins = sources() if source_pins is None else source_pins
    attest(pins)
    execution = execution_for(opening_only, learning, shop_revision_every, full_plan)
    clock = {"opening_only": opening_only, **({"shop_revision_every": shop_revision_every} if shop_revision_every else {})}
    if full_plan: clock.update(full_plan=True, plan_adapter=plan_features.identity())
    lanes = lane_order_for(len(seeds), learning, group_offset)
    plan = bound({"schema": COHORT_SCHEMA, "mode": "FRESH_SAMPLED_FULL_ECONOMIC_COHORT_FAST", "cohort_id": cohort_id,
        "seeds": seeds, "actor_rng_seeds": actor_rng_seeds, "worlds": len(seeds), "lanes": len(lanes),
        "lane_order": lanes, "plan_learning": learning, "group_offset": group_offset,
        "command_budget": command_budget, "temperature": 1.0,
        "model_sha256": _sha(model_sha256), "model_manifest_sha256": manifest["content_sha256"],
        **({"model_source_transition": model_transition} if model_transition else {}),
        "model_optimizer_age": manifest["optimizer_steps"], "rollout_iteration": manifest["rollout_iteration"]+1,
        "architecture_sha256": manifest["architecture"]["content_sha256"], "numerics": fast.NUMERICS,
        "profile_sha256": fast.identity()["content_sha256"], "source_pins": pins, "source_sha256": digest(pins),
        "environment": environment, **clock, "execution": execution,
        "decision_steps": list(decision_steps(clock)), "physical_steps_per_world": 719,
        "neural_batch": BATCH_MODE, "pad_multiple": PAD_MULTIPLE, "cublas_workspace_config": ":4096:8",
        "terminal_training_admission": False})
    return validate_cohort_plan(plan)


def validate_cohort_plan(plan):
    _sealed(plan)
    learning = validate_plan_learning(plan.get("plan_learning"))
    offset = plan.get("group_offset", 0)
    actions._integer(offset, 0, actions.MAX_INTEGER, "group offset")
    lanes = lane_order_for(len(plan["seeds"]), learning, offset)
    _registration(plan["seeds"], plan["actor_rng_seeds"], plan["command_budget"], plan["cohort_id"],
                  {**plan["environment"], "dtype": "FLOAT32", "tf32": False, "autocast": False}, learning)
    for key, expected in (("schema", COHORT_SCHEMA), ("worlds", len(plan["seeds"])), ("lanes", len(lanes)),
            ("neural_batch", BATCH_MODE), ("pad_multiple", PAD_MULTIPLE), ("temperature", 1.0),
            ("lane_order", lanes),
            ("decision_steps", list(decision_steps(plan))), ("execution", execution_for(plan["opening_only"], learning, plan.get("shop_revision_every", 0), plan.get("full_plan", False))),
            ("physical_steps_per_world", 719), ("numerics", fast.NUMERICS),
            ("profile_sha256", fast.identity()["content_sha256"]), ("source_sha256", digest(plan["source_pins"])),
            ("architecture_sha256", policy.architecture()["content_sha256"]), ("terminal_training_admission", False)):
        _equal(plan[key], expected, "cohort " + key)
    if plan.get("full_plan"):
        _equal(plan.get("plan_adapter"), plan_features.identity(), "full-plan input adapter")
    attest(plan["source_pins"])
    return plan


def world_plan(plan, index):
    if type(index) is not int or not 0 <= index < plan["worlds"]: raise ValueError("registered world index required")
    common = ("command_budget", "temperature", "model_sha256", "model_manifest_sha256", "model_optimizer_age",
        "rollout_iteration", "architecture_sha256", "numerics", "profile_sha256", "source_pins", "source_sha256",
        "environment", "decision_steps", "opening_only", "execution", "neural_batch", "pad_multiple", "terminal_training_admission")
    learning = plan.get("plan_learning")
    extra = {} if learning is None else {"plan_learning": learning, "learner_seat": plan["lane_order"][index][1],
        "group_id": plan["cohort_id"]+f"-g{index//learning['group_size']:03d}",
        "opponent": opponent_for(plan, index), "actor_rng_seeds": [plan["actor_rng_seeds"][index]]}
    if plan.get("shop_revision_every"):
        extra["shop_revision_every"] = plan["shop_revision_every"]
    if plan.get("full_plan"):
        extra.update(full_plan=True, plan_adapter=plan["plan_adapter"])
    return bound({**{k: plan[k] for k in common}, "schema": WORLD_SCHEMA, "cohort_plan_sha256": plan["content_sha256"],
        "world_index": index, "game_id": plan["cohort_id"] + f"-world{index:03d}", "seed": plan["seeds"][index],
        "actor_rng_seeds": plan["actor_rng_seeds"][2*index:2*index+2], "physical_steps": 719, **extra})


_base_finish = b32._bind(base.finish, GAME_SCHEMA=GAME_SCHEMA)


def finish(plan, env, controllers, *, turns_sha256, future=None):
    result = _base_finish(plan, env, controllers, turns_sha256=turns_sha256)
    return bound({**{k: v for k, v in result.items() if k != "content_sha256"},
                  "model_decisions": (1 if plan.get("plan_learning") else 2)*len(decision_steps(plan)),
                  "opening_only": plan["opening_only"], **({"future": future} if plan.get("plan_learning") else {}),
                  **({"frozen_opponent_model_decisions":len(decision_steps(plan))}
                     if plan.get("opponent", {}).get("kind") == NEURAL_PEER else {})})


INPUT_SIDECAR = "inputs.npz"


def capture_decision(plan, step, seat, packed, behavior, value, orders_snapshot, ordinal):
    """PRE-edit pack reference + critic; the float arrays live in the world's binary sidecar."""
    if type(step) is not int or step not in decision_steps(plan) or type(seat) is not int or seat not in (0, 1):
        raise ValueError("natural daily step and seat required")
    expected_schema = plan_features.PACK_SCHEMA if plan.get("full_plan") else features.SCHEMA
    if packed.get("schema") != expected_schema: raise ValueError("input representation differs from registered execution")
    validate_input(packed); base._trace(behavior, plan, step=step)
    reference = {k: (list(v) if k == "token_ids" else copy.deepcopy(v)) for k, v in packed.items()
                 if k not in ("rows", "knownness", "token_mask")}
    reference.update({"sidecar": INPUT_SIDECAR, "ordinal": ordinal, "shape": list(packed["rows"].shape)})
    return bound({"schema": SCHEMA+"-decision", "step": step, "seat": seat, "input": reference,
        "orders_sha256": digest(orders_snapshot), "behavior": copy.deepcopy(behavior),
        "old_log_probability": behavior["joint_logp"], "old_value": _finite(value)})


def sidecar_arrays(packed, ordinal):
    return {f"rows_{ordinal}": packed["rows"], f"knownness_{ordinal}": packed["knownness"], f"token_mask_{ordinal}": packed["token_mask"]}


def validate_input(packed):
    if isinstance(packed, dict) and packed.get("schema") == plan_features.PACK_SCHEMA:
        return plan_features.validate(packed)
    return features.validate(packed)


def recorded_pack(reference, arrays):
    """Rebuild the validated pack a decision saw from its sidecar arrays; identity is rechecked."""
    o = reference["ordinal"]
    packed = {k: v for k, v in reference.items() if k not in ("sidecar", "ordinal", "shape")}
    packed.update({"rows": np.asarray(arrays[f"rows_{o}"], np.float32), "knownness": np.asarray(arrays[f"knownness_{o}"], np.bool_),
                   "token_mask": np.asarray(arrays[f"token_mask_{o}"], np.bool_), "token_ids": tuple(reference["token_ids"])})
    if list(packed["rows"].shape) != reference["shape"]: raise ValueError("sidecar shape drift")
    return validate_input(packed)


def collate_bucket(packs):
    """Existing collate, then pad the token axis to the registered bucket."""
    schemas = {p.get("schema") for p in packs}
    if schemas == {plan_features.PACK_SCHEMA}: batch = plan_features.collate(packs)
    elif schemas == {features.SCHEMA}: batch = features.collate(packs)
    else: raise ValueError("one explicit representation per cohort; no mixed legacy/full-plan packs")
    length = batch["rows"].shape[1]
    padded = fast.pad_bucket(length, PAD_MULTIPLE)
    if padded != length:
        extra = padded-length
        batch["rows"] = np.concatenate([batch["rows"], np.zeros((len(packs), extra, features.WIDTH), np.float32)], 1)
        batch["knownness"] = np.concatenate([batch["knownness"], np.zeros((len(packs), extra, features.WIDTH), np.bool_)], 1)
        batch["token_mask"] = np.concatenate([batch["token_mask"], np.zeros((len(packs), extra), np.bool_)], 1)
    return batch


def batch_inputs(plan, step, packs, *, opponent=False):
    lanes = opponent_lanes(plan) if opponent else plan["lane_order"]
    if type(step) is not int or step not in decision_steps(plan) or len(packs) != len(lanes) or not lanes:
        raise ValueError("complete registered economic-decision cohort required")
    expected_schema = plan_features.PACK_SCHEMA if plan.get("full_plan") else features.SCHEMA
    if any(p.get("schema") != expected_schema for p in packs):
        raise ValueError("batch representation differs from registered execution")
    batch = collate_bucket(packs)
    arrays = {}
    for key in ("rows", "token_mask", "knownness"):
        value = np.ascontiguousarray(batch[key])
        arrays[key] = {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": digest_bytes(value.tobytes())}
    identity = bound({"schema": SCHEMA+"-batch", "plan_sha256": plan["content_sha256"], "step": step,
        "lane_order": lanes, "neural_batch": BATCH_MODE, "input_sha256s": list(batch["content_sha256s"]),
        **({"inference_role":NEURAL_PEER} if opponent else {}),
        "token_counts": [len(p["rows"]) for p in packs], "arrays": arrays})
    return batch, identity


def digest_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")


def _check_environment(plan, graph=None):
    if (not sys.dont_write_bytecode or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or (graph is not None and next(graph.parameters()).device.type != "cuda")):
        raise ValueError("actual CUDA cold -B :4096:8 execution required; no CPU fallback")
    _equal(environment_profile(), plan["environment"], "actual cohort numerical environment")


def sample_opponent_batch(plan, step, packs, streams, graph, encoder, sampler):
    """Same frozen round weights, separate simultaneous actor-safe inputs/RNG.

    No learner update or opponent economics is inserted by this helper. Its
    complete neural sampling is independently repeated after fresh reconstruction.
    """
    import torch
    batch, identity = batch_inputs(plan, step, packs, opponent=True)
    with torch.no_grad():
        tensors = tuple(torch.as_tensor(batch[k], device=next(graph.parameters()).device)
                        for k in ("rows", "token_mask", "knownness"))
        context, values, _ = encoder(*tensors)
        records, totals, _ = sampler.sample(context, streams, max_commands=plan["command_budget"], as_of_step=step)
        replayed, _ = fast.replay_padded(graph, context, records, expected_identity=fast.identity())
        error = float((replayed-totals).abs().max())
        if error > fast.TOLERANCES["within_profile_logp_atol"]:
            raise ValueError("opponent sampler likelihood canary failed")
        return records, values.tolist(), identity, error


# ------------------------------------------------------------------ collector
def collect(output, model_path, plan, *, canary_every=5):
    """Execute one registered fast cohort in this process; partial raw is preserved on failure."""
    import torch
    validate_cohort_plan(plan)
    target = Path(output).absolute()
    if target != target.resolve() or not target.is_relative_to(ROOT/"work/runs") or target.exists():
        raise ValueError("exclusive new output below work/runs required; partial restart refused")
    _check_environment(plan)
    graph, manifest = policy.load_model(model_path, device="cuda", expected_file_sha256=plan["model_sha256"],
        allow_remaining_window_transition=True)
    _equal(plan.get("model_source_transition"),
        policy.check_model_sources(manifest, allow_remaining_window_transition=True), "model source transition")
    for key, expected in (("content_sha256", plan["model_manifest_sha256"]), ("optimizer_steps", plan["model_optimizer_age"]),
                          ("rollout_iteration", plan["rollout_iteration"]-1)):
        _equal(manifest[key], expected, "actual model " + key)
    graph.eval()
    fast_id, head_id = fast.identity(), heads.identity()
    encoder = fast.FastEncoder(graph, expected_identity=fast_id)
    sampler = fast.GraphedSampler(graph, plan["lanes"], plan["temperature"])
    peer_lanes = opponent_lanes(plan)
    peer_indices = {tuple(pair):lane for lane,pair in enumerate(peer_lanes)}
    peer_sampler = fast.GraphedSampler(graph, len(peer_lanes), plan["temperature"]) if peer_lanes else None
    peer_streams = [fast.LaneStream(opponent_for(plan,w)["rng_seed"]) for w,s in peer_lanes]
    peer_batches, peer_seconds = [], 0.0
    target.mkdir(parents=True); _write(target/"PLAN.json", plan)
    started = time.perf_counter()
    phase = {k: 0.0 for k in ("observe_pack", "collate_hash", "encode", "sample", "canary_replay", "lower_step", "capture_write")}
    canary = {"steps": 0, "logp_max_abs": 0.0, "value_checked": True}
    envs, controllers, batches, world_results, step = [], [], [], [], -1
    sidecars = [{} for _ in range(plan["worlds"])]
    lane_indices = {tuple(pair): lane for lane, pair in enumerate(plan["lane_order"])}
    future_samples = [future_sampler(plan, w) for w in range(plan["worlds"])]
    try:
        device = next(graph.parameters()).device
        streams = [fast.LaneStream(seed) for seed in plan["actor_rng_seeds"]]
        registrations = [world_plan(plan, w) for w in range(plan["worlds"])]
        for w, seed in enumerate(plan["seeds"]):
            envs.append(ExactKaggricultureEnvironment(seed=seed)); controllers.append(initial_controllers(plan, w))
        from contextlib import ExitStack
        with ExitStack() as stack:
            tapes = []
            for w in range(plan["worlds"]):
                folder = target/f"world-{w:03d}"; folder.mkdir()
                tapes.append(stack.enter_context(gzip.open(folder/"turns.jsonl.gz", "xt", encoding="utf-8", compresslevel=6)))
            for step in range(719):
                tick = time.perf_counter()
                observations, packs, peer_packs = [], [], []
                for w, env in enumerate(envs):
                    if env.done or env.step_index != step: raise ValueError(f"world{w} left registered synchronous horizon at step{step}")
                    obs = [env.agent_observation(s) for s in (0, 1)]; observations.append(obs)
                    for s in (0, 1):
                        controllers[w][s].observe(obs[s])
                        if step in decision_steps(plan) and (w, s) in lane_indices:
                            check_decision_observation(plan, step, obs[s], env.configuration)
                            packs.append(pack_plan(plan, obs[s], controllers[w][s], env.configuration))
                        if step in decision_steps(plan) and (w,s) in peer_indices:
                            check_decision_observation(plan, step, obs[s], env.configuration)
                            peer_packs.append(pack_plan(plan, obs[s], controllers[w][s], env.configuration))
                phase["observe_pack"] += time.perf_counter()-tick
                programs, calls = [None]*plan["lanes"], [None]*plan["lanes"]
                peer_calls = {}
                if step in decision_steps(plan):
                    tick = time.perf_counter(); batch, identity = batch_inputs(plan, step, packs); batches.append(identity)
                    phase["collate_hash"] += time.perf_counter()-tick
                    with torch.no_grad():
                        tick = time.perf_counter()
                        tensors = tuple(torch.as_tensor(batch[k], device=device) for k in ("rows", "token_mask", "knownness"))
                        context, values, _ = encoder(*tensors); torch.cuda.synchronize()
                        phase["encode"] += time.perf_counter()-tick; tick = time.perf_counter()
                        records, totals, _ = sampler.sample(context, streams, max_commands=plan["command_budget"],
                            **({"as_of_step": step} if plan.get("full_plan") else {}))
                        value_list = values.tolist(); phase["sample"] += time.perf_counter()-tick
                        if canary_every and (len(batches) % canary_every == 0 or plan["opening_only"]):
                            tick = time.perf_counter()
                            replayed, _ = fast.replay_padded(graph, context, records, expected_identity=fast_id)
                            canary["logp_max_abs"] = max(canary["logp_max_abs"], float((replayed-totals).abs().max())); canary["steps"] += 1
                            phase["canary_replay"] += time.perf_counter()-tick
                        del context, values, tensors, totals
                    torch.cuda.empty_cache()   # keep resident VRAM near model+context while overlapped with the learner
                    tick = time.perf_counter()
                    for lane, (w, s) in enumerate(plan["lane_order"]):
                        programs[lane] = records[lane]
                        ordinal = 2*(len(batches)-1)+s
                        call = capture_decision(registrations[w], step, s, packs[lane], records[lane], float(value_list[lane]), controllers[w][s].snapshot(), ordinal)
                        sidecars[w].update(sidecar_arrays(packs[lane], ordinal))
                        calls[lane] = bound({**call, "lane_index": lane, "cohort_batch_sha256": identity["content_sha256"]})
                    phase["capture_write"] += time.perf_counter()-tick
                    if peer_lanes:
                        peer_tick = time.perf_counter()
                        peer_records, peer_values, peer_identity, peer_error = sample_opponent_batch(
                            plan, step, peer_packs, peer_streams, graph, encoder, peer_sampler)
                        peer_batches.append(peer_identity)
                        canary["logp_max_abs"] = max(canary["logp_max_abs"], peer_error)
                        for lane,(w,s) in enumerate(peer_lanes):
                            ordinal = 2*(len(batches)-1)+s
                            call = capture_decision(registrations[w], step, s, peer_packs[lane], peer_records[lane],
                                peer_values[lane], controllers[w][s].snapshot(), ordinal)
                            sidecars[w].update(sidecar_arrays(peer_packs[lane], ordinal))
                            peer_calls[(w,s)] = bound({**call, "lane_index":lane,
                                "cohort_batch_sha256":peer_identity["content_sha256"],
                                "opponent_identity_sha256":digest(opponent_for(plan,w))})
                        peer_seconds += time.perf_counter()-peer_tick
                        torch.cuda.empty_cache()
                for w, env in enumerate(envs):
                    tick = time.perf_counter(); rejected = []
                    if step == 0: rejected.extend(publish_peer(plan, w, controllers[w]))
                    if step in decision_steps(plan):
                        for s in (0, 1):
                            if (w,s) in peer_indices:
                                record = peer_calls[(w,s)]["behavior"]
                                controllers[w][s], failures = publish_program(plan, controllers[w][s], record["program"],
                                    step=step, observation=observations[w][s])
                                rejected.extend({"step":step,"seat":s,**failure} for failure in failures)
                                continue
                            if (w, s) not in lane_indices: continue
                            lane = lane_indices[(w, s)]
                            if plan["opening_only"] or plan.get("shop_revision_every"):
                                controllers[w][s], failures = publish_program(plan, controllers[w][s], programs[lane]["program"], step=step, observation=observations[w][s])
                                rejected.extend({"step": step, "seat": s, **failure} for failure in failures)
                            else:
                                for command in programs[lane]["program"]:
                                    try: controllers[w][s].apply(command)
                                    except ValueError as exc: rejected.append({"step": step, "seat": s, "command": command, "reason": str(exc)})
                    joint = joint_action(controllers[w], observations[w], env.configuration)
                    env.step(joint); phase["lower_step"] += time.perf_counter()-tick
                    tick = time.perf_counter()
                    if future_samples[w] is not None:
                        s = plan["lane_order"][w][1]
                        observe_future(future_samples[w], env, controllers[w][s], s)
                    row = {"schema": SCHEMA+"-turn", "step": step, "actions": joint, "rejected_edits": rejected,
                           "decisions": [calls[lane_indices[(w, s)]] if (w, s) in lane_indices else None
                                         for s in (0, 1)] if step in decision_steps(plan) else None}
                    if peer_lanes:
                        row["opponent_decision"] = peer_calls.get((w,1-plan["lane_order"][w][1]))
                    tapes[w].write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
                    phase["capture_write"] += time.perf_counter()-tick
                if step in decision_steps(plan):
                    print(json.dumps({"step": step, "worlds": plan["worlds"], "model_decisions": len(batches)*plan["lanes"]}), flush=True)
        for w, env in enumerate(envs):
            tick = time.perf_counter()
            with (target/f"world-{w:03d}"/INPUT_SIDECAR).open("xb") as stream:
                np.savez_compressed(stream, **sidecars[w])   # ~32x smaller than raw float32 packs
            phase["capture_write"] += time.perf_counter()-tick
            result = finish(registrations[w], env, controllers[w], turns_sha256=file_sha(target/f"world-{w:03d}/turns.jsonl.gz"),
                            future=finish_future(future_samples[w]))
            _write(target/f"world-{w:03d}/RESULT.json", result)
            world_results.append({"world_index": w, "result_sha256": result["content_sha256"],
                "result_file_sha256": file_sha(target/f"world-{w:03d}/RESULT.json"), "turns_sha256": result["turns_sha256"],
                "inputs_sha256": file_sha(target/f"world-{w:03d}"/INPUT_SIDECAR)})
        _write(target/"COHORTS.json", bound({"schema": SCHEMA+"-cohorts", "plan_sha256": plan["content_sha256"], "batches": batches,
            **({"opponent_batches":peer_batches} if peer_lanes else {})}))
        _equal(file_sha(model_path), plan["model_sha256"], "unchanged behavior model"); attest(plan["source_pins"])
        if canary["logp_max_abs"] > fast.TOLERANCES["within_profile_logp_atol"]:
            raise ValueError("producer sampler/replay parity canary exceeded the within-profile tolerance")
        result = bound({"schema": SCHEMA+"-collection", "status": "COMPLETE_PENDING_INDEPENDENT_ADMISSION",
            "plan_sha256": plan["content_sha256"], "model_sha256": plan["model_sha256"], "source_sha256": plan["source_sha256"],
            "world_results": world_results, "cohorts_file_sha256": file_sha(target/"COHORTS.json"), "cohorts_sha256": digest(batches),
            "physical_games": plan["worlds"], "physical_steps": 719*plan["worlds"],
            "distinct_initial_worlds": len(set(plan["seeds"])),
            "model_decisions": len(decision_steps(plan))*plan["lanes"], "opening_only": plan["opening_only"],
            **({"frozen_opponent_model_decisions":len(peer_lanes)*len(decision_steps(plan)),
                "opponent_inference_seconds":peer_seconds} if peer_lanes else {}),
            "optimizer_updates": 0, "responsive_reference_games": 0, "terminal_training_admission": False,
            "seconds": time.perf_counter()-started, "host_wall_phase_seconds": phase, "sampler_replay_canary": canary,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
        _write(target/"RESULT.json", result)
        return result
    except BaseException as exc:
        try:
            _write(target/"FAILURE.json", {"schema": SCHEMA+"-failure", "plan_sha256": plan["content_sha256"], "step": step,
                "error_type": type(exc).__name__, "message": str(exc), "terminal_training_admission": False, "optimizer_updates": 0})
        except OSError: pass
        raise


# --------------------------------------------------------------- lean witness
def reconstruct_world(plan, index, result, rows, arrays):
    """Fresh engine + recorded programs: exact executor inputs, actions and terminal cash, no model."""
    registration = world_plan(plan, index); _sealed(result)
    _equal(result["schema"], GAME_SCHEMA, "game schema"); _equal(result["plan_sha256"], registration["content_sha256"], "game plan")
    _equal(result["status"], "COMPLETE_PENDING_INDEPENDENT_ADMISSION", "complete producer")
    env = ExactKaggricultureEnvironment(seed=registration["seed"]); controllers = initial_controllers(plan, index)
    learning = plan.get("plan_learning")
    seats = [s for w, s in plan["lane_order"] if w == index]
    lane_indices = {s: lane for lane, (w, s) in enumerate(plan["lane_order"]) if w == index}
    peer_indices = {s:lane for lane,(w,s) in enumerate(opponent_lanes(plan)) if w == index}
    peer_seat = next(iter(peer_indices), None)
    lane_indices.update(peer_indices)
    future = future_sampler(plan, index)
    episodes = [{"schema": EPISODE_SCHEMA, "episode_id": registration["game_id"]+f"-seat{s}", "seed": registration["seed"], "seat": s,
        "behavior_model_sha256": plan["model_sha256"], "model_manifest_sha256": plan["model_manifest_sha256"],
        "numerics": plan["numerics"], "rollout_iteration": plan["rollout_iteration"], "source_sha256": plan["source_sha256"],
        "calls": []} for s in (0, 1)]
    fresh = [[], []]
    count = 0
    opening_summary, revision_summary, peer_summary = [], [], []
    for step, row in enumerate(rows):
        if step >= 719 or env.done: raise ValueError("extra physical suffix after terminal")
        _equal(row["step"], step, "consecutive physical step"); _equal(row["schema"], SCHEMA+"-turn", "producer turn schema")
        obs = [env.agent_observation(s) for s in (0, 1)]
        for s in (0, 1): controllers[s].observe(obs[s])
        rejected = publish_peer(plan, index, controllers) if step == 0 else []
        peer_call = row.get("opponent_decision")
        if (peer_seat is None or step not in decision_steps(plan)) and peer_call is not None:
            raise ValueError("unregistered opponent neural call")
        if step in decision_steps(plan):
            if not isinstance(row["decisions"], list) or len(row["decisions"]) != 2: raise ValueError("both complete actual neural decisions required")
            actual_calls = list(row["decisions"])
            if peer_seat is not None:
                if actual_calls[peer_seat] is not None or not isinstance(peer_call, dict):
                    raise ValueError("separate frozen opponent record required; never a learner row")
                _equal(peer_call.get("opponent_identity_sha256"), digest(opponent_for(plan,index)), "frozen opponent identity")
                actual_calls[peer_seat] = peer_call
            for s, call in enumerate(actual_calls):
                if s not in seats and s != peer_seat:
                    if call is not None: raise ValueError("frozen external opponent cannot create a learner decision")
                    continue
                _sealed(call); _equal(call["schema"], SCHEMA+"-decision", "decision schema")
                _equal([call["step"], call["seat"], call["lane_index"]], [step, s, lane_indices[s]], "decision identity")
                check_decision_observation(plan, step, obs[s], env.configuration)
                packed = pack_plan(plan, obs[s], controllers[s], env.configuration)
                recorded = recorded_pack(call["input"], arrays)
                _equal(recorded["content_sha256"], packed["content_sha256"], "fresh actor-safe full input")
                if not (np.array_equal(recorded["rows"], packed["rows"]) and np.array_equal(recorded["knownness"], packed["knownness"])
                        and np.array_equal(recorded["token_mask"], packed["token_mask"])):
                    raise ValueError("recorded actor rows differ from fresh pack")
                _equal(call["orders_sha256"], digest(controllers[s].snapshot()), "pre-edit commitments")
                base._trace(call["behavior"], plan, step=step)
                _equal(call["old_log_probability"], call["behavior"]["joint_logp"], "emitted joint logp"); _finite(call["old_value"])
                episodes[s]["calls"].append(copy.deepcopy(call)); fresh[s].append(packed)
                if plan["opening_only"] or plan.get("shop_revision_every"):
                    controllers[s], failures = publish_program(plan, controllers[s], call["behavior"]["program"], step=step, observation=obs[s])
                    rejected.extend({"step": step, "seat": s, **failure} for failure in failures)
                    summary = {"seat": s, "sampled_commands": len(call["behavior"]["program"]),
                        "stop_only": call["behavior"]["program"] == [{"op": "STOP"}],
                        "whole_plan_refused": False, "conflicting_commands_refused": len(failures),
                        "accepted_tile_orders": len(controllers[s].orders),
                        "accepted_market_rules": len(controllers[s].rules)}
                    if s == peer_seat: peer_summary.append({"step":step,**summary})
                    elif step == 0: opening_summary.append(summary)
                    else: revision_summary.append({"step": step, "visible_shops": len(obs[s]["town"]["unlocked_shops"]), **summary})
                else:
                    for command in call["behavior"]["program"]:
                        try: controllers[s].apply(command)
                        except ValueError as exc: rejected.append({"step": step, "seat": s, "command": command, "reason": str(exc)})
        elif row["decisions"] is not None: raise ValueError("unregistered extra neural decision")
        _equal(row["rejected_edits"], rejected, "actual rejected edits")
        joint = joint_action(controllers, obs, env.configuration)
        _equal(row["actions"], joint, "lowered physical actions")
        env.step(joint); count += 1
        if future is not None: observe_future(future, env, controllers[seats[0]], seats[0])
    if count != 719 or any([c["step"] for c in episodes[s]["calls"]] != list(decision_steps(plan)) for s in seats):
        raise ValueError("all719 turns and exactly the registered decision steps per seat required")
    future_labels = finish_future(future)
    expected = finish(registration, env, controllers, turns_sha256=result["turns_sha256"], future=future_labels)
    _equal(result, expected, "complete terminal/model/source/trajectory result")
    for s, episode in enumerate(episodes):
        delta = result["cash"][s]-result["cash"][1-s]; episode["terminal_reward"] = int(delta > 0)-int(delta < 0)
        if learning is not None and s in seats:
            episode.update(future=future_labels, group_id=registration["group_id"],
                group_size=learning["group_size"], opponent_sha256=digest(registration["opponent"]))
            if "reward" in learning:
                episode["terminal_cash"] = [result["cash"][s], result["cash"][1-s]]
    selected_episodes = [episodes[s] for s in seats]
    if peer_seat is not None:
        if [c["step"] for c in episodes[peer_seat]["calls"]] != list(decision_steps(plan)):
            raise ValueError("all five frozen-opponent decisions required")
        episodes[peer_seat].update(inference_role=NEURAL_PEER, opponent_identity=opponent_for(plan,index))
    witness = bound({"schema": SCHEMA+"-engine-witness", "plan_sha256": plan["content_sha256"],
        "world_plan_sha256": registration["content_sha256"], "producer_result_sha256": result["content_sha256"],
        "turns_sha256": result["turns_sha256"], "training_batch_sha256": digest({"episodes": selected_episodes}),
        "reconstructed_physical_games": 1, "responsive_reference_games": 0, "physical_steps": 719,
        "decision_calls": len(seats)*len(decision_steps(plan)), "opening_only": plan["opening_only"],
        **({"frozen_opponent_decision_calls":len(peer_summary),"opponent_programs":peer_summary,
            "opponent_episode_sha256":digest(episodes[peer_seat])} if peer_seat is not None else {}),
        "opening_programs": opening_summary,
        **({"revision_programs": revision_summary} if plan.get("shop_revision_every") else {}),
        "terminal_cash": result["cash"],
        "method": "FRESH_ENGINE_RECORDED_PROGRAMS_EXACT_EXECUTOR_INPUTS_ACTIONS_TERMINAL_CASH_LEAN_TAPE",
        "terminal_training_admission": False})
    for s, episode in enumerate(episodes):        # fresh validated arrays feed the fitter; the digest covers the sealed records
        for call, packed in zip(episode["calls"], fresh[s], strict=True):
            call["input"] = packed
    return {"episodes": selected_episodes, "engine_witness": witness,
            **({"opponent_episodes":[episodes[peer_seat]]} if peer_seat is not None else {})}


def witness_world(producer, index, expected_plan_content_sha256):
    """One world; used by the spawned CPU pool. Returns episodes plus a witness receipt."""
    producer = Path(producer)
    plan, _ = b32._read_pinned(producer/"PLAN.json"); validate_cohort_plan(plan)
    _equal(plan["content_sha256"], _sha(expected_plan_content_sha256), "registered cohort plan content")
    result, _ = b32._read_pinned(producer/"RESULT.json"); _sealed(result)
    member = result["world_results"][index]; _equal(member["world_index"], index, "world membership")
    child = producer/f"world-{index:03d}"
    _equal(file_sha(child/"RESULT.json"), member["result_file_sha256"], "exact producer world RESULT")
    _equal(file_sha(child/"turns.jsonl.gz"), member["turns_sha256"], "exact producer world tape")
    _equal(file_sha(child/INPUT_SIDECAR), member["inputs_sha256"], "exact producer input sidecar")
    world_result, _ = b32._read_pinned(child/"RESULT.json", member["result_file_sha256"])
    with np.load(child/INPUT_SIDECAR) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    location = [-1]
    def records(stream):
        for ordinal, line in enumerate(stream):
            location[0] = ordinal; yield b32._json(line)
    try:
        with gzip.open(child/"turns.jsonl.gz", "rt", encoding="utf-8") as stream:
            checked = reconstruct_world(plan, index, world_result, records(stream), arrays)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"world{index} step{location[0]} physical witness: {exc}") from exc
    return checked


def _witness_job(args):
    return witness_world(*args)


def cohort_batches(plan, episodes):
    """Recover the ORIGINAL padded daily cohorts from reconstructed episodes; identity must match."""
    if not isinstance(episodes, list) or len(episodes) != plan["lanes"]: raise ValueError("every registered world/seat episode required")
    for lane, (w, s) in enumerate(plan["lane_order"]):
        episode = episodes[lane]
        for key, expected in (("schema", EPISODE_SCHEMA), ("seat", s), ("seed", plan["seeds"][w]),
                ("episode_id", world_plan(plan, w)["game_id"]+f"-seat{s}"), ("behavior_model_sha256", plan["model_sha256"]),
                ("model_manifest_sha256", plan["model_manifest_sha256"]), ("rollout_iteration", plan["rollout_iteration"]),
                ("source_sha256", plan["source_sha256"]), ("numerics", plan["numerics"])):
            _equal(episode[key], expected, "episode " + key)
        if (type(episode["terminal_reward"]) is not int or episode["terminal_reward"] not in (-1, 0, 1)
                or [c["step"] for c in episode["calls"]] != list(decision_steps(plan))):
            raise ValueError("exact registered calls and actual terminal WDL required")
        if plan.get("plan_learning") is None and s == 1 and episode["terminal_reward"] != -episodes[lane-1]["terminal_reward"]:
            raise ValueError("opposing world rewards must be zero-sum")
        if plan.get("plan_learning") is not None:
            world = world_plan(plan, w)
            for key, expected in (("group_id", world["group_id"]), ("group_size", plan["plan_learning"]["group_size"]),
                                  ("opponent_sha256", digest(world["opponent"]))):
                _equal(episode[key], expected, "matched GRPO " + key)
            if "reward" in plan["plan_learning"]: witnessed_episode_cash(episode)
    groups = []
    if plan.get("plan_learning"):
        size = plan["plan_learning"]["group_size"]
        for i in range(0, len(episodes), size):
            group = episodes[i:i+size]
            for e in group:
                for key in ("seed", "seat", "group_id", "opponent_sha256"):
                    _equal(e[key], group[0][key], "same-condition GRPO " + key)
                _equal(e["calls"][0]["input"]["content_sha256"], group[0]["calls"][0]["input"]["content_sha256"],
                       "same actor-visible initial condition")
    for ordinal, step in enumerate(decision_steps(plan)):
        calls = [e["calls"][ordinal] for e in episodes]
        packs = [c["input"] for c in calls]   # fresh witness-validated packs
        batch, identity = batch_inputs(plan, step, packs)
        for lane, call in enumerate(calls):
            _equal(call["cohort_batch_sha256"], identity["content_sha256"], "original cohort padding and lane hashes")
            call.pop("input")   # collated arrays are the fitter's inputs; free the wire lists
        grouped = {} if plan.get("plan_learning") is None else {
            "group_size": plan["plan_learning"]["group_size"],
            "future": [e["future"]["by_step"][str(step)] if plan.get("shop_revision_every") else e["future"] for e in episodes],
            "group_ids": [e["group_id"] for e in episodes]}
        if (plan.get("plan_learning") or {}).get("schema") == PLAN_WEIGHTED_SCHEMA:
            grouped["opponent_kinds"] = [opponent_for(plan,w)["kind"] for w,s in plan["lane_order"]]
        if (plan.get("plan_learning") or {}).get("reward"):
            grouped["terminal_cash"] = [witnessed_episode_cash(e) for e in episodes]
        groups.append({"calls": calls, "returns": [e["terminal_reward"] for e in episodes], "batch": identity, **grouped,
                       "arrays": {k: np.ascontiguousarray(batch[k]) for k in ("rows", "token_mask", "knownness")}})
    return groups


def cohort_opponent_batches(plan, episodes):
    """All reconstructed peer calls in their original padding/order; never labels."""
    lanes = opponent_lanes(plan)
    if len(episodes) != len(lanes): raise ValueError("every registered neural opponent episode required")
    for episode,(w,s) in zip(episodes,lanes,strict=True):
        for key,value in (("inference_role",NEURAL_PEER),("seat",s),("seed",plan["seeds"][w]),
                ("opponent_identity",opponent_for(plan,w)),("behavior_model_sha256",plan["model_sha256"]),
                ("episode_id",world_plan(plan,w)["game_id"]+f"-seat{s}")):
            _equal(episode.get(key),value,"frozen neural episode "+key)
        if [c["step"] for c in episode["calls"]] != list(decision_steps(plan)):
            raise ValueError("complete frozen neural clock required")
    groups=[]
    if not lanes:return groups
    for ordinal,step in enumerate(decision_steps(plan)):
        calls=[copy.deepcopy(e["calls"][ordinal]) for e in episodes]
        batch,identity=batch_inputs(plan,step,[c["input"] for c in calls],opponent=True)
        for call in calls:
            _equal(call["cohort_batch_sha256"],identity["content_sha256"],"opponent original cohort")
            call.pop("input")
        groups.append({"calls":calls,"batch":identity,
                       "arrays":{k:np.ascontiguousarray(batch[k]) for k in ("rows","token_mask","knownness")}})
    return groups


# ------------------------------------------------------------------- learner
def committed_status(execution):
    if execution.get("supervised_stage") == EFFECT_STAGE:
        return io.EFFECT_STATUS
    if execution.get("supervised_stage") == SUPERVISED_STAGE:
        return io.BC_STATUS
    return io.GRPO_STATUS if execution.get("plan_learning") else io.PPO_STATUS


def supervised_execution(execution, *, loss_reduction=None, preferences=None):
    """Keep the game contract for returning to RL; describe BC honestly as BC."""
    original = execution.get("resume_game_execution", execution)
    if original.get("numerics") != fast.NUMERICS or original.get("supervised_stage"):
        raise ValueError("known fast-profile game execution required")
    reduction = execution.get("loss_reduction", "factor") if loss_reduction is None else loss_reduction
    if reduction not in ("factor", "command"):
        raise ValueError("known supervised loss reduction required")
    result = {"schema": SUPERVISED_STAGE, "supervised_stage": SUPERVISED_STAGE,
        "numerics": fast.NUMERICS, "profile": fast.SCHEMA,
        "objective": "MEAN_PER_EXAMPLE_OBSERVED_FACTOR_NLL; NO_TERMINAL_RL_TARGETS",
        "encoder": "UNCHANGED_FAST_ENCODER; OBSERVED_PREFIX_REPLAY",
        "auxiliary": "PRESERVE_HEADS_AND_ADAM_UNCHANGED; ZERO_AUXILIARY_UPDATES",
        "plan_learning": copy.deepcopy(original.get("plan_learning")),
        "resume_game_execution": copy.deepcopy(original), "training_admission": False}
    if reduction == "command":
        result.update(loss_reduction="command",
            objective="MEAN_OF_COMMAND_MEAN_OBSERVED_FACTOR_NLL; NO_TERMINAL_RL_TARGETS")
    preference = execution.get("preferences") if preferences is None else preferences
    if preference is not None:
        required = {"schema", "data_sha256", "reference_model_sha256", "coefficient"}
        if isinstance(preference, dict) and "mode" in preference:
            required.add("mode")
        if (reduction != "factor" or not isinstance(preference, dict) or set(preference) != required
                or preference["schema"] != "farm-native-program-preferences-v1"
                or preference.get("mode", "program") not in ("program", "first_stop")
                or type(preference["coefficient"]) not in (int, float)
                or not math.isfinite(preference["coefficient"]) or not 0 <= preference["coefficient"] <= 10
                or any(not isinstance(preference[k], str) or len(preference[k]) != 64
                       or any(c not in "0123456789abcdef" for c in preference[k])
                       for k in ("data_sha256", "reference_model_sha256"))):
            raise ValueError("explicit factor-BC preference configuration required")
        result.update(preferences=copy.deepcopy(preference),
            objective="FACTOR_BC_PLUS_MEAN_STATE_REFERENCE_RELATIVE_PROGRAM_PREFERENCE; OBSERVED_WDL_ORDER; NOT_PPO")
        if preference.get("mode") == "first_stop":
            result["objective"] = "FACTOR_BC_PLUS_MEAN_STATE_FIRST_STOP_BCE; OBSERVED_KEEP_EDIT_LABEL; NOT_PPO"
    return result


def validate_effect_coefficient(value, objective="joint"):
    """Explicit bounded experiment weight; never accept bools or nonfinite values."""
    if objective not in ("joint", "wdl_only", "wdl_contrast", "wdl_contrast_rank"):
        raise ValueError("known observed-outcome objective required")
    if objective in ("wdl_only", "wdl_contrast", "wdl_contrast_rank"):
        if type(value) not in (int, float) or value != 0:
            raise ValueError("WDL-only objective requires zero paired-effect coefficient")
        return 0.
    if type(value) not in (int, float) or not 0 < value <= 10000:
        raise ValueError("finite paired-effect coefficient in (0,10000] required")
    return float(value)


def validate_plan_context_gain(value):
    """Bound the declared KEEP-relative transform; one preserves historical bytes."""
    if type(value) not in (int, float) or not 1 <= value <= 10000:
        raise ValueError("finite plan context gain in [1,10000] required")
    return float(value)


def effect_execution(execution, plan_context=None, effect_coefficient=None, plan_context_gain=None, encoder_readout=None, effect_objective=None, effect_diagnostic_precision=None):
    """Observed terminal labels for authored alternatives; no sampled-policy loss."""
    original = execution.get("resume_game_execution", execution)
    if original.get("numerics") != fast.NUMERICS or original.get("supervised_stage") or not original.get("plan_learning"):
        raise ValueError("known fast game execution with existing prediction heads required")
    mode = execution.get("plan_context", "raw_factor_prefix") if plan_context is None else plan_context
    if mode not in ("raw_factor_prefix", "published_plan_encoder"):
        raise ValueError("known observed-plan context required")
    previous_objective = execution.get("effect_objective", "joint")
    objective = previous_objective if effect_objective is None else effect_objective
    default_weight = 0. if objective in ("wdl_only", "wdl_contrast", "wdl_contrast_rank") else 1.
    inherited_weight = execution.get("effect_coefficient", default_weight) if objective == previous_objective else default_weight
    weight = validate_effect_coefficient(inherited_weight if effect_coefficient is None else effect_coefficient, objective)
    precision = execution.get("effect_diagnostic_precision", "float32") if effect_diagnostic_precision is None else effect_diagnostic_precision
    if precision not in ("float32", "float64") or (precision == "float64" and objective == "joint"):
        raise ValueError("float64 cash diagnostics require the WDL-only objective")
    gain = validate_plan_context_gain(execution.get("plan_context_gain", 1.)
        if plan_context_gain is None else plan_context_gain)
    if gain != 1. and mode != "published_plan_encoder":
        raise ValueError("KEEP-relative gain requires published plan context")
    readout = execution.get("encoder_readout", "mean") if encoder_readout is None else encoder_readout
    if readout not in ("mean", "kind_mean"):
        raise ValueError("known entity encoder readout required")
    if readout != "mean" and mode != "published_plan_encoder":
        raise ValueError("kind-mean readout requires published plan context")
    result = {"schema": EFFECT_STAGE, "supervised_stage": EFFECT_STAGE,
        "numerics": fast.NUMERICS, "profile": fast.SCHEMA,
        "objective": "OBSERVED_WDL_CE_AND_TERMINAL_CASH_HUBER_PLUS_PAIRED_MARGIN_EFFECT_HUBER",
        "wdl_coefficient": .1, "cash_coefficient": 1., "effect_coefficient": weight,
        "margin_scale_dollars": 10000., "gradient_weighting": "EQUAL_COMPLETE_STATE_GROUPS; EQUAL_ALTERNATIVES_WITHIN_GROUP",
        "actor_likelihood_loss": False, "behavior_logps": "NONE_AUTHORED_ALTERNATIVES_ARE_NOT_POLICY_SAMPLES",
        "auxiliary": "UPDATE_EXISTING_HEADS_AND_SHARED_ACTOR; PRESERVE_BOTH_OPTIMIZERS_AND_GLOBAL_AGE",
        "plan_learning": copy.deepcopy(original["plan_learning"]),
        "inherited_plan_learning": "HEAD_AND_GAME_RESUME_CONTRACT; ACTUAL_FIT_COEFFICIENTS_ARE_EXPLICIT_ABOVE",
        "resume_game_execution": copy.deepcopy(original), "training_admission": False}
    if objective in ("wdl_only", "wdl_contrast", "wdl_contrast_rank"):
        result.update(effect_objective=objective, effect_objective_schema="farm-observed-plan-wdl-only-v1",
            objective="OBSERVED_WDL_CE_ONLY; CASH_AND_MARGIN_ARE_DIAGNOSTICS",
            cash_coefficient=0., auxiliary="UPDATE_EXISTING_CRITIC_AND_SHARED_ACTOR; FUTURE_GRAD_NONE; PRESERVE_BOTH_OPTIMIZERS_AND_GLOBAL_AGE")
        if objective in ("wdl_contrast", "wdl_contrast_rank"):
            result.update(effect_objective_schema="farm-observed-plan-wdl-contrast-v1",
                objective="OBSERVED_WDL_CE_PLUS_EDIT_TO_KEEP_WDL_SCORE_DELTA_MSE; CASH_AND_MARGIN_ARE_DIAGNOSTICS",
                wdl_contrast_coefficient=1.,
                wdl_contrast_target="ACTUAL_WIN_1_DRAW_0.5_LOSS_0_DIFFERENCE_TO_KEEP; NO_DOLLAR_MAGNITUDE",
                wdl_contrast_reduction="MEAN_NON_KEEP_ALTERNATIVES_THEN_EQUAL_COMPLETE_STATE_GROUPS; BOTH_KEEP_AND_EDIT_GRADIENTS_LIVE")
        if objective == "wdl_contrast_rank":
            result.update(effect_objective_schema="farm-observed-plan-wdl-contrast-rank-v1",
                objective="OBSERVED_WDL_CE_PLUS_KEEP_DELTA_MSE_PLUS_PAIRWISE_SCORE_SHORTFALL; CASH_DIAGNOSTIC",
                wdl_rank_coefficient=1.,
                wdl_rank_target="POSITIVE_OBSERVED_WDL_SCORE_GAP; SQUARED_RELU_TARGET_MINUS_PREDICTED_GAP",
                wdl_rank_reduction="MEAN_STRICTLY_ORDERED_PAIRS_THEN_EQUAL_COMPLETE_GROUPS; ALL_TIES_ZERO; BOTH_GRADIENT_PATHS_LIVE")
    elif weight != 1.:
        result["effect_weight_schema"] = "farm-explicit-paired-margin-effect-weight-v1"
    if precision == "float64":
        result.update(effect_diagnostic_precision=precision,
            effect_diagnostic_schema="farm-wdl-cash-diagnostic-float64-v1",
            diagnostic_semantics="FLOAT64_DECODE_AND_MARGIN_DIAGNOSTICS_ONLY; WDL_LOSS_AND_GRADIENTS_UNCHANGED")
    if mode == "published_plan_encoder":
        result.update(plan_context=mode, plan_context_schema="farm-published-active-plan-context-v1",
            context_semantics="SAME_OBSERVATION_WITH_PUBLISHED_PLAN; IDENTICAL_INPUTS_SHARE_CONTEXT_WITHIN_STATE",
            head_transfer="CHANGED_Q_CONTEXT; GAME_RESUME_PRESERVES_WEIGHTS_BUT_DOES_NOT_CERTIFY_HEAD_CALIBRATION")
    if gain != 1.:
        result.update(plan_context_gain=gain,
            plan_context_gain_schema="farm-keep-relative-plan-context-gain-v1",
            context_transform="KEEP_PLUS_GAIN_TIMES_CANDIDATE_MINUS_KEEP; BOTH_GRADIENT_PATHS_LIVE")
    if readout != "mean":
        result.update(encoder_readout=readout, encoder_readout_schema="farm-equal-entity-kind-readout-v1",
            readout_semantics="MEAN_WITHIN_KIND_THEN_MEAN_OVER_PRESENT_KINDS; STATE_AND_PUBLISHED_PLAN; ALL_VALID_ENTITIES_RETAINED")
    return result


def future_tensors(groups, *, device):
    """Bounded witnessed queries, with explicit zero-weight padding, never actor inputs."""
    import torch
    from farm_neat.policy.plan_learning import TARGETS
    samples = [sample for group in groups for sample in group["future"]]
    q = max(len(s["queries"]) for s in samples)
    if not 1 <= q <= 16: raise ValueError("bounded nonempty future samples required")
    horizon = torch.ones((len(samples), q))
    target = torch.zeros((len(samples), q, len(TARGETS)))
    known = torch.zeros_like(target, dtype=torch.bool)
    weight = torch.zeros_like(horizon)
    # Construct on CPU, then transfer once; no per-scalar GPU writes in the learner.
    hs, ts, ks, ws = horizon, target, known, weight
    for i, sample in enumerate(samples):
        for j, row in enumerate(sample["queries"]):
            hs[i, j], ws[i, j] = row["step"], row["weight"]
            for k, name in enumerate(TARGETS):
                value = row["values"][name]
                if value is not None: ts[i, j, k], ks[i, j, k] = value, True
    return tuple(x.to(device) for x in (hs, ts, ks, ws))


def validate_parent(saved, result, plan, manifest, *, model_sha, profile, parameter_names, target_execution=None):
    """Old parents are checked as old by their own validators; fast parents by this one."""
    target_execution = EXECUTION if target_execution is None else target_execution
    schema = saved.get("schema")
    if schema == OPTIMIZER_SCHEMA:
        # A fast parent is checked as ITS OWN recorded version (pins/execution joined across its
        # optimizer, plan, result and manifest); a later source edit is an explicit transition.
        if saved["execution"].get("supervised_stage"):
            expected = effect_execution(saved["execution"]) if saved["execution"]["supervised_stage"] == EFFECT_STAGE else supervised_execution(saved["execution"])
            if saved["execution"] != expected:
                raise ValueError("exact supervised stage and preserved game contract required")
        validate_settings(saved["settings"]); age = saved["updates"]
        # A parent whose sidecars were source-migrated into this layout still carries its
        # ORIGINAL pins/plan hash/execution inside optimizer.pt, which no tool may rewrite.
        # `joined` is what the optimizer must equal: the recorded pre-migration originals
        # when the record holds, the sidecars themselves otherwise. Nothing is loosened --
        # without a holding record this is the identical comparison it always was.
        from farm_neat.tooling import checkpoint_migration as rebind  # deferred: that module reads sources()
        joined, migrated = rebind.joined_identity(saved, plan, result, manifest, live_pins=sources())
        if (type(age) is not int or age < 1 or age != manifest["optimizer_steps"] or age != result["global_optimizer_updates"]
                or saved["model_sha256"] != model_sha or result["after_model_sha256"] != model_sha
                or result["manifest"] != manifest["content_sha256"] or saved["environment"] != profile or plan["environment"] != profile
                or saved["parameter_names"] != parameter_names or saved["optimizer_config"] != lab.OPTIMIZER_CONFIG
                or saved["settings"] != plan["schedule"] or saved["execution"] != joined["execution"]
                or manifest["lineage"].get("execution") != saved["execution"]
                or result["source_pins"] != plan["source_pins"] or joined["source_pins"] != saved["source_pins"]
                or saved["plan_sha256"] != joined["plan_sha256"] or manifest["lineage"].get("plan_sha256") != result["plan_sha256"]
                or manifest["lineage"].get("continuation_protocol" if saved["execution"].get("supervised_stage") else "ppo_protocol") != SCHEMA
                or saved["last_rollout_iteration"] != manifest["rollout_iteration"]
                or result["status"] != committed_status(saved["execution"]) or saved["execution"].get("numerics") != fast.NUMERICS):
            raise ValueError("exact fast-profile parent model/Adam/age/source/profile join required")
        for group in saved["optimizer"]["param_groups"]:
            if group["lr"] != learning_rate_at(age, saved["settings"]): raise ValueError("saved global-age learning rate required")
        if saved["source_pins"] == sources() and saved["execution"] == target_execution:
            return saved["profile_origin"], saved["recent_episode_ids"]
        transition = {"kind": "EXPLICIT_FAST_PROFILE_SOURCE_VERSION_CONTINUATION", "parent_schema": schema,
            "parent_model_sha256": model_sha, "parent_global_age": age, "old_source_pins": saved["source_pins"],
            "old_execution": saved["execution"], "new_source_pins": sources(), "new_execution": target_execution,
            "lineage_origin": saved["profile_origin"], "source_migration": None if migrated is None else
                {k: migrated[k] for k in ("kind", "reason", "origin", "path_map_entries", "accepted_drift")},
            "optimizer_conversion": "NONE_EXACT_ADAM8BIT_CPU_CUDA_RNG_AGE_AND_SCHEDULE",
            "numerical_change": "SAME_FAST_PROFILE_KERNELS; EXECUTION_AND_DECISION_CLOCK_CHANGES_EXPLICIT_ABOVE"}
        return transition, saved["recent_episode_ids"]
    reference_profile = {k: v for k, v in profile.items() if k not in ("pad_multiple", "numerics")}
    if schema == streaming.OPTIMIZER_SCHEMA:
        origin, ids, _ = streaming.validate_parent(saved, result, plan, manifest, model_sha=model_sha, source=saved["source_pins"],
                                                   profile=reference_profile, parameter_names=parameter_names)
    elif schema == b32fit.OPTIMIZER_SCHEMA:
        origin, ids, _ = b32fit.validate_parent(saved, result, plan, manifest, model_sha=model_sha, source=saved["source_pins"],
                                                profile=reference_profile, parameter_names=parameter_names)
    else:
        origin, ids = legacy.validate_parent(saved, result, plan, manifest, model_sha=model_sha, source=saved["source_pins"],
                                             profile=reference_profile, parameter_names=parameter_names)
    transition = {"kind": "EXPLICIT_REFERENCE_PROFILE_TO_FAST_PROFILE", "parent_schema": schema,
        "parent_model_sha256": model_sha, "parent_global_age": saved["updates"], "old_source_pins": saved["source_pins"],
        "old_execution": saved.get("execution"), "new_source_pins": sources(), "new_execution": target_execution,
        "bc_origin": origin, "optimizer_conversion": "NONE_EXACT_ADAM8BIT_CPU_CUDA_RNG_AGE_AND_SCHEDULE",
        "numerical_change": "FORWARD_ARITHMETIC_ORDER_AND_SAMPLER_STREAM_ONLY; WEIGHTS_BITWISE_UNCHANGED"}
    return transition, ids


def weighted_actor_sum(logp, old, advantage, entropy, weights, denominator, settings):
    """One chunk's contribution to the global weighted actor mean only.

    Positive weights commute with PPO clipping. Entropy and auxiliary losses
    retain their original unweighted decision mean in the caller.
    """
    import math
    import torch
    from farm_neat.policy.plan_learning import grpo_terms
    if (weights.shape != logp.shape or weights.device != logp.device
            or weights.requires_grad or not bool(torch.isfinite(weights).all())
            or not bool((weights > 0).all()) or not math.isfinite(denominator)
            or denominator < float(weights.sum())):
        raise ValueError("positive detached weights and full accumulation denominator required")
    return grpo_terms(logp, old, advantage * weights, entropy,
        clip_ratio=settings["clip_ratio"], entropy_coefficient=0.)["actor_loss"] * len(logp) / denominator


def actor_weight_vectors(groups, learning):
    """Weights from witnessed world roles; complete groups keep one opponent.

    Cohort construction binds these roles from the registered world identities
    after checking every episode's opponent hash. They never enter actor inputs.
    """
    if not learning or learning["schema"] != PLAN_WEIGHTED_SCHEMA:
        return None
    size = learning["group_size"]
    allowed = {p["kind"] for p in learning["opponents"]}
    result = []
    for group in groups:
        kinds = group.get("opponent_kinds")
        if (not isinstance(kinds,list) or not kinds or len(kinds) != len(group["calls"])
                or len(kinds) != len(group["returns"]) or len(kinds) % size
                or group["group_size"] != size or any(k not in allowed for k in kinds)
                or any(len(set(kinds[i:i+size])) != 1 for i in range(0,len(kinds),size))):
            raise ValueError("complete registered same-opponent group roles required for actor weights")
        result.append([float(learning["router_actor_weight"]) if k == "SHOP_ROUTER" else 1. for k in kinds])
    return result


class Learner:
    """Resident model + AdamW8bit + RNG; fits one round of fast cohorts and publishes a child.

    ``stack`` cohorts are concatenated into one forward (padded to the longest bucket);
    with per-block recomputation a whole accumulation group can be one fwd+bwd.
    A frozen behavior copy scores the within-profile parity gate when collection
    overlapped training (the collectors sampled with an earlier child).
    """
    def __init__(self, parent_dir, expected, *, stack=1, checkpoint_blocks=False, opening_only=False, plan_learning=None, shop_revision_every=0, full_plan=False):
        import bitsandbytes as bnb, torch
        self.torch, self.bnb = torch, bnb
        self.learning = validate_plan_learning(plan_learning)
        self.execution = execution_for(opening_only, self.learning, shop_revision_every, full_plan)
        if type(stack) is not int or stack < 1 or type(checkpoint_blocks) is not bool: raise ValueError("explicit stack/checkpoint settings required")
        parent_dir = io.path(ROOT, parent_dir)
        self.profile = environment_profile()
        self.graph, self.manifest = policy.load_model(parent_dir/"model.pt", device="cuda", expected_file_sha256=expected["model_sha256"],
            allow_remaining_window_transition=True)
        if self.graph.config != policy.DEFAULT: raise ValueError("intended full60M16layer graph required")
        saved = torch.load(parent_dir/"optimizer.pt", map_location="cpu", weights_only=True)
        if file_sha(parent_dir/"optimizer.pt") != expected["optimizer_sha256"]: raise ValueError("parent Adam file hash mismatch")
        self.names = [n for n, _ in self.graph.named_parameters()]
        self.origin, self.prior_ids = validate_parent(saved, io.read(parent_dir/"RESULT.json"), io.read(parent_dir/"PLAN.json"), self.manifest,
            model_sha=expected["model_sha256"], profile={**self.profile, "bitsandbytes": str(bnb.__version__)}
            if saved.get("schema") == OPTIMIZER_SCHEMA else {**lab.device_profile(), "bitsandbytes": str(bnb.__version__)},
            parameter_names=self.names, target_execution=self.execution)
        self.settings, self.age = saved["settings"], saved["updates"]
        self.optimizer = bnb.optim.AdamW8bit(self.graph.parameters(), lr=learning_rate_at(self.age, self.settings),
            betas=(.9, .999), eps=1e-8, weight_decay=0., min_8bit_size=4096, block_wise=True)
        lab.restore_bc_optimizer(self.optimizer, legacy._cpu_copy(saved)); self.graph.eval()
        self.parent = dict(expected); self.parent_dir = parent_dir
        self.fast_id, self.head_id = fast.identity(), heads.identity()
        self.stack, self.checkpoint_blocks = stack, checkpoint_blocks
        self.encoder = fast.FastEncoder(self.graph, expected_identity=self.fast_id, checkpoint_blocks=checkpoint_blocks)
        self.parameters = list(self.graph.parameters())
        self.auxiliary, self.aux_optimizer = None, None
        self.auxiliary_continuation = None
        if self.learning:
            from farm_neat.policy.plan_learning import PlanCritic, FutureHeads
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(self.learning["head_seed"])
                self.auxiliary = torch.nn.ModuleDict({"critic": PlanCritic(self.graph.config.width),
                    "future": FutureHeads(self.graph.config.width)}).to("cuda")
            self.aux_optimizer = torch.optim.AdamW(self.auxiliary.parameters(),
                lr=learning_rate_at(self.age, self.settings), betas=(.9, .999), eps=1e-8, weight_decay=0.)
            aux = saved.get("plan_learning_state")
            self.auxiliary_continuation = auxiliary_resume(aux, saved["execution"], self.learning, self.age)
            if saved["execution"].get("plan_learning"):
                self.auxiliary.load_state_dict(aux["heads"], strict=True)
                self.aux_optimizer.load_state_dict(aux["optimizer"])
            self.parameters += list(self.auxiliary.parameters())
        self.behavior_graph, self.behavior_encoder, self.behavior_sha = None, None, None

    def load_behavior(self, directory, model_sha256):
        """Frozen copy of the exact child the collectors sampled with (bitwise the learner's weights when equal)."""
        torch = self.torch
        if model_sha256 == self.parent["model_sha256"] and self.parent_dir == io.path(ROOT, directory):
            self.behavior_sha = model_sha256; return "LEARNER_WEIGHTS_ARE_THE_BEHAVIOR_WEIGHTS"
        graph, _ = policy.load_model(io.path(ROOT, directory)/"model.pt", device="cuda", expected_file_sha256=model_sha256,
            allow_remaining_window_transition=True)
        if self.behavior_graph is None:
            self.behavior_graph = graph
        else:
            self.behavior_graph.load_state_dict(graph.state_dict(), strict=True); del graph
        self.behavior_graph.eval(); self.behavior_sha = model_sha256
        self.behavior_encoder = fast.FastEncoder(self.behavior_graph, expected_identity=self.fast_id)
        return "FROZEN_BEHAVIOR_COPY_LOADED"

    def _scorer(self, behavior):
        if behavior and self.behavior_sha != self.parent["model_sha256"]:
            if self.behavior_graph is None: raise ValueError("behavior copy not loaded")
            return self.behavior_graph, self.behavior_encoder
        return self.graph, self.encoder

    def verify_opponents(self, plan, groups):
        """Fresh GPU resampling after CPU reconstruction, before any optimizer step.

        Uses original cohort shape and independent per-game opponent streams.
        Checks exact programs/factors plus existing logp/value tolerances; does
        not turn opponent calls into auxiliary/GRPO labels or consume learner RNG.
        """
        lanes=opponent_lanes(plan)
        if not lanes or len(groups)!=len(decision_steps(plan)):
            raise ValueError("complete registered opponent cohorts required")
        if self.behavior_sha!=plan["model_sha256"]:
            raise ValueError("registered frozen behavior weights must be loaded")
        graph,encoder=self._scorer(True);torch=self.torch
        streams=[fast.LaneStream(opponent_for(plan,w)["rng_seed"]) for w,s in lanes]
        sampler=fast.GraphedSampler(graph,len(lanes),plan["temperature"])
        started=time.monotonic(); max_logp=max_value=0.0
        with torch.no_grad():
            for step,group in zip(decision_steps(plan),groups,strict=True):
                _equal(group["batch"]["step"],step,"opponent decision time")
                tensors=tuple(torch.as_tensor(group["arrays"][k],device="cuda") for k in ("rows","token_mask","knownness"))
                context,values,_=encoder(*tensors)
                records,_,_=sampler.sample(context,streams,max_commands=plan["command_budget"],as_of_step=step)
                if len(records)!=len(group["calls"]):raise ValueError("opponent batch cardinality")
                for actual,call,value in zip(records,group["calls"],values.tolist(),strict=True):
                    recorded=call["behavior"]
                    for key in actual:
                        if key not in ("joint_logp","selected_logps"):
                            _equal(actual[key],recorded.get(key),"resampled opponent "+key)
                    if len(actual["selected_logps"])!=len(recorded["selected_logps"]):raise ValueError("opponent factor count")
                    max_logp=max(max_logp,abs(actual["joint_logp"]-call["old_log_probability"]),
                        *(abs(x-y) for x,y in zip(actual["selected_logps"],recorded["selected_logps"],strict=True)))
                    max_value=max(max_value,abs(value-call["old_value"]))
        if max_logp>fast.TOLERANCES["within_profile_logp_atol"] or max_value>fast.TOLERANCES["within_profile_value_atol"]:
            raise ValueError("fresh opponent neural reproduction tolerance failed")
        return bound({"schema":SCHEMA+"-neural-opponent-witness","model_sha256":plan["model_sha256"],
            "cohort_plan_sha256":plan["content_sha256"],"decision_calls":len(lanes)*len(groups),
            "opponent_programs_and_factors_exact":True,"max_logp_error":max_logp,"max_value_error":max_value,
            "seconds":time.monotonic()-started,"optimizer_updates":0,"training_rows_added":0})

    def outputs_stacked(self, groups, *, behavior=False, with_plan=False):
        """(logp, value, entropy) for the concatenated cohorts, in group order."""
        torch = self.torch
        graph, encoder = self._scorer(behavior)
        rows, mask, known, _ = fast.stack_arrays([g["arrays"] for g in groups])
        tensors = tuple(torch.as_tensor(x, device="cuda") for x in (rows, mask, known))
        context, value, _ = encoder(*tensors)
        records = [c["behavior"] for g in groups for c in g["calls"]]
        logp, entropy = fast.replay_padded(graph, context, records, expected_identity=self.fast_id)
        if with_plan:
            from farm_neat.policy.plan_learning import plan_context
            return logp, value, entropy, context, plan_context(graph, context, records)
        return logp, value, entropy

    def auxiliary_terms(self, state, plan, groups, returns):
        from farm_neat.policy.plan_learning import plan_critic_terms, future_loss
        horizon, targets, known, weights = future_tensors(groups, device=state.device)
        critic = plan_critic_terms(self.auxiliary["critic"](state, plan), returns)
        future = future_loss(self.auxiliary["future"](state, plan, horizon), targets, known, weights)
        return critic, future

    def measure(self, groups, labels, *, parity=False, behavior=False):
        """Stacked no-grad scoring; the parity gate replays each cohort at its own collection shape."""
        torch = self.torch
        tick = time.monotonic(); outs = [[], [], []]; err = [0.0, 0.0]; start = 0; aux_totals = [0., 0., 0.]
        stride = 1 if parity else self.stack
        with torch.no_grad():
            for i in range(0, len(groups), stride):
                chunk = groups[i:i+stride]
                auxiliary = self.learning is not None and not behavior
                output = self.outputs_stacked(chunk, behavior=behavior, with_plan=auxiliary)
                logp, value, entropy = output[:3]
                if auxiliary:
                    returns = labels["returns"][start:start+len(logp)]
                    critic, future = self.auxiliary_terms(output[3], output[4], chunk, returns)
                    aux_totals = [a+float(b)*len(logp) for a, b in zip(aux_totals, (critic["loss"], critic["brier"], future))]
                for j, x in enumerate((logp, value, entropy)): outs[j].append(x.detach())
                if parity:
                    sel = slice(start, start+int(logp.numel()))
                    err[0] = max(err[0], float((logp-labels["old_logp"][sel]).abs().max()))
                    err[1] = max(err[1], float((value-labels["old_value"][sel]).abs().max()))
                start += int(logp.numel())
        result = legacy.metrics(*(torch.cat(x) for x in outs), labels["old_logp"], labels["returns"])
        result.update({"cohorts": len(groups), "calls": int(labels["returns"].numel()), "stack": stride,
            "scored_by": "FROZEN_BEHAVIOR_COPY" if behavior and self.behavior_sha != self.parent["model_sha256"] else "LEARNER_WEIGHTS",
            "seconds": time.monotonic()-tick, "old_logp_max_error": err[0] if parity else None, "old_value_max_error": err[1] if parity else None})
        if self.learning and not behavior:
            result.update(plan_critic_log_loss=aux_totals[0]/start, plan_critic_brier=aux_totals[1]/start,
                          future_huber=aux_totals[2]/start, legacy_value_head_is_untrained_diagnostic=True)
        return result

    def cross_profile_canary(self, groups):
        """Frozen reference arithmetic on registered cohorts (behavior weights); reported, refused only above the coarse bound."""
        torch = self.torch
        graph, encoder = self._scorer(True)
        worst = {"logp_max_abs": 0.0, "value_max_abs": 0.0, "cohorts": 0, "seconds": 0.0}
        tick = time.monotonic()
        with torch.no_grad():
            for group in groups:
                tensors = tuple(torch.as_tensor(group["arrays"][k], device="cuda") for k in ("rows", "token_mask", "knownness"))
                records = [c["behavior"] for c in group["calls"]]
                ctx_r, val_r, _ = policy.Graph.encode(graph, *tensors); logp_r, _ = graph.replay(ctx_r, records)
                ctx_f, val_f, _ = encoder(*tensors); logp_f, _ = fast.replay_padded(graph, ctx_f, records, expected_identity=self.fast_id)
                worst["logp_max_abs"] = max(worst["logp_max_abs"], float((logp_f-logp_r).abs().max()))
                worst["value_max_abs"] = max(worst["value_max_abs"], float((val_f-val_r).abs().max())); worst["cohorts"] += 1
        worst["seconds"] = time.monotonic()-tick
        if worst["logp_max_abs"] > fast.TOLERANCES["cross_profile_logp_atol"] or worst["value_max_abs"] > fast.TOLERANCES["cross_profile_value_atol"]:
            raise ValueError("cross-profile canary exceeded the preregistered coarse bound")
        return worst

    @staticmethod
    def subset(groups, fraction):
        """Deterministic strided cohort subset for declared partial coverage (fraction 1.0 = all)."""
        count = max(1, int(round(len(groups)*fraction)))
        if count >= len(groups): return list(range(len(groups)))
        return [int(i*len(groups)/count) for i in range(count)]

    def fit(self, groups, *, accumulation, max_updates, metric_every, canary_cohorts, target, parity_fraction=1.0, metric_fraction=1.0):
        torch = self.torch; graph, optimizer, settings = self.graph, self.optimizer, self.settings
        device = torch.device("cuda")
        before = {"model": legacy._cpu_copy(graph.state_dict()), "adam": legacy._cpu_copy(optimizer.state_dict()),
                  "rng": (torch.get_rng_state().clone(), [s.clone() for s in torch.cuda.get_rng_state_all()]), "age": self.age}
        if self.learning:
            before["auxiliary"] = legacy._cpu_copy(self.auxiliary.state_dict())
            before["aux_adam"] = legacy._cpu_copy(self.aux_optimizer.state_dict())
        started = time.monotonic(); phases = {}
        try:
            actor_weights = actor_weight_vectors(groups, self.learning)
            weighted_actor = actor_weights is not None and self.learning["router_actor_weight"] != 1
            offsets, slices = [0], []
            for g in groups: slices.append(slice(offsets[-1], offsets[-1]+len(g["calls"]))); offsets.append(offsets[-1]+len(g["calls"]))
            labels = {"old_logp": torch.tensor([c["old_log_probability"] for g in groups for c in g["calls"]], dtype=torch.float64, device=device),
                      "old_value": torch.tensor([c["old_value"] for g in groups for c in g["calls"]], dtype=torch.float64, device=device),
                      "returns": torch.tensor([r for g in groups for r in g["returns"]], dtype=torch.float64, device=device), "slices": slices}
            def sub(fraction):
                chosen = self.subset(groups, fraction)
                picked = [groups[i] for i in chosen]
                sub_labels = {k: torch.cat([labels[k][slices[i]] for i in chosen]) for k in ("old_logp", "old_value", "returns")}
                return chosen, picked, sub_labels
            parity_ordinals, parity_groups, parity_labels = sub(parity_fraction)
            tick = time.monotonic(); parity = self.measure(parity_groups, parity_labels, parity=True, behavior=True); phases["parity_gate_behavior"] = time.monotonic()-tick
            parity["cohort_ordinals"] = parity_ordinals; parity["coverage_fraction"] = parity_fraction
            if (parity["old_logp_max_error"] > fast.TOLERANCES["within_profile_logp_atol"]
                    or parity["old_value_max_error"] > fast.TOLERANCES["within_profile_value_atol"]):
                raise ValueError("within-profile collect/replay parity gate failed before PPO: logp %.3e value %.3e (%s)"
                                 % (parity["old_logp_max_error"], parity["old_value_max_error"], parity["scored_by"]))
            metric_ordinals, metric_groups, metric_labels = sub(metric_fraction)
            if not self.learning and parity["scored_by"] == "LEARNER_WEIGHTS" and parity_fraction == metric_fraction:
                initial = dict(parity); phases["initial_metrics"] = 0.0
            else:
                tick = time.monotonic(); initial = self.measure(metric_groups, metric_labels); phases["initial_metrics"] = time.monotonic()-tick
            initial["cohort_ordinals"] = metric_ordinals; initial["coverage_fraction"] = metric_fraction
            tick = time.monotonic(); canary = self.cross_profile_canary([groups[i] for i in canary_cohorts]); phases["cross_profile_canary"] = time.monotonic()-tick
            group_diagnostics = []
            if self.learning:
                from farm_neat.policy.plan_learning import group_advantages, margin_group_advantages, grpo_terms
                grouped_advantages = []
                for i, g in enumerate(groups):
                    if g["group_size"] != self.learning["group_size"]:
                        raise ValueError("registered complete GRPO group width required")
                    wdl = labels["returns"][slices[i]].reshape(-1, g["group_size"])
                    if "reward" in self.learning:
                        cash = torch.tensor(g["terminal_cash"], dtype=torch.float64, device=device).reshape(*wdl.shape,2)
                        advantage, diagnostic = margin_group_advantages(wdl, cash, self.learning["reward"])
                    else:
                        advantage, diagnostic = group_advantages(wdl)
                    if self.execution.get("economic_clock", {}).get("shop_revision_every"):
                        diagnostic = {**diagnostic, "decision_step": g["batch"]["step"],
                                      "group_ids": list(dict.fromkeys(g["group_ids"]))}
                    grouped_advantages.append(advantage.flatten()); group_diagnostics.append(diagnostic)
                advantages = torch.cat(grouped_advantages)
            else:
                advantages = labels["returns"]-labels["old_value"]
            raw_std, raw_mean = float(advantages.std(unbiased=False)), float(advantages.mean())
            if not self.learning and settings["normalize_advantages"] and raw_std > 1e-8: advantages = (advantages-raw_mean)/raw_std
            order = torch.randperm(len(groups), device="cpu").tolist()
            history, position, stop = [], 0, "ONE_EPOCH_COMPLETE"
            torch.cuda.reset_peak_memory_stats(); tick = time.monotonic()
            while position < len(order):
                if len(history) >= max_updates: stop = "ROUND_UPDATE_ALLOCATION_COMPLETE"; break
                if self.age >= settings["planned_optimizer_steps"]: stop = "INHERITED_SCHEDULE_COMPLETE"; break
                chosen = order[position:position+accumulation]; count = sum(len(groups[i]["calls"]) for i in chosen)
                actor_denominator = sum(sum(actor_weights[i]) for i in chosen) if actor_weights is not None else count
                t0 = time.monotonic(); totals = [0.0]*5; optimizer.zero_grad(set_to_none=True); forwards = 0
                aux_totals = [0., 0., 0.]
                if self.learning: self.aux_optimizer.zero_grad(set_to_none=True)
                for j in range(0, len(chosen), self.stack):
                    part = chosen[j:j+self.stack]
                    weight = sum(len(groups[i]["calls"]) for i in part)/count
                    part_groups = [groups[i] for i in part]
                    output = self.outputs_stacked(part_groups, with_plan=bool(self.learning))
                    logp, value, entropy = output[:3]
                    old = torch.cat([labels["old_logp"][slices[i]] for i in part]); ret = torch.cat([labels["returns"][slices[i]] for i in part])
                    adv = torch.cat([advantages[slices[i]] for i in part])
                    if self.learning:
                        pg = grpo_terms(logp, old, adv, entropy, clip_ratio=settings["clip_ratio"],
                                        entropy_coefficient=settings["entropy_coefficient"])
                        critic, future = self.auxiliary_terms(output[3], output[4], part_groups, ret)
                        auxiliary_loss = self.learning["critic_coefficient"]*critic["loss"]+self.learning["future_coefficient"]*future
                        terms = (pg["loss"]+auxiliary_loss, pg["actor_loss"], logp.new_zeros(()), pg["entropy"], pg["clip_fraction"])
                        aux_totals = [a+float(b.detach())*weight for a, b in zip(aux_totals, (critic["loss"], critic["brier"], future))]
                    else:
                        terms = legacy.ppo_terms(logp, value, entropy, old, ret, adv, settings)
                    if not bool(torch.isfinite(terms[0])): raise ValueError("nonfinite PPO loss")
                    if weighted_actor:
                        ws = torch.tensor([w for i in part for w in actor_weights[i]], dtype=torch.float64, device=device)
                        actor = weighted_actor_sum(logp, old, adv, entropy, ws, actor_denominator, settings)
                        contributions = (actor + (-settings["entropy_coefficient"]*pg["entropy"]+auxiliary_loss)*weight,
                            actor, terms[2]*weight, terms[3]*weight, terms[4]*weight)
                        if not bool(torch.isfinite(contributions[0])): raise ValueError("nonfinite weighted GRPO loss")
                        contributions[0].backward()
                        totals = [a+float(b.detach()) for a,b in zip(totals,contributions)]
                    else:
                        (terms[0]*weight).backward(); totals = [a+float(b.detach())*weight for a, b in zip(totals, terms)]
                    forwards += 1
                norm = torch.nn.utils.clip_grad_norm_(self.parameters, settings["gradient_clip"], error_if_nonfinite=True)
                lr = learning_rate_at(self.age+1, settings)
                for group in optimizer.param_groups: group["lr"] = lr
                optimizer.step()
                if self.learning:
                    for group in self.aux_optimizer.param_groups: group["lr"] = lr
                    self.aux_optimizer.step()
                self.age += 1; position += len(chosen)
                if any(not bool(torch.isfinite(p).all()) for p in self.parameters): raise ValueError("nonfinite updated weights; round restored")
                row = {"optimizer_step": self.age, "cohort_ordinals": chosen, "microbatches": len(chosen), "forwards": forwards,
                    "effective_batch_calls": count, "learning_rate": lr, "loss": totals[0], "actor_loss": totals[1], "value_loss": totals[2],
                    "conditional_path_entropy_before": totals[3], "clip_fraction": totals[4], "gradient_norm": float(norm),
                    "update_seconds": time.monotonic()-t0, "post_update": None}
                if self.learning:
                    row.update(plan_critic_log_loss=aux_totals[0], plan_critic_brier=aux_totals[1],
                        future_huber=aux_totals[2], critic_used_for_advantages=False,
                        legacy_value_loss=0., auxiliary_head_optimizer_step=self.age)
                if actor_weights is not None:
                    row.update(actor_weight_sum=actor_denominator,
                        router_actor_weight=self.learning["router_actor_weight"],
                        router_decisions=sum(k == "SHOP_ROUTER" for i in chosen for k in groups[i]["opponent_kinds"]),
                        actor_normalization="FULL_ACCUMULATION_WEIGHT_SUM", entropy_auxiliary_normalization="UNCHANGED_CALL_COUNT")
                history.append(row)
                if metric_every and len(history) % metric_every == 0 and position < len(order):
                    row["post_update"] = self.measure(metric_groups, metric_labels)
                    if row["post_update"]["sampled_program_kl_k3"] > settings["target_kl"]: stop = "FULL_ROUND_SAMPLED_KL_STOP_NOT_HARD_BOUND"; break
            phases["updates_including_interim_metrics"] = time.monotonic()-tick
            if not history: raise ValueError("no optimizer update; no completed round fit")
            tick = time.monotonic(); final = self.measure(metric_groups, metric_labels); phases["final_metrics"] = time.monotonic()-tick
            final["cohort_ordinals"] = metric_ordinals; final["coverage_fraction"] = metric_fraction
            if final["sampled_program_kl_k3"] > settings["target_kl"] and stop == "ONE_EPOCH_COMPLETE": stop = "FULL_FINAL_SAMPLED_KL_EXCEEDED_AFTER_UPDATE_NOT_HARD_BOUND"
            changed = [k for k, v in graph.state_dict().items() if not torch.equal(v.detach().cpu(), before["model"][k])]
            if not changed: raise ValueError("unchanged tensors are not a completed fit")
            receipt = {"schema": SCHEMA+"-fit", "execution": copy.deepcopy(self.execution), "optimizer_steps": len(history),
                "global_optimizer_steps_before": before["age"], "global_optimizer_steps_after": self.age, "history": history,
                "parity": parity, "initial": initial, "final": final, "cross_profile_canary": canary, "raw_advantage_std": raw_std, "raw_advantage_mean": raw_mean,
                "behavior_model_sha256": self.behavior_sha, "changed_tensors": len(changed), "stop_reason": stop,
                "coverage": {"total_cohorts": len(groups), "unique_cohorts_used": position,
                "call_fraction": sum(len(groups[i]["calls"]) for i in order[:position])/int(labels["returns"].numel())},
                "accumulation_cohorts": accumulation, "stack_cohorts": self.stack, "checkpoint_blocks": self.checkpoint_blocks,
                "microbatch_lanes": len(groups[0]["calls"]),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()), "phase_seconds": phases,
                "kl_is_sampled_not_exact": True, "post_update_kl_is_not_a_hard_bound": True, "training_admission": False}
            if self.learning:
                initial_groups = [g for g in group_diagnostics if g.get("decision_step", 0) == 0]
                receipt.update(group_diagnostics=group_diagnostics,
                    auxiliary_continuation=self.auxiliary_continuation,
                    informative_groups=sum(g["informative_groups"] for g in initial_groups),
                    tied_groups=sum(g["tied_groups"] for g in initial_groups),
                    changed_auxiliary_tensors=sum(not torch.equal(v.detach().cpu(), before["auxiliary"][k])
                                                  for k, v in self.auxiliary.state_dict().items()),
                    critic_predictions_are_training_labels=False)
            return receipt
        except BaseException:
            optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            graph.load_state_dict(before["model"], strict=True); optimizer.load_state_dict(before["adam"])
            if self.learning:
                self.aux_optimizer.zero_grad(set_to_none=True)
                self.auxiliary.load_state_dict(before["auxiliary"], strict=True)
                self.aux_optimizer.load_state_dict(before["aux_adam"])
            torch.set_rng_state(before["rng"][0]); torch.cuda.set_rng_state_all(before["rng"][1]); self.age = before["age"]
            raise

    def publish(self, target, receipt, plan_sha, admission, episode_ids):
        torch = self.torch; target = Path(target)
        age = receipt["global_optimizer_steps_after"]
        supervised = self.execution.get("supervised_stage") in (SUPERVISED_STAGE, EFFECT_STAGE)
        iteration = self.manifest["rollout_iteration"] + (0 if supervised else 1)
        if supervised and episode_ids:
            raise ValueError("BC must not manufacture fresh RL episode IDs")
        child = policy.save_model(target/"STAGED-model.pt", self.graph, optimizer_steps=age,
            rollout_iteration=iteration, initialization_seed=self.manifest["initialization_seed"],
            lineage={"kind": "CONTINUED_FAST_PROFILE_OBSERVED_PLAN_EFFECTS" if self.execution.get("supervised_stage") == EFFECT_STAGE else "CONTINUED_FAST_PROFILE_OBSERVED_BC" if supervised else (
                         "FRESH_FAST_PROFILE_ROUND_GRPO_AUX" if self.learning else "FRESH_FAST_PROFILE_ROUND_PPO"),
                     ("continuation_protocol" if supervised else "ppo_protocol"): SCHEMA,
                     "parent_file_sha256": self.parent["model_sha256"],
                     "plan_sha256": plan_sha, "admission_sha256": admission["content_sha256"], "execution": self.execution})
        auxiliary = {} if not self.learning else {"plan_learning_state": {"configuration": self.learning,
            "global_age": age, "heads": self.auxiliary.state_dict(), "optimizer": self.aux_optimizer.state_dict()}}
        torch.save({"schema": OPTIMIZER_SCHEMA, "source_pins": sources(), "execution": self.execution, **auxiliary,
            "environment": {**self.profile, "bitsandbytes": str(self.bnb.__version__)}, "updates": age, "settings": self.settings,
            "optimizer": self.optimizer.state_dict(), "optimizer_config": lab.OPTIMIZER_CONFIG, "parameter_names": self.names,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "model_sha256": file_sha(target/"STAGED-model.pt"), "plan_sha256": plan_sha, "profile_origin": self.origin,
            "last_rollout_iteration": iteration,
            "recent_episode_ids": (self.prior_ids+episode_ids)[-4096:]}, target/"STAGED-optimizer.pt")
        # restore check: the staged file reloads to the resident tensors bit for bit
        reloaded = torch.load(target/"STAGED-model.pt", map_location="cpu", weights_only=True)
        if any(not torch.equal(reloaded[k], v.detach().cpu()) for k, v in self.graph.state_dict().items()):
            raise ValueError("staged child does not reload to the resident tensors")
        for old, new in (("STAGED-model.pt", "model.pt"), ("STAGED-model.pt.json", "model.pt.json"), ("STAGED-optimizer.pt", "optimizer.pt")):
            (target/old).rename(target/new)
        self.manifest = policy.read_manifest(target/"model.pt")
        self.parent = {"directory": io.relative(ROOT, target), "model_sha256": file_sha(target/"model.pt"),
            "optimizer_sha256": file_sha(target/"optimizer.pt"), "manifest_sha256": child["content_sha256"], "optimizer_age": age}
        self.prior_ids = (self.prior_ids+episode_ids)[-4096:]; self.parent_dir = target
        return child


# ------------------------------------------------------------------ campaign
def _gpu_watch(stop, samples, interval=2.0):
    while not stop.is_set():
        samples.append({"t": time.monotonic(), **io.gpu_sample()}); stop.wait(interval)


def _phase_gpu(samples, start, end):
    rows = [s for s in samples if start <= s["t"] <= end and "gpu_utilization_percent" in s]
    if not rows: return {"samples": 0}
    util = [r["gpu_utilization_percent"] for r in rows]
    return {"samples": len(rows), "mean_utilization_percent": sum(util)/len(util), "zero_samples": sum(1 for u in util if u == 0),
            "max_memory_MiB": max(r["gpu_memory_MiB"] for r in rows)}


def _storage_gate(plan, round_dir_or_output):
    """Refuse on missing headroom unless the owner waived the cap in the plan; always record the report."""
    report = _storage_check(plan["reserve_bytes"])
    io.write(Path(round_dir_or_output)/f"STORAGE-{int(time.time())}.json", report)
    if report.get("admissible") is not True and not plan.get("storage_cap_waived_by_owner"):
        raise ValueError("storage headroom refused before round: " + json.dumps(report)[:500])
    return report


def _storage_check(reserve):
    out = subprocess.run([sys.executable, "-B", str(ROOT/"RUN.py"), "storage", "check", "--reserve-bytes", str(reserve)],
                         cwd=ROOT, capture_output=True, text=True, timeout=300)
    try: report = json.loads(out.stdout[out.stdout.find("{"):])
    except ValueError: report = {"raw": out.stdout[-2000:], "stderr": out.stderr[-2000:]}
    return report


def prepare(output, *, parent, model_sha256, optimizer_sha256, owner, seed_start, sampler_start, games_per_round,
            processes, rounds, exclusive_seeds=False, updates_per_round=None, accumulation_cohorts=8, metric_every=0,
            canary_cohorts=2, command_budget=8, stage_seconds=1800, max_seconds=3600, reserve_bytes=1_500_000_000,
            witness_workers=4, mode="LOCAL", retain_children=2, worlds_per_process=16, stack_cohorts=None, overlap=True,
            waive_storage_cap=False, parity_fraction=1.0, metric_fraction=1.0, opening_only=False, plan_learning=None, shop_revision_every=0, full_plan=False,
            group_offset_start=0):
    output = io.path(ROOT, output, output=True)
    if output.exists(): raise ValueError("new run directory required; never overwrite a plan")
    learning = validate_plan_learning(plan_learning)
    actions._integer(group_offset_start, 0, actions.MAX_INTEGER, "initial group offset")
    if group_offset_start and learning is None:
        raise ValueError("initial group offset requires grouped learning")
    execution = execution_for(opening_only, learning, shop_revision_every, full_plan)
    clock = {"opening_only": opening_only, **({"shop_revision_every": shop_revision_every} if shop_revision_every else {})}
    if full_plan: clock.update(full_plan=True, plan_adapter=plan_features.identity())
    if (opening_only or shop_revision_every) and (overlap is not False or waive_storage_cap):
        raise ValueError("opening-only requires --no-overlap and the storage cap; fresh same-policy rounds only")
    if not waive_storage_cap and _storage_check(reserve_bytes).get("admissible") is not True:
        raise ValueError("storage headroom refused before preparation; no run directory created")
    if (exclusive_seeds is not True or not isinstance(owner, str) or not owner.strip() or mode not in ("LOCAL", "SMOKE")
            or type(games_per_round) is not int or (not learning and games_per_round % 16) or not 16 <= games_per_round <= 1024
            or type(worlds_per_process) is not int or not 2 <= worlds_per_process <= MAX_WORLDS
            or type(processes) is not int or not 1 <= processes <= 32
            or games_per_round % worlds_per_process or processes > games_per_round//worlds_per_process
            or type(rounds) is not int or not 1 <= rounds <= 1024 or type(witness_workers) is not int or not 1 <= witness_workers <= 32
            or type(retain_children) is not int or retain_children < 1):
        raise ValueError("explicit exclusive owner, complete producer cohorts (legacy 16-world multiples), rounds and bounded worker pools required")
    # Producer cohorts determine grouping and fit weight. Workers only schedule
    # them; a final wave may use fewer workers without changing any cohort.
    current = io.checkpoint(ROOT, parent)
    if current["model_sha256"] != model_sha256 or current["optimizer_sha256"] != optimizer_sha256:
        raise ValueError("external exact parent model/Adam hashes required")
    cohorts_per_round = games_per_round//worlds_per_process*len(decision_steps(clock))
    actions._integer(command_budget, 2, actions.MAX_COMMANDS, "command budget")
    if type(accumulation_cohorts) is not int or not 1 <= accumulation_cohorts <= 64:
        raise ValueError("accumulation_cohorts must be an integer in1..64")
    if stack_cohorts is None: stack_cohorts = accumulation_cohorts
    if type(stack_cohorts) is not int or not 1 <= stack_cohorts <= 64 or type(overlap) is not bool: raise ValueError("explicit stack and overlap settings required")
    if updates_per_round is None: updates_per_round = (cohorts_per_round+accumulation_cohorts-1)//accumulation_cohorts
    if shop_revision_every and updates_per_round*accumulation_cohorts < cohorts_per_round:
        raise ValueError("shop-revision allocation must cover every opening and revision cohort")
    for fraction in (parity_fraction, metric_fraction):
        if type(fraction) is not float or not 0.0 < fraction <= 1.0: raise ValueError("explicit coverage fractions in (0,1] required")
    remaining = current["schedule"]["planned_optimizer_steps"]-current["optimizer_age"]
    if not 0 < updates_per_round*rounds <= remaining: raise ValueError("declared allocation must fit the inherited schedule; no extension")
    if any(i >= cohorts_per_round for i in range(canary_cohorts)): raise ValueError("canary cohorts outside a round")
    games = games_per_round*rounds
    if learning:
        lane_order_for(worlds_per_process, learning)
        if parity_fraction != 1.0 or metric_fraction != 1.0:
            raise ValueError("initial grouped experiment requires complete parity/metric coverage")
    group_size = learning["group_size"] if learning else 1
    actions._integer(group_offset_start+games//group_size-1, 0, actions.MAX_INTEGER, "final group offset")
    lanes_per_world = 1 if learning else 2
    seeds = [seed_start+i//group_size for i in range(games)]
    samplers = list(range(sampler_start, sampler_start+lanes_per_world*games))
    if any(type(v) is not int or not 0 <= v <= actions.MAX_INTEGER for v in seeds):
        raise ValueError("bounded exact engine seed allocation required")
    pins = sources(); attest(pins)
    output.mkdir(parents=True)
    plan = bound({"schema": SCHEMA+"-plan", "owner": owner, "mode": mode, "parent": current, "execution": execution,
        **clock, "plan_learning": learning,
        "decision_steps": list(decision_steps(clock)),
        "source_pins": pins, "profile_identity": fast.identity(), "physical_games": games, "games_per_round": games_per_round,
        "processes": processes, "worlds_per_process": worlds_per_process, "rounds": rounds, "updates_per_round": updates_per_round,
        "stack_cohorts": stack_cohorts, "checkpoint_blocks": stack_cohorts > 1, "overlap": overlap,
        "storage_cap_waived_by_owner": waive_storage_cap is True,
        "parity_fraction": parity_fraction, "metric_fraction": metric_fraction,
        "coverage": "PROSPECTIVE_STRIDED_COHORT_SUBSETS_DECLARED_BEFORE_OUTCOMES; ALL_COHORTS_STILL_WITNESSED_AND_TRAINED",
        "microbatch_lanes": lanes_per_world*worlds_per_process, "cohorts_per_round": cohorts_per_round,
        "distinct_initial_worlds": len(set(seeds)), "learner_episodes_per_world": lanes_per_world,
        "accumulation_cohorts": accumulation_cohorts, "metric_every_updates": metric_every,
        "canary_cohorts": list(range(canary_cohorts)), "command_budget": command_budget, "stage_seconds": stage_seconds,
        "max_seconds": max_seconds, "reserve_bytes": reserve_bytes, "witness_workers": witness_workers,
        "retain_children": retain_children, "retirement": "OLDER_ROUND_CHILD_TENSORS_RETIRED_AFTER_NEWER_CHILD_PUBLISHED; RECEIPTS_KEPT",
        "world_seeds": seeds, "actor_rng_seeds": samplers, "schedule": current["schedule"],
        "seed_ownership": "EXPLICIT_OWNER_ASSERTION_NOT_GLOBAL_EXPOSURE_AUTHENTICATION",
        "automatic_retry": False, "automatic_cleanup": False, "promotion_or_upload": False,
        **({"group_offset_start": group_offset_start} if group_offset_start else {})})
    io.write(output/"PLAN.json", plan)
    return {"plan_file_sha256": io.sha(output/"PLAN.json"), "plan": plan, "executed_games": 0}


def _open(output, expected):
    output = io.path(ROOT, output, output=True)
    if io.sha(output/"PLAN.json") != expected: raise ValueError("externally pinned owner PLAN required")
    plan = io.read(output/"PLAN.json"); _sealed(plan)
    initial_group = plan.get("group_offset_start", 0)
    actions._integer(initial_group, 0, actions.MAX_INTEGER, "initial group offset")
    if initial_group and plan.get("plan_learning") is None:
        raise ValueError("initial group offset requires grouped learning")
    size = plan["plan_learning"]["group_size"] if plan.get("plan_learning") else 1
    actions._integer(initial_group+plan["physical_games"]//size-1, 0, actions.MAX_INTEGER, "final group offset")
    if (plan["schema"] != SCHEMA+"-plan" or plan["execution"] != execution_for(plan["opening_only"], plan.get("plan_learning"), plan.get("shop_revision_every", 0), plan.get("full_plan", False))
            or plan["decision_steps"] != list(decision_steps(plan))):
        raise ValueError("typed fast owner plan with its registered economic clock required")
    if plan.get("full_plan") and plan.get("plan_adapter") != plan_features.identity():
        raise ValueError("full-plan owner input adapter changed")
    if (plan["opening_only"] or plan.get("shop_revision_every")) and (plan["overlap"] or plan.get("storage_cap_waived_by_owner")):
        raise ValueError("opening-only requires nonoverlapped collection and storage headroom")
    attest(plan["source_pins"])
    if plan["profile_identity"] != fast.identity(): raise ValueError("fast profile source changed")
    return output, plan


def status(output, expected):
    output, plan = _open(output, expected)
    rounds = sorted(p.name for p in (output/"rounds").glob("round-*") if (p/"RESULT.json").exists()) if (output/"rounds").exists() else []
    return {"status": "CLOSED" if (output/"RESULT.json").exists() else "FAILED" if (output/"RUN-FAILURE.json").exists() else "OPEN",
            "completed_rounds": rounds, "mode": plan["mode"]}


def _launch_collectors(output, plan, round_index, learner, round_dir):
    """Start K collector subprocesses sampling with the learner's latest published child."""
    parent = learner.parent; parent_dir = io.path(ROOT, parent["directory"])
    manifest = policy.read_manifest(parent_dir/"model.pt", allow_remaining_window_transition=True)
    worlds = plan["worlds_per_process"]; producers, procs, logs, pending = [], [], [], []
    started = time.monotonic()
    for k in range(plan["games_per_round"]//worlds):
        base_index = round_index*plan["games_per_round"]+k*worlds
        learning = plan.get("plan_learning")
        lanes = 1 if learning else 2
        cohort = make_cohort_plan(manifest, parent["model_sha256"], seeds=plan["world_seeds"][base_index:base_index+worlds],
            actor_rng_seeds=plan["actor_rng_seeds"][lanes*base_index:lanes*(base_index+worlds)], command_budget=plan["command_budget"],
            cohort_id=f"{output.name}-r{round_index:03d}-p{k:02d}", environment=environment_profile(),
            source_pins=plan["source_pins"], opening_only=plan["opening_only"], plan_learning=learning,
            group_offset=plan.get("group_offset_start", 0)+base_index//learning["group_size"] if learning else 0,
            shop_revision_every=plan.get("shop_revision_every", 0), full_plan=plan.get("full_plan", False))
        plan_file = round_dir/f"plans/p{k:02d}.json"; io.write(plan_file, cohort)
        producer = round_dir/f"producers/p{k:02d}"; producers.append((producer, cohort["content_sha256"]))
        log = (round_dir/f"logs/collect-p{k:02d}.log"); log.parent.mkdir(exist_ok=True)
        handle = log.open("xb"); logs.append(handle)
        command = [sys.executable, "-B", str(Path(__file__)), "collect", "--execute", "--plan", str(plan_file),
            "--plan-sha256", io.sha(plan_file), "--model", str(parent_dir/"model.pt"), "--output", str(producer)]
        pending.append((command, handle))
    for _ in range(plan["processes"]):
        command, handle = pending.pop(0)
        procs.append(subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT))
    return {"round": round_index, "producers": producers, "procs": procs, "pending": pending, "logs": logs, "started": started,
            "behavior": dict(parent), "behavior_mode": learner.load_behavior(parent["directory"], parent["model_sha256"])}


def _wait_collectors(inflight, stage_seconds):
    deadline = inflight["started"]+stage_seconds
    try:
        for proc in inflight["procs"]:
            try: proc.wait(timeout=max(1.0, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                for p in inflight["procs"]: p.kill()
                raise TimeoutError("collector stage timeout; partial raw retained")
            if proc.returncode:
                raise RuntimeError("a collector failed; inspect rounds/round-*/logs")
            if inflight.get("pending"):
                command, handle = inflight["pending"].pop(0)
                inflight["procs"].append(subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT))
    except BaseException:
        for proc in inflight["procs"]:
            if proc.poll() is None: proc.kill()
        for proc in inflight["procs"]: proc.wait()
        raise
    finally:
        for handle in inflight["logs"]: handle.close()
    if any(p.returncode for p in inflight["procs"]): raise RuntimeError("a collector failed; inspect rounds/round-*/logs")
    return time.monotonic()


def run(output, expected, *, owner, execute=False):
    output, plan = _open(output, expected)
    if execute is not True: return status(output, expected)
    if owner != plan["owner"]: raise ValueError("exact registered owner required")
    if not sys.dont_write_bytecode or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("cold -B CUBLAS:4096:8 and all three thread counts1 required")
    with (output/"OWNER.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/"RESULT.json").exists() or (output/"RUN-FAILURE.json").exists(): raise ValueError("closed/failed run preserved; explicit owner reconciliation required")
        if io.checkpoint(ROOT, plan["parent"]["directory"]) != plan["parent"]: raise ValueError("parent checkpoint changed")
        started = time.monotonic(); samples, stop = [], threading.Event()
        watcher = threading.Thread(target=_gpu_watch, args=(stop, samples), daemon=True); watcher.start()
        rounds_done, completed_games, committed_updates = [], 0, 0
        learner, inflight, worlds = None, None, plan["worlds_per_process"]
        learner_seats = 1 if plan.get("plan_learning") else 2
        try:
            (output/"rounds"/"round-000").mkdir(parents=True); _storage_gate(plan, output/"rounds"/"round-000")
            learner = Learner(plan["parent"]["directory"], plan["parent"], stack=plan["stack_cohorts"],
                              checkpoint_blocks=plan["checkpoint_blocks"], opening_only=plan["opening_only"],
                              plan_learning=plan.get("plan_learning"), shop_revision_every=plan.get("shop_revision_every", 0), full_plan=plan.get("full_plan", False))
            inflight = _launch_collectors(output, plan, 0, learner, output/"rounds"/"round-000")
            for r in range(plan["rounds"]):
                round_dir = output/"rounds"/f"round-{r:03d}"
                current = inflight; inflight = None
                t_round = current["started"]; t_collect = _wait_collectors(current, plan["stage_seconds"])
                behavior = current["behavior"]
                # ---- overlap: the next round's collectors sample with the latest published child while this round is witnessed and fitted
                if plan["overlap"] and r+1 < plan["rounds"] and time.monotonic()-started < plan["max_seconds"]:
                    next_dir = output/"rounds"/f"round-{r+1:03d}"; next_dir.mkdir(parents=True); _storage_gate(plan, next_dir)
                    inflight = _launch_collectors(output, plan, r+1, learner, next_dir)
                # ---- witness (spawned CPU pool, no CUDA)
                jobs = [(str(producer), w, plan_sha) for producer, plan_sha in current["producers"] for w in range(worlds)]
                with multiprocessing.get_context("spawn").Pool(plan["witness_workers"]) as pool:
                    checked = pool.map(_witness_job, jobs, chunksize=1)
                t_witness = time.monotonic()
                # ---- assemble cohorts; admission receipts
                groups, episode_ids, witnesses, collections, peer_cohorts = [], [], [], [], []
                for k, (producer, plan_sha) in enumerate(current["producers"]):
                    cohort = io.read(producer/"PLAN.json"); validate_cohort_plan(cohort)
                    collection = b32._read_pinned(producer/"RESULT.json")[0]; _sealed(collection)
                    for key, value in (("plan_sha256", cohort["content_sha256"]), ("model_sha256", behavior["model_sha256"]), ("physical_games", worlds)):
                        _equal(collection[key], value, "collection " + key)
                    episodes = [e for c in checked[k*worlds:(k+1)*worlds] for e in c["episodes"]]
                    cohort_groups = cohort_batches(cohort, episodes)
                    recorded_cohorts = b32._read_pinned(producer/"COHORTS.json", collection["cohorts_file_sha256"])[0]
                    _equal([g["batch"] for g in cohort_groups], recorded_cohorts["batches"], "fresh original cohorts")
                    peer_episodes = [e for c in checked[k*worlds:(k+1)*worlds] for e in c.get("opponent_episodes", [])]
                    peer_groups = cohort_opponent_batches(cohort, peer_episodes)
                    _equal([g["batch"] for g in peer_groups], recorded_cohorts.get("opponent_batches", []), "fresh opponent cohorts")
                    _equal(collection.get("frozen_opponent_model_decisions", 0), len(peer_episodes)*len(decision_steps(cohort)), "fresh opponent call count")
                    if peer_groups: peer_cohorts.append((cohort, peer_groups))
                    for w, c in enumerate(checked[k*worlds:(k+1)*worlds]):
                        cash = c["engine_witness"]["terminal_cash"]; delta = cash[0]-cash[1]; reward = int(delta > 0)-int(delta < 0)
                        if learner_seats == 2:
                            _equal([episodes[2*w+s]["terminal_reward"] for s in (0, 1)], [reward, -reward], "fresh WDL")
                        else:
                            seat = cohort["lane_order"][w][1]
                            _equal(episodes[w]["terminal_reward"], reward if seat == 0 else -reward, "fresh learner WDL")
                            if "reward" in plan["plan_learning"]:
                                _equal(episodes[w]["terminal_cash"], [cash[seat],cash[1-seat]], "fresh learner terminal cash")
                        witnesses.append(c["engine_witness"])
                    ids = [e["episode_id"] for e in episodes]
                    if len(set(ids)) != learner_seats*worlds or set(ids) & set(learner.prior_ids+episode_ids): raise ValueError("unique unconsumed episode identities required")
                    episode_ids.extend(ids); groups.extend(cohort_groups); collections.append(collection)
                del checked
                if learner.behavior_sha != behavior["model_sha256"]: learner.load_behavior(behavior["directory"], behavior["model_sha256"])
                # Peer programs are not trusted merely because replaying them is legal.
                # Resample every peer decision using its fresh own inputs and frozen
                # behavior weights BEFORE admitting any rows or touching the optimizer.
                neural_witnesses = [learner.verify_opponents(cohort, batches) for cohort, batches in peer_cohorts]
                peer_count = sum(w["decision_calls"] for w in neural_witnesses)
                del peer_cohorts, peer_groups, peer_episodes
                peer_receipt = ({"frozen_opponent_model_decisions": peer_count,
                    "neural_opponent_witnesses": neural_witnesses} if neural_witnesses else {})
                admission = bound({"schema": SCHEMA+"-executed-admission", "round": r, "source_pins": plan["source_pins"],
                    "model_sha256": behavior["model_sha256"], "physical_games": plan["games_per_round"],
                    "engine_reconstruction_games": plan["games_per_round"], "responsive_reference_games": 0, "episode_ids": episode_ids,
                    "engine_witnesses": witnesses, "producer_canaries": [c["sampler_replay_canary"] for c in collections],
                    "behavior_mode": current["behavior_mode"], "caller_boolean_accepted": False, **peer_receipt})
                # ---- fit + publish (parity gate scored by the frozen behavior copy when the learner has moved on)
                fit_dir = round_dir/"fit"; fit_dir.mkdir()
                fit_plan = {"schema": SCHEMA+"-fit-plan", "owner_plan_file_sha256": expected, "round": r, "source_pins": plan["source_pins"],
                    "environment": {**learner.profile, "bitsandbytes": str(learner.bnb.__version__)}, "execution": learner.execution,
                    "schedule": learner.settings, "start_update": learner.age, "parent_model_sha256": learner.parent["model_sha256"],
                    "parent_optimizer_sha256": learner.parent["optimizer_sha256"], "behavior_model_sha256": behavior["model_sha256"],
                    "admission_sha256": admission["content_sha256"], "accumulation_cohorts": plan["accumulation_cohorts"],
                    "stack_cohorts": plan["stack_cohorts"], "checkpoint_blocks": plan["checkpoint_blocks"],
                    "max_updates": plan["updates_per_round"], "profile_origin": learner.origin}
                io.write(fit_dir/"PLAN.json", fit_plan); io.write(fit_dir/"ADMISSION.json", admission); plan_sha = io.sha(fit_dir/"PLAN.json")
                parent = dict(learner.parent)
                receipt = learner.fit(groups, accumulation=plan["accumulation_cohorts"], max_updates=plan["updates_per_round"],
                                      metric_every=plan["metric_every_updates"], canary_cohorts=plan["canary_cohorts"], target=fit_dir,
                                      parity_fraction=plan.get("parity_fraction", 1.0), metric_fraction=plan.get("metric_fraction", 1.0))
                child = learner.publish(fit_dir, receipt, plan_sha, admission, episode_ids)
                learner.torch.cuda.empty_cache()
                t_fit = time.monotonic(); age = receipt["global_optimizer_steps_after"]
                fit_result = {"schema": SCHEMA+"-fit-result", "status": committed_status(learner.execution), "plan_sha256": plan_sha, "source_pins": plan["source_pins"],
                    "optimizer_updates": receipt["optimizer_steps"], "global_optimizer_updates": age, "before_model_sha256": parent["model_sha256"],
                    "after_model_sha256": learner.parent["model_sha256"], "optimizer_sha256": learner.parent["optimizer_sha256"],
                    "manifest": child["content_sha256"], "admission_sha256": admission["content_sha256"], "physical_games": plan["games_per_round"],
                    "engine_reconstruction_games": plan["games_per_round"], "responsive_reference_games": 0,
                    "natural_calls": plan["games_per_round"]*learner_seats*len(decision_steps(plan)),
                    "opening_only": plan["opening_only"], "elapsed_fit_seconds": t_fit-t_witness, "fit": receipt,
                    "population_strength_established": False, **peer_receipt}
                io.write(fit_dir/"STAGED-RESULT.json", fit_result); (fit_dir/"STAGED-RESULT.json").rename(fit_dir/"RESULT.json")
                del groups
                completed_games += plan["games_per_round"]; committed_updates += receipt["optimizer_steps"]
                bytes_written = sum(p.stat().st_size for p in round_dir.rglob("*") if p.is_file())
                previous_end = rounds_done[-1]["clock"]["fit_end"] if rounds_done else started
                summary = {"round": r, "physical_games": plan["games_per_round"], "seat_episodes": 2*plan["games_per_round"],
                    "learner_episodes": learner_seats*plan["games_per_round"],
                    "distinct_initial_worlds": plan["games_per_round"]//plan["plan_learning"]["group_size"] if plan.get("plan_learning") else plan["games_per_round"],
                    "model_decisions": learner_seats*len(decision_steps(plan))*plan["games_per_round"], "opening_only": plan["opening_only"], **peer_receipt,
                    "opening_programs": {"sampled": sum(len(w["opening_programs"]) for w in witnesses),
                        "stop_only": sum(p["stop_only"] for w in witnesses for p in w["opening_programs"]),
                        "whole_plan_refused": sum(p["whole_plan_refused"] for w in witnesses for p in w["opening_programs"]),
                        "programs_with_conflicts": sum(bool(p["conflicting_commands_refused"]) for w in witnesses for p in w["opening_programs"]),
                        "conflicting_commands_refused": sum(p["conflicting_commands_refused"] for w in witnesses for p in w["opening_programs"]),
                        "accepted_tile_orders": sum(p["accepted_tile_orders"] for w in witnesses for p in w["opening_programs"]),
                        "accepted_market_rules": sum(p["accepted_market_rules"] for w in witnesses for p in w["opening_programs"])},
                    "joint_engine_turns": 719*plan["games_per_round"],
                    "microbatch_lanes": learner_seats*worlds, "accumulation_cohorts": plan["accumulation_cohorts"], "stack_cohorts": plan["stack_cohorts"],
                    "optimizer_updates": receipt["optimizer_steps"], "age_before": receipt["global_optimizer_steps_before"], "age_after": age,
                    "behavior_model_sha256": behavior["model_sha256"], "before_model_sha256": parent["model_sha256"],
                    "after_model_sha256": learner.parent["model_sha256"], "optimizer_sha256": learner.parent["optimizer_sha256"],
                    "seconds": {"collect_wall": t_collect-t_round, "witness": t_witness-t_collect, "fit_and_publish": t_fit-t_witness,
                                "round_serial": t_fit-t_round, "publish_to_publish": t_fit-previous_end},
                    "overlapped_collection": plan["overlap"], "clock": {"collect_start": t_round, "collect_end": t_collect, "fit_end": t_fit},
                    "games_per_second": {"collection_only": plan["games_per_round"]/(t_collect-t_round),
                                         "end_to_end_publish_to_publish": plan["games_per_round"]/(t_fit-previous_end)},
                    "gpu": {"collect": _phase_gpu(samples, t_round, t_collect), "witness": _phase_gpu(samples, t_collect, t_witness), "fit": _phase_gpu(samples, t_witness, t_fit)},
                    "collector_phase_seconds": [c["host_wall_phase_seconds"] for c in collections],
                    "collector_peak_cuda_bytes": max(c["peak_cuda_allocated_bytes"] for c in collections),
                    "collector_peak_rss_bytes": max(c["peak_rss_bytes"] for c in collections),
                    "learner_peak_cuda_bytes": receipt["peak_cuda_allocated_bytes"],
                    "learner_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    "round_bytes_written": bytes_written, "stop_reason": receipt["stop_reason"],
                    "parity": receipt["parity"], "initial_metrics": receipt["initial"], "final_metrics": receipt["final"], "cross_profile_canary": receipt["cross_profile_canary"],
                    "terminal_outcomes_not_strength": {"decisive_worlds": sum(1 for w in witnesses if w["terminal_cash"][0] != w["terminal_cash"][1]),
                                                       "draw_worlds": sum(1 for w in witnesses if w["terminal_cash"][0] == w["terminal_cash"][1])}}
                retired = []
                keep = {learner.parent["directory"]} | ({inflight["behavior"]["directory"]} if inflight else set())
                for old in rounds_done[:-(plan["retain_children"]-1)] if plan["retain_children"] > 1 else rounds_done:
                    old_fit = output/"rounds"/f"round-{old['round']:03d}"/"fit"
                    if io.relative(ROOT, old_fit) in keep: continue
                    for name in ("model.pt", "optimizer.pt"):
                        if (old_fit/name).exists():
                            (old_fit/name).unlink(); retired.append(io.relative(ROOT, old_fit/name))
                if retired:
                    io.write(round_dir/"RETIRED.json", {"retired_tensor_files": retired, "reason": "consumed parent; newer child published and reloaded",
                                                        "receipts_kept": True, "raw_reproduction_after_retirement": "UNAVAILABLE_FOR_RETIRED_TENSORS"})
                summary["retired_tensor_files"] = retired
                io.write(round_dir/"RESULT.json", summary); rounds_done.append(summary)
                print(json.dumps({k: v for k, v in summary.items() if k not in ("collector_phase_seconds", "initial_metrics", "final_metrics", "parity", "clock")}), flush=True)
                if inflight is None and r+1 < plan["rounds"]:
                    if time.monotonic()-started >= plan["max_seconds"]: break
                    next_dir = output/"rounds"/f"round-{r+1:03d}"; next_dir.mkdir(parents=True); _storage_gate(plan, next_dir)
                    inflight = _launch_collectors(output, plan, r+1, learner, next_dir)
            if inflight is not None:   # a collection that started but will not be fitted (time boundary); wait so nothing is orphaned
                _wait_collectors(inflight, plan["stage_seconds"])
            final = {"schema": SCHEMA+"-result", "owner_plan_file_sha256": expected,
                "status": "FRESH_FAST_PROFILE_GRPO_AUX_ROUNDS_CLOSED_NOT_STRENGTH" if plan.get("plan_learning") else "FRESH_FAST_PROFILE_PPO_ROUNDS_CLOSED_NOT_STRENGTH",
                "rounds_completed": len(rounds_done), "physical_games": completed_games, "optimizer_updates": committed_updates,
                "child": learner.parent, "elapsed_run_seconds": time.monotonic()-started, "rounds": rounds_done,
                "opening_only": plan["opening_only"], "model_decisions": learner_seats*len(decision_steps(plan))*completed_games,
                **({"frozen_opponent_model_decisions":sum(r.get("frozen_opponent_model_decisions",0) for r in rounds_done)}
                   if any("frozen_opponent_model_decisions" in r for r in rounds_done) else {}),
                "games_per_second_end_to_end_including_learner_startup": completed_games/(time.monotonic()-started) if completed_games else 0.0,
                "uncommitted_collected_games": plan["games_per_round"] if inflight is not None else 0,
                "continued_collection_not_replayed_game_credit": True, "promotion_or_upload": False}
            io.write(output/"STAGED-RESULT.json", final); (output/"STAGED-RESULT.json").rename(output/"RESULT.json")
            return final
        except BaseException as exc:
            if inflight is not None:
                for p in inflight["procs"]: p.kill()
            if not (output/"RUN-FAILURE.json").exists():
                io.write(output/"RUN-FAILURE.json", {"error": type(exc).__name__+": "+str(exc), "owner_plan_file_sha256": expected,
                    "rounds_completed": len(rounds_done), "physical_games": completed_games, "optimizer_updates": committed_updates,
                    "last_committed_child": learner.parent if learner else plan["parent"], "automatic_retry": False, "partial_artifacts_retained": True})
            raise
        finally:
            stop.set()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "status", "run", "collect"))
    p.add_argument("--output"); p.add_argument("--plan-sha256"); p.add_argument("--execute", action="store_true")
    p.add_argument("--owner"); p.add_argument("--parent"); p.add_argument("--model-sha256"); p.add_argument("--optimizer-sha256")
    p.add_argument("--seed-start", type=int); p.add_argument("--sampler-start", type=int); p.add_argument("--exclusive-seeds", action="store_true")
    p.add_argument("--group-offset-start", type=int, default=0,
                   help="explicit reconciled grouped suffix: preserve original seat/opponent rotation; not automatic resume")
    p.add_argument("--games-per-round", type=int, default=64); p.add_argument("--processes", type=int, default=4)
    p.add_argument("--rounds", type=int, default=1); p.add_argument("--updates-per-round", type=int)
    p.add_argument("--accumulation-cohorts", type=int, default=8); p.add_argument("--metric-every", type=int, default=0)
    p.add_argument("--canary-cohorts", type=int, default=2); p.add_argument("--stage-seconds", type=int, default=1800)
    p.add_argument("--max-seconds", type=int, default=3600); p.add_argument("--reserve-bytes", type=int, default=1_500_000_000)
    p.add_argument("--witness-workers", type=int, default=4); p.add_argument("--mode", choices=("LOCAL", "SMOKE"), default="LOCAL")
    p.add_argument("--retain-children", type=int, default=2); p.add_argument("--worlds-per-process", type=int, default=16)
    p.add_argument("--stack-cohorts", type=int); p.add_argument("--no-overlap", action="store_true")
    p.add_argument("--opening-only", action="store_true", help="sample one whole-season program at step0; no later economic revisions")
    p.add_argument("--full-plan", action="store_true", help="whole-plan publication, elapsed history and all-stage input; new explicit execution version")
    p.add_argument("--shop-revision-every", type=int, choices=(2,), default=0,
                   help="opening plus revisions after visible shops2/4/6/8; use instead of --opening-only, with --no-overlap")
    p.add_argument("--plan-learning", help="JSON configuration for grouped opening GRPO + witnessed plan critic/future heads")
    p.add_argument("--command-budget", type=int, default=8, help="maximum commands including STOP; choose explicitly for full opening programs")
    p.add_argument("--waive-storage-cap", action="store_true", help="owner assertion: record storage reports instead of refusing")
    p.add_argument("--parity-fraction", type=float, default=1.0); p.add_argument("--metric-fraction", type=float, default=1.0)
    p.add_argument("--plan"); p.add_argument("--model"); p.add_argument("--canary-every", type=int, default=5)
    a = p.parse_args()
    if a.command == "prepare":
        result = prepare(a.output, parent=a.parent, model_sha256=a.model_sha256, optimizer_sha256=a.optimizer_sha256, owner=a.owner,
            seed_start=a.seed_start, sampler_start=a.sampler_start, games_per_round=a.games_per_round, processes=a.processes,
            rounds=a.rounds, exclusive_seeds=a.exclusive_seeds, updates_per_round=a.updates_per_round,
            accumulation_cohorts=a.accumulation_cohorts, metric_every=a.metric_every, canary_cohorts=a.canary_cohorts,
            stage_seconds=a.stage_seconds, max_seconds=a.max_seconds, reserve_bytes=a.reserve_bytes, witness_workers=a.witness_workers, mode=a.mode,
            retain_children=a.retain_children, worlds_per_process=a.worlds_per_process, stack_cohorts=a.stack_cohorts, overlap=not a.no_overlap,
            waive_storage_cap=a.waive_storage_cap, parity_fraction=a.parity_fraction, metric_fraction=a.metric_fraction,
            opening_only=a.opening_only, command_budget=a.command_budget, shop_revision_every=a.shop_revision_every, full_plan=a.full_plan,
            plan_learning=json.loads(Path(a.plan_learning).read_text()) if a.plan_learning else None,
            group_offset_start=a.group_offset_start)
    elif a.command == "collect":
        _equal(file_sha(a.plan), _sha(a.plan_sha256), "external plan file")
        cohort = validate_cohort_plan(b32._read(a.plan))
        result = collect(a.output, a.model, cohort, canary_every=a.canary_every) if a.execute else {"status": "REGISTERED_NOT_EXECUTED", "physical_games": 0}
    elif a.command == "run" and a.execute:
        result = run(a.output, a.plan_sha256, owner=a.owner, execute=True)
    else:
        result = status(a.output, a.plan_sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False, default=str), flush=True)


if __name__ == "__main__":
    main()
