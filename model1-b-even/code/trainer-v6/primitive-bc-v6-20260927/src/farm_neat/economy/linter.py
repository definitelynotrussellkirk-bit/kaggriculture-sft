"""Actor-visible advice for full economic plans; never actions, masks or labels.

Fixed engine contract:10x10,24 turns/day,last action718,shed100,queue10.
Reuse View's exact walkable-board distances. Route/labor figures are LOWER
bounds, not joint scheduling or fill guarantees: ignore competing duties,
future prices/opponent queues and new yields. No marginal-profit inference.
Only the final day is checked for liquidation; it has no end-of-day auto-drop.
Current snapshot/report facts are not independently witnessed execution proof.
No observation, command, controller counter or report is mutated.
Live animals whose service intent was withdrawn still get consequence advice;
that is not a missed requested duty and never reinstates feeding automatically.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from farm_neat.economy.heuristic.constants import ANIMALS, CROPS, PRODUCTS
from farm_neat.economy.heuristic.state import View, manhattan, nearest_shed_tile

SCHEMA = "farm-full-economic-advisory-lint-v1"
STATUS = "ADVISORY_ONLY_NOT_ACTION_MASK_OR_TRAINING_LABEL"
LAST_STEP = 718
SHED_CAPACITY = 100
QUEUE_LIMIT = 10


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"nonnegative integer {name} required")
    return value


def _end(command):
    return LAST_STEP if command.get("until") == "GAME_END" else command.get("until", LAST_STEP)


def _active(command, step):
    return command.get("start", 0) <= step <= _end(command)


def _finding(code, severity, observed, cause, options, *, tile=None, product=None,
             confidence="OBSERVED_FACT_WITH_CONDITIONAL_CONSEQUENCE"):
    return {"code": code, "severity": severity, "tile": tile, "product": product,
            "observed": copy.deepcopy(observed), "cause": cause,
            "repair_options": list(options), "confidence": confidence}


def _view(observation, snapshot):
    if not isinstance(observation, dict) or not isinstance(snapshot, dict):
        raise ValueError("actor observation and full economic snapshot required")
    if snapshot.get("schema") != "farm-full-economic-orders-v1":
        raise ValueError("full economic snapshot schema required; old29 is incompatible")
    if not isinstance(snapshot.get("orders"), dict) or not isinstance(snapshot.get("rules"), dict):
        raise ValueError("orders/rules required")
    seat = observation.get("player")
    step = _integer(observation.get("step"), "step")
    if type(seat) is not int or seat not in (0, 1) or step > LAST_STEP + 1:
        raise ValueError("actor seat or step outside fixed engine contract")
    if observation.get("day") != step // 24 or observation.get("hour") != step % 24:
        raise ValueError("actor clock disagrees with step")
    farms = observation.get("farms")
    if not isinstance(farms, list) or len(farms) != 2 or not isinstance(farms[seat], dict):
        raise ValueError("two public farms and explicit own farm required")
    v = View(observation)
    if len(v.tiles) != 10 or any(not isinstance(row, list) or len(row) != 10 for row in v.tiles):
        raise ValueError("fixed10x10 own tiles required")
    if not isinstance(v.farm.get("hands"), list):
        raise ValueError("explicit hand positions required")
    private = observation.get("private")
    if not isinstance(private, dict) or any(k not in private for k in ("shed", "seeds", "inventories")):
        raise ValueError("own private stock required; unknown is not zero")
    if not isinstance(private["inventories"], list) or len(private["inventories"]) != v.n_units:
        raise ValueError("one own inventory per existing unit required")
    for stock in [private["shed"], private["seeds"], *private["inventories"]]:
        if not isinstance(stock, dict):
            raise ValueError("stock mapping required")
        for item, count in stock.items():
            if item not in (*PRODUCTS, *ANIMALS):
                raise ValueError("unknown stock item")
            _integer(count, "stock quantity")
    for unit in range(v.n_units):
        position = v.unit_pos(unit)
        if position is None or len(position) != 2 or any(type(x) is not int or not 0 <= x < 10 for x in position):
            raise ValueError("in-bounds own unit position required")
    if not isinstance(v.prices, dict) or any(type(price) not in (int, float) or
            not math.isfinite(price) or price < 0 for price in v.prices.values()):
        raise ValueError("finite observed market prices required")
    return v


def _service(v, snapshot, result):
    deadline = (v.step // 24 + 1) * 24 - 1
    due = []
    for x, y, tile in v.plants() + v.animals():
        animal = "animal" in tile
        species = tile.get("animal" if animal else "crop")
        order = snapshot["orders"].get(f"{x},{y}", {})
        c = order.get("command", {})
        expected_op = "SET_ANIMAL" if animal else "SET_CROP"
        matches = c.get("op") == expected_op and c.get("animal" if animal else "crop") == species
        if not matches or not _active(c, v.step):
            if animal and deadline <= LAST_STEP:
                served, counter = tile.get("fed_today"), tile.get("consecutive_unfed")
                escape = (counter + 1 >= 2 if type(counter) is int and counter >= 0 else None) if served is False else (
                    False if served is True else None)
                result.append(_finding("LIVE_ANIMAL_WITHOUT_FEED_COMMITMENT", "INFO",
                    {"step": v.step, "species": species, "service_requested": False,
                     "current_command": c.get("op"), "served_today": served,
                     "consecutive_miss_counter": counter, "next_daily_refresh_action": deadline,
                     "death_or_escape_if_unserved_at_refresh": escape},
                    "The live animal has no matching active feeding commitment. This may be intentional, "
                    "not an executor failure. If feeding is not restored, two consecutive unfed daily "
                    "refreshes remove the animal. CLEAR cannot DIG a placed animal and does not preserve feeding.",
                    ["If preservation is intended, explicitly restore a compatible animal service commitment.",
                     "If withdrawal is intentional, account for remaining yield and unsold products separately."],
                    tile=[x, y]))
            continue  # Cancelled/expired/replaced intent is not a missed service duty.
        if not animal and c.get("cycles") is not None and order.get("completed_cycles", 0) >= c["cycles"]:
            continue
        served = tile.get("fed_today" if animal else "watered_today")
        if served is True:
            continue
        counter = tile.get("consecutive_unfed" if animal else "consecutive_unwatered")
        policy = c.get("feed" if animal else "watering")
        observed = {"step": v.step, "species": species, "served_today": served,
                    "consecutive_miss_counter": counter, "policy_service": policy,
                    "next_daily_refresh_action": deadline if deadline <= LAST_STEP else None}
        if policy == "OFF":
            result.append(_finding("SERVICE_DISABLED_BY_POLICY", "INFO", observed,
                "Active commitment explicitly disables this service; not an executor failure.",
                ["Revise the service parameter only if preservation is intended."], tile=[x, y]))
            continue
        if served is not False or type(counter) is not int or counter < 0:
            result.append(_finding("SERVICE_STATE_UNKNOWN", "INFO", observed,
                "Missing typed service evidence; no deadline/death claim.",
                ["Expose the actual actor-visible service fields."], tile=[x, y], confidence="UNKNOWN"))
            continue
        if deadline > LAST_STEP or _end(c) < deadline:
            continue  # No daily reset before GAME_END/intent expiry.
        actions = deadline - v.step + 1
        lengths = []
        for unit in range(v.n_units):
            pos = v.unit_pos(unit)
            if not animal or v.unit_inv(unit).get("WHEAT", 0) > 0:
                lengths.append(manhattan(pos, (x, y)) + 1)
            else:
                # Optimistic feed availability; purchase/cash/capacity are NOT proved.
                lengths.append(min(manhattan(pos, s) + 1 + manhattan(s, (x, y)) + 1
                                   for s in ((4, 4), (5, 4), (4, 5), (5, 5))))
        observed.update(actions_left_to_refresh=actions,
                        best_existing_unit_actions_lower_bound=min(lengths),
                        death_or_escape_if_unserved_at_refresh=counter + 1 >= 2)
        due.append((animal, observed))
        result.append(_finding("UNFED_ANIMAL" if animal else "UNWATERED_CROP", "WARNING", observed,
            "Requested service is outstanding. Two consecutive miss-counter increments cause escape/weeds; "
            "a newly planted dry crop starts at1. This is conditional risk, not an already missed deadline.",
            ["Schedule service and necessary carried feed before refresh, respecting competing duties.",
             "Revise the commitment if withholding service is intentional."], tile=[x, y]))
    for x, y in v.weeds():
        order = snapshot["orders"].get(f"{x},{y}", {})
        c = order.get("command", {})
        if c.get("op") != "SET_CROP" or not _active(c, v.step):
            continue
        if c.get("cycles") is not None and order.get("completed_cycles", 0) >= c["cycles"]:
            continue
        result.append(_finding("WEED_BLOCKS_REQUESTED_CROP", "INFO" if order.get("clear_before") else "WARNING",
            {"step": v.step, "crop": c.get("crop"), "clear_before": bool(order.get("clear_before")),
             "completed_cycles": order.get("completed_cycles"), "requested_cycles": c.get("cycles")},
            "Observed weeds prevent planting the active crop request; onset/cause is unknown from one observation.",
            ["Fulfill clearing before planting, or explicitly revise/cancel the request."], tile=[x, y]))
    if due:
        available = deadline - v.step + 1
        lower = len(due)  # One service action per distinct tile; NO summed duplicate travel.
        deficit = max(0, lower - available * v.n_units)
        hands = math.ceil(deficit / (available - 1)) if deficit and available > 1 else None
        feed_need = sum(animal for animal, _ in due)
        wheat = v.shed.get("WHEAT", 0) + sum(v.unit_inv(u).get("WHEAT", 0) for u in range(v.n_units))
        result.append(_finding("SERVICE_LABOR_LOWER_BOUND", "WARNING" if deficit else "INFO",
            {"service_actions_lower_bound": lower, "existing_worker_turn_capacity": available * v.n_units,
             "additional_hands_lower_bound": hands, "hire_can_add_a_turn_before_deadline": available > 1,
             "feed_units_for_current_due_jobs": feed_need, "owned_wheat_units": wheat,
             "owned_feed_shortfall": max(0, feed_need - wheat)},
            "Optimistic aggregate bound ignores travel, hauling, other duties and hiring cash/queue limits. "
            "Hired hands first act next turn; extra hands cannot repair a last-turn deficit.",
            ["Consider additional hands only if this labor constraint matters; the bound is not a guarantee.",
             "Check carried feed, funding and routes separately; labor does not create feed."] ,
            confidence="LOWER_BOUND_NOT_SCHEDULING_CERTIFICATE"))


def _liquidation(v, snapshot, result):
    if v.step < 696:
        return
    remaining = LAST_STEP - v.step + 1
    stock = {p: [] for p in PRODUCTS}
    unsellable = []
    for item, quantity in v.shed.items():
        if quantity:
            (stock[item] if item in stock else unsellable).append(
                {"location": "shed", "quantity": quantity, "unit_actions_lower_bound": 0, "item": item})
    for unit in range(v.n_units):
        pos = v.unit_pos(unit)
        distance = manhattan(pos, nearest_shed_tile(pos, v))
        for item, quantity in v.unit_inv(unit).items():
            if quantity:
                (stock[item] if item in stock else unsellable).append(
                    {"location": f"unit/{unit}", "quantity": quantity, "unit_actions_lower_bound": distance + 1, "item": item})
    for item, quantity in v.seeds.items():
        if quantity:
            unsellable.append({"location": "seed_stock", "quantity": quantity, "item": item})
    for x, y, tile in v.plants() + v.animals():
        animal = "animal" in tile
        item = ANIMALS[tile["animal"]]["product"] if animal else tile["crop"]
        if animal:
            unsellable.append({"location": f"tile/{x},{y}", "quantity": 1, "item": tile["animal"]})
        quantity = tile.get("yield_units", 0)
        if not quantity:
            continue
        mature = animal or v.day - tile.get("planted_day", v.day) >= CROPS[item]["first_yield_day"]
        travel = min(manhattan(v.unit_pos(u), (x, y)) for u in range(v.n_units))
        shed_distance = manhattan((x, y), nearest_shed_tile((x, y), v))
        # A future hand can spawn at shed access and may beat a distant farmer.
        # Credit an optimistic next-turn hire even if its funding is unknown.
        lower = min(travel + 1 + shed_distance + 1, 1 + 2 * shed_distance + 2)
        stock[item].append({"location": f"tile/{x},{y}", "quantity": quantity,
                            "harvestable_now": mature,
                            "unit_actions_lower_bound": lower,
                            "hypothetical_next_turn_hand_considered": True})
    for item, sources in stock.items():
        if not sources:
            continue
        quantity = sum(s["quantity"] for s in sources)
        c = snapshot["rules"].get("SALE:" + item, {}).get("command", {})
        reasons = []
        active = c.get("op") == "SALE" and _active(c, v.step)
        if not active:
            reasons.append("NO_ACTIVE_SALE_REQUEST")
        quote = v.prices.get(item)
        if active and (quote is None or quote <= c["price_above"]):
            reasons.append("CURRENT_PRICE_UNKNOWN_OR_AT_BELOW_POLICY_THRESHOLD")
        reserve = c.get("reserve", 0) if active else None
        ceiling = c.get("quantity", 0) if active else None
        sale_turns = max(0, min(LAST_STEP, _end(c)) - v.step + 1) if active else 0
        if active and reserve:
            reasons.append("POLICY_RESERVE_REMAINS_UNBANKED")
        if active and ceiling * sale_turns < max(0, quantity - reserve):
            reasons.append("SALE_QUANTITY_CEILING_CANNOT_CLEAR_ELIGIBLE_STOCK")
        too_late = sum(s["quantity"] for s in sources if not s.get("harvestable_now", True)
                       or max(1, s["unit_actions_lower_bound"]) > remaining)
        if too_late:
            reasons.append("ROUTE_HARVEST_DROP_OR_MATURITY_EXCEEDS_HORIZON")
        incoming = sum(s["quantity"] for s in sources if s["location"] != "shed")
        if incoming > max(0, SHED_CAPACITY - sum(v.shed.values())):
            reasons.append("DEPOSIT_REQUIRES_SHED_ROOM_BEFORE_MARKET_SALES")
        if remaining == 0:
            reasons.append("GAME_ENDED_UNBANKED_STOCK_HAS_ZERO_TERMINAL_VALUE")
        result.append(_finding("TERMINAL_PRODUCT_INVENTORY", "WARNING" if too_late or
            "SALE_QUANTITY_CEILING_CANNOT_CLEAR_ELIGIBLE_STOCK" in reasons else "INFO",
            {"step": v.step, "remaining_action_turns": remaining, "quantity": quantity,
             "sources": sources, "quantity_unreachable_even_optimistically": too_late,
             "sale_quantity_per_turn": ceiling, "sale_reserve": reserve,
             "remaining_turns_in_active_sale_window": sale_turns,
             "quantity_beyond_sale_window_ceiling": max(0, quantity - reserve - ceiling * sale_turns) if active else None,
             "sale_price_threshold": c.get("price_above"), "current_quote": quote,
             "constraints": reasons, "minimum_market_order_slots_if_sold": 1,
             "maximum_market_order_slots_per_turn": QUEUE_LIMIT,
             "shed_room_now": max(0, SHED_CAPACITY - sum(v.shed.values()))},
            "Only shed PRODUCTS can be sold. Unit DROP precedes market in the same turn, "
            "but requires shed room before sales. No final auto-drop/liquidation. A SELL queue slot "
            "can contain multiple units; price, reserve, quantity, competing queues and route bounds "
            "are separate constraints. Policy withholding is not automatically a mechanical failure.",
            ["Inspect requested sale quantity/reserve/threshold and revise only by explicit policy choice.",
             "Check harvest, travel, deposit capacity and queue scheduling before718; reachability is not a fill guarantee."] ,
            product=item, confidence="OBSERVED_STOCK_AND_OPTIMISTIC_LOWER_BOUNDS"))
    if unsellable:
        result.append(_finding("TERMINAL_UNSELLABLE_ASSETS", "INFO", {"step": v.step, "assets": unsellable},
            "Animal bodies and seed stock are not SELL products. Placed animals cannot be DIG-cleared. "
            "They have zero terminal asset value, but this does not show their earlier investment was unprofitable.",
            ["Inspect earlier acquisition/use timing; do not invent a terminal sale or marginal-profit label."]))


def lint(observation, snapshot, report=None):
    """Return detached advice only. Missing required actor state refuses explicitly."""
    v = _view(observation, snapshot)
    result = []
    _service(v, snapshot, result)
    _liquidation(v, snapshot, result)
    report = snapshot.get("last_report", {}) if report is None else report
    if report:
        if not isinstance(report, dict) or not isinstance(report.get("blocked", []), list):
            raise ValueError("typed executor report required")
        if report.get("step") != v.step:
            result.append(_finding("STALE_EXECUTOR_REPORT", "INFO", {"report_step": report.get("step"), "observation_step": v.step},
                "Report belongs to a different observation; no current failure inferred.",
                ["Inspect that report with its original observation."], confidence="UNKNOWN_CURRENT_CAUSALITY"))
        elif report.get("blocked"):
            result.append(_finding("REPORTED_BLOCKED_JOB_TURNS", "INFO", {"step": v.step, "blocked": report["blocked"],
                "blocked_entries_this_report": len(report["blocked"]), "routes": report.get("routes", [])},
                "Reported pending/blocked entries are job-turn observations, not unique missed duties, "
                "independent replay verification or an optimal-policy counterfactual.",
                ["Separate funding, route, queue and policy constraints using actual successors."],
                confidence="REPORTED_NOT_INDEPENDENTLY_VERIFIED"))
    return result


# A named side-input schema, not an unannounced change to the trained 80-column
# encoder. Human messages and numeric facts below share the same lint findings.
ATTENTION_SCHEMA = "farm-plan-attention-v2"
ATTENTION_CODES = (
    "LIVE_ANIMAL_WITHOUT_FEED_COMMITMENT", "SERVICE_DISABLED_BY_POLICY",
    "SERVICE_STATE_UNKNOWN", "UNFED_ANIMAL", "UNWATERED_CROP",
    "WEED_BLOCKS_REQUESTED_CROP", "SERVICE_LABOR_LOWER_BOUND",
    "TERMINAL_PRODUCT_INVENTORY", "TERMINAL_UNSELLABLE_ASSETS",
    "STALE_EXECUTOR_REPORT", "REPORTED_BLOCKED_JOB_TURNS", "LAST_REVISION_HELD_STOCK")
ATTENTION_CONSTRAINTS = (
    "NO_ACTIVE_SALE_REQUEST", "CURRENT_PRICE_UNKNOWN_OR_AT_BELOW_POLICY_THRESHOLD",
    "POLICY_RESERVE_REMAINS_UNBANKED", "SALE_QUANTITY_CEILING_CANNOT_CLEAR_ELIGIBLE_STOCK",
    "ROUTE_HARVEST_DROP_OR_MATURITY_EXCEEDS_HORIZON",
    "DEPOSIT_REQUIRES_SHED_ROOM_BEFORE_MARKET_SALES",
    "GAME_ENDED_UNBANKED_STOCK_HAS_ZERO_TERMINAL_VALUE",
    "NO_REMAINING_SALE_STAGE", "EVERY_REMAINING_SALE_STAGE_RETAINS_STOCK")
ATTENTION_NUMBERS = (
    ("step", 720), ("report_step", 720), ("report_age", 720),
    ("next_daily_refresh_action", 720), ("actions_left_to_refresh", 24),
    ("served_today", 1), ("consecutive_miss_counter", 2),
    ("death_or_escape_if_unserved_at_refresh", 1),
    ("best_existing_unit_actions_lower_bound", 24),
    ("service_actions_lower_bound", 100), ("existing_worker_turn_capacity", 2400),
    ("additional_hands_lower_bound", 100),
    ("hire_can_add_a_turn_before_deadline", 1),
    ("feed_units_for_current_due_jobs", 100), ("owned_wheat_units", 100),
    ("owned_feed_shortfall", 100), ("remaining_action_turns", 720),
    ("quantity", 100), ("quantity_unreachable_even_optimistically", 100),
    ("sale_quantity_per_turn", 100), ("sale_reserve", 100),
    ("remaining_turns_in_active_sale_window", 720),
    ("quantity_beyond_sale_window_ceiling", 100),
    ("sale_price_threshold", 1000), ("current_quote", 1000),
    ("shed_room_now", 100), ("blocked_entries_this_report", 100),
    ("last_revision_action", 720), ("remaining_sale_stage_count", 100),
    ("minimum_future_sale_reserve", 100))
ATTENTION_FIELDS = (
    *(("code:" + c, 1) for c in ATTENTION_CODES),
    ("warning", 1), ("x", 9), ("y", 9),
    *(("item:" + p, 1) for p in (*PRODUCTS, *ANIMALS)),
    *(("constraint:" + c, 1) for c in ATTENTION_CONSTRAINTS),
    *ATTENTION_NUMBERS)
ATTENTION_SPEC = {
    "schema": ATTENTION_SCHEMA, "fields": ATTENTION_FIELDS,
    "numeric_units": "turns, worker-turns, item units, hands, bank dollars, boolean indicators; divide by field scale; no clipping",
    "missing": "zero storage AND false knownness, never a known zero",
    "row_order": "lint traversal order; one row per finding, no feature truncation",
    "scope": "CURRENT_SERVICE; LAST_REVISION_HELD_STOCK_AND_REMAINING_SALE_TERMS; FINAL_DAY_STOCK; NO_FUTURE_WORLD_SIMULATION",
    "visibility": "own observation and plan only; no hidden opponent or future outcomes",
    "meaning": "conditional mechanical risks and optimistic bounds, not profitability, masks, rewards or optimal repairs",
    "consumer": "human diagnostics and full-season entity v4 adapter; older v3 models/runs did not consume these rows",
}
ATTENTION_SPEC_SHA256 = hashlib.sha256(json.dumps(ATTENTION_SPEC, sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _last_revision_stock(v, plan, findings):
    """Expose a remaining-plan fact while edits are still possible, not a cash forecast.

    Additional harvest, consumption, market responses and changed commitments
    could change stock. A reserve is therefore NOT a predicted terminal loss.
    Future zero-reserve stages are considered; active-only inspection is unsound.
    """
    last = plan.publication_steps[-1]
    if plan.committed_step is None or not last <= v.step < 696:
        return
    for product in PRODUCTS:
        quantity = v.shed.get(product, 0) + sum(v.unit_inv(u).get(product, 0) for u in range(v.n_units))
        if not quantity:
            continue
        stages = [r["command"] for r in plan.executor.rule_stages.get("SALE:" + product, [])
                  if _end(r["command"]) >= v.step and r["command"].get("start", 0) <= LAST_STEP]
        minimum = min((c["reserve"] for c in stages), default=None)
        if minimum == 0:
            continue  # A release exists, not a guarantee of quantity/price/route fills.
        reasons = ["NO_REMAINING_SALE_STAGE" if not stages else "EVERY_REMAINING_SALE_STAGE_RETAINS_STOCK"]
        findings.append(_finding("LAST_REVISION_HELD_STOCK", "INFO",
            {"step": v.step, "last_revision_action": last, "quantity": quantity,
             "remaining_sale_stage_count": len(stages), "minimum_future_sale_reserve": minimum,
             "remaining_sale_terms": stages, "constraints": reasons},
            "At or after the last scheduled revision, own held stock has no remaining sale stage "
            "or every remaining sale stage retains a positive reserve. This may intentionally support "
            "feed/other uses. Future production/consumption and prices are not predicted; this is not "
            "a terminal-loss certificate. Consider the terms now, not only on the final day.",
            ["If eventual sale is intended, inspect future quantity, threshold and reserve terms before publication.",
             "If stock is reserved for service, retain that explicit economic choice."],
            product=product, confidence="OBSERVED_STOCK_AND_REMAINING_PLAN_TERMS_NOT_OUTCOME_FORECAST"))


def attention(observation, snapshot, report=None, *, max_messages=6):
    """One current-world diagnostic, readable and numeric; never mutate/advance.

    FullSeasonPlan inputs must already have observed this exact step/seat. Service
    checks use active orders; last-revision stock checks also inspect future sale
    terms. No future world is simulated; all stages and frozen past are untouched.
    The returned matrix is NOT compatible with v3 packs just because it is numeric.
    Features are complete; max_messages truncates human prose only, with a count.
    """
    from farm_neat.economy.orders import EconomicOrders, FullSeasonPlan

    if type(max_messages) is not int or not 1 <= max_messages <= 100:
        raise ValueError("max_messages must be an integer from1to100")
    plan = None
    if isinstance(snapshot, FullSeasonPlan) or (
            isinstance(snapshot, dict) and snapshot.get("schema") == FullSeasonPlan.SCHEMA):
        plan = snapshot if isinstance(snapshot, FullSeasonPlan) else FullSeasonPlan.restore(snapshot)
        if (plan.executor.stage_step != observation.get("step") or
                plan.executor.player != observation.get("player")):
            raise ValueError("observe current step and seat before plan attention")
        snapshot = EconomicOrders.snapshot(plan.executor)
    report = snapshot.get("last_report", {}) if report is None else report
    if (isinstance(report, dict) and type(report.get("step")) is int and
            report["step"] > observation["step"]):
        raise ValueError("future executor report is not current actor evidence")
    findings = lint(observation, snapshot, report)
    if plan is not None:
        _last_revision_stock(_view(observation, snapshot), plan, findings)
    rows, knownness, messages = [], [], []
    step = observation["step"]
    for finding in findings:
        code, facts = finding["code"], finding["observed"]
        if code not in ATTENTION_CODES:
            raise ValueError("version attention schema for new finding code")
        values = {"code:" + c: c == code for c in ATTENTION_CODES}
        values.update(warning=finding["severity"] == "WARNING")
        if finding["tile"] is not None:
            values.update(zip(("x", "y"), finding["tile"]))
        item = finding["product"] or facts.get("species") or facts.get("crop")
        if item is not None:
            if item not in (*PRODUCTS, *ANIMALS):
                raise ValueError("known attention item required")
            values.update({"item:" + p: p == item for p in (*PRODUCTS, *ANIMALS)})
        if "constraints" in facts:
            if set(facts["constraints"]) - set(ATTENTION_CONSTRAINTS):
                raise ValueError("version attention schema for new constraint")
            values.update({"constraint:" + c: c in facts["constraints"] for c in ATTENTION_CONSTRAINTS})
        values.update({name: facts.get(name) for name, _ in ATTENTION_NUMBERS})
        values["step"] = step
        # Preserve report age, not a fabricated current blocked-job diagnosis.
        if type(facts.get("report_step")) is int:
            values["report_age"] = step - facts["report_step"]
        deadline = facts.get("next_daily_refresh_action")
        if deadline is not None:
            values["actions_left_to_refresh"] = deadline - step + 1
        for value in values.values():
            if value is not None and (type(value) not in (int, float, bool) or not math.isfinite(value)):
                raise ValueError("finite typed attention facts required")
        rows.append([0.0 if values.get(name) is None else float(values[name]) / scale
                     for name, scale in ATTENTION_FIELDS])
        knownness.append([values.get(name) is not None for name, _ in ATTENTION_FIELDS])
        location = "" if finding["tile"] is None else " at " + str(tuple(finding["tile"]))
        detail = []
        if item: detail.append(item)
        if deadline is not None: detail.append(f"next daily refresh: action {deadline}")
        if facts.get("death_or_escape_if_unserved_at_refresh") is True:
            detail.append("lost at refresh if still unserved")
        if facts.get("quantity") is not None: detail.append(f"{facts['quantity']} units")
        if code == "LAST_REVISION_HELD_STOCK":
            detail.append(f"last revision: action {facts['last_revision_action']}")
            detail.append(f"remaining sale stages: {facts['remaining_sale_stage_count']}")
            if facts["minimum_future_sale_reserve"] is not None:
                detail.append(f"each retains at least {facts['minimum_future_sale_reserve']} units")
        if facts.get("owned_feed_shortfall") is not None:
            detail.append(f"owned feed shortfall: {facts['owned_feed_shortfall']}")
        if facts.get("service_actions_lower_bound") is not None:
            detail.append(f"at least {facts['service_actions_lower_bound']} service actions; "
                          f"{facts['existing_worker_turn_capacity']} worker-turns available")
        if code == "STALE_EXECUTOR_REPORT":
            detail.append(f"report action {facts['report_step']}; not a current fault")
        detail.extend(facts.get("constraints", []))
        messages.append(code.replace("_", " ").lower() + location +
                        (": " + "; ".join(detail) if detail else ""))
    return {"schema": ATTENTION_SCHEMA, "spec_sha256": ATTENTION_SPEC_SHA256,
        "step": step, "seat": observation["player"], "plan_revision": snapshot["revision"],
        "scope": ATTENTION_SPEC["scope"],
        "status": "ATTENTION_ITEMS_PRESENT" if findings else "NO_ITEMS_IN_IMPLEMENTED_CHECKS",
        "headline": f"Current plan: {len(findings)} diagnostic items. "
                    "These checks do not certify the whole plan or prescribe an economic response.",
        "messages": messages[:max_messages], "omitted_messages": max(0, len(messages)-max_messages),
        "findings": findings,
        "features": {"fields": [name for name, _ in ATTENTION_FIELDS],
                     "scales": [scale for _, scale in ATTENTION_FIELDS],
                     "shape": [len(rows), len(ATTENTION_FIELDS)], "rows": rows,
                     "knownness": knownness},
        "training_integration": "FULL_SEASON_ENTITY_V4_INPUT_SOURCE_NOT_PROOF_OF_TRAINING"}


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Read-only current-plan status and named numeric attention features")
    parser.add_argument("--observation", type=Path, required=True, help="actor-visible observation JSON")
    parser.add_argument("--plan", type=Path, required=True, help="observed FullSeasonPlan or active EconomicOrders snapshot JSON")
    parser.add_argument("--max-messages", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(attention(json.loads(args.observation.read_text()), json.loads(args.plan.read_text()),
        max_messages=args.max_messages), indent=2, allow_nan=False))
