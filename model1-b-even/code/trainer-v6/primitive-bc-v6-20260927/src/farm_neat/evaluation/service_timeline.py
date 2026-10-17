"""Offline, writer-bound service evidence. Does not run an engine or an executor.

Requests, dispatch, input transport and completed service are distinct. An absent
service is censored, not an invented delay. Daily tasks and asset generations
are kept separate so an escaped sheep is never joined to its replacement.
"""
from __future__ import annotations

from collections import Counter
import copy
from decimal import Decimal
import hashlib
import json
from typing import Any

LAST_STEP = 718
DAILY = {"WATER", "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def cents(value: Any) -> int:
    require(type(value) in (int, float), "money must be a finite number")
    d = Decimal(str(value)) * 100
    require(d.is_finite() and d == d.to_integral_value(), "money must resolve to cents")
    return int(d)


def cash_prefix_audit(start_cash: float, deltas: list[float], reserve: float) -> dict:
    """Audit an ORDERED realized cash tape, never an unordered aggregate.

    This proves only the supplied path's cash floor. It cannot certify future
    fills, obligations, worker capacity or survival. A raised floor may already
    be unmet before any purchase; that is distinguished from a spending breach.
    """
    cash, floor = cents(start_cash), cents(reserve)
    require(floor >= 0, "negative reserve")
    require(cash >= 0, "negative starting cash")
    path = [cash]
    for change in deltas:
        cash += cents(change)
        require(cash >= 0, "negative realized cash")
        path.append(cash)
    breaches = [i for i, v in enumerate(path) if v < floor]
    return {"start_cash_cents": path[0], "reserve_cents": floor,
            "end_cash_cents": cash, "minimum_cash_cents": min(path),
            "first_breach_prefix": breaches[0] if breaches else None,
            "floor_met_at_start": path[0] >= floor,
            "safe_on_supplied_path": not breaches,
            "scope": "ORDERED_SUPPLIED_PATH_ONLY_NOT_PLAN_FEASIBILITY"}


def cash_flow_segment(deltas_cents: list[int]) -> tuple[int, int]:
    """Return (net change, least opening working capital), in integer cents.

    Hypothetical committed debits may make a prefix negative. Unlike
    cash_prefix_audit this computes what funding a fixed full-fill path WOULD
    require; it does not assert that the path happened or will fill.
    """
    net = required = 0
    for delta in deltas_cents:
        require(type(delta) is int, "cash deltas must be integer cents")
        net += delta
        required = max(required, -net)
    return net, required


def compose_cash_segments(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    """Concatenate chronological blocks, not interleaved dependency groups.

    (d1, k1) * (d2, k2) = (d1+d2, max(k1, k2-d1)). This associative
    two-integer summary preserves net cash and every fixed-path funding test.
    It does NOT preserve event time, prices, custody, fills or group interleaving.
    """
    for pair in (left, right):
        require(isinstance(pair, (tuple, list)) and len(pair) == 2 and
                all(type(x) is int for x in pair), "invalid cash segment")
        require(pair[1] >= max(0, -pair[0]), "inconsistent cash segment")
    return left[0] + right[0], max(left[1], right[1] - left[0])


def cash_funding_bound(start_cash_cents: int, events: list[dict], *, groups: list[str],
                       policy_floor_cents: int = 0, reservations_cents: dict | None = None,
                       sunk_cost_cents: int = 0, remaining_cycles: dict | None = None,
                       last_step: int = LAST_STEP, scope_complete: bool = False,
                       allow_evaluator_inputs: bool = False) -> dict:
    """Conditional funding envelope for explicitly authorized remaining flows.

    Each event has id, group, object_id, generation, seat, obligation,
    authorization, prerequisites, visibility, kind, basis, earliest, latest,
    amount_min_cents and amount_max_cents. Moments are [step, market queue index,
    unit index], at MARKET_COMMIT (not production/deposit/town/day-end). All events
    belong to one seat. No unit/world simulator, price model or planner runs here.

    SPEND is REALIZED or COMMITTED; RECEIPT is REALIZED, GUARANTEED or FORECAST.
    REALIZED is exact. A GUARANTEED receipt is conditional on its supplied
    prerequisites/bounds, NOT inferred from biological output. FORECAST must have
    zero lower receipt. An absent/latest-after-horizon receipt cannot fund a
    guaranteed path. An unknown spend ceiling gives no finite sufficient bound.

    Debit upper bounds at earliest times and receipt lower bounds at latest times
    give a lower cash path. The opposite endpoints give an upper path. Ties are
    adverse for the sufficient bound and favorable for the necessary bound.
    K=-min(0, cumulative changes) bounds required working capital. Rectangular
    endpoints need not be jointly attainable; a gap is UNKNOWN, not infeasibility.

    External named reservations protect other groups' maximum net withdrawals;
    included groups' allocations are NOT subtracted again as costs. Joint flows
    must be merged in actual time order before reducing. scope_complete explicitly
    attests that all remaining in-scope cash obligations are included and outside
    withdrawals cannot exceed their protected allocations. No reservation is
    created/enforced by this read-only function. Sunk costs before the supplied
    opening balance are metadata, never debited twice. Remaining cycles are
    caller-declared context, not inferred production or terminal salvage.
    """
    def nonnegative(value, name):
        require(type(value) is int and value >= 0, name + " must be nonnegative integer cents/count")
    for name, value in (("start cash", start_cash_cents), ("floor", policy_floor_cents),
                        ("sunk cost", sunk_cost_cents)):
        nonnegative(value, name)
    require(type(last_step) is int and 0 <= last_step <= LAST_STEP, "invalid horizon")
    require(type(scope_complete) is bool and type(allow_evaluator_inputs) is bool, "boolean scope flags required")
    require(isinstance(groups, list) and groups and all(isinstance(g, str) and g.strip() for g in groups)
            and len(set(groups)) == len(groups), "unique nonempty groups required")
    require(isinstance(events, list) and len(events) <= 10000, "at most 10000 explicit cash events")
    reservations = dict(reservations_cents or {})
    for group, amount in reservations.items():
        require(isinstance(group, str) and bool(group.strip()), "reservation owner required")
        nonnegative(amount, "reservation")
    cycles = dict(remaining_cycles or {})
    for group, n in cycles.items():
        require(group in groups, "cycle count outside scope")
        nonnegative(n, "remaining cycles")
    horizon = (last_step, 9, 99998)

    def moment(value):
        require(isinstance(value, (list, tuple)) and len(value) == 3 and
                all(type(x) is int for x in value), "moment must be [step, queue, unit]")
        require(value[0] >= 0 and 0 <= value[1] < 10 and 0 <= value[2] < 99999, "invalid market moment")
        return tuple(value)

    pessimistic, optimistic, unbounded, unguaranteed = [], [], [], []
    ids, exact_slots, seats = set(), set(), set()
    actual_spend = committed_min = receipt_min = 0
    for event in events:
        require(isinstance(event, dict), "cash event must be an object")
        for field in ("id", "group", "object_id", "obligation", "authorization"):
            require(isinstance(event.get(field), str) and bool(event[field].strip()), "missing " + field)
        eid = event["id"]
        require(eid not in ids, "duplicate event id"); ids.add(eid)
        require(event["group"] in groups, "event outside included groups")
        nonnegative(event.get("generation"), "object generation")
        require(type(event.get("seat")) is int and event["seat"] in (0, 1), "explicit own seat required")
        seats.add(event["seat"])
        require(len(seats) == 1, "cannot pool opponents' bank accounts")
        require(isinstance(event.get("prerequisites"), list) and
                all(isinstance(x, str) and x.strip() for x in event["prerequisites"]), "prerequisite ids required")
        visibility = event.get("visibility")
        require(visibility in {"ACTOR_VISIBLE", "DECLARED_BOUND", "EVALUATOR_ONLY"}, "explicit information boundary required")
        require(visibility != "EVALUATOR_ONLY" or allow_evaluator_inputs, "evaluator-only future cannot become actor input")
        early = moment(event.get("earliest"))
        late = moment(event["latest"]) if event.get("latest") is not None else None
        require(late is None or early <= late, "reversed timing window")
        low, high = event.get("amount_min_cents"), event.get("amount_max_cents")
        nonnegative(low, "minimum flow")
        if high is not None:
            nonnegative(high, "maximum flow")
            require(high >= low, "reversed amount interval")
        kind, basis = event.get("kind"), event.get("basis")
        require(kind in {"SPEND", "RECEIPT"}, "only bank-cash commits are cash events")
        if basis == "REALIZED":
            require(early == late and low == high, "realized flows must be exact")
        if early == late:
            require(early not in exact_slots, "two exact commits share one native slot")
            exact_slots.add(early)
        if kind == "SPEND":
            require(basis in {"REALIZED", "COMMITTED"}, "spend needs realized/committed basis")
            require(late is not None and late <= horizon, "remaining spend must have an in-horizon deadline")
            optimistic.append((late, -low, eid))
            if high is None:
                unbounded.append(eid)
            else:
                pessimistic.append((early, -high, eid))
            if basis == "REALIZED":
                actual_spend += low
            else:
                committed_min += low
        else:
            require(basis in {"REALIZED", "GUARANTEED", "FORECAST"}, "receipt needs explicit basis")
            require(high is not None, "receipt requires a finite upper bound")
            require(basis != "FORECAST" or low == 0, "forecast receipt lower bound must be zero")
            require(basis != "GUARANTEED" or (late is not None and bool(event["prerequisites"])),
                    "guaranteed receipt needs delivery/fill prerequisites and a deadline")
            if early <= horizon:
                optimistic.append((early, high, eid))
            if late is not None and late <= horizon:
                pessimistic.append((late, low, eid))
                receipt_min += low
            if low == 0 or late is None or late > horizon:
                unguaranteed.append(eid)

    def reduce(rows, adverse):
        # Uncertain intervals can overlap. Never net a tied receipt against a
        # debit before checking the adverse within-slot prefix.
        rows = sorted(rows, key=lambda r: (r[0], r[1] >= 0 if adverse else r[1] < 0, r[2]))
        net = needed = 0
        binding = None
        for when, change, eid in rows:
            net += change
            if -net > needed:
                needed, binding = -net, {"event": eid, "moment": list(when)}
        return net, needed, binding

    best_net, lower, lower_at = reduce(optimistic, False)
    worst_net, upper, upper_at = reduce(pessimistic, True)
    if unbounded:
        upper = upper_at = worst_net = None
    external = sum(v for g, v in reservations.items() if g not in groups)
    internal = sum(v for g, v in reservations.items() if g in groups)
    protected = policy_floor_cents + external
    required_lower = protected + lower
    required_upper = protected + upper if upper is not None else None
    if not scope_complete:
        status = "UNKNOWN"
    elif start_cash_cents < required_lower:
        status = "INSUFFICIENT"
    elif required_upper is not None and start_cash_cents >= required_upper:
        status = "SUFFICIENT"
    else:
        status = "UNKNOWN"
    return {"schema": "farm-cash-funding-bound-v1", "status": status,
            "scope": "CONDITIONAL_FUNDING_ONLY_NOT_DELIVERY_PROFIT_OR_WDL", "scope_complete": scope_complete,
            "units": "integer cents; [zero-based step, own market queue index, unit index]",
            "groups": sorted(groups), "seat": next(iter(seats), None), "event_count": len(events),
            "last_step": last_step, "start_cash_cents": start_cash_cents,
            "policy_floor_cents": policy_floor_cents, "external_protected_cents": external,
            "included_named_reservation_cents": internal,
            "all_named_reservations_fit": protected + internal <= start_cash_cents,
            "included_reservation_covers_upper": internal >= upper if upper is not None else None,
            "working_capital_lower_cents": lower, "working_capital_upper_cents": upper,
            "required_start_lower_cents": required_lower, "required_start_upper_cents": required_upper,
            "binding_lower": lower_at, "binding_upper": upper_at,
            "terminal_net_lower_cents": worst_net, "terminal_net_upper_cents": best_net,
            "realized_spend_cents": actual_spend, "committed_spend_min_cents": committed_min,
            "guaranteed_receipt_min_cents": receipt_min,
            "sunk_cost_excluded_cents": sunk_cost_cents, "remaining_cycles": cycles,
            "unknown_spend_ceiling_ids": sorted(unbounded), "unguaranteed_receipt_ids": sorted(unguaranteed),
            "evaluator_inputs_allowed": allow_evaluator_inputs}
def cash_funding_orders(start_cash_cents: int, events: list[dict], *, groups: list[str],
                        precedence: list[list[str]], bounds_cover_orders: bool = False,
                        max_states: int = 65536, **funding_options) -> dict:
    """Refine cash_funding_bound by searching cash-event ORDER IDEALS, not paths.

    Evaluation only; no executor integration. Same event schema/units as the
    interval bound. precedence contains SOUND strict cash-commit edges [id,id],
    projected from OBJECT-GRAPH (including service/custody/authorization) or an
    enforced MARKET-TIMING queue. Do not turn a desired reordering into an edge.
    Non-overlapping windows also impose order. Noncash prerequisite names are
    preserved by the caller; this function neither resolves nor certifies them.

    At most 16 events/65536 states. Each prefix is a downward-closed event set I.
    Its net cash is path-independent for each fixed amount endpoint. Retain only
    the smallest preceding trough for each I; worst funding is the largest
    deficit of ANY ideal. This removes factorial duplicate prefixes, not shared
    money, precedence, within-market order, or endpoint uncertainty.

    Exact on the declared poset/rectangular amount relaxation, NOT a timed native
    schedule. Pairwise disjoint-window edges are necessary, not sufficient for
    slot/deadline/worker/custody feasibility. Best-order witnesses are diagnostic
    candidates, never a funding certificate for the real plan. Certification
    uses the worst-order bound; all original closure requirements still apply.
    Bounds must cover ALL allowed own orders, opponent quotes and fills, not
    just one quoted tape. Otherwise status is UNKNOWN. A budget/unsupported tail
    retains the existing interval bound but is explicitly NOT a completed search.
    """
    require(type(bounds_cover_orders) is bool, "boolean bounds_cover_orders required")
    require(type(max_states) is int and 1 <= max_states <= 65536, "state budget must be 1..65536")
    base = cash_funding_bound(start_cash_cents, events, groups=groups, **funding_options)
    ids = [e["id"] for e in events]
    index = {eid: i for i, eid in enumerate(ids)}
    require(isinstance(precedence, list) and len(precedence) <= 10000, "bounded precedence list required")
    edges = set()
    for edge in precedence:
        require(isinstance(edge, (list, tuple)) and len(edge) == 2 and
                all(isinstance(x, str) and x in index for x in edge), "precedence names unknown cash event")
        require(edge[0] != edge[1], "self precedence")
        edges.add((index[edge[0]], index[edge[1]]))
    result = {"schema": "farm-cash-funding-orders-v1", "interval": base,
              "status": base["status"] if bounds_cover_orders else "UNKNOWN",
              "bounds_cover_orders": bounds_cover_orders,
              "working_capital_lower_cents": base["working_capital_lower_cents"],
              "working_capital_upper_cents": base["working_capital_upper_cents"],
              "required_start_lower_cents": base["required_start_lower_cents"],
              "required_start_upper_cents": base["required_start_upper_cents"],
              "searched_states": 0, "max_states": max_states,
              "scope": "CONDITIONAL_CASH_ONLY; BEST_ORDER_IS_UNSCHEDULED_RELAXATION",
              "order_search_status": "NOT_RUN"}
    n = len(events)
    if n > 16:
        result["order_search_status"] = "UNKNOWN_EVENT_LIMIT"
        return result
    horizon = (base["last_step"], 9, 99998)
    if any(e["amount_max_cents"] is None or e["latest"] is None or
           tuple(e["latest"]) > horizon for e in events):
        result["order_search_status"] = "UNKNOWN_UNBOUNDED_OR_TERMINAL_TAIL"
        return result
    for i, left in enumerate(events):
        for j, right in enumerate(events):
            if i != j and tuple(left["latest"]) < tuple(right["earliest"]):
                edges.add((i, j))
    predecessors = [0] * n
    for i, j in edges:
        predecessors[j] |= 1 << i
    full = (1 << n) - 1
    seen = 0
    while seen != full:
        ready = [i for i in range(n) if not (seen >> i & 1) and not (predecessors[i] & ~seen)]
        require(bool(ready), "cyclic precedence or contradiction with fixed timing")
        for i in ready:
            seen |= 1 << i
    result["precedence"] = [[ids[i], ids[j]] for i, j in sorted(edges)]
    # Each row: optimistic net, pessimistic net, best optimistic capital,
    # best pessimistic capital, number of distinct topological prefixes.
    states = {0: [0, 0, 0, 0, 1]}
    optimistic_parent, pessimistic_parent, first_parent = {}, {}, {}
    frontier = [0]
    worst, worst_mask, transitions = 0, 0, 0
    for _ in range(n):
        next_frontier = set()
        for mask in sorted(frontier):
            row = states[mask]
            for i, event in enumerate(events):
                if mask >> i & 1 or predecessors[i] & ~mask:
                    continue
                transitions += 1
                nxt = mask | (1 << i)
                low, high = event["amount_min_cents"], event["amount_max_cents"]
                opt_delta, pess_delta = (high, low) if event["kind"] == "RECEIPT" else (-low, -high)
                opt_net, pess_net = row[0] + opt_delta, row[1] + pess_delta
                opt_need, pess_need = max(row[2], -opt_net), max(row[3], -pess_net)
                if nxt not in states:
                    if len(states) >= max_states:
                        result.update(order_search_status="UNKNOWN_STATE_BUDGET",
                                      searched_states=len(states), transitions=transitions)
                        return result  # No incomplete extrema/choice counts escape.
                    states[nxt] = [opt_net, pess_net, opt_need, pess_need, row[4]]
                    optimistic_parent[nxt] = pessimistic_parent[nxt] = first_parent[nxt] = (mask, i)
                    if -pess_net > worst:
                        worst, worst_mask = -pess_net, nxt
                else:
                    target = states[nxt]
                    target[4] += row[4]
                    if opt_need < target[2]:
                        target[2] = opt_need; optimistic_parent[nxt] = (mask, i)
                    if pess_need < target[3]:
                        target[3] = pess_need; pessimistic_parent[nxt] = (mask, i)
                next_frontier.add(nxt)
        frontier = next_frontier

    def path(mask, parents):
        selected = []
        while mask:
            mask, i = parents[mask]
            selected.append(ids[i])
        return list(reversed(selected))

    best, best_pess = states[full][2:4]
    # Intersect independent conservative bounds. Neither relaxation invents
    # a feasible native schedule or jointly attainable amount endpoints.
    lower = max(base["working_capital_lower_cents"], best)
    upper = min(base["working_capital_upper_cents"], worst)
    require(lower <= upper, "inconsistent funding assumptions")
    protected = base["policy_floor_cents"] + base["external_protected_cents"]
    required_lower, required_upper = protected + lower, protected + upper
    status = "UNKNOWN"
    if base["scope_complete"] and bounds_cover_orders:
        if start_cash_cents < required_lower:
            status = "INSUFFICIENT"
        elif start_cash_cents >= required_upper:
            status = "SUFFICIENT"
    result.update(order_search_status="COMPLETE_POSET_RELAXATION", status=status,
                  working_capital_lower_cents=lower, working_capital_upper_cents=upper,
                  required_start_lower_cents=required_lower, required_start_upper_cents=required_upper,
                  searched_states=len(states), transitions=transitions,
                  linear_extensions=states[full][4],
                  best_optimistic_order_capital_cents=best,
                  best_pessimistic_order_capital_cents=best_pess,
                  worst_pessimistic_order_capital_cents=worst,
                  best_optimistic_order=path(full, optimistic_parent),
                  best_pessimistic_order=path(full, pessimistic_parent),
                  adverse_prefix=path(worst_mask, first_parent),
                  adverse_group_deficit_cents={g: sum(
                      (e["amount_max_cents"] if e["kind"] == "SPEND" else -e["amount_min_cents"])
                      for i, e in enumerate(events) if e["group"] == g and worst_mask >> i & 1)
                      for g in sorted(groups)})
    return result




def successful_effect(effect: dict) -> bool:
    """Positive physical evidence only. PASS, attempted service and DROP aren't work."""
    action = effect["engine_received_action"]
    if not isinstance(action, list) or not action:
        return False
    op = action[0]
    before, after = effect.get("tile_before"), effect.get("tile_after")
    b, a = before if isinstance(before, dict) else {}, after if isinstance(after, dict) else {}
    carried = effect["delta"].get("carried", {})
    if op == "PICKUP":
        return len(action) >= 2 and carried.get(action[1], 0) > 0
    if op == "PLANT":
        return len(action) >= 2 and a.get("crop") == action[1] and before != after
    if op == "PLACE":
        return len(action) >= 2 and a.get("animal") == action[1] and b.get("animal") != action[1]
    if op in {"WATER", "FEED", "CARE"}:
        field = {"WATER": "watered_today", "FEED": "fed_today", "CARE": "cared_today"}[op]
        return a.get(field) is True and b.get(field) is not True
    if op == "FERTILIZE":
        return carried.get("FERTILIZER", 0) < 0 and a.get("fertilized_until_day", -1) > b.get("fertilized_until_day", -1)
    if op in {"HARVEST", "COLLECT_FERTILIZER"}:
        return any(value > 0 for value in carried.values())
    if op in {"BUILD_PASTURE", "BUILD_COOP"}:
        return a.get("kind") == op.removeprefix("BUILD_") and before != after
    if op == "DIG":
        return before is not None and after is None
    return False


def asset_identity(tile: Any) -> dict | None:
    if not isinstance(tile, dict):
        return None
    if tile.get("animal"):
        return {"animal": tile["animal"], "placed_day": tile.get("placed_day")}
    if tile.get("crop"):
        return {"crop": tile["crop"], "planted_day": tile.get("planted_day")}
    return None


def _end(state: dict) -> int:
    c = state["command"]
    until = LAST_STEP if c.get("until", "GAME_END") == "GAME_END" else c["until"]
    return min(until, state.get("stage_until", LAST_STEP), LAST_STEP)


def obligation_identity(writer: str, ref: str, state: dict, intent: list,
                        step: int, tile: Any) -> dict:
    c, op = state["command"], intent[0]
    start, end = c.get("start", 0), _end(state)
    require(start <= step <= end, "service outside effective stage window")
    scope = c.get("scope", {})
    require(scope.get("kind") == "TILE", "service must resolve to one exact tile")
    daily = op in DAILY
    return {"writer": writer, "ref": ref, "tile": [scope["x"], scope["y"]],
            "window": [start, end], "intent": intent,
            "service_day": step // 24 if daily else None,
            "deadline_step": min(end, (step // 24 + 1) * 24 - 1) if daily else end,
            "deadline_kind": "DAILY_REQUEST" if daily else "STAGE_END",
            "asset": asset_identity(tile) if op not in {"PLACE", "PLANT", "BUILD_PASTURE", "BUILD_COOP", "DIG"} else None}


def branch_timeline(data: dict) -> dict:
    """Reduce a previously provenance-verified complete trace. No new observations.

    Complete source/receipt verification is deliberately the existing runner's
    responsibility. Local identity, window, cash and inventory joins fail closed.
    Resource pools are PRE-ACTION and aggregate carried stock, not deliverability.
    """
    require(data.get("completed") is True, "incomplete trace")
    require(len(data["turns"]) == LAST_STEP + 1, "expected complete 719-turn trace")
    pubs = {p["step"]: p for p in data["publications"]}
    require(len(pubs) == len(data["publications"]), "duplicate publication step")
    stages: dict[str, dict] = {}
    holdings = copy.deepcopy(data["initial"])
    board: dict[tuple[int, int], Any] = {}
    obligations: dict[str, dict] = {}
    dispatches, transfers, services, losses, cash_rows = [], [], [], [], []
    snapshots = []

    for step, turn in enumerate(data["turns"]):
        require(turn["step"] == step, "noncontiguous service timeline")
        if step in pubs:
            for command in pubs[step]["commands"]:
                for change in command["changes"]:
                    if change["after"] is None:
                        stages.pop(change["ref"], None)
                    else:
                        stages[change["ref"]] = copy.deepcopy(change["after"])
        active_list = data["active_sets"][turn["active_set"]]
        active = {s["ref"]: s["writer"] for s in active_list}
        require(len(active) == len(active_list), "ambiguous active writer")
        require(all(ref in stages for ref in active), "active stage missing publication state")
        reserves = [(ref, writer) for ref, writer in active.items() if ref.startswith("rule_stages/CASH_RESERVE@")]
        require(len(reserves) <= 1, "ambiguous active cash reserve")
        reserve = stages[reserves[0][0]]["command"]["amount"] if reserves else 0
        require(cents(turn["cash_before"]) == cents(holdings["cash"]), "pre-action cash mismatch")
        require(cents(turn["cash_after"]) - cents(turn["cash_before"]) == cents(turn["delta"]["cash"]) ==
                sum(cents(x["cash_delta"]) for x in turn["actual_market_commits"]), "cash conservation")
        resource = {"cash_cents": cents(holdings["cash"]), "reserve_cents": cents(reserve),
                    "spendable_cents": cents(holdings["cash"]) - cents(reserve),
                    "shed_used": sum(holdings["shed"].values()),
                    "shed": copy.deepcopy(holdings["shed"]), "seeds": copy.deepcopy(holdings["seeds"]),
                    "carried_aggregate": copy.deepcopy(holdings["carried"]), "units": holdings["units"],
                    "scope": "PRE_ACTION_POOLS_NOT_ALLOCATED_OR_DELIVERABLE_CAPACITY"}
        cash_rows.append({"step": step, "before_cents": cents(holdings["cash"]),
                          "after_cents": cents(turn["cash_after"]), "reserve_cents": cents(reserve),
                          "reserve_writer": reserves[0][1] if reserves else None})
        snapshots.append(resource)
        # First physical observation at a tile supplies a local anchor; never
        # guess unknown tiles are empty. Board changes occur after unit effects.
        local_board = copy.deepcopy(board)
        for effect in turn["unit_effects"]:
            pos = tuple(effect["position"])
            if pos not in local_board:
                local_board[pos] = effect["tile_before"]
        route_by_unit = {r["unit"]: r for r in turn["executor_report"]["routes"]}
        require(len(route_by_unit) == len(turn["executor_report"]["routes"]), "unit assigned twice")

        def register(ref: str, intent: list, target: list, tile: Any) -> tuple[str, dict]:
            require(ref in active, "route/blocked task has no active origin writer")
            identity = obligation_identity(active[ref], ref, stages[ref], intent, step, tile)
            require(identity["tile"] == target, "route target differs from authorized tile")
            key = hashlib.sha256(encoded(identity)).hexdigest()[:24]
            if key not in obligations:
                obligations[key] = {"identity": identity, "first_seen": step, "last_seen": step,
                                    "dispatch_steps": [], "input_transfer_steps": [], "service_steps": [],
                                    "blocked_counts": {}, "first_resources": resource}
            record = obligations[key]
            require(record["identity"] == identity, "obligation identity collision")
            record["last_seen"] = step
            return key, record

        for route in route_by_unit.values():
            key, rec = register(route["ref"], route["intent"], route["target"], local_board.get(tuple(route["target"])))
            rec["dispatch_steps"].append(step)
            dispatches.append({"step": step, "obligation": key, "unit": route["unit"],
                               "issued": route["issued"], "intent": route["intent"],
                               "minimum_travel_and_setup": route["minimum_travel_and_setup"]})
        for blocked in turn["executor_report"]["blocked"]:
            if "op" not in blocked or "tile" not in blocked:
                continue
            refs = [r for r in blocked.get("references", []) if r["ref"].startswith("order_stages/")]
            for ref in refs:
                require(active.get(ref["ref"]) == ref["writer"], "blocked writer mismatch")
                key, rec = register(ref["ref"], blocked["op"], blocked["tile"], local_board.get(tuple(blocked["tile"])))
                counts = rec["blocked_counts"]
                counts[blocked["cause"]] = counts.get(blocked["cause"], 0) + 1
        for effect in turn["unit_effects"]:
            route = route_by_unit.get(effect["unit"])
            if route and successful_effect(effect):
                require(effect["engine_received_action"] == route["issued"], "route/action mismatch")
                key, rec = register(route["ref"], route["intent"], route["target"], local_board.get(tuple(route["target"])))
                row = {"step": step, "obligation": key, "unit": effect["unit"],
                       "action": effect["engine_received_action"], "position": effect["position"],
                       "delta": effect["delta"], "asset_after": asset_identity(effect["tile_after"])}
                if row["action"][0] == "PICKUP":
                    rec["input_transfer_steps"].append(step); transfers.append(row)
                elif row["action"] == route["intent"]:
                    require(effect["position"] == route["target"], "service at wrong tile")
                    rec["service_steps"].append(step); services.append(row)
            local_board[tuple(effect["position"])] = effect["tile_after"]
        for change in turn["board_changes"]:
            before, after = change["before"], change["after"]
            b = before if isinstance(before, dict) else {}
            a = after if isinstance(after, dict) else {}
            lost_animal = bool(b.get("animal")) and not a.get("animal")
            neglected_crop = b.get("kind") == "PLANT" and a.get("kind") == "WEED" and b.get("consecutive_unwatered", 0) >= 1 and not b.get("watered_today")
            if lost_animal or neglected_crop:
                matching = [{"ref": ref, "writer": writer} for ref, writer in active.items()
                            if ref.startswith("order_stages/") and stages[ref]["command"].get("scope") ==
                            {"kind": "TILE", "x": change["tile"][0], "y": change["tile"][1]}]
                losses.append({"step": step, "tile": change["tile"], "asset": asset_identity(before),
                               "active_origins": matching, "before": before, "after": after,
                               "classification": "OBSERVED_ANIMAL_REMOVAL" if lost_animal else "OBSERVED_NEGLECT_WEED"})
            board[tuple(change["tile"])] = after
        for pool in ("shed", "seeds", "carried"):
            for item, amount in turn["delta"][pool].items():
                holdings[pool][item] = holdings[pool].get(item, 0) + amount
                require(holdings[pool][item] >= 0, "negative reconstructed resource")
                if not holdings[pool][item]:
                    del holdings[pool][item]
        for k in ("cash", "opponent_cash", "units", "hires_today"):
            holdings[k] += turn["delta"][k]
        if "quadrants_after" in turn["delta"]:
            holdings["quadrants"] = turn["delta"]["quadrants_after"]
    require(holdings == data["terminal"], "terminal resource reconstruction mismatch")
    return {"obligations": obligations, "dispatches": dispatches, "input_transfers": transfers,
            "services": services, "losses": losses, "cash": cash_rows, "resources": snapshots}


def first_difference(left: list[dict], right: list[dict], keys: tuple[str, ...]) -> dict | None:
    # Compare physical services with unit identity intentionally removed. Two
    # workers doing the same task at the same turn have the same service outcome.
    def signature(row: dict) -> bytes:
        return encoded({k: row[k] for k in keys})
    steps = sorted({row["step"] for row in [*left, *right]})
    by_side = []
    for rows in (left, right):
        grouped: dict[int, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["step"], []).append(row)
        by_side.append(grouped)
    for step in steps:
        l, r = by_side[0].get(step, []), by_side[1].get(step, [])
        if Counter(map(signature, l)) != Counter(map(signature, r)):
            return {"step": step, "parent": l, "child": r}
    return None


def paired_timeline(parent: dict, child: dict) -> dict:
    for key in ("seed", "seat", "opponent", "initial", "runtime_sources"):
        require(parent[key] == child[key], "paired identity mismatch: " + key)
    require(len(parent["turns"]) == len(child["turns"]), "unequal trace lengths")
    left, right = branch_timeline(parent), branch_timeline(child)
    aligned = []
    for key in sorted(set(left["obligations"]) | set(right["obligations"])):
        l, r = left["obligations"].get(key), right["obligations"].get(key)
        def compact(row):
            if row is None:
                return None
            return {name: row[name] for name in ("first_seen", "last_seen", "dispatch_steps", "input_transfer_steps", "service_steps", "blocked_counts")}
        aligned.append({"id": key, "identity": (l or r)["identity"], "parent": compact(l), "child": compact(r)})
    differences = {
        "dispatch": first_difference(left["dispatches"], right["dispatches"], ("obligation", "issued", "intent")),
        "input_transfer": first_difference(left["input_transfers"], right["input_transfers"], ("obligation", "action", "delta")),
        "completed_service": first_difference(left["services"], right["services"], ("obligation", "action", "delta", "asset_after"))}
    first_market = next(({"step": a["step"], "parent": a["actual_market_commits"], "child": b["actual_market_commits"]}
                         for a, b in zip(parent["turns"], child["turns"], strict=True)
                         if a["actual_market_commits"] != b["actual_market_commits"]), None)
    differences["market_fills"] = first_market
    # Short evidence spine; full daily obligation summaries still cover all turns.
    spine_steps = sorted({0, 6, 7, 8, 9, 10, 17, 18, 23, 24, 47, 59, 144, 151, 154, 245, 358, 718} |
                         {d["step"] for d in differences.values() if d})
    spine = []
    for step in spine_steps:
        if step >= len(parent["turns"]):
            continue
        row = {"step": step}
        for label, data, reduced in (("parent", parent, left), ("child", child, right)):
            t = data["turns"][step]
            row[label] = {"resources_before": reduced["resources"][step],
                          "cash_after": t["cash_after"], "market_requests": t["action"]["market"],
                          "actual_market_fills": t["actual_market_commits"],
                          "routes": t["executor_report"]["routes"],
                          "input_blocks": [b for b in t["executor_report"]["blocked"] if b["cause"] in {"CASH_OR_SHED_CAPACITY", "HIRE_CASH", "LAND_CASH"}],
                          "unit_effects": t["unit_effects"]}
        spine.append(row)
    def summary(data):
        cash = data["cash"]
        return {"obligation_count": len(data["obligations"]), "dispatch_count": len(data["dispatches"]),
                "positive_input_transfers": len(data["input_transfers"]), "completed_services": len(data["services"]),
                "minimum_cash_after_cents": min(r["after_cents"] for r in cash),
                "below_declared_floor_after_turns": [r["step"] for r in cash if r["after_cents"] < r["reserve_cents"]],
                "already_below_floor_before_turns": [r["step"] for r in cash if r["before_cents"] < r["reserve_cents"]],
                "losses": data["losses"]}
    return {"schema": "farm-paired-service-timeline-v1", "seed": parent["seed"], "seat": parent["seat"],
            "opponent": parent["opponent"], "covered_turns_per_branch": len(parent["turns"]),
            "summary": {"parent": summary(left), "child": summary(right)},
            "first_differences": differences, "aligned_obligations": aligned, "evidence_spine": spine,
            "limits": ["Observed paired mechanism, not independently identified liquidity mediation.",
                       "Input pickup is not placement, feed, harvest, deposit or sale.",
                       "Asset generation and daily deadline split recurring duties; missing matches are censored.",
                       "Pre-action aggregate inventory is not worker-level deliverability.",
                       "Cash floors are checked at observations; within-turn ordering is not present in aggregated market commits.",
                       "Unobserved or unscheduled possible duties are not certified feasible.",
                       "Native birth-day asset identifiers are checked for same-day replacement collisions in these fixtures; not universal lifecycle IDs."]}


def tile_periodicity(tile: dict, step: int) -> dict:
    """Visible local clocks, not an authorized schedule or a yield/value forecast.

    Bundled 24-turn days, last action 718 only. At a fixed observation step,
    (next release, period, count) is the exact remaining *opportunity* calendar.
    Animal placement day may be removed from conditional biology once this clock
    and all native biological fields are retained. Generation stays in event IDs.
    Crop age is conservatively retained: maturation, bonus windows and decay are
    different clocks. Do not turn a possible release into a completed harvest.

    The neglect window is the first service needed IF retaining the asset and
    giving no intervening service. Recompute after service/replacement. It is not
    a demand for daily feed/water: one missed refresh is survivable. No final-day
    refresh exists; a one-shot crop can still gain immediate watering yield.
    """
    from farm_neat.game.constants import ANIMALS, CROPS

    require(type(step) is int and 0 <= step <= LAST_STEP, "invalid native action step")
    require(isinstance(tile, dict), "visible productive tile required")
    animal = tile.get("animal")
    crop = tile.get("crop")
    require(bool(animal) != bool(crop), "one animal or crop required")
    day, end = step // 24, (step // 24 + 1) * 24 - 1
    if animal:
        require(animal in ANIMALS, "unknown animal")
        rules = ANIMALS[animal]
        birth = tile.get("placed_day")
        served, misses, service = "fed_today", "consecutive_unfed", "FEED"
        fields = ("consecutive_unfed", "pending_care_bonus", "yield_units",
                  "fed_today", "cared_today", "fertilizer_available")
    else:
        require(crop in CROPS and tile.get("kind") == "PLANT", "unknown plant")
        rules = CROPS[crop]
        birth = tile.get("planted_day")
        served, misses, service = "watered_today", "consecutive_unwatered", "WATER"
        fields = ("planted_day", "consecutive_unwatered", "yield_units",
                  "watered_today", "fertilized_until_day", "max_lifespan_step")
    require(type(birth) is int and 0 <= birth <= day, "invalid birth day")
    require(all(k in tile for k in fields), "incomplete visible tile state")
    require(type(tile[served]) is bool and type(tile[misses]) is int and
            0 <= tile[misses] <= 1, "invalid neglect state")
    boolean_fields = {"fed_today", "cared_today", "fertilizer_available", "watered_today"}
    for field in fields:
        require(type(tile[field]) is (bool if field in boolean_fields else int),
                "invalid tile field type: " + field)
    require(0 <= tile["yield_units"] <= rules.get("max_held", rules.get("max_yield")),
            "invalid held units")
    if animal:
        require(tile["pending_care_bonus"] >= 0, "negative pending care")

    period = rules["interval"]
    first = birth + rules["first_yield_day"]
    # Future refreshes only. Current held stock already includes earlier releases.
    release_days = []
    if period:
        stop = 30 if animal else min(30, first + period * rules["max_yield"])
        release_days = [d for d in range(first, stop, period) if day < d]
    clock = {"next_release_step": 24 * release_days[0] if release_days else None,
             "period_steps": 24 * period if release_days else None,
             "remaining_events": len(release_days)}
    # If already served today, reset this day then allow one skipped refresh.
    failure_day = day + (2 if tile[served] else 1 - tile[misses])
    failure_step = (failure_day + 1) * 24 - 1
    neglect = None
    if failure_step <= LAST_STEP:
        neglect = {"op": service, "release": (day + 1) * 24 if tile[served] else step,
                   "deadline": failure_step, "duration": 1,
                   "inputs": {"WHEAT": 1} if animal else {},
                   "custody": "SERVICING_WORKER" if animal else "NONE",
                   "condition": "RETAIN_ASSET_NO_INTERVENING_SERVICE_OR_REPLACEMENT",
                   "missed_consequence": "ESCAPE" if animal else "NEGLECT_WEED"}
    decay = None
    bonus = None
    if crop:
        lifespan = tile["max_lifespan_step"]
        require(lifespan >= -1, "invalid crop lifespan")
        if lifespan >= 0:
            decay = lifespan + 2 * max(0, (step - lifespan + 1) // 2)
            if decay > LAST_STEP:
                decay = None
        if not rules["ongoing"]:
            start = max(day, birth + (rules["max_yield_day"] + 1) // 2)
            stop = min(29, birth + rules["max_yield_day"])
            bonus = [start, stop] if start <= stop else None
    return {"schema": "farm-tile-periodicity-v1", "kind": animal or crop,
            "family": "ANIMAL" if animal else "CROP", "step": step,
            "last_step": LAST_STEP, "refresh_step": end if end <= LAST_STEP else None,
            "production_clock": clock, "biology_state": {k: tile[k] for k in fields},
            "neglect_service": neglect, "next_existing_decay_step": decay,
            "one_shot_bonus_days_inclusive": bonus,
            "harvestable_now": tile["yield_units"] > 0 and (bool(animal) or day >= first),
            "daily_service_period_steps": 24,
            "scope": "VISIBLE_LOCAL_STATE_CONDITIONAL_BIOLOGY_NOT_AUTHORIZATION_DELIVERY_OR_VALUE"}


def service_capacity_screen(jobs: list[dict], capacity: list[int], *, start_step: int) -> dict:
    """Exact unit-job/interval relaxation; native success remains UNKNOWN.

    Each chosen job: id, object_id, generation, seat, op, release, deadline,
    duration=1. Other fields (position, inputs, prerequisites, authorization,
    visibility) are carried by the caller and NOT verified here. Preserve them
    when joining to OBJECT-GRAPH/PLOT-ALLOCATION. Windows are inclusive turns;
    capacity[t-start_step] is an optimistic upper bound on service unit-actions.

    Earliest-deadline-first is exact when all selected jobs are mandatory,
    independently executable by any worker, one action each, with interval windows.
    An exchange replaces a later-deadline job by the earliest available deadline
    without invalidating the displaced job. Thus failure proves this relaxation
    infeasible. A failure also returns an interval with forced demand > capacity.
    Travel, delivery, precedences, daily idempotence and unchosen economic options
    are relaxed: PASS never certifies a native plan. No worker or economic command
    is created. Never pool seats or forget generation when compiling the jobs.
    """
    import heapq

    require(type(start_step) is int and 0 <= start_step <= LAST_STEP, "invalid start step")
    require(isinstance(capacity, list) and 1 <= len(capacity) <= LAST_STEP + 1 - start_step,
            "bounded explicit capacity required")
    require(all(type(c) is int and c >= 0 for c in capacity), "nonnegative integer capacities required")
    require(isinstance(jobs, list) and len(jobs) <= 4096, "at most 4096 selected unit jobs")
    stop = start_step + len(capacity) - 1
    ids, seats = set(), set()
    for job in jobs:
        require(isinstance(job, dict), "job must be an object")
        for field in ("id", "object_id", "op"):
            require(isinstance(job.get(field), str) and bool(job[field].strip()), "missing " + field)
        require(job["id"] not in ids, "duplicate job id")
        ids.add(job["id"])
        require(type(job.get("generation")) is int and job["generation"] >= 0, "explicit generation required")
        require(type(job.get("seat")) is int and job["seat"] in (0, 1), "invalid seat")
        seats.add(job["seat"])
        require(type(job.get("duration")) is int and job["duration"] == 1, "only native unit-duration jobs supported")
        require(type(job.get("release")) is int and type(job.get("deadline")) is int and
                start_step <= job["release"] <= job["deadline"] <= stop,
                "jobs must fit supplied inclusive horizon; do not silently clip")
    require(len(seats) <= 1, "cannot pool worker capacity across farms")
    arrivals = sorted(jobs, key=lambda j: (j["release"], j["deadline"], j["id"]))
    at, queue, schedule = 0, [], []
    failed = False
    for step, slots in enumerate(capacity, start_step):
        while at < len(arrivals) and arrivals[at]["release"] <= step:
            j = arrivals[at]
            heapq.heappush(queue, (j["deadline"], j["id"]))
            at += 1
        if queue and queue[0][0] < step:
            failed = True
            break
        for _ in range(min(slots, len(queue))):
            _, jid = heapq.heappop(queue)
            schedule.append({"id": jid, "step": step})
    failed = failed or bool(queue) or at != len(arrivals)
    cut = None
    if failed:
        # O(R*(N+H)) only on failures; endpoints are job releases/deadlines.
        # Every job wholly inside [a,b] must consume one of that interval's slots.
        prefix = [0]
        for c in capacity:
            prefix.append(prefix[-1] + c)
        for a in sorted({j["release"] for j in jobs}):
            deadlines = [0] * len(capacity)
            for j in jobs:
                if j["release"] >= a:
                    deadlines[j["deadline"] - start_step] += 1
            demand = 0
            for b in range(a, stop + 1):
                demand += deadlines[b - start_step]
                slots = prefix[b - start_step + 1] - prefix[a - start_step]
                if demand > slots:
                    forced = [j["id"] for j in jobs if j["release"] >= a and j["deadline"] <= b]
                    cut = {"release": a, "deadline": b, "demand": demand,
                           "capacity": slots, "excess": demand - slots, "job_ids": sorted(forced)}
                    break
            if cut is not None:
                break
        require(cut is not None, "internal EDF/capacity-cut disagreement")
    return {"schema": "farm-service-capacity-screen-v1", "jobs": len(jobs),
            "relaxed_feasible": not failed,
            "native_status": "INFEASIBLE_SELECTED_CALENDAR" if failed else "UNKNOWN",
            "schedule": [] if failed else schedule, "cut": cut,
            "scope": "MANDATORY_UNIT_JOBS_ANY_WORKER_NO_TRAVEL_INPUT_OR_PRECEDENCE_CHECK"}


def recurring_survival_screen(objects: list[dict], capacity: list[int], wheat_capacity: list[int],
                              *, start_step: int, through_step: int = LAST_STEP) -> dict:
    """Exact count quotient for a *neglect-only*, fixed-population relaxation.

    This is NOT a production planner and never dispatches its certificate. Retain
    every supplied current generation; no births, replacements, deliberate DIG,
    harvest-removal or sales are selected here. Crops' age decay, travel, custody,
    purchases, cumulative wheat stock and other duties are relaxed. A native
    pruning implication requires the supplied per-day action/wheat capacities to
    be optimistic UPPER bounds. A pass remains UNKNOWN for native feasibility.

    Each object contains object_id, generation, seat, position, tile and an
    explicit authorization label. `tile_periodicity` validates its visible state;
    clocks/biology remain in the caller, not replaced by this count quotient.
    One list entry per remaining day-END within [start_step, through_step], never
    a fictitious refresh at 719. capacity counts available future unit-actions;
    wheat_capacity bounds successful future FEEDs independently on each such day.
    Already fed/watered flags cost no future action or wheat on the first day.

    Proof: within a service type let q be the number unserved at the last refresh.
    To avoid two consecutive misses, today's served subset must include all q;
    choosing k>=q leaves N-k misses. Which identities were missed is irrelevant
    to this relaxation: serve them first and choose any remaining extras. FEED
    and WATER share action capacity but only FEED uses wheat, so retain (qF,qW),
    not their sum. On the first partial day exclude already-served identities.
    A 2D prefix lookup finds any reachable predecessor with qF<=kF,qW<=kW.
    Time O(D*(NF+1)*(NW+1)+D*N), memory O(D*(NF+1)*(NW+1)); D<=29,N<=100.
    This eliminates labelled service subsets, NOT biological/yield distinctions.
    """
    require(type(start_step) is int and type(through_step) is int and
            0 <= start_step <= through_step <= LAST_STEP, "invalid inclusive horizon")
    require(isinstance(objects, list) and len(objects) <= 100, "at most 100 current objects")
    days = [d for d in range(start_step // 24, 29) if (d + 1) * 24 - 1 <= through_step]
    for name, values in (("capacity", capacity), ("wheat_capacity", wheat_capacity)):
        require(isinstance(values, list) and len(values) == len(days),
                name + " must have exactly one entry per included refresh")
        require(all(type(v) is int and v >= 0 for v in values),
                name + " must contain nonnegative integers")
    groups: list[list[dict]] = [[], []]
    seats, locations, identities = set(), set(), set()
    for obj in objects:
        require(isinstance(obj, dict), "object must be a mapping")
        for key in ("object_id", "authorization"):
            require(isinstance(obj.get(key), str) and bool(obj[key].strip()), "missing " + key)
        require(type(obj.get("generation")) is int and obj["generation"] >= 0,
                "explicit generation required")
        require(type(obj.get("seat")) is int and obj["seat"] in (0, 1), "invalid seat")
        pos = obj.get("position")
        require(isinstance(pos, list) and len(pos) == 2 and
                all(type(x) is int and 0 <= x < 10 for x in pos), "invalid native position")
        require(obj["object_id"] not in identities, "one current generation per object required")
        require(tuple(pos) not in locations, "two current objects cannot share a tile")
        identities.add(obj["object_id"])
        locations.add(tuple(pos))
        seats.add(obj["seat"])
        profile = tile_periodicity(obj.get("tile"), start_step)
        groups[0 if profile["family"] == "ANIMAL" else 1].append(obj)
    require(len(seats) <= 1, "cannot pool worker capacity across farms")
    groups = [sorted(g, key=lambda x: x["object_id"]) for g in groups]
    flags = ("fed_today", "watered_today")
    misses = ("consecutive_unfed", "consecutive_unwatered")
    ns = tuple(len(g) for g in groups)
    already = tuple(sum(o["tile"][flags[i]] for o in g) for i, g in enumerate(groups))
    due = tuple(sum(not o["tile"][flags[i]] and o["tile"][misses[i]] == 1 for o in g)
                for i, g in enumerate(groups))
    frontier = {due}
    history, frontier_sizes, cells = [], [], 0
    failed_day = None
    for index, day in enumerate(days):
        # Each prefix stores one actual reachable state, not an invented convex mix.
        prefix = [[None] * (ns[1] + 1) for _ in range(ns[0] + 1)]
        for f in range(ns[0] + 1):
            for w in range(ns[1] + 1):
                choices = [(f, w)] if (f, w) in frontier else []
                if f and prefix[f - 1][w] is not None:
                    choices.append(prefix[f - 1][w])
                if w and prefix[f][w - 1] is not None:
                    choices.append(prefix[f][w - 1])
                prefix[f][w] = min(choices) if choices else None
        af, aw = already if index == 0 else (0, 0)
        next_states = {}
        for f in range(ns[0] - af + 1):
            for w in range(ns[1] - aw + 1):
                cells += 1
                if f + w > capacity[index] or f > wheat_capacity[index]:
                    continue
                parent = prefix[f][w]
                if parent is not None:
                    next_states[(ns[0] - af - f, ns[1] - aw - w)] = (parent, (f, w))
        frontier_sizes.append(len(next_states))
        if not next_states:
            failed_day = day
            break
        history.append(next_states)
        frontier = set(next_states)

    feasible = failed_day is None
    daily, jobs = [], []
    if feasible and days:
        # Merely choose a feasible witness; this is not minimum cost or optimal care.
        state = max(frontier, key=lambda q: (sum(q), q))
        counts = []
        for parents in reversed(history):
            state, chosen = parents[state]
            counts.append(chosen)
        counts.reverse()
        missed_ids = [set(o["object_id"] for o in g if not o["tile"][flags[i]] and
                          o["tile"][misses[i]] == 1) for i, g in enumerate(groups)]
        for index, (day, chosen) in enumerate(zip(days, counts)):
            existing = [set(o["object_id"] for o in g if index == 0 and o["tile"][flags[i]])
                        for i, g in enumerate(groups)]
            served_ids = []
            for i, g in enumerate(groups):
                needed = missed_ids[i]
                extras = sorted(o["object_id"] for o in g
                                if o["object_id"] not in needed | existing[i])
                selected = needed | set(extras[:chosen[i] - len(needed)])
                require(len(selected) == chosen[i], "internal count lift mismatch")
                served_ids.append(sorted(selected))
                for obj in g:
                    if obj["object_id"] not in selected:
                        continue
                    op = "FEED" if i == 0 else "WATER"
                    jobs.append({"id": json.dumps([obj["seat"], obj["object_id"], obj["generation"], day, op],
                                                  separators=(",", ":")),
                                 "object_id": obj["object_id"], "generation": obj["generation"],
                                 "seat": obj["seat"], "position": list(obj["position"]), "op": op,
                                 "release": max(start_step, day * 24), "deadline": day * 24 + 23,
                                 "duration": 1, "inputs": {"WHEAT": 1} if i == 0 else {},
                                 "custody": "SERVICING_WORKER" if i == 0 else "NONE",
                                 "authorization": obj["authorization"],
                                 "realization": "HYPOTHETICAL_RELAXATION_WITNESS",
                                 "missed_consequence": "RECOMPUTE_NEGLECT_STATE_NOT_AUTOMATIC_LOSS"})
                missed_ids[i] = {o["object_id"] for o in g} - selected - existing[i]
            daily.append({"day": day, "new_feed": chosen[0], "new_water": chosen[1],
                          "feed_ids": served_ids[0], "water_ids": served_ids[1],
                          "missed_after": [len(x) for x in missed_ids]})
    return {"schema": "farm-recurring-survival-screen-v1", "objects": len(objects),
            "start_step": start_step, "through_step": through_step, "refresh_days": days,
            "relaxed_feasible": feasible,
            "native_status": "UNKNOWN" if feasible else "INFEASIBLE_SELECTED_NEGLECT_RELAXATION",
            "failed_day": failed_day, "daily_witness": daily, "jobs": jobs,
            "statistics": {"labelled_boundary_states": 2 ** len(objects),
                           "count_boundary_states": (ns[0] + 1) * (ns[1] + 1),
                           "candidate_count_pairs": cells, "reachable_states_by_day": frontier_sizes},
            "scope": "RETAIN_FIXED_GENERATIONS_NEGLECT_ONLY_UPPER_CAPACITY_NO_TRAVEL_OR_CUMULATIVE_STOCK",
            "limits": ["Reject only for explicit retention with optimistic capacity and wheat bounds.",
                       "A pass is not native survival: crop decay, inputs/custody, travel and other duties are relaxed.",
                       "This witness is not a policy action, cost optimum, production/yield quotient or cash forecast.",
                       "Keep per-object production clocks, care, held stock, generations and actual daily effects."]}
