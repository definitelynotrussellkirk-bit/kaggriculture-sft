"""CPU replay of recorded five-publication PASS programs; never samples a model.

The live FullSeasonPlan and bundled interpreter do all composition/execution.
Read-only taps delegate each engine market function exactly once, recording actual
commits rather than attributing money from a forecast or terminal cash. Engine
hooks are process-local and restored even on failure; run this CLI serially.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT / "src"))

from farm_neat.economy.action_space import validate_program
from farm_neat.economy.orders import FullSeasonPlan, ScheduledEconomicOrders
from farm_neat.economy.orders_v1 import digest
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment, PASS_ACTION

SCHEMA = "farm-fixed-program-realization-v2"
JOURNAL_SCHEMA = "farm-fixed-program-attempt-v1"
INTERVENTION_SCHEMA = "farm-fixed-program-single-field-intervention-v1"
STEPS = FullSeasonPlan.STEPS
METRICS = ("own_cash", "opponent_cash", "physical_turns", "rejected_commands")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def stage_ref(group, key, row):
    return f"{group}/{key}@{row['command'].get('start', 0)}"


def stages(executor):
    return {stage_ref(group, key, row): copy.deepcopy(row)
            for group in ("order_stages", "rule_stages")
            for key, rows in getattr(executor, group).items() for row in rows}


def active(executor, owners):
    return [{"ref": stage_ref(group, key, row),
             "writer": owners.get(stage_ref(group, key, row))}
            for group, mapping in (("order_stages", executor.orders),
                                   ("rule_stages", executor.rules))
            for key, row in mapping.items()]


def shortened_existing_window(old, new, command):
    """A future REPLACE can bound an earlier request without originating it."""
    if old is None or new is None or command["op"] != "REPLACE":
        return False
    left, right = copy.deepcopy(old), copy.deepcopy(new)
    old_end, new_end = ScheduledEconomicOrders._end(old), ScheduledEconomicOrders._end(new)
    for row in (left, right):
        row.pop("stage_until", None)
        row["command"].pop("until", None)
    return (left == right and new_end < old_end
            and new_end + 1 == command["replacement"].get("start", 0))


def publication(controller, observation, program, owners):
    """Trace commands with the existing apply(), verifying against real propose().

    This draft copy never executes actions. No refusal or installation semantics
    are reimplemented: the authoritative proposal supplies refusal indices, and
    its complete executor snapshot must equal the incrementally traced copy.
    """
    proposal = controller.propose(observation, program)
    staged = copy.deepcopy(controller.executor)
    refused = {r["command_index"]: r for r in proposal["rejected"]}
    next_owners = dict(owners)
    commands = []
    for index, command in enumerate(proposal["program"]):
        identity = f"{observation['step']}:{index}"
        record = {"id": identity, "index": index, "command": command,
                  "command_sha256": digest(command), "changes": [],
                  "revision_before": staged.revision}
        before = stages(staged)
        if index in refused:
            record.update(status="REFUSED", refusal=refused[index])
        else:
            staged.apply(command)
            after = stages(staged)
            for ref in sorted(before.keys() | after.keys()):
                old, new = before.get(ref), after.get(ref)
                if old == new:
                    continue
                prior = next_owners.get(ref)
                shortened = shortened_existing_window(old, new, command)
                record["changes"].append({"ref": ref, "before": old, "after": new,
                    "previous_writer": prior,
                    "effect": "SHORTEN_EXISTING_WINDOW" if shortened else
                              "REMOVE" if new is None else "INSTALL_OR_REVISE"})
                if new is None:
                    next_owners.pop(ref, None)
                elif not shortened:
                    next_owners[ref] = identity
            record["status"] = ("STOP" if command["op"] == "STOP" else
                                "ACCEPTED_CHANGE" if record["changes"] else "ACCEPTED_NOOP")
        record["revision_after"] = staged.revision
        commands.append(record)
    if staged.snapshot() != proposal["full_plan"]["executor"]:
        raise ValueError("diagnostic composition differs from authoritative proposal")
    published = controller.publish(observation, proposal)
    final = stages(published.executor)
    for command in commands:
        command["effective_refs_at_publication"] = [ref for ref, writer in next_owners.items()
                                                     if writer == command["id"]]
        command["execution_windows"] = [
            {"ref": ref, "from_step": max(observation["step"], published.executor._start(final[ref])),
             "until_step": published.executor._end(final[ref])}
            for ref in command["effective_refs_at_publication"]
            if published.executor._end(final[ref]) >= observation["step"]]
        command["superseded_within_publication_by"] = sorted({later["id"] for later in commands
            for change in later["changes"] if change["previous_writer"] == command["id"]
            and change["effect"] != "SHORTEN_EXISTING_WINDOW"})
        command["shortened_within_publication_by"] = sorted({later["id"] for later in commands
            for change in later["changes"] if change["previous_writer"] == command["id"]
            and change["effect"] == "SHORTEN_EXISTING_WINDOW"})
    result = {"step": observation["step"], "game_id": controller.game_id,
              "observation_sha256": proposal["observation_sha256"],
              "base_sha256": proposal["base_sha256"],
              "base_revision": proposal["base_revision"], "revision": published.revision,
              "proposal_sha256": proposal["content_sha256"],
              "published_executor_sha256": digest(published.executor.snapshot()),
              "elapsed_prefix_sha256": digest(published.past),
              "program_sha256": digest(proposal["program"]), "commands": commands,
              "effective_stages": [{"ref": ref, "writer": next_owners.get(ref), "state": row}
                                    for ref, row in stages(published.executor).items()]}
    return published, result, next_owners


def holdings(observation, seat):
    private, farm = observation["private"], observation["farms"][seat]
    carried = Counter()
    for inventory in private["inventories"]:
        carried.update(inventory)
    return {"cash": float(farm["money"]),
            "opponent_cash": float(observation["farms"][1-seat]["money"]),
            "shed": {k: v for k, v in private["shed"].items() if v},
            "seeds": {k: v for k, v in private["seeds"].items() if v},
            "carried": {k: v for k, v in carried.items() if v},
            "units": 1 + len(farm["hands"]), "hires_today": farm["hires_today"],
            "quadrants": list(farm["unlocked_quadrants"])}


def stock_delta(before, after):
    return {key: after.get(key, 0)-before.get(key, 0)
            for key in sorted(before.keys() | after.keys()) if after.get(key, 0) != before.get(key, 0)}


def delta(before, after):
    result = {key: stock_delta(before[key], after[key]) for key in ("shed", "seeds", "carried")}
    result.update({key: after[key]-before[key] for key in
                   ("cash", "opponent_cash", "units", "hires_today")})
    if before["quadrants"] != after["quadrants"]:
        result["quadrants_after"] = after["quadrants"]
    return result


@contextmanager
def market_tap(env, seat):
    """Observe real interpreter commits, not a second market implementation.

    Not thread-safe: only the dedicated serial diagnostic process may use this.
    A failed unit commit is an attempted unit, not the full unfilled order size.
    """
    engine = env.engine
    names = ("_commit_unit", "_do_hire", "_do_buy_land", "_process_market", "_apply_unit_action")
    originals = {name: getattr(engine, name) for name in names}
    if any(getattr(fn, "_realization_tap", False) for fn in originals.values()):
        raise RuntimeError("nested/concurrent market taps are unsupported")
    capture = {"commits": {}, "pre_market": None, "unit_effects": []}

    def record(op, item, farm, before_cash, units, failed):
        if farm is not env.state[0].observation.farms[seat]:
            return
        key = (op, item)
        row = capture["commits"].setdefault(key, {"op": op, "item": item,
              "units": 0, "failed_unit_attempts": 0, "cash_delta": 0.0})
        row["units"] += units
        row["failed_unit_attempts"] += failed
        row["cash_delta"] += float(farm["money"])-before_cash

    def unit(op, item, price, farm, private, market, shed_capacity=100):
        before = float(farm["money"])
        ok = originals["_commit_unit"](op, item, price, farm, private, market, shed_capacity)
        record(op, item, farm, before, int(bool(ok)), int(not ok))
        return ok

    def hire(farm, private, board_size, mult=1):
        before, count = float(farm["money"]), farm["hires_today"]
        result = originals["_do_hire"](farm, private, board_size, mult)
        got = farm["hires_today"]-count
        record("HIRE", None, farm, before, got, int(not got))
        return result

    def land(farm, board_size):
        before, count = float(farm["money"]), len(farm["unlocked_quadrants"])
        result = originals["_do_buy_land"](farm, board_size)
        got = len(farm["unlocked_quadrants"])-count
        record("BUY_LAND", None, farm, before, got, int(not got))
        return result

    def process(state, environment):
        capture["pre_market"] = holdings(environment.agent_observation(seat), seat)
        return originals["_process_market"](state, environment)

    def unit_inventory(private):
        carried = Counter()
        for inventory in private["inventories"]:
            carried.update(inventory)
        return {"shed": dict(private["shed"]), "seeds": dict(private["seeds"]),
                "carried": dict(carried)}

    def apply_unit(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity=100):
        observed = (farm is env.state[0].observation.farms[seat] and isinstance(action, list) and action
                    and action[0] not in ("PASS", "NORTH", "SOUTH", "EAST", "WEST"))
        if observed:
            before = unit_inventory(private)
            position = engine._farmer_position(farm, idx)
            tile = copy.deepcopy(farm["tiles"][position[1]][position[0]]) if position else None
        result = originals["_apply_unit_action"](
            farm, private, idx, action, board_size, day, turns_per_day, shed_capacity)
        if observed:
            after = unit_inventory(private)
            capture["unit_effects"].append({"unit": idx,
                "engine_received_action": copy.deepcopy(action), "position": copy.deepcopy(position),
                "delta": {key: stock_delta(before[key], after[key]) for key in before},
                "tile_before": tile,
                "tile_after": copy.deepcopy(farm["tiles"][position[1]][position[0]]) if position else None})
        return result

    hooks = dict(zip(names, (unit, hire, land, process, apply_unit)))
    try:
        for name, fn in hooks.items():
            fn._realization_tap = True
            setattr(engine, name, fn)
        yield capture
    finally:
        for name, fn in originals.items():
            setattr(engine, name, fn)


def sale_gates(executor, before_market, observation, owners):
    """Necessary explicit conditions, not the dispatcher's full withheld amount.

    Post-unit/pre-market stock is measured by the interpreter. This does not
    expose queue-position stock or the additional production requisition reserve.
    """
    gates = []
    for key, row in executor.rules.items():
        c = row["command"]
        if c["op"] != "SALE":
            continue
        ref = stage_ref("rule_stages", key, row)
        stock = before_market["shed"].get(c["product"], 0)
        price = observation["market"]["prices"][c["product"]]
        gates.append({"ref": ref, "writer": owners.get(ref), "product": c["product"],
            "stock": stock, "reserve": c["reserve"], "quantity": c["quantity"],
            "observed_price": price, "price_above": c["price_above"],
            "explicit_reserve_covers_stock": stock <= c["reserve"],
            "observed_price_trigger_met": price > c["price_above"]})
    return gates


def validate_case(panel, seed, seat):
    if type(seed) is not int or seed < 0 or type(seat) is not int or seat not in (0, 1):
        raise ValueError("nonnegative integer seed and seat 0 or 1 required")
    if panel.get("schema") not in ("farm-full-economic-free-running-sampled-eval-v1",
                                    "farm-full-economic-free-running-sampled-eval-v2"):
        raise ValueError("recognized saved free-running panel schema required")
    if panel.get("opponent") != "pass" or panel.get("mode") != "five":
        raise ValueError("only recorded five-publication PASS panels are supported")
    if panel.get("decision_steps") != list(STEPS):
        raise ValueError("exact five-publication clock required")
    matches = [r for r in panel.get("per_game", []) if (r.get("seed"), r.get("seat")) == (seed, seat)]
    if len(matches) != 1:
        raise ValueError("exactly one saved seed/seat row required")
    row = matches[0]
    if "error" in row:
        raise ValueError("completed saved row required; error rows are not replay targets")
    if row.get("opponent") != "pass" or set(row.get("programs", {})) != {str(s) for s in STEPS}:
        raise ValueError("saved PASS row and all five programs required")
    budget = panel.get("max_commands")
    if type(budget) is not int or not 1 <= budget <= 256:
        raise ValueError("valid recorded command budget required")
    for program in row["programs"].values():
        if len(validate_program(program)) > budget:
            raise ValueError("saved program exceeds recorded budget")
    for name in METRICS:
        value = row.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative saved metric required: " + name)
    return copy.deepcopy(row)


def outcome_comparison(saved, actual):
    differences = [{"field": key, "saved": saved[key], "replayed": actual[key]} for key in METRICS
                   if saved[key] != actual[key]]
    return {"saved_row_match": not differences, "differences": differences,
            "first_comparable_outcome_difference": ({"phase": "TERMINAL_SUMMARY", **differences[0]}
                                                     if differences else None),
            "first_historical_transition_divergence": None,
            "historical_trajectory_comparison": "UNAVAILABLE: saved rows contain programs and terminal summaries, not intermediate states/actions"}



def intervene(panel, row, spec, baseline, panel_sha256, baseline_sha256):
    """Admit only the registered opening sheep-quantity test, before any world.

    This deliberately is not a generic JSON patch or an executor modification.
    The original row remains immutable; all later publications are exact copies.
    """
    def require(ok, message):
        if not ok:
            raise ValueError("intervention: " + message)

    allowed = {"schema", "id", "seed", "seat", "panel_sha256", "baseline_receipt_sha256",
               "saved_row_sha256", "original_program_sha256", "path", "command_id", "command_sha256",
               "original_value", "replacement_value", "fixed_later_publications"}
    require(set(spec) == allowed, "unexpected or missing specification fields")
    require(spec.get("schema") == INTERVENTION_SCHEMA, "unsupported schema")
    require(spec.get("path") == ["programs", "0", 25, "quantity"], "wrong command path")
    require(spec.get("command_id") == "0:25", "wrong command identity")
    value = spec.get("replacement_value")
    require(type(value) is int and value in (1, 5), "replacement must be integer 1 or 5")
    require(spec.get("id") == f"opening-sheep-quantity-{value}", "wrong intervention identity")
    require((row["seed"], row["seat"]) == (7000155, 1), "wrong target world/seat")
    require(type(spec.get("seed")) is int and type(spec.get("seat")) is int and
            (spec["seed"], spec["seat"]) == (row["seed"], row["seat"]), "case mismatch")
    require(panel.get("temperature") == 1 and panel.get("max_commands") == 64,
            "T=1 and budget64 required")
    require(spec.get("panel_sha256") == panel_sha256, "panel binding mismatch")
    require(spec.get("baseline_receipt_sha256") == baseline_sha256, "baseline binding mismatch")
    require(spec.get("saved_row_sha256") == digest(row), "saved row binding mismatch")
    program_hashes = {k: digest(v) for k, v in row["programs"].items()}
    require(spec.get("original_program_sha256") == program_hashes, "program binding mismatch")
    require(spec.get("fixed_later_publications") is True, "later publications must be fixed")
    command = row["programs"]["0"][25]
    require((command.get("op"), command.get("kind"), command.get("item")) ==
            ("BUY", "ANIMAL", "SHEEP"), "wrong target command")
    require(spec.get("command_sha256") == digest(command), "command binding mismatch")
    require(type(spec.get("original_value")) is int and
            spec["original_value"] == command["quantity"] == 3812, "wrong quantity preimage")
    require(baseline.get("schema") == SCHEMA and baseline.get("completed") is True and
            baseline.get("status") == "COMPLETED" and baseline.get("statuses") == ["DONE", "DONE"]
            and baseline.get("simulated_steps") == len(baseline.get("turns", [])) == 719,
            "complete maintained v2 baseline required")
    require(baseline.get("panel_sha256") == panel_sha256 and baseline.get("saved_row") == row,
            "baseline must reproduce the identical original row")
    require((baseline.get("seed"), baseline.get("seat")) == (row["seed"], row["seat"]),
            "baseline case mismatch")
    require(baseline.get("comparison", {}).get("saved_row_match") is True and
            all(baseline["actual"][k] == row[k] for k in METRICS), "baseline historical mismatch")
    require([p["step"] for p in baseline.get("publications", [])] == list(STEPS), "baseline clock")
    for pub in baseline["publications"]:
        program = row["programs"][str(pub["step"])]
        require(pub["program_sha256"] == digest(program) and
                [c["command"] for c in pub["commands"]] == program, "baseline program mismatch")
    # The diagnostic itself changes; no engine/executor dependency may change.
    diagnostic = Path(__file__).resolve().relative_to(ROOT).as_posix()
    sources = baseline.get("runtime_sources", {})
    require(all(k in sources for k in (diagnostic,
        "src/farm_neat/economy/orders.py", "src/farm_neat/game/exact_engine.py",
        "src/farm_neat/game/kaggriculture_engine_reference.py", "src/farm_neat/game/kaggriculture.json")),
        "missing baseline runtime bindings")
    for name, expected in sources.items():
        target = (ROOT / name).resolve()
        require(target.is_relative_to(ROOT) and target.is_file(), "unsafe/missing runtime source")
        if name != diagnostic:
            require(sha(target) == expected, "baseline runtime source differs: " + name)
    changed = copy.deepcopy(row)
    changed["programs"]["0"][25]["quantity"] = value
    validate_program(changed["programs"]["0"])
    return changed


def counterfactual_comparison(baseline, changed):
    """First differences in saved trace channels; never invent observation fields."""
    result = {"scope": "FIXED_LATER_PUBLICATIONS_NOT_ON_POLICY_OR_SEALED_EVALUATION",
              "baseline_simulated_steps": baseline["simulated_steps"],
              "counterfactual_simulated_steps": changed["simulated_steps"],
              "first_divergences": {}}
    channels = ("observation_sha256", "action", "unit_effects", "actual_market_commits",
                "board_changes", "delta", "successor_sha256")
    for key in channels:
        first = None
        for left, right in zip(baseline["turns"], changed["turns"], strict=True):
            if left["step"] != right["step"]:
                raise ValueError("counterfactual trace step mismatch")
            if left[key] != right[key]:
                first = {"step": left["step"], "baseline": left[key], "counterfactual": right[key]}
                break
        result["first_divergences"][key] = first
    result["terminal_differences"] = [
        {"field": k, "baseline": baseline["actual"][k], "counterfactual": changed["actual"][k],
         "delta": changed["actual"][k] - baseline["actual"][k]} for k in METRICS
        if baseline["actual"][k] != changed["actual"][k]]
    return result


def custody_cash_turn(before, pre_market, after, unit_effects, actual_market_commits):
    """Exact realized-product/cash join, not biological or causal attribution.

    Inputs are the existing holdings() and market_tap() records for ONE native
    transition. Products in shed and carried inventory are physical custody;
    unharvested tile yield is deliberately outside this balance. Post-market
    disappearance is reported separately, never silently labelled overflow.
    Counts are units, money is integer cents. No engine transition is executed.
    """
    from farm_neat.evaluation.service_timeline import cents, require
    from farm_neat.game.constants import PRODUCTS

    def qty(value):
        require(type(value) is int and value >= 0, "nonnegative integer quantity required")
        return value

    def total(stock, item):
        return sum(qty(stock[pool].get(item, 0)) for pool in ("shed", "carried"))

    fields = ("harvested", "collected_fertilizer", "bought", "sold", "consumed",
              "explicit_discard", "post_market_loss", "unit_deposit", "late_deposit",
              "receipts_cents", "product_cost_cents")
    rows = {item: dict.fromkeys(fields, 0) for item in PRODUCTS}
    unit_net = Counter()
    for effect in unit_effects:
        op = effect["engine_received_action"][0]
        for item, row in rows.items():
            ds = effect["delta"]["shed"].get(item, 0)
            dc = effect["delta"]["carried"].get(item, 0)
            require(type(ds) is int and type(dc) is int, "integer effect deltas required")
            net = ds + dc
            unit_net[item] += net
            if op in ("HARVEST", "COLLECT_FERTILIZER"):
                require(net >= 0, "negative acquisition")
                require(op != "COLLECT_FERTILIZER" or item == "FERTILIZER" or net == 0,
                        "non-fertilizer collection")
                row["harvested" if op == "HARVEST" else "collected_fertilizer"] += net
            elif op in ("FEED", "FERTILIZE"):
                require(net <= 0, "input action creates custody")
                expected = "WHEAT" if op == "FEED" else "FERTILIZER"
                require(item == expected or net == 0, "unexpected consumed product")
                row["consumed"] -= net
            elif op == "DROP":
                require(ds >= 0 and dc <= 0 and net <= 0, "invalid DROP balance")
                row["unit_deposit"] += ds
                row["explicit_discard"] -= net
            elif op == "PLACE":
                require(ds >= 0 and dc <= 0 and net == 0, "product PLACE must conserve custody")
                row["unit_deposit"] += ds
            else:
                require(net == 0, "unclassified product change: " + op)
    costs = Counter()
    receipts = 0
    for commit in actual_market_commits:
        op, item = commit["op"], commit["item"]
        n, cash = qty(commit["units"]), cents(commit["cash_delta"])
        require(n > 0 or cash == 0, "cash without a successful commit")
        if op == "SELL":
            require(item in rows and cash >= n * 100, "invalid native SELL commit")
            rows[item]["sold"] += n
            rows[item]["receipts_cents"] += cash
            receipts += cash
        else:
            require(op in ("BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL", "HIRE", "BUY_LAND"),
                    "unclassified cash operation")
            require(cash <= 0, "purchase creates cash")
            costs[op + (":" + item if item else "")] -= cash
            if op == "BUY_PRODUCT":
                require(item in ("WHEAT", "FERTILIZER"), "unbuyable product")
                rows[item]["bought"] += n
                rows[item]["product_cost_cents"] -= cash
    for item, row in rows.items():
        initial, pre, final = (total(stock, item) for stock in (before, pre_market, after))
        require(pre - initial == unit_net[item], "unit-phase custody mismatch: " + item)
        require(after["shed"].get(item, 0) >= 0, "negative final shed")
        row["late_deposit"] = (after["shed"].get(item, 0) - pre_market["shed"].get(item, 0)
                               - row["bought"] + row["sold"])
        require(row["late_deposit"] >= 0, "unclassified post-market shed removal: " + item)
        row["post_market_loss"] = pre + row["bought"] - row["sold"] - final
        require(row["post_market_loss"] >= 0, "unclassified post-market acquisition: " + item)
        row["initial_units"], row["terminal_units"] = initial, final
        residual = (initial + row["harvested"] + row["collected_fertilizer"] + row["bought"]
                    - row["sold"] - row["consumed"] - row["explicit_discard"]
                    - row["post_market_loss"] - final)
        require(residual == 0, "product custody mismatch: " + item)
        row["residual_units"] = residual
    cash_delta = cents(after["cash"]) - cents(before["cash"])
    require(cents(pre_market["cash"]) == cents(before["cash"]), "cash changed before market")
    require(cash_delta == receipts - sum(costs.values()), "cash/commit mismatch")
    return {"schema": "farm-custody-cash-turn-v1", "products": rows,
            "receipts_cents": receipts, "costs_cents": dict(costs),
            "cash_delta_cents": cash_delta, "cash_residual_cents": 0,
            "scope": "REALIZED_CUSTODY_AND_COMMITS_ONLY; NOT_BIOLOGICAL_OUTPUT_OR_CAUSAL_LABEL"}


def paired_cash_bridge(parent, child, reference_prices_cents):
    """Exact two-ledger cash decomposition using a declared common price vector.

    Each input has initial_cash_cents, terminal_cash_cents, products[item] with
    sold and receipts_cents, plus costs_cents. The quantity term uses the SAME
    exogenous reference price on both sides, including a product never sold by
    one arm. The complementary price/timing term is an accounting residual, NOT
    the result of a frozen-price game or an identified causal mediator.
    """
    from farm_neat.evaluation.service_timeline import require
    from farm_neat.game.constants import PRODUCTS

    require(set(reference_prices_cents) == set(PRODUCTS), "complete common reference vector required")
    require(all(type(p) is int and p >= 0 for p in reference_prices_cents.values()),
            "nonnegative integer reference cents required")
    for data in (parent, child):
        require(set(data["products"]) == set(PRODUCTS), "complete product ledger required")
        for field in ("initial_cash_cents", "terminal_cash_cents"):
            require(type(data[field]) is int and data[field] >= 0, "integer bank cents required")
        require(all(type(v) is int and v >= 0 for v in data["costs_cents"].values()),
                "nonnegative integer costs required")
        for row in data["products"].values():
            require(all(type(row[k]) is int and row[k] >= 0 for k in ("sold", "receipts_cents")),
                    "nonnegative integer sales required")
            require(row["sold"] > 0 or row["receipts_cents"] == 0, "receipts without sales")
        net = sum(r["receipts_cents"] for r in data["products"].values()) - sum(data["costs_cents"].values())
        require(data["terminal_cash_cents"] - data["initial_cash_cents"] == net, "ledger cash mismatch")
    rows = {}
    for item in PRODUCTS:
        left, right = parent["products"][item], child["products"][item]
        dq = right["sold"] - left["sold"]
        dr = right["receipts_cents"] - left["receipts_cents"]
        quantity = dq * reference_prices_cents[item]
        rows[item] = {"delta_sold_units": dq, "delta_receipts_cents": dr,
                      "reference_quantity_cents": quantity,
                      "price_timing_residual_cents": dr - quantity,
                      "sales_support": "BOTH" if left["sold"] and right["sold"] else
                          "CHILD_ONLY" if right["sold"] else "PARENT_ONLY" if left["sold"] else "NEITHER"}
    cost_keys = parent["costs_cents"].keys() | child["costs_cents"].keys()
    costs = {k: child["costs_cents"].get(k, 0) - parent["costs_cents"].get(k, 0) for k in sorted(cost_keys)}
    initial = child["initial_cash_cents"] - parent["initial_cash_cents"]
    terminal = child["terminal_cash_cents"] - parent["terminal_cash_cents"]
    quantity = sum(r["reference_quantity_cents"] for r in rows.values())
    price = sum(r["price_timing_residual_cents"] for r in rows.values())
    require(initial + quantity + price - sum(costs.values()) == terminal, "paired cash bridge mismatch")
    return {"schema": "farm-paired-cash-bridge-v1", "products": rows,
            "reference_prices_cents": dict(reference_prices_cents),
            "delta_initial_cents": initial, "delta_costs_cents": costs,
            "reference_quantity_cents": quantity, "price_timing_residual_cents": price,
            "delta_terminal_cents": terminal, "residual_cents": 0,
            "scope": "EXACT_ACCOUNTING_IDENTITY; NOT_INDEPENDENT_QUANTITY_OR_PRICE_TREATMENTS"}

def source_identity():
    files = {Path(__file__).resolve().relative_to(ROOT).as_posix(): sha(__file__)}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("farm_neat.") and path and Path(path).is_file():
            path = Path(path).resolve()
            if path.is_relative_to(ROOT):
                files[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ("kaggriculture_engine_reference.py", "kaggriculture.json"):
        path = ROOT / "src/farm_neat/game" / name
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    return dict(sorted(files.items()))


def blocker_links(block, executor, owners):
    refs = []
    if "tile" in block:
        key = ",".join(map(str, block["tile"]))
        if key in executor.orders:
            refs.append(stage_ref("order_stages", key, executor.orders[key]))
    key = block.get("rule")
    if key in executor.rules:
        refs.append(stage_ref("rule_stages", key, executor.rules[key]))
    return [{"ref": ref, "writer": owners.get(ref)} for ref in refs]


def replay(panel, row, receipt):
    """One complete fixed-program world, retaining partial evidence on exceptions."""
    seed, seat = row["seed"], row["seat"]
    receipt.update(status="RUNNING", completed=False, simulated_steps=0,
                   publications=[], active_sets={}, turns=[])
    env = ExactKaggricultureEnvironment(seed=seed)
    receipt["runtime_sources"] = source_identity()
    controller = FullSeasonPlan(f"eval-{seed}-{seat}", publication_steps=STEPS)
    observation = env.agent_observation(seat)
    controller.observe(observation)
    receipt["initial"] = holdings(observation, seat)
    receipt["initial_observation_sha256"] = digest(observation)
    owners, total_blocked, total_markets = {}, Counter(), {}
    physical, rejected = 0, 0
    with market_tap(env, seat) as tap:
        for step in range(719):
            receipt["in_progress"] = {"step": step, "phase": "publication_or_dispatch"}
            if env.done or env.step_index != step:
                raise ValueError(f"world left expected horizon at {step}")
            if step in STEPS:
                controller, pub, owners = publication(controller, observation, row["programs"][str(step)], owners)
                pub["holdings"] = holdings(observation, seat)
                receipt["publications"].append(pub)
                rejected += sum(c["status"] == "REFUSED" for c in pub["commands"])
            selected = active(controller.executor, owners)
            selected_sha = digest(selected)
            receipt["active_sets"].setdefault(selected_sha, selected)
            before = holdings(observation, seat)
            action = controller.act(observation, env.configuration)
            report = copy.deepcopy(controller.last_report)
            for block in report["blocked"]:
                block["references"] = blocker_links(block, controller.executor, owners)
            event_start = len(controller.executor.events)
            physical += int(action != PASS_ACTION)
            tap["commits"], tap["pre_market"], tap["unit_effects"] = {}, None, []
            joint = [copy.deepcopy(PASS_ACTION), copy.deepcopy(PASS_ACTION)]
            joint[seat] = action
            receipt["in_progress"] = {"step": step, "phase": "engine_transition", "issued_action": action}
            env.step(joint)
            receipt["simulated_steps"] = env.step_index
            successor = env.agent_observation(seat)
            receipt["in_progress"]["phase"] = "reconciliation"
            gates = sale_gates(controller.executor, tap["pre_market"], observation, owners)
            controller.observe(successor)  # includes step 718 settlement at terminal 719
            after = holdings(successor, seat)
            commits = list(tap["commits"].values())
            cash_error = (after["cash"]-before["cash"]) - sum(x["cash_delta"] for x in commits)
            if cash_error != 0:
                raise ValueError(f"unattributed cash delta {cash_error} at {step}")
            report["routes"] = [{**route, "ref": next((x["ref"] for x in selected
                if x["ref"].startswith(f"order_stages/{route['target'][0]},{route['target'][1]}@")), None)}
                                 for route in report["routes"]]
            record = {"step": step, "observation_sha256": digest(observation),
                "successor_sha256": digest(successor), "active_set": selected_sha,
                "action": action, "executor_report": report,
                "cash_before": before["cash"], "cash_after": after["cash"],
                "delta": delta(before, after), "actual_market_commits": commits,
                "unit_effects": tap["unit_effects"],
                "board_changes": [{"tile": [x, y], "before": a, "after": b}
                    for y, (left, right) in enumerate(zip(observation["farms"][seat]["tiles"],
                                                         successor["farms"][seat]["tiles"]))
                    for x, (a, b) in enumerate(zip(left, right)) if a != b],
                "sale_gates": gates,
                "events": copy.deepcopy(controller.executor.events[event_start:])}
            receipt["turns"].append(record)
            total_blocked.update(b["cause"] for b in report["blocked"])
            for commit in commits:
                key = commit["op"] + (":" + commit["item"] if commit["item"] else "")
                bucket = total_markets.setdefault(key, {"units": 0, "cash_delta": 0.0, "failed_unit_attempts": 0})
                for field in bucket:
                    bucket[field] += commit[field]
            observation = successor
    receipt.pop("in_progress", None)
    receipt["terminal"] = holdings(observation, seat)
    receipt["terminal_stages"] = [{"ref": ref, "writer": owners.get(ref), "state": state}
                                   for ref, state in stages(controller.executor).items()]
    actual = {"own_cash": receipt["terminal"]["cash"],
              "opponent_cash": receipt["terminal"]["opponent_cash"],
              "physical_turns": physical, "rejected_commands": rejected}
    comparison = ({"saved_row_match": None, "differences": [],
                   "applicability": "NOT_APPLICABLE_INTENTIONAL_COUNTERFACTUAL",
                   "first_historical_transition_divergence": None}
                  if "intervention" in receipt else outcome_comparison(row, actual))
    receipt.update(actual=actual, comparison=comparison,
                   blocker_record_counts=dict(total_blocked), actual_market_totals=total_markets,
                   statuses=[str(r.status) for r in env.state])
    receipt["completed"] = env.done and env.step_index == 719 and receipt["statuses"] == ["DONE", "DONE"]
    receipt["status"] = "COMPLETED" if receipt["completed"] else "INCOMPLETE"
    if not receipt["completed"]:
        raise ValueError("natural complete 719-transition game required")


def journal_event(path, event, *, start=False, attempt_limit=12):
    """Register before reading/admitting a case; never recycle failed attempts."""
    if type(attempt_limit) is not int or not 1 <= attempt_limit <= 12:
        raise ValueError("attempt limit must be an integer from 1 to 12")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        prior = [json.loads(line) for line in stream if line.strip()]
        if any(r.get("schema") != JOURNAL_SCHEMA for r in prior):
            raise ValueError("foreign attempt journal")
        if start:
            attempts = [r for r in prior if r.get("status") == "STARTED"]
            if len(attempts) >= attempt_limit:
                raise ValueError(f"{attempt_limit}-attempt journal budget exhausted; no game started")
            event = {**event, "attempt": len(attempts)+1}
        event = {"schema": JOURNAL_SCHEMA, "utc": datetime.now(timezone.utc).isoformat(), **event}
        stream.seek(0, 2)
        stream.write(encoded(event).decode())
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return event


@contextmanager
def deadline(seconds):
    def expired(_signum, _frame):
        raise TimeoutError("fixed-program elapsed-time limit exceeded")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--panel-sha256", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--seat", type=int, choices=(0, 1), required=True)
    p.add_argument("--out", type=Path, required=True, help="new JSON or JSON.gz receipt")
    p.add_argument("--journal", type=Path, required=True, help="append-only budget: 6 for interventions, 12 otherwise")
    p.add_argument("--max-seconds", type=float, default=120.0)
    p.add_argument("--intervention", type=Path, help="registered single-field specification JSON")
    p.add_argument("--intervention-sha256")
    p.add_argument("--baseline-receipt", type=Path, help="included complete unmodified v2 receipt")
    p.add_argument("--baseline-sha256")
    a = p.parse_args(argv)
    extra = (a.intervention, a.intervention_sha256, a.baseline_receipt, a.baseline_sha256)
    if any(x is not None for x in extra) and not all(x is not None for x in extra):
        p.error("intervention requires specification and baseline receipt with both SHA-256 values")
    if a.intervention and (len({x.resolve() for x in
            (a.panel, a.out, a.journal, a.intervention, a.baseline_receipt)}) != 5):
        p.error("all intervention inputs, journal and output must be distinct")
    if len({a.panel.resolve(), a.out.resolve(), a.journal.resolve()}) != 3 or a.out.exists():
        p.error("distinct input/journal and a new output path required; nothing overwritten")
    if not math.isfinite(a.max_seconds) or a.max_seconds <= 0:
        p.error("positive finite timeout required")
    intervention_registration = ({"intervention_file": str(a.intervention),
        "intervention_sha256": a.intervention_sha256, "baseline_receipt": str(a.baseline_receipt),
        "baseline_receipt_sha256": a.baseline_sha256} if a.intervention else {})
    registration = journal_event(a.journal, {"status": "STARTED", "seed": a.seed, "seat": a.seat,
        "panel": str(a.panel), "panel_sha256": a.panel_sha256, "output": str(a.out),
        **intervention_registration}, start=True, attempt_limit=6 if a.intervention else 12)
    receipt = {"schema": SCHEMA, "attempt": registration["attempt"], "seed": a.seed, "seat": a.seat,
        "opponent": "pass", "panel": str(a.panel), "panel_sha256": a.panel_sha256,
        "python": sys.version, "replay_kind": "FIXED_RECORDED_PROGRAMS_NO_NEURAL_RESAMPLING",
        "neural_resampling": False, "eligible_for_neural_gate": False,
        "completed": False, "simulated_steps": 0, "command_argv": list(argv if argv is not None else sys.argv[1:]),
        "limitations": ["Selected, already-exposed worlds; not a policy-strength estimate.",
            "Blocker counts are repeated records, not independent failures or proven infeasibility.",
            "Routes/credit are executor reports. Aggregate production requisitions are not apportioned to commands.",
            "SALE gates omit additional requisition reserves and queue-position changes; absence of a blocker is not fulfillment.",
            "Equal terminal metrics do not establish historical trajectory parity.",
            "Market taps delegate unchanged interpreter functions and are serial/process-local."]}
    receipt.update(intervention_registration)
    began = time.monotonic()
    try:
        with deadline(a.max_seconds):
            actual_sha = sha(a.panel)
            if actual_sha != a.panel_sha256:
                raise ValueError("panel SHA-256 mismatch")
            panel = json.loads(a.panel.read_text())
            row = validate_case(panel, a.seed, a.seat)
            receipt["saved_row"] = copy.deepcopy(row)
            receipt["saved_row_sha256"] = digest(row)
            receipt["original_program_sha256"] = {k: digest(v) for k, v in row["programs"].items()}
            baseline = None
            if a.intervention:
                if sha(a.intervention) != a.intervention_sha256:
                    raise ValueError("intervention SHA-256 mismatch")
                spec = json.loads(a.intervention.read_text())
                receipt["requested_intervention"] = spec  # retained even when admission fails
                if sha(a.baseline_receipt) != a.baseline_sha256:
                    raise ValueError("baseline receipt SHA-256 mismatch")
                raw = a.baseline_receipt.read_bytes()
                baseline = json.loads(gzip.decompress(raw) if a.baseline_receipt.suffix == ".gz" else raw)
                row = intervene(panel, row, spec, baseline, a.panel_sha256, a.baseline_sha256)
                receipt["intervention"] = spec
                receipt["replay_kind"] = "REGISTERED_SINGLE_FIELD_COUNTERFACTUAL_FIXED_LATER_PROGRAMS"
                diagnostic = Path(__file__).resolve().relative_to(ROOT).as_posix()
                receipt["baseline_diagnostic_identity"] = {
                    "path": diagnostic, "baseline": baseline["runtime_sources"][diagnostic],
                    "current": sha(__file__), "other_bound_runtime_sources_match": True}
                receipt["limitations"] += [
                    "Later publications are fixed recorded programs, off-policy in altered states.",
                    "Changed occupancy can change subsequent weed/shop RNG draws despite the same initial seed.",
                    "This selected intervention cannot explain all losing pairs or establish neural strength."]
            receipt["executed_program_sha256"] = {k: digest(v) for k, v in row["programs"].items()}
            receipt["recorded_model_sha256"] = panel.get("model_file_sha256")
            receipt["sampler_seed_metadata_only"] = row.get("sampler_seed")
            current_sha = sha(ROOT / "src/farm_neat/economy/orders.py")
            saved_sha = panel.get("executor_source_sha256")
            receipt["executor_identity"] = {"saved": saved_sha, "current": current_sha,
                "matches": saved_sha == current_sha,
                "first_identity_difference": None if saved_sha == current_sha else "BEFORE_REPLAY: executor source bytes differ"}
            replay(panel, row, receipt)
            if baseline is not None:
                receipt["counterfactual_comparison"] = counterfactual_comparison(baseline, receipt)
    except Exception as exc:
        receipt.update(status="ERROR", completed=False,
                       error={"type": type(exc).__name__, "message": str(exc)})
    receipt["elapsed_seconds"] = time.monotonic()-began
    receipt["elapsed_scope"] = "ADMISSION_AND_DIAGNOSTIC_REPLAY; EXCLUDES_REGISTRATION_AND_OUTPUT_SERIALIZATION"
    payload = encoded(receipt)
    if a.out.suffix == ".gz":
        payload = gzip.compress(payload, mtime=0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("xb") as stream:
        stream.write(payload)
    ending = journal_event(a.journal, {"attempt": registration["attempt"], "status": receipt["status"],
        "seed": a.seed, "seat": a.seat, "simulated_steps": receipt["simulated_steps"],
        "completed": receipt["completed"], "outcome": receipt.get("actual"),
        "error": receipt.get("error"), "saved_row_match": receipt.get("comparison", {}).get("saved_row_match"),
        "output": str(a.out), "output_sha256": sha(a.out), **intervention_registration})
    print(json.dumps(ending, sort_keys=True))
    valid = ("counterfactual_comparison" in receipt if a.intervention
             else receipt.get("comparison", {}).get("saved_row_match") is True)
    return 0 if receipt["completed"] and valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
