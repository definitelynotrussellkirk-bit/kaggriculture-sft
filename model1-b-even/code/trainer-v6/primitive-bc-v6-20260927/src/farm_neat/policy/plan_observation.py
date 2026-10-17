"""Full-season plans compiled into the existing typed 80-column entity vocabulary.

Owner: policy representation. This is the scheduled-plan adapter, NOT another
network or a replacement for the frozen v1 snapshot packer. The run must bind
identity() in addition to the vocabulary codec/model identity. Past intervals and
all current/future stages are explicit; no active-only collapse or token truncation.
Tests live in test_full_economic_observation.py and test_full_economic_orders.py.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

from farm_neat.policy import observation as codec
from farm_neat.economy import linter as advice
from farm_neat.economy.orders import EconomicOrders, FullSeasonPlan, ScheduledEconomicOrders

SCHEMA = "farm-full-season-to-entity-v4"
PACK_SCHEMA = "farm-full-season-entity-pack-v4"
# Preserve effective dates in existing COMMAND columns. The three previously
# unused columns 75..77 expose the original terms needed for same-stage edits,
# without another token, a new model shape, or invented dates for untimed rules.
TERM_FIELDS = (("original_start", 720), ("original_until", 720), ("original_game_end", 1))
TERM_OFFSET = codec.PAYLOAD + len(codec.FIELDS["COMMAND"])
assert TERM_OFFSET + len(TERM_FIELDS) <= codec.WIDTH
# ARGUMENT was reserved, never populated by the frozen base codec. This adapter
# defines its payload explicitly; v2 packs are NOT accepted as base/v1 packs.
BLOCK_CAUSES = ("LAND_NOT_UNLOCKED", "PLACED_ANIMAL_CANNOT_BE_CLEARED",
    "OCCUPIED_BY_OTHER_USE", "TILE_DUTY_SERIALIZED", "UNASSIGNED_DUTY",
    "MARKET_ORDER_SLOTS", "HIRE_CASH", "LAND_CASH", "ACQUISITION_SALE_SERIALIZED",
    "ACQUISITION_RESET_ACCOUNTING_DEFERRED", "CASH_OR_SHED_CAPACITY")
RULE_KINDS = ("SALE", "STOCK", "BUY", "LAND", "HANDS", "CASH_RESERVE", "PREREQUISITE")
UNIT_OPS = ("PLANT", "DIG", "WATER", "FERTILIZE", "HARVEST", "BUILD_COOP",
            "BUILD_PASTURE", "PLACE", "FEED", "CARE", "COLLECT_FERTILIZER")
BLOCK_FIELDS = (*(("cause:"+c, 1) for c in BLOCK_CAUSES),
    ("report_step", 720), ("report_age", 720), ("reported_cost", 10000),
    ("free_units", 100), ("command_linked", 1), ("command_remaining_steps", 720),
    ("resource_required", 1), ("required_units", 100),
    *(("rule:"+c, 1) for c in RULE_KINDS), *(("unit:"+c, 1) for c in UNIT_OPS),
    *(("stock_kind:"+c, 1) for c in codec.commands.STOCK_KINDS))
assert len(BLOCK_FIELDS) <= codec.WIDTH-codec.PAYLOAD
# Existing ARGUMENT blockers leave column79 unused. Attention tokens retain the
# common own-seat/tile/item header and use this column to identify payload chunks.
# No existing row/column changes meaning. Chunks with entirely unknown payload
# are absent; there is no truncation of a known diagnostic fact.
ATTENTION_MARKER = codec.WIDTH - 1
ATTENTION_CHUNK = ATTENTION_MARKER - codec.PAYLOAD
assert codec.PAYLOAD + len(BLOCK_FIELDS) <= ATTENTION_MARKER
ATTENTION_FIELDS = tuple((n, s) for n, s in advice.ATTENTION_FIELDS
                        if n not in ("x", "y") and not n.startswith("item:"))
ATTENTION_SKIP_CODES = ("STALE_EXECUTOR_REPORT", "REPORTED_BLOCKED_JOB_TURNS")
ROOT = Path(__file__).resolve().parents[3]
_OWN = ("src/farm_neat/policy/plan_observation.py", "src/farm_neat/economy/orders.py",
        "src/farm_neat/economy/linter.py")
_LOADED = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in _OWN}
SPEC = {"schema": SCHEMA, "pack_schema": PACK_SCHEMA, "plan_schema": FullSeasonPlan.SCHEMA,
    "entity_vocabulary_sha256": codec.REPRESENTATION_SCHEMA_SHA256,
    "weight_conversion": "NONE_EXISTING_FIELDS_NETWORK_AND_ADAM_UNCHANGED; APPENDED_ATTENTION_TOKENS_CHANGE_INPUTS",
    "order": "WORLD_THEN_ELAPSED_SEGMENTS_THEN_CURRENT_PLAN_THEN_BLOCKERS_THEN_ATTENTION_THEN_CONTROL",
    "stages": "ORDER_MAP_THEN_RULE_MAP_INSERTION_ORDER_THEN_STAGE_START",
    "time_fields": "EFFECTIVE_INCLUSIVE_INTERVAL_INTERSECTED_WITH_PLAN_SEGMENT",
    "credit": "CUMULATIVE_AS_OF_SNAPSHOT_NOT_ADDITIVE_ACROSS_SEGMENTS",
    "plan_vs_world": "INTENDED_COMMANDS_AND_ACTUAL_CREDIT_SEPARATE_FROM_OBSERVED_ENTITIES",
    "control_counts": "ALL_ENCODED_STAGE_ENTRIES_INCLUDING_ELAPSED",
    "past_original_terms": "ORIGINAL_START_UNTIL_GAME_END_IN_COMMAND_TAIL; EFFECTIVE_DATES_UNCHANGED",
    "command_term_fields": [list(field) for field in TERM_FIELDS],
    "command_term_offset": TERM_OFFSET,
    "command_term_knownness": "ONLY_EXPLICIT_ORIGINAL_FIELDS; UNTIMED_CLEAR_CASH_RESERVE_UNKNOWN",
    "source_binding": "EXPLICIT_V4_PACK_AND_ADAPTER; V1_V2_V3_PACKS_ARE_NOT_RELABELED",
    "attention": {"spec_sha256": advice.ATTENTION_SPEC_SHA256,
        "fields": [list(x) for x in ATTENTION_FIELDS], "chunk_width": ATTENTION_CHUNK,
        "marker_column": ATTENTION_MARKER, "marker_value": "ONE_BASED_CHUNK_INDEX",
        "skipped_codes": list(ATTENTION_SKIP_CODES),
        "skip_reason": "ALREADY_REPRESENTED_BY_BLOCKER_TOKENS_AND_CONTROL_REPORT_AGE",
        "visibility": "OWN_OBSERVATION_PLUS_OBSERVED_FULL_PLAN_BEFORE_NEW_PROGRAM_SAMPLING",
        "semantics": "CONDITIONAL_MECHANICS_AND_LAST_REVISION_SALE_TERMS; NO_PROFITABILITY_MASK_OR_REWARD"},
    "argument_fields": [list(field) for field in BLOCK_FIELDS],
    "argument_semantics": "REPORTED_DISPATCH_CONSTRAINT_NOT_PROFITABILITY_OR_ACTION_MASK",
    "argument_ordinal": "LINKED_COMMAND_ORDINAL_AT_REPORT_STEP; UNKNOWN_IF_NO_UNIQUE_LINK",
    "argument_item": "REQUIRED_INPUT_ELSE_REPORTED_ITEM_ELSE_UNIT_ACTION_ITEM",
    "argument_knownness": "MISSING_IS_UNKNOWN; COMBINED_CASH_OR_CAPACITY_REASON_IS_NOT_DISAMBIGUATED",
    "argument_time": "LAST_REPORT_STEP_AND_AGE; NOT_ASSUMED_CURRENT_OR_AFTER_NEW_SHOP",
    "argument_counts": "ONE_TOKEN_PER_REPORTED_BLOCK; NO_TRUNCATION_OR_SYNTHETIC_HIRE_ADVICE"}
REPRESENTATION_SCHEMA_SHA256 = codec.digest(SPEC)


def identity():
    current = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in _OWN}
    if current != _LOADED:
        raise ValueError("loaded full-plan adapter source changed")
    result = {**SPEC, "source_pins": {**codec._LOADED_SOURCES, **_LOADED}}
    return {**result, "content_sha256": codec.digest(result)}


def _append_stage(result, row, key, group, prefix, ordinal, first, last, step):
    c = row["command"]
    start, until = max(c.get("start", 0), first), min(ScheduledEconomicOrders._end(row), last)
    if start > until:
        return False
    # Reuse the EXACT typed command/credit serializer and validation. Only the
    # temporal projection and roster assembly belong to this source version.
    legacy = EconomicOrders().snapshot()
    legacy[group] = {key: ScheduledEconomicOrders._legacy_row(row)}
    encoded = codec._Rows(); codec._state(legacy, step, encoded)
    columns = {name: (codec.PAYLOAD+i, scale) for i, (name, scale) in enumerate(codec.FIELDS["COMMAND"])}
    for values, known, name in zip(encoded.rows[:-1], encoded.known[:-1], encoded.ids[:-1], strict=True):
        values[16] = ordinal / 100
        known[16] = True
        if values[codec.KINDS.index("COMMAND")] == 1:
            # Channel meanings/units do not change. CLEAR and CASH_RESERVE gain
            # their enclosing plan's effective interval, not fictitious actions.
            for field, value in (("start", start), ("until", until), ("game_end", until == 718)):
                index, scale = columns[field]; values[index] = value / scale; known[index] = True
            terms = {}
            if "start" in c:
                terms["original_start"] = c["start"]
            if "until" in c:
                terms["original_until"] = 718 if c["until"] == "GAME_END" else c["until"]
                terms["original_game_end"] = c["until"] == "GAME_END"
            for index, (field, scale) in enumerate(TERM_FIELDS, TERM_OFFSET):
                if field in terms:
                    values[index] = terms[field] / scale; known[index] = True
        result.rows.append(values); result.known.append(known); result.ids.append(prefix+"/"+name)
    return True


def _append_blockers(result, report, stages, step):
    blocks = report.get("blocked", [])
    if not isinstance(blocks, list):
        raise ValueError("typed blocker list required")
    if not blocks:
        return
    reported = codec._number(report.get("step"), "blocker report step", integer=True, minimum=0)
    if reported > step:
        raise ValueError("future blocker report is not actor-visible")
    allowed = {"cause", "tile", "repair", "rule", "next_hire_cost", "cost", "op", "need", "free_units", "item"}
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or set(block)-allowed or block.get("cause") not in BLOCK_CAUSES:
            raise ValueError("unknown or unrepresented blocker; update the versioned adapter")
        if "repair" in block and not isinstance(block["repair"], str):
            raise ValueError("blocker repair prose must be a string")
        tile = codec._position(block["tile"]) if "tile" in block else None
        rule, kind = block.get("rule"), None
        if rule is not None:
            if not isinstance(rule, str): raise ValueError("typed blocker rule identity")
            parts = rule.split(":"); kind = parts[0]
            if kind not in RULE_KINDS:
                raise ValueError("unknown blocker rule kind")
            if kind in ("LAND", "HANDS", "CASH_RESERVE"):
                valid = len(parts) == 1
            elif kind == "SALE":
                valid = len(parts) == 2 and parts[1] in codec.PRODUCTS
            else:
                valid = len(parts) == 3 and parts[1] in codec.commands.STOCK_KINDS
                if valid:
                    # Reuse the actual stock grammar for item/kind compatibility.
                    codec.commands.validate_command(dict(op="STOCK", kind=parts[1], item=parts[2],
                        target=1, price_below=1, start=0, until="GAME_END", priority=0))
            if not valid: raise ValueError("invalid blocker rule identity")
        op = block.get("op")
        if op is not None and (not isinstance(op, list) or not op or op[0] not in UNIT_OPS
                or len(op) != (2 if op[0] in ("PLANT", "PLACE") else 1)):
            raise ValueError("typed reported unit operation required")
        if op is not None and len(op) == 2 and op[1] not in (codec.CROPS if op[0] == "PLANT" else codec.ANIMALS):
            raise ValueError("reported unit operation item does not match its family")
        need = block.get("need")
        if need is not None and need not in codec.ITEMS:
            raise ValueError("known required resource identity")
        item = need or block.get("item") or (op[1] if op is not None and len(op) == 2 else None)
        if item is None and rule is not None and ":" in rule: item = rule.split(":")[-1]
        if item is not None and item not in codec.ITEMS: raise ValueError("known blocker item identity")
        # A previous-turn report may refer to a stage that expired this morning,
        # or to the frozen prefix after publication. Do not bind it to a new stage.
        candidates = [s for s in stages if s["first"] <= reported <= s["last"] and (
            (tile is not None and s["group"] == "orders" and s["key"] == f"{tile[0]},{tile[1]}") or
            (rule is not None and s["group"] == "rules" and s["key"] == rule))]
        linked = candidates[0] if len(candidates) == 1 else None
        values = {"cause:"+c: c == block["cause"] for c in BLOCK_CAUSES}
        values.update(report_step=reported, report_age=step-reported, command_linked=linked is not None)
        if linked: values["command_remaining_steps"] = linked["last"]-step
        if "free_units" in block:
            values["free_units"] = codec._number(block["free_units"], "free units", integer=True, minimum=0)
        costs = [codec._number(block[k], "reported cost", minimum=0) for k in ("cost", "next_hire_cost") if k in block]
        if costs:
            if len(set(costs)) != 1: raise ValueError("contradictory reported costs")
            values["reported_cost"] = costs[0]
        if "need" in block:
            values["resource_required"] = need is not None
            values["required_units"] = 1 if need is not None else 0
        if kind: values.update({"rule:"+k: k == kind for k in RULE_KINDS})
        if kind in ("STOCK", "BUY", "PREREQUISITE"):
            values.update({"stock_kind:"+k: k == parts[1] for k in codec.commands.STOCK_KINDS})
        if op: values.update({"unit:"+k: k == op[0] for k in UNIT_OPS})
        # Reuse the base common header constructor; no CONTROL payload is set.
        header = codec._Rows()
        header.add("CONTROL", f"blocker/{index}", role=0, position=tile,
            ordinal=None if linked is None else linked["ordinal"], item=item)
        row, known = header.rows[0], header.known[0]
        row[codec.KINDS.index("CONTROL")] = 0; row[codec.KINDS.index("ARGUMENT")] = 1
        for column, (name, scale) in enumerate(BLOCK_FIELDS, codec.PAYLOAD):
            if name in values:
                row[column] = values[name]/scale; known[column] = True
        if not np.isfinite(row).all(): raise ValueError("blocker exceeds finite FP32 range")
        result.rows.append(row); result.known.append(known); result.ids.append(header.ids[0])


def _base_view(packed):
    """Private structural-validator view, never emitted as recorded v1 evidence."""
    result = {**packed, "schema": codec.SCHEMA,
        "representation_schema_sha256": codec.REPRESENTATION_SCHEMA_SHA256,
        "source_pins": dict(codec._LOADED_SOURCES), "source_sha256": codec.digest(codec._LOADED_SOURCES)}
    result["content_sha256"] = codec._content(result)
    return result


def _append_attention(result, observation, plan):
    status = advice.attention(observation, plan)
    fields = status["features"]["fields"]
    indices = [fields.index(name) for name, _ in ATTENTION_FIELDS]
    for index, finding in enumerate(status["findings"]):
        if finding["code"] in ATTENTION_SKIP_CODES:
            continue
        facts = finding["observed"]
        item = finding["product"] or facts.get("species") or facts.get("crop")
        values = np.asarray(status["features"]["rows"][index], np.float32)[indices]
        known = np.asarray(status["features"]["knownness"][index], np.bool_)[indices]
        for chunk, first in enumerate(range(0, len(indices), ATTENTION_CHUNK)):
            last = min(first + ATTENTION_CHUNK, len(indices))
            if not known[first:last].any():
                continue
            header = codec._Rows()
            name = f"attention/{index}/{chunk}"
            header.add("CONTROL", name, role=0, position=finding["tile"], item=item)
            row, mask = header.rows[0], header.known[0]
            row[codec.KINDS.index("CONTROL")] = 0
            row[codec.KINDS.index("ARGUMENT")] = 1
            row[codec.PAYLOAD:codec.PAYLOAD+last-first] = values[first:last]
            mask[codec.PAYLOAD:codec.PAYLOAD+last-first] = known[first:last]
            row[ATTENTION_MARKER], mask[ATTENTION_MARKER] = chunk + 1, True
            result.rows.append(row); result.known.append(mask); result.ids.append(name)


def validate(packed):
    expected = {"schema", "representation_schema_sha256", "source_pins", "source_sha256",
                "rows", "knownness", "token_mask", "token_ids", "content_sha256"}
    pins = identity()["source_pins"]
    if (not isinstance(packed, dict) or set(packed) != expected or packed["schema"] != PACK_SCHEMA
            or packed["representation_schema_sha256"] != REPRESENTATION_SCHEMA_SHA256
            or packed["source_pins"] != pins or packed["source_sha256"] != codec.digest(pins)
            or packed["content_sha256"] != codec._content(packed)):
        raise ValueError("exact full-plan v4 representation/source/content required")
    codec.validate(_base_view(packed))
    return packed


def collate(packs):
    if not isinstance(packs, (list, tuple)) or not packs:
        raise ValueError("nonempty full-plan v4 batch required")
    result = codec.collate([_base_view(validate(p)) for p in packs])
    result.update(schema=PACK_SCHEMA, content_sha256s=tuple(p["content_sha256"] for p in packs))
    return result


def pack(observation, plan, configuration=None, *, expected_identity):
    """Return an encoder-ready pack; the caller also binds this adapter identity.

    Existing columns and blocker tokens keep their meanings. V4 appends typed
    attention facts before sampling: service/feed/labor and last-revision stock.
    The 60M weights/Adam need no conversion; new source/input identity is required.
    """
    if expected_identity != identity():
        raise ValueError("explicit full-plan adapter source/semantics identity required")
    if not isinstance(plan, FullSeasonPlan):
        plan = FullSeasonPlan.restore(plan)
    if plan.executor._observation_step(observation) != plan.executor.stage_step:
        raise ValueError("full-plan input must use the current observed clock")
    if plan.executor.player != observation["player"]:
        raise ValueError("observe the current seat before full-plan encoding")
    result = codec._Rows()
    step = codec._world(observation, {} if configuration is None else configuration, result)
    order_count = rule_count = ordinal = 0
    stage_links = []
    segments = [(f"past{index}", segment["from_step"], segment["until_step"], segment["state"])
                for index, segment in enumerate(plan.past)]
    segments.append(("current", plan.committed_step or 0, 718, plan.executor.snapshot()))
    for name, first, last, state in segments:
        for stages, group in (("order_stages", "orders"), ("rule_stages", "rules")):
            for key, rows in state[stages].items():
                for index, row in enumerate(rows):
                    if _append_stage(result, row, key, group, f"{name}/stage{index}", ordinal, first, last, step):
                        stage_links.append(dict(group=group, key=key, ordinal=ordinal,
                            first=max(row["command"].get("start", 0), first),
                            last=min(ScheduledEconomicOrders._end(row), last)))
                        ordinal += 1
                        order_count += group == "orders"; rule_count += group == "rules"
    active = EconomicOrders().snapshot()
    active.update(orders={k: ScheduledEconomicOrders._legacy_row(v) for k,v in plan.orders.items()},
        rules={k: ScheduledEconomicOrders._legacy_row(v) for k,v in plan.rules.items()},
        pending=copy.deepcopy(plan.executor.pending), events=copy.deepcopy(plan.executor.events),
        last_report=copy.deepcopy(plan.last_report), revision=plan.revision)
    summary = codec._Rows(); codec._state(active, step, summary)
    control, known = summary.rows[-1], summary.known[-1]
    for name, value in (("order_count", order_count), ("rule_count", rule_count)):
        index = [field for field, _ in codec.FIELDS["CONTROL"]].index(name)
        control[codec.PAYLOAD+index] = value/codec.FIELDS["CONTROL"][index][1]
    _append_blockers(result, plan.last_report, stage_links, step)
    _append_attention(result, observation, plan)
    result.rows.append(control); result.known.append(known); result.ids.append("control")
    pins = identity()["source_pins"]
    packed = {"schema": PACK_SCHEMA, "representation_schema_sha256": REPRESENTATION_SCHEMA_SHA256,
        "source_pins": pins, "source_sha256": codec.digest(pins),
        "rows": np.stack(result.rows), "knownness": np.stack(result.known),
        "token_mask": np.ones(len(result.rows), np.bool_), "token_ids": tuple(result.ids)}
    packed["content_sha256"] = codec._content(packed)
    return validate(packed)
