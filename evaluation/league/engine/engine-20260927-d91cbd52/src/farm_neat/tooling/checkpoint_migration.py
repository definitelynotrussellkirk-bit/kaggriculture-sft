"""Rebind a farm-born checkpoint's recorded source paths to this repository's layout.

Checkpoints pin their source closure by REPOSITORY-RELATIVE PATH. The stack moved
from the farm workspace (``prototype/full_economic_policy.py``, ``heuristic/*.py``,
``scripts/train_full_economic_fast.py``, ...) into ``src/farm_neat/``, so every
imported checkpoint records paths that no longer exist and
``farm_neat.policy.network.check_model_sources`` refuses it. There is no load-time
rebind and none is added here: this is a one-off, explicit, owner-run rewrite of the
JSON sidecars of ONE checkpoint directory, recorded in the sidecars themselves.

    F=<the old workspace>/work/runs/full-economy-20260909      # nothing here reads it by default
    python3 -B src/farm_neat/tooling/runtime.py run -- \
        src/farm_neat/tooling/checkpoint_migration.py work/checkpoints/program-bc-v2-01 \
        --origin $F/program-bc-v2-01 --farm-root <the old workspace> \
        --reason "farmNEAT layout rebind of the age-1156 BC lineage"

    python3 -B src/farm_neat/tooling/runtime.py run -- \
        src/farm_neat/tooling/checkpoint_migration.py --verify <the old workspace>

What it does and does not do:

* ``model.pt`` and ``optimizer.pt`` are never opened for writing; their sha256 is
  recorded before and after and must be unchanged. Imported sidecars are HARDLINKS
  to the farm inode, so every rewrite goes to a new file that is ``os.replace``d
  over the name -- the farm workspace keeps the verbatim original bytes.
* Only ``model.pt.json``, ``PLAN.json`` and ``RESULT.json`` change: the pin maps a
  live check compares against a live closure, the full-plan adapter and frozen
  ShopRouter identity blocks, and the three hashes that join them
  (``RESULT.plan_sha256``, ``RESULT.manifest``, ``lineage.plan_sha256``).
* ``optimizer.pt`` embeds its OWN ``source_pins``, ``plan_sha256`` and ``execution``,
  which are not rewritable here. The record therefore carries those pre-migration values
  verbatim under ``pre_migration``, and ``recorded_transition`` below is the ONE reader
  that lets ``campaign.validate_parent`` and ``program_bc`` join the optimizer against
  them. That join is as strict as it ever was; it is simply made against the recorded
  originals instead of the rewritten sidecars, and only when the record holds.
* Historical blocks (``resume_binding``/``lineage_origin`` chains, ``VALIDATION.json``,
  ``RECEIPT.json``, ``ADMISSION.json``) keep their farm paths verbatim. Nothing reads
  them, and rewriting sealed history would destroy the record of where this came from.

``--farm-root`` additionally records every pin whose recorded sha is no longer what the
farm file at that path holds, i.e. every real edit the migration binds the checkpoint
across. For a pure path rebind that list is empty; when it is not, ``--accept-drift`` must
name exactly those old paths, so crossing a real source change is an owner decision and
never a side effect of moving the file.

``PATH_MAP`` is explicit data. Every entry was verified by diffing the farm file at
the old path against the file at the new path with import lines removed; ``--verify``
re-runs exactly that diff. The entries whose remainder is more than a rewritten path
literal are listed in ``KNOWN_DIFFERENCES`` and copied into the migration record.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT / "src"))

from farm_neat.economy import action_space as actions
from farm_neat.evaluation import shop_router as router
from farm_neat.policy import network as policy
from farm_neat.policy import plan_observation as plan_features
from farm_neat.policy import qwen_hybrid as hybrid
from farm_neat.training import campaign
from farm_neat.training import sustain as io

SCHEMA = "farmneat-checkpoint-source-migration-v1"
RECORD_KEY = "source_migration"
SIDECARS = ("model.pt.json", "PLAN.json", "RESULT.json")
UNTOUCHED_SIDECARS = ("ADMISSION.json", "RECEIPT.json", "STORAGE.json", "VALIDATION.json")
TENSORS = ("model.pt", "optimizer.pt")

# farm repository-relative path -> this repository's path. 47 entries: the 45 of the
# campaign closure (which contains the 18 of the model closure and the 7 of the
# full-plan adapter) plus the BC stage's own two.
PATH_MAP = {
    "agent/constants.py": "src/farm_neat/game/constants.py",
    "agent/macro.py": "src/farm_neat/game/macro.py",
    "default_plan/exact_engine.py": "src/farm_neat/game/exact_engine.py",
    "docs/upstream/kaggriculture_engine_reference.py": "src/farm_neat/game/kaggriculture_engine_reference.py",
    "economic_learning_v2/model.py": "src/farm_neat/training/schedule_model.py",
    "economic_learning_v2/training.py": "src/farm_neat/training/schedule.py",
    "external_agents/kaggle/yhay81/shop-router-0909/LICENSE.txt": "src/farm_neat/evaluation/shop_router_0909/LICENSE.txt",
    "external_agents/kaggle/yhay81/shop-router-0909/actions.json": "src/farm_neat/evaluation/shop_router_0909/actions.json",
    "external_agents/kaggle/yhay81/shop-router-0909/main.py": "src/farm_neat/evaluation/shop_router_0909/main.py",
    "heuristic/animal_dp.py": "src/farm_neat/economy/heuristic/animal_dp.py",
    "heuristic/constants.py": "src/farm_neat/economy/heuristic/constants.py",
    "heuristic/econ.py": "src/farm_neat/economy/heuristic/econ.py",
    "heuristic/market.py": "src/farm_neat/economy/heuristic/market.py",
    "heuristic/policy.py": "src/farm_neat/economy/heuristic/policy.py",
    "heuristic/search.py": "src/farm_neat/economy/heuristic/search.py",
    "heuristic/state.py": "src/farm_neat/economy/heuristic/state.py",
    "heuristic/valuation.py": "src/farm_neat/economy/heuristic/valuation.py",
    "plan_player_bridge/kaggriculture.json": "src/farm_neat/game/kaggriculture.json",
    "prototype/full_economic_action_space.py": "src/farm_neat/economy/action_space.py",
    "prototype/full_economic_batched_replay.py": "src/farm_neat/rollout/batched_replay.py",
    "prototype/full_economic_fast_profile.py": "src/farm_neat/policy/fast_profile.py",
    "prototype/full_economic_linter.py": "src/farm_neat/economy/linter.py",
    "prototype/full_economic_observation.py": "src/farm_neat/policy/observation.py",
    "prototype/full_economic_orders.py": "src/farm_neat/economy/orders.py",
    "prototype/full_economic_plan_learning.py": "src/farm_neat/policy/plan_learning.py",
    "prototype/full_economic_plan_observation.py": "src/farm_neat/policy/plan_observation.py",
    "prototype/full_economic_policy.py": "src/farm_neat/policy/network.py",
    "prototype/full_economic_rollout.py": "src/farm_neat/rollout/rollout.py",
    "prototype/semantic_commitment_context.py": "src/farm_neat/policy/semantic/commitment_context.py",
    "prototype/semantic_joint_policy_v3.py": "src/farm_neat/policy/semantic/joint_policy_v3.py",
    "prototype/semantic_orders_v1.py": "src/farm_neat/economy/orders_v1.py",
    "prototype/semantic_policy_v1.py": "src/farm_neat/policy/semantic/policy_v1.py",
    "prototype/semantic_population_v2.py": "src/farm_neat/policy/semantic/population_v2.py",
    "prototype/semantic_postedit_policy_v2.py": "src/farm_neat/policy/semantic/postedit_policy_v2.py",
    "prototype/semantic_qwen_hybrid.py": "src/farm_neat/policy/qwen_hybrid.py",
    "scripts/full_economic_program_bc.py": "src/farm_neat/training/program_bc.py",
    "scripts/onboard_shop_router_0909.py": "src/farm_neat/evaluation/shop_router.py",
    "scripts/sustain_full_economic.py": "src/farm_neat/training/sustain.py",
    "scripts/train_full_economic_fast.py": "src/farm_neat/training/campaign.py",
    "tests/test_full_economic_batched_replay.py": "tests/unit/test_batched_replay.py",
    "tests/test_full_economic_fast_profile.py": "tests/unit/test_fast_profile.py",
    "tests/test_full_economic_linter.py": "tests/unit/test_linter.py",
    "tests/test_full_economic_observation.py": "tests/unit/test_observation.py",
    "tests/test_full_economic_plan_learning.py": "tests/unit/test_plan_learning.py",
    "tests/test_full_economic_program_bc.py": "tests/contracts/test_program_bc.py",
    "tests/test_train_full_economic_fast.py": "tests/contracts/test_campaign.py",
    "train/opponents.py": "src/farm_neat/evaluation/opponents.py",
}

# Everything else differs only in import lines and in the pinned path literals the
# modules themselves declare. These six differ in more, measured 2026-09-12 against
# the old workspace with import/sys.path/ROOT lines removed (--verify re-runs it).
KNOWN_DIFFERENCES = {
    "default_plan/exact_engine.py":
        "engine lookup changed from repository-root candidate lists to the package directory; "
        "the loaded kaggriculture_engine_reference.py and kaggriculture.json are byte-identical",
    "prototype/semantic_postedit_policy_v2.py":
        "two lines added: NotImplementedError in the DEFAULT_PLAN_V03 benchmark branch and POLICY_ID = None; "
        "the un-migrated semantic benchmark harness",
    "scripts/onboard_shop_router_0909.py":
        "the notebook-extraction half (unpack_bytes/extract, ~150 lines) was not migrated; the frozen "
        "LICENSE.txt/main.py/actions.json bytes it produced are byte-identical here",
    "train/opponents.py":
        "three lines added: NotImplementedError in the neural-opponent registry and replay-tape branches",
    "scripts/sustain_full_economic.py":
        "one comment line added and the sys.path insert now points at <root>/src; the two subprocess "
        "launches name the migrated module paths",
    "scripts/full_economic_program_bc.py":
        "DEFAULT_PARENT and DEFAULT_CORPUS now name this workspace's checkpoint/dataset homes; "
        "command-line defaults only, no change to the stage",
}

# The hashed input contract embeds the encoder's path, so INPUT_SCHEMA_SHA256 -- and
# with it the architecture identity read_manifest compares -- moved with the layout even
# though no tensor, representation or factor changed. These are its path-valued fields.
INPUT_CONTRACT_PATH_FIELDS = ("encoder",)

_IMPORT_LINE = re.compile(r"^\s*(import |from |sys\.path|ROOT\s*=|if __package__)")


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bytes(value) -> bytes:
    """The exact serialization sustain.write and network.save_model already use."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _replace(path: Path, payload: bytes) -> None:
    """Break the farm hardlink: a NEW inode, renamed over the name. Never truncate."""
    staged = path.with_name(path.name + ".migrating")
    if staged.exists():
        raise ValueError("stale staged sidecar; resolve it by hand: " + str(staged))
    staged.write_bytes(payload)
    os.replace(staged, path)


def _rebind_architecture(recorded):
    """The architecture identity moved only because INPUT_CONTRACT names the encoder by
    path. Reconstruct the farm contract from the live one and the map; the recorded
    input_contract_sha256 must be exactly its digest, or this is a real model change."""
    config = hybrid.Config(**recorded["config"])
    live = json.loads(json.dumps(policy.architecture(config)))
    differing = sorted(k for k in set(recorded) | set(live) if recorded.get(k) != live.get(k))
    if not differing:
        return live, None
    if differing != ["content_sha256", "input_contract_sha256"]:
        raise ValueError("architecture changed in more than the pinned encoder path: " + repr(differing))
    inverse = {v: k for k, v in PATH_MAP.items()}
    contract = {**actions.INPUT_CONTRACT,
                **{field: inverse[actions.INPUT_CONTRACT[field]] for field in INPUT_CONTRACT_PATH_FIELDS}}
    if actions.digest(contract) != recorded["input_contract_sha256"]:
        raise ValueError("recorded input contract is not this contract with farm paths; refusing")
    return live, {"changed_fields": differing,
                  "old_input_contract_sha256": recorded["input_contract_sha256"],
                  "new_input_contract_sha256": live["input_contract_sha256"],
                  "old_content_sha256": recorded["content_sha256"],
                  "new_content_sha256": live["content_sha256"],
                  "verified": "INPUT_CONTRACT_WITH_THE_FARM_ENCODER_PATH_DIGESTS_TO_THE_RECORDED_SHA256",
                  "path_fields": {field: {"old": contract[field], "new": actions.INPUT_CONTRACT[field]}
                                  for field in INPUT_CONTRACT_PATH_FIELDS}}


def _rebind(recorded, live, label):
    """Map the recorded keys through PATH_MAP; they must then BE the live closure."""
    unmapped = sorted(k for k in recorded if k not in PATH_MAP)
    if unmapped:
        raise ValueError(label + ": no path map entry for " + repr(unmapped))
    mapped = {PATH_MAP[k]: v for k, v in recorded.items()}
    if len(mapped) != len(recorded):
        raise ValueError(label + ": path map is not injective on this closure")
    if set(mapped) != set(live):
        raise ValueError(label + ": mapped paths are not the live closure; missing "
                         + repr(sorted(set(live) - set(mapped))) + " extra "
                         + repr(sorted(set(mapped) - set(live))))
    return dict(live), {PATH_MAP[k]: {"old_path": k, "old_sha256": v, "new_sha256": live[PATH_MAP[k]]}
                        for k, v in recorded.items()}


def _rebind_block(recorded, live, label):
    """A whole identity block (full-plan adapter, frozen ShopRouter): only its pins
    and its own content digest may differ, and the difference is recorded."""
    differing = sorted(k for k in set(recorded) | set(live) if recorded.get(k) != live.get(k))
    pin_key = "source_pins" if "source_pins" in live else "assets"
    semantic = [k for k in differing if k not in (pin_key, "content_sha256")]
    _, table = _rebind(recorded[pin_key], live[pin_key], label + "." + pin_key)
    if pin_key == "assets":
        # The frozen external snapshot is never edited, so only its path may move.
        edited = sorted(k for k, e in table.items() if e["old_sha256"] != e["new_sha256"])
        if edited:
            raise ValueError(label + ": frozen external assets must be byte-identical: " + repr(edited))
    return copy.deepcopy(live), {"changed_fields": differing, "non_pin_changed_fields": semantic,
                                 "old_content_sha256": recorded.get("content_sha256"),
                                 "new_content_sha256": live.get("content_sha256"), "files": table}


def _merge(files, scopes, table, label):
    """Every scope's recorded sha is kept: a checkpoint whose PLAN and RESULT disagree
    about a file is a real inconsistency and must be visible in the record, not lost."""
    scopes[label] = {name: entry["old_sha256"] for name, entry in table.items()}
    for name, entry in table.items():
        files.setdefault(name, entry)


def _inconsistent(scopes):
    seen = {}
    for label, table in scopes.items():
        for name, sha in table.items():
            seen.setdefault(name, {}).setdefault(sha, []).append(label)
    return {name: {sha: labels for sha, labels in rows.items()}
            for name, rows in seen.items() if len(rows) > 1}


def _opponents(block, record, label, files, scopes):
    for ordinal, opponent in enumerate(block.get("opponents", []) if block else []):
        if opponent.get("kind") != "SHOP_ROUTER":
            continue
        name = f"{label}.opponents[{ordinal}].identity"
        rebound, entry = _rebind_block(opponent["identity"], router.identity(), name)
        opponent["identity"] = rebound
        record[name] = entry
        _merge(files, scopes, entry["files"], name + ".assets")


def origin_drift(pins, farm_root):
    """Pins whose recorded sha is NOT what the farm file at that path holds today.

    A pure path rebind leaves this empty: the code is the same code under a new name.
    Anything listed here is a real edit made in farm after this checkpoint was published,
    and the migration binds the checkpoint across it. It is recorded, never suppressed.
    """
    farm = Path(farm_root)
    drift = {}
    for entry in pins.values():
        candidate = farm / entry["old_path"]
        today = file_sha(candidate) if candidate.is_file() else None
        if today != entry["old_sha256"]:
            drift[entry["old_path"]] = {"recorded_sha256": entry["old_sha256"], "farm_sha256_today": today}
    return drift


def migrate(directory, *, origin, reason, root=ROOT, farm_root=None, accept_drift=()):
    """Rewrite one checkpoint directory's pin-carrying sidecars, in place, once.

    ``root`` only resolves the checkpoint directory; the live closure it rebinds to is
    always this repository's, because that is the code the checkpoint must now match.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("explicit --reason required")
    folder = io.path(root, directory)
    origin = Path(origin).resolve()
    for name in (*SIDECARS, *TENSORS):
        if not (folder / name).is_file():
            raise ValueError("complete published checkpoint required; missing " + name)
    if not (origin / "model.pt").is_file():
        raise ValueError("origin directory with the verbatim originals required: " + str(origin))
    here, there = (folder / "model.pt").stat(), (origin / "model.pt").stat()
    if (here.st_dev, here.st_ino) == (there.st_dev, there.st_ino):
        origin_binding = "SAME_INODE_HARDLINK"
    elif file_sha(folder / "model.pt") == file_sha(origin / "model.pt"):
        origin_binding = "IDENTICAL_MODEL_PT_SHA256"
    else:
        raise ValueError("origin does not hold this checkpoint's model.pt")

    manifest = io.read(folder / "model.pt.json")
    plan = io.read(folder / "PLAN.json")
    result = io.read(folder / "RESULT.json")
    for name, value in zip(SIDECARS, (manifest, plan, result)):
        if RECORD_KEY in value:
            raise ValueError("source migration already applied to " + name + "; reason was "
                             + repr(value[RECORD_KEY].get("reason")) + "; refusing to rewrite twice")
    before = {name: file_sha(folder / name) for name in (*SIDECARS, *UNTOUCHED_SIDECARS, *TENSORS)
              if (folder / name).is_file()}
    # optimizer.pt keeps these forever, so they are captured verbatim before any rewrite.
    pre_execution = copy.deepcopy(plan["execution"])
    pre_plan_pins = copy.deepcopy(plan["source_pins"])
    pre_result_pins = copy.deepcopy(result["source_pins"])

    scopes, files = {}, {}
    for label, value, key, live in (
            ("model.pt.json:source_pins", manifest, "source_pins", policy.sources()),
            ("PLAN.json:source_pins", plan, "source_pins", campaign.sources()),
            ("RESULT.json:source_pins", result, "source_pins", campaign.sources()),
            ("PLAN.json:program_bc_source_pins", plan, "program_bc_source_pins", None),
            ("RESULT.json:program_bc_source_pins", result, "program_bc_source_pins", None)):
        if key not in value:
            continue
        target = live if live is not None else {PATH_MAP[k]: file_sha(ROOT / PATH_MAP[k]) for k in value[key]}
        value[key], table = _rebind(value[key], target, label)
        _merge(files, scopes, table, label)

    blocks = {}
    for label, holder in (("PLAN.json:input_adapter", plan.get("input_adapter")),
                          ("PLAN.json:execution.resume_game_execution.full_plan.adapter",
                           plan["execution"].get("resume_game_execution", plan["execution"]).get("full_plan"))):
        if holder is None:
            continue
        recorded = holder if label.endswith("input_adapter") else holder["adapter"]
        rebound, entry = _rebind_block(recorded, plan_features.identity(), label)
        _merge(files, scopes, entry["files"], label + ".source_pins")
        if entry["non_pin_changed_fields"]:
            raise ValueError(label + ": the adapter changed in more than its pins: "
                             + repr(entry["non_pin_changed_fields"]))
        if label.endswith("input_adapter"):
            plan["input_adapter"] = rebound
        else:
            holder["adapter"] = rebound
        blocks[label] = entry
    _opponents(plan["execution"].get("plan_learning"), blocks,
               "PLAN.json:execution.plan_learning", files, scopes)
    _opponents(plan["execution"].get("resume_game_execution", {}).get("plan_learning"), blocks,
               "PLAN.json:execution.resume_game_execution.plan_learning", files, scopes)

    manifest["architecture"], architecture_record = _rebind_architecture(manifest["architecture"])

    drift = None if farm_root is None else origin_drift(files, farm_root)
    if drift and sorted(drift) != sorted(set(accept_drift or ())):
        raise ValueError("the farm file no longer holds the sha this checkpoint recorded; migrating across a "
                         "real edit needs --accept-drift naming exactly: " + repr(sorted(drift)))
    if accept_drift and drift is None:
        raise ValueError("--accept-drift needs --farm-root to establish what actually drifted")

    record = {"schema": SCHEMA, "reason": reason.strip(), "tool": "src/farm_neat/tooling/checkpoint_migration.py",
              "origin": {"directory": str(origin), "binding": origin_binding,
                         "keeps": "THE_VERBATIM_PRE_MIGRATION_SIDECARS"},
              "replaced_sidecar_sha256": {name: before[name] for name in SIDECARS},
              "unchanged_sha256": {name: sha for name, sha in before.items() if name not in SIDECARS},
              "tensors": "MODEL_PT_AND_OPTIMIZER_PT_BYTES_UNCHANGED",
              "path_map": {k: v for k, v in PATH_MAP.items() if v in files},
              "origin_workspace_drift": drift, "accepted_drift": sorted(accept_drift or ()),
              "pre_migration": {"plan_sha256": before["PLAN.json"], "result_sha256": before["RESULT.json"],
                  "manifest_sha256": before["model.pt.json"], "execution": pre_execution,
                  "execution_sha256": actions.digest(pre_execution), "source_pins": pre_plan_pins,
                  "source_pins_sha256": actions.digest(pre_plan_pins),
                  "result_source_pins_sha256": actions.digest(pre_result_pins),
                  "semantics": "WHAT_OPTIMIZER_PT_STILL_EMBEDS; THE_ONLY_VALUES_A_MIGRATED_PARENT_MAY_BE_JOINED_AGAINST"},
              "rebound_pin_maps": scopes, "rebound_identity_blocks": blocks,
              "inconsistent_recorded_pins": _inconsistent(scopes),
              "rebound_architecture_identity": architecture_record,
              "pins": files, "known_source_differences": {k: v for k, v in KNOWN_DIFFERENCES.items()
                                                          if PATH_MAP[k] in files},
              "not_rebound": "optimizer.pt embeds its own source_pins/plan_sha256/execution and is never "
                             "rewritten; resume_binding/lineage_origin history, ADMISSION.json, RECEIPT.json "
                             "and VALIDATION.json keep their farm paths verbatim",
              "read_time_checks": "UNCHANGED; check_model_sources HAS NO REBIND HOOK AND LEGACY_SOURCE_SHA256 IS UNTOUCHED"}

    plan[RECORD_KEY] = record
    plan_bytes = _bytes(plan)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    if manifest["lineage"].get("plan_sha256") == result["plan_sha256"]:
        manifest["lineage"]["plan_sha256"] = plan_sha
    result["plan_sha256"] = plan_sha
    manifest[RECORD_KEY] = record
    manifest.pop("content_sha256", None)
    manifest["content_sha256"] = actions.digest(manifest)
    result["manifest"] = manifest["content_sha256"]
    result[RECORD_KEY] = record

    # Everything read_manifest will check, before a single byte is written.
    if manifest["architecture"] != json.loads(json.dumps(policy.architecture(
            hybrid.Config(**manifest["architecture"]["config"])))):
        raise ValueError("migrated architecture is still not the live architecture")
    policy.check_model_sources(manifest)
    for name, payload in (("PLAN.json", plan_bytes), ("model.pt.json", _bytes(manifest)),
                          ("RESULT.json", _bytes(result))):
        _replace(folder / name, payload)

    after = {name: file_sha(folder / name) for name in TENSORS}
    if any(after[name] != before[name] for name in TENSORS):
        raise ValueError("checkpoint tensors changed; this tool must never touch a .pt file")
    policy.read_manifest(folder / "model.pt")
    reread = io.read(folder / "RESULT.json")
    if (reread["plan_sha256"] != file_sha(folder / "PLAN.json")
            or reread["manifest"] != io.read(folder / "model.pt.json")["content_sha256"]):
        raise ValueError("migrated plan/manifest join is inconsistent")
    return {"directory": io.relative(root, folder), "reason": record["reason"],
            "paths_rebound": len(files), "identity_blocks_rebound": sorted(blocks),
            "architecture_rebound": architecture_record is not None,
            "origin_workspace_drift": record["origin_workspace_drift"],
            "inconsistent_recorded_pins": record["inconsistent_recorded_pins"],
            "plan_sha256": plan_sha, "model_manifest_sha256": manifest["content_sha256"],
            "tensors_unchanged": after, "read_manifest": "ACCEPTS_WITH_NO_FLAGS"}


def recorded_transition(saved, plan, result, manifest, *, live_pins):
    """The ONE reader of a ``source_migration`` record, and the ONLY way a migrated parent
    is admitted. It is not a relaxation: it returns the **pre-migration** identity that
    ``optimizer.pt`` must still equal exactly, so the join is as strict as it ever was --
    it is just made against the recorded originals instead of the rewritten sidecars.

    ``None`` means "not a migrated parent, or the record does not hold", and every caller
    then compares against the sidecars exactly as before, which refuses. Four conditions,
    all required: the same record is in all three sidecars and carries a non-empty reason;
    its recorded pre-migration hashes ARE the sidecar hashes it replaced; the plan's and
    result's pins are the live closure; and ``optimizer.pt``'s embedded ``source_pins``,
    ``plan_sha256`` and ``execution`` are byte-for-byte the recorded pre-migration values.
    Nothing here reads the filesystem: an admission must not depend on another workspace.
    """
    record = plan.get(RECORD_KEY)
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return None
    if result.get(RECORD_KEY) != record or manifest.get(RECORD_KEY) != record:
        return None
    reason = record.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None
    pre, replaced = record.get("pre_migration"), record.get("replaced_sidecar_sha256")
    if not isinstance(pre, dict) or not isinstance(replaced, dict):
        return None
    if (pre.get("plan_sha256") != replaced.get("PLAN.json")
            or pre.get("result_sha256") != replaced.get("RESULT.json")
            or pre.get("manifest_sha256") != replaced.get("model.pt.json")
            or pre.get("source_pins_sha256") != actions.digest(pre.get("source_pins"))
            or pre.get("result_source_pins_sha256") != pre.get("source_pins_sha256")
            or pre.get("execution_sha256") != actions.digest(pre.get("execution"))):
        return None
    if plan.get("source_pins") != live_pins or result.get("source_pins") != live_pins:
        return None
    if (saved.get("source_pins") != pre["source_pins"] or saved.get("plan_sha256") != pre["plan_sha256"]
            or saved.get("execution") != pre["execution"]):
        return None
    return {"kind": "EXPLICIT_RECORDED_CHECKPOINT_SOURCE_MIGRATION", "reason": reason.strip(),
            "origin": record["origin"]["directory"], "path_map_entries": len(record["path_map"]),
            "accepted_drift": record.get("accepted_drift") or [],
            "source_pins": pre["source_pins"], "plan_sha256": pre["plan_sha256"],
            "execution": pre["execution"], "new_source_pins": dict(live_pins),
            "weights_optimizer_rng_age": "UNCHANGED; ONLY_RECORDED_SOURCE_IDENTITY_MOVED"}


def joined_identity(saved, plan, result, manifest, *, live_pins):
    """What ``optimizer.pt`` must agree with: the recorded originals for a migrated
    parent, the sidecars themselves for every other parent."""
    transition = recorded_transition(saved, plan, result, manifest, live_pins=live_pins)
    if transition is None:
        return {"source_pins": plan.get("source_pins"), "plan_sha256": result.get("plan_sha256"),
                "execution": plan.get("execution")}, None
    return transition, transition


def verify_paths(farm_root):
    """Re-run the diff that justifies PATH_MAP: same file, import lines removed."""
    farm = Path(farm_root)
    rows = []
    for old, new in sorted(PATH_MAP.items()):
        source, target = farm / old, ROOT / new
        if not source.is_file() or not target.is_file():
            rows.append({"old": old, "new": new, "status": "MISSING"})
            continue
        left = [l for l in source.read_text().splitlines() if not _IMPORT_LINE.match(l)]
        right = [l for l in target.read_text().splitlines() if not _IMPORT_LINE.match(l)]
        diff = [l for l in difflib.unified_diff(left, right, lineterm="", n=0)][2:]
        rows.append({"old": old, "new": new, "identical_bytes": source.read_bytes() == target.read_bytes(),
                     "non_import_diff_lines": len(diff), "known_difference": KNOWN_DIFFERENCES.get(old)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", help="checkpoint directory inside this repository")
    parser.add_argument("--origin", help="the farm directory that keeps the verbatim originals")
    parser.add_argument("--farm-root", help="the farm workspace, to record pins whose source has since changed there")
    parser.add_argument("--accept-drift", nargs="*", default=[], metavar="OLD_PATH",
                        help="old paths whose source really changed in farm; required when --farm-root finds any")
    parser.add_argument("--reason", default="", help="why this lineage is being continued here")
    parser.add_argument("--verify", metavar="FARM_ROOT", help="only diff the path map against a farm workspace")
    args = parser.parse_args(argv)
    if args.verify:
        rows = verify_paths(args.verify)
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        return 1 if any(row.get("status") == "MISSING" for row in rows) else 0
    if not args.directory or not args.origin:
        parser.error("a checkpoint directory, --origin and --reason are required")
    print(json.dumps(migrate(args.directory, origin=args.origin, reason=args.reason,
                             farm_root=args.farm_root, accept_drift=args.accept_drift),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
