"""Bounded full-economics MoE/teacher lab using the existing exact engine.

Teacher imitation is PARTIAL ACTION BC, never fabricated latent plan labels or
terminal RL admission. Production PPO continuation is a separate next step.
All outputs are new, source-bound work/runs children; no frozen actor is edited.
Public replay imports instead use the declared separate corpus mount. They reuse
observed-prefix BC, not fabricated full-plan targets or old BC resume identities.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
SCHEMA = "farm-full-economic-moe-teacher-lab-v1"
BC_PROTOCOL = "farm-full-economic-observed-bc-continuation-v2"
BC_OPTIMIZER_SCHEMA = BC_PROTOCOL + "-optimizer"
LEGACY_LAB_SHA = "0a5c789cd584a9aa48eeff371501f2dc55391413742dcbb19a0c5d8a5d2c222d"
LEGACY_BC_ANCHOR = {
    "PLAN.json": "77bf414e3231c479bd77932f60baef9a947e1041e0ee0dd26842c9e2ba5ee496",
    "RESULT.json": "63e0ad0a047631e18852fc89e41c6af49472db0530328983c762d09ea7db04d3",
    "EXAMPLES.jsonl.gz": "60cc7033dc22458b23d23a97c4b29c95a3e963161424c6981500ee11f07f091a"}
LEGACY_TEACHER_ANCHOR = {
    "PLAN.json": "5d606efa84f3f37c1a39b45f71068a732a844329b331d9d0b811070aa4d3e923",
    "RESULT.json": "e6dd8880521244527b8a94bcf2b9f28a04796b1bf6986463b11d2f3d0de60275"}
BC_EXCLUDED_EXECUTION_SOURCES = ("src/farm_neat/economy/orders.py", "src/farm_neat/economy/linter.py")
OPTIMIZER_CONFIG = {"kind": "bitsandbytes.optim.AdamW8bit", "betas": [0.9, 0.999],
                    "eps": 1e-8, "weight_decay": 0.0, "min_8bit_size": 4096, "block_wise": True}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def output_directory(path):
    path = Path(path).absolute()
    if path != path.resolve() or not path.is_relative_to(ROOT / "work/runs") or path.exists():
        raise ValueError("new output directory below work/runs required")
    path.mkdir(parents=True)
    return path


def sources():
    names = ["src/farm_neat/training/moe_lab.py", "src/farm_neat/game/exact_engine.py",
        "src/farm_neat/game/kaggriculture_engine_reference.py", "src/farm_neat/economy/orders.py",
        "src/farm_neat/economy/action_space.py", "src/farm_neat/policy/observation.py",
        "src/farm_neat/policy/network.py", "src/farm_neat/economy/linter.py",
        "src/farm_neat/training/bc.py", "src/farm_neat/policy/qwen_hybrid.py",
        "src/farm_neat/economy/heuristic/policy.py", "src/farm_neat/economy/heuristic/state.py", "src/farm_neat/economy/heuristic/constants.py", "src/farm_neat/game/constants.py",
        "src/farm_neat/evaluation/shop_router.py", "src/farm_neat/evaluation/opponents.py",
        "src/farm_neat/training/schedule.py"]
    return {name: sha(ROOT / name) for name in names if (ROOT / name).exists()}


def teacher_sources(name="shoprouter"):
    names = ("src/farm_neat/training/moe_lab.py", "src/farm_neat/game/exact_engine.py",
        "src/farm_neat/game/kaggriculture_engine_reference.py", "src/farm_neat/evaluation/shop_router.py",
        "src/farm_neat/evaluation/opponents.py", "src/farm_neat/game/constants.py")
    if name != "shoprouter":
        names += ("prototype/full_economic_teacher.py",)
    return {path: sha(ROOT / path) for path in names}


def teacher_identity(name="shoprouter"):
    from farm_neat.evaluation.shop_router import identity
    if name == "shoprouter":
        return identity()
    raise NotImplementedError('the notebook-teacher path was not migrated')
    value = notebook_members(name)[1]
    if value["license"].startswith("UNVERIFIED"):
        raise ValueError("teacher reuse license not established; inspection still available")
    return {**value, "content_sha256": digest(value)}


def bc_sources():
    """Actual BC dependencies; physical executor/linter are not invoked by BC.

    Their changes may affect subsequent games, never the already extracted
    supervised tensors. This narrow role split does not weaken model, feature,
    teacher or optimizer source identity.
    """
    from farm_neat.policy import network as policy
    from farm_neat.training import bc as bc
    from farm_neat.evaluation.shop_router import identity
    return {**policy.sources(), **bc.sources(), **teacher_sources(),
            "src/farm_neat/training/schedule.py": sha(ROOT / "src/farm_neat/training/schedule.py"),
            **identity()["assets"]}


def teacher_game(seed, stream=None, *, teacher_name="shoprouter"):
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    from farm_neat.evaluation.shop_router import ShopRouterAdapter
    env = ExactKaggricultureEnvironment(seed=seed)
    teacher_identity(teacher_name)
    if teacher_name == "shoprouter":
        bots = [ShopRouterAdapter(), ShopRouterAdapter()]
    else:
        raise NotImplementedError('the notebook-teacher path was not migrated')
        bots = [NotebookTeacher(teacher_name), NotebookTeacher(teacher_name)]
    trajectory, primitive_counts = hashlib.sha256(), Counter()
    while not env.done:
        observations = [env.agent_observation(seat) for seat in (0, 1)]
        actions = [bot.act(obs, env.configuration) for bot, obs in zip(bots, observations)]
        record = {"seed": seed, "step": env.step_index, "observations": observations, "actions": actions,
            "inventory_key_order": [[list(inv) for inv in obs["private"]["inventories"]] for obs in observations]}
        payload = canonical(record).encode()
        trajectory.update(payload)
        if stream is not None:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False).encode() + b"\n")
        for action in actions:
            primitive_counts.update(op[0] if op else "EMPTY_NOOP" for op in
                                    [action["farmer"], *action["hands"], *action["market"]])
        env.step(actions)
    terminal = [env.agent_observation(seat) for seat in (0, 1)]
    trajectory.update(canonical(terminal).encode())
    return {"seed": seed, "physical_steps": env.step_index,
        "cash": [terminal[0]["farms"][s]["money"] for s in (0, 1)],
        "trajectory_sha256": trajectory.hexdigest(), "primitive_counts": dict(primitive_counts)}


def _read_public_json(path, expected=None):
    """Bounded data parsing, no pickle/code execution or ambiguous JSON keys."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64_000_000:
        raise ValueError("ordinary public JSON file below64MB required")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError("public raw source hash mismatch")
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate public JSON key")
            out[key] = value
        return out
    def nonfinite(value):
        raise ValueError("nonfinite public JSON")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite), actual


def public_sources():
    from farm_neat.training import bc as bc
    return {**bc.sources(), "src/farm_neat/training/moe_lab.py": sha(__file__),
            "src/farm_neat/game/exact_engine.py": sha(ROOT / "src/farm_neat/game/exact_engine.py")}


def public_corpus_files(manifest):
    """One declared raw copy, source license and episode IDs; no input duplication."""
    from farm_neat.training.storage_layout import external_policy
    root = Path(external_policy(ROOT)["root"])
    manifest = Path(manifest).absolute()
    if manifest != manifest.resolve() or not manifest.is_relative_to(root):
        raise ValueError("public corpus manifest must be on the declared mount")
    value, manifest_sha = _read_public_json(manifest)
    if value.get("schema") != "farm-external-replay-corpus-v1" or value.get("license") != "CC0-1.0":
        raise ValueError("explicit supported public corpus provenance/license required")
    def member(name):
        p = manifest.parent / name
        if (not isinstance(name, str) or p != p.resolve() or not p.is_relative_to(manifest.parent)
                or ".." in Path(name).parts or Path(name).is_absolute()):
            raise ValueError("corpus members must be contained ordinary relative paths")
        return p
    metadata_path = member(value["metadata"])
    metadata, _ = _read_public_json(metadata_path, value["metadata_sha256"])
    if (metadata.get("info", {}).get("licenses") != [{"name": "CC0-1.0"}]
            or metadata["info"]["datasetId"] != value.get("dataset_id")
            or value.get("source") != "https://www.kaggle.com/datasets/kaggle/" + metadata["info"]["datasetSlug"]):
        raise ValueError("dataset metadata/source/license mismatch")
    if not isinstance(value.get("files"), list) or not 1 <= len(value["files"]) <= 10000:
        raise ValueError("bounded explicit replay file list required")
    files, seen_ids, seen_hashes = [], set(), set()
    for row in value["files"]:
        p = member(row["path"])
        if (type(row.get("episode_id")) is not int or row["episode_id"] in seen_ids
                or row["sha256"] in seen_hashes):
            raise ValueError("duplicate/invalid public episode identity")
        if p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
            raise ValueError("public corpus file byte identity mismatch")
        seen_ids.add(row["episode_id"]); seen_hashes.add(row["sha256"])
        files.append({"path": str(p), "sha256": row["sha256"], "episode_id": row["episode_id"]})
    return {"manifest": str(manifest), "manifest_sha256": manifest_sha,
            "source_uri": value["source"], "license": value["license"], "files": files}


def _public_reports(binding, stats):
    """Export one actor report per economic turn, not one farm copy per label."""
    from farm_neat.training import bc as bc
    stats.update(games=[], labeled_steps=0, examples=0, known_noop_examples_excluded=0,
                 unsupported_primitives=0, prefix_families={}, draws_skipped=0)
    families = Counter()
    for source in binding["files"]:
        replay, _ = _read_public_json(source["path"], source["sha256"])
        witness = bc.verify_public_replay(replay)
        if witness["episode_id"] != source["episode_id"]:
            raise ValueError("raw episode ID disagrees with corpus")
        stats["games"].append(witness)
        if witness["winner_seat"] is None:
            stats["draws_skipped"] += 1
            continue
        provenance = {"source_name": "PUBLIC_REPLAY_WINNER",
            "source_uri": binding["source_uri"], "source_sha256": source["sha256"],
            "license": binding["license"], "episode_id": str(witness["episode_id"]),
            "player": witness["winner_seat"]}
        for report in bc.public_winner_reports(replay, provenance=provenance):
            indices = [i for i, e in enumerate(report["examples"]) if e["diagnostic"]["known_noop"] is not True]
            stats["unsupported_primitives"] += len(report["omitted"])
            stats["known_noop_examples_excluded"] += len(report["examples"]) - len(indices)
            if not indices:
                continue
            for i in indices:
                families[report["examples"][i]["prefix"]["factors"][0]["choices"][
                    report["examples"][i]["prefix"]["factors"][0]["index"]]] += 1
            stats["labeled_steps"] += 1
            stats["examples"] += len(indices)
            yield {"step": report, "example_indices": indices}
    stats["prefix_families"] = dict(families)


def public_replay_export(output, corpus):
    """Derive hash-bound winner SFT prefixes; a separate check rebuilds them."""
    from farm_neat.training.storage_layout import external_policy
    from farm_neat.training.bc import PUBLIC_SCHEMA
    binding = public_corpus_files(corpus)
    output = Path(output).absolute()
    root = Path(external_policy(ROOT)["root"])
    if output != output.resolve() or output == root or not output.is_relative_to(root) or output.exists():
        raise ValueError("new derived-data directory on the declared corpus mount required")
    output.mkdir(parents=True)
    pins = public_sources()
    plan = {"schema": PUBLIC_SCHEMA, "mode": "PUBLIC_WINNER_OBSERVED_PREFIX_SFT",
            "source_pins": pins, "corpus": binding, "selection": "ALL_SUPPORTED_WINNER_PREFIXES_EXCEPT_PROVEN_NOOPS",
            "input": "DECISION_TIME_ACTOR_OBSERVATION_ONLY; PRIVATE_FIELDS_FROM_THIS_SEAT_ONLY",
            "targets": "ATTEMPTED_ECONOMIC_PREFIXES_NOT_FULL_PLANS_OR_PROVEN_FILLS"}
    write(output / "PLAN.json", plan)
    stats, started = {}, time.monotonic()
    with gzip.open(output / "REPORTS.jsonl.gz", "xt", compresslevel=6) as stream:
        for row in _public_reports(binding, stats):
            stream.write(canonical(row) + "\n")
    if pins != public_sources() or binding != public_corpus_files(corpus):
        raise ValueError("source/corpus changed during public import")
    result = {"schema": PUBLIC_SCHEMA, "status": "PUBLIC_WINNER_PREFIXES_EXPORTED_NOT_FITTED",
              "plan_sha256": sha(output / "PLAN.json"), "reports_sha256": sha(output / "REPORTS.jsonl.gz"),
              "source_pins": pins, **stats, "seconds": time.monotonic()-started,
              "optimizer_updates": 0, "fresh_selfplay_games": 0, "full_plan_targets": 0,
              "independent_check": "Run public-check in a fresh process before supervised use"}
    write(output / "RESULT.json", result)
    return result


def public_replay_check(folder):
    """Re-extract every example from raw and rebuild its actual feature tensors."""
    from farm_neat.training import bc as bc
    folder = Path(folder)
    plan, result = json.loads((folder/"PLAN.json").read_text()), json.loads((folder/"RESULT.json").read_text())
    if (plan.get("schema") != bc.PUBLIC_SCHEMA or result.get("schema") != bc.PUBLIC_SCHEMA
            or result.get("status") != "PUBLIC_WINNER_PREFIXES_EXPORTED_NOT_FITTED"
            or result.get("plan_sha256") != sha(folder/"PLAN.json")
            or result.get("reports_sha256") != sha(folder/"REPORTS.jsonl.gz")
            or plan.get("source_pins") != public_sources() or result.get("source_pins") != public_sources()
            or public_corpus_files(plan["corpus"]["manifest"]) != plan["corpus"]):
        raise ValueError("public export source/raw/plan/result identity mismatch")
    stats, checked = {}, 0
    with gzip.open(folder/"REPORTS.jsonl.gz", "rt") as stream:
        for expected in _public_reports(plan["corpus"], stats):
            line = stream.readline()
            if not line or json.loads(line) != expected:
                raise ValueError("public selected winner/example content mismatch")
            packed = bc.pack_step(expected["step"])
            if packed["content_sha256"] != expected["step"]["input_sha256"]:
                raise ValueError("public input tensor mismatch")
            checked += 1
        if stream.read():
            raise ValueError("extra public examples")
    if any(result.get(k) != v for k, v in stats.items()):
        raise ValueError("public export summary mismatch")
    return {"schema": bc.PUBLIC_SCHEMA, "status": "PUBLIC_PREFIX_INPUTS_REEXTRACTED_AND_REPACKED",
            "result_sha256": sha(folder/"RESULT.json"), "checked_steps": checked,
            "examples": stats["examples"], "recorded_games_reconstructed": len(stats["games"]),
            "optimizer_updates": 0, "full_plan_targets": 0, "terminal_rl_admission": False}


def public_examples(folder, limit):
    """Verified examples in the existing BC minibatch shape; not a new fitter.

    Fresh reconstruction is intentional at this import boundary. The caller must
    version the SFT/optimizer transition; this does not masquerade as an old BC
    teacher directory or reset the current GRPO learner to its BC ancestor.
    """
    check = public_replay_check(folder)
    groups = defaultdict(list)
    with gzip.open(Path(folder)/"REPORTS.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            for i in row["example_indices"]:
                primitive = row["step"]["examples"][i]["attempted_primitive"]
                key = (primitive[0], primitive[1] if len(primitive) > 1 else "", row["step"]["observation"]["step"]//24)
                groups[key].append({"step": row["step"], "example_index": i})
    return _select_bc_buckets(groups, limit), check


def teacher(output, seeds, *, teacher_name="shoprouter", external=False):
    identity = teacher_identity(teacher_name)
    if not seeds or len(seeds) > 64 or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("one to64 distinct explicit local teacher seeds required")
    if external:
        from farm_neat.training.storage_layout import external_policy
        root, output = Path(external_policy(ROOT)["root"]), Path(output).absolute()
        if output != output.resolve() or not output.is_relative_to(root) or output == root or output.exists():
            raise ValueError("new teacher data directory on declared corpus mount required")
        output.mkdir(parents=True)
    else:
        output = output_directory(output)
    source = teacher_sources(teacher_name)
    plan = {"schema": SCHEMA, "mode": "TEACHER_COLLECTION", "source_pins": source,
        "teacher": identity, "teacher_name": teacher_name, "seeds": seeds, "split": "LOCAL_TRAIN_TEACHER_NOT_HELDOUT",
        "opponents": "FRESH_SOURCE_TEACHER_IN_BOTH_SEATS_RESPONSIVE",
        "labels": "OBSERVED_PARTIAL_ACTION_ONLY_NO_LATENT_PLAN_OR_VALUE_TARGETS"}
    write(output / "PLAN.json", plan)
    started = time.monotonic()
    games = []
    with gzip.open(output / "turns.jsonl.gz", "wb", compresslevel=6) as stream:
        for seed in seeds:
            game = teacher_game(seed, stream, teacher_name=teacher_name)
            games.append(game)
            print(canonical({"teacher_game": game}), flush=True)
    witnesses = []
    for seed, expected in zip(seeds, games, strict=True):
        # Fresh interpreter process, fresh teacher instances; compare complete
        # observations AND actions, not a recorded-action/tape-only replay.
        completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
            "teacher-witness", "--seed", str(seed), "--teacher-name", teacher_name], cwd=ROOT, capture_output=True, text=True, timeout=120, check=True)
        actual = json.loads(completed.stdout)
        if actual != expected:
            raise ValueError("fresh responsive teacher reconstruction mismatch")
        witnesses.append(actual)
    if source != teacher_sources(teacher_name):
        raise ValueError("source changed during teacher collection")
    result = {"schema": SCHEMA, "status": "TEACHER_GAMES_RECONSTRUCTED",
        "physical_teacher_games": len(games), "fresh_reference_games": len(witnesses),
        "independent_worlds": len(set(seeds)), "game_perspectives": 2 * len(games),
        "games": games, "witnesses": witnesses,
        "turns_sha256": sha(output / "turns.jsonl.gz"), "source_pins": source,
        "elapsed_seconds": time.monotonic() - started, "optimizer_updates": 0,
        "strength_evaluation": False, "terminal_training_labels": 0}
    result.update(teacher_plan_sha256=sha(output / "PLAN.json"),
                  teacher_identity_sha256=plan["teacher"]["content_sha256"])
    write(output / "RESULT.json", result)
    return result


def device_profile():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise ValueError("full-size pilot requires the actual CUDA device; no tiny fallback")
    return {"python": sys.version, "numpy": str(np.__version__), "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(), "dtype": "FLOAT32", "tf32": False,
        "autocast": False}


def tensor_inputs(graph, packs):
    import torch
    from farm_neat.policy.observation import collate
    batch = collate(packs)
    device = next(graph.parameters()).device
    return tuple(torch.as_tensor(batch[k], device=device) for k in ("rows", "token_mask", "knownness"))


def teacher_data_identity(folder):
    """Owner-produced teacher chain, not authentication of arbitrary imports."""
    folder = Path(folder)
    result = json.loads((folder / "RESULT.json").read_text())
    plan = json.loads((folder / "PLAN.json").read_text())
    hashes = {name: sha(folder/name) for name in ("PLAN.json", "RESULT.json", "turns.jsonl.gz")}
    teacher = plan["teacher"]
    if (result["status"] != "TEACHER_GAMES_RECONSTRUCTED"
            or result["turns_sha256"] != hashes["turns.jsonl.gz"]
            or result.get("source_pins") != plan.get("source_pins")
            or result.get("games") != result.get("witnesses")
            or result.get("physical_teacher_games") != len(result.get("games", []))
            or result.get("fresh_reference_games") != len(result.get("witnesses", []))
            or not result.get("games") or any(g.get("physical_steps") != 719 for g in result["games"])
            or [g["seed"] for g in result["games"]] != plan.get("seeds")
            or teacher != teacher_identity(plan.get("teacher_name", "shoprouter"))):
        raise ValueError("complete owner teacher/raw/source/fresh-witness identity required")
    old = result.get("teacher_plan_sha256") is None
    if old:
        if any(hashes[name] != expected for name, expected in LEGACY_TEACHER_ANCHOR.items()):
            raise ValueError("only pinned legacy teacher may acquire explicit plan linkage")
    elif (result["teacher_plan_sha256"] != hashes["PLAN.json"]
          or result.get("teacher_identity_sha256") != teacher["content_sha256"]):
        raise ValueError("teacher PLAN/result identity link mismatch")
    return {"schema": BC_PROTOCOL + "-teacher", "files": hashes,
            "teacher_identity_sha256": teacher["content_sha256"],
            "legacy_plan_linkage_added_at_this_boundary": old,
            "source_pins": result["source_pins"]}, result, plan


def bc_examples(folder, limit):
    from farm_neat.training.bc import inspect_step
    folder = Path(folder)
    _, result, plan = teacher_data_identity(folder)
    groups = defaultdict(list)
    with gzip.open(folder / "turns.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            for seat in (0, 1):
                name = plan.get("teacher_name", "shoprouter")
                provenance = {"source_name": "SHOP_ROUTER_LOCAL_RESPONSIVE_TEACHER" if name == "shoprouter" else name.upper() + "_LOCAL_RESPONSIVE_TEACHER",
                    "source_uri": plan["teacher"].get("source_uri", "https://www.kaggle.com/code/yhay81/shop-router-0909"),
                    "episode_id": str(row["seed"]), "player": seat,
                    "source_sha256": plan["teacher"]["content_sha256"],
                    "license": "Apache-2.0"}
                report = inspect_step(row["observations"][seat], row["actions"][seat], provenance=provenance)
                for index, example in enumerate(report["examples"]):
                    primitive = example["attempted_primitive"]
                    # Breadth sample is selected by observed operation/item/day,
                    # never future cash or an ex-post winning action label.
                    key = (primitive[0], primitive[1] if len(primitive) > 1 else "", row["step"] // 24)
                    groups[key].append({"step": report, "example_index": index})
    return _select_bc_buckets(groups, limit), result


def _select_bc_buckets(groups, limit):
    """Shared unchanged operation/item/day breadth selection for both teachers."""
    if type(limit) is not int or not 1 <= limit <= 4096:
        raise ValueError("bounded positive BC example limit required")
    rng = random.Random(90909001)
    for examples in groups.values():
        rng.shuffle(examples)
    selected = []
    while groups and len(selected) < limit:
        for key in list(sorted(groups)):
            selected.append(groups[key].pop())
            if not groups[key]:
                del groups[key]
            if len(selected) >= limit:
                break
    if not selected:
        raise ValueError("no supported observed BC prefixes")
    return selected


def selection_identity(examples, examples_limit, batch_size):
    """Pin the existing order; do not silently shuffle an in-progress BC stream."""
    if (type(examples_limit) is not int or not 1 <= examples_limit <= 4096
            or type(batch_size) is not int or not 1 <= batch_size <= 64 or not examples):
        raise ValueError("bounded positive BC example limit and batch size required")
    return {"schema": BC_PROTOCOL + "-selection", "examples_limit": examples_limit,
            "batch_size": batch_size, "count": len(examples), "ordered_examples_sha256": digest(examples),
            "algorithm": "PRESERVED_SEED90909001_WITHIN_BUCKET_SHUFFLE_SORTED_BUCKET_ROUND_ROBIN"}


def verify_retained_examples(folder, examples):
    """Freshly repack each retained actor view, not just compare file hashes."""
    from farm_neat.training.bc import pack_step
    with gzip.open(Path(folder)/"EXAMPLES.jsonl.gz", "rt") as stream:
        retained = [json.loads(line) for line in stream]
    if canonical(retained) != canonical(examples):
        raise ValueError("retained ordered examples differ from current extraction")
    inputs = []
    for entry in retained:
        packed = pack_step(entry["step"])
        if packed["content_sha256"] != entry["step"]["input_sha256"]:
            raise ValueError("legacy/current BC input tensor identity mismatch")
        inputs.append(packed["content_sha256"])
    return {"examples": len(inputs), "all_input_hashes_equal": True,
            "ordered_input_sha256": digest(inputs), "ordered_examples_sha256": digest(retained)}


def validate_bc_resume(folder, saved, parent, parent_plan, manifest, *, source,
                       teacher, selection, examples, profile, parameter_names):
    """Strict v2 continuation plus ONE explicit source-bound8-update legacy join."""
    from farm_neat.training.schedule import learning_rate_at, validate_settings
    folder = Path(folder)
    legacy = saved.get("schema") == SCHEMA + "-optimizer"
    if legacy:
        if any(sha(folder/name) != expected for name, expected in LEGACY_BC_ANCHOR.items()):
            raise ValueError("unregistered legacy BC anchor")
        if (saved["source_pins"] != parent["source_pins"] or saved["source_pins"] != parent_plan["source_pins"]
                or saved["source_pins"].get("src/farm_neat/training/moe_lab.py") != LEGACY_LAB_SHA):
            raise ValueError("legacy BC source preimage mismatch")
        if parent_plan["examples"] != selection["count"] or selection["examples_limit"] != parent_plan["examples"] or parent_plan["batch_size"] != selection["batch_size"]:
            raise ValueError("legacy selection limit/order/batch size cannot change implicitly")
        old_source = dict(saved["source_pins"])
        for mapping in (manifest["source_pins"], examples[0]["step"]["source_pins"]):
            for name, value in mapping.items():
                if name in old_source and old_source[name] != value:
                    raise ValueError("conflicting legacy model/data source pin")
                old_source[name] = value
        # Teacher assets were already bound by its original PLAN identity.
        from farm_neat.evaluation.shop_router import identity
        old_source.update(identity()["assets"])
        for name, value in source.items():
            if name != "src/farm_neat/training/moe_lab.py" and old_source.get(name) != value:
                raise ValueError("BC-active source changed across legacy transition: " + name)
        excluded = {name: {"old": value, "current": sha(ROOT/name)}
                    for name, value in old_source.items() if name in BC_EXCLUDED_EXECUTION_SOURCES}
        start_index = 0  # Only the explicitly pinned genesis→BC01 is admitted.
    else:
        if saved.get("schema") != BC_OPTIMIZER_SCHEMA or saved.get("source_pins") != source:
            raise ValueError("exact v2 BC source/schema required")
        if (parent.get("source_pins") != source or parent_plan.get("source_pins") != source
                or saved.get("plan_sha256") != sha(folder/"PLAN.json")
                or parent.get("plan_sha256") != saved.get("plan_sha256")
                or parent_plan.get("bc_protocol") != BC_PROTOCOL
                or parent.get("bc_protocol") != BC_PROTOCOL):
            raise ValueError("v2 BC plan/result/source join mismatch")
        if saved.get("selection") != selection or saved.get("teacher") != teacher:
            raise ValueError("BC selection/batch/teacher plan changed without an amendment")
        if (parent_plan.get("selection") != selection or parent_plan.get("teacher") != teacher
                or parent.get("selection") != selection or parent.get("teacher") != teacher):
            raise ValueError("v2 BC plan/result selection and teacher join mismatch")
        lineage = manifest.get("lineage", {})
        if (lineage.get("bc_protocol") != BC_PROTOCOL
                or lineage.get("plan_sha256") != saved["plan_sha256"]
                or lineage.get("selection_sha256") != digest(selection)
                or parent.get("next_example_index") != saved["next_example_index"]
                or parent.get("start_example_index") != parent_plan["start_example_index"]):
            raise ValueError("v2 model lineage/selected stream cursor mismatch")
        if saved.get("examples_file_sha256") != sha(folder/"EXAMPLES.jsonl.gz"):
            raise ValueError("parent selected-example file identity mismatch")
        if saved.get("parameter_names") != parameter_names or saved.get("optimizer_config") != OPTIMIZER_CONFIG:
            raise ValueError("BC optimizer parameter order/configuration mismatch")
        excluded, start_index = {}, parent_plan["start_example_index"]
    validate_settings(saved["settings"])
    age, index = saved["updates"], saved["next_example_index"]
    if (type(age) is not int or age < 1 or type(index) is not int or index < 0
            or age != parent["global_optimizer_updates"] or age != manifest["optimizer_steps"]
            or age != parent_plan["start_update"] + parent_plan["requested_updates"]
            or index != start_index + parent_plan["requested_updates"] * parent_plan["batch_size"]
            or saved["settings"] != parent_plan["schedule"]
            or saved["model_sha256"] != parent["after_model_sha256"]
            or parent["manifest"] != manifest["content_sha256"]
            or saved["environment"] != profile or parent_plan["environment"] != profile
            or saved["teacher_result_sha256"] != teacher["files"]["RESULT.json"]):
        raise ValueError("BC model/optimizer/global-age/cursor/schedule/environment join mismatch")
    if not saved["optimizer"].get("param_groups"):
        raise ValueError("saved Adam parameter groups required")
    expected_lr = learning_rate_at(age, saved["settings"])
    for group in saved["optimizer"]["param_groups"]:
        if (group["lr"] != expected_lr or list(group["betas"]) != OPTIMIZER_CONFIG["betas"]
                or group["eps"] != OPTIMIZER_CONFIG["eps"] or group["weight_decay"] != 0.0):
            raise ValueError("saved8bit Adam schedule/hyperparameter mismatch")
    tensor_proof = verify_retained_examples(folder, examples)
    return {"schema": BC_PROTOCOL + "-resume", "legacy_transition": legacy,
            "old_source_pins": saved["source_pins"], "new_bc_source_pins": source,
            "excluded_physical_execution_sources": excluded,
            "parent_plan_sha256": sha(folder/"PLAN.json"), "parent_result_sha256": sha(folder/"RESULT.json"),
            "parent_examples_sha256": sha(folder/"EXAMPLES.jsonl.gz"),
            "tensor_proof": tensor_proof, "optimizer_age_preserved": age, "example_index_preserved": index,
            "old_receipts_modified": False}


def restore_bc_optimizer(optimizer, saved):
    """Restore Adam and both RNGs only after the caller validated the lineage.

    Isolated generators validate RNG bytes before any live generator is changed.
    No warmup, expert moment or global-age reset occurs on continuation.
    """
    import torch
    cpu = saved["torch_rng"].cpu()
    cuda = [state.cpu() for state in saved["cuda_rng"]]
    if len(cuda) != torch.cuda.device_count():
        raise ValueError("BC CUDA RNG device count mismatch")
    torch.Generator(device="cpu").set_state(cpu)
    for index, state in enumerate(cuda):
        torch.Generator(device=f"cuda:{index}").set_state(state)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state_all(cuda)


def load_bc_optimizer(folder, *, map_location="cpu"):
    """Never unrestricted-unpickle: old BC01 alone needs TorchVersion support.

    BC01 stored torch.__version__ as TorchVersion, not a plain string. Its exact
    original PLAN/RESULT/examples and optimizer hash gate this one class; v2
    serializes plain versions and uses the ordinary weights-only loader.
    """
    import torch
    folder = Path(folder)
    parent = json.loads((folder/"RESULT.json").read_text())
    if sha(folder/"optimizer.pt") != parent["optimizer_sha256"]:
        raise ValueError("parent BC optimizer file identity mismatch")
    legacy = all(sha(folder/name) == value for name, value in LEGACY_BC_ANCHOR.items())
    if legacy:
        from torch.torch_version import TorchVersion
        with torch.serialization.safe_globals([TorchVersion]):
            return torch.load(folder/"optimizer.pt", map_location=map_location, weights_only=True)
    return torch.load(folder/"optimizer.pt", map_location=map_location, weights_only=True)


def fit_bc(output, teacher_dir, updates, examples_limit, batch_size, resume=None):
    import bitsandbytes as bnb
    import torch
    from farm_neat.training.schedule import default_settings, learning_rate_at
    from farm_neat.training.bc import pack_step
    from farm_neat.policy import network as policy
    if type(updates) is not int or not 1 <= updates <= 1024:
        raise ValueError("positive bounded actual BC update count required")
    output = output_directory(output)
    profile = {**device_profile(), "bitsandbytes": str(bnb.__version__)}
    source = bc_sources()
    teacher_binding, _, _ = teacher_data_identity(teacher_dir)
    examples, data = bc_examples(teacher_dir, examples_limit)
    selection = selection_identity(examples, examples_limit, batch_size)
    seed = 90909003
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if resume is None:
        graph = policy.torch_graph(device="cuda")
        parent = None
    else:
        resume = Path(resume)
        parent = json.loads((resume / "RESULT.json").read_text())
        if (parent["status"] != "BC_UPDATES_COMPLETED_NOT_RL_STRENGTH" or
                parent["after_model_sha256"] != sha(resume / "model.pt") or
                parent["optimizer_sha256"] != sha(resume / "optimizer.pt")):
            raise ValueError("completed same-lineage BC model/optimizer required")
        graph, parent_manifest = policy.load_model(resume / "model.pt", device="cuda", expected_file_sha256=parent["after_model_sha256"])
    parameter_names = [name for name, _ in graph.named_parameters()]
    optimizer = bnb.optim.AdamW8bit(graph.parameters(), lr=3e-5, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.0, min_8bit_size=4096, block_wise=True)
    settings, age, index = default_settings(1024), 0, 0
    resume_binding = None
    if parent is not None:
        saved = load_bc_optimizer(resume, map_location="cuda")
        parent_plan = json.loads((resume / "PLAN.json").read_text())
        resume_binding = validate_bc_resume(resume, saved, parent, parent_plan, parent_manifest,
            source=source, teacher=teacher_binding, selection=selection, examples=examples,
            profile=profile, parameter_names=parameter_names)
        settings, age, index = saved["settings"], saved["updates"], saved["next_example_index"]
        restore_bc_optimizer(optimizer, saved)
        del saved  # Do not retain a second full Adam state throughout training.
    if age + updates > settings["planned_optimizer_steps"]:
        raise ValueError("continuation exceeds registered global update budget")
    start_index = index
    plan = {"schema": SCHEMA, "bc_protocol": BC_PROTOCOL,
        "mode": "PARTIAL_OBSERVED_ACTION_BC", "source_pins": source,
        "source_roles": {"BC_ACTIVE": sorted(source),
                         "NOT_EXECUTED_BY_BC": list(BC_EXCLUDED_EXECUTION_SOURCES)},
        "environment": profile, "selection": selection, "teacher": teacher_binding,
        "resume_binding": resume_binding, "start_example_index": start_index,
        "architecture": policy.architecture(), "parameters": policy.parameter_report(graph),
        "teacher_result_sha256": sha(Path(teacher_dir) / "RESULT.json"), "seed": seed,
        "optimizer": "AdamW8bit_NEW_LINEAGE_NOT_OLD_FP32_ADAM_MIGRATION",
        "schedule": settings, "requested_updates": updates, "start_update": age,
        "resume": str(resume) if resume is not None else None, "batch_size": batch_size,
        "examples": len(examples), "loss": "MEAN_PER_EXAMPLE_MEAN_OBSERVED_FACTOR_NLL",
        "no_unobserved_plan_labels": True, "no_terminal_targets": True,
        "no_rl_strength_claim": True}
    write(output / "PLAN.json", plan)
    with gzip.open(output / "EXAMPLES.jsonl.gz", "wt") as stream:
        for example in examples:
            stream.write(canonical(example) + "\n")
    selected_file_sha = sha(output / "EXAMPLES.jsonl.gz")
    plan_sha = sha(output / "PLAN.json")
    if parent is None:
        policy.save_model(output / "genesis.pt", graph, optimizer_steps=0, rollout_iteration=0,
                          initialization_seed=seed, lineage={"kind": "NEW_FULL_ECONOMIC_MOE"})
    before = sha(output / "genesis.pt") if parent is None else parent["after_model_sha256"]
    history = []
    started = time.monotonic()
    for update in range(age + 1, age + updates + 1):
        batch = [examples[(index + offset) % len(examples)] for offset in range(batch_size)]
        index += batch_size
        packs = [pack_step(e["step"]) for e in batch]
        tensors = tensor_inputs(graph, packs)
        prefixes = [e["step"]["examples"][e["example_index"]]["prefix"] for e in batch]
        counts = torch.tensor([len(p["factors"]) for p in prefixes], device="cuda")
        optimizer.zero_grad(set_to_none=True)
        start = time.monotonic()
        context, value, auxiliary = graph.encode(*tensors)
        logp, entropy = graph.replay_prefix(context, prefixes)
        loss = -(logp / counts).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite BC loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(graph.parameters(), 0.5, error_if_nonfinite=True)
        lr = learning_rate_at(update, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()
        receipt = {"update": update, "loss": float(loss.detach()), "gradient_norm": float(norm),
            "learning_rate": lr, "examples": len(batch), "supervised_factors": int(counts.sum()),
            "seconds": time.monotonic() - start, "gpu_peak_bytes": torch.cuda.max_memory_allocated()}
        history.append(receipt)
        print(canonical(receipt), flush=True)
    if (source != bc_sources() or teacher_data_identity(teacher_dir)[0] != teacher_binding
            or sha(output / "EXAMPLES.jsonl.gz") != selected_file_sha):
        raise ValueError("source changed during fit; candidate not exported")
    manifest = policy.save_model(output / "model.pt", graph, optimizer_steps=age + updates, rollout_iteration=0,
        initialization_seed=seed, lineage={"kind": "PARTIAL_OBSERVED_ACTION_BC",
        "parent_file_sha256": before, "teacher_result_sha256": plan["teacher_result_sha256"],
        "bc_protocol": BC_PROTOCOL, "plan_sha256": plan_sha, "selection_sha256": digest(selection)})
    torch.save({"schema": BC_OPTIMIZER_SCHEMA, "optimizer": optimizer.state_dict(),
        "updates": age + updates, "settings": settings, "next_example_index": index,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "model_sha256": sha(output / "model.pt"), "environment": plan["environment"],
        "teacher_result_sha256": plan["teacher_result_sha256"],
        "selection": selection, "teacher": teacher_binding, "plan_sha256": plan_sha,
        "examples_file_sha256": selected_file_sha, "parameter_names": parameter_names,
        "optimizer_config": OPTIMIZER_CONFIG,
        "source_pins": source}, output / "optimizer.pt")
    result = {"schema": SCHEMA, "bc_protocol": BC_PROTOCOL,
        "status": "BC_UPDATES_COMPLETED_NOT_RL_STRENGTH",
        "plan_sha256": plan_sha, "selection": selection, "teacher": teacher_binding,
        "start_example_index": start_index, "next_example_index": index,
        "unique_selected_examples_exposed_in_prefix": min(index, len(examples)),
        "optimizer_updates": updates, "global_optimizer_updates": age + updates,
        "selfplay_games": 0, "teacher_games": data["physical_teacher_games"],
        "before_model_sha256": before, "after_model_sha256": sha(output / "model.pt"),
        "optimizer_sha256": sha(output / "optimizer.pt"), "manifest": manifest["content_sha256"],
        "elapsed_seconds": time.monotonic() - started, "history": history,
        "source_pins": source, "population_strength_established": False}
    write(output / "RESULT.json", result)
    return result


def _state_digest(value):
    """Compare tensor state without lossy JSON conversion or pickling objects."""
    import torch
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            h.update(str((str(t.dtype), tuple(t.shape))).encode()); h.update(t.numpy().tobytes())
        elif isinstance(x, dict):
            h.update(b"dict")
            for k in sorted(x, key=lambda k: (type(k).__name__, repr(k))): visit(k); visit(x[k])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode())
            for item in x: visit(item)
        else:
            h.update(canonical([type(x).__name__, x]).encode())
    visit(value)
    return h.hexdigest()


def observed_prefix_loss(graph, encoder, examples):
    """Existing observed-prefix objective; no sampled logps or latent suffixes."""
    import torch
    from farm_neat.training.bc import pack_step
    packs = [pack_step(e["step"]) for e in examples]
    prefixes = [e["step"]["examples"][e["example_index"]]["prefix"] for e in examples]
    context, _, _ = encoder(*tensor_inputs(graph, packs))
    logp, _ = graph.replay_prefix(context, prefixes)
    counts = torch.tensor([len(p["factors"]) for p in prefixes], device=context.device)
    return -(logp/counts).mean(), int(counts.sum())


def fit_current_bc(output, teacher_dir, *, resume, model_sha256, optimizer_sha256,
                   updates=16, examples_limit=512, batch_size=1, accumulation=8,
                   reserve_bytes=450_000_000, gpu_fraction=.18):
    """Supervised stage on the resident CURRENT fast learner, then return to RL.

    No new genesis/optimizer, no aux reset, no fabricated fresh RL episodes.
    The before checkpoint remains the rollback. Failure writes no committed child.
    Small fit probes establish learning connectivity, not competitive strength.
    """
    import torch
    from farm_neat.training import campaign as campaign
    from farm_neat.policy import network as policy
    from farm_neat.training import ppo as legacy
    for name, v, hi in (("updates", updates, 1024), ("examples", examples_limit, 4096),
                        ("batch", batch_size, 64), ("accumulation", accumulation, 128)):
        if type(v) is not int or not 1 <= v <= hi:
            raise ValueError("bounded positive " + name + " required")
    if type(gpu_fraction) is not float or not 0.05 <= gpu_fraction <= 1.0:
        raise ValueError("explicit GPU memory fraction in[.05,1] required")
    parent = campaign.io.checkpoint(ROOT, resume)
    if (parent["model_sha256"] != model_sha256 or parent["optimizer_sha256"] != optimizer_sha256
            or parent["optimizer_age"]+updates > parent["schedule"]["planned_optimizer_steps"]):
        raise ValueError("exact current parent and inherited update budget required")
    data_binding, data, data_plan = teacher_data_identity(teacher_dir)
    examples, _ = bc_examples(teacher_dir, examples_limit)
    source = {**bc_sources(), **teacher_sources(data_plan.get("teacher_name", "shoprouter"))}
    selected = selection_identity(examples, examples_limit, batch_size)
    storage = campaign._storage_check(reserve_bytes)
    if not storage.get("admissible"):
        raise ValueError("current BC checkpoint output headroom refused")
    output = output_directory(output)
    torch.cuda.set_per_process_memory_fraction(gpu_fraction)
    parent_plan = campaign.io.read(campaign.io.path(ROOT, resume)/"PLAN.json")
    game = parent_plan["execution"].get("resume_game_execution", parent_plan["execution"])
    clock = game.get("economic_clock", {})
    learner = campaign.Learner(resume, parent, checkpoint_blocks=True,
        opening_only=clock.get("publication") == "ATOMIC_OPENING_ONLY_NO_LATER_ECONOMIC_REVISION",
        plan_learning=game.get("plan_learning"), shop_revision_every=clock.get("shop_revision_every", 0),
        full_plan=bool(game.get("full_plan")))
    # Preserve the actual current game contract, not an inferred alternate clock.
    if learner.execution != game:
        raise ValueError("SFT must retain the exact parent game execution contract")
    learner.execution = campaign.supervised_execution(game)
    original = torch.load(campaign.io.path(ROOT, resume)/"optimizer.pt", map_location="cpu", weights_only=True)
    entry = {"model": _state_digest(learner.graph.state_dict()),
        "main_adam": _state_digest(learner.optimizer.state_dict()),
        "cpu_rng": _state_digest(torch.get_rng_state()), "cuda_rng": _state_digest(torch.cuda.get_rng_state_all())}
    if (entry["main_adam"] != _state_digest(original["optimizer"])
            or entry["cpu_rng"] != _state_digest(original["torch_rng"])
            or entry["cuda_rng"] != _state_digest(original["cuda_rng"])):
        raise ValueError("exact optimizer/RNG entry restoration required")
    auxiliary = None if learner.auxiliary is None else {
        "heads": _state_digest(learner.auxiliary.state_dict()),
        "adam": _state_digest(learner.aux_optimizer.state_dict())}
    if auxiliary is not None and auxiliary != {
            "heads": _state_digest(original["plan_learning_state"]["heads"]),
            "adam": _state_digest(original["plan_learning_state"]["optimizer"])}:
        raise ValueError("exact auxiliary entry restoration required")
    del original
    plan = {"schema": campaign.SUPERVISED_STAGE, "execution": learner.execution,
        "source_pins": campaign.sources(), "bc_source_pins": source, "parent": parent,
        "environment": {**learner.profile, "bitsandbytes": str(learner.bnb.__version__)},
        "schedule": learner.settings, "start_update": learner.age, "requested_updates": updates,
        "teacher": data_binding, "selection": selected, "batch_size": batch_size,
        "accumulation": accumulation, "examples_per_optimizer_step": batch_size*accumulation,
        "gpu_memory_fraction": gpu_fraction, "reserve_bytes": reserve_bytes,
        "entry_restoration": entry, "preserved_auxiliary": auxiliary,
        "resume_binding": learner.origin, "sample_order": "DETERMINISTIC_CYCLING_REGISTERED_SELECTION",
        "no_optimizer_reset": True, "no_auxiliary_updates": True, "selfplay_games": 0,
        "probe_scope": "FIXED_TRAIN_PREFIX_NLL_NOT_HELDOUT_OR_STRENGTH"}
    write(output/"PLAN.json", plan); write(output/"STORAGE.json", storage)
    with gzip.open(output/"EXAMPLES.jsonl.gz", "xt") as stream:
        for e in examples: stream.write(canonical(e)+"\n")
    examples_sha = sha(output/"EXAMPLES.jsonl.gz")
    admission = campaign.bound({"schema": campaign.SUPERVISED_STAGE+"-admission",
        "teacher": data_binding, "examples_sha256": examples_sha, "selection": selected,
        "physical_teacher_games": data["physical_teacher_games"], "fresh_reference_games": data["fresh_reference_games"],
        "targets": "ONLY_OBSERVED_FACTORS; UNKNOWN_PLAN_IS_UNKNOWN", "terminal_training_labels": 0})
    write(output/"ADMISSION.json", admission)
    # Hold a CPU rollback; no foreign process/device state is modified.
    before = {"graph": legacy._cpu_copy(learner.graph.state_dict()),
        "optimizer": legacy._cpu_copy(learner.optimizer.state_dict()), "age": learner.age,
        "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
    probe = examples[:min(16, len(examples))]
    def measure():
        with torch.no_grad():
            return sum(float(observed_prefix_loss(learner.graph, learner.encoder, [e])[0]) for e in probe)/len(probe)
    started, index, history = time.monotonic(), 0, []
    try:
        initial_nll = measure()
        for _ in range(updates):
            tick = time.monotonic(); learner.optimizer.zero_grad(set_to_none=True)
            total, factors = 0.0, 0
            for _ in range(accumulation):
                batch = [examples[(index+j) % len(examples)] for j in range(batch_size)]
                index += batch_size
                loss, count = observed_prefix_loss(learner.graph, learner.encoder, batch)
                if not bool(torch.isfinite(loss)): raise ValueError("nonfinite supervised loss")
                (loss/accumulation).backward(); total += float(loss.detach()); factors += count
            norm = torch.nn.utils.clip_grad_norm_(learner.graph.parameters(), learner.settings["gradient_clip"], error_if_nonfinite=True)
            lr = campaign.learning_rate_at(learner.age+1, learner.settings)
            for group in learner.optimizer.param_groups: group["lr"] = lr
            learner.optimizer.step(); torch.cuda.synchronize(); learner.age += 1
            receipt = {"global_update": learner.age, "nll": total/accumulation, "gradient_norm": float(norm),
                "learning_rate": lr, "examples": batch_size*accumulation, "factors": factors,
                "seconds": time.monotonic()-tick, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
            history.append(receipt); print(canonical(receipt), flush=True)
        final_nll = measure()
        if auxiliary is not None and auxiliary != {"heads": _state_digest(learner.auxiliary.state_dict()), "adam": _state_digest(learner.aux_optimizer.state_dict())}:
            raise ValueError("BC unexpectedly changed auxiliary heads or Adam")
        if (source != {**bc_sources(), **teacher_sources(data_plan.get("teacher_name", "shoprouter"))}
                or campaign.sources() != plan["source_pins"] or teacher_data_identity(teacher_dir)[0] != data_binding
                or sha(output/"EXAMPLES.jsonl.gz") != examples_sha):
            raise ValueError("SFT data/source changed before publication")
        changed = [name for name,t in learner.graph.state_dict().items() if not torch.equal(t.detach().cpu(), before["graph"][name])]
        if not changed: raise ValueError("no-op fit is not a committed training update")
        receipt = {"global_optimizer_steps_after": learner.age, "optimizer_updates": updates,
            "initial_probe_nll": initial_nll, "final_probe_nll": final_nll, "history": history,
            "changed_tensor_names": changed, "auxiliary_updates": 0, "seconds": time.monotonic()-started}
        manifest = learner.publish(output, receipt, sha(output/"PLAN.json"), admission, [])
        result = {"schema": campaign.SUPERVISED_STAGE, "status": campaign.io.BC_STATUS,
            "plan_sha256": sha(output/"PLAN.json"), "source_pins": campaign.sources(), "bc_source_pins": source,
            "before_model_sha256": parent["model_sha256"], "after_model_sha256": sha(output/"model.pt"),
            "optimizer_sha256": sha(output/"optimizer.pt"), "manifest": manifest["content_sha256"],
            "optimizer_updates": updates, "global_optimizer_updates": learner.age, "selfplay_games": 0,
            "teacher_games": data["physical_teacher_games"], "teacher": data_binding,
            "auxiliary_preserved": auxiliary, "receipt": receipt, "rollout_iteration_unchanged": True}
        write(output/"RESULT.json", result)
        campaign.io.checkpoint(ROOT, output)
        return result
    except BaseException as error:
        learner.optimizer.zero_grad(set_to_none=True)
        learner.graph.load_state_dict(before["graph"], strict=True); learner.optimizer.load_state_dict(before["optimizer"])
        torch.set_rng_state(before["cpu_rng"]); torch.cuda.set_rng_state_all(before["cuda_rng"]); learner.age = before["age"]
        write(output/"FAILURE.json", {"error": str(error), "attempted_updates": len(history),
            "committed_updates": 0, "rollback_parent": parent, "history": history})
        raise


def selfplay(output, model, seed, command_budget):
    import torch
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    from farm_neat.economy.orders import EconomicOrders
    from farm_neat.policy.observation import pack
    from farm_neat.economy.linter import lint
    from farm_neat.policy import network as policy
    from farm_neat.rollout import rollout as rollout
    output = output_directory(output)
    profile = device_profile()
    source = {**sources(), **rollout.sources()}
    graph, manifest = policy.load_model(model, device="cuda", expected_file_sha256=sha(model))
    graph.eval()
    plan = rollout.make_plan(manifest, sha(model), seed=seed,
        actor_rng_seeds=[seed+1, seed+2], command_budget=command_budget,
        game_id=output.name, environment=profile, source_pins=source)
    write(output / "PLAN.json", plan)
    env = ExactKaggricultureEnvironment(seed=seed)
    controllers = [EconomicOrders(), EconomicOrders()]
    generators = [torch.Generator(device="cuda").manual_seed(seed + s + 1) for s in (0, 1)]
    started = time.monotonic()
    decisions, errors, lint_counts, command_counts = 0, [], Counter(), Counter()
    with gzip.open(output / "turns.jsonl.gz", "wt") as stream:
        while not env.done:
            obs = [env.agent_observation(s) for s in (0, 1)]
            for s in (0, 1):
                controllers[s].observe(obs[s])
            programs, captured, turn_errors = None, None, []
            if env.step_index % 24 == 0:
                packs = [pack(obs[s], controllers[s], env.configuration) for s in (0, 1)]
                with torch.no_grad():
                    context, values, aux = graph.encode(*tensor_inputs(graph, packs))
                    programs = [graph.sample(context[s:s+1], generator=generators[s], max_commands=command_budget)[0][0]
                                for s in (0, 1)]
                captured = [rollout.capture_decision(plan, env.step_index, s, packs[s],
                    programs[s], float(values[s]), controllers[s].snapshot()) for s in (0, 1)]
                for s, record in enumerate(programs):
                    for command in record["program"]:
                        command_counts[command["op"]] += 1
                        try:
                            controllers[s].apply(command)
                        except ValueError as exc:
                            turn_errors.append({"step": env.step_index, "seat": s, "command": command, "reason": str(exc)})
                    decisions += 1
            actions = [controllers[s](obs[s], env.configuration) for s in (0, 1)]
            for s in (0, 1):
                lint_counts.update(f["code"] for f in lint(obs[s], controllers[s].snapshot()))
            errors.extend(turn_errors)
            turn = env.step_index
            env.step(actions)
            stream.write(canonical(rollout.capture_turn(turn, obs, programs, captured, actions,
                [env.agent_observation(s) for s in (0,1)], controllers, turn_errors)) + "\n")
            if env.step_index % 24 == 0:
                print(canonical({"step": env.step_index, "decisions": decisions,
                    "cash": [env.agent_observation(0)["farms"][s]["money"] for s in (0, 1)]}), flush=True)
    if source != {**sources(), **rollout.sources()}:
        raise ValueError("source changed during selfplay")
    if sha(model) != plan["model_sha256"]:
        raise ValueError("behavior model changed during game")
    result = rollout.finish(plan, env, controllers, turns_sha256=sha(output/"turns.jsonl.gz"))
    write(output/"DIAGNOSTICS.json", {"commands":dict(command_counts), "rejected_edits":errors,
        "lint_counts":dict(lint_counts), "seconds":time.monotonic()-started,
        "model_decisions":decisions, "strength_claim":False, "terminal_training_admission":False})
    write(output / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("teacher")
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--teacher-name", choices=("shoprouter", "v35", "flexon", "fixed", "pasture"), default="shoprouter")
    p.add_argument("--external", action="store_true", help="teacher DATA on the declared separate mount")
    p = sub.add_parser("teacher-witness")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--teacher-name", choices=("shoprouter", "v35", "flexon", "fixed", "pasture"), default="shoprouter")
    p = sub.add_parser("public-replay", help="winner SFT prefixes from the declared external replay corpus")
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("public-check", help="fresh raw reconstruction and tensor re-extraction, no fit")
    p.add_argument("--input", required=True)
    p = sub.add_parser("bc")
    p.add_argument("--output", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--updates", type=int, default=8)
    p.add_argument("--examples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--resume")
    p = sub.add_parser("bc-current", help="supervised stage preserving current fast-model Adam/RNG/age, then resume RL")
    p.add_argument("--output", required=True); p.add_argument("--teacher", required=True)
    p.add_argument("--resume", required=True); p.add_argument("--model-sha256", required=True)
    p.add_argument("--optimizer-sha256", required=True)
    p.add_argument("--updates", type=int, default=16); p.add_argument("--examples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--reserve-bytes", type=int, default=450_000_000)
    p.add_argument("--gpu-fraction", type=float, default=.18)
    p = sub.add_parser("selfplay")
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--command-budget", type=int, default=8)
    args = parser.parse_args()
    if args.command == "teacher-witness":
        print(canonical(teacher_game(args.seed, teacher_name=args.teacher_name)))
        return
    if args.command == "public-replay":
        result = public_replay_export(args.output, args.corpus)
    elif args.command == "public-check":
        result = public_replay_check(args.input)
    elif args.command == "teacher":
        result = teacher(args.output, args.seeds, teacher_name=args.teacher_name, external=args.external)
    elif args.command == "bc-current":
        result = fit_current_bc(args.output, args.teacher, resume=args.resume,
            model_sha256=args.model_sha256, optimizer_sha256=args.optimizer_sha256,
            updates=args.updates, examples_limit=args.examples, batch_size=args.batch_size,
            accumulation=args.accumulation, reserve_bytes=args.reserve_bytes, gpu_fraction=args.gpu_fraction)
    elif args.command == "bc":
        if not 1 <= args.updates <= 1024 or not 1 <= args.batch_size <= 64 or not 1 <= args.examples <= 4096:
            raise ValueError("explicit bounded BC pilot required")
        result = fit_bc(args.output, args.teacher, args.updates, args.examples, args.batch_size, args.resume)
    else:
        result = selfplay(args.output, args.model, args.seed, args.command_budget)
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
