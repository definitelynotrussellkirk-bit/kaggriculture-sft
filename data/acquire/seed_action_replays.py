"""Reconstruct public Kaggriculture episodes from source seed and joint actions."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
import copy
from contextlib import ExitStack
import gzip
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

from farm_neat.game.exact_engine import (ExactKaggricultureEnvironment, PASS_ACTION,
                                          _copy_wire, assert_bundled_engine_identity,
                                          sha256_file)


def select_winner(row, ranks, *, own_team_id, top_n=20):
    """Return (winning seat, team ID) only for a current top-N non-self win."""
    team_ids = (row["team_id_0"], row["team_id_1"])
    if own_team_id in team_ids:
        return None
    rewards = (row["reward_0"], row["reward_1"])
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in rewards):
        return None
    if rewards[0] == rewards[1]:
        return None
    winner = int(rewards[1] > rewards[0])
    team = team_ids[winner]
    if team not in ranks or ranks[team][0] > top_n:
        return None
    return winner, team


def reconstruct_episode(row, template):
    """Produce a normal episode record, omitting unknowable wall-clock overage time."""
    episode_id = int(row["episode_id"])
    source_cfg = json.loads(row["config"])
    template_cfg = template["configuration"]
    if any(template_cfg.get(key) != value for key, value in source_cfg.items()):
        raise ValueError("source configuration differs from the pinned template")
    actions = json.loads(zstd.ZstdDecompressor().decompress(row["actions_zstd"]))
    if len(actions) != 719:
        raise ValueError(f"719 joint actions required, got {len(actions)}")
    timeout_at = {}
    for t, pair in enumerate(actions):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("two actions required at every transition")
        for seat, action in enumerate(pair):
            if isinstance(action, dict):
                continue
            if (action is None and t + 1 == row["first_bad_step"]
                    and seat not in timeout_at):
                timeout_at[seat] = t + 1
            else:
                raise ValueError("unexpected missing or non-object action")
    if (row["first_bad_step"] == -1 and timeout_at) or (
            row["first_bad_step"] != -1 and len(timeout_at) != 1):
        raise ValueError("source bad-step marker disagrees with runner timeout action")
    names = json.loads(row["team_names"])
    if not isinstance(names, list) or len(names) != 2:
        raise ValueError("two source team names required")
    env = ExactKaggricultureEnvironment(seed=int(row["seed"]), configuration=source_cfg)
    steps = []
    for t in range(720):
        boundary = []
        for seat in (0, 1):
            state = env.state[seat]
            runner_timeout = seat in timeout_at and timeout_at[seat] <= t < 719
            boundary.append({"action": _copy_wire(PASS_ACTION if t == 0 else actions[t - 1][seat]),
                             "info": _copy_wire(dict(state.info)),
                             "observation": _copy_wire(dict(state.observation)),
                             "reward": None if runner_timeout else float(state.reward),
                             "status": "TIMEOUT" if runner_timeout else str(state.status)})
        steps.append(boundary)
        if t < 719:
            env.step(actions[t])
    rewards = [float(row["reward_0"]), float(row["reward_1"])]
    if not env.done or [s.status for s in env.state] != ["DONE", "DONE"]:
        raise ValueError("source actions did not reach DONE/DONE")
    if [float(s.reward) for s in env.state] != rewards:
        raise ValueError("source terminal rewards differ from exact engine")
    output = {key: copy.deepcopy(value) for key, value in template.items()
              if key not in ("id", "info", "rewards", "statuses", "steps")}
    output["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"kaggriculture/xishengfeng/{episode_id}"))
    output["info"] = {"Agents": [{"Name": name, "ThumbnailUrl": None} for name in names],
                      "EpisodeId": episode_id, "LiveVideoPath": None,
                      "TeamNames": names, "seed": int(row["seed"])}
    output["rewards"] = rewards
    output["statuses"] = ["DONE", "DONE"]
    output["steps"] = steps
    return output


def compare_game_fields(rebuilt, official):
    """Compare every deterministic replay field required to teach recorded actions."""
    if rebuilt["info"]["EpisodeId"] != official["info"]["EpisodeId"]:
        raise ValueError("episode ID mismatch")
    if rebuilt["info"]["seed"] != official["info"]["seed"]:
        raise ValueError("seed mismatch")
    if rebuilt["info"]["TeamNames"] != official["info"]["TeamNames"]:
        raise ValueError("team identity mismatch")
    if rebuilt["configuration"] != official["configuration"]:
        raise ValueError("configuration mismatch")
    if rebuilt["rewards"] != official["rewards"] or rebuilt["statuses"] != official["statuses"]:
        raise ValueError("terminal result mismatch")
    if len(rebuilt["steps"]) != 720 or len(official["steps"]) != 720:
        raise ValueError("720 boundaries required")
    for t in range(720):
        for seat in (0, 1):
            got, want = rebuilt["steps"][t][seat], official["steps"][t][seat]
            actual_obs, expected_obs = dict(got["observation"]), dict(want["observation"])
            actual_obs.pop("remainingOverageTime", None)
            expected_obs.pop("remainingOverageTime", None)
            if actual_obs != expected_obs:
                raise ValueError(f"observation mismatch boundary {t} seat {seat}")
            if got["status"] != want["status"]:
                raise ValueError(f"status mismatch boundary {t} seat {seat}")
            if got["reward"] != want["reward"]:
                raise ValueError(f"reward mismatch boundary {t} seat {seat}")
            if t and got["action"] != want["action"]:
                raise ValueError(f"action mismatch transition {t - 1} seat {seat}")
    return {"checked_boundaries": 1440, "checked_actions": 719,
            "timing_field_excluded": "remainingOverageTime"}


def source_row_sha256(row):
    """Pin all source metadata and both compressed payloads without duplicating them."""
    meta = {key: value for key, value in row.items() if not isinstance(value, bytes)}
    return hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode()
                          + row["actions_zstd"] + row.get("daily_zstd", b"")).hexdigest()


def write_day(day, entries, out_dir, template, *, source_archive_sha256):
    """Publish one complete daily ZIP and its episode/source manifest atomically."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"kaggriculture-episodes-{day}.zip"
    manifest = out_dir / f"MANIFEST-{day}.jsonl"
    if archive.exists() or manifest.exists():
        raise FileExistsError(f"day already published or partially published: {day}")
    temp_archive = out_dir / f".{archive.name}.{os.getpid()}.tmp"
    temp_manifest = out_dir / f".{manifest.name}.{os.getpid()}.tmp"
    count = 0
    try:
        with zipfile.ZipFile(temp_archive, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as target, temp_manifest.open("w") as lines:
            for shard, row_index, row in entries:
                if row["create_time"][:10] != day:
                    raise ValueError(f"source day mismatch: {row['episode_id']}")
                if row["n_steps"] != 720 or row["final_status"] != '["DONE", "DONE"]':
                    raise ValueError(f"source game incomplete: {row['episode_id']}")
                episode = reconstruct_episode(row, template)
                raw = json.dumps(episode, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                name = f"{row['episode_id']}.json"
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                target.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
                receipt = {"episode_id": int(row["episode_id"]), "day": day,
                           "source_archive_sha256": source_archive_sha256,
                           "source_shard": shard, "source_row_index": int(row_index),
                           "source_row_sha256": source_row_sha256(row),
                           "episode_sha256": hashlib.sha256(raw).hexdigest(),
                           "episode_bytes": len(raw),
                           "checks": {"configuration_equal_reference": True,
                                      "joint_actions": 719, "actor_boundaries": 1440,
                                      "engine_terminal_equal_source": True,
                                      "wall_clock_overage_reconstructed": False}}
                lines.write(json.dumps(receipt, separators=(",", ":")) + "\n")
                count += 1
        os.replace(temp_archive, archive)
        os.replace(temp_manifest, manifest)
    finally:
        temp_archive.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)
    return {"day": day, "episodes": count, "archive": str(archive),
            "manifest": str(manifest), "archive_bytes": archive.stat().st_size}


def write_augmented_tape_labels(selected_by_day, *, original_labels, library, state_dir,
                                output, source_archive_sha256):
    """Label supplemental source actions without mutating published official labels."""
    daily_dir = Path(__file__).resolve().parent.parent / "pbc-daily-tapes"
    if str(daily_dir) not in sys.path:
        sys.path.insert(0, str(daily_dir))
    import daily_tape_labels as daily

    original_labels, output = Path(original_labels), Path(output)
    if output.exists():
        raise FileExistsError(output)
    base_sha = sha256_file(original_labels)
    published, seen = daily._label_index(original_labels)
    tapes, openings, library_sha = daily._known_modes(library, state_dir, published)
    output.parent.mkdir(parents=True, exist_ok=True)
    extra_rows = []
    new_tapes_by_day = {}
    with tempfile.TemporaryDirectory(prefix="supplemental-tapes-", dir=output.parent) as temporary:
        for day, entries in sorted(selected_by_day.items()):
            tape_file = Path(temporary) / f"tapes-{day}.jsonl.gz"
            with gzip.open(tape_file, "wt") as stream:
                for _, _, row in entries:
                    if row["create_time"][:10] != day:
                        raise ValueError(f"source day mismatch: {row['episode_id']}")
                    actions = json.loads(zstd.ZstdDecompressor().decompress(row["actions_zstd"]))
                    if (len(actions) != 719 or any(not isinstance(pair, list) or len(pair) != 2
                                                   for pair in actions)):
                        raise ValueError(f"719 joint actions required: {row['episode_id']}")
                    record = {"episode_id": int(row["episode_id"]), "day": day,
                              "teams": json.loads(row["team_names"]),
                              "rewards": [row["reward_0"], row["reward_1"]],
                              "statuses": json.loads(row["final_status"]), "seed": int(row["seed"]),
                              "actions": [[pair[seat] for pair in actions] for seat in (0, 1)]}
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            _, meta, matrix, vocab = daily.tl._load_file(str(tape_file))
            if matrix.shape[1] != 719:
                raise ValueError(f"719 action positions required on {day}")
            keys = {(m["episode_id"], m["seat"]) for m in meta}
            if len(keys) != len(meta) or keys & seen:
                raise ValueError(f"duplicate or published episode-seat on {day}")
            seen.update(keys)
            today, _, _, _ = daily.tl.build(meta, matrix, vocab, workers=1, log=lambda _: None)
            additions = daily._novel_tapes(today["tapes"], tapes)
            tapes.extend(additions)
            new_tapes_by_day[day] = len(additions)
            for length in daily.tl.OPENINGS:
                known = {tuple(item["mode"]) for item in openings[str(length)]}
                for item in today["openings"][str(length)]:
                    if tuple(item["mode"]) not in known:
                        openings[str(length)].append(item)
                        known.add(tuple(item["mode"]))
            extra_rows.extend(daily._match_rows(meta, matrix, vocab, tapes, openings))
    if sha256_file(original_labels) != base_sha:
        raise ValueError("official labels changed during supplemental classification")
    pending = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        shutil.copyfile(original_labels, pending)
        with gzip.open(pending, "ab") as stream:
            for row in extra_rows:
                stream.write((json.dumps(row, separators=(",", ":")) + "\n").encode())
        os.replace(pending, output)
    finally:
        pending.unlink(missing_ok=True)
    return {"output": str(output), "source_archive_sha256": source_archive_sha256,
            "base_labels_sha256": base_sha, "base_library_sha256": library_sha,
            "supplemental_seats": len(extra_rows),
            "static_seats": sum(row["tape_id"] is not None for row in extra_rows),
            "new_tapes_by_day": new_tapes_by_day,
            "output_sha256": sha256_file(output)}


def verify_overlaps(entries, official_map, receipt_path, *, source_archive_sha256, limit=None):
    """Check every shared game's deterministic fields, with resumable per-game receipts."""
    entries = list(entries)
    ids = [int(row["episode_id"]) for _, _, row in entries]
    if len(set(ids)) != len(ids) or any(eid not in official_map for eid in ids):
        raise ValueError("overlap entries must have unique official episode IDs")
    receipt_path = Path(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if receipt_path.exists():
        with receipt_path.open() as stream:
            for line in stream:
                record = json.loads(line)
                eid = int(record["episode_id"])
                if eid in prior:
                    raise ValueError(f"duplicate overlap receipt: {eid}")
                if record["source_archive_sha256"] != source_archive_sha256:
                    raise ValueError("source archive changed after overlap verification")
                prior[eid] = record
    new = 0
    with ExitStack() as stack, receipt_path.open("a") as output:
        archives = {}
        for shard, row_index, row in entries:
            eid = int(row["episode_id"])
            row_hash = source_row_sha256(row)
            if eid in prior:
                if prior[eid]["source_row_sha256"] != row_hash:
                    raise ValueError(f"source row changed after verification: {eid}")
                continue
            if limit is not None and new >= limit:
                break
            archive_path, member = official_map[eid]
            archive_path = Path(archive_path)
            if archive_path not in archives:
                archives[archive_path] = stack.enter_context(zipfile.ZipFile(archive_path))
            official_raw = archives[archive_path].read(member)
            official = json.loads(official_raw)
            rebuilt = reconstruct_episode(row, official)
            checks = compare_game_fields(rebuilt, official)
            receipt = {"episode_id": eid, "source_archive_sha256": source_archive_sha256,
                       "source_shard": shard, "source_row_index": int(row_index),
                       "source_row_sha256": row_hash, "official_archive": str(archive_path),
                       "official_member_sha256": hashlib.sha256(official_raw).hexdigest(),
                       "checks": checks}
            output.write(json.dumps(receipt, separators=(",", ":")) + "\n")
            prior[eid] = receipt
            new += 1
            if new % 100 == 0:
                output.flush()
                os.fsync(output.fileno())
                print(f"verified overlaps: {len(prior)}/{len(entries)}", flush=True)
    return {"total_overlaps": len(entries), "verified": len(prior), "new": new,
            "complete": len(prior) == len(entries),
            "timing_field_excluded": "remainingOverageTime"}


def catalog_source(source_zip, official_ids, ranks, *, reference_config, own_team_id,
                   since, until, exclude_ids=()):
    """Inventory downloaded rows once; hold only shared games and eligible new wins."""
    core_cfg = {key: value for key, value in reference_config.items()
                if key not in ("actTimeout", "runTimeout", "seed")}
    official_ids = set(official_ids)
    exclude_ids = set(exclude_ids)
    overlaps, nonreconstructable_overlaps, selected = [], [], defaultdict(list)
    counts, by_team = Counter(), Counter()
    seen = {}
    with zipfile.ZipFile(source_zip) as source:
        shards = sorted(name for name in source.namelist()
                        if name.startswith("replay_db/shards/ep_") and name.endswith(".parquet"))
        if not shards:
            raise ValueError("source archive has no episode shards")
        for name in shards:
            shard = Path(name).name
            table = pq.read_table(pa.BufferReader(source.read(name)))
            for row_index, row in enumerate(table.to_pylist()):
                counts["source_rows"] += 1
                eid = int(row["episode_id"])
                if eid in seen:
                    if seen[eid] != source_row_sha256(row):
                        raise ValueError(f"conflicting duplicate source episode {eid}")
                    counts["duplicate_rows"] += 1
                    continue
                seen[eid] = source_row_sha256(row)
                entry = (shard, row_index, row)
                if eid in official_ids:
                    counts["official_overlap"] += 1
                    complete = (row["n_steps"] == 720
                                and row["final_status"] == '["DONE", "DONE"]')
                    if complete and (row["first_bad_step"] == -1
                                     or row["first_bad_step"] > 0):
                        overlaps.append(entry)
                        counts["official_overlap_reconstructable"] += 1
                        if row["first_bad_step"] > 0:
                            counts["official_overlap_runner_timeout"] += 1
                    else:
                        nonreconstructable_overlaps.append({
                            "episode_id": eid, "source_shard": shard,
                            "source_row_index": row_index,
                            "first_bad_step": row["first_bad_step"],
                            "n_steps": row["n_steps"],
                            "final_status": row["final_status"],
                            "reason": "runner_failure_or_incomplete_action_stream",
                            "source_row_sha256": seen[eid],
                        })
                        counts["official_overlap_runner_failure"] += 1
                    continue
                day = row["create_time"][:10]
                if not since <= day <= until:
                    counts["outside_window"] += 1
                    continue
                chosen = select_winner(row, ranks, own_team_id=own_team_id)
                if chosen is None:
                    counts["not_top20_clean_win"] += 1
                    continue
                if (row["n_steps"] != 720 or row["first_bad_step"] != -1
                        or row["final_status"] != '["DONE", "DONE"]'):
                    counts["not_complete_clean"] += 1
                    continue
                if json.loads(row["config"]) != core_cfg:
                    counts["different_configuration"] += 1
                    continue
                if eid in exclude_ids:
                    counts["already_in_training_archive"] += 1
                    continue
                selected[day].append(entry)
                counts["selected"] += 1
                by_team[int(chosen[1])] += 1
    return {"overlaps": overlaps,
            "nonreconstructable_overlaps": nonreconstructable_overlaps,
            "selected_by_day": dict(sorted(selected.items())),
            "counts": dict(counts), "selected_by_team": dict(by_team),
            "unique_source_episodes": len(seen)}


def main(argv=None):
    """Survey, verify overlaps, then reconstruct admitted top-20-team wins."""
    base = Path("/mnt/ai/scratch/farmNEAT/primitive-bc-data")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("survey", "verify", "build", "labels"))
    parser.add_argument("--source", type=Path,
                        default=base / "extra/raw/xishengfeng/kaggriculture-replay-db.zip")
    parser.add_argument("--official-archive", type=Path, action="append")
    parser.add_argument("--leaderboard", type=Path,
                        default=base / "extra/index/kaggriculture-publicleaderboard-2026-09-25T03:37:05.csv")
    parser.add_argument("--reference-config", type=Path, default=base / "REFERENCE-CONFIG.json")
    parser.add_argument("--exclude-archive-list", type=Path,
                        default=Path("/mnt/ai/scratch/farmNEAT/primitive-bc-v1-b4-20260925/P4-ARCHIVES.txt"))
    parser.add_argument("--out", type=Path, default=base / "extra/reconstructed")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--own-team-id", type=int, default=16624026)
    parser.add_argument("--since", default="2026-09-03")
    parser.add_argument("--until", default="2026-09-23")
    parser.add_argument("--limit", type=int, help="new overlaps to verify this invocation")
    parser.add_argument("--max-days", type=int, help="daily archives to build this invocation")
    parser.add_argument("--day", action="append",
                        help="build only this selected day (repeatable; requires the full overlap proof)")
    tape_base = Path("/mnt/ai/scratch/farmNEAT/pbc-tape-corpus-20260923")
    parser.add_argument("--original-labels", type=Path,
                        default=tape_base / "EPISODE-TAPE-LABELS.jsonl.gz")
    parser.add_argument("--tape-library", type=Path, default=tape_base / "TAPE-LIBRARY.json")
    parser.add_argument("--tape-state-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "pbc-daily-tapes/state")
    args = parser.parse_args(argv)
    if args.day and args.max_days is not None:
        parser.error("--day and --max-days cannot be combined")
    if args.day and args.mode != "build":
        parser.error("--day is only valid for build")
    if args.day and len(args.day) != len(set(args.day)):
        parser.error("--day values must be unique")
    begun = time.perf_counter()
    args.work = args.work or args.out / ".work"
    args.work.mkdir(parents=True, exist_ok=True)
    with args.leaderboard.open(encoding="utf-8-sig") as stream:
        ranks = {int(row["TeamId"]): (int(row["Rank"]), row["TeamName"])
                 for row in csv.DictReader(stream)}
    reference = json.loads(args.reference_config.read_text())
    reference.pop("seed", None)
    exclude_archive_list_sha = sha256_file(args.exclude_archive_list)
    exclude_ids = set()
    for line in args.exclude_archive_list.read_text().splitlines():
        if not line.strip():
            continue
        with zipfile.ZipFile(line.strip()) as existing:
            exclude_ids.update(int(name[:-5]) for name in existing.namelist()
                               if name.endswith(".json") and name[:-5].isdigit())
    exclude_ids_sha = hashlib.sha256(json.dumps(sorted(exclude_ids)).encode()).hexdigest()
    if args.official_archive:
        archives = args.official_archive
    else:
        archives = [Path(p) for p in sorted(glob.glob(str(base / "kaggriculture-episodes-*.zip")))]
        archives += [Path(p) for p in sorted(glob.glob(str(base / "extra/sample-run/archives/*.zip")))]
    # Use a recent current-configuration body once; old daily JSONs are large and
    # cannot supply the current template anyway. Index every ZIP by central directory.
    template = None
    dated = lambda path: (re.search(r"2026-\d\d-\d\d", path.name) or [""])[0]
    for archive in sorted(archives, key=lambda p: (dated(p), p.name.startswith("kaggriculture-episodes-")),
                          reverse=True):
        with zipfile.ZipFile(archive) as source:
            member = next((name for name in source.namelist()
                           if name.endswith(".json") and name[:-5].isdigit()), None)
            if member is None:
                continue
            candidate = json.loads(source.read(member))
            cfg = dict(candidate["configuration"])
            cfg.pop("seed", None)
            if cfg == reference:
                template = candidate
                break
    official_map = {}
    for archive in archives:
        with zipfile.ZipFile(archive) as source:
            for member in source.namelist():
                if not member.endswith(".json") or not member[:-5].isdigit():
                    continue
                eid = int(member[:-5])
                official_map.setdefault(eid, (archive, member))
    if template is None:
        raise ValueError("no official replay with reference configuration")
    source_sha = sha256_file(args.source)
    catalog = catalog_source(args.source, official_map, ranks,
                             reference_config=reference, own_team_id=args.own_team_id,
                             since=args.since, until=args.until, exclude_ids=exclude_ids)
    summary = {"schema": "pbc-seed-action-replay-catalog-v1",
               "source_archive": str(args.source), "source_archive_sha256": source_sha,
               "reference_configuration": reference,
               "exclude_archive_list": str(args.exclude_archive_list),
               "exclude_archive_list_sha256": exclude_archive_list_sha,
               "exclude_episode_ids": len(exclude_ids),
               "exclude_episode_ids_sha256": exclude_ids_sha,
               "source_rows": catalog["counts"].get("source_rows", 0),
               "unique_source_episodes": catalog["unique_source_episodes"],
               "official_overlap": catalog["counts"].get("official_overlap", 0),
               "official_overlap_reconstructable": len(catalog["overlaps"]),
               "nonreconstructable_overlaps": catalog["nonreconstructable_overlaps"],
               "selected": catalog["counts"].get("selected", 0),
               "selected_by_day": {day: len(rows) for day, rows in catalog["selected_by_day"].items()},
               "selected_by_team": catalog["selected_by_team"],
               "filter_counts": catalog["counts"],
               "elapsed_catalog_seconds": round(time.perf_counter() - begun, 3)}
    _atomic_json(args.work / "CATALOG.json", summary)
    if args.mode == "survey":
        print(json.dumps(summary, sort_keys=True), flush=True)
        return summary
    if args.mode == "verify":
        result = verify_overlaps(catalog["overlaps"], official_map, args.work / "VERIFY.jsonl",
                                 source_archive_sha256=source_sha, limit=args.limit)
        result.update({"schema": "pbc-seed-action-overlap-verification-v1",
                       "source_archive_sha256": source_sha,
                       "exclude_archive_list_sha256": exclude_archive_list_sha,
                       "exclude_episode_ids_sha256": exclude_ids_sha,
                       "nonreconstructable_overlaps": catalog["nonreconstructable_overlaps"],
                       "engine": assert_bundled_engine_identity(),
                       "elapsed_seconds": round(time.perf_counter() - begun, 3)})
        _atomic_json(args.work / "VERIFY-SUMMARY.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    if args.mode == "labels":
        base_labels_sha = sha256_file(args.original_labels)
        target = args.work / ("AUGMENTED-EPISODE-TAPE-LABELS-"
                              f"{base_labels_sha[:12]}-{source_sha[:12]}.jsonl.gz")
        result = write_augmented_tape_labels(
            catalog["selected_by_day"], original_labels=args.original_labels,
            library=args.tape_library, state_dir=args.tape_state_dir, output=target,
            source_archive_sha256=source_sha)
        result.update({"schema": "pbc-seed-action-augmented-tape-labels-v1",
                       "selected_episodes": summary["selected"],
                       "elapsed_seconds": round(time.perf_counter() - begun, 3)})
        _atomic_json(args.work / "LABELS-SUMMARY.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    verification_path = args.work / "VERIFY-SUMMARY.json"
    if not verification_path.is_file():
        raise ValueError("complete overlap verification required before build")
    verification = json.loads(verification_path.read_text())
    if (not verification.get("complete") or verification.get("source_archive_sha256") != source_sha
            or verification.get("exclude_archive_list_sha256") != exclude_archive_list_sha
            or verification.get("exclude_episode_ids_sha256") != exclude_ids_sha
            or verification.get("total_overlaps") != len(catalog["overlaps"])):
        raise ValueError("complete overlap verification for this source required before build")
    confirmed = verify_overlaps(catalog["overlaps"], official_map, args.work / "VERIFY.jsonl",
                                source_archive_sha256=source_sha, limit=0)
    if not confirmed["complete"]:
        raise ValueError("overlap receipts are incomplete")
    args.out.mkdir(parents=True, exist_ok=True)
    days = list(catalog["selected_by_day"].items())
    if args.day:
        unknown = set(args.day) - set(catalog["selected_by_day"])
        if unknown:
            raise ValueError(f"requested --day not in selected source catalog: {sorted(unknown)}")
        requested_days = set(args.day)
    else:
        requested_days = {day for day, _ in
                          (days[:args.max_days] if args.max_days is not None else days)}
    # The global manifest counts every day already present in --out. Revalidate
    # those days too, or an unrelated --day build could silently admit a stray ZIP.
    present_days = {day for day, _ in days
                    if (args.out / f"kaggriculture-episodes-{day}.zip").exists()
                    or (args.out / f"MANIFEST-{day}.jsonl").exists()}
    selected_days = [(day, entries) for day, entries in days
                     if day in requested_days or day in present_days]
    receipts = []
    for day, entries in selected_days:
        archive = args.out / f"kaggriculture-episodes-{day}.zip"
        manifest = args.out / f"MANIFEST-{day}.jsonl"
        if archive.exists() or manifest.exists():
            if not archive.is_file() or not manifest.is_file():
                raise ValueError(f"partial publication of {day}; refusing overwrite")
            published = [json.loads(line) for line in manifest.read_text().splitlines()]
            expected = {int(row["episode_id"]): (shard, row_index, source_row_sha256(row))
                        for shard, row_index, row in entries}
            if (len(published) != len(expected)
                    or any("episode_id" not in line for line in published)
                    or {line["episode_id"] for line in published} != set(expected)):
                raise ValueError(f"published manifest membership differs for {day}")
            with zipfile.ZipFile(archive) as packed:
                if len(packed.namelist()) != len(expected) or set(packed.namelist()) != {
                        f"{eid}.json" for eid in expected}:
                    raise ValueError(f"published ZIP membership differs for {day}")
                for line in published:
                    eid = int(line["episode_id"])
                    shard, row_index, row_hash = expected[eid]
                    if (line["day"] != day or line["source_archive_sha256"] != source_sha
                            or line["source_shard"] != shard
                            or line["source_row_index"] != row_index
                            or line["source_row_sha256"] != row_hash):
                        raise ValueError(f"published source receipt differs for {day}/{eid}")
                    raw = packed.read(f"{eid}.json")
                    if hashlib.sha256(raw).hexdigest() != line["episode_sha256"]:
                        raise ValueError(f"episode SHA mismatch for {day}/{eid}")
                    if len(raw) != line["episode_bytes"]:
                        raise ValueError(f"episode length mismatch for {day}/{eid}")
            receipt = {"day": day, "episodes": len(published), "archive": str(archive),
                       "manifest": str(manifest), "archive_bytes": archive.stat().st_size,
                       "already_published": True}
        else:
            receipt = write_day(day, entries, args.out, template,
                                source_archive_sha256=source_sha)
        receipts.append(receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
    published_days = [day for day, _ in days
                      if (args.out / f"MANIFEST-{day}.jsonl").is_file()
                      and (args.out / f"kaggriculture-episodes-{day}.zip").is_file()]
    full = len(published_days) == len(days)
    global_manifest = {"schema": "pbc-seed-action-reconstructed-v1",
                       "source_archive_sha256": source_sha,
                       "exclude_archive_list_sha256": exclude_archive_list_sha,
                       "exclude_episode_ids_sha256": exclude_ids_sha,
                       "overlap_verification": str(verification_path),
                       "official_overlap_verified_clean": confirmed["verified"],
                       "official_overlap_nonreconstructable": len(catalog["nonreconstructable_overlaps"]),
                       "episodes": sum(len(catalog["selected_by_day"][day]) for day in published_days),
                       "days": receipts, "published_day_names": published_days,
                       "validated_this_invocation": [day for day, _ in selected_days],
                       "complete": full,
                       "elapsed_seconds": round(time.perf_counter() - begun, 3)}
    _atomic_json(args.out / "MANIFEST.json", global_manifest)
    print(json.dumps(global_manifest, sort_keys=True), flush=True)
    return global_manifest


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        pending.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


if __name__ == "__main__":
    main(sys.argv[1:])
