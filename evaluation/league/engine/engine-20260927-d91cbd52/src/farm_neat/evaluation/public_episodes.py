"""Track one team's public ladder games as small, exactly reconstructable capsules.

    discover --team NAME --seed-submission ID [...]   find the team's submission ids
    index    --team NAME --submission ID [...] --out DIR
    collect  --out DIR [--limit N]                      download, verify, capsule, delete raw
    check    CAPSULE                                    re-verify a capsule in the exact engine

Kaggle lists episodes per submission, so a team's submissions are found by walking
opponents' episode lists outward from a known submission. A capsule keeps the seed,
configuration and both seats' recorded actions: observations are recomputed by the
exact engine (both seats' actions are needed for that), and `focus` names the tracked
team's seat. The raw replay is deleted only after reconstruction matches all 1,440
actor boundaries. Capsules are public-ladder imitation data, not admitted V0 teachers.
Network calls only list and download public episodes; nothing is submitted.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

from farm_neat.evaluation.common import bound, canonical

SCHEMA = "farmneat-public-episode-capsule-v1"
INDEX_SCHEMA = "farmneat-public-episode-index-v1"
PAUSE = 0.5


def _client():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _episodes(api, submission_id: int) -> list[dict]:
    return [{"id": e.id, "end": str(e.end_time), "type": str(e.type), "state": str(e.state),
             "agents": [{"submission_id": a.submission_id, "team": a.team_name, "index": a.index,
                         "reward": a.reward} for a in e.agents]}
            for e in api.competition_list_episodes(submission_id)]


def team_rows(team: str, submission_id: int, listing: list[dict]) -> list[dict]:
    """Completed public two-seat games of `submission_id` with the team's seat and rival."""
    rows = []
    for e in listing:
        if "PUBLIC" not in e["type"] or "COMPLETED" not in e["state"] or len(e["agents"]) != 2:
            continue
        mine = [a for a in e["agents"] if a["submission_id"] == submission_id and a["team"] == team]
        if len(mine) != 1:
            continue
        seat = mine[0]["index"]
        other = next(a for a in e["agents"] if a is not mine[0])
        if {seat, other["index"]} != {0, 1} or mine[0]["reward"] is None or other["reward"] is None:
            continue
        margin = mine[0]["reward"] - other["reward"]
        rows.append({"episode_id": e["id"], "end": e["end"], "submission_id": submission_id, "seat": seat,
                     "reward": mine[0]["reward"], "rival_reward": other["reward"], "rival_team": other["team"],
                     "rival_submission_id": other["submission_id"],
                     "wdl": "WIN" if margin > 0 else "LOSS" if margin < 0 else "DRAW"})
    return rows


def discover(api, team: str, seeds: list[int], max_lists: int = 40) -> dict:
    """Breadth-first over submission episode lists; the team's own lists are read first."""
    found, seen, queue = {}, set(), list(seeds)
    while queue and len(seen) < max_lists:
        sub = queue.pop(0)
        if sub in seen:
            continue
        seen.add(sub)
        for e in _episodes(api, sub):
            for a in e["agents"]:
                if a["team"] == team and a["submission_id"] not in found:
                    found[a["submission_id"]] = e["id"]
                    queue.insert(0, a["submission_id"])
                elif a["submission_id"] not in seen:
                    queue.append(a["submission_id"])
        time.sleep(PAUSE)
    return {"team": team, "submissions": {str(k): {"first_seen_episode": v} for k, v in sorted(found.items())},
            "lists_read": len(seen),
            "note": "found by walking episode lists; a submission absent from the visited lists is not proven absent"}


def index(api, team: str, submissions: list[int], out: Path) -> dict:
    rows = []
    for sub in submissions:
        rows += team_rows(team, sub, _episodes(api, sub))
        time.sleep(PAUSE)
    rows.sort(key=lambda r: r["episode_id"])
    by_sub = {}
    for r in rows:
        s = by_sub.setdefault(str(r["submission_id"]), {"games": 0, "wins": 0, "draws": 0, "losses": 0})
        s["games"] += 1
        s[{"WIN": "wins", "DRAW": "draws", "LOSS": "losses"}[r["wdl"]]] += 1
    value = bound({"schema": INDEX_SCHEMA, "team": team, "submissions": by_sub, "episodes": rows,
                   "listed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "scope": "COMPLETED_PUBLIC_TWO_SEAT_EPISODES_LISTED_FOR_THESE_SUBMISSIONS"})
    out.mkdir(parents=True, exist_ok=True)
    (out/"INDEX.json").write_text(canonical(value) + "\n")
    return value


def capsule(replay: dict, team: str, row: dict, raw_sha256: str, raw_bytes: int) -> dict:
    """Verify in the exact engine, then keep seed, configuration and both seats' actions."""
    from farm_neat.training.bc import verify_public_replay
    header = verify_public_replay(replay)
    if header["episode_id"] != row["episode_id"]:
        raise ValueError("episode identity mismatch")
    if replay["info"].get("TeamNames", [None, None])[row["seat"]] != team:
        raise ValueError("tracked team is not in the listed seat")
    if replay["rewards"][row["seat"]] != row["reward"] or replay["rewards"][1 - row["seat"]] != row["rival_reward"]:
        raise ValueError("listed rewards disagree with the replay")
    actions = [[replay["steps"][t][s]["action"] for s in (0, 1)] for t in range(1, 720)]
    return bound({"schema": SCHEMA, "episode_id": row["episode_id"], "seed": header["seed"],
                  "configuration": replay["configuration"], "rewards": replay["rewards"],
                  "teams": replay["info"].get("TeamNames"),
                  "focus": {k: row[k] for k in ("seat", "submission_id", "wdl")},
                  "rival": {"team": row["rival_team"], "submission_id": row["rival_submission_id"]},
                  "joint_actions": actions, "engine": header["engine"],
                  "raw": {"sha256": raw_sha256, "bytes": raw_bytes},
                  "verified_actor_boundaries": header["checked_actor_boundaries"]})


def restore(value: dict):
    """Yield (step, focus-seat observation, focus-seat action) by exact re-simulation."""
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    if value.get("schema") != SCHEMA or len(value["joint_actions"]) != 719:
        raise ValueError("complete public episode capsule required")
    seat = value["focus"]["seat"]
    env = ExactKaggricultureEnvironment(seed=value["seed"], configuration=value["configuration"])
    for t, joint in enumerate(value["joint_actions"]):
        yield t, env.agent_observation(seat), joint[seat]
        env.step(joint)
    if [f["money"] for f in env.agent_observation(seat)["farms"]] != value["rewards"]:
        raise ValueError("capsule reconstruction disagrees with terminal cash")


def write_capsule(path: Path, value: dict):
    data = gzip.compress(canonical(value).encode(), mtime=0)
    with path.open("xb") as stream:
        stream.write(data)


def read_capsule(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def collect(api, out: Path, scratch: Path, limit: int | None) -> dict:
    listing = json.loads((out/"INDEX.json").read_text())
    folder = out/"episodes"
    folder.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    for row in listing["episodes"]:
        target = folder/f"{row['episode_id']}.json.gz"
        if target.exists():
            continue
        if limit is not None and done + failed >= limit:
            break
        raw = scratch/f"episode-{row['episode_id']}-replay.json"
        try:
            api.competition_episode_replay(row["episode_id"], path=str(scratch))
            payload = raw.read_bytes()
            write_capsule(target, capsule(json.loads(payload), listing["team"], row, hashlib.sha256(payload).hexdigest(), len(payload)))
            done += 1
        except Exception as error:  # keep going; the failure is journalled, the raw file removed
            failed += 1
            with (out/"FAILURES.jsonl").open("a") as stream:
                stream.write(canonical({"episode_id": row["episode_id"], "error": repr(error)}) + "\n")
        finally:
            raw.unlink(missing_ok=True)
        time.sleep(PAUSE)
    have = len(list(folder.glob("*.json.gz")))
    print(canonical({"captured_now": done, "failed_now": failed, "capsules": have, "listed": len(listing["episodes"])}), flush=True)
    return {"captured_now": done, "failed_now": failed, "capsules": have}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover"); d.add_argument("--team", required=True)
    d.add_argument("--seed-submission", type=int, nargs="+", required=True)
    i = sub.add_parser("index"); i.add_argument("--team", required=True)
    i.add_argument("--submission", type=int, nargs="+", required=True); i.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("collect"); c.add_argument("--out", type=Path, required=True)
    c.add_argument("--scratch", type=Path, required=True); c.add_argument("--limit", type=int)
    k = sub.add_parser("check"); k.add_argument("capsule", type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        value = read_capsule(args.capsule)
        steps = sum(1 for _ in restore(value))
        print(canonical({"episode_id": value["episode_id"], "focus": value["focus"], "steps": steps, "status": "PASS"}))
        return 0
    api = _client()
    if args.command == "discover":
        print(canonical(discover(api, args.team, args.seed_submission)))
    elif args.command == "index":
        value = index(api, args.team, args.submission, args.out)
        print(canonical(value["submissions"]))
    else:
        collect(api, args.out, args.scratch, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
