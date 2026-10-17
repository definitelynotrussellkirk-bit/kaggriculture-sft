#!/usr/bin/env python3
"""Pack finished league replays into the local replay catalog's fetch-folder format, so the catalog and every tool
that reads catalogued zips see league games (primitive-bc-data/catalog/README.md, "Adding data"):

  <export dir>/archives/league-<UTC hour>Z-<stamp>.zip   members <EpisodeId>.json (Kaggle replay JSON, deflated)
  <export dir>/MANIFEST.jsonl                            one row per member, appended after the zip is renamed

Only episodes that finished before the current UTC hour are packed; an episode is packed once (its EpisodeId in the
manifest), zips are never rewritten, and every member is read back and hash-checked before the rename (the order of
fetch_topteam_replays.py pack()). EpisodeIds are 7100000000000000 + episode id and both seats are SELF:<entry>, so the
catalog files them as local self-play (excluded from training queries by default). Run by league-backup.timer.

After a member is safely in its zip, the episode's replay_path becomes "zip:<zip path>#<member>" (the API serves it
from there) and the per-episode replay.json.gz is deleted, so a replay is stored once.
"""
import gzip
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import league_core as core  # noqa: E402


def manifest_ids(out):
    path = out / "MANIFEST.jsonl"
    ids = set()
    if path.is_file():
        for line in open(path):
            if line.strip():
                ids.add(int(json.loads(line)["episode_id"]))
    return ids


def export(cfg, limit_hours=None):
    ex = cfg.get("export") or {}
    if not ex.get("enabled", True):
        return {"enabled": False}
    out = Path(ex.get("dir", "/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/league"))
    (out / "archives").mkdir(parents=True, exist_ok=True)
    for tmp in (out / "archives").glob("*.tmp"):
        tmp.unlink()
    done = manifest_ids(out)
    cutoff = time.strftime("%Y-%m-%dT%H:00:00Z", time.gmtime())
    con = core.connect(cfg["service"]["db"], init=False)
    names = {r["id"]: r["name"] for r in con.execute("SELECT id, name FROM submissions")}
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM episodes WHERE state='done' AND source='league' AND replay_path IS NOT NULL "
        "AND replay_path NOT LIKE 'zip:%' AND finished_utc < ? ORDER BY id", (cutoff,))]
    by_hour = {}
    for r in rows:
        eid = core.LEAGUE_EPISODE_BASE + r["id"]
        if eid in done or not Path(r["replay_path"]).is_file():
            continue
        by_hour.setdefault(r["finished_utc"][:13], []).append(r)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    packed = 0
    for hour, eps in sorted(by_hour.items())[:limit_hours]:
        name = f"league-{hour}Z-{stamp}.zip"
        tmp = out / "archives" / (name + ".tmp")
        metas = []
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for r in eps:
                eid = core.LEAGUE_EPISODE_BASE + r["id"]
                data = gzip.decompress(Path(r["replay_path"]).read_bytes())
                z.writestr(f"{eid}.json", data)
                c0, c1 = r["cash0"], r["cash1"]
                metas.append({"episode_id": eid, "member": f"{eid}.json", "sha256": hashlib.sha256(data).hexdigest(),
                              "bytes": len(data), "day": r["finished_utc"][:10],
                              "team_names": [f"SELF:{names.get(r['seat0'])}", f"SELF:{names.get(r['seat1'])}"],
                              "team_ids": [None, None], "submission_ids": [r["seat0"], r["seat1"]],
                              "rewards": [c0, c1], "winner_seat": 0 if c0 > c1 else 1 if c1 > c0 else None,
                              "reason": "league", "versions": [names.get(r["seat0"]), names.get(r["seat1"])],
                              "league_episode": r["id"], "world_seed": r["world_seed"], "rated": bool(r["rated"]),
                              "finished_utc": r["finished_utc"]})
                if json.loads(r.get("runtime") or "{}").get("crn_rng"):
                    metas[-1]["crn_rng"] = True  # replay configuration crnRng: re-simulate with crn_rng.environment()
        with zipfile.ZipFile(tmp) as z:
            for m in metas:
                if hashlib.sha256(z.read(m["member"])).hexdigest() != m["sha256"]:
                    raise RuntimeError(f"zip readback hash mismatch {m['member']}")
        os.replace(tmp, out / "archives" / name)
        with open(out / "MANIFEST.jsonl", "a") as handle:
            handle.write("\n".join(json.dumps({**m, "archive": f"archives/{name}"}, sort_keys=True) for m in metas) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        for r, m in zip(eps, metas):
            con.execute("UPDATE episodes SET replay_path=? WHERE id=?", (f"zip:{out / 'archives' / name}#{m['member']}",
                                                                        r["id"]))
        con.commit()
        for r in eps:
            Path(r["replay_path"]).unlink(missing_ok=True)
        packed += len(metas)
    # a crash between the DB update and the unlink leaves a loose copy: remove it
    for r in con.execute("SELECT episode_dir FROM episodes WHERE replay_path LIKE 'zip:%' AND episode_dir IS NOT NULL "
                         "AND finished_utc >= ?", (time.strftime("%Y-%m-%d", time.gmtime(time.time() - 3 * 86400)),)):
        (Path(r["episode_dir"]) / "replay.json.gz").unlink(missing_ok=True)
    con.close()
    return {"enabled": True, "dir": str(out), "packed": packed, "hours": len(by_hour)}


def read_replay(replay_path):
    """Replay JSON bytes (uncompressed) from a per-episode .json.gz or a "zip:<zip>#<member>" reference."""
    if replay_path.startswith("zip:"):
        zpath, member = replay_path[4:].rsplit("#", 1)
        with zipfile.ZipFile(zpath) as z:
            return z.read(member)
    return gzip.decompress(Path(replay_path).read_bytes())


def main():
    print(json.dumps(export(core.load_config())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
