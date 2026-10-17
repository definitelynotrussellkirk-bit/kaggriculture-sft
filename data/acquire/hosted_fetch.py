"""Fetch a submission's most recent hosted episodes (replay + our agent's logs) and summarise record and per-step latency.

  work/venv/python -B work/runs/maintainer/pbc-autosubmit/hosted_fetch.py SUBMISSION_ID OUT_DIR [--n 16] [--team "Russell Kirk"]

Writes OUT_DIR/EPISODES.csv, episode-<id>-replay.json.gz, episode-<id>-agent-<seat>-logs.json.gz, EPISODE-SUMMARY.json and
HOSTED-LATENCY.json (schema pbc-hosted-latency-v1, as work/runs/deployment/pbc-v1-20000-t0/hosted/). Already-fetched files are reused.
Step latency comes from the agent's own stderr lines "[pbc] step=S ms=X fb=F"; step 0 includes the model load.
"""
import argparse
import csv
import datetime as dt
import gzip
import io
import json
import re
import subprocess
import tempfile
from pathlib import Path

STEP = re.compile(r"\[pbc\] step=(\d+) ms=([0-9.]+) fb=(\d+)")


def kaggle(*args, timeout=300):
    return subprocess.run(["kaggle", "competitions", *args], capture_output=True, text=True, timeout=timeout, check=True).stdout


CALLS = {"replay_attempts": 0}


def fetch_json(kind, episode, seat, out):
    name = out / (f"episode-{episode}-replay.json.gz" if kind == "replay" else f"episode-{episode}-agent-{seat}-logs.json.gz")
    if name.exists():
        try:
            return json.loads(gzip.decompress(name.read_bytes()))
        except (ValueError, EOFError, OSError):          # unreadable cache file: quarantine it and refetch
            name.rename(name.with_name(name.name + f".bad-{dt.datetime.now(dt.timezone.utc):%H%M%S}"))
    CALLS["replay_attempts"] += 1
    with tempfile.TemporaryDirectory() as tmp:
        if kind == "replay":
            kaggle("replay", str(episode), "-p", tmp, "-q")
        else:
            kaggle("logs", str(episode), str(seat), "-p", tmp, "-q")
        files = [p for p in Path(tmp).iterdir() if p.is_file()]
        if len(files) != 1:
            raise RuntimeError(f"{kind} {episode}: expected one file, got {[p.name for p in files]}")
        raw = files[0].read_bytes()
    obj = json.loads(raw)                                # parse first: never cache bytes that are not a replay
    tmpn = name.with_name(name.name + ".tmp"); tmpn.write_bytes(gzip.compress(raw)); tmpn.replace(name)
    return obj


def list_with_rewards(sub):
    """API listing (1 call): every episode with both seats' submission ids and rewards, so W/L/D is known without a download."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    rows = []
    for e in api.competition_list_episodes(sub):
        ag = sorted(e.agents, key=lambda x: x.index)
        mine = [i for i, x in enumerate(ag) if x.submission_id == sub]
        rw = [x.reward for x in ag]
        res = None
        if len(ag) == 2 and len(mine) == 1 and None not in rw:
            me, op = rw[mine[0]], rw[1 - mine[0]]
            res = "W" if me > op else "L" if me < op else "D"
        elif len(mine) == 2:
            res = "self"
        rows.append({"id": str(e.id), "createTime": str(e.create_time), "state": str(e.state), "type": str(e.type), "result": res or "",
                     "opp_team": (ag[1 - mine[0]].team_name if len(mine) == 1 and len(ag) == 2 else ""),
                     "opp_team_id": (str(ag[1 - mine[0]].team_id) if len(mine) == 1 and len(ag) == 2 else "")})
    return rows


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=int)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--team", default="Russell Kirk,有辣条有权,怎么做热干面?")   # renamed 2026-09-27; comma list
    ap.add_argument("--archive-sha256", default=None)
    ap.add_argument("--no-logs", action="store_true", help="skip agent-log downloads (one request per episode saved)")
    ap.add_argument("--max-new", type=int, default=None, help="download at most this many not-yet-fetched replays")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lr = list_with_rewards(a.submission)
    buf = io.StringIO(); wr = csv.DictWriter(buf, fieldnames=["id", "createTime", "state", "type", "result", "opp_team", "opp_team_id"]); wr.writeheader(); wr.writerows(lr)
    (out / "EPISODES.csv").write_text(buf.getvalue())
    rows = [r for r in lr if (r.get("state") or "").endswith("COMPLETED")]
    rec_l = {k: sum(1 for r in rows if r["result"] == k) for k in ("W", "L", "D", "self")}
    (out / "RECORD-LISTING.json").write_text(json.dumps({"submission_id": a.submission, "listed_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                          "completed": len(rows), "record": rec_l}) + "\n")
    # owner 09-30 08:4x: losses and draws first, then wins; newest first within each
    rows.sort(key=lambda r: r["createTime"], reverse=True)
    rows.sort(key=lambda r: 0 if r["result"] in ("L", "D") else 1)
    summary, lat_eps, pooled = [], [], []
    record = {"W": 0, "L": 0, "D": 0, "self_play": 0}
    new_budget = a.max_new
    failed = []
    for r in rows[: a.n]:
        eid = int(r["id"])
        if not (out / f"episode-{eid}-replay.json.gz").exists():
            if new_budget is not None and new_budget <= 0:
                continue
            if new_budget is not None:
                new_budget -= 1
        try:
            rep = fetch_json("replay", eid, None, out)
        except subprocess.CalledProcessError as ex:
            err = (ex.stderr or "") + (ex.stdout or "")
            if "429" in err or "Too Many Requests" in err:
                print(f"HTTP 429 at episode {eid}; stopping this run, unfetched episodes stay queued", flush=True)
                break                                   # rate limit: stop; nothing is marked skipped, the next run re-lists
            failed.append(eid); print(f"replay {eid} failed (non-429): {err.strip()[-160:]}", flush=True)
            continue                                    # one bad episode must not block every older one
        except (RuntimeError, ValueError, EOFError, OSError) as ex:
            failed.append(eid); print(f"replay {eid} failed: {ex}", flush=True); continue
        teams = rep.get("info", {}).get("TeamNames") or []
        ours = [i for i, t in enumerate(teams) if t in a.team.split(",")]
        rewards = rep.get("rewards") or []
        statuses = [s.get("status") for s in (rep.get("steps") or [[{}, {}]])[-1]]
        summary.append({"episode": eid, "teams": teams, "ours": ours, "reward": rewards, "status": statuses,
                        "steps": len(rep.get("steps") or [])})
        if len(ours) == 2:
            record["self_play"] += 1
        elif len(ours) == 1 and len(rewards) == 2 and None not in rewards:
            me, opp = rewards[ours[0]], rewards[1 - ours[0]]
            record["W" if me > opp else "L" if me < opp else "D"] += 1
        for seat in ours:
            if a.no_logs and not (out / f"episode-{eid}-agent-{seat}-logs.json.gz").exists():
                continue
            logs = fetch_json("logs", eid, seat, out)
            ms, fb, first = [], 0, None
            for step in logs:
                for entry in step or []:
                    for m in STEP.finditer(entry.get("stderr", "") or ""):
                        s, t, f = int(m.group(1)), float(m.group(2)), int(m.group(3))
                        fb += f
                        if s == 0:
                            first = t / 1000
                        else:
                            ms.append(t)
            pooled += ms
            opp_seat = 1 - seat
            lat_eps.append({"episode": eid, "seat": seat, "opponent": "self" if len(ours) == 2 else teams[opp_seat],
                            "cash": rewards[seat] if rewards else None, "opp_cash": rewards[opp_seat] if rewards else None,
                            "won": None if len(ours) == 2 or not rewards else rewards[seat] > rewards[opp_seat],
                            "status": statuses[seat] if seat < len(statuses) else None, "first_step_s": first,
                            "p50_ms": round(pct(ms, 0.5)) if ms else None, "p99_ms": round(pct(ms, 0.99)) if ms else None,
                            "max_s": round(max(ms) / 1000, 3) if ms else None,
                            "steps_over_1s": sum(t > 1000 for t in ms), "fb_lines": fb})
    (out / "EPISODE-SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    rep = {"schema": "pbc-hosted-latency-v1", "submission_id": a.submission, "archive_sha256": a.archive_sha256,
           "fetched": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "episodes": lat_eps,
           "pooled_after_first": {"n": len(pooled), "p50_ms": round(pct(pooled, 0.5)) if pooled else None,
                                  "p90_ms": round(pct(pooled, 0.9)) if pooled else None,
                                  "p99_ms": round(pct(pooled, 0.99)) if pooled else None,
                                  "max_s": round(max(pooled) / 1000, 3) if pooled else None},
           "record_vs_others": record}
    (out / "HOSTED-LATENCY.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(f"CALLS {3 + 2 * CALLS['replay_attempts']}", flush=True)   # import-time token + authenticate + listing; 2 per replay CLI call
    print(json.dumps({"episodes": len(summary), "failed_non429": failed, "record_vs_others": record, "pooled_after_first": rep["pooled_after_first"],
                      "fallback_lines": sum(e["fb_lines"] for e in lat_eps),
                      "steps_over_1s": sum(e["steps_over_1s"] for e in lat_eps)}))


if __name__ == "__main__":
    main()
