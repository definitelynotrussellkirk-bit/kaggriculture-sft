#!/usr/bin/env python3
"""Play ONE league episode (one world seed, one seat order) and record it as a Kaggle-format replay.

The game itself is pbc-ladder/ladder.py's ``play()`` (cold archive hosts, exact engine), called unchanged from the
engine snapshot; this runner only wraps ExactKaggricultureEnvironment.step to record the joint actions, then rebuilds
the Kaggle replay from (world seed, actions) with the same exact engine and checks that the rebuilt final cash equals
the cash the game reported. Run it with the hosted-like runtime (py3.11 + torch cpu):

  PY311 -B league_game.py --engine SNAP --code SNAP/code/primitive-bc --seat0 A.tar.gz --seat1 B.tar.gz \
      --labels nameA nameB --world-seed S --episode-id E --template replay-template.json --out DIR [--crn-rng]

--crn-rng plays the game (and rebuilds its replay) with the opt-in CRN dusk of the snapshot's
src/farm_neat/rollout/crn_rng.py (configuration crnRng: true): each tile's weed draw keyed by (seed, day, farm, x, y)
and each shop by (seed, day), so both seat orders of a match see the same shops and weeds. The replay's configuration
then carries "crnRng": true and info.LeagueEpisode.crn_rng = true; re-simulate it with crn_rng.environment().

Writes DIR/game.json (the ladder's game record + league metadata), DIR/replay.json.gz (Kaggle episode JSON) and prints
one JSON line {"ok": true/false, ...} on stdout. Exit 0 when the game ran (whatever its result), 3 on a runner error.
"""
import argparse
import gzip
import json
import os
import platform
import sys
import time
import traceback
import uuid
from pathlib import Path

SCHEMA = "league-episode-v1"


def _write_json(path, value, *, gz=False):
    tmp = Path(str(path) + ".tmp")
    if gz:
        with gzip.open(tmp, "wt", compresslevel=6) as handle:
            json.dump(value, handle, separators=(",", ":"))
    else:
        tmp.write_text(json.dumps(value, indent=1) + "\n")
    os.replace(tmp, path)


def _crn_module():
    """crn_rng from the engine snapshot (ladder._setup put its src first on sys.path)."""
    try:
        from farm_neat.rollout import crn_rng
    except ImportError as error:
        raise RuntimeError("--crn-rng needs an engine snapshot with src/farm_neat/rollout/crn_rng.py") from error
    return crn_rng


def build_replay(ee, template, *, world_seed, actions, names, episode_id, statuses=None, extra=None, crn=False):
    """Kaggle episode JSON from the world seed and the recorded joint actions (the reconstruct_episode layout of
    pbc-seed-action-replays/seed_action_replays.py: 1 + len(actions) boundaries, step 0 action = PASS, no
    remainingOverageTime). -> (replay dict, final cash per seat, engine rewards per seat). crn: rebuild with the CRN
    dusk and write "crnRng": true into the replay configuration."""
    if crn:
        env = _crn_module().environment(seed=int(world_seed), configuration={"crnRng": True})
    else:
        env = ee.ExactKaggricultureEnvironment(seed=int(world_seed))
    steps = []
    n = len(actions)
    for t in range(n + 1):
        boundary = []
        for seat in (0, 1):
            state = env.state[seat]
            boundary.append({"action": ee._copy_wire(ee.PASS_ACTION if t == 0 else actions[t - 1][seat]),
                             "info": ee._copy_wire(dict(state.info)),
                             "observation": ee._copy_wire(dict(state.observation)),
                             "reward": float(state.reward) if state.reward is not None else None,
                             "status": str(state.status)})
        steps.append(boundary)
        if t < n:
            env.step(actions[t])
    farms = env.state[0].observation.farms
    cash = [float(farms[s]["money"]) for s in (0, 1)]
    rewards = [float(s.reward) if s.reward is not None else None for s in env.state]
    final_status = [str(s.status) for s in env.state]
    if statuses and any(x != "DONE" for x in statuses):       # hosted failure: the failing seat is not DONE
        final_status = list(statuses)
        for seat in (0, 1):
            if statuses[seat] != "DONE":
                steps[-1][seat]["status"] = statuses[seat]
                steps[-1][seat]["reward"] = None
                rewards[seat] = None
    out = {k: json.loads(json.dumps(v)) for k, v in template.items() if not k.startswith("_")}
    if crn:
        out["configuration"] = {**out.get("configuration", {}), "crnRng": True}
    out["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"kaggriculture/farmneat-league/{episode_id}"))
    out["info"] = {"Agents": [{"Name": name, "ThumbnailUrl": None} for name in names], "EpisodeId": int(episode_id),
                   "LiveVideoPath": None, "TeamNames": list(names), "seed": int(world_seed)}
    if extra:
        out["info"]["LeagueEpisode"] = extra
    out["rewards"] = rewards
    out["statuses"] = final_status
    out["steps"] = steps
    return out, cash, rewards


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", required=True, help="engine snapshot root (repository layout)")
    ap.add_argument("--code", required=True, help="ladder --code directory (build_archive.py cold host)")
    ap.add_argument("--seat0", required=True)
    ap.add_argument("--seat1", required=True)
    ap.add_argument("--labels", nargs=2, required=True, help="entry names of seat 0 and seat 1")
    ap.add_argument("--world-seed", type=int, required=True)
    ap.add_argument("--sampler-seed", type=int, default=7100)
    ap.add_argument("--steps", type=int, default=719)
    ap.add_argument("--temperature", type=float, default=1.0, help="checkpoint players only; archives use MANIFEST")
    ap.add_argument("--episode-id", type=int, required=True)
    ap.add_argument("--template", required=True, help="Kaggle replay header template (replay-template.json)")
    ap.add_argument("--replay-names", nargs=2, help="TeamNames written into the replay (default SELF:<label>)")
    ap.add_argument("--hosted", action="store_true", help="ladder --hosted rules (bank, wall limit, errors end the game)")
    ap.add_argument("--crn-rng", action="store_true",
                    help="opt-in CRN dusk (configuration crnRng): shops keyed by (seed, day), weeds by (seed, day, farm, x, y)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    try:
        engine = Path(a.engine).resolve()
        sys.path.insert(0, str(engine / "work/runs/maintainer/pbc-ladder"))
        import ladder  # noqa: E402  (the snapshot's ladder.py; ROOT = the snapshot)
        ladder._setup(a.code)
        from farm_neat.game import exact_engine as ee
        recorded = []
        original_step = ee.ExactKaggricultureEnvironment.step
        original_init = ee.ExactKaggricultureEnvironment.__init__

        def recording_step(self, joint_action):
            acts = list(joint_action)
            recorded.append([ee._copy_wire(x) for x in acts])
            return original_step(self, acts)

        if a.crn_rng:
            crn_rng = _crn_module()

            def crn_init(self, *, seed, episode_steps=720, configuration=None):
                # ladder.play() builds ExactKaggricultureEnvironment(seed=...): make that game a CRN game
                original_init(self, seed=seed, episode_steps=episode_steps,
                              configuration=crn_rng.with_flag(configuration))
                crn_rng.set_crn(self, True)

            ee.ExactKaggricultureEnvironment.__init__ = crn_init
        ee.ExactKaggricultureEnvironment.step = recording_step
        hosted = dict(ladder.HOSTED_DEFAULTS) if a.hosted else None
        spec = {"code": a.code, "seats": [f"archive:{a.seat0}", f"archive:{a.seat1}"], "labels": list(a.labels),
                "world_seed": a.world_seed, "sampler_seed": a.sampler_seed, "temperature": a.temperature,
                "steps": a.steps, "preload": True, "hosted": hosted, "opts": {}}
        try:
            game = ladder.play(spec)
        finally:
            ee.ExactKaggricultureEnvironment.step = original_step
            ee.ExactKaggricultureEnvironment.__init__ = original_init
        game.pop("sources", None)
        template = json.loads(Path(a.template).read_text())
        names = a.replay_names or [f"SELF:{x}" for x in a.labels]
        import torch
        runtime = {"python": platform.python_version(), "executable": sys.executable, "torch": torch.__version__,
                   "host": platform.node(), "hosted_rules": bool(a.hosted)}
        if a.crn_rng:
            runtime["crn_rng"] = True
        extra = {"schema": SCHEMA, "labels": list(a.labels), "world_seed": a.world_seed,
                 "sampler_seed": a.sampler_seed, "runtime": runtime}
        if a.crn_rng:
            extra["crn_rng"] = True
        replay, cash, rewards = build_replay(ee, template, world_seed=a.world_seed, actions=recorded, names=names,
                                             episode_id=a.episode_id, statuses=game.get("status"), extra=extra,
                                             crn=a.crn_rng)
        verified = cash == [float(x) for x in game["cash"]]
        _write_json(out / "replay.json.gz", replay, gz=True)
        record = {"schema": SCHEMA, "episode_id": a.episode_id, "labels": list(a.labels),
                  "archives": [a.seat0, a.seat1], "world_seed": a.world_seed, "sampler_seed": a.sampler_seed,
                  "game": game, "actions_recorded": len(recorded), "replay": "replay.json.gz",
                  "replay_cash": cash, "replay_rewards": rewards, "replay_verified": verified,
                  "runtime": runtime, "ladder_sha256": ladder.LADDER_SHA256, "wall_seconds": round(time.time() - began, 1)}
        if a.crn_rng:
            record["crn_rng"] = True
        _write_json(out / "game.json", record)
        print(json.dumps({"ok": True, "episode_id": a.episode_id, "cash": game["cash"], "score": game["score"],
                          "status": game["status"], "fallbacks": game["fallbacks"], "replay_verified": verified,
                          "seconds": game["seconds"]}), flush=True)
        return 0
    except Exception as error:  # noqa: BLE001 - report the runner failure to the worker, which records it
        tb = traceback.format_exc()
        (out / "runner-error.txt").write_text(tb)
        print(json.dumps({"ok": False, "episode_id": a.episode_id, "error": f"{type(error).__name__}: {error}"[:500]}),
              flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
