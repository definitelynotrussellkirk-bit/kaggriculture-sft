"""Read-only chronological audit of pre- versus post-rule action history.

This uses exported archive bytes, real recorded observations, and the deploy
agent/rules. It does not run the game or change an archive.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import io
import json
import tarfile
from pathlib import Path

import torch

import agent
import data


HERE = Path(__file__).resolve().parent
REPLAYS = ((HERE / "fixtures/1528.json.gz", 1528, 1),
           (HERE / "fixtures/1529.json.gz", 1529, 0),
           (HERE / "fixtures/1530.json.gz", 1530, 1))


def labels_equal(entry, labels):
    return all(list(entry[i]) == [int(x) for x in labels[k]] for i, k in enumerate(
        ("unit_tok", "unit_qty", "order_tok", "order_qty"), start=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with tarfile.open(args.archive, "r:gz") as archive:
        member = archive.extractfile("policy.pt")
        if member is None:
            raise ValueError("archive has no policy.pt")
        blob = torch.load(io.BytesIO(member.read()), map_location="cpu", weights_only=True)
    policy = agent.policy_from_weights(blob)
    actor = agent.Agent("preloaded", policy=policy, temperature=0.0, threads=1, log=False)
    original_act = actor._act
    raw = [None]

    def capture(*a, **kw):
        outcome = original_act(*a, **kw)
        raw[0] = copy.deepcopy(outcome[0])
        return outcome

    actor._act = capture
    results = []
    for path, episode, seat in REPLAYS[:args.seats]:
        replay = json.loads(gzip.decompress(path.read_bytes()))
        edited = 0
        history_wrong = 0
        first = []
        for t, frame in enumerate(replay["steps"]):
            if t >= 719:
                break
            obs = frame[seat]["observation"]
            raw[0] = None
            post = actor(obs, replay["configuration"])
            if actor.last_fallback:
                raise RuntimeError(f"{episode} seat{seat} step{t}: fallback {actor.last_error}")
            if raw[0] is None:
                raise RuntimeError("actor did not produce a pre-rule action")
            if raw[0] == post:
                continue
            edited += 1
            clean = agent.actor_view(obs)
            labels = data.encode_step_lenient(post, clean)
            state = actor._states[seat]["state"]
            wrong = not labels_equal(state.hist[-1], labels)
            history_wrong += wrong
            if len(first) < 5:
                first.append({"step": t, "wrong_hist": wrong,
                              "changed_fields": [k for k in ("farmer", "hands", "market")
                                                 if raw[0].get(k) != post.get(k)],
                              "before": raw[0], "after": post})
        rec = {"episode_id": episode, "seat": seat, "decisions": 719,
               "post_rule_edits": edited, "edited_steps_with_wrong_hist": history_wrong,
               "first_edits": first}
        results.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "first_edits"}), flush=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps({"schema": "pbc-post-rule-history-audit-v1", "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
