"""Compare the deployed greedy decoder with the live trainer's encode/decode path.

The trainer's convenience ``act`` does not yet thread v8 history through its
feature extractor, so this calls its actual ``encode`` and ``decode_free`` with
the same causal features the deployed actor receives. No teacher actions enter
either decoder; recorded observations only supply identical actor-visible states.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LIVE = Path("/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc")
for directory in (ROOT / "src", HERE):
    sys.path.insert(0, str(directory))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import agent  # noqa: E402
import data  # noqa: E402
import model as deploy_model  # noqa: E402
import v8_gaps  # noqa: E402

spec = importlib.util.spec_from_file_location("pbc_live_model", LIVE / "model.py")
assert spec and spec.loader
live_model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = live_model
spec.loader.exec_module(live_model)

# Frozen before the league's rolling replay-retention cleanup.
REPLAYS = ((HERE / "fixtures/1528.json.gz", 1528, 1),
           (HERE / "fixtures/1529.json.gz", 1529, 0),
           (HERE / "fixtures/1530.json.gz", 1530, 1))


def _action(net, clean, cfg, rows, mask, known, pos, extras, hist):
    prev_unit = prev_order = None
    if net.config.v6_prev:
        prev_unit, prev_order = deploy_model.v6_history_arrays(hist, int(clean["step"]), net.U, net.O)
    ctx = net.encode(rows, mask, known, None, code_mode=net.config.infer_code_mode,
                     extra=extras[0], unit_extra=extras[1], reach_eta=extras[2],
                     prev_unit=prev_unit, prev_order=prev_order,
                     **net._last_job_eta, **net._last_v8)
    labels = net.decode_free(ctx, pos, temperature=0.0,
                             prefix_fn=net._prefix_fn(clean, cfg))
    lab = {key: [int(x) for x in labels[key][0].tolist()]
           for key in ("unit_tok", "unit_qty")}
    ot, oq = labels["order_tok"][0].tolist(), labels["order_qty"][0].tolist()
    n = next((i + 1 for i, tok in enumerate(ot) if tok == net.config.stop_index), len(ot))
    lab["order_tok"], lab["order_qty"] = [int(x) for x in ot[:n]], [int(x) for x in oq[:n]]
    return net.codec.decode_step(lab, clean), labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=719)
    parser.add_argument("--candidate-archive", type=Path,
                        help="compare checkpoint fp32 decisions against exported archive fp16 weights")
    args = parser.parse_args()
    torch.set_num_threads(1)
    deploy = agent.load_checkpoint(args.checkpoint).eval()
    if args.candidate_archive:
        with tarfile.open(args.candidate_archive, "r:gz") as archive:
            member = archive.extractfile("policy.pt")
            if member is None:
                raise ValueError("archive has no policy.pt")
            blob = torch.load(io.BytesIO(member.read()), map_location="cpu", weights_only=True)
        live = agent.policy_from_weights(blob)
    else:
        live = live_model.PolicyNet(live_model.ModelConfig(**dataclasses.asdict(deploy.config))).eval()
        live.load_state_dict(deploy.state_dict(), strict=True)
    checked = []
    with torch.inference_mode():
        for path, episode, seat in REPLAYS:
            replay = json.loads(gzip.decompress(path.read_bytes()))
            cfg = agent.public_configuration(replay["configuration"])
            history, hist = [], ()
            long_state = v8_gaps.GapsState()
            n = 0
            nonpass_units = market_orders = mismatches = 0
            first_mismatches = []
            for t, frame in enumerate(replay["steps"]):
                if t >= args.max_steps:
                    break
                clean = agent.actor_view(frame[seat]["observation"])
                if clean.get("step") is None:
                    clean["step"] = int(clean["day"]) * 24 + int(clean["hour"])
                rows, mask, known, pos = deploy.observation_tensors(
                    clean, cfg, v8_history=history, v8_long=long_state)
                live._last_extra = deploy._last_extra
                live._last_job_eta = deploy._last_job_eta
                live._last_v8 = deploy._last_v8
                extras = deploy._last_extra or (None, None, None)
                a, al = _action(deploy, clean, cfg, rows, mask, known, pos, extras, hist)
                b, bl = _action(live, clean, cfg, rows, mask, known, pos, extras, hist)
                different = [key for key in al if not torch.equal(al[key], bl[key])]
                if a != b:
                    different.append("wire_action")
                if different:
                    if not args.candidate_archive:
                        raise AssertionError(f"{episode} seat{seat} t{t}: {different} differ")
                    mismatches += 1
                    if len(first_mismatches) < 5:
                        first_mismatches.append({"step": t, "fields": different, "fp32": a, "archive": b})
                nonpass_units += sum(bool(cmd and cmd[0] != "PASS")
                                     for cmd in [a.get("farmer", ["PASS"]), *(a.get("hands") or [])])
                market_orders += len(a.get("market") or [])
                saved = data.encode_step_lenient(a, clean)
                hist = (hist + ((int(clean["step"]), saved["unit_tok"], saved["unit_qty"],
                                 saved["order_tok"], saved["order_qty"]),))[-2:]
                history = [clean, *history[:1]]
                n += 1
            checked.append({"episode_id": episode, "seat": seat, "decisions": n,
                            "nonpass_units": nonpass_units, "market_orders": market_orders,
                            "mismatched_decisions": mismatches, "first_mismatches": first_mismatches,
                            "all_greedy_labels_equal": mismatches == 0})
            print(json.dumps(checked[-1]), flush=True)
    Path(args.out).write_text(json.dumps({"schema": "v8-trainer-deploy-greedy-parity-v1",
                                         "checkpoint": args.checkpoint,
                                         "candidate_archive": str(args.candidate_archive) if args.candidate_archive else None,
                                         "results": checked}, indent=2) + "\n")
    if any(r["mismatched_decisions"] for r in checked):
        raise SystemExit("archive storage changed one or more greedy decisions (receipt written)")


if __name__ == "__main__":
    main()
