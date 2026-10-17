"""Turn one completed base-v1 chunk checkpoint into a cold-verified league entry.

The running training replica is read-only. Pass --submit only after choosing the
intended checkpoint/EMA and a unique league name. Failed verification never submits.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LIVE = Path("/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc")
LEAGUE = ROOT / "work/runs/maintainer/pbc-league-service/league_client.py"
RUNTIME = ROOT / "src/farm_neat/tooling/runtime.py"
HOSTED = Path("/mnt/ai/scratch/farmNEAT/runtime-hosted-py311/venv/bin/python")


def run(args: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def checkpoint_run(args: list[str]) -> None:
    """Select the registered training interpreter for every checkpoint read/export."""
    run([sys.executable, "-B", str(RUNTIME), "run", "--", *args])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True, help="unique league/archive name")
    parser.add_argument("--variant", choices=("trunk", "ema"), required=True)
    parser.add_argument("--out-root", type=Path, default=Path("/mnt/ai/scratch/farmNEAT/pbc-deploy"))
    parser.add_argument("--submit", action="store_true", help="register only after both seats cold-verify")
    parser.add_argument("--note", default="")
    parser.add_argument("--decode", default="greedy",
                        help='v9 step decoding recorded in MANIFEST: greedy or JSON, e.g. {"mode":"bestk","k":32,"first_n":10}')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    if not checkpoint.name.startswith("ckpt-") or checkpoint.suffix != ".pt":
        parser.error("source must be a completed ckpt-*.pt file, not LATEST or a training directory")
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.name):
        parser.error("name must contain only ASCII letters, digits, '-' or '_'")
    target = args.out_root / args.name
    if target.exists():
        parser.error(f"refusing to overwrite {target}")
    target.mkdir(parents=True)
    source = checkpoint
    if args.variant == "ema":
        source = target / "ema.pt"
        checkpoint_run([str(LIVE / "ema_export.py"), str(checkpoint), str(source)])
    archive = target / f"{args.name}.tar.gz"
    checkpoint_run([str(HERE / "build_archive.py"), "build", "--model", str(source),
                    "--out", str(archive), "--storage", "fp16", "--temperature", "0", "--no-verify",
                    "--decode", args.decode])
    receipt = target / "VERIFY.json"
    run([str(HOSTED), "-B", str(HERE / "build_archive.py"), "verify", "--archive", str(archive),
         "--opponents", "PASS", "--seats", "0,1", "--out", str(receipt)])
    verified = json.loads(receipt.read_text())
    if not verified.get("complete") or verified.get("fallbacks") != 0:
        raise SystemExit("cold verification incomplete or had fallbacks; not submitted")
    if args.submit:
        run([sys.executable, "-B", str(LEAGUE), "submit", args.name, str(archive), "--note",
             args.note or f"base-v1 {args.variant}; {checkpoint.parent.name} {checkpoint.name}; decode {args.decode}; cold both seats"])
    print(json.dumps({"archive": str(archive), "verify": str(receipt), "submitted": args.submit}), flush=True)


if __name__ == "__main__":
    main()
