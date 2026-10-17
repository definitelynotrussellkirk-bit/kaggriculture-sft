#!/bin/bash
#SBATCH -p cpu,gpu --exclude=gpu-skynet,gpu-spark,laptop-4060,gpu-3090,cpu-hub -c 2 --mem=6G -t 00:50:00 -J nflow-rel
#SBATCH -o /mnt/ai/scratch/farmNEAT/n-flow/logs/release-run-%j.out
# Release one N flow top-up run (factory / anneal / t1): EFFECTIVE receipt binding the exact final checkpoint, then the
# fail-closed P2 release chain (p2_nflow.sh: export -> archive -> cold verify -> parity -> p2_check -> finalize -> gate;
# RELEASE-DONE.json only on an all-pass LINKS receipt).
#   sbatch release_run.sbatch RUN_DIR NAME ROSTER(NAS copy of the trained roster) SEED_CKPT CMD_FILE [DEPLOY_ROOT]
set -euo pipefail
O=$1; NAME=$2; ROSTER=$3; SEED=$4; CMDF=$5; DROOT=${6:-/mnt/ai/scratch/farmNEAT/pbc-deploy}
CK=$O/$(cat $O/LATEST)
python3 - "$O" "$CK" "$ROSTER" "$SEED" "$CMDF" <<'PY'
import glob, hashlib, json, sys
o, ck, roster, seed, cmdf = sys.argv[1:6]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
run = json.load(open(sorted(glob.glob(f"{o}/RUN-*.json"))[-1]))
if run.get("roster_sha256") != sha(roster):
    sys.exit(f"REFUSED: RUN roster_sha256 {run.get('roster_sha256')} != NAS roster copy {sha(roster)}")
json.dump({"cmd": open(cmdf).read().splitlines(), "run": run, "roster": roster, "roster_sha256": run["roster_sha256"],
           "roster_receipt": json.load(open(roster + ".receipt.json")),
           "seed_checkpoint": seed, "seed_checkpoint_sha256": sha(seed),
           "final_checkpoint": ck, "final_checkpoint_sha256": sha(ck)},
          open(f"{o}/EFFECTIVE-t4.json", "w"), indent=1)
print("effective written")
PY
NFLOW_EFFECTIVE=$O/EFFECTIVE-t4.json bash /mnt/ai/scratch/farmNEAT/n-flow/code/p2_nflow.sh "$CK" "$DROOT" "$NAME" "$NAME"
