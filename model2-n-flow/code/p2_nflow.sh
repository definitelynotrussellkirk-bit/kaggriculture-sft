#!/bin/bash
# N flow release chain for one pulled Colab checkpoint. NFS paths only, so it runs on a cluster node, e.g.
#   srun --overlap --jobid <JOB on a quiet node> bash /mnt/ai/scratch/farmNEAT/n-flow/code/p2_nflow.sh [CKPT] [DEPLOY_ROOT] [NAME_PREFIX]
# default CKPT = newest pulled colab-nflow checkpoint, ROOT = pbc-deploy, prefix nflow; 4th arg = explicit name. Steps (fail-closed):
# 1. nflow_deploy.sbatch (bash): EMA export (nonfinite refused) -> fp16 archive -> cold verify both seats vs PASS
#    (every attempt logged with host + load in VERIFY-ATTEMPTS.txt) -> HUB-DONE.json
# 2. serve/train parity of that exact ema.pt on a real replay seat, 30 steps across a dawn (PARITY.json)
# 3. p2_check.py: EFFECTIVE receipt binds the exact checkpoint; archive policy.pt == fp16(checkpoint EMA)
# 4. finalize_snapshot.py, then release_gate.py; only an all-pass LINKS receipt publishes RELEASE-DONE.json
set -euo pipefail
D=/mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/colab-nflow
CODE=/mnt/ai/scratch/farmNEAT/n-flow/code
RS=${NFLOW_RS:-/mnt/ai/scratch/farmNEAT/n-flow/repo-snap}; DEP=$RS/work/runs/maintainer/n-flow-deploy
PY=/mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12; SITE=/mnt/ai/scratch/farmNEAT/runtime/site
CK=${1:-$(ls $D/ckpt-*.pt | sort | tail -1)}
ROOT=${2:-/mnt/ai/scratch/farmNEAT/pbc-deploy}
U=$((10#$(basename $CK .pt | sed 's/ckpt-//')))
# pull.sh prunes old pulled checkpoints on the NAS: release from a stable copy (HUB-DONE / BUILD name this path)
KEEP=/mnt/ai/scratch/farmNEAT/n-flow/release-ckpts${3:+/$3}
SOURCE_CK=$CK
CK=$($PY -B $CODE/pin_checkpoint.py "$SOURCE_CK" "$KEEP")  # identical existing bytes are reusable; collisions fail
NAME=${4:-${3:-nflow}-u$U}     # 4th arg: explicit archive name
OUT=$ROOT/$NAME
echo "$(date -Is) $NAME from $CK -> $OUT on $(hostname) load $(cut -d' ' -f1-3 /proc/loadavg)"
if [ -s "$OUT/RELEASE-DONE.json" ] && $PY -B $CODE/release_gate.py "$OUT" "$NAME"; then
  echo "$(date -Is) already complete $NAME"
  exit 0
fi
if [ -e "$OUT" ] && [ ! -s "$OUT/RELEASE-DONE.json" ]; then  # HUB-DONE is only an intermediate receipt
  mv "$OUT" "$OUT.failed-$(date +%H%M%S)"; echo "moved aside an incomplete release attempt of $NAME"
fi
export CUDA_VISIBLE_DEVICES=""
cd /tmp
NFLOW_VERIFY_NICE=0 NFLOW_DEPLOY_ROOT=$ROOT bash $CODE/nflow_deploy.sbatch "$CK" "$NAME"
SOURCE_EFFECTIVE=${NFLOW_EFFECTIVE:-$D/SPEC.json}
BOUND_EFFECTIVE=$OUT/EFFECTIVE-RELEASE.json
$PY -B $CODE/bind_effective.py "$SOURCE_EFFECTIVE" "$CK" "$BOUND_EFFECTIVE"
cd $DEP
PYTHONPATH=$SITE $PY -B test_nflow_parity.py --steps 30 --ckpt "$OUT/ema.pt" | tee "$OUT/PARITY.txt"
tail -2 "$OUT/PARITY.txt" | head -1 > "$OUT/PARITY.json"
grep -qx PASS "$OUT/PARITY.txt" || { echo "PARITY FAILED"; exit 3; }
PYTHONPATH=$SITE $PY -B $CODE/p2_check.py "$OUT/$NAME.tar.gz" "$CK" "$BOUND_EFFECTIVE" "$OUT/P2-CHECK.json"
PYTHONPATH=$SITE $PY -B /mnt/ai/scratch/farmNEAT/freshen/guard/finalize_snapshot.py "$NAME" "$CK" "$OUT" "$BOUND_EFFECTIVE" "$OUT"
$PY -B $CODE/release_gate.py "$OUT" "$NAME"
cat "$OUT/HUB-DONE.json"; echo
cat "$OUT/VERIFY-ATTEMPTS.txt"
echo "$(date -Is) done $NAME"
