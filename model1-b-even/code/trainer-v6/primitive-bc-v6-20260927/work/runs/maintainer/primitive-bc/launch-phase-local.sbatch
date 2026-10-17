#!/bin/bash
# One continuation phase trained entirely from the node's LOCAL disk (owner 2026-09-26: "the training files must be on a
# drive local to it"). At start: rsync the runtime, this replica (without results/) and every file in FILES_LIST from the
# NFS scratch to $LOCAL_ROOT/fs/<same relative path>, rewrite /mnt/ai/scratch/farmNEAT paths in the LOCAL copies only, train
# from there; checkpoints are written locally and copied back to NFS_OUT every 10 minutes and at exit.
#
#   RUN_ROOT     replica on NFS (/mnt/ai/scratch/farmNEAT/...)
#   FILES_LIST   NFS file listing every other input (archives, priority/force files, seed checkpoint and its LATEST, leaderboard)
#   OUT          NFS results directory (receives the copies); training writes to the same path under $LOCAL_ROOT/fs
#   SEED_FROM    NFS seed directory (its LATEST and checkpoint must be in FILES_LIST); used when OUT has no LATEST
#   PHASE_END    absolute update at which the job stops with a checkpoint
#   ARCHIVES_FILE NFS file with the training archive list (in FILES_LIST); TRAIN_ARGS extra train.py fit arguments
#   CPU_LIST     optional taskset CPU list for training and its data workers (e.g. 0-7,10-15)
#   LOCAL_ROOT   default $HOME/farmneat-local; refused if it is a network filesystem
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=farmneat-pbc-local
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CUDA_VISIBLE_DEVICES:?}" "${RUN_ROOT:?}" "${OUT:?}" "${PHASE_END:?}" "${FILES_LIST:?}" "${ARCHIVES_FILE:?}"
NFS=/mnt/ai/scratch/farmNEAT
case "$RUN_ROOT$OUT$FILES_LIST$ARCHIVES_FILE" in *"$NFS"*) ;; *) echo "paths must be under $NFS" >&2; exit 2;; esac
LOCAL=${LOCAL_ROOT:-$HOME/farmneat-local}
mkdir -p "$LOCAL/fs"
fstype=$(stat -f -c %T "$LOCAL")
case "$fstype" in nfs*|cifs|smb*|fuse*|9p|ceph|lustre|gpfs) echo "LOCAL_ROOT $LOCAL is $fstype, not a local disk" >&2; exit 2;; esac
need=$(( 40 * 1024 * 1024 ))   # KiB: runtime ~7 GB + data ~25 GB + checkpoints
free=$(df -Pk "$LOCAL" | awk 'NR==2 {print $4}')
if [ "$free" -lt "$need" ]; then echo "only ${free} KiB free on $LOCAL" >&2; exit 2; fi
loc() { printf '%s\n' "$LOCAL/fs/${1#$NFS/}"; }
t0=$(date +%s)
rsync -a "$NFS/runtime/" "$LOCAL/fs/runtime/"
rsync -a --delete --exclude 'work/runs/maintainer/primitive-bc/results/' --exclude 'slurm-*.out' --exclude 'work/venv/' \
  "$RUN_ROOT/" "$(loc "$RUN_ROOT")/"
sed "s#^$NFS/##" "$FILES_LIST" | grep -v '^[[:space:]]*$' > "$LOCAL/files-$SLURM_JOB_ID.txt"
rsync -a --files-from="$LOCAL/files-$SLURM_JOB_ID.txt" "$NFS/" "$LOCAL/fs/"
run_root=$(loc "$RUN_ROOT")
here="$run_root/work/runs/maintainer/primitive-bc"
mkdir -p "$run_root/work/venv"
ln -sfn "$LOCAL/fs/runtime/python/bin/python3.12" "$run_root/work/venv/python"
# rewrite NFS paths in the local copies only (code constants, archive/priority lists)
grep -rl "$NFS" "$here" --include=*.py --include=*.txt --include=*.tsv 2>/dev/null | xargs -r sed -i "s#$NFS/#$LOCAL/fs/#g"
while read -r f; do case "$f" in *.txt|*.tsv) sed -i "s#$NFS/#$LOCAL/fs/#g" "$LOCAL/fs/$f";; esac; done < "$LOCAL/files-$SLURM_JOB_ID.txt"
echo "{\"event\": \"staged\", \"local\": \"$LOCAL\", \"fs\": \"$fstype\", \"seconds\": $(( $(date +%s) - t0 ))}"

out=$(loc "$OUT")
mkdir -p "$out" "$OUT"
if [ ! -f "$out/LATEST" ] && [ -f "$OUT/LATEST" ]; then rsync -a "$OUT/" "$out/"; fi       # resume what NFS already has
if [ ! -f "$out/LATEST" ]; then
  : "${SEED_FROM:?OUT has no LATEST and SEED_FROM is unset}"
  seed_dir=$(loc "$SEED_FROM")
  seed=$(tr -d '[:space:]' < "$seed_dir/LATEST")
  cp "$seed_dir/$seed" "$out/$seed.tmp" && mv "$out/$seed.tmp" "$out/$seed"
  printf '%s\n' "$seed" > "$out/LATEST"
  echo "{\"event\": \"seeded\", \"from\": \"$seed_dir/$seed\"}"
fi
latest=$(tr -d '[:space:]' < "$out/LATEST"); current=${latest#ckpt-}; current=$((10#${current%.pt}))
remaining=$((PHASE_END - current))
echo "{\"event\": \"phase\", \"out\": \"$out\", \"current_update\": $current, \"phase_end\": $PHASE_END, \"stop_after\": $remaining}"
[ "$remaining" -le 0 ] && { rsync -rlt "$out/" "$OUT/" || true; exit 0; }

sync_back() { rsync -rlt --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r "$out/" "$OUT/" || true; }   # the NAS refuses chgrp: no -a
( while sleep 600; do sync_back; done ) &
syncer=$!
trap 'kill $syncer 2>/dev/null || true; sync_back' EXIT

archives=$(grep -v '^[[:space:]]*$' "$(loc "$ARCHIVES_FILE")" | tr '\n' ' ')
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$run_root/src:$here:$LOCAL/fs/runtime/site"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$run_root"
pin=()
[ -n "${CPU_LIST:-}" ] && pin=(taskset -c "$CPU_LIST")
# shellcheck disable=SC2086
"${pin[@]}" "$run_root/work/venv/python" -B "$here/train.py" fit --out "$out" --device cuda --resume --stop-after "$remaining" \
  --archives $archives --workers "${WORKERS:-12}" ${TRAIN_ARGS:-}
