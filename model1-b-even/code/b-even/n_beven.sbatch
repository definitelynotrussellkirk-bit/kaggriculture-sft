#!/bin/bash
# N = B-even (owner 09-29 22:20/22:23): B u31047 EMA with the encoder heads regrouped to $HEADS (same tensors).
#  phase 1 (N u0-1000): oa3 runner, 0.5 BC + 0.5 KD from B31 EMA (T 1.5, --unit-mask), frozen-final tiers + MMPQ step weights,
#                       lr 3e-5 flat after 100 warmup, fresh AdamW, effective batch 512 steps; saves every 500.
#  phase 2 (N u1000-):  B's trainer (chunk-10 recipe, RUN-10 PBC env) at 3e-5; weighted rosters use base-v1-pack-bfx + MMPQ,
#                       bucket rosters use base-v1-pack with no per-seat table. B31 Muon state; trainer u = N u + 30000; ckpt every 500.
# Resumable: phase 1 restarts from its newest u*.pt (--state), phase 2 from OUT/LATEST. Stop 19:40 EDT (was 12:55).
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=15:00:00
#SBATCH --job-name=n-beven
#SBATCH --no-requeue
#SBATCH --output=/mnt/ai/scratch/farmNEAT/b-even/logs/train-%j.out
set -euo pipefail
: "${HEADS:?}"
BE=/mnt/ai/scratch/farmNEAT/b-even; O=/mnt/ai/scratch/farmNEAT/oa3-astra; T=$O/tools-cont; NFS=/mnt/ai/scratch/farmNEAT
INIT=$BE/beven-h$HEADS-init.pt; TEACHER=$NFS/base-v1/runs/b/chunks/chunk9-ckpt-0031047.pt; TSHA=08a058abd9b807e02be6505dab2f42277222020a3cea9ea489d153385f1117a9
SW=$O/cont-step-w.94bc8331b271.tsv; [ "${ROWS+set}" = set ] && [ -z "$ROWS" ] && { echo "ROWS given but empty (roster build failed); refusing the weighted fallback" >&2; exit 2; }; ROWS=${ROWS:-$NFS/base-v1-oa3plan/rows-b-final-clean.63e57662a18f.tsv}; P1=$BE/p1; OUT=$BE/run
case $(basename $ROWS) in rows-buckets.*) BUCKETS=1;; *) BUCKETS=0;; esac   # owner 02:5x bucket sampler: no per-seat weights
test "$(sha256sum $TEACHER | cut -d' ' -f1)" = $TSHA; [ -s $INIT ]
test "$(sha256sum $INIT | cut -d' ' -f1)" = "${INIT_SHA:?}"
RSH=$(basename $ROWS .tsv); RSH=${RSH##*.}; test "$(sha256sum $ROWS | cut -d' ' -f1 | cut -c1-12)" = "$RSH"
# no compact-tape members anywhere in the consumed rosters (seed-only tape loader, PRO OA3-DIAGNOSE)
# ASTRA TRAINING-RUNS-R2: awk alone decides (no pipe: a large match list made grep -q SIGPIPE awk and pipefail read it as "no tape rows")
for r in $ROWS $O/frozen-final/train.tsv; do [ -s "$r" ] || { echo "roster missing: $r" >&2; exit 2; }
  if awk -F'\t' 'NR>1 && $2=="tape"{f=1; exit} END{exit !f}' "$r"; then echo "tape rows in $r" >&2; exit 2; fi; done
python3 - $BE/EFFECTIVE-$SLURM_JOB_ID.json <<PYE
import hashlib, json, sys
h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
BTR = "$NFS/base-v1-pack/nfs/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc"; BFX = "$NFS/base-v1-pack-bfx/nfs/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc"
json.dump({"model": "N = B-even (B u31047 EMA, encoder heads 36 -> $HEADS)", "init": "$INIT", "init_sha256": h("$INIT"), "teacher": "$TEACHER", "teacher_sha256": "$TSHA",
  "phase1": {"runner": "$T/oa3/cache_train.py", "runner_sha256": h("$T/oa3/cache_train.py"), "trainer_model_sha256": h(BTR + "/model.py"), "frozen_train_sha256": h("$O/frozen-final/train.tsv"),
             "step_w": "$SW", "step_w_sha256": h("$SW"), "loss": "0.5 BC + 0.5 KD(B31 EMA, T 1.5), unit mask", "lr": "3e-5 flat (100 warmup)", "updates": 1000},
  "phase2": {"trainer_train_sha256": h(BFX + "/train.py"), "trainer_model_sha256": h(BFX + "/model.py"), "roster": "$ROWS", "roster_sha256": h("$ROWS"),
             "step_w": "$SW", "lr": "${P2_LR:-3e-5}", "recipe": "chunk-10 (RUN-10 PBC env), pack 256, batch 128 windows, ckpt 500, trainer u = N u + 30000"},
  "archives": "n1-u<N>-v8g-a1-ema greedy EMA via hub_deploy (what ships); N u3000 and every 2k", "skip_state": "none (no annealed skip in B-even)"}, open(sys.argv[1], "w"), indent=1)
PYE
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); (( avail >= 16 )) || { echo "memory preflight: $avail GiB" >&2; exit 75; }
STOP="2026-09-30T19:40:00-04:00"   # coordinator 08:4x: nonstop to 19:40 like the other lines
LOCAL=/home/russ/farmneat-local/fs; mkdir -p $LOCAL
rsync -a $NFS/runtime/ $LOCAL/runtime/; PY=$LOCAL/runtime/python/bin/python3.12
BTR=$NFS/base-v1-pack/nfs/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True CPATH=/usr/include/python3.12
RJ=$NFS/base-v1/runs/b/RUN-10.json
eval "$(python3 -c "import json,shlex;e=json.load(open('$RJ'))['pbc_env'];e.update(json.load(open('$RJ'))['speed_env']);print('\n'.join(f'export {k}={shlex.quote(v)}' for k,v in e.items()))")"
export PBC_CACHE_ROWS=$ROWS
# ---------------- phase 1 ----------------
if [ ! -s $BE/P1-DONE ]; then
  mkdir -p $P1; seg=$(ls -d $P1/seg* 2>/dev/null | wc -l || true); R=$P1/seg$seg
  last=$(find "$P1" -path '*/train/u*.pt' -type f -printf '%f\t%p\n' 2>/dev/null | sort -t $'\t' -k1,1V | tail -1 | cut -f2- || true)   # PRO VERIFY-FIX: version sort (u1000 > u500)
  if [ -n "$last" ]; then ST=(--student "$last" --state "$last" --start-update $((10#$(basename $last .pt | tr -d u)))); else ST=(--student $INIT); fi
  echo "phase 1 seg$seg from ${last:-init}"
  ( cd $O/return && OA3_TRAINER=$BTR OA3_STEP_W_TSV=$SW PYTHONPATH=$BTR:$NFS/base-v1-pack/nfs/primitive-bc-v6-20260927/src:$LOCAL/runtime/site \
    $PY -B $T/oa3_lab.py --trainer $BTR train "${ST[@]}" --student-component model --teacher $TEACHER --teacher-component ema \
      --cache $NFS/base-v1-stepcache-v11 --frozen $O/frozen-final --out $R/train --stop-at $STOP --max-seconds 50000 --updates 1000 \
      --microbatch ${MICRO:-64} --effective-batch 512 --workers 3 --lr 3e-5 --min-lr 3e-5 --warmup 100 --clip 1 --alpha 0.5 --temperature 1.5 \
      --ema 0.999 --save-every 500 --unit-mask --pause-file $BE/PAUSE > $R.log 2>&1 ) || { rc=$?; echo "phase 1 FAILED rc $rc" >&2; tail -5 $R.log >&2; exit $rc; }
  python3 -c "import json,sys;s=json.load(open(sys.argv[1]));print(s);assert s['completed_updates']==1000" $R/train/summary.json || exit 3
  PYTHONPATH=$LOCAL/runtime/site $PY -B $BE/to_btrain.py $R/train/last.pt $INIT $BE/p2-seed.pt || { echo "phase 1 -> 2 conversion FAILED" >&2; exit 4; }
  sha256sum $BE/p2-seed.pt > $BE/P1-DONE
fi
# ---------------- phase 2 ----------------
mkdir -p $OUT
if [ ! -s $OUT/LATEST ]; then   # ASTRA TRAINING-RUNS-R2: a missing LATEST in a used run dir must not rewind phase 2 to the seed
  if ls $OUT/ckpt-*.pt >/dev/null 2>&1 || [ -s $OUT/metrics.jsonl ]; then echo "phase 2: $OUT has checkpoints/metrics but no LATEST; refusing to reseed" >&2; exit 2; fi
  cp $BE/p2-seed.pt $OUT/ckpt-0031000.pt; echo ckpt-0031000.pt > $OUT/LATEST; cp $NFS/base-v1/runs/b/ARM-ID $OUT/ARM-ID; fi
[ -s "$OUT/$(tr -d '[:space:]' < $OUT/LATEST)" ] || { echo "phase 2: LATEST names a missing checkpoint" >&2; exit 2; }
# TRAINER returns #2: resume checkpoint = newest by internal update counter (refuses a LATEST older than any ckpt-*.pt / last.pt)
PYTHONPATH=$NFS/runtime/site $NFS/runtime/python/bin/python3.12 -B $NFS/broad/ckpt_select.py $OUT > $BE/RESUME-SELECT-$SLURM_JOB_ID.json || exit 2
# TRAINER returns #5: finite non-negative step multipliers when a table is used; inherited OA3_*/N_* recorded, then unset for phase 2
python3 - $BE/EFFECTIVE-$SLURM_JOB_ID.json "$BUCKETS" "$SW" <<'PYPRE' || exit 2
import csv, json, math, os, sys
p, buckets, sw = sys.argv[1:4]
bad = []
if buckets != "1":
    for x in csv.DictReader(open(sw), delimiter='\t'):
        for k, v in x.items():
            if k and (k.endswith('multiplier') or k.endswith('_w')) and v not in (None, ''):
                f = float(v)
                if not math.isfinite(f) or f < 0: bad.append((k, v))
e = json.load(open(p)); e["inherited_env_phase2"] = {k: v for k, v in os.environ.items() if k.startswith(("OA3_", "N_"))}
e["step_multiplier_check"] = {"table": sw if buckets != "1" else None, "bad_values": len(bad)}
json.dump(e, open(p, "w"), indent=1)
if bad: raise SystemExit(f"step multiplier preflight: {len(bad)} negative/non-finite values, e.g. {bad[:3]}")
PYPRE
for v in $(compgen -e | grep -E '^(OA3_|N_)'); do unset "$v"; done
if [ $BUCKETS = 0 ]; then
python3 - $SW $ROWS $BE/p2-canary-seat.txt <<'PYC'
import csv, sys
rows = {(r['eid'], r['seat']) for r in csv.DictReader(open(sys.argv[2]), delimiter='\t')}
for r in csv.DictReader(open(sys.argv[1]), delimiter='\t'):
    if (r['episode_id'], r['seat']) in rows and float(r['step_multiplier']) != 1:
        open(sys.argv[3], 'w').write(f"{r['episode_id']} {r['seat']}\n"); break
else: raise SystemExit('no weighted seat in the roster')
PYC
BFX=$NFS/base-v1-pack-bfx/nfs/primitive-bc-v6-20260927
PYTHONPATH=$BFX/src:$BFX/work/runs/maintainer/primitive-bc:$NFS/runtime/site $PY -B $NFS/tools-0929/weight_canary2.py $BFX/work/runs/maintainer/primitive-bc \
  $SW $ROWS $NFS/base-v1-stepcache-v11 $(cat $BE/p2-canary-seat.txt) $BE/P2-WEIGHT-CANARY.json > /dev/null || { echo "weight canary FAIL" >&2; cat $BE/P2-WEIGHT-CANARY.json; exit 2; }
export PBC_WORKER_W_TSV=$SW; PK=base-v1-pack-bfx
else PK=${BUCKET_PK:-base-v1-pack-draws}; unset PBC_WORKER_W_TSV; export PBC_DRAW_LOG=$OUT/draws   # PRO CHECK-BUCKETS: per-worker seat draw log
echo '{"status": "not applicable (bucket sampler)"}' > $BE/P2-WEIGHT-CANARY.json; fi
secs=$(( $(date -d "$STOP" +%s) - $(date +%s) ))
rsync -a --delete $NFS/$PK/ $LOCAL/$PK/
# ASTRA CHECK-NBEVEN #5: the receipt binds the files this job actually executes (the local pack copy), not a source path
python3 - $BE/EFFECTIVE-$SLURM_JOB_ID.json $LOCAL/$PK "$PK" "$ROWS" "$BUCKETS" "${PBC_WORKER_W_TSV:-}" "${PBC_DRAW_LOG:-}" <<'PYX'
import hashlib, json, os, sys
p, pk, name, rows, buckets, ww, dl = sys.argv[1:]
h = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
tr = f"{pk}/nfs/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc"
d = json.load(open(p))
d["phase2"].update({"executed_pack": pk, "pack": name, "bucket_mode": buckets == "1",
    "executed_sha256": {f: h(f"{tr}/{f}") for f in ("train.py", "model.py", "step_cache.py", "data.py", "draw_log.py") if os.path.exists(f"{tr}/{f}")} | {"run_arm.sh": h(f"{pk}/run_arm.sh")},
    "trainer_train_sha256": h(f"{tr}/train.py"), "trainer_model_sha256": h(f"{tr}/model.py"),
    "roster": rows, "roster_sha256": h(rows), "worker_w_tsv": ww or None, "draw_log": dl or None})
if buckets == "1":
    d["phase2"]["step_w"] = None
json.dump(d, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
PYX
PY=$PY EXTRA_PYTHONPATH=$LOCAL/runtime/site WORK=$LOCAL/pbc-work-beven WORKERS=6 CKPT_EVERY=500 \
  bash $LOCAL/$PK/run_arm.sh base1-b 1000000 $OUT \
  --grad-checkpoint false --compile-model true --pack-steps 256 --step-cache $NFS/base-v1-stepcache-v11 --cache-pool-seats 6 --eval-every 0 --keep-ckpts 6 \
  --batch-windows 128 --warmup 300 --grow-lr-mult 1 --decay-updates 0 --lr-schedule wsd --ema-decay 0.999 --max-seconds $secs --mask-unit-noops true \
  --lr ${P2_LR:-3e-5} --aux-scale 2.8 >> $OUT/train.log 2>&1 || { rc=$?; echo "phase 2 FAILED rc $rc" >&2; tail -5 $OUT/train.log >&2; exit $rc; }
tail -2 $OUT/train.log | cut -c1-300
