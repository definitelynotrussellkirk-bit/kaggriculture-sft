#!/bin/bash
# One B-family winner wash on gpu-skynet (owner 15:0x "winner wash"): 15 min on the MMPQ-wins B roster, RAW export (ema := model),
# fail-closed build on the given harness (variant builder), finalize, gate checks, stage_exact_archive. Prepared option only: no upload.
#   b_wash.sh NAME BASE_CKPT BASE_SHA ROWS LR VARIANT(v9gsafe|plain) [TAG]
set -uo pipefail
NAME=$1; BASE=$2; BASE_SHA=$3; ROWS=$4; LR=$5; VAR=$6; TAG=${7:-$NAME}
F=/mnt/ai/scratch/farmNEAT; W=$F/t4w; P=$F/pbc-deploy; T=$F/tools-0929; G=$F/league/gpuserve; SP=$F/league/records/gpu
LEDGER=/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md; LOG=$W/BWASH-$NAME.log; STAGE=/home/russ/farmneat-local/stage-upload
PY312="PYTHONPATH=$F/runtime/site $F/runtime/python/bin/python3.12 -B"; FIN=$F/freshen/guard/finalize_snapshot.py
exec >> $LOG 2>&1
say() { echo "$(date '+%T') $*"; }; ledger() { echo "$1" >> $LEDGER; say "LEDGER $1"; }
waitjob() { while squeue -h -j $1 2>/dev/null | grep -q .; do sleep 15; done; sacct -j $1 -X -n -o State -P | head -1; }
[ "$(sha256sum $BASE | cut -d' ' -f1)" = "$BASE_SHA" ] || { ledger "R-$TAG ($(date +%H:%M) 09-30): FAILED - base sha mismatch"; exit 1; }
[ -e $P/$NAME ] && { ledger "R-$TAG ($(date +%H:%M) 09-30): FAILED - deploy dir $NAME exists"; exit 1; }
# free skynet's GPU: stop the game daemon(s) there; hold pending skynet-game tasks until the wash starts
true
SGH=""
BR=$W/runs/$NAME; mkdir -p $BR
U0=$((10#$(basename $BASE .pt | sed 's/.*ckpt-//; s/[^0-9]//g'))); C0=$(printf 'ckpt-%07d.pt' $U0)
[ -f $BR/LATEST ] || { cp $BASE $BR/$C0.tmp && mv $BR/$C0.tmp $BR/$C0 && echo $C0 > $BR/LATEST && cp $F/base-v1/runs/b/ARM-ID $BR/ARM-ID; }
BU=${WASH_UPDATES:-}; if [ -n "$BU" ]; then BMAX=$(( BU * 4 + 120 )); else BU=10000; BMAX=900; fi   # nb factory: fixed update count (owner 16:0x) or the 900-s cap
JB=$(cd /tmp && sbatch --parsable --export=ALL,ROWS=$ROWS,LR=$LR,RUN=$BR,UPDATES=$BU,MAX_S=$BMAX $W/t4w-bn-skynet.sbatch)
( until squeue -h -j $JB -t R 2>/dev/null | grep -q . || ! squeue -h -j $JB 2>/dev/null | grep -q .; do sleep 10; done; for j in $SGH; do scontrol release $j 2>/dev/null; done ) &
ledger "R-$TAG-START ($(date +%H:%M) 09-30): $NAME wash on gpu-skynet job $JB: base $BASE (sha ${BASE_SHA:0:12}, u$U0), rows $(basename $ROWS), lr $LR, cap 900 s, original batch; RAW export; harness $VAR"
st=$(waitjob $JB); LAST=$(cat $BR/LATEST)
# resume the skynet game daemon (runs to the lock)
JS=none   # sprinkle: no skynet daemon restart
say "train $JB $st -> $LAST; skynet daemon $JS"
[ "$LAST" != "$C0" ] || { ledger "R-$TAG ($(date +%H:%M) 09-30): FAILED - wash job $JB $st made no checkpoint"; exit 1; }
RAWD=$BR/raw; mkdir -p $RAWD
eval $PY312 - $BR/$LAST $RAWD/$LAST <<'PYR' || { ledger "R-$TAG FAILED - raw export"; exit 1; }
import hashlib, os, sys, torch
src, out = sys.argv[1:3]
b = torch.load(src, map_location="cpu", weights_only=False)
b["ema"] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in b["model"].items()}
if isinstance(b.get("ema_extra"), dict):
    b["ema_extra"] = {d: {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in b["model"].items()} for d in b["ema_extra"]}
b.setdefault("extra", {})["raw_export"] = {"source": src, "source_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(), "ema_from": "model (raw)"}
torch.save(b, out + ".tmp"); os.replace(out + ".tmp", out)
PYR
chmod 444 $RAWD/$LAST
python3 - $BR/EFFECTIVE-wash.json $BASE $BASE_SHA $ROWS $BR/$LAST $RAWD/$LAST $LR <<'PYE'
import hashlib, json, sys
e, base, bsha, rows, ck, raw, lr = sys.argv[1:8]; h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"line": "winner wash (MMPQ winning seats, mirror 4x)", "base": base, "base_sha256": bsha, "roster": rows, "roster_sha256": h(rows),
           "lr": lr, "cap_s": 900, "trainer": "base-v1-pack run_arm base1-b (t4w-bn-skynet.sbatch)", "final_checkpoint": ck,
           "final_checkpoint_sha256": h(ck), "raw_export": raw, "raw_export_sha256": h(raw)}, open(e, "w"), indent=1)
PYE
if [ "$VAR" = plain ]; then J=$(cd /tmp && sbatch --parsable -o $W/logs/build-$NAME-%j.out $T/hub_deploy_v9f.sbatch $RAWD/$LAST $NAME greedy)
else J=$(cd /tmp && sbatch --parsable -o $W/logs/build-$NAME-%j.out $T/hub_deploy_variant.failclosed.sbatch $RAWD/$LAST $NAME greedy $VAR); fi
J=${J%%;*}
FJ=$(cd /tmp && sbatch --parsable --dependency=afterany:$J -p cpu,gpu --exclude=gpu-spark,laptop-4060,gpu-3090 -c 1 --mem 8G -t 00:30:00 -o $W/logs/final-$NAME-%j.out \
     --wrap "[ -s $P/$NAME/HUB-DONE.json ] || exit 3; $PY312 $FIN $NAME $RAWD/$LAST $P/$NAME $BR/EFFECTIVE-wash.json $BR"); FJ=${FJ%%;*}
say "build $J finalize $FJ"; waitjob $J > /dev/null; waitjob $FJ > /dev/null
SHA=$(python3 - $P/$NAME $BR $NAME <<'PYG'
import json, sys, glob
p, d, n = sys.argv[1:4]
try:
    h = json.load(open(f"{p}/HUB-DONE.json")); l = json.load(open(f"{d}/LINKS-{n}.json"))
    gates = [json.load(open(g)) for g in glob.glob(f"{p}/RELEASE-GATE*.json")]
    v = json.load(open(f"{p}/VERIFY.json"))
    ok = (h.get("complete") in (True, "True") and h.get("sha_match") in (True, "True") and int(h.get("fallbacks")) == 0 and gates
          and all(g.get("complete") is True for g in gates) and l.get("all_pass") is True and v.get("complete") is True and v.get("fallbacks") == 0)
    print(h["archive_sha256"] if ok else "FAIL")
except Exception:
    print("FAIL")
PYG
)
[ "$SHA" != FAIL ] || { ledger "R-$TAG ($(date +%H:%M) 09-30): FAILED - build $J / finalize $FJ: VERIFY/gate/links not all pass; not staged"; exit 1; }
A=$P/$NAME/$NAME.tar.gz; BY=$(stat -c %s $A)
python3 -B $F/watchers/stage_exact_archive.py $A --sha256 $SHA --bytes $BY --destination-dir $STAGE > $W/stage-$NAME.json 2>&1 || { ledger "R-$TAG FAILED - stage"; exit 1; }
ledger "R-$TAG ($(date +%H:%M) 09-30): $NAME ready (NOT for upload unless the owner says): sha256 $SHA, bytes $BY, staged $STAGE/$NAME.$SHA.tar.gz. Wash $JB ($st) u$U0 -> $LAST at lr $LR on $(basename $ROWS); RAW export; harness $VAR; build $J, finalize $FJ; VERIFY complete 0 fallbacks, gate complete, LINKS all_pass"
echo "$NAME $SHA $BY" > $W/.ready-$NAME
