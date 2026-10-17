#!/bin/bash
# N B-even 4090b -> Colab #1: wait for job 7318's ckpt-0042000, cancel 7318, protect the checkpoint read-only with its sha,
# stage it as seed parts on the VM, reassemble + sha-check it there as the run's LATEST, run bpreflight.  Does not launch.
set -uo pipefail
F=/mnt/ai/scratch/farmNEAT; RUN=$F/b-even/run; CK=ckpt-0042000.pt; JOB=7318
L=$F/pbc-colab-v6-results/launch-ncolab; LOG=$L/HANDOVER.log; ST=$F/oldarch-scaleup/resume-stage-ncolab/n1
PROT=$F/hourly/PROTECTED/NBEVEN-final4090b-u42000.pt
H=/home/russ/Desktop/farmNEAT/work/runs/maintainer/oldarch-scaleup/colab; COLAB=$HOME/.local/bin/colab; S=oldarch-g4
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
vm() { local i out; for i in $(seq 1 12); do out=$(timeout 900 "$H/vm.sh" "$S" "$1" < /dev/null 2>&1) && { echo "$out"; return 0; }; sleep 10; done; echo "$out"; return 1; }
until [ "$(tr -d '[:space:]' < $RUN/LATEST 2>/dev/null)" = "$CK" ] && [ -s "$RUN/$CK" ]; do sleep 10; done
s1=$(stat -c %s "$RUN/$CK"); sleep 20; [ "$s1" = "$(stat -c %s "$RUN/$CK")" ] || { log "size moving on $CK"; exit 2; }
log "$CK present (LATEST); cancelling $JOB"
scancel $JOB; until ! squeue -h -j $JOB -o %A 2>/dev/null | grep -qx $JOB; do sleep 5; done
log "$JOB gone from the queue"
sha=$(sha256sum "$RUN/$CK" | cut -d' ' -f1)
cp "$RUN/$CK" "$PROT.tmp" && [ "$(sha256sum "$PROT.tmp" | cut -d' ' -f1)" = "$sha" ] && chmod 444 "$PROT.tmp" && mv "$PROT.tmp" "$PROT" \
  || { log "protect copy failed"; exit 3; }
echo "$sha  $(basename $PROT)  source=$RUN/$CK job=$JOB" > "$PROT.sha256"; chmod 444 "$PROT.sha256"
log "protected $PROT sha $sha"
rm -rf "$ST"; mkdir -p "$ST"; (cd "$ST" && split -b 60M -d -a 3 "$PROT" seed.part.); echo "$CK $sha" > "$ST/SEED_NAME"
n=$(ls "$ST"/seed.part.* | wc -l); log "split into $n parts"
ok=0
for i in 1 2 3 4 5; do
  if tar -C "$(dirname "$ST")" -cf - n1 | ssh -o "ProxyCommand=$COLAB ssh --proxy-mode -s $S" -o StrictHostKeyChecking=no \
       -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o Compression=no -o ServerAliveInterval=30 -c aes128-gcm@openssh.com \
       root@$S "mkdir -p /content/parts && rm -rf /content/parts/n1 && tar -C /content/parts -xf -"; then ok=1; break; fi
  log "push attempt $i failed"; sleep 15
done
[ $ok = 1 ] || { log "seed push failed"; exit 4; }
log "seed parts pushed"
r=$(vm "set -e; O=/content/results/n1-beven-colab; mkdir -p \$O; [ ! -e \$O/LATEST ] || { echo 'LATEST exists'; exit 9; }; cat /content/parts/n1/seed.part.* > \$O/$CK.tmp; got=\$(sha256sum \$O/$CK.tmp | cut -d' ' -f1); [ \"\$got\" = $sha ] || { echo sha-mismatch \$got; exit 8; }; mv \$O/$CK.tmp \$O/$CK; echo $CK > \$O/LATEST; echo reassembled \$got")
log "VM seed: $r"
grep -q "reassembled $sha" <<< "$r" || exit 5
until r=$(vm "pgrep -f 'python3 /content/fetch_root[.]py' > /dev/null && echo fetching || echo idle; tail -c 200 /content/fetch-n1.log") && grep -q '^idle' <<< "$r"; do sleep 60; done
log "seat fetch finished: $(tail -1 <<< "$r")"
timeout 300 "$COLAB" upload -s "$S" $F/broad/bpreflight.py /content/bpreflight.py > /dev/null 2>&1 || { log "bpreflight upload failed"; exit 6; }
r=$(vm "cd /content && BPF_THREADS=24 nice -n 5 python3 /content/bpreflight.py /content/results/n1-beven-colab/$CK /content/cache/rows-n.tsv /content/cache /content/code/primitive-bc-v6-20260927 /content/results/n1-beven-colab/preflight > /content/bpreflight-n1.log 2>&1; echo rc=\$?; tail -c 1500 /content/bpreflight-n1.log")
log "bpreflight: $(tr '\n' ' ' <<< "$r" | cut -c1-1500)"
grep -q "^rc=0" <<< "$r" || exit 7
log "HANDOVER-READY"
