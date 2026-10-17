#!/bin/bash
# Hourly bucket rebuild (DATA-PLAN-0930 §7): at :50 (after catalog :20, cache-new :36) submit build_buckets.sbatch (publishes
# LATEST-BUCKETS unless a bucket is > 10% uncached), then queue that build's uncached seats for the v11 cache in bucket order
# A, B, C, D, E (-p cpu; never skynet/3090/Spark). Stops after 09-30 19:00 EDT. Log: broad/buckets-loop.log
BR=/mnt/ai/scratch/farmNEAT/broad; B=/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11/backfill; L=$BR/buckets-loop.log
end=$(date -d "2026-09-30 19:00 EDT" +%s)
while [ "$(date +%s)" -lt $end ] && [ ! -e $BR/STOP-BUCKETS ]; do
  now=$(date +%s); next=$(date -d "$(date +%Y-%m-%d\ %H):50" +%s); [ $next -le $now ] && next=$((next + 3600)); sleep $((next - now))
  j=$(sbatch --parsable -p cpu --exclude=gpu-3090,gpu-skynet,gpu-spark /mnt/ai/scratch/farmNEAT/broad/build_buckets.sbatch 2>&1); rc=$?
  if [ $rc -ne 0 ] || ! [[ "$j" =~ ^[0-9]+$ ]]; then echo "$(date +%T) FAIL: bucket build submission rc=$rc: $j" >> $L; continue; fi
  echo "$(date +%T) bucket build job $j" >> $L
  while squeue -h -j $j 2>/dev/null | grep -q .; do sleep 30; done
  st=$(sacct -n -X -j $j -o State%20,ExitCode 2>/dev/null | head -1 | xargs)
  out=$(tail -c 8000 $BR/buckets-$j.out | grep '^{' | tail -1)
  if [ -z "$out" ]; then echo "$(date +%T) FAIL: bucket build $j state [$st] produced no receipt line (TOP4 or input refusal): $(grep -m1 REFUSED $BR/buckets-$j.out)" >> $L; continue; fi
  echo "$(date +%T) bucket build $j state [$st]" >> $L
  d=$(echo "$out" | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r.get("dir_written") or r["dir"])')
  echo "$(date +%T) $(echo "$out" | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["publish"], {b: (v["seats"], v["uncached"]) for b, v in r["buckets"].items()})')" >> $L
  f=$d/UNCACHED-ALL.tsv; bl=$(echo "$out" | python3 -c 'import json,sys; print(" ".join(json.load(sys.stdin)["buckets"]))')
  head -1 $d/UNCACHED-$(echo $bl | cut -d" " -f1).tsv > $f || { echo "$(date +%T) FAIL: no UNCACHED files in $d" >> $L; continue; }
  for b in $bl; do tail -n +2 $d/UNCACHED-$b.tsv >> $f || echo "$(date +%T) FAIL: missing $d/UNCACHED-$b.tsv" >> $L; done
  for u in $d/UNCACHED-*.tsv; do b=$(basename $u .tsv); b=${b#UNCACHED-}; [ "$b" = ALL ] && continue; case " $bl " in *" $b "*) ;; *) tail -n +2 $u >> $f;; esac; done   # omitted buckets too
  n=$(( $(wc -l < $f) - 1 ))
  if [ $n -gt 0 ]; then
    k=$(( n < 400 ? 2 : 4 ))
    cj=$(sbatch --parsable -J cache-buckets -p cpu --exclude=gpu-3090,gpu-skynet,gpu-spark,gpu-4090b -c 1 --mem 3000 --array=0-$((k-1))%2 --export=ALL,ROWS=$f,N=$k,FIRST=0 -o $B/slurm-buckets-%A_%a.out $B/slurm_shard.sbatch 2>&1)
    [[ "$cj" =~ ^[0-9]+$ ]] || { echo "$(date +%T) FAIL: cache job submission: $cj" >> $L; cj=""; }
    echo "$(date +%T) uncached $n (order $bl) -> cache job $cj" >> $L
  fi
  t1u=$BR/t1/UNCACHED-t1.tsv; nt=$(( $(wc -l < $t1u 2>/dev/null || echo 1) - 1 ))
  if [ "$nt" -gt 0 ]; then
    tj=$(sbatch --parsable -J cache-t1 -p cpu --exclude=gpu-3090,gpu-skynet,gpu-spark,gpu-4090b -c 1 --mem 3000 --array=0-1 --export=ALL,ROWS=$t1u,N=2,FIRST=0 -o $B/slurm-t1-%A_%a.out $B/slurm_shard.sbatch 2>&1)
    echo "$(date +%T) t1 uncached $nt -> cache job $tj; $(grep '^T1 ' $BR/buckets-$j.out | tail -1 | cut -c1-200)" >> $L
  fi
  # N flow shards (.nfz) for the newest nflow roster (nflow derives rows-nflow-v3.<bucket hash>.tsv from LATEST-BUCKETS):
  # incremental (existing .nfz skipped, rows without .stp skipped), after this hour's cache job; cpu-hub/4090b spare CPUs only.
  # N flow's roster for THIS generation is written at :55 by its switch, after this step: pre-build shards from the generation
  # itself (pseudo roster: N view winners + their games' other seats), so every seat of the next N roster has .nfz
  srun -p cpu -w cpu-hub -c 1 --mem 3G -J nfzroster -t 10 env PYTHONPATH=/mnt/ai/scratch/farmNEAT/corpus-final/site \
    /mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12 -B $BR/nfz_prebuild_roster.py "$d" $BR/NFZ-PREBUILD.tsv >> $L 2>&1
  R=$BR/NFZ-PREBUILD.tsv
  if [ -n "$R" ] && ! squeue -h -n nflow-shards-v3 2>/dev/null | grep -q .; then
    dep=""; [[ "${cj:-}" =~ ^[0-9]+ ]] && dep="--dependency=afterany:$cj"
    sj=$(sbatch --parsable $dep --array=0-11%2 --export=ALL,ROSTER=$R /mnt/ai/scratch/farmNEAT/broad/nfz-shards.sbatch 12 2>&1)
    echo "$(date +%T) nflow shards for $(basename $R) -> job $sj" >> $L
  fi
  cj=""
done
