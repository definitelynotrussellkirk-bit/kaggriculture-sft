#!/bin/bash
# Until 23:02Z: build v11 .stp (both seats) for each new copy-mmpq MANIFEST game fetched after 22:43Z. No pool changes.
C=/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/copy-mmpq; S=/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11; B=$S/backfill
L=/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md
while [ "$(date -u +%H%M)" -lt 2302 ]; do
  python3 - <<'PY' > /mnt/ai/scratch/farmNEAT/broad/.mmpq_new.tsv
import json,os
C='/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/copy-mmpq'; S='/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11'
done=set(open('/mnt/ai/scratch/farmNEAT/broad/MMPQ-NEW-SUBMITTED.txt').read().split()) if os.path.exists('/mnt/ai/scratch/farmNEAT/broad/MMPQ-NEW-SUBMITTED.txt') else set()
for l in open(C+'/MANIFEST.jsonl'):
    r=json.loads(l); e=r['episode_id']
    if r['fetched_utc']<='2026-09-30T22:43' or 'M & M & P & Q' not in (r.get('team_names') or []) or str(e) in done: continue
    if all(os.path.exists(f'{S}/{e}-{z}.stp') for z in (0,1)): continue
    for z in (0,1): print(f"{C}/{r['archive']}\t{r['member']}\t{r['day']}\t{e}\t{z}\t1\tmmpq-new")
PY
  if [ -s /mnt/ai/scratch/farmNEAT/broad/.mmpq_new.tsv ]; then
    R=/mnt/ai/scratch/farmNEAT/broad/MMPQ-NEW-$(date -u +%H%M%S).rows.tsv
    { printf 'archive\tmember\tday\teid\tseat\tweight\treason\n'; cat /mnt/ai/scratch/farmNEAT/broad/.mmpq_new.tsv; } > $R
    n=$(wc -l < /mnt/ai/scratch/farmNEAT/broad/.mmpq_new.tsv)
    j=$(sbatch --parsable -J cache-mmpqnew -p cpu -w cpu-hub -c 1 --mem 3000 --array=0-$((n-1)) --export=ALL,ROWS=$R,N=$n,FIRST=0,MIN_AVAIL_GB=6 -o $B/slurm-mmpqnew-%A_%a.out $B/slurm_shard.sbatch)
    cut -f4 /mnt/ai/scratch/farmNEAT/broad/.mmpq_new.tsv | sort -u >> /mnt/ai/scratch/farmNEAT/broad/MMPQ-NEW-SUBMITTED.txt
    echo "$(date +%T) job $j eids $(cut -f4 /mnt/ai/scratch/farmNEAT/broad/.mmpq_new.tsv | sort -u | tr '\n' ' ')"
  fi
  sleep 20
done
echo "$(date +%T) window closed"
