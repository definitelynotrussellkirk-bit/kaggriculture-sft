#!/bin/bash
# Re-switch N B-even once when data-ops publishes a newer HR2-B (the held-out-inclusive pool, expected 16:30-16:45), else at the
# deadline take whatever is newest if it differs. Usage: nb_pool_watch.sh BASELINE_SHA12 DEADLINE_HHMM
BASE=$1; DL=$2; W=/mnt/ai/scratch/farmNEAT/broad/t1/wash; L=/mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/launch-ncolab
until R=$(readlink -f $W/HR2-B) && s=$(basename $R .tsv | sed 's/rows-t1wash-b\.//') && [ "$s" != "$BASE" ] && [ -s $R.receipt.json ] || [ "$(date +%H%M)" -ge "$DL" ]; do sleep 30; done
sleep 30   # let data-ops finish writing the receipt/caching
exec bash $L/nb_pool_reswitch.sh
