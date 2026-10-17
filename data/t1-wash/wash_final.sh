#!/bin/bash
# 18:30: T1-ONLY-FINAL (owner "we train at the last step"): MMPQ winning seats incl. every held-out MMPQ game; repoint LATEST-*.
BR=/mnt/ai/scratch/farmNEAT/broad; L=$BR/t1/wash/WASH-HOURLY.log
cd $BR && T1_FINAL=1 nice -n 19 /usr/bin/python3 -B build_t1wash.py >> $L 2>&1 || { echo "$(date +%T) FAIL final wash" >> $L; exit 2; }
r=$(readlink -f $BR/t1/wash/LATEST-NFLOW)
echo "- R-WASH-FINAL ($(date +%H:%M)) T1-ONLY-FINAL: $(python3 -c "import json;d=json.load(open('$r.receipt.json'));print('nflow', '$r'.split('/')[-1], 'act', d['act_rows'], 'held-out included', len(d['held_out_included']), d['bucket_probs_arg'])"); B $(basename $(readlink -f $BR/t1/wash/LATEST-B))" >> /home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md
