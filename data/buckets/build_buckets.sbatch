#!/bin/bash
#SBATCH -J buckets
#SBATCH -c 1
#SBATCH --mem 3G
#SBATCH -t 00:45:00
#SBATCH -o /mnt/ai/scratch/farmNEAT/broad/buckets-%j.out
export PYTHONPATH=/mnt/ai/scratch/farmNEAT/corpus-final/site:/mnt/ai/scratch/farmNEAT/runtime/site
# TOP4-BEATERS refresh first (new never-trained top-4 beaters go to VAL first; appender holds them back until assessed)
set -o pipefail
if [ "${SKIP_TOP4:-0}" != 1 ]; then
  nice -n 10 /mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12 -B /mnt/ai/scratch/farmNEAT/corpus-final/special/build_top4_beaters.py | tail -c 600 \
    || { echo "REFUSED: TOP4 refresh failed; bucket build not run (live generation unchanged)"; exit 3; }
fi
nice -n 10 /mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12 -B /mnt/ai/scratch/farmNEAT/broad/build_buckets.py "$@"
rc=$?
# t1-all roster (owner 09:2x; not part of the buckets): refreshed after every build; its output line starts with "T1 ", never "{"
[ "${SKIP_T1:-0}" = 1 ] || nice -n 10 /mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12 -B /mnt/ai/scratch/farmNEAT/broad/build_t1.py 2>&1 | tail -1 || echo "T1 FAIL"
# incremental breakage scan of this generation's new episodes (applied at the next build)
nice -n 10 python3 -B /mnt/ai/scratch/farmNEAT/broad/breakage_scan.py plan | sed 's/^/SCAN /'; nice -n 10 python3 -B /mnt/ai/scratch/farmNEAT/broad/breakage_scan.py scan 0 1 || echo "SCAN FAIL"
exit $rc
