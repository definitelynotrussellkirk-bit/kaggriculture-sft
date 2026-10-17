#!/bin/bash
# skynet lane (owner's desktop, coordinator 10:1x): nice 19; no new game below 12 GB MemAvailable (waits); the kill-newest-below-8 GB guard is
# the local unit farmneat-skynet-guard. Same runner/receipts as farm_game.sbatch (exec'd unchanged).
while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 12 ]; do sleep 20; done
exec nice -n 19 bash /mnt/ai/scratch/farmNEAT/league/extfarm/farm_game.sbatch "$@"
