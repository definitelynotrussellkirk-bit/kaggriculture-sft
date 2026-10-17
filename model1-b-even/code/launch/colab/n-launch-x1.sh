#!/bin/bash
# N B-even on Colab #1 (owner 09-30: two Colabs train our best two guesses): n_beven phase-2 recipe, resumed from the protected 4090b checkpoint.
#   n-launch.sh STOP_ISO [WORKERS]      e.g. n-launch.sh 2026-09-30T19:40:00-04:00 16
set -euo pipefail
STOP=$1; WORKERS=${2:-16}
EXPECTED_ROWS_SHA=${EXPECTED_ROWS_SHA:-426f06385d73f43acf73df18fdedafa12b4e6f9e26bc0370d3e1699c4e36a99f}; ROWS_NAME=${ROWS_NAME:-rows-n.tsv}
S=nflow-x1; N=n1-beven-colab
H=/home/russ/Desktop/farmNEAT/work/runs/maintainer/oldarch-scaleup/colab
L=/mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/launch-ncolab; D=/mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/colab-$N
COLAB=$HOME/.local/bin/colab
[ -n "${SKIP_PRE:-}" ] || "$H/vm.sh" "$S" "cd /content/bcache && python3 - '$EXPECTED_ROWS_SHA' <<'PY' || exit 1
import csv, hashlib, os, sys
raw = open('$ROWS_NAME', 'rb').read()
got = hashlib.sha256(raw).hexdigest()
assert got == sys.argv[1], f'N roster hash mismatch: {got} != {sys.argv[1]}'
r = list(csv.DictReader(raw.decode('utf-8').splitlines(), delimiter='\t'))
have = sum(os.path.isfile(f\"{x['eid']}-{x['seat']}.stp\") and os.path.getsize(f\"{x['eid']}-{x['seat']}.stp\") > 0 for x in r)
print('N roster', got[:12], 'rows', len(r), 'cached on VM', have)
assert have == len(r), f'N roster cache incomplete on the VM: {have}/{len(r)}'
PY
ls /content/parts/n1b/SEED_NAME /content/parts/n1b/seed.part.000 >/dev/null" < /dev/null
secs=$(python3 -c "import datetime as d; print(int((d.datetime.fromisoformat('$STOP') - d.datetime.now(d.timezone.utc)).total_seconds()) - 300)")
[ "$secs" -gt 600 ] || { echo "stop time too close: $secs s" >&2; exit 2; }
sed -e "s/@SECS@/$secs/" -e "s/@WORKERS@/$WORKERS/" -e "s/@ROWS@/$ROWS_NAME/" "$L/worker-x1.template.py" > "$L/worker-x1.py.tmp"
grep -q "'--max-seconds', '$secs'" "$L/worker-x1.py.tmp" && grep -q "'PBC_CACHE_ROWS': '/content/bcache/$ROWS_NAME'" "$L/worker-x1.py.tmp" && grep -q ", $WORKERS, \"/content/parts/n1b\"" "$L/worker-x1.py.tmp"
mv "$L/worker-x1.py.tmp" "$L/worker-x1.py"
mkdir -p "$D"
# own systemd units: a switch/timer unit that runs this launcher must not take the loops down when it exits (13:07 incident)
for try in 1 2 3 4 5; do   # colab exec can fail at once with "runtime proxy rejected ... Retry the command" (N flow 12:59 incident)
  TS=$(date +%H%M%S)
  systemd-run --user --unit nx1-exec-$TS --collect /bin/bash -c "exec nice -n 19 $COLAB exec -s $S -f $L/worker-x1.py > $L/exec-$S.log 2>&1" > /dev/null
  systemctl --user show -p MainPID --value nx1-exec-$TS > "$L/exec.pid"
  sleep 30
  systemctl --user is-active -q nx1-exec-$TS && break
  grep -q "Retry the command" "$L/exec-$S.log" || break
  echo "$(date -u +%FT%TZ) exec retry $try: $(tail -1 "$L/exec-$S.log" | cut -c1-120)" | tee -a "$L/LAUNCH.log"
done
systemctl --user is-active -q nx1-pull || { systemctl --user stop nx1-pull 2>/dev/null || true; systemctl --user reset-failed nx1-pull 2>/dev/null || true; systemd-run --user --unit nx1-pull --collect /bin/bash -c "exec nice -n 10 $H/pull.sh $S $N --loop 120 > $D/pull-$S.out 2>&1" > /dev/null; systemctl --user show -p MainPID --value nx1-pull > "$D/pull.pid"; }
systemctl --user is-active -q nx1-keepalive || { systemctl --user stop nx1-keepalive 2>/dev/null || true; systemctl --user reset-failed nx1-keepalive 2>/dev/null || true; systemd-run --user --unit nx1-keepalive --collect /bin/bash -c "exec nice -n 19 $H/keepalive.sh $S $N 120 > /dev/null 2>&1" > /dev/null; systemctl --user show -p MainPID --value nx1-keepalive > "$D/keepalive.pid"; }
echo "$(date -u +%FT%TZ) launched $N on $S: rows $ROWS_NAME (${EXPECTED_ROWS_SHA:0:12}), workers $WORKERS, max-seconds $secs, exec pid $(cat "$L/exec.pid")" | tee -a "$L/LAUNCH.log"
