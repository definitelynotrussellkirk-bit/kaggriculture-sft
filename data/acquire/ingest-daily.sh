#!/bin/bash
# Daily replay ingestion (owner 2026-09-25: add each new daily upload to the corpus). Every hour: read the Kaggle episodes index,
# download every daily dataset dated >= 2026-09-03 that is not installed yet (404 until Kaggle builds its zip), verify json count ==
# index episode_count and zip CRC, install next to the others, and append it to ../TRAIN-ARCHIVES.txt only if the first and last
# episode's game configuration equals ../REFERENCE-CONFIG.json (a rules change is logged and never auto-added).
# 2026-09-29 (ASTRA-PRO-FINAL-TOUCHUP-0930 F5): a verifier crash/empty verdict no longer installs the zip; the list write is checked.
cd /mnt/ai/scratch/farmNEAT/primitive-bc-data/incoming || exit 1
LIST=../TRAIN-ARCHIVES.txt; REF=../REFERENCE-CONFIG.json
CNT=/home/russ/Desktop/farmNEAT/work/runs/maintainer/pbc-autosubmit
cnt() { local f=$CNT/KAGGLE-REQUESTS-$(date -u +%F).txt; ( flock 9; n=$(cat $f 2>/dev/null || echo 0); echo $((n+2)) > $f ) 9>$f.lock; }   # +2 per CLI call
while true; do
  rm -rf idx.tmp && mkdir idx.tmp
  cnt; if timeout 300 kaggle datasets download -d kaggle/kaggriculture-episodes-index -p idx.tmp --unzip > /dev/null 2>&1 && [ -f idx.tmp/manifest.csv ]; then
    mv idx.tmp/manifest.csv episodes-index-manifest.csv
  fi
  for day in $(awk -F, 'NR>1 && $1>="2026-09-03" {print $1}' episodes-index-manifest.csv); do
    D=kaggriculture-episodes-$day
    [ -e ../$D.zip ] && continue
    want=$(awk -F, -v d=$day '$1==d {print $4}' episodes-index-manifest.csv)
    rm -f $D.zip
    cnt; if ! timeout 3600 kaggle datasets download -d kaggle/$D -p . > dl-$day.log 2>&1 || [ ! -f $D.zip ]; then
      echo "$(date -u +%FT%T) not ready $day: $(grep -m1 -o '[0-9][0-9][0-9] Client Error[^f]*' dl-$day.log)"; continue
    fi
    verdict=$(nice -n 10 python3 - $D.zip "$want" $REF <<'PY'
import json, sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); names = [x for x in z.namelist() if x.endswith('.json')]
if len(names) != int(sys.argv[2]) or z.testzip() is not None:
    print('BAD count', len(names), 'want', sys.argv[2]); sys.exit()
ref = json.load(open(sys.argv[3]))
def cfg(n):
    c = dict(json.loads(z.open(n).read())['configuration']); c.pop('seed', None); return c
diff = {k for n in (names[0], names[-1]) for k in set(cfg(n)) | set(ref) if cfg(n).get(k) != ref.get(k)}
print('OK' if not diff else 'CONFIG ' + ','.join(sorted(diff)))
PY
)
    case $verdict in
      OK|CONFIG*) ;;
      *) echo "$(date -u +%FT%T) verify failed $day (not installed, retried next hour): ${verdict:-verifier error}"; rm -f $D.zip; continue ;;
    esac
    case $verdict in
      *)
        h=$(sha256sum $D.zip | cut -c1-64); mv $D.zip ../$D.zip; echo "$h  $D.zip" >> DOWNLOADS.sha256
        if [ "$verdict" = OK ]; then
          { cat $LIST; echo "/mnt/ai/scratch/farmNEAT/primitive-bc-data/$D.zip"; } | sort -u > $LIST.tmp && mv $LIST.tmp $LIST
          if grep -qx "/mnt/ai/scratch/farmNEAT/primitive-bc-data/$D.zip" $LIST; then
            echo "$(date -u +%FT%T) installed $day and added to TRAIN-ARCHIVES ($(wc -l < $LIST) days) sha $h"
          else echo "$(date -u +%FT%T) ALERT installed $day but TRAIN-ARCHIVES update FAILED; add it by hand"; fi
        else
          echo "$(date -u +%FT%T) installed $day NOT added: $verdict differs from REFERENCE-CONFIG.json sha $h"
        fi ;;
    esac
  done
  sleep 3600
done
