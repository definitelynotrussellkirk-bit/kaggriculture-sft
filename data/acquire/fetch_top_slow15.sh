#!/bin/bash
# Owner 2026-09-28 ~8:45 AM EDT: "boey submitted a new agent, we probably are going to want to copy all those": PRIO_TEAM (Boey,
# 16915014) is fetched FIRST each cycle with its own budget PRIO_MAX (60, newest submission first), outside TARGET.
# Owner 2026-09-28: "make sure we've been grabbing 100 replays every 15 min" "from top 12". Successor of fetch_top_slow2.sh:
# a cycle STARTS every CYCLE s (900; slow2 slept 900 s after a ~5 min pass). Each cycle targets TARGET new public replays (100)
# from the current top-12 teams (us excluded); team i of k remaining gets ceil(left / k), capped at PER_TEAM_MAX, so quota a
# team cannot use (no unfetched games) rolls to later teams. Same save_team.py, shared counter, 429 backoff as slow2.
cd /mnt/ai/scratch/farmNEAT/primitive-bc-data/extra || exit 1
TOP=${TOP:-12}; TARGET=${TARGET:-100}; PER_TEAM_MAX=${PER_TEAM_MAX:-30}; CAP=${CAP:-100000}; CYCLE=${CYCLE:-900}
BACKOFF=${BACKOFF:-3600}
PRIO_TEAM=${PRIO_TEAM:-16915014}; PRIO_MAX=${PRIO_MAX:-60}
# 2026-09-29 05:10 EDT owner ("we definitely want to copy his games", "his policy was able to shoot up very fast"): EXTRA_PRIO teams are
# fetched every cycle with PRIO_MAX each, like PRIO_TEAM, wherever they sit on the leaderboard (Majkel1337 16718819, M&M&P&Q 16681125).
# 2026-09-29 (ASTRA-PRO-FINAL-TOUCHUP-0930 F5): a leaderboard refresh that fails with 429 backs off instead of fetching on a stale list.
EXTRA_PRIO=${EXTRA_PRIO:-16718819 16681125}
C=/home/russ/Desktop/farmNEAT/work/runs/maintainer/pbc-autosubmit
# 2026-09-30 08:1x (coordinator, owner "grab as many replays as possible"): slow8 = slow7 + OUR OWN new hosted replays FIRST each
# cycle (newest OWN_N completed submissions, newest first; replaces farmneat-own-games' :40 freeze + :00 pass), then the top-team
# mix fills the rest of TARGET; then beaters.py gets BEATERS_PER_CYCLE calls. Every call is counted (hosted_fetch: 2 per CLI call).
REPO=/home/russ/Desktop/farmNEAT; PYV=$REPO/work/venv/python
bump() { ( flock 9; v=$(cat "$1" 2>/dev/null || echo 0); echo $(( v + $2 )) > "$1" ) 9>>"$1.lock"; }
own_step() {   # our 3 newest submissions: losses/draws first, then wins (listing rewards); budget = $left; updates own/left/got/backoff
  while read -r sub name; do
    [ -n "$sub" ] || continue
    [ $left -gt 0 ] || break
    hd=$REPO/work/runs/deployment/$name/hosted; mkdir -p "$hd"
    b0=$(ls "$hd" | grep -c 'replay.json.gz')
    o=$(cd $REPO && timeout 1500 nice -n 10 python3 -B work/runs/maintainer/pbc-autosubmit/hosted_fetch.py "$sub" "$hd" --n 400 --no-logs \
        --max-new ${OWN_MAX:-$left} 2>&1)
    n=$(( $(ls "$hd" | grep -c 'replay.json.gz') - b0 )); own=$((own + n)); left=$((left - n)); got=$((got + n))
    nc=$(echo "$o" | grep -oE '^CALLS [0-9]+' | tail -1 | cut -d' ' -f2); nc=${nc:-$(( 3 + 2 * n ))}
    bump "$C/KAGGLE-REQUESTS-$(date -u +%F).txt" $nc   # hosted_fetch's own count of attempted calls (incl. failed and 429)
    echo "$(date -u +%FT%TZ) own sub $sub ($name): +$n replays; listing $(cat "$hd/RECORD-LISTING.json" 2>/dev/null | cut -c1-160)"
    if echo "$o" | grep -q "429\|Too Many Requests\|CalledProcessError"; then echo "$(date -u +%FT%TZ) own sub $sub: HTTP 429/replay error"; backoff=1; break; fi
  done < <(own_subs)
  python3 - "$REPO/work/runs/deployment" /mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/hosted-ours/RECORD.json <<'PY'
import glob, json, os, sys
out = {}
for p in glob.glob(os.path.join(sys.argv[1], '*', 'hosted', 'RECORD-LISTING.json')):
    r = json.load(open(p)); r['name'] = p.split('/')[-3]; out[str(r['submission_id'])] = r
tmp = sys.argv[2] + '.tmp'; json.dump({'schema': 'hosted-record/v1', 'source': 'Kaggle episode listings (rewards), no downloads', 'submissions': out}, open(tmp, 'w'), indent=1); os.replace(tmp, sys.argv[2])
PY
}
own_subs() {   # newest OWN_N completed submissions (listing cached OWN_LIST_MIN minutes; 2 calls per listing)
  local f=$C/.own-subs.txt
  if [ ! -s $f ] || [ $(( $(date +%s) - $(stat -c %Y $f) )) -ge $(( ${OWN_LIST_MIN:-30} * 60 )) ]; then
    timeout 120 kaggle competitions submissions -c kaggriculture --csv 2>/dev/null | python3 -c "
import csv, sys
skip = {l.split()[0] for l in open('$C/HOSTED-EXCLUDE.txt') if l.strip()}
rows = [r for r in csv.DictReader(sys.stdin) if 'COMPLETE' in (r.get('status') or '') and r['ref'] not in skip][:${OWN_N:-3}]
for r in rows: print(r['ref'], r['fileName'].replace('.tar.gz', ''))" > $f.tmp
    bump "$C/KAGGLE-REQUESTS-$(date -u +%F).txt" 2
    [ -s $f.tmp ] && mv $f.tmp $f
  fi
  cat $f
}
declare -A SLUG=([16732748]=dsm [16915014]=boey [16681125]=mmpq [16770421]=vadim [16623559]=decem [16718819]=majkel
                 [16730612]=umg [16712258]=yizhou)
while true; do
  t0=$(date +%s); backoff=0
  lb=$(ls -t top30-losses/LEADERBOARD-*.csv | head -1)
  if [ $(( t0 - $(stat -c %Y "$lb") )) -ge ${LB_EVERY:-3600} ]; then
    if ! nice -n 10 python3 -B refresh_lb.py --counter "$C/KAGGLE-REQUESTS-$(date -u +%F).txt" --cap "$CAP" >> refresh-lb.log 2>&1 \
       && tail -30 refresh-lb.log | grep -q "429\|Too Many Requests"; then
      echo "$(date -u +%FT%TZ) leaderboard refresh HTTP 429: backing off $BACKOFF s"; sleep "$BACKOFF"; continue
    fi
    lb=$(ls -t top30-losses/LEADERBOARD-*.csv | head -1)
  fi
  mapfile -t TEAMS < <(python3 - "$lb" "$TOP" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1], encoding='utf-8-sig')) if int(r['TeamId']) != 16624026]   # us
print('\n'.join(r['TeamId'] for r in rows[:int(sys.argv[2])]))
PY
)
  if [ -n "${AUTO_TOP_PRIO:-}" ] && [ ${#TEAMS[@]} -ge "$AUTO_TOP_PRIO" ]; then
    # owner 09-29 22:58: "top 3, plus random mixes from top 12 for other 2" (live leaderboard order, refreshed hourly)
    PRIO_TEAM=${TEAMS[0]}; EXTRA_PRIO="${TEAMS[@]:1:$((AUTO_TOP_PRIO - 1))}"
    # owner 09-30 01:5x: "put a preferred random list": PREFER_TEAMS (e.g. Boey, CDE, the teams that beat MMPQ) are PREFER_W times
    # more likely in the random pick; they are eligible even outside the top 12; the live top 3 stay excluded from the pool.
    mapfile -t SEL < <( { for t in "${TEAMS[@]:$AUTO_TOP_PRIO}"; do echo "$t"; done
                          for t in ${PREFER_TEAMS:-}; do [ "$t" = "$PRIO_TEAM" ] || [[ " $EXTRA_PRIO " == *" $t "* ]] || for _ in $(seq ${PREFER_W:-4}); do echo "$t"; done; done; } \
                        | shuf | awk '!seen[$0]++' | head -n "${TEAMS_PER_CYCLE:-2}")
  else
  for x in $EXTRA_PRIO $PRIO_TEAM; do TEAMS=("${TEAMS[@]/$x}"); done; OTH=(${TEAMS[@]})
  ROT=$(cat .rot-index 2>/dev/null || echo 0); NT=${#OTH[@]}; SEL=()
  for i in $(seq 0 $(( ${TEAMS_PER_CYCLE:-3} - 1 ))); do [ $NT -gt 0 ] && SEL+=(${OTH[$(( (ROT + i) % NT ))]}); done
  echo $(( (ROT + ${TEAMS_PER_CYCLE:-3}) % (NT > 0 ? NT : 1) )) > .rot-index
  fi
  if [ "${ORDERED:-0}" = 1 ]; then
    # owner 09-30 08:4x "prioritize top 10", "in order, whatever is missing", "strictly in terms of completion": after our own games,
    # fill strictly in live leaderboard order #1..#TOP; each team is finished (all missing public games) before the next rank;
    # completion is checked on the cached per-team listing (REUSE_MIN); no prio/extra/random picks.
    mapfile -t TEAMS < <(python3 - "$lb" "$TOP" <<'PY'
import csv, sys
rows = sorted((r for r in csv.DictReader(open(sys.argv[1], encoding='utf-8-sig')) if int(r['TeamId']) != 16624026), key=lambda r: int(r['Rank']))
print('\n'.join(r['TeamId'] for r in rows[:int(sys.argv[2])]))
PY
)
    PRIO_TEAM=none; EXTRA_PRIO=""
    # owner 09-30 12:4x "sample a little bit", "for the lower ones to give more diversity", "some in the top 20": LOW_N random teams of
    # ranks LOW_FROM..LOW_TO get LOW_PER each from a reserved LOW_N*LOW_PER of TARGET; ranks 1..TOP and our own games get the rest.
    mapfile -t LOW_SEL < <(python3 - "$lb" "${LOW_FROM:-11}" "${LOW_TO:-21}" <<'PY' | shuf | head -n "${LOW_N:-2}"
import csv, sys
rows = sorted((r for r in csv.DictReader(open(sys.argv[1], encoding='utf-8-sig')) if int(r['TeamId']) != 16624026), key=lambda r: int(r['Rank']))
print('\n'.join(r['TeamId'] for r in rows[int(sys.argv[2]) - 1:int(sys.argv[3]) - 1]))
PY
)
    # coordinator 14:5x (owner "a lot of other people just updated too ... interact with the current agent"): the LOW sample prefers
    # ranks 11-40 teams whose last submission is under RECENT_H hours old (most recent first), then fills randomly as before
    RECENT=($(python3 - "$lb" "${RECENT_H:-3}" <<'PY'
import csv, sys, datetime as dt
now = dt.datetime.now(dt.timezone.utc); h = float(sys.argv[2])
rows = sorted((r for r in csv.DictReader(open(sys.argv[1], encoding='utf-8-sig')) if int(r['TeamId']) != 16624026), key=lambda r: int(r['Rank']))
rec = []
for r in rows[10:40]:
    try:
        t = dt.datetime.fromisoformat(r['LastSubmissionDate'].replace(' ', 'T')).replace(tzinfo=dt.timezone.utc)
    except Exception:
        continue
    if (now - t).total_seconds() < h * 3600: rec.append((t, r['TeamId']))
print(' '.join(i for _, i in sorted(rec, reverse=True)))
PY
))
    if [ ${#RECENT[@]} -gt 0 ]; then NL=("${RECENT[@]:0:${LOW_N:-2}}"); for t in "${LOW_SEL[@]}"; do [ ${#NL[@]} -ge ${LOW_N:-2} ] && break; [[ " ${NL[*]} " == *" $t "* ]] || NL+=("$t"); done; LOW_SEL=("${NL[@]}"); fi
    TEAMS+=("${LOW_SEL[@]}")
    # owner 13:0x "gotta add them back", "their newest submission", "nvidia's team is usually one of top": FETCH-POLICY.json front_teams
    # go right after #1 each cycle with front_max each (read every cycle; no restart needed to change)
    # coordinator 17:5x (R-FETCH-EVERYCYCLE): FRONT = live ranks 4-5 (newest leaderboard) + FETCH-POLICY front_teams, deduped, in that order
    FRONT=($(python3 - "$lb" <<'PY' 2>/dev/null
import csv, json, sys
rows = sorted((r for r in csv.DictReader(open(sys.argv[1], encoding='utf-8-sig')) if int(r['TeamId']) != 16624026), key=lambda r: int(r['Rank']))
ids = [r['TeamId'] for r in rows[3:5]] + [str(t) for t in json.load(open('FETCH-POLICY.json')).get('front_teams', [])]
print(' '.join(dict.fromkeys(ids)))
PY
))
    FRONT_MAX=$(python3 -c "import json;print(json.load(open('FETCH-POLICY.json')).get('front_max',8))" 2>/dev/null || echo 8)
    # coordinator 13:5x (adversarial check): front teams go AFTER #1-#3, so the owner order #1, #2, #3, ours (OWN_RANK 4) holds
    if [ ${#FRONT[@]} -gt 0 ]; then NT=(); for t in "${TEAMS[@]:0:3}"; do [[ " ${FRONT[*]} " == *" $t "* ]] || NT+=("$t"); done; NT+=("${FRONT[@]}"); for t in "${TEAMS[@]:3}"; do [[ " ${FRONT[*]} " == *" $t "* ]] || NT+=("$t"); done; TEAMS=("${NT[@]}"); fi
    echo "$(date -u +%FT%TZ) cycle teams (ordered): ${TEAMS[*]} (front: ${FRONT[*]:-} max ${FRONT_MAX:-}; low sample: ${LOW_SEL[*]})"
  else
  echo "$(date -u +%FT%TZ) cycle teams: prio=$PRIO_TEAM extra=[$EXTRA_PRIO] random=[${SEL[*]}]"
  TEAMS=($PRIO_TEAM $EXTRA_PRIO ${SEL[@]})
  fi
  own=0; got=0; R=$(( ${#LOW_SEL[@]} * ${LOW_PER:-2} )); left=$((TARGET - R)); lowadd=0
  [ "${ORDERED:-0}" = 1 ] || own_step
  if [ $backoff = 1 ]; then echo "$(date -u +%FT%TZ) HTTP 429 in own step: stopping this cycle, next in $BACKOFF s"; TEAMS=(); fi
  k=$(( ${#TEAMS[@]} - 1 - $(echo $EXTRA_PRIO | wc -w) )); pos=0
  for id in "${TEAMS[@]}"; do
    pos=$((pos + 1))
    islow=0; [[ " ${LOW_SEL[*]} " == *" $id "* ]] && islow=1
    [ $islow = 1 ] && [ $lowadd = 0 ] && { left=$((left + R)); lowadd=1; }
    if [ "${ORDERED:-0}" = 1 ] && [ $pos = $(( ${OWN_RANK:-4} )) ]; then   # owner 08:5x: "prioritize our own games at rank 4"
      own_step; [ $backoff = 1 ] && { echo "$(date -u +%FT%TZ) HTTP 429 in own step: pausing 90 s, then continuing with the teams (coordinator 13:5x)"; backoff=0; sleep 90; }
      [ $left -le 0 ] && [ $R = 0 ] && break
    fi
    if [ $islow = 1 ]; then per=$(( left < ${LOW_PER:-2} ? left : ${LOW_PER:-2} ))
    elif [[ " ${FRONT[*]:-} " == *" $id "* ]]; then per=$(( left < FRONT_MAX ? left : FRONT_MAX ))
    elif [ "$id" = "$PRIO_TEAM" ] || [[ " $EXTRA_PRIO " == *" $id "* ]]; then per=$PRIO_MAX
    elif [ "${ORDERED:-0}" = 1 ]; then per=$left
    else per=$(( (left + k - 1) / k )); [ $per -gt $PER_TEAM_MAX ] && per=$PER_TEAM_MAX; k=$((k - 1)); fi
    [ $per -le 0 ] && { [ $R = 0 ] && break; continue; }
    out=copy-${SLUG[$id]:-team-$id}
    day=$(date -u +%F)
    echo "$(date -u +%FT%TZ) team $id -> $out, max $per, counter $(cat $C/KAGGLE-REQUESTS-$day.txt 2>/dev/null || echo 0)"
    timeout 1800 nice -n 10 python3 -B save_team.py --team-id "$id" --out "$out" --counter "$C/KAGGLE-REQUESTS-$day.txt" \
      --cap "$CAP" --max-replays "$per" --active-only 2 $([ "${ORDERED:-0}" = 1 ] && echo --newest-sub-first) --reuse-listing-min $({ [ "${ORDERED:-0}" = 1 ] && [ $pos = 1 ]; } || [[ " ${FRONT[*]:-} " == *" $id "* ]] && echo ${REUSE_TOP1_MIN:-15} || echo ${REUSE_MIN:-30}) > "$out.slow-$(date -u +%H%M).log" 2>&1
    last=$(ls -t "$out".slow-*.log | head -1)
    f=$(grep -oE 'fetched [0-9]+' "$last" | tail -1 | grep -oE '[0-9]+'); f=${f:-0}
    { [ "$id" = "$PRIO_TEAM" ] || [[ " $EXTRA_PRIO " == *" $id "* ]]; } || left=$((left - f)); got=$((got + f))
    echo "$(date -u +%FT%TZ) team $id: $(grep -E '^packed|^public games' "$last" | tr '\n' ' ' | cut -c1-160)"
    if grep -q "http 429\|429 Client Error\|Too Many Requests" "$last"; then
      echo "$(date -u +%FT%TZ) HTTP 429: stopping this cycle, next in $BACKOFF s"; backoff=1; break
    fi
  done
  if [ "${ORDERED:-0}" = 1 ] && [ $backoff = 0 ] && [ $left -gt 0 ] && [ $pos -lt ${OWN_RANK:-4} ]; then own_step; fi
  if [ $backoff = 0 ] && [ "${BEATERS_PER_CYCLE:-4}" -gt 0 ] && [ -s $C/.own-subs.txt ]; then
    bo=$(cd $REPO && timeout 600 nice -n 10 python3 -B work/runs/maintainer/pbc-autosubmit/beaters.py --subs $(cut -d' ' -f1 $C/.own-subs.txt) \
         --max-calls ${BEATERS_PER_CYCLE:-4} --pause 3 --cap "$CAP" 2>&1 | grep -v '^ok ' | tail -1)
    echo "$(date -u +%FT%TZ) beaters: $bo"
    echo "$bo" | grep -q "429\|Too Many Requests" && backoff=1
  fi
  echo "$(date -u +%FT%TZ) cycle done: fetched $got of target $TARGET (own $own) in $(( $(date +%s) - t0 )) s"
  [ "${ONE_CYCLE:-0}" = 1 ] && break
  if [ $backoff = 1 ]; then sleep "$BACKOFF"; else s=$(( CYCLE - ($(date +%s) - t0) )); [ $s -gt 0 ] && sleep "$s"; fi
done
