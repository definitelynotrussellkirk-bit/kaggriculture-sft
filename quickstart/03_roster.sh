#!/bin/bash
# Step 3: training rosters for this episode. Imitation targets are winning seats only.
#   bash 03_roster.sh EPISODE_ID DAY
# roster-model1.tsv: the winner (format the Model 1 step-cache loader reads: archive member day eid seat weight reason).
# roster-model2.tsv: the winner as an action row, the loser as a WDL-only row (no imitation loss; value head only).
source "$(dirname "$0")/env.sh"
EID=$1; DAY=$2
"$PY" -I - "$RUN" "$EID" "$DAY" <<'EOF'
import json, sys, zipfile
run, eid, day = sys.argv[1], int(sys.argv[2]), sys.argv[3]
zipf = f"{run}/episodes/kaggriculture-episodes-{day}-{eid}.zip"
r = json.loads(zipfile.ZipFile(zipf).read(f"{eid}.json"))
money = [float(f["money"]) for f in r["steps"][-1][0]["observation"]["farms"]]
if money[0] == money[1]:
    sys.exit("draw: no winning seat to imitate")
win = int(money[1] > money[0])
with open(f"{run}/roster-model1.tsv", "w") as f:
    f.write("archive\tmember\tday\teid\tseat\tweight\treason\n")
    f.write(f"{zipf}\t{eid}.json\t{day}\t{eid}\t{win}\t1\tquickstart:win\n")
with open(f"{run}/roster-model2.tsv", "w") as f:
    f.write("path\teid\tseat\tweight\ttier\twdl\tact_loss\tfinal_cash\topp_cash\tbucket\n")
    for seat in (win, 1 - win):
        wdl, act = (2, 1) if seat == win else (0, 0)
        f.write(f"{run}/stepcache/{eid}-{seat}.stp\t{eid}\t{seat}\t1.0\t1\t{wdl}\t{act}\t{money[seat]}\t{money[1 - seat]}\t\n")
print(json.dumps({"winner_seat": win, "team": r["info"]["TeamNames"][win], "rosters": [f"{run}/roster-model1.tsv", f"{run}/roster-model2.tsv"]}))
EOF
