#!/bin/bash
# Step 1: one Kaggle replay -> a one-episode zip in the layout the pipeline reads (member <episode_id>.json).
#   bash 01_get_replay.sh EPISODE_ID DAY        DAY = the episode's UTC date, YYYY-MM-DD (from `kaggle competitions episodes`)
# Downloads with your own ~/.kaggle/kaggle.json. To use a replay you already have: REPLAY_JSON=path/to/replay.json.
source "$(dirname "$0")/env.sh"
EID=$1; DAY=$2
mkdir -p "$RUN/replays" "$RUN/episodes"
json=${REPLAY_JSON:-$RUN/replays/episode-$EID-replay.json}
if [ ! -f "$json" ]; then
  kaggle competitions replay "$EID" -p "$RUN/replays"
fi
zipf=$RUN/episodes/kaggriculture-episodes-$DAY-$EID.zip
tmp=$(mktemp -d); cp "$json" "$tmp/$EID.json"
rm -f "$zipf"; (cd "$tmp" && zip -q -9 "$zipf" "$EID.json"); rm -rf "$tmp"
"$PY" -I - "$zipf" "$EID" <<'EOF'
import json, sys, zipfile
zipf, eid = sys.argv[1], int(sys.argv[2])
r = json.loads(zipfile.ZipFile(zipf).read(f"{eid}.json"))
assert r["info"]["EpisodeId"] == eid and len(r["steps"]) == 720, "not a complete Kaggriculture episode"
money = [f["money"] for f in r["steps"][-1][0]["observation"]["farms"]]
print(json.dumps({"zip": zipf, "episode": eid, "teams": r["info"]["TeamNames"], "final_money": money,
                  "winner_seat": int(money[1] > money[0]) if money[0] != money[1] else None}))
EOF
