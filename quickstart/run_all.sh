#!/bin/bash
# All steps for one episode: bash run_all.sh EPISODE_ID DAY
set -euo pipefail
cd "$(dirname "$0")"
bash 01_get_replay.sh "$1" "$2"
bash 02_features.sh "$1" "$2"
bash 03_roster.sh "$1" "$2"
bash 04_train_model1.sh
bash 05_train_model2.sh
bash 06_archive_model2.sh
