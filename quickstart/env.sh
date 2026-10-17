# Sourced by every step. Copies the repo's code into $RUN/code (the repo itself is never modified).
# The copied code is byte-identical to what trained the models and still names the original NAS paths. When that NAS is
# not mounted (or QS_LOCAL=1), the work copies get the same rewrite base-v1-pack/run_arm.sh used on nodes without it:
# /mnt/ai/scratch/farmNEAT/ -> the repo's model1-b-even/code/trainer-v6/, where the leaderboard snapshot data.py reads lives.
set -euo pipefail
QS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$QS")
RUN=$(realpath -m "${RUN:-$REPO/qs-run}")
PY=${PY:-python3}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4}
CODE=$RUN/code
TRAINER=$CODE/trainer-v6                                     # model1-b-even/: step cache builder, trainer, model, seed writer
PBC=$TRAINER/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc
NFLOW=$CODE/nflow                                            # model2-n-flow/: trainer, model, shard format
NFLOW_DEPLOY=$CODE/nflow-deploy/work/runs/maintainer/n-flow-deploy

_qs_prepare() {
  local stamp="$CODE/.ready"
  local key
  key="$(cd "$REPO" && find model1-b-even/code/trainer-v6 model1-b-even/code/seed model2-n-flow/code \
         model2-n-flow/deploy -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16) local=${QS_LOCAL:-auto}"
  [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$key" ] && return 0
  rm -rf "$CODE"; mkdir -p "$CODE/nflow-deploy"
  cp -r "$REPO/model1-b-even/code/trainer-v6" "$TRAINER"
  cp "$REPO/model1-b-even/code/seed/make_practice_scratch.py" "$TRAINER/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/"
  cp -r "$REPO/model2-n-flow/code" "$NFLOW"
  cp -r "$REPO/model2-n-flow/deploy/." "$CODE/nflow-deploy/"
  if [ "${QS_LOCAL:-0}" = 1 ] || [ ! -d /mnt/ai/scratch/farmNEAT ]; then
    grep -rl "/mnt/ai/scratch/farmNEAT/" "$CODE" --include=*.py | xargs -r sed -i "s#/mnt/ai/scratch/farmNEAT/#$TRAINER/#g"
  fi
  echo "$key" > "$stamp"
}
mkdir -p "$RUN"
_qs_prepare
