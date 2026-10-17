# Quickstart: one replay → features → training

Takes one Kaggle replay through the same code that built the two final models: feature extraction, roster, a few training updates of each model from random init, and (optionally) a submission archive played for a full game. It runs on CPU in about two minutes. The shapes are tiny, so this shows the pipeline working, not a strong model; the real runs are in [`model1-b-even/STAGES.md`](../model1-b-even/STAGES.md) and [`model2-n-flow/STAGES.md`](../model2-n-flow/STAGES.md).

Needs Python 3.12 with `requirements.txt`, plus `zip`. Outputs go to `$RUN` (default `qs-run/` at the repo root). The repo's code is copied into `$RUN/code` and run from there; nothing in the repo is modified.

```
pip install -r quickstart/requirements.txt
bash quickstart/run_all.sh 115987062 2026-09-30      # or the steps below one at a time
```

| # | Command | Produces | Code that does the work |
|---|---|---|---|
| 1 | `bash 01_get_replay.sh EPISODE_ID DAY` | `episodes/kaggriculture-episodes-DAY-EID.zip` holding `EID.json` | `kaggle competitions replay EID` (your own `kaggle.json`); set `REPLAY_JSON=file` to use a replay you already have |
| 2 | `bash 02_features.sh EPISODE_ID DAY` | `stepcache/EID-0.stp`, `EID-1.stp`: every decision step of each seat with all input features and the action labels | `step_cache.py build --v6 --v8` → `data.py` (replay → per-step examples), `src/farm_neat/policy/observation.py` (entity tokens), `calc_features.py` + `theorem_features.py`, `prefix.py`, `job_eta.py`, `flip.py`, `v8.py` + `v8_*.py` (feature families), `codec.py` + `noop_mask.py` (action labels) |
| 3 | `bash 03_roster.sh EPISODE_ID DAY` | `roster-model1.tsv` (winning seat), `roster-model2.tsv` (winner as action row, loser as value-only row) | the script itself (winner = more money at the end) |
| 4 | `bash 04_train_model1.sh [UPDATES]` | `model1/ckpt-*.pt`, `metrics.jsonl` | `make_practice_scratch.py` (random-init seed), `train.py fit --step-cache`, `model.py` (Model 1) |
| 5 | `bash 05_train_model2.sh [UPDATES]` | `nfz/EID-*.nfz` (graph layout), `model2/ckpt-*.pt`, `train.jsonl` | `nf_format.py build` (step cache → 320-node graph), `train.py`, `nf_data.py`, `nf_model.py` (Model 2) |
| 6 | `bash 06_archive_model2.sh` | `model2-archive/quickstart-nflow.tar.gz`, `VERIFY.json` | `ema_export.py`, `build_archive.py build` / `verify` (fresh process, full game vs a PASS opponent) |

To train on more games, repeat steps 1-2 per episode and add rows to the rosters. The real rosters are in `data/` and in each model folder.

## What the test run produced

Episode 115987062 (2026-09-30, M & M & P & Q beat DECEM 107,550 to 104,168), `QS_LOCAL=1`, CPU, all six steps in 122 s. An `strace` of the whole run showed no file opened under `/mnt`.

| Step | Result |
|---|---|
| 1 | One-episode zip; winner seat 0 |
| 2 | Two `.stp` files in 34 s, **byte-identical** to the same seats in the step cache the shipped models trained on (`base-v1-stepcache-v11`) |
| 3 | Both rosters written |
| 4 | Model 1, 1,336,523 parameters (width 96, 2 layers), 30 updates in 27 s. Loss 55.9 at update 1 → 38.5 at update 30 (mean of first 5: 53.0, last 5: 44.0); unit-token accuracy 0.025 → 0.22 |
| 5 | Two `.nfz` shards; Model 2, 311,018 parameters (width 64, 1 block), 60 updates in about 10 s. Updates 1-6 score nothing: a stream that starts mid-game is warm-masked (memory only) until that game ends. Over the 54 scored updates, mean of first 5 vs last 5: unit-token accuracy 0.08 → 0.23, WDL loss 1.06 → 0.68, next-action 3.77 → 3.35, opponent 3.71 → 3.35; imitation loss stays noisy (8.3 → 9.3), since each update sees a different phase of one game |
| 6 | 774 KB archive, 38 files; one 719-step game in a fresh process, 0 fallbacks, `complete: true`. A 3,000-3,000 draw: 60 updates on one game do not learn to farm |

## Things the wrappers work around

- **NAS paths.** The copied code is byte-identical to what ran and still names `/mnt/ai/scratch/farmNEAT/...`. When that path does not exist (or with `QS_LOCAL=1`), `env.sh` rewrites it in the work copy to the repo's `model1-b-even/code/trainer-v6/`, which holds the leaderboard snapshot `data.py` reads. This is the same rewrite `run_arm.sh` did on training nodes without the NAS.
- **Feature code tree.** Step 2 runs `step_cache.py` from `model1-b-even/code/trainer-v6`. Its packing sources, including `v8_lattice.py`, are byte-identical to `data/step-cache/code` and to the hashes pinned in `data/step-cache/CACHE.json`.
- **Seed RNG.** `make_practice_scratch.py` copies a CUDA generator state from a checkpoint that is not shipped (`--rng-from`); step 4 passes a fresh CPU generator state instead.
- **Opponent league.** `play.py` imports the opponent league, which is not in this repo; `shim/league.py` provides the PASS opponent `verify` needs.
