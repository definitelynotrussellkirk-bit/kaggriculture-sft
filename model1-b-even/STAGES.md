# Model 1 (B-even): training lineage, random init to shipped archive

Shipped archive: `n1-main-sprinkle.tar.gz`, sha256 `932bb45d7fe9f0ebb486356285b43c5d079761c3905ce650d2e99c9c77986952`, 94,080,075 bytes.
It holds `policy.pt`: the fp16 EMA slot of the raw-export checkpoint at update 46,484, with 54,639,683 parameters (374 tensors after the training-only aux heads are stripped).

Everything here is copied byte for byte from the training machines. The Kaggle episodes, the step cache built from them (`base-v1-stepcache-v11`, made by `data/`) and all checkpoints are not included. A rerun will not match bit for bit: data order depends on worker timing, the loader reseeds on every resume (`data_resume=reseed`), and training ran in bf16 with `torch.compile`.

Runtime for every local stage: Python 3.12.3, torch 2.10.0+cu128 (taken from the phase-1 receipt and the RUN records), NVIDIA RTX 4090.

## Lineage

| # | Stage | Trainer updates | Hardware | Code | Input sha256 | Output sha256 | Logged time |
|---|---|---|---|---|---|---|---|
| 1 | base1-b from random init, 9 chunks | 0 → 31,047 | 1× RTX 4090 (gpu-4090b) | `code/trainer-v6` (chunks 7-9; earlier chunks ran earlier revisions, see §1) | seed `1c331996…7d44` | `08a058ab…17a9` (ckpt-0031047) | 29.54 h |
| 2 | Regroup encoder heads 36×12 → 18×24 (same tensors), model := EMA | none | CPU | `code/b-even/make_init.py` | `08a058ab…` | `9ee7a671…755e` (beven-h18-init.pt) | none |
| 3 | Phase 1: 0.5 BC + 0.5 KD from the B u31047 EMA, T = 1.5 | 1,000 (runner) | RTX 4090 | `code/phase1-kd` + `code/trainer-v6` | `9ee7a671…` | last.pt `1b0d147f…7345` → p2-seed `bdace8a9…9842` (update set to 31,000) | 1,064.7 s |
| 4 | Phase 2 on the B trainer | 31,000 → 42,000 | RTX 4090 | `trainer-v6` + overlays, per job (§4) | `bdace8a9…` | `9f26af4e…bfae` (ckpt-0042000) | 9.21 h |
| 5 | Phase 2 continued on Colab | 42,000 → 46,480 | Colab G4 (RTX PRO 6000), two VMs | `trainer-v6` + `overlay-colab` | `9f26af4e…` | `abd33c91…a523` (ckpt-0046480) | 2.11 h (45,920-46,480 only) |
| 6 | Sprinkle: 4 updates at lr 1e-5 on 5 newest #1-team wins | 46,480 → 46,484 | RTX 4090 (gpu-skynet) | `code/trainer-v6` | `abd33c91…` | `d4809051…c513`; raw export `6f1eeff1…7092` | 85 s |
| 7 | EMA export + fp16 archive build | none | CPU | `trainer-v6/.../ema_export.py`, `code/export/build_archive.py` | raw `6f1eeff1…` | ema.pt `67a4104f…9363` → archive `932bb45d…` | 22 s build |

Every checkpoint sha256 above was recomputed from the file on the NAS and matches the hash its run record wrote at the time. Updates 31,047 → 33,362 of base1-b (chunk 10) and the separate Colab run of B from u31047 are not part of this lineage: `make_init.py` reads `base-v1/runs/b/ckpt-0031047.pt`, which has the same sha256 as `chunks/chunk9-ckpt-0031047.pt`.

The run names use N-updates (B-even updates), where N u = trainer u − 30,000. So the shipped model is n1-u16484.

"Logged time" is the sum of `seconds` in that stage's metrics log over the stage's update range.

## Code trees

| Directory | What | How it is verified |
|---|---|---|
| `code/trainer-v6/` | `base-v1-pack/nfs/`: `primitive-bc-v6-20260927` (trainer, model, features, step-cache loader) + `cohort-scale-20260923`, plus `run_arm.sh` | Content key over `*.py`/`*.json` = `41d40196d6171e65`, the key `run_arm.sh` wrote for the sprinkle's work copy. 44/44 non-test files in the phase-1 receipt match. `run_arm.sh` = `e309beba…` (pinned in the phase-2 records). |
| `code/overlay-bfx/` | `train.py` `2a41cec3…` (per-seat step-weight table support) | Matches `trainer_train_sha256` in EFFECTIVE-4308/4875 |
| `code/overlay-draws/` | `train.py` `1a5e83d2…` + `draw_log.py` `870831d2…` (bucket sampler with a draw log) | Matches `executed_sha256` in EFFECTIVE-5452…7318 |
| `code/overlay-colab/` | `train.py` `ca9c9f78…` (= draws + a fail-closed roster-completeness check + a reseed-only step-cache resume) | The tree `trainer-v6` + `overlay-draws/draw_log.py` + this `train.py` gives the bpreflight code hash `ce0ba972…` recorded at the first Colab launch (HANDOVER.log) |
| `code/phase1-kd/` | `oa3_lab.py` + package `oa3/` (KD runner) | `oa3/cache_train.py` = `621f2615…` (pinned). The other files are not pinned; their mtimes predate the run. |
| `code/seed/` | `make_practice_scratch.py` (writes the update-0 random-init seed), `build_pack.py` | Not pinned; mtime 09-28 05:01, before the seed was written at 06:05 |
| `code/b-even/` | `make_init.py`, `to_btrain.py`, `n_beven.sbatch` (phase 1 + phase 2 launcher), head-count check scripts | Latest revisions only |
| `code/launch/base1-b/chunk.sbatch` | Per-chunk launcher for stage 1 | Latest revision only (accepts CHUNK 1-12) |
| `code/launch/colab/` | Colab worker, launch, switch and relaunch scripts; `bpreflight.py`; `ckpt_select.py` | Latest revisions only |
| `code/launch/sprinkle/` | `t4w-bn-skynet.sbatch`, `b_wash_sprinkle.sh` (wash, raw export, build) | Not pinned; mtimes before the 19:05 run |
| `code/export/` | `hub_deploy_v9f.sbatch`, `build_archive.py`, `SOURCE-PINS.json`, `release_gate.py`, `structured_step0.py` | `build_archive.py` = `48e9213a…` = `builder_sha256` in the archive manifest |

An overlay replaces the named files in a copy of `trainer-v6`. The `train.py` overlays sit at `primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/`.

## 1. base1-b, update 0 → 31,047

The seed is `base-v1-pack/seeds/base1-b/ckpt-0000000.pt`, sha `1c331996774bb9c3…`. That hash prefix is in `ARM-ID`. `records/1-base1-b/SCRATCH.json` lists its parameters. The command below is rebuilt from that file; the exact command line was not recorded:

```
python make_practice_scratch.py base-v1-pack/seeds/base1-b --updates 1000000 --warmup 300 --lr 3e-4 --seed 101 \
  --batch-windows 128 --width 432 --layers 10 --heads 36 --ffn swiglu --dec-layers 8 --dec-heads 12 \
  --dropout 0.05 --feature-dropout 0.1 --v6 --v8 --aux-turn --aux-turn-w 0.045,0.035,0.015,0.0123,0.03,0.026 --optimizer muon
```

Model init uses `--seed 101`. The script copies only the auxiliary CUDA generator state from another checkpoint (`--rng-from`, default `pbc-colab-v6/seeds/control/ckpt-0048456.pt`), which is not included.

Each chunk is `chunk.sbatch` → `run_arm.sh base1-b END OUT` → `train.py fit`. `run_arm.sh` builds this command:

```
python3.12 -B train.py fit --out /mnt/ai/scratch/farmNEAT/base-v1/runs/b --device cuda --resume --stop-after R --updates END \
  --priority-file WORK/rows.tsv --priority-only true --archives <base-v1-pack/data/replays-*.bin> --workers 4 \
  --ckpt-every {100|500} --log-every 10 --grad-checkpoint false --compile-model true --pack-steps 256 \
  --step-cache /mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11 --cache-pool-seats 6 --eval-every 0 --keep-ckpts 6 \
  --batch-windows 128 --warmup 300 --grow-lr-mult 1 --decay-updates D --lr-schedule wsd --ema-decay 0.999 \
  --max-seconds S --mask-unit-noops true [EXTRA]
```

With `--step-cache`, training reads the step cache only. The roster is `$PBC_CACHE_ROWS`, or the cache's `rows.tsv` when that is unset (`step_cache.py:263`). The loader re-reads the roster on every pass and skips seats without a built cache file.

| Chunk | SLURM job | Updates | Recorded in RUN-n.json | Data roster | Env and flags (source) |
|---|---|---|---|---|---|
| 1 | 1190 | 0 → 2,981 | lr 3e-4, aux-scale 6.0, ckpt 100, max 13,500 s | step-cache `rows.tsv` | PBC env not recorded |
| 2 | 1195 | 2,981 → 6,238 | as chunk 1 | step-cache `rows.tsv` | PBC_PROFILE_AT=30 (PLAN.md) |
| 3 | 1223 | 6,238 → 10,286 | aux-scale 2.8; speed env ASYNC_CKPT, DEFER_ITEMS, EAGER_CHECKS | step-cache `rows.tsv` | PLAN.md: ATTN=flex, DEALIAS_GRADS, SEAT_FRACTION .5, DEC_TRIM, BUCKET_GROUP 3, JOBBIAS fold, EMA_EXTRA .998,.9995, HIST_DROPOUT .5, TURN_NORM, TURN_W_MOVE .5, TURN_W_ORDER 2, GRAD_CLIP 10, UPDATE_STATS 50, COS_EVERY 100 + COS_ORIGIN, PROFILE_AT 30 |
| 4 | 1231 | 10,286 → 14,487 | as chunk 3 | step-cache `rows.tsv` | chunk-3 env without PROFILE_AT (PLAN.md) |
| 5 | 1233 | 14,487 → 18,703 | as chunk 3 | `rows-anneal-c5.tsv` | PBC_CACHE_ROWS (PLAN.md, ledger) |
| 6 | 1235 | 18,703 → 22,708 | updates 22,708, decay 4,000 (WSD decay 18,708 → 22,708) | `rows-anneal-c6.tsv`, edited during the chunk | END_UPDATE=22708 DECAY_UPDATES=4000 |
| 7 | 1243 | 22,708 → 27,043 | lr 3e-5 constant, ckpt 500; full `pbc_env` recorded from here | `rows-c7.tsv`, refreshed hourly | EXTRA `--aux-scale 2.8 --lr 3e-5`, CKPT_EVERY=500 |
| 8 | 1247 | 27,043 → 30,300 | max 10,200 s | `rows-c7.tsv` | as chunk 7 |
| 9 | 1745 | 30,300 → 31,047 | lr 1e-5, max 2,400 s, PBC_UPDATE_STATS=50 | `rows-c7.tsv` | as chunk 7 |

Run records for chunks 7-10 contain the same `pbc_env` apart from PBC_CACHE_ROWS and UPDATE_STATS: ASYNC_CKPT 1, ATTN flex, BUCKET_GROUP 3, COS_ORIGIN (seed), DEALIAS_GRADS 1, DEC_TRIM 1, DEFER_ITEMS 1, EAGER_CHECKS 1, EMA_EXTRA 0.998,0.9995, GRAD_CLIP 10, HIST_DROPOUT 0.5, JOBBIAS fold, SEAT_FRACTION 0.5, TURN_NORM 1, TURN_W_MOVE 0.5, TURN_W_ORDER 2. Every chunk ran Muon, bf16 autocast, seed 101, 128 windows × 24 steps per update and EMA 0.999.

Code per chunk is not pinned. Inside the pack, `train.py` was last edited at 09-29 03:30 and `model.py` at 09-28 12:14. Chunks 7-9 began at 05:18 on 09-29 or later, so by mtime they ran the code in `code/trainer-v6`. Chunks 1-6 ran earlier revisions of `train.py`/`model.py` from the same directory. Their `.bak`/`.orig` snapshots stay on the NAS and are not copied. Which revision each chunk ran is not established.

Rosters in `rosters/1-base1-b/` (gzip):
- `stepcache-rows.tsv` and its three `.bak-20260928-*` predecessors: the cache roster edited during chunks 1-2 (mtimes 06:55, 08:49, 09:31, 10:17 on 09-28).
- `rows-anneal-c5.tsv` (+ `ANNEAL.json`).
- `rows-anneal-c6.tsv` with backups `.bak-0929-0130` and `.bak-0929-0235`. Weight-3 rows and hourly appends were added during chunk 6.
- `rows-c7.<sha12>.tsv`: every content-addressed rows-c7 version live between chunk 7's start and chunk 9's end (04:32, 05:32, 07:32, 09:32, 11:32 and 12:32 on 09-29).

Which version each worker pass read is not established.

## 2. Head regroup

`python make_init.py 18 /mnt/ai/scratch/farmNEAT/b-even/beven-h18-init.pt` sets `config.heads` to 18, copies the EMA into `model` and every `ema_extra`, and keeps B31's optimizer and RNG state. No tensor changes; width 432 / 18 = 24 per head. `records/2-b-even-phase1/BEVEN-CHECK*.json` hold the 12/18/36-head comparison made before the choice.

## 3. Phase 1 (KD)

This is SLURM job 4147 on gpu-4090b, 09-29 22:50-23:08. The 1,000 updates finished, but the job's `to_btrain.py` step then failed: torch was missing from PYTHONPATH (`logs/train-4147.out`). The conversion was run again at 23:11. `p2-seed.pt` records `beven_phase1 = {runner_checkpoint: p1/seg0/train/last.pt, runner_update: 1000}` and `P1-DONE` holds its sha. The arguments come from `receipt.json` (`args`):

```
cd /mnt/ai/scratch/farmNEAT/oa3-astra/return && OA3_TRAINER=$BTR OA3_STEP_W_TSV=/mnt/ai/scratch/farmNEAT/oa3-astra/cont-step-w.94bc8331b271.tsv \
PYTHONPATH=$BTR:<trainer-v6>/primitive-bc-v6-20260927/src:<site> python3.12 -B tools-cont/oa3_lab.py --trainer $BTR train \
  --student /mnt/ai/scratch/farmNEAT/b-even/beven-h18-init.pt --student-component model \
  --teacher /mnt/ai/scratch/farmNEAT/base-v1/runs/b/chunks/chunk9-ckpt-0031047.pt --teacher-component ema \
  --cache /mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11 --frozen /mnt/ai/scratch/farmNEAT/oa3-astra/frozen-final \
  --out /mnt/ai/scratch/farmNEAT/b-even/p1/seg0/train --stop-at 2026-09-30T12:55:00-04:00 --max-seconds 50000 --updates 1000 \
  --microbatch 64 --effective-batch 512 --workers 3 --lr 3e-5 --min-lr 3e-5 --warmup 100 --clip 1 --alpha 0.5 --temperature 1.5 \
  --ema 0.999 --save-every 500 --unit-mask --pause-file /mnt/ai/scratch/farmNEAT/b-even/PAUSE
# BTR = <trainer-v6>/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc ; seed 930 ; fresh AdamW (0.9, 0.95), wd 0.05
python3.12 -B to_btrain.py p1/seg0/train/last.pt beven-h18-init.pt p2-seed.pt   # model/EMA from the runner, update := 31000
```

The data is `rosters/2-b-even-phase1/frozen-final-*` (11,196 train seats, 72,511 weighted; 300 validation seats) plus the step-weight table `cont-step-w.94bc8331b271.tsv`. The B RUN-10 PBC env was exported (receipt `pbc_environment`). `frozen_train_sha256` is `3491da7e…`.

## 4. Phase 2 on the 4090 (gpu-4090b)

Launcher: `n_beven.sbatch` (HEADS=18, INIT_SHA=9ee7a671…, ROWS=…). The run starts from `b-even/run/ckpt-0031000.pt` (a copy of p2-seed) and goes through `run_arm.sh base1-b 1000000 /mnt/ai/scratch/farmNEAT/b-even/run` with these flags:

```
--grad-checkpoint false --compile-model true --pack-steps 256 --step-cache /mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11 --cache-pool-seats 6
--eval-every 0 --keep-ckpts 6 --batch-windows 128 --warmup 300 --grow-lr-mult 1 --decay-updates 0 --lr-schedule wsd --ema-decay 0.999
--max-seconds <to 19:40 EDT> --mask-unit-noops true --lr 3e-5 --aux-scale 2.8        (WORKERS=6, CKPT_EVERY=500)
```

The env is B's RUN-10 `pbc_env` + `speed_env`, with PBC_CACHE_ROWS set to the roster. Each job resumed from the newest checkpoint (a multiple of 500):

| Job | Updates | Executed code | Roster (`rosters/3-b-even-phase2-4090/`) | Per-seat weights |
|---|---|---|---|---|
| 4308 | 31,000 → 35,000 | trainer-v6 + overlay-bfx (pinned) | `rows-b-final-clean.63e57662a18f.tsv` | `cont-step-w.94bc8331b271.tsv` (RUN-10) |
| 4875 | 35,000 → 35,500 | trainer-v6 + overlay-bfx (pinned) | `rows-b-final-clean.63e57662a18f.tsv` | as above (RUN-11) |
| 5181 | 35,500 → 37,000 | not recorded: the launcher revision then in place (backup `n_beven.sbatch.bak-draws`) picks plain `trainer-v6` for bucket rosters. EFFECTIVE-5181's train.py hash is the bfx file's, computed regardless of what ran. | `rows-buckets.9040e0bc225d.tsv` | none (RUN-12) |
| 5452 | 37,000 → 38,500 | trainer-v6 + overlay-draws (pinned) | `rows-buckets.ae8410f71ee3.tsv` | none; draw log |
| 6040 | 38,500 → 39,500 | trainer-v6 + overlay-draws (pinned) | `rows-buckets.8962d52f8bc2.tsv` | none |
| 6210 | 39,500 → 41,000 | trainer-v6 + overlay-draws (pinned) | `rows-buckets.2daa2a34a03f.tsv` | none |
| 6445 | 41,000 → 41,500 | trainer-v6 + overlay-draws (pinned) | `rows-buckets.426f06385d73.tsv` | none |
| 7318 | 41,500 → 42,000 | trainer-v6 + overlay-draws (pinned) | `rows-buckets.426f06385d73.tsv` | none |

`records/3-b-even-phase2-4090/` holds the EFFECTIVE-<job>.json, RUN-10…17.json and the metrics log. Job 7318 was cancelled at ckpt-0042000. That checkpoint was protected as `NBEVEN-final4090b-u42000.pt` (sha `9f26af4e…`) and handed to Colab.

## 5. Colab, update 42,000 → 46,480

The trainer is `train.py fit` from the worker scripts in `code/launch/colab/`, with 16 workers and step cache `/content/cache` (or `/content/bcache`). The code tar on the VM is `trainer-v6` + `overlay-colab`: the first launch's bpreflight hash is `ce0ba972`, and the same content is in `c1-code.tar`, which was built for the 19:29Z relaunch. The tar staged on nflow-x1 is not pinned. Torch is 2.10.0+cu128 / numpy 2.4.3 (pinned by VM setup); the VM Python version is not established. Hosts: `oldarch-g4` (Colab G4), whose runtime was replaced twice, then `nflow-x1` (Colab G4).

| Launch (UTC, 09-30) | Host | From checkpoint (sha256) | Roster (`rosters/4-b-even-colab/`) | Recipe |
|---|---|---|---|---|
| 16:19:25 | oldarch-g4 | 42000 `9f26af4e` | `rows-buckets.426f06385d73` (in phase-2 folder) | batch 128 windows, EMA .999, EMA_EXTRA .998/.9995, SEAT_FRACTION .5, pool seats 6, ckpt 500 |
| 16:52:37 | oldarch-g4 | 43000 `bb4d6918` | `rows-buckets.d4623759c8fe` | same |
| ~17:23 | oldarch-g4 | 44000 `18a3cc9b` | `rows-buckets.050183bc7ef3` | same |
| 18:09:39 | oldarch-g4 | 45500 `6907fafa` | `rows-buckets.050183bc7ef3` | batch 1024 windows, EMA .996, EMA_EXTRA .992/.998, SEAT_FRACTION .5, pool seats 24, ckpt 40 |
| 19:29:55 | oldarch-g4 (new runtime) | 45680 `983c9315` | `rows-buckets.60327cd812e7` | batch 1024, EMA .992, EMA_EXTRA .984/.996, SEAT_FRACTION .25, pool seats 24, ckpt 40 |
| 20:17:15 | oldarch-g4 | 45880 `91bf0a43` | `rows-t1wash-b.8f08a69ea889` | same |
| 20:46:48 | nflow-x1 | 45920 `cc45bc83` | `rows-t1wash-b.a9ab27e1ba44` | same |
| 22:11:18 | nflow-x1 | 46280 `d4c1321c` | `rows-t1wash-b.d2d605843046` (404 seats; #1 team .80, #2 .14, #3 .06) | same (`records/4-b-even-colab/COMMAND.json`) |

Every launch used lr 3e-5 (WSD, decay 0), aux-scale 2.8, pack 256, compile on, and the RUN-10 PBC env with the overrides above. ckpt-0046480 (`abd33c91`) was pulled at 22:58:42Z.

Not in the lineage:
- A switch at 18:56Z stopped training at 45720. The relaunch failed, the runtime was recycled and 45720 was lost.
- A re-switch at 20:33Z stopped at 45960. The runtime was lost and 45960 was not used.

Sources: `LAUNCH.log`, `SWITCH-*.log`, `RELAUNCH-*.log`, `HANDOVER.log`, `PULL.log` and `SHA256SUMS` in `records/4-b-even-colab/`.

The recipe for the first four launches comes from the run ledger and the backups of `worker.template.py`. Neither is copied, and those launches' `COMMAND.json` was overwritten. Metrics for 42,000-45,920 were not retained; `metrics.jsonl` covers 45,930-46,800.

## 6. Sprinkle, update 46,480 → 46,484

```
WASH_UPDATES=4 b_wash_sprinkle.sh n1-main-sprinkle \
  /mnt/ai/scratch/farmNEAT/pbc-colab-v6-results/colab-n1-beven-colab/ckpt-0046480.pt abd33c912886e5efc10f501b62197b78cecdd9b62540178e158d8d491507a523 \
  /mnt/ai/scratch/farmNEAT/broad/t1/wash/rows-b-newonly-1905.tsv 1e-5 plain
# -> sbatch --export=ALL,ROWS=..,LR=1e-5,RUN=t4w/runs/n1-main-sprinkle,UPDATES=4,MAX_S=136 t4w-bn-skynet.sbatch   (SLURM job 9560, gpu-skynet)
# -> run_arm.sh base1-b <U> RUN --grad-checkpoint false --compile-model true --pack-steps 256 --step-cache .../base-v1-stepcache-v11
#      --cache-pool-seats 4 --eval-every 0 --keep-ckpts 3 --batch-windows 128 --warmup 300 --grow-lr-mult 1 --decay-updates 0
#      --lr-schedule wsd --ema-decay 0.999 --max-seconds 136 --mask-unit-noops true --lr 1e-5 --aux-scale 2.8   (WORKERS=4)
```

The invocation line was not recorded. `WASH_UPDATES=4` is inferred from `max_seconds` 136 (= 4 × 4 + 120) in the checkpoint's train_config and from `train.log` (`"end": 46484, "remaining": 4`). The env is B's RUN-10 env with PBC_CACHE_ROWS = the roster, and the trainer is `code/trainer-v6` (content key `41d40196d6171e65`).

The roster `rosters/5-sprinkle/rows-b-newonly-1905.tsv` (sha `4763fbf5…`) has 5 winning seats from 09-30, each with weight 1. The "mirror 4x" text in `EFFECTIVE-wash.json` is a fixed string in the script, not a property of this roster.

Raw export: the inline Python in `b_wash_sprinkle.sh` copies `model` into `ema` and into every `ema_extra`. The shipped weights are therefore the trained weights after update 46,484, not an average.

## 7. Archive

```
sbatch hub_deploy_v9f.sbatch t4w/runs/n1-main-sprinkle/raw/ckpt-0046484.pt n1-main-sprinkle greedy     # job 9579
#  ema_export.py RAW ema.pt
#  build_archive.py build --model ema.pt --out n1-main-sprinkle.tar.gz --storage fp16 --temperature 0 --no-verify --decode greedy
#  build_archive.py verify (2 cold games vs PASS, both seats) ; structured_step0.py ; release_gate.py check
```

`build_archive.py` copies the inference sources (the `primitive-bc-v1-deploy-v9-decode` tree and `src/farm_neat`) into the archive. Its `MANIFEST.json` lists each member's sha256. `records/6-archive/` holds the build record, VERIFY (0 fallbacks), STEP0 and the release gate (34 source pins verified).

## Path mapping

| Path in records and scripts | In this repo |
|---|---|
| `/mnt/ai/scratch/farmNEAT/base-v1-pack/nfs/`, `~/farmneat-local/fs/base-v1-pack/nfs/`, `~/farmneat-local/fs/pbc-work-*/` | `code/trainer-v6/` |
| `.../base-v1-pack-bfx/nfs/` | `code/trainer-v6/` + `code/overlay-bfx/` |
| `.../base-v1-pack-draws/nfs/`, `~/farmneat-local/fs/base-v1-pack-draws/` | `code/trainer-v6/` + `code/overlay-draws/` |
| Colab `/content/code`, `/content/bcode` | `code/trainer-v6/` + `code/overlay-draws/draw_log.py` + `code/overlay-colab/` |
| `.../oa3-astra/tools-cont/` | `code/phase1-kd/` |
| `.../b-even/*.py`, `n_beven.sbatch` | `code/b-even/` |
| `.../pbc-colab-v6-results/launch-ncolab/`, `.../broad/bpreflight.py`, `.../broad/ckpt_select.py` | `code/launch/colab/` |
| `/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/chunk.sbatch` | `code/launch/base1-b/` |
| `/home/russ/Desktop/farmNEAT/work/runs/maintainer/pbc-colab/make_practice_scratch.py` | `code/seed/` |
| `.../t4w/` | `code/launch/sprinkle/` |
| `.../tools-0929/`, `.../tools-0929/repo-snap/.../build_archive.py` | `code/export/` |
| `.../base-v1-stepcache-v11/rows*.tsv`, `.../broad/`, `.../base-v1-oa3plan/`, `.../oa3-astra/frozen-final`, `cont-step-w.*.tsv` | `rosters/<stage>/` |
| `.../base-v1-stepcache-v11/` (`*.stp`), `.../practice-v1/data/` (`replays-*.bin`), roster `archive` column zips | built from Kaggle episodes by `data/`; not here |
| `.../runtime/python/bin/python3.12` | Python 3.12.3 + torch 2.10.0+cu128 |

## Not included

- Checkpoints (1.1 GB each), the step cache (~780 GB), replay archives, draw logs, `update_stats.jsonl`.
- `base-v1/runs/b/train.log` (21 MB) and `b-even/run/train.log` (6.4 MB). The metrics logs are included, gzipped.
- `.bak`/`.orig` code snapshots: the earlier revisions used by chunks 1-6, phase-2 job 5181's launcher, and the Colab worker templates for launches 1-4.
- The run ledger the Colab table was read from. Every fact taken from it is restated above.

## Not established

- Which code revision each of chunks 1-6 ran. Their PBC env is known only from the PLAN.md text, not from a run record.
- The executed code for phase-2 job 5181 (35,500 → 37,000).
- The code tar on `nflow-x1` (45,920 → 46,480) and the Python version on the Colab VMs.
- The exact seed-creation command line, and the auxiliary CUDA RNG state it copied.
- The exact roster version read by each worker pass while rosters were edited during chunks 1-2, 6 and 7-9.
- Training metrics for 42,000 → 45,920.
