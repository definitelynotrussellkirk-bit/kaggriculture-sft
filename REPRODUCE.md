# Reproducing the two final submissions

| Submission | Archive sha256 | Model |
|---|---|---|
| `n1-main-sprinkle` | `932bb45d7fe9f0ebb486356285b43c5d079761c3905ce650d2e99c9c77986952` | Model 1, B-even, 54,639,683 params |
| `nflow-main-wash8014` | `7661782af4ca815585945b019963b86586dbae490ba2554b592fdb080f0f4b3e` | Model 2, N flow, 7,252,244 params |

Both archives are in [`submissions/`](submissions/), byte-identical to what was submitted.

## What is and isn't included

- **Included:** every script that ran, copied byte for byte and checked against the sha256 each run recorded (`SHA256SUMS` in each folder); every roster each training stage read; run records, metrics logs and release receipts.
- **Not included:** the Kaggle episodes, the step cache built from them (~745 GiB), and checkpoints. Every roster names its seats by Kaggle episode id and seat, so the same games can be downloaded again.
- **Not bit-exact:** a rerun matches the recipe, not the weights. Data order depends on loader worker timing, loaders reseed on resume, training ran in bf16 with `torch.compile`, and Model 2's first two segments trained while its step cache was still filling.
- **Paths:** scripts keep the absolute paths they ran with (`/mnt/ai/scratch/farmNEAT/...`). Each folder has a path map. [`quickstart/`](quickstart/) shows how to run them from a checkout.

## Hardware

| Work | Hardware | Logged time |
|---|---|---|
| Step cache build | CPU SLURM shards, 1 process each | not recorded |
| Model 1, stages 1, 3, 4, 6 | 1× NVIDIA RTX 4090 (24 GB) | 29.5 h + 0.3 h + 9.2 h + 85 s |
| Model 1, stage 5 | Google Colab G4: NVIDIA RTX PRO 6000 Blackwell (96 GB), 48 vCPU | 2.1 h logged; updates 42,000-45,920 have no log |
| Model 2, S0-S9 | Colab G4 (same as above) | 11.3 h |
| Model 2, wash | 1× RTX 4090 | 304 s |
| Archive build, verification | CPU | under 1 min per archive |
| Kaggle inference | CPU, 1 thread, fp16 weights | slowest warm step over 2 cold-verify games: Model 1 212 ms, Model 2 67 ms; first step (load) 1.1-1.2 s and 0.76 s |

## Software

- Ubuntu 24.04 (local), Colab (Model 2 S0-S9 on Python 3.13).
- Python 3.12.3, torch 2.10.0+cu128, numpy 2.4.3, triton 3.6.0. Full list: [`requirements.txt`](requirements.txt); full freeze: `model2-n-flow/records/runtime/training-runtime-py312.freeze.txt`.
- The archives were verified under a Kaggle-like Python 3.11 with torch 2.6.0+cpu (`model2-n-flow/records/runtime/hosted-like-py311.freeze.txt`).
- Episode download uses the `kaggle` CLI with your own `~/.kaggle/kaggle.json`.

## Steps

| # | Step | Where | Output |
|---|---|---|---|
| 0 | One replay → features → a few training updates for each model | [`quickstart/README.md`](quickstart/README.md) | a working setup |
| 1 | Download episodes: Kaggle daily dumps (`kaggle/kaggriculture-episodes-<day>`), the replay API for 29-30 Sep and for teams that beat us, 6 hand downloads | [`data/PIPELINE.md`](data/PIPELINE.md) steps 1-6 | episode zips |
| 2 | Catalog: one row per seat with team, rank, day, result | `data/PIPELINE.md` step 7 | parquets and views |
| 3 | Step cache: one `.stp` per seat with all 719 steps' features and labels | `data/PIPELINE.md` step 8 (`step_cache.py build --v6 --v8`) | 74,857 `.stp` files |
| 4 | Rosters: which seats each stage trains on, with what weight. The exact rosters are shipped in `model1-b-even/rosters/` and `model2-n-flow/rosters/`; the builders are in `data/` | `data/PIPELINE.md` steps 9-15 | TSVs |
| 5 | Train Model 1: random init → base1-b u31,047 → head regroup → BC + distillation → phase 2 on a 4090, then on Colab → 4-update sprinkle | [`model1-b-even/STAGES.md`](model1-b-even/STAGES.md) | `ckpt-0046484.pt` |
| 6 | Train Model 2: random init → S0-S9 on Colab → 300-update wash | [`model2-n-flow/STAGES.md`](model2-n-flow/STAGES.md) (Model 2 also converts `.stp` to `.nfz` shards) | `ckpt-0056559.pt` |
| 7 | Export EMA → fp16 `policy.pt` → archive with `build_archive.py`, cold-verify | export sections of both STAGES.md | `.tar.gz` |
| 8 | Paired games on shared worlds to accept or reject a candidate | [`evaluation/README.md`](evaluation/README.md) | W-D-L, point share, margin |

## Known gaps

Each folder's doc lists its own. The ones that matter for a rerun:

- **Model 1, chunks 1-6 of base1-b:** the trainer was edited during these chunks, so only the version used for chunks 7-9 is pinned. The chunk 1-6 environment variables are known only from planning notes.
- **Model 1, Colab:** recipes for the first four launches come from ledger notes. Metrics for 42,000-45,920 were not kept.
- **Model 2:** S0 code is not pinned (a candidate copy is included). S9 has no run record; its arguments come from the checkpoint. S0-S9 command lines were rebuilt from saved arguments.
- **Data:** the builder versions for Model 1's two bucket rosters were overwritten later. Which rows-c7 version each Model 1 chunk read is not recorded. No script exists for `rows-b-newonly-1905.tsv`, the 5-game sprinkle roster; its rows match a recorded list of those five games.
- **Validation overlap:** the final pools include 131 seats with earlier validation flags, and the sprinkle roster includes one validation-split episode (115984030).

## Credits

- Some training games were rebuilt from the [`xishengfeng/kaggriculture-replay-db`](https://www.kaggle.com/datasets/xishengfeng/kaggriculture-replay-db) seed + action tapes (CC0), replayed through the engine and kept only when the final cash matched (`data/PIPELINE.md` step 6).
