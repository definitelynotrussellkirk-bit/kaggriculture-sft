# Model 1: B-even

54.6M-parameter encoder-decoder transformer trained only by SFT on winning seats. It was the stronger of the two final submissions (`n1-main-sprinkle`). Architecture and training story: [main write-up](../README.md#model-1-b-even). Every feature, column by column: [FEATURES.md](../FEATURES.md).

## Read the code

Flat reading copies of these files, same bytes and line numbers: [`../code/model1/`](../code/model1/) and [`../code/features/`](../code/features/). The links below go to the as-run tree.

All paths are under [`code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/) unless noted. That directory is the full working tree at the time, so it also holds experiments (`grow_*`, `*_v7.py`), roster lists and tests. The files below are the ones the final model used.

| What | File |
|---|---|
| Model: entity tokens → encoder ×10 → decoder ×8 → command/order/STOP heads | [`model.py`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/model.py) |
| Training loop, losses, Muon + AdamW, EMA, LR schedule | [`train.py`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/train.py) (later stages swap in the overlays under `code/overlay-*`) |
| Replay → per-step examples and labels | [`data.py`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/data.py), [`codec.py`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/codec.py) (joint action ↔ tokens) |
| Precomputed per-step features | [`step_cache.py`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/step_cache.py) |
| Feature solvers | [`flip.py`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/flip.py) (no-action forecasts), [`calc_features.py`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/calc_features.py) (market calculations), [`job_eta.py`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/job_eta.py) (job routes), [`prefix.py`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/prefix.py) (same-step prefix), [`noop_mask.py`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/noop_mask.py) (execution masks), [`v8.py`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/v8.py) and `v8_*.py` (feature families), [`v8_lattice_net.py`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/v8_lattice_net.py) (space-time lattice CNN) |
| Head regroup 36 × 12 → 18 × 24, same tensors | [`code/b-even/make_init.py`](code/b-even/make_init.py), [`head_surgery.py`](code/b-even/head_surgery.py) |
| BC + distillation phase (T = 1.5) | [`code/phase1-kd/oa3/losses.py`](code/phase1-kd/oa3/losses.py), [`cache_train.py`](code/phase1-kd/oa3/cache_train.py) |
| Last-day sprinkle: 4 updates, lr 1e-5, newest #1-team wins | [`code/launch/sprinkle/b_wash_sprinkle.sh`](code/launch/sprinkle/b_wash_sprinkle.sh) |
| EMA export, archive build and cold verify | [`ema_export.py`](code/trainer-v6/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc/ema_export.py), [`code/export/build_archive.py`](code/export/build_archive.py) |
| Inference as submitted | [`../submissions/n1-main-sprinkle/`](../submissions/n1-main-sprinkle/). Its `model.py` and `agent.py` are the serving versions and differ from the training ones here; the feature files are identical. |

## Rebuild it

[STAGES.md](STAGES.md) lists every stage from random init to the shipped archive: input and output checkpoint sha256, roster, exact command, hardware, updates, logged time and gaps.

| Folder | What |
|---|---|
| `code/` | byte-identical copies of what ran; an `overlay-*` folder replaces files in a copy of `trainer-v6` |
| `rosters/` | every roster each stage trained on (gzipped) |
| `records/` | run records, receipts, metrics logs |
| `SHA256SUMS` | hashes of every file here |
