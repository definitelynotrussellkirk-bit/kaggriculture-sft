# Model 2: N flow

7.25M-parameter graph + memory model, trained from random init in about 11 hours on the last day and submitted as `nflow-main-wash8014`. Each step the game state becomes a 320-node typed graph. Relational blocks run over it, three GRU memories (fast, slow, per-worker) carry state through the game, and a 2-layer decoder emits commands, orders and STOP one token at a time. Story: [main write-up](../README.md#model-2-n-flow). Every feature, column by column: [FEATURES.md](../FEATURES.md).

## Read the code

Flat reading copies of these files, same bytes and line numbers: [`../code/model2/`](../code/model2/). The links below go to the as-run tree.

| What | File |
|---|---|
| Design notes written before it was built | [`code/DESIGN.md`](code/DESIGN.md) |
| Model: typed graph, relation-biased attention, memories, sequential decoder, WDL/next/opp aux heads | [`code/nf_model.py`](code/nf_model.py) |
| State → graph nodes and columns (per-seat shard format) | [`code/nf_format.py`](code/nf_format.py), [`code/COLUMNS.json`](code/COLUMNS.json), [`code/DATA-FORMAT.md`](code/DATA-FORMAT.md) |
| Streaming whole games in 24-step chunks (truncated BPTT), bucket sampling | [`code/nf_data.py`](code/nf_data.py) |
| Training loop, Muon + AdamW, EMA, recipe-change guard | [`code/train.py`](code/train.py) |
| Roster from the wins-only bucket roster | [`code/roster.py`](code/roster.py) |
| Wash seed (raw := EMA) and raw export (EMA := raw) | [`code/make_t4_seed.py`](code/make_t4_seed.py), [`code/make_raw_as_ema.py`](code/make_raw_as_ema.py), [`code/ema_export.py`](code/ema_export.py) |
| Feature solvers the graph reads | shared with Model 1: [`../model1-b-even/README.md`](../model1-b-even/README.md#read-the-code) |
| Inference as submitted | [`../submissions/nflow-main-wash8014/`](../submissions/nflow-main-wash8014/). Its `nf_format.py` is the serving variant; `nf_model.py` is identical to `code/`. |

## Rebuild it

[STAGES.md](STAGES.md): random init (seed 17) → segments S0-S9 on a Colab G4 → 300-update wash on an RTX 4090 → raw export → fp16 archive, with rosters, bucket probabilities, code pins, commands and gaps for each.

| Folder | What |
|---|---|
| `code/` | the trainer as of the final segments and the wash, plus build, seed, export and release scripts |
| `code-history/` | earlier pinned versions, only files that changed |
| `rosters/` | every roster each segment trained on |
| `records/` | run records, Colab `train.log`, wash and release receipts, environment freezes |
| `deploy/` | the serving harness the archive was built from |
| `SHA256SUMS` | hashes of every file here |
