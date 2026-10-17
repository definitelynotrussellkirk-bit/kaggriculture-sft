# Code: reading copies

The files that matter, flat and grouped, so you can read them without digging through the as-run trees. Each file is a byte-identical copy of what ran (`bash code/check_copies.sh` verifies all of them against [`MANIFEST.tsv`](MANIFEST.tsv), which also gives each file's as-run path). Line numbers match the originals. To run anything, use [`../quickstart/`](../quickstart/), which runs the as-run trees.

Imports refer to the original layout (`farm_neat.*`, sibling modules), so these copies are for reading, not importing.

## Read in this order

1. [`features/observation.py`](features/observation.py): game state → entity tokens
2. [`features/flip.py`](features/flip.py), [`features/calc_features.py`](features/calc_features.py), [`features/job_eta.py`](features/job_eta.py): the main feature solvers
3. [`features/prefix.py`](features/prefix.py): what lets one decision at a time work
4. [`model1/model.py`](model1/model.py) or [`model2/nf_model.py`](model2/nf_model.py): the networks
5. [`model1/train.py`](model1/train.py) or [`model2/train.py`](model2/train.py): the training loops

Every feature column with its solver: [`../FEATURES.md`](../FEATURES.md). Every technique with links into these files: [`../HOW-IT-WORKS.md`](../HOW-IT-WORKS.md).

## `features/`: shared by both models

| File | What it does |
|---|---|
| [`observation.py`](features/observation.py) | Actor-visible game state → variable-length entity tokens (tile, worker, stock, market, shop, constant, farm, clock), 80 columns each plus known flags |
| [`data.py`](features/data.py) | Replay → per-step examples: each seat's view at step t, labelled with the action that seat sent |
| [`codec.py`](features/codec.py) | Joint action ↔ tokens: one unit token (+ quantity bin) per worker in order, then up to 10 market orders, then STOP |
| [`noop_mask.py`](features/noop_mask.py) | Exact per-unit execution masks: which commands would actually do something |
| [`flip.py`](features/flip.py) | Exact no-action forecasts: what every tile and animal does if nobody acts (weeds, decay, escapes, yields), duty counts, worker reachability |
| [`calc_features.py`](features/calc_features.py) | Market calculations: projected end inventory, clearing price of the next unit, value of one more producer |
| [`job_eta.py`](features/job_eta.py) | Nine-job route facts: resource ETAs, deadlines, route masks |
| [`prefix.py`](features/prefix.py) | Same-step prefix: the exact own state after the earlier workers' commands this step, fed to each decoder query |
| [`history_features.py`](features/history_features.py) | Causal history from the actor's own view: clock, farms, own stock, lags |
| [`theorem_features.py`](features/theorem_features.py) | Proven game facts as inputs (clocks, cash, labour, land, shed, liquidation, per product/tile/structure) |
| [`features.py`](features/features.py), [`feature_tokens.py`](features/feature_tokens.py) | theorem + history features as one schema, and as 22 extra input tokens |
| [`v8.py`](features/v8.py) | Assembles the v8 families into per-token and per-worker columns |
| [`v8_cohorts.py`](features/v8_cohorts.py) | Production cohorts, scheduled receipts, working capital to first sale, realisable equity, price trends |
| [`v8_dusk.py`](features/v8_dusk.py) | Dusk load: what is lost at the day-end deadline if nothing changes, return-to-shed feasibility |
| [`v8_flags.py`](features/v8_flags.py) | Calendar, shop-reveal timeline and demand belief, action-effect receipts, net product ledger, binary flags |
| [`v8_gaps.py`](features/v8_gaps.py) | Cheap inputs analysts kept asking for: cohort ages and holdings for both sides, one-unit market impact, producer/buy/sell/land/hire columns |
| [`v8_tileclock.py`](features/v8_tileclock.py) | Per-tile crop clock, season cliffs, service value, fertiliser opportunity |
| [`v8_lattice.py`](features/v8_lattice.py) | Both 10×10 farms as a grid with per-cell channels at t, t-1, t-2 |
| [`step_cache.py`](features/step_cache.py) | Precomputes all of the above once per seat (one `.stp` file per seat), so training doesn't pay ~9 ms per step for features |
| [`FEATURES-v1-notes.md`](features/FEATURES-v1-notes.md) | Notes on the v1 feature schema and its cost |

## `model1/`: B-even, 54.6M

| File | What it does |
|---|---|
| [`model.py`](model1/model.py) | Encoder over entity tokens, decoder that emits commands → orders → STOP one token at a time (`decode_teacher`, `decode_free`, `act`) |
| [`train.py`](model1/train.py) | Training loop: windows of contiguous steps, losses, Muon + AdamW, EMA, LR schedule. Phase 2 swapped in small variants of this file (see [`../model1-b-even/STAGES.md`](../model1-b-even/STAGES.md)) |
| [`v8_lattice_net.py`](model1/v8_lattice_net.py) | CNN over the space-time lattice, scattered onto the tile tokens |
| [`head_surgery.py`](model1/head_surgery.py), [`make_init.py`](model1/make_init.py) | Regroup 36 × 12 attention heads into 18 × 24 without changing the function, then start B-even from it |
| [`kd_losses.py`](model1/kd_losses.py) | Distillation losses for the BC + KD phase |
| [`b_wash_sprinkle.sh`](model1/b_wash_sprinkle.sh) | Last-day top-up on the newest #1-team wins, RAW export, archive build |
| [`ema_export.py`](model1/ema_export.py), [`build_archive.py`](model1/build_archive.py) | EMA export; Kaggle archive build and cold-process verify |
| [`serving/`](model1/serving/) | What ran on Kaggle: `agent.py` (observation → action), `model.py` (serving version), `step_search.py` (best-of-k whole-step decoding, shipped but off) |

## `model2/`: N flow, 7.25M

| File | What it does |
|---|---|
| [`DESIGN.md`](model2/DESIGN.md) | The design, written before it was built |
| [`nf_model.py`](model2/nf_model.py) | Typed resource-flow graph, relation-biased attention, fast/slow/worker GRU memories, sequential decoder, WDL/next/opp aux heads |
| [`nf_format.py`](model2/nf_format.py), [`COLUMNS.json`](model2/COLUMNS.json), [`DATA-FORMAT.md`](model2/DATA-FORMAT.md) | Step cache → per-seat graph shards (`.nfz`) and which columns each node type reads |
| [`nf_data.py`](model2/nf_data.py) | Streams whole games in 24-step chunks (memory carried across chunks), bucket draws |
| [`train.py`](model2/train.py) | Training loop: cross-entropy on winning seats, value-only rows for losers, Muon + AdamW, EMA |
| [`roster.py`](model2/roster.py) | Wins-only bucket roster → N flow roster |
| [`make_t4_seed.py`](model2/make_t4_seed.py), [`make_raw_as_ema.py`](model2/make_raw_as_ema.py), [`ema_export.py`](model2/ema_export.py) | Wash seed (raw := EMA), RAW export (EMA := raw), weights export |
| [`serving/`](model2/serving/) | What ran on Kaggle: `agent.py`, `model.py` (single-observation play path), `nf_format.py` (serving variant) |

## `evaluation/`: paired games

| File | What it does |
|---|---|
| [`crn_rng.py`](evaluation/crn_rng.py) | The shop fix: weed rolls keyed by (seed, day, farm, x, y) and shop unlocks by (seed, day), so a world seed fixes the shops whatever either player does |
| [`league_game.py`](evaluation/league_game.py) | One game between two archives, both seats cold-loaded, replay rebuilt and checked |
| [`gate_freezer.py`](evaluation/gate_freezer.py) | Writes the frozen gate manifest (archives, seeds, bar) before any game |
| [`panel_league.py`](evaluation/panel_league.py) | Reducer: which games count, pair lines, board |
| [`tplan.py`](evaluation/tplan.py) | World scores over both seat orders, the gate's GPU-served rule |
| [`tally.py`](evaluation/tally.py) | W-D-L, world share and margin with 95% intervals (written for this repo; reproduces the final numbers) |
