## agent-play: agent.py, play.py, build_archive.py

Pinned interpreter prefix used below:
`cd /home/russ/Desktop/farmNEAT && CUDA_VISIBLE_DEVICES= PYTHONPATH=src:work/runs/maintainer/primitive-bc:work/cache/pytest-path OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 work/venv/python -B`

- `agent.py`: `make_agent(spec, seed=, temperature=, threads=, log=)` -> callable
  `(observation, configuration) -> action`. CPU only, lazy load on the first call.
  Observation -> `bc.actor_observation`; configuration -> the `observation.CONFIG` keys +
  `marketParams` (same values `data.pack_config` gives `pack`). Calls
  `PolicyNet.act(clean, generator=, temperature=, state=, configuration=)` and keeps the
  returned `ThinkState` per seat; a new episode (step 0, or a step that does not advance)
  starts from `none`. Every action passes `sanitize_action` (plain JSON; a unit's item must
  be a string and its count an int, or the engine's unit phase can raise; unparsable
  market ops such as the codec's `NOOP` are kept because the slot still counts). Any
  exception -> PASS, `fallbacks += 1`; a failed load is not retried. `log=True` prints
  `[pbc] step=<t> seat=<s> ms=<ms> fb=<0|1>` per step to stderr.
  Specs: `stub[:seed]` (torch-free random grammar policy, harness tests), `random[:seed]`
  (random `model.PolicyNet(default_config())`), a `primitive-bc-weights-v1` file, or a
  training checkpoint (`model.load_policy`, then a generic `{config, model}` dict).
- `play.py`: complete games on the exact engine vs the SHA-pinned league (PASS,
  SHOP_ROUTER, K0006), either seat. Records per-step latency (first call separate: it
  includes the load), own/rival cash, W/D/L, fallbacks, and an engine no-op audit
  (engine functions replayed on a copy; statuses listed in the module docstring; the
  audit's post-market cash must equal the engine's, mismatching steps are counted).
  `--archive X.tar.gz` plays the archive cold. `--shadow-driver K0006` times our agent on
  K0006's observations (realistic hand counts; our actions discarded).
  `play.py --model random:0 --opponents PASS,SHOP_ROUTER,K0006 --seats 0,1 --jobs 6 --out R.json`
- `build_archive.py build --model M --out A.tar.gz [--storage fp32|fp16|int8] [--threads N]`:
  deterministic tar.gz: `main.py` (lazy load, PASS on any exception, one stderr line per
  step `[pbc] step=<t> ms=<ms> fb=<0|1>`, `agent` is the last callable), `policy.pt`
  (int8 = per-row symmetric int8 storage for float tensors with >= 4096 elements,
  dequantised to fp32 at load; fp16 otherwise), `MANIFEST.json` (member sha256/bytes),
  and the repository files a subprocess opened or imported while the agent played traced
  steps (repository-relative paths, so `parents[3]` source pins resolve). Refuses > 100 MB.
  Then it plays one cold game (disable with `--no-verify`).
  `build_archive.py verify --archive A --opponents PASS,K0006 --seats 0,1 --threads 2 --out V.json`:
  fresh `python -S -B` process, cwd = extraction, PYTHONPATH = extraction + torch/numpy
  site only, `main.py` exec'd without `__file__` and the last callable taken as the agent.
  `complete` requires 719 steps, 719 contiguous log lines, no hard errors and no agent
  module loaded from outside the extraction.
- Tests: `... -m pytest -q -p no:cacheprovider work/runs/maintainer/primitive-bc/test_agent.py work/runs/maintainer/primitive-bc/test_play.py work/runs/maintainer/primitive-bc/test_build_archive.py`

## model-train: model.py, train.py, launch-train.sbatch

- `model.PolicyNet(ModelConfig())` (vocabulary from `codec.py`). Encoder: pre-LN transformer over
  `data.features` rows (masked rows + knownness). Thinking field (`think="bottleneck"`, default):
  C=4 codes from K=64 per cell of both farms (200 cells, located from each TILE row's role/x/y)
  and per global slot (G=64), straight-through Gumbel while training, `sample`/`argmax` at
  inference, no auxiliary targets. Global slot g is written from the g-th fixed state entity in
  pack order (GLOBAL, FARM x2, STOCK x17, MARKET x9, SHOP x8, CONTROL = 38), then own carriers,
  then free tokens (`slot_anchor="entities"`; `"free"` = G free tokens). The previous step's codes
  enter the next step at the same cells/slots (`none` at episode start). Decoders see only code
  embeddings + farm/row/col or slot embeddings, the unit's x/y and own+4-neighbour cell codes
  (`local`), all G slot codes (`global_read`), and the decisions already made; order: farmer,
  hands, orders to STOP (11th slot STOP only); quantity bin after its token. `think="off"`:
  decoders read encoder states, nothing carried.
- `forward(rows, token_mask, knownness, unit_pos, unit_prefix[B,U,2], order_prefix[B,11,2], field=)`
  -> dict of unit/order token and quantity logits, `value`, `field`, `code_logits`.
  `act(obs, generator, temperature, state, configuration) -> (action, ThinkState)`; the field
  resets at step 0 or when the step does not advance. `decode_free` is a K/V-cached sequential
  decoder (one query token per decision). `load_policy(path)` reads train checkpoints and
  `PolicyNet.save` files. `latency(net, n_units=)`, `size_report()`.
- Default = `SIZES["4m"]` (width 192, 6 layers, 6 heads, 2 decoder layers, 4,854,842 params).
  `train.py bench` (1 thread, synthetic N=250 + 64 slots, greedy, all 11 order slots decoded,
  30 repeats, 1-min load 11.7 on this 32-thread host) median / p90 ms:

  | size | params | 1 unit | 2 units | 13 units | act() on a real replay (pack+model+codec, 120 steps) |
  | --- | --- | --- | --- | --- | --- |
  | 1m | 1,171,802 | 10.9 / 11.1 | 11.8 / 12.6 | 16.1 / 19.0 | 17.0 / 20.1, max 21.7 |
  | 4m | 4,854,842 | 28.8 / 30.6 | 30.9 / 32.4 | 39.0 / 42.9 | 38.1 / 41.2, max 44.6 |
  | 12m | 14,374,874 | 84.5 / 93.1 | 86.1 / 91.9 | 95.5 / 102.0 | 97.5 / 106.6, max 116.9 |

  An earlier run of the same bench at load ~40 gave about 2x these times.
- `train.py fit`: `data.iter_windows` (W=24 scored steps after up to B=24 no-grad burn-in steps,
  field carried in both) -> seeded shuffle buffer -> AdamW, warmup + cosine, bf16 autocast only
  on CUDA, per-update Gumbel tau schedule (`--tau-start 2 --tau-end 0.5`, exp, recorded in
  `RUN-*.json`), atomic `ckpt-*.pt` + `LATEST`, `--resume`, `--stop-after`, `metrics.jsonl`
  (per-head accuracy, unit/order/step exact, code usage: distinct argmax codes, per-position
  entropy, softmax entropy), held-out eval every `--eval-every`. Losses at step t+k backprop
  through the carried codes (no detach inside a window). `--grad-checkpoint` recomputes each step.
  `--usage-reg` (default 0) = per-position `log K - H(mean_batch p)`. Resume restores model,
  optimizer, Gumbel generator and window count exactly; the real-data stream restarts with a
  new shuffle seed (`--data-resume skip` replays it, exact only with `--workers 0`).
  `train.py eval --checkpoint DIR` adds free-running greedy exact accuracy.
  `train.py probe` / `train.py bench --off --replay-archive ZIP` give the numbers here.
- Carried-field probe (untrained 4m, 2 real windows of 2026-09-22, N<=260, fp32): 655 MB of
  saved activations per window (no checkpointing); gradient norm at the codes written on the
  first scored step from later losses 0.93; by lag 1/2/4/8/16/23: 0.93, 0.057, 7.4e-4, 1.1e-7,
  2.9e-15, 8.1e-22.
- Synthetic fit (CPU, tiny 32-wide model, 300 updates, 32 held-out windows): unit token accuracy
  0.19 -> 0.66, order token 0.13 -> 0.79, whole-step exact 0.00 -> 0.16.
- `launch-train.sbatch` (not submitted): gpu-4090b, pinned runtime, RUN.py storage run; needs
  `RUN_ROOT` (replica incl. SELECTION.json), `ARCHIVE_GLOB` visible from the node, optional
  `OUT`, `WORKERS`, `TRAIN_ARGS`; resumes when `OUT/LATEST` exists.
- Tests: `... -m pytest -q -p no:cacheprovider work/runs/maintainer/primitive-bc/test_model.py work/runs/maintainer/primitive-bc/test_train.py`

## codec-data: codec.py, data.py

- `codec.py`: `UNIT_VOCAB` (44) / `ORDER_VOCAB` (23, incl. `STOP`, `NOOP`) generated by
  `vocab_from_grammar(GRAMMAR)`; `GRAMMAR` is a frozen copy of `engine_grammar()` (AST of
  `_apply_unit_action`, `_parse_order`, the `_process_market` lockstep item tables) and the
  tests assert equality with the engine source and module. `MAX_UNITS=25` (observed max 16
  hands over 13,045 episodes; engine bound 24 hands at the largest observed cash 195,700).
  `MAX_ORDERS=10`, 75 `QTY_BINS`: counts 1..64, 66, 68, 72, 78, 80, 85, 89, 90, 99, 100 and one
  fraction 1/1 of the reference (`max(1, (2*num*ref+den)//(2*den))`). References are
  actor-visible at t: SELL/PICKUP own shed stock, PLACE the acting unit's carried amount,
  BUY_SEED/BUY_ANIMAL floor(cash/engine price), BUY_PRODUCT floor(cash/visible price).
  `canonical(action, obs)` is engine-equivalent (docstring lists every rule: malformed ->
  PASS/NOOP, counts clamped at the shed capacity 100, mid-queue NOOPs kept for lockstep
  positions, extra hand entries' PLANT demand applied with visible seeds).
  `encode_step(action, obs[, reasons, details])` / `decode_step(labels, obs)`.
  CLI: `codec.py measure --all` -> `VOCAB.json` (forms, quantities, bin search, label stats,
  MAX_UNITS evidence); `codec.py coverage --per-day 16` -> `COVERAGE.json` (forward grid,
  backward token x bin engine acceptance, market bounds, replay round trip, exact-engine
  state equality of decoded vs recorded actions).
- `data.py`: `iter_examples(archives, split=, rank_max=, seats=, workers=)`,
  `iter_windows(..., window=24, burn_in=24)`, `collate`, `collate_windows` (scored columns
  start at `burn_in`; `step_valid`, `loss_mask`, `first_valid`, `step`), `features(obs, cfg)`
  (the one observation -> input function; float16 rows), `count_examples`, `benchmark`.
  Split: sha256 bucket of the episode id, train 90 / val 5 / test 5 (`heldout` = val+test);
  the 16 sealed confirmation ids are removed by member name. Workers use `spawn` by default
  (safe under a torch parent). CLI: `data.py count`, `data.py bench --workers 1 4 8`.
- Tests: `... -m pytest -q -p no:cacheprovider work/runs/maintainer/primitive-bc/test_codec.py work/runs/maintainer/primitive-bc/test_data.py`
