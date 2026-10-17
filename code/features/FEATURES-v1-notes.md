# Primitive-BC input features

Input: one seat's `bc.actor_observation` view with `step` plus its own submitted actions; features only, no masks. No rival private state, seed, future shop or outcome is read (`test_features.py::test_actor_view_only`).

## Schema: `features.FEATURE_SCHEMA = primitive-bc-features-v1` (`FEATURE_SCHEMA_SHA256`), 7564 values

| block | shape | source | known |
| --- | --- | --- | --- |
| theorem.global / .product / .structure / .tile | (46,) / (9,43) / (8,18) / (200,31) | `theorem_features.compute` ÷ `SCALES`; tile row = role*100+y*10+x | all |
| history.global_now / .global_lag / .prev_action | (1,27) / (4,22) / (1,24) | `history_features` blocks (already scaled) | per value |
| history.item_now / .item_lag | (12,18) / (12,36) | rows = observation.ITEMS | per value |

Values are divided by a fixed scale and never clipped. Unknown values are stored as 0 with known=False. The largest |value| seen over 2 episodes × 2 seats is 120.5 (wheat `buyable_now` = cash/10). Relative fp16 error is below 5e-4.

## Tokens: `feature_tokens.tokens(feats)` → `values`/`known` [22,128], `type_id` [22], `tile` [200,31]
- Columns 0–3 hold the type one-hot (GLOBAL, LAG, PRODUCT, PRODUCER), 4–15 the item one-hot over observation.ITEMS, 16–19 the lag one-hot, and 20+ the payload. Padding is 0 with known=False.
- Slot 0 is GLOBAL (theorem.global, history.global_now, history.prev_action). Slots 1–4 are LAG 1/4/24/72. Slots 5–13 are PRODUCT (theorem.product plus item history). Slots 14–21 are PRODUCER (theorem.structure, plus item history for the 3 animals).
- `tile` rows use PolicyNet's own cell index. `align_tiles(tile, pack token_ids)` gives the same rows aligned to `pack` rows.
- Ablation: `tokens(f, drop=PREV_ACTION_GROUPS)` zeroes the previous-own-action values and their knownness. The schema is unchanged.

## Per step (live and dataset, same code)
`fs = FeatureState(cfg)` per seat → `f = fs.step(actor_obs)` → `tok = tokens(f)` → act → `fs.record_action(wire_action)`.
Datasets must run steps 0..718 in order before slicing (`episode_features`), because of the 72-step lags and cumulative flow. Store `values` and `tile` as float16, like `data.py` rows. Call `features.warm()` once per process; it takes 35–48 ms.

## PolicyNet wiring: `feature_tokens.POLICYNET_PATCH`, applied by `test_feature_tokens.py` to the current `model.PolicyNet.encode`
- In `__init__`, add `self.feat = FeatureTokenInput(width)`: Linear(128,w), a no-bias Linear(128,w) for knownness, and a zero-initialised Linear(31,w) for tiles.
- Change 1: `encode(..., *, feat=None, ...)`. Change 2: after `self._tiles(...)`, compute `ftok, tile_add = self.feat(...)` and add `tile_add.gather(1, cell_of…) * is_tile` to the TILE tokens.
- Change 3: before `attn = ...`, append `ftok` last, after the G slot tokens, so `src` and `tile_index` stay valid, and extend `mask`.
- Change 4: with think=off, pool the value over `mask`. With `feat=None` the patched encode is output-identical (tested); no writer change is needed.
- `data.py`/`agent.py` are not edited here; their owner adds `feature_tokens.collate(...)` beside `rows`, one token set per window step.

## Cost (`bench_features.py`: one thread, 300 steps × 2 seats per run, load average 32–33 on 32 cores)
| episode, steps | p50 ms | p95 ms | p99 ms | max ms | >5 ms (with GC) | pack p50 (reference) |
| --- | --- | --- | --- | --- | --- | --- |
| 111424871, 419–718 | 1.20 | 1.54 | 3.94 | 8.07 | 2 (2) | 4.18 |
| 111424871, 0–299 | 1.11 | 1.47 | 2.13 | 3.64 | 0 | 3.98 |
| 111419545, 419–718 | 1.35 | 1.85 | 4.73 | 7.15 | 4 (4) | 4.54 |
| 111419545, 0–299 | 1.27 | 1.73 | 2.07 | 5.82 | 1 (0) | 4.12 |

Split of the p50: history update 0.60–0.69 ms, theorem compute 0.44–0.58, tokens 0.03, record_action 0.04. Every step above 5 ms except one had a Python cyclic-GC collection inside it; the remaining one (5.82 ms wall) is in a run whose largest thread-CPU step was 4.45 ms. Effect on BC accuracy or playing strength has not been tested.
