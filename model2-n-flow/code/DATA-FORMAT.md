# N flow data format: nflow-shard-v1

Reference encoder: `nf_format.py` (`compact_seat` / `compact_tokens` / `compact_units` / `compact_labels`). The trainer
(from shards or from .stp on the fly) and the deploy agent (from `data.features` at play time) call the same functions.

## Source and alignment
- Input: v11 step-cache seats `/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11/<eid>-<seat>.stp` (primitive-bc
  `data._episode_task` with v6 + v8 keys; pairing actor observation[t] -> the seat's action stored at steps[t+1],
  as primitive-bc-v31-code/data.py). Packing sources (data.py, codec.py, prefix.py, v8.py, ...) are byte-identical to
  the B deploy `primitive-bc-v1-deploy-v9-decode` (sha256 prefixes in CACHE.json match).
- Roster: `/mnt/ai/scratch/farmNEAT/n-flow/roster/rows-nflow.1ad4d8296e84.tsv` (+ `.receipt.json`), built by
  `roster.py` from gpu-ops' wins-only `rows-broad.1ad4d8296e84.tsv` (exclusions of DATA-PLAN-0930.md already applied
  there) and catalog SEATS.parquet. Columns: path eid seat weight tier wdl act_loss final_cash opp_cash day archive member.
  - act_loss=1 rows: the 11,858 catalog-W seats, tiers 1/2/3 from the reason suffix, weight = roster weight
    (mass shares .241/.360/.399).
  - act_loss=0 rows (WDL-only): the opposite, losing seat of the same games when its .stp exists (2,423 at build
    time 01:2x; count grows as the cache backfills). No action/imitation loss, ever (test_wdl_only.py).
  - wdl = sign(final_cash - opp_cash) from the catalog: 2 W, 1 D, 0 L.

## File: `<eid>-<seat>.nfz`
numpy `.npz` (zip deflate), loaded with `allow_pickle=False`. `schema_json` = {"schema": "nflow-shard-v1",
"columns_sha256": ...}. Leading axis S = 719 steps (t = 0..718, `step` holds t). ~3.2 MB per seat (the .stp is ~11 MB).

| key | shape | dtype | meaning |
|---|---|---|---|
| node_present | S, 296 | bool | fixed node slot occupied |
| node_GLOBAL / FARM / TILE / CARRIER / STOCK / MARKET / SHOP / CONSTANT / CONTROL | S, n_type, c_type | f16 | per-type feature columns (COLUMNS.json) |
| carrier_pos | S, 50, 2 | i8 | x, y of own (0-24) and opponent (25-49) carriers by ordinal, -1 absent |
| unit_pos, unit_extra, x8_unit, reach_eta | S, 25, (2 / 12 / 41 / 100) | i8/f16/f16/i16 | own worker slots (farmer, hands in engine order) |
| unit_prefix, order_prefix, order_cash | S, 25, 27 / S, 11, 22 / S, 11, 82 | f16 | exact same-step prefix rows along the recorded action (prefix.teacher_arrays) |
| v6_job_deadline, v6_job_deadline_present | S, 100, 9 | i8 / bool | own-tile job deadlines |
| prev_unit, prev_order | S, 2, 25, 2 / S, 2, 11, 2 | i8 | own submitted action at t-1, t-2 (token, qty; hands only within their day) |
| unit_tok, unit_qty, order_tok, order_qty | S, 25 / S, 11 | i8 | codec labels of action[t+1]; -1 = none; orders end with STOP |
| unit_loss_keep, exact | S, 25 / S | bool | unit-noop-mask-v1 supervision mask; codec exactness |
| next_unit_tok, opp_unit_tok, opp_order_tok | S, 2, 25 / S, 3, 25 / S, 3, 11 | i8 | training-only targets (own t+1,t+2; opponent t..t+2) |
| eid, seat, wdl, final_cash, opp_cash, act_loss, tier, weight | scalar | | seat metadata |

Fixed node slots (296): 0 GLOBAL | 1-2 FARM own, opp | 3-202 TILE role*100 + y*10 + x | 203-252 CARRIER role*25 +
ordinal | 253-264 STOCK shed (12 items) | 265-269 STOCK seeds (5 crops) | 270-278 MARKET (9 products) | 279-286 SHOP
(ordinal) | 287-294 CONSTANT (5 crops, 3 animals) | 295 CONTROL.

Per-type columns (COLUMNS.json, `nf_format.py derive`, 48 seats, 16 per tier): indices into src = [rows 80 | extra 152 |
v6_token 8 | x8_token 354 | knownness 80], kept when the value varies across that type's tokens in the sample.
Widths: GLOBAL 180, FARM 47, TILE 168, CARRIER 67, STOCK 19, MARKET 182, SHOP 9, CONSTANT 33, CONTROL 1. A column
never seen varying in the sample is dropped identically in training and play.

## Build and verify
```
cd /mnt/ai/scratch/farmNEAT/n-flow/code        # ./sync.sh copies the desktop code dir here
PY=/mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12
R=/mnt/ai/scratch/farmNEAT/n-flow/roster/rows-nflow.1ad4d8296e84.tsv
$PY -B nf_format.py build $R /mnt/ai/scratch/farmNEAT/n-flow/shards-v1 --shard $i/$n      # one process per shard
$PY -B nf_format.py manifest $R /mnt/ai/scratch/farmNEAT/n-flow/shards-v1                  # MANIFEST.json
NFLOW_PBC_DIR=/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc \
  $PY -B test_data_alignment.py --shards /mnt/ai/scratch/farmNEAT/n-flow/shards-v1 --seats 10
```
Build cost: ~0.5 s .stp read + ~1.5-2 s compaction per seat on one CPU, < 1 GB RAM per process. BUILD-i-of-n.jsonl
logs every seat (errors are logged, never dropped silently). MANIFEST.json lists per-file sha256, bytes, tier and
act_loss, the roster sha256, the columns sha256, tier counts split into act rows and WDL-only rows, and missing files.
