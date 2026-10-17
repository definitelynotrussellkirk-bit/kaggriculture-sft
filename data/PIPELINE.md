# Data pipeline: Kaggle episodes → step cache → training rosters

Everything between the raw Kaggle replays and the rows the trainers read. The replays themselves are not shipped; every
roster names each seat by Kaggle episode id and seat, so the same games can be fetched again.

```
Kaggle daily dumps ─┐
episode API fetches ├─> episode store (zips) ─> catalog (SEATS, views) ─┬─> rosters (TSV: which seats, what weight)
manual downloads    │                                                   │
tape rebuilds ──────┘                                                   └─> step cache v11 (one .stp per seat: features + labels)
                                                                                   │
                                          trainers read roster rows, load each seat's .stp (N flow converts .stp to .nfz shards)
```

## Conventions

- Scripts are byte-identical copies of what ran. They keep their original absolute paths; see the path map below.
- `*.xz` files are xz-compressed records (`xz -dk FILE.xz`). `MANIFEST.tsv` lists every file with the sha256 and size of
  the uncompressed original and the path it was copied from.
- Roster format (trainer "priority file"): `archive  member  day  eid  seat  weight  reason`, one winning seat per row. The
  N flow wash roster uses its own format (`path eid seat weight tier wdl act_loss final_cash opp_cash bucket`, one row per
  seat, `.stp` path in column 1, losing seats as value-only rows).
- Runtime: Python 3.12, torch 2.10.0+cu128, numpy 2.4.3 (pinned in `step-cache/CACHE.json`); catalog tools ran on the
  system Python with pandas/pyarrow.

## Path map

| Original path | Here | Notes |
|---|---|---|
| `/mnt/ai/scratch/farmNEAT/primitive-bc-data/` | `acquire/`, `catalog/` (scripts only) | episode store: daily zips, `extra/<source>/archives/*.zip`, catalog parquets. Not shipped. |
| `/mnt/ai/scratch/farmNEAT/primitive-bc-v6-20260927/` | `step-cache/code/primitive-bc-v6-20260927/` | step-cache builder, its import closure and the data files it reads (39 files) |
| `/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11/` | `step-cache/`, `rows-c7/` (records) | 74,857 `.stp` files, ~745 GiB. Not shipped. |
| `/mnt/ai/scratch/farmNEAT/practice-v1/` (= `base-v1-pack/data`, `val`) | `practice-v1/` | `replays-*.bin` (zips of replays) not shipped |
| `/mnt/ai/scratch/farmNEAT/broad/` | `buckets/`, `t1-wash/`, `exclusions/` | |
| `/mnt/ai/scratch/farmNEAT/corpus-final/special/` | `buckets/build_top4_beaters.py`, `buckets/top4-beaters/` | |
| `/home/russ/Desktop/farmNEAT/` | `acquire/{hosted_fetch,beaters,seed_action_replays}.py`, `rows-c7/*.py`, `practice-v1/build_*.py` | project repo; `src/farm_neat/game/exact_engine.py` there is byte-identical to the copy under `step-cache/code/` |
| `/mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12` | — | Python 3.12 runtime used by SLURM jobs |

## Steps

| # | Step | Script(s), as run | Output | Consumed by |
|---|---|---|---|---|
| 1 | Official daily dumps | `acquire/ingest-daily.sh` (hourly loop): `kaggle datasets download -d kaggle/kaggriculture-episodes-<day>`; checks json count against `kaggle/kaggriculture-episodes-index` and the game config against `acquire/REFERENCE-CONFIG.json`. `acquire/daily_0929_retry.sh` retried the last day. | `primitive-bc-data/kaggriculture-episodes-<day>.zip`: 60 days, 07-30..09-28 (08-05 under `incoming/`). The 09-29 dump was still unpublished at 08:45 EDT 09-30 (12 attempts, `404`). | catalog |
| 2 | Episode API, top teams | `acquire/fetch_top_slow15.sh` (final loop: a cycle every 900 s, top-12 teams by live leaderboard plus priority teams, own games first, 429 back-off) → `acquire/save_team.py --team-id T --out copy-<slug> --counter … --cap …` (uses `fetch_topteam_replays.py`, `farm_top30_losses.py`); `acquire/FETCH-POLICY.json` | `extra/copy-<team>/{SUBMISSIONS,LISTINGS}-<stamp>.json`, `MANIFEST.jsonl`, `archives/kaggriculture-topteams-<day>-<stamp>.zip` (43 team folders). | catalog; t1/t1wash read `LISTINGS` for submission scores |
| 3 | Our hosted games and the games of teams that beat us | `acquire/hosted_fetch.py SUB OUT --n 400`, `acquire/beaters.py` (called from the loop in step 2) | `extra/hosted-ours`, `extra/beat-us`, `extra/beats-our-beaters` | bucket L (winners over us) |
| 4 | Leaderboard and top-30 losses | `acquire/refresh_lb.py`; `acquire/farm_top30_daily.sh` → `farm_top30_losses.py`; `acquire/run-topteam-daily.sh` → `fetch_topteam_replays.py` (earlier, 09-21..09-25) | `extra/top30-losses/LEADERBOARD-*.csv`, `extra/top30-losses/`, `extra/topteam/` | ranks in every bucket rule; early rosters |
| 5 | Hand downloads | Replays saved from the competition website, zipped one per game | `extra/manual-ingest/` (6 episodes; ids and hashes in `acquire/manual-ingest-SHA256SUMS`) | 4 of the 5 sprinkle games |
| 6 | Tape rebuilds | `acquire/seed_action_replays.py` (→ `extra/reconstructed/`); `catalog/seats.py rebuild-tapes` (→ `extra/reconstructed-xsf/`). Both replay the third-party `xishengfeng/kaggriculture-replay-db` seed+action tapes (CC0) through the exact engine and keep a game only if the final cash matches the tape. | official-format replay zips, no wall-clock field | v11 `rows.tsv` (21,386 of 40,909 rows), rows-c7 (3,430 of 21,132 rows) |
| 7 | Catalog | `catalog/refresh.sh` (hourly): `catalog.py refresh` → `seats.py build` → `seats.py facts` (exact re-play counters) → `seats.py rebuild-tapes` → `catalog.py update` → `seats.py build` → `tape_tags.py` | `catalog/{EPISODES,COPIES,SEATS,SEAT_DAYS}.parquet`, `TEAMS.tsv`, `views/*.tsv` (e.g. `WINS-2880.tsv`: winning seats rated ≥ 2880 from 09-03). Regenerable; not shipped. | every roster builder |
| 8 | Step cache v11 | `step-cache/slurm_shard.sbatch`: `cd primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc; python3.12 -B step_cache.py build "$ROWS" /mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11 --procs 1 --shard I/N --v6 --v8`. Initial list `step-cache/rows.tsv.xz` (40,909 seats), backfill list `backfill-ALL2.tsv.xz`; hourly new seats from `new_rows.py` via `cache_all_loop.sh`; every roster builder below also queued its uncached seats through the same sbatch. | `<eid>-<seat>.stp`, all 719 steps of one seat: legacy tokens + v6 history/aux + v8 families (dusk, cohorts, tileclock, flags, gaps; widths 354/41/120). Packing contract `step-cache/CACHE.json`. | every training stage of both models (N flow through `.nfz` shards, see `model2-n-flow/`) |
| 9 | practice-v1 pack | `python3 build_practice.py --out /mnt/ai/scratch/farmNEAT/practice-v1` (imports `build_pack.py`), built 2026-09-28 02:33 UTC | `rows.tsv`: 3,000 winning seats from `views/WINS-2880.tsv`, days 09-23..27, train hash split, ≤ 12% per team, seed 20260928; `VAL.tsv`: 300 val-split episodes 09-20..27; `rows-500.tsv`, `rows-beat500.tsv` (500 winners over current top-30 teams); replay subset as `replays-*.bin` | B from random init: `practice-v1/run_arm.sh` passes it as `--priority-file` |
| 10 | rows-c7 | `build_rows_c7.py` (09-29 06:29 UTC, 9,271 rows: hard / league / core / lbonly / anchor components) then `extend_rows_c7.py` hourly, in place (adds top-30 beaters, M&M&P&Q seats, winners over us, top-12 wins, Majkel1337 wins; weights by recency and beaten-opponent win share). Command in the script docstring. | `base-v1-stepcache-v11/rows-c7.tsv`; each written version kept as `rows-c7/versions/rows-c7.<sha12>.tsv` (36 versions; final 21,132 rows) | B chunks from RUN-7 (`PBC_CACHE_ROWS`) |
| 11 | Never-train lists | `exclusions/exclusion_union.py build` | `EXCLUSIONS-<sha12>.tsv`. Shipped version (17:50 EDT, used by the 18:03 pools): practice VAL 300, M&M&P&Q VAL 205, top-4-beaters VAL 202, sealed 16, leak 128, episode 115279125, 18,220 catalog episodes outside the train hash split (`data.in_split`: buckets 0-899 train, 900-949 val, 950-999 test), 8 seats of one blocked submission; decode holdout seeds 0 in this version. | buckets, t1 |
| 12 | Top-4 beaters | `buckets/build_top4_beaters.py`, run first by `build_buckets.sbatch` | `TOP4-BEATERS-<sha12>/{TRAIN,VAL}.tsv`: winner seats of games a live top-4 submission lost; new ones go to VAL first | bucket A |
| 13 | Buckets | `buckets/buckets_loop.sh` (hourly at :50) → `sbatch buckets/build_buckets.sbatch` → `build_buckets.py` (recipe `BUCKET-RECIPE.json`, schema v4) → `buckets_to_roster.py` | `buckets/<gen>/{A,BC,D,E,L,…}.tsv`, `RECEIPT.json`, `EFFECTIVE-SHARES.json` (receipts for all 40 generations in `buckets/generations/`); `rows-buckets.<sha12>.tsv` (15 rosters in `buckets/rosters/`) | B-even Colab phase 2: `426f06385d73` (gen `0f274b384748`, 12,483 rows) then `d4623759c8fe` (gen `575616a83f15`, 12,829 rows, switch 12:55 EDT). N flow builds its own roster from the same generations (`n-flow/code/roster.py`, see `model2-n-flow/`). |
| 14 | M&M&P&Q pools (t1, wash) | `build_t1.py` (after each bucket build); `build_t1wash.py` via `wash_hourly.sh` (:58) and `wash_final.sh`; uncached seats queued to step 8 | `rows-t1wash-b.<sha12>.tsv` and `rows-t1wash-nflow.<sha12>.tsv` + `.receipt.json`. Final pair built 18:03 EDT 09-30, then frozen: winning seats of the best live submission of #1 / #2 / #3 (105 / 122 / 174) plus 3 winners over #1, mixed 72 / 8 / 14 / 6 % | B-even Colab last phase (`rows-n-d2d605843046.tsv` on the VM); N flow wash |
| 15 | Last-minute new games | `t1-wash/rows-b-newonly-1905.tsv`: written by hand from `SURPRISE-5.json` (5 M&M&P&Q wins fetched after the 18:03 freeze; 4 hand downloads, 1 API) | 5 winning seats, weight 1 | B sprinkle |

## Rosters consumed by the shipped checkpoints

| Roster | sha256 | Rows | Consumer (record that pins it) |
|---|---|---|---|
| `t1-wash/rows-b-newonly-1905.tsv` | `4763fbf59299ad2121c2edb2311648dbe5c83300771bb8c4ab4279864bdf8201` | 5 | Model 1 sprinkle, u46480 → u46484 (`t4w/runs/n1-main-sprinkle/EFFECTIVE-wash.json`) |
| `t1-wash/rows-t1wash-b.d2d605843046.tsv` | `d2d6058430465a239a337c5cc11df8716f5156bd5617e3e30eace1402ecfb7e1` | 404 | Model 1 B-even Colab, last phase (from u46290; ledger, Colab `PBC_CACHE_ROWS`) |
| `buckets/rosters/rows-buckets.d4623759c8fe.tsv.xz` | `d4623759c8fef2d79b45269b10f28ec28535e701cac567d5366c961366dc5bff` | 12,829 | Model 1 B-even Colab phase 2 (`colab-n1-beven-colab/EFFECTIVE.json`) |
| `buckets/rosters/rows-buckets.426f06385d73.tsv.xz` | `426f06385d73f43acf73df18fdedafa12b4e6f9e26bc0370d3e1699c4e36a99f` | 12,483 | Model 1 B-even Colab phase 2 (same record) |
| `rows-c7/versions/*.tsv.xz` | per file | 9,271-21,132 | Model 1 B chunks RUN-7..11 (rows-c7 changed in place; which version each chunk read is in `model1-b-even/`) |
| `practice-v1/rows.tsv` | `8967940012a538ab9d3b7e305d0f4345e1bc4fc2fc2774f8fea35fdc6ccc906a` | 3,000 | Model 1 B from u0 (`practice-v1/MANIFEST.json`) |
| `t1-wash/rows-t1wash-nflow.aa146612b035.tsv` | `aa146612b035728c1d5d6695acf9a0e1a4e1fe0a3d73841b24eacf0299218381` | 806 (404 action + 402 value-only) | Model 2 wash, u56259 → u56559 (`n-flow/runs/nflow-main-wash8014/RUN-0056259.json`; its train.log: 806 of 806 rows usable) |

## Code pins

| Status | Files |
|---|---|
| Verified against a recorded sha256 | all 19 packing sources in `step-cache/CACHE.json` (`data.py`, `step_cache.py`, `codec.py`, `prefix.py`, `noop_mask.py`, `v8*.py`, `job_eta*.py`, `flip*.py`, `calc_features*.py`, `theorem_features*.py`); the 7 rosters above against their training records; `practice-v1/rows.tsv` and `VAL.tsv` against the pack manifest |
| Unpinned, last edit before the output it built | `build_t1wash.py` (18:02 EDT; final pools 18:03), `exclusion_union.py` (16:16; used at 18:03), `build_top4_beaters.py` (05:29), `build_practice.py` / `build_pack.py` (before the 22:33 EDT 09-27 build), `build_rows_c7.py` |
| Unpinned, last edit after some outputs it built | `build_buckets.py` (16:16 EDT) and `buckets_to_roster.py` (16:12) are newer than generations `0f274b384748` (07:51 EDT) and `575616a83f15` (11:51 EDT); the versions that built Model 1's two bucket rosters were earlier edits, not shipped. `extend_rows_c7.py` (last edit 09-30 04:15) wrote versions from 09-29 on. The fetch loop went through 15 revisions; only the last is shipped. |
| Not shipped | the rest of `primitive-bc-v6-20260927` (the B trainer tree; see `model1-b-even/`), `n-flow/code` (see `model2-n-flow/`), catalog parquets, the step cache, replay zips |

## Validation exposure in the last pools

- `build_t1wash.py` keeps only the data-integrity exclusion families (`leak`, `sealed`, `fixed`, `quarantine_sub`; line 84,
  `D-1611pool: no eval holdouts`). 131 of the 404 winning seats in the final pools carry earlier validation or held-out
  flags.
- The first seat of `rows-b-newonly-1905.tsv`, 115984030/1, is in the trainer's validation hash split (bucket 929).
- Earlier validation numbers on these games do not measure the checkpoints trained on them.

## Not established

- The exact `build_buckets.py` / `buckets_to_roster.py` source that built `rows-buckets.426f06385d73` and `d4623759c8fe`.
- When the v11 cache's first build started; coverage over time is only in the hourly `NEW-*.tsv` lists and `BUILD-*.jsonl` logs, not shipped.
- Which rows-c7 version each B chunk read (rows-c7 was rewritten in place while B trained).
- A builder for `rows-b-newonly-1905.tsv` (no script found; the rows match `SURPRISE-5.json`).
- Whether any reconstructed (`reconstructed`, `reconstructed-xsf`) replay reached the final two pools: none of the 404 + 5 final-stage seats points at one; earlier rosters contain them.
