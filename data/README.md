# Data: Kaggle episodes → training rosters

How games were collected, which seats were kept, how features were precomputed and how each training mix was drawn. Replays are not included; every roster names seats by Kaggle episode id and seat. Full step-by-step with exact commands: [PIPELINE.md](PIPELINE.md).

## The ideas, and where they are

| Idea | File |
|---|---|
| Pull the official daily episode dumps hourly and check them against the game config | [`acquire/ingest-daily.sh`](acquire/ingest-daily.sh) |
| Save every public game of the top teams, including submissions they later replaced, within the API rate limit | [`acquire/save_team.py`](acquire/save_team.py), [`acquire/fetch_top_slow15.sh`](acquire/fetch_top_slow15.sh), [`acquire/FETCH-POLICY.json`](acquire/FETCH-POLICY.json) |
| Track everyone who beats us, and who beats them | [`acquire/beaters.py`](acquire/beaters.py), [`acquire/hosted_fetch.py`](acquire/hosted_fetch.py) |
| Rebuild games from a public seed + action dataset, keeping a game only if the final cash matches | [`acquire/seed_action_replays.py`](acquire/seed_action_replays.py) |
| One catalog row per seat (team, rank, day, result, exact re-play checks) | [`catalog/catalog.py`](catalog/catalog.py), [`catalog/seats.py`](catalog/seats.py) |
| Precompute every step's features once per seat (74,857 seats, ~745 GiB) | [`step-cache/`](step-cache/) (builder code under `step-cache/code/`, contract in `CACHE.json`) |
| Never train on a list: validation, sealed and leak lists unioned and checked against every roster | [`exclusions/exclusion_union.py`](exclusions/exclusion_union.py), [`exclusions/leak_check.py`](exclusions/leak_check.py) |
| Winners only, sampled by recency × strength buckets with a per-seat cap | [`buckets/build_buckets.py`](buckets/build_buckets.py), [`buckets/BUCKET-RECIPE.json`](buckets/BUCKET-RECIPE.json), [`buckets/effective_shares.py`](buckets/effective_shares.py) |
| Games that beat any top-4 team: the last thing trained on | [`buckets/build_top4_beaters.py`](buckets/build_top4_beaters.py) |
| Last-day pools of top-team wins for the short top-ups | [`t1-wash/build_t1wash.py`](t1-wash/build_t1wash.py), [`t1-wash/build_t1.py`](t1-wash/build_t1.py) |

| Folder | What |
|---|---|
| `acquire/` | episode download and tracking |
| `catalog/` | per-seat catalog builders |
| `step-cache/` | feature precompute: builder, row lists, packing contract |
| `exclusions/` | never-train lists and checks |
| `practice-v1/`, `rows-c7/` | early rosters (Model 1's first stages) |
| `buckets/` | bucket sampler, recipes, generation receipts, rosters |
| `t1-wash/` | last-day pools |
| `MANIFEST.tsv` | sha256, size and origin of every file; `*.xz` files are compressed (`xz -dk`) |
