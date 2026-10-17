# Paired-game evaluation

How a candidate was judged better than the submission it would replace, and the game records behind both final uploads. Code here is copied byte-for-byte from what ran; `MANIFEST.tsv` lists every file with its sha256 and original path. `tally.py` and this README are new.

## What a paired comparison is

- **Bank, frozen first.** A gate manifest (`schema: frozen-gate/v1`) names the challenger and incumbent by outer archive sha256, the world seeds, the decision rule and a deadline. It is written before any game runs (`games_seen_at_freeze: 0`, `one_look: true`). `manifest_sha256` is the sha256 of the manifest JSON (sorted keys) without that field, so later edits are detectable.
- **Both seat orders on the same world.** Each world seed is played twice: challenger in seat 0, then in seat 1. A world is complete when both games finish DONE/DONE.
- **Played like Kaggle.** Both archives are loaded cold from their `.tar.gz` and run on CPU, for 719 steps, in a hosted-like runtime (Python 3.11.13, torch 2.6.0+cpu; package list in `../model2-n-flow/records/runtime/hosted-like-py311.freeze.txt`). One exception in the final banks: 16 sprinkle games were GPU-served (below).
- **Replay check.** After each game the runner rebuilds the Kaggle replay from (world seed, recorded actions) and requires the rebuilt final cash to equal the cash the game reported (`replay_verified: true`).
- **Receipt.** `farm_game.sbatch` hashes both archives before and after the game, hashes the runner and the engine tree, and writes `RUN-RECEIPT.json`.
- **Numbers.** W-D-L and point share over games. World share = mean over complete worlds of the average score of the two seat orders, with a Student-t 95% interval (df = worlds − 1). Margin = final-cash difference, averaged the same way.

## The shop fix: separate RNG streams for weeds and shops

In the competition engine, every dusk seeds one generator, `random.Random((seed * 1_000_003) ^ day)`. It draws a weed roll for every empty tile of farm 0 and then farm 1, in scan order, and then draws the day's shop unlock from the same generator. The number of draws before the shop draw therefore depends on how many tiles are empty. If one side weeds, plants or digs differently, the shop sequence changes from that dusk on. Two games on the same world seed could see different shops, so a comparison measured shop luck as well as policy.

`engine/engine-20260927-d91cbd52/src/farm_neat/rollout/crn_rng.py` (common random numbers) gives each draw its own key:

- **weed:** tile (x, y) of farm f at dusk d becomes a weed iff it is empty and `BLAKE2b-64("kaggriculture-crn-v1|weed|seed|d|f|x|y") / 2^64 < weedSpawnChance`;
- **shop:** the unlock at dusk d is `sorted(SHOPS)[BLAKE2b-64("kaggriculture-crn-v1|shop|seed|d") mod 8]`.

The unlock schedule, the 8-instance cap and everything else in the dusk are the reference code. Per-tile weed and per-unlock shop probabilities are unchanged; only the coupling is removed. A world seed now fixes the shop sequence regardless of what either player does, and a tile's weed roll depends only on (seed, day, farm, tile). The reference engine file is not modified; `crn_rng.py` loads a private copy and swaps `_end_of_day` and `_spawn_weeds`. The runner enables it with `--crn-rng`, which sets `crnRng: true` in the game configuration. Kaggle's hosted games do not use it (`hosted_rules: false` in every record), so CRN games are a rules variant for comparisons, not copies of hosted games.

To see the shop sequence a world will get under CRN:

```
python3 -B evaluation/league/engine/engine-20260927-d91cbd52/src/farm_neat/rollout/crn_rng.py shops 1461508636
{"seed": 1461508636, "shops": [[2, "SMOOTHIE_SHOP"], [5, "BRUNCH_SPOT"], [8, "FARMERS_MARKET"], [11, "SMOOTHIE_SHOP"], [14, "BAKERY"], [17, "BAKERY"], [20, "YARN_STORE"], [23, "FARMERS_MARKET"]]}
```

**Did the final decisions use it?** Yes. All 64 games in the two final banks record `crn_rng: true`, and `panel_league.py` (line 6) only counts a game if `crn_rng` is true on engine `engine-20260927-d91cbd52`.

Test: `tests/test_crn.py` from the runner release passes on the repo copy (6 passed, 1 skipped; the skip needs an unpatched `league_game.py` beside the test copy):

```
cd evaluation/league/releases/20260928T150017Z
LEAGUE_ENGINE_SRC=$PWD/../../engine/engine-20260927-d91cbd52/src python3 -B -m pytest -q -p no:cacheprovider tests/test_crn.py
```

## The two final decisions

| Upload | Incumbent (archive sha256) | Bank | Reported at upload | Final, from the shipped records (`tally.py`) |
|---|---|---|---|---|
| `nflow-main-wash8014` (7661782a…) | P2 `nflow-u16000-t4` (6033c855…) | `VS-P2`, 16 worlds × 2 seats, frozen 2026-09-30 22:57:35 UTC | Kaggle description "vs P2 15-2 (17 games)"; the first 17 games by finish time are 15-2 | 28-0-4, world share .875 [.721, 1.029], margin +$3,030/world [+1,789, +4,271] |
| `n1-main-sprinkle` (932bb45d…) | P1 `b-colab-u38760-d30-v9gsafe` (95e5b4e5…) | `VS-P1NB`, 16 worlds × 2 seats, frozen 2026-09-30 23:15:56 UTC, `allow_gpu_served: false` | Kaggle description "vs P1 26-0"; ledger 19:26 EDT: "n1-main-sprinkle 26-0 of 26" (raw tally of all finished games: 16 GPU-served + 10 CPU) | CPU, worlds 1-8: 16-0-0, margin +$4,745/world [+3,314, +6,177]. GPU-served, worlds 9-16: 16-0-0, margin +$5,340/world [+3,363, +7,317]. All 32: 32-0-0 |

**GPU-served sprinkle games.** In the 16 games of `gpu/val-vsp1g-n1-main-sprinkle`, `game.json` records `gpu_serve.served_seats: [true, true]`: both seats' policies ran on a CUDA server on russ-4090b (torch 2.10.0+cu128), with a CPU guard on moves whose top-two gap was under 0.005. The gate declares `allow_gpu_served: false`, and `tplan.world_scores` rejects GPU-served rows unless allowed. Under the gate's own rule only worlds 1-8 (16 CPU games, 16-0) count. The 26-0 reported at upload included all 16 GPU-served games. All 32 wash8014 games were CPU.

The write-up's "28-4 vs the previous N flow (16 worlds × 2 seats)" is the VS-P2 bank above. The live reporter (`finalbuild/vsp2.py`) wrote the same final numbers to the project ledger at 19:06 EDT: `share 0.875 [0.721, 1.029] on 16 worlds x 2 seats (final)`. The N flow upload went out at 19:04 EDT, while the bank was at 13 complete worlds (.846 [.656, 1.037]).

Reproduce the final numbers:

```
cd evaluation/league/records
python3 ../../tally.py nflow-main-wash8014 slurmlane/val-vsp2-nflow-main-wash8014__*/*/game.json
python3 ../../tally.py n1-main-sprinkle gpu/val-vsp1g-n1-main-sprinkle/*/game.json \
    slurmlane/val-vsp1cpu-n1-main-sprinkle__laptop/*/game.json colab/t-vsp1cpu-n1-main-sprinkle/*/game.json
```

Where the games ran:

| Bank | Worlds | Lane (records folder) | Launcher |
|---|---|---|---|
| VS-P2 | 6 / 4 / 3 / 3 | `slurmlane/val-vsp2-nflow-main-wash8014__{spark,skynet,laptop,4090b}` | `finalbuild/vsp2.py` → `extfarm/farm_game.sbatch` (skynet via `records/skynet_game.sbatch`, which execs it at nice 19) |
| VS-P1NB | 1-5 | `colab/t-vsp1cpu-n1-main-sprinkle` (Colab VM, 48 vCPU) | `finalbuild/vsp1cpu.py` → Colab lane script (not shipped, see below); receipts show runner 4f9e4867… |
| VS-P1NB | 6-8 | `slurmlane/val-vsp1cpu-n1-main-sprinkle__laptop` | `finalbuild/vsp1cpu.py` → `extfarm/farm_game.sbatch` |
| VS-P1NB | 9-16 | `gpu/val-vsp1g-n1-main-sprinkle` (host russ-4090b; game process torch 2.6.0+cpu, policies GPU-served by torch 2.10.0+cu128) | GPU game daemon started 19:1x (ledger D-1910gpus), SLURM job 9587; `gpu_serve.runner_sha256` 8719070b… (not the CPU runner), server f2bc8a19…, hook 4460def6…; none of these are shipped and no RUN-RECEIPT exists in these folders |

## Exact per-game command

From `league/extfarm/farm_game.sbatch`, one SLURM array task per row of `pairs.tsv`:

```
REL=/mnt/ai/scratch/farmNEAT/league/releases/20260928T150017Z
ENG=/mnt/ai/scratch/farmNEAT/league/engine/engine-20260927-d91cbd52
PY=/mnt/ai/scratch/farmNEAT/runtime-hosted-py311/venv/bin/python
nice -n 10 "$PY" -B "$REL/league_game.py" --engine "$ENG" --code "$ENG/code/primitive-bc" --seat0 "$a_arch" --seat1 "$b_arch" \
  --labels "$a_name" "$b_name" --world-seed "$seed" --sampler-seed 7100 --steps 719 --episode-id "$eid" \
  --template "$REL/replay-template.json" --out "$d" --crn-rng
```

`league_game.py` calls `play()` from the engine snapshot's `work/runs/maintainer/pbc-ladder/ladder.py` unchanged, records the joint actions, and rebuilds the replay.

## Run one comparison between the two shipped archives

```
REPO=$PWD; L=$REPO/evaluation/league; REL=$L/releases/20260928T150017Z; ENG=$L/engine/engine-20260927-d91cbd52
A=$REPO/submissions/n1-main-sprinkle/n1-main-sprinkle.tar.gz; B=$REPO/submissions/nflow-main-wash8014/nflow-main-wash8014.tar.gz
PY=python3.11   # with torch 2.6.0+cpu and numpy, as in hosted-like-py311.freeze.txt
mkdir -p out/g0 out/g1
$PY -B $REL/league_game.py --engine $ENG --code $ENG/code/primitive-bc --seat0 $A --seat1 $B --labels n1-main-sprinkle nflow-main-wash8014 \
  --world-seed 1461508636 --sampler-seed 7100 --steps 719 --episode-id 999900000000 --template $REL/replay-template.json --out out/g0 --crn-rng
$PY -B $REL/league_game.py --engine $ENG --code $ENG/code/primitive-bc --seat0 $B --seat1 $A --labels nflow-main-wash8014 n1-main-sprinkle \
  --world-seed 1461508636 --sampler-seed 7100 --steps 719 --episode-id 999900000001 --template $REL/replay-template.json --out out/g1 --crn-rng
python3 evaluation/tally.py n1-main-sprinkle out/g*/game.json
```

Run from the repo on 2026-10-16, both seat orders in parallel on CPU:

| Game | Seat 0 | Seat 1 | Final cash | Winner | Replay verified | Wall |
|---|---|---|---|---|---|---|
| g0 | n1-main-sprinkle | nflow-main-wash8014 | 105,933 vs 104,843 | seat 0 | yes, 0 fallbacks | 260 s |
| g1 | nflow-main-wash8014 | n1-main-sprinkle | 106,162 vs 106,099 | seat 0 | yes, 0 fallbacks | 258 s |

`tally.py`: 1-0-1, one complete world split, sprinkle margin +$513.5 on that world. One world is not a measurement of which model is stronger; it shows the repo copy runs end to end with `crn_rng: true`, Python 3.11.13, torch 2.6.0+cpu. Median move time: sprinkle about 287 ms, wash8014 about 72 ms.

## Files

| Path (under `league/`) | What |
|---|---|
| `releases/20260928T150017Z/league_game.py` | per-game runner (sha256 4f9e4867…, the `runner_sha256` in every receipt) |
| `releases/20260928T150017Z/{replay-template.json, league_core.py, league_export.py, league_worker.py, tests/test_crn.py, README.md, CONTRACT.md, SHA256SUMS}` | replay template, the modules the CRN test imports, release docs and sums |
| `engine/engine-20260927-d91cbd52/` | pinned engine snapshot: exact engine, `crn_rng.py`, `pbc-ladder/ladder.py` (`play()`), agent code |
| `extfarm/farm_game.sbatch`, `records/skynet_game.sbatch` | per-game SLURM scripts with run receipts |
| `records/finalbuild/vsp2.py`, `vsp1cpu.py`, `skirmish.py` | drivers that froze the final gates, launched the games and reported live tallies |
| `records/panel_league.py`, `panel_guard.py`, `colab/tplan.py`, `gates/gate_freezer.py`, `gates/verify_strict.py`, `ladder/ladder.py` | gate freezing, record eligibility rules, world scores and t intervals the drivers import |
| `records/gates/VS-P2-…json`, `records/gates/VS-P1NB-…json`, `records/banks/VS-P2.json` | the two frozen gate manifests and the world bank |
| `records/finalbuild/VSP2-STATE.json`, `VSP1CPU-STATE.json` | driver state at the end |
| `records/{slurmlane,gpu,colab}/…/` | every game of both banks: `game.json`, `RUN-RECEIPT.json`, `PAIR.tsv`, `game.log`, `pairs.tsv` |

The drivers import each other through absolute paths on the farm's NFS and were not run from the repo. `tally.py` reads the shipped `game.json` files directly.

## Pins checked on the repo copy (2026-10-16)

| Item | Result |
|---|---|
| `league_game.py` vs receipts' `runner_sha256` 4f9e4867… | match |
| release `SHA256SUMS` for the 7 copied release files it lists | all OK |
| `pbc-ladder/ladder.py` vs the games' `ladder_sha256` fb2a09af… | match |
| engine tree hash (receipt algorithm) | recorded 43a878a6… includes 8 `__pycache__` files present at run time; the copy has no `__pycache__` and hashes to f75cb9b7…, which the NFS original also gives with them excluded |
| gate `manifest_sha256` (both) | match |
| `game.json` vs `RUN-RECEIPT.json` `game_json_sha256` | 48 of 48 receipts match; the 16 games of `gpu/val-vsp1g-…` have no receipt |
| unpinned | `farm_game.sbatch` (last edit 09-29 23:59, before both banks), `skynet_game.sbatch` (09-30 10:17), `vsp1cpu.py` (19:09, before the VS-P1NB start at 19:15), `vsp2.py` (19:03, after the VS-P2 start at 18:57), `panel_league.py`, `panel_guard.py`, `tplan.py`, `gate_freezer.py`, `verify_strict.py`, `records/ladder/ladder.py`, `tests/test_crn.py` (not in the release `SHA256SUMS`) |

## Path mapping

| Original | Repo |
|---|---|
| `/mnt/ai/scratch/farmNEAT/league/` | `evaluation/league/` |
| `/mnt/ai/scratch/farmNEAT/pbc-deploy/<name>/<name>.tar.gz` (the two final uploads) | `submissions/<name>/<name>.tar.gz` |
| `/mnt/ai/scratch/farmNEAT/runtime-hosted-py311/venv/bin/python` | Python 3.11.13 + torch 2.6.0+cpu (`../model2-n-flow/records/runtime/hosted-like-py311.freeze.txt`) |
| `/home/russ/Desktop/farmNEAT/work/runs/maintainer/base-v1/LEDGER-0929.md` | not shipped; the lines quoted above are from it |

## Not shipped or not established

- **Replays:** the 64 `replay.json.gz` files (about 1 MB each) are not shipped. `NOT-SHIPPED.tsv` lists each with its sha256; every game is re-runnable from its world seed and the two archives.
- **Incumbent archives:** P1 `b-colab-u38760-d30-v9gsafe` and P2 `nflow-u16000-t4` are not in the repo; the manifests give their sha256.
- **Colab lane launcher:** `selfplay-s/colab-lane/p0batch3.sh` is not shipped because it reads a private serve token and tunnel URL. Its games' receipts show the same runner hash.
- **`gpu/val-vsp1g-n1-main-sprinkle` lane:** the GPU game daemon and policy server that played those 16 games are not shipped; the exact launch is not established.
- **`vsp2.py`:** the copy was edited 6 minutes after the VS-P2 bank started, so it may differ from the version that launched it. Its lane split (6/4/3/3 worlds) matches the records.
