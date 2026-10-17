# How it works: what we did, with the code

Each technique behind the two final models, how it works, and links to the exact code that shipped and to any evidence in the repo.

1. [One decision at a time](#1-one-decision-at-a-time)
2. [Imitate winners only, including whoever beat you](#2-imitate-winners-only-including-whoever-beat-you)
3. [Sample by recency and strength buckets](#3-sample-by-recency-and-strength-buckets)
4. [Last-day top-ups and RAW export](#4-last-day-top-ups-and-raw-export)
5. [Paired-game evaluation](#paired-game-evaluation)
6. [Features and the step cache](#6-features-and-the-step-cache)
7. [Model 2: typed graph with three memories](#7-model-2-typed-graph-with-three-memories)
8. [Model 1: wide transformer, heads regrouped](#8-model-1-wide-transformer-heads-regrouped)
9. [Optimizer, schedule, EMA](#9-optimizer-schedule-ema)
10. [Serving: rule guards and train/serve parity](#10-serving-rule-guards-and-trainserve-parity)
11. [Getting more games](#11-getting-more-games)
12. [What didn't work](#12-what-didnt-work)

Code links go to the flat reading copies in [`code/`](code/): byte-identical to what ran, same line numbers ([`code/MANIFEST.tsv`](code/MANIFEST.tsv) maps each to its as-run path). Data scripts, records and figures link to their own folders.

## 1. One decision at a time

Each turn is generated like a sentence: the farmer's command first, then each hand in engine order, then market orders, then STOP. Each choice is one token, and each token's query sees the choices already made that turn. A quantity bin is predicted after its token and conditioned on it. Training is teacher-forced on the winner's actual commands. Play is greedy with a K/V cache, so each decision is one query.

The step that made this work: before unit *k* decides, a prefix executor replays the earlier units' commands of the same turn through the engine's own unit logic, and the query gets the resulting state. Planting and buying conflicts within a turn are visible to later tokens.

| What | Model 1 | Model 2 |
|---|---|---|
| Design note | [`code/model1/model.py#L13-L20`](code/model1/model.py#L13-L20) | [`code/model2/nf_model.py#L17-L22`](code/model2/nf_model.py#L17-L22) |
| Teacher-forced decoder (training) | [`decode_teacher`](code/model1/model.py#L1068-L1132) | [`queries_teacher`](code/model2/nf_model.py#L484-L497), [`forward_chunk`](code/model2/nf_model.py#L535-L557) |
| Greedy sequential decode (play) | [`decode_free`](code/model1/model.py#L1210-L1296), [`act`](code/model1/model.py#L1349-L1389) | [`decide`](code/model2/serving/model.py#L179-L244), [`act`](code/model2/serving/model.py#L147-L177) |
| Same-step prefix state | [`PrefixExecutor`](code/features/prefix.py#L49-L292), [`teacher_arrays`](code/features/prefix.py#L295-L315) | same prefix rows, stored in the shards ([`code/model2/nf_format.py#L57-L65`](code/model2/nf_format.py#L57-L65)) |
| Loss | [`step_loss`](code/model1/train.py#L808-L866) | [`chunk_loss`](code/model2/train.py#L68-L113) |

Sizes: up to 25 unit slots and up to 10 orders plus STOP. The write-up's framing is in [README.md](README.md#abandoned).

## 2. Imitate winners only, including whoever beat you

- **Winners only:** a seat is a training target only if its final reward is strictly greater than the rival's. No losing action is ever imitated.
- **Losses:** when one of our submissions lost (on 29-30 Sep), the opponent's winning seat goes into its own bucket L. Our losing seat is never an action target. L's per-seat draw rate is 4× bucket A's.
- **Excluded:** mirror games (same team on both sides), our own wins, validation/sealed/leak lists.
- **Model 2 also learns from losing seats,** but only as value targets. WDL-only rows train the win/draw/loss head; their action masks are off. The wash roster had 806 rows: 404 imitation seats plus 402 WDL-only seats ([`model2-n-flow/STAGES.md`](model2-n-flow/STAGES.md)).

| What | Code |
|---|---|
| `wins_only` seat filter | [`code/features/data.py` `select_seats`](code/features/data.py#L203-L226) |
| Bucket L: winners over us | [`data/buckets/build_buckets.py#L154-L156`](data/buckets/build_buckets.py#L154-L156); rule text [`BUCKET-RECIPE.json#L58`](data/buckets/BUCKET-RECIPE.json#L58) |
| Qualifying rules (winner seat, not ours, not a mirror, not held out) | [`data/buckets/build_buckets.py#L1-L12`](data/buckets/build_buckets.py#L1-L12) |
| Model 2 action masks off on WDL-only rows | [`code/model2/train.py` `masks`](code/model2/train.py#L51-L60) |
| Exclusion lists | [`data/exclusions/`](data/exclusions/) |

## 3. Sample by recency and strength buckets

Each qualifying winning seat lands in one bucket by day and strength, e.g. A = wins by the top teams on the last day plus winners over the top 4, BC = other 29-30 Sep wins, D/E = top 10% per day of older days. A draw picks a bucket by share, then a seat uniformly inside it. Empty buckets are renormalised away. A small bucket is capped at 4× bucket A's per-seat draw rate, so a handful of games can't be drawn hundreds of times. Buckets were rebuilt every hour as new games arrived.

- **Model 2** draws buckets directly in its loader. Shares per segment are in [`model2-n-flow/STAGES.md`](model2-n-flow/STAGES.md), e.g. S3 `A=0.3495,BC=0.3995,D=0.1498,E=0.0499,L=0.0513`.
- **Model 1's** step-cache loader reads integer row weights, so the bucket draw is emulated with weights proportional to share / bucket size.

| What | Code |
|---|---|
| Bucket builder and recipe | [`build_buckets.py#L1-L12`](data/buckets/build_buckets.py#L1-L12), [`BUCKET-RECIPE.json`](data/buckets/BUCKET-RECIPE.json) |
| Dynamic share for L | [`build_buckets.py#L298-L312`](data/buckets/build_buckets.py#L298-L312) |
| 4× small-bucket cap | [`build_buckets.py#L313-L331`](data/buckets/build_buckets.py#L313-L331), [`BUCKET-RECIPE.json#L48-L49`](data/buckets/BUCKET-RECIPE.json#L48-L49) |
| Bucket draw (Model 2) | [`code/model2/nf_data.py` `Streams._draw`](code/model2/nf_data.py#L111-L127) |
| Weighted-roster emulation (Model 1) | [`data/buckets/buckets_to_roster.py#L1-L9`](data/buckets/buckets_to_roster.py#L1-L9); loader repeats a row `weight` times: [`code/features/step_cache.py` `StepChunks`](code/features/step_cache.py#L253-L315) |

The shipped `build_buckets.py` and `buckets_to_roster.py` are their final versions. The versions that built Model 1's two bucket rosters were overwritten ([`data/PIPELINE.md`](data/PIPELINE.md)). The rosters themselves are shipped byte-identical.

## 4. Last-day top-ups and RAW export

Both final models got a very short, low-LR fine-tune on the newest top-team wins right before the deadline:

| | Model 1 sprinkle | Model 2 wash |
|---|---|---|
| Start | Colab checkpoint u46,480 | u56,259, raw weights := EMA first, so the run starts at the policy that was already shipping |
| Updates, LR | 4 updates, lr 1e-5 | 300 updates, lr 3e-5 constant |
| Data | 5 winning seats from 30 Sep, weight 1 | 806 rows: 30 Sep wins by the top three teams + WDL-only losing seats; `T1W=0.718, BEAT1=0.082, T2W=0.14, T3W=0.06` |
| Wall time | 85 s on an RTX 4090 | 304 s on an RTX 4090 |
| Export | RAW: EMA := trained weights | RAW: EMA := trained weights |

- **Why RAW:** the export chain ships the EMA, and with EMA decay 0.999 a 300-update top-up moves the EMA only partway (0.999^300 ≈ 0.74 of it is still the pre-top-up average). Copying the raw weights into the EMA slot ships the top-up itself.
- **Model 1's last Colab phase** also ran on a top-3 pool: 404 seats, #1 team .80, #2 .14, #3 .06 ([`model1-b-even/STAGES.md`](model1-b-even/STAGES.md)).

| What | Code |
|---|---|
| Model 1 sprinkle launcher | [`b_wash_sprinkle.sh`](code/model1/b_wash_sprinkle.sh), [`t4w-bn-skynet.sbatch`](model1-b-even/code/launch/sprinkle/t4w-bn-skynet.sbatch) |
| Model 1 RAW export | [`b_wash_sprinkle.sh#L35-L38`](code/model1/b_wash_sprinkle.sh#L35-L38) |
| Model 2 seed (raw := EMA) | [`code/model2/make_t4_seed.py`](code/model2/make_t4_seed.py#L1-L29) |
| Model 2 RAW export (EMA := raw) | [`code/model2/make_raw_as_ema.py`](code/model2/make_raw_as_ema.py#L1-L12) |
| Model 2 wash launcher | [`model2-n-flow/code/arm_skynet.sbatch`](model2-n-flow/code/arm_skynet.sbatch) |
| Wash pool builder | [`data/t1-wash/build_t1wash.py#L1-L8`](data/t1-wash/build_t1wash.py#L1-L8) |

Game results for both top-ups are under [Paired-game evaluation](#paired-game-evaluation).

## Paired-game evaluation

A candidate replaced a live submission only if it beat that submission's exact archive in paired games. Both archives play the same world seeds, each from both seats, loaded cold and run on CPU like Kaggle. The pass bar, seeds and archive hashes are frozen in a manifest before the first game. After every game the replay is rebuilt from (seed, actions) and must reproduce the reported final cash.

**The shop fix.** In the competition engine, each dusk drew a weed roll for every empty tile and then the day's shop unlock from one generator. A different number of empty tiles (one side weeded or planted differently) shifted the shop draw, so two games on the same seed could see different shops, and a comparison measured shop luck as well as play. Common random numbers give each draw its own key: a tile's weed roll hashes (seed, day, farm, x, y), and a shop unlock hashes (seed, day). The per-draw probabilities are unchanged. A world seed now fixes the shop sequence whatever either player does. Only CRN games count.

| What | Code |
|---|---|
| Weed roll and shop draw, each keyed independently | [`weed_uniform`](code/evaluation/crn_rng.py#L77-L80), [`shop_index`](code/evaluation/crn_rng.py#L83-L88), [`shop_sequence`](code/evaluation/crn_rng.py#L91-L98) |
| The dusk with the split streams, swapped into a private copy of the engine | [`_crn_spawn_weeds`](code/evaluation/crn_rng.py#L122-L129), [`_crn_end_of_day`](code/evaluation/crn_rng.py#L132-L160), [`install`](code/evaluation/crn_rng.py#L177-L195) |
| One game: both archives cold, 719 steps, replay rebuilt and checked (`--crn-rng`) | [`league_game.py` `main`](code/evaluation/league_game.py#L101-L192) |
| Freeze the gate before any game | [`gate_freezer.py` `gate_one`](code/evaluation/gate_freezer.py#L119-L174) |
| A game counts only if CRN, replay verified, DONE/DONE, 0 fallbacks | [`panel_league.py#L6-L7`](code/evaluation/panel_league.py#L6-L7) |
| World scores (mean of both seat orders) and the gate's GPU rule | [`tplan.py` `world_scores`](code/evaluation/tplan.py#L420-L469) |
| W-D-L, world share and margin with Student-t 95% intervals | [`tally.py`](code/evaluation/tally.py#L18-L56) |

| Release decision | Bank | Result |
|---|---|---|
| `nflow-main-wash8014` vs the previous N flow (P2) | 16 worlds × 2 seats, CPU | 28-0-4, world share .875 [.721, 1.029], margin +$3,030/world [+1,789, +4,271] |
| `n1-main-sprinkle` vs P1 | 16 worlds × 2 seats | worlds 1-8 on CPU: 16-0-0. Worlds 9-16 were GPU-served, also 16-0-0, but the frozen gate disallowed GPU-served games, so only the 16 CPU games count under its own rule. The "26-0" in the upload note mixed both. |

Commands, lanes, receipts and the full record: [`evaluation/README.md`](evaluation/README.md).

## 6. Features and the step cache

**Every feature, column by column: [FEATURES.md](FEATURES.md).**

Both models read the same features, computed once per seat and cached. All features are actor-visible: no rival private state, no future shop draws, no replay outcome. Most families are exact engine arithmetic rather than learned guesses: timers, price curves, reachability, what tonight's day end will discard.

| Family | What it captures | Code |
|---|---|---|
| Entity tokens | every farm, tile, worker, market item, shop and constant as one 80-column row, with a separate knownness mask (absent ≠ zero) | [`observation.py`](code/features/observation.py#L1-L6), [`pack`](code/features/observation.py#L436-L454) |
| Proven game facts | engine-exact price curves, visible town demand, crop/animal clocks, liquidation value | [`theorem_features.py`](code/features/theorem_features.py#L1-L8) |
| Market projection | projected end-of-game inventory per product if committed supply sells | [`calc_features.py`](code/features/calc_features.py#L1-L8) |
| No-action forecasts | what happens if nobody acts, duty counts, worker reachability / ETA | [`flip.py`](code/features/flip.py#L1-L7) |
| History | causal features from this seat's past observations and its own past actions | [`history_features.py`](code/features/history_features.py#L1-L8), previous two turns as input [`_history`](code/model1/model.py#L910-L924) |
| v8 families | dusk (day-end shed overflow, return feasibility), cohorts (production cohorts, working capital), tileclock (per-tile crop clock, season cliffs), flags (calendar, shop reveals, action receipts), gaps (asks from game analysis), lattice (both farms as a 10×10 grid over t, t-1, t-2) | [`v8.py`](code/features/v8.py#L1-L3), [`v8_dusk.py`](code/features/v8_dusk.py#L1-L8), [`v8_cohorts.py`](code/features/v8_cohorts.py#L1-L7), [`v8_tileclock.py`](code/features/v8_tileclock.py#L1-L8), [`v8_flags.py`](code/features/v8_flags.py#L1-L6), [`v8_gaps.py`](code/features/v8_gaps.py#L1-L7), [`v8_lattice.py`](code/features/v8_lattice.py#L1-L8) |
| Same-step prefix | state after earlier commands of the same turn (section 1) | [`prefix.py`](code/features/prefix.py#L1-L7) |
| No-op supervision mask | don't train on unit commands the engine would not execute; the teacher action itself is never changed | [`noop_mask.py`](code/features/noop_mask.py#L1-L8), used in [`step_loss`](code/model1/train.py#L808-L866) |

- **Inputs are added zero-initialised.** Most auxiliary groups enter through their own `grow_*` projection that starts at zero, with per-group feature dropout 0.1 in Model 1 ([growth inputs in `PolicyNet.__init__`](code/model1/model.py#L598-L658), [`zero_growth`](code/model1/model.py#L660-L667), `feature_dropout` in [`SCRATCH.json`](model1-b-even/records/1-base1-b/SCRATCH.json)). New inputs can be added to a trained network without changing its outputs at graft time.
- **Step cache:** `step_cache.py build --v6 --v8` writes one `.stp` file per seat holding all 719 steps' features and labels. Training reads only the cache. v11 has 74,857 files, ~745 GiB ([`data/PIPELINE.md`](data/PIPELINE.md), step 8). Code: [`code/features/step_cache.py` `build`](code/features/step_cache.py#L156-L191), [`seat_steps`](code/features/step_cache.py#L56-L69), [`data.features`](code/features/data.py#L283-L315).
- **Figures:** [which model uses which family](images/15-figI-feature-table.png); [loss increase when each group is zeroed](images/16-figD-features.png) (earlier B/C checkpoints).

## 7. Model 2: typed graph with three memories

A 7,252,244-parameter model trained from random weights in one day: 11.3 h logged on one Colab G4 for S0-S9, plus the 304 s wash ([`model2-n-flow/STAGES.md`](model2-n-flow/STAGES.md)).

- **Graph:** 320 nodes per step. 296 fixed entity slots (each tile, worker, stock item, market item, shop and constant always sits in the same slot) + FAST and SLOW memory nodes + 8 calendar landmarks (now, next shop tick, dusk, +2d … terminal) + pad.
- **Relational attention:** 4 blocks, width 256, 8 heads. 4 heads add a learned bias per (relation type, head) over 17 relation types (tile adjacency, mirror tile, farm membership, carrier-tile, custody, market-market, shop-market, calendar sweeps, …). The other 4 heads are plain global attention.
- **Memories:** a fast GRU (128) updated every step from a pooled graph summary and our previous action; a slow GRU (256) updated at dawn from the previous day's mean summary; one worker GRU (64) per own worker slot, reset at dawn for hands (hands are dismissed at dusk), never for the farmer.
- **Training:** truncated BPTT. Streams walk whole games in 24-step chunks and carry memory across chunks. The first seat of each stream starts at a random chunk and its losses are masked, since its memory can't be reconstructed.
- **Aux heads:** win/draw/loss, own next-step unit tokens, opponent unit tokens; loss weights 0.2 / 0.1 / 0.1.
- **Same features as Model 1:** the shard format is a pure projection of Model 1's step cache into the fixed slots. Per node type, a frozen column subset is listed in `COLUMNS.json`, and the same `compact_tokens` function runs at train and play time.

| What | Code |
|---|---|
| Design note | [`code/model2/nf_model.py#L1-L23`](code/model2/nf_model.py#L1-L23), [`DESIGN.md`](code/model2/DESIGN.md) |
| 17 relation types | [`REL_NAMES`](code/model2/nf_model.py#L48-L50), [`relation_table`](code/model2/nf_model.py#L128-L184) |
| Relational block (rel heads + global heads) | [`RelBlock`](code/model2/nf_model.py#L209-L236) |
| Memories | [`memory_step`](code/model2/nf_model.py#L419-L437), [`encode_post`](code/model2/nf_model.py#L439-L447) |
| Aux heads, loss weights | [`aux_heads`](code/model2/nf_model.py#L528-L532), [`chunk_loss`](code/model2/train.py#L68-L113) |
| Streams / truncated BPTT | [`code/model2/nf_data.py#L1-L12`](code/model2/nf_data.py#L1-L12), [`Streams.__iter__`](code/model2/nf_data.py#L129-L182) |
| Graph layout and column freeze | [`code/model2/nf_format.py#L1-L19`](code/model2/nf_format.py#L1-L19), [`slot_index`](code/model2/nf_format.py#L97-L125), [`compact_tokens`](code/model2/nf_format.py#L139-L181), [`cmd_derive`](code/model2/nf_format.py#L365-L404), [`COLUMNS.json`](code/model2/COLUMNS.json) |

Evidence: 28-4 for the shipped N flow against the previous N flow over 16 worlds × 2 seats. An earlier N flow (16k updates + top-up) went 60-4 against an older 18.9M model and 35-29 against P1, a 54.6M B-family submission, over 32 worlds × 2 seats. Both are drawn in [`images/14-m2-d-evidence.png`](images/14-m2-d-evidence.png), with numbers typed into [`figures/model_options.py#L256-L282`](figures/model_options.py#L256-L282).

## 8. Model 1: wide transformer, heads regrouped

54,639,683 parameters: width 432, 10 encoder blocks, 8 decoder blocks with 12 heads × 36, SwiGLU, dropout 0.05. It trained 31,047 updates as "B" with 36 encoder heads × 12, then continued as "B-even" to 46,484 ([`model1-b-even/STAGES.md`](model1-b-even/STAGES.md)).

- **Regroup:** change the encoder from 36 heads × 12 to 18 heads × 24 with no tensor changes. The same weights are just split into fewer, wider heads. On 160 steps this alone changed 55.6% of whole-step decisions vs the parent; 12 × 36 changed 73.8% ([`BEVEN-CHECK.json`](model1-b-even/records/2-b-even-phase1/BEVEN-CHECK.json)).
- **Repair with distillation:** 1,000 updates of 0.5 BC + 0.5 KD from the 36-head parent's EMA at temperature 1.5. Validation bits/step went 3.79 → 2.69 (teacher 2.44) on 512 fixed steps ([`validation-initial.json`](model1-b-even/records/2-b-even-phase1/validation-initial.json), [`validation-final-ema.json`](model1-b-even/records/2-b-even-phase1/validation-final-ema.json)). After that, plain SFT on the bucket rosters.
- **Aux heads (training only):** own t+1/t+2 and opponent t/t+1/t+2 worker commands and order programs. They are stripped from the shipped archive (374 tensors remain; [`model1-b-even/STAGES.md`](model1-b-even/STAGES.md)).

| What | Code |
|---|---|
| Config fields | [`ModelConfig`](code/model1/model.py#L100-L216); the run's values in [`SCRATCH.json`](model1-b-even/records/1-base1-b/SCRATCH.json) |
| Encoder / decoder blocks | [`EncoderBlock`](code/model1/model.py#L362-L380), [`DecoderLayer`](code/model1/model.py#L383-L434), [`PolicyNet`](code/model1/model.py#L521-L658) |
| Head regroup | [`model1-b-even/code/b-even/make_init.py`](code/model1/make_init.py#L1-L12) |
| BC + KD objective | [`oa3/training.py` `objective`](code/model1/kd_training.py#L34-L73), [`oa3/losses.py` `masked_kl`](code/model1/kd_losses.py#L20-L33) |
| Aux heads and loss | [aux heads in `PolicyNet.__init__`](code/model1/model.py#L640-L657), [`aux_loss`](code/model1/train.py#L967-L1004) |

Evidence: B-even at u5000 (trainer update 35,000) beat its 36-head parent B u31,047 45-19, +$1,463 per world, over 32 worlds × 2 seats ([`images/06-m1-e-signature.png`](images/06-m1-e-signature.png); numbers in [`figures/model_options.py#L231-L241`](figures/model_options.py#L231-L241)). u5000 had also had the KD phase and 4,000 further SFT updates, so this does not isolate the regroup.

## 9. Optimizer, schedule, EMA

- **Muon + AdamW:** Muon (Nesterov, 5 Newton-Schulz steps, `match_rms_adamw` LR scaling) on every 2-D weight matrix inside the encoder/decoder blocks; AdamW on embeddings, norms, biases and heads. Model 2's optimizer code is copied from Model 1's.
- **WSD schedule:** warmup, then constant LR, with an optional linear decay over the last `decay_updates`. Model 2 can also start its decay at a wall-clock time so the run ends cooling down at a deadline; [`model2-n-flow/STAGES.md`](model2-n-flow/STAGES.md) notes that decay never applied in the shipped lineage.
- **EMA 0.999 with a warmup:** `d = min(decay, (1+u)/(10+u))`, so the early EMA does not hold on to the random init. Model 1 can keep extra EMAs alongside (`PBC_EMA_EXTRA`, e.g. 0.998 and 0.9995).

| What | Model 1 | Model 2 |
|---|---|---|
| Muon/AdamW split | [`make_optimizer`](code/model1/train.py#L1381-L1416), [`MultiOptimizer`](code/model1/train.py#L1441-L1467), optional [`SplitMuon`](code/model1/train.py#L108-L150) | [`is_muon_matrix`](code/model2/train.py#L145-L146), [`make_optimizer`](code/model2/train.py#L149-L172) |
| LR schedule | [`lr_at`](code/model1/train.py#L394-L407) | [`lr_at`](code/model2/train.py#L175-L190) |
| EMA update | [`train.py#L2353-L2361`](code/model1/train.py#L2353-L2361) | [`train.py#L494-L497`](code/model2/train.py#L494-L497) |

## 10. Serving: rule guards and train/serve parity

The network's greedy action passes through a fixed chain of engine-aware rules before it is sent. Both archives use the same chain:

| Rule | What it does | Code |
|---|---|---|
| `plant_clamp` | drop PLANT commands beyond the seeds held (the engine would cancel all of them) | [`agent.py#L441-L461`](code/model1/serving/agent.py#L441-L461) |
| `land_rule` | with 3 quadrants, enough cash and the right days, append BUY_LAND | [`agent.py#L143-L176`](code/model1/serving/agent.py#L143-L176) |
| `hire_cap` | cap hires per day | [`agent.py#L518-L535`](code/model1/serving/agent.py#L518-L535) |
| `overflow_guard` | from hour 20, if tonight's deposit would overflow the shed, send idle workers to deposit | [`agent.py#L354-L438`](code/model1/serving/agent.py#L354-L438) |
| `terminal_salvage` | from t=700, walk carried products to the shed in time for the final sell-off | [`agent.py#L467-L509`](code/model1/serving/agent.py#L467-L509) |
| `sell_rule` | dusk and terminal selling | [`agent.py#L289-L330`](code/model1/serving/agent.py#L289-L330) |

- **History:** the history fed back to the model records the action actually sent, not the network's proposal, matching how training labels were made ([`Agent.step`](code/model1/serving/agent.py#L928-L971)).
- **Fallback:** any exception returns PASS ([`submissions/n1-main-sprinkle/main.py#L54-L56`](submissions/n1-main-sprinkle/main.py#L54-L56)).
- **Parity check:** the Model 2 play path (features at play time, step-by-step memory, cached decoder) is checked against the training path on a real replay seat. Result: 248 decisions, max abs logit difference 4.5e-5, tolerance 2e-3 ([`test_nflow_parity.py`](model2-n-flow/deploy/work/runs/maintainer/n-flow-deploy/test_nflow_parity.py#L1-L10), [`PARITY.json`](model2-n-flow/records/release/PARITY.json)). Deployed v8 inputs vs training examples: [`parity_v8.py`](model2-n-flow/deploy/work/runs/maintainer/n-flow-deploy/parity_v8.py).
- **Cold verify:** each archive played 2 games vs PASS before release with 0 fallbacks. Slowest warm step: Model 1 212 ms, Model 2 67 ms; first step including the model load: 1.1-1.2 s and 0.76 s ([`LINKS-n1-main-sprinkle.json#L33-L47`](model1-b-even/records/5-sprinkle/LINKS-n1-main-sprinkle.json#L33-L47), [`LINKS-nflow-main-wash8014.json#L39-L53`](model2-n-flow/records/release/LINKS-nflow-main-wash8014.json#L39-L53)).

## 11. Getting more games

- **Sources:** Kaggle's daily episode dumps; a replay-API loop over the top teams (a cycle every 900 s, own games first, back-off on 429); our own hosted games and the games of whoever beat us ([`data/PIPELINE.md`](data/PIPELINE.md), steps 1-4; [`fetch_top_slow15.sh`](data/acquire/fetch_top_slow15.sh), [`save_team.py`](data/acquire/save_team.py), [`hosted_fetch.py`](data/acquire/hosted_fetch.py), [`beaters.py`](data/acquire/beaters.py)).
- **Tape rebuilds:** games rebuilt from the public `xishengfeng/kaggriculture-replay-db` seed + action tapes (CC0) by replaying them through the exact engine, kept only when the replayed final cash equals the recorded one. They make up 21,386 of the 40,909 rows in the v11 cache's initial list ([`data/PIPELINE.md`](data/PIPELINE.md), step 6; [`seed_action_replays.py`](data/acquire/seed_action_replays.py), [`seats.py` `rebuild_task`](data/catalog/seats.py#L1465-L1532)).

## 12. What didn't work

| Idea | What happened | Code in this repo |
|---|---|---|
| Macro decisions: factor the game's micro decisions into macro ones, backed by a page of theorems (311 claims, 919 linked pairs) | two weeks of work; too much for agents to keep consistent; abandoned for micro decisions | [`figures/theorem-graph.json`](figures/theorem-graph.json), [`figures/theorem_graph.py`](figures/theorem_graph.py), [`images/01-figA-theorems.png`](images/01-figA-theorems.png) |
| Chain of thought: write an unsupervised code onto every tile (and 64 global slots) before deciding; the decoder reads only the codes | dropped; both shipped models run without it (`think='off'`, which the history inputs require) | [`model.py#L1-L26`](code/model1/model.py#L1-L26), [`think` field](code/model1/model.py#L123), [code writer in `encode`](code/model1/model.py#L872-L875), [check](code/model1/model.py#L204-L205) |
| Recurrence in depth: rerun the middle encoder blocks several times | some of the best early losses, then worse late with the chosen config; too many ablations needed | [`recur_*` config](code/model1/model.py#L156-L163), [core loop in `encode`](code/model1/model.py#L844-L866) |
| Depth over width | worse loss immediately | — |
| 36 heads of width 12 | did well, then started doing worse later; regrouped into Model 1 (section 8) | [`make_init.py`](code/model1/make_init.py#L1-L12) |

Source for this table: [README.md](README.md#abandoned).
