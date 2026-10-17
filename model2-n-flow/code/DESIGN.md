# N flow pilot: design, shapes, data, known gaps

Owner approval 2026-09-30 ~01:05 EDT. The sources are the two design reviews:
- /home/russ/dot/n-moonshot-reviews/gpt61-review-20260930.md (GPT)
- /home/russ/dot/n-moonshot-reviews/astra-review-20260930.md (Astra)

All weights, memory states, optimizer moments and EMA start from fresh random initialisation. Nothing is loaded from B, O or any other checkpoint; there is no loader for foreign weights. The objective is supervised BC plus auxiliary heads, with no RL.

## Code
| file | role |
|---|---|
| n-flow/nf_format.py | Shard format and reference encoder (DATA-FORMAT.md) |
| n-flow/nf_model.py | Network |
| n-flow/nf_data.py | Stream loader |
| n-flow/train.py | Trainer |
| n-flow/roster.py | Roster builder |
| n-flow/ema_export.py | Exports EMA weights for deploy |
| n-flow-deploy/ | Copy of the B v9-decode harness (agent.py, data.py, codec.py, prefix.py and the rest, unchanged) plus model.py (the N flow PolicyNet with its play path), the staged nf_model.py, nf_format.py and COLUMNS.json (stage_deploy.sh), and build_archive.py |

## Graph (per physical step)
- **Nodes: N = 320.**
  - 296 fixed entity slots:
    - GLOBAL;
    - FARM own/opp;
    - 200 TILE (both farms);
    - 50 CARRIER (own and opponent, by ordinal);
    - 17 STOCK (12 shed items, 5 seed crops);
    - 9 MARKET;
    - 8 SHOP;
    - 8 CONSTANT (5 crops, 3 animals);
    - CONTROL.
  - Plus FAST and SLOW memory nodes, 8 calendar landmarks and 14 pad nodes.
  - Absent slots (missing hands, etc.) are nodes with a learned ABSENT embedding, not masked keys, so flash attention stays applicable.
- **Node inputs.** Each node type has its own Linear over its own column subset of the trainer's actor features: observation rows, flip extra, v6 job token, v8 token families and knownness bits. Widths:
  - GLOBAL 180, FARM 47, TILE 168, CARRIER 67, STOCK 19, MARKET 182, SHOP 9, CONSTANT 33, CONTROL 1.
  - Added to the input: a slot embedding, and x/y position embeddings on carriers.
- **Calendar.** Landmarks sit at: now, the next shop tick (4-step cycle), dusk, +2 d, +4 d, +8 d, +12 d and terminal (step 718). Each has features for offset, target step, beyond-end flag, log offset, target day and remaining steps. The landmarks carry time only; see Gaps.
- **Relational blocks: 4, width 256, 8 heads (head dim 32).** Each block is pre-RMSNorm, then attention, then a SwiGLU FFN (hidden 704 = 8/3·256 rounded up to a multiple of 64, no bias).
  - Heads 0-3 are relation-typed. They add a learned bias per (relation type, head) from a static typed relation table with 17 types:
    - other, self, tile adjacency, mirror tile, farm membership, carrier-tile, carrier-carrier, custody (own carrier-stock);
    - market-market, shop-market, resource-resource, hub, memory;
    - calendar-later, calendar-earlier (the forward/backward calendar sweep), calendar-resource, pad.
  - Heads 4-7 are global and unbiased (flash SDPA).
  - Relation-specific messages come from head specialisation, not from per-edge W_r matrices.
- **Memory**, sequential over steps:
  - fast GRU (128): input is a pooled graph summary (GLOBAL, own FARM, mean own tiles, mean market) plus an embedding of the actor's own submitted action at t-1 and t-2.
  - slow GRU (256): updated at dawn from the previous day's mean summary.
  - worker GRU (64), shared across the 25 own worker slots: input is the slot's carrier node plus its own previous command. Hands reset at dawn, since they are dismissed at dusk; absent slots are held at zero; the farmer never resets.
  - The GRU inputs come from blocks 1-3, which are stateless and run in parallel over B·L steps.
  - Block 4 reads the memory: FAST/SLOW node inputs, and worker state added to the own carriers.
  - Choice, GPT fast/slow/worker vs Astra persistent + per-day worker: the two agree in substance; we use GPT's split. Memory is read in the last block and the decoder only, so the other three blocks batch over all B·L steps.
- **Decoder**: 2 layers of causal self-attention (flash, is_causal) over the decision positions, cross-attention to the block-4 node states, then SwiGLU.
  - Positions: 25 unit positions (farmer, then hands in engine order), then ≤ 11 order positions ending in STOP; the last slot is forced to STOP.
  - Each unit query adds:
    - the exact same-step prefix row (prefix.PrefixExecutor, the one B uses; teacher rows in training);
    - its worker memory and its carrier node;
    - x/y position, unit_extra and x8_unit;
    - its own previous commands;
    - a per-layer, per-head cross-attention bias from the unit's real ETA (reach_eta) to every own tile.
  - Each order query adds the order-stock interval row and the cash ladder (prefix.order_features / cash_features).
  - Absent unit slots emit ABSENT in both training and play, so teacher-forced and free-running prefixes have identical length.
  - Play is greedy with a K/V cache per layer; node K/V is computed once per step (prefix caching).
- **Heads.**
  - Grammar heads: unit token (44), unit quantity (75 codec bins, conditioned on the token), order token (23), order quantity (75). Output is exactly the codec grammar; agent.py's sanitize_action, fallbacks and play-time rules wrap it unchanged.
  - Aux heads, training targets only: WDL (3-class sign of final cash margin), own next-step unit tokens, opponent current unit tokens.
- **Bias lookups are one-hot matmuls, not index gathers.** The relation bias uses a static [N², 17] one-hot × [17, heads] table; the reach bias uses a one-hot of the ETA bucket × [13, layers·heads]. On GPU the backward of an index gather serialises ~1e5-2e6 duplicate indices, which B's notes measured at 85 % of GPU time in the same pattern. This was fixed at 02:1x, after the 3090 smoke's 1.4 s/update.
- **Parameters: 7,252,244**, measured with `NFlowNet.parameter_count()` at the pilot size. GPT's "pilot" size is width 256, 4 blocks, fast/slow 128/256, worker 64. Astra's mechanism probe (width 192, 6 blocks, 8-12M) was not built.

## Objective and optimisation
- **Per-step loss** (act rows only): mean unit CE (token + quantity) over scored units, plus the sum of order CE (token + quantity) over the order sequence including STOP, divided by 2.
  - A scored unit is present, keep = unit-noop-mask-v1, and on an act row; this matches B's turn-normalised form.
  - Aux weights: WDL 0.2 (all valid rows, including WDL-only), own-next 0.1 (act rows), opponent 0.1 (all rows).
- **Data rule (owner, 09-30).** Action and imitation targets come from winning seats only. Roster v2 has two WDL-only kinds, drawn together for 10 % of stream seats (`--wdl-frac`):
  - wdl-loser: the losing seats of the same games, 2,872 rows;
  - own-loss: seats our teams lost on 09-29/09-30, 7 new rows + 65 already present as wdl-loser + 71 more being built. Our losses from before 09-29 are removed, including 338 that had been wdl-loser rows.
  test_wdl_only.py and test_roster_weights.py prove these rows never enter the action loss.
- **WDL closeness weight (owner, 09-30).** w = 1/(1+|margin|/M), M = median |margin| over the roster (6,097), mean-normalised to 1. The weight is per row (roster column wdl_w, also in the MANIFEST). The WDL loss is Σ CE·w / #valid steps. Range 0.048-1.99, ESS 11,706 of 14,737.
- **Optimiser.**
  - torch.optim.Muon (match_rms_adamw, momentum .95, Nesterov, 5 NS steps) on the 2-D weights of `blocks.*` and `dec.*`; fused AdamW (β2 .95, wd .05) on everything else. Same split and hyperparameters as B's train.py.
  - lr 3e-4, warmup 300, cosine or WSD schedule.
  - Grad clip 10, B's value (coordinator 02:2x). The 3090 smoke clipped every update at 1.0. Clip rate is logged; `--adamw-lr-mult` scales the AdamW groups only.
  - EMA .999, bf16 autocast.
  - Gradient noise scale (McCandlish two-batch estimator over the `--accum` micro-batches) is logged as gns_B_simple_steps (in graph-steps) and grad_snr.
  - `--compile` wraps the stateless encoder and the post-memory encoder+decoder in torch.compile. Shapes are static: N = 320, 25 + 11 positions, fixed B·L.
- **Truncated BPTT.** B streams (workers × streams per worker) each walk a whole game in chunks of L = 24 steps (one day per chunk). Memory is carried, detached, across chunks, so it is the exact causal memory from step 0, stale by at most one update's parameter change.
  - A stream's first seat starts at a random chunk to decorrelate streams. Its memory cannot be reconstructed, so all its chunks are loss-masked (warm).
  - Choice, Astra "reconstruct causal memory per chunk" vs GPT "do not zero-init arbitrary windows": streaming satisfies both without per-window burn-in.
- **Guards (ASTRA CHECK-NFLOW P0s, added 03:0x):**
  - every loaded seat must match its roster row (eid, seat, wdl, act_loss) with steps contiguous from 0, and an action row must be a win by cash, else the seat is skipped and logged;
  - compaction fails closed on any step gap;
  - an update with a nonfinite gradient norm or with no scored step is skipped and logged; memory state is zeroed where nonfinite;
  - a checkpoint with nonfinite model/EMA weights is refused and the run stops;
  - `<out>/STOP` makes the trainer checkpoint and exit at its next log boundary.
- **Bucket sampler (owner 09-30):** `--bucket-probs A=.35,B=.20,C=.20,D=.15,E=.05` draws act rows by their roster bucket, uniformly within the bucket; empty buckets are renormalised away. WDL-only rows keep their own 10 % draw. `roster.py --buckets DIR` builds the roster from gpu-ops' A..E seat lists.
- **Resume binding (PRO/ASTRA R2 #6):** a resume compares 23 recipe keys against the checkpoint and refuses undeclared drift: lr, schedule and decay, clip, optimizer, EMA, chunk, accum, wdl_frac, bucket_probs, aux weights, total streams, roster sha256. `--allow-recipe-change` declares a change and RUN-*.json records it. `--new-seed` declares a fork (fresh RNG and data streams). Deploy refuses a checkpoint whose columns_sha256 differs from the shipped COLUMNS.json.
- **Resumability.** A checkpoint every `--ckpt-every` updates holds model, EMA, optimiser, update, args, RNG and config. `LATEST` names the newest; a restart resumes from it. `--max-seconds` stops cleanly with a checkpoint. Data streams restart on resume, and their first seats are warm-masked.

## Data
- Roster `rows-nflow.1ad4d8296e84.tsv` comes from gpu-ops' wins-only broad roster (exclusions per DATA-PLAN-0930.md). Weights are B's winner, opponent-strength and recency weights × tier factor.
  - Tier mass: tier1 (09-24..27) .241, tier2 (09-28..30 non-grabber + 09-28 grabber) .360, tier3 (grabber/top-team from 09-29) .399.
  - Rows: 6,476 / 2,919 / 2,463 act rows, plus 657 / 841 / 925 WDL-only rows.
- Shards are nflow-shard-v1 (DATA-FORMAT.md): ~3.2 MB per seat, ~45 GB in total. Writes are deterministic (sorted members, fixed zip timestamps). `content_sha256` identifies the content across zlib builds.
- The loader takes each seat's .nfz when it exists, else its .stp compacted on the fly by the same encoder (`--stp-dir`). It re-lists every 10 min, so the Colab run starts on partially staged data.

## Choices where the reviews differ
| topic | GPT | Astra | chosen, reason |
|---|---|---|---|
| size | pilot w256 / 4 blocks | probe w192 / 6 blocks, 8-12M | GPT pilot (owner instruction) |
| calendar | 8 landmarks + backward sweep | ≤30 day nodes × 20 resource classes | 8 landmarks. Fixed shape and cheap; the sweep is the calendar-later/earlier relation types |
| relations | typed edges | relation-specific W_r with shared bases | learned relation biases on 4 of 8 heads, 4 heads global (flash). Per-edge W_r over 320² edges × B·L is too costly for the timeline |
| memory | fast/slow/worker | persistent farm + per-day worker | GPT's split. Reads in the last block and the decoder |
| recurrence training | causal summaries, no zero windows | 24-step TBPTT with reconstructed memory | whole-game streams, 24-step chunks, carried memory |
| outcome target | 3-class WDL | strict win for RL, draws reported | 3-class WDL aux (owner) |
| quantities | 1-100 classes + decimal digits | same | existing codec 75 bins (hard rule: emit the existing grammar) |
| losing seats | not as imitation targets | may train outcome prediction | WDL-only rows (owner approved) |

## Measured
| run | hardware | steps/update | s/update | steps/s | GPU mem | notes |
|---|---|---|---|---|---|---|
| smoke 4593 | 3090, eager | 768 | 1.40 | 549 | 21.1 GB | index-gather bias backward, since replaced |
| colab-nflow | RTX PRO 6000 (Blackwell), torch.compile | 3,072 (64 streams × 24 × accum 2) | 0.60 | 5,110-5,180 | 35.4 GB | u690: loss 3.29, unit-token acc .754, order-token acc .728, exact step .155. Restarted 03:26 at u3000 onto the bucket roster v3 with the P0 guards: 0 skips through u4160 |
| nflow-3090-a (gpu-ops, job 5198) | 3090, eager | 1,728 (18 × 24 × accum 4) | 1.92 | ~900 | 11.8 GB | seeded from Colab u2500 |

Free-running receipts (cold verify vs PASS, whose cash stays $3,000; EMA weights; own cash seat 0 / seat 1):
| checkpoint | own cash | fallbacks |
|---|---|---|
| init | $64 / $64 | 0 |
| u2000 | $9,639 / $3,340 | 0 |
| u2500 | $67,351 / $65,210 | 0 |
The u2500 run went through the full P2 chain: archive → verify → parity → finalize_snapshot all_pass.

## Known gaps (not done or unproven)
- No playing strength is measured. The only games so far are cold-verify games of a random-init archive vs PASS.
- Forecast resource buckets are not computed. Landmarks carry time features only; any production/deadline-to-landmark link must be learned through attention to tile fields.
- There is no rival-holdings belief head. The opponent aux head covers current unit tokens only; opponent order/quantity labels are stored but unused.
- COLUMNS.json keeps columns seen varying in 48 sampled seats. A rarer column is dropped, identically in training and play.
- Slow memory gets gradient only inside a chunk: the previous day's summaries entering the dawn update are detached, which is truncated BPTT.
- GRU state runs under bf16 autocast in training and fp32 in CPU play. Serve/train parity is proven in fp32 only.
- Hands beyond 25 PASS (codec MAX_UNITS), as in B.
- WDL-only draw fraction (10 %), aux weights and order normalisation (/2) are unmeasured choices.
- Resume restarts data streams: up to ~30 updates of partly warm-masked loss after every restart.
