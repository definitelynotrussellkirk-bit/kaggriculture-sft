# Feature ideas from the old farm repo

Scope: ideas only; no farm encoder is ported. Input contract for every feature: one seat's
`training.bc.actor_observation` (clock, both public farms, own shed/seeds/carried, public
market, visible shops) plus that seat's own past wire actions. Nothing reads the rival's
private state, the episode seed, future shop draws, future actions or outcomes.

"pack" = `src/farm_neat/policy/observation.py` `pack(actor_obs, None, cfg)` (GLOBAL, FARM,
TILE, CARRIER, MARKET, STOCK, SHOP, CONSTANT tokens). "theorem" = `theorem_features.py`
(other worker, same build). "history" = `history_features.py` (this file set).

## Sources read (sha256)

| File | sha256 |
| --- | --- |
| /home/russ/Desktop/farm/train/spatiotemporal.py | 9f61b2c6a2c32f52846ea3049f5dad3bb9d4c625af8f5ac658ed81c4ae9e04e6 |
| /home/russ/Desktop/farm/train/fused_v64.py | 66f9ef555e6a04a698e98192d5d5260ae65a12bd3b3377b91ea67fa69bfe0faa |
| /home/russ/Desktop/farm/agent/planning_trackers.py | b846c6e3500cb292bea67c23ecafed02067ff610b68ac285ab067f63e7e33eb1 |
| /home/russ/Desktop/farm/train/encoder.py | 3835a3c29e4e5fdd1ac143ceb4d20384225957d88f4fe422ae0bd59e470446b9 |
| /home/russ/Desktop/farm/neural/current_neural_features.py | 5ba8ef1ee8a84d87f34628425c7d2122271e390965a1f6ada4c7b8da9b923c3f |
| /home/russ/Desktop/farm/neural/current_neural_history_knownness.py | fa7f7a53dd9070a089e79d0282887078cebbb0164e6a9974e5d5766ac9ad8082 |
| /home/russ/Desktop/farm/plan_player_bridge/opponent_projection.py | aedbb64fae73a91f27dd1efbf998dc13d5b3fedc82077fa52f77e3b333faddd7 |
| /home/russ/Desktop/farm/agent/market_analysis.py | 467e9b4ff2b0cadb0e36f57ea175335771b8ca47e2136d4402dee48a6a96c461 |
| /home/russ/Desktop/farm/agent/relative_heuristics.py | 7e30503fd9698126725b88df35b022a6bcade25596bffe39a8ffc9986ed123a0 |
| /home/russ/Desktop/farm/train/resource_features.py | 332d893ef297585c2d6f97db5eeb1d4c1662ce3cd55abfbb4a121f0864f3a516 |
| /home/russ/Desktop/farm/train/opponent_model.py | f7cb78becee9a7d30375bcb315c3a765d8646abe2e9fbeb862b943c93ef925f2 |
| /home/russ/Desktop/farm/train/daily_tom.py | 920d29748da2f09842f35575d9ac78bce60dbdaa09c7e338a24d229e46eb0b50 |
| /home/russ/Desktop/farm/neural/economic_residual_history.py | e7d731f10ba66b83fc251c82953156c55858e0b79ba534e0847882bdb453c4ed |
| /home/russ/Desktop/farm/neural/current_neural_v2_1_public_opponent_context.py | 618e4c73e123e5e2231d9ea762768a3221db39d0bb4436f1be6283fa3120866e |
| /home/russ/Desktop/farm/agent/net.py | 7851a2418177353079c4f8470ee35e338b4e30050e641a27b2a20d42fc1a1d8c |

Abbreviations: ST = train/spatiotemporal.py, FV = train/fused_v64.py, PT =
agent/planning_trackers.py, EN = train/encoder.py, CN = neural/current_neural_features.py,
HK = neural/current_neural_history_knownness.py, OP = plan_player_bridge/opponent_projection.py.

## V64 view (SpatialHistoryCodec + FusedV64)

| # | Idea | Encodes | Source | In pack? | Decision |
| --- | --- | --- | --- | --- | --- |
| 1 | Board channels | tile kind / crop / animal one-hots; tile booleans and numerics with applicability | ST:79-93, ST:581-603 | yes: TILE kind/item one-hot + TILE_FIELDS with knownness | skip: carried |
| 2 | Unit occupancy maps | farmer/hand positions per role | ST:605-633 | yes: CARRIER xy | skip: carried |
| 3 | tanh tile scaling | squashed tile numerics | ST:566-579 | pack divides, no clip | skip: normalisation choice, not information |
| 4 | Scalar clock/config | step/day/hour fractions, turnsPerDay, shedCapacity, hire mult | ST:638-652 | yes: GLOBAL | skip: carried |
| 5 | Scalar farm/private | money, hires, units, quadrants, shed/seed per item, per-unit inventory + key-order rank, shop counts | ST:653-710, ST:386-419 | yes: FARM, STOCK, CARRIER (incl. key_rank), SHOP | skip: carried |
| 6 | Market base channels | price/base-1, (inv-I0)/T, curve params, curve function one-hots | ST:712-742 | params and raw price/inventory yes; the two ratios no | keep ratios: history `mkt.price_vs_base`, `mkt.inv_vs_target` (no tanh) |
| 7 | Market trend channels | per-step price and inventory first differences, trend_available | ST:95-99, ST:794-803 | no | keep: history `d.price_rel`, `d.mkt_inv` at lags 1/4/24/72; availability as knownness |
| 8 | History window | last 3 full frames, left-padded, mask, time ids | ST:532, ST:744-756, ST:788 | no | keep idea as lag deltas (not full frames: token/CPU cost) |
| 9 | Absolute seat id | player_ids | ST:787 | yes: GLOBAL seat | skip: carried |
| 10 | clock.day/hour_remaining | 1 - fraction | ST:289-290 | no | theorem (`steps_remaining`, `day_actions_remaining`) |
| 11 | money self/opp/lead | cash and difference | ST:291-293 | cash yes | numeric: theorem `cash_diff`; binary sign: history `flag.cash_lead`, `flag.cash_tie` |
| 12 | hires / unit count / unlocked fraction, relative | differences of FARM fields | ST:294-302 | yes | skip: carried; theorem has hands/land counts |
| 13 | shed_used / fill / free | shed occupancy | ST:303-305 | no | theorem `shed_used`, `shed_room` |
| 14 | seed_total, carried_total | sums | ST:306-307 | no | theorem `own_seed_total`, `carried_total` |
| 15 | carried mean/max per unit, loaded fraction | load distribution over units | ST:308-310 | per-carrier yes | skip: per-carrier loads carried; return-trip form kept (#40) |
| 16 | board fractions per kind/crop/animal per role | tile counts | ST:312-314 | per tile yes | current counts: theorem `own/rival_producers`; changes: history `d.{own,rival}_{plants,animals,weeds}` |
| 17 | tile boolean true/false fractions per role | e.g. share unwatered / unfed / uncared / fertilizer available | ST:315-322 | per tile yes | theorem `*_unserviced_today`, `*_uncared_animals`, `*_fertilizer_collectable` |
| 18 | tile numeric means per role | mean planted_day etc. | ST:323-329 | per tile yes | skip: low information; per-tile clocks in theorem tile block |
| 19 | unit centre Manhattan | each unit's distance to board centre | ST:330-333 | positions yes | replaced by exact shed-access return steps (#40) |
| 20 | market summary mean/min/max | across-product pooling | ST:342-350 | no | skip: FV:443-470 records that pooling across products lost crop identity (78% then 85% crop accuracy); keep per-product rows instead |
| 21 | has_money flags | cash > 0 | ST:357-358 | cash yes | skip: near-constant |
| 22 | shed_empty / shed_full / has_seed / has_carried | binary flags | ST:360-363 | no | keep: history `flag.*` |
| 23 | quadrant / unit present flags | binary | ST:366-369 | yes | skip: carried |
| 24 | has_crop / has_animal per role | any tile of a kind | ST:371-372 | per tile yes | theorem producer counts (>0) |
| 25 | has_false per tile boolean | any unwatered/unfed/... tile | ST:373 | per tile yes | theorem counts (>0) |
| 26 | price_rising / price_falling | sign of last price change | ST:377-380 | no | keep: history `mkt.price_up1`, `mkt.price_down1` |
| 27 | inventory_below_target | scarcity regime | ST:379 | no | keep: history `mkt.scarce` |
| 28 | Numeric basis expansion | value, abs(x), x², sign, log, sin/cos pi, is_zero; FV: x, signed log1p, tanh, x²/(1+x²) | ST:140-149, ST:3462-3478, FV:44-59 | n/a | skip as feature: model-side input transform (model.py owner) |
| 29 | Predicted embeddings / macro successor heads | self-supervised successor targets | ST:150-157, FV:298-313 | n/a | skip: targets, not inputs |
| 30 | Preference goals | objective conditioning names | ST:158-171 | n/a | skip: not observation information |
| 31 | Semantic effects, quantity/money signs per op | per-action effect targets | ST:172-220 | n/a | skip as target; realised own effects are history lag-1 deltas |
| 32 | Legality field roles | legality model | ST:245-265 | n/a | skip: constraint, owner asked for features not constraints |
| 33 | Behavior conditions | date and viewer/opponent rating | ST:266-273 | no | skip: ladder ratings are not actor observation |
| 34 | Opponent action codec / opponent model | rival's next wire action | ST:123-138, train/opponent_model.py:139-261 | no | skip as input: the rival's action is not visible; the public inference is kept (#44-47) |
| 35 | Planning axes | space / transportation / money labels | ST:280 | n/a | skip |
| 36 | Product identity preserved | lossless crop-market channels beside pooled tokens | FV:443-470 | per-product MARKET tokens yes | keep as structure: history item blocks are one row per item |
| 37 | History-mean tokens | current + history-mean board/scalar/market tokens | FV:473-486 | no | covered by lag deltas (#8) |
| 38 | Autoregressive head conditioning | earlier decoded heads condition later heads | FV:188-203, agent/net.py:68-79 | n/a | skip: decoder design (model.py) |
| 39 | Self/opponent fusion | concat, difference, product of role embeddings | FV:265-269 | n/a | skip: model-side |

## Planning trackers, encoder, public-history and other farm feature code

| # | Idea | Encodes | Source | In pack? | Decision |
| --- | --- | --- | --- | --- | --- |
| 40 | transport return demand / shed distance / terminal stranding | per loaded unit: moves to nearest shed-access tile + 1 PLACE/DROP; items that cannot be deposited before the last action | PT:97-99, PT:112-113, PT:140-143, PT:348-398 | positions and loads yes; distances no | keep: history `own.loaded_return_steps`, `own.max_return_steps`, `own.stranded_carried` (Manhattan is exact: locked tiles traversable, units never block; stranding only on the final day, which has no day-end deposit) |
| 41 | feed.cover_days | own wheat (shed + carried) / own animals | PT:107, PT:339-346 | no | keep: history `own.feed_cover_days` |
| 42 | feed.urgent_shortfall | urgent animals - wheat | PT:108 | no | skip: theorem `*_dying_tonight` and wheat quantities give it |
| 43 | production.output_at_cap_tiles | tiles whose stored yield is at cap | PT:110, PT:325 | per tile yes | keep: history `own.at_cap_tiles`, `rival.at_cap_tiles` |
| 44 | clock.* trackers | turns to terminal, actions until EOD, day-ends left | PT:80-82, PT:242-250 | no | theorem |
| 45 | liquidity.shed_spot_sale_value / floor value | integrated sale revenue of shed stock | PT:86-87, PT:284-298 | no | theorem `own_shed_liquidation`, `sell_receipts_k` |
| 46 | threshold.next_hire_cost / next_land_cost | Fibonacci hire cost, next land price | PT:88-89, PT:299-306 | no | theorem |
| 47 | space.* | shed used/slack/overflow, EOD projected slack | PT:90-94 | no | theorem `shed_room`, `dayend_overflow` |
| 48 | service.* | plants/animals due and urgent, minimum actions, action slack | PT:100-105, PT:320-323 | per tile yes | theorem counts; action slack skipped (derivable) |
| 49 | production.harvest_ready_tiles / output_headroom | harvestable tiles, capacity left | PT:109, PT:111 | per tile yes | theorem `*_harvestable_units`, tile `yield_room` |
| 50 | investment.latest_physical_plant_day_slack | days left to plant each crop and still harvest | PT:121-128 | no | theorem structure block (`first_output_days`, season units) |
| 51 | market spot / next buy price | quote now and after one buy | PT:131-134 | price yes | theorem `buy_quote` |
| 52 | opponent public unit count / occupied tiles | rival labour and board use | PT:114-115 | yes | skip current; changes in history `d.rival_units`, `d.rival_{plants,animals,weeds}` |
| 53 | Encoder aggregates | money tanh, tile-kind counts, plant_ready, plant_thirsty, plant_mean_age, n_animals, opp crop counts | EN:47-66, EN:93-159 | mostly | skip: carried or theorem; plant_mean_age low information |
| 54 | Encoder aux targets | future prices, money, opp money, next features, win, margin | EN:169-219 | n/a | skip: future information; allowed only as training targets |
| 55 | Sparse actor projection | tile-token counts, hand position mean/min/max, market, shops, private stocks | CN:44-133 | mostly | skip: carried or theorem |
| 56 | Public-opponent lag deltas at 1/24/72 | rival cash, units, land, crop/animal tiles, visible yield, unfed/urgent animals | CN:21, CN:136-207, OP:269-283 | current values yes | keep: history lags (1, 4, 24, 72): `d.rival_cash`, `d.rival_units`, `d.rival_quadrants`, `d.rival_{plants,animals,weeds}`, `d.rival_yield` per product |
| 57 | Lag availability | farm fell back to the earliest snapshot silently (CN:187); later fixed with exact-lag flag + realised age | CN:187, HK:18-25 | no | keep: history `lag.exact`, `lag.age`; values use the oldest snapshot within the lag, knownness false only when no earlier step exists |
| 58 | Rival visible held yield per product | public supply the rival can sell soon | OP:141-196, OP:267 | per tile yes | theorem `rival_field_stock`; its changes in history `d.rival_yield` |
| 59 | Rival earliest sale step bound | min rival unit->tile + tile->shed + 1 per supply tile | OP:84-100, OP:205-220 | no | skip: theorem tile block carries per-tile `unit_distance`, stored yield and harvestability |
| 60 | Painted intents / sensitivity queue | plan-level opponent intents | OP:40-66, OP:222-262 | n/a | skip: needs a planner |
| 61 | Market geometry | regime, finite differences, units to floor, pairwise flips | agent/market_analysis.py:110-155, :173-211 | no | theorem `sell_quote_after_k`, `floor_distance`, `floor_locked` |
| 62 | Grid path / capacity slack / latest start | BFS path, slack, deadline start | agent/relative_heuristics.py:317, :401, :409 | n/a | used as idea for #40 (BFS unnecessary: movement is unblocked) |
| 63 | Candidate resource ledgers | budget, slack, shadow price, contention per candidate | train/resource_features.py:56-99 | n/a | skip: needs a candidate set; the per-step policy has none |
| 64 | Residual history / forecast context | projected-vs-reality residuals, candidate-conditioned rival forecasts | neural/economic_residual_history.py:22-50, neural/current_neural_v2_1_public_opponent_context.py:55 | n/a | skip: plan/forecast-model inputs |
| 65 | Daily top-ladder selection | episode selection by terminal income | train/daily_tom.py:1-10 | n/a | skip: outcome-selected metadata, not an input |

## Added here (not in the farm code)

| # | Idea | Exactness | Fields (history_features.py) |
| --- | --- | --- | --- |
| 66 | Phase encodings | exact | `phase.hour_sin/cos`, `phase.tick4_{0..3}` (shop consumption period), `phase.day_mod3_{0..2}` (shop unlock period), `phase.day_end_follows`, `phase.final_action` |
| 67 | Town consumption per product over each lag | exact given the visible shop list (engine `_town_consume` uses the shops visible at that step) | `town_consumed`, `cum.town_consumed` |
| 68 | Trade flow per product: market inventory change + town consumption = both players' net trade supply | exact, except a sale at the $1 floor adds no supply | `trade_flow`, `cum.trade_flow`, `cum.floor_steps` |
| 69 | Own / rival split of the trade flow | own: holdings identity bounded by own submitted orders, $1-floor cap, total flow; rival = flow - own | `own.trade_est`, `rival.trade_est` |
| 70 | Rival goods obtained from public tiles | inferred harvests (stored yield taken), fertilizer collections, feeds and fertilizer uses | `rival.obtained_est`, `rival.obtained_cum_est` |
| 71 | Rival hidden stock estimate | running obtained - used - traded, clamped at 0, from episode start only | `rival.stock_est` |
| 72 | Rival cash spent on trades | rival cash change + visible hire and land costs | `rival.trade_cash_est`; own counterpart `own.trade_cash` |
| 73 | Own discard risk | steps whose own shed ended at capacity | `own.discard_risk_steps` |
| 74 | Previous own joint action | own action at step-1: unit op counts, market order counts, per-item quantities (clamped like the engine/codec), and `codec.encode_step` tokens when codec.py imports | `prev.*` fields, `HistoryOutput.prev_tokens` |

Known blind spots of #69-71 (all public-information limits): the rival's sales at the $1
floor, collections in the hour-23 step (erased by the day-end refresh), harvests hidden by a
same-night refill, DIG of a one-shot crop (counted as a harvest), and rival discards when its
shed is full.

## Measured (test_history_features.py, 2026-09-23)

Games on the exact engine, one world seed each; values are per (step, product) cell.

| Game | Cells | own `trade_est` exact / on nonzero truth | rival `trade_est` exact / on nonzero truth | `rival.stock_est` exact | mean abs error | zero-predictor exact / mean abs |
| --- | --- | --- | --- | --- | --- | --- |
| SHOP_ROUTER vs K0006, seed 7 | 12,924 / 12,942 | 0.9991 / 0.9858 (n=634) | 0.9991 / 0.9842 (n=634) | 0.8429 | 4.51 (max 159) | 0.5967 / 5.75 |
| SHOP_ROUTER vs engine starter, seed 11 | 12,924 / 12,942 | 0.9999 / 0.9970 (n=335) | 0.9999 / 1.0000 (n=335) | 0.9523 | 0.143 (max 9) | 0.7752 / 2.85 |

In the seed-7 game 369 of 719 joint actions are identical between the two seats; in the
seed-11 game 0 of 719. The largest seed-7
errors are MILK and FERTILIZER while their price sat at the $1 floor.

Town consumption and trade flow matched engine instrumentation (patched `_town_consume`,
`_process_market`, `_commit_unit`) in every cell of the seed-7 game: 718 transitions x 2
seats x 9 products. Cost of `update` + `record_action`
(including codec tokens) over 719 steps x 2 repeats, one thread, host load average 40-80
from parallel builds, 6 runs: mean 0.65-0.85 ms, p99 1.2-5.0 ms. Output: 5 blocks, 787
values (schema `primitive-bc-history-features-v1`).

Unestablished: whether any of these features improves BC accuracy or play strength; the
previous-own-action fields are a known copycat risk in behaviour cloning and have not been
ablated.
