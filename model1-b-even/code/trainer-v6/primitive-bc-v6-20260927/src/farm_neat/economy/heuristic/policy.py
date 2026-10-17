"""Kaggriculture heuristic opponent -- value-driven hierarchical policy.

The strongest heuristic in the repo. Decisions combine exact engine constraints, bounded search,
and dollar values computed from engine constants plus the live market state through the exact
price-curve replica (`econ.py`):

  * Strategy (per turn, closed-form): what each empty tile is worth under every crop,
    whether another goose out-earns the coop tile it would occupy, what fertilizer is worth
    in use versus sold, and how much wheat cover the flock needs. There are no crop quotas
    or buy-cutoff days; late-season and crashed-market behaviour falls out of the NPVs.
  * Tactics (per unit, per turn): every candidate action is a job carrying its computed value;
    the scheduler selects by value density over exact shortest-path distance and gives a still-
    legal route target a measured soft 10-turn commitment bonus. A bounded exhaustive 5--10 turn
    job-order solver models item gains/consumption, deadlines and destructive operations; full
    assignment replacement remains experimental because its economic terminal value is not exact.

Hard engine rules are kept as structure, not values: the final day has no end-of-day
(harvest-haul-sell only), PLANT requests are capped per crop by seeds held (the atomic
rule), hands spawn on locked tiles, and unloading is room-safe (PLACE, never a DROP that
would destroy overflow). Jobs worth less than nothing are not taken.

Design notes inherited from the executor (the things that dominate everything else):

1. **Actions, not money, are the constraint.** Hands cost fib dollars and each buys 24
   actions. The executor's job is to spend those actions on the highest-value work, which
   is exactly what value-density assignment does.

2. **Both players share one market.** Premium curves crash to $1 within a few hundred units
   of glut; selling is marginal-revenue planning (`market.plan_sales`), and the same live
   prices feed every value in `econ.py`, so a crashed crop devalues itself everywhere at
   once -- planting, watering, fertilizing and harvesting all stand down together.

3. **Day `total_days-1` has no end-of-day.** The episode ends at step `episodeSteps-2` (day
   29 hour 22), before the final end-of-day would run: nothing produces, nothing
   auto-drops, and anything still carried is simply lost.
"""

from . import econ
from .constants import (
    ANIMALS,
    CROPS,
    LAND_PRICES,
    MAX_MARKET_ORDERS,
    MOVES,
    PRICE_FLOOR,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    hire_cost,
    is_shed_adjacent,
    shed_access_tiles,
    water_bonus_window,
)
from .market import plan_sales, price_at as market_price_at
from .valuation import Valuation
from .search import (
    enumerate_job_schedules,
    maximum_weight_assignment,
    select_joint_schedules,
    shortest_routes,
)
from .state import View, manhattan, nearest_shed_tile, step_toward
from farm_neat.game.macro import MACRO_DIRECTIVE_VERSION, MacroDirective

# --- tunables ---------------------------------------------------------------
# Held per-agent rather than as module globals, so a sweep or a mirror match can pit two
# differently-configured agents against each other without one clobbering the other.
DEFAULTS = {
    "ACTION_COST": econ.ACTION_COST,  # shadow price of one unit-action; the labour market
    "HANDS_PER_JOB": 0.0,             # if >0, the daily hand target is the job surface divided
                                      # by this, clamped to TARGET_HANDS -- labour that ramps
                                      # with the work instead of a constant from day 1.
                                      # Measured motivation: on episode-102123196 we hire 11
                                      # hands on day 1 against 16 standing plants while the
                                      # frontier tape hires 5 against 19, and the tape then
                                      # ramps 5 -> 10 across days 7-10 as its board goes 19 ->
                                      # 37 plants and 6 -> 13 animals. Every previous sweep
                                      # (6/8/10/12/14) tested a CONSTANT target; the shape of
                                      # the ramp has never been tested. 0.0 keeps the constant.
                                      # ANSWER: the constant wins. Sizing the crew to the work
                                      # costs our OWN score $8,628 [-13,432, -3,931] at 2 and
                                      # $1,415 (unmeasured) at 3. And at 5 the hire curve
                                      # matches the tape's almost exactly -- 3,4,4,4,5,3,5,6,8,8
                                      # against their 5,4,4,5,4,5,8,8,10,10 -- while scoring
                                      # $50,357 against $84,746 on the probe seed. Whatever
                                      # drives their day-7 acceleration, it is not how they
                                      # hire. artifacts/evaluations/hands-per-job-panel-20260831.json
    "TARGET_HANDS": 12,               # expanded-farm mirror winner; exact daily target
    "HIRES_PER_TURN": 4,              # locked spawn tiles are passable since engine 1.32.3
    "HIRE_BUDGET_FLOOR": 0,           # spend at least this on hands even when cash-starved.
                                      # Measured 2026-08-30: 40/80/150 all lose 0/24 vs the
                                      # early-herd solver -- in the cash-starved window every
                                      # dollar is herd feed; keep 0 (named negative control).
    "LAST_HIRE_HOUR": 8,              # hiring later buys too few working turns to pay off
    "SEED_BUFFER": 2,                 # spare seeds beyond the tiles we can plant right now
    "CASH_FLOOR": 150,                # never spend down to where we cannot buy seed
    "DROP_INVENTORY_AT": 12,          # carry this much and a shed trip beats another job.
                                      # Swept 2026-08-30 with the mixed herd vs the frozen
                                      # 7f9fcc7 solver: 12 -> 55/72 (+$3.4k, mirror $84.2k);
                                      # 6 -> 6/24 and 16 -> 3/24, so 12 is a real optimum.
    "BUY_LAND": True,                 # capacity/affordability-constrained expansion
    "MAX_QUADRANTS": 4,               # stop expanding at this many owned quadrants.
                                      # Measured motivation: on episode-102123196 we buy all
                                      # four ($7,000, of which the fourth alone is $4,000)
                                      # while the frontier tape stops at three -- and we then
                                      # weed out 81 tiles across the season, so the extra land
                                      # is not obviously serviceable. This is the direct test.
                                      # ANSWER: buy all four. Against the default, stopping at
                                      # three costs our OWN score $3,288 [-5,216, -1,454] and
                                      # stopping at two costs $10,582 [-13,675, -7,576], both
                                      # significant on 66 held-out cases. The fourth quadrant
                                      # earns its $4,000; the tape's three is not a target.
                                      # artifacts/evaluations/max-quadrants-panel-20260831.json
    "PLANTING_ENABLED": True,         # learned policies switch this off at the end
    "CASH_CROP": "AUTO",             # AUTO preserves the production solver's dynamic NPV crop
    "FERTILIZE_DISCOUNT": 1.0,        # multiplier on the FERTILIZE job's value. Measured
                                      # motivation: against a real opponent FERTILIZE is the
                                      # worst-calibrated operation on the board -- assigned
                                      # $370 against an exact marginal worth of $136 (ratio
                                      # 0.37) and in the exact optimum only 38% of the time,
                                      # where every other operation sits at 0.92-1.00 and
                                      # 98-100%. 0.0 disables the job entirely.
                                      # ANSWER: keep it at 1.0. Discounting to 0.37 costs our
                                      # own score $3,597 [-388, +7,704] (unmeasured) and
                                      # removing it costs $6,919 [+1,895, +12,097], significant
                                      # and surviving Holm. The low ratio is the WINDOW, not a
                                      # mispricing: FERTILIZE covers three days and an 8-turn
                                      # oracle cannot see past its own horizon. This is the
                                      # deferred-vs-mispriced limitation, measured.
                                      # artifacts/evaluations/fertilize-discount-panel-20260831.json
    "SELL_ORDER_BUDGET": 5,           # SELL orders per turn. The market cap is 10 and the
                                      # queue never reaches it -- 0% of turns hit the cap and
                                      # 82% issue 0 or 1 order -- so this is a self-imposed
                                      # limit, not an engine one. Raising it changes nothing:
                                      # budgets 8 and 10 produce byte-identical panels, and
                                      # both sit at +$69 [-90, +259] against 5. The budget
                                      # never binds. artifacts/evaluations/sell-budget-panel-20260831.json
    "SELL_MIN_RATIO": 0.0,            # floor on sale price as a multiple of base price.
                                      # Holding for a better price loses monotonically: 0.15
                                      # costs own score $2,569 [-4,213, -1,013] and 0.35 costs
                                      # $18,875 [-22,314, -15,389], both significant, and both
                                      # hand the opponent $2,631 and $4,404 of the market we
                                      # declined to take. Sell on sight.
                                      # artifacts/evaluations/sell-min-ratio-panel-20260831.json
    "MAX_GEESE": 6,                   # hard cap; NPV decides below it. Swept 6/8/10 vs v1:
                                      # 6 best -- the NPV overvalues geese 7+ (congestion
                                      # and feed-logistics risk it does not price in)
    "FEED_PLANT_COVER_DAYS": None,    # days of feed the growing wheat should cover when
                                      # deciding what to PLANT / which seeds to buy; None =
                                      # FEED_COVER_DAYS. Measured 2026-08-30: 2/3/4 lose
                                      # 9/24, 3/24, 5/24 vs the early-herd solver -- tiles
                                      # earn more as cash crops than as feed (negative control).
    "FEED_COVER_OPENING": 3,          # feed cover to hold during the OPENING only (None = off).
                                      # Distilled from the grpo4/grpo5 macro networks, whose
                                      # ENTIRE learned difference from production was the feed
                                      # head: every other head matched the solver.  Swept
                                      # 2026-08-30 vs production over two seed blocks x both
                                      # seats: cover 3 for days 0-1 won 44/48 (+$7.1k/+$12.2k
                                      # mean score), cover 3 for days 0-2 won 35/48, and cover
                                      # 2 did nothing (24/48).  No regression on the seven-tape
                                      # frontier panel (-$51.4k vs -$52.6k mean margin).
    "FEED_COVER_OPENING_DAYS": 2,     # days the opening cover applies (day < N).  Stopping
                                      # after day 1 beats running through day 2: opening wheat
                                      # pays while the opponent is not also buying it, and the
                                      # shared market reprices once both do.
    "FEED_COVER_DAYS": 1,             # days of feed the wheat pipeline must cover. Swept
                                      # 2026-08-30 with the mixed herd: 1 beat 2 by 63/72
                                      # (+$8.2k); with ~20 animals two days of cover hoards
                                      # 40 wheat in a 100-slot shed. Adaptive scarcity
                                      # triggers (1.2x/1.6x base) were measured harmful.
    "SUNK_ANIMAL_PLACEMENT": True,    # value PICKUP/PLACE of an animal we already own at its
                                      # remaining production, with the purchase treated as the
                                      # sunk cost it is. False restores the old behaviour, in
                                      # which a bought animal became unmovable as soon as a
                                      # FRESH purchase would have been unprofitable.
    "HERD_SERVICE_DISCOUNT": 0.0,     # weight on discounting a prospective animal's NPV by how
                                      # well we are actually servicing the herd we already have.
                                      # `animal_npv` values an animal assuming it is fed and
                                      # survives; measured on episode-102123196 we buy 23 and
                                      # keep 14 -- COW 9/9, SHEEP 3/7, GOOSE 2/7 -- so the NPV
                                      # ranking (SHEEP $7,797 > COW $6,781 > GOOSE $3,758)
                                      # inverts once multiplied by survival ($3,342 / $6,781 /
                                      # $1,074). The discount uses the share of the standing
                                      # herd that is NOT one miss from escaping, which is
                                      # observable at the moment of purchase. 0.0 disables it.
                                      # ANSWER: no. Inert at 0.5 (identical herd and score to
                                      # the dollar), and at 0.75 it costs our OWN score $2,865
                                      # [-5,285, -439], significant. The NPV-times-survival
                                      # reframing is arithmetically true and not actionable,
                                      # because survival is ENDOGENOUS: buying fewer sheep does
                                      # not make the remaining ones survive, it just means
                                      # fewer animals. At 1.0 the herd collapses to 9 cows and
                                      # 5 geese with no sheep at all, for $59,091.
                                      # artifacts/evaluations/herd-service-discount-panel-20260831.json
    "HERD_SCALE": 1.0,                # multiplies every herd cap at once. Measured motivation:
                                      # on episode-102123196 the animal line returns ~$2,950
                                      # per animal (MILK $35,545 alone is 43.2% of all our
                                      # revenue, plus EGG, WOOL and the free FERTILIZER at
                                      # $8,107) while a crop tile nets $52-$211 -- strawberry
                                      # $311/tile against a $100 seed, melon $132 against $80.
                                      # That is a 14-56x difference per tile, and the caps have
                                      # only ever been swept DOWNWARD (MAX_COWS 8 vs 2,
                                      # MAX_GEESE 3/4/6/16). Milk is not glutted either: we
                                      # sold 172 units at a $207 mean against a $160 base, so
                                      # the town was draining it faster than we supplied it.
                                      # ANSWER: no. A single seed read $66,054 -> $86,780 at
                                      # 1.5x and $88,209 at 3x, the largest probe movement of
                                      # the session; the 66-case panel reversed it
                                      # monotonically -- own score -$4,112 [-7,059, -1,226] at
                                      # 1.5x, -$4,430 at 2x, -$7,119 [-10,381, -3,872] at 3x,
                                      # every arm significant and surviving Holm. The per-tile
                                      # return argument is sound and the conclusion drawn from
                                      # it is wrong; the herd is cash-bound to day 20, not
                                      # cap-bound. Fifth single-seed reversal recorded here.
                                      # artifacts/evaluations/herd-scale-panel-20260831.json
    "MAX_COWS": 8,                    # pasture herd caps; NPV decides below them. Cows and
    "EARLY_HERD_TARGET": 6,           # animals to fund ahead of the seed budget while
    "EARLY_HERD_DAYS": 3,             # day <= EARLY_HERD_DAYS (cows need 8 days to produce).
                                      # Re-gated 2026-08-31 on 66 held-out cases because the
                                      # frontier tape goes 6 -> 13 animals across days 7-10
                                      # where we go 3 -> 5, which looked like an acquisition
                                      # ORDER difference rather than a threshold one. It is
                                      # not: TARGET 9 and 12 cost our OWN score $10,151
                                      # [-16,081, -4,055] and $9,233 [-15,505, -2,854], both
                                      # significant, and extending DAYS to 6 or 10 moves own
                                      # score by +$372 and -$4 -- nothing at all. The opening
                                      # is at its optimum in both size and duration.
                                      # artifacts/evaluations/early-herd-{target,days}-panel-20260831.json
                                      # Swept 2026-08-30 vs the frozen d75732f solver: 6 ->
                                      # 72/72 (+$32k mean, mirror $88.2k); 4 -> +$6.7k;
                                      # 2 and 10 lose (a half-herd delays seeds for nothing;
                                      # ten starves them); 8 is a wash.
    "SAME_TURN_FEED_CARE": False,     # let one turn both FEED and CARE an unfed animal. The
                                      # engine's CARE has no `fed_today` precondition, so the
                                      # pair IS legal in either order and does bank at end of
                                      # day -- three review returns were right that the job
                                      # surface could not express it. Taking it greedily is a
                                      # different question and the answer is no: on 66 held-out
                                      # cases our OWN score falls $6,082 [-9,712, -2,577] with
                                      # the opponent's unmoved, so the extra CARE actions cost
                                      # more than the bank they buy. Kept as a measured control.
                                      # artifacts/evaluations/same-turn-feed-care-panel-20260831.json
    "DYNAMIC_ACTION_COST": False,     # price a unit-action as the marginal job it displaces
                                      # (`econ.action_shadow_price`) instead of the flat $7.
                                      # All six review returns asked for this; it is a control
                                      # until measured, because $7 is calibrated and a dual
                                      # that moves every turn changes every downstream NPV.
    "UNIT_AFFINITY": 0.0,             # soft per-unit preference for the KIND of work it did
                                      # last, as a fractional bonus to value density. This is
                                      # the errand-specialisation idea from Oxygen Not Included,
                                      # where each Duplicant carries a per-errand priority and
                                      # tends to keep doing the same class of work. Our units
                                      # are fully interchangeable, which is the one structural
                                      # difference from that design worth testing -- and the
                                      # positional sibling, ROUTE_STICKINESS, is measured
                                      # non-monotonic with an optimum at 0.25, so a soft
                                      # preference is known to work here. 0.0 disables it.
                                      # ANSWER: unmeasured at every strength. Own score
                                      # -$1,918 / -$1,959 / +$59 at 0.15 / 0.35 / 0.6, all
                                      # intervals spanning zero, none surviving Holm. It also
                                      # fails to produce the behaviour it was built for: mean
                                      # top-class share per unit FALLS 56.1% -> 49.6%, because
                                      # every unit then prefers the same class, only one can
                                      # take each job, and the losers scatter. Specialisation
                                      # in ONI comes from priorities assigned per worker, not
                                      # from an emergent same-class bonus.
                                      # artifacts/evaluations/unit-affinity-panel-20260831.json
    "ONGOING_FERTILIZE": 0.0,         # weight on fertilizing an ONGOING crop. `econ` returned
                                      # 0 for these unconditionally ("hard to time"), so 0% of
                                      # our 61 strawberry tiles is ever fertilized -- while a
                                      # watered AND fertilized production day yields +2 instead
                                      # of +1, i.e. double that event's output. 1.0 prices the
                                      # productions that actually land inside the three-day
                                      # cover window; 0.0 is the historical behaviour.
                                      # ANSWER: the hardcoded zero was right. Own score -$2,127
                                      # [-3,479, -732] and margin -$3,780 [-4,838, -2,730], both
                                      # significant. The fertilizer really does pay better sold.
                                      # artifacts/evaluations/ongoing-fertilize-panel-20260831.json
    "ONGOING_HARVEST_AT": 1,          # held units an ONGOING crop must carry before we make a
                                      # harvest trip. 1 is the historical behaviour: harvest the
                                      # instant anything is held, which on episode-102123196
                                      # means 181 STRAWBERRY harvests at exactly 1.00 unit each
                                      # against a cap of 4 -- four visits for what one could
                                      # carry. Raising it trades trips against decay risk.
                                      # RESULT, and it is the first genuinely ambiguous gate of
                                      # the session -- all three statistics move and they do not
                                      # agree. At AT=4 our OWN score rises $2,392
                                      # [+1,466, +3,344] (significant, surviving Holm) and wins
                                      # go 3/132 -> 5/132, while the OPPONENT's rises more,
                                      # $3,896 [+3,034, +4,776], so margin falls $1,504
                                      # [-2,728, -281]. Holding strawberry off the market longer
                                      # leaves them a less glutted market to sell into. Left at
                                      # 1. REPLICATED on 66 fresh seeds and the picture got
                                      # clearer, not better: pooled over 132 cases own score
                                      # +$1,872 [+1,066, +2,655], opponent +$4,429
                                      # [+3,833, +5,039], margin -$2,557 [-3,551, -1,577] --
                                      # and the win-rate gain (3/132 -> 5/132) did NOT
                                      # replicate, coming back at exactly +0.000. So the
                                      # ladder statistic does not move and the margin
                                      # statistic gets worse; only our own cash improves, and
                                      # it improves less than we hand over. Left at 1.
                                      # artifacts/evaluations/ongoing-harvest-at-{panel,replication}-20260831.json
    "CROP_TILE_CAP": 0,               # max standing tiles of any ONE crop; 0 disables it.
                                      # Measured motivation, not a guess: on
                                      # episode-102123196 we stand 60-61 STRAWBERRY tiles from
                                      # day 15 where the frontier tape holds 33 falling to 16,
                                      # and the seed bill for them is $6,100 of a $9,990 total
                                      # against a $25,403 total spend. STRAWBERRY's glut curve
                                      # is linear/1.60 with T=100, so it reaches the $1 floor
                                      # about 100 units past I0 -- and 61 ongoing tiles at
                                      # max_yield 4 produce multiples of that. The cap is the
                                      # direct test of whether the count is the problem.
                                      # ANSWER: no. A single-seed probe put cap 30 at $80,213
                                      # against $66,054 uncapped, and the 66-case panel reversed
                                      # it monotonically -- own score -$2,791 [-4,764, -912] at
                                      # 40, -$7,815 [-10,492, -5,237] at 30, -$10,152 at 20, all
                                      # surviving Holm. Capping the count is strictly worse.
                                      # Default 0. The fourth single-seed reversal recorded here.
                                      # artifacts/evaluations/crop-tile-cap-panel-20260831.json
    "MARGINAL_CROP_PRICING": 0,       # re-price the crop menu as empty tiles are assigned
                                      # within a turn, instead of offering all of them the
                                      # price the FIRST tile faced.
                                      #   0  off -- one ranking per turn for every tile
                                      #   1  full -- the marginal price sets both the choice
                                      #      and the accept threshold, so tiles stop being
                                      #      planted once the running supply prices them out
                                      #   2  choice only -- rank marginally, keep the
                                      #      unmarginal value for the >0 test and the job
                                      #      value, so the MIX changes without the tile COUNT
                                      # Mode 1 conflates the two: on episode-102123196 it moves
                                      # strawberry 60 -> 42 (the tape holds 33) but also drops
                                      # the board from 79 tiles to 50 (the tape holds 57).
                                      # Mode 2 moves strawberry only 60 -> 59, so the running
                                      # supply does NOT change which crop ranks first -- it
                                      # changes whether a tile is worth planting at all.
                                      # Both UNMEASURED on 66 held-out cases: mode 1 margin
                                      # +$328 [-2,012, +2,635] p=0.780, mode 2 -$666
                                      # [-2,484, +1,215] p=0.469; neither survives Holm across
                                      # the swept family. Default stays 0.
                                      # artifacts/evaluations/marginal-crop-pricing-panel-20260831.json
    "FERTILIZER_GROSS_MARGINAL": True,  # value one owned FERTILIZER as sale + max(0, use
                                      # advantage) rather than max(sale, advantage). R5's P0.1:
                                      # the two have different baselines, so the old max()
                                      # understated fertilizer whenever using it was attractive,
                                      # and `animal_daily_net` inherited that. ON for
                                      # CORRECTNESS, not for score: the first panel read margin
                                      # +$3,555 [+110, +7,236] p=0.042, and a replication on 66
                                      # fresh seeds read -$1,583 [-4,736, +1,244] p=0.292.
                                      # Pooled over 132 cases: own score -$14 [-2,451, +2,382],
                                      # margin +$986 [-1,396, +3,363] p=0.411 -- UNMEASURED. The
                                      # first reading was the winner's curse the fresh-seed gate
                                      # exists to catch. `max(sale, net_advantage)` is still a
                                      # category error, so the arithmetic stays fixed and the
                                      # score claim is withdrawn.
                                      # artifacts/evaluations/fertilizer-baseline-{panel,replication}-20260831.json
    "ANIMAL_EXACT_DP": False,         # price animals with the exact finite-state service DP
                                      # (`heuristic/animal_dp.py`) instead of the closed form.
                                      # R2/R3/R4/R5 each found the closed form's event calendar
                                      # wrong; the DP matches the engine step by step. It is
                                      # OFF by default anyway: on 66 held-out cases the paired
                                      # cluster bootstrap puts own score at -$2,576
                                      # [-7,073, +1,895] and margin at -$3,432 [-7,018, +158],
                                      # i.e. UNMEASURED in both directions, and a season costs
                                      # 27s against the closed form's 4.2s. A provably exact
                                      # calculator that buys no measured score does not earn a
                                      # 6x latency bill. Evidence:
                                      # artifacts/evaluations/exact-animal-dp-panel-20260831.json
    "OPPONENT_SUPPLY_WEIGHT": 0.5,    # how much of the opponent's VISIBLE standing pipeline
                                      # to price into our market forecast. Both players sell into
                                      # one market and `farms` is public, so their planted tiles
                                      # are observable supply our NPVs otherwise ignore. Swept
                                      # 2026-08-31 on the 66 held-out frontier seats (132 games,
                                      # never tuned on): 0.5 gains +$5,346 margin and +$4,437 of
                                      # our own score over 0.0 and takes the panel's first wins
                                      # (2 vs 0); 0.25 and 0.75 also gain but by less. Exactly
                                      # neutral head-to-head against production over 608 games
                                      # in three seed ranges (304W/304L, mean -$49).
    "TOWN_DRAIN_EXACT": False,        # estimate the drain rate as (exact town demand) plus a
                                      # smoothed PLAYER component instead of one EMA over the
                                      # sum. Measured NEGATIVE 2026-08-31 and kept as a named
                                      # control: -$534 and 0/132 vs 2/132 wins on the held-out
                                      # frontier seats, despite a single-tape probe showing
                                      # +$14.9k of margin. With a fixed shop roster the two
                                      # estimators agree exactly; the difference is that this one
                                      # jumps the moment a shop unlocks instead of ramping over
                                      # ~1/alpha days -- so it prices the coming scarcity
                                      # immediately and HOLDS stock for it, and holding is the
                                      # thing a shared market punishes.
    "EXACT_MARKET_LEDGER": False,     # commit every market order against ONE projected
                                      # cash/market/shed state, in the order the engine will
                                      # resolve it, instead of testing each line against the
                                      # pre-market observation plus an ad-hoc counter. The
                                      # 2026-09-01 reviews reproduced four defects in that
                                      # counter -- see `_MarketLedger` -- of which the sharpest
                                      # are that every crop line sized against the same cash
                                      # ($100 emits a $100 strawberry seed AND an $80 melon
                                      # seed) and that the sale proceeds this queue deliberately
                                      # front-loads were never spendable by the purchases they
                                      # were front-loaded for. The engine then silently clipped
                                      # whatever we could not afford, which made ENGINE ORDER,
                                      # not our valuation, the effective crop priority. This is
                                      # a correctness fix and a behaviour change, so it ships
                                      # off until both gates have scored it.
    "FEED_DEBITS_SEEDS": True,        # charge the feed top-up against the cash the seed order
                                      # is sized from. Market orders resolve in LIST order and
                                      # the feed BUY_PRODUCT is appended before the seed
                                      # BUY_SEED, so without this the seed plan is sized against
                                      # money the wheat purchase is about to spend. Measured
                                      # NEUTRAL (+$38, wins unchanged, on the 66 held-out
                                      # frontier seats): the top-up's `want` is small next to a
                                      # seed order, so the defect rarely binds. Adopted anyway as
                                      # an invariant -- committed_capital must account for
                                      # everything already appended -- so the class of bug cannot
                                      # come back when feed volumes change.
    "TOUR_STOPS": 0,                  # extra stops chained onto a unit's assignment as a
                                      # committed local TOUR (0 = one job per turn, re-decided,
                                      # the behaviour before 2026-08-31). Measured that day: the
                                      # frontier tapes pay 0.90-1.07 travel turns per task and we
                                      # pay 1.74, and per task type the difference is that they
                                      # never leave -- CARE costs them 0.02 travel and us 0.66,
                                      # COLLECT_FERTILIZER 0.11 vs 1.88. Greedy value/(1+d)
                                      # re-picks a global best every turn, so a $600 job two
                                      # steps away ties a $200 job underfoot and pulls the unit
                                      # off the tile it is standing on. Measured NEGATIVE and
                                      # kept as a named control: 2/3/5 stops score -$16,583 /
                                      # -$10,438 / -$10,823 on the 66 held-out frontier seats.
                                      # The mechanism works -- travel/task falls 1.74 -> 1.50 and
                                      # idle 18.5% -> 14.7% -- and the farm still loses, because
                                      # a committed tour spends turns on cheap nearby entries
                                      # while a valuable distant one goes unserved. Low travel is
                                      # a CONSEQUENCE of good task selection on a dense board,
                                      # not a cause of score.
    "TOUR_RADIUS": 3,                 # a stop further than this is a new trip, not a stop.
    "FEED_CARRIERS": 2,               # units that may carry wheat for FEED at once. A FEED job
                                      # needs a wheat-CARRYING unit and each carrier can only
                                      # feed one animal per turn while walking the ring, so this
                                      # is the hard ceiling on feeding throughput. 2 was tuned
                                      # in July for a six-goose flock and the mixed herd is ~23
                                      # head, so this looked stale. It is not: measured
                                      # 2026-08-31 on the held-out frontier seats, 3/4/6 carriers
                                      # score -$499 / -$3,730 / -$2,523. The carriers cost more
                                      # in diverted actions than the marginal animals they save.
    "HERD_FEED_GATE": False,          # refuse to buy another animal while the wheat we hold and
                                      # are growing cannot already feed the herd we have.
                                      # Probed 2026-08-31: the goose flock falls from 7 on day 18
                                      # to 2 by day 24 -- escape needs two consecutive unfed days
                                      # -- while 23 head compete for ~19 wheat/day, and the feed
                                      # top-up only buys while wheat is cheaper than a head's
                                      # daily output. Measured INERT: it does not bind where the
                                      # flock actually dies (identical trace and score on the
                                      # probe seed) and is worth +$806 on the held-out seats,
                                      # inside the noise band. We are not short of wheat when the
                                      # geese starve -- we hold 18-23 units every day from day 15.
    "FERTILIZER_SUPPLY_DAYS": 0,      # forecast horizon, in days, for our own herd's FERTILIZER
                                      # output. Separate from ANIMAL_SUPPLY_DAYS, which bundled
                                      # every animal product and measured negative. Fertilizer is
                                      # the line worth isolating: `econ.animal_daily_net` credits
                                      # $100/day per head -- more than a goose's eggs -- at the
                                      # spot price, while every animal makes 1/day regardless of
                                      # kind, so a 20-head herd sells 500 a season into a
                                      # linear/0.40 curve that reaches $20 at +400 and $1 at
                                      # +600. Measured average realised price: $85.10 at 6 head,
                                      # $70.10 at 12, $50.10 at 20.
                                      # Measured NEGATIVE and kept as a named control: 5/15/30
                                      # days score -$127 / -$1,338 / -$2,325 on the 66 held-out
                                      # frontier seats. Isolating the largest, most saturating
                                      # line does not rescue the idea that ANIMAL_SUPPLY_DAYS
                                      # tested: pricing our own future output into the forecast
                                      # makes the farm buy fewer animals, and fewer animals is
                                      # worse even though each is worth less than econ claims.
    "ANIMAL_SUPPLY_DAYS": 0,          # forecast horizon, in days, for livestock PRODUCTION
                                      # supply. Measured NEGATIVE 2026-08-31 and kept as a named
                                      # control: on the 66 held-out frontier seats, 5/10/20 days
                                      # score -$229 / -$275 / -$2,718 against 0 and give back
                                      # every win (0/132 vs 2/132). Unlike a crop harvest, a herd
                                      # produces steadily from its first yield day, so the
                                      # observed drain EMA already carries it -- adding standing
                                      # production double-counts and suppresses the pools that
                                      # actually pay (EGG is log/0.20, the flattest curve there
                                      # is). The public information is real; the forecast already
                                      # had it.
    "ANIMAL_UNPLACED_ALLOWANCE": 2,   # order the next animal while this many are still
                                      # bought-but-unplaced (0 = strict one-at-a-time drip).
                                      # Swept 2026-08-30 vs the frozen 84fd5b1 solver: 2 ->
                                      # 51/72 (+$2.4k, mirror $85.6k); 1 and 3 similar.
    "MAX_SHEEP": 6,                   # swept for the first time 2026-08-31 (it was the one
                                      # herd cap never tested): 9 costs margin $3,619
                                      # [-6,628, -705] with own score unmeasured, and 3 moves
                                      # nothing. 6 stands.
                                      # artifacts/evaluations/max-sheep-panel-20260831.json
                                      # sheep make the same free fertilizer as geese plus
                                      # milk/wool pools that saturate independently of eggs.
                                      # Swept 2026-08-30 vs the frozen baac447 solver over 60
                                      # games / 3 seed sets: 8+6 -> 36/60, +$6.0k mean; 12+8
                                      # equal (+$6.1k); cows alone LOSE (2/12, milk gluts).
    "ASSIGNMENT_SEARCH": False,        # experimental global router; temporal gate required
    "LAND_RESERVE": 300,               # cash to keep BEYOND the quadrant price before expanding.
                                       # Was an unnamed 1500, making the first quadrant cost
                                       # $2,500 -- which our peak cash first clears on day 10, so
                                       # we bought land on days 11 and 14 where the frontier tape
                                       # buys on 6 and 11, with NW full (0 empty tiles) from day
                                       # 9. The gain is REAL but the board-density story is not:
                                       # on the probe seed, reserve 100 and 0 do buy land on days
                                       # 8 and 7, and plant counts (19/29/49 at days 9/12/15) and
                                       # score are IDENTICAL. Once land is unlocked we are seed-
                                       # and cash-limited, not tile-limited, so this most likely
                                       # belongs with CASH_FLOOR and SEED_BUFFER: every form of
                                       # holding capital back loses.
                                       # Swept 2026-08-31: 750/300/0 gain +$1,002/+$1,217/+$1,009
                                       # of margin and about +$2,000 of our own score on the 66
                                       # held-out frontier seats, and 300 wins 227/98/187 head to
                                       # head over 512 games (54.8% of decisive, +$872). It does
                                       # give back the panel's 2 wins, which at 2/132 is noise.
    "CAPITAL_SEARCH": False,           # compare land NPV with its exact next purchase price
    "LOOKAHEAD_HORIZON": 0,            # 0=greedy; 5--10=exact terminal daily search window
    "LOOKAHEAD_CANDIDATES": 8,         # exhaustive per-unit job shortlist
    "LOOKAHEAD_SCHEDULES": 24,         # nondominated schedules retained per unit
    "LOOKAHEAD_NODES": 12_000,         # deterministic enumeration budget across workers
    "LOOKAHEAD_STICKINESS": 0.50,       # relative bonus for the still-legal prior itinerary
    "LOOKAHEAD_PRESERVE_GREEDY": True,  # search only repairs jobs that miss the reset deadline
    "LOOKAHEAD_REPLANT": False,         # do not let terminal repair alter crop/weed strategy
    "LOOKAHEAD_REPAIR_RADIUS": 0,       # conservative override must execute immediately
    "LOOKAHEAD_CRITICAL_ONLY": True,    # override only imminent crop/animal survival failures
    "DAY_END_FEASIBILITY": True,        # reject guaranteed-dead plants/incomplete haul routes
    "LOOKAHEAD_FINAL_DAY_ONLY": False,  # restrict exhaustive planner to exact season terminal
    "ROUTE_STICKINESS": 0.25,           # gated temporal-search winner; avoids goal thrashing
    "ROUTE_STICKINESS_PROGRESS": 0.0,   # scale stickiness by how far the unit has ALREADY
                                        # walked toward its target (0 = the flat preference
                                        # measured optimal at 0.25). Hard commitment loses three
                                        # ways (TOUR_STOPS, continue_or_revise, PLAN_COMMIT_TURNS)
                                        # because it freezes a stale value; a preference that
                                        # grows with sunk travel keeps the repricing while making
                                        # an almost-finished walk hard to abandon -- and 155
                                        # abandoned walks burn 467 turns a season for nothing.
                                        # Measured NEGATIVE and kept as a named control: 0.5/1/3
                                        # score +$239 / -$3,614 / -$5,345 on the 66 held-out
                                        # frontier seats, and 0.5 gives back both panel wins. The
                                        # flat preference is optimal in SHAPE as well as
                                        # magnitude: sunk travel is not evidence about value, and
                                        # weighting by it only makes the policy slower to react.
    "ROUTE_COMMIT_TURNS": 10,           # maximum life of one spatial commitment
}


class Config(dict):
    """Attribute-style access over DEFAULTS plus overrides."""

    def __init__(self, **overrides):
        super().__init__(DEFAULTS)
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown policy tunables: {sorted(unknown)}")
        self.update(overrides)
        if self["CASH_CROP"] not in (None, "AUTO", *CROPS):
            raise ValueError(f"unknown CASH_CROP {self['CASH_CROP']!r}")

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _last_useful_plant_day(crop, total_days):
    """Latest day a fresh planting still reaches a harvestable yield before the season ends."""
    return total_days - 1 - CROPS[crop]["first_yield_day"]


def _wants_water(v, tile, total_days):
    """'urgent' | 'bonus' | None -- whether watering this tile tonight changes an outcome."""
    if tile.get("watered_today"):
        return None
    final_day = v.day >= total_days - 1
    # Survival. consecutive_unwatered resets to 0 on a watered day and increments otherwise,
    # so ">= 1" naturally yields alternate-day watering -- and fires on the planting day
    # too, because a fresh plant starts at 1 (CLAUDE.md 3.4). Pointless on the last day:
    # there is no end-of-day left for the plant to die in.
    urgent = tile.get("consecutive_unwatered", 0) >= 1
    window = water_bonus_window(tile.get("crop"))
    in_window = False
    if window:
        age = v.day - tile.get("planted_day", 0)
        in_window = (window[0] <= age <= window[1]
                     and tile.get("yield_units", 0) < CROPS[tile["crop"]]["max_yield"])
    if final_day:
        # Only a bonus water on an already-harvestable tile still converts to money today.
        age = v.day - tile.get("planted_day", 0)
        if in_window and age >= CROPS[tile["crop"]]["first_yield_day"]:
            return "bonus"
        return None
    if urgent:
        return "urgent"
    if in_window:
        return "bonus"
    return None


def _ready_to_harvest(v, tile, total_days, cfg=None):
    cd = CROPS.get(tile.get("crop"))
    if cd is None:                       # animal tile -- handled in _build_jobs
        return tile.get("yield_units", 0) > 0
    if tile.get("yield_units", 0) <= 0:
        return False
    age = v.day - tile.get("planted_day", 0)
    if age < cd["first_yield_day"]:
        return False
    if cd["ongoing"]:
        # An ongoing crop accrues up to `max_yield` HELD and harvest leaves the plant standing,
        # so harvesting the instant it holds anything spends one visit per unit. Measured on
        # episode-102123196: 181 STRAWBERRY harvests at exactly 1.00 unit each against a cap of
        # 4, where MELON -- a one-time crop with a real harvest deadline -- runs at 94% of its
        # cap. Batching to the cap buys the same units for a quarter of the trips. The risk is
        # the other side of the same rule: past `max_yield` productions the plant decays, and
        # anything still held when it weeds is lost, so this is a threshold, not a hold.
        held = tile.get("yield_units", 0)
        at = max(1, min(int(getattr(cfg, "ONGOING_HARVEST_AT", 1)), cd["max_yield"]))
        if held >= at:
            return True
        # Always take it if the plant is near the end of its life, cap or no cap.
        productions = (age - cd["first_yield_day"]) // max(1, cd["interval"]) + 1
        return productions >= cd["max_yield"] or v.day >= total_days - 2
    # At the yield cap further watering does nothing: clear the tile NOW and restart the
    # cycle. For melon this is age 10 instead of 12 -- two days of extra throughput.
    if tile.get("yield_units", 0) >= cd["max_yield"]:
        return True
    # Otherwise wait for the last bonus day -- and wait until that day's watering has
    # landed, or the final unit is thrown away.
    if age > cd["max_yield_day"]:
        return True
    if age == cd["max_yield_day"]:
        return tile.get("watered_today", False)
    return v.day >= total_days - 2      # season ending: take what is there


def _feed_cover_days(cfg, day):
    """Feed cover to hold on ``day``.

    Buying extra opening wheat is strong against an opponent that does not, and the advantage
    shrinks once both players do it -- the shared market reprices immediately -- so the cover is
    a short opening schedule rather than a constant.  ``FEED_COVER_OPENING=None`` keeps the
    measured constant-cover behaviour exactly.
    """
    opening = getattr(cfg, "FEED_COVER_OPENING", None)
    if opening is not None and int(day) < int(getattr(cfg, "FEED_COVER_OPENING_DAYS", 3)):
        return int(opening)
    return int(cfg.FEED_COVER_DAYS)


def _feed_plant_cover_days(cfg):
    """Feed cover targeted by planting/seed purchases (defaults to the holding cover)."""
    value = getattr(cfg, "FEED_PLANT_COVER_DAYS", None)
    return int(cfg.FEED_COVER_DAYS if value is None else value)


def _service_discount(v, cfg):
    """How much of a new animal's NPV we should expect to actually collect.

    `animal_npv` prices an animal that is fed every day and survives the season. Whether that
    happens is a labour question, and the board answers it: an animal already carrying
    `consecutive_unfed >= 1` is one miss from escaping, and a herd where many are in that state
    is a herd we are not servicing. Buying into it adds a competitor for the same feed trips.

    Deliberately uses only what is visible when the purchase is decided, and returns 1.0 for an
    empty herd so the first animals are never discounted.
    """
    weight = float(getattr(cfg, "HERD_SERVICE_DISCOUNT", 0.0))
    if weight <= 0:
        return 1.0
    animals = [t for _x, _y, t in v.animals()]
    if not animals:
        return 1.0
    safe = sum(1 for t in animals if t.get("consecutive_unfed", 0) == 0)
    healthy = safe / len(animals)
    return 1.0 - weight * (1.0 - healthy)


def _herd_caps(cfg):
    """Configured herd ceiling per animal kind (kinds capped at zero are never bought)."""
    return {
        "GOOSE": int(cfg.MAX_GEESE * float(getattr(cfg, "HERD_SCALE", 1.0))),
        "COW": int(getattr(cfg, "MAX_COWS", 0) * float(getattr(cfg, "HERD_SCALE", 1.0))),
        "SHEEP": int(getattr(cfg, "MAX_SHEEP", 0) * float(getattr(cfg, "HERD_SCALE", 1.0))),
    }


def _animals_by_kind(v, herd_caps, *, projected_tiles=None):
    """Count placed animals at one boundary; caps name kinds, not inventory limits.

    Market planning supplies the unit-phase tile overlay because its shed and carried
    inventories are already post-unit. A successful PLACE transfers ownership; it must
    not create acquisition headroom by disappearing between two different phase counts.
    Job planning omits the overlay and retains the turn-start view.
    """
    counts = {k: 0 for k in herd_caps}
    for _, _, t in v.animals():
        kind = t.get("animal")
        if kind in counts:
            counts[kind] += 1
    for position, final in (projected_tiles or {}).items():
        for tile, delta in ((v.tile(*position), -1), (final, 1)):
            kind = tile.get("animal") if isinstance(tile, dict) else None
            if kind in counts:
                counts[kind] += delta
    return counts


def _coop_sites(v, n):
    """The n cheapest-to-service tiles for coops: closest to a reachable shed-access tile.

    (4,4) itself is deliberately eligible -- FEED acts on the tile you stand on, and
    PICKUP/DROP only need shed adjacency, so a coop on a shed-access tile is serviced
    without any walking.
    """
    anchors = [t for t in shed_access_tiles(v.board) if v.walkable(*t)]
    if not anchors or n <= 0:
        return []
    def near(p):
        return min(manhattan(p, a) for a in anchors)
    tiles = [(x, y) for x, y, t in v.iter_tiles()
             if t != "LOCKED" and (t is None or (isinstance(t, dict)
                                                 and t.get("kind") in ("COOP", "PASTURE", "WEED"))
                                   or (isinstance(t, dict) and "animal" in t))]
    tiles.sort(key=lambda p: (near(p), p[1], p[0]))
    return tiles[:n]


def _build_jobs(v, cfg, total_days, fc):
    """Every action worth taking this turn, as {value, pos, op[, need]}.

    `value` is the computed dollar worth of doing the job (see econ.py); `need` names an
    item the assigned unit must be carrying (FEED wheat, PLACE a goose, FERTILIZE). The
    assignment loop skips non-positive jobs and units that do not hold the needed item.
    """
    jobs = []
    final_day = v.day >= total_days - 1

    # One valuation object per turn decides HOW everything below is priced (fertilizer
    # baseline, exact-vs-legacy animals, action shadow price); the call sites only say WHAT.
    val = Valuation(fc, v.day, total_days, cfg)
    # The best USE advantage of one fertilizer right now. Recorded first because every
    # animal value below credits the fertilizer it produces.
    for _, _, t in v.plants():
        val.note_fertilizer_use(econ.fertilizer_use_advantage(
            t, v.day, fc, float(cfg.ONGOING_FERTILIZE)))
    fert_use = val._use_advantage

    for x, y, t in v.plants():
        kind = _wants_water(v, t, total_days)
        if kind == "urgent":
            jobs.append({"value": econ.standing_plant_value(t, v.day, total_days, fc),
                         "pos": (x, y), "op": ["WATER"], "critical": True})
        elif kind == "bonus":
            jobs.append({"value": econ.water_bonus_value(t, v.day, fc),
                         "pos": (x, y), "op": ["WATER"]})
        if _ready_to_harvest(v, t, total_days, cfg):
            units = t.get("yield_units", 0)
            harvest = {
                "value": units * fc.avg_sell_price(t["crop"], units),
                "pos": (x, y),
                "op": ["HARVEST"],
            }
            # A one-time crop disappears when HARVEST resolves. WATER/FERTILIZE on the same
            # tile may be independently attractive in the pre-turn observation, but only one
            # of those alternatives can have value this turn: if the lower-index unit harvests
            # first, every later tile action silently no-ops. Mark the position as an exclusive
            # current-turn choice while leaving ongoing crops and animal multi-service alone.
            if not CROPS[t["crop"]]["ongoing"]:
                harvest["invalidates_position"] = True
            jobs.append(harvest)
        fv = econ.fertilize_value(
            t, v.day, fc, float(cfg.ONGOING_FERTILIZE)) * float(cfg.FERTILIZE_DISCOUNT)
        if fv > 0 and t.get("fertilized_until_day", -1) < v.day:
            jobs.append({"value": fv, "pos": (x, y), "op": ["FERTILIZE"],
                         "need": "FERTILIZER"})

    # Labour's dual, priced off the crop surface. Crop job values (water bonus, harvest,
    # fertilize) contain no ACTION_COST term at all, so taking the marginal one as the price of
    # an action is exact rather than circular -- which is what lets this run in the same pass
    # that the animal values, which DO contain an action term, are computed in.
    val.note_labour([j["value"] for j in jobs], v.n_units)

    # --- livestock ----------------------------------------------------------
    carried = [v.unit_inv(i) for i in range(v.n_units)]
    wheat_carried = sum(inv.get("WHEAT", 0) for inv in carried)
    herd_caps = _herd_caps(cfg)
    carried_by_kind = {k: sum(inv.get(k, 0) for inv in carried) for k in herd_caps}
    unfed = 0
    best_feed_value = 0.0

    for x, y, t in v.animals():
        yields = t.get("yield_units", 0)
        if yields >= 2 or (yields > 0 and v.day >= total_days - 2):
            product = ANIMALS[t["animal"]]["product"]
            jobs.append({"value": yields * fc.avg_sell_price(product, yields),
                         "pos": (x, y), "op": ["HARVEST"]})
        if t.get("fertilizer_available"):
            # What one more owned FERTILIZER is worth: its sale plus whatever using it
            # instead would add. R5's P0.1 -- the old max() compared a gross sale against a
            # net advantage, two different baselines.
            jobs.append({"value": val.fertilizer_marginal(),
                         "pos": (x, y), "op": ["COLLECT_FERTILIZER"]})
        if final_day:
            continue                    # no end-of-day left: feeding and caring pay nothing
        # FEED and CARE are independent engine operations on the same tile: CARE checks only
        # `cared_today`, never `fed_today` (engine reference, the CARE branch), and the bank
        # increments at end-of-day when BOTH flags are set. Two units may therefore feed and
        # care the same animal in one turn, in either order. Emitting CARE only after
        # `fed_today` was already observed -- as this did until 2026-08-31 -- gates a
        # conjunction the engine allows on sequential observation, and costs a whole day of
        # bank on every animal fed late. Reported independently by three review returns.
        cared = t.get("cared_today")
        cv = econ.care_value(t, fc, assume_fed=True)
        if not t.get("fed_today"):
            unfed += 1
            urgent = t.get("consecutive_unfed", 0) >= 1
            value = val.feed_value(t, urgent=urgent)
            best_feed_value = max(best_feed_value, value)
            jobs.append({"value": value, "pos": (x, y), "op": ["FEED"], "need": "WHEAT"})
            if urgent:
                jobs[-1]["critical"] = True
            if cv > 0 and not cared and cfg.SAME_TURN_FEED_CARE:
                # A bundle member: worth the bank only if the FEED lands this turn too, so it
                # is dropped after assignment if its companion was not also assigned.
                jobs.append({"value": cv, "pos": (x, y), "op": ["CARE"],
                             "companion": ((x, y), "FEED")})
        elif cv > 0 and not cared:
            jobs.append({"value": cv, "pos": (x, y), "op": ["CARE"]})

    # Restock feed: two carriers, not one big pickup -- FEED jobs need a wheat-CARRYING
    # unit, and a single carrier can only feed one goose per turn while it walks the coop
    # ring. The pickup is worth the best feed it enables.
    shed_tile = nearest_shed_tile((v.board // 2 - 1, v.board // 2 - 1), v)
    if not final_day and unfed > wheat_carried and v.shed.get("WHEAT", 0) > 0 and shed_tile:
        n = min(unfed - wheat_carried, v.shed.get("WHEAT", 0))
        parts = max(1, min(int(getattr(cfg, "FEED_CARRIERS", 2)), n))
        share, extra = divmod(n, parts)
        for index in range(parts):
            take = share + (1 if index < extra else 0)
            if take <= 0:
                continue
            jobs.append({"value": best_feed_value, "pos": shed_tile,
                         "op": ["PICKUP", "WHEAT", take]})

    # Herd: each kind sizes itself by NPV. One more animal is worth buying/placing while
    # its season NPV beats the crop value of the prime tile its structure would occupy.
    # Structures are built ONLY for animals we already own (bought, shedded or carried):
    # pre-building for hypothetical animals once burned days of labour while crops died
    # unwatered. A carried animal can only be placed on its own structure kind.
    best_tile_crop, best_tile_value = econ.best_crop(v.day, total_days, fc)
    animals_by_kind = _animals_by_kind(v, herd_caps)
    owned_by_kind = {
        k: animals_by_kind[k] + v.shed.get(k, 0) + carried_by_kind[k] for k in herd_caps
    }
    total_owned = sum(owned_by_kind.values())
    total_cap = sum(herd_caps.values())
    for kind, cap in herd_caps.items():
        if cap <= 0:
            continue
        structure = ANIMALS[kind]["structure"]
        npv = val.animal_npv(kind)
        # Moving an animal we ALREADY OWN is not the same decision as buying one. `animal_npv`
        # subtracts the purchase price, which is correct for BUY and wrong for PICKUP/PLACE:
        # by then the money is spent and the only question is whether the remaining production
        # beats the tile. Valuing the move at the full NPV means that once the season is far
        # enough along for a fresh purchase to be unprofitable, `value > 0` filters the move
        # out and the animal stays in the shed for good -- and an animal in the shed can only
        # be picked up and placed (gotcha 11), so it is dead capital plus its whole lost
        # production stream. Observed on episode-102123196: three animals stranded from day 20
        # with four empty structures standing free.
        placement = npv + ANIMALS[kind]["cost"] if cfg.SUNK_ANIMAL_PLACEMENT else npv
        empties = [
            (x, y) for x, y, t in v.empty_structures() if t.get("kind") == structure
        ]
        in_shed = v.shed.get(kind, 0)
        if carried_by_kind[kind] > 0:
            for pos in empties[:carried_by_kind[kind]]:
                jobs.append({"value": placement, "pos": pos, "op": ["PLACE", kind],
                             "need": kind})
        elif in_shed > 0 and shed_tile:
            n = min(in_shed, max(1, len(empties)))
            jobs.append({"value": placement, "pos": shed_tile,
                         "op": ["PICKUP", kind, n]})
        if owned_by_kind[kind] > animals_by_kind[kind] + len(empties):
            sites = _coop_sites(v, min(total_cap, total_owned))
            for pos in sites:
                if v.tile(*pos) is None:
                    jobs.append({"value": npv, "pos": pos, "op": ["BUILD_" + structure]})
                    break

    # --- planting -----------------------------------------------------------
    # Each empty tile goes to the crop with the highest tile NPV today, with one hard
    # constraint and one hard rule on top of the values:
    #   * feed security: if the wheat pipeline (shed + carried + growing) does not cover
    #     FEED_COVER_DAYS of flock demand, wheat gets a premium worth the goose output it
    #     protects -- feeding is why wheat is grown at all;
    #   * per-crop jobs are capped by seeds held of THAT crop (atomic PLANT rule).
    # A plant starts with consecutive_unwatered=1. Planting on hour 23 is therefore a
    # guaranteed seed/action loss: end-of-day immediately increments it to 2 and turns it into
    # a weed before any later WATER can exist. This is an exact one-step feasibility rule.
    if (
        cfg.PLANTING_ENABLED
        and (not cfg.DAY_END_FEASIBILITY or v.hour < TURNS_PER_DAY - 1)
    ):
        n_animals = len(v.animals())
        wheat_pipeline = (v.shed.get("WHEAT", 0)
                          + sum(inv.get("WHEAT", 0) for inv in carried)
                          + 3 * sum(1 for _, _, t in v.plants() if t.get("crop") == "WHEAT"))
        cover_needed = n_animals * _feed_plant_cover_days(cfg)
        feed_premium = 0.0
        if not final_day and n_animals and wheat_pipeline < cover_needed:
            feed_premium = econ.herd_daily_net(
                [t for _, _, t in v.animals()], fc, fert_use
            )

        if cfg.CASH_CROP == "AUTO":
            candidate_crops = list(CROPS)
        elif cfg.CASH_CROP is None:
            candidate_crops = []
        else:
            candidate_crops = [cfg.CASH_CROP]
        # Feed wheat is an orthogonal allocation, not a mutually exclusive cash-crop choice.
        if feed_premium > 0 and "WHEAT" not in candidate_crops:
            candidate_crops.append("WHEAT")
        ranked = sorted(
            (
                (
                    econ.crop_tile_value(c, v.day, total_days, fc)
                    + (feed_premium if c == "WHEAT" else 0.0),
                    c,
                )
                for c in candidate_crops
            ),
            reverse=True,
        )
        seeds_left = {c: v.seeds.get(c, 0) for c in CROPS}
        n_coops = len(v.animals()) + len(list(v.empty_structures()))
        planned_coops = max(n_coops, min(total_cap, total_owned + 1))
        reserved = set(_coop_sites(v, planned_coops))
        centre = (v.board // 2 - 1, v.board // 2 - 1)
        empties = sorted((p for p in v.empty_tiles() if p not in reserved),
                         key=lambda p: manhattan(p, centre))
        # `ranked` is computed ONCE and then applied to every empty tile, so on a turn with
        # forty open squares all forty are offered the same crop at the same price -- the price
        # the FIRST one faces. The forecast does carry our own pipeline, but only across turns,
        # so within a turn the fortieth strawberry is priced as though the other thirty-nine
        # did not exist. Measured consequence on episode-102123196: we stand 60-61 strawberry
        # tiles from day 15 while the frontier tape holds 33 falling to 16 and runs 24-42 wheat.
        # With MARGINAL_CROP_PRICING the running supply is charged as tiles are assigned, which
        # is the same correction `neural.allocate.supply_of` makes for the allocator.
        mode = int(cfg.MARGINAL_CROP_PRICING)
        flat = {crop: value for value, crop in ranked}   # the once-per-turn value, by crop
        running = dict(fc.supply)
        cap = int(cfg.CROP_TILE_CAP)
        standing = {}
        if cap > 0:
            for _x, _y, tile in v.plants():
                crop = tile.get("crop")
                standing[crop] = standing.get(crop, 0) + 1
        for pos in empties:
            order = ranked
            if mode:
                marginal = econ.Forecast(fc.market, fc.rates, running)
                order = sorted(
                    ((econ.crop_tile_value(c, v.day, total_days, marginal)
                      + (feed_premium if c == "WHEAT" else 0.0), c)
                     for c in candidate_crops),
                    reverse=True,
                )
            for value, crop in order:
                # Mode 2 ranks on the marginal price but accepts and reports the unmarginal
                # one, so the CHOICE moves and the board size does not.
                accept = value if mode != 2 else flat.get(crop, value)
                if accept <= 0:
                    break
                if v.day > _last_useful_plant_day(crop, total_days):
                    continue
                if cap > 0 and standing.get(crop, 0) >= cap:
                    continue
                if seeds_left[crop] > 0:
                    jobs.append({"value": accept, "pos": pos, "op": ["PLANT", crop]})
                    seeds_left[crop] -= 1
                    if cap > 0:
                        standing[crop] = standing.get(crop, 0) + 1
                    if mode:
                        running[crop] = running.get(crop, 0.0) + _expected_units(crop)
                    break

    dig_value = 0.8 * best_tile_value
    for pos in v.weeds():
        jobs.append({"value": dig_value, "pos": pos, "op": ["DIG"]})

    return [j for j in jobs if j["value"] > 0]


def _expected_units(crop):
    """Units one tile of `crop` will put on the market, for the within-turn running supply.

    The engine's own cap, not a guess: a one-time crop is harvested once at up to `max_yield`,
    an ongoing one accumulates up to `max_yield` per harvest across its productions.
    """
    return float(CROPS[crop]["max_yield"])


def _haul_action(v, pos, inv, *, shed_capacity=SHED_CAPACITY, room=None):
    """One step of getting a carried load into the shed without destroying any of it.

    DROP dumps the WHOLE inventory and silently destroys whatever exceeds shed room
    (the engine deletes the remainder, it does not keep it carried) -- so when room is
    short, PLACE the largest stack clamped to room instead.
    """
    room = int(shed_capacity) - v.shed_used() if room is None else int(room)
    if room <= 0:
        return ["PASS"]                 # selling will free room; try again next turn
    if is_shed_adjacent(pos, v.board) and v.walkable(*pos):
        carrying = sum(inv.values())
        if room >= carrying:
            return ["DROP"]
        item = max(inv, key=inv.get)
        return ["PLACE", item, min(inv[item], room)]
    target = nearest_shed_tile(pos, v)
    op = step_toward(pos, target, v) if target else None
    return [op] if op else ["PASS"]


def _capacity_safe_actions(
    v, actions, positions, *, shed_capacity=SHED_CAPACITY
):
    """Rewrite unsafe unloads against one ordered projection of the unit phase."""
    del positions  # The projector reads canonical unit positions from the View.
    return _project_unit_market_ledger(
        v, actions, shed_capacity=shed_capacity, rewrite_unsafe_unloads=True
    )["actions"]


def _shed_place_admission(v, position, action):
    """Return ``(item, requested)`` when PLACE uses the engine's shed fallback.

    Animal PLACE is structure placement only on a matching empty structure. Elsewhere it falls
    through to ordinary shed admission with a default quantity of one, and must reserve shared
    same-turn shed capacity just like any other PLACE.
    """
    if (
        not action
        or action[0] != "PLACE"
        or len(action) < 2
        or position is None
        or not is_shed_adjacent(position, v.board)
        or not v.walkable(*position)
    ):
        return None
    item = action[1]
    tile = v.tile(*position)
    if (
        item in ANIMALS
        and isinstance(tile, dict)
        and tile.get("kind") == ANIMALS[item]["structure"]
        and "animal" not in tile
    ):
        return None
    requested = int(action[2]) if len(action) >= 3 else 1
    return item, max(0, requested)


def _project_unit_market_ledger(
    v,
    unit_actions,
    *,
    shed_capacity=SHED_CAPACITY,
    rewrite_unsafe_unloads=False,
):
    """Project market-relevant unit effects in exact unit-index order.

    The shared result is used by capacity guards, same-turn sales, animal/feed cover, and feed
    consumption. This prevents those decisions from each inventing a different post-unit state.
    """
    actions = [
        list(action) if isinstance(action, (list, tuple)) else action
        for action in (unit_actions or ())
    ]
    shed = dict(v.shed)
    seeds = dict(v.seeds)
    inventories = [dict(v.unit_inv(unit)) for unit in range(v.n_units)]
    initial_tiles, tiles = {}, {}
    admissions = {}
    destroyed = {}
    feed_wheat_spent = 0

    # The engine validates PLANT requests atomically before resolving unit presence, position,
    # tile legality, or hand count. A syntactic request in an ignored hand tail still contributes
    # to demand; if demand exceeds the turn-start seeds, every PLANT for that crop becomes PASS.
    plant_demand = {}
    for action in actions:
        if isinstance(action, list) and len(action) >= 2 and action[0] == "PLANT":
            crop = action[1]
            plant_demand[crop] = plant_demand.get(crop, 0) + 1
    blocked_plant_crops = {
        crop
        for crop, demand in plant_demand.items()
        if demand > seeds.get(crop, 0)
    }

    def tile_at(position):
        if position not in tiles:
            original = v.tile(*position)
            original = dict(original) if isinstance(original, dict) else original
            initial_tiles[position] = dict(original) if isinstance(original, dict) else original
            tiles[position] = original
        return tiles[position]

    def take(inventory, item, amount=1):
        if inventory.get(item, 0) < amount:
            return False
        inventory[item] -= amount
        if inventory[item] == 0:
            del inventory[item]
        return True

    def add(inventory, item, amount):
        inventory[item] = inventory.get(item, 0) + amount

    for unit, original_action in enumerate(actions):
        if unit >= v.n_units or not isinstance(original_action, list) or not original_action:
            continue
        position = v.unit_pos(unit)
        if position is None:
            continue
        position = tuple(position)
        if not v.walkable(*position):
            continue
        tile = tile_at(position)
        inventory = inventories[unit]
        action = original_action
        adjacent = is_shed_adjacent(position, v.board)

        if rewrite_unsafe_unloads and adjacent and action[0] == "DROP":
            room = max(0, int(shed_capacity) - sum(shed.values()))
            if sum(inventory.values()) > room:
                if room <= 0 or not inventory:
                    action = ["PASS"]
                else:
                    item = max(inventory, key=inventory.get)
                    amount = min(inventory[item], room)
                    action = ["PLACE", item, amount] if amount > 0 else ["PASS"]
                actions[unit] = action
        if rewrite_unsafe_unloads and adjacent and action[0] == "PLACE" and len(action) >= 2:
            item = action[1]
            animal_placement = (
                item in ANIMALS
                and isinstance(tile, dict)
                and tile.get("kind") == ANIMALS[item]["structure"]
                and "animal" not in tile
            )
            if not animal_placement:
                requested = int(action[2]) if len(action) >= 3 else 1
                room = max(0, int(shed_capacity) - sum(shed.values()))
                amount = min(max(0, requested), inventory.get(item, 0), room)
                action = ["PLACE", item, amount] if amount > 0 else ["PASS"]
                actions[unit] = action

        op = action[0]
        if op in MOVES or op == "PASS":
            continue
        if op == "PICKUP" and adjacent and len(action) >= 2:
            item = action[1]
            requested = int(action[2]) if len(action) >= 3 else 1
            amount = min(max(0, requested), shed.get(item, 0))
            if amount:
                shed[item] -= amount
                add(inventory, item, amount)
            continue
        if op == "DROP" and adjacent:
            for item, carried in list(inventory.items()):
                room = max(0, int(shed_capacity) - sum(shed.values()))
                admitted = min(max(0, carried), room)
                if admitted:
                    shed[item] = shed.get(item, 0) + admitted
                    admissions[item] = admissions.get(item, 0) + admitted
                # The engine destroys overflow; it does not leave it carried. Record what was
                # lost -- a silently destroyed harvest is invisible in the score and is one of
                # the diagnostics the review returns asked to be able to see.
                lost = max(0, carried) - admitted
                if lost:
                    destroyed[item] = destroyed.get(item, 0) + lost
                del inventory[item]
            continue
        if op == "BUILD_COOP" and tile is None:
            tiles[position] = {"kind": "COOP"}
            continue
        if op == "BUILD_PASTURE" and tile is None:
            tiles[position] = {"kind": "PASTURE"}
            continue
        if op == "DIG" and tile is not None and not (
            isinstance(tile, dict) and "animal" in tile
        ):
            tiles[position] = None
            continue
        if op == "PLACE" and len(action) >= 2:
            item = action[1]
            if (
                item in ANIMALS
                and isinstance(tile, dict)
                and tile.get("kind") == ANIMALS[item]["structure"]
                and "animal" not in tile
                and take(inventory, item)
            ):
                tiles[position] = {
                    "kind": ANIMALS[item]["structure"],
                    "animal": item,
                    "placed_day": v.day,
                    "yield_units": 0,
                    "consecutive_unfed": 0,
                    "fed_today": False,
                    "cared_today": False,
                    "fertilizer_available": False,
                    "pending_care_bonus": 0,
                }
                continue
            if not adjacent:
                continue
            requested = int(action[2]) if len(action) >= 3 else 1
            room = max(0, int(shed_capacity) - sum(shed.values()))
            amount = min(max(0, requested), inventory.get(item, 0), room)
            if amount:
                take(inventory, item, amount)
                shed[item] = shed.get(item, 0) + amount
                admissions[item] = admissions.get(item, 0) + amount
            continue
        if op == "FEED" and (
            isinstance(tile, dict)
            and "animal" in tile
            and not tile.get("fed_today", False)
            and take(inventory, "WHEAT")
        ):
            tile["fed_today"] = True
            feed_wheat_spent += 1
            continue
        if op == "FERTILIZE" and (
            isinstance(tile, dict)
            and tile.get("kind") == "PLANT"
            and take(inventory, "FERTILIZER")
        ):
            tile["fertilized_until_day"] = max(
                int(tile.get("fertilized_until_day", -1)), v.day + 2
            )
            continue
        if op == "WATER" and (
            isinstance(tile, dict)
            and tile.get("kind") == "PLANT"
            and not tile.get("watered_today", False)
        ):
            tile["watered_today"] = True
            crop = tile.get("crop")
            crop_data = CROPS.get(crop)
            if crop_data is not None and not crop_data["ongoing"]:
                age = v.day - int(tile.get("planted_day", 0))
                window_start = (crop_data["max_yield_day"] + 1) // 2
                if window_start <= age <= crop_data["max_yield_day"]:
                    bonus = 2 if tile.get("fertilized_until_day", -1) >= v.day else 1
                    tile["yield_units"] = min(
                        crop_data["max_yield"],
                        int(tile.get("yield_units", 0)) + bonus,
                    )
            continue
        if op == "CARE" and (
            isinstance(tile, dict)
            and "animal" in tile
            and not tile.get("cared_today", False)
        ):
            tile["cared_today"] = True
            continue
        if op == "PLANT" and len(action) >= 2:
            crop = action[1]
            if (
                crop not in blocked_plant_crops
                and crop in CROPS
                and tile is None
                and seeds.get(crop, 0) > 0
            ):
                seeds[crop] -= 1
                tiles[position] = {
                    "kind": "PLANT", "crop": crop, "planted_day": v.day,
                    "watered_today": False, "consecutive_unwatered": 1,
                    "yield_units": 0 if CROPS[crop]["ongoing"] else 1,
                    "max_lifespan_step": (
                        -1
                        if CROPS[crop]["ongoing"]
                        else (v.day + CROPS[crop]["max_yield_day"] + 1)
                        * TURNS_PER_DAY
                    ),
                    "fertilized_until_day": -1,
                }
            continue
        if op == "HARVEST" and isinstance(tile, dict) and tile.get("yield_units", 0) > 0:
            if tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                if crop in CROPS and v.day - tile.get("planted_day", 0) >= CROPS[crop]["first_yield_day"]:
                    add(inventory, crop, int(tile["yield_units"]))
                    tile["yield_units"] = 0
                    if not CROPS[crop]["ongoing"]:
                        tiles[position] = None
            elif "animal" in tile:
                add(inventory, ANIMALS[tile["animal"]]["product"], int(tile["yield_units"]))
                tile["yield_units"] = 0
            continue
        if op == "COLLECT_FERTILIZER" and (
            isinstance(tile, dict) and "animal" in tile
            and tile.get("fertilizer_available", False)
        ):
            tile["fertilizer_available"] = False
            add(inventory, "FERTILIZER", 1)

    animal_count = len(v.animals())
    wheat_plants = sum(
        1 for _x, _y, tile in v.plants() if tile.get("crop") == "WHEAT"
    )
    for position, final in tiles.items():
        initial = initial_tiles[position]
        animal_count += int(isinstance(final, dict) and "animal" in final)
        animal_count -= int(isinstance(initial, dict) and "animal" in initial)
        wheat_plants += int(
            isinstance(final, dict) and final.get("kind") == "PLANT"
            and final.get("crop") == "WHEAT"
        )
        wheat_plants -= int(
            isinstance(initial, dict) and initial.get("kind") == "PLANT"
            and initial.get("crop") == "WHEAT"
        )
    return {
        "actions": actions,
        "shed": shed,
        "seeds": seeds,
        "inventories": inventories,
        "tiles": tiles,
        "shed_admissions": admissions,
        "shed_destroyed": destroyed,
        "feed_wheat_spent": feed_wheat_spent,
        "animal_count": animal_count,
        "wheat_plants": wheat_plants,
    }


def _turns_before_reset(v, total_days):
    """Number of actions including this one before positions/hands reset (or the game ends)."""

    if v.day >= total_days - 1:
        # The final actionable observation is hour 22; hour 23 is never played.
        return max(0, TURNS_PER_DAY - 1 - v.hour)
    return max(0, TURNS_PER_DAY - v.hour)


def _horizon_jobs(v, jobs, total_days, horizon):
    """Add exact short-horizon legality/effect metadata to current jobs."""

    turns_left = _turns_before_reset(v, total_days)
    effective = min(int(horizon), turns_left)
    prepared = []
    for original in jobs:
        job = dict(original)
        op = job["op"][0]
        target = tuple(job["pos"])
        deadline = effective
        # A newly planted tile needs one later action before end-of-day to survive. The WATER is
        # generated from the next real observation, so reserve that turn here.
        if op == "PLANT":
            deadline = min(deadline, turns_left - 1)
        tile = v.tile(*target)
        if op == "HARVEST" and isinstance(tile, dict):
            lifespan = int(tile.get("max_lifespan_step", -1))
            if lifespan >= 0:
                # Decay is applied after the action at lifespan_step. Completion number 1 means
                # acting now, hence the +1 conversion from absolute step to relative deadline.
                deadline = min(deadline, max(1, lifespan - v.step + 1))
            if tile.get("kind") == "PLANT":
                job["gain"] = {tile["crop"]: int(tile.get("yield_units", 0))}
                if not CROPS[tile["crop"]]["ongoing"]:
                    job["invalidates_position"] = True
            elif "animal" in tile:
                product = ANIMALS[tile["animal"]]["product"]
                job["gain"] = {product: int(tile.get("yield_units", 0))}
        elif op == "PICKUP" and len(job["op"]) >= 3:
            job["gain"] = {job["op"][1]: int(job["op"][2])}
        elif op == "COLLECT_FERTILIZER":
            job["gain"] = {"FERTILIZER": 1}
        elif op in ("FEED", "FERTILIZE"):
            job["consume"] = {job["need"]: 1}
        elif op == "PLACE" and job.get("need"):
            job["consume"] = {job["need"]: 1}
            job["invalidates_position"] = True
        elif op in ("DIG", "BUILD_COOP", "BUILD_PASTURE"):
            job["invalidates_position"] = True
        if v.day >= total_days - 1 and job.get("gain"):
            # Final reward is money only. A harvested/collected unit is worthless unless there is
            # still time after this job to reach a shed-access tile and DROP it. Reserve those
            # exact shortest-path actions inside the job's completion deadline.
            target_routes = shortest_routes(v, target)
            shed_distances = [
                target_routes[pos][0]
                for pos in shed_access_tiles(v.board)
                if pos in target_routes
            ]
            if shed_distances:
                deadline = min(deadline, turns_left - min(shed_distances) - 1)
        job["deadline"] = deadline
        if deadline >= 1:
            prepared.append(job)
    return prepared, effective


def _job_key(job):
    """Stable identity for reconciling a speculative itinerary with the next observation."""

    op = tuple(job["op"][:2])
    return (tuple(job["pos"]), op)


def _lookahead_assignments(
    v,
    cfg,
    total_days,
    jobs,
    available,
    positions,
    routes,
    planner_state=None,
    fixed_assignments=(),
):
    """Solve legal current-job sequences and return each unit's first committed job."""

    prepared, horizon = _horizon_jobs(v, jobs, total_days, cfg.LOOKAHEAD_HORIZON)
    if horizon <= 0 or not prepared or not available:
        return []

    route_cache = {}

    def route_map(source):
        source = tuple(source)
        cached = route_cache.get(source)
        if cached is None:
            cached = shortest_routes(v, source)
            route_cache[source] = cached
        return cached

    def distance(source, target):
        route = route_map(source).get(tuple(target))
        return None if route is None else route[0]

    schedule_sets = []
    enumeration_nodes = 0
    enumeration_truncated = False
    for u in available:
        schedules, stats = enumerate_job_schedules(
            positions[u],
            v.unit_inv(u),
            prepared,
            distance,
            horizon,
            action_cost=cfg.ACTION_COST,
            candidate_limit=cfg.LOOKAHEAD_CANDIDATES,
            schedule_limit=cfg.LOOKAHEAD_SCHEDULES,
            node_limit=max(1_000, cfg.LOOKAHEAD_NODES // max(1, len(available))),
        )
        schedule_sets.append(schedules)
        enumeration_nodes += stats["nodes"]
        enumeration_truncated = enumeration_truncated or stats["truncated"]
    # The rollout after the first action is a value estimate, not a reservation: every real next
    # observation may create WATER/CARE jobs or invalidate a speculative continuation. Collapse
    # schedules by their first action and solve that executable joint choice exactly. Reserving
    # all speculative future jobs here caused workers to stand idle for plans that the next
    # observation correctly replaced.
    current_by_key = {_job_key(job): index for index, job in enumerate(prepared)}
    available_slots = {u: slot for slot, u in enumerate(available)}
    fixed = {}
    for u, original_index in fixed_assignments:
        slot = available_slots.get(u)
        prepared_index = current_by_key.get(_job_key(jobs[original_index]))
        if slot is None or prepared_index is None:
            continue
        route = routes[u].get(tuple(prepared[prepared_index]["pos"]))
        if route is None or route[0] + 1 > prepared[prepared_index]["deadline"]:
            continue
        fixed[slot] = prepared_index
    fixed_jobs = set(fixed.values())
    desired = {}
    if planner_state is not None:
        prior_plans = planner_state.setdefault("plans", {})
        for u in available:
            plan = list(prior_plans.get(u, ()))
            # Completed or invalidated prefixes disappear from the freshly-built exact job set.
            while plan and plan[0] not in current_by_key:
                plan.pop(0)
            if plan:
                desired[u] = current_by_key[plan[0]]
            prior_plans[u] = tuple(plan)

    edges = {}
    edge_schedules = {}
    for slot, schedules in enumerate(schedule_sets):
        if slot in fixed:
            continue
        for schedule in schedules:
            if not schedule["jobs"]:
                continue
            if cfg.LOOKAHEAD_CRITICAL_ONLY and not prepared[schedule["jobs"][0]].get(
                "critical", False
            ):
                continue
            if not cfg.LOOKAHEAD_REPLANT and any(
                prepared[index]["op"][0] in ("PLANT", "DIG")
                for index in schedule["jobs"]
            ):
                continue
            first = schedule["jobs"][0]
            if first in fixed_jobs:
                continue
            first_pos = tuple(prepared[first]["pos"])
            travel = distance(positions[available[slot]], first_pos)
            if (
                cfg.LOOKAHEAD_PRESERVE_GREEDY
                and travel > cfg.LOOKAHEAD_REPAIR_RADIUS
            ):
                continue
            utility = schedule["score"]
            if desired.get(available[slot]) == first:
                utility *= 1.0 + cfg.LOOKAHEAD_STICKINESS
            edge = (utility, travel)
            prior = edges.get((slot, first))
            if prior is None or edge[0] > prior[0]:
                edges[slot, first] = edge
                edge_schedules[slot, first] = schedule
    selected = maximum_weight_assignment(len(available), len(prepared), edges)
    selected.update(fixed)
    # The solver is an override, not an idling policy. If it cannot prove an immediately better
    # repair for a greedy route, retain that exact greedy target: intermediate movement can put a
    # worker near a different job that appears on the next observation even when today's original
    # target itself was beyond the reset deadline.
    if cfg.LOOKAHEAD_PRESERVE_GREEDY:
        for u, original_index in fixed_assignments:
            slot = available_slots.get(u)
            prepared_index = current_by_key.get(_job_key(jobs[original_index]))
            if slot is not None and prepared_index is not None and slot not in selected:
                selected[slot] = prepared_index
    assignments = [(available[slot], job) for slot, job in sorted(selected.items())]
    if planner_state is not None:
        next_plans = {}
        for slot, first in selected.items():
            u = available[slot]
            schedule = edge_schedules.get(
                (slot, first),
                {"jobs": (first,), "mask": 1 << first, "score": 0.0, "time": 1},
            )
            sequence = list(schedule["jobs"])
            # A job underfoot executes now. Keep only genuinely future actions; a movement target
            # remains the head until the worker reaches and executes it.
            if tuple(positions[u]) == tuple(prepared[first]["pos"]):
                sequence = sequence[1:]
            next_plans[u] = tuple(_job_key(prepared[index]) for index in sequence)
        planner_state["plans"] = next_plans
        planner_state["last_stats"] = {
            "step": v.step,
            "horizon": horizon,
            "jobs": len(prepared),
            "units": len(available),
            "enumeration_nodes": enumeration_nodes,
            "enumeration_truncated": enumeration_truncated,
            "assignment_edges": len(edges),
            "selected": len(assignments),
            "preserved_greedy": len(fixed),
        }
    return assignments, prepared


def _build_tour(cfg, jobs, start, taken, routes_from):
    """Greedy nearest-value chain from `start`: the local cluster, in visiting order.

    Ordering by locality is the thing greedy per-turn density structurally cannot do, because it
    re-derives a global best every turn instead of finishing the cluster it is standing in.  Each
    next stop maximises value per MARGINAL step from the previous stop, not from the unit's
    original position, so a run of same-tile work (FEED then CARE then COLLECT_FERTILIZER, all
    acting on the tile underfoot) chains at zero travel and wins outright.
    """
    # Later stops are a PREFERENCE, never a reservation.  Claiming them exclusively starves
    # every other unit -- twelve units times three stops removes more jobs from the pool than the
    # pool usually holds, measured 2026-08-31 as idle 18.5% -> 39.9% and 396 fewer tasks done.
    # Another unit taking a stop is fine: the tour simply advances past it next turn.
    stops, position = [], tuple(start)
    mine = set(taken)
    radius = int(getattr(cfg, "TOUR_RADIUS", 3))
    for _ in range(int(getattr(cfg, "TOUR_STOPS", 0))):
        best, best_density = None, 0.0
        for index, job in enumerate(jobs):
            if index in mine or job["value"] <= 0 or job.get("need"):
                continue
            route = routes_from(position).get(tuple(job["pos"]))
            if route is None or route[0] > radius:
                continue
            density = job["value"] / (1 + route[0])
            if density > best_density:
                best, best_density = index, density
        if best is None:
            break
        stops.append(best)
        mine.add(best)
        position = tuple(jobs[best]["pos"])
    return stops


def _plan_units(
    v,
    cfg,
    total_days,
    fc,
    planner_state=None,
    *,
    shed_capacity=SHED_CAPACITY,
    assign=None,
    stickiness=None,
):
    """Receding-horizon unit plan with measured persistence and optional exhaustive search.

    Production uses greedy live value density plus an expiring soft preference for the prior legal
    target. ``LOOKAHEAD_HORIZON`` enables the separately gated time-expanded enumerator near an
    exact reset boundary; ``ASSIGNMENT_SEARCH`` retains the rejected current-turn global-router
    ablation. Every mode replans from the next real observation.
    """
    n = v.n_units
    actions = [None] * n
    positions = [v.unit_pos(i) for i in range(n)]
    free = []

    for i in range(n):
        pos = positions[i]
        if pos is None:
            actions[i] = ["PASS"]
            continue
        free.append(i)

    # Final-day deadline: the last actionable step is day `total_days-1`, hour 22, and
    # there is no end-of-day auto-drop after it -- whatever is still carried scores ZERO.
    # Force each carrying unit back in time: distance + 1 (drop) + 1 margin before hour 22.
    forced = set()
    if v.day >= total_days - 1:
        for u in free:
            inv = v.unit_inv(u)
            if sum(inv.values()) > 0 and v.hour >= 20 - manhattan(
                    positions[u], nearest_shed_tile(positions[u], v) or positions[u]):
                actions[u] = _haul_action(
                    v, positions[u], inv, shed_capacity=shed_capacity
                )
                forced.add(u)
    free = [u for u in free if u not in forced]

    prior_tours = dict((planner_state or {}).get("tours") or {})

    jobs = _build_jobs(v, cfg, total_days, fc)
    available = []
    routes = {}
    turns_left = _turns_before_reset(v, total_days)
    for u in free:
        inv = v.unit_inv(u)
        # A unit at the drop threshold hauls instead of taking another job (see
        # _haul_action).
        routes[u] = shortest_routes(v, positions[u])
        if sum(inv.values()) >= cfg.DROP_INVENTORY_AT:
            shed_tile = nearest_shed_tile(positions[u], v)
            route = routes[u].get(shed_tile) if shed_tile else None
            # End-of-day auto-drops from anywhere. Do not spend the last turns walking toward a
            # DROP that cannot execute before that reset; a local farm job can still pay now.
            if (
                route is not None
                and (not cfg.DAY_END_FEASIBILITY or route[0] + 1 <= turns_left)
            ):
                actions[u] = _haul_action(
                    v, positions[u], inv, shed_capacity=shed_capacity
                )
                continue
        available.append(u)

    pairs = []
    desired_targets = {}
    origin_distance = {}
    prior_targets = {}
    # Per-unit errand affinity, carried across turns in the same planner state the positional
    # stickiness uses. Reset at the day boundary because hands are dismissed each night and the
    # unit index no longer refers to the same worker.
    affinity = float(getattr(cfg, "UNIT_AFFINITY", 0.0))
    last_class = {}
    if affinity and planner_state is not None:
        record = planner_state.setdefault("unit_class", {"day": v.day, "by_unit": {}})
        if record.get("day") != v.day:
            record.update(day=v.day, by_unit={})
        last_class = record["by_unit"]
    if planner_state is not None and cfg.ROUTE_STICKINESS > 0:
        current_by_key = {_job_key(job): index for index, job in enumerate(jobs)}
        prior_targets = planner_state.setdefault("greedy_targets", {})
        for u, record in prior_targets.items():
            if record.get("expires", -1) >= v.step and record.get("key") in current_by_key:
                desired_targets[u] = current_by_key[record["key"]]
                origin_distance[u] = record.get("d0", 0)
    for slot, u in enumerate(available):
        inv = v.unit_inv(u)
        for k, job in enumerate(jobs):
            need = job.get("need")
            if need and inv.get(need, 0) <= 0:
                continue
            route = routes[u].get(tuple(job["pos"]))
            if route is None:
                continue
            d, _first = route
            density = job["value"] / (1 + d)
            if affinity and last_class.get(u) == _op_class(job["op"][0]):
                # Same KIND of work, not the same tile. A unit that fed an animal last turn is
                # already carrying wheat and already standing among the coops, so the next
                # animal job is cheaper for it than the distance term alone can express.
                density *= 1.0 + affinity
            if desired_targets.get(u) == k:
                # Sunk travel is not a reason to finish a job on its own -- but an almost-arrived
                # walk is one step from paying and a fresh one is not, and abandoning it wastes
                # every step already taken.
                start = origin_distance.get(u, 0)
                progress = max(0.0, 1.0 - d / start) if start > 0 else 0.0
                # The learnable quantity.  Four hand-designed HARD commitment schemes all lose
                # because they freeze a stale value; the soft preference keeps per-turn repricing,
                # so its STRENGTH is what a policy should own. `stickiness=None` is the measured
                # constant exactly.
                sticky = (
                    cfg.ROUTE_STICKINESS if stickiness is None
                    else float(stickiness(v, cfg, u, job, d, start, progress))
                )
                density *= 1.0 + sticky * (
                    1.0 + float(cfg.ROUTE_STICKINESS_PROGRESS) * progress
                )
            pairs.append((density, d, slot, u, k))

    # Jobs on a position that one candidate destroys are alternatives, not independent work.
    # This matters for capped one-time crops: the same observation can correctly advertise both
    # survival WATER and HARVEST, but assigning them to two co-located hands makes the later
    # action a deterministic no-op. Pick the highest-density alternative and let the freed unit
    # take the next useful job.
    exclusive_positions = {
        tuple(job["pos"])
        for job in jobs
        if job.get("invalidates_position", False)
    }
    greedy_assignments = []
    used_u, used_j = set(), set()
    used_exclusive_underfoot_positions = set()
    for _density, _d, _slot, u, k in sorted(
        pairs, key=lambda row: (-row[0], row[1], row[3], row[4])
    ):
        if u in used_u or k in used_j:
            continue
        position = tuple(jobs[k]["pos"])
        if (
            _d == 0
            and position in exclusive_positions
            and position in used_exclusive_underfoot_positions
        ):
            continue
        used_u.add(u)
        used_j.add(k)
        if _d == 0 and position in exclusive_positions:
            used_exclusive_underfoot_positions.add(position)
        greedy_assignments.append((u, k))

    # Bundles: a job whose value only exists alongside a companion this turn (same-turn
    # FEED + CARE) is dropped when the companion did not get a unit. Dropping frees the unit
    # for the next pass rather than spending an action on a flag that banks nothing.
    greedy_assignments = _drop_unpaired_bundles(greedy_assignments, jobs)
    if affinity and planner_state is not None:
        for u, k in greedy_assignments:
            last_class[u] = _op_class(jobs[k]["op"][0])

    # Search only when its horizon reaches the exact terminal boundary. Earlier in the day a
    # truncated rollout has no faithful terminal value for unfinished routes and empirically
    # becomes short-sighted; the proven receding greedy rule remains the stronger open-horizon
    # policy. Once the whole remaining day fits, exhaustive schedules can safely solve through
    # the position/hand reset and rule out work that cannot possibly complete.
    search_day = not cfg.LOOKAHEAD_FINAL_DAY_ONLY or v.day >= total_days - 1
    if (
        cfg.LOOKAHEAD_HORIZON
        and search_day
        and turns_left <= cfg.LOOKAHEAD_HORIZON
    ):
        searched = _lookahead_assignments(
            v,
            cfg,
            total_days,
            jobs,
            available,
            positions,
            routes,
            planner_state,
            fixed_assignments=(greedy_assignments if cfg.LOOKAHEAD_PRESERVE_GREEDY else ()),
        )
        if searched:
            assignments, jobs = searched
        else:
            assignments = []
        for u, k in assignments:
            job = jobs[k]
            if tuple(positions[u]) == tuple(job["pos"]):
                actions[u] = job["op"]
            else:
                op = routes[u][tuple(job["pos"])][1]
                actions[u] = [op] if op else ["PASS"]

        # Match the control for genuinely idle units; the search changes assignments only.
        for u in free:
            if actions[u] is not None:
                continue
            inv = v.unit_inv(u)
            actions[u] = (
                _haul_action(v, positions[u], inv, shed_capacity=shed_capacity)
                if sum(inv.values()) else ["PASS"]
            )
        return _capacity_safe_actions(
            v, actions, positions, shed_capacity=shed_capacity
        )

    assignments = greedy_assignments
    if cfg.ASSIGNMENT_SEARCH and greedy_assignments:
        # Job selection is held fixed. The search only answers which capable unit should perform
        # each selected job. A constant reward for filling a job plus negative distance makes the
        # maximum-weight matching exactly a minimum-total-travel assignment; the constant exceeds
        # every possible board path, so no selected job can be dropped to save distance.
        selected_jobs = sorted(k for _u, k in greedy_assignments)
        route_cap = v.board * v.board + 1
        edges = {}
        for slot, u in enumerate(available):
            inv = v.unit_inv(u)
            for selected_index, k in enumerate(selected_jobs):
                job = jobs[k]
                need = job.get("need")
                if need and inv.get(need, 0) <= 0:
                    continue
                route = routes[u].get(tuple(job["pos"]))
                if route is None:
                    continue
                distance, _first = route
                edges[slot, selected_index] = (route_cap - distance, distance)
        chosen = maximum_weight_assignment(
            len(available), len(selected_jobs), edges
        )
        if len(chosen) == len(selected_jobs):
            assignments = [
                (available[slot], selected_jobs[selected_index])
                for slot, selected_index in sorted(chosen.items())
            ]

    if planner_state is not None and cfg.ROUTE_STICKINESS > 0:
        next_targets = {}
        for u, k in assignments:
            job = jobs[k]
            if tuple(positions[u]) == tuple(job["pos"]):
                continue
            key = _job_key(job)
            prior = prior_targets.get(u, {})
            expires = (
                prior.get("expires")
                if prior.get("key") == key
                else v.step + cfg.ROUTE_COMMIT_TURNS
            )
            route = routes[u].get(tuple(job["pos"]))
            d0 = (
                prior.get("d0", 0) if prior.get("key") == key
                else (route[0] if route else 0)
            )
            next_targets[u] = {"key": key, "expires": expires, "d0": d0}
        planner_state["greedy_targets"] = next_targets

    if int(getattr(cfg, "TOUR_STOPS", 0)) > 0:
        by_key = {_job_key(job): index for index, job in enumerate(jobs)}
        route_cache = {}

        def routes_from(position):
            position = tuple(position)
            cached = route_cache.get(position)
            if cached is None:
                cached = shortest_routes(v, position)
                route_cache[position] = cached
            return cached

        # A unit already on a tour keeps it: the stop it is walking to is the commitment.  A
        # tour entry that vanished was completed by this unit or taken by another, so drop it and
        # advance rather than abandoning the whole cluster.
        held, claimed, tours = {}, set(), {}
        for u in available:
            tour = [k for k in (by_key.get(key) for key in prior_tours.get(u, ())) if k is not None]
            tour = [k for k in tour if k not in claimed]
            if not tour:
                continue
            held[u] = tour[0]
            claimed.add(tour[0])
            tours[u] = tour
        # Critical work preempts a tour outright; it is the one thing worth breaking a cluster for.
        critical = {k for k, job in enumerate(jobs) if job.get("critical")}
        if critical - claimed:
            held, claimed, tours = {}, set(), {}
        merged = dict(held)
        for u, k in assignments:
            if u in merged or k in claimed:
                continue
            merged[u] = k
            claimed.add(k)
            tours[u] = [k] + _build_tour(
                cfg, jobs, jobs[k]["pos"], claimed, routes_from
            )
        assignments = sorted(merged.items())
        if planner_state is not None:
            planner_state["tours"] = {
                u: tuple(_job_key(jobs[k]) for k in tour) for u, tour in tours.items()
            }

    # The seam an external plan acts through.  `assign` receives the entries and what the
    # executor decided, and returns the assignment to actually run -- so a queue policy expresses
    # CONTINUE by returning the default and REVISE by returning something else, and a policy that
    # returns the default is byte-identical to production by construction.
    if assign is not None:
        assignments = assign({
            "view": v, "cfg": cfg, "total_days": total_days, "jobs": jobs,
            "available": available, "positions": positions, "routes": routes,
            "planner_state": planner_state,
        }, assignments)

    for u, k in assignments:
        job = jobs[k]
        if tuple(positions[u]) == tuple(job["pos"]):
            actions[u] = job["op"]
        else:
            op = routes[u][tuple(job["pos"])][1]
            actions[u] = [op] if op else ["PASS"]

    # Units with nothing to do: haul produce to the shed, otherwise sit still. Idling
    # beats wandering -- a unit that drifts is a unit that starts tomorrow far from work.
    for u in free:
        if actions[u] is None:
            inv = v.unit_inv(u)
            if (
                sum(inv.values())
                and is_shed_adjacent(positions[u], v.board)
                and v.walkable(*positions[u])
            ):
                # The common ordered ledger below decides whether this is a safe DROP or must
                # become a bounded PLACE/PASS after earlier PICKUP/PLACE/DROP effects.
                actions[u] = ["DROP"]
            else:
                actions[u] = (
                    _haul_action(
                        v, positions[u], inv, shed_capacity=shed_capacity
                    )
                    if sum(inv.values()) else ["PASS"]
                )

    return _capacity_safe_actions(
        v, actions, positions, shed_capacity=shed_capacity
    )


def _incoming_drops(v, actions, *, shed_capacity=SHED_CAPACITY):
    """Items that this turn's unit actions will land in the shed BEFORE market orders run.

    Units act before the market queue, so a DROP/PLACE this turn is sellable this turn --
    include it in the sell plan or (on the last step) it is never sold at all.
    """
    projection = _project_unit_market_ledger(
        v, actions, shed_capacity=shed_capacity
    )
    items = set(v.shed) | set(projection["shed"])
    return {
        item: projection["shed"].get(item, 0) - v.shed.get(item, 0)
        for item in items
        if projection["shed"].get(item, 0) != v.shed.get(item, 0)
    }


def _successful_feed_wheat_spent(v, unit_actions):
    """Count only FEED commands that the ordered unit phase will actually commit."""
    return _project_unit_market_ledger(v, unit_actions)["feed_wheat_spent"]


class _MarketLedger:
    """One projected cash/market/shed state, committed order by order in QUEUE ORDER.

    The 2026-09-01 reviews reproduced four cash-accounting defects in `_market_orders`, all of
    them the same shape: feasibility was tested against `v.money` (the PRE-MARKET observation)
    while an ad-hoc `committed_capital` counter tracked only some of what had already been
    appended. Concretely, on the deployed policy:

      * every crop line in the seed loop sized itself against the same cash, so $100 emitted a
        $100 STRAWBERRY seed AND an $80 MELON seed;
      * the queue sells first specifically so proceeds fund later purchases, and the purchases
        never added the proceeds, so $0 plus a $50 egg sale emitted no $10 wheat seed;
      * HIRE costs went into a local `spent` that nothing else read;
      * BUY_ANIMAL was appended without debiting anything.

    The engine then clipped whatever it could not afford, which means the ENGINE was choosing
    our crop priority by accident -- and silently, because an unaffordable order is a no-op, not
    an error. That is the failure mode this repository has recorded more than any other.

    This replaces the counter with the engine's own transition, applied in the order the engine
    will apply it. It is exact for OUR OWN queue and does not model the opponent, who commits at
    the same index from the same pre-commit inventory; `contract` says so rather than implying
    more.

    IT IS NOT UNIFORMLY CONSERVATIVE, and an earlier version of this docstring said it was. The
    direction depends on which way the opponent trades: their SELL of an item we are buying
    lowers our price (we are pessimistic, which is safe), but their BUY of the same item RAISES
    it, and then we have planned to spend money we will not have. The same asymmetry runs the
    other way on sales. So the projection is safe against our own queue and merely
    approximate against theirs; the claim of one-sided safety was wrong and is withdrawn.
    Reported 2026-09-01.
    """

    contract = "own-only"

    def __init__(self, money, market, shed, hires_today, capacity=SHED_CAPACITY):
        self.cash = float(money)
        self.inventory = {k: int(v) for k, v in
                          (market.get("inventory") or {}).items()}
        self.params = market.get("params")
        self.stored = int(sum(int(n) for n in (shed or {}).values()))
        self.hires_today = int(hires_today)
        self.capacity = int(capacity)
        self.proceeds = 0.0
        self.spent = 0.0

    @property
    def room(self):
        return max(0, self.capacity - self.stored)

    def _price(self, op, item):
        if op == "SELL":
            return market_price_at(item, self.inventory.get(item, 0), self.params)
        if op == "BUY_PRODUCT":
            return market_price_at(item, self.inventory.get(item, 0) - 1, self.params)
        if op == "BUY_SEED":
            return CROPS[item]["seed"]
        if op == "BUY_ANIMAL":
            return ANIMALS[item]["cost"]
        return None

    def commit(self, order):
        """Apply `order` unit by unit exactly as the engine will. Returns units committed."""
        op = order[0]
        if op == "HIRE":
            cost = hire_cost(self.hires_today)
            if self.cash < cost:
                return 0
            self.cash -= cost
            self.spent += cost
            self.hires_today += 1
            return 1
        if op == "BUY_LAND":
            # Price depends on how many quadrants are already unlocked; the caller passes the
            # exact sequential price it computed, because the ledger does not track the board.
            cost = float(order[1]) if len(order) > 1 else 0.0
            if self.cash < cost:
                return 0
            self.cash -= cost
            self.spent += cost
            return 1
        item, want = order[1], int(order[2])
        done = 0
        for _ in range(want):
            price = self._price(op, item)
            if price is None:
                break
            if op == "SELL":
                self.cash += price
                self.proceeds += price
                self.stored -= 1
                if price > PRICE_FLOOR:
                    self.inventory[item] = self.inventory.get(item, 0) + 1
            else:
                if self.cash < price:
                    break
                if op in ("BUY_PRODUCT", "BUY_ANIMAL") and self.stored >= self.capacity:
                    break
                self.cash -= price
                self.spent += price
                if op == "BUY_PRODUCT":
                    self.stored += 1
                    self.inventory[item] = self.inventory.get(item, 0) - 1
                elif op == "BUY_ANIMAL":
                    self.stored += 1
            done += 1
        return done

    def affordable(self, op, item, want, *, reserve=0.0):
        """How many units of `op item` this state can actually pay for, keeping `reserve`.

        Prices are re-quoted per unit, so this is exact for a seed line (flat price) and for a
        wheat top-up whose curve moves as it fills.
        """
        cash, inventory, stored = self.cash - float(reserve), dict(self.inventory), self.stored
        n = 0
        for _ in range(max(0, int(want))):
            if op == "BUY_SEED":
                price = CROPS[item]["seed"]
            elif op == "BUY_PRODUCT":
                price = market_price_at(item, inventory.get(item, 0) - 1, self.params)
            elif op == "BUY_ANIMAL":
                price = ANIMALS[item]["cost"]
            else:
                return int(want)
            if cash < price:
                break
            if op in ("BUY_PRODUCT", "BUY_ANIMAL") and stored >= self.capacity:
                break
            cash -= price
            if op == "BUY_PRODUCT":
                inventory[item] = inventory.get(item, 0) - 1
                stored += 1
            elif op == "BUY_ANIMAL":
                stored += 1
            n += 1
        return n


def _market_orders(
    v,
    cfg,
    total_days,
    incoming=None,
    fc=None,
    unit_actions=None,
    unit_projection=None,
    shed_capacity=SHED_CAPACITY,
):
    orders = []
    committed_capital = 0
    final_day = v.day >= total_days - 1
    fc = fc or econ.Forecast(v.market)

    projection = unit_projection
    n_animals = (
        int(projection["animal_count"]) if projection is not None else len(v.animals())
    )
    wheat_plants = (
        int(projection["wheat_plants"])
        if projection is not None
        else sum(1 for _, _, tile in v.plants() if tile.get("crop") == "WHEAT")
    )
    wheat_pipeline = 3 * wheat_plants
    carried_wheat = (
        sum(inventory.get("WHEAT", 0) for inventory in projection["inventories"])
        if projection is not None
        else sum(v.unit_inv(i).get("WHEAT", 0) for i in range(v.n_units))
    )
    # FEED executes before the market. Conservatively reserve one carried wheat for every
    # feasible requested FEED so tonight's consumption cannot also count as future cover.
    feed_wheat_spent = (
        int(projection["feed_wheat_spent"])
        if projection is not None
        else _successful_feed_wheat_spent(v, unit_actions)
    )
    # A projection already contains the committed consumption; the fallback starts pre-unit.
    post_unit_carried_wheat = (
        carried_wheat if projection is not None
        else max(0, carried_wheat - feed_wheat_spent)
    )
    cover_needed = n_animals * _feed_cover_days(cfg, v.day)

    # 1. Liquidate FIRST: sale proceeds land before later orders in the same queue are
    #    priced, so selling up front funds this very turn's purchases. Both players share
    #    one market and the premium curves are steep -- holding stock is a bet that the
    #    opponent will not dump. The wheat keep is the computed feed cover the pipeline
    #    does NOT already provide.
    endgame = v.day >= total_days - 2
    shed_view = dict(projection["shed"]) if projection is not None else dict(v.shed)
    if projection is None:
        for item, cnt in (incoming or {}).items():
            shed_view[item] = shed_view.get(item, 0) + cnt
    keep = 0
    if n_animals and not final_day:
        # Future plants may offset later cover but cannot feed tonight. Wheat carried by a unit
        # that is not landing it this turn is already protected from market sales; incoming
        # wheat becomes sellable and therefore belongs in the shed keep calculation.
        desired_total_holdings = max(n_animals, cover_needed - wheat_pipeline)
        protected_carried = max(0, post_unit_carried_wheat)
        keep = max(0, desired_total_holdings - protected_carried)
    sales = plan_sales(
        shed_view,
        v.market,
        budget_orders=int(cfg.SELL_ORDER_BUDGET),
        min_price=1 if final_day else 2,
        keep={"WHEAT": keep},
        min_ratio=0.0 if endgame else cfg.SELL_MIN_RATIO,
    )
    orders.extend(sales)
    sold_units = sum(int(order[2]) for order in sales)
    shed_room = max(0, shed_capacity - sum(shed_view.values()) + sold_units)

    # One projected cash/market/shed state, advanced in the exact order the engine resolves
    # this queue. See `_MarketLedger` for the four defects this replaces. The sales above are
    # committed first, which is the whole point of selling first: the proceeds are spendable by
    # the purchases below, in this same turn.
    exact = bool(getattr(cfg, "EXACT_MARKET_LEDGER", False))
    ledger = _MarketLedger(v.money, v.market, shed_view, v.farm.get("hires_today", 0),
                           capacity=shed_capacity)
    if exact:
        for order in sales:
            ledger.commit(order)

    # 2. Hire early. Locked shed-access spawns are passable, so same-turn hire count is no
    #    longer constrained by quadrant ownership; the configured order budget is the cap.
    if v.hour <= cfg.LAST_HIRE_HOUR:
        hires = v.farm.get("hires_today", 0)
        # PROD-006: the budget was a fraction of the PRE-MARKET cash even though the queue has
        # just sold. With the ledger it is a fraction of what we will actually hold.
        budget = max((ledger.cash if exact else v.money) * 0.25,
                     float(cfg.HIRE_BUDGET_FLOOR))
        spent = 0
        n_hire_orders = 0
        target = int(cfg.TARGET_HANDS)
        per_job = float(getattr(cfg, "HANDS_PER_JOB", 0.0))
        if per_job > 0:
            # Size the crew to the work that exists. A hand hired on day 1 onto a sixteen-tile
            # board has almost nothing to do and still costs cash at the moment the farm has
            # $20 of it.
            work = len(v.plants()) + 2 * len(v.animals())
            target = max(1, min(target, int(work / per_job)))
        while hires < target and n_hire_orders < cfg.HIRES_PER_TURN:
            c = hire_cost(hires)
            if spent + c > budget:
                break
            # PROD-003: hire spend used to live in this local `spent` alone, so the animal,
            # feed and seed decisions below sized themselves against cash the crew had taken.
            if exact and ledger.commit(["HIRE"]) != 1:
                break
            orders.append(["HIRE"])
            spent += c
            n_hire_orders += 1
            hires += 1

    # 3. Search the currently executable expansion choice. Unlike the legacy empty-tile gate,
    # this compares the next quadrant's modeled season value with its exact sequential price.
    # Only one quadrant is bought per turn; the next observation then exposes its real geometry
    # and the planner resolves the next choice instead of pretending all three are independent.
    if cfg.BUY_LAND and cfg.CAPITAL_SEARCH and len(orders) < MAX_MARKET_ORDERS:
        n_extra = len(v.farm.get("unlocked_quadrants", ["NW"])) - 1
        owned = n_extra + 1
        if n_extra < len(LAND_PRICES) and owned < int(cfg.MAX_QUADRANTS):
            price = LAND_PRICES[n_extra]
            value = econ.land_npv(v.day, total_days, fc, price=price)
            # PROD-005: affordability was tested against the pre-market observation, so a
            # sale-funded expansion in this same turn was rejected.
            have = ledger.cash if exact else v.money
            if value > 0 and have >= price + cfg.CASH_FLOOR:
                if not exact or ledger.commit(["BUY_LAND", price]) == 1:
                    orders.append(["BUY_LAND"])
                    committed_capital += price

    # 4. Geese by NPV, drip-fed one per day and capital-rationed: a goose is only bought
    #    when (a) its season NPV beats the crop tile its coop would occupy, and (b) the
    #    cash left over still covers this turn's seed plan and the feed shortfall. Buying
    #    the whole flock on day 0 once left the farm without seed money -- capital, not
    #    NPV, is the binding constraint early. A goose whose remaining producing days
    #    cannot repay the $300 has negative NPV: the late-season cutoff is computed.
    seed_plan = _seed_plan(v, cfg, total_days, fc, projection=projection)
    seed_budget = sum(CROPS[c]["seed"] * n for c, n in seed_plan)
    herd_reserve = 0
    herd_caps = _herd_caps(cfg)
    if any(cap > 0 for cap in herd_caps.values()) and len(orders) < MAX_MARKET_ORDERS:
        _, best_tile_value = econ.best_crop(v.day, total_days, fc)
        projected_inventories = (
            projection["inventories"] if projection is not None
            else [v.unit_inv(i) for i in range(v.n_units)]
        )
        placed_by_kind = _animals_by_kind(
            v, herd_caps,
            projected_tiles=projection["tiles"] if projection is not None else None,
        )
        unplaced = sum(
            shed_view.get(kind, 0)
            + sum(inventory.get(kind, 0) for inventory in projected_inventories)
            for kind in herd_caps
        )
        reserve = cfg.CASH_FLOOR + seed_budget
        # Early herd: production starts days after the purchase, so the first animals are
        # worth more than the seeds competing for the opening cash; fund them ahead of the
        # seed budget while the herd is below EARLY_HERD_TARGET in the first days.
        herd_owned = sum(
            placed_by_kind[kind] + shed_view.get(kind, 0)
            + sum(inventory.get(kind, 0) for inventory in projected_inventories)
            for kind in herd_caps
        )
        early_herd = (
            v.day <= int(cfg.EARLY_HERD_DAYS)
            and herd_owned < int(cfg.EARLY_HERD_TARGET)
        )
        if early_herd:
            reserve = cfg.CASH_FLOOR
        # Drip: only order the next animal once the previous one stands on its structure;
        # among the kinds still below their caps, buy the one with the best season NPV.
        candidates = []
        for kind, cap in herd_caps.items():
            if cap <= 0:
                continue
            have = (
                placed_by_kind[kind] + shed_view.get(kind, 0)
                + sum(inventory.get(kind, 0) for inventory in projected_inventories)
            )
            npv = econ.animal_npv(kind, v.day, total_days, fc)
            npv *= _service_discount(v, cfg)
            if npv > best_tile_value and have < cap:
                candidates.append((npv, kind))
        # A herd we cannot feed is worse than a smaller one: an animal that misses two
        # consecutive days escapes and takes its capital with it.  `animal_npv` values each
        # candidate on its own, so it cannot see that the wheat is already spoken for.
        feed_short = (
            bool(getattr(cfg, "HERD_FEED_GATE", False))
            and n_animals
            and wheat_pipeline + post_unit_carried_wheat + shed_view.get("WHEAT", 0)
            < cover_needed
        )
        if (
            candidates
            and not feed_short
            and unplaced <= int(cfg.ANIMAL_UNPLACED_ALLOWANCE)
            and shed_room > 0
        ):
            _, kind = max(candidates)
            # PROD-004: the animal was appended without debiting anything, so the feed and
            # seed lines below could spend its money a second time.
            have = (ledger.cash if exact else v.money - committed_capital)
            if have - ANIMALS[kind]["cost"] >= reserve:
                if not exact or ledger.commit(["BUY_ANIMAL", kind, 1]) == 1:
                    orders.append(["BUY_ANIMAL", kind, 1])
                    shed_room -= 1
                    herd_owned += 1
        if early_herd and candidates:
            # Hold the seed order back from the cash the remaining early animals need.
            cheapest = min(ANIMALS[kind]["cost"] for _, kind in candidates)
            herd_reserve = max(0, int(cfg.EARLY_HERD_TARGET) - herd_owned) * cheapest

    # 5. Feed top-up. Homegrown wheat is the primary supply (the pipeline above); this
    #    only bridges a shortfall, and only while feed costs less than the goose output
    #    it protects -- during a wheat price spike we dip into cover instead.
    if n_animals and not final_day and len(orders) < MAX_MARKET_ORDERS:
        sold_wheat = sum(
            int(order[2])
            for order in sales
            if len(order) >= 3 and order[0] == "SELL" and order[1] == "WHEAT"
        )
        have = (
            shed_view.get("WHEAT", 0)
            + post_unit_carried_wheat
            - sold_wheat
        )
        desired_total_holdings = max(n_animals, cover_needed - wheat_pipeline)
        want = desired_total_holdings - have
        want = min(want, shed_room)
        herd = [t for _, _, t in v.animals()]
        per_head = (
            econ.herd_daily_net(herd, fc) / len(herd) if herd else econ.goose_daily_net(fc)
        )
        if want > 0 and fc.buy_price("WHEAT") < per_head:
            if exact:
                # PROD-008: accounting used to be conditional on FEED_DEBITS_SEEDS, so turning
                # a strategy flag off silently reintroduced stale-cash sizing. The ledger makes
                # it unconditional and exact -- the wheat curve moves as the order fills.
                want = ledger.affordable("BUY_PRODUCT", "WHEAT", want)
            if want > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", want])
                if exact:
                    ledger.commit(["BUY_PRODUCT", "WHEAT", want])
                elif getattr(cfg, "FEED_DEBITS_SEEDS", False):
                    # Approximate: the curve moves as the order fills, but a reserve that is
                    # roughly right beats one that is exactly zero.
                    committed_capital += want * fc.buy_price("WHEAT")
                shed_room -= want

    # 6. Seeds: the plan computed for capital rationing above, clamped to what is left.
    if cfg.PLANTING_ENABLED and len(orders) < MAX_MARKET_ORDERS:
        for crop, n in seed_plan:
            if len(orders) >= MAX_MARKET_ORDERS:
                break
            # PROD-001: every crop line used to size against the same `v.money`, so a $100
            # balance emitted a $100 strawberry seed AND an $80 melon seed and the engine's
            # silent clipping picked which one we actually got. The ledger commits each line
            # before the next is priced.
            if exact:
                n = min(n, ledger.affordable("BUY_SEED", crop, n,
                                             reserve=cfg.CASH_FLOOR + herd_reserve))
            else:
                n = min(n, int(max(0, v.money - committed_capital - cfg.CASH_FLOOR
                                   - herd_reserve) // CROPS[crop]["seed"]))
            if n > 0:
                orders.append(["BUY_SEED", crop, n])
                if exact:
                    ledger.commit(["BUY_SEED", crop, n])

    # 7. Legacy expansion gate. Kept as the control for CAPITAL_SEARCH=False: the extra
    #    quadrants lengthen every walk and the cash is worth more as seed.
    if cfg.BUY_LAND and not cfg.CAPITAL_SEARCH and len(orders) < MAX_MARKET_ORDERS:
        n_extra = len(v.farm.get("unlocked_quadrants", ["NW"])) - 1
        owned = n_extra + 1
        if n_extra < len(LAND_PRICES) and owned < int(cfg.MAX_QUADRANTS):
            price = LAND_PRICES[n_extra]
            have = ledger.cash if exact else v.money
            if len(v.empty_tiles()) <= 2 and have >= price + cfg.LAND_RESERVE:
                if not exact or ledger.commit(["BUY_LAND", price]) == 1:
                    orders.append(["BUY_LAND"])

    return orders[:MAX_MARKET_ORDERS]


def _seed_plan(v, cfg, total_days, fc, *, projection=None):
    """[(crop, n)] seed purchases wanted this turn, before affordability clamping.

    Enough viable cash-crop seed to fill the open tiles, plus wheat seed when the feed
    pipeline does not cover the flock (same cover computation as the planting premium).

    AUTO ranks every held seed type again before each PLANT assignment.  Starting two days before
    half-season, count its existing non-wheat cash-crop seeds as fungible *capacity cover* here,
    even though the engine keeps the inventories separate.  Otherwise an intraday price flip buys
    a full second crop while viable seeds can already fill every open tile; those replay purchases
    then remain worthless at termination.  The two-day lead preserves the productive day-12
    planting wave while covering the first repeated surplus switch on day 13.  Fixed macro
    directives remain exact and feed wheat is not consumed as generic cash-crop cover.
    """
    plan = []
    if not cfg.PLANTING_ENABLED:
        return plan
    if cfg.CASH_CROP == "AUTO":
        crop, value = econ.best_crop(v.day, total_days, fc)
    elif cfg.CASH_CROP is None:
        crop, value = None, 0.0
    else:
        crop = cfg.CASH_CROP
        value = econ.crop_tile_value(crop, v.day, total_days, fc)
    if crop is not None and value > 0 and v.day <= _last_useful_plant_day(crop, total_days):
        seeds = projection["seeds"] if projection is not None else v.seeds
        empty_tiles = len(v.empty_tiles())
        if projection is not None:
            for position, final in projection["tiles"].items():
                initial = v.tile(*position)
                empty_tiles += int(final is None) - int(initial is None)
        have = seeds.get(crop, 0)
        if cfg.CASH_CROP == "AUTO" and v.day >= max(0, total_days // 2 - 2):
            have = sum(
                max(0, int(seeds.get(candidate, 0)))
                for candidate in CROPS
                if candidate != "WHEAT"
            )
            if crop == "WHEAT":
                have += max(0, int(seeds.get("WHEAT", 0)))
        want = max(0, empty_tiles + cfg.SEED_BUFFER - have)
        if want > 0:
            plan.append((crop, want))
    n_animals = (
        int(projection["animal_count"]) if projection is not None else len(v.animals())
    )
    wheat_plants = (
        int(projection["wheat_plants"])
        if projection is not None
        else sum(1 for _, _, tile in v.plants() if tile.get("crop") == "WHEAT")
    )
    wheat_pipeline = 3 * wheat_plants
    cover_needed = n_animals * _feed_plant_cover_days(cfg)
    if (
        n_animals
        and wheat_pipeline < cover_needed
        and v.day <= _last_useful_plant_day("WHEAT", total_days)
    ):
        seeds = projection["seeds"] if projection is not None else v.seeds
        have = seeds.get("WHEAT", 0)
        tiles_short = (cover_needed - wheat_pipeline + 2) // 3
        want = max(0, tiles_short + cfg.SEED_BUFFER - have)
        if want > 0:
            plan.append(("WHEAT", want))
    return plan


def _episode_steps(config):
    if config is None:
        return 720
    got = getattr(config, "episodeSteps", None)
    if got is None and hasattr(config, "get"):
        got = config.get("episodeSteps")
    return int(got or 720)


def _shed_capacity(config):
    if config is None:
        return SHED_CAPACITY
    got = getattr(config, "shedCapacity", None)
    if got is None and hasattr(config, "get"):
        got = config.get("shedCapacity")
    value = SHED_CAPACITY if got is None else int(got)
    if value <= 0:
        raise ValueError("shedCapacity must be positive")
    return value


def _own_supply(v):
    """Our unsold pipeline per product: shed + carried + expected standing-crop harvests.

    Ongoing crops count their remaining productions; one-time crops count what they will
    still gain (discounted) plus what is already on the plant.
    """
    supply = dict(v.shed)
    for i in range(v.n_units):
        for item, cnt in v.unit_inv(i).items():
            supply[item] = supply.get(item, 0) + cnt
    for _, _, t in v.plants():
        cd = CROPS.get(t.get("crop"))
        if cd is None:
            continue
        if cd["ongoing"]:
            prods_done = max(0, (v.day - t.get("planted_day", 0)
                                 - cd["first_yield_day"]) // cd["interval"] + 1)
            expected = t.get("yield_units", 0) + max(0, cd["max_yield"] - prods_done)
        else:
            expected = t.get("yield_units", 0) \
                + 0.8 * max(0, cd["max_yield"] - t.get("yield_units", 0))
        supply[t["crop"]] = supply.get(t["crop"], 0) + expected
    return supply


def _animal_supply(tiles, day, horizon, into, weight=1.0):
    """Add livestock production over the next `horizon` days to a supply estimate.

    Animal tiles are public on both farms and carry everything needed: `placed_day` fixes the
    schedule, `yield_units` is what is already standing on the animal.  Counts the base unit per
    scheduled production only -- the CARE bonus needs `fed_today` and is not ours to assume on
    the opponent's herd -- so this under-states rather than over-suppresses.
    """
    if horizon <= 0 or weight <= 0:
        return into
    for row in tiles or []:
        for tile in row:
            if not isinstance(tile, dict) or "animal" not in tile:
                continue
            ad = ANIMALS.get(tile["animal"])
            if ad is None:
                continue
            first = tile.get("placed_day", day) + ad["first_yield_day"]
            interval = max(1, ad["interval"])
            productions = sum(
                1 for k in range(int(horizon) // interval + 2)
                if day < first + k * interval <= day + horizon
            )
            units = tile.get("yield_units", 0) + productions
            if units:
                item = ad["product"]
                into[item] = into.get(item, 0) + weight * units
    return into


def _opponent_supply(v, weight):
    """The opponent's VISIBLE standing pipeline, weighted into our forecast.

    `_own_supply` is what stops the calculator pricing every new tile as if it were the only
    seller.  It is only half the correction: both players sell into ONE market, and `farms` is
    public, so the opponent's planted tiles are observable future supply.  Measured 2026-08-31,
    a frontier tape holds 12 melon and 20-33 strawberry tiles by day 10 -- enough to bury both
    premium curves -- while our AUTO crop choice prices them as though nobody else grew any.

    Their shed and carried stock are private, so this counts standing crops only; that is the
    public part, and it is the part that has not been sold yet.  `weight` discounts it because
    the observed drain rate already extrapolates whatever they have been dumping, so the two
    signals overlap for a crop they are already harvesting.
    """
    if weight <= 0:
        return {}
    supply = {}
    for row in v.opponent.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                continue
            cd = CROPS.get(tile.get("crop"))
            if cd is None:
                continue
            if cd["ongoing"]:
                prods_done = max(0, (v.day - tile.get("planted_day", 0)
                                     - cd["first_yield_day"]) // cd["interval"] + 1)
                expected = tile.get("yield_units", 0) + max(0, cd["max_yield"] - prods_done)
            else:
                expected = tile.get("yield_units", 0) \
                    + 0.8 * max(0, cd["max_yield"] - tile.get("yield_units", 0))
            supply[tile["crop"]] = supply.get(tile["crop"], 0) + weight * expected
    return supply


#: Work of the same class shares carried items, tooling and usually a region of the board.
_OP_CLASS = {
    "FEED": "animal", "CARE": "animal", "COLLECT_FERTILIZER": "animal",
    "BUILD_COOP": "animal", "BUILD_PASTURE": "animal",
    "WATER": "crop", "FERTILIZE": "crop", "PLANT": "crop", "DIG": "crop",
    "HARVEST": "harvest",
    "PICKUP": "haul", "PLACE": "haul", "DROP": "haul",
}


def _op_class(op):
    """Which kind of work an operation is. HARVEST is its own class because it applies to both
    crops and animals and always ends with the unit holding produce."""
    return _OP_CLASS.get(op, "other")


def _drop_unpaired_bundles(assignments, jobs):
    """Remove assignments whose `companion` job was not also assigned this turn."""
    if not any(jobs[k].get("companion") for _u, k in assignments):
        return assignments
    landed = {(tuple(jobs[k]["pos"]), jobs[k]["op"][0]) for _u, k in assignments}
    return [
        (u, k) for u, k in assignments
        if not jobs[k].get("companion")
        or (tuple(jobs[k]["companion"][0]), jobs[k]["companion"][1]) in landed
    ]


def plan_turn(
    v,
    cfg,
    total_days,
    fc=None,
    planner_state=None,
    *,
    shed_capacity=SHED_CAPACITY,
    assign=None,
    stickiness=None,
):
    """Tactical executor: given a View and a Config, produce this turn's full action dict.

    `fc` is the market Forecast (spot prices when omitted); it is re-wrapped with our own
    unsold supply so every value below prices in what we are already going to sell. All
    job and order values are computed against it.
    """
    rates = fc.rates if fc else None
    supply = _own_supply(v)
    horizon = int(getattr(cfg, "ANIMAL_SUPPLY_DAYS", 0))
    opponent_weight = float(getattr(cfg, "OPPONENT_SUPPLY_WEIGHT", 0.0))
    _animal_supply(v.farm.get("tiles"), v.day, horizon, supply)
    # Fertilizer is not in `_animal_supply`: it is not any animal's `product`, every surviving
    # animal makes exactly one per end-of-day regardless of kind, and it is the largest revenue
    # line in the animal NPV.
    fertilizer_days = int(getattr(cfg, "FERTILIZER_SUPPLY_DAYS", 0))
    if fertilizer_days > 0:
        herd = len(v.animals())
        if herd:
            days = min(fertilizer_days, max(0, total_days - 1 - v.day))
            supply["FERTILIZER"] = supply.get("FERTILIZER", 0) + herd * days
    for item, units in _opponent_supply(v, opponent_weight).items():
        supply[item] = supply.get(item, 0) + units
    _animal_supply(v.opponent.get("tiles"), v.day, horizon, supply, weight=opponent_weight)
    fc = econ.Forecast(v.market, rates, supply=supply)
    actions = _plan_units(
        v, cfg, total_days, fc, planner_state, shed_capacity=shed_capacity, assign=assign,
        stickiness=stickiness,
    )
    unit_projection = _project_unit_market_ledger(
        v, actions, shed_capacity=shed_capacity
    )
    return {
        "farmer": actions[0] if actions else ["PASS"],
        "hands": actions[1:],
        "market": _market_orders(
            v,
            cfg,
            total_days,
            incoming=_incoming_drops(v, actions, shed_capacity=shed_capacity),
            fc=fc,
            unit_actions=actions,
            unit_projection=unit_projection,
            shed_capacity=shed_capacity,
        ),
    }


#: How fast the online drain-rate estimate tracks the observed net inventory change.
RATE_EMA_ALPHA = 0.25

# ``main.py`` deploys this conservative seven-turn repair solver.  Learned directives change
# strategy targets, but not the production executor's already-gated routing/search behavior.
PRODUCTION_EXECUTOR_OVERRIDES = {"LOOKAHEAD_HORIZON": 7}


def macro_directive_to_config(directive, base=None):
    """Adapt the executor-neutral directive to the current production solver contract."""
    if not isinstance(directive, MacroDirective):
        directive = MacroDirective.from_head_values(directive)
    cfg = {**DEFAULTS, **PRODUCTION_EXECUTOR_OVERRIDES}
    if base is not None:
        cfg.update(dict(base))
    cfg.update(
        {
            "CASH_CROP": directive.cash_crop,
            "PLANTING_ENABLED": (
                directive.cash_crop is not None or directive.feed_cover_days > 0
            ),
            "TARGET_HANDS": directive.target_hands,
            "SELL_MIN_RATIO": directive.minimum_sell_ratio,
            "BUY_LAND": directive.allow_land_purchase,
            "MAX_GEESE": directive.target_goose_population,
            "FEED_COVER_DAYS": directive.feed_cover_days,
            "MAX_COWS": directive.target_cow_population,
            "MAX_SHEEP": directive.target_sheep_population,
            "EARLY_HERD_TARGET": directive.early_herd_target,
        }
    )
    return Config(**cfg)


def _owned_animals(v, kind):
    """Placed + shed + carried animals of one kind."""
    placed = sum(1 for _x, _y, tile in v.animals() if tile.get("animal") == kind)
    carried = sum(v.unit_inv(index).get(kind, 0) for index in range(v.n_units))
    return placed + int(v.shed.get(kind, 0)) + carried


def _strategy_snapshot(v):
    crops = {crop: 0 for crop in CROPS}
    for _x, _y, tile in v.plants():
        crops[tile["crop"]] += 1
    carried_wheat = sum(v.unit_inv(index).get("WHEAT", 0) for index in range(v.n_units))
    return {
        "money": float(v.money),
        "hires_today": int(v.farm.get("hires_today", 0)),
        "hands": max(0, int(v.n_units) - 1),
        "unlocked_quadrants": len(v.farm.get("unlocked_quadrants", ["NW"])),
        # v.animals() returns EVERY placed animal, so before the mixed herd this happened to
        # equal the goose count and afterwards silently counted cows and sheep as geese.  Each
        # kind is now counted on its own, placed + shed + carried.
        "goose_owned": _owned_animals(v, "GOOSE"),
        "cow_owned": _owned_animals(v, "COW"),
        "sheep_owned": _owned_animals(v, "SHEEP"),
        "wheat_available": int(v.shed.get("WHEAT", 0)) + carried_wheat,
        "crop_tiles": crops,
    }


def _empty_emitted_effects():
    return {
        "hire_orders": 0,
        "land_orders": 0,
        "animal_buys": {},
        "seed_buys": {},
        "sell_requests": {},
        "plant_requests": {},
    }


def _accumulate_emitted(effects, action):
    unit_actions = [action.get("farmer", ["PASS"]), *(action.get("hands", []) or [])]
    for wire in unit_actions:
        if isinstance(wire, list) and len(wire) >= 2 and wire[0] == "PLANT":
            crop = str(wire[1])
            effects["plant_requests"][crop] = effects["plant_requests"].get(crop, 0) + 1
    for wire in action.get("market", []) or []:
        if not isinstance(wire, list) or not wire:
            continue
        if wire[0] == "HIRE":
            effects["hire_orders"] += 1
        elif wire[0] == "BUY_LAND":
            effects["land_orders"] += 1
        elif len(wire) >= 3:
            target = {
                "BUY_ANIMAL": "animal_buys",
                "BUY_SEED": "seed_buys",
                "SELL": "sell_requests",
            }.get(wire[0])
            if target is not None:
                item = str(wire[1])
                effects[target][item] = effects[target].get(item, 0) + int(wire[2])


def _realized_effects(day_state, current):
    start = day_state["start"]
    return {
        "money_delta": float(current["money"] - start["money"]),
        "peak_hires_today": int(day_state["peak_hires"]),
        "peak_hands": int(day_state["peak_hands"]),
        "land_quadrants_delta": int(
            current["unlocked_quadrants"] - start["unlocked_quadrants"]
        ),
        "goose_owned_delta": int(current["goose_owned"] - start["goose_owned"]),
        "cow_owned_delta": int(current["cow_owned"] - start["cow_owned"]),
        "sheep_owned_delta": int(current["sheep_owned"] - start["sheep_owned"]),
        "wheat_available_delta": int(
            current["wheat_available"] - start["wheat_available"]
        ),
        "crop_tile_delta": {
            crop: int(current["crop_tiles"][crop] - start["crop_tiles"][crop])
            for crop in CROPS
        },
    }


class MacroExecutor:
    """Stateful adapter from MacroDirective v3 to the production search/value executor."""

    contract_version = MACRO_DIRECTIVE_VERSION

    def __init__(self):
        self.seats = {}

    def act(self, obs, directive, env_config=None, *, return_audit=False):
        if not isinstance(directive, MacroDirective):
            directive = MacroDirective.from_head_values(directive)
        v = View(obs)
        if not v.farm:
            action = {"farmer": ["PASS"], "hands": [], "market": []}
            return (action, {}) if return_audit else action
        seat = self.seats.setdefault(
            v.player,
            {
                "step": None,
                "inv": {},
                "rates": {},
                "planner": {"plans": {}},
                "day_state": None,
            },
        )
        if seat["step"] is not None and v.step < seat["step"]:
            seat = {
                "step": None,
                "inv": {},
                "rates": {},
                "planner": {"plans": {}},
                "day_state": None,
            }
            self.seats[v.player] = seat
        current = _strategy_snapshot(v)
        completed_previous = None
        if seat["day_state"] is None or seat["day_state"]["day"] != v.day:
            if seat["day_state"] is not None:
                previous = seat["day_state"]
                completed_previous = {
                    "day": previous["day"],
                    "emitted": previous["emitted"],
                    "realized": _realized_effects(previous, current),
                }
            seat["day_state"] = {
                "day": v.day,
                "start": current,
                "peak_hires": current["hires_today"],
                "peak_hands": current["hands"],
                "emitted": _empty_emitted_effects(),
            }
            seat["planner"] = {"plans": {}}
        day_state = seat["day_state"]
        day_state["peak_hires"] = max(day_state["peak_hires"], current["hires_today"])
        day_state["peak_hands"] = max(day_state["peak_hands"], current["hands"])

        cfg = macro_directive_to_config(directive)
        forecast = econ.Forecast(v.market, _update_drain_rates(seat, v, cfg))
        total_days = max(1, _episode_steps(env_config) // TURNS_PER_DAY)
        action = plan_turn(
            v,
            cfg,
            total_days,
            forecast,
            planner_state=seat["planner"],
            shed_capacity=_shed_capacity(env_config),
        )
        _accumulate_emitted(day_state["emitted"], action)
        audit = {
            "executor": "heuristic.policy:MacroExecutor",
            "contract_version": self.contract_version,
            "day": v.day,
            "requested": directive.to_payload(),
            "adapted_config": {
                "CASH_CROP": cfg.CASH_CROP,
                "PLANTING_ENABLED": cfg.PLANTING_ENABLED,
                "TARGET_HANDS": cfg.TARGET_HANDS,
                "SELL_MIN_RATIO": cfg.SELL_MIN_RATIO,
                "BUY_LAND": cfg.BUY_LAND,
                "MAX_GEESE": cfg.MAX_GEESE,
                "MAX_COWS": cfg.MAX_COWS,
                "MAX_SHEEP": cfg.MAX_SHEEP,
                "EARLY_HERD_TARGET": cfg.EARLY_HERD_TARGET,
                "FEED_COVER_DAYS": cfg.FEED_COVER_DAYS,
                "LOOKAHEAD_HORIZON": cfg.LOOKAHEAD_HORIZON,
            },
            "emitted": day_state["emitted"],
            "realized": _realized_effects(day_state, current),
            "first_turn_action": action if v.hour == 0 else None,
            "completed_previous": completed_previous,
        }
        return (action, audit) if return_audit else action


def _update_drain_rates(seat, v, cfg):
    """Refresh this seat's net drain-rate estimate from the observed inventory change.

    Observed change = players' net selling MINUS the town's consumption.  With
    ``TOWN_DRAIN_EXACT`` the known half is subtracted before smoothing and added back after, so
    a shop unlock moves the forecast on the turn it becomes visible instead of bleeding into the
    EMA over the following days, and the smoothed quantity is the part that is actually
    uncertain.  With the flag off this is exactly the previous single-EMA behaviour.
    """
    inventory = v.market.get("inventory", {})
    if seat["step"] is not None and v.step > seat["step"]:
        days = (v.step - seat["step"]) / TURNS_PER_DAY
        exact = bool(getattr(cfg, "TOWN_DRAIN_EXACT", False))
        known = econ.town_daily_drain(v.town) if exact else {}
        players = seat.setdefault("players", {})
        for item, now in inventory.items():
            observed = (now - seat["inv"].get(item, now)) / days
            if exact:
                drain = known.get(item, 0.0)
                supplied = observed + drain
                previous = players.get(item, supplied)
                players[item] = (1 - RATE_EMA_ALPHA) * previous + RATE_EMA_ALPHA * supplied
                seat["rates"][item] = players[item] - drain
            else:
                previous = seat["rates"].get(item, observed)
                seat["rates"][item] = (
                    (1 - RATE_EMA_ALPHA) * previous + RATE_EMA_ALPHA * observed
                )
    seat["step"] = v.step
    seat["inv"] = dict(inventory)
    return seat["rates"]


def make_agent(assign=None, stickiness=None, **overrides):
    """Build an agent callable with its own tunables. Use this for sweeps and frozen opponents.

    `assign` is the external-plan seam (see `plan_turn`): a callable that receives this turn's
    entries and the assignment the executor chose, and returns the assignment to run.  Left None,
    this is exactly the deployed solver.
    """
    cfg = Config(**overrides)
    # Per-seat market tracker: net inventory drain per product (signed units/day), EMA of
    # observed changes. Keyed by seat because the same callable may play BOTH seats of a
    # mirror match, and the two seats' observations interleave at the same step.
    market_track = {}

    def _agent(obs, config=None):
        v = View(obs)
        if not v.farm:
            return {"farmer": ["PASS"], "hands": [], "market": []}
        seat = market_track.setdefault(
            v.player, {"step": None, "inv": {}, "rates": {}, "planner": {"plans": {}}}
        )
        if seat["step"] is not None and v.step < seat["step"]:
            # Arena/Kaggle may reuse one callable across episodes. Forecast rates describe one
            # particular shared market and must never leak into the next seed.
            seat = {"step": None, "inv": {}, "rates": {}, "planner": {"plans": {}}}
            market_track[v.player] = seat
        if v.hour == 0:
            # Unit indices and all hand positions reset every day.
            seat["planner"] = {"plans": {}}
        fc = econ.Forecast(v.market, _update_drain_rates(seat, v, cfg))
        return plan_turn(
            v,
            cfg,
            max(1, _episode_steps(config) // TURNS_PER_DAY),
            fc,
            planner_state=seat["planner"],
            shed_capacity=_shed_capacity(config),
            assign=assign,
            stickiness=stickiness,
        )

    _agent.config = cfg
    _agent.market_track = market_track
    return _agent


#: Base greedy/default control. Current ``main.py`` explicitly deploys ``lookahead_agent``.
agent = make_agent()

# Fixed control for arenas: same economics and actions, pre-commitment greedy routing.
greedy_agent = make_agent(ASSIGNMENT_SEARCH=False, ROUTE_STICKINESS=0.0)

# Experimental exact current-turn router. It is not the production default until it beats the
# greedy control against fixed opponents; current results expose multi-turn goal thrashing.
search_agent = make_agent(ASSIGNMENT_SEARCH=True, ROUTE_STICKINESS=0.0)

# Time-expanded development solver. It exhaustively enumerates bounded legal job orders per
# worker and resolves the joint conflicts with deterministic branch-and-bound. Kept separate
# until disjoint fixed-opponent and mirror gates establish that the extra planning pays.
lookahead_agent = make_agent(LOOKAHEAD_HORIZON=7)

# Fully replacing greedy assignments with exhaustive seven-turn rollout is intentionally exposed
# as a rejected ablation. It finds more locally valuable completions but perturbs crop supply and
# regresses true-mirror money; use `lookahead_agent` for the conservative repair gate.
exhaustive_agent = make_agent(
    LOOKAHEAD_HORIZON=7,
    LOOKAHEAD_PRESERVE_GREEDY=False,
    LOOKAHEAD_REPLANT=True,
    LOOKAHEAD_REPAIR_RADIUS=10,
    LOOKAHEAD_CRITICAL_ONLY=False,
    ROUTE_STICKINESS=0.0,
)

# Aggressive search is safest when the model has an exact terminal value: the final ten actions.
# Unlike daily search, there is no future planting mix to perturb, and every inventory-producing
# job has a proved route-to-shed deadline.
endgame_search_agent = make_agent(
    LOOKAHEAD_HORIZON=10,
    LOOKAHEAD_FINAL_DAY_ONLY=True,
    LOOKAHEAD_PRESERVE_GREEDY=False,
    LOOKAHEAD_REPLANT=True,
    LOOKAHEAD_REPAIR_RADIUS=10,
    LOOKAHEAD_CRITICAL_ONLY=False,
    ROUTE_STICKINESS=0.0,
)

# Lightweight temporal-search ablation: retain a viable route target for up to ten observations
# unless a materially higher-value job wins the density comparison.
sticky_route_agent = make_agent()
sticky50_route_agent = make_agent(ROUTE_STICKINESS=0.50, ROUTE_COMMIT_TURNS=10)

# Experimental aggressive capital search. It over-expands before labour/cash can service the new
# frontier and is retained as a measured counterexample, not the production policy.
early_capital_agent = make_agent(BUY_LAND=True, CAPITAL_SEARCH=True)

# Historical compatibility alias for the base control; production imports ``lookahead_agent``.
solver_agent = agent

# Proven expansion control: the pre-search rule waits until the current quadrant is nearly full.
land_agent = make_agent(BUY_LAND=True, CAPITAL_SEARCH=False)

# Labour controls used by true mirrors and holdout comparisons for the expanded four-quadrant
# policy. They remain explicit callables so a sweep cannot accidentally compare against an
# opponent whose supposedly matching config stayed at the default.
hands10_agent = make_agent(TARGET_HANDS=10)
hands12_agent = make_agent(TARGET_HANDS=12)

# Ablation for the pre-lookahead policy's known last-hour no-ops.
legacy_day_agent = make_agent(DAY_END_FEASIBILITY=False, ROUTE_STICKINESS=0.0)
