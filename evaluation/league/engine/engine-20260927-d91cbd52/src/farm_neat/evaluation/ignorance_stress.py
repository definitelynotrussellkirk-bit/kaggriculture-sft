"""Exact finite ignorance bounds on native MARKET-PHASE fixtures; zero games.

This is an evaluator, not an actor or a second simulator. It enumerates a declared
one-batch stopping problem using the pinned market/town kernels, solves pure
minimax regret with pre-action information, and compresses replayable evidence.
No full-game, RNG-population, actual tape0 performance, or global-optimality claim.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from farm_neat.evaluation.common import bound, canonical, digest, file_sha, read, write_new
from farm_neat.game.exact_engine import (AttrDict, DEFAULT_CONFIGURATION,
                                         assert_bundled_engine_identity, load_engine)

SCHEMA = 'farm-ignorance-stress-v1'
TIMES = tuple(range(568, 593, 4))
QUANTITIES = (0, 1, 8, 24, 100)
INVENTORIES = (10000, 9900)
TAPE = ROOT / 'src/farm_neat/evaluation/shop_router_0909/actions.json'
SENSORS = ('clock', 'shops', 'market', 'public')


def stopping_regret(payoffs, observations, baseline=0):
    """Exact PURE minimax regret for a finite one-shot stopping menu.

    payoffs[s][t]: complete realized utility of committing at t in scenario s.
    observations[s][t]: hashable sensor value BEFORE the t-th decision on the
    all-WAIT path. Until stopping, past own actions are identical. The history
    partition prevents merging distinct memories or using a future signal.
    All stopping times must be feasible or have their real failed payoff recorded.
    A nonfinite/missing cell is UNKNOWN, never replaced by a penalty.

    Returns a clock-only minimax action and an observation-contingent stopping
    policy. Policies are pure; randomized policies may improve their bound.
    Restricted actions AND scenarios mean neither minimax result is a global
    upper/lower certificate. For a fixed included tape, a witnessed regret is a
    lower bound on unrestricted worst-case oracle regret, subject to reachability.
    """
    if not payoffs or not payoffs[0]:
        raise ValueError('nonempty complete payoff matrix required')
    n, horizon = len(payoffs), len(payoffs[0])
    if len(observations) != n or type(baseline) is not int or not 0 <= baseline < horizon:
        raise ValueError('matching observations and included baseline required')
    for row, obs in zip(payoffs, observations):
        if len(row) != horizon or len(obs) != horizon:
            raise ValueError('UNKNOWN: incomplete matrix/history')
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in row):
            raise ValueError('UNKNOWN: missing/nonfinite payoff')
        for value in obs:
            hash(value)
    oracle = [max(row) for row in payoffs]
    regret = [[v - x for x in row] for v, row in zip(oracle, payoffs)]
    static = [max(row[t] for row in regret) for t in range(horizon)]
    best_static = min(range(horizon), key=lambda t: (static[t], t))
    histories = [[tuple(obs[:t+1]) for t in range(horizon)] for obs in observations]

    def partition(ids, t):
        groups = defaultdict(list)
        for s in ids:
            groups[histories[s][t]].append(s)
        return tuple(tuple(g) for g in groups.values())

    @lru_cache(None)
    def solve_node(t, ids):
        stop = max(regret[s][t] for s in ids)
        if t == horizon - 1:
            return stop, 'SELL'
        wait = max(solve_node(t+1, g)[0] for g in partition(ids, t+1))
        return (stop, 'SELL') if stop <= wait else (wait, 'WAIT')

    roots = partition(tuple(range(n)), 0)
    optimum = max(solve_node(0, g)[0] for g in roots)
    selected = [None] * n
    policy = []

    def trace(t, ids):
        value, action = solve_node(t, ids)
        policy.append({'slot': t, 'history_sha256': digest(histories[ids[0]][t]),
                       'action': action, 'conditional_worst_regret': value})
        if action == 'SELL':
            for s in ids:
                selected[s] = t
        else:
            for g in partition(ids, t+1):
                trace(t+1, g)
    for g in roots:
        trace(0, g)
    realized = [regret[s][t] for s, t in enumerate(selected)]
    if max(realized) != optimum or optimum > static[best_static]:
        raise AssertionError('nonanticipative policy certificate failed')
    worst = max(range(n), key=lambda s: regret[s][baseline])
    return {'scenario_count': n, 'action_count': horizon,
            'baseline_slot': baseline, 'baseline_regret_min': min(r[baseline] for r in regret),
            'baseline_regret_max': static[baseline], 'baseline_worst_scenario': worst,
            'baseline_worst_payoff': min(r[baseline] for r in payoffs),
            'baseline_worst_payoff_scenario': min(range(n), key=lambda s: payoffs[s][baseline]),
            'all_static_worst_regrets': static, 'best_static_slot': best_static,
            'best_static_worst_regret': static[best_static],
            'causal_worst_regret': optimum, 'causal_selected_slots': selected,
            'causal_policy': policy, 'oracle_payoffs': oracle,
            'scope': 'EXACT_DECLARED_FINITE_PURE_STOPPING_CLASS'}


def state_fixture(item='MILK', inventory=10000, seat=0):
    engine = load_engine()
    farms = [engine._new_farm(10, 3000) for _ in range(2)]
    privates = [engine._new_private() for _ in range(2)]
    privates[seat]['shed'][item] = 24
    # Same hidden endowment in every scenario. Quantity is a chosen sale, not
    # supplied to the observation partition as a hidden inventory label.
    privates[1-seat]['shed'][item] = 100
    market = engine._new_market()
    market['inventory'][item] = inventory
    engine._refresh_prices(market)
    town = {'unlocked_shops': ['BAKERY'] * 7}
    state = [AttrDict(observation=AttrDict(farms=farms, private=p, market=market,
                                        town=town, player=i), action={})
             for i, p in enumerate(privates)]
    env = SimpleNamespace(configuration=AttrDict(DEFAULT_CONFIGURATION.copy()), info={'seed': 0})
    return engine, state, env


def market_case(item, inventory, seat, shop, rival_quantity, rival_step, sell_step):
    """Sparse native phase calls: intervening turns have no market/town events.

    Fixed seven BAKERIES precede the eighth shop at step576, before observation.
    Real engine end-of-day575 unlock precedes actions576. No biology, new delivery,
    labor or capacity competition is asserted. Endowments are INJECTED, not replayed.
    Opponent script is selected before our policy, never best-responded per cell.
    sell_step=None produces only the legitimate waiting-path sensor histories.
    """
    engine, state, env = state_fixture(item, inventory, seat)
    market, farms, town = (state[0].observation[k] for k in ('market', 'farms', 'town'))
    history = {name: [] for name in SENSORS}
    for step in TIMES:
        if step == 576:
            town['unlocked_shops'].append(shop)
        # The signal cannot contain a current simultaneous opponent action.
        history['clock'].append(step)
        history['shops'].append(canonical([step, town['unlocked_shops']]))
        history['market'].append(canonical([step, market['inventory'][item]]))
        history['public'].append(canonical([step, market['inventory'],
                                            town['unlocked_shops'], [farms[seat]['money'], farms[1-seat]['money']]]))
        for s in state:
            s.action = {'market': []}
        if step == sell_step:
            state[seat].action['market'] = [['SELL', item, 24]]
        if rival_quantity and step == rival_step:
            state[1-seat].action['market'] = [['SELL', item, rival_quantity]]
        engine._process_market(state, env)
        engine._town_consume(env, state, step)
    cash = [farms[seat]['money'], farms[1-seat]['money']]
    filled = [24 - state[seat].observation.private['shed'][item],
              100 - state[1-seat].observation.private['shed'][item]]
    if filled != [0 if sell_step is None else 24, rival_quantity]:
        raise AssertionError('declared endowment failed to realize requested sale')
    return cash, history, filled


def market_panel(item, inventory, seat):
    engine = load_engine()
    cases = [{'shop': shop, 'quantity': q, 'step': t}
             for shop in sorted(engine.SHOPS) for q in QUANTITIES
             for t in ((TIMES[0],) if q == 0 else TIMES)]
    rows, sensors = [], {k: [] for k in SENSORS}
    for case in cases:
        _, hist, _ = market_case(item, inventory, seat, case['shop'], case['quantity'], case['step'], None)
        for sensor in sensors:
            sensors[sensor].append(hist[sensor])
        rows.append([market_case(item, inventory, seat, case['shop'], case['quantity'],
                                 case['step'], t)[0] for t in TIMES])
    channels = {}
    for metric in ('own_cash', 'cash_margin'):
        utility = [[c[0] if metric == 'own_cash' else c[0]-c[1] for c in row] for row in rows]
        all_results = {sensor: stopping_regret(utility, obs) for sensor, obs in sensors.items()}
        base = all_results['public']
        worst, poor = base['baseline_worst_scenario'], base['baseline_worst_payoff_scenario']
        oracle_slot = max(range(len(TIMES)), key=lambda t: utility[worst][t])
        frontier = []
        for budget in QUANTITIES:
            ids = [i for i, c in enumerate(cases) if c['quantity'] <= budget]
            # Full <=budget set includes PASS and every smaller tested size.
            r = stopping_regret([utility[i] for i in ids], [sensors['public'][i] for i in ids])
            frontier.append({'opponent_sale_budget': budget, 'scenarios': len(ids),
                             'fixed_regret': r['baseline_regret_max'],
                             'best_static_regret': r['best_static_worst_regret'],
                             'public_causal_regret': r['causal_worst_regret']})
        channels[metric] = {k: base[k] for k in ('baseline_regret_min', 'baseline_regret_max',
            'baseline_worst_payoff', 'all_static_worst_regrets', 'best_static_slot',
            'best_static_worst_regret', 'causal_worst_regret')}
        channels[metric].update(
            sensor_regrets={k: r['causal_worst_regret'] for k, r in all_results.items()},
            robust_tape_improvement=base['baseline_regret_max']-base['best_static_worst_regret'],
            information_improvement=base['best_static_worst_regret']-base['causal_worst_regret'],
            budget_frontier=frontier,
            worst_regret_witness={'scenario': cases[worst], 'baseline_sell_step': TIMES[0],
                'oracle_sell_step': TIMES[oracle_slot], 'cash_own_opponent_baseline': rows[worst][0],
                'cash_own_opponent_oracle': rows[worst][oracle_slot]},
            worst_payoff_witness={'scenario': cases[poor], 'cash_own_opponent': rows[poor][0]},
            causal_policy_sha256=digest(base['causal_policy']),
            causal_selected_slots_sha256=digest(base['causal_selected_slots']))
    return {'item': item, 'initial_market_inventory': inventory, 'seat': seat,
            'scenarios': len(cases), 'payoff_cells': len(cases)*len(TIMES),
            'matrix_sha256': digest({'scenarios': cases, 'cash_own_opponent': rows}),
            'waiting_observations_sha256': digest(sensors), 'channels': channels}


def demand_envelope():
    """Exact marginal shop-demand envelope, NOT a joint/cash/RNG-seed bound."""
    engine = load_engine()
    weights = [len(range(t, 719, 4)) for t in range(72, 577, 72)]
    rows = {}
    for item in engine.PRODUCTS:
        rates = {s: (2 if len(p) == 1 else 1) if item in p else 0
                 for s, p in sorted(engine.SHOPS.items())}
        center = len(range(0, 719, 24)) if item in engine.TOWN_CENTER_PRODUCTS else 0
        lo, hi = min(rates.values()), max(rates.values())
        rows[item] = {'shop_units_min': sum(weights)*lo, 'shop_units_max': sum(weights)*hi,
                      'center_units': center, 'min_total_units': sum(weights)*lo+center,
                      'max_total_units': sum(weights)*hi+center,
                      'per_shop_tick': rates}
    return {'unlock_steps': list(range(72, 577, 72)), 'remaining_shop_ticks': weights,
            'shop_sequences': len(engine.SHOPS)**8, 'products': rows,
            'scope': 'MARGINAL_EXACT_OVER_8_CHOICE_SCHEDULES; NOT_JOINTLY_ATTAINABLE_OR_SEED_CERTIFIED'}


def weed_witnesses():
    engine = load_engine()
    out = []
    # Native action budget and day-end deadline; no value attached to unsold yield.
    for remaining in (3, 2):
        for weed in (False, True):
            for repair in (False, True):
                farm, private = engine._new_farm(10, 3000), engine._new_private()
                x, y = farm['farmer']
                farm['tiles'][y][x] = {'kind': 'WEED'} if weed else None
                private['seeds']['WHEAT'] = 1
                actions = ([['DIG'], ['PLANT', 'WHEAT'], ['WATER']] if repair and weed
                           else [['PLANT', 'WHEAT'], ['WATER'], ['PASS']])[:remaining]
                for a in actions:
                    engine._apply_unit_action(farm, private, 0, a, 10, 0, 24, 100)
                engine._daily_refresh_plants(farm, 0, 24)
                tile = farm['tiles'][y][x]
                out.append({'turns_before_day_end': remaining, 'weed': weed, 'repair': repair,
                            'actions': actions, 'tile_after_day_end': tile,
                            'seeds_remaining': private['seeds']['WHEAT']})
    return out


def prerequisite_witnesses():
    """A paid 2-unit wheat buy can break a nearly funded 2-cow feeding tape.

    Same initial stock/cash/assets per control; rescue observes the realized own
    fill, explicitly sells one fertilizer, and uses an extra turn. Does not claim
    to optimize all possible rescues or translate a lost animal into terminal cash.
    """
    rows = []
    for seat in (0, 1):
        for turns in (6, 5):
            for rival_buy in (0, 2):
                for guarded in (False, True):
                    engine, state, env = state_fixture(seat=seat)
                    farms = state[0].observation.farms
                    state[0].observation.town['unlocked_shops'] = []
                    for i in (0, 1):
                        state[i].observation.private = engine._new_private()
                    private = state[seat].observation.private
                    private['shed']['FERTILIZER'] = 1
                    farms[seat]['money'] = 52.0
                    for x in (4, 3):
                        tile = engine._new_animal('COW', 0)
                        tile['consecutive_unfed'] = 1
                        farms[seat]['tiles'][4][x] = tile
                    actions, refill = [], False
                    post_purchase = None
                    baseline = [['PASS'], ['PICKUP', 'WHEAT', 2], ['FEED'], ['WEST'], ['FEED'], ['PASS']]
                    rescued = [['PASS'], ['PASS'], ['PICKUP', 'WHEAT', 2], ['FEED'], ['WEST'], ['FEED']]
                    for offset in range(turns):
                        if offset == 1:
                            refill = guarded and private['shed']['WHEAT'] < 2
                        unit = (rescued if refill else baseline)[offset]
                        own_queue = [['BUY_PRODUCT', 'WHEAT', 2]] if offset == 0 else []
                        if offset == 1 and refill:
                            own_queue = [['SELL', 'FERTILIZER', 1], ['BUY_PRODUCT', 'WHEAT', 1]]
                        for st in state:
                            st.action = {'market': []}
                        state[seat].action['market'] = own_queue
                        if offset == 0 and rival_buy:
                            state[1-seat].action['market'] = [['BUY_PRODUCT', 'WHEAT', rival_buy]]
                        engine._apply_unit_action(farms[seat], private, 0, unit, 10, 10, 24, 100)
                        engine._process_market(state, env)
                        engine._town_consume(env, state, 10*24 + 24-turns + offset)
                        actions.append({'unit': unit, 'market': own_queue})
                        if offset == 0:
                            post_purchase = {'own_wheat_fill': private['shed']['WHEAT'],
                                             'own_cash': farms[seat]['money'],
                                             'rival_cash': farms[1-seat]['money']}
                    engine._daily_refresh_animals(farms[seat], 10)
                    alive = sum(farms[seat]['tiles'][4][x].get('animal') == 'COW' for x in (4, 3))
                    rows.append({'seat': seat, 'turns_before_day_end': turns,
                                 'rival_buy_wheat': rival_buy, 'guarded': guarded,
                                 'post_purchase': post_purchase, 'repair_fired': refill,
                                 'cows_surviving': alive, 'cash_own_opponent':
                                 [farms[seat]['money'], farms[1-seat]['money']],
                                 'fertilizer_sold': 1-private['shed']['FERTILIZER'],
                                 'actions': actions, 'native_market_calls': turns,
                                 'native_town_calls': turns, 'native_animal_refresh_calls': 1})
    return {'scope': 'INJECTED_PREREQUISITE_CHAIN; NOT_TERMINAL_VALUE_OR_OPTIMAL_RESCUE',
            'initial_cash_own_opponent': [52, 3000], 'initial_cows': 2,
            'initial_fertilizer': 1, 'cows_already_consecutive_unfed': 1,
            'controls': 'Same state within seat/deadline; paid rival BUY2 versus PASS; actual own-fill guard.',
            'cases': rows}


def seed_coupling():
    engine = load_engine()
    rows = []
    for seed in range(32):
        outcomes = []
        for occupied in (False, True):
            _, state, env = state_fixture()
            env.info['seed'] = seed
            state[0].observation.town['unlocked_shops'] = []
            if occupied:
                state[0].observation.farms[0]['tiles'][0][0] = {'kind': 'WEED'}
            engine._end_of_day(state, env, 2)
            outcomes.append(state[0].observation.town['unlocked_shops'][-1])
        rows.append([seed, *outcomes])
    changed = [r for r in rows if r[1] != r[2]]
    return {'injected_state_seed_pairs': 32, 'native_day_end_calls': 64,
            'changed_last_shop': len(changed), 'all_rows_sha256': digest(rows),
            'witness': changed[0] if changed else None,
            'scope': 'SAME_SEED_DIFFERENT_OCCUPANCY; INJECTED_STATES; NO_SEED_SEARCH_OR_GAME'}


def tape_audit():
    actions = read(TAPE, binding=False)[0]
    units, orders, amounts = Counter(), Counter(), Counter()
    for row in actions:
        for a in [row.get('farmer', ['PASS']), *row.get('hands', [])]:
            if a:
                units[a[0]] += 1
        for a in row.get('market', []):
            if a:
                key = ' '.join(str(v) for v in a[:2])
                orders[key] += 1
                if len(a) >= 3 and type(a[2]) is int:
                    amounts[key] += a[2]
    return {'file_sha256': file_sha(TAPE), 'tape_index': 0, 'turns': len(actions),
            'unit_requests': dict(sorted(units.items())), 'market_orders': dict(sorted(orders.items())),
            'requested_quantities': dict(sorted(amounts.items())),
            'scope': 'STATIC_REQUESTS_ONLY; NOT_ACCEPTED_ACTIONS_FILLS_EXPOSURE_OR_PERFORMANCE'}


def source_pins():
    paths = [Path(__file__), ROOT/'src/farm_neat/evaluation/common.py',
             ROOT/'src/farm_neat/game/exact_engine.py',
             ROOT/'src/farm_neat/game/kaggriculture_engine_reference.py',
             ROOT/'src/farm_neat/game/kaggriculture.json', TAPE]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def plan():
    return {'schema': SCHEMA, 'source_pins': source_pins(), 'engine': assert_bundled_engine_identity(),
            'items': list(load_engine().PRODUCTS), 'initial_market_inventories': list(INVENTORIES),
            'seats': [0, 1], 'sale_steps': list(TIMES), 'opponent_sale_quantities': list(QUANTITIES),
            'own_units': 24, 'rival_endowment_units': 100, 'initial_cash_each': 3000,
            'known_shops': ['BAKERY']*7, 'new_shop_step': 576,
            'max_games': 0, 'initialized_worlds': 0, 'max_cpu_seconds': 1800,
            'observation_contract': 'PRE_ACTION_ALL_WAIT_HISTORY; NO_CURRENT_RIVAL_ACTION_OR_FUTURE_SHOP',
            'disturbance_contract': 'COMMITTED_OPPONENT_SALE_SCRIPT; NATIVE_REQUOTE_NOT_FROZEN_OUTCOMES',
            'reachability': 'INJECTED_COMPONENT_STATES; PREFIX_REACHABILITY_UNKNOWN',
            'distribution': 'FINITE_ADVERSARIAL_SET; NO_PROBABILITY_OR_TAIL_RISK_CLAIM',
            'prerequisite_panel': '2seats x 2deadlines x 2paid_attack_arms x 2guard_arms; 16 component cases'}


def run_panel(p):
    panels = [market_panel(item, inv, seat) for item in p['items']
              for inv in p['initial_market_inventories'] for seat in p['seats']]
    cells = sum(r['payoff_cells'] for r in panels)
    scenarios = sum(r['scenarios'] for r in panels)
    # The paired seat runs are actual checks, not a copied symmetry claim.
    for left, right in zip(panels[::2], panels[1::2]):
        if left['matrix_sha256'] != right['matrix_sha256'] or left['channels'] != right['channels']:
            raise AssertionError('seat-dependent result in pure paired-sale fixture')
    return {'schema': SCHEMA, 'plan_sha256': digest(p), 'source_pins': source_pins(),
            'physical_games': 0, 'initialized_worlds': 0, 'optimizer_updates': 0,
            'independent_worlds': 0, 'market_payoff_cells': cells,
            'waiting_paths': scenarios, 'market_kernel_calls': (cells+scenarios)*len(TIMES),
            'town_kernel_calls': (cells+scenarios)*len(TIMES), 'panels': panels,
            'kernel_count_scope': 'Market matrix only; prerequisite native calls are counted separately per case.',
            'demand_envelope': demand_envelope(), 'weed_witnesses': weed_witnesses(),
            'prerequisite_witnesses': prerequisite_witnesses(),
            'seed_coupling': seed_coupling(), 'actual_tape0_static_audit': tape_audit(),
            'limits': ['Exact only over declared one-batch stopping menus and finite disturbances.',
                       'Own cash/margin are component-horizon balances, not completed-game WDL.',
                       'No biological deliveries, purchase squeeze or adaptive rival is in the matrix.',
                       'Shop schedule legality does not certify reachability from the official seeded generator.',
                       'No actual tape0 rollout; audit contains requests, not fills.']}



def compact_result(result):
    """Retain literals, controls and reconstructible hashes, not duplicate panels.

    Both seats are evaluated before deduplication. This is lossless for the
    declared summary columns, not a claim that hashes replace raw replication.
    market_panel() reconstructs a full stratum, including policies/frontiers.
    """
    panels = result['panels']
    retained = []
    if len(panels) % 2:
        raise ValueError('unpaired seat panel')
    for left, right in zip(panels[::2], panels[1::2]):
        keys = ('item', 'initial_market_inventory', 'matrix_sha256',
                'waiting_observations_sha256', 'channels')
        if (left['seat'], right['seat']) != (0, 1) or any(left[k] != right[k] for k in keys):
            raise ValueError('seat deduplication requires checked equality')
        row = [left[k] for k in keys[:-1]]
        for metric in ('own_cash', 'cash_margin'):
            c = left['channels'][metric]
            row.append([c['baseline_regret_min'], c['baseline_regret_max'],
                        c['baseline_worst_payoff'], c['best_static_slot'],
                        c['all_static_worst_regrets'],
                        [c['sensor_regrets'][k] for k in SENSORS]])
        retained.append(row)
    out = {k: v for k, v in result.items() if k not in ('panels', 'content_sha256')}
    out.update(evaluated_strata=len(panels), full_panel_summaries_sha256=digest(panels),
               paired_seats_equal=True, panels=retained,
               panel_columns=['item', 'initial_market_inventory', 'matrix_sha256',
                              'waiting_observations_sha256', 'own_cash', 'cash_margin'],
               metric_columns=['baseline_regret_min', 'baseline_regret_max',
                               'baseline_worst_payoff', 'best_static_slot',
                               'all_static_worst_regrets', 'sensor_regrets'],
               sensor_order=list(SENSORS))
    # Preserve one complete adverse distinction and one quantity-budget curve.
    milk = next((r for r in panels if r['item'] == 'MILK' and
                 r['initial_market_inventory'] == 10000 and r['seat'] == 0), None)
    out['milk_witnesses'] = {} if milk is None else {
        metric: {k: c[k] for k in ('worst_regret_witness', 'worst_payoff_witness', 'budget_frontier')}
        for metric, c in milk['channels'].items()}
    feed = result['prerequisite_witnesses']
    cases = feed['cases']
    normalize = lambda c: {k: v for k, v in c.items() if k != 'seat'}
    if ([normalize(c) for c in cases if c['seat'] == 0] !=
            [normalize(c) for c in cases if c['seat'] == 1]):
        raise ValueError('prerequisite seats differ')
    out['prerequisite_witnesses'] = {
        **{k: v for k, v in feed.items() if k != 'cases'},
        'evaluated_cases': len(cases), 'all_cases_sha256': digest(cases),
        'paired_seats_equal': True,
        'cases': [{k: v for k, v in c.items() if k not in ('actions', 'seat')}
                  for c in cases if c['seat'] == 0],
        'action_witnesses': [{k: c[k] for k in ('guarded', 'actions')}
                             for c in cases if c['seat'] == 0 and
                             c['rival_buy_wheat'] == 2 and c['turns_before_day_end'] == 6]}
    if len(canonical(out)) > 24000:
        raise ValueError('compact receipt exceeds 24KB; do not raise retention cap')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'check', 'run', 'verify'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.absolute()
    if not out.is_relative_to(ROOT/'work/runs') or any(p.is_symlink() for p in (out, *out.parents)):
        parser.error('non-symlink work/runs output required')
    if args.command == 'register':
        if out.exists():
            parser.error('new output directory required; no silent overwrite')
        write_new(out/'PLAN.json', plan())
        print('REGISTERED: zero games; finite native component panel')
        return
    p = read(out/'PLAN.json')
    body = {k: v for k, v in p.items() if k != 'content_sha256'}
    if canonical(body) != canonical(plan()):
        raise ValueError('source/plan drift; original registration must remain unchanged')
    if args.command == 'check':
        print('CHECKED: source-bound finite panel, no execution')
        return
    # CPU cap also applies to accidental regressions; no timeout becomes a result.
    import resource
    cpu_now = math.ceil(resource.getrusage(resource.RUSAGE_SELF).ru_utime +
                         resource.getrusage(resource.RUSAGE_SELF).ru_stime)
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    limit = cpu_now + body['max_cpu_seconds']
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_CPU, (limit, hard))
    result = compact_result(run_panel(body))
    if args.command == 'run':
        write_new(out/'RESULT.json', result)
    elif canonical(read(out/'RESULT.json')) != canonical(bound(result)):
        raise ValueError('retained result differs from exact replay')
    print(canonical({'status': 'VERIFIED' if args.command == 'verify' else 'COMPLETE',
                     'payoff_cells': result['market_payoff_cells'],
                     'native_market_calls': result['market_kernel_calls'], 'physical_games': 0}))


if __name__ == '__main__':
    main()
