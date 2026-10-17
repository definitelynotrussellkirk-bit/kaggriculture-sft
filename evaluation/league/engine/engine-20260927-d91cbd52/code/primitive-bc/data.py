"""Streaming per-step primitive-BC examples from raw public Kaggle replays (v3).

Pairing (bc.from_replay_step): the actor view at boundary t of seat s
(bc.public_observation -> actor-visible whitelist) is labelled with the action
that seat submitted after observing it, replay["steps"][t+1][s]["action"],
for t = 0..718. Both seats of an episode are separate perspectives.

Split: whole episodes, deterministic sha256 bucket of the episode id
(train 90% / val 5% / test 5%). The 16 sealed confirmation episodes
(work/datasets/public-episodes/daily/SELECTION.json, split == "confirmation")
are dropped by member name before any byte of them is read.

v3 (ARCH-PBC-v3.md):
  * canonical_observation(obs): copy with each private.inventories dict sorted by key.
    THE observation function of primitive-BC: features()/step_inputs() apply it, and
    agent.py calls it before pack/features (bc.actor_observation is unchanged).
  * canonical_labels(replay): the episode is replayed through the exact engine; a lean
    copy of play.audit_step (same order statuses/fills, tested equal) gives per-order
    fills. Orders after the last order with fill > 0 are dropped (interior orders are
    kept as codec.canonical gives them: malformed -> NOOP, zero-fill orders literal);
    a SELL whose audited fill empties the item becomes SELL_ALL (100, the canonical
    SELL clamp: shed stock never exceeds the shed capacity). A None action is masked
    (loss_mask False; the engine executes it as PASS). If the engine diverges from the
    replay (actor views differ, or the audit's post-market cash differs from the
    engine's), the whole episode keeps literal labels (codec.canonical) and is counted.
  * Inputs per step: pack rows (+knownness, float16) of the canonical observation,
    feature_tokens.tokens(FeatureState.compute()) (float16), tile_row (pack row of
    each of the 200 cells), prev_unit_tok (own previous canonical unit tokens; -1 at
    step 0, for hands at hour 0 and for units absent at t-1). One FeatureState per
    perspective from step 0: update(canonical obs with step) -> compute() -> tokens;
    after the step, record_action(canonical label action).
  * Teacher metadata from STRENGTH.json (strength.py): team_id, team-day Bradley-Terry
    strength and its decile; strength_top keeps perspectives in the top fraction.
  * iter_perspectives(...) -> Perspective objects (zlib-compressed per-step blobs);
    Perspective.chunks(W) -> lists of W per-step example dicts; collate_chunks(chunks).
  * Pretraining source: a directory holding episode-*.jsonl.gz (pbc-pretrain batch)
    in archive_paths is read by the same pipeline (engine audit via the recorded
    seed/rule; PASS seats are never teachers).

v3.1 data rules (ARCH-PBC-v3.md owner decisions 2026-09-23), decided per seat from the
episode header before any replay step is read (filter_reason; same function for the live
stream, the step cache and the header report):
  * wins_only (default True): a seat is a teacher only if its final reward (header
    rewards[seat]; pretraining: terminal cash) is strictly greater than the rival's.
  * exclude_static_tapes (default True): seats labelled static in EPISODE-TAPE-LABELS
    (tape_id set or match >= 0.95) give no perspective; the labels file is required
    unless allow_no_tape_labels=True.
  * strength_top (unchanged, default 0.6).
  * Curriculum.weight(meta, f): perspective sampling weight at training progress f
    (uniform until 0.5, exponential tilt toward high strength deciles and recent days
    until 0.8, then only strength_decile >= 8 from the last 5 days; Curriculum.fit
    relaxes that final set when it holds too few perspectives).
  Every Kaggle perspective's meta carries won, tie, rewards, day_ordinal and its own and
  the rival's tape_id / tape_match / tape_static (None when the seat is unlabelled).

The v1 per-example/window API (iter_examples, iter_windows, collate, collate_windows)
is kept; it now packs canonical observations too.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import glob
import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import struct
import sys
import time
import zipfile
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import codec  # noqa: E402
import feature_tokens  # noqa: E402
import features as FS  # noqa: E402  (primitive-bc FeatureState; `features` below is the pack function)
import strength as ST  # noqa: E402
from farm_neat.game.exact_engine import DEFAULT_CONFIGURATION, _copy_wire  # noqa: E402
from farm_neat.policy import observation as features_mod  # noqa: E402
from farm_neat.training import bc  # noqa: E402

try:
    import orjson as _json_fast
except ImportError:  # pragma: no cover
    _json_fast = None

ARCHIVE_GLOB = "/mnt/steam/farm-data/kaggle/kaggriculture-episodes-2026-09-*/kaggriculture-episodes-2026-09-*.zip"
PRETRAIN_DIR = Path("/mnt/ai/scratch/farmNEAT/pbc-pretrain-20260923/batch-2000-v1")
LEADERBOARD = Path("/mnt/ai/scratch/farmNEAT/cohort-scale-20260923/leaderboard/kaggriculture.zip")
TAPE_LABELS = Path("/mnt/ai/scratch/farmNEAT/pbc-tape-corpus-20260923/EPISODE-TAPE-LABELS.jsonl.gz")
TAPE_STATIC_MATCH = 0.95  # per-step match to a mined tape at or above this = static tape (owner rule)
SELECTION = ROOT / "work/datasets/public-episodes/daily/SELECTION.json"
SPLIT_SALT = b"primitive-bc-split-v1:"
SPLITS = {"train": (0, 900), "val": (900, 950), "test": (950, 1000),
          "heldout": (900, 1000), "all": (0, 1000)}
LAST_STEP = 718          # last boundary with a submitted successor action
STEPS = LAST_STEP + 1    # examples per perspective in a complete episode
TURNS_PER_DAY = 24
N_ORDER_SLOTS = codec.N_ORDER_SLOTS
MAX_UNITS = codec.MAX_UNITS
DIGIT_SLOTS = codec.DIGIT_SLOTS
N_CELLS = feature_tokens.N_CELLS
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def _loads(raw):
    return _json_fast.loads(raw) if _json_fast is not None else json.loads(raw)


# ---------------------------------------------------------------- sources
def default_archives():
    return sorted(glob.glob(ARCHIVE_GLOB))


def sealed_ids(path=SELECTION):
    """IDs only; the sealed replays themselves are never opened."""
    entries = json.loads(Path(path).read_text())["episodes"]
    sealed = {int(e["episode_id"]) for e in entries if e["split"] == "confirmation"}
    if len(sealed) != 16:
        raise ValueError("expected the 16 sealed confirmation episode ids")
    return frozenset(sealed)


def split_bucket(episode_id):
    digest = hashlib.sha256(SPLIT_SALT + str(int(episode_id)).encode()).hexdigest()
    return int(digest, 16) % 1000


def in_split(episode_id, split):
    lo, hi = SPLITS[split]
    return lo <= split_bucket(episode_id) < hi


def load_leaderboard(path=LEADERBOARD):
    """TeamName -> best (smallest) public-leaderboard rank. Accepts a CSV or the Kaggle zip (v1 filter)."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            text = z.read(name).decode("utf-8-sig")
    else:
        text = path.read_text(encoding="utf-8-sig")
    ranks = {}
    for row in csv.DictReader(io.StringIO(text)):
        name, rank = row["TeamName"], int(row["Rank"])
        ranks[name] = min(rank, ranks.get(name, rank))
    return ranks


def archive_day(path):
    name = Path(path).name
    return name.split("episodes-")[-1][:10]


def list_members(archive_paths=None, *, split="all", sealed=None):
    """[(archive, member, day, episode_id)] in archive then member order; sealed removed by name."""
    sealed = sealed_ids() if sealed is None else sealed
    out = []
    for path in archive_paths or default_archives():
        with zipfile.ZipFile(path) as z:
            names = sorted(n for n in z.namelist() if n.endswith(".json") and n[:-5].isdigit())
        for name in names:
            eid = int(name[:-5])
            if eid in sealed or not in_split(eid, split):
                continue
            out.append((str(path), name, archive_day(path), eid))
    return out


def read_header(archive, member, limit=65536):
    """Replay fields before "steps" (info, rewards, statuses, configuration) without the steps."""
    with zipfile.ZipFile(archive) as z, z.open(member) as f:
        head = f.read(limit)
    cut = head.find(b'"steps"')
    if cut < 0:
        raise ValueError("replay header not found in the first bytes")
    return json.loads(head[:cut].rstrip().rstrip(b",") + b"}")


def read_replay(archive, member):
    with zipfile.ZipFile(archive) as z:
        return _loads(z.read(member))


def seat_ranks(header, leaderboard):
    teams = header.get("info", {}).get("TeamNames") or [None, None]
    return teams, [leaderboard.get(t) if t is not None else None for t in teams]


def keep_seat(rank, rank_max):
    return rank_max is None or (rank is not None and rank <= rank_max)


def seat_list(seats):
    if seats == "both":
        return (0, 1)
    if seats in (0, 1):
        return (seats,)
    raise ValueError("seats must be 'both', 0 or 1")


# ---------------------------------------------------------------- observation -> inputs
def canonical_observation(actor_obs):
    """Copy of an actor observation with each private.inventories dict sorted by key.

    The replay JSON stores inventories key-sorted while the engine keeps insertion order,
    and pack() ranks carried items in dict order (852/1440 pack-row differences on one
    engine-replayed episode). Every primitive-BC train and play input goes through this."""
    out = dict(actor_obs)
    private = actor_obs.get("private")
    if isinstance(private, dict):
        private = dict(private)
        inventories = private.get("inventories")
        if isinstance(inventories, list):
            private["inventories"] = [dict(sorted(inv.items())) if isinstance(inv, dict) else inv
                                      for inv in inventories]
        out["private"] = private
    return out


def pack_config(configuration):
    cfg = {k: configuration[k] for k in features_mod.CONFIG if k in (configuration or {})}
    if configuration and "marketParams" in configuration:
        cfg["marketParams"] = {item: dict(p) for item, p in configuration["marketParams"].items()}
    return cfg


def features(actor_obs, configuration):
    """Pack inputs of an actor observation (canonicalized here): rows float16 [N,80],
    token_mask [N], knownness [N,80], unit_pos [MAX_UNITS,2] (-1 absent; farmer, then hands
    in observation order), tile_row [200] (pack row of cell role*100+y*10+x)."""
    obs = canonical_observation(actor_obs)
    packed = features_mod.pack(obs, None, pack_config(configuration))
    pos = np.full((MAX_UNITS, 2), -1, np.int64)
    units = codec.unit_positions(obs)[:MAX_UNITS]
    if units:
        pos[:len(units)] = np.asarray(units, np.int64)
    return {"rows": packed["rows"].astype(np.float16), "token_mask": packed["token_mask"],
            "knownness": packed["knownness"], "unit_pos": pos,
            "tile_row": feature_tokens.pack_tile_rows(packed["token_ids"])}


def feature_inputs(feature_state):
    """feature_tokens arrays of the FeatureState's current step, float16-rounded values/tile."""
    return feature_tokens.tokens(feature_state.compute(), dtype=np.float16)


def prev_unit_tokens(prev_unit_tok, actor_obs):
    """[MAX_UNITS] int64: the actor's own unit tokens (codec unit_tok) of its previous step's
    canonical action; -1 when there is none (step 0), for every hand at hour 0 (hands are
    dismissed at day end) and for units absent now or at the previous step."""
    out = np.full(MAX_UNITS, -1, np.int64)
    step = int(actor_obs["step"])
    if prev_unit_tok is None or step == 0:
        return out
    hour = int(actor_obs.get("hour", step % TURNS_PER_DAY))
    n = 1 if hour == 0 else min(MAX_UNITS, 1 + codec.hand_count(actor_obs))
    out[:n] = np.asarray(prev_unit_tok, np.int64)[:n]
    return out


def step_inputs(actor_obs, configuration, feature_state, prev_unit_tok=None):
    """THE per-step input function shared by data.py and agent.py.

    Returns (canonical observation, inputs dict: features() arrays + feat + prev_unit_tok).
    Updates feature_state with the canonical observation (must carry 'step'); the caller
    records the canonical action it submits with feature_state.record_action afterwards."""
    obs = canonical_observation(actor_obs)
    ex = features(obs, configuration)
    feature_state.update(obs)
    ex["feat"] = feature_inputs(feature_state)
    ex["prev_unit_tok"] = prev_unit_tokens(prev_unit_tok, obs)
    return obs, ex


def check_replay(replay):
    if (replay.get("name") != "kaggriculture" or not isinstance(replay.get("steps"), list)
            or len(replay["steps"]) != 720 or replay.get("statuses") != ["DONE", "DONE"]):
        raise ValueError("complete two-seat Kaggriculture replay required")
    cfg = replay.get("configuration", {})
    if any(cfg.get(k) != v for k, v in features_mod.CONFIG.items()):
        raise ValueError("replay mechanics configuration differs from the fixed schema")


# ---------------------------------------------------------------- engine audit
def audit_step(env, actions):
    """Per-order fills of one joint action, from the engine's own functions on copies.

    Same unit phase (atomic PLANT seed rule, farmer then hands) and per-index lockstep
    market as the interpreter, like play.audit_step (test_data asserts equal order
    statuses/requested/filled and cash) but without per-unit state diffs. Per seat:
    {"orders": [(label, status, requested, filled, post)], "truncated_orders": n,
    "shed_after_units": dict}; post = own shed stock of the order's item after the
    order (SELL / BUY_PRODUCT / BUY_ANIMAL), else None. Also returns the simulated
    post-market cash per seat (the engine changes cash only in the market phase)."""
    E = env.engine
    cfg = env.configuration
    tpd = max(1, int(E.get(cfg, "turnsPerDay", 24)))
    board = int(E.get(cfg, "boardSize", 10))
    shed_cap = int(E.get(cfg, "shedCapacity", 100))
    max_orders = max(1, int(E.get(cfg, "maxMarketOrdersPerTurn", 10)))
    hire_mult = int(E.get(cfg, "farmHandCostMult", E.FARM_HAND_COST_MULT))
    obs0 = env.state[0].observation
    day = int(E.get(obs0, "step", 0)) // tpd
    farms = [_copy_wire(dict(f)) for f in obs0.farms]
    privates = [_copy_wire(dict(env.state[p].observation.private)) for p in range(2)]
    market = _copy_wire(dict(obs0.market))
    audits = []
    for i in range(2):
        action = actions[i] if isinstance(actions[i], dict) else {}
        farmer_action = action.get("farmer", ["PASS"])
        hands_actions = action.get("hands", [])
        if not isinstance(hands_actions, list):
            hands_actions = []
        demand = {}
        for a in (farmer_action, *hands_actions):
            if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT":
                demand[a[1]] = demand.get(a[1], 0) + 1
        seeds = privates[i].get("seeds", {})
        blocked = {crop for crop, n in demand.items() if n > seeds.get(crop, 0)}

        def allowed(a):
            if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] in blocked:
                return ["PASS"]
            return a

        E._apply_unit_action(farms[i], privates[i], 0, allowed(farmer_action), board, day, tpd, shed_cap)
        for h, hand_action in enumerate(hands_actions):
            E._apply_unit_action(farms[i], privates[i], h + 1, allowed(hand_action), board, day, tpd, shed_cap)
        audits.append({"orders": [], "shed_after_units": dict(privates[i]["shed"])})
    queues, records = [], []
    for i in range(2):
        action = actions[i] if isinstance(actions[i], dict) else {}
        m = action.get("market", [])
        q = list(m) if isinstance(m, list) else []
        audits[i]["truncated_orders"] = max(0, len(q) - max_orders)
        q = q[:max_orders]
        queues.append(q)
        records.append([[_order_label(o), None, 0, 0, None] for o in q])
    for k in range(max((len(q) for q in queues), default=0)):
        states = []
        for i, q in enumerate(queues):
            st = None
            if k < len(q):
                st = E._parse_order(q[k])
                if st is None:
                    records[i][k][1] = "unparsed"
                else:
                    records[i][k][2] = st.get("remaining", 1)
            states.append(st)
        for i, st in enumerate(states):
            if st is not None and st["type"] in ("HIRE", "BUY_LAND"):
                before = (farms[i]["money"], len(farms[i]["hands"]), len(farms[i]["unlocked_quadrants"]))
                if st["type"] == "HIRE":
                    E._do_hire(farms[i], privates[i], board, hire_mult)
                else:
                    E._do_buy_land(farms[i], board)
                after = (farms[i]["money"], len(farms[i]["hands"]), len(farms[i]["unlocked_quadrants"]))
                records[i][k][3] = int(after != before)
                states[i] = None
        stock_items = [st["item"] if st is not None and st["type"] in ("SELL", "BUY_PRODUCT", "BUY_ANIMAL")
                       else None for st in states]
        guard = 0
        while True:
            guard += 1
            if guard >= 100_000:
                break
            quoted = [None, None]
            for i, st in enumerate(states):
                if st is None or st["remaining"] <= 0:
                    continue
                op, item = st["type"], st["item"]
                if op == "SELL" and item in E.PRODUCTS:
                    quoted[i] = (op, item, E.market_price(item, market["inventory"][item], market.get("params")), st)
                elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
                    quoted[i] = (op, item, E.market_price(item, market["inventory"][item] - 1, market.get("params")), st)
                elif op == "BUY_SEED" and item in E.CROPS:
                    quoted[i] = (op, item, E.CROPS[item]["seed"], st)
                elif op == "BUY_ANIMAL" and item in E.ANIMALS:
                    quoted[i] = (op, item, E.ANIMALS[item]["cost"], st)
                else:
                    records[i][k][1] = "bad_item"
                    states[i] = None
            if all(x is None for x in quoted):
                break
            committed_any = False
            for i, x in enumerate(quoted):
                if x is None:
                    continue
                op, item, price, st = x
                if E._commit_unit(op, item, price, farms[i], privates[i], market, shed_cap):
                    st["remaining"] -= 1
                    records[i][k][3] += 1
                    committed_any = True
                else:
                    states[i] = None
            if not committed_any:
                break
        for i, item in enumerate(stock_items):
            if item is not None and records[i][k][1] is None:
                records[i][k][4] = int(privates[i]["shed"].get(item, 0))
        E._refresh_prices(market)
    for i in range(2):
        for r in records[i]:
            if r[1] is None:
                r[1] = "ignored" if r[3] == 0 else "partial" if r[3] < r[2] else "effective"
        audits[i]["orders"] = [tuple(r) for r in records[i]]
    return audits, [float(f["money"]) for f in farms]


def _order_label(order):
    if isinstance(order, list) and order and isinstance(order[0], str):
        return order[0]
    return "MALFORMED"


def _engine_view(env, seat):
    return bc.actor_observation(env.agent_observation(seat))


class EpisodeLabels:
    """canonical_labels / label_episode result: per seat per step the label action (codec
    canonical form; None = masked), the action FeatureState records, and counters."""

    __slots__ = ("actions", "history", "masked", "literal", "first_divergence", "divergence", "stats")

    def __init__(self, actions, history, masked, literal, first_divergence, divergence, stats):
        self.actions, self.history, self.masked, self.literal = actions, history, masked, literal
        self.first_divergence, self.divergence, self.stats = first_divergence, divergence, stats

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


LABEL_COUNTERS = ("steps", "masked_none", "masked_unrepresentable", "raw_orders", "raw_orders_truncated",
                  "canonical_orders", "trailing_noop_removed", "tail_removed", "kept_orders",
                  "interior_zero_fill", "interior_zero_fill_noop", "sell_emptied", "sell_relabel",
                  "codec_engine_mismatch")


def _strip(lit, raw, audit, stats):
    """Tail strip + SELL relabel of one canonical action with its audited order fills."""
    orders = audit["orders"]
    fills = [o[3] for o in orders]
    last = max((k for k, f in enumerate(fills) if f > 0), default=-1)
    market = lit["market"]
    if len(market) < last + 1 or len(orders) != min(len(raw_market(raw)), codec.MAX_ORDERS):
        stats["codec_engine_mismatch"] += 1
        return None
    kept = []
    for k, o in enumerate(market[:last + 1]):
        o = list(o)
        if fills[k] > 0 and o == ["NOOP"]:
            stats["codec_engine_mismatch"] += 1
            return None
        if fills[k] == 0:
            stats["interior_zero_fill"] += 1
            stats["interior_zero_fill_noop"] += o == ["NOOP"]
        elif o[0] == "SELL" and orders[k][4] == 0:
            stats["sell_emptied"] += 1
            if o[2] != codec.SELL_ALL:
                o[2] = codec.SELL_ALL
                stats["sell_relabel"] += 1
        kept.append(o)
    stats["tail_removed"] += len(market) - len(kept)
    stats["kept_orders"] += len(kept)
    return {"farmer": list(lit["farmer"]), "hands": [list(h) for h in lit["hands"]], "market": kept}


def raw_market(action):
    m = action.get("market", []) if isinstance(action, dict) else []
    return m if isinstance(m, list) else []


def label_episode(env, views, raw, *, check_views=True):
    """Engine-audited canonical labels of one episode (both seats).

    env: a fresh exact environment at step 0 of this episode; views[s][t]: canonical actor
    observation of seat s at boundary t; raw[s][t]: the action seat s submitted after it.
    env is stepped through every step with the raw joint actions."""
    T = len(views[0])
    audits = []
    first_div, divergence = None, None
    for t in range(T):
        acts = [raw[0][t], raw[1][t]]
        if check_views and first_div is None:
            for s in (0, 1):
                if _engine_view(env, s) != views[s][t]:
                    first_div, divergence = t, f"view:seat{s}"
                    break
        audit, sim_cash = audit_step(env, acts)
        env.step(acts)
        if first_div is None and sim_cash != [float(env.state[0].observation.farms[p]["money"]) for p in (0, 1)]:
            first_div, divergence = t, "audit_cash"
        audits.append(audit)
    literal = first_div is not None
    actions = [[None] * T for _ in (0, 1)]
    history = [[None] * T for _ in (0, 1)]
    masked = [[None] * T for _ in (0, 1)]
    stats = [Counter({k: 0 for k in LABEL_COUNTERS}) for _ in (0, 1)]
    for s in (0, 1):
        st = stats[s]
        for t in range(T):
            obs, a = views[s][t], raw[s][t]
            st["steps"] += 1
            m = raw_market(a)
            st["raw_orders"] += min(len(m), codec.MAX_ORDERS)
            st["raw_orders_truncated"] += max(0, len(m) - codec.MAX_ORDERS)
            if a is None:
                st["masked_none"] += 1
                masked[s][t] = "none"
                history[s][t] = codec.canonical(None, obs)      # the engine executes None as PASS
                continue
            try:
                lit = codec.canonical(a, obs)
            except codec.Unrepresentable:
                st["masked_unrepresentable"] += 1
                masked[s][t] = "unrepresentable"
                history[s][t] = a
                continue
            st["canonical_orders"] += len(lit["market"])
            st["trailing_noop_removed"] += min(len(m), codec.MAX_ORDERS) - len(lit["market"])
            label = None if literal else _strip(lit, a, audits[t][s], st)
            if label is None:            # literal episode, or codec/engine disagreement at this step
                label = lit
                st["kept_orders"] += len(lit["market"])
            actions[s][t] = history[s][t] = label
    return EpisodeLabels(actions, history, masked, literal, first_div, divergence, [dict(x) for x in stats])


def replay_views(replay, steps=STEPS):
    """[seat][t] canonical actor views and [seat][t] submitted actions of a public replay."""
    views = [[canonical_observation(bc.public_observation(replay, t, s)) for t in range(steps)] for s in (0, 1)]
    raw = [[replay["steps"][t + 1][s].get("action") for t in range(steps)] for s in (0, 1)]
    return views, raw


def canonical_labels(replay, *, check_views=True):
    """Per seat per step canonical label action of a complete public replay (EpisodeLabels)."""
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    check_replay(replay)
    views, raw = replay_views(replay)
    env = ExactKaggricultureEnvironment(seed=replay["info"]["seed"], configuration=replay["configuration"])
    return label_episode(env, views, raw, check_views=check_views)


# ---------------------------------------------------------------- per-step blob
_HDR = struct.Struct("<HHB3x")           # N, step, flags(exact|loss_mask<<1), pad to 8 bytes


def _bits(a):
    return np.packbits(np.asarray(a, np.bool_).reshape(-1)).tobytes()


def _labels_arrays(labels):
    """codec labels -> fixed-size int arrays; None -> all -1 (exact False)."""
    unit_tok = np.full(MAX_UNITS, -1, np.int8)
    unit_qty = np.full(MAX_UNITS, -1, np.int8)
    unit_digits = np.full((MAX_UNITS, DIGIT_SLOTS), -1, np.int8)
    order_tok = np.full(N_ORDER_SLOTS, -1, np.int8)
    order_qty = np.full(N_ORDER_SLOTS, -1, np.int8)
    order_digits = np.full((N_ORDER_SLOTS, DIGIT_SLOTS), -1, np.int8)
    exact = False
    if labels is not None:
        unit_tok[:] = labels["unit_tok"]
        unit_qty[:] = labels["unit_qty"]
        unit_digits[:] = labels["unit_digits"]
        k = len(labels["order_tok"])
        order_tok[:k] = labels["order_tok"]
        order_qty[:k] = labels["order_qty"]
        order_digits[:] = labels["order_digits"]
        exact = bool(labels["exact"])
    return unit_tok, unit_qty, unit_digits, order_tok, order_qty, order_digits, exact


def pack_step(ex, step, labels, loss_mask, level=1):
    """One step's inputs + labels -> compressed bytes (see unpack_step)."""
    rows = np.ascontiguousarray(ex["rows"], np.float16)
    n = rows.shape[0]
    ut, uq, ud, ot, oq, od, exact = _labels_arrays(labels)
    parts = [_HDR.pack(n, step, int(exact) | (int(bool(loss_mask)) << 1)),
             rows.tobytes(),
             np.asarray(ex["tile_row"], np.int16).tobytes(),
             np.asarray(ex["feat"]["values"], np.float16).tobytes(),
             np.asarray(ex["feat"]["tile"], np.float16).tobytes(),
             _bits(ex["knownness"]), _bits(ex["token_mask"]), _bits(ex["feat"]["known"]),
             np.asarray(ex["unit_pos"], np.int8).tobytes(), np.asarray(ex["prev_unit_tok"], np.int8).tobytes(),
             ut.tobytes(), uq.tobytes(), ud.tobytes(), ot.tobytes(), oq.tobytes(), od.tobytes()]
    raw = b"".join(parts)
    return zlib.compress(raw, level) if level else raw


_FW, _FK = feature_tokens.WIDTH, feature_tokens.K
_TW = feature_tokens.TILE_WIDTH
_W = features_mod.WIDTH


def unpack_step(blob, meta=None, compressed=True):
    """Inverse of pack_step -> per-step example dict (numpy; int64 indices, bool masks)."""
    buf = zlib.decompress(blob) if compressed else blob
    n, step, flags = _HDR.unpack_from(buf, 0)
    off = _HDR.size

    def take(count, dtype, shape):
        nonlocal off
        a = np.frombuffer(buf, dtype, count, off).reshape(shape)
        off += count * np.dtype(dtype).itemsize
        return a

    def bits(count, shape):
        nonlocal off
        nbytes = (count + 7) // 8
        a = np.unpackbits(np.frombuffer(buf, np.uint8, nbytes, off), count=count).astype(np.bool_).reshape(shape)
        off += nbytes
        return a

    rows = take(n * _W, np.float16, (n, _W))
    tile_row = take(N_CELLS, np.int16, (N_CELLS,)).astype(np.int64)
    fvalues = take(_FK * _FW, np.float16, (_FK, _FW))
    ftile = take(N_CELLS * _TW, np.float16, (N_CELLS, _TW))
    known = bits(n * _W, (n, _W))
    token_mask = bits(n, (n,))
    fknown = bits(_FK * _FW, (_FK, _FW))
    i64 = lambda count, shape: take(count, np.int8, shape).astype(np.int64)  # noqa: E731
    unit_pos = i64(MAX_UNITS * 2, (MAX_UNITS, 2))
    prev = i64(MAX_UNITS, (MAX_UNITS,))
    labels = {"unit_tok": i64(MAX_UNITS, (MAX_UNITS,)), "unit_qty": i64(MAX_UNITS, (MAX_UNITS,)),
              "unit_digits": i64(MAX_UNITS * DIGIT_SLOTS, (MAX_UNITS, DIGIT_SLOTS)),
              "order_tok": i64(N_ORDER_SLOTS, (N_ORDER_SLOTS,)), "order_qty": i64(N_ORDER_SLOTS, (N_ORDER_SLOTS,)),
              "order_digits": i64(N_ORDER_SLOTS * DIGIT_SLOTS, (N_ORDER_SLOTS, DIGIT_SLOTS)),
              "exact": bool(flags & 1)}
    if off != len(buf):
        raise ValueError("step blob layout mismatch")
    step_meta = dict(meta or {})
    step_meta.update(step=step, hour=step % TURNS_PER_DAY, game_day=step // TURNS_PER_DAY)
    return {"rows": rows, "token_mask": token_mask, "knownness": known, "unit_pos": unit_pos,
            "tile_row": tile_row,
            "feat": {"values": fvalues, "known": fknown, "type_id": feature_tokens.TYPE_ID, "tile": ftile},
            "prev_unit_tok": prev, "labels": labels, "loss_mask": bool(flags & 2), "meta": step_meta}


class Perspective:
    """One seat of one episode: meta + one (compressed) blob per step from step 0.

    chunks(W) yields lists of W consecutive per-step example dicts (the last may be
    shorter); every perspective starts at step 0 so a carried state starts fresh."""

    __slots__ = ("meta", "blobs", "compressed")

    def __init__(self, meta, blobs, compressed=True):
        self.meta, self.blobs, self.compressed = meta, blobs, compressed

    def __len__(self):
        return len(self.blobs)

    @property
    def n_steps(self):
        return len(self.blobs)

    @property
    def nbytes(self):
        return sum(len(b) for b in self.blobs)

    def example(self, t):
        return unpack_step(self.blobs[t], self.meta, self.compressed)

    def examples(self, start=0, stop=None):
        return [self.example(t) for t in range(start, len(self.blobs) if stop is None else stop)]

    def chunks(self, W=24, start=0):
        if W < 1:
            raise ValueError("chunk width must be positive")
        for lo in range(start, len(self.blobs), W):
            yield self.examples(lo, min(lo + W, len(self.blobs)))

    def __repr__(self):
        m = self.meta
        return f"Perspective({m.get('episode_id')}, seat={m.get('seat')}, steps={len(self.blobs)})"


def build_perspective(views_s, labels, seat, configuration, meta, *, compress=True, timings=None):
    """Per-step inputs (one FeatureState from step 0) + labels of one seat -> Perspective."""
    fs = FS.FeatureState(configuration)
    prev = None
    blobs = []
    level = 1 if compress else 0
    clock = time.perf_counter
    for t, obs in enumerate(views_s):
        t0 = clock()
        _, ex = step_inputs(obs, configuration, fs, prev)
        t1 = clock()
        action = labels.actions[seat][t]
        hist = labels.history[seat][t]
        why = labels.masked[seat][t]
        # masked None: labels of the executed PASS (loss off); unrepresentable: no labels
        enc = None if why == "unrepresentable" else codec.encode_step(
            None, obs, canonical_action=action if action is not None else hist)
        loss = action is not None and enc is not None and enc["exact"]
        blobs.append(pack_step(ex, t, enc, loss, level))
        fs.record_action(hist)
        prev = enc["unit_tok"] if enc is not None else None
        if timings is not None:
            t2 = clock()
            timings["inputs"] += t1 - t0
            timings["labels_pack"] += t2 - t1
    meta = dict(meta)
    meta["label_stats"] = labels.stats[seat]
    meta["literal_labels"] = labels.literal
    meta["first_divergence"] = labels.first_divergence
    return Perspective(meta, blobs, compress)


# ---------------------------------------------------------------- teacher metadata
class _NoStrength(ST.Strength):
    def __init__(self):
        super().__init__({"schema": ST.SCHEMA, "team_day": {}, "team": {}, "quantiles": [0.0] * 101,
                          "team_vocab": ["<unknown>"], "pretrain": {}})


def load_strength(strength=None, required=True):
    if isinstance(strength, ST.Strength):
        return strength
    path = Path(strength) if strength is not None else ST.DEFAULT_PATH
    if not path.is_file() and not required:
        return _NoStrength()
    return ST.Strength.load(path)


# ---------------------------------------------------------------- v3.1 data rules
def outcome_meta(rewards, seat):
    """won (rewards[seat] strictly greater than the rival's), tie, rewards; None when the
    rewards are missing or not two numbers."""
    if not isinstance(rewards, (list, tuple)) or len(rewards) != 2 or any(
            not isinstance(r, (int, float)) or isinstance(r, bool) for r in rewards):
        return {"won": None, "tie": None, "rewards": None}
    return {"won": rewards[seat] > rewards[1 - seat], "tie": rewards[0] == rewards[1], "rewards": list(rewards)}


def winner_seat(rewards):
    """0 or 1 for a decisive two-number rewards list, else None."""
    won = outcome_meta(rewards, 0)
    if won["won"] is None or won["tie"]:
        return None
    return 0 if won["won"] else 1


def day_ordinal(day):
    """date.toordinal() of an ISO archive date ('2026-09-22'); None for anything else
    (pretraining batch names)."""
    try:
        return datetime.date.fromisoformat(str(day)).toordinal()
    except ValueError:
        return None


class TapeLabels:
    """EPISODE-TAPE-LABELS.jsonl.gz (pbc-tape-corpus/tape_library.py): one row per public
    episode-seat {episode_id, seat, day, team, opponent, won, tie, tape_id, nearest_tape_id,
    match, nearest_prefix, opening24_id, opening72_id}. A seat is static when tape_id is set
    or match >= TAPE_STATIC_MATCH (in the 2026-09-23 file both hold for the same 3,481 rows)."""

    def __init__(self, rows, *, path=None, sha256=None):
        self.labels = {}
        for r in rows:
            key = (int(r["episode_id"]), int(r["seat"]))
            if key in self.labels:
                raise ValueError(f"duplicate tape label for {key}")
            self.labels[key] = (r.get("tape_id"), float(r.get("match") or 0.0))
        self.path, self.sha256 = path, sha256

    @classmethod
    def load(cls, path=TAPE_LABELS):
        path = Path(path)
        raw = path.read_bytes()
        rows = [json.loads(line) for line in gzip.decompress(raw).splitlines() if line.strip()]
        return cls(rows, path=str(path), sha256=hashlib.sha256(raw).hexdigest())

    def __len__(self):
        return len(self.labels)

    def get(self, episode_id, seat):
        """(tape_id, match) or None when the seat is not labelled."""
        try:
            return self.labels.get((int(episode_id), int(seat)))
        except (TypeError, ValueError):
            return None

    def static(self, episode_id, seat):
        """True / False, or None when the seat is not labelled."""
        lab = self.get(episode_id, seat)
        return None if lab is None else (lab[0] is not None or lab[1] >= TAPE_STATIC_MATCH)

    def meta(self, episode_id, seat):
        """Perspective meta fields of a seat and its rival (all None when unlabelled)."""
        out = {}
        for prefix, s in (("", seat), ("opponent_", 1 - seat)):
            lab = self.get(episode_id, s)
            out[prefix + "tape_id"] = None if lab is None else lab[0]
            out[prefix + "tape_match"] = None if lab is None else lab[1]
            out[prefix + "tape_static"] = self.static(episode_id, s)
        return out

    def describe(self):
        return {"path": self.path, "sha256": self.sha256, "rows": len(self),
                "static": sum(1 for k in self.labels if self.static(*k))}


def load_tape_labels(tape_labels=None, *, required=True):
    """TapeLabels from an object, a path or TAPE_LABELS. A missing file raises when required
    (static-tape exclusion is on) and gives None otherwise."""
    if isinstance(tape_labels, TapeLabels):
        return tape_labels
    path = Path(tape_labels) if tape_labels is not None else TAPE_LABELS
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"{path} missing: static-tape exclusion needs EPISODE-TAPE-LABELS; pass "
                "allow_no_tape_labels=True (--allow-no-tape-labels) to run without it")
        return None
    return TapeLabels.load(path)


_TAPE_KEYS = ("tape_id", "tape_match", "tape_static", "opponent_tape_id", "opponent_tape_match",
              "opponent_tape_static")


def rule_meta(rewards, seat, day, episode_id=None, tape_labels=None):
    """Meta fields the data rules and the curriculum read: won/tie/rewards, day_ordinal and
    the tape fields (None without labels)."""
    out = outcome_meta(rewards, seat)
    out["day_ordinal"] = day_ordinal(day)
    tape = tape_labels.meta(episode_id, seat) if tape_labels is not None else dict.fromkeys(_TAPE_KEYS)
    out.update(tape)
    return out


def complete_meta(meta, tape_labels=None):
    """Copy of a perspective meta with the rule fields filled in (older metas lack them);
    the tape fields are re-read from tape_labels when given (Kaggle perspectives only)."""
    m = dict(meta)
    if m.get("won") is None:
        m.update(outcome_meta(m.get("rewards"), int(m["seat"])))
    if m.get("day_ordinal") is None:
        m["day_ordinal"] = day_ordinal(m.get("day"))
    for k in _TAPE_KEYS:
        m.setdefault(k, None)
    if tape_labels is not None and m.get("source", "kaggle") == "kaggle":
        m.update(tape_labels.meta(m["episode_id"], int(m["seat"])))
    return m


def filter_reason(meta, *, wins_only=True, exclude_static_tapes=True, strength_threshold=-math.inf):
    """First data rule that drops a perspective, in chain order: 'no_reward' / 'draw' /
    'loss' (wins_only), 'static_tape', 'strength'; None = kept."""
    if wins_only:
        won = meta.get("won")
        if won is None:
            return "no_reward"
        if not won:
            return "draw" if meta.get("tie") else "loss"
    if exclude_static_tapes and meta.get("tape_static"):
        return "static_tape"
    if meta.get("strength", 0.0) < strength_threshold:
        return "strength"
    return None


FILTER_KEYS = ("seats", "dropped_no_reward", "dropped_draw", "dropped_loss", "dropped_static_tape",
               "dropped_strength", "kept", "tape_unlabelled", "kept_tape_unlabelled")


class FilterRecord(Counter):
    """Per-episode data-rule counts; _kaggle_task/_pretrain_task yield one before their
    perspectives when report_filters=True (iter_perspectives sums them into stats['filters'])."""

    def add(self, meta, reason, labelled=True):
        self["seats"] += 1
        self["kept" if reason is None else f"dropped_{reason}"] += 1
        if not labelled:
            self["tape_unlabelled"] += 1
            if reason is None:
                self["kept_tape_unlabelled"] += 1


# ---------------------------------------------------------------- curriculum
@dataclasses.dataclass(frozen=True)
class Curriculum:
    """Perspective sampling weight as a function of training progress f in [0, 1].

    f < uniform_until: 1 for every perspective.
    uniform_until <= f < final_from: tilt(meta) ** alpha, alpha = (f - uniform_until) /
      (final_from - uniform_until); tilt = exp(-decile_gain * (9 - strength_decile))
      * 2^(-days_back / day_half_life), days_back = last_day - day (undated perspectives,
      i.e. pretraining batches, count as undated_days_back).
    f >= final_from: 1 for the final set (strength_decile >= final_min_decile and
      days_back < final_days), else 0.
    fit(metas) sets last_day to the newest day of the population and, when the final set
    holds fewer than min_final perspectives (default 512 = 2 x the 256-slot pool), widens it
    one step at a time (final_days + 1, then final_min_decile - 1, alternating) and records
    the steps in `relaxed`. pick() falls back when a candidate buffer has no positive weight."""
    last_day: object = None
    uniform_until: float = 0.5
    final_from: float = 0.8
    final_min_decile: int = 8
    final_days: int = 5
    decile_gain: float = math.log(2.0)
    day_half_life: float = 5.0
    undated_days_back: int = 30
    relaxed: tuple = ()

    def __post_init__(self):
        if isinstance(self.last_day, str):
            object.__setattr__(self, "last_day", day_ordinal(self.last_day))
        if not 0.0 <= self.uniform_until < self.final_from <= 1.0:
            raise ValueError("need 0 <= uniform_until < final_from <= 1")

    def days_back(self, meta):
        d = meta.get("day_ordinal")
        d = day_ordinal(meta.get("day")) if d is None else d
        if d is None:
            return None
        if self.last_day is None:
            raise ValueError("Curriculum.last_day is unset: build it with Curriculum.fit(metas) or last_day=")
        return max(0, self.last_day - int(d))

    def in_final(self, meta):
        back = self.days_back(meta)
        return (back is not None and back < self.final_days
                and int(meta.get("strength_decile", 0)) >= self.final_min_decile)

    def log_tilt(self, meta):
        back = self.days_back(meta)
        back = self.undated_days_back if back is None else back
        dec = int(meta.get("strength_decile", 0))
        return -self.decile_gain * (9 - dec) - math.log(2.0) * back / self.day_half_life

    def weight(self, meta, f):
        f = min(1.0, max(0.0, float(f)))
        if f < self.uniform_until:
            return 1.0
        if f >= self.final_from:
            return 1.0 if self.in_final(meta) else 0.0
        alpha = (f - self.uniform_until) / (self.final_from - self.uniform_until)
        return math.exp(alpha * self.log_tilt(meta))

    def weights(self, metas, f):
        return [self.weight(m, f) for m in metas]

    def pick(self, metas, f, rng, k=1):
        """k indices into metas drawn with replacement by weight (rng: random.Random). A
        buffer without any positive weight (no final-set perspective in it) falls back to
        the full-tilt weights, then to uniform."""
        w = self.weights(metas, f)
        if not any(w):
            w = [math.exp(self.log_tilt(m)) for m in metas]
        if not any(w):
            w = None
        return rng.choices(range(len(metas)), weights=w, k=k)

    def distribution(self, metas, f):
        """{(strength_decile, day): normalised weight} over a population of metas."""
        mass = Counter()
        for m in metas:
            w = self.weight(m, f)
            if w > 0:
                mass[(int(m.get("strength_decile", 0)), m.get("day"))] += w
        total = sum(mass.values())
        return {k: v / total for k, v in sorted(mass.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))} if total else {}

    def describe(self):
        d = dataclasses.asdict(self)
        d["last_day"] = (datetime.date.fromordinal(self.last_day).isoformat()
                         if isinstance(self.last_day, int) else self.last_day)
        d["relaxed"] = [list(x) for x in self.relaxed]
        return d

    @classmethod
    def fit(cls, metas, *, min_final=512, **params):
        metas = list(metas)
        days = [d for d in (m.get("day_ordinal") if m.get("day_ordinal") is not None else day_ordinal(m.get("day"))
                            for m in metas) if d is not None]
        cur = cls(**{"last_day": max(days) if days else None, **params})
        if not days:
            return cur
        span = cur.last_day - min(days) + 1
        need = min(min_final, len(metas))
        steps, turn = [], 0
        while sum(cur.in_final(m) for m in metas) < need:
            can_days, can_dec = cur.final_days < span, cur.final_min_decile > 0
            if not (can_days or can_dec):
                break
            if (turn % 2 == 0 and can_days) or not can_dec:
                cur = dataclasses.replace(cur, final_days=cur.final_days + 1)
                steps.append(("final_days", cur.final_days))
            else:
                cur = dataclasses.replace(cur, final_min_decile=cur.final_min_decile - 1)
                steps.append(("final_min_decile", cur.final_min_decile))
            turn += 1
        return dataclasses.replace(cur, relaxed=tuple(steps))


# ---------------------------------------------------------------- v3 sources
def choose_seats(teams, day, strength, strength_top, seats="both"):
    """(kept seats, [(strength, source)] per seat): seats whose team-day strength is at
    least the strength_top threshold (None keeps all). Losers are kept."""
    info = [strength.lookup(t, day) for t in teams]
    thr = strength.threshold(strength_top)
    return [s for s in seat_list(seats) if info[s][0] >= thr], info


def _kaggle_task(task, *, strength, strength_top, seats, compress, wins_only=False, exclude_static_tapes=False,
                 tape_labels=None, report_filters=False, timings=None, **_):
    """Worker body for one public episode: yields Perspectives of the kept seats (preceded
    by one FilterRecord when report_filters). Seats are chosen from the header alone
    (filter_reason: wins_only, static tapes, strength_top) before the replay is read."""
    archive, member, day, eid = task
    header = read_header(archive, member)
    teams = (header.get("info") or {}).get("TeamNames") or [None, None]
    info = [strength.lookup(t, day) for t in teams]
    thr = strength.threshold(strength_top)
    record = FilterRecord()
    kept = []
    for s in seat_list(seats):
        meta = {"source": "kaggle", "episode_id": eid, "seat": s, "day": day, "team": teams[s],
                "team_id": strength.team_id(teams[s]), "strength": info[s][0], "strength_source": info[s][1],
                "strength_decile": strength.decile(info[s][0]), "opponent": teams[1 - s],
                "opponent_strength_decile": strength.decile(info[1 - s][0]),
                **rule_meta(header.get("rewards"), s, day, eid, tape_labels)}
        why = filter_reason(meta, wins_only=wins_only, exclude_static_tapes=exclude_static_tapes,
                            strength_threshold=thr)
        record.add(meta, why, labelled=tape_labels is None or meta["tape_static"] is not None)
        if why is None:
            kept.append(meta)
    if report_filters:
        yield record
    if not kept:
        return
    t0 = time.perf_counter()
    replay = read_replay(archive, member)
    check_replay(replay)
    if replay["info"]["EpisodeId"] != eid:
        raise ValueError("member name / EpisodeId mismatch")
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    cfg = replay["configuration"]
    views, raw = replay_views(replay)
    env = ExactKaggricultureEnvironment(seed=replay["info"]["seed"], configuration=cfg)
    del replay
    t1 = time.perf_counter()
    labels = label_episode(env, views, raw)
    t2 = time.perf_counter()
    if timings is not None:
        timings["parse_views"] += t1 - t0
        timings["audit"] += t2 - t1
    for meta in kept:
        yield build_perspective(views[meta["seat"]], labels, meta["seat"], cfg, meta, compress=compress,
                                timings=timings)


def list_pretrain(directory, *, split="train", include_pass=False):
    """[(path, None, batch_name, key)] of a pretraining batch (RESULTS.jsonl index), split by
    physical game exactly as pbc-pretrain/stream_pretrain.py."""
    out = []
    lo, hi = SPLITS[split]
    for row in ST.pretrain_results(directory):
        if lo <= ST.pretrain_split_bucket(row["path"]) < hi:
            out.append((row["path"], None, Path(directory).resolve().name, ST.pretrain_key(row["path"])))
    return out


def pretrain_configuration(header):
    config = dict(DEFAULT_CONFIGURATION)
    rule = header["rule"]
    if rule["kind"] == "fixed_shop" and not rule.get("random_weeds", False):
        config["weedSpawnChance"] = 0.0
    elif rule["kind"] not in ("original", "fixed_shop"):
        raise ValueError(f"unknown pretraining rule: {rule['kind']}")
    return config


def pretrain_env(header):
    """Fresh exact environment of a pretraining game from its recorded seed and rule."""
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    rule = header["rule"]
    if rule["kind"] == "original":
        return ExactKaggricultureEnvironment(seed=header["seed"])
    from farm_neat.evaluation.scenarios import ScenarioEnvironment, catalog
    spec = dict(catalog()[rule["scenario_id"]])
    if list(spec["shops"]) != list(rule["shops"]) or bool(spec["random_weeds"]) != bool(rule["random_weeds"]):
        raise ValueError("scenario catalog differs from the recorded rule")
    spec["seed"] = header["seed"]
    return ScenarioEnvironment(spec)


def read_pretrain(path):
    """(header, views[s][t] canonical, raw[s][t], terminal) of one pretraining episode file."""
    import gzip
    views, raw = [[], []], [[], []]
    header = terminal = None
    with gzip.open(path, "rb") as stream:
        for line in stream:
            row = _loads(line)
            kind = row.get("kind")
            if kind == "episode":
                header = row
            elif kind == "step":
                s, obs = row["seat"], row["observation"]
                if row["step"] != len(views[s]) or obs.get("player") != s or obs.get("step") != row["step"]:
                    raise ValueError(f"noncontiguous pretraining step in {path}")
                views[s].append(canonical_observation(obs))
                raw[s].append(row["action"])
            elif kind == "terminal":
                terminal = row
    if (header is None or header.get("schema") != "farm-pbc-raw-episode-v1" or terminal is None
            or len(views[0]) != STEPS or len(views[1]) != STEPS):
        raise ValueError(f"incomplete pretraining episode: {path}")
    return header, views, raw, terminal


def _pretrain_task(task, *, strength, pretrain_strength_top, seats, compress, wins_only=False, report_filters=False,
                   timings=None, **_):
    """Worker body for one pretraining game. Rewards = the terminal cash; PASS is never a
    teacher and is not counted; static-tape labels do not cover pretraining games."""
    path, _, batch, key = task
    t0 = time.perf_counter()
    header, views, raw, terminal = read_pretrain(path)
    actors = header["actors"]
    record = FilterRecord()
    kept = []
    for s in seat_list(seats):
        if actors[s] == "PASS":
            continue                      # PASS is never a teacher
        st, dec, tid, q = strength.pretrain_actor(batch, actors[s])
        thr = (ST.Strength._threshold(q, pretrain_strength_top)
               if pretrain_strength_top is not None and q is not None else -math.inf)
        meta = {"source": "pretrain", "episode_id": key, "seat": s, "day": batch,
                "team": f"pretrain:{actors[s]}", "team_id": tid, "strength": st, "strength_source": "pretrain_actor",
                "strength_decile": dec, "opponent": f"pretrain:{actors[1 - s]}",
                "opponent_strength_decile": strength.pretrain_actor(batch, actors[1 - s])[1],
                "rule": header["rule"]["kind"], "outcome": terminal["outcomes"][s],
                **rule_meta(terminal.get("cash"), s, batch)}
        why = filter_reason(meta, wins_only=wins_only, exclude_static_tapes=False, strength_threshold=thr)
        record.add(meta, why)
        if why is None:
            kept.append(meta)
    if report_filters:
        yield record
    if not kept:
        return
    cfg = pretrain_configuration(header)
    env = pretrain_env(header)
    t1 = time.perf_counter()
    labels = label_episode(env, views, raw)
    t2 = time.perf_counter()
    if timings is not None:
        timings["parse_views"] += t1 - t0
        timings["audit"] += t2 - t1
    for meta in kept:
        yield build_perspective(views[meta["seat"]], labels, meta["seat"], cfg, meta, compress=compress,
                                timings=timings)


def _source_task(task, **kwargs):
    return (_pretrain_task if task[1] is None else _kaggle_task)(task, **kwargs)


def perspective_tasks(archive_paths=None, *, split="train", shuffle_seed=None, max_episodes=None):
    """Episode tasks of zip archives (public replays) and pretraining batch directories."""
    tasks = []
    zips = []
    for p in (archive_paths or default_archives()):
        if Path(p).is_dir():
            tasks += list_pretrain(p, split=split)
        else:
            zips.append(p)
    if zips:
        tasks += list_members(zips, split=split)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(tasks)
    return tasks if max_episodes is None else tasks[:max_episodes]


def iter_perspectives(archive_paths=None, *, split="train", strength_top=0.6, seats="both", workers=0,
                      shuffle_seed=None, max_episodes=None, strength=None, pretrain_strength_top=None,
                      repeat=False, compress=True, stats=None, start_method="spawn",
                      wins_only=True, exclude_static_tapes=True, tape_labels=None, allow_no_tape_labels=False):
    """Yield Perspective objects (episode order shuffled by shuffle_seed; with workers the
    completion order also varies). strength_top=None keeps every Kaggle perspective;
    pretraining perspectives keep every non-PASS seat unless pretrain_strength_top is set.
    wins_only (default True): only seats whose final reward is strictly greater than the
    rival's. exclude_static_tapes (default True): drop seats labelled static in tape_labels
    (TapeLabels, a path, or TAPE_LABELS); a missing labels file raises unless
    allow_no_tape_labels (then nothing is excluded and stats['data_rules'] says so). The
    labels are also read, when present, with exclusion off: the tape meta fields are filled.
    repeat=True streams epochs forever (shuffle seed + epoch). stats (dict) receives
    episode errors, perspective count, summed label counters, 'data_rules' and 'filters'
    (FILTER_KEYS: per seat, the first rule that dropped it)."""
    stats = {} if stats is None else stats
    strength = load_strength(strength, required=strength_top is not None)
    tapes = load_tape_labels(tape_labels, required=exclude_static_tapes and not allow_no_tape_labels)
    if exclude_static_tapes and tapes is None:
        print("data.py: static-tape exclusion OFF (tape labels missing, allow_no_tape_labels)", file=sys.stderr)
    kwargs = {"strength": strength, "strength_top": strength_top, "seats": seats, "compress": compress,
              "pretrain_strength_top": pretrain_strength_top, "wins_only": wins_only,
              "exclude_static_tapes": bool(exclude_static_tapes and tapes is not None), "tape_labels": tapes,
              "report_filters": True}
    stats["data_rules"] = {"wins_only": wins_only, "exclude_static_tapes": kwargs["exclude_static_tapes"],
                           "exclude_static_tapes_requested": exclude_static_tapes, "strength_top": strength_top,
                           "pretrain_strength_top": pretrain_strength_top,
                           "tape_labels": tapes.describe() if tapes is not None else None}
    stats.setdefault("filters", Counter())
    stats.setdefault("label_stats", Counter())
    stats.setdefault("perspectives", 0)
    stats.setdefault("literal_episodes", 0)
    epoch = 0
    while True:
        seed = None if shuffle_seed is None else shuffle_seed + epoch
        tasks = perspective_tasks(archive_paths, split=split, shuffle_seed=seed, max_episodes=max_episodes)
        for p in _stream(tasks, workers, kwargs, stats, start_method, fn=_source_task, unit="perspective"):
            if isinstance(p, FilterRecord):
                stats["filters"].update(p)
                continue
            stats["perspectives"] += 1
            stats["label_stats"].update(p.meta["label_stats"])
            stats["literal_episodes"] += bool(p.meta["literal_labels"])
            yield p
        epoch += 1
        if not repeat:
            return


# ---------------------------------------------------------------- collate (v3)
def collate_chunks(chunks, *, rows_dtype=None):
    """List (B) of chunks (lists of per-step examples of one perspective, W_i <= W) ->
    dict of torch tensors [B, W, ...], padded (steps with step_valid False, rows with
    token_mask False, labels -1, loss_mask False)."""
    import torch
    B = len(chunks)
    W = max(len(c) for c in chunks)
    N = max(ex["rows"].shape[0] for c in chunks for ex in c)
    rows = np.zeros((B, W, N, _W), np.float16)
    known = np.zeros((B, W, N, _W), np.bool_)
    mask = np.zeros((B, W, N), np.bool_)
    unit_pos = np.full((B, W, MAX_UNITS, 2), -1, np.int64)
    tile_row = np.full((B, W, N_CELLS), -1, np.int64)
    fvalues = np.zeros((B, W, _FK, _FW), np.float16)
    fknown = np.zeros((B, W, _FK, _FW), np.bool_)
    ftile = np.zeros((B, W, N_CELLS, _TW), np.float16)
    prev = np.full((B, W, MAX_UNITS), -1, np.int64)
    lab = {"unit_tok": np.full((B, W, MAX_UNITS), -1, np.int64),
           "unit_qty": np.full((B, W, MAX_UNITS), -1, np.int64),
           "unit_digits": np.full((B, W, MAX_UNITS, DIGIT_SLOTS), -1, np.int64),
           "order_tok": np.full((B, W, N_ORDER_SLOTS), -1, np.int64),
           "order_qty": np.full((B, W, N_ORDER_SLOTS), -1, np.int64),
           "order_digits": np.full((B, W, N_ORDER_SLOTS, DIGIT_SLOTS), -1, np.int64)}
    exact = np.zeros((B, W), np.bool_)
    loss_mask = np.zeros((B, W), np.bool_)
    valid = np.zeros((B, W), np.bool_)
    step = np.full((B, W), -1, np.int64)
    for b, chunk in enumerate(chunks):
        for w, ex in enumerate(chunk):
            n = ex["rows"].shape[0]
            rows[b, w, :n] = ex["rows"]
            known[b, w, :n] = ex["knownness"]
            mask[b, w, :n] = ex["token_mask"]
            unit_pos[b, w] = ex["unit_pos"]
            tile_row[b, w] = ex["tile_row"]
            fvalues[b, w] = ex["feat"]["values"]
            fknown[b, w] = ex["feat"]["known"]
            ftile[b, w] = ex["feat"]["tile"]
            prev[b, w] = ex["prev_unit_tok"]
            for k in lab:
                lab[k][b, w] = ex["labels"][k]
            exact[b, w] = ex["labels"]["exact"]
            loss_mask[b, w] = ex["loss_mask"]
            valid[b, w] = True
            step[b, w] = ex["meta"]["step"]
    metas = [c[0]["meta"] for c in chunks]
    t = torch.from_numpy
    rows_t = t(rows) if rows_dtype == torch.float16 else t(rows).to(rows_dtype or torch.float32)
    labels = {k: t(v) for k, v in lab.items()}
    labels["exact"] = t(exact)
    hour = np.where(step >= 0, step % TURNS_PER_DAY, -1)
    out = {"rows": rows_t, "knownness": t(known), "token_mask": t(mask), "unit_pos": t(unit_pos),
           "tile_row": t(tile_row),
           "feat": {"values": t(fvalues).float(), "known": t(fknown),
                    "type_id": torch.from_numpy(np.broadcast_to(feature_tokens.TYPE_ID, (B, W, _FK)).copy()),
                    "tile": t(ftile).float()},
           "prev_unit_tok": t(prev), "labels": labels, "loss_mask": t(loss_mask), "step_valid": t(valid),
           "step": t(step), "hour": t(hour),
           "seat": torch.tensor([m["seat"] for m in metas], dtype=torch.long),
           "team_id": torch.tensor([m.get("team_id", 0) for m in metas], dtype=torch.long),
           "strength_decile": torch.tensor([m.get("strength_decile", 0) for m in metas], dtype=torch.long),
           "starts_episode": t(step[:, 0] == 0), "meta": metas}
    out.update(labels)       # flat aliases of the label tensors
    return out


# ---------------------------------------------------------------- v1 per-example API (kept)
def episode_examples(replay, seat, *, meta_base, steps=None):
    """Yield one v1 example per boundary t (default 0..718) for one perspective (literal labels)."""
    cfg = replay["configuration"]
    for t in (range(STEPS) if steps is None else steps):
        obs = canonical_observation(bc.public_observation(replay, t, seat))
        action = replay["steps"][t + 1][seat].get("action")
        ex = features(obs, cfg)
        ex.update(codec.encode_step(action, obs))
        ex["meta"] = {**meta_base, "step": t}
        yield ex


def _episode_task(task, *, rank_max, seats, leaderboard, windows=None):
    """v1 worker body: yields lists of examples (or windows) for one episode."""
    archive, member, day, eid = task
    header = read_header(archive, member)
    teams, ranks = seat_ranks(header, leaderboard)
    chosen = [s for s in seat_list(seats) if keep_seat(ranks[s], rank_max)]
    if not chosen:
        return
    replay = read_replay(archive, member)
    check_replay(replay)
    if replay["info"]["EpisodeId"] != eid:
        raise ValueError("member name / EpisodeId mismatch")
    for seat in chosen:
        meta = {"episode_id": eid, "seat": seat, "team": teams[seat], "rank": ranks[seat],
                "opponent_rank": ranks[1 - seat], "day": day}
        if windows is None:
            chunk = []
            for ex in episode_examples(replay, seat, meta_base=meta):
                chunk.append(ex)
                if len(chunk) == 64:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
        else:
            examples = list(episode_examples(replay, seat, meta_base=meta))
            yield list(make_windows(examples, **windows))


# ---------------------------------------------------------------- streaming
_END = "__end__"


def _worker(task_q, out_q, kwargs, fn, unit):
    while True:
        task = task_q.get()
        if task is None:
            out_q.put((_END, None))
            return
        try:
            for item in fn(task, **kwargs):
                out_q.put(("data", item))
        except Exception as error:  # a bad replay is skipped and reported, never silently
            out_q.put(("error", (task, repr(error))))


def _stream(tasks, workers, kwargs, stats, start_method="spawn", fn=None, unit="chunk"):
    fn = _episode_task if fn is None else fn
    stats.setdefault("episode_errors", [])
    if workers <= 0:
        for task in tasks:
            try:
                for item in fn(task, **kwargs):
                    yield item
            except Exception as error:
                stats["episode_errors"].append((task, repr(error)))
                print(f"data.py: skipped episode {task[3]}: {error!r}", file=sys.stderr)
        return
    ctx = mp.get_context(start_method)
    depth = max(2, 2 * workers) if unit == "perspective" else max(4, 4 * workers)
    task_q, out_q = ctx.Queue(), ctx.Queue(maxsize=depth)
    for task in tasks:
        task_q.put(task)
    for _ in range(workers):
        task_q.put(None)
    procs = [ctx.Process(target=_worker, args=(task_q, out_q, kwargs, fn, unit), daemon=True)
             for _ in range(workers)]
    for p in procs:
        p.start()
    finished = 0
    try:
        while finished < workers:
            kind, value = out_q.get()
            if kind == _END:
                finished += 1
            elif kind == "error":
                stats["episode_errors"].append(value)
                print(f"data.py: skipped episode {value[0][3]}: {value[1]}", file=sys.stderr)
            else:
                yield value
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        # An abandoned stream leaves undelivered tasks in the queue buffers; without
        # this the interpreter blocks at exit flushing them into a pipe nobody reads.
        for q in (task_q, out_q):
            q.cancel_join_thread()
            q.close()


def _tasks(archive_paths, split, shuffle_seed, max_episodes):
    tasks = list_members(archive_paths, split=split)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(tasks)
    return tasks if max_episodes is None else tasks[:max_episodes]


def iter_examples(archive_paths=None, *, split="train", rank_max=None, seats="both", workers=0,
                  shuffle_seed=None, max_episodes=None, leaderboard=None, stats=None,
                  start_method="spawn"):
    """v1: yield per-step examples (literal codec labels, no features). workers=0 runs
    in-process (deterministic order); workers>0 uses that many processes."""
    stats = {} if stats is None else stats
    lb = load_leaderboard() if leaderboard is None else leaderboard
    tasks = _tasks(archive_paths, split, shuffle_seed, max_episodes)
    kwargs = {"rank_max": rank_max, "seats": seats, "leaderboard": lb}
    for chunk in _stream(tasks, workers, kwargs, stats, start_method):
        yield from chunk


def make_windows(examples, *, window=24, burn_in=24, stride=None):
    """v1: contiguous per-perspective windows over one perspective's examples (step order).

    Scored steps [s, s+window) for s = 0, stride, ...; each is preceded by up to
    burn_in no-loss steps [max(0, s-burn_in), s). A window whose burn-in starts at
    step 0 begins from the 'none' field exactly as at inference."""
    stride = window if stride is None else stride
    if window < 1 or burn_in < 0 or stride < 1:
        raise ValueError("positive window/stride and non-negative burn_in required")
    steps = [ex["meta"]["step"] for ex in examples]
    if steps != list(range(steps[0], steps[0] + len(steps))):
        raise ValueError("examples must be one perspective's contiguous steps")
    for start in range(0, len(examples), stride):
        lo = max(0, start - burn_in)
        part = examples[lo:start + window]
        yield {"steps": part, "n_burn": start - lo, "n_scored": len(part) - (start - lo),
               "start_step": examples[start]["meta"]["step"],
               "from_episode_start": examples[lo]["meta"]["step"] == 0,
               "meta": {k: v for k, v in examples[0]["meta"].items() if k != "step"}}


def iter_windows(archive_paths=None, *, split="train", rank_max=None, seats="both", workers=0,
                 window=24, burn_in=24, stride=None, shuffle_seed=None, max_episodes=None,
                 leaderboard=None, stats=None, start_method="spawn"):
    """v1: yield contiguous windows (see make_windows) of one perspective each."""
    stats = {} if stats is None else stats
    lb = load_leaderboard() if leaderboard is None else leaderboard
    tasks = _tasks(archive_paths, split, shuffle_seed, max_episodes)
    kwargs = {"rank_max": rank_max, "seats": seats, "leaderboard": lb,
              "windows": {"window": window, "burn_in": burn_in, "stride": stride}}
    for chunk in _stream(tasks, workers, kwargs, stats, start_method):
        yield from chunk


LABEL_KEYS = ("unit_tok", "unit_qty", "order_tok", "order_qty")


def _pad_orders(values):
    out = np.full(N_ORDER_SLOTS, -1, np.int64)
    out[:len(values)] = values
    return out


def collate(batch):
    """v1: pad entities to the batch's longest roster; torch tensors, rows float32."""
    import torch
    n = max(len(ex["rows"]) for ex in batch)
    b = len(batch)
    rows = np.zeros((b, n, features_mod.WIDTH), np.float32)
    known = np.zeros((b, n, features_mod.WIDTH), np.bool_)
    mask = np.zeros((b, n), np.bool_)
    for i, ex in enumerate(batch):
        k = len(ex["rows"])
        rows[i, :k], known[i, :k], mask[i, :k] = ex["rows"], ex["knownness"], ex["token_mask"]
    return {"rows": torch.from_numpy(rows), "knownness": torch.from_numpy(known),
            "token_mask": torch.from_numpy(mask),
            "unit_pos": torch.from_numpy(np.stack([ex["unit_pos"] for ex in batch])),
            "unit_tok": torch.tensor([ex["unit_tok"] for ex in batch], dtype=torch.long),
            "unit_qty": torch.tensor([ex["unit_qty"] for ex in batch], dtype=torch.long),
            "order_tok": torch.from_numpy(np.stack([_pad_orders(ex["order_tok"]) for ex in batch])),
            "order_qty": torch.from_numpy(np.stack([_pad_orders(ex["order_qty"]) for ex in batch])),
            "exact": torch.tensor([bool(ex["exact"]) for ex in batch]),
            "meta": [ex["meta"] for ex in batch]}


def collate_windows(windows):
    """v1: [B, T] step layout with T = burn_in + window: scored steps always start at
    column burn_in; shorter burn-ins are left-padded, short tails right-padded.
    step_valid marks real steps, loss_mask scored steps, first_valid the column
    where the carried field starts from 'none'."""
    import torch
    burn = max(w["n_burn"] for w in windows)
    width = max(w["n_scored"] for w in windows)
    t_len = burn + width
    flat, where = [], []
    for i, w in enumerate(windows):
        for j, ex in enumerate(w["steps"]):
            flat.append(ex)
            where.append((i, burn - w["n_burn"] + j))
    steps = collate(flat)
    b = len(windows)
    n = steps["rows"].shape[1]
    out = {"rows": torch.zeros(b, t_len, n, features_mod.WIDTH),
           "knownness": torch.zeros(b, t_len, n, features_mod.WIDTH, dtype=torch.bool),
           "token_mask": torch.zeros(b, t_len, n, dtype=torch.bool),
           "unit_pos": torch.full((b, t_len, MAX_UNITS, 2), -1, dtype=torch.long),
           "unit_tok": torch.full((b, t_len, MAX_UNITS), -1, dtype=torch.long),
           "unit_qty": torch.full((b, t_len, MAX_UNITS), -1, dtype=torch.long),
           "order_tok": torch.full((b, t_len, N_ORDER_SLOTS), -1, dtype=torch.long),
           "order_qty": torch.full((b, t_len, N_ORDER_SLOTS), -1, dtype=torch.long),
           "exact": torch.zeros(b, t_len, dtype=torch.bool),
           "step_valid": torch.zeros(b, t_len, dtype=torch.bool),
           "loss_mask": torch.zeros(b, t_len, dtype=torch.bool),
           "first_valid": torch.tensor([burn - w["n_burn"] for w in windows]),
           "from_episode_start": torch.tensor([w["from_episode_start"] for w in windows]),
           "step": torch.full((b, t_len), -1, dtype=torch.long),
           "meta": [w["meta"] for w in windows]}
    ii = torch.tensor([i for i, _ in where])
    cc = torch.tensor([col for _, col in where])
    for key in ("rows", "knownness", "token_mask", "unit_pos", "unit_tok", "unit_qty",
                "order_tok", "order_qty", "exact"):
        out[key][ii, cc] = steps[key]
    out["step_valid"][ii, cc] = True
    out["loss_mask"][ii, cc] = cc >= burn
    out["step"][ii, cc] = torch.tensor([ex["meta"]["step"] for ex in flat])
    return out


# ---------------------------------------------------------------- reports
def _header_task(task):
    archive, member, day, eid = task
    try:
        h = read_header(archive, member)
        return {"episode_id": eid, "day": day, "teams": h.get("info", {}).get("TeamNames"),
                "statuses": h.get("statuses"), "complete": h.get("statuses") == ["DONE", "DONE"]}
    except Exception as error:
        return {"episode_id": eid, "day": day, "error": repr(error)}


def count_examples(archive_paths=None, *, thresholds=(8, 50, 200, None), workers=8, leaderboard=None):
    """v1 header-only count of available perspectives/examples per rank threshold and split."""
    lb = load_leaderboard() if leaderboard is None else leaderboard
    tasks = list_members(archive_paths, split="all")
    with mp.get_context("fork").Pool(workers) as pool:
        headers = pool.map(_header_task, tasks, chunksize=16)
    out = {"episodes_listed": len(tasks), "header_errors": sum("error" in h for h in headers),
           "incomplete_episodes": sum(1 for h in headers if not h.get("complete")),
           "sealed_excluded": sorted(sealed_ids()), "thresholds": {}}
    teams_seen, unmatched = set(), set()
    for h in headers:
        for t in h.get("teams") or []:
            teams_seen.add(t)
            if t not in lb:
                unmatched.add(t)
    out["distinct_teams"], out["teams_not_on_leaderboard"] = len(teams_seen), len(unmatched)
    for thr in thresholds:
        key = "all" if thr is None else f"top{thr}"
        per_split = {s: {"perspectives": 0, "examples": 0, "episodes": 0} for s in ("train", "val", "test")}
        per_day = {}
        for h in headers:
            if not h.get("complete"):
                continue
            ranks = [lb.get(t) for t in h["teams"]]
            k = sum(keep_seat(r, thr) for r in ranks)
            if not k:
                continue
            split = next(s for s in ("train", "val", "test") if in_split(h["episode_id"], s))
            per_split[split]["perspectives"] += k
            per_split[split]["examples"] += k * STEPS
            per_split[split]["episodes"] += 1
            per_day[h["day"]] = per_day.get(h["day"], 0) + k * STEPS
        total = {f: sum(v[f] for v in per_split.values()) for f in ("perspectives", "examples", "episodes")}
        out["thresholds"][key] = {"total": total, "by_split": per_split, "examples_by_day": per_day}
    return out


def benchmark(archive_paths=None, *, workers=(1, 8), examples=4000, split="train", rank_max=None,
              start_method="spawn"):
    """v1 examples/s from a cold start (includes worker start and replay parse)."""
    result = {}
    for w in workers:
        start, n = time.perf_counter(), 0
        for _ in iter_examples(archive_paths, split=split, rank_max=rank_max, workers=w,
                               shuffle_seed=0, start_method=start_method):
            n += 1
            if n >= examples:
                break
        seconds = time.perf_counter() - start
        result[str(w)] = {"examples": n, "seconds": round(seconds, 2),
                          "examples_per_s": round(n / seconds, 1)}
    return result


def _qty_ops(labels, counter):
    for kind, toks, qtys, vocab in (("unit", labels["unit_tok"], labels["unit_qty"], codec.UNIT_VOCAB),
                                    ("order", labels["order_tok"], labels["order_qty"], codec.ORDER_VOCAB)):
        for tok, q in zip(toks, qtys):
            if tok >= 0 and q >= 0:
                op = vocab[tok].split(":")[0]
                counter[f"{op}|{'exact' if q == codec.QTY_EXACT else 'bin'}"] += 1


def audit_episode_labels(task, *, semantic=True):
    """Label audit of one source episode, both seats: canonical labels, codec round trip of
    every label (exact share, decode == label), escape usage per op, and (semantic) an
    exact-engine twin stepped with the decoded labels (raw action where masked) whose full
    state must equal the recorded-action engine's at every step."""
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    t0 = time.perf_counter()
    if task[1] is None:
        header, views, raw, _ = read_pretrain(task[0])
        make_env = lambda: pretrain_env(header)  # noqa: E731
        eid = task[3]
    else:
        replay = read_replay(task[0], task[1])
        check_replay(replay)
        views, raw = replay_views(replay)
        seed, cfg = replay["info"]["seed"], replay["configuration"]
        make_env = lambda: ExactKaggricultureEnvironment(seed=seed, configuration=cfg)  # noqa: E731
        eid = task[3]
        del replay
    t1 = time.perf_counter()
    labels = label_episode(make_env(), views, raw)
    t2 = time.perf_counter()
    rt = Counter()
    esc = Counter()
    decoded = [[None] * len(views[0]) for _ in (0, 1)]
    for s in (0, 1):
        for t, obs in enumerate(views[s]):
            action = labels.actions[s][t]
            if action is None:
                rt["masked"] += 1
                decoded[s][t] = raw[s][t]
                continue
            rt["labels"] += 1
            enc = codec.encode_step(None, obs, canonical_action=action)
            rt["exact"] += enc["exact"]
            out = codec.decode_step(enc, obs)
            rt["decode_equal"] += out == action
            decoded[s][t] = out
            _qty_ops(enc, esc)
    sem = {}
    if semantic:
        a, b = make_env(), make_env()
        first = None
        for t in range(len(views[0])):
            a.step([raw[0][t], raw[1][t]])
            b.step([decoded[0][t], decoded[1][t]])
            if first is None and codec._state_wire(a) != codec._state_wire(b):
                first = t
                break
        sem = {"first_state_difference": first}
    return {"episode_id": eid, "literal": labels.literal, "first_divergence": labels.first_divergence,
            "divergence": labels.divergence, "stats": [dict(x) for x in labels.stats],
            "roundtrip": dict(rt), "qty_classes": dict(esc), **sem,
            "seconds": {"load_views": t1 - t0, "audit": t2 - t1, "total": time.perf_counter() - t0}}


def audit_labels(tasks, *, workers=8, semantic=True):
    """Aggregate audit_episode_labels over tasks (fork pool)."""
    import functools
    fn = functools.partial(audit_episode_labels, semantic=semantic)
    if workers <= 1:
        results = [fn(t) for t in tasks]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(fn, tasks, chunksize=1)
    agg = Counter()
    rt = Counter()
    esc = Counter()
    for r in results:
        for s in (0, 1):
            agg.update(r["stats"][s])
        rt.update(r["roundtrip"])
        esc.update(r["qty_classes"])
    by_op = {}
    for key, n in esc.items():
        op, kind = key.split("|")
        by_op.setdefault(op, {"bin": 0, "exact": 0})[kind] = n
    for op, v in by_op.items():
        v["escape_share"] = v["exact"] / max(1, v["bin"] + v["exact"])
    literal = [r for r in results if r["literal"]]
    sem_diff = [r for r in results if r.get("first_state_difference") is not None]
    return {"episodes": len(results), "perspectives": 2 * len(results),
            "label_steps": rt["labels"], "masked_steps": rt["masked"],
            "exact_labels": rt["exact"], "exact_share": rt["exact"] / max(1, rt["labels"]),
            "decode_equals_label": rt["decode_equal"],
            "escape_by_op": dict(sorted(by_op.items())),
            "literal_episodes": len(literal),
            "divergences": [{"episode_id": r["episode_id"], "step": r["first_divergence"],
                             "kind": r["divergence"]} for r in literal],
            "semantic_checked": semantic,
            "semantic_state_differences": [{"episode_id": r["episode_id"], "step": r["first_state_difference"]}
                                           for r in sem_diff],
            "orders": {k: agg[k] for k in ("raw_orders", "raw_orders_truncated", "trailing_noop_removed",
                                            "canonical_orders", "tail_removed", "kept_orders",
                                            "interior_zero_fill", "interior_zero_fill_noop")},
            "tail_removed_share_of_raw": agg["tail_removed"] / max(1, agg["raw_orders"]),
            "removed_share_of_raw": (agg["tail_removed"] + agg["trailing_noop_removed"]) / max(1, agg["raw_orders"]),
            "sell": {"emptied": agg["sell_emptied"], "relabelled": agg["sell_relabel"]},
            "masked": {"none": agg["masked_none"], "unrepresentable": agg["masked_unrepresentable"]},
            "codec_engine_mismatch": agg["codec_engine_mismatch"],
            "seconds_per_episode": {k: float(np.mean([r["seconds"][k] for r in results]))
                                    for k in ("load_views", "audit", "total")}}


def header_metas(archive_paths=None, *, split="train", workers=6, strength=None, tape_labels=None, headers=None):
    """Header-only perspective metas (both seats of every complete episode of a split) with
    the fields filter_reason and Curriculum read; -> (metas, counts)."""
    strength = load_strength(strength)
    headers = ST.kaggle_headers(archive_paths, workers=workers, split=split) if headers is None else headers
    counts = Counter(headers=len(headers))
    metas = []
    for h in headers:
        if "error" in h:
            counts["header_errors"] += 1
            continue
        teams = h.get("teams")
        if h.get("statuses") != ["DONE", "DONE"] or not teams or len(teams) != 2:
            counts["incomplete"] += 1
            continue
        counts["complete"] += 1
        for s in (0, 1):
            st, src = strength.lookup(teams[s], h["date"])
            metas.append({"source": "kaggle", "episode_id": h["episode_id"], "seat": s, "day": h["date"],
                          "team": teams[s], "strength": st, "strength_source": src,
                          "strength_decile": strength.decile(st),
                          **rule_meta(h.get("rewards"), s, h["date"], h["episode_id"], tape_labels)})
    return metas, counts


def _cell_table(counter):
    return {f"{d}|{day}": n for (d, day), n in sorted(counter.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))}


def data_rules_report(archive_paths=None, *, split="train", workers=6, strength_top=0.6, tape_labels=None,
                      progress=(0.0, 0.5, 0.65, 0.8, 1.0), draws=4096, seed=0, min_final=512, strength=None,
                      headers=None):
    """Header-only measurement of the v3.1 data rules on one split: perspectives before and
    after each rule per archive day (chain order wins_only -> static tapes -> strength_top,
    and each rule alone), tape-label coverage and agreement, and the curriculum's
    distribution over (strength_decile, day) at each progress f: expected weight shares
    and `draws` seeded Curriculum.pick draws (random.Random(seed))."""
    strength = load_strength(strength)
    tapes = load_tape_labels(tape_labels, required=True)
    thr = strength.threshold(strength_top)
    metas, counts = header_metas(archive_paths, split=split, workers=workers, strength=strength,
                                 tape_labels=tapes, headers=headers)
    days = sorted({m["day"] for m in metas})
    rows = {d: Counter() for d in days + ["all"]}
    agree = Counter()
    kept = {"wins_static_strength": [], "wins_static": []}
    for m in metas:
        won, static = m["won"], m["tape_static"]
        c = Counter(perspectives=1)
        c["won"] += won is True
        c["draw"] += bool(m["tie"])
        c["loss"] += won is False and not m["tie"]
        c["no_reward"] += won is None
        c["tape_unlabelled"] += static is None
        c["static"] += bool(static)
        c["strength_ok"] += m["strength"] >= thr
        c["static_among_won"] += bool(static) and won is True
        r1 = won is True
        r2 = r1 and not static
        r3 = r2 and m["strength"] >= thr
        c["after_wins_only"] += r1
        c["after_wins_only_static"] += r2
        c["after_wins_only_static_strength"] += r3
        c["only_static_excluded"] += not static
        c["only_strength"] += m["strength"] >= thr
        for d in (m["day"], "all"):
            rows[d].update(c)
        lab = tapes.labels.get((int(m["episode_id"]), m["seat"]))
        if lab is not None:
            agree["labelled"] += 1
        if r2:
            kept["wins_static"].append(m)
        if r3:
            kept["wins_static_strength"].append(m)
    # the labels file's own 'won' column against the header rewards
    raw = {}
    with gzip.open(tapes.path, "rt") as f:
        for line in f:
            r = json.loads(line)
            raw[(int(r["episode_id"]), int(r["seat"]))] = r
    for m in metas:
        r = raw.get((int(m["episode_id"]), m["seat"]))
        if r is not None:
            agree["won_equal" if bool(r["won"]) == bool(m["won"]) else "won_differs"] += 1
            agree["day_equal" if r["day"] == m["day"] else "day_differs"] += 1
    out = {"schema": "primitive-bc-data-rules-report-v1", "split": split,
           "archives": [str(p) for p in (archive_paths or default_archives())],
           "strength_top": strength_top, "strength_threshold": thr, "tape_labels": tapes.describe(),
           "tape_static_rule": f"tape_id set or match >= {TAPE_STATIC_MATCH}",
           "header_counts": dict(counts), "label_agreement": dict(agree),
           "by_day": {d: dict(rows[d]) for d in days + ["all"]}, "curriculum": {}}
    for name, population in kept.items():
        cur = Curriculum.fit(population, min_final=min_final)
        cells = Counter((m["strength_decile"], m["day"]) for m in population)
        final = [m for m in population if cur.in_final(m)]
        block = {"population": len(population), "curriculum": cur.describe(), "final_set": len(final),
                 "final_set_teams": len({m["team"] for m in final}),
                 "final_set_team_days": len({(m["team"], m["day"]) for m in final}),
                 "population_teams": len({m["team"] for m in population}),
                 "population_cells": _cell_table(cells), "by_f": {}}
        for f in progress:
            w = np.array(cur.weights(population, f), np.float64)
            p = w / w.sum()
            exp_cells, exp_dec, exp_day = Counter(), Counter(), Counter()
            for m, pi in zip(population, p):
                exp_cells[(m["strength_decile"], m["day"])] += pi
                exp_dec[m["strength_decile"]] += pi
                exp_day[m["day"]] += pi
            picks = cur.pick(population, f, random.Random(seed), k=draws)
            got = Counter((population[i]["strength_decile"], population[i]["day"]) for i in picks)
            block["by_f"][str(f)] = {
                "positive_weight": int((w > 0).sum()), "effective_n": float(1.0 / (p ** 2).sum()),
                "expected_by_decile": {str(k): round(v, 5) for k, v in sorted(exp_dec.items())},
                "expected_by_day": {k: round(v, 5) for k, v in sorted(exp_day.items())},
                "expected_cells": {k: round(v, 6) for k, v in _cell_table(exp_cells).items() if v > 0},
                "draws": draws, "seed": seed,
                "drawn_by_decile": {str(d): sum(n for (dd, _), n in got.items() if dd == d)
                                    for d in sorted({d for d, _ in got})},
                "drawn_by_day": {day: sum(n for (_, dd), n in got.items() if dd == day)
                                 for day in sorted({day for _, day in got})},
                "drawn_distinct": len(set(picks)),
                "drawn_cells": _cell_table(got)}
        out["curriculum"][name] = block
    return out


def bench_perspectives(archive_paths=None, *, workers=1, episodes=4, split="all", shuffle_seed=0,
                       strength_top=None, start_method="spawn"):
    """Steps/s of the v3 perspective stream (parse + engine audit + pack + features + labels +
    compression). workers=0: in-process with a per-stage breakdown."""
    stats = {}
    timings = Counter()
    start = time.perf_counter()
    n_steps = n_persp = nbytes = 0
    if workers <= 0:
        strength = load_strength(None, required=False)
        tasks = perspective_tasks(archive_paths, split=split, shuffle_seed=shuffle_seed, max_episodes=episodes)
        for task in tasks:
            for p in _source_task(task, strength=strength, strength_top=strength_top, seats="both",
                                  compress=True, pretrain_strength_top=None, timings=timings):
                n_steps += len(p)
                n_persp += 1
                nbytes += p.nbytes
    else:
        for p in iter_perspectives(archive_paths, split=split, strength_top=strength_top, workers=workers,
                                   shuffle_seed=shuffle_seed, max_episodes=episodes, stats=stats,
                                   start_method=start_method, wins_only=False, exclude_static_tapes=False):
            n_steps += len(p)
            n_persp += 1
            nbytes += p.nbytes
    seconds = time.perf_counter() - start
    out = {"workers": workers, "episodes": episodes, "perspectives": n_persp, "steps": n_steps,
           "seconds": round(seconds, 2), "steps_per_s": round(n_steps / seconds, 1),
           "steps_per_s_per_worker": round(n_steps / seconds / max(1, workers), 1),
           "bytes_per_step": round(nbytes / max(1, n_steps), 1),
           "errors": len(stats.get("episode_errors", []))}
    if timings:
        out["ms_per_step"] = {k: round(1000 * v / max(1, n_steps), 3) for k, v in timings.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--out")
    b = sub.add_parser("bench")
    b.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    b.add_argument("--examples", type=int, default=4000)
    b.add_argument("--rank-max", type=int)
    b.add_argument("--start-method", default="spawn", choices=("spawn", "fork", "forkserver"))
    b.add_argument("--out")
    a = sub.add_parser("audit-labels", help="engine-audited label report on a per-day sample")
    a.add_argument("--per-day", type=int, default=16)
    a.add_argument("--pretrain-files", type=int, default=0)
    a.add_argument("--pretrain-dir", default=str(PRETRAIN_DIR))
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--no-semantic", action="store_true")
    a.add_argument("--out")
    p = sub.add_parser("bench-perspectives")
    p.add_argument("--workers", type=int, nargs="+", default=[0, 1])
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--archives", nargs="*")
    p.add_argument("--out")
    r = sub.add_parser("rules-report", help="header-only counts of the v3.1 data rules + curriculum shares")
    r.add_argument("--split", default="train")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--strength-top", type=float, default=0.6)
    r.add_argument("--tape-labels", default=str(TAPE_LABELS))
    r.add_argument("--progress", type=float, nargs="+", default=[0.0, 0.5, 0.65, 0.8, 1.0])
    r.add_argument("--draws", type=int, default=4096)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--min-final", type=int, default=512)
    r.add_argument("--out")
    args = ap.parse_args(argv)
    workers = args.workers if isinstance(args.workers, list) else [args.workers]
    if max(workers) > 8:
        raise SystemExit("at most 8 worker processes")
    if args.cmd == "count":
        report = count_examples(workers=args.workers)
    elif args.cmd == "bench":
        report = benchmark(workers=tuple(args.workers), examples=args.examples, rank_max=args.rank_max,
                           start_method=args.start_method)
    elif args.cmd == "audit-labels":
        report = {}
        if args.per_day:
            tasks = codec.sample_tasks(args.per_day)
            report["public"] = audit_labels(tasks, workers=args.workers, semantic=not args.no_semantic)
            report["public"]["sample"] = ("per archive day, members sorted by sha256(member name), first "
                                          f"{args.per_day}; all splits; sealed excluded")
        if args.pretrain_files:
            tasks = list_pretrain(args.pretrain_dir, split="all")
            tasks = sorted(tasks, key=lambda t: hashlib.sha256(t[3].encode()).hexdigest())[:args.pretrain_files]
            report["pretrain"] = audit_labels(tasks, workers=args.workers, semantic=not args.no_semantic)
    elif args.cmd == "rules-report":
        report = data_rules_report(split=args.split, workers=args.workers, strength_top=args.strength_top,
                                   tape_labels=args.tape_labels, progress=tuple(args.progress), draws=args.draws,
                                   seed=args.seed, min_final=args.min_final)
    else:
        report = {str(w): bench_perspectives(args.archives or None, workers=w, episodes=args.episodes)
                  for w in workers}
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if getattr(args, "out", None):
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
