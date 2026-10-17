"""Streaming per-step primitive-BC examples from raw public Kaggle replays.

Pairing (bc.from_replay_step): the actor view at boundary t of seat s
(bc.public_observation -> actor-visible whitelist) is labelled with the action
that seat submitted after observing it, replay["steps"][t+1][s]["action"],
for t = 0..718. Both seats of an episode are separate perspectives.

Split: whole episodes, deterministic sha256 bucket of the episode id
(train 90% / val 5% / test 5%). The 16 sealed confirmation episodes
(work/datasets/public-episodes/daily/SELECTION.json, split == "confirmation")
are dropped by member name before any byte of them is read.

Rank filter: a perspective is kept when its team's rank on the public
leaderboard CSV is <= rank_max (rank_max=None keeps every perspective,
including teams missing from the leaderboard).

Seat filters (select_seats, from the episode header before any step is read;
continuation replica 2026-09-24):
  wins_only (default True = the 2026-09-23 wins-only patch): keep a seat only if its
    final reward is strictly greater than the rival's; an episode with a missing
    reward gives no perspective. wins_only=False is the pre-patch behaviour.
  min_decile (default 0 = off): keep a seat only if its team-day Bradley-Terry
    strength decile (lineage A's strength.py + STRENGTH.json, same lookup:
    team-day, else team, else 0.0) is >= min_decile.
  recent_days (default 0 = off): keep only episodes from the last N archive days
    among the archives given (archive day = date in the zip name).

Inputs: observation.pack(actor_obs, None, cfg) rows [N,80] stored as float16,
token_mask [N] bool, knownness [N,80] bool; unit_pos [MAX_UNITS,2] (x, y),
-1 for absent units, farmer first then hands in observation order.
Labels: codec.encode_step (unit_tok, unit_qty, order_tok, order_qty, exact).
features() is the single observation->input function; agent.py must use it so
training and inference see identical float16-rounded inputs.
"""
from __future__ import annotations

import argparse
import csv
import codecs
import glob
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import random
import queue
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import codec  # noqa: E402
import flip  # noqa: E402
import job_eta  # noqa: E402
import calc_features  # noqa: E402
import prefix  # noqa: E402
import noop_mask  # noqa: E402
from farm_neat.policy import observation as features_mod  # noqa: E402
from farm_neat.training import bc  # noqa: E402

ARCHIVE_GLOB = "/mnt/steam/farm-data/kaggle/kaggriculture-episodes-2026-09-*/kaggriculture-episodes-2026-09-*.zip"
LEADERBOARD = Path("/mnt/ai/scratch/farmNEAT/cohort-scale-20260923/leaderboard/kaggriculture.zip")
SELECTION = ROOT / "work/datasets/public-episodes/daily/SELECTION.json"
SPLIT_SALT = b"primitive-bc-split-v1:"
SPLITS = {"train": (0, 900), "val": (900, 950), "test": (950, 1000),
          "heldout": (900, 1000), "all": (0, 1000)}
LAST_STEP = 718          # last boundary with a submitted successor action
STEPS = LAST_STEP + 1    # examples per perspective in a complete episode
N_ORDER_SLOTS = codec.MAX_ORDERS + 1


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
    """TeamName -> best (smallest) public-leaderboard rank. Accepts a CSV or the Kaggle zip."""
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


_DAY = re.compile(r"(20\d\d-\d\d-\d\d)")


def archive_day(path):
    """First YYYY-MM-DD in the archive file name ('unknown' if none). The old rule
    name.split("episodes-")[-1][:10] returned 'kaggricult' for kaggriculture-topteams-* / -top30local-* archives."""
    m = _DAY.search(Path(path).name)
    return m.group(1) if m else "unknown"


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
    # Decode a bounded prefix; an incomplete UTF-8 code point at the end may
    # belong to the step payload and must not invalidate an intact header.
    text = codecs.getincrementaldecoder("utf-8-sig")().decode(head, final=False)
    decoder = json.JSONDecoder()
    pos = len(text) - len(text.lstrip())
    if pos >= len(text) or text[pos] != "{":
        raise ValueError("replay header must be a JSON object")
    pos += 1
    header = {}
    while pos < len(text):
        pos = re.compile(r"\s*").match(text, pos).end()
        key, pos = decoder.raw_decode(text, pos)
        if not isinstance(key, str):
            raise ValueError("replay header keys must be strings")
        pos = re.compile(r"\s*").match(text, pos).end()
        if pos >= len(text) or text[pos] != ":":
            raise ValueError("replay header key missing colon")
        pos += 1
        if key == "steps":
            return header
        pos = re.compile(r"\s*").match(text, pos).end()
        header[key], pos = decoder.raw_decode(text, pos)
        pos = re.compile(r"\s*").match(text, pos).end()
        if pos >= len(text) or text[pos] != ",":
            break
        pos += 1
    raise ValueError("replay header not found in the first bytes")


def seat_ranks(header, leaderboard):
    teams = header.get("info", {}).get("TeamNames") or [None, None]
    return teams, [leaderboard.get(t) if t is not None else None for t in teams]


def keep_seat(rank, rank_max):
    return rank_max is None or (rank is not None and rank <= rank_max)


STRENGTH_PATH = HERE / "STRENGTH.json"


def load_strength(path=STRENGTH_PATH):
    """Lineage A's team-day strength table (strength.Strength); used only when min_decile > 0."""
    import strength
    return strength.Strength.load(path)


def strength_decile(strength, team, day):
    return strength.decile(strength.lookup(team, day)[0])


def select_seats(header, day, *, rank_max, seats, leaderboard, wins_only=True, min_decile=0, strength=None,
                 force_teams=frozenset()):
    """(teams, ranks, kept seats) from an episode header: rank filter, then wins_only, then min_decile.
    A winning seat of a team in force_teams is kept regardless of rank and decile (stopgap for teams missing from the
    strength table, 2026-09-26)."""
    teams, ranks = seat_ranks(header, leaderboard)
    forced = []
    if force_teams:
        rw = header.get("rewards") or [None, None]
        if None not in rw:
            forced = [s for s in seat_list(seats) if teams[s] in force_teams and rw[s] > rw[1 - s]]
    chosen = [s for s in seat_list(seats) if keep_seat(ranks[s], rank_max)]
    if wins_only:
        rewards = header.get("rewards") or [None, None]
        if None in rewards:
            return teams, ranks, []
        chosen = [s for s in chosen if rewards[s] > rewards[1 - s]]  # owner 2026-09-23: BC targets from wins only
    if min_decile:
        if strength is None:
            raise ValueError("min_decile > 0 needs the strength table")
        chosen = [s for s in chosen if strength_decile(strength, teams[s], day) >= min_decile]
    if forced:
        chosen = sorted(set(chosen) | set(forced))
    return teams, ranks, chosen


def seat_list(seats):
    if seats == "both":
        return (0, 1)
    if seats in (0, 1):
        return (seats,)
    raise ValueError("seats must be 'both', 0 or 1")


# ---------------------------------------------------------------- features
def pack_config(configuration):
    cfg = {k: configuration[k] for k in features_mod.CONFIG if k in (configuration or {})}
    if configuration and "marketParams" in configuration:
        cfg["marketParams"] = {item: dict(p) for item, p in configuration["marketParams"].items()}
    return cfg


def canonical_inventories(actor_obs):
    """Carrier inventories in alphabetical item order. Saved replays store them sorted, so training always saw that
    order; applying it at play keeps the inputs identical whatever order the live observation arrives in."""
    private = actor_obs.get("private")
    if isinstance(private, dict) and isinstance(private.get("inventories"), list):
        private["inventories"] = [dict(sorted(inv.items())) if isinstance(inv, dict) else inv
                                  for inv in private["inventories"]]
    return actor_obs


def canonical_inventories_v7(actor_obs):
    """Canonical token view, retaining raw inventory order for prefix execution."""
    private = actor_obs.get("private")
    if isinstance(private, dict) and isinstance(private.get("inventories"), list):
        inventories = [dict(sorted(inv.items())) if isinstance(inv, dict) else inv
                       for inv in private["inventories"]]
        return {**actor_obs, "private": {**private, "inventories": inventories}}
    return actor_obs


CALC_WIDTH = calc_features.N_PROD + calc_features.N_GLOB     # v4: appended after flip.WIDTH (v3 models never read them)
EXTRA_WIDTH = flip.WIDTH + CALC_WIDTH


def calc_columns(actor_obs, configuration, token_ids, calculator=calc_features):
    """[N, CALC_WIDTH] float16: the calc_features product row on its MARKET token (first N_PROD columns), the global row on
    the GLOBAL token (last N_GLOB columns); zeros elsewhere."""
    c = calculator.compute(actor_obs, configuration)
    out = np.zeros((len(token_ids), CALC_WIDTH), np.float16)
    row_of = {p: i for i, p in enumerate(calc_features.PRODUCTS)}
    for n, tid in enumerate(token_ids):
        if tid == "global":
            out[n, calc_features.N_PROD:] = c["calc_glob"]
        elif tid.startswith("market/") and tid[7:] in row_of:
            out[n, :calc_features.N_PROD] = c["calc_prod"][row_of[tid[7:]]]
    return out


def features(actor_obs, configuration, *, feature_schema="legacy", v8_history=None, v8_long=None):
    """Model inputs from an actor-visible observation (bc.actor_observation output with step). v8_history (the seat's
    previous actor observations, most recent first, <= 2) adds the v8 families (v8.py)."""
    if feature_schema == "legacy":
        # V8 builds a canonical token VIEW while preserving the raw actor snapshot
        # for execution-order-sensitive labels/prefixes. Legacy non-V8 is unchanged.
        actor_obs = (canonical_inventories_v7(actor_obs) if v8_history is not None
                     else canonical_inventories(actor_obs))
        flip_module, job_module, calculator = flip, job_eta, calc_features
    elif feature_schema == "v7":
        import flip_v7
        import job_eta_v7
        import calc_features_v7
        actor_obs = canonical_inventories_v7(actor_obs)
        flip_module, job_module, calculator = flip_v7, job_eta_v7, calc_features_v7
    else:
        raise ValueError("feature_schema must be legacy|v7")
    packed = features_mod.pack(actor_obs, None, pack_config(configuration))
    pos = np.full((codec.MAX_UNITS, 2), -1, np.int64)
    units = codec.unit_positions(actor_obs)[:codec.MAX_UNITS]
    if units:
        pos[:len(units)] = np.asarray(units, np.int64)
    extra, unit_extra, reach_eta = flip_module.token_features(actor_obs, packed["token_ids"], codec.MAX_UNITS, configuration)
    extra = np.concatenate([extra.astype(np.float16), calc_columns(actor_obs, configuration, packed["token_ids"], calculator)], axis=1)
    jobs = job_module.extract(actor_obs, packed["token_ids"], codec.MAX_UNITS)
    return {"rows": packed["rows"].astype(np.float16), "token_mask": packed["token_mask"],
            "knownness": packed["knownness"], "unit_pos": pos, "extra": extra, "unit_extra": unit_extra,
            "reach_eta": reach_eta, "v6_token": jobs["token"],
            "v6_resource_eta": jobs["resource_eta"], "v6_resource_route": jobs["resource_route"],
            "v6_job_deadline": jobs["job_deadline"],
            "v6_job_deadline_present": jobs["job_deadline_present"],
            **(__import__("v8").compute(actor_obs, configuration, packed["token_ids"], v8_history, long_state=v8_long)
               if v8_history is not None else {})}


def check_replay(replay):
    if (replay.get("name") != "kaggriculture" or not isinstance(replay.get("steps"), list)
            or len(replay["steps"]) != 720 or replay.get("statuses") != ["DONE", "DONE"]):
        raise ValueError("complete two-seat Kaggriculture replay required")
    cfg = replay.get("configuration", {})
    if any(cfg.get(k) != v for k, v in features_mod.CONFIG.items()):
        raise ValueError("replay mechanics configuration differs from the fixed schema")


def _scrub_action(action):
    """The action with every unit/order that codec.canonical_* rejects replaced by PASS / NOOP."""
    a = dict(action) if isinstance(action, dict) else {}

    def unit(u):
        try:
            codec.canonical_unit(u)
            return u
        except codec.Unrepresentable:
            return ["PASS"]

    def order(o):
        try:
            codec.canonical_order(o)
            return o
        except codec.Unrepresentable:
            return ["NOOP"]
    a["farmer"] = unit(a.get("farmer", ["PASS"]))
    a["hands"] = [unit(h) for h in a["hands"]] if isinstance(a.get("hands"), list) else []
    a["market"] = [order(o) for o in a["market"]] if isinstance(a.get("market"), list) else []
    return a


def encode_step_lenient(action, obs):
    """codec.encode_step; an Unrepresentable part (e.g. PLACE SHEEP 0) becomes PASS/NOOP with exact=False
    instead of aborting the whole episode task (the production worker skipped every such episode)."""
    try:
        return codec.encode_step(action, obs)
    except codec.Unrepresentable:
        lab = codec.encode_step(_scrub_action(action), obs)
        lab["exact"] = False
        return lab


def episode_examples(replay, seat, *, meta_base, steps=None, feature_schema="legacy", v8=False):
    """Yield one example per boundary t (default 0..718) for one perspective. v8: add the v8 families, which read the
    seat's previous two actor observations (as the agent keeps them)."""
    cfg = replay["configuration"]
    seen = {}
    gl = __import__("v8_gaps").GapsState() if v8 else None      # v8 long columns: one running state per seat
    wanted = None if steps is None else set(steps)
    order = range(STEPS) if steps is None else (range(max(steps) + 1) if v8 and steps else steps)
    for t in order:
        if wanted is not None and t not in wanted:           # v8 only: the running state must see every earlier step
            o = bc.public_observation(replay, t, seat)
            __import__("v8_gaps").raw_long(canonical_inventories_v7(o), cfg, gl)
            seen = {t: o}
            continue
        obs = seen.pop(t, None) or bc.public_observation(replay, t, seat)
        action = replay["steps"][t + 1][seat].get("action")
        hist = None
        if v8:
            hist = [seen.get(t - k) or bc.public_observation(replay, t - k, seat) for k in (1, 2) if t - k >= 0]
            seen = {t: obs, t - 1: hist[0] if hist else None}
            seen = {k: v for k, v in seen.items() if v is not None}
        ex = features(obs, cfg, feature_schema=feature_schema, v8_history=hist, v8_long=gl)
        ex.update(encode_step_lenient(action, obs))
        wire = codec.decode_step({k: ex[k] for k in LABEL_KEYS}, obs)      # what the token prefix means at play
        ex["unit_loss_keep"], _ = noop_mask.teacher_supervision_mask(
            obs, action, wire, max_units=codec.MAX_UNITS, engine=prefix.E, configuration=cfg)
        ex["unit_noop_mask_schema"] = noop_mask.SCHEMA
        ex["unit_prefix"], ex["order_prefix"], ex["order_cash"] = prefix.teacher_arrays(
            obs, wire, codec.MAX_UNITS, N_ORDER_SLOTS, configuration=cfg, include_cash=True)
        ex["meta"] = {**meta_base, "step": t}
        yield ex


# ---------------------------------------------------------------- v6 (owner 2026-09-27; pbc-v6-design/DESIGN.md)
V6_LAGS = (1, 2)          # previous own turns fed as input, in sequence (t-1, t-2)
V6_AHEAD = (1, 2)         # own future turns predicted by training-only heads
V6_OPP = (0, 1, 2)        # opponent turns predicted by training-only heads
V6_KEYS = ("prev_unit", "prev_order", "next_unit_tok", "opp_unit_tok",
           "next_unit_qty", "next_order_tok", "next_order_qty", "opp_unit_qty", "opp_order_tok", "opp_order_qty")
OPP_LABEL_KEYS = ("unit_tok", "unit_qty", "order_tok", "order_qty")
OPP_LABEL_MASKED = __import__("collections").Counter()   # reason -> opponent steps whose labels could not be encoded


def _same_hand(k, t_a, t_b):
    """Worker slot k names the same worker at steps t_a and t_b: the farmer always; a hand only within one day (hands are
    dismissed at dusk, new hires append in order)."""
    return k == 0 or t_a // 24 == t_b // 24


def _orders_row(lab, O):
    tok, qty = np.full(O, -1, np.int64), np.full(O, -1, np.int64)
    ot, oq = list(lab["order_tok"]), list(lab["order_qty"])
    if len(ot) != len(oq) or not 1 <= len(ot) <= O:
        raise ValueError("order labels must have equal token/quantity lengths in 1..O")
    if ot[-1] != codec.STOP or codec.STOP in ot[:-1]:
        raise ValueError("order labels require exactly one terminal STOP")
    tok[:len(ot)], qty[:len(oq)] = ot, oq
    return tok, qty


def add_v6(examples, opp_unit_tok=None):
    """In place, for one perspective's contiguous examples (step order): the previous own turns (V6_LAGS) as codec tokens in
    their original order (worker slot for commands, lockstep slot for orders), the own worker commands (token, quantity) and
    orders at t+1, t+2 and, when given, the opponent's worker commands and orders at t, t+1, t+2 (training-only targets).
    opp_unit_tok[j]: the opponent's codec labels at examples[j] (dict with OPP_LABEL_KEYS; a bare unit_tok array gives
    worker tokens only), or None. -1 = no such step, worker, order or quantity."""
    n = len(examples)
    U, O = codec.MAX_UNITS, N_ORDER_SLOTS
    steps = [ex["meta"]["step"] for ex in examples]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("example steps must be unique and strictly increasing")
    for key in ("episode_id", "seat"):
        if len({ex["meta"].get(key) for ex in examples}) > 1:
            raise ValueError(f"add_v6 cannot mix {key} across steps")
    by_step = {t: i for i, t in enumerate(steps)}
    lab = None
    if opp_unit_tok is not None:
        if len(opp_unit_tok) != n:
            raise ValueError("opponent labels must align one-for-one with examples")
        # Convert once per seat, not once per step (previously O(n**2)).
        lab = [o if (o is None or isinstance(o, dict)) else {"unit_tok": o} for o in opp_unit_tok]
    for i, ex in enumerate(examples):
        t = steps[i]
        pu = np.full((len(V6_LAGS), U, 2), -1, np.int64)
        po = np.full((len(V6_LAGS), O, 2), -1, np.int64)
        for li, lag in enumerate(V6_LAGS):
            j = by_step.get(t - lag)
            if j is None:                         # preserve available t-2 even when t-1 is missing
                continue
            prev = examples[j]
            for k in range(U):
                if _same_hand(k, steps[j], t):
                    pu[li, k] = (prev["unit_tok"][k], prev["unit_qty"][k])
            ot, oq = list(prev["order_tok"])[:O], list(prev["order_qty"])[:O]
            po[li, :len(ot), 0] = ot
            po[li, :len(oq), 1] = oq
        nu = np.full((len(V6_AHEAD), U), -1, np.int64)
        nq = np.full((len(V6_AHEAD), U), -1, np.int64)
        no = np.full((len(V6_AHEAD), O), -1, np.int64)
        noq = np.full((len(V6_AHEAD), O), -1, np.int64)
        for li, a in enumerate(V6_AHEAD):
            j = by_step.get(t + a)
            if j is not None:
                for k in range(U):
                    if ex["unit_tok"][k] >= 0 and _same_hand(k, t, steps[j]):
                        nu[li, k] = examples[j]["unit_tok"][k]
                        if nu[li, k] >= 0:
                            nq[li, k] = examples[j]["unit_qty"][k]
                no[li], noq[li] = _orders_row(examples[j], O)
        ex["prev_unit"], ex["prev_order"], ex["next_unit_tok"] = pu, po, nu
        ex["next_unit_qty"], ex["next_order_tok"], ex["next_order_qty"] = nq, no, noq
        ou = np.full((len(V6_OPP), U), -1, np.int64)
        oq = np.full((len(V6_OPP), U), -1, np.int64)
        oo = np.full((len(V6_OPP), O), -1, np.int64)
        ooq = np.full((len(V6_OPP), O), -1, np.int64)
        if lab is not None:
            for li, a in enumerate(V6_OPP):
                j = by_step.get(t + a)
                if j is not None and lab[j] is not None:
                    for k in range(U):
                        if (lab[i] is not None and lab[i]["unit_tok"][k] >= 0
                                and _same_hand(k, t, steps[j])):
                            ou[li, k] = lab[j]["unit_tok"][k]
                            if ou[li, k] >= 0 and "unit_qty" in lab[j]:
                                oq[li, k] = lab[j]["unit_qty"][k]
                    if "order_tok" in lab[j]:
                        oo[li], ooq[li] = _orders_row(lab[j], O)
        ex["opp_unit_tok"], ex["opp_unit_qty"], ex["opp_order_tok"], ex["opp_order_qty"] = ou, oq, oo, ooq
    return examples


def opponent_unit_tokens(replay, seat, steps):
    """Codec labels (OPP_LABEL_KEYS) of the opponent's recorded turn at each step, from the opponent's own observation."""
    out = []
    for t in steps:
        try:
            obs = bc.public_observation(replay, t, 1 - seat)
            lab = encode_step_lenient(replay["steps"][t + 1][1 - seat].get("action"), obs)
            out.append({k: lab[k] for k in OPP_LABEL_KEYS})
        except Exception as e:  # noqa: BLE001 - masked, never a crash; counted so a silent label loss is visible
            OPP_LABEL_MASKED[type(e).__name__] += 1
            out.append(None)
    return out


def tape_examples(tape_path, seat, *, meta_base, min_final_quadrants=0, v6=False, feature_schema="legacy", v8=False):
    """Per-step examples of one seat from a compact tape (seed + 719 joint actions: self-league games, hosted or league games
    stored as tapes), re-simulated in the exact engine. Inputs are the serve-time actor view (bc.actor_observation of the
    engine observation; features() sorts inventories); labels, lenient encoding and prefix arrays as in episode_examples.
    Returns [] if the seat ends with fewer quadrants than asked."""
    import gzip
    from farm_neat.game.exact_engine import ExactKaggricultureEnvironment
    tape = json.load(gzip.open(tape_path, "rt"))
    env = ExactKaggricultureEnvironment(seed=int(tape["seed"]))
    cfg = dict(env.configuration)
    out, opp, history = [], [], []
    gl = __import__("v8_gaps").GapsState() if v8 else None
    for t, acts in enumerate(tape["actions"][:STEPS]):
        obs = bc.actor_observation(json.loads(json.dumps(env.agent_observation(seat))))
        if v6:
            try:
                o2 = bc.actor_observation(json.loads(json.dumps(env.agent_observation(1 - seat))))
                if o2.get("step") is None:
                    o2["step"] = t
                lab = encode_step_lenient(acts[1 - seat], o2)
                opp.append({k: lab[k] for k in OPP_LABEL_KEYS})
            except Exception as e:  # noqa: BLE001 - masked and counted (see OPP_LABEL_MASKED)
                OPP_LABEL_MASKED[type(e).__name__] += 1
                opp.append(None)
        if obs.get("step") is None:
            obs["step"] = t
        ex = features(obs, cfg, feature_schema=feature_schema, v8_history=history if v8 else None, v8_long=gl)
        if v8:
            history = [obs, *history[:1]]  # actor snapshots, not labels; newest first
        ex.update(encode_step_lenient(acts[seat], obs))
        wire = codec.decode_step({k: ex[k] for k in LABEL_KEYS}, obs)
        ex["unit_loss_keep"], _ = noop_mask.teacher_supervision_mask(
            obs, acts[seat], wire, max_units=codec.MAX_UNITS, engine=prefix.E, configuration=cfg)
        ex["unit_noop_mask_schema"] = noop_mask.SCHEMA
        ex["unit_prefix"], ex["order_prefix"], ex["order_cash"] = prefix.teacher_arrays(
            obs, wire, codec.MAX_UNITS, N_ORDER_SLOTS, configuration=cfg, include_cash=True)
        ex["meta"] = {**meta_base, "step": t}
        out.append(ex)
        env.step(acts)
    if min_final_quadrants and len(env.state[0].observation.farms[seat].get("unlocked_quadrants") or []) < min_final_quadrants:
        return []
    return add_v6(out, opp) if v6 else out


OUR_TEAM = "Russell Kirk"


def beat_seats(header):
    """Seats kept from a beat game (owner 2026-09-25: games a strong policy lost): the winner, including our own policy's wins
    (owner 2026-09-26 ~00:00: "train on our own POLICY wins against other top 30 too"); not our games against ourselves."""
    teams = header.get("info", {}).get("TeamNames") or [None, None]
    rewards = header.get("rewards") or [None, None]
    if None in rewards or teams[0] == teams[1] == OUR_TEAM:
        return []
    return [s for s in (0, 1) if rewards[s] > rewards[1 - s]]


WORTH_FILE = HERE / "WORTH.tsv.gz"     # built by pbc-filter-review-agent/worth.py; one row per winner seat
WORTH_TEAM_CAP = 0.10                  # no team above this share of the sampled mass


def load_worth(theta, path=None, team_cap=WORTH_TEAM_CAP):
    """[(archive, member, day, eid, seat, weight)]: rows with W >= theta (zip copies only), weight = W after
    scaling any team above team_cap of the total mass down to the cap (water-filling)."""
    rows = []
    with gzip.open(WORTH_FILE if path is None else path, "rt", encoding="utf-8") as f:
        cols = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(cols, line.rstrip("\n").split("\t")))
            w = float(r["W"])
            if w <= 0 or w < theta or not r["archive"].endswith(".zip"):
                continue
            rows.append([r["archive"], r["member"], r["day"], int(r["eid"]), int(r["seat"]), r["team"], w])
    for _ in range(50):
        total = sum(r[6] for r in rows)
        mass = {}
        for r in rows:
            mass[r[5]] = mass.get(r[5], 0.0) + r[6]
        over = {t: m for t, m in mass.items() if m > team_cap * total * 1.0001}
        if not over or team_cap * len(over) >= 1:
            break
        target = team_cap * (total - sum(over.values())) / (1 - team_cap * len(over))
        for r in rows:
            if r[5] in over:
                r[6] *= target / over[r[5]]
    return [(a, m, d, e, s, w) for a, m, d, e, s, t, w in rows]


def _worth_tasks(rows, split, shuffle_seed, max_episodes):
    """One task per row, drawn with probability weight / max weight (a fresh draw per epoch seed)."""
    sealed = sealed_ids()
    rng = random.Random(shuffle_seed)
    top = max(r[5] for r in rows)
    tasks = [(a, m, d, e, (s,)) for a, m, d, e, s, w in rows
             if e not in sealed and in_split(e, split) and rng.random() < w / top]
    rng.shuffle(tasks)
    return tasks if max_episodes is None else tasks[:max_episodes]


def _episode_task(task, *, rank_max, seats, leaderboard, windows=None, wins_only=True, min_decile=0,
                  strength=None, beat_archives=frozenset(), beat_ids=frozenset(), force_teams=frozenset(),
                  min_final_quadrants=0, v6=False, feature_schema="legacy", v8=False):
    """Worker body: yields lists of examples (or windows) for one episode. member "tape": a compact tape from a priority
    row (seat given). min_final_quadrants: keep only seats that own at least that many quadrants at the end."""
    archive, member, day, eid = task[:4]
    if member == "tape":
        for seat in task[4]:
            meta = {"episode_id": eid, "seat": seat, "team": None, "rank": None, "opponent_rank": None, "day": day}
            examples = tape_examples(archive, seat, meta_base=meta, min_final_quadrants=min_final_quadrants,
                                     v6=v6, feature_schema=feature_schema, v8=v8)
            if not examples:
                continue
            if windows is None:
                for i in range(0, len(examples), 64):
                    yield examples[i:i + 64]
            else:
                yield list(make_windows(examples, **windows))
        return
    # Explicit priority/worth rows carry an eid independently of the ZIP member.
    # Check their agreement BEFORE read_header opens any replay bytes: the task
    # planners applied the split/sealed filters to eid, not to this member.
    name = Path(member).name
    if not (name.endswith(".json") and name[:-5].isdigit() and int(name[:-5]) == int(eid)):
        raise ValueError(f"member name / task episode_id mismatch: {member!r} vs {eid}")
    header = read_header(archive, member)
    if len(task) > 4:                       # worth task: the seat was chosen by the worth file
        teams, ranks = seat_ranks(header, leaderboard)
        chosen = list(task[4])
    elif archive in beat_archives or eid in beat_ids:
        teams, ranks = seat_ranks(header, leaderboard)
        chosen = beat_seats(header)
    else:
        teams, ranks, chosen = select_seats(header, day, rank_max=rank_max, seats=seats, leaderboard=leaderboard,
                                            wins_only=wins_only, min_decile=min_decile, strength=strength,
                                            force_teams=force_teams)
    if not chosen:
        return
    with zipfile.ZipFile(archive) as z:
        replay = json.loads(z.read(member))
    check_replay(replay)
    if replay["info"]["EpisodeId"] != eid:
        raise ValueError("member name / EpisodeId mismatch")
    if min_final_quadrants:
        final = replay["steps"][-1][0]["observation"]["farms"]
        chosen = [s for s in chosen if len(final[s].get("unlocked_quadrants") or []) >= min_final_quadrants]
    for seat in chosen:
        meta = {"episode_id": eid, "seat": seat, "team": teams[seat], "rank": ranks[seat],
                "opponent_rank": ranks[1 - seat], "day": day}
        examples = list(episode_examples(replay, seat, meta_base=meta, feature_schema=feature_schema, v8=v8))
        if v6:
            add_v6(examples, opponent_unit_tokens(replay, seat, [ex["meta"]["step"] for ex in examples]))
        if windows is None:
            for i in range(0, len(examples), 64):
                yield examples[i:i + 64]
        else:
            yield list(make_windows(examples, **windows))


# ---------------------------------------------------------------- streaming
_END = "__end__"


def _worker(task_q, out_q, kwargs):
    while True:
        task = task_q.get()
        if task is None:
            out_q.put((_END, None))
            return
        try:
            for chunk in _episode_task(task, **kwargs):
                out_q.put(("data", chunk))
        except Exception as error:  # a bad replay is skipped and reported, never silently
            out_q.put(("error", (task, repr(error))))


def _stream(tasks, workers, kwargs, stats, start_method="spawn"):
    stats.setdefault("episode_errors", [])
    if workers <= 0:
        for task in tasks:
            try:
                for chunk in _episode_task(task, **kwargs):
                    yield chunk
            except Exception as error:
                stats["episode_errors"].append((task, repr(error)))
        return
    ctx = mp.get_context(start_method)
    task_q, out_q = ctx.Queue(), ctx.Queue(maxsize=max(4, 4 * workers))
    for task in tasks:
        task_q.put(task)
    for _ in range(workers):
        task_q.put(None)
    procs = [ctx.Process(target=_worker, args=(task_q, out_q, kwargs), daemon=True) for _ in range(workers)]
    for p in procs:
        p.start()
    finished = 0
    try:
        while finished < workers:
            try:
                kind, value = out_q.get(timeout=0.5)
            except queue.Empty:
                # A killed/OOM/native-crashing child cannot enqueue _END. Also
                # detect a clean exit that bypassed the completion protocol.
                failed = [(p.pid, p.exitcode) for p in procs if p.exitcode not in (None, 0)]
                if failed or all(p.exitcode is not None for p in procs):
                    raise RuntimeError(f"data worker exited without completion: failures={failed}; "
                                       f"completed={finished}/{workers}")
                continue
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


def recent_day_set(tasks, recent_days):
    """The last ``recent_days`` distinct archive days among the tasks (all days when 0)."""
    days = sorted({t[2] for t in tasks})
    return set(days[-recent_days:] if recent_days else days)


def _tasks(archive_paths, split, shuffle_seed, max_episodes, recent_days=0, beat_archives=(), beat_ids=frozenset(),
           beat_lookup=(), priority_rows=(), priority_only=False):
    tasks = [] if priority_only else list_members(archive_paths, split=split)
    if recent_days:
        keep = recent_day_set(tasks, recent_days)
        tasks = [t for t in tasks if t[2] in keep or t[3] in beat_ids]
    if beat_ids and beat_lookup:   # beat episodes found only in lookup archives (those archives add nothing else)
        have = {t[3] for t in tasks}
        seen = set()
        for t in list_members(list(beat_lookup), split=split):
            if t[3] in beat_ids and t[3] not in have and t[3] not in seen:
                seen.add(t[3])
                tasks.append(t)
    if beat_archives:   # no recent_days / rank / decile filter; a beat copy replaces the same episode elsewhere
        beat = list_members(list(beat_archives), split=split)
        ids = {t[3] for t in beat}
        tasks = [t for t in tasks if t[3] not in ids] + beat
    sealed = sealed_ids() if priority_rows else frozenset()   # the 16 confirmation episodes stay unseen here too
    for archive, member, day, eid, seat, weight in priority_rows:   # explicit (episode, seat) rows repeated `weight` times,
        if in_split(int(eid), split) and int(eid) not in sealed:     # no recent/rank/decile/wins filter (seat is chosen)
            tasks += [(archive, member, day, int(eid), (int(seat),))] * int(weight)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(tasks)
    return tasks if max_episodes is None else tasks[:max_episodes]


def _filter_kwargs(wins_only, min_decile, strength):
    if min_decile and strength is None:
        strength = load_strength()
    return {"wins_only": wins_only, "min_decile": min_decile, "strength": strength if min_decile else None}


def iter_examples(archive_paths=None, *, split="train", rank_max=None, seats="both", workers=0,
                  shuffle_seed=None, max_episodes=None, leaderboard=None, stats=None,
                  start_method="spawn", wins_only=True, recent_days=0, min_decile=0, strength=None,
                  feature_schema="legacy", v8=False):
    """Yield per-step examples. workers=0 runs in-process (deterministic order);
    workers>0 uses that many processes with a bounded queue (episode order then
    depends on timing; the example set does not). start_method "spawn" (default)
    is safe under a multi-threaded (torch) parent; "fork" starts faster."""
    stats = {} if stats is None else stats
    lb = load_leaderboard() if leaderboard is None else leaderboard
    tasks = _tasks(archive_paths, split, shuffle_seed, max_episodes, recent_days)
    kwargs = {"rank_max": rank_max, "seats": seats, "leaderboard": lb, "feature_schema": feature_schema, "v8": v8,
              **_filter_kwargs(wins_only, min_decile, strength)}
    for chunk in _stream(tasks, workers, kwargs, stats, start_method):
        yield from chunk


def make_windows(examples, *, window=24, burn_in=24, stride=None):
    """Contiguous per-perspective windows over one perspective's examples (step order).

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
                 leaderboard=None, stats=None, start_method="spawn", wins_only=True, recent_days=0,
                 min_decile=0, strength=None, beat_archives=(), beat_ids=(), beat_lookup=(), force_teams=(),
                 worth_min=0.0, priority_rows=(), priority_only=False, min_final_quadrants=0, v6=False,
                 feature_schema="legacy", v8=False):
    """Yield contiguous windows (see make_windows) of one perspective each. worth_min > 0: tasks come from
    WORTH_FILE (rows with W >= worth_min, drawn in proportion to W, team-capped); archives, recent_days,
    rank_max, wins_only, min_decile and the beat/force lists are then not used."""
    stats = {} if stats is None else stats
    lb = load_leaderboard() if leaderboard is None else leaderboard
    beat_archives = tuple(str(p) for p in beat_archives)
    beat_ids = frozenset(int(e) for e in beat_ids)
    if worth_min > 0:
        tasks = _worth_tasks(load_worth(worth_min), split, shuffle_seed, max_episodes)
    else:
        tasks = _tasks(archive_paths, split, shuffle_seed, max_episodes, recent_days, beat_archives, beat_ids,
                       tuple(str(p) for p in beat_lookup), tuple(priority_rows), priority_only)
    kwargs = {"rank_max": rank_max, "seats": seats, "leaderboard": lb, "feature_schema": feature_schema, "v8": v8,
              "windows": {"window": window, "burn_in": burn_in, "stride": stride},
              **_filter_kwargs(wins_only, min_decile, strength)}
    if beat_archives:
        kwargs["beat_archives"] = frozenset(beat_archives)
    if beat_ids:
        kwargs["beat_ids"] = beat_ids
    if force_teams:
        kwargs["force_teams"] = frozenset(force_teams)
    if min_final_quadrants:
        kwargs["min_final_quadrants"] = int(min_final_quadrants)
    if v6:
        kwargs["v6"] = True
    for chunk in _stream(tasks, workers, kwargs, stats, start_method):
        yield from chunk


# ---------------------------------------------------------------- collate
LABEL_KEYS = ("unit_tok", "unit_qty", "order_tok", "order_qty")


def _pad_orders(values):
    out = np.full(N_ORDER_SLOTS, -1, np.int64)
    out[:len(values)] = values
    return out


def _collate_noop_masks(batch):
    import torch
    if not any("unit_loss_keep" in ex for ex in batch):
        return {}
    if any("unit_loss_keep" not in ex or ex.get("unit_noop_mask_schema") != noop_mask.SCHEMA for ex in batch):
        raise ValueError("mixed or stale unit no-op mask examples")
    return {"unit_loss_keep": torch.tensor([ex["unit_loss_keep"] for ex in batch], dtype=torch.bool),
            "unit_noop_mask_schema": noop_mask.SCHEMA}


def collate(batch):
    """Pad entities to the batch's longest roster; torch tensors, rows float32."""
    import torch
    if not batch:
        raise ValueError("cannot collate an empty batch")
    v8_keys = ("x8_token", "x8_unit", "lattice")
    schemas = {tuple(k for k in v8_keys if k in ex) for ex in batch}
    if len(schemas) != 1 or (next(iter(schemas)) and
                            not {"x8_token", "x8_unit"} <= set(next(iter(schemas)))):
        raise ValueError("mixed or incomplete v8 schemas in one batch")
    n = max(len(ex["rows"]) for ex in batch)
    b = len(batch)
    rows = np.zeros((b, n, features_mod.WIDTH), np.float32)
    known = np.zeros((b, n, features_mod.WIDTH), np.bool_)
    mask = np.zeros((b, n), np.bool_)
    extra = np.zeros((b, n, EXTRA_WIDTH), np.float32)
    v6_token = np.zeros((b, n, job_eta.TOKEN_WIDTH), np.float32)
    for i, ex in enumerate(batch):
        k = len(ex["rows"])
        rows[i, :k], known[i, :k], mask[i, :k] = ex["rows"], ex["knownness"], ex["token_mask"]
        extra[i, :k] = ex["extra"]
        v6_token[i, :k] = ex["v6_token"]
    return {**_collate_noop_masks(batch), "rows": torch.from_numpy(rows), "knownness": torch.from_numpy(known),
            "token_mask": torch.from_numpy(mask), "extra": torch.from_numpy(extra),
            "v6_token": torch.from_numpy(v6_token),
            "v6_resource_eta": torch.from_numpy(np.stack([ex["v6_resource_eta"] for ex in batch]).astype(np.int64)),
            "v6_resource_route": torch.from_numpy(np.stack([ex["v6_resource_route"] for ex in batch])),
            "v6_job_deadline": torch.from_numpy(np.stack([ex["v6_job_deadline"] for ex in batch]).astype(np.int64)),
            "v6_job_deadline_present": torch.from_numpy(np.stack([ex["v6_job_deadline_present"] for ex in batch])),
            "unit_extra": torch.from_numpy(np.stack([ex["unit_extra"] for ex in batch]).astype(np.float32)),
            "reach_eta": torch.from_numpy(np.stack([ex["reach_eta"] for ex in batch]).astype(np.int64)),
            **({"unit_prefix": torch.from_numpy(np.stack([ex["unit_prefix"] for ex in batch]).astype(np.float32)),
                "order_prefix": torch.from_numpy(np.stack([ex["order_prefix"] for ex in batch]).astype(np.float32)),
                "order_cash": torch.from_numpy(np.stack([ex["order_cash"] for ex in batch]).astype(np.float32))}
               if "unit_prefix" in batch[0] else {}),
            "unit_pos": torch.from_numpy(np.stack([ex["unit_pos"] for ex in batch])),
            "unit_tok": torch.tensor([ex["unit_tok"] for ex in batch], dtype=torch.long),
            "unit_qty": torch.tensor([ex["unit_qty"] for ex in batch], dtype=torch.long),
            "order_tok": torch.from_numpy(np.stack([_pad_orders(ex["order_tok"]) for ex in batch])),
            "order_qty": torch.from_numpy(np.stack([_pad_orders(ex["order_qty"]) for ex in batch])),
            "exact": torch.tensor([bool(ex["exact"]) for ex in batch]),
            **({k: torch.from_numpy(np.stack([ex[k] for ex in batch]).astype(np.int64)) for k in V6_KEYS}
               if "prev_unit" in batch[0] else {}),
            **(_collate_v8(batch, n) if "x8_token" in batch[0] else {}),
            "meta": [ex["meta"] for ex in batch]}


def _collate_v8(batch, n):
    import torch
    if not batch:
        raise ValueError("cannot collate an empty v8 batch")
    first = batch[0]
    keys = {k for k in ("x8_token", "x8_unit", "lattice") if k in first}
    if not {"x8_token", "x8_unit"} <= keys:
        raise ValueError("v8 requires both x8_token and x8_unit")
    token_shape = first["x8_token"].shape
    unit_shape = first["x8_unit"].shape
    if len(token_shape) != 2 or len(unit_shape) != 2 or unit_shape[0] != 25:
        raise ValueError("invalid v8 token/unit rank or unit count")
    for ex in batch:
        if {k for k in ("x8_token", "x8_unit", "lattice") if k in ex} != keys:
            raise ValueError("mixed v8 schemas in one batch")
        shape = ex["x8_token"].shape
        if len(shape) != 2 or shape[1] != token_shape[1] or shape[0] > n:
            raise ValueError("v8 token width/rank/padding mismatch")
        if ex["x8_unit"].shape != unit_shape:
            raise ValueError("v8 unit shape mismatch")
        if "lattice" in keys and (ex["lattice"].shape != first["lattice"].shape
                                  or len(ex["lattice"].shape) != 4
                                  or ex["lattice"].shape[:3] != (2, 10, 10)):
            raise ValueError("v8 lattice shape mismatch")
    tok = np.zeros((len(batch), n, token_shape[1]), np.float32)
    for i, ex in enumerate(batch):
        tok[i, :len(ex["x8_token"])] = ex["x8_token"]
    out = {"x8_token": torch.from_numpy(tok),
           "x8_unit": torch.from_numpy(np.stack([ex["x8_unit"] for ex in batch]).astype(np.float32))}
    if "lattice" in keys:
        out["lattice"] = torch.from_numpy(np.stack([ex["lattice"] for ex in batch]).astype(np.float32))
    return out


def collate_windows(windows):
    """[B, T] step layout with T = burn_in + window: scored steps always start at
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
           "extra": torch.zeros(b, t_len, n, EXTRA_WIDTH),
           "v6_token": torch.zeros(b, t_len, n, job_eta.TOKEN_WIDTH),
           "v6_resource_eta": torch.zeros(b, t_len, codec.MAX_UNITS, 100, 2, dtype=torch.long),
           "v6_resource_route": torch.zeros(b, t_len, codec.MAX_UNITS, 100, 2, dtype=torch.bool),
           "v6_job_deadline": torch.zeros(b, t_len, 100, job_eta.N_JOBS, dtype=torch.long),
           "v6_job_deadline_present": torch.zeros(b, t_len, 100, job_eta.N_JOBS, dtype=torch.bool),
           "unit_extra": torch.zeros(b, t_len, codec.MAX_UNITS, flip.UNIT_WIDTH),
           "reach_eta": torch.zeros(b, t_len, codec.MAX_UNITS, 100, dtype=torch.long),
           "unit_prefix": torch.zeros(b, t_len, codec.MAX_UNITS, prefix.F_UP),
           "order_prefix": torch.zeros(b, t_len, N_ORDER_SLOTS, prefix.F_OP),
           "order_cash": torch.zeros(b, t_len, N_ORDER_SLOTS, prefix.F_CASH),
           "unit_pos": torch.full((b, t_len, codec.MAX_UNITS, 2), -1, dtype=torch.long),
           "unit_tok": torch.full((b, t_len, codec.MAX_UNITS), -1, dtype=torch.long),
           "unit_qty": torch.full((b, t_len, codec.MAX_UNITS), -1, dtype=torch.long),
           "order_tok": torch.full((b, t_len, N_ORDER_SLOTS), -1, dtype=torch.long),
           "order_qty": torch.full((b, t_len, N_ORDER_SLOTS), -1, dtype=torch.long),
           "exact": torch.zeros(b, t_len, dtype=torch.bool),
           "step_valid": torch.zeros(b, t_len, dtype=torch.bool),
           "loss_mask": torch.zeros(b, t_len, dtype=torch.bool),
           "first_valid": torch.tensor([burn - w["n_burn"] for w in windows]),
           "from_episode_start": torch.tensor([w["from_episode_start"] for w in windows]),
           "step": torch.full((b, t_len), -1, dtype=torch.long),
           "meta": [w["meta"] for w in windows]}
    if "prev_unit" in steps:
        for k in V6_KEYS:
            out[k] = torch.full((b, t_len, *steps[k].shape[1:]), -1, dtype=torch.long)
    V8 = tuple(k for k in ("x8_token", "x8_unit", "lattice") if k in steps)
    for k in V8:
        out[k] = torch.zeros((b, t_len, *steps[k].shape[1:]))
    ii = torch.tensor([i for i, _ in where])
    cc = torch.tensor([col for _, col in where])
    for key in ("rows", "knownness", "token_mask", "extra", "v6_token", "v6_resource_eta", "v6_resource_route",
                "v6_job_deadline", "v6_job_deadline_present", "unit_extra", "reach_eta", "unit_prefix", "order_prefix", "order_cash",
                "unit_pos", "unit_tok", "unit_qty",
                "order_tok", "order_qty", "exact", *(V6_KEYS if "prev_unit" in steps else ()), *V8):
        out[key][ii, cc] = steps[key]
    if "unit_loss_keep" in steps:
        out["unit_loss_keep"] = torch.zeros(b, t_len, codec.MAX_UNITS, dtype=torch.bool)
        out["unit_loss_keep"][ii, cc] = steps["unit_loss_keep"]
        out["unit_noop_mask_schema"] = steps["unit_noop_mask_schema"]
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
    """Header-only count of available perspectives/examples per rank threshold and split.
    Each complete (DONE, DONE) perspective contributes 719 labelled steps."""
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
    """Examples/s from a cold start (includes worker start and replay parse)."""
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
    args = ap.parse_args(argv)
    if args.workers and max(args.workers if isinstance(args.workers, list) else [args.workers]) > 8:
        print("note: more than 8 processes requested", file=sys.stderr)
    if args.cmd == "count":
        report = count_examples(workers=args.workers)
    else:
        report = benchmark(workers=tuple(args.workers), examples=args.examples, rank_max=args.rank_max,
                           start_method=args.start_method)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
