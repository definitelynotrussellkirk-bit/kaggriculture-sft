"""Per-seat model input features: theorem_features + history_features, one versioned schema.

Usage, once per engine step for ONE seat (actor view = ``training.bc.actor_observation`` with
``step``, i.e. what ``bc.public_observation`` / ``agent.actor_view`` return)::

    fs = FeatureState(configuration)       # one per seat; SeatFeatures dispatches two seats
    fs.update(actor_obs)                   # ingest this step (history is causal)
    feats = fs.compute()                   # dict of normalized float32 arrays (+ knownness)
    action = policy(pack_rows, feature_tokens.tokens(feats))
    fs.record_action(action)               # the wire action actually submitted this step

Dataset builders call ``episode_features(observations, actions, configuration)``, which runs the
same code path; features for any step need every earlier step of that seat's episode (lags up
to 72 steps, cumulative market flow since step 0), so compute steps 0..718 in order, then slice.

``compute()`` blocks (all float32, divided by a fixed per-column scale, never clipped):

  theorem.global     (46,)      theorem_features global      (all known)
  theorem.product    (9, 43)    rows PRODUCTS                (all known)
  theorem.structure  (8, 18)    rows PRODUCERS               (all known)
  theorem.tile       (200, 31)  row = role*100 + y*10 + x; role 0 own, 1 rival (all known)
  history.global_now (1, 27)    history_features blocks, with knownness
  history.global_lag (4, 22)    rows lag 1/4/24/72
  history.prev_action(1, 24)
  history.item_now   (12, 18)   rows ITEMS = PRODUCTS + ANIMALS
  history.item_lag   (12, 36)

``feats["known"][block]`` is the bool knownness of each block (False = unknown, stored 0).
No rival private state, seed, future shop or outcome is read by either source module.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

import history_features as HF
import theorem_features as TF

FEATURE_SCHEMA = "primitive-bc-features-v1"

_HIST = {"history.global_now": "history_global_now", "history.global_lag": "history_global_lag",
         "history.prev_action": "history_prev_action", "history.item_now": "history_item_now",
         "history.item_lag": "history_item_lag"}
_TILE_ROWS = tuple(f"own:{x},{y}" for y in range(10) for x in range(10)) + \
             tuple(f"rival:{x},{y}" for y in range(10) for x in range(10))

# block -> (row labels, column names, column scales)
BLOCKS = {
    "theorem.global": (("global",), TF.NAMES["global"], TF.SCALES["global"]),
    "theorem.product": (TF.ROWS["product"], TF.NAMES["product"], TF.SCALES["product"]),
    "theorem.structure": (TF.ROWS["structure"], TF.NAMES["structure"], TF.SCALES["structure"]),
    "theorem.tile": (_TILE_ROWS, TF.NAMES["tile_own"], TF.SCALES["tile_own"]),
    **{name: (tuple(HF.BLOCKS[src][0]), tuple(f for f, _ in HF.BLOCKS[src][1]),
              np.asarray([s for _, s in HF.BLOCKS[src][1]], np.float32))
       for name, src in _HIST.items()},
}
SHAPES = {name: (len(cols),) if name == "theorem.global" else (len(rows), len(cols))
          for name, (rows, cols, _) in BLOCKS.items()}
THEOREM_BLOCKS = ("theorem.global", "theorem.product", "theorem.structure", "theorem.tile")
HISTORY_BLOCKS = tuple(_HIST)
N_VALUES = sum(int(np.prod(s)) for s in SHAPES.values())

SCHEMA_PAYLOAD = {
    "schema": FEATURE_SCHEMA,
    "blocks": {name: {"rows": list(rows), "columns": list(cols), "scales": [float(s) for s in scales]}
               for name, (rows, cols, scales) in BLOCKS.items()},
    "theorem_engine_sha256": TF.ENGINE_SHA256,
    "history_schema": HF.SCHEMA, "history_schema_sha256": HF.SCHEMA_SHA256,
    "normalization": "DIVIDE_THEN_FP32_WITHOUT_CLIPPING",
    "knownness": "FALSE_MEANS_UNKNOWN_AND_STORED_ZERO",
    "tile_row": "role*100+y*10+x; role 0 = observing seat",
}
FEATURE_SCHEMA_SHA256 = hashlib.sha256(json.dumps(SCHEMA_PAYLOAD, sort_keys=True).encode()).hexdigest()


def flat_names():
    """'block/row/column' for every value, in BLOCKS order, row-major."""
    return tuple(f"{name}/{r}/{c}" for name, (rows, cols, _) in BLOCKS.items() for r in rows for c in cols)


def _config(configuration):
    if configuration is None:
        return None
    return configuration if hasattr(configuration, "get") else dict(configuration)


class FeatureState:
    """Causal feature state for one seat of one episode stream.

    A step 0 observation, or any step that does not follow the last one, restarts the
    history (history_features' own reset rules; ``history.global_now`` hist.* fields say so).
    """

    def __init__(self, configuration=None):
        self.configuration = _config(configuration)
        self.history = HF.HistoryFeatures()
        self._obs = None
        self._hist_out = None
        self._cache = None

    @property
    def seat(self):
        return self.history.seat

    def update(self, actor_obs, configuration=None):
        """Ingest one step of this seat's actor observation (must carry ``step``)."""
        if configuration is not None:
            self.configuration = _config(configuration)
        self._hist_out = self.history.update(actor_obs, self.configuration)
        self._obs = actor_obs
        self._cache = None
        return self

    def record_action(self, action):
        """The exact wire action this seat submitted after the last ``update``."""
        self.history.record_action(action)

    def theorem_raw(self):
        """Unscaled theorem_features.compute() output for the current step."""
        if self._obs is None:
            raise ValueError("update() must precede compute()")
        return TF.compute(self._obs, self.configuration)

    def compute(self):
        """Dict: ``schema``, ``step``, one normalized float32 array per BLOCKS entry,
        ``known`` (block -> bool array) and ``prev_tokens`` (codec labels of the own
        action at step-1, or None). Cached until the next ``update``."""
        if self._cache is not None:
            return self._cache
        raw = self.theorem_raw()
        out = {"schema": FEATURE_SCHEMA, "step": int(self._hist_out.step)}
        known = {}
        sc = TF.SCALES
        out["theorem.global"] = raw["global"] / sc["global"]
        out["theorem.product"] = raw["product"] / sc["product"]
        out["theorem.structure"] = raw["structure"] / sc["structure"]
        out["theorem.tile"] = np.concatenate((raw["tile_own"], raw["tile_rival"])) / sc["tile_own"]
        for name in THEOREM_BLOCKS:
            known[name] = np.ones(out[name].shape, np.bool_)
        blocks = self._hist_out.blocks
        for name, src in _HIST.items():
            _, _, values, k = blocks[src]
            out[name] = values
            known[name] = k
        out["known"] = known
        out["prev_tokens"] = self._hist_out.prev_tokens
        self._cache = out
        return out

    def step(self, actor_obs, configuration=None):
        """update() then compute()."""
        return self.update(actor_obs, configuration).compute()


class SeatFeatures:
    """One callable playing both seats keeps one FeatureState per seat."""

    def __init__(self, configuration=None):
        self.configuration = configuration
        self.by_seat = {}

    def state(self, seat):
        return self.by_seat.setdefault(int(seat), FeatureState(self.configuration))

    def step(self, actor_obs, configuration=None):
        return self.state(actor_obs["player"]).step(actor_obs, configuration)

    def record_action(self, seat, action):
        self.by_seat[int(seat)].record_action(action)


def episode_features(observations, actions=None, configuration=None):
    """Offline helper: one seat's observations 0..T in order (and the actions it submitted
    after each) -> list of compute() dicts, via the same path as live play."""
    state = FeatureState(configuration)
    out = []
    for t, obs in enumerate(observations):
        out.append(state.step(obs))
        if actions is not None and t < len(actions) and actions[t] is not None:
            state.record_action(actions[t])
    return out


def flat(features):
    """(values f32[N_VALUES], known bool[N_VALUES]) in flat_names() order."""
    values = [np.asarray(features[n], np.float32).reshape(-1) for n in BLOCKS]
    known = [np.asarray(features["known"][n]).reshape(-1) for n in BLOCKS]
    return np.concatenate(values), np.concatenate(known)


def warm(configuration=None):
    """One-time costs before the first timed step: theorem price tables (~40 ms) and the
    lazy codec import history_features uses for prev_tokens."""
    TF.warm(configuration)
    HF._codec()
