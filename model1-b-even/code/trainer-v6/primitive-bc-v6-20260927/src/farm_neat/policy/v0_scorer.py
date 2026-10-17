"""Fresh V0 legal-candidate actor and canonical-prefix sample/replay reference.

Public interface: ``V0Scorer.score(state, prefix_plan, legal_ids, phi)`` returns
one raw logit per supplied legal ID (including STOP). ``score_batch`` and
``score_padded`` run the shared transformer once per context batch, NOT once
per action. A Plan means the start of an epoch; use Prefix to retain key/cap
state. Supplied training supports/phi must be exporter-verified. build_menu stays
the unchanged reference; decode_epoch/replay_epoch use the exact fast backend by
default. backend="reference" selects rollback without changing trace semantics.

Only hybrid.Block/norm architecture is reused: no legacy Graph, GRU history,
option head, economic prior, value loss, checkpoint or optimizer is imported.
All actor fields are explicitly listed below. This is a lossy bootstrap
representation, NOT a sufficient-state theorem or a playing-strength result.
FP32 reference; CPU fixtures may use diagnostic=True. Production defaults are
16 layers / 60M-class. No persistent hidden cache and no implicit RNG seed.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from farm_neat.economy.v0 import vocabulary as V
from farm_neat.economy.v0.plan import Plan
from farm_neat.economy.v0.projection import DEMAND_SPEC, Masks, PHI_NAMES, State, phi as project_phi
from farm_neat.economy.heuristic.market import sell_revenue
from farm_neat.game.constants import ANIMALS, CROPS as CROP_PARAMS, LAND_PRICES, MARKET_PARAMS, hire_cost
from farm_neat.policy import qwen_hybrid as hybrid

SCHEMA = 'farm-v0-sparse-canonical-scorer-v1'
TRACE_SCHEMA = 'farm-v0-canonical-class-path-v1'
STEP_SCHEMA = 'farm-v0-canonical-step-v1'
FEATURE_WIDTH = 40
SHAPES = ('linear', 'sq', 'sqrt', 'log', 'log10', 'hinge')
CROPS = tuple(V.CROPS)
TOKEN_PRODUCT = 1
TOKEN_SEED = TOKEN_PRODUCT + len(V.PRODUCTS)
TOKEN_LAND = TOKEN_SEED + len(CROPS)
TOKEN_MODULE = TOKEN_LAND + 1
N_TOKENS = TOKEN_MODULE + V.N_MODULES
CASH_PERIODS = (300., 700., 1500., 3100., 6300.)
CASH_COLUMN, CASH_PERIODIC_COLUMN = 1, 8
FEATURE_SPEC = {
    'schema': 'farm-v0-aggregate-actor-input-v2',
    'width': FEATURE_WIDTH,
    'order': 'GLOBAL,PRODUCTS,SEEDS,LAND,SORTED_PRESENT_MODULES; no truncation',
    'global': ('remaining_dawns', 'own_cash', 'quadrants', 'hands_today',
               'hires_today', 'occupied_sites', 'last_key_plus_one', 'edit_count'),
    'product': ('own_aggregate_stock', 'market_inventory', 'base', 'I0', 'T',
                'below_target', 'above_target',
                *('below_' + s for s in SHAPES), *('above_' + s for s in SHAPES),
                'next_demand_distance', 'next_reveal_distance', 'next_dawn_distance',
                'terminal_distance', 'known_shop_demand_now', 'known_shop_demand_next_dawn',
                'overlay_present', 'primitive', 'horizon', 'rank',
                'hold_memory_known', 'hold_binding_present', 'hold_primitive',
                'hold_horizon', 'hold_release_distance'),
    'seed': ('own_seed_count',),
    'land': ('authorization_present', *V.LAND_TIMINGS),
    'module': ('count', 'explicit_regime', *V.REGIMES, 'birth_distance_days', 'animal'),
    'values': ('numeric values sign(x)*log1p(abs(x))/10 except the declared '
               'unit-scale cash_periodic coordinates; no fitted statistics or clipping'),
    'cash_periodic': {
        'source': 'own_cash',
        'periods': CASH_PERIODS,
        'columns': tuple(range(CASH_PERIODIC_COLUMN,
                               CASH_PERIODIC_COLUMN + 2 * len(CASH_PERIODS))),
        'transform': 'sin/cos(2*pi*own_cash/P), written on the GLOBAL token at unit scale',
    },
    'missing': 'zero storage plus separate false knownness; padding excluded before projection',
    'phi_names': PHI_NAMES,
    'phi_transform': 'same signed log1p / 10; exact input rows retained in menus',
    'memory_allowlist': ('hold_releases',),
    'excluded': ('tile coordinates', 'worker identities', 'absolute turn tokens',
                 'rival-private information', 'hidden future', 'teacher targets', 'outcomes'),
    'limits': ('State already aggregates custody and defaults some missing facts; no new sufficiency claim',
               'no public rival economic features or biological tile details in this bootstrap encoder'),
}


# V2 remains the default and retains its exact declared schema. V3 is a fresh,
# explicitly selected actor; neither model identity nor weights are rebound.
FEATURE_WIDTH_V3 = 64
TOKEN_RIVAL = N_TOKENS
TOKEN_RIVAL_PRODUCT = TOKEN_RIVAL + 1
N_TOKENS_V3 = TOKEN_RIVAL_PRODUCT + len(V.PRODUCTS)
CASH_SCALES = (100., 1000., 10000., 100000.)
CASH_MONOTONE_COLUMN = 8
CASH_THRESHOLDS = tuple(sorted({c["seed"] for c in CROP_PARAMS.values()}
                              | {a["cost"] for a in ANIMALS.values()} | set(LAND_PRICES)))
FEATURE_SPEC_V3 = {
    **{k: v for k, v in FEATURE_SPEC.items() if k not in ("cash_periodic", "limits")},
    "schema": "farm-v0-aggregate-actor-input-v3", "width": FEATURE_WIDTH_V3,
    "order": FEATURE_SPEC["order"] + ";RIVAL_PUBLIC,RIVAL_PRODUCTS (relative other seat, no ID)",
    "global": (*FEATURE_SPEC["global"], *(f"cash_scale_{int(s)}" for s in CASH_SCALES),
               *(f"afford_{p}" for p in CASH_THRESHOLDS), "afford_next_hire", "afford_next_land"),
    "product": (*FEATURE_SPEC["product"], "target_inventory_gap", "unsatisfied_demand",
                "total_demand_created", "created_early", "created_mid", "created_late",
                "known_remaining_near", "known_remaining_mid", "known_remaining_late",
                "expected_unrevealed_near", "expected_unrevealed_mid", "expected_unrevealed_late",
                "remaining_reveals", "sell_1_revenue", "sell_10_revenue", "sell_100_revenue",
                "sell_own_stock_revenue"),
    "rival": ("cash", "quadrants", "hands_today", "hires_today", "occupied_sites"),
    "rival_product": ("producer_sites", "visible_yield", "serviced_today", "fertilizer_available"),
    "cash_monotone": {"scales": CASH_SCALES, "column": CASH_MONOTONE_COLUMN,
                      "transform": "cash/(scale+abs(cash)), unit scale; strictly increasing in real arithmetic",
                      "thresholds": CASH_THRESHOLDS, "affordability": "cash >= public cost, unit 0/1"},
    "market": DEMAND_SPEC,
    "sale_quantities": (1, 10, 100, "own_aggregate_stock"),
    "sale_semantics": "exact sequential solo marginal revenue including floor inventory nonmovement; not an action rule",
    "values": "v2 signed log1p/10, except GLOBAL columns 8 onward use declared unit-scale cash/affordability",
    "history": "State.demand_history explicit public prefix cursor snapshot; no endpoint reconstruction",
    "limits": ("phase aggregates are lossy, not a sufficient state or playing-strength claim",
               "fixed V0 calendar only; full-season realized demand is not an actor field",
               "rival public tile yields are not rival shed/seeds/carried stock"),
}


def _feature_version(value: str) -> str:
    if type(value) is not str or value not in ("v2", "v3"):
        raise ValueError("explicit feature_version must be v2 or v3")
    return value


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


FEATURE_SCHEMA_SHA256 = _digest(FEATURE_SPEC)
FEATURE_SCHEMA_SHA256_V3 = _digest(FEATURE_SPEC_V3)
READOUT_SCHEMA = 'farm-v0-candidate-readout-v1'


def _readout_mode(value: str) -> str:
    if type(value) is not str or value not in ('mean', 'pooled_control', 'candidate_attention'):
        raise ValueError('readout must be mean, pooled_control or candidate_attention')
    return value


def cash_periodic(cash: torch.Tensor) -> torch.Tensor:
    """Return unit-scale periodic coordinates for one or more own-cash values."""
    periods = torch.tensor(CASH_PERIODS, device=cash.device, dtype=cash.dtype)
    angle = 2 * math.pi * cash[..., None] / periods
    return torch.cat([angle.sin(), angle.cos()], -1)


def cash_monotone(cash: torch.Tensor) -> torch.Tensor:
    """Bounded monotone cash coordinates; the signed-log own_cash is also retained."""
    scales = torch.tensor(CASH_SCALES, device=cash.device, dtype=cash.dtype)
    return cash[..., None] / (scales + cash[..., None].abs())


def source_pins() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    names = ('src/farm_neat/policy/v0_scorer.py', 'src/farm_neat/policy/v0_menu_fast.py',
             'src/farm_neat/policy/qwen_hybrid.py',
             'src/farm_neat/economy/v0/vocabulary.py', 'src/farm_neat/economy/v0/plan.py',
             'src/farm_neat/economy/v0/projection.py', 'src/farm_neat/economy/v0/modules.json',
             'src/farm_neat/economy/heuristic/market.py', 'src/farm_neat/game/constants.py')
    return {**hybrid.sources(), **{name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in names}}


def _menu_backend(backend: str) -> str:
    if type(backend) is not str or backend not in ('fast', 'reference'):
        raise ValueError("backend must be 'fast' or 'reference'")
    return backend


@dataclass(frozen=True)
class Prefix:
    """Canonical continuation only. No selected raw alias or recurrent vector."""
    plan: Plan
    last_key: int = -1
    edit_count: int = 0
    terminated: bool = False
    _snapshot: str = field(init=False, repr=False)

    def __post_init__(self):
        if (not isinstance(self.plan, Plan) or type(self.last_key) is not int
                or not -1 <= self.last_key < V.N_KEYS or type(self.edit_count) is not int
                or not 0 <= self.edit_count <= V.MAX_EDITS_PER_EPOCH
                or type(self.terminated) is not bool
                or (self.edit_count == 0) != (self.last_key == -1)
                or self.last_key + 1 < self.edit_count):
            raise ValueError('canonical Plan, key cursor and 0..16 edit count required')
        object.__setattr__(self, 'plan', self.plan.copy())
        object.__setattr__(self, '_snapshot', _json(self.plan.canonical()))

    def payload(self) -> dict:
        if _json(self.plan.canonical()) != self._snapshot:
            raise ValueError('canonical prefix Plan was mutated')
        return {'plan': self.plan.canonical(), 'last_key': self.last_key,
                'edit_count': self.edit_count, 'terminated': self.terminated}


def _prefix(value: Prefix | Plan) -> Prefix:
    value = Prefix(value) if isinstance(value, Plan) else value
    if not isinstance(value, Prefix):
        raise ValueError('Plan or explicit Prefix required')
    value.payload()
    if value.terminated:
        raise ValueError('cannot score/decode a terminated prefix')
    return value


def _support(ids: Sequence[int], prefix: Prefix | None = None) -> tuple[int, ...]:
    ids = tuple(ids)
    if (not ids or any(type(a) is not int or not 0 <= a < V.N_ACTIONS for a in ids)
            or ids != tuple(sorted(set(ids))) or ids[0] != V.STOP_ID):
        raise ValueError('nonempty sorted unique frozen legal IDs including STOP required')
    if prefix is not None:
        if prefix.edit_count == V.MAX_EDITS_PER_EPOCH and ids != (V.STOP_ID,):
            raise ValueError('cap support must be forced STOP only')
        if any(a != V.STOP_ID and V.plan_key(a) <= prefix.last_key for a in ids):
            raise ValueError('support contains an expired canonical key')
    return ids


def _hold_memory(memory: dict | None) -> dict | None:
    if memory is None:
        return None
    if not isinstance(memory, dict) or set(memory) - {'hold_releases'}:
        raise ValueError('only declared own hold_releases memory is accepted')
    holds = memory.get('hold_releases', {})
    if not isinstance(holds, dict):
        raise ValueError('hold_releases must be a mapping')
    result = {}
    for product, value in holds.items():
        if type(product) is not int or not 0 <= product < len(V.PRODUCTS):
            raise ValueError('invalid hold product')
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError('invalid hold binding')
        pair, release = value
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(type(x) is not int for x in pair)
                or not 0 <= pair[0] < len(V.PRIMITIVES)
                or not 0 <= pair[1] < len(V.HORIZONS)
                or type(release) is not int or not 0 <= release <= V.LAST_STEP):
            raise ValueError('invalid hold binding')
        result[product] = (tuple(pair), release)
    return result


def actor_tokens(state: State, prefix_plan: Prefix | Plan, *, memory: dict | None = None,
                 feature_version: str = "v2") -> tuple:
    """Actor-only numeric rows and knownness. Reads an explicit State allowlist.

    Relative event/horizon distances are allowed; no absolute time, coordinates,
    arbitrary state.__dict__, teacher target, rival private data or alias ID is read.
    """
    version = _feature_version(feature_version)
    width = FEATURE_WIDTH_V3 if version == "v3" else FEATURE_WIDTH
    prefix = _prefix(prefix_plan)
    if (not isinstance(state, State) or state.hour != 0 or state.step != state.day * 24
            or not 0 <= state.day < V.SEASON_DAYS):
        raise ValueError('V0 actor input requires an actor-visible dawn State')
    holds = _hold_memory(memory)
    tokens = []

    def add(identity, values):
        if len(values) > width:
            raise ValueError('feature schema exceeds declared width')
        if any(v is not None and (not isinstance(v, (int, float, bool)) or not math.isfinite(v))
               for v in values):
            raise ValueError('finite actor facts required')
        values = list(values) + [None] * (width - len(values))
        tokens.append((identity, tuple(0. if v is None else float(v) for v in values),
                       tuple(v is not None for v in values)))

    global_values = [V.SEASON_DAYS - state.day, state.cash, state.quadrants, state.hands_today,
                     state.hires_today, state.occupied, prefix.last_key + 1, prefix.edit_count]
    market = state.market_features() if version == "v3" else None
    if version == "v3":
        global_values += [state.cash / (scale + abs(state.cash)) for scale in CASH_SCALES]
        global_values += [state.cash >= cost for cost in CASH_THRESHOLDS]
        global_values += [state.cash >= hire_cost(state.hires_today),
                          state.quadrants < 4 and state.cash >= LAND_PRICES[state.quadrants - 1]]
    add(0, global_values)
    for index, product in enumerate(V.PRODUCTS):
        params = (state.params or MARKET_PARAMS)[product]
        shapes = [params['below_func'], params['above_func']]
        if any(s not in SHAPES for s in shapes):
            raise ValueError('unknown public market shape')
        distances = [state.horizon_step(index, h) for h in range(1, len(V.HORIZONS))]
        primitive, horizon, rank = prefix.plan.overlay(index)
        bound = None if holds is None else holds.get(index)
        values = [state.stock[product], state.inventory[product],
            *(params[n] for n in ('base', 'I0', 'T', 'below_target', 'above_target')),
            *(shape == s for shape in shapes for s in SHAPES),
            *(None if t is None else t - state.step for t in distances),
            state.demand(product, state.step), state.demand(product, state.step + 24),
            index in prefix.plan.market, primitive, horizon, rank, holds is not None,
            None if holds is None else bound is not None,
            None if bound is None else bound[0][0], None if bound is None else bound[0][1],
            None if bound is None else bound[1] - state.step]
        if version == "v3":
            row = market[product]
            values += [row["target_inventory_gap"], row["unsatisfied_demand"], row["total_demand_created"],
                       *row["created_by_phase"], *row["known_remaining"], *row["expected_unrevealed"],
                       row["remaining_reveals"],
                       *(sell_revenue(product, state.inventory[product], q, state.params)
                         for q in (1, 10, 100, state.stock[product]))]
        add(TOKEN_PRODUCT + index, values)
    for index, crop in enumerate(CROPS):
        add(TOKEN_SEED + index, [state.seeds.get(crop, 0)])
    add(TOKEN_LAND, [prefix.plan.land is not None,
                    *(prefix.plan.land == r for r in range(len(V.LAND_TIMINGS)))])
    for module, count in sorted(prefix.plan.cohorts.items()):
        add(TOKEN_MODULE + module, [count, module in prefix.plan.regimes,
            *(prefix.plan.regime(module) == r for r in range(len(V.REGIMES))),
            V.MODULES[module]['birth'] - state.day, V.MODULES[module]['animal']])
    if version == "v3":
        rival = state.rival_public
        add(TOKEN_RIVAL, [rival["summary"][name] for name in FEATURE_SPEC_V3["rival"]])
        for i, product in enumerate(V.PRODUCTS):
            add(TOKEN_RIVAL_PRODUCT + i, [rival["products"][product][name]
                                         for name in FEATURE_SPEC_V3["rival_product"]])
    return tuple(tokens)


def _action_fields() -> tuple[list[list[int]], int]:
    # Disjoint typed feature IDs preserve the entire frozen raw action tuple.
    # Zero pads fields; neither raw-ID order nor aliases become continuation state.
    kinds = ('STOP', 'PROD', 'SERVICE', 'LAND', 'MARKET')
    sizes = {'kind': len(kinds), 'module': V.N_MODULES, 'delta': len(V.DELTAS),
             'regime': len(V.REGIMES), 'land': len(V.LAND_TIMINGS),
             'product': len(V.PRODUCTS), 'primitive': len(V.PRIMITIVES),
             'horizon': len(V.HORIZONS), 'rank': V.RANKS}
    offsets, next_id = {}, 1
    for name, size in sizes.items():
        offsets[name] = next_id; next_id += size
    fields_by_kind = {'STOP': (), 'PROD': ('module', 'delta'),
                      'SERVICE': ('module', 'regime'), 'LAND': ('land',),
                      'MARKET': ('product', 'primitive', 'horizon', 'rank')}
    rows = []
    for action in range(V.N_ACTIONS):
        kind, *values = V.decode(action)
        row = [offsets['kind'] + kinds.index(kind)]
        row += [offsets[name] + value for name, value in zip(fields_by_kind[kind], values)]
        rows.append(row + [0] * (5 - len(row)))
    return rows, next_id


_ACTION_FIELDS, N_ACTION_FIELDS = _action_fields()


def masked_log_probs(logits: torch.Tensor, valid: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
    if (type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0
            or logits.ndim != 2 or not logits.is_floating_point()
            or valid.shape != logits.shape or valid.dtype != torch.bool
            or valid.device != logits.device):
        raise ValueError('floating BxK logits, boolean mask and positive finite temperature required')
    if logits.shape[0] == 0 or logits.shape[1] == 0 or not bool(valid.any(-1).all()):
        raise ValueError('each row requires a nonempty legal support; no all-masked fallback')
    if not bool(torch.isfinite(logits[valid]).all()):
        raise ValueError('finite legal logits required')
    scaled = logits.masked_fill(~valid, 0) / temperature
    if not bool(torch.isfinite(scaled[valid]).all()):
        raise ValueError('finite temperature-scaled legal logits required')
    return scaled.masked_fill(~valid, -torch.inf).log_softmax(-1)


def class_nll(logits: torch.Tensor, target_indices: Sequence[int], *, valid: torch.Tensor | None = None) -> torch.Tensor:
    """One prefix's exact successor-class NLL, NOT representative-ID CE.

    Sum these over a declared class PATH (sampled STOP included; forced STOP=0),
    then let the consuming trainer average by trajectories, never padded rows.
    """
    if logits.ndim != 1:
        raise ValueError('one-dimensional prefix logits required')
    target = tuple(target_indices)
    if (not target or any(type(i) is not int or not 0 <= i < len(logits) for i in target)
            or len(set(target)) != len(target)):
        raise ValueError('nonempty unique in-support target indices required')
    valid = torch.ones_like(logits, dtype=torch.bool) if valid is None else valid
    logp = masked_log_probs(logits[None], valid[None])[0]
    if not bool(valid[list(target)].all()):
        raise ValueError('target includes illegal padding')
    return -torch.logsumexp(logp[list(target)], 0)


@dataclass
class ScoreBatch:
    logits: torch.Tensor
    legal_ids: torch.Tensor
    valid: torch.Tensor
    routing: tuple = field(repr=False)

    def log_probs(self, temperature: float = 1.) -> torch.Tensor:
        return masked_log_probs(self.logits, self.valid, temperature)


class V0Scorer(nn.Module):
    """Fresh V0 input/head on the existing 16-layer hybrid/MoE block design."""
    def __init__(self, config: hybrid.Config | None = None, *, diagnostic: bool = False,
                 backend: str = 'fast', feature_version: str = 'v2', readout: str = 'mean'):
        super().__init__()
        self.feature_version = _feature_version(feature_version)
        self.feature_width = FEATURE_WIDTH_V3 if self.feature_version == 'v3' else FEATURE_WIDTH
        self.readout = _readout_mode(readout)
        # Execution choice only: exact backends share weights, traces and continuation.
        # ScorerPolicy inherits this choice without changing the evaluator API.
        self.menu_backend = _menu_backend(backend)
        config = hybrid.Config() if config is None else config
        if not isinstance(config, hybrid.Config) or type(diagnostic) is not bool:
            raise ValueError('explicit hybrid config and diagnostic boolean required')
        self.config, self.diagnostic = config, diagnostic
        width = config.width
        self.input = nn.Linear(self.feature_width, width, bias=False)
        self.knownness_input = nn.Linear(self.feature_width, width, bias=False)
        self.token_embedding = nn.Embedding(N_TOKENS_V3 if self.feature_version == 'v3' else N_TOKENS, width)
        self.blocks = nn.ModuleList(hybrid.Block(config, i) for i in range(config.layers))
        self.final_norm = hybrid.ZeroCenteredRMSNorm(width, config.rms_epsilon)
        self.action_embedding = nn.Embedding(N_ACTION_FIELDS, width, padding_idx=0)
        self.register_buffer('action_fields', torch.tensor(_ACTION_FIELDS, dtype=torch.long), persistent=False)
        self.phi_input = nn.Linear(len(PHI_NAMES), width, bias=False)
        self.context_query = nn.Linear(width, width)
        self.score_head = nn.Linear(width, 1)
        if self.readout != 'mean':
            # Both experimental modes use the exact same parameterization. The
            # control changes only whether K/V see every valid token or their mean.
            self.readout_query = nn.Linear(width, width, bias=False)
            self.readout_key = nn.Linear(width, width, bias=False)
            self.readout_value = nn.Linear(width, width, bias=False)
            self.readout_output = nn.Linear(width, width, bias=False)
        nn.init.normal_(self.token_embedding.weight, std=.03)
        nn.init.normal_(self.action_embedding.weight, std=.03)
        with torch.no_grad():
            self.action_embedding.weight[0].zero_()
        report = self.parameter_report()
        if not diagnostic and (config.layers < 14 or report['trainable_parameters'] < 30_000_000):
            raise ValueError('production requires >=30M trainable parameters and >=14 layers; use diagnostic=True for fixtures')
        self._identity = {'schema': SCHEMA, 'config': asdict(config), 'diagnostic': diagnostic,
            'action_schema_sha256': V.ACTION_SCHEMA_SHA256,
            'feature_schema_sha256': (FEATURE_SCHEMA_SHA256_V3 if self.feature_version == 'v3' else FEATURE_SCHEMA_SHA256),
            'source_pins': source_pins(),
            'legacy_checkpoint_compatible': False, 'initialization': 'FRESH_TORCH_NO_WEIGHT_IMPORT',
            'numerics': 'FP32_NO_AUTOCAST_NO_TF32', 'continuation': 'REENCODE_CANONICAL_PREFIX_NO_RAW_ALIAS_CACHE'}
        if self.readout != 'mean':
            self._identity['readout'] = {
                'schema': READOUT_SCHEMA, 'version': 'v1', 'mode': self.readout,
                'method': ('POOLED_GATED_MLP' if self.readout == 'pooled_control'
                           else 'MASKED_TOKEN_ATTENTION'),
                'context': ('MASKED_MEAN_ONLY' if self.readout == 'pooled_control'
                            else 'MASKED_ACTOR_TOKENS'),
                'candidate': 'ACTION_FIELDS_PLUS_PHI',
                'residual': 'MASKED_MEAN'}

    def get_extra_state(self):
        return copy.deepcopy(self._identity)

    def set_extra_state(self, state):
        if state != self._identity:
            raise ValueError('V0 identity mismatch; legacy/drifted weights need a separately reviewed migration')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Accept only complete, source-identical native V0 state dictionaries.

        Preflight before copying any parameter: PyTorch's ordinary strict=False
        permits missing extra_state, and strict=True alone can mutate parameters
        before reporting missing keys. This is not a legacy migration interface.
        """
        if strict is not True or assign is not False:
            raise ValueError('V0 identity requires strict=True and assign=False')
        if not isinstance(state_dict, Mapping) or state_dict.get('_extra_state') != self._identity:
            raise ValueError('V0 identity missing or mismatched')
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            raise ValueError('V0 identity requires the complete native parameter key set')
        for name, destination in expected.items():
            if name == '_extra_state':
                continue
            source = state_dict[name]
            if (not isinstance(source, torch.Tensor) or source.shape != destination.shape
                    or source.dtype != destination.dtype):
                raise ValueError(f'V0 parameter shape/dtype mismatch: {name}')
        return super().load_state_dict(state_dict, strict=True, assign=False)

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        routed = sum(p.numel() for name, p in self.named_parameters() if '.moe.experts.' in name)
        per_expert = 3 * self.config.width * self.config.expert_hidden
        if routed != self.config.layers * self.config.experts * per_expert:
            raise ValueError('routed parameter accounting changed')
        return {'schema': SCHEMA, 'layers': self.config.layers, 'total_parameters': total,
            'trainable_parameters': trainable, 'routed_parameters_total': routed,
            'active_routed_per_token': self.config.layers * self.config.top_k * per_expert,
            'active_per_token_upper_bound': total - routed + self.config.layers * self.config.top_k * per_expert,
            'active_convention': 'all non-routed tables counted in full plus top-2 experts/layer; upper bound, not batch union/FLOPs',
            'diagnostic': self.diagnostic, 'parameter_counts_are_training_evidence': False}

    def _encode_tokens(self, states, prefixes, memories):
        device = self.input.weight.device
        packs = [actor_tokens(s, p, memory=m, feature_version=self.feature_version)
                 for s, p, m in zip(states, prefixes, memories)]
        batch, length = len(packs), max(map(len, packs))
        rows = torch.zeros(batch, length, self.feature_width, device=device, dtype=torch.float32)
        known = torch.zeros_like(rows, dtype=torch.bool)
        enabled = torch.zeros(batch, length, device=device, dtype=torch.bool)
        identities = torch.zeros(batch, length, device=device, dtype=torch.long)
        for i, pack in enumerate(packs):
            n = len(pack)
            identities[i, :n] = torch.tensor([t[0] for t in pack], device=device)
            rows[i, :n] = torch.tensor([t[1] for t in pack], device=device)
            known[i, :n] = torch.tensor([t[2] for t in pack], device=device)
            enabled[i, :n] = True
        visible = rows.masked_fill(~known | ~enabled[..., None], 0)
        visible = visible.sign() * visible.abs().log1p() / 10
        is_global = (identities == 0) & enabled
        column = torch.arange(self.feature_width, device=device)
        if self.feature_version == "v2":
            periodic = cash_periodic(rows[..., CASH_COLUMN])
            padded = torch.zeros_like(visible)
            padded[..., CASH_PERIODIC_COLUMN:CASH_PERIODIC_COLUMN + periodic.shape[-1]] = periodic
            slot = (is_global[..., None] & (column >= CASH_PERIODIC_COLUMN)
                    & (column < CASH_PERIODIC_COLUMN + periodic.shape[-1]))
            visible = torch.where(slot, padded, visible)
            known = known | slot
        else:
            slot = (is_global[..., None] & (column >= CASH_MONOTONE_COLUMN)
                    & (column < len(FEATURE_SPEC_V3["global"])))
            visible = torch.where(slot, rows.masked_fill(~known, 0), visible)
        x = self.input(visible) + self.knownness_input(known.float()) + self.token_embedding(identities)
        x = x.masked_fill(~enabled[..., None], 0)
        residual = x[..., None, :].expand(*x.shape[:2], self.config.branches, self.config.width)
        routes = []
        for block in self.blocks:
            residual, routing = block(residual, enabled)
            routes.append(routing)
        hidden = self.final_norm(residual.mean(-2)).masked_fill(~enabled[..., None], 0)
        pooled = hidden.sum(1) / enabled.sum(1, keepdim=True)
        return hidden, enabled, pooled, tuple(routes)

    def _encode(self, states, prefixes, memories):
        """Compatibility pooling interface used by existing trainers and probes."""
        _hidden, _enabled, pooled, routing = self._encode_tokens(states, prefixes, memories)
        return pooled, routing

    def _readout_context(self, hidden, enabled, pooled, candidates):
        """Return one residual actor context per candidate for a nondefault readout."""
        if self.readout == 'mean':
            return pooled[:, None].expand_as(candidates)
        queries = self.readout_query(candidates)
        if self.readout == 'pooled_control':
            keys = self.readout_key(pooled)[:, None]
            values = self.readout_value(pooled)[:, None]
            attended = torch.tanh(queries + keys) * values
        else:
            keys = self.readout_key(hidden)
            values = self.readout_value(hidden)
            weights = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.config.width)
            weights = weights.masked_fill(~enabled[:, None], -torch.inf).softmax(-1)
            attended = torch.matmul(weights, values)
        return pooled[:, None] + self.readout_output(attended)

    def score(self, state, prefix_plan, legal_ids, phi, *, memory=None) -> torch.Tensor:
        return self.score_batch([state], [prefix_plan], [legal_ids], [phi], memories=[memory]).logits[0]

    def score_batch(self, states, prefix_plans, legal_ids, phi, *, memories=None) -> ScoreBatch:
        if not (len(states) == len(prefix_plans) == len(legal_ids) == len(phi) > 0):
            raise ValueError('aligned nonempty ragged context/candidate batches required')
        prefixes = [_prefix(p) for p in prefix_plans]
        supports = [_support(ids, p) for ids, p in zip(legal_ids, prefixes)]
        device = self.input.weight.device
        maximum = max(map(len, supports))
        ids = torch.zeros(len(states), maximum, dtype=torch.long, device=device)
        features = torch.zeros(len(states), maximum, len(PHI_NAMES), dtype=torch.float32, device=device)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for i, support in enumerate(supports):
            n = len(support)
            try:
                row = torch.as_tensor(phi[i], dtype=torch.float32, device=device)
            except (TypeError, RuntimeError) as exc:
                raise ValueError('finite aligned PHI_NAMES rows required') from exc
            if row.shape != (n, len(PHI_NAMES)):
                raise ValueError('phi shape must match support in frozen PHI_NAMES order')
            ids[i, :n] = torch.tensor(support, device=device)
            features[i, :n] = row
            valid[i, :n] = True
        return self.score_padded(states, prefixes, ids, features, valid, memories=memories)

    def score_padded(self, states, prefix_plans, legal_ids, phi, valid, *, memories=None) -> ScoreBatch:
        device = self.input.weight.device
        prefixes = [_prefix(p) for p in prefix_plans]
        if (legal_ids.ndim != 2 or legal_ids.shape[0] != len(states) or len(states) == 0
                or len(prefixes) != len(states) or legal_ids.dtype != torch.long
                or valid.shape != legal_ids.shape or valid.dtype != torch.bool
                or phi.shape != (*legal_ids.shape, len(PHI_NAMES)) or phi.dtype != torch.float32
                or any(t.device != device for t in (legal_ids, phi, valid))
                or self.input.weight.dtype != torch.float32 or torch.is_autocast_enabled(device.type)):
            raise ValueError('aligned FP32/long/bool padded candidates on the model device required')
        if device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError('FP32 reference requires TF32 disabled')
        for row, enabled, prefix in zip(legal_ids, valid, prefixes):
            _support(row[enabled].tolist(), prefix)
        if not bool(torch.isfinite(phi[valid]).all()):
            raise ValueError('finite legal phi required')
        memories = [None] * len(states) if memories is None else list(memories)
        if len(memories) != len(states):
            raise ValueError('memory batch length mismatch')
        safe_ids = legal_ids.masked_fill(~valid, V.STOP_ID)
        safe_phi = phi.masked_fill(~valid[..., None], 0)
        transformed = safe_phi.sign() * safe_phi.abs().log1p() / 10
        candidates = self.action_embedding(self.action_fields[safe_ids]).sum(-2) + self.phi_input(transformed)
        if self.readout == 'mean':
            context, routing = self._encode(states, prefixes, memories)
            logits = self.score_head(torch.tanh(
                self.context_query(context)[:, None] + candidates)).squeeze(-1)
        else:
            hidden, enabled, pooled, routing = self._encode_tokens(states, prefixes, memories)
            context = self._readout_context(hidden, enabled, pooled, candidates)
            logits = self.score_head(torch.tanh(self.context_query(context) + candidates)).squeeze(-1)
        logits = logits.masked_fill(~valid, -torch.inf)
        # Validate finite legal outputs now, not only when a downstream loss is called.
        if not bool(torch.isfinite(logits[valid]).all()):
            raise ValueError('nonfinite neural legal logits')
        return ScoreBatch(logits, legal_ids, valid, routing)


@dataclass(frozen=True)
class SuccessorClass:
    class_id: str
    indices: tuple[int, ...]
    successor: Prefix


@dataclass(frozen=True)
class Menu:
    prefix: Prefix
    legal_ids: tuple[int, ...]
    phi: tuple[tuple[float, ...], ...]
    classes: tuple[SuccessorClass, ...]


def partition_menu(prefix_plan, legal_ids, phi, resolver=None) -> Menu:
    """Partition an already-verified sparse support by exact continuation state.

    This lower-level adapter does NOT certify economic legality/completeness of
    caller-supplied training supports. Use build_menu for free-running production.
    Group by exact canonical JSON; hashes name classes, not equality tests.
    """
    prefix = _prefix(prefix_plan)
    ids = _support(legal_ids, prefix)
    try:
        rows = tuple(tuple(float(v) for v in row) for row in phi)
    except (TypeError, ValueError) as exc:
        raise ValueError('aligned finite phi rows required') from exc
    if (len(rows) != len(ids) or any(len(row) != len(PHI_NAMES) for row in rows)
            or any(not math.isfinite(v) for row in rows for v in row)):
        raise ValueError('aligned finite phi rows required')
    groups = {}
    for i, action in enumerate(ids):
        successor = (Prefix(prefix.plan, prefix.last_key, prefix.edit_count, True) if action == V.STOP_ID
                     else Prefix(prefix.plan.apply(action, resolver), V.plan_key(action), prefix.edit_count + 1))
        signature = _json(successor.payload())
        if signature not in groups:
            groups[signature] = (successor, [])
        groups[signature][1].append(i)
    classes = tuple(sorted((SuccessorClass(hashlib.sha256(s.encode()).hexdigest(), tuple(indices), successor)
                            for s, (successor, indices) in groups.items()), key=lambda c: c.class_id))
    if len({c.class_id for c in classes}) != len(classes):
        raise ValueError('class hash collision across distinct exact continuations')
    return Menu(prefix, ids, rows, classes)


def _projection_options(options=None):
    options = dict(options or {})
    if set(options) - {'labor_factor', 'belief', 'hold_releases'}:
        raise ValueError('unknown projection options')
    result = {'labor_factor': 4., 'belief': {p: 0 for p in V.PRODUCTS}, 'hold_releases': {}}
    result.update(options)
    factor = result['labor_factor']
    if type(factor) not in (float, int) or not math.isfinite(factor) or factor <= 0:
        raise ValueError('finite positive labor_factor required')
    belief = result['belief']
    if belief is None:
        belief = {}
    if not isinstance(belief, dict) or set(belief) - set(V.PRODUCTS) or any(v != 0 for v in belief.values()):
        raise ValueError('v1 bootstrap requires explicit zero-flow belief')
    result['belief'] = {p: 0 for p in V.PRODUCTS}
    result['hold_releases'] = _hold_memory({'hold_releases': result['hold_releases']})
    return result


def build_menu(state: State, prefix_plan, *, projection_kwargs=None) -> Menu:
    """Recompute masks, phi and successor classes on THIS prefix, including cap."""
    prefix = _prefix(prefix_plan)
    options = _projection_options(projection_kwargs)
    masks = Masks(state, prefix.plan, **options)
    ids = ((V.STOP_ID,) if prefix.edit_count == V.MAX_EDITS_PER_EPOCH else
           tuple(sorted(a for a in masks.legal_actions() if a == V.STOP_ID or V.plan_key(a) > prefix.last_key)))
    rows = [project_phi(state, prefix.plan, a, masks, **options) for a in ids]
    return partition_menu(prefix, ids, rows, masks.resolver)


def _raw_log_probs(logits, menu, temperature=1.):
    if logits.ndim != 1 or len(logits) != len(menu.legal_ids):
        raise ValueError('logits must align exactly with the recorded support')
    return masked_log_probs(logits[None], torch.ones_like(logits, dtype=torch.bool)[None], temperature)[0]


def class_log_probs(logits: torch.Tensor, menu: Menu, temperature: float = 1.) -> torch.Tensor:
    raw = _raw_log_probs(logits, menu, temperature)
    return torch.stack([torch.logsumexp(raw[list(c.indices)], 0) for c in menu.classes])


@dataclass
class Choice:
    action: int
    prefix: Prefix
    raw_logp: torch.Tensor
    class_logp: torch.Tensor
    record: dict


def _choice(menu, raw, classes, index, mode, temperature):
    action = menu.legal_ids[index]
    ci = next(i for i, c in enumerate(menu.classes) if index in c.indices)
    chosen = menu.classes[ci]
    forced = menu.prefix.edit_count == V.MAX_EDITS_PER_EPOCH
    raw_logp, class_logp = raw[index], classes[ci]
    record = {'schema': STEP_SCHEMA, 'prefix': menu.prefix.payload(), 'support': list(menu.legal_ids),
        'phi_sha256': _digest(menu.phi), 'raw_id': action, 'class_id': chosen.class_id,
        'class_members': [menu.legal_ids[i] for i in chosen.indices],
        'continuation': chosen.successor.payload(), 'temperature': float(temperature), 'mode': mode,
        'raw_logp': float(raw_logp.detach()), 'class_logp': float(class_logp.detach()),
        'behavior_logp': 0. if mode == 'greedy_class' else float(raw_logp.detach()),
        'forced_stop': forced, 'termination': ('CAP' if forced else 'STOP') if action == 0 else 'CONTINUE'}
    return Choice(action, chosen.successor, raw_logp, class_logp, record)


def choose(menu: Menu, logits: torch.Tensor, *, generator: torch.Generator | None = None,
           mode: str = 'sample_raw', temperature: float = 1.) -> Choice:
    """Sample raw categorical IDs, or greedily select total successor CLASS mass.

    Greedy ties use class digest order; its representative is the lowest raw ID
    in that class. Neither choice is embedded into the next canonical prefix.
    Greedy behavior likelihood is 1; softmax logps remain policy diagnostics.
    """
    if mode not in ('sample_raw', 'greedy_class'):
        raise ValueError('mode must be sample_raw or greedy_class')
    raw = _raw_log_probs(logits, menu, temperature)
    classes = torch.stack([torch.logsumexp(raw[list(c.indices)], 0) for c in menu.classes])
    if menu.prefix.edit_count == V.MAX_EDITS_PER_EPOCH:
        index = 0  # Forced administrative STOP has probability one and consumes no RNG.
    elif mode == 'sample_raw':
        if not isinstance(generator, torch.Generator):
            raise ValueError('explicit torch.Generator required for sampling')
        index = int(torch.multinomial(raw.exp(), 1, generator=generator).item())
    else:
        ci = int(classes.argmax().item())
        index = menu.classes[ci].indices[0]
    return _choice(menu, raw, classes, index, mode, temperature)


def replay_step(menu: Menu, logits: torch.Tensor, record: dict, *, verify_logps: bool = True) -> Choice:
    """Replay on a freshly reconstructed menu; localize the first mismatch.

    verify_logps=False permits changed model weights for BC/RL evaluation, not
    changed supports, phi, canonical continuation or cap/STOP semantics.
    """
    try:
        mode, temperature, action = record['mode'], record['temperature'], record['raw_id']
        if mode not in ('sample_raw', 'greedy_class') or type(action) is not int:
            raise ValueError('invalid mode/raw ID')
        index = menu.legal_ids.index(action)
        raw = _raw_log_probs(logits, menu, temperature)
        classes = torch.stack([torch.logsumexp(raw[list(c.indices)], 0) for c in menu.classes])
        result = _choice(menu, raw, classes, index, mode, temperature)
        if set(record) != set(result.record):
            raise ValueError('record fields')
        floats = {'raw_logp', 'class_logp', 'behavior_logp'}
        if len(menu.legal_ids) == 1 and any(record[key] != 0 for key in floats):
            # A singleton support has probability one under any weights (return f3b9d839's adverse case).
            raise ValueError('singleton support log-probabilities must be zero')
        for key, value in result.record.items():
            if key in floats:
                if verify_logps and (type(record[key]) not in (int, float)
                        or not math.isfinite(record[key])
                        or not math.isclose(record[key], value, rel_tol=2e-6, abs_tol=2e-6)):
                    raise ValueError(key)
            elif record[key] != value:
                raise ValueError(key)
        if verify_logps and mode == 'greedy_class':
            expected = menu.classes[int(classes.argmax())].indices[0]
            if index != expected:
                raise ValueError('greedy class/representative')
        return result
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f'first replay divergence: {exc}') from exc


@dataclass
class EpochResult:
    plan: Plan
    actions: tuple[int, ...]
    raw_logp: torch.Tensor
    class_logp: torch.Tensor
    trace: dict


def _state_digest(state, feature_version="v2"):
    names = ('player', 'step', 'cash', 'quadrants', 'hands_today', 'hires_today',
             'stock', 'seeds', 'inventory', 'params', 'shops', 'occupied')
    payload = {name: getattr(state, name) for name in names}
    if _feature_version(feature_version) == "v3":
        payload.update(demand_history=state.demand_history, rival_public=state.rival_public)
    return _digest(payload)


def _epoch(scorer, state, prefix_plan, *, generator=None, mode='sample_raw', temperature=1.,
           projection_kwargs=None, replay=None, verify_logps=True, backend=None):
    from farm_neat.policy import v0_menu_fast
    backend = _menu_backend(getattr(scorer, 'menu_backend', 'fast') if backend is None else backend)
    if (mode not in ('sample_raw', 'greedy_class') or type(temperature) not in (int, float)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('V0 epoch requires a valid decoding mode and finite positive temperature')
    prefix = _prefix(prefix_plan)
    options = _projection_options(projection_kwargs)
    memory = {'hold_releases': options['hold_releases']}
    header = {'schema': TRACE_SCHEMA, 'model_identity': scorer.get_extra_state(),
        'state_sha256': _state_digest(state, 'v3' if scorer.get_extra_state().get('feature_schema_sha256')
                                     == FEATURE_SCHEMA_SHA256_V3 else 'v2'),
        'projection_sha256': _digest(options),
        'initial_prefix': prefix.payload(), 'mode': mode, 'temperature': float(temperature)}
    if replay is not None:
        for key, value in header.items():
            if replay.get(key) != value:
                raise ValueError(f'first replay divergence: epoch {key}')
        if not isinstance(replay.get('steps'), list):
            raise ValueError('first replay divergence: steps required')
    choices = []
    while not prefix.terminated:
        menu = v0_menu_fast.build_menu(state, prefix, projection_kwargs=options, backend=backend)
        logits = scorer.score(state, prefix, menu.legal_ids, menu.phi, memory=memory)
        if replay is None:
            choice = choose(menu, logits, generator=generator, mode=mode, temperature=temperature)
        else:
            if len(choices) >= len(replay['steps']):
                raise ValueError(f'first replay divergence: missing step {len(choices)} / STOP')
            try:
                record = replay['steps'][len(choices)]
                for key in ('mode', 'temperature'):
                    if not isinstance(record, dict) or record.get(key) != header[key]:
                        raise ValueError(f'first replay divergence: step {key} differs from epoch')
                choice = replay_step(menu, logits, record, verify_logps=verify_logps)
            except ValueError as exc:
                raise ValueError(f'prefix {len(choices)}: {exc}') from exc
        choices.append(choice)
        prefix = choice.prefix  # Reconstruct from own sampled/replayed successor, never teacher prefixes.
    raw = torch.stack([c.raw_logp.double() for c in choices]).sum()
    classes = torch.stack([c.class_logp.double() for c in choices]).sum()
    actions = tuple(c.action for c in choices)
    trace = {**header, 'steps': [c.record for c in choices], 'actions': list(actions),
        'final_plan': prefix.plan.canonical(), 'raw_path_logp': float(raw.detach()),
        'class_path_logp': float(classes.detach()),
        'class_path_not_final_plan_marginal': True}
    if replay is not None:
        if len(replay['steps']) != len(choices):
            raise ValueError('first replay divergence: extra records after STOP')
        for key in ('actions', 'final_plan', 'class_path_not_final_plan_marginal'):
            if replay.get(key) != trace[key]:
                raise ValueError(f'first replay divergence: epoch {key}')
        if verify_logps:
            for key in ('raw_path_logp', 'class_path_logp'):
                saved = replay.get(key)
                if type(saved) not in (int, float) or not math.isclose(saved, trace[key], abs_tol=3e-5, rel_tol=2e-6):
                    raise ValueError(f'first replay divergence: epoch {key}')
    return EpochResult(prefix.plan.copy(), actions, raw, classes, trace)


def decode_epoch(scorer: V0Scorer, state: State, prefix_plan: Prefix | Plan, *,
                 generator: torch.Generator | None = None, mode: str = 'sample_raw',
                 temperature: float = 1., projection_kwargs=None, backend: str | None = None) -> EpochResult:
    """Native free-running epoch; caller owns no_grad/eval and generator state.

    backend overrides scorer.menu_backend (default fast); reference is exact rollback.
    Compiler/executor publication and complete-game evaluation are separate gates.
    This function does not train, publish, run games or change V0Agent economics.
    """
    return _epoch(scorer, state, prefix_plan, generator=generator, mode=mode,
                  temperature=temperature, projection_kwargs=projection_kwargs, backend=backend)


def replay_epoch(scorer: V0Scorer, state: State, prefix_plan: Prefix | Plan, trace: dict, *,
                 projection_kwargs=None, verify_logps: bool = True, backend: str | None = None) -> EpochResult:
    """Differentiable exact class-PATH replay with freshly recomputed prefixes."""
    if not isinstance(trace, dict) or trace.get('schema') != TRACE_SCHEMA:
        raise ValueError('V0 trace required; legacy program likelihood is not native V0')
    return _epoch(scorer, state, prefix_plan, mode=trace.get('mode'), temperature=trace.get('temperature'),
                  projection_kwargs=projection_kwargs, replay=trace, verify_logps=verify_logps, backend=backend)
