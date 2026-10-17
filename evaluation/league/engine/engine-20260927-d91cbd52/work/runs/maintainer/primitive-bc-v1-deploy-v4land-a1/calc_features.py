"""Saturation-aware comparison calculations for lineage B (prototype, review scratch only).

Pure function of ONE seat's actor observation (no history, no own past action, no rival
private state, no future shop identity). Built on theorem_features.compute (engine-exact
price curves, visible demand law, per-tile season units) plus a projected market balance:

  E[p]  = (inv[p]-I0) + S_own[p] + S_riv[p] - D_vis[p] - D_new[p]
          projected end-of-game inventory offset if all committed supply is sold and the
          visible + expected town demand is consumed (WHEAT supply net of animal feed).
  P*[p] = price(I0 + E[p])            projected clearing quote of the marginal unit.
  M_r(n)= sum_{j in [(n-1)u, nu)} price(I0+E+j)   receipts of the n-th ADDITIONAL producer r
          (u = its season units if started today), net_r(n) = M_r(n) - input cost.
  n*_r  = #additional producers r with net_r(n) >= 0 (cap 40).

Output: prod [9, N_PROD] (row = product; producer columns filled for the 8 producing
products, FERTILIZER row has has_producer=0) and glob [N_GLOB]; signed-log scaled
slog(x, s) = sign(x) * log1p(|x|/s), float32, finite.
"""
from __future__ import annotations

import math

import numpy as np

import theorem_features as TF

PRODUCTS = TF.PRODUCTS
PRODUCER_OF = {"WHEAT": "WHEAT", "CARROT": "CARROT", "TOMATO": "TOMATO", "STRAWBERRY": "STRAWBERRY",
               "MELON": "MELON", "EGG": "GOOSE", "MILK": "COW", "WOOL": "SHEEP"}
ANIMALS = ("GOOSE", "COW", "SHEEP")
N_STAR_CAP = 40
K_SELL_CAP = 100

# (name, scale for slog)
PROD_COLS = (
    ("inv_offset", 10.0), ("sell_quote", 10.0), ("visible_rate", 1.0), ("demand_rest_visible", 10.0),
    ("demand_rest_expected_new", 10.0), ("demand24_visible", 1.0), ("supply_own", 10.0),
    ("supply_rival", 10.0), ("projected_offset", 10.0), ("projected_quote", 10.0),
    ("sell_premium", 10.0), ("k_sell_above_projected", 1.0),
    ("has_producer", 1.0), ("units_if_started", 1.0), ("input_cost", 100.0), ("first_output_days", 1.0),
    ("marginal_receipts_1", 100.0), ("marginal_net_1", 100.0), ("marginal_net_5", 100.0),
    ("n_star", 1.0),
)
GLOB_COLS = (
    ("days_remaining", 1.0), ("shops_visible", 1.0), ("shop_unlocks_remaining", 1.0),
    ("steps_to_next_shop_unlock", 10.0), ("liquidation_lead", 1000.0), ("own_shed_liquidation", 100.0),
    ("own_pipeline_value", 1000.0), ("rival_pipeline_value", 1000.0), ("projected_margin", 1000.0),
    ("own_free_tiles", 1.0), ("own_animals", 1.0), ("feed_cover", 0.1), ("shed_room", 10.0),
    ("dayend_overflow", 1.0), ("next_land_price", 1000.0), ("land_affordable", 1.0),
)
N_PROD, N_GLOB = len(PROD_COLS), len(GLOB_COLS)
_PC = {n: i for i, (n, _) in enumerate(PROD_COLS)}
_GC = {n: i for i, (n, _) in enumerate(GLOB_COLS)}
PROD_SCALE = np.asarray([s for _, s in PROD_COLS], np.float32)
GLOB_SCALE = np.asarray([s for _, s in GLOB_COLS], np.float32)


def slog(x, scale):
    return (np.sign(x) * np.log1p(np.abs(x) / scale)).astype(np.float32)


def _k_sell(cv, inv, threshold):
    """max k <= K_SELL_CAP with quote(inv+k-1) >= threshold (quotes are non-increasing)."""
    lo, hi = 0, K_SELL_CAP
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cv.quote(inv + mid - 1) >= threshold:
            lo = mid
        else:
            hi = mid - 1
    return lo


def raw(actor_obs, config=None, tf=None):
    """Unscaled (prod [9, N_PROD], glob [N_GLOB]) float64 arrays; tf = TF.compute output if cached."""
    tf = TF.compute(actor_obs, config) if tf is None else tf
    P, S, G = TF._P, TF._S, TF._G
    prod_tf, struct_tf, glob_tf = tf["product"], tf["structure"], tf["global"]
    curves = TF._curves(TF._market_params(actor_obs, config))
    inventory = actor_obs["market"]["inventory"]
    days = float(glob_tf[G["days_remaining"]])
    own_animals = sum(float(struct_tf[TF._RIDX[a], S["own_count"]]) for a in ANIMALS)
    riv_animals = sum(float(struct_tf[TF._RIDX[a], S["rival_count"]]) for a in ANIMALS)
    out = np.zeros((len(PRODUCTS), N_PROD), np.float64)
    own_value = riv_value = 0.0
    for i, p in enumerate(PRODUCTS):
        cv = curves[i]
        I0 = int(cv.p["I0"])
        inv = int(inventory[p])
        row, t = out[i], prod_tf[i]
        held = float(t[P["shed_qty"]] + t[P["carried_qty"]])
        if p == "FERTILIZER":
            s_own = held + float(t[P["own_field_stock"]])
            s_riv = float(t[P["rival_field_stock"]])
            d_vis = d_new = 0.0
        else:
            s_own = held + float(t[P["own_season_units"]])
            s_riv = float(t[P["rival_season_units"]])
            d_vis = float(t[P["demand_rest_visible"]])
            d_new = float(t[P["demand_rest_expected_new"]])
        if p == "WHEAT":
            s_own -= own_animals * days
            s_riv -= riv_animals * days
        E = (inv - I0) + s_own + s_riv - d_vis - d_new
        Ei = I0 + int(round(E))
        q = float(cv.quote(inv))
        pstar = float(cv.quote(Ei))
        row[_PC["inv_offset"]] = inv - I0
        row[_PC["sell_quote"]] = q
        row[_PC["visible_rate"]] = t[P["visible_rate"]]
        row[_PC["demand_rest_visible"]] = d_vis
        row[_PC["demand_rest_expected_new"]] = d_new
        row[_PC["demand24_visible"]] = t[P["demand24_visible"]]
        row[_PC["supply_own"]] = s_own
        row[_PC["supply_rival"]] = s_riv
        row[_PC["projected_offset"]] = E
        row[_PC["projected_quote"]] = pstar
        row[_PC["sell_premium"]] = q - pstar
        row[_PC["k_sell_above_projected"]] = _k_sell(cv, inv, pstar)
        own_value += max(0.0, s_own) * pstar
        riv_value += max(0.0, s_riv) * pstar
        producer = PRODUCER_OF.get(p)
        if producer is None:
            continue
        r = TF._RIDX[producer]
        units = int(struct_tf[r, S["season_units_watered"]])
        cost = float(struct_tf[r, S["season_input_cost"]])
        row[_PC["has_producer"]] = 1.0
        row[_PC["units_if_started"]] = units
        row[_PC["input_cost"]] = cost
        row[_PC["first_output_days"]] = struct_tf[r, S["first_output_days"]]
        if units <= 0:
            row[_PC["marginal_receipts_1"]] = 0.0
            row[_PC["marginal_net_1"]] = row[_PC["marginal_net_5"]] = -cost
            continue
        m1 = float(cv.total(Ei, Ei + units))
        row[_PC["marginal_receipts_1"]] = m1
        row[_PC["marginal_net_1"]] = m1 - cost
        row[_PC["marginal_net_5"]] = float(cv.total(Ei + 4 * units, Ei + 5 * units)) - cost
        n = 0
        while n < N_STAR_CAP and cv.total(Ei + n * units, Ei + (n + 1) * units) - cost >= 0:
            n += 1
        row[_PC["n_star"]] = n
    g = np.zeros(N_GLOB, np.float64)
    own = actor_obs["farms"][actor_obs["player"]]
    rival = actor_obs["farms"][1 - actor_obs["player"]]
    wheat = prod_tf[PRODUCTS.index("WHEAT")]
    own_wheat = float(wheat[P["shed_qty"]] + wheat[P["carried_qty"]] + wheat[P["own_season_units"]])
    tile = tf["tile_own"]
    free = float(tile[:, TF._T["plantable"]].sum() + tile[:, TF._T["structure_free"]].sum())
    g[_GC["days_remaining"]] = days
    g[_GC["shops_visible"]] = glob_tf[G["shops_visible"]]
    g[_GC["shop_unlocks_remaining"]] = glob_tf[G["shop_unlocks_remaining"]]
    g[_GC["steps_to_next_shop_unlock"]] = glob_tf[G["steps_to_next_shop_unlock"]]
    g[_GC["liquidation_lead"]] = glob_tf[G["liquidation_lead"]]
    g[_GC["own_shed_liquidation"]] = glob_tf[G["own_shed_liquidation"]]
    g[_GC["own_pipeline_value"]] = own_value
    g[_GC["rival_pipeline_value"]] = riv_value
    g[_GC["projected_margin"]] = float(own["money"]) + own_value - float(rival["money"]) - riv_value
    g[_GC["own_free_tiles"]] = free
    g[_GC["own_animals"]] = own_animals
    g[_GC["feed_cover"]] = own_wheat / max(1.0, own_animals * days) if own_animals else 0.0
    g[_GC["shed_room"]] = glob_tf[G["shed_room"]]
    g[_GC["dayend_overflow"]] = glob_tf[G["dayend_overflow"]]
    g[_GC["next_land_price"]] = glob_tf[G["own_next_land_price"]]
    g[_GC["land_affordable"]] = glob_tf[G["own_land_affordable"]]
    return out, g


def compute(actor_obs, config=None, tf=None):
    """Scaled float16-exact float32 arrays: {'calc_prod': [9, N_PROD], 'calc_glob': [N_GLOB]}."""
    prod, glob = raw(actor_obs, config, tf)
    cp = slog(prod, PROD_SCALE).astype(np.float16)
    cg = slog(glob, GLOB_SCALE).astype(np.float16)
    if not (np.isfinite(cp).all() and np.isfinite(cg).all()):
        raise ValueError("non-finite calculation")
    return {"calc_prod": cp, "calc_glob": cg}


def warm(config=None):
    TF.warm(config)
