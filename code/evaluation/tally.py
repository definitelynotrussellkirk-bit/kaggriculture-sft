#!/usr/bin/env python3
"""Tally paired games: W-D-L, point share, world-clustered share and margin with Student-t 95% intervals.

usage: python3 tally.py CHALLENGER GAME_JSON [GAME_JSON ...]
Each game.json is a league_game.py output. A world counts as complete when both seat orders are present.
Written for this repo; it reproduces the reported numbers from the shipped records (see README.md).
"""
import json
import math
import sys
from collections import defaultdict

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201,
        12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060,
        30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t975(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:
        return T975.get(df) or T975[min(k for k in T975 if k >= df)] if df <= 120 else 1.96


def ci(xs):
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, float("nan"), float("nan")
    h = t975(n - 1) * math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) / math.sqrt(n)
    return mu, mu - h, mu + h


def main(challenger, paths):
    worlds = defaultdict(dict)
    w = d = l = 0
    for p in paths:
        g = json.load(open(p))["game"]
        if challenger not in g["seats"] or g["status"] != ["DONE", "DONE"]:
            continue
        me = g["seats"].index(challenger)
        s = g["score"][me]
        w += s == 1.0
        d += s == 0.5
        l += s == 0.0
        worlds[g["world_seed"]][me] = (s, g["cash"][me] - g["cash"][1 - me])
    full = [v for v in worlds.values() if len(v) == 2]
    share = ci([(v[0][0] + v[1][0]) / 2 for v in full]) if full else None
    margin = ci([(v[0][1] + v[1][1]) / 2 for v in full]) if full else None
    games = w + d + l
    print(json.dumps({"challenger": challenger, "games": games, "W-D-L": [w, d, l],
                      "point_share": round((w + 0.5 * d) / games, 3) if games else None,
                      "complete_worlds": len(full),
                      "world_share_t95": [round(x, 3) for x in share] if share else None,
                      "margin_per_world_t95": [round(x, 1) for x in margin] if margin else None}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
