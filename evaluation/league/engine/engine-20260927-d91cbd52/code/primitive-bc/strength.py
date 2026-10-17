"""Bradley-Terry teacher strength for primitive-BC -> STRENGTH.json.

Kaggle: one strength per (team, archive date), fitted on the complete, decisive
(rewards differ) TRAIN-split episode headers only (data.list_members(split="train"),
sealed ids removed by name; no replay steps are read). Model, as in
pbc-arch-review/imitation/strength.py: logistic regression of "seat 0 wins" on
+1 (seat-0 key) / -1 (seat-1 key) indicators with an intercept (seat advantage), L2
penalty C=1.0 (sklearn LogisticRegression, lbfgs). A team-level fit is the fallback
for a (team, date) absent from the train split, then 0.0 (the penalty's prior mean).

Deciles and the strength filter use the distribution of team-day strength over the
train perspectives (two per complete episode): decile d = number of the 10%..90%
quantiles <= s; strength_top=f keeps s >= the (1-f) quantile.

Team vocabulary: id 0 = unknown; ids 1.. = train teams with >= min_team_perspectives
train perspectives, by descending count then name. Pretraining actors of batch
directories (RESULTS.jsonl index; split as pbc-pretrain/stream_pretrain.py) are
appended as "pretrain:<actor>" (PASS excluded; it is never a teacher), with their own
per-actor fit and deciles over the non-PASS pretraining train perspectives.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_PATH = HERE / "STRENGTH.json"
SCHEMA = "primitive-bc-strength-v1"
PRETRAIN_SPLIT_SALT = b"farm-pbc-local-pretrain-split-v1:"   # == stream_pretrain.SPLIT_SALT


# ---------------------------------------------------------------- pretraining split
def pretrain_key(path):
    path = Path(path)
    return f"{path.parent.name}/{path.name}"


def pretrain_split_bucket(path):
    """Physical-game bucket; both seats of one file share it (stream_pretrain.split_bucket)."""
    key = pretrain_key(path).encode()
    return int.from_bytes(hashlib.sha256(PRETRAIN_SPLIT_SALT + key).digest()[:8], "big") % 1000


def pretrain_results(directory):
    """Completed games of a pretraining batch from its RESULTS.jsonl index (no gzip opened)."""
    directory = Path(directory).resolve()
    rows = []
    with (directory / "RESULTS.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("status") != "COMPLETE":
                continue
            path = Path(row["path"]).resolve()
            if path.parent != directory or len(row.get("actors", [])) != 2:
                raise ValueError(f"foreign pretraining result row: {row.get('path')}")
            rows.append(row)
    rows.sort(key=lambda r: r["path"])
    summary = directory / "SUMMARY.json"
    if summary.is_file() and json.loads(summary.read_text())["physical_games_complete"] != len(rows):
        raise ValueError(f"{directory}: RESULTS.jsonl and SUMMARY.json disagree")
    return rows


# ---------------------------------------------------------------- fitting
def fit_bt(games, C=1.0):
    """games: [(key0, key1, seat0_won)] -> ({key: strength}, intercept, {key: games})."""
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    keys = sorted({g[0] for g in games} | {g[1] for g in games}, key=repr)
    ix = {k: i for i, k in enumerate(keys)}
    n = len(games)
    rows = np.repeat(np.arange(n), 2)
    cols = np.array([ix[k] for a, b, _ in games for k in (a, b)])
    vals = np.tile(np.array([1.0, -1.0]), n)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(keys)))
    y = np.array([bool(w) for _, _, w in games])
    model = LogisticRegression(C=C, fit_intercept=True, max_iter=5000).fit(X, y)
    counts = Counter(k for a, b, _ in games for k in (a, b))
    return {k: float(v) for k, v in zip(keys, model.coef_[0])}, float(model.intercept_[0]), counts


def _quantiles(values):
    v = np.sort(np.asarray(values, np.float64))
    return [float(np.quantile(v, p / 100)) for p in range(101)] if len(v) else [0.0] * 101


def _header_task(task):
    import data
    archive, member, date, eid = task
    try:
        h = data.read_header(archive, member)
    except Exception as error:  # reported, never silently dropped
        return {"episode_id": eid, "date": date, "error": repr(error)}
    return {"episode_id": eid, "date": date, "teams": (h.get("info") or {}).get("TeamNames"),
            "rewards": h.get("rewards"), "statuses": h.get("statuses")}


def kaggle_headers(archive_paths=None, *, workers=8, split="train"):
    import data
    tasks = data.list_members(archive_paths, split=split)
    if workers <= 1:
        return [_header_task(t) for t in tasks]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(_header_task, tasks, chunksize=32)


def build(archive_paths=None, *, workers=8, min_team_perspectives=20, pretrain_dirs=(), C=1.0):
    """STRENGTH.json payload from train-split headers (and pretraining batch indexes)."""
    import data
    headers = kaggle_headers(archive_paths, workers=workers)
    errors = [h for h in headers if "error" in h]
    complete = [h for h in headers if "error" not in h and h["statuses"] == ["DONE", "DONE"]
                and h["teams"] and len(h["teams"]) == 2 and h["rewards"] and len(h["rewards"]) == 2
                and None not in h["rewards"]]
    decisive = [h for h in complete if h["rewards"][0] != h["rewards"][1]]
    td_games = [((h["teams"][0], h["date"]), (h["teams"][1], h["date"]), h["rewards"][0] > h["rewards"][1])
                for h in decisive]
    team_day, b_td, n_td = fit_bt(td_games, C)
    team, b_t, n_t = fit_bt([(a[0], b[0], w) for a, b, w in td_games], C)
    per_team = Counter(t for h in complete for t in h["teams"])
    vocab = [t for t, n in sorted(per_team.items(), key=lambda kv: (-kv[1], kv[0])) if n >= min_team_perspectives]
    persp = [team_day.get((t, h["date"]), team.get(t, 0.0)) for h in complete for t in h["teams"]]
    out = {"schema": SCHEMA,
           "source": {"archives": [str(p) for p in (archive_paths or data.default_archives())],
                      "split": "train", "split_salt": data.SPLIT_SALT.decode(),
                      "sealed_excluded": len(data.sealed_ids())},
           "method": ("Bradley-Terry: sklearn LogisticRegression(C=%g, lbfgs, intercept) of seat-0 win on "
                      "+1/-1 key indicators; decisive complete train-split headers only" % C),
           "counts": {"train_headers": len(headers), "header_errors": len(errors), "complete": len(complete),
                      "decisive": len(decisive), "draws": len(complete) - len(decisive),
                      "team_days": len(team_day), "teams": len(team), "train_perspectives": len(persp)},
           "seat0_intercept": {"team_day": b_td, "team": b_t},
           "team_day": {t: {d: round(team_day[(t, d)], 6) for (tt, d) in sorted(team_day) if tt == t}
                        for t in sorted({k[0] for k in team_day})},
           "team_day_games": {f"{t}|{d}": n for (t, d), n in sorted(n_td.items())},
           "team": {t: round(v, 6) for t, v in sorted(team.items())},
           "quantiles": _quantiles(persp),
           "min_team_perspectives": min_team_perspectives,
           "team_vocab": ["<unknown>"] + vocab,
           "pretrain": {}}
    for directory in pretrain_dirs:
        rows = [r for r in pretrain_results(directory) if pretrain_split_bucket(r["path"]) < 900]
        games = [(r["actors"][0], r["actors"][1], r["outcomes"][0] == "WIN") for r in rows
                 if r["outcomes"][0] != "DRAW"]
        actor, b_a, n_a = fit_bt(games, C)
        p = [actor[a] for r in rows for a in r["actors"] if a != "PASS"]
        name = Path(directory).resolve().name
        out["pretrain"][name] = {"directory": str(Path(directory).resolve()), "train_games": len(rows),
                                 "decisive": len(games), "seat0_intercept": b_a,
                                 "actor": {a: round(v, 6) for a, v in sorted(actor.items())},
                                 "actor_games": dict(sorted(n_a.items())), "quantiles": _quantiles(p)}
        for a in sorted(actor):
            if a != "PASS" and f"pretrain:{a}" not in out["team_vocab"]:
                out["team_vocab"].append(f"pretrain:{a}")
    return out


# ---------------------------------------------------------------- lookup
class Strength:
    """Read-only view of STRENGTH.json used by data.py."""

    def __init__(self, payload):
        if payload.get("schema") != SCHEMA:
            raise ValueError("STRENGTH.json schema mismatch")
        self.payload = payload
        self.team_day = payload["team_day"]
        self.team = payload["team"]
        self.quantiles = payload["quantiles"]
        self.vocab = payload["team_vocab"]
        self.team_index = {t: i for i, t in enumerate(self.vocab)}
        self.pretrain = payload.get("pretrain", {})

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; build it with `python strength.py build`")
        return cls(json.loads(path.read_text()))

    def lookup(self, team, date):
        """(strength, source) for a Kaggle team on an archive date."""
        by_day = self.team_day.get(team)
        if by_day is not None and date in by_day:
            return by_day[date], "team_day"
        if team in self.team:
            return self.team[team], "team"
        return 0.0, "prior"

    @staticmethod
    def _decile(quantiles, s):
        return min(9, bisect.bisect_right(quantiles[10:100:10], s))

    @staticmethod
    def _threshold(quantiles, top):
        if top is None:
            return -np.inf
        if not 0 < top <= 1:
            raise ValueError("strength_top must be in (0, 1]")
        return quantiles[int(round(100 * (1 - top)))] if top < 1 else -np.inf

    def decile(self, s):
        return self._decile(self.quantiles, s)

    def threshold(self, top):
        return self._threshold(self.quantiles, top)

    def team_id(self, team):
        return self.team_index.get(team, 0)

    def pretrain_actor(self, batch, actor):
        """(strength, decile, team_id, threshold_fn) for a pretraining actor of batch dir name."""
        part = self.pretrain.get(batch)
        if part is None or actor not in part["actor"]:
            return 0.0, 0, self.team_index.get(f"pretrain:{actor}", 0), None
        s = part["actor"][actor]
        return s, self._decile(part["quantiles"], s), self.team_index.get(f"pretrain:{actor}", 0), part["quantiles"]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--min-team-perspectives", type=int, default=20)
    b.add_argument("--pretrain", nargs="*", default=["/mnt/ai/scratch/farmNEAT/pbc-pretrain-20260923/batch-2000-v1"])
    b.add_argument("--out", default=str(DEFAULT_PATH))
    args = ap.parse_args(argv)
    if args.workers > 8:
        raise SystemExit("at most 8 worker processes")
    payload = build(workers=args.workers, min_team_perspectives=args.min_team_perspectives,
                    pretrain_dirs=[d for d in args.pretrain if Path(d).is_dir()])
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=False, ensure_ascii=False) + "\n")
    s = Strength(payload)
    top = sorted(((v, t, d) for t, by in payload["team_day"].items() for d, v in by.items()), reverse=True)[:10]
    print(json.dumps({"counts": payload["counts"], "vocab": len(payload["team_vocab"]),
                      "quantiles_10": payload["quantiles"][::10], "threshold_top0.6": s.threshold(0.6),
                      "top_team_days": [[t, d, round(v, 3)] for v, t, d in top],
                      "pretrain": {k: {"actor": v["actor"], "train_games": v["train_games"]}
                                   for k, v in payload["pretrain"].items()}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    for _p in (HERE.parents[3] / "src", HERE):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    main()
