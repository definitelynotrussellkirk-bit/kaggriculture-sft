#!/usr/bin/env python3
"""Generate FEATURES.md (every model input, every column, how each is computed) from the shipped code.

    python3 tools/list_features.py            # writes FEATURES.md at the repo root
    python3 tools/list_features.py --check    # regenerates in memory and checks every #Lnn link in FEATURES.md
    python3 tools/list_features.py --extract-docs DIR   # refresh the embedded column definitions from COLUMNS-*.md

Needs numpy (the v8 feature modules are imported to read their column lists). Column names, widths, transforms,
scales, inline comments, code expressions and line numbers all come from the code below MODEL1/MODEL2; the per-column
definitions of the v8 families come from the family authors' COLUMNS-*.md tables, embedded at the end of this file
and used only where the column name matches the code exactly.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import pprint
import re
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL1 = REPO / "model1-b-even/code/trainer-v6/primitive-bc-v6-20260927"
PBC = MODEL1 / "work/runs/maintainer/primitive-bc"
SRC = MODEL1 / "src"
MODEL2 = REPO / "model2-n-flow/code"
DEPLOY = REPO / "submissions/n1-main-sprinkle/work/runs/maintainer/primitive-bc-v1-deploy-v9-decode"
M1_CONFIG = REPO / "model1-b-even/records/3-b-even-phase2-4090/RUN-17.json"
M2_RUN = REPO / "model2-n-flow/records/wash/RUN-0056259.json"
OUT = REPO / "FEATURES.md"

for p in (str(PBC), str(SRC), str(MODEL2)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _reading_copies():
    """{as-run path: code/ reading copy} from code/MANIFEST.tsv, kept only where the bytes match."""
    import hashlib
    out = {}
    man = REPO / "code/MANIFEST.tsv"
    if not man.exists():
        return out
    for line in man.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        copy, sha, as_run = parts[0], parts[1], parts[2]
        a, c = REPO / as_run, REPO / "code" / copy
        if a.exists() and c.exists() and hashlib.sha256(a.read_bytes()).hexdigest() == sha == \
                hashlib.sha256(c.read_bytes()).hexdigest():
            out[as_run] = "code/" + copy
    return out


READING = _reading_copies()
COLS_LINK = READING.get("model2-n-flow/code/COLUMNS.json", "model2-n-flow/code/COLUMNS.json")


# ----------------------------------------------------------------------------- source helpers
class Src:
    """One source file: AST, inline comments by line, symbol -> line."""
    _cache: dict = {}

    def __new__(cls, path):
        path = Path(path)
        if path not in cls._cache:
            obj = super().__new__(cls)
            obj.path = path
            obj.text = path.read_text()
            obj.lines = obj.text.splitlines()
            obj.tree = ast.parse(obj.text)
            obj.comments = {}
            for tok in tokenize.generate_tokens(io.StringIO(obj.text).readline):
                if tok.type == tokenize.COMMENT:
                    obj.comments[tok.start[0]] = tok.string.lstrip("#").strip()
            cls._cache[path] = obj
        return cls._cache[path]

    @property
    def rel(self):
        r = self.path.relative_to(REPO).as_posix()
        return READING.get(r, r)

    def node(self, qualname):
        """FunctionDef/ClassDef for 'name' or 'Class.method', or an Assign/AnnAssign for a module-level NAME."""
        parts = qualname.split(".")
        body = self.tree.body
        found = None
        for i, part in enumerate(parts):
            found = None
            for n in body:
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part:
                    found = n
                elif i == len(parts) - 1 and isinstance(n, (ast.Assign, ast.AnnAssign)):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    for t in targets:
                        names = [e.id for e in (t.elts if isinstance(t, ast.Tuple) else [t]) if isinstance(e, ast.Name)]
                        if part in names:
                            found = n
                if found is not None:
                    break
            if found is None:
                raise KeyError(f"{self.rel}: symbol {qualname!r} not found")
            body = getattr(found, "body", [])
        return found

    def line(self, qualname):
        return self.node(qualname).lineno

    def link(self, qualname, label=None):
        return f"[`{label or qualname}`]({self.rel}#L{self.line(qualname)})"

    def file_link(self, label=None):
        return f"[`{label or self.path.name}`]({self.rel})"


def P(name):
    return Src(PBC / name)


OBS = Src(SRC / "farm_neat/policy/observation.py")
ENGINE = Src(SRC / "farm_neat/game/kaggriculture_engine_reference.py")
MARKET = Src(SRC / "farm_neat/economy/heuristic/market.py")
NFM = Src(MODEL2 / "nf_model.py")
NFF = Src(MODEL2 / "nf_format.py")
STEP_SEARCH = Src(DEPLOY / "step_search.py")


def md(s):
    """Escape a code expression for a markdown table cell."""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s.replace("|", "\\|")


def code(s, limit=140):
    s = md(s)
    if len(s) > limit:
        s = s[:limit - 1] + "…"
    return f"`{s}`" if s else ""


# ----------------------------------------------------------------------------- array-fill extraction
def _eval(node, env):
    try:
        return eval(compile(ast.Expression(node), "<idx>", "eval"), dict(env))
    except Exception:
        return None


class Resolver:
    """Expression text with single-assignment locals expanded one level and loop variables bound to constants."""

    def __init__(self, fn):
        counts, exprs = {}, {}
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
        for n in ast.walk(fn):
            targets = []
            if isinstance(n, ast.Assign):
                targets = [(t, n.value) for t in n.targets]
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
                targets = [(n.target, None)]
            elif isinstance(n, ast.For):
                targets = [(n.target, None)]
            for t, v in targets:
                if isinstance(t, ast.Name):
                    counts[t.id] = counts.get(t.id, 0) + 1
                    exprs[t.id] = v
                elif isinstance(t, ast.Tuple):
                    for k, e in enumerate(t.elts):
                        if isinstance(e, ast.Name):
                            counts[e.id] = counts.get(e.id, 0) + 1
                            if isinstance(v, ast.Tuple) and len(v.elts) == len(t.elts):
                                exprs[e.id] = v.elts[k]
                            elif v is not None:
                                exprs[e.id] = ast.Subscript(value=v, slice=ast.Constant(k), ctx=ast.Load())
                            else:
                                exprs[e.id] = None
        mutated = {n.value.id for n in ast.walk(fn) if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)
                   and isinstance(n.value, ast.Name)}
        self.locals = {k: v for k, v in exprs.items()
                       if counts.get(k) == 1 and v is not None and k not in params and k not in mutated}

    def text(self, node, bind=None):
        bind = bind or {}
        loc = self.locals

        class T(ast.NodeTransformer):
            def __init__(self, depth):
                self.depth = depth

            def visit_Name(self, n):
                if not isinstance(n.ctx, ast.Load):
                    return n
                if n.id in bind and isinstance(bind[n.id], (str, int, float)):
                    return ast.Constant(bind[n.id])
                if self.depth and n.id in loc:
                    return T(self.depth - 1).visit(_copy(loc[n.id]))
                return n

        return ast.unparse(T(1).visit(_copy(node)))


def _copy(node):
    return ast.parse(ast.unparse(node), mode="eval").body


def fills(src, qualname, var, env):
    """Every write to `var[...]` inside a function, in source order, with loops over known tuples expanded.

    Returns [(column index or None, index source, value source, condition text, slice end)]."""
    fn = src.node(qualname)
    res = Resolver(fn)
    out = []

    def index_of(sub, bind):
        s = sub.slice
        if isinstance(s, ast.Tuple):
            s = s.elts[-1]
        if isinstance(s, ast.Slice):
            lo = 0 if s.lower is None else _eval(s.lower, bind)
            hi = None if s.upper is None else _eval(s.upper, bind)
            return ("slice", lo, hi, ast.unparse(sub.slice))
        return ("idx", _eval(s, bind), None, ast.unparse(sub.slice))

    def add(target, value, bind, cond):
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == var):
            return False
        kind, a, b, text = index_of(target, bind)
        lb = {k: v for k, v in bind.items() if k in loopvars}
        if kind == "slice" and isinstance(value, (ast.Tuple, ast.List)) and a is not None:
            for j, e in enumerate(value.elts):
                out.append((a + j, f"{text}[{j}]", res.text(e, lb), cond, None))
        elif kind == "slice":
            out.append((a, text, res.text(value, lb) if value is not None else "?", cond, b))
        else:
            out.append((a, text, res.text(value, lb) if value is not None else "?", cond, None))
        return True

    loopvars = set()

    def walk(stmts, bind, cond):
        for st in stmts:
            if isinstance(st, ast.Assign):
                for t in st.targets:
                    if isinstance(t, ast.Tuple) and isinstance(st.value, ast.Tuple) and len(t.elts) == len(st.value.elts):
                        for te, ve in zip(t.elts, st.value.elts):
                            add(te, ve, bind, cond)
                    else:
                        add(t, st.value, bind, cond)
            elif isinstance(st, ast.For):
                it = st.iter
                seq = None
                if isinstance(it, ast.Call) and getattr(it.func, "id", None) == "enumerate":
                    seq = _eval(it.args[0], bind)
                    seq = None if seq is None else list(enumerate(seq))
                else:
                    seq = _eval(it, bind)
                    seq = None if seq is None else list(seq)
                tnames = [e.id for e in (st.target.elts if isinstance(st.target, ast.Tuple) else [st.target])
                          if isinstance(e, ast.Name)]
                if seq is None or len(seq) > 64:
                    walk(st.body, bind, cond + [f"for {ast.unparse(st.target)} in {ast.unparse(it)}"])
                    continue
                for item in seq:
                    b2 = dict(bind)
                    vals = item if isinstance(item, tuple) and len(tnames) > 1 else (item,)
                    for n, v in zip(tnames, vals):
                        b2[n] = v
                        loopvars.add(n)
                    walk(st.body, b2, cond)
            elif isinstance(st, ast.If):
                t = ast.unparse(st.test)
                walk(st.body, bind, cond + [t])
                walk(st.orelse, bind, cond + [f"not ({t})"])
            elif isinstance(st, (ast.With, ast.Try)):
                walk(st.body, bind, cond)

    walk(fn.body, dict(env), [])
    return out


def list_elements(src, qualname, match):
    """Elements of the first list literal inside `qualname` for which match(node, parent) is true."""
    fn = src.node(qualname)
    res = Resolver(fn)
    for n in ast.walk(fn):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, ast.List) and match(child, n):
                return [res.text(e) for e in child.elts]
    raise KeyError(f"{src.rel}:{qualname}: list literal not found")


def column_comments(src, name):
    """{column name: inline comment} for a module-level tuple of string literals / (name, ...) tuples, with group
    comments (a comment on its own line inside the tuple) carried onto the following names."""
    node = src.node(name)
    value = node.value
    out, group = {}, {}
    if not isinstance(value, (ast.Tuple, ast.List)):
        return out, group
    lo, hi = value.lineno, value.end_lineno
    pending = None
    elts = sorted(value.elts, key=lambda e: (e.lineno, e.col_offset))
    k = 0
    for ln in range(lo, hi + 1):
        line = src.lines[ln - 1].strip()
        if line.startswith("#"):
            pending = line.lstrip("#").strip()
        while k < len(elts) and elts[k].lineno == ln:
            e = elts[k]
            nm = e.value if isinstance(e, ast.Constant) and isinstance(e.value, str) else (
                e.elts[0].value if isinstance(e, ast.Tuple) and e.elts and isinstance(e.elts[0], ast.Constant) else None)
            if nm is not None:
                c = src.comments.get(e.end_lineno)
                if c and sum(1 for x in elts if x.end_lineno == e.end_lineno) == 1:
                    out[nm] = re.sub(r"^\d+(?:-\d+)?\s+", "", c)
                if pending:
                    group[nm] = pending
            k += 1
    return out, group


# ----------------------------------------------------------------------------- column definitions from the docs
def _doc_names(cell, index_cell, code_names):
    """Code column names a doc row describes: explicit names (a / _b, {x,y}, A..B ranges, prefix inheritance) or,
    for tables keyed by index, code_names[index]."""
    names = []
    cell = cell.replace("`", "")
    parts = [p.strip() for p in re.split(r",\s*(?![^{]*\})|\s+/\s+", cell) if p.strip()]
    first = None
    prefix = ""
    for part in parts:
        if first and part.startswith("_") and "_" in first:
            part = first.rsplit("_", 1)[0] + part
        if ":" in part:
            prefix = part.split(":", 1)[0] + ":"
        elif prefix and part.upper() == part:
            part = prefix + part
        m = re.search(r"\{([^{}]*)\}", part)
        if m and "," in m.group(1):
            for o in m.group(1).split(","):
                names.append(part[:m.start()] + o.strip() + part[m.end():])
        elif m:
            rx = re.compile(re.escape(part[:m.start()]) + r"[A-Za-z0-9_]+" + re.escape(part[m.end():]) + "$")
            names.extend(c for c in code_names if rx.match(c) or rx.match(c.split(".", 1)[-1]))
        elif ".." in part:
            a, b = part.split("..", 1)
            pre = a.split(":", 1)[0] + ":" if ":" in a else ""
            b = pre + b if pre and not b.startswith(pre) else b
            if a in code_names and b in code_names:
                i, j = code_names.index(a), code_names.index(b)
                names.extend(code_names[i:j + 1])
        else:
            names.append(part)
        first = first or part
    if index_cell is not None:
        idx = []
        for chunk in re.split(r"[,\s]+", index_cell.strip()):
            m = re.fullmatch(r"(\d+)(?:-(\d+))?", chunk)
            if m:
                a = int(m.group(1))
                idx.extend(range(a, int(m.group(2) or a) + 1))
        if idx and all(i < len(code_names) for i in idx) and not any(n in code_names for n in names):
            names = [code_names[i] for i in idx]
    return names


def doc_definitions(family, code_names):
    """{code column: definition} for one v8 family from the embedded COLUMNS-*.md rows."""
    out = {}
    for names_cell, index_cell, definition in DOC_DEFS.get(family, ()):
        for n in _doc_names(names_cell, index_cell, code_names):
            for key in (n, "m." + n, "g." + n):
                if key in code_names and key not in out:
                    out[key] = definition
    return out


def extract_docs(docs_dir):
    """Parse COLUMNS-<family>.md tables -> {family: [(names cell, index cell, definition)]}."""
    files = {"dusk": "COLUMNS-dusk.md", "cohorts": "COLUMNS-cohorts.md", "flags": "COLUMNS-flags.md",
             "gaps": "COLUMNS-gaps.md", "tileclock": "COLUMNS-tileclock.md", "lattice": "COLUMNS-lattice.md"}
    data = {}
    for fam, name in files.items():
        rows, head = [], None
        for line in (Path(docs_dir) / name).read_text().splitlines():
            s = line.strip()
            if not s.startswith("|"):
                head = None
                continue
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", s.strip("|"))]
            low = [c.lower() for c in cells]
            if any(c in ("column", "name") for c in low) and any(c.startswith("definition") for c in low):
                head = (low.index("column") if "column" in low else low.index("name"),
                        next(i for i, c in enumerate(low) if c.startswith("definition")),
                        next((i for i, c in enumerate(low) if c in ("#", "c", "col")), None))
                continue
            if head is None or set("".join(cells)) <= set("-: ") or len(cells) <= max(h for h in head if h is not None):
                continue
            ni, di, ii = head
            rows.append((cells[ni], cells[ii] if ii is not None else None, cells[di]))
        data[fam] = rows
    return data


# ----------------------------------------------------------------------------- the inputs, read from code
def load():
    import numpy as np  # noqa: F401  (v8 modules need it)
    import calc_features
    import codec
    import data
    import flip
    import job_eta
    import prefix
    import v8
    import v8_cohorts
    import v8_dusk
    import v8_flags
    import v8_gaps
    import v8_lattice
    import v8_tileclock
    from farm_neat.policy import observation
    return dict(calc=calc_features, codec=codec, data=data, flip=flip, job_eta=job_eta, prefix=prefix, v8=v8,
                dusk=v8_dusk, cohorts=v8_cohorts, flags=v8_flags, gaps=v8_gaps, lattice=v8_lattice,
                tileclock=v8_tileclock, obs=observation)


def base_header(obs):
    """Names of base-row columns 0..34 (same for every kind), from observation._Rows.add."""
    names = [f"kind:{k}" for k in obs.KINDS] + ["role:own", "role:rival", "x/9", "y/9", "ordinal/100"]
    names += [f"item:{i}" for i in obs.ITEMS] + [f"tile:{t}" for t in obs.TILES]
    assert len(names) == obs.PAYLOAD, (len(names), obs.PAYLOAD)
    return names


def base_names_for(obs, kind):
    """80 base-row column names for one entity kind (payload fields depend on the kind)."""
    names = base_header(obs) + [f"{n}" for n, _ in obs.FIELDS.get(kind, ())]
    return names + [f"(unused {i})" for i in range(len(names), obs.WIDTH)]


def flip_columns(m):
    """[(col, block, expression, condition)] for the 116 flip columns, extracted from flip.py."""
    flip = m["flip"]
    src = P("flip.py")
    env = vars(flip)
    cols = []
    g = list_elements(src, "global_features", lambda n, parent: isinstance(parent, ast.Return))
    cols += [(flip.G0 + i, "GLOBAL", e, "") for i, e in enumerate(g)]
    f = list_elements(src, "farm_view", lambda n, parent: isinstance(parent, ast.Assign)
                      and ast.unparse(parent.targets[0]) == "view['farm']")
    cols += [(flip.F0 + i, "FARM", e, "") for i, e in enumerate(f)]
    for i, text, val, cond, _ in fills(src, "_tile_row", "row", env):
        if text == "20 + kind":
            for k, nm in enumerate(("NONE", "WATER_TONIGHT", "FEED_TONIGHT", "HARVEST_DECAYING", "WATER_TODAY",
                                    "FEED_TODAY", "HARVEST", "COLLECT", "FERTILIZER", "PLANT_EMPTY", "WEED", "CARE")):
                assert getattr(flip, nm) == k
                cols.append((flip.T0 + 20 + k, "TILE", f"job kind == {nm}", ""))
        else:
            cols.append((flip.T0 + i, "TILE", val, " and ".join(cond)))
    w = list_elements(src, "farm_view", lambda n, parent: isinstance(parent, ast.Call)
                      and ast.unparse(parent.func) == "per_worker.append")
    cols += [(flip.C0 + i, "CARRIER (and unit_extra)", e, "") for i, e in enumerate(w)]
    e = list_elements(src, "economics", lambda n, parent: isinstance(parent, ast.Assign)
                      and ast.unparse(parent.targets[0]) == "row")
    over = {i: (val, " and ".join(cond)) for i, _, val, cond, _ in fills(src, "economics", "row", env) if i is not None}
    for i, ex in enumerate(e):
        if i in over:
            ex = f"{ex}; own farm: {over[i][0]}"
        cols.append((flip.E0 + i, "FARM", ex, ""))
    for i, text, val, cond, _ in fills(src, "market_row", "row", env):
        if text == "k":
            cols += [(flip.M0 + k, "MARKET", f"(price after consumption tick {k + 1} - now) / base", "")
                     for k in range(flip.HORIZON_TICKS)]
        else:
            cols.append((flip.M0 + i, "MARKET", val, ""))
    for i, text, val, cond, _ in fills(src, "job_bits", "b", env):
        if i is not None:
            cols.append((flip.J0 + i, "TILE", val, " and ".join(c for c in cond if not c.startswith("not ("))))
    s = list_elements(src, "startup_rows", lambda n, parent: isinstance(parent, ast.Assign)
                      and ast.unparse(parent.targets[0]) == "out[x, y]")
    cols += [(flip.S0 + i, "TILE (own empty)", e, "") for i, e in enumerate(s)]
    o = list_elements(src, "overflow_tonight", lambda n, parent: isinstance(parent, ast.Return))
    cols += [(flip.N0 + i, "FARM (own)", e, "") for i, e in enumerate(o)]
    cols.append((flip.KIND_COL, "TILE", "job kind code (raw integer; Model 1 uses it only in the decoder reach bias)", ""))
    cols.append((flip.DEADLINE_COL, "TILE", "job deadline ETA (raw integer, 0 = none; Model 1 decoder reach bias only)", ""))
    seen = {}
    for c in cols:
        seen.setdefault(c[0], c)
    cols = [seen[i] for i in sorted(seen)]
    return cols


def calc_columns(m):
    calc = m["calc"]
    src = P("calc_features.py")
    expr = {}
    for var, key in (("row", "_PC"), ("g", "_GC")):
        for _, text, val, cond, _ in fills(src, "raw", var, vars(calc)):
            mm = re.fullmatch(rf"{key}\['(\w+)'\]", text)
            if mm:
                c = [x for x in cond if not x.startswith("for ")]
                expr.setdefault(mm.group(1), []).append(f"{val}  [if {' and '.join(c)}]" if c else val)
    expr = {k: "; ".join(dict.fromkeys(v)) for k, v in expr.items()}
    rows = [(i, "MARKET", n, s, expr.get(n, "")) for i, (n, s) in enumerate(calc.PROD_COLS)]
    rows += [(calc.N_PROD + i, "GLOBAL", n, s, expr.get(n, "")) for i, (n, s) in enumerate(calc.GLOB_COLS)]
    return rows


def job_columns(m):
    src = P("job_eta.py")
    out = []
    for i, text, val, cond, _ in fills(src, "extract", "token", vars(m["job_eta"])):
        if i is None:
            continue
        where = "FARM (own)" if "tid == 'farm/0'" in cond else "TILE (own)"
        out.append((i, where, val))
    return out


def prefix_columns(m):
    pre = m["prefix"]
    src = P("prefix.py")
    env = dict(vars(pre))
    unit = [(i, val, " and ".join(c)) for i, _, val, c, _ in fills(src, "PrefixExecutor.unit_features", "f", env)
            if i is not None]
    order = [(i, val, " and ".join(c)) for i, _, val, c, _ in fills(src, "PrefixExecutor.order_features", "f", env)
             if i is not None]
    cash = []
    for i, text, val, c, hi in fills(src, "PrefixExecutor.cash_features", "f", env):
        if i is None:
            continue
        if text == "CASH_SCALARS:":
            for j, nm in enumerate(pre.CASH_THRESHOLD_NAMES):
                for k, part in enumerate(("z clipped to [-2, 2]", "sign(z)*log1p(|z|)", "z >= 0")):
                    cash.append((pre.CASH_SCALARS + 3 * j + k, f"margin to {nm}: {part}, z = (available - required) / max(1, required)", ""))
        else:
            cash.append((i, val, " and ".join(c)))
    return sorted(unit, key=lambda r: r[0]), sorted(order, key=lambda r: r[0]), sorted(cash, key=lambda r: r[0])


# ----------------------------------------------------------------------------- rendering
def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render(m, docs_ok=True):
    obs, flip, calc, job, pre, v8m = m["obs"], m["flip"], m["calc"], m["job_eta"], m["prefix"], m["v8"]
    dusk, coh, tc, flags, gaps, lat = m["dusk"], m["cohorts"], m["tileclock"], m["flags"], m["gaps"], m["lattice"]
    data = m["data"]
    cfg1 = _find(json.loads(M1_CONFIG.read_text()), "model_config")
    run2 = json.loads(M2_RUN.read_text())
    cfg2 = run2["config"]
    cols_json = json.loads((MODEL2 / "COLUMNS.json").read_text())

    tok_w, unit_w, lat_w = v8m.widths()
    fam_w = {mod.__name__[3:]: (mod.WIDTH_TOKEN, mod.WIDTH_UNIT) for mod in v8m.available()}
    long_w = gaps.WIDTH_LONG
    checks = []

    def check(label, got, want):
        checks.append((label, got, want, "ok" if got == want else "MISMATCH"))

    check("v8 token columns (dusk+cohorts+tileclock+flags+gaps+gaps long) vs Model 1 `x8_token_width`",
          sum(t for t, _ in fam_w.values()) + long_w, cfg1["x8_token_width"])
    check("same vs Model 2 `x8_token_width`", tok_w, cfg2["x8_token_width"])
    check("v8 unit columns (dusk+flags) vs Model 1 `x8_unit_width`", sum(u for _, u in fam_w.values()), cfg1["x8_unit_width"])
    check("same vs Model 2 `x8_unit_width`", unit_w, cfg2["x8_unit_width"])
    check("lattice channels (3 slots x 40) vs Model 1 `lattice_channels`", lat_w, cfg1["lattice_channels"])
    check("Model 2 `lattice_channels`", cfg2["lattice_channels"], 0)
    check("flip linear columns vs Model 1 `extra_width`", flip.LIN, cfg1["extra_width"])
    check("calc columns vs Model 1 `calc_width`", calc.N_PROD + calc.N_GLOB, cfg1["calc_width"])
    check("flip WIDTH (LIN + kind + deadline) + calc = `data.EXTRA_WIDTH` = Model 2 `extra` source block",
          flip.WIDTH + calc.N_PROD + calc.N_GLOB, dict(cols_json["src_keys"])["extra"])
    check("flip per-worker columns vs Model 1 `unit_extra_width`", flip.UNIT_WIDTH, cfg1["unit_extra_width"])
    check("prefix unit / order / cash widths vs Model 1 config",
          (pre.F_UP, pre.F_OP, pre.F_CASH), (cfg1["prefix_unit_width"], cfg1["prefix_order_width"], cfg1["prefix_cash_width"]))
    check("job_eta token width vs Model 2 `v6_token` source block", job.TOKEN_WIDTH, dict(cols_json["src_keys"])["v6_token"])
    check("base row width vs Model 2 `rows` source block", obs.WIDTH, dict(cols_json["src_keys"])["rows"])
    check("Model 2 source width (rows+extra+v6_token+x8_token+knownness)", sum(w for _, w in cols_json["src_keys"]),
          cols_json["src_width"])
    check("Model 2 per-type column counts vs RUN `type_widths`", {k: len(v) for k, v in cols_json["types"].items()},
          cfg2["type_widths"])

    toks, units = v8m.names()
    defs = {}
    if docs_ok:
        by_fam = {}
        for n in toks + units:
            fam, c = n.split(":", 1)
            by_fam.setdefault("gaps" if fam == "gaps_long" else fam, []).append(c)
        for fam, names in by_fam.items():
            defs[fam] = doc_definitions(fam, names)
        defs["lattice"] = doc_definitions("lattice", list(lat.SLOT_CHANNELS))

    L = []
    w = L.append
    w("# Features\n")
    w("<!-- Generated by tools/list_features.py from the shipped code. Do not edit by hand. -->")
    w("*Generated by [`tools/list_features.py`](tools/list_features.py) from the code the final models trained with. "
      "Regenerate with `python3 tools/list_features.py` (needs numpy); `--check` verifies every line link.*\n")
    w("Every input the two final models read: what each column is, how it is computed, and which model uses it. "
      "Code links point at the flat reading copies in [`code/`](code/), byte-identical to the files that ran and "
      "mapped to their as-run paths in [`code/MANIFEST.tsv`](code/MANIFEST.tsv); files without a reading copy (the "
      "game engine) link into Model 1's training tree "
      f"([`{MODEL1.relative_to(REPO).as_posix()}`]({MODEL1.relative_to(REPO).as_posix()})). "
      "Both models read the same per-step arrays (the step cache stores them); they differ in which arrays they use "
      "and how they consume them.\n")
    w("![Every feature family, coloured by which final model uses it](images/15-figI-feature-table.png)\n")

    toc = ["How to read this", "Solvers at a glance", "Overview", "Width check", "Base entity rows",
           "flip: no-action forecasts, jobs, reach, economics, price path", "calc: market calculations",
           "job routes (v6)", "Decoder reach bias", "Same-step prefix and cash ladder", "Own last two turns",
           "v8 dusk", "v8 cohorts", "v8 tileclock", "v8 flags", "v8 gaps", "v8 lattice (Model 1 only)",
           "Model 2 graph: relation types, calendar, memory", "Model 2 node columns", "Training-only: no-op loss mask",
           "Built but not used by the shipped models"]
    w(" · ".join(f"[{t}](#{_slug(t)})" for t in toc) + "\n")

    # --- how to read
    w("## How to read this\n")
    w(f"- **Entity tokens.** Each decision step, {P('data.py').link('features', 'data.features')} calls "
      f"{OBS.link('pack', 'observation.pack')}, which emits one token per entity: the global clock, both farms, "
      "200 tiles (own and rival), every worker (`CARRIER`), 12 shed and 5 seed `STOCK` items, 9 `MARKET` products, "
      "8 `SHOP` types and 8 crop/animal `CONSTANT`s. Each token is an 80-wide row plus an 80-wide known mask "
      "(absent or hidden values are 0 with known = False).")
    w("- **Families.** Every other feature family adds columns to particular token types (`GLOBAL`, `MARKET`, `TILE`, "
      "`FARM`, `CARRIER`, `STOCK`) or per-worker *unit rows* (25 rows, farmer first) that the decoder reads when that "
      "worker's command is generated. A family leaves zeros on token types it does not describe.")
    w(f"- **Model 1** adds each family through its own zero-initialised `Linear` into the token stream "
      f"({P('model.py').link('PolicyNet.encode')}); unit rows go into the decoder's per-worker queries. "
      f"**Model 2** keeps, per node type, the source columns that vary within that type and projects them with a "
      f"per-type `Linear` ({NFF.link('compact_tokens')}, [`COLUMNS.json`]({COLS_LINK})).")
    w("- **Transforms** in the column tables: `lin` = x / scale (tileclock clips to ±8, gaps to [-3, 3]); "
      "`slog` = sign(x)·log1p(|x| / scale); `rlog` = log1p(min(ratio, 1e6)) for a cost or sale as a fraction of "
      "own money; `flag` = 0/1; `clip` = clip(x / scale, -1, 1). Code: "
      f"{P('v8_cohorts.py').link('scale', 'v8_cohorts.scale')}, {P('v8_gaps.py').link('_scale', 'v8_gaps._scale')}, "
      f"{P('v8_gaps.py').link('_ratio', 'v8_gaps._ratio')}, {P('v8_tileclock.py').link('scale', 'v8_tileclock.scale')}, "
      f"{P('calc_features.py').link('slog', 'calc_features.slog')}, {P('v8_dusk.py').link('slog', 'v8_dusk.slog')}. "
      "Families without a transform column scale inside the expression shown.")
    w("- **Meanings.** Where a table shows *computed as*, the text is the code expression itself. Where it shows "
      "*meaning*, the text is the inline comment next to the column in the code, or the definition from the family "
      "author's column table (used only where the column name matches the code exactly).\n")

    # --- solvers
    w("## Solvers at a glance\n")
    w("The computations behind the columns. Most are exact replays of the game engine's rules from the actor's own "
      "view; none reads the rival's private state, the episode seed or future shop draws.\n")
    sol = [
        ("Price law and price tables", "quote at any inventory; total receipts for selling or buying a run of units",
         "market params, inventory", f"{P('theorem_features.py').link('market_price', 'theorem_features.market_price')}, "
         f"{P('theorem_features.py').link('_Curve')} (exact table ±2048 around I0 with prefix sums), "
         f"{ENGINE.link('market_price', 'engine market_price')}"),
        ("Season formula", "units, cycles and first output day of a producer started today",
         "producer, days left, fertilized?", P("theorem_features.py").link("season_formula")),
        ("Projected clearing quote", "end-of-game inventory offset E, its quote P*, receipts and net of one more producer, n*",
         "public market, both boards, own stock", P("calc_features.py").link("raw", "calc_features.raw")),
        ("No-action tile forecast", "water/feed deadline, WEED/escape ETA, decay start, next production of each tile",
         "one tile, step", f"{P('flip.py').link('plant_forecast')}, {P('flip.py').link('_plant_no_action')}, "
         f"{P('flip.py').link('animal_forecast')}"),
        ("Priority job per tile", "one job code + deadline per tile (WATER_TONIGHT first)", "tile forecasts",
         P("flip.py").link("_jobs")),
        ("Worker reach", "decisions for each worker to act on each tile; FEED via a shed WHEAT pickup", "positions, inventories",
         f"{P('flip.py').link('farm_view')}, {P('flip.py').link('reach_matrix')}"),
        ("No-trade price path", "price of every product after each of the next 6 consumption ticks", "market, visible shops",
         P("flip.py").link("price_path")),
        ("Planting startup certificate", "can two present workers PLANT then WATER an empty tile before tonight",
         "positions, seeds", P("flip.py").link("startup_rows")),
        ("Nine-job routes", "min ETA per (cell, job) over workers with a WHEAT/FERTILIZER route; deadlines; feasibility",
         "own farm, own inventories", P("job_eta.py").link("extract", "job_eta.extract")),
        ("Same-step prefix", "own state after the earlier workers' commands this step; shed stock intervals after orders",
         "snapshot + decoded commands", f"{P('prefix.py').link('PrefixExecutor.apply_unit')}, "
         f"{P('prefix.py').link('PrefixExecutor.unit_features')}, {P('prefix.py').link('PrefixExecutor.apply_order')}"),
        ("Cash ladder", "money after earlier orders, affordable counts, margins to 17 price thresholds",
         "snapshot + decoded orders", f"{P('prefix.py').link('PrefixExecutor.cash_features')}, "
         f"{P('prefix.py').link('PrefixExecutor._apply_cash_order')}, "
         f"{P('prefix.py').link('PrefixExecutor.affordable_product_count')}"),
        ("No-op mask", "which teacher worker commands provably do nothing (dropped from the loss)",
         "snapshot + teacher action", f"{P('noop_mask.py').link('unit_reason')}, "
         f"{P('noop_mask.py').link('training_unit_mask')}, {P('noop_mask.py').link('teacher_supervision_mask')}"),
        ("Dusk return", "deadline, return slack per carrier, deposit queue, lost-unit bounds, harvest detours",
         "own carriers, shed, market", P("v8_dusk.py").link("raw", "v8_dusk.raw")),
        ("Cohort schedules and sell-as-produced path", "full-service production by day, pro-rata receipts on the price "
         "law, working capital and payback of one more producer", "both boards, market, shops",
         f"{P('v8_cohorts.py').link('_entry')}, {P('v8_cohorts.py').link('_new_schedule')}, "
         f"{P('v8_cohorts.py').link('raw', 'v8_cohorts.raw')}"),
        ("Tile clock simulation", "per-crop value of starting today vs later; per-tile service, skip and fertilizer value",
         "tiles, quotes", f"{P('v8_tileclock.py').link('plant_sim')}, {P('v8_tileclock.py').link('animal_sim')}, "
         f"{P('v8_tileclock.py').link('clock_table')}, {P('v8_tileclock.py').link('clock_raw')}"),
        ("Step receipts", "what each worker's last action did, from two consecutive observations; product ledger",
         "own observations t-1, t", P("v8_flags.py").link("analyse_step")),
        ("Shop-reveal demand belief", "visible vs expected future town demand per product", "visible shops, step",
         P("v8_flags.py").link("_shop_block")),
        ("Market impact and cash identity", "quote drop of one more unit, cannibalisation; running cash identity and "
         "rival flow", "observation; per-seat state over the game",
         f"{P('v8_gaps.py').link('raw', 'v8_gaps.raw')}, {P('v8_gaps.py').link('GapsState')}, "
         f"{P('v8_gaps.py').link('_update')}"),
        ("Space-time lattice", "both farms as 10x10 grids at t, t-1, t-2, through a small CNN onto TILE tokens",
         "own observations t, t-1, t-2", f"{P('v8_lattice.py').link('compute_frame')}, "
         f"{P('v8_lattice.py').link('stack_frames')}, {P('v8_lattice_net.py').link('LatticeCNN')}"),
        ("Decoder reach bias (Model 1)", "attention bias from each worker to each own tile by ETA, slack, job kind and "
         "per-job route facts", "unit positions, flip and job_eta outputs",
         f"{P('model.py').link('PolicyNet._reach_bias')}, {P('model.py').link('PolicyNet._v6_job_bias')}"),
        ("Graph relations and memory (Model 2)", "relation type of every node pair; calendar landmarks; fast/slow/worker GRUs",
         "node layout, step", f"{NFM.link('relation_table')}, {NFM.link('NFlowNet.calendar')}, "
         f"{NFM.link('NFlowNet.memory_step')}"),
        ("Step search (not used)", "joint log-prob best-of-k or beam over a whole step",
         "model, observation", f"{STEP_SEARCH.link('act', 'step_search.act')} (shipped archives declare `decode: greedy`)"),
    ]
    w(table(["solver", "what it answers", "inputs", "code"], sol))
    w("")
    w("Engine functions the solvers call or replicate: "
      + ", ".join(ENGINE.link(n) for n in ("_apply_unit_action", "_commit_unit", "_refresh_prices", "_parse_order",
                                            "_do_hire", "_do_buy_land", "_hire_cost", "_town_consume", "_decay_plants",
                                            "_daily_refresh_plants", "_daily_refresh_animals",
                                            "_drop_inventories_to_shed", "_end_of_day"))
      + f"; price helper {MARKET.link('price_at')}.\n")

    # --- overview
    w("## Overview\n")
    yes, no = "yes", "–"
    ov = [
        ("Base entity rows", "one row per entity: identity one-hots, position, kind-specific fields", "every token",
         f"{obs.WIDTH} + {obs.WIDTH} known", yes, "yes (per-type subset)", OBS.link("_Rows.add")),
        ("flip", "no-action forecasts, priority jobs, worker reach, economics, no-trade price path", "GLOBAL, FARM, TILE, CARRIER, MARKET",
         f"{flip.LIN} + kind + deadline", "yes", "yes (subset)", P("flip.py").link("token_features")),
        ("flip unit rows", "per-worker reach summaries", "unit rows", str(flip.UNIT_WIDTH), "yes", "yes",
         P("flip.py").link("farm_view")),
        ("calc", "projected clearing quote, marginal producer receipts, pipeline values", "MARKET, GLOBAL",
         str(calc.N_PROD + calc.N_GLOB), "yes", "yes (subset)", P("calc_features.py").link("PROD_COLS")),
        ("job routes (v6)", "nine-job ETAs, routes, deadlines, feasibility", "TILE (own), FARM (own)",
         f"{job.TOKEN_WIDTH} + per-(worker, cell, job) arrays", "yes", "token columns only", P("job_eta.py").link("extract")),
        ("reach bias", "worker → own tile attention bias by ETA, slack, job", "decoder", "–", "yes", "yes (ETA only)",
         P("model.py").link("PolicyNet._reach_bias")),
        ("same-step prefix", "state after earlier workers' commands this step; order stock intervals", "decoder queries",
         f"{pre.F_UP} unit, {pre.F_OP} order", "yes", "yes", P("prefix.py").link("PrefixExecutor")),
        ("cash ladder", "money, affordability and threshold margins after earlier orders", "decoder order queries",
         str(pre.F_CASH), "yes", "yes", P("prefix.py").link("PrefixExecutor.cash_features")),
        ("own last two turns", "previous commands and orders as tokens", "decoder / memory", "2 × (25 + 11) tokens", "yes",
         "yes", P("data.py").link("add_v6")),
        ("v8 dusk", "dusk deposit and return-to-shed feasibility", "GLOBAL, STOCK (shed), unit rows",
         f"{fam_w['dusk'][0]} + {fam_w['dusk'][1]} unit", "yes", "yes (subset)", P("v8_dusk.py").link("TOKEN_COLS")),
        ("v8 cohorts", "production cohorts, scheduled receipts, working capital, equity, trends", "MARKET, GLOBAL",
         str(fam_w["cohorts"][0]), "yes", "yes (subset)", P("v8_cohorts.py").link("PROD_COLS")),
        ("v8 tileclock", "crop start clocks and season cliffs; per-tile service value", "TILE, GLOBAL",
         str(fam_w["tileclock"][0]), "yes", "yes (subset)", P("v8_tileclock.py").link("CLOCK_FIELDS")),
        ("v8 flags", "calendar, shop-reveal timeline, action receipts, ledger, binary flags", "MARKET, GLOBAL, unit rows",
         f"{fam_w['flags'][0]} + {fam_w['flags'][1]} unit", "yes", "yes (subset)", P("v8_flags.py").link("PROD_COLS")),
        ("v8 gaps", "cohort maturity, one-more-unit market impact, costs as a fraction of money", "MARKET, GLOBAL",
         str(fam_w["gaps"][0]), "yes", "yes (subset)", P("v8_gaps.py").link("PROD_COLS")),
        ("v8 gaps long", "running cash identity, product conservation, rival flow (whole-game state)", "MARKET, GLOBAL",
         str(long_w), "yes", "yes (subset)", P("v8_gaps.py").link("LONG_PROD_COLS")),
        ("v8 lattice", "both farms as grids at t, t-1, t-2 through a CNN", "TILE (via CNN)", str(lat_w), "yes", no,
         P("v8_lattice.py").link("SLOT_CHANNELS")),
        ("relation types", "17 typed node-pair relations as attention biases", "graph attention", "17 × 4 heads", no, "yes",
         NFM.link("REL_NAMES")),
        ("calendar landmarks", "now, next shop tick, dusk, +2d, +4d, +8d, +12d, end", "8 graph nodes", "6 each", no, "yes",
         NFM.link("NFlowNet.calendar")),
        ("game memory", "fast, slow and per-worker GRU state", "2 graph nodes + own carriers",
         f"{cfg2['fast']} / {cfg2['slow']} / {cfg2['worker']}", no, "yes", NFM.link("NFlowNet.memory_step")),
        ("theorem feature tokens", "7,564 values per step from the macro theorems", "–", "7,564", no, no,
         P("features.py").link("FEATURE_SCHEMA")),
        ("long-lag history", "clock and item history 1, 4, 24, 72 steps back", "–", "–", no, no,
         P("history_features.py").link("LAGS")),
    ]
    w(table(["family", "what it captures", "attaches to", "width", "Model 1", "Model 2", "defined at"], ov))
    w("")
    w(f"Model 1's config ({_rel(M1_CONFIG)}): `extra_width` {cfg1['extra_width']}, `calc_width` {cfg1['calc_width']}, "
      f"`unit_extra_width` {cfg1['unit_extra_width']}, `reach_bias` {cfg1['reach_bias']}, `reach_job` {cfg1['reach_job']}, "
      f"`job_eta_v6` {cfg1['job_eta_v6']}, `v6_prev` {cfg1['v6_prev']}, `x8_token_width` {cfg1['x8_token_width']}, "
      f"`x8_unit_width` {cfg1['x8_unit_width']}, `lattice_channels` {cfg1['lattice_channels']}, `think` {cfg1['think']!r}. "
      f"Model 2's config ({_rel(M2_RUN)}): `x8_token_width` {cfg2['x8_token_width']}, `x8_unit_width` "
      f"{cfg2['x8_unit_width']}, `lattice_channels` {cfg2['lattice_channels']}, `v6_prev` {cfg2['v6_prev']}. "
      "Neither config has a theorem feature-token input.\n")

    w("## Width check\n")
    w("Recomputed from the code each time this file is generated.\n")
    w(table(["check", "from code", "recorded", "result"], [(a, f"`{b}`", f"`{c}`", d) for a, b, c, d in checks]))
    w("")

    # --- base rows
    w("## Base entity rows\n")
    w(f"**How it's computed.** {OBS.link('_world')} walks the actor observation in a fixed order (global; for each farm, "
      "own first: FARM, its 100 tiles, one CARRIER per worker; then shed and seed STOCK, 9 MARKET, 8 SHOP, 8 CONSTANT) "
      f"and {OBS.link('_Rows.add')} writes the header one-hots and each payload field as `value / scale`, never clipped. "
      "A field that is absent or hidden stays 0 with known = False; carried inventories are filled for own workers only. "
      f"Field lists: {OBS.link('FIELDS')}, {OBS.link('TILE_FIELDS')}.\n")
    w("Columns 0-34, every token:\n")
    hdr = base_header(obs)
    w(table(["#", "column"], [(i, f"`{n}`") for i, n in enumerate(hdr)]))
    w("\nColumns 35+, by token kind (`field / scale`):\n")
    kinds = ["GLOBAL", "FARM", "TILE", "CARRIER", "STOCK", "MARKET", "SHOP", "CONSTANT", "CONTROL"]
    w(table(["kind", "payload columns from 35"], [(k, ", ".join(f"`{n}` /{_num(s)}" for n, s in obs.FIELDS[k])) for k in kinds]))
    w(f"\nThe actor view carries no order plan, so {OBS.link('_state')} emits a single `CONTROL` token with only "
      "`commitments_known` = False set. Nothing in it varies, so Model 2 keeps column 0 for that type as a placeholder "
      f"(`vary or [0]` in {NFF.link('cmd_derive')}).\n")

    # --- flip
    w("## flip: no-action forecasts, jobs, reach, economics, price path\n")
    fs = P("flip.py")
    w(f"**How it's computed.** For every tile, {fs.link('plant_forecast')} and {fs.link('animal_forecast')} simulate "
      f"that tile alone with no further action: {fs.link('_plant_no_action')} steps through yield-decay ticks (every 2 "
      "steps from `max_lifespan_step`) and day ends (unwatered counter, WEED at 2, ongoing production) for up to 3 days, "
      f"giving the water/feed deadline, the WEED/escape ETA, decay start and the next production. {fs.link('_jobs')} turns "
      "each tile into one priority job and deadline (WATER_TONIGHT, HARVEST_DECAYING, WATER_TODAY, HARVEST, …). "
      f"{fs.link('farm_view')} computes every worker's ETA to every tile as Manhattan distance + 1, routing FEED through "
      "the shed-access tile that minimises the whole trip (+2 to pick up and act) when the worker carries no WHEAT, then per-worker summaries and farm duty counts. "
      f"{fs.link('price_path')} replays the next 6 passive consumption ticks (shops every 4 steps, town centre every 24) "
      f"with no trades; {fs.link('economics')} computes hire (Fibonacci wage) and land affordability on raw integers; "
      f"{fs.link('startup_rows')} certifies the fastest PLANT-then-WATER by two workers before tonight's refresh. "
      f"Assembled per token by {fs.link('token_features')}.\n")
    w(f"Model 1 reads columns 0-{flip.LIN - 1} through one linear layer and uses the job kind and deadline only in the "
      "decoder reach bias; Model 2 keeps the varying columns per node type, including the raw kind and deadline.\n")
    rows = [(i, blk, code(ex), code(cond, 80)) for i, blk, ex, cond in flip_columns(m)]
    w(table(["#", "token", "computed as", "when"], rows))
    used = {r[0] for r in rows}
    missing = [i for i in range(flip.WIDTH) if i not in used]
    w(f"\nColumns written by no code path (always 0): {', '.join(map(str, missing)) or 'none'}.\n")

    # --- calc
    w("## calc: market calculations\n")
    cs = P("calc_features.py")
    w(f"**How it's computed.** {cs.link('raw', 'calc_features.raw')} builds on "
      f"{P('theorem_features.py').link('compute', 'theorem_features.compute')} (exact price curves, visible town demand, "
      "per-tile season units). Per product it projects the end-of-game inventory offset "
      "E = (inv − I0) + own supply (held + season units) + rival supply − visible demand − expected new-shop demand "
      "(WHEAT supply net of the animals' remaining feed), then P* = quote(I0 + E), the projected clearing quote of the "
      "marginal unit. The receipts of one more producer are the sum of quotes over the next `units` inventory steps from "
      f"I0 + E ({P('theorem_features.py').link('_Curve.total')}, prefix sums); net = receipts − input cost; n* counts how "
      f"many extra producers still net ≥ 0 (cap 40); {cs.link('_k_sell')} binary-searches how many units sell above P*. "
      f"Scaled with `slog(x, scale)` in {cs.link('compute')}. Product rows sit on `market/<P>` tokens, the global row on "
      f"`global` ({P('data.py').link('calc_columns', 'data.calc_columns')}).\n")
    w(table(["#", "token", "column", "slog scale", "computed as (unscaled)"],
            [(i, t, f"`{n}`", _num(s), code(e)) for i, t, n, s, e in calc_columns(m)]))
    w("")

    # --- job routes
    w("## job routes (v6)\n")
    js = P("job_eta.py")
    w(f"**How it's computed.** {js.link('extract', 'job_eta.extract')} builds direct ETAs (Manhattan + 1) from every own "
      "worker to every own cell, and resource ETAs for WHEAT and FERTILIZER: direct when the worker carries it, else via "
      "the shed-access tile that minimises the whole trip (+2) when the shed holds it. Nine job types per cell come from "
      f"{fs.link('job_bits', 'flip.job_bits')}; WATER and FEED take their deadline from the flip forecast, HARVEST the "
      "earliest of decay, dehydration and game end. Per cell and job it keeps the minimum ETA over workers with a route. "
      "Token columns summarise each own tile's primary job; the FARM columns total tonight's feasible travel and count "
      f"tonight's jobs with no route. Model 1 also feeds the per-(worker, cell, job) facts into the decoder as an edge "
      f"bias ({P('model.py').link('PolicyNet._v6_job_bias')}); Model 2 reads the token columns only.\n")
    w(table(["#", "token", "computed as"], [(i, t, code(v)) for i, t, v in job_columns(m)]))
    w("")

    # --- reach bias
    w("## Decoder reach bias\n")
    w(f"**How it's computed.** {fs.link('reach_matrix', 'flip.reach_matrix')} gives `reach_eta [25, 100]`: decisions for "
      "each own worker to act on each own tile, FEED on an unfed animal routed through a WHEAT pickup, 999 when no route "
      f"exists. Model 1 ({P('model.py').link('PolicyNet._reach_bias')}) adds, per decoder layer and head, a learned bias "
      "to each worker query's attention over own TILE tokens, looked up from (ETA bucket of Manhattan + 1, slack bucket to "
      "the tile's flip deadline, flip job kind) plus a `reach_job` bucket from `reach_eta`, plus the per-job edge bias "
      f"above. Model 2 ({NFM.link('NFlowNet.dec_context')}) adds a learned per-layer, per-head bias from "
      f"{NFM.link('eta_bucket')} (12 buckets of `reach_eta`) on the own TILE nodes.\n")

    # --- prefix / cash
    w("## Same-step prefix and cash ladder\n")
    ps = P("prefix.py")
    w(f"**How it's computed.** {ps.link('PrefixExecutor')} starts from the actor-visible snapshot and applies the earlier "
      f"workers' commands of the same step with the engine's own {ENGINE.link('_apply_unit_action')}. A PLANT that "
      "over-subscribes the seeds held at step start becomes PASS and the prefix is replayed from the snapshot "
      f"({ps.link('PrefixExecutor.apply_unit')}). Before worker k decides, {ps.link('PrefixExecutor.unit_features')} "
      "reads the state at k's tile, seeds left, blocked crops and shed room. After the workers, each market order updates "
      f"per-product shed stock intervals [L, U] (SELL lowers both, buys raise only U) "
      f"({ps.link('PrefixExecutor.apply_order')}, {ps.link('PrefixExecutor.order_features')}). The cash ladder "
      f"({ps.link('PrefixExecutor.cash_features')}) replays the orders on a copy of farm and market with the engine's "
      f"{ENGINE.link('_commit_unit')}, {ENGINE.link('_do_hire')}, {ENGINE.link('_do_buy_land')} and price refresh, "
      "assuming no rival trades: money after earlier orders, affordable counts (product buys walk the price curve unit by "
      "unit) and a margin triplet for 17 thresholds. Training computes these along the teacher's decoded action "
      f"({ps.link('teacher_arrays')}); play computes them along the model's own decoded tokens.\n")
    unit_c, order_c, cash_c = prefix_columns(m)
    w(f"Unit prefix ({pre.F_UP} columns, one row per worker):\n")
    w(table(["#", "computed as", "when"], [(i, code(v), code(c, 60)) for i, v, c in unit_c]))
    w(f"\nOrder prefix ({pre.F_OP} columns, one row per order slot):\n")
    w(table(["#", "computed as", "when"], [(i, code(v), code(c, 60)) for i, v, c in order_c]))
    w(f"\nCash ladder ({pre.F_CASH} columns, one row per order slot):\n")
    w(table(["#", "computed as", "when"], [(i, code(v) if not v.startswith("margin") else md(v), code(c, 60))
                                            for i, v, c in cash_c]))
    w("")

    # --- history
    w("## Own last two turns\n")
    w(f"**How it's computed.** {P('data.py').link('add_v6', 'data.add_v6')} stores the codec tokens (token, quantity) of "
      "the seat's previous two turns: each worker slot's command (only when it is the same hand) and the orders. "
      f"Model 1 embeds them as history tokens and per-worker previous-command vectors "
      f"({P('model.py').link('PolicyNet._history')}); Model 2 sums lag embeddings into a step summary and per-worker "
      f"vectors that feed its fast and worker memories ({NFM.link('NFlowNet.prev_action_emb')}). At play the agent "
      f"builds the same arrays ({P('model.py').link('v6_history_arrays')}).\n")

    # --- v8 families
    def v8_table(fam, mod, cols, comments, groups, where, transform=True):
        d = defs.get(fam, {})
        rows = []
        for i, col in enumerate(cols):
            if isinstance(col, tuple):
                name, mode, scale = col if len(col) == 3 else (col[0], "slog", col[1])
            else:
                name, mode, scale = col, "", ""
            meaning = comments.get(name) or d.get(name) or d.get(f"{where(i)}{name}") or ""
            grp = groups.get(name)
            if grp and not meaning:
                meaning = re.sub(r"\b[HSCE]\d+(?:/[HSCE]\d+)?\b\s*", "", grp).strip(" +")
            r = [i, f"`{name}`"]
            if transform:
                r += [mode, _num(scale)]
            r.append(md(meaning))
            rows.append(r)
        return rows

    ds = P("v8_dusk.py")
    w("## v8 dusk\n")
    w(f"**How it's computed.** {ds.link('raw', 'v8_dusk.raw')} takes the deadline (the dusk step 24d + 23, or step 718 on "
      "day 29 when nothing is deposited), each own carrier's load, its shed ETA (Manhattan to the nearest access tile + "
      "1) and return slack. It simulates the day-end deposit in engine order (farmer first, overflow discarded) to get "
      f"units lost if nothing changes, with min/max value bounds over the unknown order inside a carrier "
      f"({ds.link('_lost_value_bounds')}); then a queue where every loaded carrier walks home now and drops on arrival; "
      "then one harvest detour per carrier (walk, HARVEST, walk to the shed, DROP) with yield decay counted on the way. "
      f"Values use spot quotes and calc's P* ({ds.link('unit_values')}). Shed-item columns sit on `stock/shed/<ITEM>`, "
      f"globals on `global` ({ds.link('compute')}).\n")
    c1, _ = column_comments(ds, "GLOBAL_COLS")
    c2, _ = column_comments(ds, "ITEM_COLS")
    c3, _ = column_comments(ds, "UNIT_COLS")
    allc = {**c1, **c2, **c3}
    w(table(["#", "column", "meaning"], v8_table("dusk", dusk, list(dusk.TOKEN_COLS), allc, {}, lambda i: "", False)))
    w(f"\nUnit rows ({dusk.WIDTH_UNIT}, one per own worker):\n")
    w(table(["#", "column", "meaning"], v8_table("dusk", dusk, list(dusk.UNIT_COLS), allc, {}, lambda i: "", False)))
    w("")

    cos = P("v8_cohorts.py")
    w("## v8 cohorts\n")
    w(f"**How it's computed.** Scenario: full service of every visible producer on both farms. {cos.link('_entry')} "
      f"(via {cos.link('_plant_entry')} / {cos.link('_animal_entry')}) gives each tile's units by availability day, cached "
      f"per tile state; {cos.link('_farm')} sums them per product with bucket counts by time to first receipt. "
      f"{cos.link('raw', 'v8_cohorts.raw')} then walks a sell-as-produced market path day by day: both farms' units of "
      "a product are sold together on the exact price law and split pro rata, animal feed is bought as WHEAT, and "
      f"visible plus expected town demand drains inventory between days ({cos.link('_demand')}). For one more producer "
      f"of each kind ({cos.link('_new_schedule')}) it finds the nearest site and route (DIG or BUILD first if needed), "
      "decides whether it can start today, and accumulates costs and receipts on the post-sale path to get cash to first "
      "sale, payback day and net to end; the receipts/cost ratio ranks the eight producers. Product columns (`m.`) sit on "
      "`market/<P>`, globals (`g.`) on `global`.\n")
    cols = [("m." + n, mo, s) for n, mo, s in coh.PROD_COLS] + [("g." + n, mo, s) for n, mo, s in coh.GLOB_COLS]
    w(table(["#", "column", "transform", "scale", "meaning"], v8_table("cohorts", coh, cols, {}, {}, lambda i: "")))
    w("")

    tcs = P("v8_tileclock.py")
    w("## v8 tileclock\n")
    w(f"**How it's computed.** {tcs.link('plant_sim')} and {tcs.link('animal_sim')} roll one tile forward day by day "
      "under the engine's rules to the last day end, for four plans: full service, skip today, fertilize now, and "
      f"fertilize on every day that raises the season (greedy interval cover). {tcs.link('clock_table')} runs the plant "
      f"simulation for every crop and start day; {tcs.link('clock_raw')} reads off harvests, units, the loss from waiting "
      "a day, days to the next harvest-count cliff and to the last useful start, and value at today's quote; these sit "
      f"on own empty TILE tokens. {tcs.link('_plant_facts')} and {tcs.link('_animal_facts')} give each producer tile's "
      f"season units and the gain or loss of serving, skipping or fertilizing today ({tcs.link('raw', 'v8_tileclock.raw')}).\n")
    tcols = list(zip(tc.TOKEN_COLS, [s[1] for s in tc._SPEC], [s[2] for s in tc._SPEC]))
    w(table(["#", "column", "transform", "scale", "meaning"], v8_table("tileclock", tc, tcols, {}, {}, lambda i: "")))
    w("")

    fls = P("v8_flags.py")
    w("## v8 flags\n")
    w(f"**How it's computed.** {fls.link('analyse_step')} compares the seat's observations at t-1 and t (same day only) and "
      "attributes every inventory and tile change to the worker that caused it (one action per worker per step; DROP "
      "overflow simulated in worker order), giving per-worker effect codes, a per-product ledger (produced, consumed, "
      f"bought, sold) and the cash split. {fls.link('_calendar')} encodes the hour and day landmarks; "
      f"{fls.link('_shop_block')} dates every visible shop that consumes a product and computes the expected and "
      f"worst-case demand from future uniform shop draws; {fls.link('_season')} asks whether a producer started at the "
      "earliest possible step still yields before the end. Product columns sit on `market/<P>`, globals on `global`, "
      f"unit columns on unit rows ({fls.link('compute')}). Scaling is inside `compute`.\n")
    g1, grp1 = column_comments(fls, "PROD_COLS")
    g2, grp2 = column_comments(fls, "GLOB_COLS")
    g3, grp3 = column_comments(fls, "UNIT_COLS")
    grp = {**grp1, **grp2, **grp3}
    w(table(["#", "column", "meaning"], v8_table("flags", flags, list(flags.TOKEN_COLS), {}, grp, lambda i: "", False)))
    w(f"\nUnit rows ({flags.WIDTH_UNIT}):\n")
    w(table(["#", "column", "meaning"], v8_table("flags", flags, list(flags.UNIT_COLS), {}, grp, lambda i: "", False)))
    w("")

    gs = P("v8_gaps.py")
    w("## v8 gaps\n")
    w(f"**How it's computed.** {gs.link('raw', 'v8_gaps.raw')} scans both boards ({gs.link('_scan')}, reusing the cohorts "
      "per-tile schedule) for cohort maturity and held yield; for the market it takes the first-order effect of one more "
      "unit in inventory (the quote drop between inventory and inventory + 1, which persists because town demand does not "
      "depend on price) times own and rival future units; every cost and sale is also expressed as a ratio to own money. "
      f"The long block ({gs.link('compute_long')}) carries one {gs.link('GapsState')} per seat through the game: "
      f"{gs.link('_update')} reconstructs a running cash identity for both seats (land, wages, seeds, animals, trade) "
      "from board and inventory differences, own per-product conservation, and the rival's net market flow over the "
      "last 24 and 72 steps.\n")
    gcols = [(n, mo, s) for n, mo, s in gaps.PROD_COLS] + [(n, mo, s) for n, mo, s in gaps.GLOB_COLS]
    w(table(["#", "column", "transform", "scale", "meaning"], v8_table("gaps", gaps, gcols, {}, {}, lambda i: "")))
    w(f"\nLong block ({long_w} columns, appended after the five families):\n")
    lcols = [(n, mo, s) for n, mo, s in gaps.LONG_PROD_COLS] + [(n, mo, s) for n, mo, s in gaps.LONG_GLOB_COLS]
    w(table(["#", "column", "transform", "scale", "meaning"], v8_table("gaps", gaps, lcols, {}, {}, lambda i: "")))
    w("")

    las = P("v8_lattice.py")
    w("## v8 lattice (Model 1 only)\n")
    w(f"**How it's computed.** {las.link('compute_frame')} paints both farms as 10x10 grids with "
      f"{len(lat.SLOT_CHANNELS)} channels per cell (tile kind, crop/animal, service flags, counters, workers, carried "
      f"units, the flip job kind from {las.link('job_kind')}, shed geometry); {las.link('stack_frames')} folds the frames "
      f"at t, t-1 and t-2 into {lat_w} channels. {P('v8_lattice_net.py').link('LatticeCNN')} runs each farm through two "
      "3x3 convolutions with a farm embedding and a zero-initialised per-cell output layer, and "
      f"{P('v8_lattice_net.py').link('scatter_to_tokens')} adds each cell's output to its TILE token. Model 2's "
      "`lattice_channels` is 0. Channel index = slot × 40 + channel (slot 0 = t).\n")
    lc, _ = column_comments(las, "SLOT_CHANNELS")
    rows = []
    for i, ch in enumerate(lat.SLOT_CHANNELS):
        meaning = lc.get(ch) or defs.get("lattice", {}).get(ch, "")
        rows.append((i, f"`{ch}`", md(meaning)))
    w(table(["#", "channel", "meaning"], rows))
    w("")

    # --- model 2 graph
    w("## Model 2 graph: relation types, calendar, memory\n")
    w(f"**How it's computed.** {NFM.link('NFlowNet.embed')} builds {_const(NFM, 'N_NODES')} nodes: the 296 fixed entity "
      f"slots from {NFF.link('compact_tokens')} (one per tile, worker, stock item, market, shop, constant, farm, global "
      "and control), FAST and SLOW memory nodes, 8 calendar landmarks and padding. "
      f"{NFM.link('relation_table')} gives every node pair one of {len(_names(NFM, 'REL_NAMES'))} relation types "
      "(general relations first, specific ones override); each relational block adds a learned bias per (relation "
      f"type, head) on {cfg2['rel_heads']} of its {cfg2['heads']} heads. {NFM.link('NFlowNet.calendar')} gives each "
      "landmark 6 features: offset / 720, target step / 720, past the end, log offset, target day / 30, steps left after "
      f"the target / 720. {NFM.link('NFlowNet.memory_step')}: a fast GRU ({cfg2['fast']}) updated every step from a "
      f"pooled graph summary and the actor's previous action; a slow GRU ({cfg2['slow']}) updated at dawn from the "
      f"previous day's mean summary; one worker GRU ({cfg2['worker']}) per own worker slot from its carrier node and "
      f"previous command, reset at dawn for hands. {NFM.link('NFlowNet.encode_post')} writes the memories back into the "
      "graph before the last relational block.\n")
    pairs = {
        "other": "pairs no rule assigns, e.g. two tiles of one farm that are not adjacent",
        "self": "a node with itself (assigned last)",
        "tile_adj": "two tiles of the same farm at Manhattan distance 1",
        "tile_mirror": "the tile at the same (x, y) on the other farm",
        "farm_member": "a FARM node and the tiles and workers of that farm",
        "carrier_tile": "a worker and any tile of its own farm",
        "carrier_carrier": "two workers of the same farm",
        "custody": "an own worker and an own shed or seed STOCK node",
        "market_market": "two MARKET nodes",
        "shop_market": "a SHOP node and a MARKET node",
        "resource": "two nodes among STOCK, MARKET, CONSTANT and SHOP",
        "hub": "GLOBAL, CONTROL or a FARM node with any non-pad node",
        "memory": "the FAST or SLOW memory node with any non-pad node",
        "cal_later": "a calendar landmark and a later landmark",
        "cal_earlier": "a calendar landmark and an earlier landmark",
        "cal_resource": "a calendar landmark and a STOCK, MARKET, CONSTANT, SHOP, TILE or FARM node",
        "pad": "any pair that includes a padding node",
    }
    assert list(pairs) == list(_names(NFM, "REL_NAMES")), "relation descriptions out of date with REL_NAMES"
    rel_rows = []
    rt = NFM.node("relation_table")
    for i, name in enumerate(_names(NFM, "REL_NAMES")):
        lines = [n.lineno for n in ast.walk(rt) if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
                 and n.slice.value == name and ast.unparse(n.value) == "idx"]
        where = ", ".join(f"[L{ln}]({NFM.rel}#L{ln})" for ln in sorted(set(lines))) or "default (0)"
        rel_rows.append((i, f"`{name}`", pairs[name], where))
    w("Rules run in source order and a later rule overrides an earlier one for the same pair, so the general "
      "relations (hub, memory, resource) only hold where nothing more specific applies.\n")
    w(table(["#", "relation", "pairs", "assigned at"], rel_rows))
    w("")
    cal_names = ("now", "next shop tick", "dusk", "+2 days", "+4 days", "+8 days", "+12 days", "terminal (step 718)")
    w("Calendar landmarks in order: " + ", ".join(cal_names) + f" ({NFM.link('NFlowNet.calendar')}).\n")

    # --- model 2 node columns
    w("## Model 2 node columns\n")
    w(f"Model 2 builds a 674-wide source row per entity token: `rows` 80 | `extra` 152 | `v6_token` 8 | `x8_token` 354 | "
      f"`knownness` 80 ({NFF.link('SRC_KEYS')}). [`COLUMNS.json`]({COLS_LINK}) keeps, per node type, "
      f"the columns whose value varies across that type's tokens on a sample of seats ({NFF.link('cmd_derive')}); "
      "dropped columns are dropped identically at play time. Per-worker inputs bypass this: `unit_extra` 12 + `x8_unit` "
      f"41 feed the unit queries together with the prefix rows ({NFM.link('NFlowNet.dec_context')}, "
      f"{NFM.link('NFlowNet.queries_teacher')}).\n")
    flip_cols = {i: f"flip[{i}]" for i in range(flip.WIDTH)}
    calc_names = [n for n, _ in calc.PROD_COLS] + [n for n, _ in calc.GLOB_COLS]
    x8_names = toks
    type_kind = {"GLOBAL": "GLOBAL", "FARM": "FARM", "TILE": "TILE", "CARRIER": "CARRIER", "STOCK": "STOCK",
                 "MARKET": "MARKET", "SHOP": "SHOP", "CONSTANT": "CONSTANT", "CONTROL": "CONTROL"}
    fl = {i: (blk, ex) for i, blk, ex, _ in flip_columns(m)}
    seg_rows = []
    for t, idx in cols_json["types"].items():
        base = base_names_for(obs, type_kind[t])
        groups = {"rows": [], "extra": [], "v6_token": [], "x8_token": [], "knownness": []}
        for i in idx:
            o = 0
            for nm, wd in cols_json["src_keys"]:
                if i < o + wd:
                    j = i - o
                    if nm == "rows":
                        groups[nm].append(base[j])
                    elif nm == "knownness":
                        groups[nm].append("known:" + base[j])
                    elif nm == "extra":
                        groups[nm].append(f"flip[{j}]" if j < flip.WIDTH else f"calc.{calc_names[j - flip.WIDTH]}")
                    elif nm == "v6_token":
                        groups[nm].append(f"job[{j}]")
                    else:
                        groups[nm].append(x8_names[j])
                    break
                o += wd
        for nm, lst in groups.items():
            if lst:
                seg_rows.append((t, nm, len(lst), ", ".join(f"`{x}`" for x in lst)))
    w(table(["node type", "source block", "kept", "columns"], seg_rows))
    w("\n`flip[i]` and `job[i]` refer to the numbered rows of the flip and job-route tables above; x8 names are "
      "`family:column`.\n")

    # --- noop mask
    w("## Training-only: no-op loss mask\n")
    ns = P("noop_mask.py")
    w(f"**How it's computed.** {ns.link('training_unit_mask')} replays the teacher's worker commands on the engine state "
      f"in order and asks {ns.link('unit_reason')} whether each one provably has no effect (a move off the board edge, an "
      "action on the wrong tile kind, a PLANT blocked because the step's PLANT requests exceed the seeds held, …), "
      f"following the engine's own check order. {ns.link('teacher_supervision_mask')} drops a worker's label only when it "
      "is a no-op in both the raw replay action and its codec encoding. Both models use it in the loss: Model 1 with "
      "`--mask-unit-noops true` (see `model1-b-even/STAGES.md`), Model 2 through `unit_loss_keep` in "
      f"{Src(MODEL2 / 'train.py').file_link('train.py')}. It is not an input feature and never a mask at play time.\n")

    # --- not used
    w("## Built but not used by the shipped models\n")
    w(f"- **Theorem feature tokens.** {P('features.py').link('FEATURE_SCHEMA', 'features.py')} packs 7,564 values per "
      f"step from {P('theorem_features.py').file_link()} and {P('history_features.py').file_link()} into 22 feature "
      f"tokens ({P('feature_tokens.py').link('tokens', 'feature_tokens.tokens')}). Neither shipped config has that "
      "input. `theorem_features.py` itself is still used as a library: its price curves, demand law and season "
      "arithmetic feed calc, cohorts and tileclock.")
    w(f"- **Long-lag history.** {P('history_features.py').link('LAGS', 'history_features.py')} (lags 1, 4, 24, 72) is "
      "only imported by `features.py`.")
    w(f"- **Step search.** {STEP_SEARCH.link('sample_rows', 'step_search.sample_rows')} (best of k complete steps by "
      f"joint log-probability) and {STEP_SEARCH.link('beam_search', 'step_search.beam_search')} ship in both "
      "archives' agents, but both archives declare `decode: greedy`, so they are never called.\n")
    w("![Loss increase on earlier B/C checkpoints when each feature group is removed](images/16-figD-features.png)\n")
    w("*Feature-drop test from the write-up: earlier B/C checkpoints, whole projection zeroed at fixed weights.*\n")
    if Path(REPO / "HOW-IT-WORKS.md").exists():
        w("See also [HOW-IT-WORKS.md](HOW-IT-WORKS.md) for where each idea lives in the code.\n")
    return "\n".join(L).rstrip() + "\n", checks


def _find(o, key):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def _rel(p):
    r = Path(p).relative_to(REPO).as_posix()
    return f"[`{r}`]({r})"


def _num(x):
    if x == "" or x is None:
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def _slug(t):
    t = t.lower()
    t = re.sub(r"[^\w\- ]", "", t)
    return t.replace(" ", "-")


def _names(src, name):
    return ast.literal_eval(src.node(name).value)


def _const(src, name):
    return ast.literal_eval(src.node(name).value)


# ----------------------------------------------------------------------------- link check
def check_links(text):
    bad = []
    text = re.sub(r"(?<!\[)`[^`]*`(?!\])", "", text)        # code spans outside link labels
    for label, target in re.findall(r"\[([^\]]*)\]\(([^)\s]+)\)", text):
        if target.startswith(("http", "#")):
            continue
        path, _, frag = target.partition("#")
        f = REPO / path
        if not f.exists():
            bad.append((target, "missing file"))
            continue
        if frag.startswith("L") and f.is_file():
            ln = int(frag[1:].split("-")[0])
            lines = f.read_text().splitlines()
            if ln > len(lines):
                bad.append((target, "line past end"))
                continue
            sym = label.strip("`").split(" ")[-1].split(".")[-1]
            if sym and not sym.startswith("L") and sym not in lines[ln - 1] and label.strip("`") not in (f.name,):
                bad.append((target, f"line {ln} does not contain {sym!r}: {lines[ln - 1].strip()[:80]}"))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="regenerate in memory, compare with FEATURES.md, check links")
    ap.add_argument("--extract-docs", metavar="DIR", help="refresh the embedded DOC_DEFS from COLUMNS-*.md in DIR")
    a = ap.parse_args()
    if a.extract_docs:
        data = extract_docs(a.extract_docs)
        me = Path(__file__).read_text()
        block = "DOC_DEFS = " + pprint.pformat(data, width=118, sort_dicts=False) + "\n"
        me = re.sub(r"(# --- BEGIN DOC_DEFS\n).*?(# --- END DOC_DEFS)", lambda m_: m_.group(1) + block + m_.group(2),
                    me, flags=re.S)
        Path(__file__).write_text(me)
        print("embedded", {k: len(v) for k, v in data.items()}, "doc rows")
        return
    mods = load()
    text, checks = render(mods)
    bad = check_links(text)
    for b in bad:
        print("LINK", *b, file=sys.stderr)
    mism = [c for c in checks if c[3] != "ok"]
    for c in mism:
        print("WIDTH", c, file=sys.stderr)
    if a.check:
        same = OUT.exists() and OUT.read_text() == text
        print(f"FEATURES.md up to date: {same}; bad links: {len(bad)}; width mismatches: {len(mism)}")
        sys.exit(0 if same and not bad else 1)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)}: {len(text.splitlines())} lines; bad links: {len(bad)}; "
          f"width mismatches: {len(mism)}")


# ----------------------------------------------------------------------------- embedded column definitions
# From the v8 family authors' COLUMNS-<family>.md tables (pbc-features-v8/, 2026-09-28). The modules those tables
# describe are byte-identical to the shipped v8_cohorts/flags/gaps/tileclock/lattice; v8_dusk differs only in which
# calc module it imports. Refresh with --extract-docs.
# --- BEGIN DOC_DEFS
DOC_DEFS = {'dusk': [('dusk_left', None, 'decisions left until the deadline, current one included'),
          ('dusk_final_day',
           None,
           '1 on day 29 (no day-end deposit remains; carried goods are lost unless sold by t718)'),
          ('dusk_shed_fill', None, 'shed units, animals included / shedCapacity'),
          ('dusk_shed_room', None, 'shedCapacity - fill'),
          ('dusk_carried_units', None, 'all own carried units'),
          ('dusk_carried_value_spot', None, 'carried units x spot'),
          ('dusk_carried_value_proj', None, 'carried units x P*: total carried value at risk'),
          ('dusk_lost_units_idle',
           None,
           'units lost at the deadline if nothing changes (engine deposit order; day 29: all carried)'),
          ('dusk_lost_value_idle_min',
           None,
           'spot $ of those losses, taking the cheapest units first within each carrier'),
          ('dusk_lost_value_idle_max', None, 'same, taking the dearest units first'),
          ('dusk_overflow_after_sellout',
           None,
           'units lost even if the whole sellable shed is sold by the dusk step: max(0, carried + shed animals - '
           'cap); day 29: carried'),
          ('dusk_feasible_payload', None, 'units on carriers with slack >= 0'),
          ('dusk_late_payload', None, 'units on carriers with slack < 0'),
          ('dusk_overflow_unavoidable',
           None,
           'lower bound on loss if every feasible carrier deposits and all is sold: max(0, late payload + shed '
           'animals - cap); day 29: late payload'),
          ('dusk_n_late_loaded', None, 'number of loaded carriers with slack < 0'),
          ('dusk_min_loaded_slack', None, 'smallest slack among loaded carriers; 1 if none is loaded'),
          ('dusk_pending_ready_units',
           None,
           'harvest-ready own units surviving plant decay until a carrier reaches and harvests the tile by the '
           'deadline (d + 1 <= left)'),
          ('dusk_pending_return_units',
           None,
           'the surviving units, but only where some carrier can also harvest and DROP by the deadline'),
          ('dusk_overflow_with_pending', None, 'max(0, fill + carried + pending_ready - cap); 0 on day 29'),
          ('dusk_late_value_proj', None, 'P* $ carried by carriers with slack < 0'),
          ('dusk_item_carried', None, 'this item, summed over all own carriers'),
          ('dusk_item_lost_idle_min',
           None,
           'fewest units of this item lost at the deadline if nothing changes, over the unknown within-carrier '
           'insertion order'),
          ('dusk_item_lost_idle_max', None, 'most units of this item lost, same conditions'),
          ('dusk_item_carried_late', None, 'this item on carriers with slack < 0'),
          ('dusk_u_carried', None, 'units carried'),
          ('dusk_u_wheat', None, 'WHEAT carried'),
          ('dusk_u_fertilizer', None, 'FERTILIZER carried'),
          ('dusk_u_crop', None, 'CARROT + TOMATO + STRAWBERRY + MELON carried'),
          ('dusk_u_animal_product', None, 'EGG + MILK + WOOL carried'),
          ('dusk_u_live_animal', None, 'GOOSE + COW + SHEEP carried'),
          ('dusk_u_shed_eta',
           None,
           'd + 1: decisions until a DROP lands (Manhattan to the nearest access tile, via `flip._shed_dist`)'),
          ('dusk_u_return_slack', None, 'left - (d + 1); latest deposit start = now + slack'),
          ('dusk_u_can_return',
           None,
           '1 if slack >= 0: the carrier can reach shed access and attempt DROP by the deadline; NOT a storage or '
           'sale guarantee'),
          ('dusk_u_lost_idle',
           None,
           'units of this carrier lost at the deadline if nothing changes. Exact per carrier, because the day end '
           'deposits in carrier order (E843-857). This is the loss if the carrier keeps working away from the shed.'),
          ('dusk_u_lost_value_min', None, 'spot $ of those losses, cheapest units first'),
          ('dusk_u_lost_value_max', None, 'spot $ of those losses, dearest units first'),
          ('dusk_u_value_spot', None, 'carried value at spot: cash if dropped and sold now, ignoring price impact'),
          ('dusk_u_value_proj', None, 'carried value at P*: value at risk'),
          ('dusk_u_queue_room',
           None,
           "room at this carrier's deposit minus its load, assuming every carrier that can arrive walks home now and "
           'DROPs on arrival (ETA order, ties by index) and late carriers deposit at the day end; no sales. A '
           'negative value means its DROP destroys that many units. On day 29, late carriers get -load.'),
          ('dusk_u_detour_cost',
           None,
           'fewest extra decisions for a one-tile detour before the deposit (walk, HARVEST, walk to the nearest '
           'access tile, DROP), among detours that still fit by the deadline: d(w,j) + d(j,S) + 1 - d(w,S). 1 if '
           'none fits.'),
          ('dusk_u_pending_return',
           None,
           'ready units on the tiles this carrier could harvest and still DROP by the deadline'),
          ('dusk_u_home_delta',
           None,
           'sign(previous shed distance - current): +1 moved closer since t-1; 0 if unmoved or if there is no '
           'same-day history'),
          ('dusk_u_carried_delta', None, 'change in units carried since t-1 (same day only; 0 otherwise)')],
 'cohorts': [('m.own_producing',
              '0',
              '# own producer tiles of this product, already producing: plants age >= first_yield_day, animals day '
              '>= placed_day+first_yield_day (FERTILIZER row: animals past their first refresh), with scheduled '
              'units left'),
             ('m.own_first_le2d',
              '1',
              '# own producer tiles of this product, not yet producing; first scheduled receipt in 1-2 days'),
             ('m.own_first_3to5d',
              '2',
              '# own producer tiles of this product, not yet producing; first scheduled receipt in 3-5 days'),
             ('m.own_first_6plusd',
              '3',
              '# own producer tiles of this product, not yet producing; first scheduled receipt in 6+ days and by '
              'day 29'),
             ('m.own_no_output_left',
              '4',
              '# own producer tiles of this product, no scheduled unit by day 29 (e.g. cow placed on day >= 22, '
              'exhausted ongoing crop)'),
             ('m.rival_producing',
              '5',
              '# rival (public tiles only) producer tiles of this product, already producing: plants age >= '
              'first_yield_day, animals day >= placed_day+first_yield_day (FERTILIZER row: animals past their first '
              'refresh), with scheduled units left'),
             ('m.rival_first_le2d',
              '6',
              '# rival (public tiles only) producer tiles of this product, not yet producing; first scheduled '
              'receipt in 1-2 days'),
             ('m.rival_first_3to5d',
              '7',
              '# rival (public tiles only) producer tiles of this product, not yet producing; first scheduled '
              'receipt in 3-5 days'),
             ('m.rival_first_6plusd',
              '8',
              '# rival (public tiles only) producer tiles of this product, not yet producing; first scheduled '
              'receipt in 6+ days and by day 29'),
             ('m.rival_no_output_left',
              '9',
              '# rival (public tiles only) producer tiles of this product, no scheduled unit by day 29 (e.g. cow '
              'placed on day >= 22, exhausted ongoing crop)'),
             ('m.own_units_1d',
              '10',
              'own board units available on days d..d+1 (stored harvestable yield counted today; one-shot crops at '
              'their full-service harvest day; animals incl. FERTILIZER row)'),
             ('m.own_units_3d',
              '11',
              'own board units available on days d..d+3 (stored harvestable yield counted today; one-shot crops at '
              'their full-service harvest day; animals incl. FERTILIZER row)'),
             ('m.own_units_7d',
              '12',
              'own board units available on days d..d+7 (stored harvestable yield counted today; one-shot crops at '
              'their full-service harvest day; animals incl. FERTILIZER row)'),
             ('m.own_units_end',
              '13',
              'own board units available on days d..29 (stored harvestable yield counted today; one-shot crops at '
              'their full-service harvest day; animals incl. FERTILIZER row)'),
             ('m.own_value_1d',
              '14',
              '$ receipts of own_units_1d on the sell-as-produced path (per day: exact integrated price law over '
              "both farms' same-day units, shared pro rata; floor units pay $1)"),
             ('m.own_value_3d',
              '15',
              '$ receipts of own_units_3d on the sell-as-produced path (per day: exact integrated price law over '
              "both farms' same-day units, shared pro rata; floor units pay $1)"),
             ('m.own_value_7d',
              '16',
              '$ receipts of own_units_7d on the sell-as-produced path (per day: exact integrated price law over '
              "both farms' same-day units, shared pro rata; floor units pay $1)"),
             ('m.own_value_end',
              '17',
              '$ receipts of own_units_end on the sell-as-produced path (per day: exact integrated price law over '
              "both farms' same-day units, shared pro rata; floor units pay $1)"),
             ('m.rival_units_1d',
              '18',
              'rival (public tiles only) board units available on days d..d+1 (stored harvestable yield counted '
              'today; one-shot crops at their full-service harvest day; animals incl. FERTILIZER row)'),
             ('m.rival_units_3d',
              '19',
              'rival (public tiles only) board units available on days d..d+3 (stored harvestable yield counted '
              'today; one-shot crops at their full-service harvest day; animals incl. FERTILIZER row)'),
             ('m.rival_units_7d',
              '20',
              'rival (public tiles only) board units available on days d..d+7 (stored harvestable yield counted '
              'today; one-shot crops at their full-service harvest day; animals incl. FERTILIZER row)'),
             ('m.rival_units_end',
              '21',
              'rival (public tiles only) board units available on days d..29 (stored harvestable yield counted '
              'today; one-shot crops at their full-service harvest day; animals incl. FERTILIZER row)'),
             ('m.rival_value_1d',
              '22',
              '$ receipts of rival_units_1d on the sell-as-produced path (per day: exact integrated price law over '
              "both farms' same-day units, shared pro rata; floor units pay $1)"),
             ('m.rival_value_3d',
              '23',
              '$ receipts of rival_units_3d on the sell-as-produced path (per day: exact integrated price law over '
              "both farms' same-day units, shared pro rata; floor units pay $1)"),
             ('m.rival_value_7d',
              '24',
              '$ receipts of rival_units_7d on the sell-as-produced path (per day: exact integrated price law over '
              "both farms' same-day units, shared pro rata; floor units pay $1)"),
             ('m.rival_value_end',
              '25',
              '$ receipts of rival_units_end on the sell-as-produced path (per day: exact integrated price law over '
              "both farms' same-day units, shared pro rata; floor units pay $1)"),
             ('m.wc_cash_to_first_sale',
              '26',
              'cash out before the first product-sale day of ONE more producer started on its earliest feasible day: '
              'seed or animal price + WHEAT feed at path buy quotes (animals); all costs to day 29 if no sale is '
              'reachable'),
             ('m.wc_days_to_first_sale',
              '27',
              "days from today to that producer's first product unit: start day + first_yield_day (animals, ongoing "
              'crops) or + one-shot harvest age; = days remaining (29-d+1) if unreachable'),
             ('m.wc_first_sale_reachable', '28', '1 if that first product unit lands by day 29'),
             ('m.wc_start_today',
              '29',
              '1 if it can be placed/planted today: crops BUY_SEED now, PLANT from s+1 on the nearest empty tile (+1 '
              'step to DIG a weed) and WATER the same day (planting day counts unwatered, L222/L783); animals '
              'BUY_ANIMAL now (needs shed room, L679-688), PICKUP at a shed-access tile, walk, BUILD if needed, '
              "PLACE by the day's last step. Else the start is tomorrow"),
             ('m.wc_site_state',
              '30',
              '1 free matching site (empty owned tile for crops, free COOP/PASTURE for animals); 0.5 needs DIG/BUILD '
              'first; 0 none (needs land)'),
             ('m.wc_reserve_after', '31', 'own_liquid_budget - wc_cash_to_first_sale - own_obligations_3d'),
             ('m.wc_payback_days',
              '32',
              "first day offset at which the new producer's cumulative marginal receipts (product + animal "
              "fertilizer, sold after the day's scheduled sales on the path) cover its cumulative costs (seed per "
              'replant cycle, animal, daily feed); days remaining if never'),
             ('m.wc_net_to_end',
              '33',
              'marginal receipts - costs of that producer to day 29 (timing-aware; calc marginal_net_1 is the single '
              'end-quote version)'),
             ('m.attract_ratio',
              '34',
              'owner ratio: receipts / full input cost of one more producer (same schedule as wc_*: product + animal '
              'fertilizer at per-day projected quotes / seed x replant cycles or animal + daily WHEAT feed). '
              'Land/space NOT included (no exact cheap price for a site bundle)'),
             ('m.attract_rank',
              '35',
              'rank of attract_ratio among the 8 producers (1 = best; ties by producer order)'),
             ('m.count_own',
              '36',
              'visible own producer tiles of this type (planted crop tiles / animals of the species)'),
             ('m.count_rival', '37', 'visible rival producer tiles of this type'),
             ('m.count_total', '38', 'count_own + count_rival'),
             ('m.vis_production_1d',
              '39',
              'own + rival board units available on days d..d+1 (to-end version: own_units_end + rival_units_end '
              'here; calc supply_own + supply_rival also counts own shed/carried)'),
             ('m.vis_production_3d',
              '40',
              'own + rival board units available on days d..d+3 (to-end version: own_units_end + rival_units_end '
              'here; calc supply_own + supply_rival also counts own shed/carried)'),
             ('m.vis_production_7d',
              '41',
              'own + rival board units available on days d..d+7 (to-end version: own_units_end + rival_units_end '
              'here; calc supply_own + supply_rival also counts own shed/carried)'),
             ('m.demand_1d',
              '42',
              'town units consumed from step s to the end of day d+1: visible shops x rate per shop tick + centre '
              '1/day (not FERTILIZER) + expected-new-shop prior after each unlock morning (TF formula); WHEAT adds '
              "both farms' scheduled feed. To-end version = calc demand_rest_visible + demand_rest_expected_new (no "
              'feed)'),
             ('m.demand_3d',
              '43',
              'town units consumed from step s to the end of day d+3: visible shops x rate per shop tick + centre '
              '1/day (not FERTILIZER) + expected-new-shop prior after each unlock morning (TF formula); WHEAT adds '
              "both farms' scheduled feed. To-end version = calc demand_rest_visible + demand_rest_expected_new (no "
              'feed)'),
             ('m.demand_7d',
              '44',
              'town units consumed from step s to the end of day d+7: visible shops x rate per shop tick + centre '
              '1/day (not FERTILIZER) + expected-new-shop prior after each unlock morning (TF formula); WHEAT adds '
              "both farms' scheduled feed. To-end version = calc demand_rest_visible + demand_rest_expected_new (no "
              'feed)'),
             ('m.delta_1d', '45', 'vis_production_1d - demand_1d (to-end analogue: calc projected_offset)'),
             ('m.delta_3d', '46', 'vis_production_3d - demand_3d (to-end analogue: calc projected_offset)'),
             ('m.delta_7d', '47', 'vis_production_7d - demand_7d (to-end analogue: calc projected_offset)'),
             ('m.delta_quote_1d',
              '48',
              'price-law quote at market inventory now + round(delta_1d) (L192-206; to-end: calc projected_quote)'),
             ('m.delta_quote_3d',
              '49',
              'price-law quote at market inventory now + round(delta_3d) (L192-206; to-end: calc projected_quote)'),
             ('m.delta_quote_7d',
              '50',
              'price-law quote at market inventory now + round(delta_7d) (L192-206; to-end: calc projected_quote)'),
             ('m.flow_production_rate', '51', 'board units (both farms) maturing on the next 3 mornings / 3'),
             ('m.flow_sell_pressure',
              '52',
              "units available now: both farms' stored/harvest-today board units + own shed + own carried (rival "
              'shed/carried are private: not counted)'),
             ('m.flow_demand_rate',
              '53',
              "town demand at steps s..s+71 (visible shops, centre, expected-new prior; WHEAT adds both farms' feed) "
              '/ 3'),
             ('m.net_flow_per_day', '54', 'flow_production_rate + flow_sell_pressure/3 - flow_demand_rate'),
             ('m.dq_per_day',
              '55',
              '[quote(inv + round(3 x net_flow_per_day)) - quote(inv)] / 3 by the price law (L192-206)'),
             ('m.dq_rel_per_day', '56', 'dq_per_day / max(1, current quote)'),
             ('m.trend_class',
              '57',
              '+2 fast up (dq_rel_per_day >= +0.05), +1 slow up (>= +0.01), 0 flat, -1 slow down (<= -0.01), -2 fast '
              'down (<= -0.05)'),
             ('m.obs_dq_1', '58', 'quote now - quote at t-1 (own history; 0 when hist_lag1_avail = 0)'),
             ('m.obs_dq_2', '59', 'quote now - quote at t-2 (0 unless both history steps are contiguous)'),
             ('m.obs_net_sales_1',
              '60',
              'market inventory now - at t-1 + exact town consumption at step t-1 with the shops visible at t-1 '
              '(L736-749): net units both players sold (+) or bought (-); $1 sales add no inventory (L658-660) so '
              'they are invisible'),
             ('m.obs_net_sales_2', '61', 'same over steps t-2..t-1'),
             ('g.own_liquid_budget',
              '62',
              'own cash + receipts of selling the whole own shed now (empty rival queue, exact law)'),
             ('g.own_obligations_3d',
              '63',
              "(own animals' feed for days d..d+2 - own WHEAT in shed+carried)+ x WHEAT buy quote + 3 x today's wage "
              "bill (labour plan unknown: today's hires used as the daily proxy)"),
             ('g.own_held_value', '64', 'path receipts of own shed + carried products sold today'),
             ('g.own_board_value_end', '65', 'sum of own_value_end over products'),
             ('g.own_feed_liability',
              '66',
              "WHEAT for own animals' daily feed to day 28 at path buy quotes (signed cost, not floored)"),
             ('g.own_cash_plus_pipeline',
              '67',
              'own cash + own_held_value + own_board_value_end - own_feed_liability'),
             ('g.rival_board_value_end', '68', 'sum of rival_value_end (public board)'),
             ('g.rival_feed_liability', '69', "WHEAT for rival animals' daily feed to day 28 at path buy quotes"),
             ('g.rival_cash_plus_pipeline',
              '70',
              'rival cash + rival_board_value_end - rival_feed_liability (rival shed/carried unknown, excluded)'),
             ('g.realizable_equity_lead', '71', 'own_cash_plus_pipeline - rival_cash_plus_pipeline'),
             ('g.board_equity_lead',
              '72',
              'same lead without own held goods (both sides on the same public-information basis)'),
             ('g.cash_equity_disagree',
              '73',
              '+1 cash lead but realizable equity behind; -1 cash behind but equity ahead; else 0'),
             ('g.pipeline_lead_1d', '74', '(own - rival) board receipts minus feed cost on days d..d+1'),
             ('g.pipeline_lead_3d', '75', '(own - rival) board receipts minus feed cost on days d..d+3'),
             ('g.pipeline_lead_7d', '76', '(own - rival) board receipts minus feed cost on days d..d+7'),
             ('g.pipeline_lead_end', '77', '(own - rival) board receipts minus feed cost on days d..29'),
             ('g.best_is_WHEAT', '78', '1 if WHEAT holds attract_ratio rank 1'),
             ('g.best_is_CARROT', '79', '1 if CARROT holds attract_ratio rank 1'),
             ('g.best_is_TOMATO', '80', '1 if TOMATO holds attract_ratio rank 1'),
             ('g.best_is_STRAWBERRY', '81', '1 if STRAWBERRY holds attract_ratio rank 1'),
             ('g.best_is_MELON', '82', '1 if MELON holds attract_ratio rank 1'),
             ('g.best_is_GOOSE', '83', '1 if GOOSE holds attract_ratio rank 1'),
             ('g.best_is_COW', '84', '1 if COW holds attract_ratio rank 1'),
             ('g.best_is_SHEEP', '85', '1 if SHEEP holds attract_ratio rank 1'),
             ('g.best_ratio', '86', 'attract_ratio of the rank 1 producer'),
             ('g.best_count_own', '87', 'visible own producer tiles of the rank 1 producer'),
             ('g.best_count_rival', '88', 'visible rival producer tiles of the rank 1 producer'),
             ('g.best_count_total', '89', 'visible own + rival producer tiles of the rank 1 producer'),
             ('g.second_is_WHEAT', '90', '1 if WHEAT holds attract_ratio rank 2'),
             ('g.second_is_CARROT', '91', '1 if CARROT holds attract_ratio rank 2'),
             ('g.second_is_TOMATO', '92', '1 if TOMATO holds attract_ratio rank 2'),
             ('g.second_is_STRAWBERRY', '93', '1 if STRAWBERRY holds attract_ratio rank 2'),
             ('g.second_is_MELON', '94', '1 if MELON holds attract_ratio rank 2'),
             ('g.second_is_GOOSE', '95', '1 if GOOSE holds attract_ratio rank 2'),
             ('g.second_is_COW', '96', '1 if COW holds attract_ratio rank 2'),
             ('g.second_is_SHEEP', '97', '1 if SHEEP holds attract_ratio rank 2'),
             ('g.second_ratio', '98', 'attract_ratio of the rank 2 producer'),
             ('g.second_count_own', '99', 'visible own producer tiles of the rank 2 producer'),
             ('g.second_count_rival', '100', 'visible rival producer tiles of the rank 2 producer'),
             ('g.second_count_total', '101', 'visible own + rival producer tiles of the rank 2 producer'),
             ('g.worst_is_WHEAT', '102', '1 if WHEAT holds attract_ratio rank 8'),
             ('g.worst_is_CARROT', '103', '1 if CARROT holds attract_ratio rank 8'),
             ('g.worst_is_TOMATO', '104', '1 if TOMATO holds attract_ratio rank 8'),
             ('g.worst_is_STRAWBERRY', '105', '1 if STRAWBERRY holds attract_ratio rank 8'),
             ('g.worst_is_MELON', '106', '1 if MELON holds attract_ratio rank 8'),
             ('g.worst_is_GOOSE', '107', '1 if GOOSE holds attract_ratio rank 8'),
             ('g.worst_is_COW', '108', '1 if COW holds attract_ratio rank 8'),
             ('g.worst_is_SHEEP', '109', '1 if SHEEP holds attract_ratio rank 8'),
             ('g.worst_ratio', '110', 'attract_ratio of the rank 8 producer'),
             ('g.worst_count_own', '111', 'visible own producer tiles of the rank 8 producer'),
             ('g.worst_count_rival', '112', 'visible rival producer tiles of the rank 8 producer'),
             ('g.worst_count_total', '113', 'visible own + rival producer tiles of the rank 8 producer'),
             ('g.second_worst_is_WHEAT', '114', '1 if WHEAT holds attract_ratio rank 7'),
             ('g.second_worst_is_CARROT', '115', '1 if CARROT holds attract_ratio rank 7'),
             ('g.second_worst_is_TOMATO', '116', '1 if TOMATO holds attract_ratio rank 7'),
             ('g.second_worst_is_STRAWBERRY', '117', '1 if STRAWBERRY holds attract_ratio rank 7'),
             ('g.second_worst_is_MELON', '118', '1 if MELON holds attract_ratio rank 7'),
             ('g.second_worst_is_GOOSE', '119', '1 if GOOSE holds attract_ratio rank 7'),
             ('g.second_worst_is_COW', '120', '1 if COW holds attract_ratio rank 7'),
             ('g.second_worst_is_SHEEP', '121', '1 if SHEEP holds attract_ratio rank 7'),
             ('g.second_worst_ratio', '122', 'attract_ratio of the rank 7 producer'),
             ('g.second_worst_count_own', '123', 'visible own producer tiles of the rank 7 producer'),
             ('g.second_worst_count_rival', '124', 'visible rival producer tiles of the rank 7 producer'),
             ('g.second_worst_count_total', '125', 'visible own + rival producer tiles of the rank 7 producer'),
             ('g.best_ratio_gap', '126', 'best_ratio - second_ratio'),
             ('g.hist_lag1_avail', '127', '1 if history[0] is the step t-1 observation'),
             ('g.hist_lag2_avail', '128', '1 if history[0], history[1] are steps t-1, t-2')],
 'flags': [('lg1_produced',
            None,
            'units of this product entering own inventories by HARVEST (E L446-473) or COLLECT_FERTILIZER (E '
            'L515-522) in step t-1 -> t'),
           ('lg1_consumed', None, 'WHEAT fed (E L505-513) or FERTILIZER spent (E L475-482) in the last step'),
           ('lg1_bought',
            None,
            'max(0, market_net): shed change minus worker deposits plus pickups (DROP overflow simulated in worker '
            "order, E L343-356) = units bought in the last step's market"),
           ('lg1_sold',
            None,
            "max(0, -market_net) = units sold in the last step's market (net of same-step buys of that product)"),
           ('lg2_produced / lg2_consumed / lg2_bought / lg2_sold',
            None,
            'same over the last two steps (t-2 -> t), both same-day'),
           ('lg1_seeds_bought',
            None,
            'crop tokens: seeds bought last step = seed change + new PLANT tiles of that crop (E L425-428, '
            'L673-678)'),
           ('can_buy_producer',
            None,
            'crops: cash >= seed cost (E L673-678); EGG/MILK/WOOL: cash >= GOOSE/COW/SHEEP cost and shed below '
            'capacity (E L679-686); pre-step cash only'),
           ('n_producer_affordable', None, 'floor(cash / seed or animal cost)'),
           ('can_buy_unit',
            None,
            'WHEAT/FERTILIZER: cash >= quote at inventory-1 and shed below capacity (E L598-601, L662-672)'),
           ('free_structure',
            None,
            'EGG: empty own COOPs; MILK/WOOL: empty own PASTUREs (placeable now, E L384-392)'),
           ('producer_can_yield',
            None,
            'a producer started at the earliest step still yields before the end: crops planted now (seed held) or '
            'next step, harvestable day pd+first_yield_day <= 29 (E L453, L789-802); animals placed now/next '
            'step/t+2 (carried/shed/bought), first pulse at next_day pd+first_yield_day <= 29 (E L822-827)'),
           ('producer_full_schedule',
            None,
            'crops only: the full schedule fits (non-ongoing: pd+max_yield_day <= 29; ongoing: last of max_yield '
            'pulses by next_day 29); 0 for animals'),
           ('producer_pulses_left',
            None,
            'production events left for a producer started now (crops <= max_yield, animals every `interval` days), '
            'min(·,30)/10'),
           ('shop_src_count', None, 'visible shop instances consuming this product / 8'),
           ('shop_src_none', None, 'no visible shop consumes it (town centre only)'),
           ('shop_src_newest_age',
            None,
            'days since the newest consuming shop became visible: shop k is visible from step 72(k+1) (E L884-891), '
            '(t-72(k+1))/24/30'),
           ('shop_src_first_age', None, 'same for the oldest consuming shop'),
           ('shop_src_new_last_unlock', None, 'the most recent unlock consumes this product'),
           ('dem_p_new_source',
            None,
            'P(at least one more consuming shop before the end) = 1-(1-f)^n, f = share of the 8 types consuming it, '
            'n = unlocks left (uniform draw, E L891)'),
           ('dem_future_hi',
            None,
            'largest possible extra shop demand from future unlocks: max multiplier x remaining shop ticks of each '
            'future shop (ticks s % 4 == 0, s <= 718, E L736-743)'),
           ('dem_future_sd', None, 'standard deviation of that future demand under the draw law'),
           ('dem_supported_share',
            None,
            'visible remaining demand (visible shops + town centre, E L736-747) / (visible + expected future)'),
           ('cal_hour_sin, cal_hour_cos', None, 'sin/cos(2π·hour/24)'),
           ('cal_phase_night / _morning / _afternoon / _evening', None, 'hour in [0,6) / [6,12) / [12,18) / [18,24)'),
           ('cal_dawn', None, 'hour == 0 (hands gone, hires reset, farmer respawned, E L879-882)'),
           ('cal_dusk', None, 'this decision is the last before a day end ((t+1) % 24 == 0, t <= 695; E L945)'),
           ('cal_final_action', None, 't == 718, the last applied action (E L960)'),
           ('cal_no_dusk_ahead', None, 't > 695: no deposit/refresh left; carried goods are never deposited'),
           ('cal_day_ends_left', None, 'day ends (production dawns) still to come / 29'),
           ('cal_shop_unlock_tonight',
            None,
            "tonight's day end appends a shop ((day+1) % 3 == 0, fewer than 8; E L884-891)"),
           ('cal_shop_tick_now', None, "shops consume after this step's market (t % 4 == 0, E L736)"),
           ('cal_center_tick_now', None, "town centre consumes after this step's market (t % 24 == 0, E L745)"),
           ('rc_valid', None, 'the last step is comparable (history[0] = t-1, same day)'),
           ('rc2_valid', None, 'the step before is comparable too (history[1] = t-2, same day)'),
           ('rc_any_idle', None, 'some worker had no visible effect last step (PASS or failed command)'),
           ('rc_n_idle', None, 'number of such workers / 25'),
           ('rc_n_effect', None, 'workers with a certain effect last step / 25'),
           ('rc_n_ambiguous', None, 'workers whose effect cannot be attributed / 25'),
           ('rc_n_new_hands', None, 'hands hired in the last market (E L702-709) / 10'),
           ('rc_land_bought', None, 'a quadrant was bought last step (valid across a day end)'),
           ('rc_goose_bought / rc_cow_bought / rc_sheep_bought',
            None,
            'animals bought last step (shed change net of worker pickups/deposits)'),
           ('rc_seeds_bought', None, 'all seeds bought last step'),
           ('rc_cash_delta', None, 'own cash change t-1 -> t (valid across a day end)'),
           ('rc_trade_cash',
            None,
            'cash change + seed + animal + hire + land spend = product sale receipts - WHEAT/FERT purchase cost last '
            'step (exact when lg_exact)'),
           ('rc_hire_cash', None, 'hire cost paid last step (Fibonacci, E L690-709)'),
           ('lg_exact',
            None,
            "the last step's per-product flows are exact (no DROP overflow split, no ambiguous FERT/deposit, "
            'same-amount exclusive conflicts)'),
           ('lg_discard', None, 'units destroyed by DROP overflow last step (E L346-355)'),
           ('lg2_trade_cash', None, 'rc_trade_cash over the last two steps'),
           ('lg2_cash_delta', None, 'cash change over the last two steps'),
           ('labour_spent_today', None, 'hire cost paid so far today = Σ_{i<hires_today} fib(i) (E L698-709)'),
           ('can_afford_hire',
            None,
            'cash >= next hire cost (same as flip economics col 4 on farm/0; kept on request)'),
           ('can_afford_land',
            None,
            'a quadrant is left and cash >= its price (E L712-725; same as flip economics col 8 on farm/0)'),
           ('q4_owned', None, 'all four quadrants owned'),
           ('q4_affordable', None, 'three quadrants owned and cash >= 4000'),
           ('shed_full', None, 'shed total >= capacity: BUY_ANIMAL/BUY_PRODUCT fail (E L667, L682)'),
           ('any_animal_unfed', None, 'an own animal not fed today'),
           ('any_plant_unwatered', None, 'an own plant not watered today'),
           ('any_loss_tonight',
            None,
            'an unwatered plant with consecutive_unwatered >= 1 or unfed animal with consecutive_unfed >= 1 '
            "dies/escapes at tonight's day end (E L777-785, L813-820); 0 after t = 695"),
           ('u_valid', None, 'worker existed at t-1 and the step is comparable'),
           ('u_new_hand', None, "worker hired in the last step's market"),
           ('u_effect', None, 'a certain visible effect last step'),
           ('u_no_effect', None, 'no visible effect and not ambiguous: PASS or a failed command'),
           ('u_ambiguous',
            None,
            'co-located workers share a tile change that cannot be attributed, or a WATER may have left no trace '
            '(plant harvested/decayed in the same step)'),
           ('u_moved', None, 'position changed (E L323-332)'),
           ('u_plant, u_water, u_harvest, u_feed, u_care, u_fertilize, u_collect_fert, u_place_animal, u_build, '
            'u_dig, u_pickup, u_deposit',
            None,
            'the effect kind (several bits when ambiguous among them); u_place_animal from the board transition, '
            'u_deposit = DROP or PLACE into the shed'),
           ('u_other_effect', None, 'an inventory change no single engine action explains (never seen in 60 seats)'),
           ('u_load_delta', None, 'carried-unit change of this worker'),
           ('u_idle_streak', None, 'consecutive observed no-effect steps (0, 1, 2) / 2'),
           ('u_undo_move', None, 'moved on both of the last two steps and is back on its t-2 tile')],
 'gaps': [('c_own_mean_age',
           '0',
           'mean days since planting/placement of own producers of P (0 if none; `cohorts:m.count_own` says whether '
           'any)'),
          ('c_rival_mean_age', '1', 'same, rival'),
          ('c_own_mean_first_day',
           '2',
           'mean absolute first-output day, start + first_yield_day (e.g. cows placed d2/d3/d4 = 11.0)'),
          ('c_rival_mean_first_day', '3', 'same, rival'),
          ('c_own_producer_days',
           '4',
           'Σ over own producers of max(0, day - first-output day): days in production so far (animal-days of the '
           'current herd)'),
          ('c_rival_producer_days', '5', 'same, rival'),
          ('c_own_pulses_left',
           '6',
           'Σ production events on days d+1..29 under full service (animals every interval; ongoing crops < '
           'max_yield; one-shot: its harvest day). Engine-checked'),
          ('c_rival_pulses_left', '7', 'same, rival'),
          ('c_own_held_units',
           '8',
           'units a HARVEST gives now (animals y; ongoing crops y; one-shot y once age >= first_yield_day); '
           'FERTILIZER row = animals with fertilizer_available. Engine-checked'),
          ('c_rival_held_units', '9', 'same, rival'),
          ('c_own_held_value', '10', 'c_own_held_units x current quote'),
          ('c_rival_held_value', '11', 'same, rival'),
          ('mk_unit_impact',
           '12',
           'quote(inv) - quote(inv+1): the quote drop one more unit sold now causes (0 at the $1 floor, E L658-660). '
           'Engine-checked'),
          ('mk_own_cannibal_1u',
           '13',
           'mk_unit_impact x (own board units to day 29 + own shed + carried): own later receipts lost to one more '
           "unit sold now. Town demand is price-independent (E L728-749), so the +1 persists; first order at today's "
           'inventory'),
          ('mk_rival_ext_1u',
           '14',
           'mk_unit_impact x rival board units to day 29: rival receipts lost (rival stock is private, not counted)'),
          ('mk_rel_value_1u',
           '15',
           'quote - mk_own_cannibal_1u + mk_rival_ext_1u: first-order Δown - Δrival of selling (or producing and '
           'selling) one more unit now'),
          ('an_own_unplaced', '16', 'EGG/MILK/WOOL: own GOOSE/COW/SHEEP in shed + carried (bought, not placed)'),
          ('an_site_backlog',
           '17',
           'unplaced animals needing that structure (COOP: geese; PASTURE: cows + sheep, shared) - free own '
           'structures of that type'),
          ('mr_producer_unit',
           '18',
           'rlog of the BUY_SEED price (crops) or BUY_ANIMAL price (EGG/MILK/WOOL) over own money'),
          ('mr_producer_unit_cannot',
           '19',
           'price > money (engine refuses, E L673-688). Crops: equals not `flags:can_buy_producer`; animals: pure '
           'money test (flags also needs shed room)'),
          ('mr_producer_lot',
           '20',
           'lot = crops: seed x own empty tiles (a fill of every empty tile); animals: price x free matching '
           'structures'),
          ('mr_producer_lot_cannot', '21', 'lot cost > money (engine buys fewer than the lot; test-checked)'),
          ('mr_buy_unit', '22', 'WHEAT/FERTILIZER: BUY_PRODUCT quote at inv-1 (E L598-601) over money'),
          ('mr_buy_unit_cannot',
           '23',
           'that quote > money (pure money test; `flags:can_buy_unit` also needs shed room)'),
          ('mr_buy_lot',
           '24',
           'WHEAT only: one day of feed for own board animals, bought unit by unit at quotes inv-1, inv-2, ... '
           '(engine-checked); FERTILIZER: 0 (no natural lot)'),
          ('mr_buy_lot_cannot', '25', 'that feed cost > money'),
          ('mr_sell_unit', '26', 'current quote / money: percent increase in money from selling one unit'),
          ('mr_sell_held',
           '27',
           'quote x own shed units / money: percent increase in money if the shed stock of P is sold at the quote '
           '(no price impact; carried goods excluded because SELL takes shed stock)'),
          ('g_rival_labour_today',
           '28',
           'rival wages paid today = Σ fib(i), i < rival hires_today (own: `flags:labour_spent_today`)'),
          ('g_own_board_capital',
           '29',
           'Σ own live producers x input price (seed / animal): capital on the board at cost'),
          ('g_rival_board_capital', '30', 'same, rival'),
          ('g_held_value_lead', '31', 'Σ over products (own - rival) held units x quote'),
          ('mr_money_zero', '32', 'own money <= 0'),
          ('mr_land',
           '33',
           'rlog of the next BUY_LAND price over money (0 when all quadrants owned). Affordability: reuse '
           '`flags:can_afford_land`'),
          ('mr_hire',
           '34',
           'rlog of the next HIRE wage fib(hires_today) over money. Affordability: reuse `flags:can_afford_hire`'),
          ('mr_shed_value', '35', 'Σ quote x own shed units over money'),
          ('L_own_produced', '0', 'units of P harvested/collected, same-day steps'),
          ('L_own_consumed', '1', 'WHEAT fed / FERTILIZER applied, same-day steps'),
          ('L_own_bought', '2', 'units bought, steps where own market flows are known'),
          ('L_own_sold', '3', 'units sold, same steps'),
          ('L_own_trade_account',
           '4',
           '$ of own sales - product buys attributed to P: exact on steps trading one product; on multi-product '
           'steps each product gets -net x quote(t-1) (buys at the inv-1 quote) and the remainder goes to '
           'L_own_trade_unattributed'),
          ('L_own_capital_spent',
           '5',
           '$ of seeds (crops) or animals (EGG/MILK/WOOL) bought for P. WHEAT feed is on the WHEAT trade account'),
          ('L_own_net_account', '6', 'L_own_trade_account - L_own_capital_spent'),
          ('L_rival_net_24',
           '7',
           'rival net units into market inventory over the last 24 steps (sales +, BUY_PRODUCT -), known steps only: '
           "total = inv(t) - inv(t-1) + town demand(t-1), rival = total - own. Equals the engine's exact "
           'displacement on 18,992/18,992 known product-steps'),
          ('L_rival_deficit_24', '8', 'largest net-buy run inside that window: max(0, -min prefix sum)'),
          ('L_rival_known_24',
           '9',
           'share of window steps where the rival flow of P is known (unknown: cross-day step with own trading, an '
           'inexact own step, or an own sale that may have hit the $1 floor)'),
          ('L_rival_net_72, L_rival_deficit_72, L_rival_known_72', '10-12', 'same over 72 steps'),
          ('L_rival_net_all', '13', 'rival net units since the first observed step (known steps)'),
          ('L_sell_held_vs_inc24',
           '14',
           "rlog of quote x own shed units over max(L_inc24, 1): the sale as a share of the last day's income"),
          ('L_own_land', '15', 'Σ land prices paid (from quadrants owned, E L712-725)'),
          ('L_own_labour',
           '16',
           "Σ wages: each day's fib sum over the largest hires_today seen that day. A hire in a day's last step is "
           'invisible (the day end resets hires_today); its cash then lands in trade (0 such hires in the 3 test '
           'seats)'),
          ('L_own_seeds', '17', 'Σ seed price x (seed inventory change + new plantings)'),
          ('L_own_animals',
           '18',
           'Σ animal price x (change in board + shed + carried animals + escapes). Misses an animal destroyed by '
           'shed overflow'),
          ('L_own_trade_net',
           '19',
           'money - start + land + labour + seeds + animals = product sales - WHEAT/FERTILIZER buys'),
          ('L_own_trade_unattributed',
           '20',
           'part of L_own_trade_net not attributed to one product (multi-product estimate remainder, unknown steps)'),
          ('L_rival_land', '21', 'same as 15 for the rival (public)'),
          ('L_rival_labour', '22', "same as 16 from the rival's public hires_today (engine-exact in test 6)"),
          ('L_rival_seeds_seen',
           '23',
           'Σ seed price x rival new plantings: lower bound (seeds bought but not planted are private)'),
          ('L_rival_animals_seen', '24', 'Σ animal price x rival new placements: lower bound'),
          ('L_rival_trade_residual',
           '25',
           'rival money - start + 21 + 22 + 23 + 24 = rival trade net - unseen seed/animal purchases '
           '(engine-checked)'),
          ('L_trade_lead', '26', 'L_own_trade_net - L_rival_trade_residual'),
          ('L_spend_lead',
           '27',
           'own (land + labour + seeds + animals) - rival (land + labour + seen seeds + seen animals)'),
          ('L_inc24',
           '28',
           'own trade cash over the last 24 observed steps (sales - product buys): the recent income rate'),
          ('L_inc24_nonpos', '29', 'L_inc24 <= 0 (then L_sell_held_vs_inc24 carries log1p(value))'),
          ('L_own_flow_known_frac', '30', 'share of observed transitions with known own market flows'),
          ('L_contiguous', '31', 'every step since 0 was observed in order (0 after a missed step)')],
 'tileclock': [('tc_clk_{wheat,carrot,tomato,strawberry,melon}_harvests',
                '0,9,18,27,36',
                'harvest events a single planting started today still reaches by LASTDAY under full service '
                '(ongoing: production mornings ≤ LASTDAY, max 4; one-shot: 1 if age ≥ first_yield_day is reachable, '
                'else 0). lin/4'),
               ('tc_clk_{crop}_units',
                '1,10,19,28,37',
                'units that planting yields by LASTDAY, watered daily, no fertilizer, horizon-adjusted (a one-shot '
                'harvested on LASTDAY with fewer waterings counts: WHEAT from d27 = 2). lin/10'),
               ('tc_clk_{crop}_units_fert',
                '2,11,20,29,38',
                'same with every useful day fertilized (fert all). lin/10'),
               ('tc_clk_{crop}_wait_loss',
                '3,12,21,30,39',
                'units(start today) − units(start tomorrow), unfertilized: the cost of waiting one day. lin/10'),
               ('tc_clk_{crop}_days_to_cliff',
                '4,13,22,31,40',
                "number of start days from today (incl. today) that still give today's harvest count; 1 = today is "
                'the last such day; 0 = a start today yields nothing (strawberry: 4 harvests through d13, 3 through '
                'd15, 2 through d17, 1 through d19). lin/30'),
               ('tc_clk_{crop}_days_to_last_start',
                '5,14,23,32,41',
                'number of start days from today (incl. today) that still give ≥1 harvest; 0 = none. lin/30'),
               ('tc_clk_{crop}_first_harvest_days',
                '6,15,24,33,42',
                'days from today to the first harvest (ongoing: first production morning; one-shot: earliest '
                'harvestable day already holding the final units); 0 when none. lin/30'),
               ('tc_clk_{crop}_net_quote',
                '7,16,25,34,43',
                'units × sell quote(crop) − seed price ($; −seed when it yields nothing). slog/100'),
               ('tc_clk_{crop}_net_fert_quote',
                '8,17,26,35,44',
                'units_fert × quote − seed − fertilizers used by the fert-all rule × FERTILIZER buy quote ($; melon '
                'uses 0 fertilizers: waterings alone reach the cap). slog/100'),
               ('tc_svc_season_units',
                '45',
                'units this PLANT/animal tile yields by LASTDAY under full service incl. stored yield (= TF '
                'season_units_remaining, tested equal on every tile). lin/10'),
               ('tc_svc_season_value', '46', 'season_units × sell quote ($). slog/100'),
               ('tc_svc_gain_now_units',
                '47',
                "immediate units from today's WATER/FEED: one-shot water_gain_now (0 if it can never be harvested); "
                "ongoing/animal TF gain_tonight_serviced − gain_tonight_current (tonight's production with service "
                'minus as-is; 0 if it dies). lin/10'),
               ('tc_svc_gain_now_value', '48', 'gain_now_units × quote ($). slog/100'),
               ('tc_svc_skip_loss_units',
                '49',
                'season units(full) − season units(skip): what leaving the tile alone today costs, incl. death '
                'tonight (whole remaining season, salvage harvest allowed), lost water bonus, lost fertilizer bonus '
                'on a production night, lost pending care bonus. 0 when already serviced. lin/10'),
               ('tc_svc_skip_loss_value',
                '50',
                'skip_loss_units × quote − FEEDs saved × WHEAT buy quote ($; negative = skipping pays, e.g. a $1 '
                'product). slog/100'),
               ('tc_svc_care_gain_units',
                '51',
                'animals not cared today: season units(FEED+CARE today) − units(FEED only today); 0 if cared, if no '
                'production remains to consume the bonus, or if the cap absorbs it. lin/10'),
               ('tc_svc_care_gain_value', '52', 'care_gain_units × quote ($). slog/100'),
               ('tc_svc_fert_gain_units',
                '53',
                'PLANT: season units(fert now) − units(full): what one FERTILIZE applied now adds (it covers today, '
                '+1, +2). 0 on animals. lin/10'),
               ('tc_svc_fert_gain_net_value',
                '54',
                'fert_gain_units × quote − FERTILIZER sell quote ($; the fertilizer could be sold instead; −quote '
                'when it adds nothing). slog/100'),
               ('tc_svc_fert_takes_tonight',
                '55',
                "1 if fertilizing now adds a unit that the harvest keeps to today's watering (one-shot) or tonight's "
                'production (ongoing, harvested before the day end). lin/1'),
               ('tc_svc_fert_upside_units',
                '56',
                "PLANT: season units(fert all) − units(full): the tile's remaining fertilizer upside. lin/10"),
               ('tc_svc_collect_fert_value',
                '57',
                'animal tile with fertilizer_available: value of the fertilizer COLLECT_FERTILIZER yields = '
                "max(FERTILIZER sell quote, best fert_gain_units × quote over that farm's plants) ($). slog/100"),
               ('tc_svc_cap_loss_tonight_units',
                '58',
                "units the cap discards at tonight's production that a HARVEST today would save (ongoing: "
                'y+bonus−max_yield; animals: y+1+pending−max_held minus the part that overflows even when empty), '
                'under service. lin/10'),
               ('tc_g_own_fert_held', '59', 'own shed + carried FERTILIZER (private, own). lin/10'),
               ('tc_g_own_fert_collectable', '60', 'own animal tiles with fertilizer_available. lin/10'),
               ('tc_g_own_fert_gain_units_total',
                '61',
                'Σ own tiles fert_gain_units: units one fertilizer per plant would add now. lin/10'),
               ('tc_g_own_fert_takes_tonight_total', '62', 'Σ own tiles fert_takes_tonight. lin/10'),
               ('tc_g_own_skip_loss_value_total',
                '63',
                'Σ own tiles skip_loss_value: $ at risk if nothing more is serviced today. slog/100'),
               ('tc_g_rival_skip_loss_value_total', '64', "same over the rival's public tiles. slog/100"),
               ('tc_g_own_care_gain_units_total', '65', 'Σ own tiles care_gain_units. lin/10')],
 'lattice': [('kind:EMPTY, kind:LOCKED, kind:WEED, kind:PLANT, kind:COOP, kind:PASTURE',
              '0-5',
              'tile kind one-hot (EMPTY = None tile, LOCKED = "LOCKED"; COOP/PASTURE with or without an animal)'),
             ('item:WHEAT..MELON, item:GOOSE, item:COW, item:SHEEP',
              '6-13',
              'crop (PLANT) or animal one-hot; 0 on empty structures'),
             ('watered_today', '14', 'plant watered_today, 0/1'),
             ('fed_today', '15', 'animal fed_today, 0/1'),
             ('cared_today', '16', 'animal cared_today, 0/1'),
             ('days_unwatered', '17', 'consecutive_unwatered / 2, clip [0, 1] (1 = WEED at dusk)'),
             ('days_unfed', '18', 'consecutive_unfed / 2, clip [0, 1] (1 = escape at dusk)'),
             ('units_held', '19', 'yield_units / 6, clip [0, 1] (max_yield, max_held <= 6)'),
             ('fertilizer_available', '20', 'animal fertilizer_available, 0/1'),
             ('workers', '21', 'log1p(units standing here) / log1p(25): own units on farm 0, rival units on farm 1'),
             ('farmer', '22', "1 where that farm's farmer stands"),
             ('carried_units',
              '23',
              'own farm only: log1p(sum of inventories of own units here) / log1p(100), clip [0, 2]; 0 on farm 1 '
              '(private)'),
             ('carried_wheat',
              '24',
              'own farm only: log1p(WHEAT carried by own units here) / log1p(100), clip [0, 2]; 0 on farm 1'),
             ('job:WATER_TONIGHT, FEED_TONIGHT, HARVEST_DECAYING, WATER_TODAY, FEED_TODAY, HARVEST, COLLECT, '
              'FERTILIZER, PLANT_EMPTY, WEED, CARE',
              '25-35',
              'legacy flip.py code numbering (NONE = all 0), with terminal-aware V8 priorities: no fatal WATER/FEED '
              'or future-only FEED/CARE when no dusk remains; immediate one-shot WATER gains remain possible; rival '
              'empty tiles are PLANT_EMPTY (seeds unknown)'),
             ('shed_access', '36', 'static: tile is one of (4,4) (5,4) (4,5) (5,5)'),
             ('shed_dist', '37', 'static: Manhattan distance to the nearest shed-access tile / 9'),
             ('present', '38', '1 on every cell when the slot has an observation'),
             ('same_day',
              '39',
              "1 when the slot's day equals day(t): no dusk dismissal/respawn between the slot and t")]}
# --- END DOC_DEFS

if __name__ == "__main__":
    main()
