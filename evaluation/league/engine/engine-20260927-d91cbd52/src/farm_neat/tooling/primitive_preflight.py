"""Read-only source preflight for the current v3.1 primitive-policy entrypoints.

This does not import any project module, load weights, install dependencies, or
execute a game. It follows statically declared primitive-local imports (including
imports inside functions), and checks directly imported farm_neat package paths.
It deliberately does not search for unrelated namesakes elsewhere under work/.

A present source closure is NOT a working runtime. Dynamic import targets,
transitive imports inside farm_neat, installed third-party versions, native
binaries, source-pin compatibility, datasets and checkpoints remain unchecked.

Usage: PYTHONPATH=src python -m farm_neat.tooling.primitive_preflight --root .
Exit 2 means a missing/unreadable/invalid source dependency, 0 means only that the
statically checked source paths are present. Neither result certifies execution.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ENTRY_MODULES = ("agent", "codec", "data", "feature_tokens", "features", "model", "train", "unit_index")
HOME = Path("work/runs/maintainer/primitive-bc-v31-code")
EXTERNAL_MODULES = frozenset({"numpy", "torch", "orjson"})


def _module_path(base: Path, module: str) -> Path:
    path = base.joinpath(*module.split("."))
    package = path / "__init__.py"
    return package if package.is_file() else path.with_suffix(".py")


def audit_sources(root: Path | str) -> dict:
    """Return an auditable static source inventory; never return a runtime-ready claim."""
    root = Path(root).resolve()
    home = root / HOME
    queue = [(home / f"{name}.py", name, "entrypoint", None) for name in ENTRY_MODULES]
    seen: set[Path] = set()
    checked: dict[str, dict] = {}
    missing: list[dict] = []
    external: set[str] = set()

    def relative(path: Path) -> str:
        return path.relative_to(root).as_posix()

    def inventory(path: Path, *, parsed: bool) -> bytes:
        raw = path.read_bytes()
        checked[relative(path)] = {"path": relative(path), "sha256": hashlib.sha256(raw).hexdigest(),
                                   "static_imports_parsed": parsed}
        return raw

    while queue:
        path, module, imported_by, line = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        origin = {"module": module, "expected_path": relative(path), "imported_by": imported_by, "line": line}
        try:
            raw = inventory(path, parsed=True)
            tree = ast.parse(raw, filename=relative(path))
        except (OSError, SyntaxError, UnicodeError, ValueError) as exc:
            missing.append({**origin, "reason": type(exc).__name__, "detail": str(exc)[:300]})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    # None of the current entrypoints use relative imports. Do
                    # not silently certify a future form without resolving it.
                    missing.append({**origin, "module": "." * node.level + (node.module or ""),
                                    "line": node.lineno, "reason": "unresolved_relative_import"})
                    continue
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                top = name.split(".", 1)[0]
                if top in sys.stdlib_module_names:
                    continue
                if top in EXTERNAL_MODULES:
                    external.add(top)
                    continue
                if top == "farm_neat":
                    target = _module_path(root / "src", name)
                    try:
                        inventory(target, parsed=False)
                    except OSError as exc:
                        missing.append({"module": name, "expected_path": relative(target),
                                        "imported_by": relative(path), "line": node.lineno,
                                        "reason": type(exc).__name__})
                    continue
                target = (_module_path(home.parent / "pbc-step-cache", name) if top == "build_cache"
                          else _module_path(path.parent, name))
                queue.append((target, name, relative(path), node.lineno))

    return {
        "schema": "primitive-source-closure-v1",
        "status": "SOURCE_CLOSURE_BLOCKED" if missing else "SOURCE_CLOSURE_PRESENT_NOT_EXECUTED",
        "execution_verified": False,
        "root": str(root),
        "entry_modules": list(ENTRY_MODULES),
        "checked_files": [checked[key] for key in sorted(checked)],
        "missing": sorted(missing, key=lambda row: (row["module"], row["imported_by"])),
        "external_modules_not_verified": sorted(external),
        "scope": "static primitive-local imports and directly imported farm_neat module paths",
        "not_verified": ["dynamic imports", "transitive imports inside farm_neat", "dependency versions",
                         "native binaries", "source pins", "datasets", "weights", "runtime behavior", "playing strength"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, help="optional JSON report path (parent must exist)")
    args = parser.parse_args(argv)
    report = audit_sources(args.root)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.write_text(text)
    print(text, end="")
    return 2 if report["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
