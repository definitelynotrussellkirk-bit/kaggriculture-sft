"""Static import graph from project/entrypoints.toml; names every unused module.

A module under src/ that no declared entry point can reach is dead weight: delete it
and its tests rather than justifying it. This never imports research code -- it parses
it -- so it runs under any Python 3.12 with no third-party dependency.

Nested and function-local imports count, because the lazy imports in this codebase are
real dependencies of the code paths that run.

    python3 -B src/farm_neat/tooling/reachability.py
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src"


def module_name(path: Path) -> str:
    rel = path.relative_to(SOURCE).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def modules() -> dict[str, Path]:
    return {module_name(p): p for p in sorted(SOURCE.rglob("*.py"))}


def imports(path: Path) -> set[str]:
    """Every dotted name this file imports: nested, relative and package-attribute."""
    package = module_name(path)
    if path.name != "__init__.py":
        package = package.rsplit(".", 1)[0] if "." in package else ""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".") if package else []
                anchor = anchor[:len(anchor) - node.level + 1]
                base = ".".join([*anchor, base] if base else anchor)
            if not base:
                continue
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def entry_points(root: Path = ROOT) -> list[str]:
    spec = tomllib.loads((root / "project/entrypoints.toml").read_text())
    names = [entry["module"] for entry in spec.get("step", [])]
    names += [entry["module"] for entry in spec.get("tool", [])]
    if not names:
        raise ValueError("project/entrypoints.toml declares no entry point")
    return names


def task_states(root: Path = ROOT) -> dict[str, str]:
    """Assignment state by task id when the authoritative review registry is present."""
    registry = root / "project/review.toml"
    if not registry.is_file():
        return {}
    spec = tomllib.loads(registry.read_text())
    states: dict[str, str] = {}
    for task in spec.get("assignments", []):
        task_id = task.get("id")
        state = task.get("schedule", {}).get("state")
        if isinstance(task_id, str) and task_id and isinstance(state, str) and state:
            states[task_id] = state
    return states


def planned(root: Path = ROOT) -> list[str]:
    """Temporary roots owned by unfinished tasks only.

    A planned entry is an escape hatch for a dispatched task while its module is
    being created.  Once the owning task is ACCEPTED, the module must be reachable
    from a real [[step]] or [[tool]] command instead of remaining a planned seed.
    """
    spec = tomllib.loads((root / "project/entrypoints.toml").read_text())
    states = task_states(root)
    names = []
    for entry in spec.get("planned", []):
        if set(entry) != {"module", "task", "purpose"} or not all(isinstance(v, str) and v for v in entry.values()):
            raise ValueError("planned entry needs module, task and purpose")
        if states.get(entry["task"]) == "ACCEPTED":
            raise ValueError(
                f"planned entry owned by accepted task {entry['task']}: {entry['module']}"
            )
        names.append(entry["module"])
    if len(names) != len(set(names)):
        raise ValueError("duplicate planned module")
    return names


def _walk(roots: list[str]) -> set[str]:
    known = modules()
    queue = [name for name in roots if name in known]
    seen: set[str] = set()
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        # A package is reached by any of its modules being reached.
        parent = name.rsplit(".", 1)[0] if "." in name else ""
        if parent and parent in known and parent not in seen:
            queue.append(parent)
        for target in imports(known[name]):
            while target:
                if target in known and target not in seen:
                    queue.append(target)
                    break
                target = target.rsplit(".", 1)[0] if "." in target else ""
    return seen


def command_reachable(root: Path = ROOT) -> set[str]:
    """Modules reachable from real [[step]] / [[tool]] roots only."""
    roots = entry_points(root)
    known = modules()
    missing = [name for name in roots if name not in known]
    if missing:
        raise ValueError("entry point is not a module under src/: " + ", ".join(sorted(missing)))
    return _walk(roots)


def reachable(root: Path = ROOT) -> set[str]:
    """Modules reachable from real commands plus unfinished planned roots."""
    roots = entry_points(root)
    roots += [name for name in planned(root) if name in modules()]
    # command_reachable performs the fail-closed check for real roots.
    command_reachable(root)
    return _walk(roots)


def assets(root: Path = ROOT) -> set[str]:
    """Files an entry point loads by path; exempt from the module-reachability rule."""
    spec = tomllib.loads((root / "project/entrypoints.toml").read_text())
    names = set()
    for entry in spec.get("asset", []):
        path = root / entry["path"]
        if not path.is_file():
            raise ValueError("declared asset is missing: " + entry["path"])
        if path.suffix == ".py":
            names.add(module_name(path))
    return names


def unused(root: Path = ROOT) -> list[str]:
    known = modules()
    return sorted(set(known) - reachable(root) - assets(root))


def main() -> int:
    commands = command_reachable()
    all_reached = reachable()
    dead = unused()
    if dead:
        print("Unused modules under src/ (delete them and their tests):", file=sys.stderr)
        print("\n".join("  " + name for name in dead), file=sys.stderr)
        return 1
    planned_only = all_reached - commands
    path_assets = assets()
    print(f"PASS: all {len(modules())} modules under src/ are accounted for by "
          f"{len(entry_points())} real entry points, unfinished planned roots, or "
          f"{len(path_assets)} declared path assets.")
    print(f"Command-reachable modules: {len(commands)}; "
          f"unfinished-planned-only modules: {len(planned_only)}; "
          f"declared path assets: {len(path_assets)}; accepted planned seeds: 0.")
    print("Reachable is not the same as exercised; the test suite is what shows behavior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
