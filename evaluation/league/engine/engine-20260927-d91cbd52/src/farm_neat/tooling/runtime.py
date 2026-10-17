"""Reference the pinned interpreters declared in project/runtime/interpreter.toml.

This never installs, builds or copies an environment: it only creates local
symlinks under work/ and execs an existing interpreter with the declared
environment variables. Standard library only; safe to run with plain python3.

    python3 -B src/farm_neat/tooling/runtime.py setup
    python3 -B src/farm_neat/tooling/runtime.py check
    python3 -B src/farm_neat/tooling/runtime.py suite
    python3 -B src/farm_neat/tooling/runtime.py run -- src/farm_neat/training/program_bc.py --help
    python3 -B src/farm_neat/tooling/runtime.py pytest -- tests -q

`suite` is the one command that runs everything: the layout check, the scaffold's
standard-library checks and the whole imported pytest suite on the pinned interpreter.
`run` and `check` select exactly the current machine's training_by_machine entry.
The registered site precedes inherited PYTHONPATH sites; only the legacy training
profile uses the shared pytest-path links. `setup`, `pytest` and `suite` refuse
other profiles before any subprocess or filesystem mutation because no portable
pytest setup is registered for them. On legacy training, `check` also reports the
engine-only interpreter, which must never publish a checkpoint. This launcher
selects commands; the trainer's runtime/hash/site/identity gates remain authoritative.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / "project/runtime/interpreter.toml"


def lock() -> dict:
    spec = tomllib.loads(LOCK.read_text())
    if spec.get("schema") != "farmneat-runtime-interpreters-v1":
        raise ValueError("unsupported runtime lock schema")
    return spec


def link(target: Path, name: Path) -> str:
    if not target.exists():
        raise ValueError(f"declared interpreter/source is missing: {target}")
    name.parent.mkdir(parents=True, exist_ok=True)
    if name.is_symlink() or name.exists():
        name.unlink()
    name.symlink_to(target)
    return f"{name.relative_to(ROOT)} -> {target}"


def training_section(spec: dict) -> str:
    """Use the existing lock's machine registration, with no inferred fallback."""
    machine = platform.machine()
    mapping = spec.get("training_by_machine")
    if not isinstance(mapping, dict) or machine not in mapping:
        raise ValueError(f"no pinned training runtime registered for machine: {machine}")
    section = mapping[machine]
    if not isinstance(section, str) or not isinstance(spec.get(section), dict):
        raise ValueError(f"unknown training runtime section for {machine}: {section!r}")
    profile = spec[section]
    if (not isinstance(profile.get("path"), str) or not profile["path"]
            or not isinstance(profile.get("environment"), dict)
            or not all(isinstance(k, str) and isinstance(v, str)
                       for k, v in profile["environment"].items())):
        raise ValueError(f"invalid training runtime section for {machine}: {section!r}")
    return section


def require_pytest_setup(section: str, command: str) -> None:
    # pytest_path belongs to legacy training; do not use its x86 source or
    # rewrite its shared links on a profile with no registered pytest setup.
    if section != "training":
        raise ValueError(f"{command} unsupported for runtime {section}: "
                         "architecture-specific pytest/setup is not registered; "
                         "use run for the registered interpreter")


def require_executable(path: Path, section: str) -> Path:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"selected runtime {section} executable is missing or not executable: {path}")
    return path


def setup(spec: dict) -> int:
    section = training_section(spec)
    require_pytest_setup(section, "setup")
    path = require_executable(Path(spec[section]["path"]), section)
    print(link(path, ROOT / spec[section]["link"]))
    source, home = Path(spec["pytest_path"]["source"]), ROOT / spec["pytest_path"]["directory"]
    for member in spec["pytest_path"]["members"]:
        print(link(source / member, home / member))
    # Git does not track empty directories, so a fresh clone lacks the declared homes.
    subprocess.run([sys.executable, "-B", str(ROOT / "src/farm_neat/tooling/layout.py"),
                    "--create"], check=True)
    print("No environment was built, installed or copied.")
    return 0


def environment(spec: dict, *, section: str | None = None) -> dict:
    section = training_section(spec) if section is None else section
    profile = spec[section]
    env = dict(os.environ, **profile["environment"])
    parts = [str(ROOT / "src")]
    if "pythonpath" in profile:
        parts.append(profile["pythonpath"])
    if section == "training":
        home = ROOT / spec["pytest_path"]["directory"]
        if home.is_dir():
            parts.append(str(home))
    env["PYTHONPATH"] = os.pathsep.join(parts + list(filter(None, [env.get("PYTHONPATH")])))
    return env


def interpreter(spec: dict, *, section: str) -> Path:
    linked = spec[section].get("link")
    local = ROOT / linked if linked else None
    if local is not None and local.exists():
        return local
    return Path(spec[section]["path"])


PROBE = ("import sys, importlib\n"
         "def version(name):\n"
         "    try: return importlib.import_module(name).__version__\n"
         "    except Exception as error: return f'ABSENT ({type(error).__name__})'\n"
         "print(sys.version.split()[0], 'torch', version('torch'),"
         " 'numpy', version('numpy'), 'pytest', version('pytest'))\n")


def check(spec: dict) -> int:
    selected = training_section(spec)
    # Fail before any probe, rather than diagnosing engine-only after a missing
    # training runtime and inadvertently making that look like a fallback.
    require_executable(interpreter(spec, section=selected), selected)
    profiles = [(selected, f"{selected} (checkpoints, campaign, BC)")]
    if selected == "training":
        profiles.append(("engine_only", "engine-only (never publishes a checkpoint)"))
    failures = 0
    for section, label in profiles:
        path = interpreter(spec, section=section)
        print(f"\n{label}\n  {path}")
        require_executable(path, section)
        done = subprocess.run([str(path), "-B", "-c", PROBE], capture_output=True, text=True,
                              env=environment(spec, section=selected) if section == selected else None)
        observed = done.stdout.strip() or done.stderr.strip()
        print(f"  {observed}")
        if done.returncode:
            print(f"  FAILED: probe exited {done.returncode}")
            failures += 1
        for key in ("torch", "numpy"):
            if spec[section].get(key) and spec[section][key] not in observed:
                print(f"  MISMATCH: lock declares {key} {spec[section][key]}")
                failures += 1
    print("\nVersions are read from the live interpreters; this is not a checkpoint validation.")
    return 1 if failures else 0


def suite(spec: dict) -> int:
    """Layout, no-waste scaffold checks, module reachability, then the whole pytest suite."""
    section = training_section(spec)
    require_pytest_setup(section, "suite")
    training = str(require_executable(interpreter(spec, section=section), section))
    stages = [
        ("layout", [sys.executable, "-B", "src/farm_neat/tooling/layout.py", "--check"], None),
        ("scaffold", [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests",
                      "-p", "test_workspace.py"], None),
        ("reachability", [sys.executable, "-B", "src/farm_neat/tooling/reachability.py"], None),
        ("imported stack", [training, "-B", "-m", "pytest", "tests", "-q",
                            "-p", "no:cacheprovider"], environment(spec, section=section)),
    ]
    failures = []
    for label, command, env in stages:
        print(f"\n=== {label}: {' '.join(command)}")
        if subprocess.call(command, cwd=ROOT, env=env) != 0:
            failures.append(label)
    print("\nFAILED: " + ", ".join(failures) if failures else "\nAll stages passed.")
    print("Tests prove the migrated code behaves as it did in farm; they prove no playing strength.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("setup", "check", "suite", "run", "pytest"))
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        spec = lock()
        if args.command == "setup":
            return setup(spec)
        if args.command == "check":
            return check(spec)
        if args.command == "suite":
            return suite(spec)
        section = training_section(spec)
        if args.command == "pytest":
            require_pytest_setup(section, "pytest")
        rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
        if not rest:
            print("nothing to run; pass the command after --", file=sys.stderr)
            return 2
        path = require_executable(interpreter(spec, section=section), section)
        # no:cacheprovider: a root .pytest_cache is undeclared and fails layout --check.
        prefix = ["-B", "-m", "pytest", "-p", "no:cacheprovider"] if args.command == "pytest" else ["-B"]
        return subprocess.call([str(path), *prefix, *rest], cwd=ROOT,
                               env=environment(spec, section=section))
    except (ValueError, OSError) as error:
        print(f"runtime: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
