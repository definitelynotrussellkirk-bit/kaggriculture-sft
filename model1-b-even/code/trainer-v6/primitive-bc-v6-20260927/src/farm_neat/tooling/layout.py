"""Find file owners and check/materialize the declared layout; standard library only.

This is a placement check, not a security sandbox, storage quota or code audit.
It never imports research code and never deletes, moves or copies files.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[3]
CONTRACTS = "docs/contracts"


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name
            or any(p in (".", "..") for p in name.split("/"))
            or path.as_posix() != name or name.startswith(".git/")):
        raise ValueError(f"Expected a normalized repository-relative path: {name!r}")
    return name


def checked_path(root: Path, name: str) -> Path:
    safe_name(name)
    result = root
    for part in PurePosixPath(name).parts:
        result = result / part
        if result.is_symlink():
            raise ValueError(f"Refusing symlink in managed path: {name}")
    return result


def read_layout(root: Path = ROOT) -> dict:
    spec = tomllib.loads(checked_path(root, "project/layout.toml").read_text())
    if spec.get("version") != 1:
        raise ValueError("Unsupported layout version")
    keys = set()
    for section in ("files", "containers", "areas"):
        entries = spec.get(section)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Missing layout section: {section}")
        for name, value in entries.items():
            safe_name(name)
            if name == ".git" or name in keys:
                raise ValueError(f"Duplicate or reserved path: {name}")
            keys.add(name)
            purpose = value.get("purpose") if isinstance(value, dict) else value
            if not isinstance(purpose, str) or not purpose.strip():
                raise ValueError(f"Missing purpose: {name}")
            if section == "areas":
                if not isinstance(value, dict):
                    raise ValueError(f"Area needs purpose and contract fields: {name}")
                contract = value.get("contract", "")
                if not contract or "/" in contract:
                    raise ValueError(f"Invalid contract: {name}")
                target = checked_path(root, f"{CONTRACTS}/{contract}.md")
                if not target.is_file():
                    raise ValueError(f"Missing contract: {target}")
                if "local" in value and not isinstance(value["local"], bool):
                    raise ValueError(f"local must be boolean: {name}")
                for field in value:
                    if field.startswith("max_") and (
                            field not in LIMIT_FIELDS or type(value[field]) is not int
                            or value[field] < 1):
                        raise ValueError(f"Invalid cap {field}: {name}")
    directories = set(spec["containers"]) | set(spec["areas"])
    for name in keys:
        parent = str(PurePosixPath(name).parent)
        if parent != "." and parent not in directories:
            raise ValueError(f"Undeclared parent of {name}: {parent}")
    cleanup = spec.get("cleanup", {})
    if not isinstance(cleanup, dict) or "default" not in cleanup:
        raise ValueError("Missing default cleanup rule")
    for name, rule in cleanup.items():
        if name != "default":
            owner(spec, name)
        if not isinstance(rule, str) or not rule.strip():
            raise ValueError(f"Empty cleanup rule: {name}")
    return spec


def owner(spec: dict, name: str) -> tuple[str, str, str, bool]:
    """Return nearest declared home, purpose, contract path, local flag."""
    safe_name(name)
    for section in ("files", "containers"):
        if name in spec[section]:
            return name, spec[section][name], f"{CONTRACTS}/layout.md", False
    matches = [p for p in spec["areas"] if name == p or name.startswith(p + "/")]
    if not matches:
        raise ValueError(f"No declared owner for {name}; do not create it yet")
    home = max(matches, key=len)
    value = spec["areas"][home]
    return home, value["purpose"], f"{CONTRACTS}/{value['contract']}.md", value.get("local", False)


def create_homes(root: Path, spec: dict) -> int:
    directories = sorted(set(spec["containers"]) | set(spec["areas"]),
                         key=lambda p: (p.count("/"), p))
    # Validate all targets before adding anything. Existing content is untouched.
    for name in directories:
        target = checked_path(root, name)
        if target.exists() and not target.is_dir():
            raise ValueError(f"A file occupies directory home: {name}")
    count = 0
    for name in directories:
        target = checked_path(root, name)
        if not target.exists():
            target.mkdir()
            count += 1
    return count


def cleanup_rule(spec: dict, name: str) -> str:
    """Inherit one rule, using the same longest-prefix convention as placement."""
    safe_name(name)
    matches = [p for p in spec["cleanup"] if p != "default"
               and (name == p or name.startswith(p + "/"))]
    return spec["cleanup"][max(matches, key=len) if matches else "default"]


LIMIT_FIELDS = {"max_files", "max_entries", "max_markdown_files", "max_bytes", "max_file_bytes"}
NOTE_HOMES = {"docs/guides", "review/tasks", "review/discussions"}
NOTE_FIELDS = ("Purpose", "Owner", "Status", "Evidence", "Next", "Remove-when")
DOCUMENT_HOMES = {"docs/contracts", "docs/guides", "docs/reference", "kagglewriteup",
                  "kagglesub", "review/tasks", "review/discussions", "review/continuations"}
ENTRY_DOCS = {"README.md", "AGENTS.md", "CLAUDE.md"}


def note_fields(path: Path) -> dict[str, str]:
    """The six lifecycle fields, or an empty string where one is missing/duplicated."""
    text = path.read_text(encoding="utf-8")
    fields = {}
    for field in NOTE_FIELDS:
        values = re.findall(rf"^{field}:[ \t]*(.*)$", text, re.MULTILINE)
        fields[field] = values[0].strip() if len(values) == 1 else ""
    return fields


def check_note(path: Path, name: str, home: str) -> list[str]:
    """Syntactic handoff checks, not proof that its evidence/claims are true."""
    text = path.read_text(encoding="utf-8")
    errors = []
    for field in NOTE_FIELDS:
        values = re.findall(rf"^{field}:[ \t]*(.*)$", text, re.MULTILINE)
        if len(values) != 1 or not values[0].strip():
            errors.append(f"{name}: needs exactly one nonempty {field}: field")
    statuses = re.findall(r"^Status:[ \t]*(.*)$", text, re.MULTILINE)
    allowed = {"ACTIVE", "BLOCKED"} if home == "docs/guides" else {"OPEN", "BLOCKED"}
    if statuses and statuses[0].strip() not in allowed:
        errors.append(f"{name}: Status must be {sorted(allowed)}; consolidate resolved notes")
    return errors


def check_links(root: Path, documents: list[Path]) -> list[str]:
    errors = []
    for path in documents:
        text = path.read_text(encoding="utf-8")
        # Our small Markdown convention uses ordinary relative links, no titles.
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", link):
                continue
            filename, _, anchor = link.partition("#")
            target = path.parent / filename if filename else path
            target = target.resolve()
            label = f"{path.relative_to(root)}: {link}"
            if not target.is_relative_to(root.resolve()) or not target.exists():
                errors.append(f"Broken or outside-workspace link: {label}")
            elif anchor and target.suffix == ".md":
                headings = re.findall(r"^#+ (.+)$", target.read_text(), re.MULTILINE)
                anchors = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}
                if anchor not in anchors:
                    errors.append(f"Broken heading link: {label}")
    return errors


def usage(path: Path) -> dict[str, int]:
    """Logical retained bytes, no symlink traversal or archive expansion.

    Hardlinks count at each pathname intentionally: they still represent duplicate
    artifacts in a portable package. This is not the shared allocated-byte quota.
    """
    result = {field: 0 for field in LIMIT_FIELDS}
    result["max_entries"] = sum(1 for _ in path.iterdir())

    def fail(error: OSError) -> None:
        raise error  # An unreadable tree is not zero usage.

    for base, dirs, files in os.walk(path, followlinks=False, onerror=fail):
        links = [d for d in dirs if (Path(base) / d).is_symlink()]
        for name in files + links:
            item = Path(base) / name
            size = item.lstat().st_size
            result["max_files"] += 1
            result["max_markdown_files"] += int(item.suffix.lower() == ".md")
            result["max_bytes"] += size
            result["max_file_bytes"] = max(result["max_file_bytes"], size)
    return result


def check_caps(root: Path, spec: dict) -> list[str]:
    errors = []
    for name, area in spec["areas"].items():
        if not LIMIT_FIELDS.intersection(area):
            continue
        try:
            path = checked_path(root, name)
            if not path.is_dir():
                continue  # Structural check reports missing homes.
            measured = usage(path)
            for field in sorted(LIMIT_FIELDS.intersection(area)):
                if measured[field] > area[field]:
                    errors.append(f"Cap exceeded: {name} {field}={measured[field]} > {area[field]}")
        except (ValueError, OSError) as error:
            errors.append(str(error))
    archive = root / "review/inbox/FARM-REVIEW.zip"
    if archive.is_file() and archive.stat().st_size >= 500_000_000:
        errors.append("Cap exceeded: outgoing ZIP must be below 500000000 bytes")
    return errors


def check(root: Path, spec: dict) -> list[str]:
    errors = []
    documents = []
    for name in sorted(set(spec["containers"]) | set(spec["areas"])):
        try:
            if not checked_path(root, name).is_dir():
                errors.append(f"Missing directory: {name}; run --create")
        except ValueError as error:
            errors.append(str(error))

    def walk(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            name = path.relative_to(root).as_posix()
            if name == ".git":
                continue  # Git can be a directory or a worktree metadata file.
            try:
                checked_path(root, name)
                home, _, _, local = owner(spec, name)
                if name in spec["files"] and not path.is_file():
                    raise ValueError(f"Expected file: {name}")
                if name in spec["containers"] and not path.is_dir():
                    raise ValueError(f"Expected directory: {name}")
                if path.is_dir() and not local:
                    walk(path)
                # Contents of local data/environment homes are intentionally opaque.
                # Empty directories are not proof of an implementation.
                if path.is_file() and home in NOTE_HOMES:
                    if path.suffix != ".md":
                        raise ValueError(f"Notes use Markdown, not payloads: {name}")
                    errors.extend(check_note(path, name, home))
                if path.is_file() and path.suffix == ".md":
                    # A document is placed either in a document home, or named
                    # individually in the map with its own purpose (home == name).
                    if name not in ENTRY_DOCS and home not in DOCUMENT_HOMES and home != name:
                        raise ValueError(f"Misplaced document: {name}; use docs/ or review/, not source folders")
                    documents.append(path)
            except (ValueError, OSError, UnicodeError) as error:
                errors.append(str(error))
    walk(root)
    errors.extend(check_caps(root, spec))
    errors.extend(check_links(root, documents))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--create", action="store_true", help="Add only declared empty directory homes")
    mode.add_argument("--check", action="store_true", help="Read-only structure check (default)")
    mode.add_argument("--why", metavar="PATH", help="Explain an existing or proposed relative path")
    mode.add_argument("--find", metavar="NEED", help="Find homes by responsibility or path text")
    mode.add_argument("--cleanup", action="store_true", help="Show retention/removal rules; never delete")
    mode.add_argument("--notes", action="store_true", help="One line per open note: path, Status, Next, Remove-when")
    args = parser.parse_args(argv)
    try:
        spec = read_layout()
        if args.create:
            print(f"Created {create_homes(ROOT, spec)} directories; no files copied/moved/deleted.")
        elif args.why:
            home, purpose, contract, local = owner(spec, args.why)
            target = checked_path(ROOT, args.why)
            print(f"Home: {home}\nPurpose: {purpose}\nContract: {contract}")
            print(f"Cleanup: {cleanup_rule(spec, args.why)}")
            print(f"Path: {'present' if target.exists() else 'not created'}; "
                  f"{'local data' if local else 'maintained source/documentation'}")
            for name, area in spec["areas"].items():
                if args.why == name or args.why.startswith(name + "/"):
                    caps = ", ".join(f"{field}={area[field]}"
                                     for field in sorted(LIMIT_FIELDS.intersection(area)))
                    if caps:
                        print(f"Caps ({name}): {caps}")
            print("Placement does not certify implementation or measured capability.")
        elif args.notes:
            rows = [path for home in sorted(NOTE_HOMES)
                    for path in sorted((ROOT/home).glob("*.md")) if (ROOT/home).is_dir()]
            for path in rows:
                fields = note_fields(path)
                print(f"{path.relative_to(ROOT).as_posix()} | {fields['Status']} | "
                      f"next: {fields['Next']} | remove-when: {fields['Remove-when']}")
            print(f"{len(rows)} open note(s). A note is deleted once its conclusion lives in "
                  f"code, a contract or a test; it is never marked DONE.")
        elif args.cleanup:
            for name, rule in spec["cleanup"].items():
                print(f"{name}: {rule}")
            print("Read-only rules, not a deletion list. Check live users and dependencies before cleanup.")
        elif args.find:
            query = args.find.casefold()
            matches = []
            for name in [*spec["files"], *spec["containers"], *spec["areas"]]:
                _, purpose, _, _ = owner(spec, name)
                if query in f"{name} {purpose}".casefold():
                    matches.append(f"{name}: {purpose}")
            print("\n".join(matches) if matches else f"No home matches: {args.find}")
            return 0 if matches else 1
        else:
            errors = check(ROOT, spec)
            if errors:
                print("\n".join(errors), file=sys.stderr)
                return 1
            print("PASS: homes, document placement/links, note lifecycle fields and local count/size caps.")
            print("Not storage admission, an OS quota, a content audit, a game test or a capability claim.")
    except (ValueError, OSError, tomllib.TOMLDecodeError) as error:
        print(f"Layout error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
