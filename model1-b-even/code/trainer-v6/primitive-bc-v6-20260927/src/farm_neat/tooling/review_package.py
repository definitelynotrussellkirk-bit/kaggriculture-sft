"""Build the current source, semantic evidence and explicit-witness review ZIP.

Canonical publication and non-executing intake; no games or training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
import re
import secrets
import shutil
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT / "src"))

from farm_neat.tooling import layout, work_schedule
from farm_neat.tooling import review_intake, evidence_compaction as evidence
from farm_neat.training import storage_layout

OUTPUT = "review/inbox/FARM-REVIEW.zip"
DISPATCH_PROMPT = "review/DISPATCH-PROMPT.txt"
MANIFEST = "work/runs/review-package/MANIFEST.json"
SCHEMA = "farmneat-source-review-package-v1"
CONTEXT_SPEC = "project/review.toml"
ASSIGNMENT = "work/runs/reviewer-play/ASSIGNMENT.json"
SELF_TEST_PROGRESS = "work/runs/reviewer-play/review-selftest-progress/SELF-TEST-PROGRESS.json"
SELF_TEST_STATUS = "review/SELF-TEST-STATUS.json"
SELF_TEST_LAST_PASS = "work/runs/review-package/SELF-TEST-LAST-PASS.json"
SELF_TEST_LAST_RUN = "work/runs/review-package/SELF-TEST-LAST-RUN.json"
SELF_TEST_RECORD_SCHEMA = "farm-review-published-self-test-record-v1"
# Exact publisher/reviewer state, not portable fixture dependencies.
PACKAGING_OMISSIONS = {
    ASSIGNMENT: "Per-extraction ticket; each reviewer creates their own source-bound assignment.",
    SELF_TEST_PROGRESS: "Mutable local test progress, not an input to the published suite.",
    SELF_TEST_LAST_PASS: "Local last-green record; its bound coverage ships in SELF-TEST-STATUS.json.",
    SELF_TEST_LAST_RUN: "Local latest-run record, including failures; never a portable fixture.",
    evidence.POLICY: "Producer-only disposition index; its identity and aggregates ship in the manifest.",
}
DISPATCH = "project/dispatch.toml"
LOTTERY_TICKET = "farm-review-theory-ticket-v1"
DISPATCH_TICKET = "farm-review-dispatch-ticket-v1"
CONTINUATION_SCHEMA = "farm-review-continuation-v3"
LEGACY_CONTINUATION_SCHEMAS = {"farm-review-continuation-v1", "farm-review-continuation-v2"}
NATIVE_START_SCHEMA = "farm-review-native-start-v2"
LEGACY_NATIVE_START_SCHEMA = "farm-review-native-start-v1"
CONTINUATION_ROOT = "review/continuations"
DISPATCH_REGISTRY_ROOT = "review/inbox/dispatch-registry"
ACCEPTANCE_OUTCOMES = {"PASS", "CLOSED_NEGATIVE"}
TARGET_PARALLELISM = 8
CONTINUATION_LIMIT = 10_000_000
CONTINUATION_MEMBER_LIMIT = 1_024
# The accounted workspace contains many small retained evidence files.  Its full
# no-follow inode scan can exceed storage.status()'s 60-second interactive default;
# package publication is a bounded maintainer operation and must wait for a
# complete measurement rather than refusing an otherwise valid partial scan.
BUILD_STORAGE_SCAN_TIMEOUT = 300
# The owner permits review ZIPs below 500 MB. Keep a substantially tighter source
# packet while leaving room for complete accepted return evidence (never trim a
# worker's witnesses merely to satisfy the transport bound).
LIMIT = 400_000_000
WORK_LIMIT = 300_000_000
MEMBER_LIMIT = 8192
WITNESS_LIMIT = 2_000_000
WITNESS_COUNT = 8
QUICK_TEST_FILES = (
    "tests/unit/test_review_package.py",
    "tests/unit/test_review_context.py",
    "tests/unit/test_review_dispatch.py",
    "tests/unit/test_review_dispatch_lint.py",
    "tests/unit/test_review_intake.py",
    "tests/unit/test_search_ledger.py",
    "tests/unit/test_work_schedule.py",
)
THEOREMS = "review/THEOREMS.md"
FRONTIER = "review/FRONTIER.md"
ROOT_FILES = (".gitignore", "AGENTS.md", "CLAUDE.md", "README.md", "RUN.py",
              "pyproject.toml", "pytest.ini", "review/INSTRUCTIONS.md", DISPATCH_PROMPT, THEOREMS, FRONTIER)
TASK_DATA_FILES = ("work/datasets/corpus/games-v1-plus-repairs.jsonl.gz",)
FORMS = {"conditional_identity": "REPRESENTATION", "exhaustive_finite_check": "REPRESENTATION, finite-checked",
         "counterexample": "BOUNDARY", "data_contract": "STRUCTURE", "empirical_paired": "MEASUREMENT",
         "empirical_panel": "MEASUREMENT", "hypothesis": "HYPOTHESIS"}
TREES = ("src", "tests", "project", "docs", "kagglewriteup",
         "review/tasks", "review/discussions")
SUFFIXES = {".py", ".md", ".toml", ".json", ".ini", ".txt", ".svg", ".png",
            ".rs", ".c", ".cpp", ".h", ".cu", ".cuh", ".lock"}
CONTINUATION_SUFFIXES = SUFFIXES | {".csv", ".jsonl"}
# Exact derived presentation exports, never a filename heuristic for evidence.
DERIVED_FIGURES = {f"kagglewriteup/{name}.png": f"kagglewriteup/{name}.svg"
                   for name in ("architecture", "interaction", "training-objectives")}
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "target"}

def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _continuation_task(task_id: str) -> str:
    if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", task_id):
        raise ValueError("unsafe continuation task id")
    return task_id


def _continuation_path(task_id: str) -> str:
    return f"{CONTINUATION_ROOT}/{_continuation_task(task_id)}/CONTINUE.json"


def _ticket_task_id(ticket: dict) -> str:
    return _continuation_task(ticket.get("task_id", ticket["assignment"]["id"]))


def continuation_members(root: Path) -> list[str]:
    """Return the bounded tool-owned continuation bytes that must survive repackaging.

    These are control-plane handoff bytes, not admitted scientific evidence.  Work payloads
    needed for a resume live under ``workspace/`` copies, so publication never has to weaken
    the semantic-evidence policy for arbitrary ``work/`` artifacts.
    """
    base = layout.checked_path(root, CONTINUATION_ROOT)
    if not base.exists():
        return []
    if base.is_symlink() or not base.is_dir():
        raise ValueError("continuation root must be a regular directory")
    members, total = [], 0
    for task_dir in sorted(base.iterdir()):
        if task_dir.is_symlink() or not task_dir.is_dir():
            raise ValueError("continuation entries must be task directories")
        _continuation_task(task_dir.name)
        files = []
        for path in sorted(task_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError("symlink in continuation workspace: " + str(path))
            if not path.is_file():
                continue
            relative = path.relative_to(task_dir).as_posix()
            control = relative == "CONTINUE.json" or re.fullmatch(r"START-\d{6}\.json", relative)
            workspace = relative.startswith("workspace/") and path.suffix in CONTINUATION_SUFFIXES
            if not (control or workspace):
                raise ValueError("unsupported continuation member: " + path.relative_to(root).as_posix())
            files.append(path)
        if any(path.relative_to(task_dir).parts[0] == "workspace" for path in files) and not (task_dir / "CONTINUE.json").is_file():
            raise ValueError("continuation workspace lacks CONTINUE.json: " + task_dir.name)
        _native_start_receipts(root, task_dir.name)
        for path in files:
            total += path.stat().st_size
            members.append(path.relative_to(root).as_posix())
    if len(members) > CONTINUATION_MEMBER_LIMIT or total > CONTINUATION_LIMIT:
        raise ValueError("continuation workspace exceeds bounded review handoff budget")
    return members


def inventory(root: Path) -> list[str]:
    """Include current uncommitted files too; never infer membership from git."""
    names = set(ROOT_FILES)
    names.update(name for name in TASK_DATA_FILES if layout.checked_path(root, name).is_file())
    for tree in TREES:
        folder = layout.checked_path(root, tree)
        if not folder.is_dir():
            raise ValueError(f"missing source tree: {tree}")
        for base, dirs, files in os.walk(folder, followlinks=False):
            for name in dirs + files:
                if (Path(base) / name).is_symlink():
                    raise ValueError(f"symlink in package inputs: {Path(base) / name}")
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                path = Path(base) / name
                if path.suffix in {".pyc", ".pyo"}:
                    continue
                if path.suffix not in SUFFIXES or name.startswith("."):
                    raise ValueError(f"file outside source allowlist: {path}")
                relative = path.relative_to(root).as_posix()
                if relative in DERIVED_FIGURES:
                    if not layout.checked_path(root, DERIVED_FIGURES[relative]).is_file():
                        raise ValueError("derived figure lacks its SVG source: " + relative)
                    continue
                names.add(relative)
    view = evidence.packet_view(root)
    names.add(evidence.CATALOG)
    retained = {item['path'] for item in view['retained']}
    conflicts = retained & PACKAGING_OMISSIONS.keys()
    if conflicts:
        raise ValueError("packaging omission retained by evidence policy: " + ", ".join(sorted(conflicts)))
    names.update(retained)
    names.update(continuation_members(root))
    for name in names:
        path = layout.checked_path(root, name)
        if not path.is_file():
            raise ValueError(f"not a regular package input: {name}")
    return sorted(names)


def source_versions(root: Path, transitions: list[str]) -> dict[str, set[str]]:
    """Authenticate reversed source diffs as data; never import or execute them.

    Every hunk must exactly match its current endpoint. Only declared files may
    appear, and the reconstructed bytes must match their recorded old digest.
    This is review provenance, never training/checkpoint source admission.
    """
    restored = {}; versions = {}
    for relative in transitions:
        transition = json.loads(layout.checked_path(root, relative).read_text())
        patch = layout.checked_path(root, transition['patch']).read_bytes()
        if digest(patch) != transition['patch_sha256']:
            raise ValueError('source patch hash mismatch: '+relative)
        lines = patch.decode('utf-8').splitlines(keepends=True); i = 0; seen = set()
        while i < len(lines):
            if not lines[i].startswith('--- a/') or i+1 == len(lines) or not lines[i+1].startswith('+++ b/'):
                raise ValueError('only exact unified source diffs are supported')
            name = lines[i][6:].rstrip('\n')
            if lines[i+1][6:].rstrip('\n') != name or name not in transition['files'] or name in seen:
                raise ValueError('undeclared or duplicate source patch path')
            live = layout.checked_path(root, name)
            payload = restored.get(name, live.read_bytes()); endpoints = transition['files'][name]
            if digest(payload) != endpoints['after']:
                raise ValueError('source transition endpoint mismatch: '+name)
            current = payload.decode('utf-8').splitlines(keepends=True); output = []; cursor = 0; i += 2
            while i < len(lines) and not lines[i].startswith('--- a/'):
                match = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n?', lines[i])
                if not match: raise ValueError('malformed source hunk')
                old_start, old_count, new_start, new_count = match.groups()
                old_count = int(old_count) if old_count is not None else 1
                new_count = int(new_count) if new_count is not None else 1
                at = int(new_start) - (1 if new_count else 0)
                before = []; after = []; i += 1
                while len(before) < old_count or len(after) < new_count:
                    if i == len(lines) or not lines[i] or lines[i][0] not in ' +-':
                        raise ValueError('truncated source hunk')
                    kind, value = lines[i][0], lines[i][1:]; i += 1
                    if kind in ' -': before.append(value)
                    if kind in ' +': after.append(value)
                    if len(before) > old_count or len(after) > new_count:
                        raise ValueError('source hunk line count mismatch')
                if at < cursor or current[at:at+new_count] != after:
                    raise ValueError('source hunk does not match endpoint')
                output.extend(current[cursor:at]); output.extend(before); cursor = at+new_count
            output.extend(current[cursor:]); payload = ''.join(output).encode('utf-8')
            if digest(payload) != endpoints['before']:
                raise ValueError('historical source digest mismatch: '+name)
            restored[name] = payload; versions.setdefault(name,set()).add(endpoints['before']); seen.add(name)
        if seen != set(transition['files']): raise ValueError('source patch omits declared files')
    return versions


def note_fields(text: str, name: str) -> dict[str, str]:
    """A repeated or empty field is ambiguity, not a second valid continuation."""
    labels = {"Purpose", "Owner", "Status", "Evidence", "Next", "Remove-when"}
    fields = {}
    for line in text.splitlines():
        label, sep, value = line.partition(":")
        if sep and label in labels:
            if label in fields or not value.strip():
                raise ValueError("duplicate or empty six-field continuation value: " + name)
            fields[label] = value.strip()
    if set(fields) != labels:
        raise ValueError("complete six-field continuation note required: " + name)
    if fields["Status"] not in {"OPEN", "BLOCKED"}:
        raise ValueError("resolved notes must leave the active continuation map: " + name)
    return fields


def reading_context(root: Path, spec: dict, describe) -> dict:
    """Bound the COMPLETE maintained narrative, not just its entrypoint.

    Receipts/source are on-demand evidence, not silently elided parts of this
    reading set. The human view reads these exact bytes; there is no editable
    generated summary and no transcript store.
    """
    first = spec["read_first"]
    if not isinstance(first, list) or not first or len(set(first)) != len(first):
        raise ValueError("distinct nonempty read_first paths required")
    paths = list(dict.fromkeys([spec["entrypoint"], *first,
        *[n["path"] for n in spec["threads"]], *[n["path"] for n in spec["tasks"]]]))
    records = []
    for name in paths:
        if not name.endswith(".md"):
            raise ValueError("maintained reading must be Markdown: " + name)
        record = describe(name)
        text = layout.checked_path(root, name).read_text(encoding="utf-8")
        records.append({**record, "words": len(text.split())})
        if name != spec["entrypoint"] and "max_note_bytes" in spec:
            limit = spec["max_note_bytes"]
            if type(limit) is not int or limit <= 0 or record["bytes"] > limit:
                raise ValueError("continuation note exceeds reading budget; consolidate: " + name)
    if spec.get("require_all_active_notes", False):
        for folder in ("review/tasks", "review/discussions"):
            for note in layout.checked_path(root, folder).rglob("*.md"):
                relative = note.relative_to(root).as_posix()
                if relative not in paths:
                    raise ValueError("unregistered active note would be unread: " + relative)
    totals = {"bytes": sum(r["bytes"] for r in records), "words": sum(r["words"] for r in records)}
    for key in ("bytes", "words"):
        configured = "max_context_" + key
        if configured in spec:
            limit = spec[configured]
            if type(limit) is not int or limit <= 0 or totals[key] > limit:
                raise ValueError("complete continuation exceeds reading budget (" + key + "); summarize owning notes")
    return {"files": records, "total_bytes": totals["bytes"], "total_words": totals["words"],
            "budget_bytes": spec.get("max_context_bytes"), "budget_words": spec.get("max_context_words"),
            "render_budget_bytes": spec.get("max_handoff_bytes"),
            "identity_sha256": digest(json.dumps(records, sort_keys=True).encode()),
            "scope": "FULL_CURRENT_BRIEF_AND_ALL_ACTIVE_NOTES; SOURCE_AND_RECEIPTS_ARE_ON_DEMAND"}


def theory_assignments(root: Path, spec: dict, describe) -> list[dict]:
    """A bounded lottery over explicit extensions of existing claim nodes."""
    items = spec.get("assignments", [])
    # Historical ACCEPTED/PARKED records are source-bound evidence, not live work.
    # Keep their serialization bound finite while enforcing the much smaller OPEN
    # concurrency limit independently below.
    if not isinstance(items, list) or len(items) > 512:
        raise ValueError("at most 512 assignments required")
    # Owner rule: at most 32 problems open at once; close or park one before opening another.
    if sum(1 for i in items if isinstance(i, dict) and isinstance(i.get("schedule"), dict)
           and i["schedule"].get("state") == "OPEN") > 32:
        raise ValueError("at most 32 OPEN assignments required")
    result, seen = [], set()
    for item in items:
        fields = {"id", "topic", "title", "operation", "question", "parents", "inputs",
                  "acceptance", "avoid", "allowed_changes", "max_games", "cpu_seconds"}
        if not isinstance(item, dict) or set(item) - {"weight", "schedule"} != fields:
            raise ValueError("complete closed theory-assignment fields required")
        if type(item.get("weight", 1)) is not int or not 1 <= item.get("weight", 1) <= 8:
            raise ValueError("assignment weight must be an integer 1..8")
        if not isinstance(item["id"], str) or not re.fullmatch(r"[A-Z][A-Z0-9-]{0,31}", item["id"]) or item["id"] in seen:
            raise ValueError("distinct theory-assignment ids required")
        seen.add(item["id"])
        if item["topic"] not in {t["id"] for t in spec["threads"]}:
            raise ValueError("assignment must continue a standing thread")
        if item["operation"] not in {"extend", "connect", "bound", "challenge", "prove", "validate"}:
            raise ValueError("explicit theory-building operation required")
        for key in ("title", "question"):
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError("nonempty assignment " + key)
        for key in ("parents", "inputs", "acceptance", "avoid", "allowed_changes"):
            value = item[key]
            if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x.strip() for x in value) or len(value) != len(set(value)):
                raise ValueError("distinct nonempty assignment " + key)
        if type(item["max_games"]) is not int or not 0 <= item["max_games"] <= 4:
            raise ValueError("reviewer native-game sprawl target required")
        if type(item["cpu_seconds"]) is not int or not 1 <= item["cpu_seconds"] <= 3600:
            raise ValueError("bounded reviewer CPU allowance required")
        if len(json.dumps({k: v for k, v in item.items() if k != "schedule"}).encode()) > 6000:
            raise ValueError("consolidate oversized theory assignment")
        parents = []
        for claim_id in item["parents"]:
            if not re.fullmatch(r"[0-9a-f]{64}", claim_id):
                raise ValueError("assignment needs full parent claim hashes")
            record = describe("project/registries/search-ledger/" + claim_id + ".json")
            claim = json.loads(layout.checked_path(root, record["path"]).read_text())
            if record["sha256"] != claim_id or claim.get("type") != "claim":
                raise ValueError("assignment parent must be an intact claim node")
            parents.append({**record, "id": claim_id, "key": claim["key"], "title": claim["title"]})
        result.append({**item, "parent_claims": parents,
                       "input_files": [describe(name) for name in item["inputs"]]})
    schedule_inspect(root, result)
    for item in result:
        acceptance_paths = set()
        for record in item.get("schedule", {}).get("evidence", []):
            acceptance_paths.add(record["path"])
        # Historical acceptance does not turn every old source pin into a live
        # publication dependency. Current assignment inputs own that closure.
        item["acceptance_files"] = [describe(name) for name in sorted(acceptance_paths)]
    return result


def theorem_index(root: Path, spec: dict) -> str:
    """One generated page of the ledger's claims by layer; the immutable records stay the authority."""
    folder = layout.checked_path(root, "project/registries/search-ledger")
    claims = {}
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        record = json.loads(path.read_text())
        if record.get("type") == "claim":
            claims[record["key"]] = (path.stem, record)
    closed = {}  # stem of a refuted/superseded claim -> "REFUTED by KEY" / "SUPERSEDED by KEY"
    for key, (stem, rec) in claims.items():
        for rel in rec.get("relations", []) or []:
            if rel.get("kind") in ("refutes", "supersedes"):
                closed[rel["target"]] = f"{rel['kind'].upper()[:-1]}D by {key}"

    def line(key):
        stem, rec = claims[key]
        form = FORMS.get(rec.get("classification"), str(rec.get("classification")))
        removes = (rec.get("reuse") or [""])[0]
        status = f" ~~{closed[stem]}~~ —" if stem in closed else ""
        return (f"- **{key}** [{form}]{status} {rec.get('title', '')} — the model need not learn: {removes} "
                f"([record](../project/registries/search-ledger/{stem}.json))")

    lines = ["# What the model need not learn", "",
             "Generated from the immutable ledger by `review_package.py --theorems`; never edit by hand. "
             "Every claim removes something from what a learner must discover: the compiler owns it, or a "
             "theorem fixes it. Form, key, title, then the removal. A record is the authority; "
             "`search_ledger.py show KEY` prints statement, scope, limits and reproduction. A claim marked "
             "REFUTED or SUPERSEDED stays listed because records are immutable; the marking names the claim that closed it.", ""]
    placed = set()
    for layer in spec.get("layers", []):
        lines += [f"## {layer['name']}", "", layer.get("description", ""), ""]
        for key in layer.get("claims", []):
            if key in claims:
                placed.add(key); lines.append(line(key))
            else:
                lines.append(f"- {key}: not in the ledger")
        lines.append("")
    quotients = spec.get("quotients", [])
    if quotients:
        lines += ["## Permutation groups factored out", "",
                  "What permutes, what stays invariant, under which condition; the claim that proves it and the claim that bounds it. "
                  "A group not in this table is still open. The method that has worked on every layer: "
                  "enumerate states, not actions; quotient what does not change the outcome; preserve product mix and timing; compile through the "
                  "executor and turn each failure into a reusable cut; rank by demand-aware receipts, never by one scalar.", "",
                  "| group | invariant | condition | claim | boundary |", "|---|---|---|---|---|"]
        for q in quotients:
            ok = "" if q["claim"] in claims else " (not in the ledger)"
            lines.append(f"| {q['group']} | {q['invariant']} | {q['condition']} | {q['claim']}{ok} | {q['boundary']} |")
        lines.append("")
    unplaced = [k for k in sorted(claims) if k not in placed]
    if unplaced:
        lines += ["## Unplaced", "", "Admitted claims not yet assigned to a layer in project/review.toml.", ""]
        lines += [line(k) for k in unplaced] + [""]
    lines += ["", f"Claims: {len(claims)}; placed: {len(placed)}. Open work: review/FRONTIER.md and the tickets in project/review.toml.", ""]
    return "\n".join(lines)


def _pool(root: Path) -> tuple[bytes, dict, list, str]:
    manifest_bytes = layout.checked_path(root, MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("an extracted review manifest is required for an assignment")
    pool = manifest.get("continuation", {}).get("assignments", [])
    if not pool:
        raise ValueError("this source ZIP has no prepared theory assignments")
    return manifest_bytes, manifest, pool, digest(json.dumps(pool, sort_keys=True, separators=(",", ":")).encode())


def _dispatch_ticket(source_sha: str, dispatch_sha: str, slot: int, task: str) -> str:
    return digest(f"{source_sha}:{dispatch_sha}:{slot}:{task}".encode())[:32]


_DEPENDENCY_RE = re.compile(r"^(?P<task>\S+)(?:\s+outcome=(?P<outcome>PASS|CLOSED_NEGATIVE))?$")


def _dependency_requirement(value: str) -> dict:
    """Parse one backwards-compatible dependency edge.

    Plain ``TASK`` retains the historical meaning: the prerequisite must be ACCEPTED,
    irrespective of whether the accepted result was positive or a measured negative.
    ``TASK outcome=PASS`` (or CLOSED_NEGATIVE) additionally predicates the edge on the
    accepted outcome.  The task id remains the graph identity used by work_schedule.
    """
    if not isinstance(value, str):
        raise ValueError("dependency requirement must be text")
    match = _DEPENDENCY_RE.fullmatch(value)
    if match is None:
        raise ValueError("dependency must be TASK or TASK outcome=PASS|CLOSED_NEGATIVE: " + repr(value))
    return {"task": match.group("task"), "outcome": match.group("outcome"), "requirement": value}


def _accepted_outcome(root: Path, item: dict) -> str | None:
    """Return a source-bound accepted outcome; legacy accepted receipts default PASS."""
    schedule = item.get("schedule", {})
    if schedule.get("state") != "ACCEPTED":
        return None
    outcomes = []
    for record in schedule.get("evidence", []):
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError("receipt identity requires path and sha256")
        payload = layout.checked_path(root, record["path"]).read_bytes()
        if digest(payload) != record["sha256"]:
            raise ValueError("receipt identity mismatch: " + record["path"])
        receipt = review_intake.parse_json(payload)
        outcome = receipt.get("outcome", "PASS")
        if outcome not in ACCEPTANCE_OUTCOMES:
            raise ValueError("accepted receipt outcome must be PASS or CLOSED_NEGATIVE")
        outcomes.append(outcome)
    # work_schedule verifies that ACCEPTED work has at least one valid receipt.  Keep
    # this explicit so contradictory reviewed receipts cannot silently collapse.
    if outcomes and len(set(outcomes)) != 1:
        raise ValueError("accepted task has contradictory outcomes: " + item["id"])
    return outcomes[0] if outcomes else "PASS"


def _base_schedule_assignments(assignments: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    """Strip dependency predicates for the legacy DAG validator, retaining their meaning."""
    normalized, requirements = [], {}
    for item in assignments:
        copy = dict(item)
        if isinstance(item.get("schedule"), dict):
            schedule = dict(item["schedule"])
            parsed = [_dependency_requirement(value) for value in schedule.get("depends_on", [])]
            schedule["depends_on"] = [row["task"] for row in parsed]
            copy["schedule"] = schedule
            requirements[item["id"]] = parsed
        else:
            requirements[item["id"]] = []
        normalized.append(copy)
    return normalized, requirements


def schedule_inspect(root: Path, assignments: list[dict]) -> dict:
    """Outcome-aware facade over work_schedule without creating a second backlog."""
    normalized, requirements = _base_schedule_assignments(assignments)
    view = work_schedule.inspect(root, normalized)
    source = {item["id"]: item for item in assignments}
    rows = {row["id"]: row for row in view["tasks"]}
    outcomes = {task_id: _accepted_outcome(root, source[task_id]) for task_id in rows}
    for task_id, row in rows.items():
        parsed = requirements.get(task_id, [])
        row["dependency_requirements"] = parsed
        row["outcome"] = outcomes[task_id]
        waiting = []
        for req in parsed:
            parent = rows[req["task"]]
            unsatisfied = parent["state"] != "ACCEPTED"
            if not unsatisfied and req["outcome"] is not None:
                unsatisfied = outcomes[req["task"]] != req["outcome"]
            if unsatisfied:
                waiting.append(req["requirement"])
        if row["state"] == "ACCEPTED" and waiting:
            raise ValueError("accepted task has unsatisfied dependency requirement: " + task_id)
        row["waiting_for"] = waiting
        row["status"] = (row["state"] if row["state"] != "OPEN"
                         else "WAITING" if waiting else "READY")
    ordered = view["tasks"]
    ready = [row for row in ordered if row["status"] == "READY"]
    review = [row for row in ready if row["lane"] != "MAINTAINER"]
    draw = work_schedule.draw_weights(review)
    view.update({
        "schema": "farm-review-work-schedule-v2",
        "ready_for_review": [row["id"] for row in review],
        "ready_for_maintainer": [row["id"] for row in ready if row["lane"] == "MAINTAINER"],
        "review_lottery_weight": sum(row["weight"] for row in review),
        "build_lottery_weight": sum(row["weight"] for row in review if row["lane"] == "BUILD"),
        "draw_probability": ({row["id"]: round(weight / sum(draw), 4)
                              for row, weight in zip(review, draw)} if review else {}),
        "outcome_semantics": {
            "accepted_default": "PASS",
            "accepted_outcomes": sorted(ACCEPTANCE_OUTCOMES),
            "dependency_predicate": "TASK outcome=PASS|CLOSED_NEGATIVE",
        },
    })
    return view


def _descendants(view: dict) -> dict[str, set[str]]:
    children = {row["id"]: set() for row in view["tasks"]}
    for row in view["tasks"]:
        for dep in row["depends_on"]:
            children[dep].add(row["id"])
    result = {}
    for key in children:
        seen, stack = set(), list(children[key])
        while stack:
            child = stack.pop()
            if child in seen:
                continue
            seen.add(child); stack.extend(children[child])
        result[key] = seen
    return result


def _maintainer_gate_pressure(view: dict) -> list[dict]:
    rows = {row["id"]: row for row in view["tasks"]}
    memo: dict[str, set[str]] = {}

    def gates(task_id: str) -> set[str]:
        if task_id in memo:
            return memo[task_id]
        result = set()
        for dep in rows[task_id]["depends_on"]:
            parent = rows[dep]
            if parent["lane"] == "MAINTAINER" and parent["state"] != "ACCEPTED":
                result.add(dep)
            if parent["state"] != "ACCEPTED":
                result.update(gates(dep))
        memo[task_id] = result
        return result

    pressure = []
    for row in view["tasks"]:
        if row["state"] != "OPEN":
            continue
        blocked = sorted(gates(row["id"]))
        if blocked:
            pressure.append({"task": row["id"], "maintainer_gates": blocked})
    return pressure


def _dispatch_numbers(spec: dict) -> list[int]:
    """Validate issuance metadata without requiring the old slots to still be READY."""
    if (not isinstance(spec, dict) or type(spec.get("version")) is not int or spec["version"] != 1
            or set(spec) - {"version", "target_parallelism", "next_slot", "slot", "hold"}):
        raise ValueError("dispatch needs version = 1 with only [[slot]] and [[hold]] tables, "
                         "target_parallelism and next_slot")
    target = spec.get("target_parallelism", TARGET_PARALLELISM)
    if type(target) is not int or target <= 0:
        raise ValueError("target_parallelism must be a positive integer")
    slots, holds = spec.get("slot", []), spec.get("hold", [])
    if not isinstance(slots, list) or not isinstance(holds, list):
        raise ValueError("dispatch slots and holds must be arrays of tables")
    numbers, tasks = [], set()
    for row in slots:
        if not isinstance(row, dict) or set(row) != {"number", "task"}:
            raise ValueError("dispatch slot needs exactly number and task")
        number, task = row["number"], row["task"]
        if "next_slot" not in spec:
            if type(number) is not int or number != len(numbers) + 1:
                raise ValueError("dispatch slot numbers must run 1, 2, 3, ... in order; "
                                 "sparse slots require explicit next_slot migration")
        elif type(number) is not int or number <= 0 or (numbers and number <= numbers[-1]):
            raise ValueError("dispatch slot numbers must be positive, distinct and increasing")
        if not isinstance(task, str) or not task:
            raise ValueError("dispatch slot needs a task id")
        if task in tasks:
            raise ValueError("task dispatched twice: " + task)
        numbers.append(number); tasks.add(task)
    for row in holds:
        if not isinstance(row, dict) or set(row) != {"task", "reason"}:
            raise ValueError("dispatch hold needs exactly task and reason")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError("dispatch hold needs a reason")
        if not isinstance(row["task"], str) or not row["task"]:
            raise ValueError("dispatch hold needs a task id")
        if row["task"] in tasks:
            raise ValueError("task both dispatched and held, or held twice: " + row["task"])
        tasks.add(row["task"])
    if "next_slot" in spec:
        counter = spec["next_slot"]
        if type(counter) is not int or counter <= max(numbers, default=0):
            raise ValueError("next_slot must be a positive integer strictly greater than every live slot")
    return numbers


def _dispatch_with_metrics(view: dict, spec: dict) -> dict:
    numbers = _dispatch_numbers(spec)
    target = spec.get("target_parallelism", TARGET_PARALLELISM)
    # Keep schedule validation in its existing owner. Ordinals exist ONLY inside this
    # adapter; plans, manifests, prompts, tickets and returns always use literal IDs.
    legacy = {key: value for key, value in spec.items() if key not in {"target_parallelism", "next_slot"}}
    legacy["slot"] = [{**row, "number": i} for i, row in enumerate(spec.get("slot", []), 1)]
    try:
        plan = work_schedule.dispatch(view, legacy)
    except ValueError as error:
        message = re.sub(r"slot (\d+) names", lambda m: f"slot {numbers[int(m[1]) - 1]} names", str(error))
        raise ValueError(message) from error
    descendants = _descendants(view)
    slots = [{**slot, "number": number, "descendant_count": len(descendants[slot["task"]])}
             for slot, number in zip(plan["slots"], numbers)]
    blocked = _maintainer_gate_pressure(view)
    return {**plan, **({"next_slot": spec["next_slot"]} if "next_slot" in spec else {}), "slots": slots,
            "target_parallelism": target,
            "underfill_by": max(0, target - len(slots)),
            "underfilled": len(slots) < target,
            "open_blocked_by_maintainer_gates": blocked,
            "open_blocked_by_maintainer_gate_count": len(blocked)}


def _dispatch_history(registry: Path, *, required: bool = False) -> dict:
    """Read the EXISTING immutable-baseline registry; never create another task index."""
    if not registry.exists() and not required:
        return {"floor": 1, "sources": {}, "tickets": [], "snapshot": {}}
    home, state = review_intake._load_registry(registry)
    if required and not state["sources"]:
        raise ValueError("dispatch registry must contain retained issued manifests before allocation")
    state_path = home / review_intake.DISPATCH_REGISTRY_FILE
    state_bytes = state_path.read_bytes()
    if review_intake.parse_json(state_bytes) != state:
        raise ValueError("dispatch registry changed while loading; retry from current state")
    snapshot = {state_path: state_bytes}
    sources, floor = {}, 1
    for source_sha, source in state["sources"].items():
        baseline = review_intake._registry_baseline(home, state, source_sha)
        path = Path(baseline["file"])
        raw, manifest, dispatch_sha, slots = review_intake._manifest_dispatch(path)
        if (digest(raw) != source_sha or source.get("dispatch_sha256") != dispatch_sha
                or source.get("slots") != slots or source.get("source_manifest_sha256") != source_sha):
            raise ValueError("retained dispatch registry source identity mismatch")
        snapshot[path] = raw
        counter = manifest["continuation"]["dispatch"].get("next_slot")
        maximum = max((row["number"] for row in slots), default=0)
        if counter is not None and (type(counter) is not int or counter <= maximum):
            raise ValueError("retained next_slot must exceed every retained issued slot")
        floor = max(floor, maximum + 1, counter or 1)
        sources[source_sha] = {"slots": {row["number"]: row["task"] for row in slots},
                               "next_slot": counter, "dispatch_sha256": dispatch_sha,
                               "issued_at": review_intake._parse_time(source["registered_at"])}
        for row in slots:
            exact = _dispatch_ticket(source_sha, dispatch_sha, row["number"], row["task"])
            if exact not in state["tickets"]:
                raise ValueError("retained dispatch source lacks its exact issued ticket")
    tickets = []
    for key, row in state["tickets"].items():
        source = sources.get(row.get("source_manifest_sha256"))
        number, task = row.get("slot"), row.get("task_id")
        if (source is None or type(number) is not int or source["slots"].get(number) != task
                or row.get("dispatch_sha256") != source["dispatch_sha256"]
                or key != row.get("ticket")
                or key != _dispatch_ticket(row["source_manifest_sha256"], row["dispatch_sha256"], number, task)):
            raise ValueError("retained dispatch ticket identity mismatch")
        review_intake._parse_time(row["issued_at"])
        review_intake._parse_time(row["lost_after"])
        if row.get("returned_at") is not None:
            review_intake._parse_time(row["returned_at"])
        tickets.append(row)
    return {"floor": floor, "sources": sources, "tickets": tickets, "snapshot": snapshot}


def _check_dispatch_history(spec: dict, history: dict) -> None:
    if "next_slot" not in spec:
        return  # Already-issued legacy ZIPs keep their source-local alias semantics.
    if spec["next_slot"] < history["floor"]:
        raise ValueError(f"next_slot must be at least {history['floor']} to exceed retained issued slots/counters")
    modern = [source for source in history["sources"].values() if source["next_slot"] is not None]
    if not modern:
        return
    latest = max(modern, key=lambda row: (row["next_slot"], row["issued_at"]))
    for slot in spec.get("slot", []):
        number, task = slot["number"], slot["task"]
        if any(number in source["slots"] and source["slots"][number] != task for source in modern):
            raise ValueError(f"reused retained slot {number}: cannot reassign it to {task}")
        if number < latest["next_slot"] and latest["slots"].get(number) != task:
            raise ValueError(f"reused retired slot {number}: allocate from next_slot instead")


def rebuild_dispatch(root: Path, *, registry: Path | None = None, now: str | None = None,
                     migrate_next_slot: bool = False) -> dict:
    """Maintainer-only reconciliation; task state, registry history and tickets are read-only.

    Explicit migration attests the retained registry is complete. Ready unheld tasks
    are issued by descending weight, then dependency wave, then task ID. A recorded
    return ends that numeric issuance even for standing OPEN work (NEXT-TICKETS).
    """
    import fcntl
    from datetime import datetime, timezone
    instant = review_intake._parse_time(now or datetime.now(timezone.utc).isoformat())
    path = layout.checked_path(root, DISPATCH)
    registry = Path(registry) if registry is not None else root / DISPATCH_REGISTRY_ROOT
    # Lock the directory, not the replaced file's inode. A second rebuild re-reads
    # next_slot after acquiring this lock; no persistent second allocation file.
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        fcntl.flock(directory, fcntl.LOCK_EX)
        before = path.read_bytes()
        spec = tomllib.loads(before.decode())
        numbers = _dispatch_numbers(spec)
        history = _dispatch_history(registry, required=True)
        if "next_slot" not in spec:
            if not migrate_next_slot:
                raise ValueError("legacy dispatch allocation requires explicit --migrate-next-slot")
            if not any(source["dispatch_sha256"] == digest(before) for source in history["sources"].values()):
                raise ValueError("migration requires the current legacy dispatch in the retained registry")
            counter = max(history["floor"], max(numbers, default=0) + 1)
        else:
            _check_dispatch_history(spec, history)
            counter = spec["next_slot"]
        context_path = layout.checked_path(root, CONTEXT_SPEC)
        context_bytes = context_path.read_bytes()
        assignments = tomllib.loads(context_bytes.decode()).get("assignments", [])
        view = schedule_inspect(root, assignments)
        rows = {row["id"]: row for row in view["tasks"]}
        preserved, retired = [], []
        for slot in spec.get("slot", []):
            row = rows.get(slot["task"])
            if row is None:
                raise ValueError("dispatch names an unknown task: " + slot["task"])
            reason = None
            if row["status"] != "READY" or row["lane"] == "MAINTAINER":
                reason = row["status"] if row["lane"] != "MAINTAINER" else "MAINTAINER"
            else:
                records = [r for r in history["tickets"] if (r["slot"], r["task_id"]) == (slot["number"], slot["task"])]
                if records:
                    latest = max(records, key=lambda r: (review_intake._parse_time(r["issued_at"]), r["ticket"]))
                    # Legacy waves could reuse the SAME (number, task). Only their
                    # latest exact issuance is current. Modern sources sharing this
                    # pair are continuations, including late returns from older ZIPs.
                    modern = [r for r in records
                              if history["sources"][r["source_manifest_sha256"]]["next_slot"] is not None]
                    legacy = [r for r in records if r not in modern]
                    eligible = [latest]
                    if "next_slot" in spec:
                        latest_legacy = max(legacy, key=lambda r: (
                            review_intake._parse_time(r["issued_at"]), r["ticket"]), default=None)
                        eligible = modern + ([latest_legacy] if latest_legacy is not None else [])
                    returned = any(r.get("returned_at") is not None for r in eligible)
                    status = review_intake._ticket_status(latest, instant)
                    if returned:
                        reason = "RETURNED"
                    elif status in {"LOST", "SUPERSEDED"}:
                        reason = status
            if reason is None:
                preserved.append(slot)
            else:
                retired.append({**slot, "reason": reason})
        holds = []
        for hold in spec.get("hold", []):
            row = rows.get(hold["task"])
            if row is None:
                raise ValueError("dispatch hold names an unknown task: " + hold["task"])
            if row["status"] in {"READY", "WAITING"} and row["lane"] != "MAINTAINER":
                holds.append(hold)
        claimed = {r["task"] for r in preserved + holds}
        ready = sorted((r for r in view["tasks"] if r["id"] in view["ready_for_review"] and r["id"] not in claimed),
                       key=lambda r: (-r["weight"], r["wave"], r["id"]))
        allocated = [{"number": counter + i, "task": row["id"]} for i, row in enumerate(ready)]
        counter += len(allocated)
        candidate = {**spec, "next_slot": counter, "slot": preserved + allocated}
        if holds or "hold" in spec:
            candidate["hold"] = holds
        plan = _dispatch_with_metrics(view, candidate)
        _check_dispatch_history(candidate, history)
        lint = lint_dispatch(root, assignments, plan)
        if lint["status"] != "PASS":
            raise ValueError("dispatch rebuild lint failed: " + json.dumps(lint["errors"], sort_keys=True))
        changed = candidate != spec
        payload = before
        if changed:
            lines = ["version = 1", f"next_slot = {counter}"]
            if "target_parallelism" in candidate:
                lines.append(f"target_parallelism = {candidate['target_parallelism']}")
            lines += ["", "# Monotonic issuance IDs. Reconcile with review_package.py --rebuild-dispatch.",
                      "# Task state and priority remain in project/review.toml; never recycle a number."]
            def quote(value: str) -> str:
                # TOML needs Unicode scalars, not JSON surrogate pairs; DEL must stay escaped.
                return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
            for slot in candidate["slot"]:
                lines += ["", "[[slot]]", f"number = {slot['number']}", "task = " + quote(slot["task"])]
            for hold in holds:
                lines += ["", "[[hold]]", "task = " + quote(hold["task"]), "reason = " + quote(hold["reason"])]
            payload = ("\n".join(lines) + "\n").encode()
        snapshot = {path: before, context_path: context_bytes, **history["snapshot"]}
        if any(p.read_bytes() != data for p, data in snapshot.items()):
            raise ValueError("dispatch inputs changed during rebuild; retry from current state")
        if changed:
            review_intake._atomic_write(path, payload)
            os.fsync(directory)
    finally:
        os.close(directory)
    return {"schema": "farm-review-dispatch-rebuild-v1", "changed": changed,
            "migrated": "next_slot" not in spec, "next_slot": counter,
            "preserved": preserved, "retired": retired, "allocated": allocated, "holds": holds,
            "before_sha256": digest(before), "after_sha256": digest(payload),
            "registry": str(registry), "as_of": review_intake._time_text(instant),
            "priority": "descending weight, then ascending dependency wave, then task ID",
            "task_state_changes": 0, "registry_changes": 0}


def render_dispatch_summary(plan: dict) -> str:
    lines = [f"Slot {s['number']}: {s['task']} [{s['track']} {s['lane']}] descendants={s.get('descendant_count', 0)} {s['title']}"
             for s in plan["slots"]]
    lines += [f"HOLD: {h['task']} ({h['status']}): {h['reason']}" for h in plan["holds"]]
    lines += [f"PARALLELISM: {len(plan['slots'])}/{plan['target_parallelism']} dispatched; underfill={plan['underfill_by']}",
              f"MAINTAINER-GATE PRESSURE: {plan['open_blocked_by_maintainer_gate_count']} OPEN task(s) blocked"]
    if plan["open_blocked_by_maintainer_gates"]:
        lines += [f"BLOCKED: {row['task']} <- {', '.join(row['maintainer_gates'])}"
                  for row in plan["open_blocked_by_maintainer_gates"]]
    return "\n".join(lines)


def saved_assignment(root: Path) -> dict:
    """The one saved ticket of this copy, drawn or dispatched; source, pool or dispatch drift refuses."""
    from farm_neat.evaluation.common import read
    manifest_bytes, _, pool, pool_sha = _pool(root)
    target = layout.checked_path(root, ASSIGNMENT)
    if not target.exists():
        raise ValueError("no saved assignment; run --assign-slot N once before scoped navigation")
    value = read(target)
    if (value.get("schema") not in (LOTTERY_TICKET, DISPATCH_TICKET)
            or value["source_manifest_sha256"] != digest(manifest_bytes) or value["pool_sha256"] != pool_sha):
        raise ValueError("saved assignment belongs to a different source/pool; preserve its evidence")
    if value["schema"] == LOTTERY_TICKET:
        number = value["random_number"]
        if type(number) is not int or not 1 <= number <= len(pool) or value["assignment"] != pool[number - 1]:
            raise ValueError("saved theory assignment is inconsistent")
        return value
    dispatch_bytes = layout.checked_path(root, DISPATCH).read_bytes()
    if value["dispatch_sha256"] != digest(dispatch_bytes):
        raise ValueError("saved assignment belongs to a different dispatch; preserve its evidence")
    try:
        selected, chosen = _selected_dispatch_assignment(
            pool, tomllib.loads(dispatch_bytes.decode()), slot=value["slot"])
    except ValueError as error:
        raise ValueError("saved dispatch assignment is inconsistent: " + str(error)) from error
    ticket = _dispatch_ticket(value["source_manifest_sha256"], value["dispatch_sha256"], value["slot"], value["task_id"])
    if (selected["task"] != value["task_id"] or value["assignment"].get("id") != value["task_id"]
            or chosen != value["assignment"]
            or value["ticket"] != ticket or value["return_name"] != f"FARM-RETURN-{value['task_id']}-{ticket[:8]}.zip"):
        raise ValueError("saved dispatch assignment is inconsistent")
    return value


def _save_ticket(root: Path, value: dict) -> dict:
    """Atomic hard-link publication: concurrent calls in one copy keep one complete ticket."""
    target = layout.checked_path(root, ASSIGNMENT)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".assignment-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())
            stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
    finally:
        os.unlink(temporary)
    return saved_assignment(root)


def dispatch_plan(root: Path, pool: list[dict]) -> dict:
    """Validated slots and holds joined to the schedule of this pool."""
    path = layout.checked_path(root, DISPATCH)
    if not path.is_file():
        raise ValueError("missing dispatch: " + DISPATCH)
    spec = tomllib.loads(path.read_text())
    plan = _dispatch_with_metrics(schedule_inspect(root, pool), spec)
    if "next_slot" in spec:
        _check_dispatch_history(spec, _dispatch_history(root / DISPATCH_REGISTRY_ROOT))
    return plan


def _selected_dispatch_assignment(pool: list[dict], spec: dict, *, slot=None, task=None) -> tuple[dict, dict]:
    """Resolve only the requested reviewer slot; unrelated dispatch/pool rows are inert."""
    if (slot is None) == (task is None):
        raise ValueError("give exactly one dispatch slot or task id")
    rows = spec.get("slot", []) if isinstance(spec, dict) else []
    if not isinstance(rows, list):
        raise ValueError("dispatch slots must be an array of tables")
    if slot is not None:
        if type(slot) is not int or slot <= 0:
            raise ValueError(f"unknown slot {slot}; dispatch slot must be a positive integer")
        matches = [row for row in rows if isinstance(row, dict) and row.get("number") == slot]
        requested = f"slot {slot}"
    else:
        if not isinstance(task, str) or not task:
            raise ValueError("dispatch task id must be nonempty")
        matches = [row for row in rows if isinstance(row, dict) and row.get("task") == task]
        requested = f"task {task}"
    if not matches:
        if task is not None:
            held = [r for r in spec.get("hold", []) if isinstance(r, dict) and r.get("task") == task]
            if held:
                raise ValueError("task is on hold: " + str(held[0].get("reason", task)))
            selected = next((r for r in pool if isinstance(r, dict) and r.get("id") == task), None)
            if selected:
                schedule = selected.get("schedule", {})
                if schedule.get("lane") == "MAINTAINER":
                    raise ValueError("task is maintainer-only: " + task)
                raise ValueError(f"task {task} is not dispatched ({schedule.get('state', 'OPEN')})")
        listing = ", ".join(f"{r.get('number')}={r.get('task')}" for r in rows if isinstance(r, dict))
        raise ValueError("unknown " + requested + "; this ZIP dispatches " + listing)
    if len(matches) != 1:
        raise ValueError("ambiguous " + requested + " in this ZIP dispatch")
    row = matches[0]
    number, task_id = row.get("number"), row.get("task")
    if type(number) is not int or number <= 0 or not isinstance(task_id, str) or not task_id:
        raise ValueError("selected dispatch row needs a positive number and task id")
    chosen = [item for item in pool if isinstance(item, dict) and item.get("id") == task_id]
    if len(chosen) != 1:
        raise ValueError("selected dispatch task is absent or ambiguous in the source-bound pool: " + task_id)
    schedule = chosen[0].get("schedule", {})
    if schedule.get("state") != "OPEN":
        raise ValueError(f"selected task {task_id} is {schedule.get('state', 'not OPEN')}")
    if schedule.get("lane") == "MAINTAINER":
        raise ValueError("selected task is maintainer-only: " + task_id)
    return {"number": number, "task": task_id}, chosen[0]


def _verify_selected_assignment_inputs(root: Path, chosen: dict) -> None:
    for record in [*chosen.get("parent_claims", []), *chosen.get("input_files", [])]:
        name = record.get("path")
        target = layout.checked_path(root, name)
        if not target.is_file():
            raise ValueError("selected assignment input is missing: " + str(name))
        payload = target.read_bytes()
        if digest(payload) != record.get("sha256") or len(payload) != record.get("bytes", len(payload)):
            raise ValueError("selected assignment input changed from source ZIP: " + str(name))


def lint_dispatch(root: Path, assignments: list[dict], plan: dict) -> dict:
    """Run the existing structural lint after outcome-aware schedule validation."""
    schedule_inspect(root, assignments)
    normalized, _ = _base_schedule_assignments(assignments)
    return work_schedule.lint_dispatch(root, normalized, plan)


def assign(root: Path, *, slot: int | None = None, task: str | None = None) -> dict:
    """Save the supplied dispatch slot (or its task id) as this copy's source-bound ticket.

    No randomness: the same ZIP and slot give the same ticket and return name. Repeating
    the selection returns the saved ticket; another task refuses. There is no lock across
    copies; the maintainer hands out distinct numbers.
    """
    from farm_neat.evaluation.common import bound
    manifest_bytes, manifest, pool, pool_sha = _pool(root)
    dispatch_bytes = layout.checked_path(root, DISPATCH).read_bytes()
    if manifest["members"].get(DISPATCH, {}).get("sha256") != digest(dispatch_bytes):
        raise ValueError("dispatch differs from the one this ZIP shipped; use a fresh extraction")
    dispatch_identity = manifest.get("continuation", {}).get("dispatch", {})
    if dispatch_identity.get("sha256") != digest(dispatch_bytes):
        raise ValueError("source manifest dispatch identity differs from the shipped dispatch")
    try:
        chosen_slot, chosen = _selected_dispatch_assignment(
            pool, tomllib.loads(dispatch_bytes.decode()), slot=slot, task=task)
    except ValueError as error:
        raise ValueError(f"{error}; source_manifest_sha256={digest(manifest_bytes)}") from error
    if layout.checked_path(root, ASSIGNMENT).exists():
        saved = saved_assignment(root)
        if saved["assignment"]["id"] != chosen_slot["task"]:
            held = f" (slot {saved['slot']})" if "slot" in saved else ""
            raise ValueError(f"this copy is already assigned {saved['assignment']['id']}{held}; refusing "
                             f"{chosen_slot['task']} (slot {chosen_slot['number']}). Use a fresh extraction.")
        return saved
    _verify_selected_assignment_inputs(root, chosen)
    source_sha, dispatch_sha = digest(manifest_bytes), digest(dispatch_bytes)
    ticket = _dispatch_ticket(source_sha, dispatch_sha, chosen_slot["number"], chosen["id"])
    return _save_ticket(root, bound({
        "schema": DISPATCH_TICKET, "ticket": ticket,
        "selection_method": "explicit_slot" if slot is not None else "explicit_task",
        "slot": chosen_slot["number"], "task_id": chosen["id"],
        "source_manifest_sha256": source_sha, "pool_sha256": pool_sha, "dispatch_sha256": dispatch_sha,
        "assignment": chosen,
        "return_name": f"FARM-RETURN-{chosen['id']}-{ticket[:8]}.zip",
        "protocol": "Do the selected task within its write scope. Read relevant inputs, implement the deliverable and run affected checks. Return changes, results and any remaining limitation. No global review or full suite is required. Keep this ticket and source identity; do not switch tasks.",
        "retention": {"purpose": "Persist one source-bound dispatched assignment through reviewer interruption",
                      "dependencies": [MANIFEST, CONTEXT_SPEC, DISPATCH],
                      "remove_when": "Maintainer has integrated this review return and retained its ticket and findings."}}))


def _read_continuation(root: Path, task_id: str) -> tuple[dict | None, str, str | None]:
    """Read one task-keyed PARTIAL.  Ticket/slot changes are intentionally irrelevant."""
    relative = _continuation_path(task_id)
    target = layout.checked_path(root, relative)
    if not target.exists():
        return None, relative, None
    if target.is_symlink() or not target.is_file():
        raise ValueError("CONTINUE.json must be a regular file")
    value = review_intake.parse_json(target.read_bytes())
    legacy_required = {
        "schema", "status", "task_id", "source_ticket", "partial_return_name", "final_return_name",
        "generation", "done", "exact_state", "next_command", "native_start_cap",
        "native_starts_charged", "do_not_redo", "workspace", "workspace_sha256", "retirement_rule",
    }
    v2_required = legacy_required | {"remaining", "blockers", "checkpoint"}
    current_required = (legacy_required - {"native_start_cap", "native_starts_charged"}
                        | {"soft_target_games", "starts_recorded", "remaining", "blockers", "checkpoint"})
    schema = value.get("schema")
    if schema == "farm-review-continuation-v1":
        expected = legacy_required
    elif schema == "farm-review-continuation-v2":
        expected = v2_required
    elif schema == CONTINUATION_SCHEMA:
        expected = current_required
    else:
        expected = None
    if expected is None or set(value) != expected:
        raise ValueError("CONTINUE.json has incomplete or unknown fields")
    if value["status"] != "PARTIAL" or value["task_id"] != task_id:
        raise ValueError("CONTINUE.json does not describe this task's PARTIAL state")
    if schema == "farm-review-continuation-v1":
        value = {**value,
                 "remaining": ["Legacy PARTIAL: continue from exact_state and next_command."],
                 "blockers": [], "checkpoint": "LEGACY_PARTIAL"}
    if schema in LEGACY_CONTINUATION_SCHEMAS:
        value = {key: item for key, item in value.items()
                 if key not in {"native_start_cap", "native_starts_charged"}} | {
                     "soft_target_games": value["native_start_cap"],
                     "starts_recorded": value["native_starts_charged"],
                 }
    if not isinstance(value["source_ticket"], str) or not re.fullmatch(r"[0-9a-f]{32}", value["source_ticket"]):
        raise ValueError("CONTINUE.json source_ticket must be a dispatch/lottery ticket id")
    expected_partial = f"FARM-PARTIAL-{task_id}-{value['source_ticket'][:8]}.zip"
    if value["partial_return_name"] != expected_partial or not isinstance(value["final_return_name"], str):
        raise ValueError("CONTINUE.json return names are inconsistent")
    if type(value["generation"]) is not int or value["generation"] < 1:
        raise ValueError("CONTINUE.json generation must be positive")
    if (not isinstance(value["done"], list) or not value["done"]
            or any(not isinstance(x, str) or not x.strip() for x in value["done"])):
        raise ValueError("CONTINUE.json done must be a nonempty string list")
    if not isinstance(value["exact_state"], str) or not value["exact_state"].strip():
        raise ValueError("CONTINUE.json exact_state must be nonempty text")
    if not isinstance(value["next_command"], str) or not value["next_command"].strip():
        raise ValueError("CONTINUE.json next_command must be nonempty text")
    if (type(value["soft_target_games"]) is not int or value["soft_target_games"] < 0
            or type(value["starts_recorded"]) is not int or value["starts_recorded"] < 0):
        raise ValueError("CONTINUE.json native start accounting must be nonnegative integers")
    if (not isinstance(value["do_not_redo"], list)
            or any(not isinstance(x, str) or not x.strip() for x in value["do_not_redo"])):
        raise ValueError("CONTINUE.json do_not_redo must be a string list")
    for field in ("remaining", "blockers"):
        if (not isinstance(value[field], list)
                or any(not isinstance(x, str) or not x.strip() for x in value[field])):
            raise ValueError(f"CONTINUE.json {field} must be a string list")
    if not isinstance(value["checkpoint"], str) or not value["checkpoint"].strip():
        raise ValueError("CONTINUE.json checkpoint must be nonempty text")
    if not isinstance(value["workspace"], list):
        raise ValueError("CONTINUE.json workspace must be a list")
    if value["workspace_sha256"] != digest(review_intake.canonical(value["workspace"])):
        raise ValueError("CONTINUE.json workspace hash mismatch")
    if not isinstance(value["retirement_rule"], str) or not value["retirement_rule"].strip():
        raise ValueError("CONTINUE.json retirement_rule must be nonempty")
    for row in value["workspace"]:
        if not isinstance(row, dict) or "path" not in row or row.get("status") not in {"FILE", "DIRECTORY", "MISSING"}:
            raise ValueError("malformed CONTINUE.json workspace record")
        layout.checked_path(root, row["path"])
        if row["status"] == "FILE":
            if (type(row.get("bytes")) is not int or row["bytes"] < 0
                    or not isinstance(row.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
                raise ValueError("malformed CONTINUE.json file record")
            saved_copy = row.get("saved_copy")
            if saved_copy is not None:
                copy = layout.checked_path(root, saved_copy)
                if (not copy.is_file() or copy.stat().st_size != row["bytes"]
                        or digest(copy.read_bytes()) != row["sha256"]):
                    raise ValueError("saved continuation workspace copy drifted: " + row["path"])
    return value, relative, digest(target.read_bytes())


def _native_start_receipts(root: Path, task_id: str) -> list[dict]:
    folder = layout.checked_path(root, f"{CONTINUATION_ROOT}/{_continuation_task(task_id)}")
    if not folder.is_dir():
        return []
    rows = []
    for path in sorted(folder.glob("START-*.json")):
        if path.is_symlink() or not path.is_file() or not re.fullmatch(r"START-\d{6}\.json", path.name):
            raise ValueError("malformed native start receipt path")
        row = review_intake.parse_json(path.read_bytes())
        if not isinstance(row, dict):
            raise ValueError("malformed native start receipt: " + path.name)
        if row.get("schema") == LEGACY_NATIVE_START_SCHEMA:
            expected = {"schema", "task_id", "source_ticket", "ordinal", "native_start_cap"}
            target = row.get("native_start_cap")
        elif row.get("schema") == NATIVE_START_SCHEMA:
            expected = {"schema", "task_id", "source_ticket", "ordinal", "soft_target_games"}
            target = row.get("soft_target_games")
        else:
            raise ValueError("malformed native start receipt: " + path.name)
        if (set(row) != expected or row["task_id"] != task_id
                or not isinstance(row["source_ticket"], str)
                or type(row["ordinal"]) is not int or row["ordinal"] < 1
                or type(target) is not int or target < 0
                or path.name != f"START-{row['ordinal']:06d}.json"):
            raise ValueError("malformed native start receipt: " + path.name)
        rows.append({**row, "soft_target_games": target})
    if len({row["ordinal"] for row in rows}) != len(rows):
        raise ValueError("duplicate native start ordinals")
    # A PARTIAL may carry a legacy accounting high-water mark. Without that
    # control file, the receipts themselves must cover every historical start.
    if not (folder / "CONTINUE.json").exists() and any(
            row["ordinal"] != ordinal for ordinal, row in enumerate(rows, 1)):
        raise ValueError("native start receipt ordinals must be contiguous from 1")
    return rows


def _native_start_budget(root: Path, ticket: dict, continuation: dict | None = None) -> dict:
    task_id = ticket["assignment"]["id"]
    current_target = ticket["assignment"]["max_games"]
    if type(current_target) is not int or current_target < 0:
        raise ValueError("assignment max_games must be a nonnegative integer")
    receipts = _native_start_receipts(root, task_id)
    starts = max([0, *[r["ordinal"] for r in receipts]])
    if continuation is not None:
        starts = max(starts, continuation["starts_recorded"])
    return {"soft_target_games": current_target, "starts_recorded": starts,
            "remaining_to_target": max(0, current_target - starts),
            "above_target_by": max(0, starts - current_target), "may_continue": True,
            "meaning": "soft_target_games is the current ticket's anti-sprawl target, not a cap. Historical starts remain recorded across reissues; a newer ticket may raise the soft target without erasing them."}


def continuation_state(root: Path, ticket: dict | None = None) -> dict:
    """Describe resumable state before IMPLEMENT without mutating it."""
    ticket = ticket or saved_assignment(root)
    task_id = ticket["assignment"]["id"]
    value, relative, sha = _read_continuation(root, task_id)
    budget = _native_start_budget(root, ticket, value)
    partial_name = f"FARM-PARTIAL-{task_id}-{ticket['ticket'][:8]}.zip"
    if value is None:
        return {"status": "NO_SAVED_PARTIAL", "path": relative, "sha256": None,
                "partial_return_name": partial_name, "native_start_budget": budget,
                "retirement_rule": "No saved workspace exists for this task."}
    checks = []
    for row in value["workspace"]:
        target = layout.checked_path(root, row["path"])
        if row["status"] == "FILE":
            current = target.is_file() and target.stat().st_size == row["bytes"] and digest(target.read_bytes()) == row["sha256"]
            checks.append({"path": row["path"], "state": "CURRENT" if current else
                           ("RESTORABLE" if row.get("saved_copy") else "DRIFT_OR_MISSING")})
        elif row["status"] == "DIRECTORY":
            checks.append({"path": row["path"], "state": "CURRENT" if target.is_dir() else "DRIFT_OR_MISSING"})
        else:
            checks.append({"path": row["path"], "state": "CURRENT" if not target.exists() else "DRIFT_OR_MISSING"})
    return {"status": "SAVED_PARTIAL", "path": relative, "sha256": sha,
            "source_ticket": value["source_ticket"], "source_partial_return": value["partial_return_name"],
            "current_partial_return": partial_name, "current_final_return": ticket["return_name"],
            "generation": value["generation"], "done": value["done"], "exact_state": value["exact_state"],
            "next_command": value["next_command"], "do_not_redo": value["do_not_redo"],
            "remaining": value["remaining"], "blockers": value["blockers"], "checkpoint": value["checkpoint"],
            "workspace_sha256": value["workspace_sha256"], "workspace_check": checks,
            "native_start_budget": budget, "retirement_rule": value["retirement_rule"],
            "resume_command": "python3 -B src/farm_neat/tooling/review_package.py --resume-continuation"}


def _atomic_json(target: Path, value: dict, *, replace: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = review_intake.canonical(value) + b"\n"
    if replace:
        fd, temporary = tempfile.mkstemp(prefix=".continue-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    else:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def charge_native_start(root: Path) -> dict:
    """Charge one native start *before* launch; interrupted starts remain spent."""
    ticket = saved_assignment(root)
    task_id = _ticket_task_id(ticket)
    continuation, _, _ = _read_continuation(root, task_id)
    budget = _native_start_budget(root, ticket, continuation)
    ordinal = budget["starts_recorded"] + 1
    relative = f"{CONTINUATION_ROOT}/{task_id}/START-{ordinal:06d}.json"
    receipt = {"schema": NATIVE_START_SCHEMA, "task_id": task_id,
               "source_ticket": ticket["ticket"], "ordinal": ordinal,
               "soft_target_games": budget["soft_target_games"]}
    _atomic_json(layout.checked_path(root, relative), receipt, replace=False)
    over = max(0, ordinal - budget["soft_target_games"])
    return {"status": "NATIVE_START_OVER_TARGET" if over else "NATIVE_START_CHARGED",
            "path": relative, "ordinal": ordinal, "starts_recorded": ordinal,
            "remaining_to_target": max(0, budget["soft_target_games"] - ordinal),
            "soft_target_games": budget["soft_target_games"],
            "above_target_by": over, "may_continue": True,
            "checkpoint_recommended": over > 0,
            "meaning": "The start is recorded even beyond the anti-sprawl target. Over-target work is allowed only inside the same assigned edge; checkpoint and explain why another start is needed instead of broadening the search."}


def _inside_write_scope(path: str, writes: list[str]) -> bool:
    return any(path == base.rstrip("/") or path.startswith(base.rstrip("/") + "/") for base in writes)


def _snapshot_workspace(root: Path, ticket: dict) -> tuple[list[dict], dict[str, bytes]]:
    """Validate and capture the whole write scope without changing any saved bytes."""
    writes = ticket["assignment"].get("schedule", {}).get("writes", [])
    manifest = json.loads(layout.checked_path(root, MANIFEST).read_text())
    preimages = manifest.get("members", {})
    task_id = _ticket_task_id(ticket)
    records, copies = [], {}

    def add_file(path: Path):
        relative = path.relative_to(root).as_posix()
        carry = relative.startswith("work/")
        if carry and path.suffix not in CONTINUATION_SUFFIXES:
            raise ValueError("cannot carry unsupported work artifact in PARTIAL: " + relative)
        payload = path.read_bytes()
        row = {"path": relative, "status": "FILE", "bytes": len(payload), "sha256": digest(payload)}
        if relative in preimages:
            row["preimage_sha256"] = preimages[relative]["sha256"]
        if carry:
            saved = f"{CONTINUATION_ROOT}/{task_id}/workspace/{relative}"
            layout.checked_path(root, saved)
            row["saved_copy"] = saved
            copies[saved] = payload
            if len(copies) > CONTINUATION_MEMBER_LIMIT or sum(map(len, copies.values())) > CONTINUATION_LIMIT:
                raise ValueError("continuation workspace exceeds bounded review handoff budget")
        records.append(row)

    for name in writes:
        target = layout.checked_path(root, name)
        if target.is_symlink():
            raise ValueError("symlink in scheduled continuation write scope: " + name)
        if not target.exists():
            records.append({"path": name, "status": "MISSING"}); continue
        if target.is_file():
            add_file(target); continue
        if not target.is_dir():
            raise ValueError("unsupported scheduled continuation write path: " + name)
        records.append({"path": name, "status": "DIRECTORY"})
        for path in sorted(target.rglob("*")):
            if path.is_symlink():
                raise ValueError("symlink in scheduled continuation workspace: " + str(path))
            if path.is_file() and not any(part in SKIP_DIRS for part in path.relative_to(target).parts):
                add_file(path)
    return records, copies


def _install_continuation_task(root: Path, stage_root: Path, task_id: str) -> tuple[dict | None, str | None]:
    """Validate a complete staged task, then replace it with rollback on install failure."""
    prefix = f"{CONTINUATION_ROOT}/{task_id}/"
    staged_members = continuation_members(stage_root)
    live_others = [name for name in continuation_members(root) if not name.startswith(prefix)]
    size = sum(layout.checked_path(stage_root, name).stat().st_size for name in staged_members)
    size += sum(layout.checked_path(root, name).stat().st_size for name in live_others)
    if len(staged_members) + len(live_others) > CONTINUATION_MEMBER_LIMIT or size > CONTINUATION_LIMIT:
        raise ValueError("continuation workspace exceeds bounded review handoff budget")
    continuation, _, staged_sha = _read_continuation(stage_root, task_id)
    starts = _native_start_receipts(stage_root, task_id)
    parent = layout.checked_path(root, CONTINUATION_ROOT)
    target = layout.checked_path(root, prefix.rstrip("/"))
    stage_task = layout.checked_path(stage_root, prefix.rstrip("/"))
    backup = parent.parent / ("." + task_id + ".adopt-backup")
    if backup.exists():
        raise ValueError("stale continuation adoption backup exists: " + backup.name)
    parent.mkdir(parents=True, exist_ok=True)
    had_target = target.is_dir()
    if had_target:
        os.replace(target, backup)
    try:
        os.replace(stage_task, target)
        installed, _, installed_sha = _read_continuation(root, task_id)
        if installed != continuation or installed_sha != staged_sha or _native_start_receipts(root, task_id) != starts:
            raise ValueError("installed continuation differs from verified staged state")
    except Exception:
        if target.exists():
            shutil.rmtree(target)
        if had_target:
            os.replace(backup, target)
        raise
    if had_target:
        shutil.rmtree(backup)
    return installed, installed_sha


def save_partial(root: Path, draft: dict) -> dict:
    """Install a canonical task-keyed CONTINUE file; this never changes schedule state."""
    ticket = saved_assignment(root)
    required = {"done", "exact_state", "next_command", "do_not_redo"}
    optional = {"remaining", "blockers", "checkpoint"}
    if not isinstance(draft, dict) or not required <= set(draft) or set(draft) - required - optional:
        raise ValueError("PARTIAL draft requires done, exact_state, next_command and do_not_redo; remaining, blockers and checkpoint are optional")
    if (not isinstance(draft["done"], list) or not draft["done"]
            or any(not isinstance(x, str) or not x.strip() for x in draft["done"])):
        raise ValueError("PARTIAL done must be a nonempty string list")
    if not isinstance(draft["exact_state"], str) or not draft["exact_state"].strip():
        raise ValueError("PARTIAL exact_state must be nonempty text")
    if not isinstance(draft["next_command"], str) or not draft["next_command"].strip():
        raise ValueError("PARTIAL next_command must be nonempty text")
    if (not isinstance(draft["do_not_redo"], list)
            or any(not isinstance(x, str) or not x.strip() for x in draft["do_not_redo"])):
        raise ValueError("PARTIAL do_not_redo must be a string list")
    remaining = draft.get("remaining", [])
    blockers = draft.get("blockers", [])
    for field, value in (("remaining", remaining), ("blockers", blockers)):
        if (not isinstance(value, list)
                or any(not isinstance(x, str) or not x.strip() for x in value)):
            raise ValueError(f"PARTIAL {field} must be a string list")
    checkpoint = draft.get("checkpoint", "UNPLANNED_PARTIAL")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("PARTIAL checkpoint must be nonempty text")
    task_id = _ticket_task_id(ticket)
    previous, relative, _ = _read_continuation(root, task_id)
    budget = _native_start_budget(root, ticket, previous)
    workspace, copies = _snapshot_workspace(root, ticket)
    generation = 1 if previous is None else previous["generation"] + 1
    partial_name = f"FARM-PARTIAL-{task_id}-{ticket['ticket'][:8]}.zip"
    value = {"schema": CONTINUATION_SCHEMA, "status": "PARTIAL", "task_id": task_id,
             "source_ticket": ticket["ticket"], "partial_return_name": partial_name,
             "final_return_name": ticket["return_name"], "generation": generation,
             "done": draft["done"], "exact_state": draft["exact_state"],
             "next_command": draft["next_command"], "soft_target_games": budget["soft_target_games"],
             "starts_recorded": budget["starts_recorded"], "do_not_redo": draft["do_not_redo"],
             "remaining": remaining, "blockers": blockers, "checkpoint": checkpoint,
             "workspace": workspace, "workspace_sha256": digest(review_intake.canonical(workspace)),
             "retirement_rule": "Persist across waves and slot changes until the task is ACCEPTED, its result is admitted as a ledger claim, or the maintainer explicitly retires it; a wave ending never retires it."}
    # No live writes or stale-copy deletions until every member has been captured.
    # A directory swap keeps CONTINUE.json and its workspace copies in one generation.
    review_dir = layout.checked_path(root, "review")
    review_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".partial-save-", dir=review_dir) as temporary:
        stage_root = Path(temporary)
        task_dir = layout.checked_path(root, f"{CONTINUATION_ROOT}/{task_id}")
        for path in sorted(task_dir.glob("START-*.json")):
            copies[path.relative_to(root).as_posix()] = path.read_bytes()
        for name, raw in copies.items():
            destination = layout.checked_path(stage_root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        _atomic_json(layout.checked_path(stage_root, relative), value, replace=True)
        _install_continuation_task(root, stage_root, task_id)
    state = continuation_state(root, ticket)
    return {"status": "PARTIAL_SAVED", "archive_name": partial_name, "path": relative,
            "sha256": state["sha256"], "continuation": value,
            "required_return_files": [ASSIGNMENT, "RESULT.md", relative],
            "difference_from_final": "PARTIAL preserves unfinished workspace and charges but does not accept/reject the task; the final return keeps ticket.return_name and must satisfy task acceptance.",
            "next": draft["next_command"]}


def resume_continuation(root: Path) -> dict:
    """Restore work-only saved copies, refusing to overwrite divergent current bytes."""
    ticket = saved_assignment(root)
    task_id = _ticket_task_id(ticket)
    value, _, _ = _read_continuation(root, task_id)
    if value is None:
        raise ValueError("no saved PARTIAL continuation for assigned task")
    writes = ticket["assignment"].get("schedule", {}).get("writes", [])
    restored, already = [], []
    for row in value["workspace"]:
        saved = row.get("saved_copy") if row.get("status") == "FILE" else None
        if not saved:
            continue
        if not _inside_write_scope(row["path"], writes):
            raise ValueError("saved workspace path is outside current task write scope: " + row["path"])
        source = layout.checked_path(root, saved); target = layout.checked_path(root, row["path"])
        if target.exists():
            if not target.is_file() or target.stat().st_size != row["bytes"] or digest(target.read_bytes()) != row["sha256"]:
                raise ValueError("refusing to overwrite divergent resumed workspace: " + row["path"])
            already.append(row["path"]); continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        restored.append(row["path"])
    return {"status": "CONTINUATION_RESTORED", "task_id": task_id,
            "restored": restored, "already_current": already,
            "next_command": value["next_command"], "native_start_budget": _native_start_budget(root, ticket, value)}


def _verified_continuation_adoption(archive: Path, registry: Path) -> dict:
    """Use canonical intake authentication, including returns containing only STARTs.

    Intake owns archive safety, baseline and issuance joins. Its PARTIAL routing
    stays unchanged: receipt-only returns are FINAL/UNCLASSIFIED, never PARTIAL.
    """
    report = review_intake.scan([archive], registry)
    if report["errors"] or len(report["returns"]) != 1:
        error = report["errors"][0]["error"] if report["errors"] else "expected exactly one return"
        raise ValueError("continuation adoption refused: " + error)
    row = report["returns"][0]
    if row.get("return_kind") == "PARTIAL":
        return review_intake.verified_partial_adoption(archive, registry)
    if row["status"] != "INVENTORIED":
        raise ValueError("START receipt adoption requires one clean return")
    if row.get("baseline_comparison") != "EXACT_DECLARED_BASELINE":
        raise ValueError("START receipt adoption requires the exact registered baseline")
    if row.get("ticket_registry_status") != "EXACT_ISSUANCE":
        raise ValueError("START receipt adoption requires an exact registered ticket issuance")
    task_id = _continuation_task(row.get("assignment_id"))
    prefix = f"{CONTINUATION_ROOT}/{task_id}/"
    selected = {}
    for name, record in row["members"].items():
        if not name.startswith(prefix):
            continue
        if not re.fullmatch(r"START-\d{6}\.json", name[len(prefix):]):
            raise ValueError("receipt-only adoption cannot carry workspace without CONTINUE.json: " + name)
        selected[name] = record
    if not selected:
        raise ValueError("START receipt adoption requires native start receipts")
    payloads = {}
    with archive.open("rb") as stream:
        with zipfile.ZipFile(stream) as zipped:
            for name, record in selected.items():
                raw = zipped.read(record["member"])
                if len(raw) != record["bytes"] or digest(raw) != record["sha256"]:
                    raise ValueError("START receipt member changed after intake scan: " + name)
                payloads[name] = raw
        stream.seek(0)
        if hashlib.file_digest(stream, "sha256").hexdigest() != row["zip_sha256"]:
            raise ValueError("START receipt archive changed during adoption")
    return {"task_id": task_id, "ticket": row["ticket"],
            "source_manifest_sha256": row["resolved_baseline_manifest_sha256"],
            "zip_sha256": row["zip_sha256"], "members": payloads}


def adopt_partial(root: Path, archive: Path, registry: Path | None = None) -> dict:
    """Install registered PARTIAL state or START receipts; never apply a code delta."""
    registry = registry or layout.checked_path(root, DISPATCH_REGISTRY_ROOT)
    payload = _verified_continuation_adoption(archive, registry)
    task_id = _continuation_task(payload["task_id"])
    incoming_partial = _continuation_path(task_id) in payload["members"]
    parent = layout.checked_path(root, CONTINUATION_ROOT)
    parent.parent.mkdir(parents=True, exist_ok=True)
    target = layout.checked_path(root, f"{CONTINUATION_ROOT}/{task_id}")
    if target.exists() and not target.is_dir():
        raise ValueError("existing continuation target must be a regular directory")
    existing, _, _ = _read_continuation(root, task_id)
    # Stage outside review/continuations: no reader should see a temporary task.
    with tempfile.TemporaryDirectory(prefix=".partial-adopt-", dir=parent.parent) as temporary:
        stage_root = Path(temporary)
        stage_task = layout.checked_path(stage_root, f"{CONTINUATION_ROOT}/{task_id}")
        if target.is_dir():
            continuation_members(root)  # Refuse live symlinks before copytree follows them.
            shutil.copytree(target, stage_task)
        for relative, raw in payload["members"].items():
            destination = layout.checked_path(stage_root, relative)
            if destination.name.startswith("START-") and destination.exists() and destination.read_bytes() != raw:
                raise ValueError("conflicting native start receipt: " + relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)

        continuation, _, _ = _read_continuation(stage_root, task_id)
        if incoming_partial:
            if continuation is None or continuation["source_ticket"] != payload["ticket"]:
                raise ValueError("PARTIAL continuation source_ticket does not match registered return ticket")
            if existing is not None:
                if continuation["generation"] < existing["generation"]:
                    raise ValueError("PARTIAL continuation generation would regress installed state")
                if continuation["generation"] == existing["generation"] and continuation != existing:
                    raise ValueError("PARTIAL continuation conflicts with installed generation")
        starts = _native_start_receipts(stage_root, task_id)
        expected_count = continuation["starts_recorded"] if incoming_partial else len(starts)
        if [row["ordinal"] for row in starts] != list(range(1, expected_count + 1)):
            raise ValueError("continuation START receipts must exactly cover historical starts")
        installed, installed_sha = _install_continuation_task(root, stage_root, task_id)
    result = {"status": "PARTIAL_ADOPTED" if incoming_partial else "START_RECEIPTS_ADOPTED",
              "task_id": task_id, "ticket": payload["ticket"],
              "source_manifest_sha256": payload["source_manifest_sha256"],
              "archive_sha256": payload["zip_sha256"], "continuation_sha256": installed_sha,
              "starts_recorded": max(len(starts), installed["starts_recorded"] if installed else 0)}
    if incoming_partial:
        result.update({"generation": installed["generation"], "path": _continuation_path(task_id),
                       "next": "Redispatch the same task; its assignment context will surface this saved PARTIAL."})
    else:
        result.update({"path": f"{CONTINUATION_ROOT}/{task_id}",
                       "next": "START receipts preserved; no PARTIAL was created or task state changed."})
    return result


def _readiness_snapshot(root: Path) -> dict:
    """Byte/metadata boundary for one readiness call, never a path-only cache.

    Readiness owns an unassigned disposable tree. From first validation through
    the last projection it may change only ASSIGNMENT (and its temporary file,
    which _save_ticket removes before returning). All other files AND directory
    entries must stay fixed, including unmanifested/missing-delivery inputs.
    SHA256 plus inode/ctime checks catch rewrites even with restored bytes/mtime.
    Symlinks, special files, unreadable or changing reads fail closed. This is a
    checked no-writer precondition, not an OS lock against hostile writers.
    """
    rows = {}
    ticket_parent = PurePosixPath(ASSIGNMENT).parent.as_posix()

    def token(info, *, directory=False, owned_parent=False):
        stable = (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)
        # Only this directory's times/size may change while saving/removing a
        # ticket. Its identity, permissions and all non-ticket entries still join.
        if owned_parent:
            return stable
        return stable + (() if directory else (info.st_size, info.st_nlink)) + (
            info.st_mtime_ns, info.st_ctime_ns)

    def visit(path, name):
        before = path.lstat()
        if stat.S_ISDIR(before.st_mode):
            stamp = token(before, directory=True, owned_parent=name == ticket_parent)
            rows[name] = stamp
            for child in sorted(path.iterdir()):
                relative = child.relative_to(root).as_posix()
                if relative != ASSIGNMENT:
                    visit(child, relative)
            after = token(path.lstat(), directory=True, owned_parent=name == ticket_parent)
        elif stat.S_ISREG(before.st_mode):
            stamp = token(before)
            with path.open("rb") as stream:
                if token(os.fstat(stream.fileno())) != stamp:
                    raise ValueError("source changed during dispatch readiness: " + name)
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
                if token(os.fstat(stream.fileno())) != stamp:
                    raise ValueError("source changed during dispatch readiness: " + name)
            rows[name] = stamp + (sha,)
            after = token(path.lstat())
        else:
            raise ValueError("dispatch readiness needs regular files/directories: " + name)
        if after != stamp:
            raise ValueError("source changed during dispatch readiness: " + name)

    try:
        visit(root, ".")
    except OSError as error:
        raise ValueError("cannot establish unchanged dispatch readiness tree: " + str(error)) from error
    return rows


def _readiness_unchanged(root: Path, snapshot: dict) -> None:
    current = _readiness_snapshot(root)
    if current != snapshot:
        changed = next(name for name in sorted(snapshot.keys() | current.keys())
                       if snapshot.get(name) != current.get(name))
        raise ValueError("source changed during dispatch readiness; use an unchanged copy: " + changed)


def dispatch_readiness(root: Path) -> dict:
    """Exercise every slot, sharing only one globally validated context.

    Standalone assign/context/evidence validation is unchanged. Each slot still
    authenticates the full manifest, checks the saved ticket and assignment join,
    and validates parent bytes/catalog. Reuse is invocation-local and bracketed
    by byte/metadata checks; no result escapes if the no-mutation window breaks.
    Tickets are written one at a time and removed; an existing ticket refuses.
    """
    target = layout.checked_path(root, ASSIGNMENT)
    if target.exists():
        raise ValueError("dispatch readiness needs a copy without a saved assignment")
    _, _, pool, _ = _pool(root)
    rows = []
    context = snapshot = None
    try:
        for slot in dispatch_plan(root, pool)["slots"]:
            ticket = assign(root, slot=slot["number"])
            if context is None:
                # First assign creates the ticket home and fully authenticates
                # source before capturing the no-mutation window.
                snapshot = _readiness_snapshot(root)
                context = review_context(root)
            _readiness_unchanged(root, snapshot)
            view = _assignment_context(root, saved_assignment(root), context)
            evidence_view = assignment_evidence(root, context, slot["task"])
            _readiness_unchanged(root, snapshot)
            rows.append({"slot": slot["number"], "task": slot["task"], "return_name": ticket["return_name"],
                         "context": view["schema"], "parent_receipts": evidence_view["selected_receipts"],
                         "unrepresented_parents": len(evidence_view["unrepresented_parents"])})
            target.unlink()
    finally:
        if target.exists():
            target.unlink()
    return {"schema": "farm-review-dispatch-readiness-v1", "slots": rows, "commands_executed": 0}


def draw_assignment(root: Path, *, create: bool = True) -> dict:
    """Legacy lottery: draw once per extracted workspace; independent copies may overlap.

    Uniform OS randomness selects a prepared question, not a random proof claim.
    An existing ticket, drawn or dispatched, is returned instead of drawing again.
    No shared cross-reviewer lock, uniqueness guarantee or work execution is implied.
    """
    from farm_neat.evaluation.common import bound
    manifest_bytes, manifest, pool, pool_sha = _pool(root)
    if layout.checked_path(root, ASSIGNMENT).exists():
        return saved_assignment(root)
    if not create:
        raise ValueError("no saved assignment; run --assign-slot N once before scoped navigation")
    unchanged(root, manifest)
    lottery_view = schedule_inspect(root, pool)
    ready = {row["id"]: row for row in lottery_view["tasks"]
             if row["id"] in set(lottery_view["ready_for_review"])}
    ready_rows = [ready[item["id"]] for item in pool if item["id"] in ready]
    by_id = dict(zip([row["id"] for row in ready_rows], work_schedule.draw_weights(ready_rows))) if ready_rows else {}
    weights = [by_id.get(item["id"], 0) for item in pool]
    if not sum(weights):
        raise ValueError("no ready reviewer work; do not reroll or reopen parked theory")
    draw = secrets.randbelow(sum(weights)) + 1
    number = next(i for i, edge in enumerate(itertools.accumulate(weights), 1) if draw <= edge)
    ticket = secrets.token_hex(16)
    chosen = pool[number - 1]
    return _save_ticket(root, bound({"schema": LOTTERY_TICKET, "ticket": ticket,
                   "random_number": number, "pool_size": len(pool), "weighted_draw": draw, "pool_weight": sum(weights),
                   "random_source": "secrets.randbelow using OS randomness; each ready reviewer task drawn with its track share (work_schedule.TRACK_SHARES) times its weight within the track; blocked, parked and maintainer-only work excluded",
                   "source_manifest_sha256": digest(manifest_bytes), "pool_sha256": pool_sha,
                   "assignment": chosen,
                   "return_name": "FARM-RETURN-" + chosen["id"] + "-" + ticket[:8] + ".zip",
                   "protocol": "Keep this ticket. Read the parent claims and shared handoff; do only the selected extension. Return explicit dependencies, assumptions, evidence, counterexamples and unresolved limits. Do not reroll for a preferred question. Overlapping independent reviews are consolidated by the maintainer; no unique claim is promised.",
                   "retention": {"purpose": "Persist one source-bound theory assignment through reviewer interruption",
                                 "dependencies": [MANIFEST, CONTEXT_SPEC],
                                 "remove_when": "Maintainer has integrated this review return and retained its ticket and findings."}}))


def review_context(root: Path, names: list[str] | None = None, *, require_deliveries: bool = False) -> dict:
    """Validate owning notes and source/evidence closure, without executing commands.

    A missing future delivery is explicitly blocked, not a successful experiment.
    --require-deliveries is an optional ready-for-outcome-review gate; unfinished
    implementation work may still be reviewed with its exact blocker.
    """
    path = layout.checked_path(root, CONTEXT_SPEC)
    if not path.is_file():
        raise ValueError("missing continuation contract: " + CONTEXT_SPEC)
    spec = tomllib.loads(path.read_text())
    if spec.get("version") != 1:
        raise ValueError("unsupported review context version")
    if 'evidence' in spec:
        expected = {'catalog': evidence.CATALOG, 'producer_policy': evidence.POLICY,
                    'max_zip_bytes': LIMIT, 'max_work_bytes': WORK_LIMIT, 'max_members': MEMBER_LIMIT}
        if spec['evidence'] != expected:
            raise ValueError('canonical evidence paths and fixed publication budgets required')
    selected = set(inventory(root) if names is None else names)
    required = set(spec["required_files"]) | set(spec["read_first"]) | {CONTEXT_SPEC}
    entrypoint = spec["entrypoint"]
    required.add(entrypoint)
    instruction = layout.checked_path(root, entrypoint)
    limit = spec["max_instruction_bytes"]
    if type(limit) is not int or limit <= 0 or instruction.stat().st_size > limit:
        raise ValueError("active instructions exceed the declared reading budget; consolidate existing notes")

    def describe(name):
        if name not in selected:
            raise ValueError("required continuation dependency is not packaged: " + name)
        payload = layout.checked_path(root, name).read_bytes()
        return {"path": name, "sha256": digest(payload), "bytes": len(payload)}

    threads = []
    tasks = []
    note_paths = set()
    sections = spec.get("thread_sections", [])
    if not isinstance(sections, list) or len(sections) != len(set(sections)):
        raise ValueError("distinct thread sections required")
    for key, target in (("threads", threads), ("tasks", tasks)):
        seen = set()
        for item in spec[key]:
            if item["id"] in seen:
                raise ValueError("duplicate continuation note id: " + item["id"])
            seen.add(item["id"])
            name = item["path"]
            if name in note_paths:
                raise ValueError("duplicate continuation note path: " + name)
            note_paths.add(name)
            required.add(name)
            record = describe(name)
            text = layout.checked_path(root, name).read_text(encoding="utf-8")
            fields = note_fields(text, name)
            if key == "threads":
                for section in sections:
                    marker = "**" + section + ".**"
                    matching = [line for line in text.splitlines() if line.startswith(marker)]
                    if len(matching) != 1 or not matching[0][len(marker):].strip():
                        raise ValueError("one nonempty thread section required: " + name + " -> " + section)
            target.append({**record, "id": item["id"], "purpose": fields["Purpose"],
                           "owner": fields["Owner"], "status": fields["Status"],
                           "next": fields["Next"], "evidence": fields["Evidence"],
                           "remove_when": fields["Remove-when"]})

    reading = reading_context(root, spec, describe)
    required.update(record["path"] for record in reading["files"])
    history = spec.get('source_history', [])
    for relative in history:
        required.add(relative); describe(relative)
        transition = json.loads(layout.checked_path(root, relative).read_text())
        required.add(transition['patch']); describe(transition['patch'])
        for name in transition['files']:required.add(name);describe(name)
    versions = source_versions(root, history)
    joins = []
    for binding in spec.get("bindings", []):
        name, field, base = binding["path"], binding["field"], binding["base"]
        required.add(name)
        value = json.loads(layout.checked_path(root, name).read_text())
        pins = value[field]
        if "path_key" in binding:
            if not isinstance(pins, list) or not pins:
                raise ValueError("nonempty evidence records required: " + name)
            pairs = [(row[binding["path_key"]], row[binding["sha256_key"]]) for row in pins]
            if len({p for p, _ in pairs}) != len(pairs):
                raise ValueError("duplicate evidence record path: " + name)
            pins = dict(pairs)
        if not isinstance(pins, dict) or not pins:
            raise ValueError("nonempty source/evidence binding required: " + name)
        historical = 0
        for relative, expected in pins.items():
            dependency = str(PurePosixPath(base) / relative)
            record = describe(dependency)
            if not isinstance(expected, str) or (record["sha256"] != expected and expected not in versions.get(dependency,set())):
                raise ValueError("stale continuation binding: " + name + " -> " + dependency)
            historical += record['sha256'] != expected
            required.add(dependency)
        joins.append({"path": name, "field": field, "verified_dependencies": len(pins),
                      "historical_sources_verified_as_data": historical})

    deliveries = []
    delivery_ids = set()
    for item in spec.get("deliveries", []):
        if not item.get("id") or item["id"] in delivery_ids:
            raise ValueError("distinct delivery ids required")
        delivery_ids.add(item["id"])
        if item["note"] not in note_paths:
            raise ValueError("delivery must name its owning active note")
        files = item["files"]
        if not isinstance(files, list) or not files or len(files) != len(set(files)):
            raise ValueError("distinct nonempty delivery paths required")
        present, missing = [], []
        for name in files:
            candidate = layout.checked_path(root, name)
            if candidate.exists():
                present.append(describe(name)); required.add(name)
            else:
                missing.append(name)
        references = []
        for key in ("request", "snapshot"):
            if key in item:
                references.append(describe(item[key])); required.add(item[key])
        deliveries.append({"id": item["id"], "note": item["note"], "present": present,
            "missing": missing, "references": references,
            "status": "WAITING_FOR_EVIDENCE" if missing else "FILES_PRESENT_NOT_SCIENTIFIC_VERIFICATION",
            "limitation": "Scheduler records are saved snapshots, never proof of current live execution."})
    if require_deliveries and any(d["missing"] for d in deliveries):
        raise ValueError("outcome review blocked by missing delivery evidence: " + "; ".join(
            d["id"] + ": " + ", ".join(d["missing"]) for d in deliveries if d["missing"]))
    assignments = theory_assignments(root, spec, describe)
    for item in assignments:
        required.update(record["path"] for record in item["parent_claims"] + item["input_files"] + item["acceptance_files"])
    plan = None
    if layout.checked_path(root, DISPATCH).is_file():
        plan = {**describe(DISPATCH), **dispatch_plan(root, assignments)}
        required.add(DISPATCH)
    required_records = [describe(name) for name in sorted(required)]
    result = {"schema": "farmneat-review-continuation-v1", "mode": spec["mode"],
            "entrypoint": entrypoint, "read_first": [describe(n) for n in spec["read_first"]],
            "instructions_bytes": instruction.stat().st_size, "instruction_budget_bytes": limit,
            "threads": threads, "tasks": tasks, "bindings": joins, "reading": reading,
            "deliveries": deliveries, "assignments": assignments, "required_files": required_records,
            "checks": spec.get("checks", []), "commands_executed": 0,
            **({"dispatch": plan} if plan is not None else {}),
            "ownership": "Tasks and discussion notes own conclusions and next actions; this map is generated.",
            "limitations": ["No cross-session memory is assumed; ship this manifest and its referenced files.",
                            "A recorded source hash is not access to omitted weights or a training environment."]}
    # The readable output, including framing and delivery blockers, has its own
    # hard bound. Validate this during build too, not only when someone reads it.
    render_handoff(root, result)
    return result


def assignment_evidence(root: Path, context: dict, assignment_id: str) -> dict:
    """Read-only navigation: return intact receipts bound to exact direct parents.

    This does not draw a ticket, assess findings, or claim transitive coverage.
    Retain every matching receipt, including adverse results and interpretations.
    """
    matches = [item for item in context["assignments"] if item["id"] == assignment_id]
    if len(matches) != 1:
        raise ValueError("one known assignment id required: " + assignment_id)
    assignment = matches[0]
    parents = []
    for parent in assignment["parent_claims"]:
        source = {key: parent[key] for key in ("path", "sha256", "bytes")}
        evidence.binding(source)
        if evidence.identity(root, source["path"]) != source:
            raise ValueError("assignment parent source drift: " + source["path"])
        parents.append(source)
    wanted = {(p["path"], p["sha256"], p["bytes"]) for p in parents}
    records = evidence.catalog(root)
    selected, represented = [], set()
    for record in records:
        bound = {(s["path"], s["sha256"], s["bytes"]) for s in record["source"]}
        joined = bound & wanted
        if joined:
            selected.append(record)
            represented.update(joined)
    return {
        "schema": "farm-review-assignment-evidence-view-v1",
        "assignment": assignment,
        "catalog": evidence.identity(root, evidence.CATALOG),
        "catalog_receipts": len(records),
        "selected_receipts": len(selected),
        "selected_receipt_bytes": len(evidence.encode(selected)),
        "not_selected_receipts": len(records) - len(selected),
        "unrepresented_parents": [p for p in parents
            if (p["path"], p["sha256"], p["bytes"]) not in represented],
        "selection": "EXACT_DIRECT_PARENT_BINDINGS; NOT_TRANSITIVE_GRAPH_CLOSURE",
        "limitations": [
            "Navigation only. This command does not draw or change an assignment ticket.",
            "All matching receipts remain intact; nonmatching receipts remain in the catalog.",
            "Related or contradicting claims not bound to these parents may be outside this view.",
            "Read linked claims and the assigned Q-thread before making a decision.",
            "Omitted producer evidence remains an attestation, not a replay or verified statistic.",
        ],
        "commands_executed": 0,
        "receipts": selected,
    }


def render_handoff(root: Path, context: dict) -> str:
    """Render every canonical reading byte; never truncate or summarize by slicing."""
    reading = context["reading"]
    chunks = ["# farmNEAT — complete maintained review handoff\n",
              "Reading identity: " + reading["identity_sha256"] + "\n",
              "This view contains every current brief/thread/task in the declared reading set. "
              "Source and raw receipts remain linked evidence, not implied reading. "
              "No commands were executed and no live scheduler access is claimed.\n"]
    for item in reading["files"]:
        payload = layout.checked_path(root, item["path"]).read_bytes()
        if digest(payload) != item["sha256"]:
            raise ValueError("note changed while generating handoff: " + item["path"])
        chunks.append("\n---\n## " + item["path"] + "\n\n" + payload.decode("utf-8") + "\n")
    for item in context.get("deliveries", []):
        chunks.append("\n## Delivery: " + item["id"] + "\n" + item["status"] + "\n")
        if item["missing"]:
            chunks.append("Missing: " + ", ".join(item["missing"]) + "\n")
        chunks.append("Owner note: " + item["note"] + ". " + item["limitation"] + "\n")
    chunks.append("\n---\nEND OF COMPLETE MAINTAINED HANDOFF. "
                  "Integrate changes into the owning notes; do not append this view to the next ZIP.\n")
    rendered = "".join(chunks)
    limit = reading.get("render_budget_bytes")
    if limit is not None and (type(limit) is not int or limit <= 0 or len(rendered.encode()) > limit):
        raise ValueError("rendered handoff exceeds reading budget; consolidate owning notes")
    return rendered


def write_archive(root: Path, names: list[str], target: Path, admission: dict,
                  context: dict | None = None, *, generated: dict[str, bytes] | None = None) -> dict:
    if len(names) != len(set(names)) or MANIFEST in names:
        raise ValueError('distinct input names; manifest is generated')
    generated = generated or {}
    if set(generated) - {SELF_TEST_STATUS} or set(generated) & set(names):
        raise ValueError('only the distinct self-test status may be generated')
    if len(names) + len(generated) + 1 >= MEMBER_LIMIT:
        raise ValueError('review member budget exceeded')
    work = {n for n in names if n.startswith('work/')}
    view = evidence.packet_view(root) if work else None
    if view is not None:
        classified = {evidence.CATALOG, *[i['path'] for i in view['retained']]}
        if work != classified:
            raise ValueError('UNKNOWN or missing selected work artifact')
        if view['witness_count'] > WITNESS_COUNT or view['witness_bytes'] >= WITNESS_LIMIT:
            raise ValueError('explicit raw witness budget exceeded')
    members = {}
    unpacked_bytes = work_bytes = 0
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        def add(name, payload):
            nonlocal unpacked_bytes, work_bytes
            unpacked_bytes += len(payload)
            if name.startswith("work/"):
                work_bytes += len(payload)
                if work_bytes >= WORK_LIMIT:
                    raise ValueError("review work bytes including generated manifest exceed budget")
            if unpacked_bytes >= LIMIT:
                raise ValueError(f"review contents including generated manifest must be below {LIMIT:,} bytes")
            info = zipfile.ZipInfo("farmNEAT/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload, compresslevel=9)

        for name in [*names, *generated]:
            payload = generated[name] if name in generated else layout.checked_path(root, name).read_bytes()
            members[name] = {"sha256": digest(payload), "bytes": len(payload)}
            add(name, payload)
        manifest = {
            "schema": SCHEMA, "mode": "SOURCE_AND_SEMANTIC_EVIDENCE_REVIEW",
            "start_here": "review/INSTRUCTIONS.md",
            "capabilities": ["Inspect current source, contracts, tests and linked theory questions",
                "Inspect source-bound semantic results, limits and explicit raw witnesses",
                "Play scripted-policy games with the bundled exact engine, PASS and ShopRouter",
                "Run the advertised portable checks without producer history"],
            "not_included": ["Full actor and optimizer tensors", "Historical trajectories, matrices, journals, predictions and bulk PLAN/RESULT payloads",
                "Producer-only provenance index, environments, caches and incoming returns"],
            "limitations": ["Not a checkpoint continuation or full-policy neural game package",
                "Historical raw replay requires maintainer evidence; a hash is not access or proof of a statistic",
                "Only checks declared in project/review.toml are advertised as self-contained",
                "review/INSTRUCTIONS.md owns external authority; maintainer deadlines are historical context for reviewers"],
            "checks": ["Member hashes, safe paths, exact byte/member budgets and fresh extraction",
                "Extracted layout, Markdown links, entrypoint reachability and game dependencies"],
            "presentation_omissions": [
                {"path": png, "source": svg,
                 "reason": "Derived PNG export; the editable SVG renders the same figure. Regenerate with export_figure.py."}
                for png, svg in DERIVED_FIGURES.items() if svg in names],
            "storage_admission": admission,
            "retention": {"purpose": "One current external source and semantic-evidence packet",
                "remove_when": "Replace only after the next packet verifies; no ZIP history"},
            "members": members,
        }
        if view is not None:
            manifest['evidence'] = view
        if context is not None:
            manifest["continuation"] = context
        add(MANIFEST, (json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n").encode())
    if target.stat().st_size >= LIMIT:
        raise ValueError(f"review ZIP must be below {LIMIT:,} bytes")
    return manifest


def verify_archive(path: Path, destination: Path) -> dict:
    """Validate all names and hashes before extracting into a fresh directory."""
    if path.stat().st_size >= LIMIT:
        raise ValueError(f"review ZIP must be below {LIMIT:,} bytes")
    if destination.exists():
        raise ValueError("extraction destination must be new")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if (len(names) != len(set(names)) or len(names) >= MEMBER_LIMIT
                or sum(i.file_size for i in infos) >= LIMIT
                or sum(i.file_size for i in infos if i.filename.startswith("farmNEAT/work/")) >= WORK_LIMIT):
            raise ValueError("duplicate members or excessive extracted size")
        for info in infos:
            name = info.filename
            parts = PurePosixPath(name).parts
            if (not name.startswith("farmNEAT/") or "\\" in name or ":" in name
                    or len(parts) < 2 or any(p in {".", ".."} for p in parts)
                    or PurePosixPath(name).as_posix() != name
                    or not stat.S_ISREG(info.external_attr >> 16) or info.flag_bits & 1
                    or any(ord(c) < 32 for c in name)):
                raise ValueError(f"unsafe archive member: {name}")
        manifest = review_intake.parse_json(archive.read("farmNEAT/" + MANIFEST))
        if manifest.get("schema") != SCHEMA:
            raise ValueError("unsupported review manifest")
        expected = {"farmNEAT/" + n for n in manifest["members"]} | {"farmNEAT/" + MANIFEST}
        if set(names) != expected:
            raise ValueError("archive membership differs from manifest")
        for name, record in manifest["members"].items():
            payload = archive.read("farmNEAT/" + name)
            if len(payload) != record["bytes"] or digest(payload) != record["sha256"]:
                raise ValueError(f"archive hash mismatch: {name}")
        archive.extractall(destination)
    return manifest


def _archive_payloads(path: Path) -> tuple[dict, dict[str, bytes]]:
    """Read one previously published packet without extracting or executing it."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("dispatch refresh requires an existing regular baseline ZIP")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or len(names) >= MEMBER_LIMIT:
            raise ValueError("baseline review ZIP has duplicate or excessive members")
        manifest_name = "farmNEAT/" + MANIFEST
        if manifest_name not in names:
            raise ValueError("baseline review ZIP lacks its manifest")
        manifest = review_intake.parse_json(archive.read(manifest_name))
        if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("members"), dict):
            raise ValueError("unsupported baseline review manifest")
        for name in manifest["members"]:
            try:
                safe = layout.safe_name(name)
            except (TypeError, ValueError) as exc:
                raise ValueError("unsafe baseline archive member: " + repr(name)) from exc
            if safe != name:
                raise ValueError("unsafe baseline archive member: " + repr(name))
        expected = {"farmNEAT/" + name for name in manifest["members"]} | {manifest_name}
        if set(names) != expected:
            raise ValueError("baseline archive membership differs from manifest")
        payloads = {}
        for name, record in manifest["members"].items():
            payload = archive.read("farmNEAT/" + name)
            if len(payload) != record.get("bytes") or digest(payload) != record.get("sha256"):
                raise ValueError("baseline archive hash mismatch: " + name)
            payloads[name] = payload
    return manifest, payloads


def _dispatch_slots(payload: bytes) -> list[dict]:
    spec = tomllib.loads(payload.decode())
    rows, numbers, tasks = [], set(), set()
    for row in spec.get("slot", []):
        number, task = row.get("number"), row.get("task")
        if type(number) is not int or number <= 0 or number in numbers:
            raise ValueError("dispatch slots require distinct positive numbers")
        if (not isinstance(task, str) or not re.fullmatch(r"[A-Z][A-Z0-9-]{0,31}", task)
                or task in tasks):
            raise ValueError("dispatch slots require distinct safe task ids")
        numbers.add(number); tasks.add(task)
        rows.append({"slot": number, "task": task})
    if not rows:
        raise ValueError("dispatch publication requires at least one slot")
    return rows


def _snapshot_assignment_pool(spec: dict, payloads: dict[str, bytes]) -> list[dict]:
    """Bind current assignment metadata only to bytes captured for this snapshot."""
    def describe(name: str) -> dict:
        payload = payloads.get(name)
        if payload is None:
            raise ValueError("dispatch snapshot assignment input is absent: " + name)
        return {"path": name, "sha256": digest(payload), "bytes": len(payload)}

    pool = []
    for item in spec.get("assignments", []):
        parents = []
        for claim_id in item.get("parents", []):
            name = "project/registries/search-ledger/" + claim_id + ".json"
            record = describe(name)
            claim = review_intake.parse_json(payloads[name])
            parents.append({**record, "id": claim_id, "key": claim.get("key"),
                            "title": claim.get("title")})
        evidence_paths = [row["path"] for row in item.get("schedule", {}).get("evidence", [])]
        pool.append({**item, "parent_claims": parents,
                     "input_files": [describe(name) for name in item.get("inputs", [])],
                     "acceptance_files": [describe(name) for name in sorted(set(evidence_paths))]})
    return pool


def _capture_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("dispatch snapshot requires regular files: " + str(path))
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("source changed during dispatch snapshot: " + str(path))
    return payload


def prepare_dispatch_archive(root: Path, baseline: Path, destination: Path) -> dict:
    """Overlay current source/control files onto the last published immutable packet.

    This is deliberately an assembler, not a preflight: it performs no layout,
    evidence, scientific, extracted-runtime, storage-scan or test gate.  It keeps
    baseline work evidence, adds current work inputs named by the registry, and
    derives the prompt slots from the exact dispatch bytes placed in the ZIP.
    """
    root, baseline, destination = Path(root), Path(baseline), Path(destination)
    manifest, payloads = _archive_payloads(baseline)

    mutable_prefixes = tuple(tree.rstrip("/") + "/" for tree in TREES)
    for name in list(payloads):
        if name in ROOT_FILES or name.startswith(mutable_prefixes) or name.startswith(CONTINUATION_ROOT + "/"):
            del payloads[name]
    for name in ROOT_FILES:
        path = root / name
        if path.exists():
            payloads[name] = _capture_regular(path)
    for tree in TREES:
        base = root / tree
        if not base.is_dir() or base.is_symlink():
            continue
        for path in sorted(base.rglob("*")):
            relative_parts = path.relative_to(base).parts
            if any(part in SKIP_DIRS for part in relative_parts):
                continue
            if path.is_symlink():
                raise ValueError("symlink in dispatch snapshot: " + str(path))
            if path.is_file() and path.suffix in SUFFIXES:
                payloads[path.relative_to(root).as_posix()] = _capture_regular(path)
    continuation = root / CONTINUATION_ROOT
    if continuation.is_dir() and not continuation.is_symlink():
        for path in sorted(continuation.rglob("*")):
            if path.is_symlink():
                raise ValueError("symlink in dispatch continuation: " + str(path))
            if path.is_file() and path.suffix in CONTINUATION_SUFFIXES:
                payloads[path.relative_to(root).as_posix()] = _capture_regular(path)

    registry = tomllib.loads(payloads[CONTEXT_SPEC].decode())
    task_by_id = {row.get("id"): row for row in registry.get("assignments", [])}
    dispatch_spec = tomllib.loads(payloads[DISPATCH].decode())
    _dispatch_numbers(dispatch_spec)
    slots = _dispatch_slots(payloads[DISPATCH])
    work_names = {evidence.CATALOG}
    for task in registry.get("assignments", []):
        for row in task.get("schedule", {}).get("evidence", []):
            if isinstance(row, dict) and isinstance(row.get("path"), str):
                work_names.add(row["path"])
    for slot in slots:
        task = task_by_id.get(slot["task"])
        if task is None:
            raise ValueError("dispatch task absent from current registry: " + slot["task"])
        work_names.update(name for name in task.get("inputs", []) if name.startswith("work/"))
    catalog_path = root / evidence.CATALOG
    if catalog_path.is_file():
        for line in catalog_path.read_bytes().splitlines():
            record = review_intake.parse_json(line)
            work_names.update(row["path"] for row in record.get("source", [])
                              if isinstance(row.get("path"), str) and row["path"].startswith("work/"))
            work_names.update(row["copy"] for row in record.get("result", {}).get("historical_sources", [])
                              if isinstance(row.get("copy"), str) and row["copy"].startswith("work/"))
    for name in sorted(work_names):
        path = root / name
        if path.is_file():
            payloads[name] = _capture_regular(path)

    assignment_pool = _snapshot_assignment_pool(registry, payloads)

    status = {"schema": "farm-review-self-test-status-v1", "status": "SKIPPED",
              "suite_for_this_zip": "NOT_RUN_BY_DISPATCH_PUBLICATION",
              "reason": "Dispatch publication assembles bytes only; run --full-build or --self-test-zip separately."}
    payloads[SELF_TEST_STATUS] = review_intake.canonical(status) + b"\n"
    work_bytes = sum(len(payload) for name, payload in payloads.items() if name.startswith("work/"))
    if work_bytes >= WORK_LIMIT:
        raise ValueError("dispatch snapshot exceeds work byte limit")
    if len(payloads) + 1 >= MEMBER_LIMIT or sum(map(len, payloads.values())) >= LIMIT:
        raise ValueError("dispatch snapshot exceeds package member or byte limit")

    next_manifest = dict(manifest)
    next_manifest["members"] = {name: {"sha256": digest(payload), "bytes": len(payload)}
                                for name, payload in sorted(payloads.items())}
    continuation = dict(next_manifest.get("continuation", {}))
    continuation["assignments"] = assignment_pool
    continuation["dispatch"] = {
        "schema": "farm-review-dispatch-v1",
        "authority": "project/review.toml [[assignments]]",
        "path": DISPATCH,
        "sha256": digest(payloads[DISPATCH]),
        "bytes": len(payloads[DISPATCH]),
        "next_slot": dispatch_spec.get("next_slot"),
        "target_parallelism": dispatch_spec.get("target_parallelism", TARGET_PARALLELISM),
        "slots": [{"number": row["slot"], "task": row["task"]} for row in slots],
        "holds": list(dispatch_spec.get("hold", [])),
    }
    next_manifest["continuation"] = continuation
    next_manifest["storage_admission"] = {
        "schema": "farm-review-dispatch-snapshot-v1", "status": "NOT_SCANNED",
        "admissible": True, "limits": "Assembler enforces archive byte/member limits only.",
    }
    retained = {row["path"]: dict(row) for row in next_manifest.get("evidence", {}).get("retained", [])}
    work_payloads = {name: payload for name, payload in payloads.items()
                     if name.startswith("work/") and name != evidence.CATALOG}
    next_retained = []
    for name, payload in sorted(work_payloads.items()):
        row = retained.get(name, {"path": name, "tier": "ACTIVE_DEPENDENCY", "receipts": [],
                                  "reason": "Current registry input or retained baseline review evidence."})
        row.update(path=name, sha256=digest(payload), bytes=len(payload))
        next_retained.append(row)
    view = dict(next_manifest.get("evidence", {}))
    catalog_payload = payloads.get(evidence.CATALOG, b"")
    view.update(catalog={"path": evidence.CATALOG, "sha256": digest(catalog_payload),
                         "bytes": len(catalog_payload)},
                semantic_receipts=len(catalog_payload.splitlines()), unclassified=0,
                retained=next_retained,
                witness_count=sum(row.get("tier") == "WITNESS" for row in next_retained),
                witness_bytes=sum(row["bytes"] for row in next_retained if row.get("tier") == "WITNESS"))
    next_manifest["evidence"] = view

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("dispatch snapshot destination must be new")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(payloads.items()):
            info = zipfile.ZipInfo("farmNEAT/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload, compresslevel=9)
        info = zipfile.ZipInfo("farmNEAT/" + MANIFEST, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        archive.writestr(info, (json.dumps(next_manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
                         compresslevel=9)
    if destination.stat().st_size >= LIMIT:
        destination.unlink()
        raise ValueError("dispatch ZIP exceeds package limit")
    return {"archive": str(destination), "slots": slots, "members": len(payloads) + 1,
            "bytes": destination.stat().st_size, "preflights": "NOT_RUN"}


def unchanged(root: Path, manifest: dict, *, generated: dict[str, bytes] | None = None) -> None:
    generated = generated or {}
    for name, record in manifest["members"].items():
        payload = generated[name] if name in generated else layout.checked_path(root, name).read_bytes()
        if digest(payload) != record["sha256"]:
            raise ValueError(f"source changed during packaging; rebuild: {name}")


def publish_dispatch_bundle(root: Path, archive: Path, slots: list[dict]) -> dict:
    """Transactionally replace the public ZIP, checksum and complete prompt set.

    Every byte is staged before the first live mutation.  If any replacement or
    stale-prompt removal fails, the exact prior public bundle is restored while
    unrelated inbox files remain untouched.
    """
    root = Path(root)
    archive = Path(archive)
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("dispatch publication requires a regular staged ZIP")
    template = layout.checked_path(root, DISPATCH_PROMPT).read_text()
    rendered, numbers = {}, set()
    for slot in slots:
        content = render_slot_prompt(template, slot).encode()
        number, task = slot["slot"], slot["task"]
        if number in numbers:
            raise ValueError("duplicate dispatch prompt slot number")
        numbers.add(number)
        name = f"SLOT-{number}-{task}.txt"
        if name in rendered:
            raise ValueError("duplicate dispatch prompt target")
        rendered[name] = content

    target = layout.checked_path(root, OUTPUT)
    inbox = target.parent
    inbox.mkdir(parents=True, exist_ok=True)
    companion = target.with_name(target.name + ".sha256")
    checksum = digest(archive.read_bytes())
    current_prompts = set(inbox.glob("SLOT-*.txt"))
    if any(path.is_symlink() or not path.is_file() for path in current_prompts):
        raise ValueError("published slot prompts must be regular files")
    live = {target, companion, *current_prompts}
    expected_prompts = {inbox / name for name in rendered}
    touched = live | expected_prompts
    if any(path.is_symlink() for path in touched):
        raise ValueError("published dispatch paths must not be symlinks")

    with tempfile.TemporaryDirectory(prefix=".dispatch-bundle-", dir=inbox) as temporary:
        stage = Path(temporary)
        staged = stage / "staged"
        backup = stage / "backup"
        staged.mkdir(); backup.mkdir()
        staged_zip = staged / target.name
        shutil.copyfile(archive, staged_zip)
        staged_checksum = staged / companion.name
        staged_checksum.write_text(f"{checksum}  {target.name}\n")
        staged_prompts = {}
        for name, payload in rendered.items():
            path = staged / name
            path.write_bytes(payload)
            staged_prompts[inbox / name] = path

        prior = {}
        for path in sorted(live):
            if path.exists():
                copy = backup / path.name
                os.link(path, copy)
                prior[path] = copy

        try:
            os.replace(staged_zip, target)
            os.replace(staged_checksum, companion)
            for destination in sorted(staged_prompts):
                os.replace(staged_prompts[destination], destination)
            for stale in sorted(current_prompts - expected_prompts):
                stale.unlink()
        except BaseException:
            for path in sorted(touched):
                if path.exists():
                    path.unlink()
            for path, copy in sorted(prior.items()):
                os.replace(copy, path)
            raise
    return {
        "file": str(target), "sha256": checksum,
        "prompts": [str(path) for path in sorted(expected_prompts)],
        "slots": [{"slot": row["slot"], "task": row["task"]} for row in slots],
    }


def publish_dispatch(root: Path = ROOT) -> dict:
    """Refresh the owner-facing ZIP/prompts and retain its exact dispatch identity."""
    root = Path(root)
    target = root / OUTPUT
    inbox = target.parent
    inbox.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dispatch-snapshot-", dir=inbox) as temporary:
        archive = Path(temporary) / "FARM-REVIEW.next.zip"
        snapshot = prepare_dispatch_archive(root, target, archive)
        with zipfile.ZipFile(archive) as source:
            names = [name for name in source.namelist() if name.endswith("/" + MANIFEST)]
            if len(names) != 1:
                raise ValueError("dispatch snapshot must contain exactly one manifest")
            manifest = Path(temporary) / "MANIFEST.json"
            manifest.write_bytes(source.read(names[0]))
        registration = review_intake.register_dispatch_manifest(
            manifest,
            root / DISPATCH_REGISTRY_ROOT,
            issued_at=datetime.now(timezone.utc).isoformat(),
        )
        published = publish_dispatch_bundle(root, archive, snapshot["slots"])
    return {
        **published,
        "source_manifest_sha256": registration["source"]["source_manifest_sha256"],
        "registered_tickets": len(registration["tickets"]),
        "members": snapshot["members"],
        "bytes": (root / OUTPUT).stat().st_size,
        "preflights": "NOT_RUN",
        "verification": "ARCHIVE_MEMBER_HASHES_FROM_BASELINE_AND_CAPTURED_BYTES",
    }


def render_slot_prompt(template: str, slot: dict) -> str:
    """Bind one owner-facing prompt to exactly one validated numeric dispatch slot."""
    if template.count("<SLOT>") != 1:
        raise ValueError("dispatch prompt must contain exactly one <SLOT> marker")
    number, task = slot.get("slot"), slot.get("task")
    if type(number) is not int or number <= 0:
        raise ValueError("dispatch prompt slot must be a positive integer")
    if not isinstance(task, str) or not re.fullmatch(r"[A-Z][A-Z0-9-]{0,31}", task):
        raise ValueError("dispatch prompt task id is unsafe")
    command = f"python3 -B src/farm_neat/tooling/review_package.py --assign-slot {number} --summary"
    return (template.replace("<SLOT>", str(number)).replace("<TASK>", task)
            .replace("<COMMAND>", command))


def packaging_closure(root: Path, names: list[str], *, omissions: list[dict] | None = None) -> dict:
    """Check concrete work-file references without importing shipped source.

    Text references, Python string constants and constant Path joins are checked.
    Computed filenames, directory enumeration and absent future outputs
    are not a static dependency proof. Omissions are exact paths with reasons;
    no prefix/glob exemption can hide a newly added fixture.
    """
    import ast
    declared = dict(PACKAGING_OMISSIONS)
    if layout.checked_path(root, evidence.POLICY).is_file():
        declared.update({item['path']: item['reason'] for item in evidence.policy(root)
                         if item['tier'] not in evidence.RETAINED})
    for item in omissions or []:
        if (not isinstance(item, dict) or set(item) != {'path', 'reason'}
                or not isinstance(item['reason'], str) or not item['reason'].strip()):
            raise ValueError('packaging omissions need an exact path and nonempty reason')
        name = layout.safe_name(item['path'])
        if not name.startswith('work/') or any(c in name for c in '*?[]'):
            raise ValueError('packaging omissions need exact work/ paths, not globs')
        if name in declared and declared[name] != item['reason']:
            raise ValueError('conflicting packaging omission: ' + name)
        declared[name] = item['reason']
    def literal_path(node):
        # Interpret only literal string/path assembly; never execute source.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
            left, right = literal_path(node.left), literal_path(node.right)
            if isinstance(node.op, ast.Div) and left is None and right == 'work':
                return right  # ROOT / 'work' / ...
            if left is not None and right is not None:
                return left + ('/' if isinstance(node.op, ast.Div) else '') + right
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {'Path', 'PurePath', 'PurePosixPath'} and len(node.args) == 1):
            return literal_path(node.args[0])
        return None

    pattern = re.compile(r'(?<![\w./-])work/[A-Za-z0-9_./+-]+')
    references: dict[str, set[str]] = {}
    scanned = 0
    for name in names:
        if not name.startswith(('src/', 'tests/')) or Path(name).suffix in {'.png', '.svg'}:
            continue
        text = layout.checked_path(root, name).read_text(encoding='utf-8')
        scanned += 1
        fragments = [(text, 1)]
        if name.endswith('.py'):
            fragments.extend((value, node.lineno) for node in ast.walk(ast.parse(text, filename=name))
                             if (value := literal_path(node)) is not None)
        for fragment, line in fragments:
            paths = [(match.group().rstrip('.'), line + fragment.count(chr(10), 0, match.start()))
                     for match in pattern.finditer(fragment)]
            if fragment.startswith('work/') and '\n' not in fragment:
                paths.append((fragment, line))  # whole literal: spaces and Unicode are valid
            for path, at in paths:
                if path.endswith('/'):
                    continue  # directory/prefix literal, not a named file
                if path in declared or (root / path).is_file():
                    if path not in declared:
                        layout.checked_path(root, path)  # reject unsafe/symlink references only when they exist
                    references.setdefault(path, set()).add(f'{name}:{at}')
    packed = set(names) | {MANIFEST}
    missing = sorted(set(references) - packed - declared.keys())
    if missing:
        raise ValueError('packaging closure: unpackaged existing work file(s): ' + '; '.join(
            f'{name} referenced by {", ".join(sorted(references[name]))}' for name in missing))
    return {'status': 'PASS', 'sources_scanned': scanned, 'referenced_files': len(references),
            'omitted': [{'path': name, 'reason': declared[name], 'references': sorted(references[name])}
                        for name in sorted(set(references) - packed)],
            'scope': 'Concrete existing work-file references; not dynamic dependency discovery.'}


def _portable_record_errors(portable: dict, selection: dict) -> list[str]:
    """A quick/partial/empty result can never masquerade as the full contract."""
    if not isinstance(portable, dict):
        return ['portable result must be an object']
    errors = []
    if portable.get('schema') != 'farm-review-self-test-v1':
        errors.append('not a full portable self-test result')
    if not selection['files'] or [row['path'] for row in portable['tests']] != selection['files']:
        errors.append('portable test selection is empty, incomplete or different')
    if [row['name'] for row in portable['checks']] != [row['name'] for row in selection['checks']]:
        errors.append('advertised checks are incomplete or different')
    if portable['maintainer_only'] != selection['maintainer_only'] or portable['deselected'] != selection['deselected']:
        errors.append('portable exclusions differ from the tested ZIP')
    if portable['test_files'] != len(portable['tests']):
        errors.append('test-file accounting mismatch')
    rows = portable['checks'] + portable['tests']
    if any(type(row['returncode']) is not int for row in rows):
        errors.append('every completed command needs an integer return code')
    if portable.get('status') == 'PASS' and (portable['failed'] or portable['shipped_files_changed']
                                          or any(row['returncode'] for row in rows)):
        errors.append('PASS contradicts failed commands or changed shipped files')
    return errors


def _portable_record_counts(portable: dict | None, selection: dict) -> dict:
    portable = portable if isinstance(portable, dict) else {}
    test_rows = portable.get('tests')
    check_rows = portable.get('checks')
    tests = [row for row in (test_rows if isinstance(test_rows, list) else []) if isinstance(row, dict)
             and isinstance(row.get('path'), str) and type(row.get('returncode')) is int]
    results = {row['name']: row for row in (check_rows if isinstance(check_rows, list) else [])
               if isinstance(row, dict) and isinstance(row.get('name'), str)}
    return {'files_run': len(tests), 'files_passed': sum(row['returncode'] == 0 for row in tests),
            # Count the canonical file run once, not duplicated pytest checks.
            'tests_passed': sum(int(match.group(1)) for row in tests
                                for match in re.finditer(r'\b(\d+) passed\b', str(row.get('result', '')))),
            'maintainer_only_count': len(selection['maintainer_only']),
            'deselected_count': len(selection['deselected']),
            'checks': [{**check, **results.get(check['name'], {'returncode': None})}
                       for check in selection['checks']]}


def _record_seal(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != 'record_sha256'}
    return {**body, 'record_sha256': digest(review_intake.canonical(body))}


def _record_path(path: Path) -> Path:
    path = Path(path).absolute()
    if path.suffix != '.json' or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('self-test record needs a regular JSON path, not a symlink')
    return Path(os.path.abspath(path))  # normalize aliases before comparing run/cache destinations


def _run_published_suite(command: list[str], *, cwd: Path, env: dict, timeout: int):
    """Bound the runner and, on POSIX, its test subprocesses to one lifetime."""
    import signal
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=os.name == 'posix') as child:
        try:
            stdout, stderr = child.communicate(timeout=timeout)
        except BaseException:
            if os.name == 'posix':
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                child.kill()
            child.communicate()
            raise
        return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)


def published_self_test(archive: Path, record_path: Path, *, jobs: int = 0,
                        last_pass_path: Path | None = None, timeout: int = 5400) -> dict:
    """Test exactly a copied published ZIP with its own runner in a fresh tree.

    Full results (including failures) are retained. Only a complete PASS promotes
    the last-green cache. Receipts bind bytes and execution scope, not authenticity.
    This explicitly executes the supplied ZIP; --verify-zip remains non-executing.
    """
    import time
    from datetime import datetime, timezone
    if jobs < 0 or timeout <= 0:
        raise ValueError('nonnegative jobs and positive timeout required')
    target = _record_path(record_path)
    green = _record_path(last_pass_path) if last_pass_path is not None else None
    if green == target:
        raise ValueError('run record and last-PASS cache must have distinct paths')
    archive = Path(archive)
    if archive.is_symlink() or not archive.is_file() or archive.stat().st_size >= LIMIT:
        raise ValueError('bounded regular published ZIP required')
    if archive.resolve() in {target, green}:
        raise ValueError('record paths must not overwrite the tested ZIP')
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix='review-published-selftest-') as temporary:
        stage = Path(temporary)
        # A concurrently replaced outgoing ZIP must not change what gets tested.
        raw = archive.read_bytes()
        snapshot = stage / 'tested.zip'
        snapshot.write_bytes(raw)
        tested_zip = {'name': archive.name, 'sha256': digest(raw), 'bytes': len(raw)}
        del raw
        manifest = verify_archive(snapshot, stage / 'extracted')
        extracted = stage / 'extracted/farmNEAT'
        spec = tomllib.loads(layout.checked_path(extracted, CONTEXT_SPEC).read_text())
        files, excluded, deselected = portable_tests(extracted, spec)
        selection = {'files': files, 'checks': spec.get('checks', []),
                     'maintainer_only': excluded, 'deselected': deselected}
        value = {'schema': SELF_TEST_RECORD_SCHEMA, 'status': 'RUNNING',
                 'tested_zip': tested_zip,
                 'manifest_sha256': digest((extracted / MANIFEST).read_bytes()),
                 'members': {name: row for name, row in manifest['members'].items() if name != SELF_TEST_STATUS},
                 'started_at': started_at, 'finished_at': None,
                 'interpreter': {'executable': sys.executable, 'version': sys.version.split()[0]},
                 'selection': selection, 'portable': None, 'runner_returncode': None, 'errors': []}
        _atomic_json(target, _record_seal(value), replace=True)
        env = _portable_environment(extracted)
        # Never join candidate/pristine fixtures or modules outside the tested ZIP.
        for key in ('FARMNEAT_TEST_SOURCE_ROOT', 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS'):
            env.pop(key, None)
        env['PYTHONPATH'] = str(extracted / 'src')
        try:
            done = _run_published_suite([sys.executable, '-B', str(extracted / 'src/farm_neat/tooling/review_package.py'),
                                        '--self-test', '--jobs', str(jobs or min(8, os.cpu_count() or 1))],
                                       cwd=extracted, env=env, timeout=timeout)
            value['runner_returncode'] = done.returncode
            portable = review_intake.parse_json(done.stdout.encode())
            value['portable'] = portable
            value['errors'] = _portable_record_errors(portable, selection)
            if not isinstance(portable, dict):
                raise ValueError('portable result must be an object')
            if portable.get('interpreter') != value['interpreter']['version']:
                value['errors'].append('runner interpreter mismatch')
            changed = _manifest_drift(extracted)
            if changed:
                value['errors'].append('post-run shipped-file drift: ' + ', '.join(changed))
            if done.returncode == 0 and portable.get('status') == 'PASS' and not value['errors']:
                value['status'] = 'PASS'
            else:
                value['status'] = 'FAIL'
                value['output_tail'] = (done.stdout + done.stderr)[-8000:]
        except (ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            value['status'] = 'ERROR'
            value['errors'].append(f'{type(error).__name__}: {error}')
            progress = layout.checked_path(extracted, SELF_TEST_PROGRESS)
            if progress.is_file():
                try:
                    value['partial_progress'] = review_intake.parse_json(progress.read_bytes())
                except (ValueError, OSError) as progress_error:
                    value['errors'].append('unreadable partial progress: ' + str(progress_error))
        value.update(_portable_record_counts(value['portable'] or value.get('partial_progress'), selection))
        value['seconds'] = round(time.monotonic() - started, 3)
        value['finished_at'] = datetime.now(timezone.utc).isoformat()
        value = _record_seal(value)
        _atomic_json(target, value, replace=True)
        if value['status'] == 'PASS' and green is not None:
            _atomic_json(green, value, replace=True)
        return value


def self_test_status(root: Path, names: list[str], record_path: Path | None = None) -> dict:
    """Historical coverage only: even matching payloads are not this ZIP's PASS."""
    path = Path(record_path) if record_path is not None else layout.checked_path(root, SELF_TEST_LAST_PASS)
    status = {'schema': 'farm-review-self-test-status-v1', 'status': 'MISSING',
              'suite_for_this_zip': 'NOT_ATTESTED_BY_THIS_STATUS', 'last_pass': None,
              'changed_paths': [], 'other_changed_paths': [],
              'meaning': 'Last verified published ZIP, not a PASS for this ZIP. Captured before the current build gate; generated status and manifest are excluded from payload comparison.'}
    if not path.exists() and not path.is_symlink():
        return status
    try:
        path = _record_path(path)
        if path.stat().st_size > 4_000_000:
            raise ValueError('self-test record exceeds 4 MB')
        raw = path.read_bytes()
        record = review_intake.parse_json(raw)
        if (record.get('schema') != SELF_TEST_RECORD_SCHEMA or record.get('status') != 'PASS'
                or record != _record_seal(record) or record['runner_returncode'] != 0
                or record['errors'] or _portable_record_errors(record['portable'], record['selection'])):
            raise ValueError('not an intact complete last-PASS record')
        if any(record.get(key) != value for key, value in _portable_record_counts(record['portable'], record['selection']).items()):
            raise ValueError('record counts/checks contradict portable result')
        if record['portable']['status'] != 'PASS':
            raise ValueError('last-PASS record did not pass')
        old = record['members']
        for name, row in old.items():
            evidence.binding({'path': name, **row})
        current = {name: {'sha256': digest(layout.checked_path(root, name).read_bytes()),
                          'bytes': layout.checked_path(root, name).stat().st_size}
                   for name in names if name not in {MANIFEST, SELF_TEST_STATUS}}
        changed = sorted(name for name in current.keys() | old.keys() if current.get(name) != old.get(name))
        status['status'] = 'STALE' if changed else 'UNCHANGED_SINCE_PASS'
        status['changed_paths'] = [name for name in changed if name.startswith(('src/', 'tests/'))]
        status['other_changed_paths'] = [name for name in changed if not name.startswith(('src/', 'tests/'))]
        status['last_pass'] = {key: record[key] for key in (
            'tested_zip', 'manifest_sha256', 'started_at', 'finished_at', 'interpreter',
            'files_run', 'files_passed', 'tests_passed', 'maintainer_only_count', 'deselected_count', 'checks')}
        status['last_pass']['record'] = {'path': str(path), 'sha256': digest(raw), 'bytes': len(raw)}
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
        status.update(status='INVALID', error=str(error))
    return status


def full_build(root: Path = ROOT, *, last_self_test: Path | None = None) -> dict:
    """Run the explicit strict audit lane and publish only after every gate passes."""
    spec = layout.read_layout(root)
    errors = layout.check(root, spec)
    if errors:
        raise ValueError("\n".join(errors))
    layout.checked_path(root, THEOREMS).write_text(theorem_index(root, tomllib.loads(layout.checked_path(root, CONTEXT_SPEC).read_text())))
    names = inventory(root)
    review_spec = tomllib.loads(layout.checked_path(root, CONTEXT_SPEC).read_text())
    closure = packaging_closure(root, names, omissions=review_spec.get('packaging_omissions', []))
    coverage = self_test_status(root, names, last_self_test)
    generated = {SELF_TEST_STATUS: review_intake.canonical(coverage) + b"\n"}
    context = review_context(root, names)
    total = sum((root / n).stat().st_size for n in names) + sum(map(len, generated.values()))
    # The writer counts actual metadata, preserving the strict contents/ZIP limits
    # without substituting a fixed five-megabyte estimate for the manifest.
    if total >= LIMIT:
        raise ValueError("selected evidence exceeds the bounded source-review package; review the allowlist")
    # Worst case: compressed staging + clean extraction, with metadata headroom.
    reserve = 2 * total + 20_000_000
    admission = storage_layout.bindings().status(
        root, reserve_bytes=reserve,
        scan_timeout=BUILD_STORAGE_SCAN_TIMEOUT,
    )
    if not admission["admissible"]:
        raise ValueError(f"storage admission refused: {admission['status']}")
    scratch = layout.checked_path(root, "work/scratch")
    with tempfile.TemporaryDirectory(prefix="review-package-", dir=scratch) as temporary:
        stage = Path(temporary)
        archive = stage / "package.zip"
        manifest = write_archive(root, names, archive, admission, context, generated=generated)
        verify_archive(archive, stage / "extracted")
        extracted = stage / "extracted/farmNEAT"
        layout.create_homes(extracted, layout.read_layout(extracted))
        for script, args in (("layout.py", ["--check"]), ("reachability.py", []),
                             ("review_package.py", ["--context"]), ("search_ledger.py", ["verify"])):
            done = subprocess.run([sys.executable, "-B", str(extracted / "src/farm_neat/tooling" / script), *args],
                                  cwd=extracted, capture_output=True, text=True, timeout=30)
            if done.returncode:
                raise ValueError(f"extracted {script} failed:\n{done.stdout}{done.stderr}")
        # Run without site packages: the shipped game must not rely on a local
        # installation, weights, or an opponent asset omitted from the ZIP.
        # Verify dependencies and editable example inputs without advancing a game.
        game_check = """
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from farm_neat.evaluation.eval_vs import play_program
from farm_neat.game.exact_engine import assert_bundled_engine_identity
from farm_neat.economy.action_space import validate_program
from farm_neat.evaluation.shop_router import ShopRouterAdapter
assert_bundled_engine_identity()
opponent = ShopRouterAdapter()
hall = json.loads(Path('work/datasets/corpus/teachers-v5-hall.json').read_text())
assert hall, 'bundled editable examples required'
for row in hall:
    program = row['program']
    if not program or program[-1]['op'] != 'STOP':
        program = [*program, {'op': 'STOP'}]
    assert len(program) <= 64, 'example exceeds advertised command budget'
    validate_program(program)
"""
        done = subprocess.run([sys.executable, "-B", "-S", "-c", game_check],
                              cwd=extracted, capture_output=True, text=True, timeout=30)
        if done.returncode:
            raise ValueError(f"extracted playable game dependencies failed:\n{done.stdout}{done.stderr}")
        if not layout.checked_path(root, DISPATCH).is_file():
            raise ValueError("a published packet needs " + DISPATCH)
        # Check the published inputs, never the mutable producer tree (which has
        # no extracted manifest). The full portable suite is opt-in below.
        readiness = """
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from farm_neat.tooling import review_package as package
print(json.dumps(package.dispatch_readiness(Path.cwd())))
"""
        done = subprocess.run([sys.executable, "-B", "-c", readiness], cwd=extracted,
                              capture_output=True, text=True, timeout=1200)
        if done.returncode:
            raise ValueError(f"extracted dispatch readiness failed:\n{done.stdout[-4000:]}{done.stderr[-4000:]}")
        dispatch_view = json.loads(done.stdout.splitlines()[-1])
        done = subprocess.run([sys.executable, "-B", str(extracted / "src/farm_neat/tooling/review_package.py"),
                               "--self-test", "--jobs", str(min(16, os.cpu_count() or 1))],
                              cwd=extracted, capture_output=True, text=True, timeout=5400)
        portable = json.loads(done.stdout) if done.stdout.startswith("{") else None
        if done.returncode or portable is None or portable["status"] != "PASS":
            failed = portable and {"failed": portable["failed"], "changed": portable["shipped_files_changed"],
                                   "tails": [t.get("output_tail") for t in portable["checks"] + portable["tests"] if t["returncode"]]}
            raise ValueError(f"extracted self-test failed: {json.dumps(failed)[-8000:] if failed else done.stderr[-4000:]}")
        if inventory(root) != names:
            raise ValueError("package membership changed during build; rebuild")
        if evidence.packet_view(root) != manifest['evidence']:
            raise ValueError('evidence classification/provenance changed during build; rebuild')
        unchanged(root, manifest, generated=generated)
        target = layout.checked_path(root, OUTPUT)
        size = archive.stat().st_size
        publication = publish_dispatch_bundle(root, archive, dispatch_view["slots"])
        checksum, prompts = publication["sha256"], publication["prompts"]
    return {"file": str(target), "bytes": size, "sha256": checksum,
            "members": len(manifest["members"]) + 1, "mode": manifest["mode"], "verification": "PASS",
            "packaging_closure": closure, "self_test_status": coverage,
            "storage_admission": admission,
            "dispatch": dispatch_view["slots"], "prompts": prompts,
            "self_test": {key: portable[key] for key in ("status", "seconds", "workers", "test_files", "interpreter")}
                         | {"checks": len(portable["checks"]), "maintainer_only": len(portable["maintainer_only"]),
                            "deselected": len(portable["deselected"])},
            "evidence": manifest["evidence"],
            "uncompressed_bytes": total + len((json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n").encode()),
            "members_by_tree": {tree: sum(1 for n in [*manifest["members"], MANIFEST] if n.split("/")[0] == tree)
                                for tree in sorted({n.split("/")[0] for n in names if "/" in n})}}


def context_summary(context: dict) -> dict:
    """Navigation-only projection after full validation; never a second authority."""
    reading = context["reading"]
    return {
        "schema": "farm-review-context-summary-v1",
        "status": "VALIDATED_CONTEXT",
        "meaning": "Navigation only. Read the saved assignment and its Q-thread before work; "
                   "this summary neither replaces their requirements nor replays historical evidence.",
        "full_result_sha256": digest(review_intake.canonical(context)),
        "mode": context["mode"],
        "entrypoint": context["entrypoint"],
        "read_first": [item["path"] for item in context["read_first"]],
        "reading": {key: reading.get(key) for key in
                    ("identity_sha256", "total_bytes", "total_words", "budget_bytes", "budget_words")},
        "reading_files": len(reading["files"]),
        "assignments": [{key: item.get(key) for key in ("id", "topic", "max_games", "cpu_seconds")}
                        for item in context.get("assignments", [])],
        "notes": [{key: item[key] for key in ("id", "path", "status")}
                  for kind in ("threads", "tasks") for item in context[kind]],
        # Preserve every registered delivery and blocker, not just a favourable count.
        "deliveries": context.get("deliveries", []),
        "required_files": len(context["required_files"]),
        "checks": len(context["checks"]),
        "limitations": context["limitations"],
        "commands_executed": context["commands_executed"],
        "details": "python3 -B src/farm_neat/tooling/review_package.py --context",
        "handoff": "python3 -B src/farm_neat/tooling/review_package.py --handoff",
    }


def _execution_policy(chosen: dict) -> dict:
    """Render anti-sprawl semantics and an expected checkpoint without changing acceptance."""
    lane = chosen.get("schedule", {}).get("lane", "THEOREM")
    long = chosen.get("cpu_seconds", 0) >= 1800 or chosen.get("max_games", 0) > 0 or len(chosen.get("acceptance", [])) >= 4
    milestone = {
        "BUILD": "Reproduce the named blocker and preserve the first focused failing test or first independently reviewable implementation slice.",
        "THEOREM": "Freeze the exact control/domain and preserve the first decisive witness, counterexample, bound, or clean negative result.",
        "MAINTAINER": "Finish preflight and preserve the exact runnable command plus bound inputs before the expensive run.",
    }.get(lane, "Preserve the first durable intermediate result and exact next command.")
    return {
        "budget_semantics": {
            "soft_target_games": chosen.get("max_games"), "starts_recorded": 0,
            "above_target_by": 0, "may_continue": True, "cpu_seconds": chosen.get("cpu_seconds"),
            "meaning": "These are anti-sprawl planning targets, not intake validity gates. Going over does not invalidate a return or forbid continuation inside the same task. Historical starts remain recorded when a later ticket raises the soft target. Do not broaden the question; record the overage and why the next action is still the cheapest decisive evidence.",
            "hard_limits": "Explicit avoid/prohibited-shortcut rules, safety/storage integrity checks, and task write scope remain hard."
        },
        "expected_checkpoint": {
            "recommended_for_long_task": long, "return_type": "PARTIAL", "milestone": milestone,
            "turn_in_when": "Save usable partial work when stopping or handing off. Continue implementation while it is productive; the first test or milestone is not a mandatory return gate.",
            "fields": ["done", "exact_state", "remaining", "blockers", "next_command", "do_not_redo", "checkpoint"],
            "intake_action": "Preserve usable work and the exact next command. Continue the same task when useful; do not require a fresh audit, perfect form or restart from scratch."
        }
    }


def assignment_context(root: Path, *, require_deliveries: bool = False) -> dict:
    """Project only the saved source-bound task; unrelated packet state is not a startup gate."""
    ticket = saved_assignment(root)
    _, manifest, _, _ = _pool(root)
    chosen = ticket["assignment"]
    instructions = manifest.get("members", {}).get("review/INSTRUCTIONS.md")
    reading = ([{"path": "review/INSTRUCTIONS.md", **instructions}] if isinstance(instructions, dict) else [])
    continuation = continuation_state(root, ticket)
    return {
        "schema": "farm-review-assignment-context-v1",
        "status": "SOURCE_BOUND_SELECTED_ASSIGNMENT_CONTEXT",
        "ticket": {key: ticket[key] for key in
                   ("ticket", "selection_method", "slot", "task_id", "source_manifest_sha256",
                    "pool_sha256", "dispatch_sha256", "return_name") if key in ticket},
        "ticket_path": ASSIGNMENT,
        "assignment": {key: chosen[key] for key in
                       ("id", "topic", "title", "operation", "question", "max_games",
                        "cpu_seconds", "acceptance", "avoid", "allowed_changes")},
        "execution_policy": _execution_policy(chosen),
        "stages": [
            {"stage": "READ", "files": reading,
             "purpose": "Read the shipped reviewer instructions and this selected ticket."},
            {"stage": "EVIDENCE", "parents": [
                {key: p[key] for key in ("key", "title", "path", "sha256", "bytes")}
                for p in chosen.get("parent_claims", [])],
             "purpose": "Open only the selected task's direct parent records as needed."},
            {"stage": "CONTINUE", **continuation,
             "purpose": "Resume this task-keyed saved workspace before IMPLEMENT."},
            {"stage": "IMPLEMENT", "inputs": chosen.get("inputs", []),
             "input_files": chosen.get("input_files", []),
             "purpose": "Use the selected task's source-bound inputs and preserve its change limits."},
            {"stage": "CHECK_AND_RETURN", "checks": "FOCUSED_CHANGE_ONLY",
             "purpose": "Run checks relevant to the selected change; no global suite is a startup or return gate."},
        ],
        "burden": {
            "active_assignments": 1,
            "parent_claims": len(chosen.get("parent_claims", [])),
            "input_paths": len(chosen.get("inputs", [])),
            "requirement_items": sum(len(chosen.get(key, [])) for key in
                                     ("acceptance", "avoid", "allowed_changes")),
        },
        "reference_commands": {"work_map": "python3 -B src/farm_neat/tooling/review_package.py --work-map --summary"},
        "limitations": [
            "Only the selected source-bound task is projected; unrelated assignments, deliveries and acceptance receipts are not validated.",
            "Direct-parent metadata is not a transitive evidence closure.",
        ],
        "commands_executed": 0,
    }


def _assignment_context(root: Path, ticket: dict, context: dict) -> dict:
    """Existing projection/equality owner; callers must first validate globally."""
    chosen = ticket["assignment"]
    matches = [item for item in context["assignments"] if item["id"] == chosen["id"]]
    if matches != [chosen]:
        raise ValueError("saved assignment differs from the validated context")
    threads = [item for item in context["threads"] if item["id"] == chosen["topic"]]
    if len(threads) != 1:
        raise ValueError("saved assignment needs exactly one owning thread")
    paths = {context["entrypoint"], *[r["path"] for r in context["read_first"]], threads[0]["path"]}
    reading = [r for r in context["reading"]["files"] if r["path"] in paths]
    if {r["path"] for r in reading} != paths:
        raise ValueError("scoped reading is not in the validated reading set")
    tool = "python3 -B src/farm_neat/tooling/review_package.py"
    continuation = continuation_state(root, ticket)
    return {
        "schema": "farm-review-assignment-context-v1",
        "status": "VALIDATED_SAVED_ASSIGNMENT_CONTEXT",
        "ticket": {key: ticket[key] for key in
                   ("ticket", "selection_method", "slot", "task_id", "source_manifest_sha256",
                    "pool_sha256", "dispatch_sha256", "return_name") if key in ticket},
        "ticket_path": ASSIGNMENT,
        "full_context_sha256": digest(review_intake.canonical(context)),
        # Requirements are copied exactly, not summarized into a weaker gate.
        "assignment": {key: chosen[key] for key in
                       ("id", "topic", "title", "operation", "question", "max_games",
                        "cpu_seconds", "acceptance", "avoid", "allowed_changes")},
        "execution_policy": _execution_policy(chosen),
        "stages": [
            {"stage": "READ", "files": reading,
             "purpose": "Current brief, shared question index and assigned Q-thread; read these files intact."},
            {"stage": "EVIDENCE", "parents": [
                {key: p[key] for key in ("key", "title", "path", "sha256", "bytes")}
                for p in chosen["parent_claims"]],
             "command": tool + " --evidence-for-assignment " + chosen["id"],
             "purpose": "Open intact direct-parent receipts, then follow linked counterevidence. Not graph closure."},
            {"stage": "CONTINUE", **continuation,
             "purpose": "Resume this task-keyed saved workspace before IMPLEMENT; slot/ticket changes do not reset it or its native-start charges."},
            {"stage": "IMPLEMENT", "inputs": chosen["inputs"],
             "purpose": "Open inputs as needed; exact hashes remain in the saved ticket. Preserve its change limits."},
            {"stage": "CHECK_AND_RETURN", "checks": context["checks"],
             "purpose": "Read review-release.md for pristine/candidate checks; do not rebind the original manifest."},
        ],
        # Topic filtering must not conceal a waiting delivery in another thread.
        "deliveries": context["deliveries"],
        "burden": {
            "meaning": "Counts of displayed structure, not measured human memory, comprehension or time.",
            "active_assignments": 1, "active_questions": 1,
            "parent_claims": len(chosen["parent_claims"]),
            "parent_aliases": len({p["key"] for p in chosen["parent_claims"]}),
            "input_paths": len(chosen["inputs"]),
            "requirement_items": sum(len(chosen[k]) for k in ("acceptance", "avoid", "allowed_changes")),
            "initial_reading_files": len(reading),
            "initial_reading_bytes": sum(r["bytes"] for r in reading),
            "initial_reading_words": sum(r["words"] for r in reading),
            "full_reading_files": len(context["reading"]["files"]),
            "full_reading_bytes": context["reading"]["total_bytes"],
            "other_assignments_deferred": len(context["assignments"]) - 1,
        },
        "reference_commands": {"work_map": tool + " --work-map --summary",
                               "all_context": tool + " --context",
                               "all_notes": tool + " --handoff",
                               "all_requests": "python3 -B src/farm_neat/tooling/search_ledger.py queue"},
        "limitations": context["limitations"] + [
            "Only presentation is scoped. The full context and all available packet dependencies were validated.",
            "Other questions and queue items are reference, not extra assignments or proven irrelevant.",
            "Direct-parent receipts may omit related contradictions; follow their links and the assigned thread.",
            "Omitted producer evidence remains historical attestation, not a raw replay or playing-strength result.",
        ],
        "commands_executed": 0,
    }


def _work_path_record(root: Path, name: str) -> dict:
    """Describe one repository-relative path without importing or executing it."""
    target = layout.checked_path(root, name)
    if not target.exists():
        return {"path": name, "status": "MISSING"}
    if target.is_file():
        payload = target.read_bytes()
        return {"path": name, "status": "FILE", "bytes": len(payload), "sha256": digest(payload)}
    if target.is_dir():
        return {"path": name, "status": "DIRECTORY"}
    return {"path": name, "status": "OTHER"}


def manifest_file_preflight(root: Path, manifest_name: str) -> list[dict]:
    """Check a conventional hash-bound top-level ``files`` manifest, if present.

    This is intentionally narrow.  A JSON input is treated as a file manifest only
    when every ``files`` value is a mapping with a full sha256.  Relative members
    are resolved beside the manifest.  The check never invents dependencies from
    arbitrary strings elsewhere in the JSON.
    """
    manifest = layout.checked_path(root, manifest_name)
    if not manifest.is_file() or manifest.suffix != ".json":
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict) or not files:
        return []
    for name, meta in files.items():
        if (not isinstance(name, str) or not isinstance(meta, dict)
                or not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("sha256", "")))):
            return []
        relative = PurePosixPath(name)
        if (relative.is_absolute() or "\\" in name
                or any(part in ("", ".", "..") for part in relative.parts)):
            return []
        if "bytes" in meta and (type(meta["bytes"]) is not int or meta["bytes"] < 0):
            return []

    parent = PurePosixPath(manifest_name).parent
    rows = []
    for name, meta in sorted(files.items()):
        joined = (parent / PurePosixPath(name)).as_posix()
        target = layout.checked_path(root, joined)
        record = {"path": joined, "expected_sha256": meta["sha256"]}
        if "bytes" in meta:
            record["expected_bytes"] = meta["bytes"]
        if not target.exists():
            rows.append({**record, "status": "MISSING"})
            continue
        if not target.is_file():
            rows.append({**record, "status": "NOT_FILE"})
            continue
        data = target.read_bytes()
        actual_sha = digest(data)
        actual_bytes = len(data)
        status = "MATCH"
        if actual_sha != meta["sha256"]:
            status = "HASH_MISMATCH"
        elif "bytes" in meta and actual_bytes != meta["bytes"]:
            status = "SIZE_MISMATCH"
        rows.append({**record, "status": status, "actual_sha256": actual_sha,
                     "actual_bytes": actual_bytes})
    return rows


def _assignment_authority_line(root: Path, task_id: str) -> int | None:
    path = layout.checked_path(root, CONTEXT_SPEC)
    pattern = re.compile(r'^\s*id\s*=\s*"' + re.escape(task_id) + r'"\s*$')
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if pattern.match(line):
            return number
    return None


def _focused_assignment_tests(root: Path, inputs: list[str], writes: list[str]) -> list[dict]:
    """Small test hint set: declared test writes plus basename-conventional siblings."""
    found: dict[str, str] = {}
    for name in writes:
        if name.startswith("tests/") and name.endswith(".py"):
            found[name] = "declared_write"
    stems = {PurePosixPath(name).stem for name in [*inputs, *writes]
             if name.startswith("src/") and name.endswith(".py")}
    tests_root = layout.checked_path(root, "tests")
    if tests_root.is_dir():
        for stem in sorted(stems):
            for path in sorted(tests_root.rglob(f"test_{stem}*.py")):
                if path.is_file() and not path.is_symlink():
                    found.setdefault(path.relative_to(root).as_posix(), "source_stem")
    rows = []
    for name in sorted(found):
        row = _work_path_record(root, name)
        rows.append({**row, "discovery": found[name]})
    return rows


def assignment_work_map(root: Path, *, require_deliveries: bool = False) -> dict:
    """One-command orientation for a saved assignment; navigation, never authority.

    It surfaces the schedule fields that assignment-context intentionally does not
    duplicate (especially write ownership), hashes explicit inputs, names likely
    focused tests, and preflights conventional hash-bound input manifests.  No
    research code, tests, games or reproduction commands are executed.
    """
    ticket = saved_assignment(root)
    _, _, pool, _ = _pool(root)
    chosen = next((item for item in pool if item["id"] == ticket["assignment"]["id"]), None)
    if chosen != ticket["assignment"]:
        raise ValueError("saved assignment differs from the source-bound assignment pool")
    # A navigation view must remain usable after the reviewer edits source.
    # Dispatch readiness belongs to the maintainer, not this read-only command.
    task = {"id": chosen["id"], "title": chosen["title"], "track": "LEARNER",
            "lane": "THEOREM", "wave": 0, "owner": "reviewer", "state": "OPEN",
            "writes": [], "depends_on": [], **chosen.get("schedule", {}),
            "status": "DISPATCHED"}
    writes = [_work_path_record(root, name) for name in task["writes"]]
    inputs = [_work_path_record(root, name) for name in chosen["inputs"]]
    focused_tests = _focused_assignment_tests(root, chosen["inputs"], task["writes"])

    manifests = []
    transitive_problems = []
    for name in chosen["inputs"]:
        members = manifest_file_preflight(root, name)
        if not members:
            continue
        manifests.append({"manifest": name, "files": members})
        transitive_problems.extend({"manifest": name, **row}
                                   for row in members if row["status"] != "MATCH")

    dependencies = []
    for requirement in task["depends_on"]:
        dep = _dependency_requirement(requirement)
        row = next((item for item in pool if item.get("id") == dep["task"]), {})
        dependencies.append({"id": dep["task"], "title": row.get("title", dep["task"]),
                             "state": row.get("schedule", {}).get("state", "UNKNOWN"),
                             "status": "NOT_REVALIDATED", "required_outcome": dep["outcome"]})
    test_paths = [row["path"] for row in focused_tests]
    tool = "python3 -B src/farm_neat/tooling/review_package.py"
    runtime = "python3 -B src/farm_neat/tooling/runtime.py"
    return {
        "schema": "farm-review-assignment-work-map-v1",
        "status": "VALIDATED_ASSIGNMENT_WORK_MAP",
        "meaning": "Navigation only. project/review.toml and the saved assignment remain authoritative.",
        "task": {
            "id": task["id"], "title": task["title"], "topic": chosen["topic"],
            "operation": chosen["operation"], "track": task["track"], "lane": task["lane"],
            "wave": task["wave"], "owner": task["owner"], "state": task["state"],
            "status": task["status"], "writes": task["writes"],
            "soft_target_games": chosen["max_games"], "cpu_seconds": chosen["cpu_seconds"],
            "authority": CONTEXT_SPEC, "authority_line": _assignment_authority_line(root, task["id"]),
        },
        "writes": writes,
        "inputs": inputs,
        "missing_inputs": [row["path"] for row in inputs if row["status"] == "MISSING"],
        "input_manifests": manifests,
        "transitive_input_problems": transitive_problems,
        "dependencies": dependencies,
        "focused_tests": focused_tests,
        "commands": {
            "context": tool + " --assignment-context",
            "evidence": tool + " --evidence-for-assignment " + chosen["id"],
            "focused_tests": (runtime + " pytest -- " + " ".join(test_paths) + " -q") if test_paths else None,
            "layout_check": "python3 -B src/farm_neat/tooling/layout.py --check",
            "self_test": tool + " --self-test",
        },
        "limitations": [
            "Focused tests are basename-conventional hints plus declared test write paths, not a complete dependency graph.",
            "Transitive input preflight recognizes only top-level JSON files maps whose members carry full sha256 values.",
            "This command executes no tests, games, imports or reproduction commands and cannot establish scientific sufficiency.",
        ],
        "commands_executed": 0,
    }


def render_work_map(view: dict) -> str:
    task = view["task"]
    where = f"{task['authority']}:{task['authority_line']}" if task["authority_line"] else task["authority"]
    lines = [f"Task {task['id']} [{task['track']} {task['lane']} wave {task['wave']}; {task['status']}] {where}"]
    for row in view["writes"]:
        lines.append(f"WRITE {row['status']:9} {row['path']}")
    for row in view["inputs"]:
        suffix = f" sha256={row['sha256'][:12]}" if "sha256" in row else ""
        lines.append(f"INPUT {row['status']:9} {row['path']}{suffix}")
    for row in view["transitive_input_problems"]:
        lines.append(f"MANIFEST {row['status']:13} {row['path']} (from {row['manifest']})")
    for row in view["focused_tests"]:
        lines.append(f"TEST  {row['status']:9} {row['path']} [{row['discovery']}]")
    if view["commands"]["focused_tests"]:
        lines.append("CHECK " + view["commands"]["focused_tests"])
    lines.append("EVIDENCE " + view["commands"]["evidence"])
    return "\n".join(lines) + "\n"


def evidence_summary(view: dict) -> dict:
    """Summarize validated availability, never promote attestations to raw replay."""
    return {
        "schema": "farm-review-evidence-summary-v1",
        "status": "VALIDATED_AVAILABLE_EVIDENCE",
        "meaning": "No scientific replay. On a thin extraction, omitted producer evidence is a "
                   "historical attestation, not locally verified raw bytes.",
        "scope": view["scope"],
        "full_result_sha256": digest(review_intake.canonical(view)),
        "catalog": view["catalog"],
        "semantic_receipts": view["semantic_receipts"],
        "retained_files": len(view["retained"]),
        "retained_bytes": sum(item["bytes"] for item in view["retained"]),
        "witness_count": view["witness_count"],
        "witness_bytes": view["witness_bytes"],
        "unclassified": view["unclassified"],
        "omitted_count": view["omitted_count"],
        "omitted_bytes": view["omitted_bytes"],
        "omitted_set_sha256": view["omitted_set_sha256"],
        "commands_executed": 0,
        "details": "python3 -B src/farm_neat/tooling/review_package.py --evidence-check",
    }


def portable_tests(root: Path, spec: dict) -> tuple[list[str], list[dict], list[dict]]:
    """Every test is portable unless [self_test] names its file or its test with a reason.

    Exclusions live here, not in test files: ledger records pin test bytes. A named test
    must exist, so a stale exclusion cannot hide a renamed test.
    """
    table = spec.get("self_test")
    if (not isinstance(table, dict) or set(table) != {"maintainer_only", "deselect"}
            or not all(isinstance(table[key], list) for key in table)):
        raise ValueError("[self_test] needs maintainer_only and deselect lists")
    files = sorted(p.relative_to(root).as_posix() for p in layout.checked_path(root, "tests").rglob("test_*.py"))
    for key, field in (("maintainer_only", "path"), ("deselect", "test")):
        for item in table[key]:
            if (not isinstance(item, dict) or set(item) != {field, "reason"} or not isinstance(item[field], str)
                    or not isinstance(item["reason"], str) or not item["reason"].strip()):
                raise ValueError(f"{key} entries need {field} and a reason: {item!r}")
        if len({item[field] for item in table[key]}) != len(table[key]):
            raise ValueError(f"duplicate {key} entry")
    excluded = {item["path"] for item in table["maintainer_only"]}
    if excluded - set(files):
        raise ValueError("maintainer_only entries need an existing test path: " + ", ".join(sorted(excluded - set(files))))
    for item in table["deselect"]:
        path, _, name = item["test"].partition("::")
        if (path not in files or path in excluded or not re.fullmatch(r"test_\w+", name)
                or not re.search(rf"^def {name}\(", layout.checked_path(root, path).read_text(), re.M)):
            raise ValueError("deselect entries need path::test_name of an existing portable test: " + item["test"])
    return [name for name in files if name not in excluded], table["maintainer_only"], table["deselect"]


def _portable_environment(root: Path) -> dict[str, str]:
    """CPU-only environment shared by full and quick review-package verification."""
    inherited = [str(Path(entry).resolve()) for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                 if entry and not (Path(entry) / "farm_neat").is_dir()]
    return {**os.environ, "PYTHONPATH": os.pathsep.join(["src", *inherited]), "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}


def _run_checks(root: Path, spec: dict, env: dict[str, str], timeout: int) -> list[dict]:
    import shlex
    import time
    python = shlex.quote(sys.executable)
    rows = []
    for check in spec.get("checks", []):
        command = re.sub(r"(?<![\w/.-])python3(?= )", python, check["command"])
        command = re.sub(r"\bPYTHONPATH=(\S+) ", r'PYTHONPATH="\1:$PYTHONPATH" ', command)
        began = time.monotonic()
        done = subprocess.run(command, shell=True, cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
        rows.append({"name": check["name"], "returncode": done.returncode,
                     "seconds": round(time.monotonic() - began, 1),
                     **({"output_tail": (done.stdout + done.stderr)[-2000:]} if done.returncode else {})})
    return rows


def _run_test_files(root: Path, files: list[str], deselected: list[dict], env: dict[str, str],
                    *, jobs: int, timeout: int, on_progress=None) -> list[dict]:
    import time
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    def run(name):
        began = time.monotonic()
        skip = [arg for item in deselected if item["test"].startswith(name + "::")
                for arg in ("--deselect", item["test"])]
        done = subprocess.run([sys.executable, "-B", "-m", "pytest", name, "-q", "-p", "no:cacheprovider", "-rs", *skip],
                              cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
        lines = [line for line in done.stdout.splitlines() if line.strip()]
        return {"path": name, "returncode": done.returncode, "seconds": round(time.monotonic() - began, 1),
                "result": lines[-1] if lines else "",
                "skipped": [line for line in lines if line.startswith("SKIPPED")],
                **({"output_tail": (done.stdout + done.stderr)[-3000:]} if done.returncode else {})}

    workers = jobs or min(8, os.cpu_count() or 1)
    if workers < 1:
        raise ValueError("self-test workers must be positive")
    completed: dict[str, dict] = {}
    next_index = 0
    running = {}

    def ordered_completed() -> list[dict]:
        return [completed[name] for name in files if name in completed]

    def submit_available(pool) -> None:
        nonlocal next_index
        while len(running) < workers and next_index < len(files):
            name = files[next_index]
            next_index += 1
            running[pool.submit(run, name)] = name

    with ThreadPoolExecutor(max_workers=workers) as pool:
        submit_available(pool)
        while running:
            ready, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
            for future in sorted(ready, key=lambda item: files.index(running[item])):
                name = running.pop(future)
                completed[name] = future.result()
                if on_progress is not None:
                    on_progress(ordered_completed(),
                                [candidate for candidate in files if candidate in running.values()],
                                files[next_index:])
                submit_available(pool)
    return ordered_completed()


def _manifest_drift(root: Path) -> list[str]:
    path = layout.checked_path(root, MANIFEST)
    if not path.is_file():
        return []
    manifest = json.loads(path.read_bytes())
    changed = []
    for name, record in manifest["members"].items():
        candidate = layout.checked_path(root, name)
        if not candidate.is_file() or digest(candidate.read_bytes()) != record["sha256"]:
            changed.append(name)
    return changed


def quick_self_test(root: Path, *, jobs: int = 0, timeout: int = 120) -> dict:
    """Optional control-plane smoke test; reviewers run checks affected by their task."""
    import time
    started = time.monotonic()
    spec = tomllib.loads(layout.checked_path(root, CONTEXT_SPEC).read_text())
    portable, excluded, deselected = portable_tests(root, spec)
    selected = [name for name in QUICK_TEST_FILES if name in portable]
    if not selected:
        raise ValueError("quick self-test selected no portable test files")
    env = _portable_environment(root)
    checks = _run_checks(root, spec, env, timeout)
    tests = _run_test_files(root, selected, deselected, env, jobs=jobs, timeout=timeout)
    changed = _manifest_drift(root)
    failed = [c["name"] for c in checks if c["returncode"]] + [t["path"] for t in tests if t["returncode"]]
    return {"schema": "farm-review-quick-self-test-v1", "status": "FAIL" if failed or changed else "PASS",
            "meaning": "Fast control-plane checks only; not scientific replay and not a replacement for --self-test.",
            "interpreter": sys.version.split()[0], "workers": jobs or min(8, os.cpu_count() or 1),
            "seconds": round(time.monotonic() - started, 1), "checks": checks,
            "test_files": len(tests), "failed": failed, "shipped_files_changed": changed,
            "tests": tests, "maintainer_only": excluded, "deselected": deselected,
            "commands_executed": len(checks) + len(tests)}


def package_budget(root: Path) -> dict:
    """Read-only file/byte pressure report for the current tree, with work hot spots."""
    rows = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        rows.append((path.relative_to(root).as_posix(), size))
    top = {}
    for name, size in rows:
        key = name.split("/", 1)[0]
        bucket = top.setdefault(key, {"files": 0, "bytes": 0})
        bucket["files"] += 1; bucket["bytes"] += size
    groups = {}
    for name, size in rows:
        parts = name.split("/")
        if len(parts) < 3 or parts[0] != "work" or parts[1] != "runs":
            continue
        depth = 4 if len(parts) >= 4 and parts[2] == "reviewer-play" else 3
        key = "/".join(parts[:depth])
        bucket = groups.setdefault(key, {"path": key, "files": 0, "bytes": 0})
        bucket["files"] += 1; bucket["bytes"] += size
    largest = sorted(groups.values(), key=lambda r: (-r["bytes"], -r["files"], r["path"]))[:20]
    return {"schema": "farm-review-package-budget-v1", "files": len(rows),
            "bytes": sum(size for _, size in rows), "member_limit": MEMBER_LIMIT,
            "top_level": dict(sorted(top.items())), "largest_work_trees": largest,
            "commands_executed": 0}


def _stale_return_signals(root: Path, view: dict) -> list[dict]:
    """Conservative advisory only: matching return artifacts never auto-accept or block work."""
    base = layout.checked_path(root, "work/runs/reviewer-play")
    if not base.is_dir():
        return []
    rows = []
    for task in view["tasks"]:
        if task["state"] in {"ACCEPTED", "PARKED"}:
            continue
        slug = task["id"].lower()
        artifacts = []
        for child in base.iterdir():
            if not child.is_dir() or not child.name.lower().startswith(slug):
                continue
            for path in child.rglob("*"):
                if path.is_file() and ("return" in path.parts or path.name in {"RESULT.md", "ANSWER_COUNTS.md"}):
                    artifacts.append(path.relative_to(root).as_posix())
        if artifacts:
            rows.append({"task": task["id"], "status": task["status"], "artifacts": sorted(set(artifacts)),
                         "meaning": "ADVISORY_RETURN_SIGNAL_ONLY; maintainer intake still decides ACCEPT/REJECT/RERUN"})
    return rows


def _partial_continuation_tasks(root: Path, actionable_tasks: set[str] | None = None) -> list[dict]:
    base = layout.checked_path(root, CONTINUATION_ROOT)
    if not base.is_dir():
        return []
    rows = []
    for child in sorted(base.iterdir()):
        if child.is_symlink() or not child.is_dir():
            continue
        task_id = _continuation_task(child.name)
        if actionable_tasks is not None and task_id not in actionable_tasks:
            continue
        value, relative, sha = _read_continuation(root, task_id)
        if value is None:
            continue
        starts = max(value["starts_recorded"], max([0, *[r["ordinal"] for r in _native_start_receipts(root, task_id)]]))
        rows.append({"task": task_id, "status": "PARTIAL", "generation": value["generation"],
                     "checkpoint": value["checkpoint"], "starts_recorded": starts,
                     "path": relative, "sha256": sha, "next_command": value["next_command"]})
    return rows


def coordinator_status(root: Path) -> dict:
    """Small coordinator view: dispatch, holds, maintainer gates, and suspicious landed-return signals."""
    spec = tomllib.loads(layout.checked_path(root, CONTEXT_SPEC).read_text())
    assignments = spec.get("assignments", [])
    view = schedule_inspect(root, assignments)
    plan = dispatch_plan(root, assignments)
    rows = {r["id"]: r for r in view["tasks"]}
    slotted = {s["task"] for s in plan["slots"]}
    held = {h["task"] for h in plan["holds"]}
    returned = _stale_return_signals(root, view)
    actionable = {r["id"] for r in view["tasks"] if r["state"] == "OPEN"}
    return {"schema": "farm-review-coordinator-status-v2",
            "dispatch": {"slots": plan["slots"], "holds": plan["holds"]},
            "parallelism": {"target": plan["target_parallelism"], "dispatched": len(plan["slots"]),
                            "underfill_by": plan["underfill_by"]},
            "open_blocked_by_maintainer_gates": plan["open_blocked_by_maintainer_gates"],
            "maintainer_ready": view["ready_for_maintainer"],
            "reviewer_ready_unassigned": [t for t in view["ready_for_review"] if t not in slotted | held],
            "waiting": [{"task": r["id"], "waiting_for": r["waiting_for"]} for r in rows.values() if r["status"] == "WAITING"],
            "returned_but_undecided": returned,
            "partial_continuations": _partial_continuation_tasks(root, actionable),
            # Compatibility alias for callers that consumed the advisory signal before it was named precisely.
            "stale_return_signals": returned,
            "warning": "returned_but_undecided signals are advisory and never change schedule state",
            "commands_executed": 0}


def render_coordinator_status(view: dict) -> str:
    lines = ["# Coordinator status", "", "DISPATCHED"]
    lines += [f"{s['number']:>2}  {s['task']}  descendants={s.get('descendant_count', 0)}" for s in view["dispatch"]["slots"]] or ["none"]
    parallel = view["parallelism"]
    lines += ["", f"PARALLELISM {parallel['dispatched']}/{parallel['target']}  underfill={parallel['underfill_by']}"]
    lines += ["", "HELD"] + [f"- {h['task']}: {h['reason']}" for h in view["dispatch"]["holds"]]
    lines += ["", "MAINTAINER READY", ", ".join(view["maintainer_ready"]) or "none"]
    lines += ["", "OPEN BLOCKED BY MAINTAINER GATES"]
    lines += [f"- {row['task']} <- {', '.join(row['maintainer_gates'])}" for row in view["open_blocked_by_maintainer_gates"]] or ["none"]
    lines += ["", "RETURNED BUT UNDECIDED"]
    if view["returned_but_undecided"]:
        for row in view["returned_but_undecided"]:
            lines.append(f"- {row['task']} [{row['status']}]: {len(row['artifacts'])} return-like artifact(s); INTAKE OR EXPLICIT RERUN")
    else:
        lines.append("none")
    lines += ["", "PARTIAL CONTINUATIONS"]
    lines += [f"- {row['task']} generation={row['generation']} starts_recorded={row['starts_recorded']} checkpoint={row['checkpoint']}"
              for row in view["partial_continuations"]] or ["none"]
    lines += ["", "These signals are advisory; no task state was changed.", ""]
    return "\n".join(lines)


def _self_test_progress(root: Path, *, started: float, files: list[str], checks: list[dict],
                        tests: list[dict], running: list[str], not_yet_run: list[str], workers: int,
                        excluded: list[dict], deselected: list[dict], progress_path: Path,
                        final_result: dict | None = None) -> None:
    import time
    passed = [row["path"] for row in tests if not row["returncode"]]
    failed = [row["path"] for row in tests if row["returncode"]]
    value = {
        "schema": "farm-review-self-test-progress-v1",
        "status": "COMPLETE" if final_result is not None else "RUNNING",
        "meaning": "Durable per-file progress for --self-test; RUNNING is not a PASS result.",
        "interpreter": sys.version.split()[0],
        "workers": workers,
        "seconds": round(time.monotonic() - started, 1),
        "checks": checks,
        "test_files": len(files),
        "completed_test_files": len(tests),
        "passed": passed,
        "failed": failed,
        "running": running,
        "not_yet_run": not_yet_run,
        "tests": tests,
        "maintainer_only": excluded,
        "deselected": deselected,
        "commands_executed": len(checks) + len(tests),
        "final_result": final_result,
    }
    _atomic_json(progress_path, value, replace=True)


def self_test(root: Path, *, jobs: int = 0, timeout: int = 1800, progress_path: Path | None = None) -> dict:
    """Run every advertised check, then the portable test files in parallel, from this tree.

    A failure, an empty selection or a changed shipped file fails the self-test. The
    commands use this interpreter; nothing launches games beyond the tests themselves.
    File completion is checkpointed atomically so an externally interrupted run retains
    a readable partial result without changing the final PASS/FAIL contract.
    """
    import time
    spec = tomllib.loads(layout.checked_path(root, CONTEXT_SPEC).read_text())
    files, excluded, deselected = portable_tests(root, spec)
    env = _portable_environment(root)
    started = time.monotonic()
    workers = jobs or min(8, os.cpu_count() or 1)
    target = progress_path or layout.checked_path(root, SELF_TEST_PROGRESS)
    _self_test_progress(root, started=started, files=files, checks=[], tests=[], running=[],
                        not_yet_run=files, workers=workers, excluded=excluded, deselected=deselected,
                        progress_path=target)
    checks = _run_checks(root, spec, env, timeout)
    _self_test_progress(root, started=started, files=files, checks=checks, tests=[], running=[],
                        not_yet_run=files, workers=workers, excluded=excluded, deselected=deselected,
                        progress_path=target)

    def checkpoint(tests: list[dict], running: list[str], not_yet_run: list[str]) -> None:
        _self_test_progress(root, started=started, files=files, checks=checks, tests=tests, running=running,
                            not_yet_run=not_yet_run, workers=workers, excluded=excluded, deselected=deselected,
                            progress_path=target)

    tests = _run_test_files(root, files, deselected, env, jobs=workers, timeout=timeout, on_progress=checkpoint)
    changed = _manifest_drift(root)
    failed = [c["name"] for c in checks if c["returncode"]] + [t["path"] for t in tests if t["returncode"]]
    result = {"schema": "farm-review-self-test-v1", "status": "FAIL" if failed or changed else "PASS",
              "meaning": "Advertised checks and portable tests ran from this tree with this interpreter; no scientific replay.",
              "interpreter": sys.version.split()[0], "workers": workers,
              "seconds": round(time.monotonic() - started, 1),
              "checks": checks, "test_files": len(tests), "failed": failed, "shipped_files_changed": changed,
              "tests": tests, "maintainer_only": excluded, "deselected": deselected,
              "limitations": ["Portable means runnable from a fresh extraction with Python 3.12, pytest, numpy and CPU torch.",
                              "maintainer_only files and deselected tests need omitted evidence or the GPU runtime; each names why."],
              "commands_executed": len(checks) + len(tests)}
    _self_test_progress(root, started=started, files=files, checks=checks, tests=tests, running=[],
                        not_yet_run=[], workers=workers, excluded=excluded, deselected=deselected,
                        progress_path=target, final_result=result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--schedule", action="store_true", help="Validate the existing assignment DAG and show ready work; no jobs or status writes")
    modes.add_argument("--theorems", action="store_true", help="Regenerate review/THEOREMS.md from the ledger and project/review.toml layers; no execution")
    modes.add_argument("--handoff", action="store_true", help="Validate and print the COMPLETE bounded maintained brief and notes; no execution")
    modes.add_argument("--assignment-context", action="store_true", help="Validate globally and show a staged view of the saved ticket; never draw or write")
    modes.add_argument("--work-map", action="store_true", help="Show the saved task's scheduled writes, input hashes, focused tests and manifest gaps; navigation only")
    modes.add_argument("--context", action="store_true", help="Validate and display the generated continuation map; no packaging or execution")
    modes.add_argument("--evidence-for-assignment", metavar="ID", help="Print intact receipts bound to one assignment's direct parents; navigation only, no ticket draw")
    modes.add_argument("--dispatch", action="store_true", help="Validate and show this wave's numbered slots and holds; no writes")
    modes.add_argument("--rebuild-dispatch", action="store_true", help="Maintainer: preserve outstanding IDs and atomically allocate replacements from next_slot using the retained registry")
    modes.add_argument("--lint-dispatch", action="store_true", help="Preflight write ownership and planned entrypoints for the numbered wave; no writes")
    modes.add_argument("--assign-slot", type=int, metavar="N", help="Save dispatch slot N as this copy's ticket; repeat calls return it, another slot refuses")
    modes.add_argument("--assign", metavar="TASK-ID", help="Save the dispatched task TASK-ID as this copy's ticket; held or blocked work refuses")
    modes.add_argument("--save-partial", type=Path, metavar="DRAFT.json", help="Save canonical task-keyed PARTIAL continuation state from a four-field draft; never changes schedule state")
    modes.add_argument("--adopt-partial", type=Path, metavar="RETURN.zip", help="Verify one registered PARTIAL return and transactionally install its continuation workspace")
    modes.add_argument("--charge-native-start", action="store_true", help="Atomically record one native-game start before launch; soft targets never hard-refuse continuation")
    modes.add_argument("--resume-continuation", action="store_true", help="Restore work-only files saved by the assigned task's PARTIAL, refusing divergent overwrites")
    modes.add_argument("--publish-dispatch", "--fast-build", dest="publish_dispatch", action="store_true",
                       help="Atomically refresh ZIP/checksum/prompts without running preflights")
    modes.add_argument("--full-build", action="store_true", help="Opt in to gating publication on the full extracted portable self-test")
    modes.add_argument("--self-test-zip", type=Path, metavar="PUBLISHED.zip", help="Execute a published ZIP's full portable suite in a fresh extraction and record its exact identity")
    modes.add_argument("--self-test", action="store_true", help="Run the advertised checks and portable tests from this tree; exits nonzero on any failure")
    modes.add_argument("--quick-self-test", action="store_true", help="Fast control-plane smoke test; not a replacement for --self-test")
    modes.add_argument("--status", action="store_true", help="Show compact coordinator state, returned-but-undecided signals and PARTIAL continuations; no writes")
    modes.add_argument("--package-budget", action="store_true", help="Report file/byte pressure and largest work trees; no writes")
    modes.add_argument("--draw-assignment", action="store_true", help="Legacy: draw one ready review assignment at random; an existing ticket is returned instead")
    modes.add_argument("--verify-extracted", action="store_true", help="Verify extracted files against their original manifest; never rebind it")
    modes.add_argument("--verify-zip", type=Path, help="Verify and safely extract an archive without executing its contents")
    modes.add_argument("--evidence-check", action="store_true", help="Verify semantic evidence and classifications, without executing reproduction commands")
    modes.add_argument("--evidence-candidate", help="Print an UNREVIEWED mechanical candidate for one relative source path")
    modes.add_argument("--intake-scan", nargs="+", type=Path, help="Mechanically scan return ZIPs/directories; never extract, execute, accept or delete arrivals")
    parser.add_argument("--baseline", type=Path, help="Exact outgoing ZIP or original retained MANIFEST.json used for intake comparison")
    parser.add_argument("--registry", type=Path, help=f"Retained registry for --adopt-partial/--rebuild-dispatch (default {DISPATCH_REGISTRY_ROOT})")
    parser.add_argument("--migrate-next-slot", action="store_true", help="With --rebuild-dispatch: explicitly initialize the high-water counter from complete retained issuance history")
    parser.add_argument("--now", help="With --rebuild-dispatch: timezone-aware lifecycle timestamp (default current UTC)")
    parser.add_argument("--intake-report", type=Path, help="Write the mechanical JSON report; compact human summary goes to stdout")
    parser.add_argument("--extract", type=Path, help="New destination for --verify-zip")
    parser.add_argument("--require-deliveries", action="store_true", help="Refuse outcome-review readiness when registered final evidence is absent")
    parser.add_argument("--summary", action="store_true", help="Compact validated navigation for --context, --evidence-check, --schedule, --dispatch, --work-map or an assignment; complete ticket remains saved")
    parser.add_argument("--jobs", type=int, default=0, help="Parallel test files for --self-test (default min(8, CPUs))")
    parser.add_argument("--record-output", type=Path, help=f"Run record for --self-test-zip (default {SELF_TEST_LAST_RUN})")
    parser.add_argument("--last-self-test", type=Path, help=f"Last-PASS record read by builds or updated by --self-test-zip (default {SELF_TEST_LAST_PASS})")
    args = parser.parse_args()
    assigning = args.draw_assignment or args.assign_slot is not None or args.assign
    if args.summary and not (args.context or args.evidence_check or assigning or args.schedule or args.dispatch or args.work_map or args.status):
        parser.error("--summary requires --context, --evidence-check, --schedule, --dispatch, --work-map or an assignment")
    if args.jobs < 0 or (args.jobs and not (args.self_test or args.quick_self_test or args.self_test_zip)):
        parser.error("nonnegative --jobs requires --self-test, --quick-self-test or --self-test-zip")
    if args.record_output and not args.self_test_zip:
        parser.error("--record-output requires --self-test-zip")
    if args.last_self_test and not (args.self_test_zip or args.full_build):
        parser.error("--last-self-test requires --full-build or --self-test-zip")
    if args.require_deliveries and not (args.context or args.handoff or args.assignment_context or args.work_map
                                        or (assigning and args.summary)):
        parser.error("--require-deliveries requires a context view")
    if bool(args.verify_zip) != bool(args.extract):
        parser.error("--verify-zip and --extract must be used together")
    if (args.baseline or args.intake_report) and not args.intake_scan:
        parser.error("--baseline/--intake-report require --intake-scan")
    if args.registry and not (args.adopt_partial or args.rebuild_dispatch):
        parser.error("--registry requires --adopt-partial or --rebuild-dispatch")
    if (args.migrate_next_slot or args.now) and not args.rebuild_dispatch:
        parser.error("--migrate-next-slot/--now require --rebuild-dispatch")
    try:
        if args.intake_scan:
            result = review_intake.scan(args.intake_scan, args.baseline)
            if args.intake_report:
                target = args.intake_report
                if target.suffix != '.json' or any(p.is_symlink() for p in (target,*target.parents)):
                    raise ValueError('regular JSON intake-report path required')
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix='.intake-', dir=target.parent)
                try:
                    with os.fdopen(fd,'wb') as stream:
                        stream.write(review_intake.canonical(result)+b'\n')
                    os.replace(temporary,target)
                finally:
                    if os.path.exists(temporary): os.unlink(temporary)
                print(review_intake.summary(result),end='')
            else:
                print(json.dumps(result,indent=2))
            return 1 if result['errors'] or any(r['status'] != 'INVENTORIED' for r in result['returns']) else 0
        elif args.adopt_partial:
            result = adopt_partial(ROOT, args.adopt_partial, args.registry)
        elif args.rebuild_dispatch:
            result = rebuild_dispatch(ROOT, registry=args.registry, now=args.now,
                                      migrate_next_slot=args.migrate_next_slot)
        elif args.lint_dispatch:
            spec = tomllib.loads(layout.checked_path(ROOT, CONTEXT_SPEC).read_text())
            assignments = spec.get("assignments", [])
            plan = dispatch_plan(ROOT, assignments)
            result = lint_dispatch(ROOT, assignments, plan)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "PASS" else 1
        elif args.schedule or args.dispatch:
            spec = tomllib.loads(layout.checked_path(ROOT, CONTEXT_SPEC).read_text())
            result = schedule_inspect(ROOT, spec.get("assignments", []))
            plan = dispatch_plan(ROOT, spec.get("assignments", [])) if args.dispatch or layout.checked_path(ROOT, DISPATCH).is_file() else None
            if args.dispatch:
                if args.summary:
                    print(render_dispatch_summary(plan))
                    return 0
                result = plan
            elif args.summary:
                text = work_schedule.summary(result, plan)
                if plan is not None:
                    text = text.replace(work_schedule.dispatch_summary(plan), render_dispatch_summary(plan))
                print(text, end="")
                return 0
        elif args.self_test_zip:
            result = published_self_test(args.self_test_zip, args.record_output or ROOT / SELF_TEST_LAST_RUN,
                                         jobs=args.jobs, last_pass_path=args.last_self_test or ROOT / SELF_TEST_LAST_PASS)
            print(json.dumps(result, indent=2))
            return 0 if result['status'] == 'PASS' else 1
        elif args.self_test:
            result = self_test(ROOT, jobs=args.jobs)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "PASS" else 1
        elif args.quick_self_test:
            result = quick_self_test(ROOT, jobs=args.jobs)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "PASS" else 1
        elif args.status:
            result = coordinator_status(ROOT)
            if args.summary:
                print(render_coordinator_status(result), end="")
                return 0
        elif args.package_budget:
            result = package_budget(ROOT)
        elif args.save_partial:
            if args.save_partial.is_symlink() or not args.save_partial.is_file():
                raise ValueError("--save-partial requires a regular JSON draft")
            result = save_partial(ROOT, review_intake.parse_json(args.save_partial.read_bytes()))
        elif args.charge_native_start:
            result = charge_native_start(ROOT)
        elif args.resume_continuation:
            result = resume_continuation(ROOT)
        elif args.assign_slot is not None or args.assign:
            ticket = assign(ROOT, slot=args.assign_slot, task=args.assign)
            result = {key: ticket[key] for key in ("schema", "ticket", "selection_method", "slot", "task_id", "return_name")}
            result.update({"ticket_path": ASSIGNMENT, "next": "python3 -B src/farm_neat/tooling/review_package.py --assignment-context"})
            if args.summary:
                result = assignment_context(ROOT, require_deliveries=args.require_deliveries)
        elif args.evidence_for_assignment:
            context = review_context(ROOT)
            result = assignment_evidence(ROOT, context, args.evidence_for_assignment)
            print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            return 0
        elif args.evidence_candidate:
            result = evidence.candidate(ROOT, args.evidence_candidate)
        elif args.evidence_check:
            result = evidence.packet_view(ROOT)
            if args.summary:
                result = evidence_summary(result)
        elif args.draw_assignment:
            result = draw_assignment(ROOT)
            if args.summary:
                result = assignment_context(ROOT, require_deliveries=args.require_deliveries)
        elif args.assignment_context:
            result = assignment_context(ROOT, require_deliveries=args.require_deliveries)
        elif args.work_map:
            result = assignment_work_map(ROOT, require_deliveries=args.require_deliveries)
            if args.summary:
                print(render_work_map(result), end="")
                return 0
        elif args.theorems:
            text = theorem_index(ROOT, tomllib.loads(layout.checked_path(ROOT, CONTEXT_SPEC).read_text()))
            layout.checked_path(ROOT, THEOREMS).write_text(text)
            print(json.dumps({"file": THEOREMS, "bytes": len(text.encode()), "commands_executed": 0}))
            return 0
        elif args.handoff:
            result = review_context(ROOT, require_deliveries=args.require_deliveries)
            print(render_handoff(ROOT, result), end="")
            return 0
        elif args.context:
            result = review_context(ROOT, require_deliveries=args.require_deliveries)
            if args.summary:
                result = context_summary(result)
        elif args.verify_extracted:
            manifest = json.loads((ROOT / MANIFEST).read_text())
            if manifest.get("schema") != SCHEMA:
                raise ValueError("unsupported review manifest")
            unchanged(ROOT, manifest)
            result = {"status": "EXTRACTED_MEMBER_HASHES_VERIFIED", "members": len(manifest["members"]),
                      "commands_executed": 0}
        elif args.verify_zip:
            manifest = verify_archive(args.verify_zip, args.extract)
            result = {"status": "ARCHIVE_VERIFIED_AND_EXTRACTED", "members": len(manifest["members"]),
                      "commands_executed": 0, "destination": str(args.extract)}
        elif args.full_build:
            result = full_build(last_self_test=args.last_self_test)
        else:
            result = publish_dispatch(ROOT)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, zipfile.BadZipFile, subprocess.TimeoutExpired) as error:
        print(f"Review package refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
