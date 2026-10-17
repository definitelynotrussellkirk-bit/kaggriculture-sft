"""Source-bound search knowledge and explicit reviewer handoffs; stdlib, no execution.

Records are immutable, content-addressed JSON. Commands in records are inert text.
The ledger is an index over receipts, not a replacement for Q1-Q4 or training-state.
Run ``python3 -B src/farm_neat/tooling/search_ledger.py --help`` from any directory.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any

USER_GUIDE = 'Purpose: reuse bounded search knowledge and route falsifiable follow-ups without another campaign or status diary.\nOwner: tooling; integration maintainer arbitrates scientific acceptance and overlapping work.\nStatus: ACTIVE\nEvidence: project/registries/search-ledger/ contains source-bound claims/requests; tests/unit/test_search_ledger.py checks the local protocol, not scientific truth.\nNext: run search_ledger.py verify, then queue; generate the selected handoff and have the operator deliver it.\nRemove-when: replace this guide when the tool interface changes; never retire a cited source without checking dependencies.\n\n# Search knowledge ledger\n\n## One job\n\nAnswer: **What have we actually eliminated, counted, factored, or disproved; under\nwhat conditions; and what is the next smallest check?** The registry links existing\nreceipts and functions. It does not store copies of raw games, replace Q1–Q4,\nbecome another training framework, or require paperwork before permitted work.\nKeep conclusions in the existing threads and current measurements in training-state.\n\nRun from the repository root. The routing candidate remains an optional verified overlay:\n\n```bash\npython3 -B src/farm_neat/tooling/search_ledger.py verify\npython3 -B src/farm_neat/tooling/search_ledger.py render\npython3 -B src/farm_neat/tooling/search_ledger.py list --query assignment\npython3 -B src/farm_neat/tooling/search_ledger.py queue\npython3 -B src/farm_neat/tooling/search_ledger.py show K03\n```\n\n`render` is a generated human view; `list`, `show`, `queue` and `verify` emit JSON.\nNo command launches a process, runs a stored command, initializes a game, loads a\ncheckpoint, installs anything, sends a message, or starts another agent.\n\n## What belongs in a claim\n\nA claim is a reusable restriction on search, not a diary entry such as “looked at\nrouting.” It must name a bounded question, the measured/predicted quantity and\nunits, the domain, fixed and varied factors, information boundary and exclusions.\nIt declares what can be reused, what remains unproved, and an exact reproduction\ncommand or a clearly unexecuted hypothesis protocol.\n\nCoverage separates the number evaluated, a known total or `null`, the unit counted,\nindependent units, method and data exposure. A count of matrix edges is not a count\nof independent worlds; represented DP histories are not individually run games.\n`exhaustive: true` requires equal evaluated/total counts **within the declared\ndomain**, never the whole game by implication. Unknown denominators stay unknown.\nAn exhausted budget is not completion or a proof of infeasibility.\n\nClassifications are `conditional_identity`, `exhaustive_finite_check`,\n`empirical_paired`, `empirical_panel`, `counterexample`, `data_contract` and\n`hypothesis`. These are author assertions, not promotions by the validator. A\nfinite test suite does not confer theorem status. The seed claims are source-bound\nimports from the earlier review, not newly independent replications.\n\nEvidence entries contain a repository-relative path, SHA-256 and role (`source`,\n`receipt`, `table`, `test`, `plan`, `journal`). Optional JSON Pointer checks bind\nindividual quantities to the file, including gzip JSON and exact array lengths:\n\n```json\n{\n  "path": "work/runs/reviewer-play/routing-factor/AUDIT.json",\n  "sha256": "33530e1c3e756f2296c1e5369c8aa324e7a174a7609caa58aa25dfe9c81a2d99",\n  "role": "receipt",\n  "checks": [{"pointer": "/unique_cost_matrices", "equals": 754}]\n}\n```\n\nGenerate bindings with `search_ledger.py bind PATH --role source`. Pin the exact\nimplementation whose scope you are claiming, not only the nice-looking result.\nHashes establish byte identity, not correctness or provenance authenticity.\n\n## Add a finding without destroying the old one\n\n```bash\nmkdir -p work/scratch/ledger-contribution\npython3 -B src/farm_neat/tooling/search_ledger.py template claim \\\n  > work/scratch/ledger-contribution/claim.json\n# Fill the bounded claim; replace CHANGE-ME with a distinct descriptive key.\n# Add actual evidence paths/hashes and honest coverage. A hypothesis can have no data.\npython3 -B src/farm_neat/tooling/search_ledger.py add \\\n  work/scratch/ledger-contribution/claim.json\npython3 -B src/farm_neat/tooling/search_ledger.py verify\n```\n\nRecords are canonical JSON files named by their full content hash under\n`project/registries/search-ledger/`. Re-adding identical content is idempotent;\nindependent concurrent writes do not share a mutable JSONL append point. The full\nhash is the durable reference; short `K01`/`F01` keys are human aliases. Multiple\nrecords with an alias are reported as ambiguous, not resolved by timestamp.\n\nA correction is a **new** claim with a `relations` entry such as\n`{"kind":"refutes","target":"FULL_OLD_CLAIM_HASH"}`. Other relations are\n`narrows`, `corroborates` and `supersedes`. True dependencies go in `depends_on`.\nA counterclaim is not dependent on the truth of the claim it refutes.\n\n`list` exposes incoming/outgoing relationships. Same-scope claims produce a\nwarning: they may be a useful replication, a duplicate or a disagreement; sharing\na scope or a differently worded conclusion does not establish independent data.\nMerging offline branches means unioning immutable JSON records after source and\nmembership review, then running `verify`. Do not overwrite a conflicting file or\nsilently select the most recent conclusion. There is no distributed assignment\nlock between isolated ZIPs; the maintainer must explicitly allocate work.\n\nBefore source edits or retention cleanup:\n\n```bash\npython3 -B src/farm_neat/tooling/search_ledger.py impact src/farm_neat/economy/orders.py\n```\n\n`verify` detects missing/drifted evidence and propagates stale dependencies to\nfollow-ups. Handoff refuses stale dependencies. Stale means “revalidate for this\nsource version,” not “the old finding was false.” Do not fix it by substituting a\nnew hash without rerunning the relevant check. Keep receipt retention bounded;\nmove established abstractions into maintained code/contracts/tests and check\nincoming references before retiring unique evidence. Immutable does not mean keep\nall experimental payloads forever.\n\n## Ask another reviewer, then bring the result back\n\nThe first prepared request, **F01**, asks for state-field ablation over the retained\nanimal transition table. It is a pure table analysis, not a new game or model.\nIts acceptance test excludes a common false proof: when removing a state field,\nalso remove that field from the next-state target. Otherwise simply copying the\nomitted field would create a trivial, irrelevant “necessity” counterexample.\n\n```bash\npython3 -B src/farm_neat/tooling/search_ledger.py handoff F01 \\\n  > work/scratch/ledger-contribution/HANDOFF.md\n```\n\nThe output contains the exact request ID, originating reviewer, requested role,\nQ-thread, starting claims/hashes, permitted resources, first steps, success and\nnegative criteria, and expected return fields. **It is prepared, not sent.** The\noperator copies it into another agent\'s task or supplies it with the existing\nsource-bound review packet. This chat cannot wake another agent or receive a\nlater result without another user turn or an actual supported integration.\n\nThe recipient adds any genuinely new claim/evidence and then a response:\n\n```bash\npython3 -B src/farm_neat/tooling/search_ledger.py template response \\\n  > work/scratch/ledger-contribution/response.json\n# Set request to F01\'s FULL hash; name the actual reviewer; give an answer and limits.\n# Use supports/refutes/inconclusive/blocked; substantive answers need evidence.\n# List newly added claim hashes in new_claims.\npython3 -B src/farm_neat/tooling/search_ledger.py add \\\n  work/scratch/ledger-contribution/response.json\npython3 -B src/farm_neat/tooling/search_ledger.py verify\n```\n\nReturn only the new records, code/tests and necessary evidence as the existing\nsource-bound delta. The maintainer merges it and gives it back to the originating\nreviewer. The originating review session reads:\n\n```bash\npython3 -B src/farm_neat/tooling/search_ledger.py inbox --for "GPT-6 Astra Pro"\npython3 -B src/farm_neat/tooling/search_ledger.py template review \\\n  > work/scratch/ledger-contribution/review.json\n# Set response to the received hash, then accept / needs_work / reject with a reason.\npython3 -B src/farm_neat/tooling/search_ledger.py add \\\n  work/scratch/ledger-contribution/review.json\n```\n\nA response alone is `AWAITING_REVIEW`. An explicit accepted review closes an answer,\nnot an entire topic. Accepted blockers remain `BLOCKER_ACKNOWLEDGED`. Conflicting\nreviews and parallel responses remain visible. An amended response or review\nuses `supersedes: ["FULL_PREVIOUS_HASH"]`, only within the same request/response.\nThis is explicit reconciliation, never last-write-wins; all old records remain\nreadable. `queue --all` includes accepted answers; `inbox` keeps the return trail.\nSelf-declared author names are not authentication; maintainer intake supplies the\nhuman permission boundary. No record grants compute or scientific authority.\n\n## Order work by the decision it resolves\n\nThe prepared order favors cheap, discriminating checks before more games:\nF01 state sufficiency/minimal counterexamples, F02 routing inventory boundaries,\nF03 schedule-stratified interaction rankings. F04 is producer learner integration;\nF05 is a separately authorized opponent-response panel design. F04/F05 remain\n`MAINTAINER_AUTHORIZATION_REQUIRED`, even after a proposal is accepted.\n\nPriority 1 means first; S/M/L are qualitative costs, not promised runtimes. There\nis deliberately no fabricated automatic “research value” score. A useful next\nrequest should identify a live decision, competing explanations, the smallest\ncheck that separates them, a budget, the negative outcome to retain, and what\nsearch work its answer will eliminate. Do not let ledger growth replace learning.\n\nThe earlier routing run used all four permitted attempts. Every seed request\nretains zero new games, optimizer updates, GPU jobs and network calls. Future\nmaintainer-authorized work requires its normal independent registration and\nreceipts; editing a budget here does not authorize it. No blanket hold is imposed\non separately authorized producer work.\n\n## Thin external packets\n\n`EVIDENCE.jsonl` under `work/runs/review-package/` separates literal results,\ncontrols, interpretation, limits, decisions and original source identities. The\nproducer checks actual bytes/selectors. In a thin extraction, `verify` reports\n`PASS_WITH_SEMANTIC_REFERENCES` for declared historical attestations; this is not\nraw replay. Missing undeclared evidence still fails. Only `project/review.toml`\nchecks are promised portable. Historical first steps may need maintainer data.\nReturn compact receipts and a justified witness; UNKNOWN prevents publication.\n'

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT / "src"))
from farm_neat.tooling import evidence_compaction
STORE = "project/registries/search-ledger"
VERSION = 1
MAX_RECORD = 200_000
MAX_JSON = 32_000_000
KINDS = {"claim", "followup", "response", "review"}
CLASSES = {"conditional_identity", "exhaustive_finite_check", "empirical_paired",
           "empirical_panel", "counterexample", "data_contract", "hypothesis"}
ROLES = {"source", "receipt", "table", "test", "plan", "journal"}
BUDGET_KEYS = {"new_games", "optimizer_updates", "gpu_jobs", "network_calls"}
HEX = re.compile(r"^[0-9a-f]{64}$")


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f"duplicate JSON key: {k}")
        result[k] = v
    return result


def parse(raw: bytes) -> Any:
    def bad(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=bad)


def safe_path(root: Path, name: str) -> Path:
    """Reject traversal, absolute names, Windows paths and symlinks, even within root."""
    if not isinstance(name, str) or not name or name == "." or "\\" in name or ":" in name:
        raise ValueError(f"unsafe relative path: {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or p.as_posix() != name or any(x in {".", ".."} for x in p.parts):
        raise ValueError(f"unsafe relative path: {name!r}")
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink is not an evidence/ledger path: {name}")
    return current


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pointer(value: Any, path: str) -> Any:
    """Small JSON Pointer reader. Empty pointer means the document itself."""
    if path == "":
        return value
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with /")
    for part in path[1:].split("/"):
        if re.search(r"~(?![01])", part):
            raise ValueError("invalid JSON pointer escape")
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", key):
                raise ValueError("invalid array index")
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise ValueError("pointer traverses a scalar")
    return value


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def texts(value: Any, label: str, *, nonempty: bool = True) -> None:
    require(isinstance(value, list) and (bool(value) or not nonempty)
            and all(isinstance(v, str) and v.strip() for v in value), label)


def shape(rec: dict) -> None:
    require(isinstance(rec, dict), "record must be an object")
    require(rec.get("version") == VERSION and type(rec.get("version")) is int,
            "unsupported version")
    require(rec.get("type") in KINDS, "unknown record type")
    for k in ("key", "author", "title"):
        require(isinstance(rec.get(k), str) and bool(rec[k].strip()), f"missing {k}")
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,79}", rec["key"])), "invalid key")
    require(rec.get("topic") in {"Q1", "Q2", "Q3", "Q4", "Q5"}, "topic must route to Q1-Q5")
    require(isinstance(rec.get("evidence"), list), "evidence must be a list")
    for e in rec["evidence"]:
        require(isinstance(e, dict), "evidence must be an object")
        safe_path(Path("/"), e.get("path"))
        require(isinstance(e.get("sha256"), str) and bool(HEX.fullmatch(e["sha256"])),
                "evidence needs a full SHA-256")
        require(e.get("role") in ROLES, "invalid evidence role")
        for c in e.get("checks", []):
            require(isinstance(c, dict) and isinstance(c.get("pointer"), str)
                    and "equals" in c and c.get("op", "value") in {"value", "length"},
                    "invalid selector check")
    kind = rec["type"]
    if kind == "claim":
        require(rec.get("classification") in CLASSES, "invalid claim classification")
        require(isinstance(rec.get("statement"), str) and bool(rec["statement"].strip()),
                "missing claim statement")
        scope = rec.get("scope", {})
        require(isinstance(scope, dict), "scope must be an object")
        for k in ("question", "estimand", "domain", "information_boundary"):
            require(isinstance(scope.get(k), str) and bool(scope[k].strip()), f"scope.{k}")
        for k in ("fixed", "varied", "excluded"):
            texts(scope.get(k), f"scope.{k}")
        coverage = rec.get("coverage", {})
        require(isinstance(coverage, dict), "coverage must be an object")
        for k in ("evaluated", "total", "independent_units"):
            v = coverage.get(k)
            require(v is None or (type(v) is int and v >= 0), f"coverage.{k}")
        require(type(coverage.get("evaluated")) is int, "evaluated count required")
        require(type(coverage.get("exhaustive")) is bool, "exhaustive must be boolean")
        total, evaluated = coverage.get("total"), coverage["evaluated"]
        require(total is None or evaluated <= total, "evaluated exceeds defined total")
        require(not coverage["exhaustive"] or total == evaluated,
                "exhaustive requires equal evaluated and total in the declared domain")
        for k in ("unit", "exposure", "method"):
            require(isinstance(coverage.get(k), str) and bool(coverage[k].strip()), f"coverage.{k}")
        for k in ("reuse", "limits", "reproduce"):
            texts(rec.get(k), k)
        if rec["classification"] != "hypothesis":
            require(bool(rec["evidence"]), "non-hypothesis claim requires evidence")
        require(isinstance(rec.get("depends_on", []), list), "depends_on must be a list")
        for relation in rec.get("relations", []):
            require(relation.get("kind") in {"refutes", "narrows", "supersedes", "corroborates"}
                    and HEX.fullmatch(str(relation.get("target", ""))), "invalid relation")
    elif kind == "followup":
        texts(rec.get("claims"), "followup needs claims")
        for k in ("question", "requested_of", "reply_to", "why_next"):
            require(isinstance(rec.get(k), str) and bool(rec[k].strip()), f"missing {k}")
        for k in ("acceptance", "return_fields", "first_steps"):
            texts(rec.get(k), k)
        require(rec.get("gate") in {"READ_ONLY", "MAINTAINER_AUTHORIZATION_REQUIRED"}, "invalid gate")
        budget = rec.get("budget", {})
        require(isinstance(budget, dict) and set(budget) == BUDGET_KEYS
                and all(type(v) is int and v >= 0 for v in budget.values()), "invalid budget")
        require(rec["gate"] != "READ_ONLY" or not any(budget.values()),
                "READ_ONLY requires zero games/updates/GPU/network")
        require(type(rec.get("priority")) is int and 1 <= rec["priority"] <= 5,
                "priority must be 1 (first) to 5")
        require(rec.get("cost") in {"S", "M", "L"}, "cost must be S/M/L")
    elif kind == "response":
        require(HEX.fullmatch(str(rec.get("request", ""))) is not None, "response needs request hash")
        require(rec.get("outcome") in {"supports", "refutes", "inconclusive", "blocked"},
                "invalid response outcome")
        for k in ("answer", "limits"):
            require(isinstance(rec.get(k), str) and bool(rec[k].strip()), f"missing {k}")
        require(rec["outcome"] == "blocked" or bool(rec["evidence"]),
                "substantive response needs evidence")
        texts(rec.get("new_claims", []), "new_claims must be a list", nonempty=False)
    else:
        require(HEX.fullmatch(str(rec.get("response", ""))) is not None, "review needs response hash")
        require(rec.get("decision") in {"accept", "needs_work", "reject"}, "invalid review decision")
        require(isinstance(rec.get("reason"), str) and bool(rec["reason"].strip()), "missing reason")
    if kind in {"response", "review"}:
        previous = rec.get("supersedes", [])
        require(isinstance(previous, list) and len(previous) == len(set(previous))
                and all(isinstance(v, str) and HEX.fullmatch(v) for v in previous),
                "supersedes must contain distinct full record hashes")
    canonical(rec)  # Also reject non-JSON and nonfinite objects supplied through Python.


def links(rec: dict, *, dependent_only: bool = False) -> list[tuple[str, str]]:
    if rec["type"] == "claim":
        out = [(v, "claim") for v in rec.get("depends_on", [])]
        if not dependent_only:
            out += [(v["target"], "claim") for v in rec.get("relations", [])]
        return out
    if rec["type"] == "followup":
        return [(v, "claim") for v in rec["claims"]]
    if rec["type"] == "response":
        return ([(rec["request"], "followup")] + [(v, "claim") for v in rec.get("new_claims", [])]
                + ([] if dependent_only else [(v, "response") for v in rec.get("supersedes", [])]))
    return [(rec["response"], "response")] + ([] if dependent_only else
        [(v, "review") for v in rec.get("supersedes", [])])


def scope_id(rec: dict) -> str:
    return digest(canonical({"scope": rec["scope"],
        "sources": sorted((e["path"], e["sha256"]) for e in rec["evidence"] if e["role"] == "source"),
        "unit": rec["coverage"]["unit"]}))


class Ledger:
    def __init__(self, root: Path = ROOT, store: str = STORE):
        self.root = Path(root).resolve()
        self.directory = safe_path(self.root, store)

    def read(self) -> dict[str, dict]:
        if not self.directory.exists():
            return {}
        require(self.directory.is_dir(), "ledger path is not a directory")
        result = {}
        for p in sorted(self.directory.iterdir()):
            require(p.is_file() and not p.is_symlink() and p.suffix == ".json"
                    and bool(HEX.fullmatch(p.stem)), f"unexpected ledger member: {p.name}")
            require(p.stat().st_size <= MAX_RECORD, f"record too large: {p.name}")
            raw = p.read_bytes()
            require(digest(raw) == p.stem, f"modified immutable record: {p.name}")
            rec = parse(raw)
            shape(rec)
            require(canonical(rec) == raw, f"noncanonical record: {p.name}")
            result[p.stem] = rec
        return result

    def audit(self, records: dict[str, dict] | None = None) -> dict:
        records = self.read() if records is None else records
        errors, warnings, stale, checked = [], [], {}, {}
        semantic = evidence_compaction.semantic_references(self.root)
        source_copies = evidence_compaction.historical_sources(self.root)
        historical = {}
        attested = {}
        scopes: dict[str, list[str]] = {}
        aliases: dict[str, list[str]] = {}
        for rid, rec in records.items():
            shape(rec)
            aliases.setdefault(rec["key"], []).append(rid)
            if rec["type"] == "claim":
                scopes.setdefault(scope_id(rec), []).append(rid)
            for target, wanted in links(rec):
                if target not in records or records[target]["type"] != wanted:
                    errors.append(f"{rec['key']}: missing/wrong-type reference {target}")
            if rec["type"] in {"response", "review"}:
                target_field = "request" if rec["type"] == "response" else "response"
                for prior in rec.get("supersedes", []):
                    if prior in records and records[prior].get(target_field) != rec[target_field]:
                        errors.append(f"{rec['key']}: supersedes belongs to a different {target_field}")
            for e in rec["evidence"]:
                key = (e["path"], e["sha256"], canonical(e.get("checks", [])).decode(), rid)
                if key not in checked:
                    problems = []
                    try:
                        p = safe_path(self.root, e["path"])
                        copied = source_copies.get((e['path'], e['sha256']))
                        if copied and e['role'] in {'source', 'test'}:
                            p = safe_path(self.root, copied)
                            historical[key] = copied
                        if not p.exists() and semantic:
                            candidates = [r for r in semantic.get((e['path'], e['sha256']), [])
                                if any(s['path'] == STORE + '/' + rid + '.json' and s['sha256'] == rid for s in r['source'])]
                            require(bool(candidates), 'missing raw evidence without a source-bound semantic reference for this record')
                            for c in e.get('checks', []):
                                require(any(any(o.get('source') == e['path'] and o.get('pointer') == c['pointer']
                                    and o.get('op') == c.get('op', 'value') and canonical(o.get('value')) == canonical(c['equals'])
                                    for o in r['result'].get('observations', [])) for r in candidates),
                                    'semantic reference does not retain required selector')
                            attested[key] = [r['id'] for r in candidates]
                            checked[key] = []
                            continue
                        require(p.is_file(), "missing file")
                        require(file_sha(p) == e["sha256"], "SHA-256 mismatch")
                        if e.get("checks"):
                            opener = gzip.open if p.suffix == ".gz" else open
                            with opener(p, "rb") as stream:
                                raw = stream.read(MAX_JSON + 1)
                            require(len(raw) <= MAX_JSON, "JSON selector source exceeds size limit")
                            data = parse(raw)
                            for c in e["checks"]:
                                found = pointer(data, c["pointer"])
                                if c.get("op") == "length":
                                    require(isinstance(found, (list, dict, str)), "length needs a container")
                                    found = len(found)
                                require(canonical(found) == canonical(c["equals"]),
                                        f"selector mismatch at {c['pointer']}")
                    except (ValueError, OSError, KeyError, IndexError, TypeError, EOFError) as exc:
                        problems.append(f"{e['path']}: {exc}")
                    checked[key] = problems
                if checked[key]:
                    stale.setdefault(rid, []).extend(checked[key])
        # An amended response or review carries the current bindings; the record it
        # supersedes stays readable but its drifted evidence no longer blocks the ledger.
        superseded = {prior for rec in records.values() if rec["type"] in {"response", "review"}
                      for prior in rec.get("supersedes", [])}
        superseded_stale = {rid: stale.pop(rid) for rid in list(stale) if rid in superseded}
        # Cycles make derived dependency validity ill-defined. Never select a newest record.
        active, done = set(), set()
        def visit(rid):
            if rid in active:
                errors.append(f"dependency cycle at {records[rid]['key']}")
                return
            if rid in done:
                return
            active.add(rid)
            for ref, _ in links(records[rid], dependent_only=True):
                if ref in records:
                    visit(ref)
            active.remove(rid)
            done.add(rid)
        for rid in records:
            visit(rid)
        invalid = set(stale)
        changed = True
        while changed:
            before = len(invalid)
            invalid.update(rid for rid, rec in records.items()
                if any(ref in invalid for ref, _ in links(rec, dependent_only=True)))
            changed = len(invalid) != before
        for key, ids in aliases.items():
            if len(ids) > 1:
                warnings.append(f"ambiguous alias {key}: use full content hashes")
        for ids in scopes.values():
            if len(ids) > 1:
                warnings.append("same declared scope (not independent replication by itself): " +
                                ", ".join(records[x]["key"] for x in ids))
        semantic_ids = {key[3] for key in attested}
        while True:
            expanded = semantic_ids | {rid for rid, rec in records.items()
                if any(ref in semantic_ids for ref, _ in links(rec, dependent_only=True))}
            if expanded == semantic_ids: break
            semantic_ids = expanded
        return {"status": ("PASS_WITH_SEMANTIC_REFERENCES" if attested else "PASS") if not errors and not stale else "NEEDS_ATTENTION",
                "records": len(records), "claims": sum(r["type"] == "claim" for r in records.values()),
                "followups": sum(r["type"] == "followup" for r in records.values()),
                "errors": errors, "warnings": warnings, "stale_direct": stale,
                "superseded_stale": superseded_stale,
                "stale_transitive": sorted(invalid), "evidence_bindings_checked": len(checked),
                "raw_bindings_verified": sum(not problems and key not in attested for key, problems in checked.items()),
                "semantic_transitive": sorted(semantic_ids),
                "semantic_references": [{'path': k[0], 'sha256': k[1], 'record': k[3], 'receipts': v} for k, v in sorted(attested.items())],
                "historical_source_bindings": [{'path': k[0], 'sha256': k[1], 'record': k[3], 'copy': v} for k, v in sorted(historical.items())],
                "commands_executed": 0, "meaning": "Schema/reference and available byte/selector checks only. Semantic references attest omitted historical evidence; they do not verify its raw bytes, rerun its reducer, prove a claim, or grant authority."}

    def add(self, rec: dict) -> dict:
        shape(rec)
        payload = canonical(rec)
        require(len(payload) <= MAX_RECORD, "record too large")
        rid = digest(payload)
        records = self.read()
        if rid in records:
            return {"id": rid, "key": rec["key"], "status": "ALREADY_PRESENT"}
        records[rid] = rec
        audit = self.audit(records)
        require(not audit["errors"], "; ".join(audit["errors"]))
        require(rid not in audit["stale_transitive"], "new record or its dependencies have stale/missing evidence")
        self.directory.mkdir(parents=True, exist_ok=True)
        # Write fully, then install with a no-clobber atomic link. Concurrent equal records dedupe.
        fd, temp = tempfile.mkstemp(prefix=".ledger-", dir=self.directory.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temp, self.directory / (rid + ".json"))
                status = "ADDED"
            except FileExistsError:
                require((self.directory / (rid + ".json")).read_bytes() == payload,
                        "content-address collision or concurrent corruption")
                status = "ALREADY_PRESENT"
        finally:
            os.unlink(temp)
        return {"id": rid, "key": rec["key"], "status": status}

    @staticmethod
    def resolve(records: dict[str, dict], query: str) -> str:
        if query in records:
            return query
        matches = [rid for rid, r in records.items() if r["key"] == query or rid.startswith(query)]
        require(len(matches) == 1, f"unknown or ambiguous record {query!r}; use a full hash")
        return matches[0]

    @staticmethod
    def request_status(records: dict[str, dict], request: str) -> str:
        responses = [rid for rid, r in records.items()
                     if r["type"] == "response" and r["request"] == request]
        superseded = {x for rid in responses for x in records[rid].get("supersedes", [])}
        responses = [rid for rid in responses if rid not in superseded]
        if not responses:
            return "OPEN"
        decision_sets = []
        for response in responses:
            reviews = {rid: r for rid, r in records.items()
                       if r["type"] == "review" and r["response"] == response}
            superseded_reviews = {x for r in reviews.values() for x in r.get("supersedes", [])}
            decisions = {r["decision"] for rid, r in reviews.items() if rid not in superseded_reviews}
            if len(decisions) > 1:
                return "REVIEW_CONFLICT"
            decision_sets.append(decisions)
        if len(responses) > 1:
            return "MULTIPLE_RESPONSES_REVIEW_REQUIRED"
        if not decision_sets[0]:
            return "AWAITING_REVIEW"
        if decision_sets[0] == {"accept"} and records[responses[0]]["outcome"] == "blocked":
            return "BLOCKER_ACKNOWLEDGED"
        return {"accept": "ANSWER_ACCEPTED", "needs_work": "CHANGES_REQUESTED",
                "reject": "ANSWER_REJECTED"}[next(iter(decision_sets[0]))]

    def handoff(self, query: str) -> str:
        records = self.read()
        rid = self.resolve(records, query)
        rec = records[rid]
        require(rec["type"] == "followup", "handoff requires a followup")
        audit = self.audit(records)
        require(not audit["errors"] and rid not in audit["stale_transitive"],
                "cannot hand off stale, missing, or broken dependencies")
        lines = [f"# Follow-up: {rec['title']}", "", "**Prepared, not sent. No worker/job has been started.**",
                 f"Request ID: `{rid}`", f"Key: `{rec['key']}` | Thread: {rec['topic']}",
                 f"From: {rec['author']}", f"Requested of: {rec['requested_of']}",
                 f"Return to: {rec['reply_to']}", f"Gate: **{rec['gate']}**",
                 "", "## Question", rec["question"], "", "## Why this next", rec["why_next"],
                 "", "## Scope and starting evidence"]
        if audit['semantic_references']:
            lines += ['This thin packet uses source-bound semantic references for omitted historical evidence.',
                      'The original reproduction/first-step commands below are historical maintainer recipes; do not assume their raw inputs ship.',
                      'Use project/review.toml for portable checks. Request a named minimal witness if the assignment cannot proceed from the catalog and current source.']
        for cid in rec["claims"]:
            c = records[cid]
            lines += [f"### {c['key']} — {c['title']}", f"Claim ID: `{cid}`",
                      c["statement"], f"Classification: {c['classification']}",
                      "Scope: " + json.dumps(c["scope"], ensure_ascii=False),
                      "Coverage: " + json.dumps(c["coverage"], ensure_ascii=False),
                      "Limits: " + "; ".join(c["limits"])]
            for e in c["evidence"]:
                lines.append(f"- `{e['path']}` — SHA-256 `{e['sha256']}` ({e['role']})")
        lines += ["", "## First steps (text, never executed by the ledger)"]
        lines += [f"{i}. {v}" for i, v in enumerate(rec["first_steps"], 1)]
        lines += ["", "## Acceptance checks"] + [f"- {v}" for v in rec["acceptance"]]
        lines += ["", "## Return fields"] + [f"- {v}" for v in rec["return_fields"]]
        lines += ["", "## Resource boundary", json.dumps(rec["budget"], sort_keys=True),
                  "No ledger record grants compute, new games, training, data access, or promotion.",
                  "The prior four-attempt routing allowance is exhausted; do not restart it.",
                  "", "## Return to the originating reviewer", "```bash",
                  "python3 -B src/farm_neat/tooling/search_ledger.py template response > work/scratch/response.json",
                  "# Fill the response, including the request hash above and real evidence hashes.",
                  "python3 -B src/farm_neat/tooling/search_ledger.py add work/scratch/response.json",
                  "python3 -B src/farm_neat/tooling/search_ledger.py verify", "```",
                  "Return only changed/new paths and source/payload hashes through the existing delta intake.",
                  "The maintainer/operator must deliver that delta to the originating reviewer. The ledger",
                  "cannot message another chat or wake an agent. A response does not close a request:",
                  "an explicit review record is required; multiple/conflicting replies stay visible.",
                  "Author fields are self-declared labels, not authenticated signatures."]
        return "\n".join(lines) + "\n"


def template(kind: str) -> dict:
    common = {"version": 1, "type": kind, "key": "CHANGE-ME", "author": "ACTUAL-REVIEWER",
              "topic": "Q2", "title": "Replace with a bounded question/result", "evidence": []}
    if kind == "claim":
        common.update(classification="hypothesis", statement="Proposed, not yet measured.",
            scope={"question": "What changes?", "estimand": "Define the quantity and units.",
                   "domain": "Define a finite domain or explicitly state unknown.",
                   "information_boundary": "Actor-visible versus evaluator-only.",
                   "fixed": ["List controls."], "varied": ["List interventions."],
                   "excluded": ["List excluded effects."]},
            coverage={"evaluated": 0, "total": None, "exhaustive": False,
                      "unit": "case", "independent_units": 0,
                      "exposure": "UNMEASURED", "method": "Not run."},
            reuse=["What future work this could eliminate."], limits=["Unmeasured hypothesis."],
            reproduce=["Not run; replace with a bounded protocol."], depends_on=[], relations=[])
    elif kind == "followup":
        common.update(claims=["FULL-CLAIM-HASH"], question="One answerable follow-up question.",
            requested_of="integration-maintainer or explicitly assigned reviewer",
            reply_to="originating reviewer via existing delta intake", why_next="Which live decision this resolves.",
            gate="READ_ONLY", budget=dict.fromkeys(sorted(BUDGET_KEYS), 0), priority=2, cost="S",
            acceptance=["A falsifiable success/negative criterion."], return_fields=["Answer, evidence, limits."],
            first_steps=["Read cited artifacts; do not execute a historical campaign."])
    elif kind == "response":
        common.update(request="FULL-REQUEST-HASH", outcome="blocked", answer="Concrete result or blocker.",
                      limits="State the exact unfinished scope.", new_claims=[], supersedes=[])
    else:
        common.update(response="FULL-RESPONSE-HASH", decision="needs_work", reason="State the missing acceptance check.", supersedes=[])
    return common


def verification_summary(audit: dict) -> dict:
    """Keep failure details and epistemic scope while omitting successful binding rows."""
    return {
        "schema": "farm-search-ledger-verification-summary-v1",
        "full_result_sha256": digest(canonical(audit)),
        **{key: audit[key] for key in
           ("status", "records", "claims", "followups", "evidence_bindings_checked", "superseded_stale",
            "raw_bindings_verified", "errors", "warnings", "stale_direct", "stale_transitive",
            "meaning", "commands_executed")},
        "semantic_bindings_attested": len(audit["semantic_references"]),
        "historical_source_bindings_verified": len(audit.get("historical_source_bindings", [])),
        "details": "python3 -B src/farm_neat/tooling/search_ledger.py verify",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("verify")
    q.add_argument("--summary", action="store_true", help="Compact counts; preserve all warnings, failures and raw/semantic distinctions")
    sub.add_parser("guide", help="Print the full contribution and return protocol")
    sub.add_parser("render", help="Generate a compact Markdown view; never store a second authority")
    q = sub.add_parser("list"); q.add_argument("--query", default=""); q.add_argument("--topic", choices=["Q1", "Q2", "Q3", "Q4", "Q5"])
    q = sub.add_parser("show"); q.add_argument("ref")
    q = sub.add_parser("add"); q.add_argument("file", type=Path)
    q = sub.add_parser("template"); q.add_argument("kind", choices=sorted(KINDS))
    q = sub.add_parser("bind"); q.add_argument("path"); q.add_argument("--role", choices=sorted(ROLES), default="receipt")
    q = sub.add_parser("queue"); q.add_argument("--for", dest="recipient"); q.add_argument("--all", action="store_true")
    q = sub.add_parser("inbox"); q.add_argument("--for", dest="recipient", required=True)
    q = sub.add_parser("handoff"); q.add_argument("ref")
    q = sub.add_parser("impact"); q.add_argument("path")
    args = p.parse_args(argv)
    ledger = Ledger(args.root)
    try:
        if args.command == "guide":
            print(USER_GUIDE, end="")
            return 0
        elif args.command == "template":
            result = template(args.kind)
        elif args.command == "bind":
            result = {"path": args.path, "sha256": file_sha(safe_path(ledger.root, args.path)), "role": args.role}
        elif args.command == "add":
            require(args.file.stat().st_size <= MAX_RECORD, "record too large")
            result = ledger.add(parse(args.file.read_bytes()))
        elif args.command == "verify":
            result = ledger.audit()
            if args.summary:
                result = verification_summary(result)
        elif args.command == "handoff":
            print(ledger.handoff(args.ref), end="")
            return 0
        else:
            records = ledger.read()
            if args.command == "show":
                rid = ledger.resolve(records, args.ref)
                result = {"id": rid, "record": records[rid]}
            else:
                audit = ledger.audit(records)
                if args.command == "render":
                    lines = ["# Search knowledge ledger", "", "Generated view, not a second source of truth. Byte bindings are not proof or external review.",
                             "", "## Enumerated findings"]
                    for rid, r in sorted(records.items(), key=lambda kv: kv[1]["key"]):
                        if r["type"] != "claim": continue
                        c = r["coverage"]
                        lines += [f"### {r['key']} — {r['title']}",
                                  f"{r['topic']} | {r['classification']} | " + ("STALE" if rid in audit["stale_transitive"] else "SEMANTIC_REFERENCE; historical raw unavailable" if rid in audit["semantic_transitive"] else "BOUND, not independently reviewed"),
                                  r["statement"],
                                  f"Coverage: {c['evaluated']} / {c['total'] if c['total'] is not None else 'unknown'} {c['unit']}; exhaustive only in declared domain: {c['exhaustive']}.",
                                  "Reuse: " + " ".join(r["reuse"]), "Limits: " + " ".join(r["limits"]),
                                  f"Record: `{rid}`", ""]
                    lines += ["## Follow-up queue", "", "Prepared requests only; no messages sent or jobs launched."]
                    requests = [(rid, r) for rid, r in records.items() if r["type"] == "followup"]
                    for rid, r in sorted(requests, key=lambda kv: (kv[1]["gate"] != "READ_ONLY", kv[1]["priority"], kv[1]["key"])):
                        lines += [f"### {r['key']} — {r['title']}",
                                  f"{Ledger.request_status(records, rid)} | {r['gate']} | priority {r['priority']} | cost {r['cost']}",
                                  r["question"], f"Requested of: {r['requested_of']}", f"Return to: {r['reply_to']}",
                                  f"Generate packet: `python3 -B src/farm_neat/tooling/search_ledger.py handoff {r['key']}`", ""]
                    print("\n".join(lines))
                    return int(bool(audit["errors"] or audit["stale_direct"]))
                elif args.command == "list":
                    result = [{"key": r["key"], "id": rid, "topic": r["topic"], "title": r["title"],
                        "classification": r["classification"], "coverage": r["coverage"],
                        "validity": "STALE" if rid in audit["stale_transitive"] else "SEMANTIC_REFERENCE" if rid in audit["semantic_transitive"] else "BOUND_UNREVIEWED",
                        "scope_id": scope_id(r), "relations": r.get("relations", []),
                        "incoming_relations": [{"from": oid, "kind": rel["kind"]} for oid, other in records.items()
                           if other["type"] == "claim" for rel in other.get("relations", []) if rel["target"] == rid]}
                        for rid, r in records.items()
                        if r["type"] == "claim" and (not args.topic or r["topic"] == args.topic)
                        and args.query.lower() in canonical(r).decode().lower()]
                elif args.command == "queue":
                    result = [{"key": r["key"], "id": rid, "topic": r["topic"], "question": r["question"],
                        "requested_of": r["requested_of"], "priority": r["priority"], "cost": r["cost"],
                        "state": Ledger.request_status(records, rid), "gate": r["gate"],
                        "validity": "STALE" if rid in audit["stale_transitive"] else "SEMANTIC_REFERENCE" if rid in audit["semantic_transitive"] else "BOUND_UNREVIEWED"}
                        for rid, r in records.items() if r["type"] == "followup"
                        and (args.all or Ledger.request_status(records, rid) != "ANSWER_ACCEPTED")
                        and (not args.recipient or args.recipient in {r["author"], r["requested_of"]})]
                    result.sort(key=lambda v: (v["gate"] != "READ_ONLY", v["priority"], v["cost"], v["key"]))
                elif args.command == "inbox":
                    result = [{"response_id": rid, "response": r, "state": Ledger.request_status(records, r["request"])}
                        for rid, r in records.items() if r["type"] == "response" and r["request"] in records
                        and records[r["request"]]["author"] == args.recipient]
                else:
                    affected = {rid for rid, r in records.items() if any(e["path"] == args.path for e in r["evidence"])}
                    changed = True
                    while changed:
                        n = len(affected)
                        affected.update(rid for rid, r in records.items()
                                        if any(ref in affected for ref, _ in links(r, dependent_only=True)))
                        changed = len(affected) != n
                    result = [{"id": rid, "key": records[rid]["key"], "type": records[rid]["type"]}
                              for rid in sorted(affected)]
        print(canonical(result).decode(), end="")
        return int(args.command == "verify" and bool(result["errors"] or result["stale_direct"]))
    except (ValueError, OSError, KeyError, TypeError, IndexError, EOFError) as exc:
        print(f"Search ledger refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
