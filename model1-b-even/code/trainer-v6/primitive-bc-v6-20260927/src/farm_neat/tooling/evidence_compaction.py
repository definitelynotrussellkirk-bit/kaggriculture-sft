"""Bounded semantic evidence for the canonical review packet; never executes code.

The producer verifies raw bytes and literal selectors. An extracted packet can
verify the catalog and shipped sources only: omitted sources are attestations,
not a replay or a cryptographic proof of the reported measurements.
"""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
from pathlib import Path
import re

from farm_neat.tooling import review_intake as safe

CATALOG = 'work/runs/review-package/EVIDENCE.jsonl'
POLICY = 'work/runs/review-package/EVIDENCE-POLICY.json'
MANIFEST = 'work/runs/review-package/MANIFEST.json'
SCHEMA = 'farm-semantic-evidence-v1'
TIERS = {'SEMANTIC_ONLY', 'WITNESS', 'ACTIVE_DEPENDENCY', 'MAINTAINER_ONLY', 'UNKNOWN'}
RETAINED = {'WITNESS', 'ACTIVE_DEPENDENCY'}
MAX_RECEIPT = 24_000
MAX_CATALOG = 2_000_000
MAX_POLICY = 12_000_000
MAX_SOURCE_JSON = 64_000_000
HEX = re.compile(r'[0-9a-f]{64}')


def path(root: Path, name: str) -> Path:
    safe.relative_path(name)
    p = root
    for part in name.split('/'):
        p = p / part
        if p.is_symlink():
            raise ValueError('symlink evidence path: ' + name)
    return p


def identity(root: Path, name: str) -> dict:
    p = path(root, name)
    if not p.is_file():
        raise ValueError('missing regular evidence file: ' + name)
    h = hashlib.sha256()
    count = 0
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block); count += len(block)
    return {'path': name, 'sha256': h.hexdigest(), 'bytes': count}


def binding(value: dict) -> None:
    if not isinstance(value, dict) or set(value) != {'path', 'sha256', 'bytes'}:
        raise ValueError('exact path/hash/bytes identity required')
    safe.relative_path(value['path'])
    if (not isinstance(value['sha256'], str) or not HEX.fullmatch(value['sha256'])
            or type(value['bytes']) is not int or value['bytes'] < 0):
        raise ValueError('invalid evidence identity')


def bounded(value, depth=0, budget=None) -> None:
    """No unbounded arrays, nested matrices, or giant strings renamed summaries."""
    budget = [0] if budget is None else budget
    budget[0] += 1
    if depth > 8 or budget[0] > 1000:
        raise ValueError('semantic value exceeds depth/node budget')
    if isinstance(value, dict):
        if len(value) > 64 or any(not isinstance(k, str) or len(k) > 512 for k in value):
            raise ValueError('semantic object exceeds key budget')
        for v in value.values(): bounded(v, depth + 1, budget)
    elif isinstance(value, list):
        if len(value) > 64:
            raise ValueError('semantic array exceeds budget; use count/digest or explicit witness')
        for v in value: bounded(v, depth + 1, budget)
    elif isinstance(value, str) and len(value.encode()) > 5000:
        raise ValueError('semantic string exceeds budget')
    safe.canonical(value)  # Finite, JSON-compatible values only.


def seal(record: dict) -> dict:
    result = {k: v for k, v in record.items() if k != 'content_sha256'}
    return {**result, 'content_sha256': safe.digest(safe.canonical(result))}


def validate(record: dict, *, publish=False) -> None:
    required = {'schema', 'id', 'kind', 'question', 'delta', 'control', 'result',
                'interpretation', 'limits', 'decision', 'repro', 'source',
                'raw_policy', 'raw_reason', 'needs_semantic_review', 'reviewed_by',
                'content_sha256'}
    if not isinstance(record, dict) or set(record) != required or record['schema'] != SCHEMA:
        raise ValueError('closed semantic receipt schema required')
    if not isinstance(record['id'], str) or not re.fullmatch(r'E-[A-Za-z0-9_.-]{1,100}', record['id']):
        raise ValueError('distinct semantic evidence id required')
    if record['kind'] not in {'experiment', 'audit', 'counterexample', 'benchmark', 'migration'}:
        raise ValueError('invalid semantic kind')
    for key in ('question', 'delta', 'control'):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError('semantic receipt requires ' + key)
    if not isinstance(record['result'], dict) or not record['result']:
        raise ValueError('literal result object required separately from interpretation')
    for key in ('interpretation', 'decision', 'reviewed_by'):
        if not isinstance(record[key], str): raise ValueError('text required: ' + key)
    for key in ('limits', 'repro'):
        if not isinstance(record[key], list) or not record[key] or any(not isinstance(x, str) or not x.strip() for x in record[key]):
            raise ValueError('explicit nonempty ' + key)
    if type(record['needs_semantic_review']) is not bool:
        raise ValueError('needs_semantic_review must be boolean')
    if publish and (record['needs_semantic_review'] or any(not record[k].strip() for k in ('interpretation', 'decision', 'reviewed_by'))):
        raise ValueError('UNKNOWN: receipt needs semantic review: ' + record['id'])
    if record['raw_policy'] not in {'omit_from_review', 'retain_witness', 'active_dependency'}:
        raise ValueError('explicit raw policy required')
    if not isinstance(record['raw_reason'], str) or not record['raw_reason'].strip():
        raise ValueError('explicit raw reason required')
    if not isinstance(record['source'], list) or not record['source']:
        raise ValueError('semantic provenance sources required')
    for item in record['source']: binding(item)
    names = [x['path'] for x in record['source']]
    if len(set(names)) != len(names): raise ValueError('duplicate semantic source')
    bounded(record)
    if len(safe.canonical(record)) > MAX_RECEIPT:
        raise ValueError('semantic receipt exceeds byte budget')
    if seal(record) != record:
        raise ValueError('semantic receipt content hash mismatch')


def pointer(value, selector: str):
    if selector == '': return value
    if not isinstance(selector, str) or not selector.startswith('/'):
        raise ValueError('JSON pointer required')
    for token in selector[1:].split('/'):
        if re.search(r'~(?![01])', token): raise ValueError('invalid pointer escape')
        key = token.replace('~1', '/').replace('~0', '~')
        if isinstance(value, list):
            if not re.fullmatch(r'0|[1-9][0-9]*', key): raise ValueError('invalid array index')
            value = value[int(key)]
        else: value = value[key]
    return value


def literal(root: Path, name: str, selector: str, op='value'):
    p = path(root, name)
    with (gzip.open if p.suffix == '.gz' else open)(p, 'rb') as stream:
        raw = stream.read(MAX_SOURCE_JSON + 1)
    if len(raw) > MAX_SOURCE_JSON: raise ValueError('literal source exceeds JSON budget')
    value = pointer(safe.parse_json(raw), selector)
    if op == 'length': value = len(value)
    elif op == 'digest': value = {'sha256': safe.digest(safe.canonical(value)), 'json_bytes': len(safe.canonical(value))}
    elif op != 'value': raise ValueError('unknown literal operation')
    bounded(value)
    return value


def candidate(root: Path, name: str) -> dict:
    """Mechanical inspection candidate, deliberately unable to pass publish."""
    source = identity(root, name)
    result = {'source_bytes': source['bytes']}
    p = path(root, name)
    if p.suffix == '.json':
        if p.stat().st_size > MAX_SOURCE_JSON: raise ValueError('candidate JSON exceeds budget')
        data = safe.parse_json(p.read_bytes())
        if isinstance(data, dict):
            result['top_level_fields'] = len(data)
            result['fields'] = {}
            # No filename/field name is taken as evidence of scientific meaning.
            for key in sorted(data):
                if len(key) > 512: continue
                value = data[key]
                if isinstance(value, (dict, list)):
                    result['fields'][key] = {'omitted': True, 'items': len(value), 'sha256': safe.digest(safe.canonical(value))}
                elif isinstance(value, str) and len(value.encode()) > 512:
                    result['fields'][key] = {'omitted': True, 'utf8_bytes': len(value.encode()), 'sha256': safe.digest(value.encode())}
                else: result['fields'][key] = value
                if len(result['fields']) >= 32: break
            result['omitted_field_count'] = len(data) - len(result['fields'])
    record = seal({'schema': SCHEMA, 'id': 'E-candidate-' + source['sha256'][:16],
        'kind': 'migration', 'question': 'UNREVIEWED source extraction', 'delta': 'UNKNOWN',
        'control': 'UNKNOWN', 'result': result, 'interpretation': '', 'limits': ['Literal extraction only; no scientific assessment.'],
        'decision': '', 'repro': ['Inspect the source and specify a bounded reproduction protocol.'],
        'source': [source], 'raw_policy': 'omit_from_review', 'raw_reason': 'Candidate only; producer evidence must remain unchanged.',
        'needs_semantic_review': True, 'reviewed_by': ''})
    validate(record)
    return record


def encode(records: list[dict]) -> bytes:
    ids = [r['id'] for r in records]
    if len(set(ids)) != len(ids): raise ValueError('duplicate semantic id')
    for r in records: validate(r)
    raw = b''.join(safe.canonical(r) + b'\n' for r in sorted(records, key=lambda r: r['id']))
    if len(raw) > MAX_CATALOG: raise ValueError('catalog exceeds budget')
    return raw


def catalog(root: Path, *, publish=True, verify_sources=False) -> list[dict]:
    p = path(root, CATALOG)
    if p.stat().st_size > MAX_CATALOG: raise ValueError('catalog exceeds budget')
    raw = p.read_bytes()
    records = [safe.parse_json(line) for line in raw.splitlines()]
    if encode(records) != raw: raise ValueError('noncanonical semantic catalog')
    checked = {}
    copies = historical_sources(root, records)
    for r in records:
        validate(r, publish=publish)
        sources = {s['path']: s for s in r['source']}
        if verify_sources:
            for name, source in sources.items():
                key = (name, source['sha256'], source['bytes'])
                if key not in checked:
                    actual = identity(root, copies.get((name, source['sha256']), name))
                    checked[key] = {**actual, 'path': name}
                if checked[key] != source: raise ValueError('semantic source drift: ' + name)
            for observation in r['result'].get('observations', []):
                if set(observation) != {'source', 'pointer', 'op', 'value'} or observation['source'] not in sources:
                    raise ValueError('literal observation must name a bound source')
                if safe.canonical(literal(root, observation['source'], observation['pointer'], observation['op'])) != safe.canonical(observation['value']):
                    raise ValueError('literal result mismatch: ' + r['id'])
    return records


def historical_sources(root: Path, records=None) -> dict:
    """Exact source copies explicitly admitted by reviewed migration receipts.

    This preserves old source identity, not compatibility with current code.
    It cannot redirect outcomes, selectors, checkpoints or arbitrary raw evidence.
    Copies are byte-checked even when the original path still exists.
    """
    if records is None:
        if not path(root, CATALOG).exists(): return {}
        records = catalog(root)
    result = {}
    for record in records:
        entries = record.get('result', {}).get('historical_sources', [])
        if not entries: continue
        validate(record, publish=True)
        if record['kind'] != 'migration' or not isinstance(entries, list):
            raise ValueError('historical source copies require reviewed migration')
        sources = {s['path']: s for s in record['source']}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {'path', 'sha256', 'bytes', 'copy'}:
                raise ValueError('closed historical source copy required')
            original = {k: entry[k] for k in ('path', 'sha256', 'bytes')}
            binding(original)
            if not entry['path'].startswith(('src/', 'tests/')) or not entry['path'].endswith('.py'):
                raise ValueError('only historical Python source may use a copy')
            copy = entry['copy']
            if not isinstance(copy, str) or not copy.startswith('work/') or not copy.endswith('.py'):
                raise ValueError('historical source copy must be explicit work Python source')
            expected = {**original, 'path': copy}
            if sources.get(copy) != expected or identity(root, copy) != expected:
                raise ValueError('historical source copy bytes/binding mismatch')
            key = (entry['path'], entry['sha256'])
            if key in result and result[key] != copy:
                raise ValueError('ambiguous historical source copy')
            result[key] = copy
    return result


def omitted_digest(items: list[dict]) -> str:
    ordered = sorted(({k: i[k] for k in ('path', 'sha256', 'bytes')} for i in items), key=lambda i: i['path'])
    if len({i['path'] for i in ordered}) != len(ordered): raise ValueError('duplicate omitted path')
    for i in ordered: binding(i)
    return safe.digest(b''.join(safe.canonical(i) + b'\n' for i in ordered))


def policy(root: Path) -> list[dict]:
    p = path(root, POLICY)
    if p.stat().st_size > MAX_POLICY: raise ValueError('evidence policy exceeds budget')
    value = safe.parse_json(p.read_bytes())
    if value.get('schema') != 'farm-review-evidence-policy-v1': raise ValueError('unknown policy schema')
    items = value['artifacts']
    if not isinstance(items, list) or len(items) > 30_000: raise ValueError('bounded artifact policy required')
    seen = set()
    for item in items:
        if set(item) != {'path', 'sha256', 'bytes', 'tier', 'reason', 'receipts'}:
            raise ValueError('closed artifact disposition required')
        binding({k: item[k] for k in ('path', 'sha256', 'bytes')})
        name = item['path']
        if not name.startswith('work/') or name in {CATALOG, POLICY, MANIFEST} or name in seen:
            raise ValueError('distinct producer work artifact required: ' + name)
        seen.add(name)
        if item['tier'] not in TIERS or item['tier'] == 'UNKNOWN':
            raise ValueError('UNKNOWN evidence blocks publication: ' + name)
        if not isinstance(item['reason'], str) or not item['reason'].strip():
            raise ValueError('explicit evidence/witness reason required: ' + name)
        if item['tier'] in RETAINED and Path(name).suffix.lower() in {'.pt', '.pth', '.zip', '.tar', '.tgz', '.whl', '.so', '.pkl', '.pickle'}:
            raise ValueError('weights, archives and executable binary payloads are outside this source packet: ' + name)
        if not isinstance(item['receipts'], list) or any(not isinstance(r, str) for r in item['receipts']):
            raise ValueError('receipt identifiers required')
        if item['tier'] in {'SEMANTIC_ONLY', 'WITNESS'} and not item['receipts']:
            raise ValueError('semantic-only/witness requires a reviewed receipt: ' + name)
    return items


def producer_view(root: Path) -> dict:
    records = catalog(root, verify_sources=True)
    by_id = {r['id']: r for r in records}
    items = policy(root)
    for item in items:
        if identity(root, item['path']) != {k: item[k] for k in ('path', 'sha256', 'bytes')}:
            raise ValueError('classified artifact drift: ' + item['path'])
        for rid in item['receipts']:
            if rid not in by_id or item['path'] not in {s['path'] for s in by_id[rid]['source']}:
                raise ValueError('disposition receipt does not bind artifact: ' + item['path'])
    retained = [i for i in items if i['tier'] in RETAINED]
    omitted = [i for i in items if i['tier'] not in RETAINED]
    categories = Counter()
    for item in omitted:
        # These categories report byte accounting only; they never decide retention.
        categories['/'.join(item['path'].split('/')[:3])] += item['bytes']
    witnesses = [i for i in retained if i['tier'] == 'WITNESS']
    return {'schema': 'farm-review-evidence-view-v1', 'catalog': identity(root, CATALOG),
            'semantic_receipts': len(records), 'unclassified': 0,
            'retained': retained, 'witness_count': len(witnesses), 'witness_bytes': sum(i['bytes'] for i in witnesses),
            'omitted_count': len(omitted), 'omitted_bytes': sum(i['bytes'] for i in omitted),
            'omitted_set_sha256': omitted_digest(omitted),
            'omitted_digest_encoding': 'SHA256 of sorted-by-path compact canonical UTF-8 JSON {path,sha256,bytes} lines, each with LF',
            'top_omitted_categories': dict(categories.most_common(10)),
            'top_retained': sorted(retained, key=lambda i: (-i['bytes'], i['path']))[:10],
            'producer_policy': identity(root, POLICY),
            'scope': 'Producer verified raw identities and selected literals. Omitted bytes stay maintainer-side; no deletion or experiment executed.'}


def packet_view(root: Path) -> dict:
    """Use a producer policy or the original, self-consistent extracted manifest.

    The fallback is intentionally unavailable to a producer that still has its
    policy: a stale policy must fail, not quietly fall back to the last release.
    """
    if path(root, POLICY).exists(): return producer_view(root)
    p = path(root, MANIFEST)
    if not p.is_file(): raise ValueError('missing explicit evidence policy or extracted manifest')
    if p.stat().st_size > MAX_POLICY: raise ValueError('manifest exceeds budget')
    manifest = safe.parse_json(p.read_bytes())
    view = manifest.get('evidence')
    if manifest.get('mode') != 'SOURCE_AND_SEMANTIC_EVIDENCE_REVIEW' or not isinstance(view, dict):
        raise ValueError('thin extracted manifest required')
    if view.get('unclassified') != 0 or view.get('catalog') != identity(root, CATALOG):
        raise ValueError('extracted semantic catalog mismatch')
    # Verify current dependencies, never silently rebind changed receipts.
    for name, expected in manifest['members'].items():
        if identity(root, name) != {'path': name, **expected}:
            raise ValueError('extracted member hash mismatch: ' + name)
    ids = {r['id']: r for r in catalog(root)}
    for item in view['retained']:
        if (set(item) != {'path','sha256','bytes','tier','reason','receipts'}
                or item['tier'] not in RETAINED or not isinstance(item['reason'],str) or not item['reason'].strip()):
            raise ValueError('UNKNOWN or unjustified extracted work disposition')
        binding({k:item[k] for k in ('path','sha256','bytes')})
        if not isinstance(item['receipts'],list) or (item['tier']=='WITNESS' and not item['receipts']):
            raise ValueError('extracted witness requires a receipt')
        for rid in item['receipts']:
            if rid not in ids or item['path'] not in {s['path'] for s in ids[rid]['source']}:
                raise ValueError('unbound extracted work disposition')
        if manifest['members'].get(item['path']) != {k:item[k] for k in ('sha256','bytes')}:
            raise ValueError('extracted work identity mismatch')
    witnesses = [i for i in view['retained'] if i['tier']=='WITNESS']
    if (view['witness_count'] != len(witnesses) or view['witness_bytes'] != sum(i['bytes'] for i in witnesses)
            or view['semantic_receipts'] != len(ids)):
        raise ValueError('extracted evidence accounting mismatch')
    wanted = {CATALOG, *[i['path'] for i in view['retained']]}
    actual = {n for n in manifest['members'] if n.startswith('work/')}
    if wanted != actual: raise ValueError('unclassified extracted work member')
    return view


def semantic_references(root: Path) -> dict:
    """Only an original thin extraction may substitute attestations for raw joins."""
    if path(root, POLICY).exists() or not path(root, MANIFEST).exists(): return {}
    packet_view(root)
    result = {}
    for r in catalog(root):
        for s in r['source']:
            key = (s['path'], s['sha256'])
            result.setdefault(key, []).append(r)
    return result
