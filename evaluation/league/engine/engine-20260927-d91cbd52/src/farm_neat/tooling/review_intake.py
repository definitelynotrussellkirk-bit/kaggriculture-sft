"""Bounded, non-executing transport scan for the canonical review packager.

Adapted from the owner's FARM_INTAKE_REDESIGN_KIT. This module never extracts,
imports returned code, applies patches, accepts science, or removes an arrival.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import tempfile
import zipfile

MAX_ARCHIVE = 100_000_000
MAX_EXPANDED = 300_000_000
MAX_MEMBER = 64_000_000
MAX_MEMBERS = 10_000
MAX_METADATA = 1_000_000
BASELINE_LIMIT = 500_000_000  # Read legacy packets; new publication has tighter caps.
ROOTS = {'src','tests','project','docs','review','work','kagglewriteup','payload','overlay'}
ARCHIVES = ('.zip','.tar','.tar.gz','.tgz','.7z','.rar')
CODE = {'.py','.rs','.c','.cpp','.h','.cu','.cuh','.toml'}
DISPATCH_REGISTRY_SCHEMA = 'farm-review-dispatch-registry-v1'
DISPATCH_REGISTRY_FILE = 'DISPATCH-REGISTRY.json'
BASELINE_DIR = 'baselines'
DEFAULT_LOST_AFTER_SECONDS = 24 * 60 * 60
LOST_AFTER_RULE = ('An unresolved exact issuance is LOST when now >= lost_after; reissuing the same '
                   'task supersedes the earlier issuance instead of counting both. Transport status never '
                   'accepts or rejects the task.')
IDENTITY_SCHEME = {
    'logical_work': 'task_id',
    'source': 'full source_manifest_sha256 identifies one immutable review ZIP baseline',
    'issuance': 'ticket = sha256(source_manifest, dispatch, slot, task)[:32]',
    'slot': 'ZIP-local alias only; never a cross-ZIP identity',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def bad(_): raise ValueError('nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)



def _parse_time(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('timestamp must be nonempty ISO-8601 text')
    text = value.strip()
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError('timestamp must be ISO-8601') from error
    if parsed.tzinfo is None:
        raise ValueError('timestamp must include a timezone')
    return parsed.astimezone(timezone.utc)


def _time_text(value):
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _manifest_dispatch(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size > 16_000_000:
        raise ValueError('bounded regular source manifest required')
    raw = p.read_bytes()
    value = parse_json(raw)
    if value.get('schema') != 'farmneat-source-review-package-v1' or not isinstance(value.get('members'), dict):
        raise ValueError('original outgoing manifest schema required')
    dispatch = value.get('continuation', {}).get('dispatch')
    if not isinstance(dispatch, dict):
        raise ValueError('source manifest has no dispatch identity')
    dispatch_sha = dispatch.get('sha256')
    if not isinstance(dispatch_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', dispatch_sha):
        raise ValueError('source manifest dispatch sha256 must be a full hash')
    shipped = value['members'].get('project/dispatch.toml')
    if shipped is not None and (not isinstance(shipped, dict) or shipped.get('sha256') != dispatch_sha):
        raise ValueError('source manifest dispatch identity differs from shipped project/dispatch.toml')
    slots = dispatch.get('slots')
    if not isinstance(slots, list):
        raise ValueError('source manifest dispatch slots must be a list')
    normalized, numbers, tasks = [], set(), set()
    for row in slots:
        if not isinstance(row, dict):
            raise ValueError('source manifest dispatch slot must be an object')
        number, task = row.get('number'), row.get('task')
        if type(number) is not int or number <= 0 or not isinstance(task, str) or not task:
            raise ValueError('source manifest dispatch slot needs positive number and task')
        if number in numbers or task in tasks:
            raise ValueError('source manifest dispatch slots must be unique by number and task')
        numbers.add(number); tasks.add(task)
        normalized.append({'number': number, 'task': task})
    normalized.sort(key=lambda row: row['number'])
    return raw, value, dispatch_sha, normalized


def _dispatch_ticket(source_sha, dispatch_sha, slot, task):
    """Existing v1 ticket derivation; kept byte-for-byte compatible with review_package."""
    return digest(f'{source_sha}:{dispatch_sha}:{slot}:{task}'.encode())[:32]


def source_dispatch_identity(manifest):
    """Describe immutable source + logical task + exact issuance identities.

    task_id survives ZIP rebuilds.  source_manifest_sha256 identifies one baseline.
    ticket identifies one issuance on that baseline.  A slot is only a local alias.
    """
    raw, _, dispatch_sha, slots = _manifest_dispatch(manifest)
    source_sha = digest(raw)
    return {
        'schema': 'farm-review-source-dispatch-identity-v1',
        'source_manifest_sha256': source_sha,
        'source_identity': f'FARM-REVIEW@{source_sha}',
        'dispatch_sha256': dispatch_sha,
        'slots': slots,
        'identity_scheme': dict(IDENTITY_SCHEME),
    }


def resolve_source_dispatch(manifest, *, slot=None, task=None):
    """Resolve one source-local alias and refuse stale slot/task requests with source identity."""
    if (slot is None) == (task is None):
        raise ValueError('give exactly one of a slot number or a task id')
    identity = source_dispatch_identity(manifest)
    matches = [row for row in identity['slots']
               if row['number'] == slot] if slot is not None else [row for row in identity['slots'] if row['task'] == task]
    if not matches:
        listing = ', '.join(f"{row['number']}={row['task']}" for row in identity['slots']) or 'no reviewer slots'
        requested = f'slot {slot}' if slot is not None else f'task {task}'
        raise ValueError(f"this ZIP source_manifest_sha256={identity['source_manifest_sha256']} does not dispatch "
                         f"{requested}; it dispatches {listing}")
    row = matches[0]
    ticket = _dispatch_ticket(identity['source_manifest_sha256'], identity['dispatch_sha256'], row['number'], row['task'])
    return {
        'source_manifest_sha256': identity['source_manifest_sha256'],
        'source_identity': identity['source_identity'],
        'dispatch_sha256': identity['dispatch_sha256'],
        'slot': row['number'],
        'task_id': row['task'],
        'ticket': ticket,
        'return_name': f"FARM-RETURN-{row['task']}-{ticket[:8]}.zip",
    }


def _registry_paths(registry):
    root = Path(registry)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError('dispatch registry must be a directory')
    return root, root / DISPATCH_REGISTRY_FILE, root / BASELINE_DIR


def _empty_registry():
    return {
        'schema': DISPATCH_REGISTRY_SCHEMA,
        'default_lost_after_seconds': DEFAULT_LOST_AFTER_SECONDS,
        'lost_after_rule': LOST_AFTER_RULE,
        'identity_scheme': dict(IDENTITY_SCHEME),
        'sources': {},
        'tickets': {},
    }


def _load_registry(registry, *, missing_ok=False):
    root, state_path, _ = _registry_paths(registry)
    if not state_path.exists():
        if missing_ok:
            return root, _empty_registry()
        raise ValueError(f'dispatch registry missing {DISPATCH_REGISTRY_FILE}')
    if state_path.is_symlink() or not state_path.is_file() or state_path.stat().st_size > 16_000_000:
        raise ValueError('bounded regular dispatch registry required')
    value = parse_json(state_path.read_bytes())
    if (not isinstance(value, dict) or value.get('schema') != DISPATCH_REGISTRY_SCHEMA
            or not isinstance(value.get('sources'), dict) or not isinstance(value.get('tickets'), dict)):
        raise ValueError('invalid dispatch registry schema')
    if value.get('lost_after_rule') != LOST_AFTER_RULE:
        raise ValueError('dispatch registry lost-after rule mismatch')
    return root, value


def _atomic_write(target, payload):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + target.name + '-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_registry(registry, value):
    root, state_path, _ = _registry_paths(registry)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write(state_path, canonical(value) + b'\n')


def _registry_baseline(root, registry, source_sha):
    source = registry['sources'].get(source_sha)
    if source is None:
        return None
    rel = source.get('manifest')
    if not isinstance(rel, str):
        raise ValueError('dispatch registry source has no manifest path')
    relative_path(rel)
    target = root / rel
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError('dispatch registry baseline escapes registry root') from error
    base = baseline_inventory(target)
    if base['baseline_manifest_sha256'] != source_sha:
        raise ValueError('dispatch registry baseline hash mismatch')
    return base


def register_dispatch_manifest(manifest, registry, *, issued_at, lost_after_seconds=DEFAULT_LOST_AFTER_SECONDS):
    """Retain one exact outgoing manifest and record all exact ticket issuances in its dispatch.

    This replaces hand copies of old ZIPs with hash-addressed manifests.  Reissuing the
    same logical task supersedes an older unresolved ticket without deleting history.
    """
    if type(lost_after_seconds) is not int or lost_after_seconds <= 0:
        raise ValueError('lost_after_seconds must be a positive integer')
    issued = _parse_time(issued_at)
    raw, _, dispatch_sha, slots = _manifest_dispatch(manifest)
    # Validate member identities before persisting the baseline bytes.
    base = baseline_inventory(manifest)
    source_sha = base['baseline_manifest_sha256']
    root, state = _load_registry(registry, missing_ok=True)
    _, _, baseline_dir = _registry_paths(root)
    baseline_rel = f'{BASELINE_DIR}/{source_sha}.json'
    baseline_path = baseline_dir / f'{source_sha}.json'
    if baseline_path.exists():
        if baseline_path.is_symlink() or baseline_path.read_bytes() != raw:
            raise ValueError('registered baseline bytes conflict with source manifest hash')
    else:
        _atomic_write(baseline_path, raw)
    source = {
        'source_manifest_sha256': source_sha,
        'source_identity': f'FARM-REVIEW@{source_sha}',
        'dispatch_sha256': dispatch_sha,
        'manifest': baseline_rel,
        'registered_at': _time_text(issued),
        'slots': slots,
    }
    existing_source = state['sources'].get(source_sha)
    if existing_source is not None:
        comparable = {k: existing_source.get(k) for k in ('source_manifest_sha256','source_identity','dispatch_sha256','manifest','slots')}
        expected = {k: source[k] for k in comparable}
        if comparable != expected:
            raise ValueError('dispatch registry source identity conflict')
        source = existing_source
    else:
        state['sources'][source_sha] = source

    made = []
    for row in slots:
        exact = resolve_source_dispatch(manifest, slot=row['number'])
        ticket = exact['ticket']
        current = state['tickets'].get(ticket)
        if current is None:
            current = {
                **exact,
                'issued_at': _time_text(issued),
                'lost_after': _time_text(issued + timedelta(seconds=lost_after_seconds)),
                'lost_after_seconds': lost_after_seconds,
                'returned_at': None,
                'superseded_by': None,
                'superseded_at': None,
            }
            # Registration can happen out of order. The newest unresolved issuance is current.
            unresolved = [old for old in state['tickets'].values()
                          if old.get('task_id') == row['task'] and old.get('returned_at') is None
                          and old.get('superseded_by') is None]
            newer = [old for old in unresolved if _parse_time(old['issued_at']) > issued]
            if newer:
                newest = max(newer, key=lambda old: (_parse_time(old['issued_at']), old['ticket']))
                current['superseded_by'] = newest['ticket']
                current['superseded_at'] = newest['issued_at']
            else:
                for old in unresolved:
                    old['superseded_by'] = ticket
                    old['superseded_at'] = _time_text(issued)
            state['tickets'][ticket] = current
        else:
            for key in ('source_manifest_sha256','dispatch_sha256','slot','task_id','return_name'):
                if current.get(key) != exact.get(key):
                    raise ValueError('dispatch registry ticket identity conflict')
        made.append(dict(current))
    _save_registry(root, state)
    return {
        'schema': 'farm-review-dispatch-registration-v1',
        'source': state['sources'][source_sha],
        'tickets': made,
        'lost_after_rule': LOST_AFTER_RULE,
        'registry': str(root / DISPATCH_REGISTRY_FILE),
    }


def _ticket_status(record, now):
    if record.get('returned_at') is not None and record.get('superseded_by') is not None:
        return 'RETURNED_AFTER_REISSUE'
    if record.get('returned_at') is not None:
        return 'RETURNED'
    if record.get('superseded_by') is not None:
        return 'SUPERSEDED'
    return 'LOST' if now >= _parse_time(record['lost_after']) else 'OUTSTANDING'


def outstanding_tickets(registry, *, now):
    """Read-only transport lifecycle view; LOST is a reissue signal, never task acceptance state."""
    instant = _parse_time(now)
    _, state = _load_registry(registry)
    rows = []
    for ticket, record in sorted(state['tickets'].items(), key=lambda item: (item[1]['issued_at'], item[0])):
        rows.append({**record, 'status': _ticket_status(record, instant)})
    active = [row for row in rows if row.get('returned_at') is None and row.get('superseded_by') is None]
    return {
        'schema': 'farm-review-outstanding-tickets-v1',
        'as_of': _time_text(instant),
        'lost_after_rule': LOST_AFTER_RULE,
        'identity_scheme': dict(IDENTITY_SCHEME),
        'tickets': rows,
        'active_ticket_count': len(active),
        'active_task_count': len({row['task_id'] for row in active}),
        'lost_ticket_count': sum(row['status'] == 'LOST' for row in rows),
    }


def reconcile_returns(report, registry, *, received_at):
    """Record transport arrival by exact ticket only; never infer scientific acceptance/rejection."""
    received = _parse_time(received_at)
    root, state = _load_registry(registry)
    matched, unmatched = [], []
    for row in report.get('returns', []):
        ticket = row.get('ticket')
        record = state['tickets'].get(ticket) if isinstance(ticket, str) else None
        if record is None:
            unmatched.append(ticket)
            continue
        if row.get('assignment_id') != record['task_id']:
            raise ValueError('return task identity does not match registered ticket')
        if row.get('claimed_baseline_manifest_sha256') != record['source_manifest_sha256']:
            raise ValueError('return source identity does not match registered ticket')
        if record.get('returned_at') is None:
            record['returned_at'] = _time_text(received)
        matched.append(ticket)
    _save_registry(root, state)
    return {
        'schema': 'farm-review-return-reconciliation-v1',
        'matched_tickets': matched,
        'unmatched_tickets': unmatched,
        'scientific_intake': 'PENDING',
        'outstanding': outstanding_tickets(root, now=_time_text(received)),
    }


def relative_path(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or any(ord(c) < 32 for c in name)):
        raise ValueError(f'unsafe relative path: {name!r}')
    p = PurePosixPath(name)
    if p.is_absolute() or p.as_posix() != name or any(x in {'.','..'} for x in p.parts):
        raise ValueError(f'unsafe relative path: {name!r}')
    return name


def safe_member(info):
    relative_path(info.filename[:-1] if info.is_dir() else info.filename)
    mode = stat.S_IFMT(info.external_attr >> 16)
    if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError('special/symlink ZIP member refused: ' + info.filename)
    if (info.is_dir() and mode == stat.S_IFREG) or (not info.is_dir() and mode == stat.S_IFDIR):
        raise ValueError('inconsistent ZIP member type: ' + info.filename)
    if info.flag_bits & 1: raise ValueError('encrypted ZIP member refused')


def normalized_names(names):
    """Strip a transport wrapper, never a lone repository component such as src/."""
    parts = [PurePosixPath(n).parts for n in names]
    wrapper = ''
    if parts and all(len(p) > 1 for p in parts):
        roots = {p[0] for p in parts}
        if len(roots) == 1 and next(iter(roots)) not in ROOTS:
            wrapper = next(iter(roots)) + '/'
    result = {}
    for name in names:
        rel = name[len(wrapper):] if wrapper else name
        for prefix in ('payload/', 'overlay/'):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
                if rel.startswith('farmNEAT/'): rel = rel[len('farmNEAT/'):]
                break
        result[name] = relative_path(rel)
    if len(set(result.values())) != len(result):
        raise ValueError('ambiguous normalized ZIP paths')
    files = set(result.values())
    if any(str(parent) in files for n in files for parent in PurePosixPath(n).parents):
        raise ValueError('file/directory prefix collision in ZIP paths')
    return wrapper, result


def patch_paths(raw):
    """Read unified-diff target declarations; never interpret hunk contents as code."""
    paths = set()
    for line in raw.decode('utf-8').splitlines():
        if not line.startswith(('--- ', '+++ ')): continue
        value = line[4:].split('\t', 1)[0]
        if value.startswith('"'):
            fields = shlex.split(value)
            if len(fields) != 1: raise ValueError('ambiguous patch header')
            value = fields[0]
        if value == '/dev/null': continue
        if value.startswith(('a/','b/')): value = value[2:]
        paths.add(relative_path(value))
    if not paths:
        raise ValueError('patch has no inspectable unified-diff target declarations')
    return sorted(paths)


def archive_inventory(path, *, baseline=False):
    path = Path(path)
    limit = BASELINE_LIMIT if baseline else MAX_ARCHIVE
    expanded_limit = BASELINE_LIMIT if baseline else MAX_EXPANDED
    member_limit = BASELINE_LIMIT if baseline else MAX_MEMBER
    if path.is_symlink() or not path.is_file() or path.stat().st_size >= limit:
        raise ValueError('regular bounded archive required')
    members, assignments, continuations, nested, manifests = {}, [], [], [], []
    metadata_errors, deletions, targets, artifacts = [], set(), set(), []
    original_manifest = None
    with path.open('rb') as stream:
        archive_sha = hashlib.file_digest(stream, 'sha256').hexdigest(); stream.seek(0)
        with zipfile.ZipFile(stream) as z:
            infos = z.infolist()
            if len(infos) > (20_000 if baseline else MAX_MEMBERS):
                raise ValueError('ZIP member count limit exceeded')
            if len({i.filename for i in infos}) != len(infos):
                raise ValueError('duplicate ZIP members')
            for info in infos: safe_member(info)  # Directories are checked too.
            files = [i for i in infos if not i.is_dir()]
            if {i.filename for i in files} & {i.filename.rstrip('/') for i in infos if i.is_dir()}:
                raise ValueError('file/directory name collision')
            if sum(i.file_size for i in files) >= expanded_limit or any(i.file_size >= member_limit for i in files):
                raise ValueError('ZIP expanded/member byte limit exceeded')
            wrapper, normalized = normalized_names([i.filename for i in files])
            total = 0
            for info in files:
                rel = normalized[info.filename]; name = PurePosixPath(rel).name
                ext = PurePosixPath(rel).suffix.lower()
                metadata = name in {'ASSIGNMENT.json','CONTINUE.json','DELETIONS.json','MANIFEST.json','BASELINE.json','EXPERIMENT.json'}
                interesting_json = ext == '.json' and (metadata or rel.startswith('project/registries/search-ledger/'))
                keep_limit = 16_000_000 if baseline and rel == 'work/runs/review-package/MANIFEST.json' else MAX_METADATA
                keep = (interesting_json or ext in {'.patch','.diff'}) and info.file_size <= keep_limit
                payload, first, h, count = bytearray(), b'', hashlib.sha256(), 0
                with z.open(info) as source:
                    while chunk := source.read(1 << 20):
                        if not first: first = chunk[:8]
                        count += len(chunk); total += len(chunk)
                        if count > info.file_size or count >= member_limit or total >= expanded_limit:
                            raise ValueError('ZIP actual byte limit exceeded')
                        h.update(chunk)
                        if keep: payload.extend(chunk)
                if count != info.file_size: raise ValueError('ZIP actual length mismatch')
                record = dict(sha256=h.hexdigest(), bytes=count, member=info.filename)
                members[rel] = record
                if rel.lower().endswith(ARCHIVES) or first[:4] in {b'PK\x03\x04',b'PK\x05\x06',b'PK\x07\x08'}:
                    nested.append(dict(path=rel, **record))
                roles = []
                if ext in CODE: roles.append('source_or_configuration')
                if ext in {'.patch','.diff'}:
                    roles.append('patch')
                    if not keep: metadata_errors.append(rel + ': patch too large to inspect targets')
                    else:
                        try: targets.update(patch_paths(payload))
                        except (ValueError, UnicodeError) as error:
                            metadata_errors.append(rel + ': ' + str(error))
                if name in {'RESULT.md','FEEDBACK.md','DISCUSSION_REPLIES.md'}: roles.append('report_or_discussion')
                if interesting_json:
                    if not keep: metadata_errors.append(rel + ': metadata too large')
                    else:
                        try:
                            obj = parse_json(payload)
                            if name == 'ASSIGNMENT.json':
                                if not isinstance(obj, dict): raise ValueError('assignment must be an object')
                                assignments.append(dict(path=rel, value=obj)); roles.append('assignment')
                            if name == 'CONTINUE.json':
                                if not isinstance(obj, dict): raise ValueError('continuation must be an object')
                                continuations.append(dict(path=rel, value=obj)); roles.append('continuation')
                            if name in {'MANIFEST.json','BASELINE.json'} and isinstance(obj, dict):
                                manifests.append(dict(path=rel, value=obj)); roles.append('manifest')
                            if baseline and rel == 'work/runs/review-package/MANIFEST.json':
                                original_manifest = dict(sha256=digest(payload), value=obj)
                            if name == 'DELETIONS.json':
                                values = obj.get('deletions') if isinstance(obj, dict) else obj
                                if not isinstance(values, list): raise ValueError('deletions must be a list')
                                deletions.update(relative_path(v) for v in values); roles.append('deletions')
                            if isinstance(obj, dict) and obj.get('type') in {'claim','followup','response','review'}:
                                roles.append('ledger_' + obj['type'])
                                if obj['type'] == 'followup': roles.append('experiment_proposal')
                            if name == 'EXPERIMENT.json': roles.append('experiment_proposal')
                        except (ValueError, TypeError, UnicodeError) as error:
                            metadata_errors.append(rel + ': ' + str(error))
                if roles: artifacts.append(dict(path=rel, roles=roles))
        stream.seek(0)
        if hashlib.file_digest(stream, 'sha256').hexdigest() != archive_sha:
            raise ValueError('archive changed during scan')
    if len(assignments) > 1: metadata_errors.append('multiple ASSIGNMENT.json files; no ticket selected')
    if len(continuations) > 1: metadata_errors.append('multiple CONTINUE.json files; no continuation selected')
    # Verify supplied baseline bytes against its own exact membership when present.
    if baseline and original_manifest is not None:
        declared = original_manifest['value'].get('members')
        if not isinstance(declared, dict): raise ValueError('invalid baseline manifest')
        actual = {n:{k:r[k] for k in ('sha256','bytes')} for n,r in members.items()
                  if n != 'work/runs/review-package/MANIFEST.json'}
        if actual != declared: raise ValueError('baseline membership/hash mismatch')
    signature = digest(canonical({n:{k:r[k] for k in ('sha256','bytes')} for n,r in sorted(members.items())}))
    return dict(file=str(path), zip_sha256=archive_sha, bytes=path.stat().st_size,
        expanded_bytes=total, content_signature=signature, root_prefix=wrapper,
        members=members, assignments=assignments, continuations=continuations, manifests=manifests,
        baseline_manifest_sha256=original_manifest['sha256'] if original_manifest else None,
        nested_archives=nested, metadata_errors=metadata_errors,
        declared_deletions=sorted(deletions), patch_targets=sorted(targets), artifacts=artifacts)


def baseline_inventory(baseline):
    """An exact retained outgoing manifest suffices for hash comparison, not replay."""
    p = Path(baseline)
    if p.suffix.lower() != '.json':
        return archive_inventory(p, baseline=True)
    if p.is_symlink() or not p.is_file() or p.stat().st_size > 16_000_000:
        raise ValueError('bounded regular baseline manifest required')
    raw = p.read_bytes(); value = parse_json(raw)
    if value.get('schema') != 'farmneat-source-review-package-v1' or not isinstance(value.get('members'),dict):
        raise ValueError('original outgoing manifest schema required')
    members = value['members']
    if len(members)>20_000: raise ValueError('baseline member limit exceeded')
    for name, record in members.items():
        relative_path(name)
        if (not isinstance(record,dict) or set(record) != {'sha256','bytes'}
                or not isinstance(record['sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',record['sha256'])
                or type(record['bytes']) is not int or record['bytes']<0):
            raise ValueError('invalid original baseline member identity')
    members = {**members, 'work/runs/review-package/MANIFEST.json':dict(sha256=digest(raw),bytes=len(raw))}
    return dict(file=str(p),zip_sha256=None,baseline_manifest_sha256=digest(raw),members=members)


def scan(paths, baseline=None):
    """Return identity/difference evidence only; every scientific intake remains pending.

    ``baseline`` may be one exact outgoing ZIP/manifest (legacy mode) or a dispatch
    registry directory. Registry mode resolves each return by its own declared
    source_manifest_sha256, so mixed-baseline arrivals compare to the right source.
    """
    registry_root = registry_state = None
    base = None
    if baseline:
        baseline_path = Path(baseline)
        if baseline_path.is_dir():
            registry_root, registry_state = _load_registry(baseline_path)
        else:
            base = baseline_inventory(baseline_path)
    files = set()
    for value in paths:
        path = Path(value)
        if path.is_symlink(): raise ValueError('symlink input refused')
        files.update(path.glob('*.zip') if path.is_dir() else [path])
    reports, errors = [], []
    baseline_cache = {}
    for path in sorted(files, key=str):
        try:
            item = archive_inventory(path)
        except (ValueError,OSError,zipfile.BadZipFile,RuntimeError,EOFError) as error:
            h = None
            if path.is_file() and not path.is_symlink() and path.stat().st_size < MAX_ARCHIVE:
                with path.open('rb') as stream: h=hashlib.file_digest(stream,'sha256').hexdigest()
            errors.append(dict(file=str(path),zip_sha256=h,error=str(error),status='REJECTED_UNREAD'))
            continue

        assignment = item['assignments'][0]['value'] if len(item['assignments']) == 1 else {}
        continuation = item['continuations'][0]['value'] if len(item['continuations']) == 1 else None
        ticket = assignment.get('ticket')
        topic = assignment.get('task_id')
        if not isinstance(topic, str):
            topic = assignment.get('assignment',{}).get('id') if isinstance(assignment.get('assignment',{}),dict) else None
        claimed = assignment.get('source_manifest_sha256')
        continuation_summary = None
        if continuation is not None:
            ctask = continuation.get('task_id')
            if continuation.get('status') != 'PARTIAL' or not isinstance(ctask, str) or not ctask:
                item['metadata_errors'].append('CONTINUE.json must describe a nonempty PARTIAL task_id')
            elif topic is not None and ctask != topic:
                item['metadata_errors'].append(f'CONTINUE.json task_id {ctask} disagrees with assignment {topic}')
            for field in ('done', 'do_not_redo'):
                if field in continuation and (not isinstance(continuation[field], list)
                        or any(not isinstance(x, str) or not x.strip() for x in continuation[field])):
                    item['metadata_errors'].append(f'CONTINUE.json {field} must be a string list')
            if 'next_command' in continuation and (not isinstance(continuation['next_command'], str)
                    or not continuation['next_command'].strip()):
                item['metadata_errors'].append('CONTINUE.json next_command must be nonempty text')
            continuation_summary = {key: continuation.get(key) for key in
                                    ('schema','status','task_id','generation','done','exact_state','remaining',
                                     'blockers','next_command','do_not_redo','checkpoint','soft_target_games',
                                     'starts_recorded','native_start_cap','native_starts_charged')
                                    if key in continuation}
        if claimed is not None and (not isinstance(claimed,str) or not re.fullmatch(r'[0-9a-f]{64}',claimed)):
            item['metadata_errors'].append('assignment source_manifest_sha256 must be a full hash')

        row_base = base
        resolved_baseline = None
        baseline_status = 'NOT_SUPPLIED'
        ticket_registry_status = None
        if registry_state is not None:
            baseline_status = 'UNDECLARED_SOURCE; REGISTRY_NOT_RESOLVED'
            if isinstance(claimed, str) and re.fullmatch(r'[0-9a-f]{64}', claimed):
                if claimed in registry_state['sources']:
                    if claimed not in baseline_cache:
                        baseline_cache[claimed] = _registry_baseline(registry_root, registry_state, claimed)
                    row_base = baseline_cache[claimed]
                    resolved_baseline = claimed
                    baseline_status = 'EXACT_DECLARED_BASELINE'
                else:
                    row_base = None
                    baseline_status = 'DECLARED_BASELINE_NOT_REGISTERED'
            else:
                row_base = None
            if isinstance(ticket, str):
                registered = registry_state['tickets'].get(ticket)
                if registered is None:
                    ticket_registry_status = 'UNREGISTERED_TICKET'
                elif (registered.get('source_manifest_sha256') == claimed
                      and (topic is None or registered.get('task_id') == topic)):
                    ticket_registry_status = 'EXACT_ISSUANCE'
                else:
                    ticket_registry_status = 'TICKET_IDENTITY_MISMATCH'
                    item['metadata_errors'].append('assignment ticket identity disagrees with dispatch registry')
        elif row_base is not None:
            baseline_status = 'UNDECLARED_SOURCE; COMPARED_TO_SUPPLIED_BASELINE_ONLY'
            if claimed:
                baseline_status = ('EXACT_DECLARED_BASELINE'
                                   if claimed == row_base['baseline_manifest_sha256']
                                   else 'DECLARED_BASELINE_MISMATCH')
            resolved_baseline = row_base['baseline_manifest_sha256']

        changed, new, identical, uncompared = [], [], [], []
        for name, record in item['members'].items():
            if row_base is None:
                uncompared.append(name); continue
            old = row_base['members'].get(name)
            if old is None: new.append(name)
            elif (old['sha256'],old['bytes']) == (record['sha256'],record['bytes']): identical.append(name)
            else: changed.append(name)
        metadata_names = {a['path'] for a in item['artifacts'] if set(a['roles']) & {'assignment','manifest','continuation'}}
        payload = {n:{k:r[k] for k in ('sha256','bytes')} for n,r in item['members'].items() if n not in metadata_names}
        touched = sorted(set(changed+new+uncompared+item['declared_deletions']+item['patch_targets']))
        status = 'REJECTED_NESTED_ARCHIVE' if item['nested_archives'] else ('NEEDS_METADATA_REVIEW' if item['metadata_errors'] else 'INVENTORIED')
        has_result = any(PurePosixPath(a['path']).name == 'RESULT.md' for a in item['artifacts'])
        return_kind = 'PARTIAL' if continuation is not None else ('FINAL' if has_result else 'UNCLASSIFIED')
        routing = ('PRESERVE_CONTINUATION_AND_REDISPATCH_SAME_TASK' if return_kind == 'PARTIAL'
                   else 'SCIENTIFIC_REVIEW_REQUIRED')
        reports.append({k:v for k,v in item.items() if k not in {'assignments','continuations','manifests'}} | dict(
            status=status, scientific_intake='PENDING', return_kind=return_kind, routing=routing,
            continuation=continuation_summary, ticket=ticket if isinstance(ticket,str) else None,
            assignment_id=topic if isinstance(topic,str) else None,
            claimed_baseline_manifest_sha256=claimed, baseline_comparison=baseline_status,
            resolved_baseline_manifest_sha256=resolved_baseline,
            ticket_registry_status=ticket_registry_status,
            changed_vs_baseline=sorted(changed),new_vs_baseline=sorted(new),
            identical_vs_baseline=sorted(identical),uncompared=sorted(uncompared),
            changed_or_proposed_paths=touched, payload_signature=digest(canonical(payload)) if payload else None))
    def groups(field):
        grouped=defaultdict(list)
        for item in reports:
            if item.get(field): grouped[item[field]].append(item['file'])
        return {k:v for k,v in sorted(grouped.items()) if len(v)>1}
    overlaps=defaultdict(list)
    for item in reports:
        if item['status']=='REJECTED_NESTED_ARCHIVE': continue
        for name in item['changed_or_proposed_paths']:
            overlaps[name].append(dict(file=item['file'],content_signature=item['content_signature']))
    overlaps={k:[v['file'] for v in values] for k,values in sorted(overlaps.items())
              if len({v['content_signature'] for v in values})>1}
    wrapped=[]
    by_sha={r['zip_sha256']:r['file'] for r in reports}
    for item in reports:
        for nested in item['nested_archives']:
            if nested['sha256'] in by_sha:
                wrapped.append(dict(wrapper=item['file'],member=nested['path'],same_bytes_as=by_sha[nested['sha256']]))
    if registry_state is not None:
        baseline_view = dict(mode='DISPATCH_REGISTRY', path=str(registry_root),
                             sources=len(registry_state['sources']), tickets=len(registry_state['tickets']),
                             lost_after_rule=registry_state['lost_after_rule'])
    elif base is not None:
        baseline_view = dict(file=base['file'],zip_sha256=base['zip_sha256'],
                             manifest_sha256=base['baseline_manifest_sha256'],members=len(base['members']))
    else:
        baseline_view = None
    return dict(schema='farm-return-intake-v2', baseline=baseline_view,
        returns=reports,errors=errors,
        duplicate_zip_groups=groups('zip_sha256'),duplicate_content_groups=groups('content_signature'),
        duplicate_payload_groups=groups('payload_signature'),ticket_collisions=groups('ticket'),
        assignment_collisions=groups('assignment_id'),changed_path_overlaps=overlaps,wrapped_duplicates=wrapped,
        commands_executed=0,archives_extracted=0,arrivals_modified=0,
        limits=dict(archive_bytes=MAX_ARCHIVE,expanded_bytes=MAX_EXPANDED,member_bytes=MAX_MEMBER,members=MAX_MEMBERS),
        scope='Transport evidence only. PARTIAL is a routing state, not scientific acceptance or failure. Execution budgets are not intake validity gates. Patches, metadata claims and science are untrusted pending maintainer reading. No code executed, acceptance inferred, archive unpacked or arrival deleted.')


def verified_partial_adoption(path, registry):
    """Return verified continuation bytes for one registered PARTIAL arrival.

    This is deliberately narrower than extraction: the exact dispatch ticket must resolve
    to a registered immutable baseline, and only the tool-owned continuation subtree is
    returned.  The caller owns transactional installation into its repository.
    """
    path = Path(path)
    report = scan([path], registry)
    if report['errors'] or len(report['returns']) != 1:
        error = report['errors'][0]['error'] if report['errors'] else 'expected exactly one return'
        raise ValueError('PARTIAL adoption refused: ' + error)
    row = report['returns'][0]
    if row['status'] != 'INVENTORIED' or row.get('return_kind') != 'PARTIAL':
        raise ValueError('PARTIAL adoption requires one clean PARTIAL return')
    if row.get('baseline_comparison') != 'EXACT_DECLARED_BASELINE':
        raise ValueError('PARTIAL adoption requires the exact registered baseline')
    if row.get('ticket_registry_status') != 'EXACT_ISSUANCE':
        raise ValueError('PARTIAL adoption requires an exact registered ticket issuance')
    task = row.get('assignment_id')
    ticket = row.get('ticket')
    if not isinstance(task, str) or not task or not isinstance(ticket, str) or not ticket:
        raise ValueError('PARTIAL adoption requires task and ticket identity')

    inventory = archive_inventory(path)
    prefix = f'review/continuations/{task}/'
    canonical_continue = prefix + 'CONTINUE.json'
    continuation_paths = [item['path'] for item in inventory['continuations']]
    candidates = [name for name in continuation_paths if name in {canonical_continue, 'CONTINUE.json'}]
    if len(candidates) != 1:
        raise ValueError('PARTIAL adoption requires exactly one canonical/root CONTINUE.json')
    source_continue = candidates[0]
    selected = {}
    for rel, record in inventory['members'].items():
        if rel == source_continue:
            selected[canonical_continue] = record
            continue
        if not rel.startswith(prefix):
            continue
        tail = rel[len(prefix):]
        if re.fullmatch(r'START-\d{6}\.json', tail) or tail.startswith('workspace/'):
            selected[rel] = record
        elif tail != 'CONTINUE.json':
            raise ValueError('unsupported continuation adoption member: ' + rel)
    if canonical_continue not in selected:
        raise ValueError('PARTIAL adoption continuation bytes missing')

    payloads = {}
    before = inventory['zip_sha256']
    with path.open('rb') as stream:
        with zipfile.ZipFile(stream) as archive:
            for destination, record in selected.items():
                original_rel = source_continue if destination == canonical_continue else destination
                original_member = inventory['members'][original_rel]['member']
                raw = archive.read(original_member)
                if len(raw) != record['bytes'] or digest(raw) != record['sha256']:
                    raise ValueError('PARTIAL adoption member changed after intake scan: ' + original_rel)
                payloads[destination] = raw
        stream.seek(0)
        if hashlib.file_digest(stream, 'sha256').hexdigest() != before:
            raise ValueError('PARTIAL archive changed during adoption')
    return {
        'schema': 'farm-review-partial-adoption-payload-v1',
        'task_id': task, 'ticket': ticket,
        'source_manifest_sha256': row['resolved_baseline_manifest_sha256'],
        'zip_sha256': before, 'members': payloads,
    }


def summary(report):
    lines=[f"Intake scan: {len(report['returns'])} inventoried; {len(report['errors'])} unsafe/unreadable."]
    for row in report['returns']:
        suffix = (f" PARTIAL -> preserve and redispatch {row.get('assignment_id') or row.get('continuation',{}).get('task_id','same task')};"
                  if row.get('return_kind') == 'PARTIAL' else '')
        lines.append(f"- {Path(row['file']).name}: {row['status']}; {row.get('return_kind','UNCLASSIFIED')};"
                     f"{suffix} {len(row['changed_vs_baseline'])} changed, "
                     f"{len(row['new_vs_baseline'])} new, {len(row['identical_vs_baseline'])} identical; "
                     f"baseline {row['baseline_comparison']}; scientific intake PENDING.")
    lines.append(f"Duplicate contents: {len(report['duplicate_content_groups'])}; ticket collisions: "
                 f"{len(report['ticket_collisions'])}; assignment overlaps: {len(report['assignment_collisions'])}; "
                 f"overlapping changed paths: {len(report['changed_path_overlaps'])}.")
    lines += [f"- Rejected {Path(e['file']).name}: {e['error']}" for e in report['errors']]
    return '\n'.join(lines)+'\nNo returned code executed; arrivals untouched. Game/CPU targets do not invalidate intake.\n'
