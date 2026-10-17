#!/usr/bin/python3
"""Catalog of every locally stored Kaggriculture game. Read-only on every source; writes only this directory and
extra/hosted-ours/ (sync-hosted). Runs under /usr/bin/python3 (pyarrow); lowers itself to nice 19, <= 6 processes.

  catalog.py find boey                   # alias / team text / episode id / submission id / team id -> games + local copies
  catalog.py find us --version abR       # our games per deployed version, every loss with opponent and rank
  catalog.py query --team dsm --day-from 2026-09-20 --out dsm-wins.tsv   # trainer TSV (archive member day eid seat weight reason)
  catalog.py build                       # full rescan of every source
  catalog.py update                      # rescan only new/changed archives, then rejoin all metadata
  catalog.py sync-hosted                 # copy new hosted replays of ours into extra/hosted-ours (zip + MANIFEST.jsonl)
  catalog.py refresh                     # sync-hosted + update
  catalog.py teams [--top 40]            # per-team summary (TEAMS.tsv)
  catalog.py stats                       # per-source counts (SUMMARY.json)

Outputs here: COPIES.parquet (one row per stored copy), EPISODES.parquet (one row per episode: stored, or only known from
an index/listing), TEAMS.tsv, SUMMARY.json; state/ = per-archive scan cache. ALIASES.tsv maps owner shorthand to names.
A replay header is everything before the "steps" key; only that is read (first 64 KiB of a member), never whole replays.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import datetime as dt
import fcntl
import glob
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import struct
import sys
import time
import unicodedata
import zipfile
import zlib
from pathlib import Path

DATA = Path('/mnt/ai/scratch/farmNEAT/primitive-bc-data')
EXTRA = DATA / 'extra'
SCR = Path('/mnt/ai/scratch/farmNEAT')
REPO = Path('/home/russ/Desktop/farmNEAT')
FARM = Path('/home/russ/Desktop/farm')
STEAM = Path('/mnt/steam/farm-data/kaggle')
DEPLOY = REPO / 'work/runs/deployment'
AUTOSUBMIT = REPO / 'work/runs/maintainer/pbc-autosubmit'
HOSTED_OURS = EXTRA / 'hosted-ours'
CAT = Path(os.environ.get('CATALOG_DIR', str(Path(__file__).resolve().parent)))
HERE = Path(__file__).resolve().parent
OUR_TEAM, OUR_TID = 'Russell Kirk', 16624026
# every display name team 16624026 has shown (renamed on Kaggle 2026-09-27: hosted games show 有辣条有权; 怎么做热干面? also
# appears at our seat). "Ours" is decided by team id / our submission ids; these names cover sources that carry names only.
OUR_NAMES = frozenset({'Russell Kirk', '有辣条有权', '怎么做热干面?'})
REFERENCE = json.loads((DATA / 'REFERENCE-CONFIG.json').read_text())
LOCAL_ID_MIN = 10 ** 12          # Kaggle episode ids are 9 digits; local games use 16-digit ids
WORKERS = 5                      # + the main process = 6
HEAD = 65536
MAX_ID_GAP = 600_000             # day estimates only from a known episode id this close (~1 ladder day)
TAPE_LABELS = [SCR / 'pbc-tape-corpus-20260923/EPISODE-TAPE-LABELS.jsonl.gz',
               REPO / 'work/runs/maintainer/pbc-seed-action-replays/labels-full/'
                      'AUGMENTED-EPISODE-TAPE-LABELS-86a24b766a27-c564ee4dcfaf.jsonl.gz']

# preferred copy = lowest rank; any extra/<folder>/archives source not named here ranks API_RANK (original Kaggle bytes)
SOURCE_RANK = {'official': 0, 'hosted-ours': 1, 'reconstructed': 30, 'reconstructed-xsf': 31, 'selfplay-zip': 40,
               'mirror-steam': 50, 'repo-hosted': 60, 'manual-ingest': 61, 'repo-misc': 62, 'farm-repo': 63,
               'skirmishes-loose': 64, 'tapes-compact': 70, 'selfplay-tapes': 71, 'recorded-ladder': 72, 'capsules': 73,
               'selfplay-league': 74, 'guided-branches': 75, 'xsf-raw': 80}
# repo-misc / farm-repo may also hold tapes (pbc-puzzles/data/tapes); the format column says which
API_RANK = 10
LOCAL_SOURCES = {'selfplay-tapes', 'selfplay-zip', 'recorded-ladder', 'selfplay-league', 'guided-branches'}
REPLAY_FORMATS = {'zip-json', 'json', 'json-gz'}

NAME_RE = re.compile(r'^(episode-\d+[^/]*|\d{8,17}|[^/]*replay[^/]*|HOSTED-REPLAY|REPLAY)\.json(\.gz)?$', re.I)
SKIP_NAME_RE = re.compile(r'agent-\d-logs|own-logs|\.tape\.json|capsule|-events\.json', re.I)
SCHEMA_RE = re.compile(rb'^\s*\{\s*"schema"\s*:\s*"([^"]+)"')
DAY_RE = re.compile(r'(20\d\d-\d\d-\d\d)')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, file=sys.stderr, flush=True)


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def be_nice():
    try:
        cur = os.nice(0)
        if cur < 19:
            os.nice(19 - cur)
    except OSError:
        pass


# ------------------------------------------------------------------ small helpers
def parse_header(head):
    """Replay fields before "steps" (info, rewards, statuses, configuration). Tries each "steps" occurrence."""
    start = 0
    while True:
        cut = head.find(b'"steps"', start)
        if cut < 0:
            raise ValueError('no "steps" key in header window')
        try:
            h = json.loads(head[:cut].rstrip().rstrip(b',') + b'}')
            if isinstance(h, dict):
                return h
        except ValueError:
            pass
        start = cut + 1


def cut_prefix(head, key):
    """JSON object prefix before `key` (a large trailing field), closed with a brace."""
    cut = head.find(b'"' + key + b'"')
    if cut < 0:
        raise ValueError(f'no "{key.decode()}" key in window')
    return json.loads(head[:cut].rstrip().rstrip(b',') + b'}')


def config_diff(cfg):
    """Keys whose value differs from REFERENCE-CONFIG.json (seed ignored; timeouts compared only when present)."""
    if not isinstance(cfg, dict) or not cfg:
        return None
    c = dict(cfg)
    c.pop('seed', None)
    return sorted(k for k in set(c) | set(REFERENCE)
                  if c.get(k) != REFERENCE.get(k) and not (k in ('actTimeout', 'runTimeout') and k not in c))


def winner(rewards):
    if not isinstance(rewards, (list, tuple)) or len(rewards) != 2 or None in rewards:
        return None
    try:
        a, b = float(rewards[0]), float(rewards[1])
    except (TypeError, ValueError):
        return None
    return 0 if a > b else 1 if b > a else None


def pair(x):
    return list(x) if isinstance(x, (list, tuple)) and len(x) == 2 else [None, None]


def as_int(x):
    try:
        return None if x is None or x == '' else int(x)
    except (TypeError, ValueError):
        return None


def valid_catalog_id(x):
    """Catalog keys use Arrow int64; retain but do not index invalid local smoke IDs."""
    return isinstance(x, int) and not isinstance(x, bool) and 0 <= x < (1 << 63)


def as_float(x):
    try:
        return None if x is None or x == '' else float(x)
    except (TypeError, ValueError):
        return None


def fold(s):
    return unicodedata.normalize('NFKC', str(s)).casefold()


def tokens(s):
    return [t for t in re.split(r'[^\w]+', fold(s)) if t]


def local_label(s):
    """Seat label of a local (non-Kaggle) game: SELF:<model version> or LEAGUE:<bot>."""
    s = str(s)
    if s.startswith(('archive:', 'archive/')) or s.startswith('SELF:'):
        base = s[5:] if s.startswith('SELF:') else s.split('/')[-1]
        for suf in ('.tar.gz', '.tgz', '.tar', '.zip'):
            if base.endswith(suf):
                base = base[:-len(suf)]
        return 'SELF:' + base
    return s if s.startswith('LEAGUE:') else 'LEAGUE:' + s


def local_id(text):
    return 2 ** 60 + int(hashlib.sha256(text.encode()).hexdigest()[:14], 16)


def mtime_day(path):
    return dt.datetime.fromtimestamp(os.stat(path).st_mtime, dt.timezone.utc).strftime('%Y-%m-%d')


def gz_trailer(path):
    with open(path, 'rb') as f:
        f.seek(-8, 2)
        crc, isize = struct.unpack('<II', f.read(8))
    return crc, isize


def read_head(path, n=HEAD):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as f:
        return f.read(n)


def header_fields(h):
    info = h.get('info') if isinstance(h.get('info'), dict) else {}
    return {'header_eid': info.get('EpisodeId'), 'team_names': info.get('TeamNames'), 'rewards': h.get('rewards'),
            'statuses': h.get('statuses'), 'seed': info.get('seed'), 'module_version': h.get('module_version'),
            'name': h.get('name'), 'config_diff': config_diff(h.get('configuration')), 'replay_uuid': h.get('id')}


def is_kaggle_header(h):
    return isinstance(h, dict) and (h.get('name') == 'kaggriculture' or 'EpisodeId' in (h.get('info') or {}))


# ------------------------------------------------------------------ discovery
def walk_named(root, skip_parts, skip_prefixes=()):
    """Files under root whose name looks like a replay (NAME_RE), skipping dirs named in skip_parts."""
    out = []
    root = str(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        dirnames[:] = sorted(d for d in dirnames if d not in skip_parts and not d.startswith('python-')
                             and not any(os.path.normpath(os.path.join(rel, d)).startswith(p) for p in skip_prefixes))
        for f in sorted(filenames):
            if NAME_RE.match(f) and not SKIP_NAME_RE.search(f):
                p = os.path.join(dirpath, f)
                if not os.path.islink(p):
                    out.append(p)
    return out


def repo_hosted_version(path):
    """Deployed version folder of a replay under work/runs/deployment (hosted/, replays/, hosted-<sub>/ dropped)."""
    rel = Path(path).parent.relative_to(DEPLOY).parts
    keep = [p for p in rel if p not in ('hosted', 'replays') and not re.fullmatch(r'(hosted|kaggle)-\d+', p)]
    return '/'.join(keep) if keep else 'deployment-root'


def discover(rediscover=False):
    out = []

    def add(path, kind, source, subset=''):
        out.append({'path': str(path), 'kind': kind, 'source': source, 'subset': subset})

    for p in sorted(DATA.glob('kaggriculture-episodes-*.zip')):
        add(p, 'zip', 'official')
    for p in sorted((DATA / 'incoming').glob('kaggriculture-episodes-*.zip')):
        add(p, 'zip', 'official', 'incoming')
    for p in sorted((EXTRA / 'reconstructed').glob('*.zip')):
        add(p, 'zip', 'reconstructed')
    special = {'reconstructed', 'raw', 'selfplay', 'index', 'manual-ingest', '__pycache__', 'skirmishes',
               'selfplay-league', 'guided-branches'}
    for folder in sorted(x for x in EXTRA.iterdir() if x.is_dir() and not x.is_symlink() and x.name not in special):
        for p in sorted((folder / 'archives').glob('*.zip')):
            if not p.is_symlink():
                add(p, 'zip', folder.name)
    sk = EXTRA / 'skirmishes'     # games between our versions (Kaggle CPU notebooks): archives/*.zip or loose files
    if sk.is_dir():
        for p in sorted(sk.rglob('*')):
            if p.is_symlink() or not p.is_file() or 'staging' in p.parts:
                continue
            if p.suffix == '.zip':
                add(p, 'zip', 'skirmishes', str(p.parent.relative_to(sk)))
            elif p.name.endswith(('.json', '.json.gz')) and not p.name.startswith(('MANIFEST', 'LISTINGS', 'SUBMISSIONS')):
                add(p, 'loose', 'skirmishes-loose', str(p.parent.relative_to(sk)))
    for p in sorted((EXTRA / 'selfplay').glob('*/*.tape.json')):
        add(p, 'loose', 'selfplay-tapes', p.parent.name)
    for p in sorted((EXTRA / 'selfplay').glob('*/*.zip')):
        add(p, 'zip', 'selfplay-zip', p.parent.name)
    for folder in ('selfplay-league', 'guided-branches'):  # compact tapes (seed + actions) of our models' local games
        for p in sorted((EXTRA / folder).glob('*/*.tape.json.gz')):
            add(p, 'loose', folder, p.parent.name)
    for p in sorted((EXTRA / 'manual-ingest').glob('*.json')):
        add(p, 'loose', 'manual-ingest')
    for p in sorted((EXTRA / 'raw').glob('*/*.zip')):
        add(p, 'xsf-raw', 'xsf-raw', p.parent.name)
    for p in sorted(SCR.glob('pbc-replays/*/*.json.gz')):
        add(p, 'loose', 'recorded-ladder', p.parent.name)
    if STEAM.is_dir():
        for p in sorted(STEAM.glob('*/*.zip')):
            add(p, 'zip', 'mirror-steam', p.parent.name)
        for p in sorted(STEAM.glob('*/*.json')):
            if re.fullmatch(r'\d{8,10}\.json', p.name):
                add(p, 'loose', 'mirror-steam', p.parent.name)
    for p in sorted(DEPLOY.glob('*/hosted/episode-*-replay.json.gz')):
        add(p, 'loose', 'repo-hosted', p.parent.parent.name)
    for p in sorted((REPO / 'work/runs/maintainer').rglob('*.tape.json.gz')):
        add(p, 'loose', 'tapes-compact', str(p.parent.relative_to(REPO / 'work/runs/maintainer')))
    for p in sorted((REPO / 'work/datasets/public-episodes').glob('*/episodes/*.json.gz')):
        add(p, 'capsule', 'capsules', p.parent.parent.name)
    covered = {c['path'] for c in out}
    for p in walk_named(REPO / 'work', {'venv', 'review', '__pycache__', 'node_modules', '.git'},
                        ('datasets/public-episodes',)):
        if p not in covered:
            add(p, 'loose', 'repo-misc', os.path.relpath(os.path.dirname(p), REPO / 'work'))
    for p in sorted((REPO / 'work/datasets/public-episodes').glob('*/*.json')):   # e.g. daily/<id>.json
        if re.fullmatch(r'\d{8,10}\.json', p.name) and str(p) not in covered:
            add(p, 'loose', 'repo-misc', os.path.relpath(str(p.parent), REPO / 'work'))
    if FARM.is_dir():                   # the old workspace is frozen: its file list is walked once and cached
        cache = CAT / 'state/farm-discovery.json'
        if rediscover or not cache.exists():
            files = walk_named(FARM, {'.git', 'review', '__pycache__', 'node_modules', 'venv', 'logs'})
            tmp = cache.with_suffix('.tmp')
            tmp.write_text(json.dumps({'walked_utc': utc_now().isoformat(), 'files': files}))
            os.replace(tmp, cache)
        for p in json.loads(cache.read_text())['files']:
            if os.path.exists(p):
                add(p, 'loose', 'farm-repo', os.path.relpath(os.path.dirname(p), FARM))
    seen, uniq = set(), []
    for c in out:
        if c['path'] not in seen:
            seen.add(c['path'])
            uniq.append(c)
    return uniq


# ------------------------------------------------------------------ scanners (worker side)
def scan_zip(path):
    rows, others = [], []
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = {zi.filename for zi in infos}
        csvmeta = {}
        if 'manifest.csv' in names:
            for r in csv.DictReader(io.StringIO(z.read('manifest.csv').decode('utf-8-sig'))):
                csvmeta[as_int(r.get('episode_id'))] = r
        for zi in infos:
            base = zi.filename.rsplit('/', 1)[-1]
            if not re.fullmatch(r'\d+\.json', base):
                if not zi.is_dir():
                    others.append(zi.filename)
                continue
            eid = int(base[:-5])
            row = {'member': zi.filename, 'episode_id': eid, 'format': 'zip-json', 'bytes': zi.file_size,
                   'stored_bytes': zi.compress_size, 'crc32': zi.CRC}
            try:
                with z.open(zi) as f:
                    head = f.read(HEAD)
                    try:
                        h = parse_header(head)
                    except ValueError:
                        h = parse_header(head + f.read(1 << 20))
                row.update(header_fields(h))
                names = row.get('team_names') or []
                if eid >= LOCAL_ID_MIN or any(str(n).startswith(('SELF:', 'LEAGUE:')) for n in names):
                    row['local'] = True
                    row['team_names'] = [local_label(n) for n in names]
                if not is_kaggle_header(h):
                    row['error'] = 'no kaggriculture header'
                elif row.get('header_eid') is not None and as_int(row['header_eid']) != eid:
                    row['error'] = f'header EpisodeId {row["header_eid"]} != member {eid}'
            except Exception as e:  # noqa: BLE001 - reported per member, never dropped
                row['error'] = repr(e)[:300]
            m = csvmeta.get(eid)
            if m:
                row['official'] = {'create_time': m.get('create_time'), 'avg_score': as_float(m.get('avg_score')),
                                   'min_score': as_float(m.get('min_score')), 'sum_score': as_float(m.get('sum_score')),
                                   'size_bytes': as_int(m.get('size_bytes'))}
            rows.append(row)
    note = {'other_members': others[:20], 'n_other_members': len(others)}
    if csvmeta:                       # official daily zips: manifest.csv vs the members actually present
        have = {r['episode_id'] for r in rows}
        listed = {k for k in csvmeta if k is not None}
        note.update({'manifest_csv_rows': len(listed), 'members': len(have),
                     'manifest_csv_without_member': sorted(listed - have)[:100],
                     'members_not_in_manifest_csv': sorted(have - listed)[:100]})
    return rows, note


def scan_loose(path):
    """One loose file: a Kaggle replay (.json / .json.gz) or a known tape format (sniffed from its first bytes)."""
    size = os.path.getsize(path)
    head = read_head(path)
    m = SCHEMA_RE.match(head[:400])
    schema = m.group(1).decode() if m else None
    base = {'member': '', 'stored_bytes': size}
    if schema == 'farm-hosted-tape-v1':            # seed + both seats' 719 actions (compact_tapes.py)
        t = cut_prefix(head, b'actions')
        eid = as_int(t.get('episode_id'))
        if eid is None:                            # a local game between our checkpoints (selfplay-league, guided-branches)
            rel = os.path.relpath(path, EXTRA) if str(path).startswith(str(EXTRA)) else str(path)
            return [dict(base, format='tape-compact', episode_id=local_id('tape:' + rel), local=True,
                         team_names=['SELF:' + str(n) for n in (t.get('teams') or [])], rewards=t.get('rewards'),
                         seed=t.get('seed'), statuses=None, day_mtime=mtime_day(path))], {}
        return [dict(base, format='tape-compact', episode_id=eid, team_names=t.get('teams'),
                     rewards=t.get('rewards'), seed=t.get('seed'), statuses=None)], {}
    if schema == 'pbc-puzzle-tape-v1':             # puzzle tapes cut from archived episodes (pbc-puzzles)
        t = cut_prefix(head, b'actions')
        row = dict(base, format='tape-puzzle', episode_id=as_int(t.get('episode_id')), rewards=t.get('rewards'),
                   seed=t.get('seed'), config_diff=config_diff(t.get('configuration')), extra={'from': t.get('source')})
        if str(t.get('source') or '').startswith('fresh'):      # a fresh local game (e.g. v1 vs PASS), not a Kaggle episode
            row.update(local=True, day_mtime=mtime_day(path))
        return [row], {}
    if schema == 'farmneat-public-episode-capsule-v1':
        return scan_capsule(path)
    if schema == 'pbc-recorded-game-v1':           # locally recorded ladder game
        t = cut_prefix(head, b'final_farms')
        names = [local_label(x) for x in (t.get('names') or t.get('seats') or [])]
        return [dict(base, format='recorded-game', episode_id=local_id('recorded:' + str(t.get('tag'))), local=True,
                     team_names=names, rewards=t.get('final_cash'), seed=t.get('world_seed'), tag=t.get('tag'),
                     day_mtime=mtime_day(path), extra={'sampler_seed': t.get('sampler_seed'),
                                                       'temperature': t.get('temperature'),
                                                       'matches_ladder': t.get('matches_ladder')})], {}
    if head.lstrip().startswith(b'{"world_seed"'):  # pbc-win-farm tape (task42 panels)
        if size > 20_000_000:
            return [], {'skipped': 'tape too large to parse'}
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rb') as f:
            t = json.load(f)
        names = [local_label(x) for x in (t.get('seats') or [])]
        return [dict(base, format='tape-selfplay', episode_id=as_int(t.get('episode_id')), local=True, team_names=names,
                     rewards=t.get('rewards'), statuses=t.get('statuses'), seed=t.get('world_seed'),
                     config_diff=config_diff(t.get('configuration')), day_mtime=mtime_day(path),
                     extra={'valid': t.get('valid'), 'sampler_seed': t.get('sampler_seed'),
                            'fallbacks': t.get('fallbacks'), 'hard_errors': t.get('hard_errors'),
                            'seats_raw': t.get('seats')})], {}
    try:
        h = parse_header(head)
    except ValueError:
        if size <= HEAD:
            return [], {'skipped': 'not a replay (no header)'}
        try:
            h = parse_header(read_head(path, 1 << 20))
        except ValueError:
            return [], {'skipped': 'not a replay (no header)'}
    if not is_kaggle_header(h):
        return [], {'skipped': 'not a kaggriculture replay'}
    row = dict(base, **header_fields(h))
    if not isinstance(row.get('rewards'), list) or None in row['rewards']:   # e.g. notebook replays: keys after steps
        row.update({k: v for k, v in tail_fields(path).items() if k != 'seed' or row.get('seed') is None})
    gz = str(path).endswith('.gz')
    row['format'] = 'json-gz' if gz else 'json'
    if gz:
        crc, isize = gz_trailer(path)
        row['crc32'], row['bytes'] = crc, isize
    else:
        row['bytes'] = size
    eid = as_int(row.get('header_eid'))
    if eid is None:
        mm = re.search(r'(\d{8,})', Path(path).name)
        eid = int(mm.group(1)) if mm else local_id('replay:' + str(row.get('replay_uuid') or path))
    row['episode_id'] = eid
    names = row.get('team_names') or []
    if eid >= LOCAL_ID_MIN or any(str(n).startswith(('SELF:', 'LEAGUE:')) for n in names):
        row['local'] = True
        row['team_names'] = [local_label(n) for n in names]
        row['day_mtime'] = mtime_day(path)
    return [row], {}


TAIL_RE = {k: re.compile(rb'"' + k.encode() + rb'"\s*:\s*(\[[^\]]*\])') for k in ('rewards', 'statuses')}
TAIL_INFO_RE = re.compile(rb'"info"\s*:\s*(\{[^{}]*\})')


def tail_fields(path, n=1 << 16):
    """Top-level rewards / statuses written after "steps" (the replay's last bytes)."""
    if str(path).endswith('.gz'):
        tail = b''
        try:
            with gzip.open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 22), b''):
                    tail = (tail + chunk)[-n:]
        except (EOFError, OSError, zlib.error):      # truncated / corrupt file: only the header fields are known
            return {}
    else:
        with open(path, 'rb') as f:
            f.seek(max(0, os.path.getsize(path) - n))
            tail = f.read()
    return tail_values(tail)


def tail_values(tail):
    out = {}
    for k, rx in TAIL_RE.items():
        found = rx.findall(tail)
        if found:
            try:
                out[k] = json.loads(found[-1])
            except ValueError:
                pass
    found = TAIL_INFO_RE.findall(tail)          # top-level "info" written last ({"seed": N} in notebook replays)
    if found:
        try:
            info = json.loads(found[-1])
            if isinstance(info, dict) and info.get('seed') is not None:
                out['seed'] = info['seed']
        except ValueError:
            pass
    return out


def scan_capsule(path):
    """farmneat-public-episode-capsule-v1: keys are sorted, so configuration..focus precede the joint actions."""
    head = read_head(path, 16384)
    t = cut_prefix(head, b'joint_actions')
    focus = t.get('focus') or {}
    return [{'member': '', 'format': 'capsule', 'episode_id': as_int(t.get('episode_id')),
             'stored_bytes': os.path.getsize(path), 'config_diff': config_diff(t.get('configuration')),
             'extra': {'focus': focus, 'content_sha256': t.get('content_sha256')}}], {}


XSF_COLS = ['episode_id', 'create_time', 'seed', 'team_names', 'team_id_0', 'team_id_1', 'submission_id_0',
            'submission_id_1', 'initial_score_0', 'initial_score_1', 'updated_score_0', 'updated_score_1', 'reward_0',
            'reward_1', 'n_steps', 'first_bad_step', 'final_status', 'config', 'actions_zstd']


def scan_xsf_raw(path):
    """xishengfeng replay-db: one row per stored episode (seed + both seats' actions, zstd) in replay_db/shards/*."""
    import pyarrow.parquet as pq
    rows = []
    with zipfile.ZipFile(path) as z:
        shards = sorted(n for n in z.namelist() if re.search(r'shards/[^/]+\.parquet$', n))
        for s in shards:
            t = pq.read_table(io.BytesIO(z.read(s)), columns=XSF_COLS)
            acts = t.column('actions_zstd').to_pylist()
            t = t.drop(['actions_zstd'])
            for i, r in enumerate(t.to_pylist()):
                try:
                    names = json.loads(r['team_names']) if r['team_names'] else None
                except ValueError:
                    names = None
                try:
                    status = json.loads(r['final_status']) if r['final_status'] else None
                except ValueError:
                    status = None
                try:
                    cfg = json.loads(r['config']) if r['config'] else None
                except ValueError:
                    cfg = None
                rows.append({'member': f'{s}#{i}', 'format': 'xsf-tape', 'episode_id': as_int(r['episode_id']),
                             'stored_bytes': len(acts[i]) if acts[i] is not None else None,
                             'team_names': names, 'team_ids': [r['team_id_0'], r['team_id_1']],
                             'submission_ids': [r['submission_id_0'], r['submission_id_1']],
                             'rewards': [r['reward_0'], r['reward_1']], 'statuses': status, 'seed': r['seed'],
                             'config_diff': config_diff(cfg), 'create_time': r['create_time'],
                             'extra': {'n_steps': r['n_steps'], 'first_bad_step': r['first_bad_step'],
                                       'initial_scores': [r['initial_score_0'], r['initial_score_1']],
                                       'updated_scores': [r['updated_score_0'], r['updated_score_1']],
                                       'has_actions': acts[i] is not None}})
    return rows, {'shards': len(shards)}


def scan_task(c):
    t0 = time.time()
    rec = dict(c)
    try:
        st = os.stat(c['path'])
        rec['size'], rec['mtime_ns'] = st.st_size, st.st_mtime_ns
        fn = {'zip': scan_zip, 'loose': scan_loose, 'capsule': scan_capsule, 'xsf-raw': scan_xsf_raw}[c['kind']]
        rows, note = fn(c['path'])
        rec['note'] = json.dumps(note, ensure_ascii=False) if note else ''
        rec['error'] = ''
    except Exception as e:  # noqa: BLE001 - container-level failure is recorded, rescanned next update
        rows = []
        rec.setdefault('size', None)
        rec.setdefault('mtime_ns', None)
        rec['note'], rec['error'] = '', repr(e)[:500]
    rec['n_rows'] = len(rows)
    rec['seconds'] = round(time.time() - t0, 3)
    rec['scanned_utc'] = utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')
    return rec, rows


# ------------------------------------------------------------------ scan cache
def _pq():
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa, pq


def write_table(rows, schema, path):
    pa, pq = _pq()
    t = pa.Table.from_pylist(rows, schema=schema)
    tmp = Path(str(path) + '.tmp')
    pq.write_table(t, tmp, compression='zstd')
    os.replace(tmp, path)


CONT_FIELDS = [('path', 'string'), ('kind', 'string'), ('source', 'string'), ('subset', 'string'), ('size', 'int64'),
               ('mtime_ns', 'int64'), ('n_rows', 'int64'), ('seconds', 'float64'), ('scanned_utc', 'string'),
               ('note', 'string'), ('error', 'string')]


def load_cache():
    _, pq = _pq()
    cp, sp = CAT / 'state/CONTAINERS.parquet', CAT / 'state/SCAN.parquet'
    if not cp.exists() or not sp.exists():
        return {}, collections.defaultdict(list)
    conts = {r['path']: r for r in pq.read_table(cp).to_pylist()}
    rows = collections.defaultdict(list)
    for path, raw in zip(*[pq.read_table(sp, columns=['container', 'raw']).column(k).to_pylist()
                           for k in ('container', 'raw')]):
        rows[path].append(json.loads(raw))
    return conts, rows


def save_cache(conts, rows):
    pa, _ = _pq()
    cs = pa.schema([(k, getattr(pa, t)()) for k, t in CONT_FIELDS])
    write_table([{k: c.get(k) for k, _ in CONT_FIELDS} for c in conts.values()], cs, CAT / 'state/CONTAINERS.parquet')
    ss = pa.schema([('container', pa.string()), ('raw', pa.string())])
    flat = [{'container': p, 'raw': json.dumps(r, ensure_ascii=False, separators=(',', ':'))}
            for p in sorted(rows) for r in rows[p]]
    write_table(flat, ss, CAT / 'state/SCAN.parquet')


def run_scan(full=False, rediscover=False, only=None):
    conts, rows = ({}, collections.defaultdict(list)) if full else load_cache()
    found = discover(rediscover=rediscover or full)
    if only:
        found = [c for c in found if c['source'] in only]
    todo, keep = [], {}
    for c in found:
        try:
            st = os.stat(c['path'])
        except OSError:
            continue
        old = conts.get(c['path'])
        if (not only and old and not old.get('error') and old.get('size') == st.st_size
                and old.get('mtime_ns') == st.st_mtime_ns and old.get('kind') == c['kind']):
            keep[c['path']] = dict(old, source=c['source'], subset=c['subset'])
        else:
            todo.append(c)
    if only:                            # rescan only the named sources; keep every other cached container
        for p, old in conts.items():
            if old.get('source') not in only and p not in keep and os.path.exists(p):
                keep[p] = old
    gone = [p for p in conts if p not in keep and p not in {c['path'] for c in todo}]
    log(f'containers: {len(found)} found{" in " + ",".join(sorted(only)) + " (forced rescan)" if only else ""}, '
        f'{len(keep)} kept from cache, {len(todo)} to scan, {len(gone)} gone')
    new_rows = {p: rows[p] for p in keep}
    todo.sort(key=lambda c: -os.path.getsize(c['path']))
    done = 0
    t0 = time.time()
    if todo:
        with mp.get_context('fork').Pool(WORKERS, initializer=be_nice) as pool:
            for rec, rs in pool.imap_unordered(scan_task, todo, chunksize=1):
                keep[rec['path']] = rec
                new_rows[rec['path']] = rs
                done += 1
                if done % 200 == 0 or done == len(todo):
                    log(f'  scanned {done}/{len(todo)} containers ({time.time() - t0:.0f}s)')
    save_cache(keep, new_rows)
    return keep, new_rows, {'found': len(found), 'unchanged': len(found) - len(todo), 'scanned': len(todo),
                            'gone': len(gone)}


# ------------------------------------------------------------------ metadata (read-only)
def load_extra_manifests():
    """(archive, member) -> (row, manifest path); per-folder manifest rows for reconciliation."""
    by_copy, per_folder, dups = {}, collections.defaultdict(list), collections.Counter()
    for mf in sorted(EXTRA.glob('*/MANIFEST*.jsonl')) + sorted(EXTRA.glob('skirmishes/*/MANIFEST*.jsonl')):
        folder = mf.parent
        recon = folder.name == 'reconstructed'
        day = DAY_RE.search(mf.name)
        for n, line in enumerate(open(mf, encoding='utf-8')):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                dups['unparseable:' + str(mf)] += 1
                continue
            eid = as_int(r.get('episode_id'))
            if eid is None:
                continue
            if recon:
                if not day:
                    continue
                archive = folder / f'kaggriculture-episodes-{day.group(1)}.zip'
            else:
                archive = folder / r.get('archive', '')
            member = r.get('member') or f'{eid}.json'
            key = (str(archive), member)
            if key in by_copy:
                dups[str(mf)] += 1
            by_copy[key] = (r, str(mf))
            per_folder[str(mf)].append((str(archive), member, eid))
    return by_copy, per_folder, dups


def newest(paths, stamp_re):
    best = None
    for p in paths:
        m = re.search(stamp_re, Path(p).name)
        if m and (best is None or m.group(1) > best[0]):
            best = (m.group(1), p)
    return best[1] if best else None


def load_xsf_index():
    import pyarrow.parquet as pq
    path = newest(glob.glob(str(EXTRA / 'index/xsf-episodes_index-*.parquet')), r'(\d{8}T\d{4})')
    if not path:
        return {}, None
    return {int(r['episode_id']): r for r in pq.read_table(path).to_pylist()}, path


def load_xsf_teams():
    import pyarrow.parquet as pq
    for z in sorted((EXTRA / 'raw').glob('*/*.zip')):
        try:
            with zipfile.ZipFile(z) as zz:
                n = next((x for x in zz.namelist() if x.endswith('index/teams.parquet')), None)
                if n:
                    return {int(r['team_id']): r['team_name'] for r in pq.read_table(io.BytesIO(zz.read(n))).to_pylist()}
        except Exception:  # noqa: BLE001
            pass
    return {}


def _lb_stamp(name):
    m = re.search(r'(\d{8}T\d{4})', name)
    if m:
        return m.group(1)
    m = re.search(r'(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d)', name)
    return f'{m.group(1)}{m.group(2)}{m.group(3)}T{m.group(4)}{m.group(5)}' if m else None


def load_leaderboards():
    files = sorted(set(glob.glob(str(EXTRA / '**/LEADERBOARD-*.csv'), recursive=True)
                       + glob.glob(str(EXTRA / 'index/*leaderboard*.csv'))))
    snaps = []
    for p in files:
        stamp = _lb_stamp(Path(p).name)
        if not stamp:
            continue
        try:
            rows = list(csv.DictReader(io.StringIO(Path(p).read_text(encoding='utf-8-sig'))))
        except Exception:  # noqa: BLE001
            continue
        snaps.append((stamp, p, rows))
    snaps.sort()
    current, best, names = {}, {}, collections.defaultdict(set)
    for stamp, p, rows in snaps:
        for r in rows:
            tid, rank = as_int(r.get('TeamId')), as_int(r.get('Rank'))
            if tid is None or rank is None:
                continue
            names[tid].add(r.get('TeamName'))
            if tid not in best or rank < best[tid][0]:
                best[tid] = (rank, stamp)
    latest = snaps[-1] if snaps else None
    if latest:
        for r in latest[2]:
            tid = as_int(r.get('TeamId'))
            if tid is not None:
                current[tid] = (as_int(r.get('Rank')), as_float(r.get('Score')), r.get('TeamName'))
    return {'current': current, 'best': best, 'names': names, 'path': latest[1] if latest else None,
            'stamp': latest[0] if latest else None, 'files': [s[1] for s in snaps]}


def load_our_subs():
    """Our submission id -> deployed version (folder with SUBMISSION.json; VERSIONS.tsv adds the ones without)."""
    subs = {}
    extra = HERE / 'VERSIONS.tsv'
    if extra.exists():
        for r in csv.DictReader(open(extra, encoding='utf-8'), delimiter='\t'):
            if as_int(r.get('submission_id')) and r.get('version'):
                subs[int(r['submission_id'])] = r['version']
    for p in sorted(DEPLOY.rglob('SUBMISSION.json')):
        try:
            sid = as_int(json.loads(p.read_text()).get('submission_id'))
        except Exception:  # noqa: BLE001
            continue
        if sid:
            rel = p.parent.relative_to(DEPLOY)
            subs[sid] = str(rel) if str(rel) != '.' else 'deployment-root'
    return subs


def _listing_entry(e, label):
    return {'from': label, 'create_time': str(e.get('create_time') or '') or None, 'type': e.get('type'),
            'state': e.get('state'), 'team_ids': pair(e.get('team_ids')), 'team_names': pair(e.get('team_names')),
            'sub_ids': pair(e.get('sub_ids') or e.get('submission_ids')), 'rewards': pair(e.get('rewards'))}


def load_listings():
    """episode id -> [known facts] from every API listing, capsule index, beaters log and hosted episode list."""
    K = collections.defaultdict(list)
    sub_scores, team_policies = {}, {}
    for p in sorted(EXTRA.glob('**/LISTINGS-*.json')):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        label = f'listing:{p.parent.name}'
        for e in d.get('episodes') or []:
            if as_int(e.get('episode_id')) is not None:
                K[int(e['episode_id'])].append(_listing_entry(e, label))
        for tid, t in (d.get('teams') or {}).items():
            team_policies.setdefault(int(tid), {}).update({int(s): float(sc) for s, sc in t.get('policies') or []})
    for p in sorted(EXTRA.glob('*/listings/*.json')):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        for e in d if isinstance(d, list) else d.get('episodes') or []:
            if as_int(e.get('episode_id')) is not None:
                K[int(e['episode_id'])].append(_listing_entry(e, f'listing:{p.parent.parent.name}'))
    for p in sorted(EXTRA.glob('**/SUBMISSIONS-*.json')):
        try:
            for s in json.loads(p.read_text()):
                sc = as_float(s.get('publicScore'))
                if sc is not None and as_int(s.get('id')):
                    sub_scores[int(s['id'])] = max(sc, sub_scores.get(int(s['id']), sc))
        except Exception:  # noqa: BLE001
            continue
    bp = AUTOSUBMIT / 'BEATERS.jsonl'
    if bp.exists():
        for line in open(bp, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                e = _listing_entry({**r, 'sub_ids': r.get('sub_ids')}, 'beaters')
                s = as_int(r.get('our_seat'))
                if s in (0, 1):
                    e['team_names'][s] = OUR_TEAM
                    e['team_names'][1 - s] = r.get('opponent')
                    e['rewards'][s], e['rewards'][1 - s] = r.get('our_cash'), r.get('opp_cash')
                e['type'] = 'EPISODE_TYPE_PUBLIC'
                K[int(r['episode_id'])].append(e)
    for p in sorted((REPO / 'work/datasets/public-episodes').glob('*/INDEX*.json')):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        team = d.get('team')
        for e in d.get('episodes') or []:
            s, eid = as_int(e.get('seat')), as_int(e.get('episode_id'))
            if s not in (0, 1) or eid is None:
                continue
            names, subs, rw = [None, None], [None, None], [None, None]
            names[s], names[1 - s] = team, e.get('rival_team')
            subs[s], subs[1 - s] = as_int(e.get('submission_id')), as_int(e.get('rival_submission_id'))
            rw[s], rw[1 - s] = e.get('reward'), e.get('rival_reward')
            K[eid].append({'from': f'capsule-index:{p.parent.name}', 'create_time': None, 'end_time': e.get('end'),
                           'type': 'EPISODE_TYPE_PUBLIC', 'state': 'COMPLETED', 'team_ids': [None, None],
                           'team_names': names, 'sub_ids': subs, 'rewards': rw})
    for p in sorted(DEPLOY.glob('**/EPISODES.csv')):
        version = repo_hosted_version(p)
        for r in csv.DictReader(open(p, encoding='utf-8')):
            eid = as_int(r.get('id'))
            if eid is None:
                continue
            K[eid].append({'from': f'hosted-episodes:{version}', 'create_time': r.get('createTime'),
                           'type': str(r.get('type') or '').split('.')[-1] or None,
                           'state': str(r.get('state') or '').split('.')[-1] or None, 'team_ids': [None, None],
                           'team_names': [None, None], 'sub_ids': [None, None], 'rewards': [None, None],
                           'our_version': version})
    hp = DEPLOY / 'HOSTED-INDEX.jsonl'
    if hp.exists():
        for line in open(hp, encoding='utf-8'):
            if not line.strip():
                continue
            r = json.loads(line)
            s = as_int(r.get('our_seat'))
            names, rw = [None, None], [None, None]
            if s in (0, 1):
                names[s], names[1 - s] = OUR_TEAM, r.get('opponent')
                rw[s], rw[1 - s] = r.get('cash'), r.get('opp_cash')
            K[int(r['episode_id'])].append({'from': 'hosted-index', 'create_time': None, 'type': None, 'state': None,
                                            'team_ids': [None, None], 'team_names': names, 'sub_ids': [None, None],
                                            'rewards': rw, 'our_version': r.get('archive')})
    return K, sub_scores, team_policies


def load_static_labels():
    static, labelled = set(), set()
    for p in TAPE_LABELS:
        if not p.exists():
            continue
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (int(r['episode_id']), int(r['seat']))
                labelled.add(key)
                if r.get('tape_id') is not None or float(r.get('match') or 0.0) >= 0.95:
                    static.add(key)
    return static, labelled


def load_daily_index():
    p = DATA / 'incoming/episodes-index-manifest.csv'
    if not p.exists():
        return {}
    return {r['date']: as_int(r['episode_count']) for r in csv.DictReader(open(p, encoding='utf-8'))}


# ------------------------------------------------------------------ join
def source_rank(source):
    return SOURCE_RANK.get(source, API_RANK)


def fill_pair(cur, new):
    cur, new = pair(cur), pair(new)
    return [cur[i] if cur[i] is not None else new[i] for i in (0, 1)]


def is_ours_name(n, names=OUR_NAMES):
    return n in names or (isinstance(n, str) and n.startswith('SELF:'))


def our_team_names(K=None):
    """Every display name our team id has had (the team was renamed on Kaggle: replays show e.g. '怎么做热干面?')."""
    names = set(OUR_NAMES)
    for fs in (K or {}).values():
        for f in fs:
            ids, ns = pair(f.get('team_ids')), pair(f.get('team_names'))
            names |= {ns[s] for s in (0, 1) if as_int(ids[s]) == OUR_TID and ns[s]}
    names |= {n for n in load_leaderboards()['names'].get(OUR_TID, ()) if n}
    return names


SKIRMISH_SOURCES = ('skirmishes', 'skirmishes-loose')
SKIRMISH_FILE_RE = re.compile(r'^(\d+)_(.+?)__(.+?)\.json(?:\.gz)?$')


def skirmish_copy(path, r, eid, mrow, mpath, skirmish_raw):
    """Skirmishes are games between our own versions (Kaggle CPU notebooks): local games, seats SELF:<version>. A raw
    replay (`raw/<run>/replays/<seed>_<A>__<B>.json.gz`, no EpisodeId: the file-name number is a seed) is the same game
    as the packed member whose manifest row names it in `raw`; an unpacked one gets its own local id."""
    r = dict(r, local=True)
    if r.get('format') != 'zip-json' and not r.get('header_eid'):
        packed = skirmish_raw.get(str(path))      # its bytes differ from the packed member (header added): no sha reuse
        eid = as_int(packed[0].get('episode_id')) if packed else local_id(
            'skirmish:' + (os.path.relpath(path, EXTRA) if str(path).startswith(str(EXTRA)) else str(path)))
        mm = SKIRMISH_FILE_RE.match(Path(path).name)
        if mm and not any(r.get('team_names') or []):
            r['team_names'] = [mm.group(2), mm.group(3)]
        if not r.get('day_mtime') and os.path.exists(path):
            r['day_mtime'] = mtime_day(path)
    names = r.get('team_names') or (mrow or {}).get('team_names')
    if names:
        r['team_names'] = [n if n is None or str(n).startswith(('SELF:', 'LEAGUE:')) else 'SELF:' + str(n)
                           for n in names]
    return r, eid, mrow, mpath


def join(conts, rows):
    t0 = time.time()
    manifests, per_folder, mdups = load_extra_manifests()
    xsf, xsf_path = load_xsf_index()
    xsf_teams = load_xsf_teams()
    K, sub_scores, team_policies = load_listings()
    lb = load_leaderboards()
    our_subs = load_our_subs()
    static, labelled = load_static_labels()
    log(f'metadata loaded ({time.time() - t0:.0f}s): manifests {len(manifests)}, xsf {len(xsf)}, listings {len(K)}, '
        f'leaderboard {lb["path"]}')
    for x in xsf.values():                 # our submissions seen in the crawl
        for s in (0, 1):
            if x.get(f'team_id_{s}') == OUR_TID and x.get(f'submission_id_{s}') and x[f'submission_id_{s}'] not in our_subs:
                our_subs[x[f'submission_id_{s}']] = f'sub:{x[f"submission_id_{s}"]}'

    # ---- copies
    skirmish_raw = {}                  # extra/skirmishes/raw replay path -> its packed copy's manifest row
    for (arch, member), (m, mpath) in manifests.items():
        if '/skirmishes/' in str(mpath) and m.get('raw'):
            skirmish_raw[str(m['raw'])] = (m, mpath)
    copies = []
    for path, rs in rows.items():
        c = conts[path]
        for r in rs:
            eid = as_int(r.get('episode_id'))
            if eid is None:
                continue
            if not valid_catalog_id(eid):
                log(f'skip out-of-range episode_id {eid} from {path}:{r.get("member") or ""} '
                    '(source file retained; catalog keys are int64)')
                continue
            mrow, mpath = manifests.get((path, r.get('member') or ''), (None, None))
            if mrow is None and c['kind'] == 'zip':
                mrow, mpath = manifests.get((path, f'{eid}.json'), (None, None))
            if c['source'] in SKIRMISH_SOURCES:
                r, eid, mrow, mpath = skirmish_copy(path, r, eid, mrow, mpath, skirmish_raw)
            copies.append({'r': r, 'c': c, 'eid': eid, 'mrow': mrow, 'mpath': mpath})
    log(f'{len(copies)} stored copies')

    # ---- per-episode merge
    E = {}

    def ep(eid):
        e = E.get(eid)
        if e is None:
            e = E[eid] = {'episode_id': eid, 'copies': [], 'known_from': set(), 'team_names': [None, None],
                          'team_ids': [None, None], 'submission_ids': [None, None], 'rewards': [None, None],
                          'statuses': None, 'seed': None, 'config_diff': None, 'module_version': None,
                          'create_time': None, 'day': None, 'day_from': None, 'episode_type': None,
                          'game_scores': [None, None], 'official': None, 'fetch_reasons': set(), 'local': False,
                          'versions': set(), 'end_time': None}
        return e

    copies.sort(key=lambda x: (source_rank(x['c']['source']), x['c']['path'], x['r'].get('member') or ''))
    for x in copies:
        r, c, e = x['r'], x['c'], ep(x['eid'])
        e['copies'].append(x)
        if r.get('local') or c['source'] in LOCAL_SOURCES:
            e['local'] = True
        ok = not r.get('error')
        if ok and r.get('team_names'):
            e['team_names'] = fill_pair(e['team_names'], r['team_names'])
        if ok:
            e['rewards'] = fill_pair(e['rewards'], r.get('rewards'))
            e['team_ids'] = fill_pair(e['team_ids'], r.get('team_ids'))
            e['submission_ids'] = fill_pair(e['submission_ids'], r.get('submission_ids'))
            for k in ('statuses', 'seed', 'module_version'):
                if e[k] is None and r.get(k) is not None:
                    e[k] = r[k]
            if e['config_diff'] is None and r.get('config_diff') is not None:
                e['config_diff'] = r['config_diff']
            if r.get('create_time') and not e['create_time']:
                e['create_time'], e['day_from'] = r['create_time'], 'xsf-raw'
        if r.get('official'):
            e['official'] = r['official']
            if r['official'].get('create_time'):
                e['create_time'], e['day_from'] = r['official']['create_time'], 'official-manifest.csv'
        if x['mrow']:
            m = x['mrow']
            e['team_ids'] = fill_pair(e['team_ids'], [as_int(v) for v in pair(m.get('team_ids'))])
            e['submission_ids'] = fill_pair(e['submission_ids'], [as_int(v) for v in pair(m.get('submission_ids'))])
            e['team_names'] = fill_pair(e['team_names'], m.get('team_names'))
            e['rewards'] = fill_pair(e['rewards'], m.get('rewards'))
            if m.get('scores'):
                e['game_scores'] = fill_pair(e['game_scores'], m.get('scores'))
            reason = m.get('reason') or m.get('cand_source')
            if reason:
                e['fetch_reasons'].add(str(reason))
            if m.get('version'):
                e['versions'].add(m['version'])
        if c['source'] == 'repo-hosted':
            e['versions'].add(c['subset'])
        if c['source'] in ('repo-misc',) and str(c['path']).startswith(str(DEPLOY)):
            e['versions'].add(repo_hosted_version(c['path']))
        if r.get('format') == 'capsule':
            f = (r.get('extra') or {}).get('focus') or {}
            s = as_int(f.get('seat'))
            if s in (0, 1) and f.get('submission_id'):
                subs = [None, None]
                subs[s] = as_int(f['submission_id'])
                e['submission_ids'] = fill_pair(e['submission_ids'], subs)
        if r.get('format') == 'xsf-tape':
            upd = (r.get('extra') or {}).get('updated_scores')
            if upd:
                e['game_scores'] = fill_pair(e['game_scores'], upd)

    for eid, x in xsf.items():
        e = ep(eid)
        e['known_from'].add('xsf-index')
        e['team_ids'] = fill_pair(e['team_ids'], [x.get('team_id_0'), x.get('team_id_1')])
        e['submission_ids'] = fill_pair(e['submission_ids'], [x.get('submission_id_0'), x.get('submission_id_1')])
        e['game_scores'] = [x.get('score_0'), x.get('score_1')]     # rating after the game (xsf crawl)
        e['rewards'] = fill_pair(e['rewards'], [x.get('reward_0'), x.get('reward_1')])
        e['episode_type'] = e['episode_type'] or x.get('type')
        if x.get('create_time') and e['day_from'] != 'official-manifest.csv':
            e['create_time'], e['day_from'] = x['create_time'], 'xsf-index'
    for eid, facts in K.items():
        e = ep(eid)
        for f in facts:
            e['known_from'].add(f['from'].split(':')[0] if f['from'].startswith('hosted-episodes') else f['from'])
            e['team_ids'] = fill_pair(e['team_ids'], [as_int(v) for v in f['team_ids']])
            e['team_names'] = fill_pair(e['team_names'], f['team_names'])
            e['submission_ids'] = fill_pair(e['submission_ids'], [as_int(v) for v in f['sub_ids']])
            e['rewards'] = fill_pair(e['rewards'], f['rewards'])
            e['episode_type'] = e['episode_type'] or f.get('type')
            if f.get('create_time') and not e['create_time']:
                e['create_time'], e['day_from'] = f['create_time'], f['from'].split(':')[0]
            if f.get('end_time') and not e['end_time']:
                e['end_time'] = f['end_time']
            if f.get('our_version'):
                e['versions'].add(f['our_version'])

    # team names <-> ids
    name_ids = collections.defaultdict(set)
    id_names = collections.defaultdict(set)
    for e in E.values():
        for s in (0, 1):
            n, t = e['team_names'][s], e['team_ids'][s]
            if n and t is not None and not e['local']:
                name_ids[n].add(t)
                id_names[t].add(n)
    for tid, names in lb['names'].items():
        for n in names:
            name_ids[n].add(tid)
            id_names[tid].add(n)
    for tid, n in xsf_teams.items():
        id_names[tid].add(n)
    ours_names = set(OUR_NAMES) | {n for n in id_names.get(OUR_TID, set()) if n}
    for n in ours_names:
        name_ids[n] = {OUR_TID}

    # create-time neighbours for day estimates (episode ids increase with time)
    known_ct = sorted((eid, e['create_time'][:10]) for eid, e in E.items()
                      if e['create_time'] and eid < LOCAL_ID_MIN and DAY_RE.match(e['create_time'][:10]))
    kid = [k for k, _ in known_ct]

    for eid, e in E.items():
        e['kaggle'] = not e['local'] and eid < LOCAL_ID_MIN
        if e['kaggle']:
            for s in (0, 1):
                n = e['team_names'][s]
                if e['team_ids'][s] is None and n and len(name_ids.get(n, ())) == 1:
                    e['team_ids'][s] = next(iter(name_ids[n]))
                if n is None and e['team_ids'][s] is not None:
                    e['team_names'][s] = xsf_teams.get(e['team_ids'][s]) or next(iter(sorted(id_names.get(e['team_ids'][s], ()))), None)
        if e['create_time']:
            e['day'] = e['create_time'][:10]
        else:
            day = None
            for x in e['copies']:
                if x['mrow'] and DAY_RE.fullmatch(str(x['mrow'].get('day') or '')):
                    day, e['day_from'] = x['mrow']['day'], 'manifest-day'
                    break
            if not day:
                for x in e['copies']:
                    if x['r'].get('day_mtime'):
                        day, e['day_from'] = x['r']['day_mtime'], 'file-mtime'
                        break
            if not day and e['kaggle'] and kid:
                i = bisect.bisect_left(kid, eid)
                cands = [known_ct[j] for j in (i - 1, i) if 0 <= j < len(known_ct)]
                near = min(cands, key=lambda kv: abs(kv[0] - eid)) if cands else None
                if near and abs(near[0] - eid) <= MAX_ID_GAP:
                    day, e['day_from'] = near[1], 'estimated-from-neighbour-id'
            e['day'] = day
        # ours
        ours = set()
        for s in (0, 1):
            n, t, sb = e['team_names'][s], e['team_ids'][s], e['submission_ids'][s]
            if (n and is_ours_name(n, ours_names)) or t == OUR_TID or (sb is not None and sb in our_subs and e['kaggle']):
                ours.add(s)
        e['involves_us'] = bool(ours)
        e['self_play'] = len(ours) == 2
        e['our_seat'] = next(iter(ours)) if len(ours) == 1 else None
        e['opponent'] = e['team_names'][1 - e['our_seat']] if e['our_seat'] is not None else None
        e['our_submission'] = e['submission_ids'][e['our_seat']] if e['our_seat'] is not None else None
        ver = None
        if e['kaggle'] and e['our_submission'] in our_subs:
            ver = our_subs[e['our_submission']]
        if not ver and e['versions']:
            ver = sorted(e['versions'])[0]
        if not ver and e['local'] and ours:
            ver = '+'.join(sorted({e['team_names'][s][5:] for s in ours if e['team_names'][s]}))
        if not ver and e['involves_us'] and any(x['c']['source'] == 'farm-repo' for x in e['copies']):
            ver = 'farm-repo'
        e['our_version'] = ver
        e['winner_seat'] = winner(e['rewards'])
        e['static_seats'] = [(eid, s) in static if (eid, s) in labelled else None for s in (0, 1)]
        cur = []
        for s in (0, 1):
            t = e['team_ids'][s]
            cur.append(lb['current'].get(t, (None,))[0] if t is not None else None)
        e['current_ranks'] = cur
        # copy agreement / content variants: distinct (bytes, CRC32) among replay copies that have both
        sigs, heads = set(), set()
        for x in e['copies']:
            r = x['r']
            if r.get('format') in REPLAY_FORMATS and r.get('crc32') is not None and r.get('bytes') is not None:
                sigs.add((r['bytes'], r['crc32']))
            if r.get('format') in REPLAY_FORMATS and not r.get('error') and r.get('team_names'):
                heads.add((json.dumps(r.get('team_names'), ensure_ascii=False),
                           json.dumps([as_float(v) for v in pair(r.get('rewards'))])))
        e['n_content_signatures'] = len(sigs)
        e['copies_disagree'] = len(heads) > 1

    # ---- rows
    copy_rows = []
    for e in E.values():
        for x in e['copies']:
            r, c, m = x['r'], x['c'], x['mrow'] or {}
            sha = m.get('sha256') or m.get('episode_sha256')
            detail = {k: v for k, v in m.items() if k not in (
                'archive', 'member', 'episode_id', 'day', 'sha256', 'episode_sha256', 'bytes', 'episode_bytes',
                'team_names', 'team_ids', 'submission_ids', 'rewards', 'winner_seat', 'seed', 'module_version',
                'fetched_utc')}
            if r.get('extra'):
                detail['scan'] = r['extra']
            if r.get('tag'):
                detail['tag'] = r['tag']
            fmt = r.get('format') or ('zip-json' if c['kind'] == 'zip' else c['kind'])
            names = r.get('team_names') if r.get('team_names') else e['team_names']
            rw = r.get('rewards') if r.get('rewards') else e['rewards']
            cd = r.get('config_diff')
            copy_rows.append({
                'episode_id': e['episode_id'], 'kaggle': e['kaggle'], 'source': c['source'], 'subset': c['subset'],
                'format': fmt, 'tsv_ready': fmt == 'zip-json' and not r.get('error'), 'archive': c['path'],
                'member': r.get('member') or '', 'bytes': as_int(m.get('bytes') or m.get('episode_bytes') or r.get('bytes')),
                'stored_bytes': as_int(r.get('stored_bytes')), 'crc32': as_int(r.get('crc32')), 'sha256': sha,
                'sha256_from': Path(x['mpath']).name if sha else None, 'day': e['day'], 'day_from': e['day_from'],
                'create_time': e['create_time'], 'team_names': [None if v is None else str(v) for v in pair(names)],
                'team_ids': [as_int(v) for v in e['team_ids']],
                'submission_ids': [as_int(v) for v in e['submission_ids']],
                'rewards': [as_float(v) for v in pair(rw)], 'winner_seat': winner(rw),
                'statuses': [str(v) for v in r.get('statuses')] if isinstance(r.get('statuses'), list) else None,
                'seed': as_int(r.get('seed') if r.get('seed') is not None else e['seed']),
                'config_ok': None if cd is None else not cd, 'config_diff': ','.join(cd) if cd else None,
                'module_version': r.get('module_version'), 'involves_us': e['involves_us'], 'our_seat': e['our_seat'],
                'self_play': e['self_play'], 'opponent': e['opponent'], 'our_version': e['our_version'],
                'our_submission': as_int(e['our_submission']),
                'fetch_reason': m.get('reason') or m.get('cand_source'),
                'reason_detail': json.dumps(detail, ensure_ascii=False, sort_keys=True) if detail else None,
                'fetched_utc': m.get('fetched_utc'), 'manifest': x['mpath'],
                'game_scores': [as_float(v) for v in e['game_scores']],
                'official_avg_score': (e['official'] or {}).get('avg_score'),
                'official_min_score': (e['official'] or {}).get('min_score'),
                'scan_error': r.get('error'), 'rank': source_rank(c['source'])})

    ep_rows = []
    for e in E.values():
        cs = sorted(e['copies'], key=lambda x: (not (x['r'].get('format', 'zip-json') == 'zip-json' and not x['r'].get('error')),
                                                source_rank(x['c']['source']),
                                                bool(x['r'].get('config_diff')), x['c']['path'], x['r'].get('member') or ''))
        pref = cs[0] if cs else None
        cd = e['config_diff']
        tsv_ready = any((x['r'].get('format') or 'zip-json') == 'zip-json' and not x['r'].get('error') for x in cs)
        ep_rows.append({
            'episode_id': e['episode_id'], 'kaggle': e['kaggle'], 'stored': bool(cs), 'n_copies': len(cs),
            'sources': sorted({x['c']['source'] for x in cs}, key=source_rank),
            'formats': sorted({x['r'].get('format') or 'zip-json' for x in cs}), 'tsv_ready': tsv_ready,
            'preferred_source': pref['c']['source'] if pref else None,
            'preferred_format': (pref['r'].get('format') or 'zip-json') if pref else None,
            'preferred_archive': pref['c']['path'] if pref else None,
            'preferred_member': (pref['r'].get('member') or '') if pref else None,
            'day': e['day'], 'day_from': e['day_from'], 'create_time': e['create_time'], 'episode_type': e['episode_type'],
            'team_names': [None if v is None else str(v) for v in e['team_names']],
            'team_ids': [as_int(v) for v in e['team_ids']], 'submission_ids': [as_int(v) for v in e['submission_ids']],
            'rewards': [as_float(v) for v in e['rewards']], 'winner_seat': e['winner_seat'],
            'statuses': [str(v) for v in e['statuses']] if isinstance(e['statuses'], list) else None,
            'seed': as_int(e['seed']), 'config_ok': None if cd is None else not cd,
            'config_diff': ','.join(cd) if cd else None, 'module_version': e['module_version'],
            'involves_us': e['involves_us'], 'our_seat': e['our_seat'], 'self_play': e['self_play'],
            'opponent': e['opponent'], 'our_version': e['our_version'], 'our_submission': as_int(e['our_submission']),
            'game_scores': [as_float(v) for v in e['game_scores']],
            'official_avg_score': (e['official'] or {}).get('avg_score'),
            'official_min_score': (e['official'] or {}).get('min_score'),
            'current_ranks': e['current_ranks'], 'static_seats': e['static_seats'],
            'fetch_reasons': sorted(e['fetch_reasons']), 'known_from': sorted(e['known_from']),
            'copies_disagree': e['copies_disagree'], 'n_content_signatures': e['n_content_signatures']})
    meta = {'xsf_index': xsf_path, 'leaderboard': lb['path'], 'leaderboard_stamp': lb['stamp'],
            'manifest_duplicate_rows': dict(mdups), 'our_subs': {str(k): v for k, v in sorted(our_subs.items())},
            'sub_scores': sub_scores, 'team_policies': team_policies, 'lb': lb, 'id_names': id_names,
            'name_ids': name_ids, 'per_folder': per_folder, 'xsf_teams': xsf_teams}
    log(f'joined {len(ep_rows)} episodes ({time.time() - t0:.0f}s)')
    return copy_rows, ep_rows, meta


# ------------------------------------------------------------------ schemas / writers
def copy_schema():
    import pyarrow as pa
    L = pa.list_
    return pa.schema([
        ('episode_id', pa.int64()), ('kaggle', pa.bool_()), ('source', pa.string()), ('subset', pa.string()),
        ('format', pa.string()), ('tsv_ready', pa.bool_()), ('archive', pa.string()), ('member', pa.string()),
        ('bytes', pa.int64()), ('stored_bytes', pa.int64()), ('crc32', pa.int64()), ('sha256', pa.string()),
        ('sha256_from', pa.string()), ('day', pa.string()), ('day_from', pa.string()), ('create_time', pa.string()),
        ('team_names', L(pa.string())), ('team_ids', L(pa.int64())), ('submission_ids', L(pa.int64())),
        ('rewards', L(pa.float64())), ('winner_seat', pa.int8()), ('statuses', L(pa.string())), ('seed', pa.int64()),
        ('config_ok', pa.bool_()), ('config_diff', pa.string()), ('module_version', pa.string()),
        ('involves_us', pa.bool_()), ('our_seat', pa.int8()), ('self_play', pa.bool_()), ('opponent', pa.string()),
        ('our_version', pa.string()), ('our_submission', pa.int64()), ('fetch_reason', pa.string()),
        ('reason_detail', pa.string()), ('fetched_utc', pa.string()), ('manifest', pa.string()),
        ('game_scores', L(pa.float64())), ('official_avg_score', pa.float64()), ('official_min_score', pa.float64()),
        ('scan_error', pa.string()), ('rank', pa.int32())])


def episode_schema():
    import pyarrow as pa
    L = pa.list_
    return pa.schema([
        ('episode_id', pa.int64()), ('kaggle', pa.bool_()), ('stored', pa.bool_()), ('n_copies', pa.int32()),
        ('sources', L(pa.string())), ('formats', L(pa.string())), ('tsv_ready', pa.bool_()),
        ('preferred_source', pa.string()), ('preferred_format', pa.string()), ('preferred_archive', pa.string()),
        ('preferred_member', pa.string()), ('day', pa.string()), ('day_from', pa.string()), ('create_time', pa.string()),
        ('episode_type', pa.string()), ('team_names', L(pa.string())), ('team_ids', L(pa.int64())),
        ('submission_ids', L(pa.int64())), ('rewards', L(pa.float64())), ('winner_seat', pa.int8()),
        ('statuses', L(pa.string())), ('seed', pa.int64()), ('config_ok', pa.bool_()), ('config_diff', pa.string()),
        ('module_version', pa.string()), ('involves_us', pa.bool_()), ('our_seat', pa.int8()), ('self_play', pa.bool_()),
        ('opponent', pa.string()), ('our_version', pa.string()), ('our_submission', pa.int64()),
        ('game_scores', L(pa.float64())), ('official_avg_score', pa.float64()), ('official_min_score', pa.float64()),
        ('current_ranks', L(pa.int64())), ('static_seats', L(pa.bool_())), ('fetch_reasons', L(pa.string())),
        ('known_from', L(pa.string())), ('copies_disagree', pa.bool_()), ('n_content_signatures', pa.int32())])


def team_key(name, tid):
    return f'id:{tid}' if tid is not None else f'name:{name}'


def build_teams(ep_rows, meta):
    lb = meta['lb']
    T = {}
    for e in ep_rows:
        if not e['kaggle']:
            continue
        for s in (0, 1):
            n, tid = e['team_names'][s], e['team_ids'][s]
            if n is None and tid is None:
                continue
            k = team_key(n, tid)
            t = T.get(k)
            if t is None:
                t = T[k] = {'team_id': tid, 'names': collections.Counter(), 'stored': 0, 'tsv_ready': 0, 'wins': 0,
                            'losses': 0, 'ties': 0, 'known_only': 0, 'known_wins': 0, 'known_losses': 0,
                            'beat_us': 0, 'lost_to_us': 0, 'days': set(), 'subs': collections.Counter(),
                            'sources': collections.Counter(), 'best_game_score': None, 'last_day': ''}
            if n:
                t['names'][n] += 1
            w = e['winner_seat']
            res = None if w is None and None in (e['rewards'] or [None]) else ('W' if w == s else 'L' if w == 1 - s else 'T')
            if e['stored']:
                t['stored'] += 1
                t['tsv_ready'] += bool(e['tsv_ready'])
                if res:
                    t[{'W': 'wins', 'L': 'losses', 'T': 'ties'}[res]] += 1
                if e['day']:
                    t['days'].add(e['day'])
                for src in e['sources']:
                    t['sources'][src] += 1
            else:
                t['known_only'] += 1
                if res == 'W':
                    t['known_wins'] += 1
                elif res == 'L':
                    t['known_losses'] += 1
            if e['day'] and e['day'] > t['last_day']:
                t['last_day'] = e['day']
            if e['involves_us'] and e['our_seat'] == 1 - s and res in ('W', 'L'):
                t['beat_us' if res == 'W' else 'lost_to_us'] += 1
            if e['submission_ids'][s] is not None:
                t['subs'][e['submission_ids'][s]] += 1
            gs = e['game_scores'][s]
            if gs is not None and (t['best_game_score'] is None or gs > t['best_game_score']):
                t['best_game_score'] = gs
    out = []
    for k, t in T.items():
        tid = t['team_id']
        cur = lb['current'].get(tid) if tid is not None else None
        best = lb['best'].get(tid) if tid is not None else None
        name = cur[2] if cur and cur[2] else (t['names'].most_common(1)[0][0] if t['names'] else '')
        other = sorted((set(t['names']) | (meta['id_names'].get(tid, set()) if tid is not None else set())) - {name})
        days = sorted(t['days'])
        policies = meta['team_policies'].get(tid, {}) if tid is not None else {}
        best_policy = max(policies.values()) if policies else None
        sub_best = max((meta['sub_scores'][s] for s in t['subs'] if s in meta['sub_scores']), default=None)
        out.append({'team': name, 'team_id': tid if tid is not None else '', 'other_names': '|'.join(o for o in other if o),
                    'current_rank': cur[0] if cur else '', 'current_score': cur[1] if cur else '',
                    'best_rank_seen': best[0] if best else '',
                    'best_known_score': round(max([v for v in (t['best_game_score'], best_policy, sub_best,
                                                               cur[1] if cur else None) if v is not None], default=0.0), 1)
                    or '',
                    'games_stored': t['stored'], 'tsv_ready': t['tsv_ready'], 'wins': t['wins'], 'losses': t['losses'],
                    'ties': t['ties'], 'known_not_stored': t['known_only'], 'known_not_stored_wins': t['known_wins'],
                    'known_not_stored_losses': t['known_losses'], 'beat_us': t['beat_us'], 'lost_to_us': t['lost_to_us'],
                    'first_day': days[0] if days else '', 'last_day': days[-1] if days else t['last_day'],
                    'days_stored': len(days),
                    'submissions': ' '.join(f'{s}:{n}' for s, n in t['subs'].most_common(6)),
                    'sources': ' '.join(f'{s}:{n}' for s, n in t['sources'].most_common(6))})
    out.sort(key=lambda r: (r['current_rank'] if r['current_rank'] != '' else 10 ** 9, -r['games_stored'], r['team']))
    return out


TEAM_COLS = ['team', 'team_id', 'other_names', 'current_rank', 'current_score', 'best_rank_seen', 'best_known_score',
             'games_stored', 'tsv_ready', 'wins', 'losses', 'ties', 'known_not_stored', 'known_not_stored_wins',
             'known_not_stored_losses', 'beat_us', 'lost_to_us', 'first_day', 'last_day', 'days_stored', 'submissions',
             'sources']


def write_tsv(path, cols, rows):
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write('\t'.join(cols) + '\n')
        for r in rows:
            f.write('\t'.join('' if r.get(c) is None else str(r.get(c)).replace('\t', ' ') for c in cols) + '\n')
    os.replace(tmp, path)


def summarize(conts, copy_rows, ep_rows, meta, scan_info, seconds):
    by_src = collections.defaultdict(lambda: {'containers': 0, 'container_bytes': 0, 'copies': 0, 'episodes': set(),
                                              'tsv_ready': 0, 'errors': 0, 'days': set(), 'formats': collections.Counter(),
                                              'skipped_files': 0})
    for c in conts.values():
        s = by_src[c['source']]
        if c.get('n_rows'):
            s['containers'] += 1
            s['container_bytes'] += c.get('size') or 0
        elif c.get('note'):
            s['skipped_files'] += 1
    for r in copy_rows:
        s = by_src[r['source']]
        s['copies'] += 1
        s['episodes'].add(r['episode_id'])
        s['tsv_ready'] += bool(r['tsv_ready'])
        s['errors'] += bool(r['scan_error'])
        s['formats'][r['format']] += 1
        if r['day']:
            s['days'].add(r['day'])
    sources = {}
    for k, s in sorted(by_src.items(), key=lambda kv: source_rank(kv[0])):
        days = sorted(s['days'])
        sources[k] = {'containers': s['containers'], 'copies': s['copies'], 'unique_episodes': len(s['episodes']),
                      'tsv_ready_copies': s['tsv_ready'], 'scan_errors': s['errors'],
                      'disk_bytes': s['container_bytes'], 'disk_gb': round(s['container_bytes'] / 1e9, 2),
                      'first_day': days[0] if days else None, 'last_day': days[-1] if days else None,
                      'formats': dict(s['formats']), 'files_skipped_not_replays': s['skipped_files']}
    stored = [e for e in ep_rows if e['stored']]
    multi = [e for e in stored if e['n_copies'] > 1]
    pairs = collections.Counter()
    for e in multi:
        srcs = e['sources']
        if len(srcs) > 1:
            pairs[' + '.join(srcs)] += 1
    differ = collections.Counter(' + '.join(e['sources']) for e in stored if e['n_content_signatures'] > 1)
    official = []
    for c in conts.values():
        if c['source'] == 'official' and c.get('note'):
            n = json.loads(c['note'])
            if n.get('manifest_csv_without_member') or n.get('members_not_in_manifest_csv'):
                official.append({'archive': c['path'], **{k: n.get(k) for k in (
                    'manifest_csv_rows', 'members', 'manifest_csv_without_member', 'members_not_in_manifest_csv')}})
    ours = collections.defaultdict(lambda: collections.Counter())
    for e in stored:
        if e['involves_us'] and e['kaggle']:
            v = e['our_version'] or 'unknown'
            if e['self_play']:
                ours[v]['self_play'] += 1
            elif e['winner_seat'] is None:
                ours[v]['T' if None not in (e['rewards'] or [None]) else 'unknown'] += 1
            else:
                ours[v]['W' if e['winner_seat'] == e['our_seat'] else 'L'] += 1
    cont_errors = [{'path': c['path'], 'error': c['error']} for c in conts.values() if c.get('error')]
    return {
        'built_utc': utc_now().strftime('%Y-%m-%dT%H:%M:%SZ'), 'seconds': round(seconds, 1), 'scan': scan_info,
        'leaderboard': meta['leaderboard'], 'xsf_index': meta['xsf_index'],
        'totals': {'containers': len(conts), 'copies': len(copy_rows), 'episodes_rows': len(ep_rows),
                   'stored_episodes': len(stored), 'stored_kaggle_episodes': sum(e['kaggle'] for e in stored),
                   'stored_local_episodes': sum(not e['kaggle'] for e in stored),
                   'stored_with_tsv_ready_copy': sum(e['tsv_ready'] for e in stored),
                   'stored_only_as_tape': sum(not e['tsv_ready'] and not (set(e['formats']) & {'json', 'json-gz'})
                                              for e in stored),
                   'known_not_stored': sum(not e['stored'] for e in ep_rows),
                   'episodes_with_multiple_copies': len(multi),
                   'episodes_in_multiple_sources': sum(len(e['sources']) > 1 for e in stored),
                   'episodes_with_differing_bytes': sum(e['n_content_signatures'] > 1 for e in stored),
                   'episodes_with_disagreeing_headers': sum(bool(e['copies_disagree']) for e in stored),
                   'config_ok_stored': sum(e['config_ok'] is True for e in stored),
                   'config_differs_stored': sum(e['config_ok'] is False for e in stored),
                   'disk_bytes': sum(c.get('size') or 0 for c in conts.values() if c.get('n_rows')),
                   'container_errors': len(cont_errors), 'copy_scan_errors': sum(bool(r['scan_error']) for r in copy_rows)},
        'sources': sources, 'duplicate_source_sets': dict(pairs.most_common(40)),
        'differing_content_source_sets': dict(differ.most_common(40)),
        'official_manifest_csv_mismatches': official,
        'our_games_by_version': {v: dict(c) for v, c in sorted(ours.items())},
        'container_errors': cont_errors[:50], 'manifest_duplicate_rows': meta['manifest_duplicate_rows']}


def write_outputs(conts, copy_rows, ep_rows, meta, scan_info, t0):
    write_table(copy_rows, copy_schema(), CAT / 'COPIES.parquet')
    write_table(ep_rows, episode_schema(), CAT / 'EPISODES.parquet')
    teams = build_teams(ep_rows, meta)
    write_tsv(CAT / 'TEAMS.tsv', TEAM_COLS, teams)
    summary = summarize(conts, copy_rows, ep_rows, meta, scan_info, time.time() - t0)
    tmp = CAT / 'SUMMARY.json.tmp'
    tmp.write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, CAT / 'SUMMARY.json')
    (CAT / 'state').mkdir(exist_ok=True)
    tmp = CAT / 'state/META.json.tmp'
    tmp.write_text(json.dumps({'our_subs': meta['our_subs'], 'sub_scores': meta['sub_scores'],
                               'leaderboard': meta['leaderboard'], 'leaderboard_stamp': meta['leaderboard_stamp'],
                               'per_folder_manifest_rows': {k: len(v) for k, v in meta['per_folder'].items()}},
                              ensure_ascii=False, indent=1))
    os.replace(tmp, CAT / 'state/META.json')
    return summary


class Lock:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.f = open(self.path, 'w')
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'another run holds {self.path}')
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def cmd_build(a, full):
    be_nice()
    t0 = time.time()
    (CAT / 'state').mkdir(parents=True, exist_ok=True)
    with Lock(CAT / 'state/.lock'):
        only = set(getattr(a, 'only', None) or []) or None
        conts, rows, info = run_scan(full=full, rediscover=a.rediscover, only=only)
        copy_rows, ep_rows, meta = join(conts, rows)
        s = write_outputs(conts, copy_rows, ep_rows, meta, info, t0)
    tot = s['totals']
    print(json.dumps({'copies': tot['copies'], 'stored_episodes': tot['stored_episodes'],
                      'known_not_stored': tot['known_not_stored'], 'scan': info, 'seconds': s['seconds']}))


# ------------------------------------------------------------------ hosted-ours (copy, never move)
def our_repo_replays():
    """Every Kaggle replay file under work/runs/deployment with our team in a seat: (path, version)."""
    out = []
    for p in sorted(DEPLOY.rglob('*')):
        n = p.name
        if not p.is_file() or p.is_symlink() or not n.endswith(('.json', '.json.gz')):
            continue
        if not (NAME_RE.match(n) and not SKIP_NAME_RE.search(n)):
            continue
        out.append((str(p), repo_hosted_version(p)))
    return out


def sync_hosted(dry_run=False):
    be_nice()
    HOSTED_OURS.mkdir(parents=True, exist_ok=True)
    (HOSTED_OURS / 'archives').mkdir(exist_ok=True)
    man = HOSTED_OURS / 'MANIFEST.jsonl'
    with Lock(HOSTED_OURS / '.lock'):
        done = set()
        if man.exists():
            done = {json.loads(l)['episode_id'] for l in open(man, encoding='utf-8') if l.strip()}
        for old in (HOSTED_OURS / 'archives').glob('*.zip.tmp'):   # our own leftovers of an interrupted sync
            old.unlink()
        cands = our_repo_replays()
        todo = []
        for path, version in cands:
            m = re.search(r'(\d{8,10})', Path(path).name)
            if m and int(m.group(1)) in done:
                continue
            todo.append((path, version))
        if not todo:
            print(json.dumps({'candidates': len(cands), 'new': 0}))
            return 0
        xsf, _ = load_xsf_index()
        K, _, _ = load_listings()
        our_names = our_team_names(K)
        our_subs = load_our_subs()
        ver_sub = {v: s for s, v in our_subs.items()}
        known_ct = sorted({(eid, x['create_time'][:10]) for eid, x in xsf.items() if x.get('create_time')}
                          | {(eid, f['create_time'][:10]) for eid, fs in K.items() for f in fs
                             if f.get('create_time') and DAY_RE.match(str(f['create_time'])[:10])})
        kid = [k for k, _ in known_ct]
        stamp = utc_now().strftime('%Y%m%dT%H%M%S')
        groups = collections.defaultdict(list)
        skipped = collections.Counter()
        seen_now = set()
        for path, version in todo:
            try:                                  # header pass only; full bytes are read one replay at a time below
                h = parse_header(read_head(path))
            except Exception as e:  # noqa: BLE001 - e.g. a replay still being written; retried next sync
                skipped['unreadable'] += 1
                log(f'skip {path}: {e!r}'[:300])
                continue
            info = h.get('info') or {}
            eid = as_int(info.get('EpisodeId'))
            names = info.get('TeamNames') or []
            m = re.search(r'(\d{8,10})', Path(path).name)
            if eid is None or (m and int(m.group(1)) != eid):
                skipped['episode id mismatch'] += 1
                continue
            seats = [i for i, n in enumerate(names) if n in our_names]
            if not seats and version in ver_sub:     # a display name we have not seen yet: our submission's seat
                for f in K.get(eid, []):
                    seats = [i for i, sb in enumerate(pair(f['sub_ids'])) if as_int(sb) == ver_sub[version]] or seats
            if not seats:
                skipped['not our game'] += 1
                continue
            if eid in done:
                skipped['already copied'] += 1
                continue
            if eid in seen_now:
                skipped['second file of the same episode'] += 1
                continue
            seen_now.add(eid)
            x = xsf.get(eid) or {}
            facts = K.get(eid, [])
            ct = next((f['create_time'] for f in facts if f['from'].startswith('hosted-episodes') and f.get('create_time')),
                      None) or x.get('create_time') or next((f['create_time'] for f in facts if f.get('create_time')), None)
            day, day_from = (ct[:10], 'create_time') if ct else (None, None)
            if not day and kid:
                i = bisect.bisect_left(kid, eid)
                near = [known_ct[j] for j in (i - 1, i) if 0 <= j < len(known_ct)]
                best = min(near, key=lambda kv: abs(kv[0] - eid)) if near else None
                if best and abs(best[0] - eid) <= MAX_ID_GAP:
                    day, day_from = best[1], 'estimated-from-neighbour-id'
            tids = pair([x.get('team_id_0'), x.get('team_id_1')] if x else None)
            subs = pair([x.get('submission_id_0'), x.get('submission_id_1')] if x else None)
            for f in facts:
                tids = fill_pair(tids, [as_int(v) for v in f['team_ids']])
                subs = fill_pair(subs, [as_int(v) for v in f['sub_ids']])
            our_seat = seats[0] if len(seats) == 1 else None
            for s in seats:
                tids[s] = OUR_TID
                if subs[s] is None and version in ver_sub:
                    subs[s] = ver_sub[version]
            rw = h.get('rewards')
            cd = config_diff(h.get('configuration'))
            meta = {'episode_id': eid, 'day': day or 'unknown', 'day_from': day_from, 'create_time': ct,
                    'source': 'repo-hosted-copy', 'cand_source': 'hosted-ours', 'reason': 'hosted-ours',
                    'team_names': names, 'team_ids': tids, 'submission_ids': subs, 'rewards': rw,
                    'winner_seat': winner(rw), 'statuses': h.get('statuses'), 'seed': info.get('seed'),
                    'module_version': h.get('module_version'), 'config_ok': None if cd is None else not cd,
                    'version': version, 'our_seat': our_seat, 'self_play': len(seats) == 2,
                    'opponent': names[1 - our_seat] if our_seat is not None else None,
                    'our_sub': subs[our_seat] if our_seat is not None else (subs[0] if len(seats) == 2 else None),
                    'origin': os.path.relpath(path, REPO)}
            groups[(meta['day'], version)].append((eid, path, meta))
        if dry_run:
            print(json.dumps({'candidates': len(cands), 'new': sum(len(v) for v in groups.values()),
                              'zips': len(groups), 'skipped': dict(skipped)}))
            return 0
        n = 0
        for (day, version), items in sorted(groups.items()):
            vtag = re.sub(r'[^A-Za-z0-9._-]+', '__', version)
            name = f'kaggriculture-hosted-{day}-{vtag}-{stamp}.zip'
            final = HOSTED_OURS / 'archives' / name
            tmp = HOSTED_OURS / 'archives' / (name + '.tmp')
            items, written = sorted(items, key=lambda t: t[0]), []
            with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for eid, path, meta in items:
                    try:                          # gzip CRC/length check: a replay still being written is skipped
                        raw = open(path, 'rb').read()
                        if path.endswith('.gz'):
                            raw = gzip.decompress(raw)
                    except Exception as e:  # noqa: BLE001
                        skipped['unreadable'] += 1
                        log(f'skip {path}: {e!r}'[:300])
                        continue
                    meta['sha256'], meta['bytes'] = hashlib.sha256(raw).hexdigest(), len(raw)
                    z.writestr(f'{eid}.json', raw)
                    written.append((eid, path, meta))
                    del raw
            items = written
            if not items:
                os.remove(tmp)
                continue
            with zipfile.ZipFile(tmp) as z:       # read back every member and compare its hash before committing
                for eid, path, meta in items:
                    hs = hashlib.sha256()
                    with z.open(f'{eid}.json') as f:
                        for chunk in iter(lambda: f.read(1 << 22), b''):
                            hs.update(chunk)
                    if hs.hexdigest() != meta['sha256']:
                        raise RuntimeError(f'zip readback hash mismatch {eid}')
            os.replace(tmp, final)
            now = utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')
            lines = [json.dumps(dict(meta, archive=f'archives/{name}', member=f'{eid}.json', fetched_utc=now),
                                sort_keys=True, ensure_ascii=False) for eid, path, meta in items]
            with open(man, 'a', encoding='utf-8') as f:   # manifest rows right after their zip is in place
                f.write('\n'.join(lines) + '\n')
                f.flush()
                os.fsync(f.fileno())
            n += len(items)
            log(f'hosted-ours: {name} ({len(items)} replays)')
        print(json.dumps({'candidates': len(cands), 'copied': n, 'zips': len(groups), 'skipped': dict(skipped)}))
        return n


# ------------------------------------------------------------------ readers for find / query
def load_aliases():
    p = HERE / 'ALIASES.tsv'
    out = {}
    if p.exists():
        for line in open(p, encoding='utf-8'):
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 2 and parts[0].strip() and parts[0].strip() != 'alias':
                out[fold(parts[0].strip())] = [n for n in parts[1].split('|') if n]
    return out


def load_episodes(columns=None, where=None):
    import pyarrow.parquet as pq
    t = pq.read_table(CAT / 'EPISODES.parquet', columns=columns)
    rows = t.to_pylist()
    return [r for r in rows if where(r)] if where else rows


def load_copies(eids=None, columns=None):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(CAT / 'COPIES.parquet', columns=columns)
    if eids is not None:
        t = t.filter(pc.is_in(t.column('episode_id'), value_set=pa.array(sorted(eids), type=pa.int64())))
    out = collections.defaultdict(list)
    for r in t.to_pylist():
        out[r['episode_id']].append(r)
    for v in out.values():
        v.sort(key=lambda r: (not r['tsv_ready'], r['rank'], r['archive'], r['member']))
    return out


def load_teams():
    p = CAT / 'TEAMS.tsv'
    return list(csv.DictReader(open(p, encoding='utf-8'), delimiter='\t')) if p.exists() else []


def match_teams(text, teams, aliases):
    """[(team row, how)] for alias / exact / substring / token-prefix matches of text against team names."""
    q = fold(text.strip())
    qt = tokens(text)
    found, seen = [], set()

    def names_of(t):
        return [t['team']] + [n for n in t['other_names'].split('|') if n]

    def add(t, how):
        k = (t['team'], t['team_id'])
        if k not in seen:
            seen.add(k)
            found.append((t, how))

    for target in aliases.get(q, []):
        for t in teams:
            if any(fold(n) == fold(target) for n in names_of(t)):
                add(t, f'alias {text!r} -> {target}')
    for t in teams:
        if any(fold(n) == q for n in names_of(t)):
            add(t, 'exact name')
    for t in teams:
        if any(q and q in fold(n) for n in names_of(t)):
            add(t, 'substring')
    if qt:
        for t in teams:
            for n in names_of(t):
                nt = tokens(n)
                if nt and all(any(w.startswith(x) for w in nt) for x in qt):
                    add(t, 'token')
                    break
    return found


def fmt_copy(c):
    where = f"{c['archive']}:{c['member']}" if c['member'] else c['archive']
    tags = [c['source'], c['format']]
    if c['format'] not in REPLAY_FORMATS:
        tags.append('tape: re-simulate')
    if c['scan_error']:
        tags.append('ERROR ' + c['scan_error'][:60])
    return f"{where}  [{', '.join(tags)}]"


def seat_result(e, s):
    w = e['winner_seat']
    if w is None:
        return 'T' if e['rewards'] and None not in e['rewards'] else '?'
    return 'W' if w == s else 'L'


def team_seats(e, ids, names):
    return [s for s in (0, 1) if (e['team_ids'][s] is not None and e['team_ids'][s] in ids)
            or (e['team_names'][s] is not None and fold(e['team_names'][s]) in names)]


EP_COLS = ['episode_id', 'kaggle', 'stored', 'n_copies', 'sources', 'tsv_ready', 'day', 'team_names', 'team_ids',
           'submission_ids', 'rewards', 'winner_seat', 'involves_us', 'our_seat', 'self_play', 'opponent', 'our_version',
           'our_submission', 'current_ranks', 'known_from', 'config_ok', 'episode_type', 'game_scores', 'static_seats',
           'preferred_archive', 'preferred_member', 'preferred_source']
COPY_COLS = ['episode_id', 'source', 'format', 'archive', 'member', 'tsv_ready', 'rank', 'scan_error']


def rank_lookup(teams):
    by_name = collections.defaultdict(list)
    for t in teams:
        by_name[fold(t['team'])].append(t)

    def rank_of(e, s):
        if e['current_ranks'] and e['current_ranks'][s]:
            return e['current_ranks'][s]
        n = e['team_names'][s]
        c = by_name.get(fold(n)) if n else None
        return (as_int(c[0]['current_rank']) if c and len(c) == 1 else None)
    return rank_of


def version_match(v, want):
    return v is not None and fold(want) in fold(v)


def filter_rows(rows, a):
    """rows = [(episode, seat)] seen from `seat`; apply the shared find filters."""
    if a.result:
        want = {'win': 'W', 'loss': 'L', 'tie': 'T'}[a.result]
        rows = [(e, s) for e, s in rows if seat_result(e, s) == want]
    if a.vs:
        vv = fold(a.vs)
        rows = [(e, s) for e, s in rows if e['team_names'][1 - s] and vv in fold(e['team_names'][1 - s])]
    if a.day_from:
        rows = [(e, s) for e, s in rows if (e['day'] or '') >= a.day_from]
    if a.day_to:
        rows = [(e, s) for e, s in rows if (e['day'] or '9999') <= a.day_to]
    if a.stored_only:
        rows = [(e, s) for e, s in rows if e['stored']]
    return sorted(rows, key=lambda es: (es[0]['day'] or '', es[0]['episode_id']), reverse=True)


def print_rows(rows, copies, rank_of, a, what='games'):
    st = [(e, s) for e, s in rows if e['stored']]
    res = collections.Counter(seat_result(e, s) for e, s in st)
    print(f"   {len(rows)} {what}: {len(st)} stored ({res['W']}W-{res['L']}L-{res['T']}T), "
          f"{len(rows) - len(st)} known with no local copy")
    shown = rows if a.all else rows[:a.limit]
    for e, s in shown:
        o = 1 - s
        rw = e['rewards'] or [None, None]
        score = f"{rw[s]:.0f}-{rw[o]:.0f}" if None not in rw else '?'
        r = rank_of(e, o)
        ver = f"  [{e['our_version']}]" if e['involves_us'] and e['our_seat'] == s and e['our_version'] else ''
        sub = e['submission_ids'][s]
        print(f"   {e['day'] or '?':10} {e['episode_id']:>9}  seat {s} {seat_result(e, s)} {score:>15}  sub {sub or '?':>8}"
              f"  vs {e['team_names'][o] or '?'}{f' (#{r})' if r else ''}{ver}")
        cs = copies.get(e['episode_id'], [])
        if not cs:
            print(f"       no local copy (known from: {', '.join(e['known_from']) or '?'})")
        for c in (cs if a.copies == 'all' else cs[:1]):
            print(f"       {fmt_copy(c)}")
        if a.copies == 'first' and len(cs) > 1:
            print(f"       (+{len(cs) - 1} more copies)")
    if len(rows) > len(shown):
        print(f"   ... {len(rows) - len(shown)} older {what} not shown (--all or --limit N)")


def team_summary(t, how):
    print(f"== {t['team']}  team id {t['team_id'] or '?'}  rank {t['current_rank'] or '-'}"
          f"{' (score ' + t['current_score'] + ')' if t['current_score'] else ''}  best rank seen "
          f"{t['best_rank_seen'] or '-'}  best known score {t['best_known_score'] or '-'}   [{how}]")
    print(f"   stored {t['games_stored']} ({t['wins']}W-{t['losses']}L-{t['ties']}T; {t['tsv_ready']} in zips), days "
          f"{t['first_day'] or '?'}..{t['last_day'] or '?'} ({t['days_stored']}), known not stored {t['known_not_stored']}; "
          f"vs us: beat us {t['beat_us']}, lost to us {t['lost_to_us']}")
    if t['other_names']:
        print(f"   other names: {t['other_names']}")
    if t['submissions']:
        print(f"   submissions (games): {t['submissions']}")


def cmd_find(a):
    text = ' '.join(a.text).strip()
    aliases, teams = load_aliases(), load_teams()
    rank_of = rank_lookup(teams)
    if text.isdigit():
        n = int(text)
        eps = load_episodes(EP_COLS, where=lambda e: e['episode_id'] == n or n in (e['submission_ids'] or [])
                            or n in (e['team_ids'] or []))
        exact = [e for e in eps if e['episode_id'] == n]
        if exact:
            e = exact[0]
            cs = load_copies({n}, COPY_COLS).get(n, [])
            print(f"episode {n}  day {e['day']}  {' vs '.join(str(x) for x in e['team_names'])}  rewards {e['rewards']}  "
                  f"winner seat {e['winner_seat']}")
            print(f"   submissions {e['submission_ids']}  team ids {e['team_ids']}  current ranks {e['current_ranks']}  "
                  f"type {e['episode_type']}  config_ok {e['config_ok']}")
            if e['involves_us']:
                print(f"   ours: seat {e['our_seat']}{' (self-play)' if e['self_play'] else ''}, version {e['our_version']}, "
                      f"submission {e['our_submission']}")
            print(f"   known from: {', '.join(e['known_from']) or '-'}")
            print(f"   {len(cs)} local copies:" if cs else '   no local copy')
            for c in cs:
                print(f"     {fmt_copy(c)}")
            return
        rows = [(e, e['submission_ids'].index(n)) for e in eps if n in (e['submission_ids'] or [])]
        if rows:
            e0, s0 = rows[0]
            print(f"submission {n}: team {e0['team_names'][s0]} (team id {e0['team_ids'][s0]})")
            rows = filter_rows(rows, a)
            copies = load_copies({e['episode_id'] for e, _ in rows}, COPY_COLS)
            print_rows(rows, copies, rank_of, a)
            return
        tm = [t for t in teams if t['team_id'] == str(n)]
        if not tm:
            print(f'{n}: no episode, submission or team id known locally')
            return
        text = tm[0]['team']
    found = match_teams(text, teams, aliases)
    if not found:
        print(f'no team matches {text!r} ({len(teams)} teams in TEAMS.tsv, {len(aliases)} aliases in ALIASES.tsv)')
        return
    direct = [th for th in found if th[1].startswith(('alias', 'exact'))]
    rest = [th for th in found if th not in direct]
    if direct and rest:
        if len(rest) <= 8:
            print(f"also matching {text!r}: " + ', '.join(f"{t['team']} ({t['games_stored']} stored)" for t, _ in rest))
        else:
            print(f"{len(rest)} other team names contain {text!r} (showing the alias/exact match only)")
        found = direct
    if len(found) > a.max_teams:
        found = sorted(found, key=lambda th: -int(th[0]['games_stored']))
        print(f"{len(found)} teams match {text!r}; details for the {a.max_teams} with most stored games (--max-teams N):")
        for t, how in found[:40]:
            print(f"   {t['team'][:40]:40} id {t['team_id'] or '?':>9} rank {t['current_rank'] or '-':>5} "
                  f"stored {t['games_stored']:>5}  [{how}]")
        found = found[:a.max_teams]
    for t, how in found:
        ids = {int(t['team_id'])} if t['team_id'] else set()
        names = {fold(x) for x in [t['team']] + t['other_names'].split('|') if x}
        eps = load_episodes(EP_COLS, where=lambda e: bool(team_seats(e, ids, names)))
        team_summary(t, how)
        if t['team'] == OUR_TEAM or OUR_TID in ids:
            print_ours(eps, rank_of, a)
            continue
        rows = filter_rows([(e, s) for e in eps for s in team_seats(e, ids, names)], a)
        copies = load_copies({e['episode_id'] for e, _ in (rows if a.all else rows[:a.limit])}, COPY_COLS)
        print_rows(rows, copies, rank_of, a)


def print_ours(eps, rank_of, a):
    ver_sub = {v: s for s, v in load_our_subs().items()}
    mine = [e for e in eps if e['involves_us'] and e['kaggle']]
    if a.version:
        mine = [e for e in mine if version_match(e['our_version'], a.version)]
    by_v = collections.defaultdict(list)
    for e in mine:
        by_v[e['our_version'] or 'unknown'].append(e)
    print(f"   {'version':40} {'submission':>10} {'stored':>6} {'W':>4} {'L':>4} {'T':>3} {'self':>4} {'win%':>5} "
          f"{'known':>5}  days")
    for v, es in sorted(by_v.items(), key=lambda kv: max((e['day'] or '') for e in kv[1]), reverse=True):
        st = [e for e in es if e['stored']]
        one = [e for e in st if not e['self_play'] and e['our_seat'] is not None]
        w = sum(seat_result(e, e['our_seat']) == 'W' for e in one)
        l_ = sum(seat_result(e, e['our_seat']) == 'L' for e in one)
        t_ = sum(seat_result(e, e['our_seat']) == 'T' for e in one)
        sp = sum(e['self_play'] for e in st)
        subs = collections.Counter(e['our_submission'] for e in es if e['our_submission'])
        days = sorted(e['day'] for e in es if e['day'])
        pct = f'{100 * w / (w + l_):.0f}%' if w + l_ else '-'
        sub = subs.most_common(1)[0][0] if subs else ver_sub.get(v, '?')
        print(f"   {v[:40]:40} {sub:>10} {len(st):>6} {w:>4} {l_:>4} {t_:>3} "
              f"{sp:>4} {pct:>5} {len(es) - len(st):>5}  {days[0] if days else '?'}..{days[-1] if days else '?'}")
    local = load_episodes(['episode_id', 'kaggle', 'involves_us', 'our_version', 'our_seat', 'self_play', 'winner_seat',
                           'rewards', 'sources'], where=lambda e: e['involves_us'] and not e['kaggle'])
    if a.version:
        local = [e for e in local if version_match(e['our_version'], a.version)]
    if local:
        c = collections.Counter()
        for e in local:
            c[(e['our_version'], 'self' if e['self_play'] else seat_result(e, e['our_seat']) if e['our_seat'] is not None
               else '?')] += 1
        vs = sorted({v for v, _ in c})
        print(f"   local games (not on Kaggle: selfplay tapes, recorded ladder, skirmishes): {len(local)}")
        for v in vs:
            print(f"     {v}: " + ' '.join(f"{k}={c[(v, k)]}" for k in ('W', 'L', 'T', 'self', '?') if c[(v, k)]))
    rows = [(e, e['our_seat']) for e in mine if e['our_seat'] is not None and not e['self_play']]
    if a.result is None and not a.list_games:
        a.result = 'loss'
        what = 'losses'
    else:
        what = 'games'
    rows = filter_rows(rows, a)
    copies = load_copies({e['episode_id'] for e, _ in (rows if a.all else rows[:a.limit])}, COPY_COLS)
    print_rows(rows, copies, rank_of, a, what=what)


# ------------------------------------------------------------------ query -> trainer TSV
def resolve_team_arg(text, teams, aliases):
    found = match_teams(text, teams, aliases)
    exactish = [(t, h) for t, h in found if h.startswith(('alias', 'exact'))]
    use = exactish or found
    if not use:
        raise SystemExit(f'no team matches {text!r}')
    if len(use) > 1 and not exactish:
        raise SystemExit(f'{text!r} matches {len(use)} teams: ' + ', '.join(t["team"] for t, _ in use[:12])
                         + ' (be more specific or add an ALIASES.tsv row)')
    ids = {int(t['team_id']) for t, _ in use if t['team_id']}
    names = {fold(n) for t, _ in use for n in [t['team']] + t['other_names'].split('|') if n}
    return ids, names, [t['team'] for t, _ in use]


def cmd_query(a):
    aliases, teams = load_aliases(), load_teams()
    team = resolve_team_arg(a.team, teams, aliases) if a.team else None
    beat = resolve_team_arg(a.beat, teams, aliases) if a.beat else None
    vs = resolve_team_arg(a.vs, teams, aliases) if a.vs else None
    if a.us:
        team = ({OUR_TID}, {fold(n) for n in OUR_NAMES}, sorted(OUR_NAMES))
    cols = EP_COLS + ['config_diff', 'official_avg_score', 'official_min_score']
    eps = load_episodes(cols, where=lambda e: e['stored'] and e['tsv_ready'])
    days = sorted({e['day'] for e in eps if e['day'] and DAY_RE.fullmatch(e['day'])})
    recent = set(days[-a.recent_days:]) if a.recent_days else None
    lb_rank = rank_lookup(teams)
    out = []
    for e in eps:
        if not a.include_local and not e['kaggle']:
            continue
        if not a.any_config and e['config_ok'] is not True:
            continue
        if e['self_play'] and not a.include_selfplay:
            continue
        if a.day_from and (e['day'] or '') < a.day_from:
            continue
        if a.day_to and (e['day'] or '9999') > a.day_to:
            continue
        if recent is not None and e['day'] not in recent:
            continue
        if a.source and not set(a.source) & set(e['sources']):
            continue
        if a.exclude_source and set(e['sources']) <= set(a.exclude_source):
            continue
        if a.version and not version_match(e['our_version'], a.version):
            continue
        if a.episode_type and e['episode_type'] and a.episode_type.upper() not in e['episode_type']:
            continue
        w = e['winner_seat']
        cand = []
        if beat:
            ls = team_seats(e, beat[0], beat[1])
            if ls and w is not None and w not in ls:
                cand = [(w, f"beat:{e['team_names'][ls[0]]}:{e['team_names'][w]}")]
        elif team:
            ts = team_seats(e, team[0], team[1])
            for s in ts:
                res = seat_result(e, s)
                if a.result == 'any' or (a.result or 'win') == {'W': 'win', 'L': 'loss', 'T': 'tie'}.get(res):
                    cand.append((s, f"{ {'W': 'win', 'L': 'loss', 'T': 'tie'}.get(res, 'res?')}:{e['team_names'][s]}"))
        else:
            if a.seat == 'both':
                cand = [(s, f"{'win' if w == s else 'loss' if w == 1 - s else 'tie'}:{e['team_names'][s]}") for s in (0, 1)]
            elif a.seat == 'loser':
                cand = [(1 - w, f"loss:{e['team_names'][1 - w]}")] if w is not None else []
            else:
                cand = [(w, f"win:{e['team_names'][w]}")] if w is not None else []
        keep = []
        for s, reason in cand:
            o = 1 - s
            if vs and not team_seats(e, vs[0], vs[1]) == [o]:
                continue
            if a.rank_max is not None:
                r = lb_rank(e, s)
                if r is None or int(r) > a.rank_max:
                    continue
            if a.opp_rank_max is not None:
                r = lb_rank(e, o)
                if r is None or int(r) > a.opp_rank_max:
                    continue
            if a.min_game_score is not None:
                gs = (e['game_scores'] or [None, None])[s]
                if gs is None or gs < a.min_game_score:
                    continue
            if a.min_official_score is not None:
                if e['official_min_score'] is None or e['official_min_score'] < a.min_official_score:
                    continue
            if a.no_static and (e['static_seats'] or [None, None])[s]:
                continue
            if a.exclude_ours and e['involves_us'] and e['our_seat'] == s:
                continue
            keep.append((s, reason))
        for s, reason in keep:
            out.append((e, s, (a.reason + ':' if a.reason else '') + reason))
    # the copy to emit: preferred tsv-ready copy, or the preferred one among --source
    need = {e['episode_id'] for e, _, _ in out}
    copies = load_copies(need, columns=['episode_id', 'source', 'archive', 'member', 'tsv_ready', 'rank', 'format',
                                        'scan_error', 'config_ok']) if need else {}
    rows = []
    for e, s, reason in sorted(out, key=lambda t: (t[0]['day'] or '', t[0]['episode_id'], t[1])):
        cs = [c for c in copies.get(e['episode_id'], []) if c['tsv_ready']]
        if a.source:
            cs = [c for c in cs if c['source'] in a.source] or cs
        if a.exclude_source:
            cs = [c for c in cs if c['source'] not in a.exclude_source] or cs
        if not cs:
            continue
        c = cs[0]
        rows.append((c['archive'], c['member'], e['day'] or 'unknown', e['episode_id'], s, a.weight, reason))
    if a.limit:
        rows = rows[-a.limit:]
    text = 'archive\tmember\tday\teid\tseat\tweight\treason\n' + ''.join('\t'.join(str(x) for x in r) + '\n' for r in rows)
    if a.out:
        tmp = Path(a.out + '.tmp')
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, a.out)
        by = collections.Counter(r[6].split(':')[0 if not a.reason else 1] for r in rows)
        print(json.dumps({'rows': len(rows), 'episodes': len({r[3] for r in rows}), 'out': a.out,
                          'days': [min(r[2] for r in rows), max(r[2] for r in rows)] if rows else None,
                          'reasons': dict(by)}, ensure_ascii=False), file=sys.stderr)
    else:
        sys.stdout.write(text)


def have(eids):
    """{episode_id: EPISODES row} for the given ids that have any local copy (fetchers: skip these ids)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(CAT / 'EPISODES.parquet', columns=['episode_id', 'stored', 'tsv_ready', 'formats', 'preferred_source',
                                                         'preferred_archive', 'preferred_member', 'day'])
    t = t.filter(pc.is_in(t.column('episode_id'), value_set=pa.array(sorted(set(eids)), type=pa.int64())))
    return {r['episode_id']: r for r in t.to_pylist() if r['stored']}


def cmd_have(a):
    ids = [int(x) for x in a.ids]
    if a.file:
        ids += [int(x) for x in re.findall(r'\d{8,}', Path(a.file).read_text())]
    got = have(ids)
    print('eid\tlocal\tzip_ready\tformats\tpreferred_source\tarchive\tmember')
    for e in ids:
        r = got.get(e)
        print(f"{e}\t{'yes' if r else 'no'}\t{'yes' if r and r['tsv_ready'] else 'no'}\t"
              f"{','.join(r['formats']) if r else ''}\t{r['preferred_source'] if r else ''}\t"
              f"{r['preferred_archive'] if r else ''}\t{r['preferred_member'] if r else ''}")
    print(f"# {len(got)}/{len(set(ids))} ids have a local copy", file=sys.stderr)


def cmd_missing(a):
    """Episodes known from an index/listing with no local replay (fetch candidates): eid day seats subs known_from."""
    aliases, teams = load_aliases(), load_teams()
    team = resolve_team_arg(a.team, teams, aliases) if a.team else None
    cols = ['episode_id', 'stored', 'formats', 'day', 'team_names', 'team_ids', 'submission_ids', 'rewards', 'known_from',
            'involves_us', 'episode_type']
    eps = load_episodes(cols, where=lambda e: (not e['stored'] or (
        a.include_tape_only and not set(e['formats']) & REPLAY_FORMATS)))
    rows = []
    for e in eps:
        if team and not team_seats(e, team[0], team[1]):
            continue
        if a.us and not e['involves_us']:
            continue
        if a.day_from and (e['day'] or '') < a.day_from:
            continue
        if a.day_to and (e['day'] or '9999') > a.day_to:
            continue
        if not a.include_validation and e['episode_type'] and 'VALIDATION' in e['episode_type']:
            continue
        rows.append(e)
    rows.sort(key=lambda e: (e['day'] or '', e['episode_id']), reverse=True)
    out = ['eid\tday\tteam0\tteam1\tsub0\tsub1\treward0\treward1\tlocal_formats\tknown_from']
    for e in rows[:a.limit] if a.limit else rows:
        n, sb, rw = e['team_names'], e['submission_ids'], e['rewards']
        out.append('\t'.join(str(x if x is not None else '') for x in (
            e['episode_id'], e['day'], n[0], n[1], sb[0], sb[1], rw[0], rw[1], ','.join(e['formats']),
            ','.join(e['known_from']))))
    text = '\n'.join(out) + '\n'
    if a.out:
        Path(a.out).write_text(text, encoding='utf-8')
        print(json.dumps({'rows': len(out) - 1, 'out': a.out}), file=sys.stderr)
    else:
        sys.stdout.write(text)


def cmd_teams(a):
    teams = load_teams()
    if a.min_games:
        teams = [t for t in teams if int(t['games_stored']) >= a.min_games]
    teams = teams[:a.top] if a.top else teams
    print(f"{'rank':>5} {'team':34} {'id':>9} {'score':>7} {'stored':>6} {'W':>5} {'L':>5} {'known':>6} {'beat us':>7} "
          f"{'lost us':>7}  days")
    for t in teams:
        print(f"{t['current_rank'] or '-':>5} {t['team'][:34]:34} {t['team_id'] or '?':>9} {t['current_score'] or '-':>7} "
              f"{t['games_stored']:>6} {t['wins']:>5} {t['losses']:>5} {t['known_not_stored']:>6} {t['beat_us']:>7} "
              f"{t['lost_to_us']:>7}  {t['first_day']}..{t['last_day']}")


def cmd_stats(a):
    s = json.loads((CAT / 'SUMMARY.json').read_text())
    t = s['totals']
    print(f"built {s['built_utc']} ({s['seconds']}s); leaderboard {Path(s['leaderboard'] or '-').name}")
    print(json.dumps(t, indent=1))
    print(f"{'source':26} {'files':>6} {'copies':>7} {'episodes':>8} {'zip':>7} {'GB':>7}  days")
    for k, v in s['sources'].items():
        print(f"{k:26} {v['containers']:>6} {v['copies']:>7} {v['unique_episodes']:>8} {v['tsv_ready_copies']:>7} "
              f"{v['disk_gb']:>7}  {v['first_day']}..{v['last_day']}")
    print('our games by version (stored, Kaggle):')
    for v, c in s['our_games_by_version'].items():
        print(f"   {v:44} {json.dumps(c)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    for name in ('build', 'update', 'refresh'):
        p = sp.add_parser(name)
        p.add_argument('--rediscover', action='store_true', help='re-walk the frozen old farm workspace')
        if name != 'build':
            p.add_argument('--only', nargs='*', help='force a rescan of these sources; every other source keeps its cache')
    p = sp.add_parser('sync-hosted')
    p.add_argument('--dry-run', action='store_true')
    p = sp.add_parser('find')
    p.add_argument('text', nargs='+')
    p.add_argument('--version', help='our deployed version folder / model tag substring (with "us")')
    p.add_argument('--result', choices=['win', 'loss', 'tie'])
    p.add_argument('--vs', help='opponent name substring')
    p.add_argument('--day-from')
    p.add_argument('--day-to')
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--all', action='store_true')
    p.add_argument('--copies', choices=['all', 'first'], default='all')
    p.add_argument('--stored-only', action='store_true')
    p.add_argument('--max-teams', type=int, default=4)
    p.add_argument('--list-games', action='store_true', help='with "us": also list every game, not only losses')
    p = sp.add_parser('query')
    p.add_argument('--team', help='team name / alias; emits that team\'s seat (default: its wins)')
    p.add_argument('--us', action='store_true', help='our team (Russell Kirk)')
    p.add_argument('--beat', help='team name / alias; emits the WINNER seat of games that team lost')
    p.add_argument('--vs', help='the other seat must be this team')
    p.add_argument('--result', choices=['win', 'loss', 'tie', 'any'])
    p.add_argument('--seat', choices=['winner', 'loser', 'both'], default='winner', help='without --team/--beat')
    p.add_argument('--day-from')
    p.add_argument('--day-to')
    p.add_argument('--recent-days', type=int)
    p.add_argument('--source', nargs='*', help='episode must have a copy in one of these sources (that copy is emitted)')
    p.add_argument('--exclude-source', nargs='*')
    p.add_argument('--rank-max', type=int, help='selected seat\'s team current leaderboard rank <= N')
    p.add_argument('--opp-rank-max', type=int, help='opponent\'s current leaderboard rank <= N')
    p.add_argument('--min-game-score', type=float, help='selected seat\'s rating after the game (xsf) >= X')
    p.add_argument('--min-official-score', type=float, help='official daily manifest min_score >= X')
    p.add_argument('--version', help='our deployed version substring')
    p.add_argument('--episode-type', help='PUBLIC or VALIDATION')
    p.add_argument('--no-static', action='store_true', help='drop seats labelled static tapes')
    p.add_argument('--exclude-ours', action='store_true', help='never emit our own seat')
    p.add_argument('--include-selfplay', action='store_true', help='keep games where both seats are ours')
    p.add_argument('--include-local', action='store_true', help='keep non-Kaggle (local) games with zip copies')
    p.add_argument('--any-config', action='store_true', help='keep games whose configuration differs from reference')
    p.add_argument('--weight', type=int, default=1)
    p.add_argument('--reason', default='')
    p.add_argument('--limit', type=int)
    p.add_argument('--out')
    p = sp.add_parser('teams')
    p.add_argument('--top', type=int, default=40)
    p.add_argument('--min-games', type=int, default=0)
    sp.add_parser('stats')
    p = sp.add_parser('have', help='which episode ids already have a local copy (fetchers: skip them)')
    p.add_argument('ids', nargs='*')
    p.add_argument('--file', help='text file with episode ids (any separators)')
    p = sp.add_parser('missing', help='episodes known from an index/listing but not stored (fetch candidates)')
    p.add_argument('--team')
    p.add_argument('--us', action='store_true')
    p.add_argument('--day-from')
    p.add_argument('--day-to')
    p.add_argument('--include-tape-only', action='store_true', help='also list games held only as tapes')
    p.add_argument('--include-validation', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--out')
    a = ap.parse_args()
    if a.cmd == 'build':
        cmd_build(a, full=True)
    elif a.cmd == 'update':
        cmd_build(a, full=False)
    elif a.cmd == 'refresh':
        sync_hosted()
        cmd_build(a, full=False)
    elif a.cmd == 'sync-hosted':
        sync_hosted(dry_run=a.dry_run)
    elif a.cmd == 'find':
        cmd_find(a)
    elif a.cmd == 'query':
        cmd_query(a)
    elif a.cmd == 'teams':
        cmd_teams(a)
    elif a.cmd == 'stats':
        cmd_stats(a)
    elif a.cmd == 'have':
        cmd_have(a)
    elif a.cmd == 'missing':
        cmd_missing(a)


if __name__ == '__main__':
    main()
