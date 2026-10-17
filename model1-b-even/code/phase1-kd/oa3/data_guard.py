"""Frozen weighted seats, episode-level holdouts, and enabled-feature contracts."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import shutil
from .source import sha256

HEADER = ['archive', 'member', 'team', 'episode', 'seat', 'weight']

def read_roster(path: str | Path) -> list[tuple]:
    rows = []
    with Path(path).open(newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, [])
        if len(header) < 5 or header[0] != 'archive':
            raise ValueError(f'Expected a headered archive/member/team/episode/seat roster: {path}')
        seen = set()
        for lineno, r in enumerate(reader, 2):
            if not r:
                continue
            if len(r) not in (5, 6):
                raise ValueError(f'{path}:{lineno}: expected 5 or 6 columns')
            eid, seat, weight = int(r[3]), int(r[4]), int(r[5]) if len(r) == 6 else 1
            if eid < 0 or seat not in (0, 1) or weight < 1:
                raise ValueError(f'{path}:{lineno}: invalid episode, seat or integer weight')
            if (eid, seat) in seen:
                raise ValueError(f'{path}:{lineno}: duplicate seat; express multiplicity as a weight')
            seen.add((eid, seat))
            rows.append((r[0], r[1], r[2], eid, seat, weight))
    if not rows:
        raise ValueError(f'Empty roster: {path}')
    return rows

def _write_roster(path, rows):
    with Path(path).open('x', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(HEADER); w.writerows(rows)

def freeze_rosters(train: str | Path, validation: str | Path, output: str | Path,
                   exclude_episodes=(), cache_dir: str | Path | None = None) -> dict:
    tr, va = read_roster(train), read_roster(validation)
    a, b = {r[3] for r in tr}, {r[3] for r in va}
    excluded = set(map(int, exclude_episodes))
    if a & b:
        raise ValueError(f'Episode leakage, including opposite seats: {sorted(a & b)[:20]}')
    if (a | b) & excluded:
        raise ValueError(f'Quarantined/held-out episodes present: {sorted((a | b) & excluded)[:20]}')
    sizes = {}
    if cache_dir is not None:
        cache = Path(cache_dir)
        missing = [f'{r[3]}-{r[4]}.stp' for r in tr + va if not (cache / f'{r[3]}-{r[4]}.stp').is_file()]
        if missing:
            raise ValueError(f'{len(missing)} missing cache seats; refusing silent reweighting: {missing[:10]}')
        sizes = {f'{r[3]}-{r[4]}.stp': (cache / f'{r[3]}-{r[4]}.stp').stat().st_size for r in tr+va}
    dest = Path(output)
    if dest.exists():
        raise FileExistsError(f'Frozen output must not exist: {dest}')
    dest.mkdir(parents=True)
    _write_roster(dest / 'train.tsv', tr); _write_roster(dest / 'validation.tsv', va)
    receipt = {'train_unique_episodes': len(a), 'validation_unique_episodes': len(b),
        'train_seats': len(tr), 'validation_seats': len(va),
        'train_weighted_seats': sum(r[5] for r in tr), 'validation_weighted_seats': sum(r[5] for r in va),
        'excluded_episodes': sorted(excluded),
        'source_hashes': {'train': sha256(train), 'validation': sha256(validation)},
        'frozen_hashes': {name: sha256(dest / name) for name in ('train.tsv', 'validation.tsv')},
        'cache_seat_sizes': sizes,
        'limitations': ['Exclusions are owner-supplied, not an inferred complete quarantine registry.',
                       'Size inventory is not a content hash of large caches. Keep the cache immutable.']}
    (dest / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt

def verify_frozen(path: str | Path) -> dict:
    p = Path(path); rec = json.loads((p / 'receipt.json').read_text())
    for name, digest in rec['frozen_hashes'].items():
        if sha256(p / name) != digest:
            raise ValueError(f'Frozen roster changed: {name}')
    tr, va = read_roster(p/'train.tsv'), read_roster(p/'validation.tsv')
    if {r[3] for r in tr} & {r[3] for r in va}:
        raise ValueError('Episode overlap in frozen rows')
    return rec

def validate_feature_subset(meta: dict, cfg) -> None:
    """Supplement, never replace, upstream validate_metadata(source hashes)."""
    if meta.get('version') != 2 or meta.get('feature_schema', 'legacy') != cfg.feature_schema:
        raise ValueError('Cache version/feature schema mismatch')
    if (cfg.v6_prev or cfg.job_eta_v6) and not meta.get('v6'):
        raise ValueError('Enabled v6 features absent from cache')
    widths = [cfg.x8_token_width, cfg.x8_unit_width, cfg.lattice_channels]
    if any(widths):
        actual = meta.get('v8_widths')
        if not meta.get('v8') or not isinstance(actual, list) or len(actual) != 3:
            raise ValueError('Enabled v8 features require declared cache widths')
        if any(w != 0 and w != actual[i] for i, w in enumerate(widths)):
            raise ValueError(f'Enabled v8 widths {widths} mismatch cache {actual}')
