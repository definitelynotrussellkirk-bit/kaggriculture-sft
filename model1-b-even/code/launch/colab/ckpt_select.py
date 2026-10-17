"""Resume preflight. Select only LATEST, check internal counters, refuse with exit 2.

Requires the trainer's PyTorch environment. This admits a checkpoint; the trainer
must still load these same bytes. It does not prove optimizer/data compatibility.
"""
import json
from pathlib import Path
import sys
import torch


def identity(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def select(run):
    run = Path(run).resolve(strict=True)
    lp = run / 'LATEST'
    latest_bytes = lp.read_bytes()
    name = latest_bytes.decode().strip()
    if not name or Path(name).name != name or '\\' in name or not name.endswith('.pt'):
        raise ValueError(f'LATEST missing/empty/unsafe: {name!r}')
    files = sorted(set(run.glob('ckpt-*.pt')) | set(run.glob('last.pt')) | {run / name})
    before = {}
    for f in files:
        if not f.is_file() or f.resolve(strict=True).parent != run:
            raise ValueError(f'checkpoint is missing or outside RUN_DIR: {f}')
        before[f] = identity(f)
    ups, skipped = {}, {}
    for f in files:
        required = f.name == name   # LATEST's target must load; other candidates are optional (EASY-FILE-CKPT-SELECT)
        try:
            # Old-format torch.save checkpoints cannot be memory mapped.
            try:
                b = torch.load(f, map_location='cpu', mmap=True, weights_only=False)
            except RuntimeError as e:
                if 'mmap' not in str(e):
                    raise
                b = torch.load(f, map_location='cpu', mmap=False, weights_only=False)
            u = b.get('update') if isinstance(b, dict) else None
            if type(u) is not int or u < 0:
                raise ValueError(f'{f} has no nonnegative integer update counter')
        except Exception as e:
            if required:
                raise
            if identity(f) != before[f]:
                raise ValueError(f'checkpoint changed during preflight: {f}')
            skipped[f.name] = f'{type(e).__name__}: {e}'[:200]   # stable, unloadable non-LATEST candidate (e.g. a crash mid-save)
            continue
        ups[f.name] = u
        if identity(f) != before[f]:
            raise ValueError(f'checkpoint changed during preflight: {f}')
    if (lp.read_bytes() != latest_bytes or any(identity(f) != before[f] for f in files)
            or set(run.glob('ckpt-*.pt')) | set(run.glob('last.pt')) | {run / name} != set(files)):
        raise ValueError('LATEST/checkpoint set changed during preflight; retry')
    chosen = ups[name]
    newest = max(ups, key=ups.get)
    if chosen < ups[newest]:
        raise ValueError(f'LATEST {name} is update {chosen} but {newest} holds update {ups[newest]}')
    return {'run': str(run), 'latest': name, 'update': chosen, 'scanned': ups, 'skipped_unloadable': skipped}


def main():
    try:
        if len(sys.argv) != 2:
            raise ValueError('usage: ckpt_select.py RUN_DIR')
        print(json.dumps(select(sys.argv[1])))
        return 0
    except Exception as e:
        print(f'resume refused: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
