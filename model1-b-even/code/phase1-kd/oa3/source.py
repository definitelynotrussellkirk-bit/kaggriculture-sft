"""Use an explicit trainer tree; refuse silently mixed source modules."""
from __future__ import annotations
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def trainer_dir(path: str | Path | None = None) -> Path:
    p = Path(path or os.environ.get('OA3_TRAINER', ROOT / 'reference/bc_trainer')).resolve()
    if not (p / 'model.py').is_file() or not (p / 'codec.py').is_file():
        raise ValueError(f'Not a trainer source directory: {p}')
    return p

def activate(path: str | Path | None = None) -> Path:
    p = trainer_dir(path)
    for name in ('codec', 'v8_lattice_net', 'step_cache', 'data', 'v8', 'model'):
        old = sys.modules.get(name)
        target = p / f'{name}.py'
        if old is not None and target.exists():
            oldfile = getattr(old, '__file__', None)
            if not oldfile or sha256(oldfile) != sha256(target):
                raise RuntimeError(f'{name} already loaded from different source; use a fresh process')
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p

def model_module(path: str | Path | None = None) -> ModuleType:
    p = activate(path)
    name = '_oa3_model_' + hashlib.sha256(str(p).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, p / 'model.py')
        if spec is None or spec.loader is None:
            raise ImportError(p / 'model.py')
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]

def source_receipt(path: str | Path | None = None) -> dict:
    p = trainer_dir(path)
    return {'trainer': str(p), 'files': {f.name: sha256(f) for f in sorted(p.glob('*.py'))}}
