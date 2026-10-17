"""Minimal v8 actor wiring for the supplied stale serving source.

Writes only to a NEW isolated directory. Does not change cache-hashed data.py or
any training feature provider, and does not alter the model's parameter schema.
"""
from __future__ import annotations
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
from .source import ROOT, sha256

def patched_source(source: str) -> str:
    expected=sha256(ROOT/'reference/bc_trainer/model.py')
    if hashlib.sha256(source.encode()).hexdigest()!=expected:
        raise ValueError('Source differs from reviewed model.py (or is already patched); do not blindly apply this patch')
    edits=[
        ('def observation_tensors(self, clean, configuration=None):',
         'def observation_tensors(self, clean, configuration=None, *, v8_history=None, v8_long=None):'),
        ('        try:\n            import data as _data\n            feats = _data.features(clean, configuration or {}, feature_schema=self.config.feature_schema)',
         '        needs_v8 = bool(self.config.x8_token_width or self.config.x8_unit_width or self.config.lattice_channels)\n'
         '        if needs_v8 and (v8_history is None or v8_long is None):\n'
         '            raise ValueError("v8 serving requires explicit seat-local observation history and GapsState")\n'
         '        try:\n            import data as _data\n'
         '            feats = _data.features(clean, configuration or {}, feature_schema=self.config.feature_schema,\n'
         '                                   v8_history=v8_history if needs_v8 else None,\n'
         '                                   v8_long=v8_long if needs_v8 else None)'),
        ('or c.reach_bias or c.reach_job or c.job_eta_v6 or c.feature_schema != "legacy"):',
         'or c.reach_bias or c.reach_job or c.job_eta_v6 or c.feature_schema != "legacy" or needs_v8):'),
        ('        self._last_job_eta = None\n        if self.config.extra_width',
         '        self._last_job_eta = None\n'
         '        self._last_v8 = {}\n'
         '        for name, width in (("x8_token", self.config.x8_token_width),\n'
         '                            ("x8_unit", self.config.x8_unit_width),\n'
         '                            ("lattice", self.config.lattice_channels)):\n'
         '            if width:\n'
         '                if name not in feats:\n'
         '                    raise ValueError("Missing enabled v8 serving feature: " + name)\n'
         '                self._last_v8[name] = torch.from_numpy(feats[name].astype("float32"))[None]\n'
         '        if self.config.extra_width'),
        ('            code_mode=None):\n        """One decision',
         '            code_mode=None, v8_history=None, v8_long=None):\n        """One decision'),
        ('        rows, token_mask, known, pos = self.observation_tensors(clean, configuration)\n',
         '        rows, token_mask, known, pos = self.observation_tensors(clean, configuration,\n'
         '                                                                   v8_history=v8_history, v8_long=v8_long)\n'),
        ('prev_unit=prev_unit, prev_order=prev_order, **(self._last_job_eta or {}))',
         'prev_unit=prev_unit, prev_order=prev_order, **(self._last_job_eta or {}), **self._last_v8)'),
    ]
    for before,after in edits:
        if source.count(before)!=1:raise ValueError(f'Expected exactly one reviewed patch anchor: {before[:80]}')
        source=source.replace(before,after)
    ast.parse(source)
    return source

def write_isolated(source_dir: str|Path, output: str|Path) -> dict:
    src=Path(source_dir).resolve();dest=Path(output).resolve()
    if len(dest.parents)<4:
        raise ValueError('Upstream source expects repository-root/work/runs/serving depth; destination is too shallow')
    original=(src/'model.py').read_text();patched=patched_source(original)
    if dest.exists():raise FileExistsError('Serving destination must be a new directory')
    if src==dest or src in dest.parents:raise ValueError('Write outside the source tree')
    dest.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix in ('.py','.md','.json'):shutil.copy2(f,dest/f.name)
    (dest/'model.py').write_text(patched)
    patch=''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='a/model.py',tofile='b/model.py'))
    (dest/'v8-serving.patch').write_text(patch)
    receipt={'source_model_sha256':sha256(src/'model.py'),'patched_model_sha256':sha256(dest/'model.py'),
             'changed_methods':['PolicyNet.observation_tensors','PolicyNet.act'],
             'weights_changed':False,'source_tree_modified':False,
             'deployment_gate':'Use the current v9 agent that carries v8_history/GapsState. Full cold-archive game verification still required.'}
    (dest/'PATCH-RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
