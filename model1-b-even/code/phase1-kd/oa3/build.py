"""Explicit EMA selection and strict, lossy depth reduction of PolicyNet."""
from __future__ import annotations
import dataclasses
import math
import re
from pathlib import Path
import torch
from .source import model_module, sha256

def choose_layers(old: int, new: int) -> list[int]:
    if not isinstance(old, int) or not isinstance(new, int) or not 1 <= new <= old:
        raise ValueError('Only positive depth reduction is supported')
    if new == 1:
        return [old - 1]
    return [round(i * (old - 1) / (new - 1)) for i in range(new)]

def selected_state(blob: dict, component: str) -> dict[str, torch.Tensor]:
    if component not in ('model', 'ema'):
        raise ValueError('Select component explicitly: model or ema (no auto fallback)')
    if not isinstance(blob.get('model'), dict):
        raise ValueError('Checkpoint must contain a model state dict')
    base = blob['model']
    if any(not torch.is_tensor(v) for v in base.values()):
        raise ValueError('Model state must contain only tensors')
    state = dict(base)
    if component == 'ema':
        ema = blob.get('ema')
        if not isinstance(ema, dict) or not ema:
            raise ValueError('Explicit EMA requested, but no EMA state is present; no raw fallback')
        missing = [k for k, v in base.items() if (v.is_floating_point() or v.is_complex()) and k not in ema]
        if missing:
            raise ValueError(f'Incomplete EMA floating state: {missing[:5]}')
        for k, v in ema.items():
            if k not in base or not torch.is_tensor(v) or v.shape != base[k].shape:
                raise ValueError(f'Unknown or incompatible EMA tensor: {k}')
        state.update(ema)
    if any((v.is_floating_point() or v.is_complex()) and not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Checkpoint contains nonfinite weights')
    return state

def load_checkpoint(path: str | Path, component: str, trainer=None):
    # weights_only deliberately refuses arbitrary pickle execution. Trusted legacy
    # checkpoints should be converted by the owner, not loaded with a silent fallback.
    blob = torch.load(path, map_location='cpu', weights_only=True)
    cfg = blob.get('config') or blob.get('model_config')
    if not isinstance(cfg, dict):
        raise ValueError('Missing model configuration')
    m = model_module(trainer)
    unknown = set(cfg) - {f.name for f in dataclasses.fields(m.ModelConfig)}
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    net = m.PolicyNet(m.ModelConfig.from_dict(cfg))
    net.load_state_dict(selected_state(blob, component), strict=True)
    net.eval()
    return net, {'path': str(Path(path).resolve()), 'sha256': sha256(path), 'component': component,
                 'config': net.config.to_dict(), 'parameters': net.parameter_count()}

def candidate_config(parent, enc_layers: int, dec_layers: int, feature_mode: str = 'v8'):
    choose_layers(parent.layers, enc_layers); choose_layers(parent.dec_layers, dec_layers)
    if parent.think != 'off' or parent.recur_core or parent.recur_loops != 1:
        raise ValueError('This builder only supports plain, think=off donors, not recurrent/bottleneck donors')
    if feature_mode not in ('inherit', 'v8'):
        raise ValueError('feature_mode must be inherit or v8')
    c = dataclasses.replace(parent, layers=enc_layers, dec_layers=dec_layers)
    if feature_mode == 'v8':
        c = dataclasses.replace(c, extra_width=114, unit_extra_width=12, calc_width=36,
            reach_bias=True, reach_job=True, carrier_read=True,
            prefix_unit_width=27, prefix_order_width=22, prefix_cash_width=82,
            job_eta_v6=True, input_schema='pbc-job-input-v6.0', feature_schema='legacy',
            v6_prev=True, v6_aux=False, v6_aux_turn=False,
            x8_token_width=354, x8_unit_width=41, lattice_channels=0,
            dropout=0.0, feature_dropout=0.0)
    return c.resolve()

def build_student(parent, *, enc_layers=6, dec_layers=2, feature_mode='v8'):
    enc = choose_layers(parent.config.layers, enc_layers)
    dec = choose_layers(parent.config.dec_layers, dec_layers)
    cfg = candidate_config(parent.config, enc_layers, dec_layers, feature_mode)
    student = type(parent)(cfg)
    source, dest = parent.state_dict(), student.state_dict()
    copied, initialized, removed = [], [], []
    used = set()
    for k, v in dest.items():
        src = k
        match = re.match(r'(blocks|dec)\.(\d+)\.(.+)', k)
        if match:
            group, i, tail = match.groups()
            src = f'{group}.{(enc if group == "blocks" else dec)[int(i)]}.{tail}'
        if src not in source:
            # New growth/history modules use the model's inert output initialization.
            if not (k.startswith('grow_') or k.startswith('v6_')):
                raise ValueError(f'Unexpected new tensor: {k}')
            initialized.append(k)
            continue
        x = source[src]
        if k in ('grow_reach_eta', 'grow_reach_kind', 'grow_reach_slack', 'grow_reach_job'):
            x = x[dec]
        elif k.startswith('grow_v6_job_out.'):
            h = parent.config.dec_heads or parent.config.heads
            x = x.reshape(parent.config.dec_layers, h, *x.shape[1:])[dec].reshape(-1, *x.shape[1:])
        if x.shape != v.shape:
            raise ValueError(f'Unsafe shape change {src} {tuple(x.shape)} -> {k} {tuple(v.shape)}')
        dest[k] = x.detach().clone()
        copied.append({'from': src, 'to': k}); used.add(src)
    removed = sorted(set(source) - used)
    student.load_state_dict(dest, strict=True)
    student.eval()
    preserving = cfg.to_dict() == parent.config.to_dict() and not initialized and not removed
    return student, {'encoder_indices': enc, 'decoder_indices': dec,
        'parent_parameters': parent.parameter_count(), 'student_parameters': student.parameter_count(),
        'config': cfg.to_dict(), 'copied': copied, 'initialized': initialized, 'not_copied': removed,
        'function_preserving': preserving,
        'warning': 'Depth thinning is lossy. Fresh optimizer; no playing-strength claim.'}
