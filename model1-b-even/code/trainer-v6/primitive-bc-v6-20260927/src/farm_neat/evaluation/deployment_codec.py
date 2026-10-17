"""Deployment-only grouped-int8 storage for the complete full-economic state.

Owner: evaluation maintainer. Ported from farm full_economic_weight_codec v2; no new-artifact
behavior parity, hosted acceptance or training admission follows from unit tests.
Consumer: the explicitly typed compact export/CPU-entry adapter, NOT the original
strict model loader. Only routed-expert gate_up/down matrices are quantized;
routers, shared experts, GDN, norms and every other tensor remain exact FP32.

Payload is dense little-endian bytes plus a separately externally pinned JSON
manifest. Decoding produces all original state_dict keys as fresh FP32 NumPy
arrays, once at load; inference itself is not int8. No pickle, file writer,
optimizer, graph construction, sampling or GPU operation exists in this module.
The caller must authenticate the original checkpoint FILE SHA and original
manifest with its maintained lineage loader; matching tensor hashes alone does
not prove those tensors came from a particular file or constitute training.
V2 separates exact encoder provenance from source/format-bound decoding. Host
package versions are diagnostics, not proof of decoded or inference equivalence:
every reconstructed tensor and the complete state still need exact bound hashes.
V1 artifacts require their unchanged archived v1 decoder, never this new schema.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'farmneat-full-economic-group64-int8-deployment-v2'
GROUP = 64
ARCHIVE_CAP_BYTES = 100_000_000
ENVELOPE_RESERVE_BYTES = 8_000_000
MAX_MANIFEST_BYTES = 3_000_000
MAX_DECODED_BYTES = 300_000_000
_ROUTED = re.compile(r'blocks\.(0|[1-9][0-9]*)\.moe\.experts\.(0|[1-9][0-9]*)\.(gate_up|down)\.weight\Z')
_NAME = re.compile(r'[A-Za-z_][A-Za-z_0-9]*(?:\.(?:[A-Za-z_][A-Za-z_0-9]*|0|[1-9][0-9]*))*\Z')
_ORIGIN_SCOPE = 'CALLER_AUTHENTICATES_ORIGINAL_FILE_AND_LINEAGE; CODEC_VERIFIES_ALL_MANIFEST_TENSORS'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _bound(value):
    body = dict(value); body.pop('content_sha256', None)
    return {**body, 'content_sha256': digest(body)}


def _equal(a, b, label):
    if canonical(a) != canonical(b):
        raise ValueError('codec mismatch: '+label)


def _sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('explicit lowercase SHA256 required')
    return value


def _bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def _source_pins():
    paths = ('src/farm_neat/evaluation/deployment_codec.py', 'tests/unit/test_deployment_codec.py')
    pins = {}
    for name in paths:
        path = ROOT/name
        if not path.is_file() or path.is_symlink():
            raise ValueError('regular codec source/test required')
        pins[name] = _bytes_sha(path.read_bytes())
    return pins


def decoder_identity():
    """Exact source/format contract, independent of the encoder's package build."""
    return _bound({'schema': SCHEMA+'-decoder', 'source_pins': _source_pins(), 'group_size': GROUP,
        'selection': _ROUTED.pattern, 'order': 'LEXICOGRAPHIC_TENSOR_NAMES_C_ORDER_ROWS',
        'quantization': 'SYMMETRIC_INT8_MINUS127_TO127; NO_ZERO_POINT',
        'scale': 'FP32(MAX_ABS_FP64/127); ZERO_GROUP_1; NONZERO_UNDERFLOW_MIN_POSITIVE_FP32',
        'rounding': 'NUMPY_RINT_TIES_TO_EVEN_OF_FP64_INPUT_OVER_STORED_FP32_SCALE; CLIP',
        'decode': 'FP32_INT8_TIMES_FP32_SCALE; NO_FP16_OR_INT8_INFERENCE',
        'other_tensors': 'EXACT_LITTLE_ENDIAN_FP32_BITS', 'tail': 'PARTIAL_LAST_ROW_GROUP_NO_STORED_PADDING',
        'archive_cap_bytes': ARCHIVE_CAP_BYTES, 'envelope_reserve_bytes': ENVELOPE_RESERVE_BYTES,
        'training_admission': False, 'optimizer_payload': False, 'lossless_checkpoint_claim': False,
        'compatibility': 'EXACT_SOURCE_AND_FORMAT_AND_ALL_DECODED_TENSOR_HASHES; PACKAGE_VERSIONS_DIAGNOSTIC_ONLY'})


def _encoder_identity(decoder, provenance):
    return _bound({**{k: v for k, v in decoder.items() if k not in ('schema', 'content_sha256')},
        'schema': SCHEMA+'-execution', 'encoder_provenance': provenance,
        'decoder_compatibility_sha256': decoder['content_sha256']})


def identity():
    """Frozen encoder source/runtime provenance; standard library only."""
    return _encoder_identity(decoder_identity(), {
        'numpy_version': importlib.metadata.version('numpy'),
        'python_version': '.'.join(map(str, sys.version_info[:3])),
        'implementation': sys.implementation.name})


def validate_decoder_identity(expected):
    """Validate v2's entire contract, retaining—not replacing—encoder provenance.

    No runtime-version wildcard is applied to v1. The caller must additionally
    authenticate the deployment manifest and verify every decoded tensor hash.
    This metadata check alone is not numerical or behavioral certification.
    """
    if type(expected) is not dict:
        raise ValueError('explicit v2 codec execution identity required')
    provenance = expected.get('encoder_provenance')
    if (type(provenance) is not dict or set(provenance) != {'numpy_version', 'python_version', 'implementation'}
            or any(type(v) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.+_!\-]{0,79}', v)
                   for v in provenance.values())):
        raise ValueError('bounded exact v2 encoder provenance required')
    current = decoder_identity()
    _equal(expected, _encoder_identity(current, provenance), 'v2 decoder sources/format compatibility')
    return current


def runtime_diagnostics():
    """Bounded version-only observability; no package imports, paths or environment."""
    def version(value):
        return str(value) if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.+_!\-]{0,79}', value) else 'UNAVAILABLE'
    def installed(name):
        try:
            return version(importlib.metadata.version(name))
        except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
            return 'UNAVAILABLE'
    return {'schema': SCHEMA+'-runtime-diagnostics',
        'python_version': '.'.join(map(str, sys.version_info[:3])),
        'implementation': version(sys.implementation.name), 'byteorder': sys.byteorder,
        'numpy_distribution_version': installed('numpy'),
        'numpy_loaded_version': version(getattr(sys.modules.get('numpy'), '__version__', None)),
        'torch_distribution_version': installed('torch'),
        'torch_loaded_version': version(getattr(sys.modules.get('torch'), '__version__', None)),
        'scope': 'DIAGNOSTIC_ONLY; NOT_DECODER_ADMISSION_OR_INFERENCE_PARITY'}


def _identity(expected):
    _equal(expected, identity(), 'externally frozen codec execution identity')
    import numpy as np
    _equal(np.__version__, expected['encoder_provenance']['numpy_version'], 'loaded encoder NumPy version')
    return np


def _original(manifest):
    if type(manifest) is not dict or len(canonical(manifest).encode()) > MAX_MANIFEST_BYTES:
        raise ValueError('bounded original manifest required')
    _equal(manifest, _bound(manifest), 'original manifest content binding')
    tensors = manifest.get('tensors')
    if type(tensors) is not dict or not 0 < len(tensors) <= 4096:
        raise ValueError('complete bounded original tensor map required')
    config = manifest.get('architecture', {}).get('config', {})
    if any(type(config.get(k)) is not int or config[k] < 1
           for k in ('layers', 'experts', 'width', 'expert_hidden')):
        raise ValueError('original full-economic dimensions required')
    total, size, routed = 0, 0, 0
    for name in sorted(tensors):
        if type(name) is not str or len(name) > 200 or not _NAME.fullmatch(name):
            raise ValueError('canonical tensor name required')
        value = tensors[name]
        if type(value) is not dict or set(value) != {'shape', 'sha256'}:
            raise ValueError('exact original tensor shape/hash fields required')
        shape = value['shape']; _sha(value['sha256'])
        if (type(shape) is not list or not 0 < len(shape) <= 8
                or any(type(n) is not int or n < 1 for n in shape)):
            raise ValueError('positive integer original tensor shape required')
        count = math.prod(shape); total += count
        match = _ROUTED.fullmatch(name)
        if '.experts.' in name and match is None:
            raise ValueError('unsupported routed-expert tensor name')
        if match:
            block, expert, kind = match.groups()
            wanted = [2*config['expert_hidden'], config['width']] if kind == 'gate_up' else [config['width'], config['expert_hidden']]
            if int(block) >= config['layers'] or int(expert) >= config['experts'] or shape != wanted:
                raise ValueError('routed-expert index/shape mismatch')
            size += count + 4*shape[0]*math.ceil(shape[1]/GROUP); routed += count
        else:
            size += 4*count
    report = manifest.get('parameter_report', {})
    _equal(report.get('total_trainable_parameters'), total, 'complete parameter count')
    _equal(report.get('parameter_tensors'), len(tensors), 'complete tensor count')
    if total*4 > MAX_DECODED_BYTES or size+ENVELOPE_RESERVE_BYTES > ARCHIVE_CAP_BYTES:
        raise ValueError('shape-derived deployment/decoded budget exceeded')
    return tensors, {'total_parameters': total, 'tensor_count': len(tensors),
        'quantized_parameters': routed, 'exact_fp32_parameters': total-routed,
        'payload_bytes': size, 'decoded_state_bytes': total*4,
        'archive_upper_bound_with_reserved_envelope_bytes': size+ENVELOPE_RESERVE_BYTES,
        'archive_cap_bytes': ARCHIVE_CAP_BYTES,
        'reserve_scope': 'CALLER_MUST_KEEP_METADATA_SOURCE_TAR_GZIP_OVERHEAD_WITHIN_RESERVED_8MB_AND_CHECK_ACTUAL_ARCHIVE'}


def estimate(original_manifest):
    """Shape-only bound. Does not load weights or promise a built archive size."""
    return _original(original_manifest)[1]


def _array(value, np):
    if not isinstance(value, np.ndarray):
        # Torch is optional and imported only for a supplied Torch state_dict.
        import torch
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.dtype != torch.float32 or value.layout != torch.strided):
            raise ValueError('CPU FP32 tensor required; no device/dtype conversion')
        value = value.detach().numpy()
    if value.dtype != np.dtype('<f4') or not value.flags.c_contiguous or not np.isfinite(value).all():
        raise ValueError('finite C-contiguous little-endian FP32 tensor required')
    return value


def _quantize(value, np):
    rows, columns = value.shape; groups = math.ceil(columns/GROUP)
    padded = np.zeros((rows, groups*GROUP), dtype=np.float64)
    padded[:, :columns] = value
    grouped = padded.reshape(rows, groups, GROUP)
    maxima = np.max(np.abs(grouped), axis=-1)
    scales = (maxima/127.).astype('<f4')
    scales[maxima == 0] = np.float32(1)
    scales[(maxima != 0) & (scales == 0)] = np.nextafter(np.float32(0), np.float32(1))
    quantized = np.clip(np.rint(grouped/scales.astype(np.float64)[..., None]), -127, 127).astype(np.int8)
    quantized = quantized.reshape(rows, -1)[:, :columns].copy(order='C')
    return quantized, scales, _dequantize(quantized, scales, np)


def _dequantize(quantized, scales, np):
    rows, columns = quantized.shape; groups = math.ceil(columns/GROUP)
    if (scales.dtype != np.dtype('<f4') or scales.shape != (rows, groups)
            or not np.isfinite(scales).all() or not (scales > 0).all()
            or (quantized == -128).any()):
        raise ValueError('finite positive FP32 scales and canonical int8 range required')
    padded = np.zeros((rows, groups*GROUP), dtype=np.int8)
    padded[:, :columns] = quantized
    grouped = padded.reshape(rows, groups, GROUP)
    if ((np.max(np.abs(grouped.astype(np.int16)), axis=-1) == 0) & (scales != 1)).any():
        raise ValueError('zero groups require canonical scale1')
    with np.errstate(over='ignore', invalid='ignore'):
        result = (grouped.astype('<f4')*scales[..., None]).reshape(rows, -1)[:, :columns].copy(order='C')
    if not np.isfinite(result).all():
        raise ValueError('nonfinite reconstructed FP32 tensor')
    return result


def encode_state(state_dict, original_manifest, *, original_checkpoint_sha256, expected_identity):
    """Return (immutable payload bytes, bound deployment manifest); no file writes.

    Incoming mapping insertion order is irrelevant. Serialized records are in a
    single canonical lexical order. Every input is checked before quantization.
    The original file SHA is provenance supplied by the authenticated caller,
    never a claim that lossy bytes reconstruct the original file or tensor values.
    """
    np = _identity(expected_identity); _sha(original_checkpoint_sha256)
    expected, sizing = _original(original_manifest)
    if not isinstance(state_dict, Mapping) or set(state_dict) != set(expected):
        raise ValueError('complete exact original state_dict keys required')
    arrays = {}
    for name in sorted(expected):
        value = _array(state_dict[name], np)
        if list(value.shape) != expected[name]['shape'] or _bytes_sha(value.tobytes()) != expected[name]['sha256']:
            raise ValueError('original tensor shape/hash mismatch: '+name)
        arrays[name] = value
    payload, records = bytearray(), []
    for name, value in arrays.items():
        routed = _ROUTED.fullmatch(name) is not None
        if routed:
            data, scales, decoded = _quantize(value, np)
            scale_bytes = scales.tobytes(order='C')
        else:
            data, decoded, scale_bytes = value, value, b''
        raw = data.tobytes(order='C'); offset = len(payload); payload.extend(raw)
        scale_offset = len(payload) if routed else None; payload.extend(scale_bytes)
        records.append({'name': name, 'shape': list(value.shape),
            'storage': 'INT8_GROUP64' if routed else 'FP32_EXACT',
            'data_offset': offset, 'data_bytes': len(raw), 'data_sha256': _bytes_sha(raw),
            'scales_offset': scale_offset, 'scales_bytes': len(scale_bytes),
            'scales_sha256': _bytes_sha(scale_bytes) if routed else None,
            'original_tensor_sha256': expected[name]['sha256'],
            'decoded_tensor_sha256': _bytes_sha(decoded.tobytes(order='C')),
            'maximum_absolute_error': float(np.max(np.abs(decoded.astype(np.float64)-value.astype(np.float64))))})
    payload = bytes(payload)
    _equal(len(payload), sizing['payload_bytes'], 'actual shape-derived payload')
    manifest = _bound({'schema': SCHEMA, 'codec_identity': expected_identity,
        'original_checkpoint_sha256': original_checkpoint_sha256,
        'original_manifest': json.loads(canonical(original_manifest)), 'origin_scope': _ORIGIN_SCOPE,
        'payload_sha256': _bytes_sha(payload), 'sizing': sizing, 'tensors': records,
        'decoded_state_sha256': digest([{k: r[k] for k in ('name', 'shape', 'decoded_tensor_sha256')} for r in records]),
        'lossy_deployment_only': True, 'training_admission': False, 'optimizer_payload': False})
    if len(canonical(manifest).encode()) > MAX_MANIFEST_BYTES:
        raise ValueError('deployment manifest budget exceeded')
    return payload, manifest


def decode_state(payload, manifest, *, expected_manifest_sha256, expected_identity):
    """Validate before returning a complete fresh OrderedDict of FP32 arrays.

    Pin expected_manifest_sha256 outside the artifact. A self-rehashed manifest
    is not an authority. No decoded file or original-loader acceptance is created.
    """
    validate_decoder_identity(expected_identity); _sha(expected_manifest_sha256)
    import numpy as np
    fields = {'schema', 'codec_identity', 'original_checkpoint_sha256', 'original_manifest',
        'origin_scope', 'payload_sha256', 'sizing', 'tensors', 'decoded_state_sha256',
        'lossy_deployment_only', 'training_admission', 'optimizer_payload', 'content_sha256'}
    if (type(payload) is not bytes or len(payload) > ARCHIVE_CAP_BYTES
            or type(manifest) is not dict or set(manifest) != fields
            or len(canonical(manifest).encode()) > MAX_MANIFEST_BYTES):
        raise ValueError('bounded immutable payload and exact manifest fields required')
    _equal(manifest, _bound(manifest), 'deployment content binding')
    _equal(manifest['content_sha256'], expected_manifest_sha256, 'external deployment identity')
    _equal(manifest['codec_identity'], expected_identity, 'codec profile')
    for key, value in {'schema': SCHEMA, 'origin_scope': _ORIGIN_SCOPE, 'lossy_deployment_only': True,
                       'training_admission': False, 'optimizer_payload': False}.items():
        _equal(manifest[key], value, key)
    _sha(manifest['original_checkpoint_sha256'])
    original, sizing = _original(manifest['original_manifest'])
    _equal(manifest['sizing'], sizing, 'shape-derived size/counts')
    _equal(len(payload), sizing['payload_bytes'], 'payload length')
    _equal(_bytes_sha(payload), _sha(manifest['payload_sha256']), 'encoded payload hash')
    records = manifest['tensors']
    if type(records) is not list or len(records) != len(original):
        raise ValueError('complete ordered deployment tensor records required')
    common = {'name','shape','storage','data_offset','data_bytes','data_sha256','scales_offset',
              'scales_bytes','scales_sha256','original_tensor_sha256','decoded_tensor_sha256','maximum_absolute_error'}
    offset = 0
    # Validate the whole dense layout before allocating decoded tensors.
    for record, name in zip(records, sorted(original), strict=True):
        if type(record) is not dict or set(record) != common:
            raise ValueError('exact tensor metadata required')
        shape = original[name]['shape']; count = math.prod(shape); routed = _ROUTED.fullmatch(name) is not None
        data_bytes = count if routed else 4*count
        scale_bytes = 4*shape[0]*math.ceil(shape[1]/GROUP) if routed else 0
        for key, value in {'name':name, 'shape':shape, 'storage':'INT8_GROUP64' if routed else 'FP32_EXACT',
                'data_offset':offset, 'data_bytes':data_bytes, 'scales_offset':offset+data_bytes if routed else None,
                'scales_bytes':scale_bytes, 'original_tensor_sha256':original[name]['sha256']}.items():
            _equal(record[key], value, 'ordered tensor layout '+key)
        _sha(record['data_sha256']); _sha(record['decoded_tensor_sha256'])
        if routed: _sha(record['scales_sha256'])
        else:
            _equal(record['scales_sha256'], None, 'unquantized tensor has no scales')
            _equal(record['decoded_tensor_sha256'], original[name]['sha256'], 'exact retained tensor')
            _equal(record['maximum_absolute_error'], 0.0, 'exact retained error')
        error = record['maximum_absolute_error']
        if type(error) is not float or not math.isfinite(error) or error < 0:
            raise ValueError('finite nonnegative recorded tensor error required')
        offset += data_bytes+scale_bytes
    _equal(offset, len(payload), 'complete dense payload extent')
    output = OrderedDict(); view = memoryview(payload)
    for record in records:
        start, size, shape = record['data_offset'], record['data_bytes'], record['shape']
        raw = view[start:start+size]
        _equal(_bytes_sha(raw), record['data_sha256'], 'encoded tensor digest')
        if record['storage'] == 'INT8_GROUP64':
            start, size = record['scales_offset'], record['scales_bytes']; raw_scale = view[start:start+size]
            _equal(_bytes_sha(raw_scale), record['scales_sha256'], 'scale digest')
            quantized = np.frombuffer(raw, dtype=np.int8).reshape(shape)
            scales = np.frombuffer(raw_scale, dtype='<f4').reshape(shape[0], math.ceil(shape[1]/GROUP))
            value = _dequantize(quantized, scales, np)
        else:
            value = np.frombuffer(raw, dtype='<f4').reshape(shape).copy(order='C')
        if not np.isfinite(value).all() or _bytes_sha(value.tobytes()) != record['decoded_tensor_sha256']:
            raise ValueError('reconstructed tensor finiteness/hash mismatch')
        output[record['name']] = value
    _equal(digest([{'name':n, 'shape':list(a.shape), 'decoded_tensor_sha256':_bytes_sha(a.tobytes())}
                   for n,a in output.items()]), manifest['decoded_state_sha256'], 'complete decoded state digest')
    return output
