#!/usr/bin/env python3
"""Extra release checks for an N flow archive (coordinator 04:5x, EASY-17):
1. the EFFECTIVE training receipt is a non-empty JSON object;
2. the archive's policy.pt equals fp16(checkpoint EMA) tensor by tensor (same key set, shapes, dtypes).
  python3 p2_check.py ARCHIVE.tar.gz CKPT EFFECTIVE.json OUT.json
"""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile

import torch

arc, ck, eff, out = sys.argv[1:5]
res = {"archive": arc, "checkpoint": ck, "effective": eff}
e = json.loads(Path(eff).read_text())
res["effective_nonempty"] = isinstance(e, dict) and bool(e) and bool(e.get("cmd"))
with tarfile.open(arc, "r:gz") as t:
    raw = t.extractfile("policy.pt").read()
res["policy_pt_sha256"] = hashlib.sha256(raw).hexdigest()
blob = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
tens = blob["tensors"]
# Hash and deserialize the SAME bytes: bind top-up intent to the actual export source.
checkpoint_bytes = Path(ck).read_bytes()
res["checkpoint_sha256"] = hashlib.sha256(checkpoint_bytes).hexdigest()
checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
ema = checkpoint["ema"]
res["checkpoint_update"] = checkpoint.get("update")
# review PRO NFLOW-LIVE #3: the EFFECTIVE receipt must name these exact checkpoint bytes (release_/final_ identities)
identities = {}
for prefix in ("release", "final"):
    declared_path = e.get(f"{prefix}_checkpoint")
    declared_sha = e.get(f"{prefix}_checkpoint_sha256")
    if declared_path is not None or declared_sha is not None:
        identities[prefix] = {"path": declared_path, "sha256": declared_sha}
preferred = identities.get("release") or identities.get("final") or {}
res["effective_checkpoint_declared"] = bool(identities)
res["effective_checkpoint_identities"] = identities
res["effective_checkpoint_path"] = preferred.get("path")
res["effective_checkpoint_sha256"] = preferred.get("sha256")
res["effective_binds_checkpoint"] = bool(identities) and all(
    isinstance(identity.get("path"), str)
    and isinstance(identity.get("sha256"), str)
    and os.path.basename(identity["path"]) == os.path.basename(ck)
    and identity["sha256"] == res["checkpoint_sha256"]
    for identity in identities.values()
)
# Reuse the already-hashed policy and checkpoint objects. Never mix reads across a file replacement.

res["policy_config_equals_checkpoint"] = (isinstance(checkpoint.get("config"), dict)
    and bool(checkpoint["config"]) and blob.get("config") == checkpoint["config"])
res["checkpoint_schema_valid"] = checkpoint.get("schema") == "nflow-policy-v1"
expected = e.get("final_checkpoint_sha256") if isinstance(e, dict) else None
# Generic legacy SPEC receipts have no checkpoint digest; report that limitation honestly.
# A top-up EFFECTIVE receipt, however, MUST supply and match its final-checkpoint digest.
requires_binding = isinstance(e, dict) and any(k in e for k in ("final_checkpoint", "final_checkpoint_sha256", "seed_checkpoint"))
res["effective_checkpoint_sha256"] = expected
res["effective_checkpoint_matches"] = (expected == res["checkpoint_sha256"]) if requires_binding else None
res["effective_binding_required"] = requires_binding
res["storage"] = blob.get("storage")
missing, extra, bad = sorted(set(ema) - set(tens)), sorted(set(tens) - set(ema)), []
nonfinite_ema, nonfinite_policy = [], []
for k in sorted(set(ema) & set(tens)):
    a, b = ema[k], tens[k]
    if a.is_floating_point() and not bool(torch.isfinite(a).all()):
        nonfinite_ema.append(k)
    want = a.half() if a.is_floating_point() else a
    if isinstance(b, torch.Tensor) and b.is_floating_point() and not bool(torch.isfinite(b).all()):
        nonfinite_policy.append(k)
    if not isinstance(b, torch.Tensor) or b.shape != want.shape or b.dtype != want.dtype or not torch.equal(b, want):
        bad.append(k)
res.update(tensors=len(tens), ema_tensors=len(ema), missing=missing[:5], extra=extra[:5], mismatched=bad[:5])
res["policy_equals_fp16_ema"] = not missing and not extra and not bad and res["storage"] == "fp16"
res["nonfinite_ema"] = nonfinite_ema[:5]
res["nonfinite_policy"] = nonfinite_policy[:5]
res["weights_finite"] = not nonfinite_ema and not nonfinite_policy
res["pass"] = res["effective_binds_checkpoint"] and bool(res["effective_nonempty"] and res["policy_equals_fp16_ema"] and res["weights_finite"]
    and res["policy_config_equals_checkpoint"] and res["checkpoint_schema_valid"]
    and (not requires_binding or res["effective_checkpoint_matches"]))
Path(out).write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: res[k] for k in ("pass", "effective_nonempty", "effective_binds_checkpoint", "policy_equals_fp16_ema", "tensors", "policy_pt_sha256")}))
sys.exit(0 if res["pass"] else 2)
