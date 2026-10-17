"""Coordinator 07:1x (V9-FINAL-HARNESS: HUB-DONE.json certifies nothing): an archive is freezable only when its own VERIFY.json says
complete=true, fallbacks=0, and both seats' verification games have fallbacks 0, DONE/DONE and are not censored, and the VERIFY
archive_sha256 equals a fresh rehash of the tar (and the sha a builder names, when given). Returns (ok, why)."""
import hashlib, json, os
from pathlib import Path
DEP = Path("/mnt/ai/scratch/farmNEAT/pbc-deploy"); _H = {}


def tar_sha(p):
    st = os.stat(p); k = (str(p), st.st_size, st.st_mtime_ns)
    if k not in _H:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        _H[k] = h.hexdigest()
    return _H[k]


def verify_strict(n, named_sha=None):
    t, v = DEP / n / f"{n}.tar.gz", DEP / n / "VERIFY.json"
    try:
        d = json.loads(v.read_text()); sha = tar_sha(t)
    except (OSError, ValueError) as e:
        return False, f"unreadable ({type(e).__name__})"
    if d.get("complete") is not True:
        return False, "VERIFY complete != true"
    if d.get("fallbacks") != 0:
        return False, f"VERIFY fallbacks {d.get('fallbacks')}"
    g = d.get("games")
    if not isinstance(g, list) or {x.get("our_seat") for x in g if isinstance(x, dict)} != {0, 1}:
        return False, "VERIFY lacks one verification game per seat"
    for x in g:
        if x.get("fallbacks") != 0 or list(x.get("statuses") or []) != ["DONE", "DONE"] or x.get("censored") is True:
            return False, f"seat {x.get('our_seat')}: fallbacks {x.get('fallbacks')}, statuses {x.get('statuses')}, censored {x.get('censored')}"
    if d.get("archive_sha256") != sha:
        return False, "VERIFY archive_sha256 != rehash of the tar (sha_match false)"
    if named_sha and not sha.startswith(named_sha):
        return False, "rehash != the sha the builder named"
    return True, "verified"
