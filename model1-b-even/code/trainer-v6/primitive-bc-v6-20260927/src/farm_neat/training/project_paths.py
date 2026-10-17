"""Explicit current registry paths; optional read-only legacy packet resolution.

No aliases are written. Historical byte-bound records retain their original path
strings; callers may explicitly resolve their known registry names for inspection.
"""
from pathlib import Path, PurePosixPath

REGISTRIES = frozenset({"AUTHORITY.json", "PROJECT-STATE.json", "WORKSPACES.json",
    "TASKS.json", "PROJECT-STORAGE.json", "DEEP-REVIEW-ASKS.json", "REVIEW_MAP.json",
    "REPRODUCTION-MAP.json", "BUNDLE.json", "SHA256SUMS.txt", "UPSTREAM-MANIFEST.sha256",
    "PACKAGE-MANIFEST.sha256"})
MOVED_PATHS = {
    "RUN_MODEL.py": "scripts/run_reference_model.py",
    **{name: "project/requirements/" + name for name in (
        "requirements.txt", "requirements.lock.txt", "requirements-oracle.lock.txt",
        "requirements-planning-review.txt", "requirements-planning-review.lock.txt",
        "requirements-review.lock.txt", "requirements-train-cu128.lock.txt")},
    **{name: "work/reference/root-artifacts/" + name for name in (
        "submission-current-neural-v1.tar.gz", "submission-current-neural-v1.preflight.json",
        "submission-current.tar.gz", "submission-current.preflight.json",
        "submission-neural-macro.tar.preflight.json", "submission-neural-macro.tar.gz",
        "current-neural-submitted-interactions.html", "current-neural-submitted-interactions.json")},
}
DOCUMENT_MOVES = {
    **{n + ".md": "docs/guides/start.md" for n in (
        "START-HERE", "START-INDEPENDENT-REVIEW", "START-CONTINUING-WORK", "AGENT-WORKFLOW", "REVIEWER-NAVIGATION")},
    "DEEP-REVIEW.md": "docs/guides/review.md",
    "RETURN-MINIMAL.md": "docs/contracts/review-return.md",
    "REVIEW-RETURN-TEMPLATE.md": "docs/contracts/review-return.md",
    "WORKSPACES.md": "docs/guides/components.md",
    "RESEARCH-TOOLS.md": "docs/guides/tools.md",
    "MODEL-RUN-README.md": "docs/guides/model-use.md",
    **{n + ".md": "docs/guides/release.md" for n in ("PACKET-CONTENTS", "ZIP-BUNDLES", "OPTIONAL-HISTORY")},
    "PLAY.html": "docs/plan-player.standalone.html",
}
MOVED_PATHS.update(DOCUMENT_MOVES)


def workspace_path(root, name, *, legacy=False):
    """Return an in-root path; registry fallback requires explicit legacy=True.

    Current registries live only in project/. Mixed root/project copies refuse.
    A fully legacy fixture/packet can be read without mutating or relabeling it.
    Missing canonical files never fall back once any current registry exists.
    """
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("symlinked repository root refused")
    if (not isinstance(name, str) or not name or "\\" in name
            or any(ord(c) < 32 for c in name) or ":" in name):
        raise ValueError("plain repository-relative path required")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name or name == ".":
        raise ValueError("normalized repository-relative path required")
    basename = name[8:] if name.startswith("project/") else name
    if basename in REGISTRIES:
        canonical, previous = root / "project" / basename, root / basename
        for path in (root / "project", canonical, previous):
            if path.is_symlink():
                raise ValueError("symlinked project registry refused")
        if canonical.exists() and previous.exists():
            raise ValueError("ambiguous root/project registry copies: " + basename)
        current = any((root / "project" / n).exists() for n in REGISTRIES)
        if legacy and not current and previous.exists():
            return previous
        return canonical
    if name in MOVED_PATHS:
        current, previous = root / MOVED_PATHS[name], root / name
        if current.exists() and previous.exists():
            raise ValueError("ambiguous previous/current artifact copies: " + name)
        path = previous if legacy and not current.exists() and previous.exists() else current
    else:
        path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlinked repository path refused")
    return path


def project_file(root, name, *, legacy=False):
    """Restricted registry-name variant, useful for callers with external input."""
    if name not in REGISTRIES and name not in {"project/" + n for n in REGISTRIES}:
        raise ValueError("unknown project registry")
    return workspace_path(root, name, legacy=legacy)
