"""Teacher search, cleaning, corpus recording and repair tools.

Inputs come from ``work/datasets`` and generated output goes under ``work/``; neither is
ever resolved against the tool's own directory, which would read paths that do not exist
and write into the source tree. ``FARM_NEAT_OUT_DIR`` redirects the output home for a
smoke test or a parallel run without touching the tool.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATASETS = ROOT / "work/datasets"


def output_dir(default: str) -> Path:
    """Generated-output home for a tool; ``default`` is repository-relative."""
    directory = Path(os.environ.get("FARM_NEAT_OUT_DIR") or ROOT / default)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
