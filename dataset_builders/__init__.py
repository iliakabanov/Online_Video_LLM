"""Pipelines that build the datasets (one package per dataset)."""
import sys
from pathlib import Path

# `python -m dataset_builders...` from the repository root: make online_vllm (src/) importable without installing.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
