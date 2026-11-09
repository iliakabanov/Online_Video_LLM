"""Makes the repository's packages importable without installing it: online_vllm (src/) and dataset_builders (root).
Imported by every script before the project's own imports."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for path in (str(REPO), str(REPO / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)
