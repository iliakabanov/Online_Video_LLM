"""Where the CLEVRER-Proactive pipeline reads and writes. Defaults live under <repo>/data (not in git):

    data/raw/clevrer/                {train,validation,test}.json, annotations/{train,validation}/, videos/
    data/cache/clevrer/              cache_<split>.pkl (compact annotation arrays)
    data/processed/clevrer_proactive/  <split>.json (generated), <split>_balanced.json, <split>.jsonl (common schema)

DATA_ROOT moves everything; CLEVRER_RAW points at an existing copy of the raw CLEVRER files.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO / "data"))
RAW = Path(os.environ.get("CLEVRER_RAW", DATA_ROOT / "raw" / "clevrer"))
CACHE = DATA_ROOT / "cache" / "clevrer"
PROCESSED = DATA_ROOT / "processed" / "clevrer_proactive"
