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


def video_path(split, name):
    """Where video_XXXXX.mp4 is: the extracted CLEVRER zip as is (<RAW>/video_<split>/video_10000-11000/video_10001.mp4)
    if it is there, else the flat <RAW>/videos/ of download.py --videos all."""
    lo = int(name[6:11]) // 1000 * 1000
    unzipped = RAW / f"video_{split}" / f"video_{lo:05d}-{lo + 1000:05d}" / name
    return unzipped if unzipped.exists() else RAW / "videos" / name
