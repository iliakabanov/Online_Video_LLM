"""Download the raw CLEVRER files (http://clevrer.csail.mit.edu/, CC0) into <RAW> (see paths.py):

    <RAW>/{train,validation,test}.json                       questions with programs and answers (test: no answers)
    <RAW>/annotations/{train,validation}/annotation_*/...     scene annotations (224 MB zipped, 1.9 GB extracted)
    <RAW>/videos/video_*.mp4                                 only with --videos all (train 12.4 GB + validation 6.2 GB)

Without --videos the review page (viewer.py) fetches the videos it shows one by one from the remote zips.

    python -m dataset_builders.clevrer_proactive.download [--videos all] [--keep-zips]
"""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .paths import RAW

BASE_URL = "https://data.csail.mit.edu/clevrer/"
SPLITS = ("train", "validation")


def fetch(url, dst: Path):
    if dst.exists():
        print(f"[skip] {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 2**22)
    tmp.replace(dst)
    print(f"[ok  ] {dst} ({dst.stat().st_size / 2**20:.0f} MiB)")


def unzip(zp: Path, dst: Path, keep):
    """Extract the zip; members keep their paths (annotation_00000-01000/annotation_00000.json, video_.../video_*.mp4)."""
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        bad = z.testzip()
        if bad:
            raise SystemExit(f"corrupt member {bad} in {zp}")
        z.extractall(dst)
    if not keep:
        zp.unlink()
    print(f"[ok  ] extracted {zp.name} -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", choices=["none", "all"], default="none")
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()

    for split in SPLITS + ("test",):
        fetch(BASE_URL + f"questions/{split}.json", RAW / f"{split}.json")
    for split in SPLITS:
        out = RAW / "annotations" / split
        if out.exists() and any(out.iterdir()):
            print(f"[skip] {out}")
            continue
        zp = RAW / "annotations" / f"annotation_{split}.zip"
        fetch(BASE_URL + f"annotations/{split}/annotation_{split}.zip", zp)
        unzip(zp, out, args.keep_zips)
    if args.videos == "all":
        for split in SPLITS:
            zp = RAW / "videos" / f"video_{split}.zip"
            fetch(BASE_URL + f"videos/{split}/video_{split}.zip", zp)
            tmp = RAW / "videos" / f"_{split}"
            unzip(zp, tmp, args.keep_zips)
            for mp4 in tmp.rglob("*.mp4"):  # flatten: <RAW>/videos/video_XXXXX.mp4, as the viewer expects
                mp4.replace(RAW / "videos" / mp4.name)
            shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
