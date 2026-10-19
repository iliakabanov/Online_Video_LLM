"""Feature shards on Google Drive (features.remote / features.remote_folder of the training config, rclone).

upload    copy the finished shards not copied yet, with meta.json and done (extract_features.py already does this
          during extraction when features.upload is set); the local copies are deleted unless --keep
download  fetch a split from Drive, e.g. to train the head on another machine (only what is missing)

python scripts/sync_features.py upload --split train eval
python scripts/sync_features.py download --split train eval
"""
import argparse
import os
from pathlib import Path

import _paths  # noqa: F401  (src/ on the path)
from online_vllm.train.config import TrainConfig
from online_vllm.train.remote import DriveUploader, download

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--train", default="configs/train/group1.yaml")
    parser.add_argument("--split", choices=["train", "eval"], nargs="+", default=["train", "eval"])
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--keep", action="store_true", help="upload: keep the local copies")
    args = parser.parse_args()

    features = TrainConfig.load(args.train).features
    if not features.remote or not features.remote_folder:
        raise SystemExit("Set features.remote and features.remote_folder in the training config")
    for split in args.split:
        local = Path(args.data_root) / getattr(features, split)
        if args.action == "download":
            download(features.remote, features.remote_folder, local)
            continue
        if not (local / "meta.json").exists():
            print(f"{local}: nothing extracted, skipped")
            continue
        uploader = DriveUploader(features.remote, features.remote_folder, local, keep_local=args.keep)
        uploader.put(local / "meta.json")
        uploaded = uploader.uploaded()
        for shard in sorted(local.glob("shard_*")):
            if (shard / "complete").exists() and shard.name not in uploaded:
                uploader.submit(shard)
        uploader.wait()
        uploader.put(local / "done")
        print(f"{local}: uploaded")


if __name__ == "__main__":
    main()
