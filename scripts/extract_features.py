"""Extract the frozen backbone's features for the state head (src/online_vllm/train/FEATURES.md).

Takes the records of data.train / data.eval, the frames of video.* and the layers of features.* from the training
config, and writes to features.train / features.eval in shards of features.shard_gb. With features.upload every
finished shard is copied to Google Drive (rclone) and deleted locally. Rerun the same command to resume an
interrupted extraction (and an interrupted upload).

python scripts/extract_features.py --split train --device cuda
python scripts/extract_features.py --split eval --device cuda
"""
import argparse
import logging
import os
from pathlib import Path

import torch

import _paths  # noqa: F401  (src/ on the path)
from online_vllm.data.inputs import InputBuilder
from online_vllm.models import ModelConfig, build_model
from online_vllm.train.config import TrainConfig
from online_vllm.train.features import extract
from online_vllm.train.remote import DriveUploader

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="configs/model/qwen_3b_head.yaml")
    parser.add_argument("--train", default="configs/train/group1.yaml")
    parser.add_argument("--split", choices=["train", "eval"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--num-workers", type=int, default=1, help="each worker is a process with its own copy of "
                        "torch and the records (~2 GB RAM); one keeps up with decoding")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N records (a trial run), written next to the full run as <dir>_firstN")
    parser.add_argument("--no-upload", action="store_true", help="keep every shard locally whatever features.upload")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = TrainConfig.load(args.train)
    root = Path(args.data_root)
    out = root / getattr(config.features, args.split)
    if args.limit is not None:  # a trial run never occupies the directory of the full one
        out = out.with_name(f"{out.name}_first{args.limit}")
    uploader = None
    if config.features.upload and not args.no_upload:  # before loading the model: fail early without rclone
        uploader = DriveUploader(config.features.remote, config.features.remote_folder, out)
    model_config = ModelConfig.load(args.model)
    if args.device == "cpu":
        model_config.backbone.dtype = "float32"
    model = build_model(model_config, local_files_only=args.local_files_only, device=args.device)
    if torch.device(args.device).type == "cuda":
        # Past the physical memory Windows silently moves GPU allocations into RAM and everything slows down tenfold;
        # fail with an out-of-memory error instead.
        torch.cuda.set_per_process_memory_fraction(0.95)
    builder = InputBuilder.from_config(model.config, local_files_only=args.local_files_only)
    out = extract(model, builder, root / getattr(config.data, args.split), root, config.video, config.features, out,
                  args.device, args.num_workers, limit=args.limit, uploader=uploader)
    print("Features:", out)


if __name__ == "__main__":
    main()
