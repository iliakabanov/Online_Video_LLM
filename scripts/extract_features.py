"""Extract the backbone's features for the state head (src/online_vllm/train/FEATURES.md).

Takes the records of data.train / data.eval, the frames of video.* and the layers of features.* from the training
config, and writes to features.train / features.eval in shards of features.shard_gb. With features.upload every
finished shard is copied to Google Drive (rclone) and deleted locally. Rerun the same command to resume an
interrupted extraction (and an interrupted upload).

By default the backbone is the frozen base model of --model. With --adapter it is the model fine-tuned by
scripts/train_lora_sft.py: built from the model.yaml of its run, with the adapter loaded; the output directory gets
the suffix _<run>_<adapter> and meta.json records the adapter, so its features never mix with the base model's.

python scripts/extract_features.py --split train --device cuda
python scripts/extract_features.py --split eval --device cuda
python scripts/extract_features.py --split train --videos 300                        # -> <features.train>_videos300
python scripts/extract_features.py --split train --adapter outputs/lora_1000/best   # -> <features.train>_lora_1000_best
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import torch

import _paths  # noqa: F401  (src/ on the path)
from lora import load_lora_adapter
from online_vllm.data.inputs import InputBuilder
from online_vllm.models import ModelConfig, build_model
from online_vllm.train.config import TrainConfig
from online_vllm.train.features import extract
from online_vllm.train.remote import DriveUploader

REPO = Path(__file__).resolve().parents[1]


def adapter_description(path: Path) -> dict:
    """What identifies an adapter in meta.json: where it is, which run and epoch, and the hash of its weights (best/
    and last/ are overwritten while a run trains: resuming with other weights must fail)."""
    weights = path / "adapter_model.safetensors"
    digest = hashlib.sha256()
    with open(weights, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    metadata = path / "sft_metadata.json"
    epoch = json.loads(metadata.read_text(encoding="utf-8")).get("epoch") if metadata.exists() else None
    return {"path": str(path.resolve()).replace("\\", "/"), "run": path.parent.name, "name": path.name,
            "epoch": epoch, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="configs/model/qwen_3b_head.yaml",
                        help="the base model (ignored with --adapter: the model.yaml of the adapter's run is used)")
    parser.add_argument("--adapter", help="LoRA adapter directory of train_lora_sft.py, e.g. outputs/lora_1000/best")
    parser.add_argument("--train", default="configs/train/group1.yaml")
    parser.add_argument("--split", choices=["train", "eval"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--num-workers", type=int, default=1, help="each worker is a process with its own copy of "
                        "torch and the records (~2 GB RAM); one keeps up with decoding")
    parser.add_argument("--videos", type=int, default=None,
                        help="only the first N videos of the file, every question of each (~6.5 per video): a trial "
                             "or pilot run, written next to the full run as <dir>_videosN")
    parser.add_argument("--no-upload", action="store_true", help="keep every shard locally whatever features.upload")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = TrainConfig.load(args.train)
    root = Path(args.data_root)
    out = root / getattr(config.features, args.split)
    adapter = None
    if args.adapter:
        adapter_path = Path(args.adapter)
        if not (adapter_path / "adapter_model.safetensors").exists() or not (adapter_path.parent / "model.yaml").exists():
            raise FileNotFoundError(f"{adapter_path} is not an adapter of a train_lora_sft.py run "
                                    "(adapter_model.safetensors here, model.yaml in the run directory)")
        adapter = adapter_description(adapter_path)
        out = out.with_name(f"{out.name}_{adapter['run']}_{adapter['name']}")
    if args.videos is not None:  # a trial run never occupies the directory of the full one
        if args.videos < 1:
            parser.error("--videos must be positive")
        out = out.with_name(f"{out.name}_videos{args.videos}")
    uploader = None
    if config.features.upload and not args.no_upload:  # before loading the model: fail early without rclone
        uploader = DriveUploader(config.features.remote, config.features.remote_folder, out)
    model_config = ModelConfig.load(Path(args.adapter).parent / "model.yaml" if adapter else args.model)
    if args.device == "cpu":
        model_config.backbone.dtype = "float32"
    model = build_model(model_config, local_files_only=args.local_files_only, device=args.device)
    if adapter:
        load_lora_adapter(model, Path(args.adapter))
        logging.info("LoRA adapter %s (run %s, epoch %s)", adapter["path"], adapter["run"], adapter["epoch"])
    if torch.device(args.device).type == "cuda":
        # Past the physical memory Windows silently moves GPU allocations into RAM and everything slows down tenfold;
        # fail with an out-of-memory error instead.
        torch.cuda.set_per_process_memory_fraction(0.95)
    builder = InputBuilder.from_config(model.config, local_files_only=args.local_files_only)
    out = extract(model, builder, root / getattr(config.data, args.split), root, config.video, config.features, out,
                  args.device, args.num_workers, videos=args.videos, uploader=uploader, adapter=adapter)
    print("Features:", out)


if __name__ == "__main__":
    main()
