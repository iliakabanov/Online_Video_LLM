"""Train the model on proactive-QA sequences (see src/online_vllm/train/README.md).

python scripts/train.py --model configs/model/qwen_3b_head.yaml --train configs/train/group1.yaml \
    --output outputs/group1_head --device cuda
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
from online_vllm.train.loop import train

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="configs/model/qwen_3b_head.yaml")
    parser.add_argument("--train", default="configs/train/group1.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    model_config = ModelConfig.load(args.model)
    if args.device == "cpu":
        model_config.backbone.dtype = "float32"
    model = build_model(model_config, local_files_only=args.local_files_only, device=args.device)
    builder = InputBuilder.from_config(model.config, local_files_only=args.local_files_only)
    final = train(model, builder, TrainConfig.load(args.train), Path(args.data_root), Path(args.output), args.device)
    print("Final checkpoint:", final)


if __name__ == "__main__":
    main()
