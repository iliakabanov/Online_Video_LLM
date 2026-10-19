"""Train the state head on extracted features (scripts/extract_features.py); no backbone is loaded.

python scripts/train_head.py --output outputs/group1_head --device cuda
"""
import argparse
import logging
import os
from pathlib import Path

import torch

import _paths  # noqa: F401  (src/ on the path)
from online_vllm.models import ModelConfig
from online_vllm.train.config import TrainConfig
from online_vllm.train.head import train_head

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="configs/model/qwen_3b_head.yaml")
    parser.add_argument("--train", default="configs/train/group1.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    final = train_head(ModelConfig.load(args.model), TrainConfig.load(args.train), Path(args.data_root),
                       Path(args.output), args.device)
    print("Final checkpoint:", final)


if __name__ == "__main__":
    main()
