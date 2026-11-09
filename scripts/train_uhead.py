"""Обучить causal UHead на ранее извлечённых признаках, без загрузки Qwen.

python scripts/train_uhead.py --output outputs/group1_uhead --device cuda
"""
import argparse
import logging
import os
from pathlib import Path

import torch

import _paths  # noqa: F401
from online_vllm.train.config import TrainConfig
from uhead.config import UHeadConfig
from uhead.train import train

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uhead", default="configs/uhead/default.yaml")
    parser.add_argument("--train", default="configs/train/group1.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = train(UHeadConfig.load(args.uhead), TrainConfig.load(args.train), Path(args.data_root), Path(args.output),
                   args.device, tags="hard")
    print("Final checkpoint:", result)


if __name__ == "__main__":
    main()
