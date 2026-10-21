"""Train Qwen2.5-VL LoRA SFT on the documented CLEVRER group-1 video split.

The first 5,000 scene-indexed train videos train LoRA. The first 2,000 scene-indexed validation videos are halved
by video (with a fixed seed, the same for every run): the validation half selects the adapter by accuracy on a prefix ending uniformly between the target
and the end of the video (fixed per question; start, target and end are reported too), evaluated before training and
after every epoch at the training frame size; the test half is left for scripts/evaluate_lora_sft.py (split.json).
Batches of 4 sequences, no accumulation, constant learning rate.
"""
import argparse
import os
from pathlib import Path

import torch

import _paths  # noqa: F401  (src/ on the path)
from lora import (
    ClevrerLoRASplitConfig,
    LoRASFTConfig,
    build_clevrer_lora_data,
    prepare_lora_sft_model,
    train_lora_sft,
)
from online_vllm.data.inputs import InputBuilder
from online_vllm.models import ModelConfig, build_model


REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="configs/model/qwen_3b_lora_sft.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-videos", type=int, default=5000)
    parser.add_argument("--validation-videos", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true", help="delete a nonempty output directory for a new run")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model_config = ModelConfig.load(args.model)
    sft = LoRASFTConfig(overwrite=args.overwrite)
    torch.manual_seed(sft.seed)  # the random LoRA A initialization happens in build_model
    model = build_model(model_config, local_files_only=args.local_files_only, device=args.device)
    model = prepare_lora_sft_model(model, model.config, gradient_checkpointing=sft.gradient_checkpointing)
    builder = InputBuilder.from_config(model.config, local_files_only=args.local_files_only)
    split = ClevrerLoRASplitConfig(train_videos=args.train_videos, validation_videos=args.validation_videos)
    loader, validation, split_info = build_clevrer_lora_data(data_root=args.data_root, builder=builder, split=split,
                                                             seed=sft.seed, batch_size=sft.batch_size)
    train_lora_sft(model, loader, sft, output_dir=args.output, device=args.device, validation_provider=validation,
                   validation_resolution=split.train_frame_size, split_info=split_info)


if __name__ == "__main__":
    main()
