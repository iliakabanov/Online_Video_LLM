"""Train Qwen2.5-VL LoRA SFT on the documented CLEVRER group-1 video split.

Data: the LoRA halves of the group-1 subsets (README §11; the other halves are for the features of the readiness
heads). The videos of train_group1_lora.jsonl train LoRA (--train-videos: only the first n). The videos of
val_group1_lora.jsonl (--validation-videos: only the first m) are halved by video (with a fixed seed, the same for
every run with the same m): the validation half selects the adapter by accuracy on a prefix
ending uniformly between the target and the end of the video (fixed per question; start, target and end are reported
too), evaluated before training and after every epoch at the training frame size; the test half is left for
scripts/evaluate_lora_sft.py (split.json).
By default: micro-batches of 2 sequences of similar length, 2 accumulated (the optimizer steps of batches of 4), no
gradient checkpointing, constant learning rate; every default comes from LoRASFTConfig and can be overridden below.
The frozen vision encoder runs once per video into a cache (lora/vision_cache.py; the missing videos are encoded
before training, ~14 MB per video at 252x168), and training and validation read its output from there.
"""
import argparse
import os
from dataclasses import replace
from pathlib import Path

# Before torch touches CUDA: fewer out-of-memory errors from fragmentation with sequences of varying length.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

import _paths  # noqa: F401,E402  (src/ on the path)
from lora import (  # noqa: E402
    ClevrerLoRASplitConfig,
    LoRASFTConfig,
    build_clevrer_lora_data,
    prepare_lora_sft_model,
    train_lora_sft,
)
from lora.lora import lora_records  # noqa: E402
from lora.vision_cache import VisionCache, build_vision_cache  # noqa: E402
from online_vllm.data.inputs import InputBuilder  # noqa: E402
from online_vllm.models import ModelConfig, build_model  # noqa: E402


REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    defaults, split_defaults = LoRASFTConfig(), ClevrerLoRASplitConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="configs/model/qwen_3b_lora_sft.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-videos", type=int, default=split_defaults.train_videos,
                        help="the first n videos of the training file (default: all)")
    parser.add_argument("--validation-videos", type=int, default=split_defaults.validation_videos,
                        help="the first m videos of the validation file, halved into validation and test (default: "
                             "all). Keep m the same in runs to be compared: it fixes their validation and test")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate, help="constant learning rate")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="sequences per forward pass")
    parser.add_argument("--gradient-accumulation", type=int, default=defaults.gradient_accumulation,
                        help="micro-batches per optimizer step")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=defaults.gradient_checkpointing,
                        help="recompute activations in backward: far less memory, ~4x slower steps")
    parser.add_argument("--num-workers", type=int, default=split_defaults.num_workers,
                        help="DataLoader workers decoding videos, for training and validation")
    parser.add_argument("--vision-cache", help="vision-encoder cache directory; default "
                                               "<data root>/cache/lora_vision/<model>_<width>x<height>_s<stride>")
    parser.add_argument("--no-vision-cache", action="store_true", help="decode and encode every video at every step")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--overwrite", action="store_true", help="delete a nonempty output directory for a new run")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    sft = replace(defaults, epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size,
                  gradient_accumulation=args.gradient_accumulation, gradient_checkpointing=args.gradient_checkpointing,
                  seed=args.seed, overwrite=args.overwrite)
    sft.validate()
    torch.manual_seed(sft.seed)  # the random LoRA A initialization happens in build_model
    model = build_model(ModelConfig.load(args.model), local_files_only=args.local_files_only, device=args.device)
    model = prepare_lora_sft_model(model, model.config, gradient_checkpointing=sft.gradient_checkpointing)
    builder = InputBuilder.from_config(model.config, local_files_only=args.local_files_only)
    split = ClevrerLoRASplitConfig(train_videos=args.train_videos, validation_videos=args.validation_videos,
                                   num_workers=args.num_workers)
    if not args.no_vision_cache:
        width, height = split.train_frame_size
        path = args.vision_cache or Path(args.data_root) / "cache" / "lora_vision" / \
            f"{model.config.backbone.model_id.split('/')[-1]}_{width}x{height}_s{split.frame_stride}"
        split = replace(split, vision_cache=str(path))
        cache = VisionCache.for_model(path, model, frame_size=split.train_frame_size, frame_stride=split.frame_stride)
        added = build_vision_cache(model, builder, lora_records(split, args.data_root), cache, data_root=args.data_root,
                                   frame_size=split.train_frame_size, frame_stride=split.frame_stride,
                                   device=args.device, num_workers=split.num_workers)
        print(f"Vision cache {path}: {added} videos encoded now")
    loader, validation, split_info = build_clevrer_lora_data(data_root=args.data_root, builder=builder, split=split,
                                                             seed=sft.seed, batch_size=sft.batch_size)
    train_lora_sft(model, loader, sft, output_dir=args.output, device=args.device, validation_provider=validation,
                   validation_resolution=split.train_frame_size, split_info=split_info)


if __name__ == "__main__":
    main()
