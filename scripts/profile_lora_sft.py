"""Where the time of a LoRA SFT step goes: waiting for data, the vision encoder, the forward and backward passes.

Runs a few training steps without updating the weights, on the same data, model and settings as train_lora_sft.py,
and prints one line per batch and the mean without the first (warm-up) batch. The vision encoder is timed by an
extra forward of its own; the loss forward includes it as well.

python scripts/profile_lora_sft.py --local-files-only [--batches 6] [--num-workers 2] [--gradient-checkpointing]
"""
import argparse
import importlib.util
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # as train_lora_sft.py

import torch  # noqa: E402
import transformers  # noqa: E402

import _paths  # noqa: F401,E402  (src/ on the path)
from lora import (  # noqa: E402
    ClevrerLoRASplitConfig, LoRASFTConfig, build_clevrer_lora_data, masked_sft_loss, prepare_lora_sft_model,
)
from lora.lora import _move_inputs, lora_records  # noqa: E402
from lora.vision_cache import VisionCache, build_vision_cache  # noqa: E402
from online_vllm.data.inputs import InputBuilder  # noqa: E402
from online_vllm.models import ModelConfig, build_model  # noqa: E402
from online_vllm.models.model import attention_kernels  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="configs/model/qwen_3b_lora_sft.yaml")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=LoRASFTConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=ClevrerLoRASplitConfig.num_workers)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=LoRASFTConfig.gradient_checkpointing)
    parser.add_argument("--vision-cache", help="time with a vision-encoder cache in this directory (filled as needed)")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    sft = replace(LoRASFTConfig(), batch_size=args.batch_size, gradient_checkpointing=args.gradient_checkpointing)
    torch.manual_seed(sft.seed)
    model = build_model(ModelConfig.load(args.model), local_files_only=args.local_files_only, device=args.device)
    model = prepare_lora_sft_model(model, model.config, gradient_checkpointing=sft.gradient_checkpointing)
    model.train()
    builder = InputBuilder.from_config(model.config, local_files_only=args.local_files_only)
    split = ClevrerLoRASplitConfig(train_videos=50, validation_videos=2, num_workers=args.num_workers,
                                   vision_cache=args.vision_cache)
    if args.vision_cache:
        cache = VisionCache.for_model(args.vision_cache, model, frame_size=split.train_frame_size,
                                      frame_stride=split.frame_stride)
        build_vision_cache(model, builder, lora_records(split, args.data_root), cache, data_root=args.data_root,
                           frame_size=split.train_frame_size, frame_stride=split.frame_stride, device=args.device,
                           num_workers=args.num_workers)
    loader, _, _ = build_clevrer_lora_data(data_root=args.data_root, builder=builder, split=split, seed=sft.seed,
                                           batch_size=sft.batch_size)
    qwen = model.qwen.model
    print(f"GPU {torch.cuda.get_device_name()}, torch {torch.__version__}, transformers {transformers.__version__}, "
          f"attention {model.qwen.config._attn_implementation}, "
          f"flash_attn {'installed' if importlib.util.find_spec('flash_attn') else 'not installed'}, "
          f"batch {sft.batch_size}, workers {args.num_workers}, checkpointing {sft.gradient_checkpointing}")

    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        return result, time.perf_counter() - start

    rows, iterator = [], iter(loader)
    for index in range(args.batches):
        started = time.perf_counter()
        batch = next(iterator)
        wait = time.perf_counter() - started
        moved = _move_inputs(batch, args.device)
        vision = 0.0  # with the cache the encoder does not run at all
        if "pixel_values_videos" in moved:
            with torch.no_grad(), attention_kernels():
                _, vision = timed(lambda: qwen.get_video_features(moved["pixel_values_videos"],
                                                                  moved["video_grid_thw"]))
        torch.cuda.reset_peak_memory_stats()
        loss, forward = timed(lambda: masked_sft_loss(model, moved))

        def backward():
            with attention_kernels():
                loss.backward()

        _, backward_time = timed(backward)
        model.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        lengths = batch["attention_mask"].sum(1).tolist()
        rows.append((wait, vision, forward, backward_time))
        print(f"batch {index}: data wait {wait:6.2f}s | vision {vision:5.2f}s | loss forward {forward:5.2f}s | "
              f"backward {backward_time:5.2f}s | tokens {min(lengths)}-{max(lengths)} | peak {peak:.1f} GB")
    if len(rows) > 1:
        mean = [sum(column) / (len(rows) - 1) for column in zip(*rows[1:])]
        print(f"mean without batch 0: data wait {mean[0]:.2f}s | vision {mean[1]:.2f}s | "
              f"loss forward {mean[2]:.2f}s | backward {mean[3]:.2f}s")


if __name__ == "__main__":
    main()
