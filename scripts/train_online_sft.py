"""Online SFT with <listen>/<speak> control tokens on top of an offline LoRA SFT run (lora/online.py).

The offline adapter (--init-run, --init-adapter) is merged into Qwen; a new LoRA of the same settings (model.yaml of
that run) and the rows of the two new tokens are trained. Data: the LoRA halves of the group-1 subsets, as the offline
run (split.json of --init-run). Training takes the videos of train_group1_lora.jsonl after the first n the offline
adapter learned (data.train_videos; --train-videos: only the first of those), since on its own videos it knows the
answers; validation and test are the same halves as the offline run's, so both models are compared on the same
questions. Every question is one sequence: the whole video with a slot after every pair (<listen> on the input,
target <listen> before the target pair and <speak> from it on) and two answer branches, after a pair s ~ U(t*, P - 1)
drawn every epoch and after the last pair. The video comes from the vision cache of the offline run (the encoder is
unchanged; missing videos are encoded first).

Before training and after every epoch the validation half is evaluated with a branch after every pair: loss, answer
accuracy at the start / target / s / end, and the online metrics per threshold (tau 0.5 is the native rule
P(<speak>) > P(<listen>)); predictions_validation_epoch_<n>.json is in the format of eval/thresholds.py and
eval/readiness_plot.py. best/ is the adapter of the lowest validation loss, last/ that of the last epoch. The test half:
scripts/evaluate_online_sft.py.

python scripts/train_online_sft.py --init-run outputs/lora_sft --output outputs/online_sft --tags hard
"""
import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

import _paths  # noqa: F401,E402  (src/ on the path)
from lora.lora import (  # noqa: E402
    VALIDATION_SPLIT_SEED,
    ClevrerLoRASplitConfig,
    LoRASFTConfig,
    _records_for_first_videos,
    split_validation_videos,
)
from lora.online import (  # noqa: E402
    TAGS,
    OnlineObjective,
    RopeSpec,
    assert_online_isolation,
    build_online_loader,
    build_online_model,
    evaluate_online,
    expected_lora_targets,
    online_training_records,
    train_online_sft,
)
from lora.vision_cache import VisionCache, build_vision_cache  # noqa: E402
from online_vllm.data.inputs import InputBuilder  # noqa: E402
from online_vllm.models import ModelConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    defaults, objective_defaults = LoRASFTConfig(), OnlineObjective()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init-run", required=True, help="offline run of train_lora_sft.py (model.yaml, split.json)")
    parser.add_argument("--init-adapter", default="best",
                        help="adapter of --init-run merged into Qwen: best, last, or none (start from the base Qwen)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tags", choices=TAGS, default=objective_defaults.tags,
                        help="slot targets: hard 0/1 at the target pair; step 0.5 in --step-window pairs before it; "
                             "sigmoid with --sigmoid-slope, 0.9 at the target pair")
    parser.add_argument("--step-window", type=int, default=objective_defaults.step_window)
    parser.add_argument("--sigmoid-slope", type=float, default=objective_defaults.sigmoid_slope)
    parser.add_argument("--answer-weight", type=float, default=objective_defaults.answer_weight,
                        help="loss = slot loss + answer_weight * answer loss")
    parser.add_argument("--taus", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--train-videos", type=int,
                        help="the first n training videos after those of the offline adapter (default: all of them)")
    parser.add_argument("--reuse-offline-videos", action="store_true",
                        help="train on the videos of the offline adapter too (it knows their answers: a leak)")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate, help="constant learning rate")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="sequences per forward pass")
    parser.add_argument("--gradient-accumulation", type=int, default=defaults.gradient_accumulation)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=defaults.gradient_checkpointing)
    parser.add_argument("--num-workers", type=int, help="DataLoader workers (default: as the offline run)")
    parser.add_argument("--vision-cache", help="vision cache directory (default: that of the offline run)")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--overwrite", action="store_true", help="delete a nonempty output directory for a new run")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    init_run = Path(args.init_run)
    sft = replace(defaults, epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size,
                  gradient_accumulation=args.gradient_accumulation, gradient_checkpointing=args.gradient_checkpointing,
                  seed=args.seed, overwrite=args.overwrite)
    objective = OnlineObjective(tags=args.tags, step_window=args.step_window, sigmoid_slope=args.sigmoid_slope,
                                answer_weight=args.answer_weight)
    sft.validate()
    objective.validate()

    data = json.loads((init_run / "split.json").read_text(encoding="utf-8"))["data"]
    data["train_frame_size"] = tuple(data["train_frame_size"])
    split = ClevrerLoRASplitConfig(**data)
    split = replace(split, **{key: value for key, value in (("num_workers", args.num_workers),
                                                            ("vision_cache", args.vision_cache)) if value is not None})
    config = ModelConfig.load(init_run / "model.yaml")
    if split.vision_cache is None:  # the default of train_lora_sft.py
        width, height = split.train_frame_size
        split = replace(split, vision_cache=str(Path(args.data_root) / "cache" / "lora_vision" /
                        f"{config.backbone.model_id.split('/')[-1]}_{width}x{height}_s{split.frame_stride}"))
    split.validate()

    adapter = None if args.init_adapter == "none" else init_run / args.init_adapter
    builder = InputBuilder.from_config(config, local_files_only=args.local_files_only)
    torch.manual_seed(sft.seed)  # the random LoRA A initialization happens in build_online_model
    model, tokens = build_online_model(config, builder.processor.tokenizer, init_adapter=adapter,
                                       local_files_only=args.local_files_only, device=args.device)
    assert_online_isolation(model, expected_targets=expected_lora_targets(model))

    root = Path(args.data_root)
    # split.train_videos stays that of the offline run: the first n videos, which its adapter learned. None: all.
    offline_videos = len({r.video for r in _records_for_first_videos(root / split.train, split.train_videos)})
    skip = 0 if args.reuse_offline_videos else offline_videos
    train = online_training_records(root / split.train, skip_videos=skip, videos=args.train_videos)
    train_videos = sorted({record.video for record in train})
    scenes = sorted({record.meta["scene_index"] for record in train})
    print(f"Online training: {len(train_videos)} videos, {len(train)} questions, scenes {scenes[0]}..{scenes[-1]} "
          f"(skipped the first {skip} videos, those of the offline adapter)")
    validation, _, info = split_validation_videos(
        _records_for_first_videos(root / split.validation, split.validation_videos), VALIDATION_SPLIT_SEED)
    cache = VisionCache.for_model(split.vision_cache, model, frame_size=split.train_frame_size,
                                  frame_stride=split.frame_stride)
    added = build_vision_cache(model, builder, train + validation, cache, data_root=root,
                               frame_size=split.train_frame_size, frame_stride=split.frame_stride, device=args.device,
                               num_workers=split.num_workers)
    print(f"Vision cache {split.vision_cache}: {added} videos encoded now")

    spec = RopeSpec.of(model)
    loader = build_online_loader(train, builder=builder, split=split, tokens=tokens, spec=spec, objective=objective,
                                 seed=sft.seed, batch_size=sft.batch_size)

    def evaluate():
        return evaluate_online(model, validation, builder=builder, split=split, tokens=tokens, spec=spec,
                               objective=objective, device=args.device, taus=args.taus, desc="Validation")

    train_online_sft(model, loader, sft, output_dir=args.output, device=args.device, tokens=tokens,
                     objective=objective, evaluate=evaluate,
                     run_info={"init_run": str(init_run), "init_adapter": None if adapter is None else str(adapter),
                               "taus": args.taus},
                     split_info={"data": asdict(split), **info,
                                 "online_train": {"skipped_offline_videos": skip, "videos": len(train_videos),
                                                  "questions": len(train), "train_videos": train_videos}})


if __name__ == "__main__":
    main()
