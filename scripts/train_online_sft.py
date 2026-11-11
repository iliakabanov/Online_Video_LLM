"""Online SFT with <listen>/<speak> control tokens on top of an offline LoRA SFT run (lora/online.py).

The LoRA starts from the offline adapter (--init-run, --init-adapter) and keeps training (same settings, model.yaml of
that run), together with the rows of the two new tokens. Data: the LoRA halves of the group-1 subsets, as the offline
run (split.json of --init-run). Training takes the same videos as the offline adapter, the first n of
train_group1_lora.jsonl (data.train_videos; --skip-offline-videos: the videos after them instead); validation and test
are the same halves as the offline run's, so both models are compared on the same questions. Every question is one
sequence cut as the offline prefixes, after a pair s ~ U(t*, P - 1) drawn every epoch: a slot after every pair before
s (<listen> on the input, target <listen> before the target pair and <speak> from it on), then <speak> and the answer.
The video comes from the vision cache of the offline run (the encoder is unchanged; missing videos are encoded first).

Before training and after every epoch the validation half is evaluated with a branch after every pair: loss, answer
accuracy at the start / target / s / end, the latency / accuracy curve over all thresholds, prefix classification and
the online metrics per threshold (tau 0.5 is the native rule P(<speak>) > P(<listen>));
predictions_validation_epoch_<n>.json is in the format of eval/thresholds.py and eval/readiness_plot.py. best/ is the
adapter of the lowest prefix-classification BCE over the pairs within 2 of the target pair (--selection), last/ that
of the last epoch. The test half: scripts/evaluate_online_sft.py.

--resume-run continues an online run instead, from its last/ (--resume-adapter) into a new --output: the same model,
data, objective and optimization (each can be overridden), epochs numbered on from that of the adapter (new training
cuts), AdamW from its state when last/ has one (optimizer.pt; older runs: moments from zero). The evaluation before
training is that of the adapter's epoch again; best/ is chosen among it and the new epochs.

python scripts/train_online_sft.py --init-run outputs/lora_sft --output outputs/online_sft --tags hard
python scripts/train_online_sft.py --resume-run outputs/online_sft --output outputs/online_sft_e6 --epochs 3
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
    SELECTIONS,
    TAGS,
    OnlineObjective,
    RopeSpec,
    assert_online_isolation,
    build_online_loader,
    build_online_model,
    evaluate_online,
    expected_lora_targets,
    online_training_records,
    resume_online_run,
    train_online_sft,
)
from lora.vision_cache import VisionCache, build_vision_cache  # noqa: E402
from online_vllm.data.inputs import InputBuilder  # noqa: E402
from online_vllm.models import ModelConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]


def _pick(value, saved):
    """A flag given on the command line, else the value of the run (resumed) or the default."""
    return saved if value is None else value


def main() -> None:
    defaults, objective_defaults = LoRASFTConfig(), OnlineObjective()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--init-run", help="offline run of train_lora_sft.py (model.yaml, split.json)")
    start.add_argument("--resume-run", help="online run of this script to continue (online.json, split.json, last/)")
    parser.add_argument("--init-adapter", default="best",
                        help="adapter of --init-run the LoRA starts from: best, last, or none (a fresh LoRA)")
    parser.add_argument("--resume-adapter", default="last",
                        help="adapter of --resume-run to continue from: last (with the optimizer state) or best")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Flags that a resumed run takes from the run unless given (default None); a new run from the defaults shown.
    parser.add_argument("--tags", choices=TAGS,
                        help=f"slot targets (default {objective_defaults.tags}): hard 0/1 at the target pair; step 0.5 "
                             "in --step-window pairs before it; sigmoid with --sigmoid-slope, 0.9 at the target pair")
    parser.add_argument("--step-window", type=int, help=f"default {objective_defaults.step_window}")
    parser.add_argument("--sigmoid-slope", type=float, help=f"default {objective_defaults.sigmoid_slope}")
    parser.add_argument("--answer-weight", type=float,
                        help=f"loss = slot loss + answer_weight * answer loss (default {objective_defaults.answer_weight})")
    parser.add_argument("--taus", type=float, nargs="+", help=f"default {DEFAULT_TAUS}")
    parser.add_argument("--train-videos", type=int,
                        help="the first n training videos (default: as many as the offline adapter; with "
                             "--skip-offline-videos all the rest)")
    parser.add_argument("--skip-offline-videos", action="store_true",
                        help="train on the videos after those of the offline adapter, whose answers it has not learned")
    parser.add_argument("--selection", choices=SELECTIONS,
                        help="best/ (default bce_pm2): lowest validation prefix-classification BCE over the pairs "
                             "|k - t*| <= 2 (bce_pm2); largest area under the latency / accuracy curve over mean "
                             "lateness 0..2 pairs (auc_2) or the whole range (auc); or lowest validation loss")
    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="epochs to train (resumed: more epochs)")
    parser.add_argument("--lr", type=float, help=f"constant learning rate (default {defaults.learning_rate})")
    parser.add_argument("--batch-size", type=int, help=f"sequences per forward pass (default {defaults.batch_size})")
    parser.add_argument("--gradient-accumulation", type=int, help=f"default {defaults.gradient_accumulation}")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        help=f"default {defaults.gradient_checkpointing}")
    parser.add_argument("--num-workers", type=int, help="DataLoader workers (default: as the offline run)")
    parser.add_argument("--vision-cache", help="vision cache directory (default: that of the offline run)")
    parser.add_argument("--seed", type=int, help=f"default {defaults.seed}")
    parser.add_argument("--overwrite", action="store_true", help="delete a nonempty output directory for a new run")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    resume_run = None if args.resume_run is None else Path(args.resume_run)
    if resume_run is not None:
        if args.train_videos is not None or args.skip_offline_videos:
            parser.error("--train-videos and --skip-offline-videos: a resumed run keeps the training videos of the run")
        if Path(args.output).resolve() == resume_run.resolve():
            parser.error("--output must be a new directory, not --resume-run itself")
        source = resume_run
        saved_sft = json.loads((resume_run / "sft_config.json").read_text(encoding="utf-8"))
        saved_sft["betas"] = tuple(saved_sft["betas"])
        base_sft = LoRASFTConfig(**saved_sft)
        saved_run = json.loads((resume_run / "online.json").read_text(encoding="utf-8"))
        base_objective = OnlineObjective(**saved_run["objective"])
        base_taus, base_selection = saved_run.get("taus", DEFAULT_TAUS), saved_run.get("selection", "bce_pm2")
    else:
        source, base_sft, base_objective = Path(args.init_run), defaults, objective_defaults
        base_taus, base_selection = DEFAULT_TAUS, "bce_pm2"
    taus, selection = _pick(args.taus, base_taus), _pick(args.selection, base_selection)

    sft = replace(base_sft, epochs=args.epochs, learning_rate=_pick(args.lr, base_sft.learning_rate),
                  batch_size=_pick(args.batch_size, base_sft.batch_size),
                  gradient_accumulation=_pick(args.gradient_accumulation, base_sft.gradient_accumulation),
                  gradient_checkpointing=_pick(args.gradient_checkpointing, base_sft.gradient_checkpointing),
                  seed=_pick(args.seed, base_sft.seed), overwrite=args.overwrite)
    objective = replace(base_objective, tags=_pick(args.tags, base_objective.tags),
                        step_window=_pick(args.step_window, base_objective.step_window),
                        sigmoid_slope=_pick(args.sigmoid_slope, base_objective.sigmoid_slope),
                        answer_weight=_pick(args.answer_weight, base_objective.answer_weight))
    sft.validate()
    objective.validate()

    saved_split = json.loads((source / "split.json").read_text(encoding="utf-8"))
    data = saved_split["data"]
    data["train_frame_size"] = tuple(data["train_frame_size"])
    split = ClevrerLoRASplitConfig(**data)
    split = replace(split, **{key: value for key, value in (("num_workers", args.num_workers),
                                                            ("vision_cache", args.vision_cache)) if value is not None})
    config = ModelConfig.load(source / "model.yaml")
    if split.vision_cache is None:  # the default of train_lora_sft.py
        width, height = split.train_frame_size
        split = replace(split, vision_cache=str(Path(args.data_root) / "cache" / "lora_vision" /
                        f"{config.backbone.model_id.split('/')[-1]}_{width}x{height}_s{split.frame_stride}"))
    split.validate()

    torch.manual_seed(sft.seed)  # a fresh LoRA (--init-adapter none) draws its A in build_online_model
    if resume_run is not None:
        model, builder, tokens, _, resume = resume_online_run(resume_run, args.resume_adapter,
                                                              local_files_only=args.local_files_only, device=args.device)
        run_info = {"init_run": saved_run.get("init_run"), "init_adapter": saved_run.get("init_adapter"),
                    "resumed_from": str(resume_run / args.resume_adapter), "resumed_epoch": resume.epoch,
                    "resumed_optimizer_state": resume.optimizer_state is not None}
        print(f"Resuming {resume_run / args.resume_adapter}: epoch {resume.epoch}, {resume.global_step} optimizer "
              f"steps, AdamW {'state loaded' if resume.optimizer_state is not None else 'moments from zero'}; "
              f"training epochs {resume.epoch + 1}..{resume.epoch + sft.epochs}")
    else:
        adapter = None if args.init_adapter == "none" else source / args.init_adapter
        builder = InputBuilder.from_config(config, local_files_only=args.local_files_only)
        model, tokens = build_online_model(config, builder.processor.tokenizer, init_adapter=adapter,
                                           local_files_only=args.local_files_only, device=args.device)
        resume = None
        run_info = {"init_run": str(source), "init_adapter": None if adapter is None else str(adapter)}
    assert_online_isolation(model, expected_targets=expected_lora_targets(model))

    root = Path(args.data_root)
    if resume_run is not None:  # the training videos of the run, the same list
        online_train = saved_split["online_train"]
        skip = online_train["skipped_offline_videos"]
        train = online_training_records(root / split.train, skip_videos=skip, videos=online_train["videos"])
        if sorted({record.video for record in train}) != online_train["train_videos"]:
            raise ValueError(f"The training videos differ from those in {resume_run / 'split.json'}")
        offline_videos = len({r.video for r in _records_for_first_videos(root / split.train, split.train_videos)})
    else:
        # split.train_videos stays that of the offline run: the first n videos, which its adapter learned. None: all.
        offline_videos = len({r.video for r in _records_for_first_videos(root / split.train, split.train_videos)})
        skip = offline_videos if args.skip_offline_videos else 0
        videos = args.train_videos if args.train_videos or args.skip_offline_videos else offline_videos
        train = online_training_records(root / split.train, skip_videos=skip, videos=videos)
    train_videos = sorted({record.video for record in train})
    scenes = sorted({record.meta["scene_index"] for record in train})
    print(f"Online training: {len(train_videos)} videos, {len(train)} questions, scenes {scenes[0]}..{scenes[-1]} "
          f"(skipped the first {skip}; the offline adapter learned the first {offline_videos})")
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
                               objective=objective, device=args.device, taus=taus, desc="Validation")

    train_online_sft(model, loader, sft, output_dir=args.output, device=args.device, tokens=tokens,
                     objective=objective, evaluate=evaluate, selection=selection,
                     run_info={**run_info, "taus": taus, "selection": selection},
                     split_info={"data": asdict(split), **info,
                                 "online_train": {"skipped_offline_videos": skip, "videos": len(train_videos),
                                                  "questions": len(train), "train_videos": train_videos}},
                     resume=resume)


if __name__ == "__main__":
    main()
