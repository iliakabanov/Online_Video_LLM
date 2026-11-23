"""Separate decision module (StreamBridge-style, separate/decision.py): a VLM of its own with a fresh LoRA, the token
<act> after every pair and a binary head, trained to tell when M_off should answer. The backbone comes from
--model-config: LLaVA-OneVision-0.5B (SB-0.5B, configs/model/llava_ov_0.5b_separate.yaml) or Qwen2.5-VL-3B without
the adapter of M_off (SB-3B, configs/model/qwen_3b_lora_sft.yaml).

Data and settings are those of the readiness heads on M_off's features (experiment_launcher/offline_10ep), so that the
module is compared with them on the same questions:
- training: every question with a target pair of the first --train-videos (1000) videos of the features half
  (train_group1_features.jsonl, by scene_index), the training set of the heads (extract_features.py --videos 1000);
- validation: exactly the questions of a head's validation predictions (--validation-from, default
  results/lora/mlp/last_hidden_mlp_off10/predictions_validation.json), taken from --records (val_group1_features.jsonl),
  with M_off's answer at every pair (answer_correct), so the online metrics are those of "the module decides, M_off
  answers". The test half is evaluated once afterwards: scripts/evaluate_separate.py.

Model and optimization: LoRA with M_off's settings (q/k/v/o of every language-model layer, r 32, alpha 64), the head of
the detector "MLP last_hidden" (--head-config, default configs/uhead/default.yaml on the group last_hidden); AdamW,
constant learning rates: LoRA and <act> at --lr (1e-4, as M_off and M_on), the head at --head-lr (5e-4, weight decay 0,
as the heads); batches 2 x 2 accumulated; --epochs 10; best/ by the largest validation AUROC over |k - t*| <= 2
(--selection auroc_pm2, as the heads). Before training and after every epoch the validation half is evaluated
(validation_metrics.jsonl, predictions_validation_epoch_<n>.json).

Visual tokens come from a vision cache (the encoder is frozen); videos missing from it are encoded first. Qwen: the cache
of the LoRA runs at 252x168 (--frame-size). LLaVA-OV: frames at its encoder's size (384x384), SigLIP features before
the projector, pooled as LLaVA-OV pools (14 x 14 = 196 per frame; --tokens-side smaller pools further), both frames of a
pair or only the last (--frames-per-pair); the projector runs in the forward pass, frozen or trained
(--train-projector).

python scripts/train_separate.py --model-config configs/model/llava_ov_0.5b_separate.yaml --output outputs/separate/sb05b
python scripts/train_separate.py --model-config configs/model/llava_ov_0.5b_separate.yaml \
    --output outputs/separate/sb05b_proj --train-projector
python scripts/train_separate.py --output outputs/separate/sb3b --gradient-checkpointing
python scripts/train_separate.py --output outputs/separate/sb05b --resume        # continue an interrupted run
Then the threshold on validation and the test: bash experiment_launcher/separate/run.sh outputs/separate/sb05b
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

import _paths  # noqa: F401,E402  (src/ on the path)
from lora.lora import ClevrerLoRASplitConfig, LoRASFTConfig  # noqa: E402
from lora.online import check_same_questions, records_of_predictions  # noqa: E402
from online_vllm.data.schema import load_jsonl  # noqa: E402
from online_vllm.models import ModelConfig  # noqa: E402
from online_vllm.train.loop import seed_everything  # noqa: E402
from separate.decision import (  # noqa: E402
    QWEN,
    RUN_FILE,
    SELECTIONS,
    TAGS,
    DecisionObjective,
    answers_of_predictions,
    assert_separate_isolation,
    build_decision_loader,
    build_separate_model,
    decision_records,
    evaluate_decision,
    expected_lora_targets,
    llava_cache_settings,
    load_processor,
    make_family,
    open_vision_cache,
    tokenizer_of,
    train_separate,
    trainable_parameters,
)
from uhead.config import UHeadConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]
FULL_LR = 1.0e-5  # full fine-tuning: LLaVA-OV's own fine-tuning rate for the language model


def main() -> None:
    defaults, objective = LoRASFTConfig(), DecisionObjective()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-config", default=REPO / "configs/model/qwen_3b_lora_sft.yaml",
                        help="backbone and LoRA settings (default: Qwen2.5-VL-3B with those of M_off)")
    parser.add_argument("--head-config", default=REPO / "configs/uhead/default.yaml",
                        help="readiness-head settings; the head is its MLP on the group last_hidden")
    parser.add_argument("--train-records", default="processed/clevrer_proactive/train_group1_features.jsonl",
                        help="training questions, relative to --data-root (the features half)")
    parser.add_argument("--train-videos", type=int, default=1000, help="the first n videos by scene_index")
    parser.add_argument("--records", default="processed/clevrer_proactive/val_group1_features.jsonl",
                        help="dataset file of the --validation-from questions, relative to --data-root")
    parser.add_argument("--validation-from",
                        default=REPO / "results/lora/mlp/last_hidden_mlp_off10/predictions_validation.json",
                        help="a readiness head's validation predictions: its questions and M_off's answers")
    parser.add_argument("--validation-questions", type=int,
                        help="only the first n questions of --validation-from (a smoke run; default: all)")
    parser.add_argument("--tags", choices=TAGS, default=objective.tags,
                        help="slot targets: hard 0/1 at the target pair; step 0.5 in --step-window pairs before it; "
                             "sigmoid with --sigmoid-slope")
    parser.add_argument("--step-window", type=int, default=objective.step_window)
    parser.add_argument("--sigmoid-slope", type=float, default=objective.sigmoid_slope)
    parser.add_argument("--selection", choices=SELECTIONS, default="auroc_pm2")
    parser.add_argument("--taus", type=float, nargs="+", default=DEFAULT_TAUS)
    parser.add_argument("--epochs", type=int,
                        help="epochs in all (default 10; with --resume: those of the run, more to extend it)")
    parser.add_argument("--lr", type=float, help=f"LoRA, <act> and a trained projector, or with --full the whole "
                                                 f"language model and projector (constant; default "
                                                 f"{defaults.learning_rate}, with --full {FULL_LR})")
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay, help="the backbone's trained weights")
    parser.add_argument("--head-lr", type=float, default=5.0e-4, help="the head (constant)")
    parser.add_argument("--head-weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="sequences per forward pass")
    parser.add_argument("--gradient-accumulation", type=int, default=defaults.gradient_accumulation)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=defaults.gradient_checkpointing)
    parser.add_argument("--eval-batch-size", type=int,
                        help="questions per forward pass in evaluation (default 4; with --resume: that of the run)")
    parser.add_argument("--frame-size", type=int, nargs=2, default=[252, 168], metavar=("WIDTH", "HEIGHT"),
                        help="Qwen: as M_off")
    parser.add_argument("--frames-per-pair", type=int, choices=(1, 2), default=2,
                        help="LLaVA-OV: both frames of a pair, or only its last")
    parser.add_argument("--tokens-side", type=int, default=7,
                        help="LLaVA-OV: tokens per frame = side^2 (default 7: 49 per frame, 98 per pair, near SB-3B's "
                             "54; 14 = its own 2 x 2 pooling, 196 per frame)")
    parser.add_argument("--train-projector", action=argparse.BooleanOptionalAction, default=False,
                        help="LLaVA-OV: train its whole projector too (at --lr, with the LoRA); default frozen")
    parser.add_argument("--full", action="store_true",
                        help="LLaVA-OV: full fine-tuning instead of LoRA — the whole language model (its embedding with "
                             "<act> too) and the projector, float32 weights in bfloat16 autocast; the vision encoder "
                             "stays frozen (its cache is the same). Checkpoints are the trained weights, ~2 GB each: "
                             "consider --no-save-every-epoch")
    parser.add_argument("--encoder-frames", type=int, default=32,
                        help="LLaVA-OV: frames per pass of the vision encoder when filling the cache (less: less memory)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--vision-cache", help="vision cache directory (default: Qwen — that of the LoRA runs, "
                                               "<data-root>/cache/lora_vision/<model>_<w>x<h>_s1; LLaVA-OV — "
                                               "<data-root>/cache/separate_vision/<model>_f<frames>_t<side>)")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--save-every-epoch", action=argparse.BooleanOptionalAction, default=True,
                        help="also save the adapter of every epoch as epoch_<n>/")
    parser.add_argument("--overwrite", action="store_true", help="delete a nonempty output directory for a new run")
    parser.add_argument("--resume", action="store_true",
                        help="continue the interrupted run in --output from its last/ with the run's own settings "
                             "(model.yaml, separate.json, split.json, sft_config.json; of the flags only --epochs, "
                             "--device, --data-root, --num-workers, --encoder-frames, --eval-batch-size apply); a run "
                             "without a finished epoch starts anew, a complete one is left as it is")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume continues a run, --overwrite deletes it: one of them")
    root, run = Path(args.data_root), Path(args.output)
    resuming = args.resume and all((run / name).exists() for name in (RUN_FILE, "split.json", "sft_config.json",
                                                                      "model.yaml"))
    if resuming:  # everything as the run was started, so that it goes on as one run
        info = _read(run / RUN_FILE)
        split_info = _read(run / "split.json")
        saved = _read(run / "sft_config.json")
        saved["betas"] = tuple(saved["betas"])
        sft = replace(LoRASFTConfig(**saved), epochs=args.epochs or saved["epochs"], overwrite=False)
        objective = DecisionObjective(**info["objective"])
        config, head_config = ModelConfig.load(run / "model.yaml"), UHeadConfig.from_dict(info["head"])
        train_projector, selection, taus = info.get("train_projector", False), info["selection"], info["taus"]
        full = info.get("full", False)
        head_lr, head_weight_decay = info["head_lr"], info["head_weight_decay"]
        save_every_epoch = info.get("save_every_epoch", True)
        eval_batch_size = args.eval_batch_size or info["eval_batch_size"]
        validation_from, validation_questions = split_info["validation_from"], split_info["validation_questions"]
        model_config, head_config_path = info["model_config"], info["head_config"]
    else:
        if args.resume:
            print(f"{run}: no run to resume, a new one starts")
        full = args.full
        lr = args.lr if args.lr is not None else (FULL_LR if full else defaults.learning_rate)
        sft = LoRASFTConfig(learning_rate=lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
                            gradient_accumulation=args.gradient_accumulation, epochs=args.epochs or 10,
                            gradient_checkpointing=args.gradient_checkpointing, seed=args.seed,
                            overwrite=args.overwrite)
        objective = DecisionObjective(args.tags, args.step_window, args.sigmoid_slope)
        config, head_config = ModelConfig.load(args.model_config), UHeadConfig.load(args.head_config)
        train_projector, selection, taus = args.train_projector or args.full, args.selection, args.taus
        head_lr, head_weight_decay, save_every_epoch = args.head_lr, args.head_weight_decay, args.save_every_epoch
        eval_batch_size = args.eval_batch_size or 4
        validation_from, validation_questions = args.validation_from, args.validation_questions
        model_config, head_config_path = args.model_config, args.head_config
    sft.validate()
    objective.validate()

    seed_everything(sft.seed)  # the fresh LoRA's A and the head draw from it (a resumed run then loads last/)
    processor = load_processor(config, local_files_only=args.local_files_only)
    model = build_separate_model(config, tokenizer_of(processor), head_config, local_files_only=args.local_files_only,
                                 device=args.device, train_projector=train_projector, full=full)
    family = make_family(model, processor)
    assert_separate_isolation(model, expected_targets=None if full else expected_lora_targets(model))
    if resuming:
        data = dict(split_info["data"])
        data.update(train_frame_size=tuple(data["train_frame_size"]), num_workers=args.num_workers)
        split, settings = ClevrerLoRASplitConfig(**data), split_info.get("vision")
    else:
        name = config.backbone.model_id.split("/")[-1]
        if model.family == QWEN:
            width, height = args.frame_size
            settings = None
            cache_dir = args.vision_cache or str(root / "cache" / "lora_vision" / f"{name}_{width}x{height}_s1")
        else:
            width, height = 252, 168  # not used by LLaVA-OV (its frames are at its encoder's size); kept valid
            settings = llava_cache_settings(model, frames_per_pair=args.frames_per_pair, tokens_side=args.tokens_side)
            cache_dir = args.vision_cache or str(  # features before the projector
                root / "cache" / "separate_vision" /
                f"{name}_f{settings['frames_per_pair']}_t{settings['tokens_side']}_vision")
        split = ClevrerLoRASplitConfig(train=args.train_records, validation=args.records,
                                       train_videos=args.train_videos, train_frame_size=(width, height),
                                       num_workers=args.num_workers, vision_cache=cache_dir)
    split.validate()
    print(f"Decision module: {config.backbone.model_id} ({model.family}), <act> = {model.act}, "
          f"{'full fine-tuning (language model + projector)' if model.full else 'LoRA'}, "
          f"projector {'trained' if model.train_projector else 'frozen'}, lr {sft.learning_rate:g}, "
          f"{trainable_parameters(model):,} trainable parameters")

    train = decision_records(root / split.train, split.train_videos, family=family, split=split)
    reference = _read(Path(validation_from))
    if validation_questions is not None:
        reference["predictions"] = reference["predictions"][:validation_questions]
    validation = records_of_predictions(reference, load_jsonl(root / split.validation))
    check_same_questions(reference, validation, builder=family, split=split)
    answers = answers_of_predictions(reference)
    train_videos = sorted({record.video for record in train})
    validation_videos = sorted({record.video for record in validation})
    if set(train_videos) & set(validation_videos):
        raise ValueError("Training and validation share videos")
    if resuming and (train_videos != split_info["train_videos"]
                     or validation_videos != split_info["validation_videos"]):
        raise ValueError(f"The training or validation videos differ from those in {run / 'split.json'}")
    print(f"Training: {len(train)} questions of {len(train_videos)} videos ({split.train}); validation: "
          f"{len(validation)} questions of {len(validation_videos)} videos (those of {validation_from})")
    cache = open_vision_cache(model, family, split, train + validation, data_root=root, device=args.device,
                              settings=settings, frames_per_pass=args.encoder_frames)

    loader = build_decision_loader(train, family=family, split=split, act=model.act, objective=objective,
                                   seed=sft.seed, batch_size=sft.batch_size)

    def evaluate():
        return evaluate_decision(model, validation, family=family, split=split, objective=objective,
                                 device=args.device, answers=answers, taus=taus, batch_size=eval_batch_size,
                                 desc="Validation")

    train_separate(model, loader, sft, family=family, output_dir=run, device=args.device, objective=objective,
                   evaluate=evaluate, selection=selection, head_lr=head_lr, head_weight_decay=head_weight_decay,
                   save_every_epoch=save_every_epoch, resume=args.resume,
                   run_info={"model_config": str(model_config), "head_config": str(head_config_path),
                             "validation_from": str(validation_from), "taus": taus,
                             "eval_batch_size": eval_batch_size, "save_every_epoch": save_every_epoch},
                   split_info={"data": asdict(split), "vision": cache.settings, "train_videos": train_videos,
                               "train_questions": len(train), "validation_from": str(validation_from),
                               "validation_videos": validation_videos, "validation_questions": len(validation)})


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
