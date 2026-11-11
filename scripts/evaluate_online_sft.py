"""Online SFT run (scripts/train_online_sft.py) on its test half (or validation half): the half of the LoRA validation
videos that neither training nor the choice of the best epoch saw (split.json of the run).

One pass per question with an answer branch after every pair: p_k = P(<speak>) / (P(<speak>) + P(<listen>)) and the
answer at every pair. Writes <run>/<part>_<adapter>.json (loss, answer accuracy at the start / target / s / end, online
metrics per threshold) and predictions_<part>_<adapter>.json in the format of eval/thresholds.py and
eval/readiness_plot.py (tau = 0.5 is the native rule P(<speak>) > P(<listen>)).

--records-from evaluates instead exactly the questions of another model's predictions file, e.g. the test half of a
readiness head (uhead.evaluate: predictions_test_best_tau<tau>.json), taken from --records (default the features half
of the group-1 validation, where the heads' questions come from). The target pair and the number of pairs of every
question are checked against that file first, so the two models are compared question by question. --records-from may
also be the split.json of the head (uhead.train, in its checkpoint directory): the questions of its --records-part
half, without a predictions file of the head. --name names the
output: <run>/<name>_<adapter>.json, predictions_<name>_<adapter>.json. Videos missing from the vision cache are
encoded first.

The threshold tau_val of the largest --tau-criterion (default on_time_0_4: the answer comes 0..4 pairs after the
target pair, not before it; on_target_pm2: within 2 pairs of it) is chosen on validation predictions of the online
model: by default
those of the run (the LoRA validation half, at the epoch of the adapter); --tau-from another file, e.g. the online
model on the validation half of the head (an earlier --records-from run). Applied to the test it gives an honest number
(the best threshold of the test itself is an upper bound).

python scripts/evaluate_online_sft.py --run outputs/online_sft --adapter best
python scripts/evaluate_online_sft.py --run outputs/online_sft --name head_test \\
    --records-from results/lora/transformer/<id>/predictions_test_best_tau0.7.json
"""
import argparse
import json
import os
from pathlib import Path

import torch

import _paths  # noqa: F401  (src/ on the path)
from lora.lora import ClevrerLoRASplitConfig, clevrer_lora_part_provider
from lora.online import (OnlineObjective, RopeSpec, _best_threshold_line, _classification_line, _latency_line,
                         check_same_questions, evaluate_online, load_online_run, records_of_head_split,
                         records_of_predictions)
from online_vllm.data.schema import load_jsonl
from online_vllm.eval.metrics import THRESHOLD_WINDOWS
from online_vllm.eval.thresholds import sweep
from lora.vision_cache import VisionCache, build_vision_cache

REPO = Path(__file__).resolve().parents[1]
HEAD_RECORDS = "processed/clevrer_proactive/val_group1_features.jsonl"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_split(run: Path) -> ClevrerLoRASplitConfig:
    data = dict(_read(run / "split.json")["data"])
    data["train_frame_size"] = tuple(data["train_frame_size"])
    split = ClevrerLoRASplitConfig(**data)
    split.validate()
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="output directory of train_online_sft.py")
    parser.add_argument("--adapter", choices=("best", "last"), default="best")
    parser.add_argument("--part", choices=("test", "validation"), default="test",
                        help="half of the run's LoRA validation videos (without --records-from)")
    parser.add_argument("--records-from", help="predictions file of another model (evaluate exactly its questions) or "
                                               "split.json of a readiness head (its --records-part half)")
    parser.add_argument("--records-part", choices=("validation", "test"), default="test",
                        help="half of a head's split.json given as --records-from")
    parser.add_argument("--records", default=HEAD_RECORDS,
                        help=f"dataset file of the --records-from questions, relative to --data-root ({HEAD_RECORDS})")
    parser.add_argument("--name", help="output name (default: --part; required with --records-from)")
    parser.add_argument("--tau-from", help="validation predictions of the online model to choose tau_val on "
                                           "(default: the run's predictions_validation_epoch_<epoch of the adapter>)")
    parser.add_argument("--tau-criterion", choices=tuple(THRESHOLD_WINDOWS), default="on_time_0_4",
                        help="tau_val maximizes this share on validation: on_time_0_4 = answered 0..4 pairs after the "
                             "target pair (not before it); on_target_pm2 = within 2 pairs of it")
    parser.add_argument("--taus", type=float, nargs="+", help="default: those of the run")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.records_from and not args.name:
        parser.error("--records-from needs --name for its output files")
    name = args.name or args.part

    run = Path(args.run)
    taus = list(args.taus or _read(run / "online.json")["taus"])
    epoch = _read(run / args.adapter / "sft_metadata.json")["epoch"]
    tau_from = Path(args.tau_from) if args.tau_from else run / f"predictions_validation_epoch_{epoch}.json"
    criterion = args.tau_criterion
    validation_best = sweep(_read(tau_from), [0.5])["best_thresholds"][criterion]
    if validation_best["tau"] not in taus:
        taus.append(validation_best["tau"])
    model, builder, tokens, info = load_online_run(run, args.adapter, local_files_only=args.local_files_only,
                                                   device=args.device)
    metadata = {"run": str(run), "adapter": args.adapter}
    if args.records_from:
        reference = _read(Path(args.records_from))
        split = _run_split(run)
        dataset = load_jsonl(Path(args.data_root) / args.records)
        if "predictions" in reference:
            records = records_of_predictions(reference, dataset)
            check_same_questions(reference, records, builder=builder, split=split)
        else:  # split.json of uhead.train
            records = records_of_head_split(reference, args.records_part, dataset, builder=builder, split=split)
            metadata["records_part"] = args.records_part
        metadata.update(records_from=args.records_from, records=args.records)
    else:
        provider, split = clevrer_lora_part_provider(run, args.part, data_root=args.data_root, builder=builder)
        records = provider.records
        metadata["split"] = args.part
    metadata.update(questions=len(records), videos=len({record.video for record in records}))
    cache = VisionCache.for_model(split.vision_cache, model, frame_size=split.train_frame_size,
                                  frame_stride=split.frame_stride)
    build_vision_cache(model, builder, records, cache, data_root=args.data_root, frame_size=split.train_frame_size,
                       frame_stride=split.frame_stride, device=args.device, num_workers=split.num_workers)
    objective = OnlineObjective(**info["objective"])
    summary, predictions = evaluate_online(model, records, builder=builder, split=split, tokens=tokens,
                                           spec=RopeSpec.of(model), objective=objective, device=args.device,
                                           taus=taus, desc=f"{name}, adapter {args.adapter}")
    validation_tau = summary["online"][str(validation_best["tau"])]
    summary["validation_threshold"] = {"criterion": criterion, "epoch": epoch, "tau_from": str(tau_from),
                                       "validation": validation_best, f"{name}_{criterion}": validation_tau[criterion]}
    adapter_metadata = run / args.adapter / "sft_metadata.json"
    if adapter_metadata.exists():
        metadata["adapter_metadata"] = _read(adapter_metadata)
    output = run / f"{name}_{args.adapter}.json"
    output.write_text(json.dumps({"metadata": metadata, **summary}, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    predictions["metadata"].update(metadata)
    (run / f"predictions_{name}_{args.adapter}.json").write_text(
        json.dumps(predictions, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{name}, adapter {args.adapter}: {metadata['questions']} questions / {metadata['videos']} videos, "
          f"loss {summary['loss']:.4f}, answer at the end {100 * summary['end_accuracy']:.2f}%")
    print("  " + _latency_line(summary["latency_accuracy"]))
    print("  " + _classification_line(summary["classification"]))
    print(f"  tau of the best validation {criterion} ({tau_from.name}): {validation_best['tau']:.6g} -> validation "
          f"{100 * validation_best[criterion]:.1f}%, {name} {100 * validation_tau[criterion]:.1f}%")
    print(f"  ({name} itself: " + _best_threshold_line(summary["best_thresholds"][criterion]) + ", an upper bound)")
    for tau, values in summary["online"].items():
        print(f"  tau={tau}: on_time_0_4 {100 * values['on_time_0_4']:.1f}%, on_time_0_9 "
              f"{100 * values['on_time_0_9']:.1f}%, on_target_pm2 {100 * values['on_target_pm2']:.1f}%, premature "
              f"{100 * values['premature_rate']:.1f}%, late {100 * values['late_rate']:.1f}%, missed "
              f"{100 * values['missed_rate']:.1f}%, answer {100 * values['answer_accuracy']:.1f}%, timely_correct "
              f"0_4 {100 * values['timely_correct_0_4']:.1f}% / 0_9 {100 * values['timely_correct_0_9']:.1f}%")
    print(f"-> {output}")


if __name__ == "__main__":
    main()
