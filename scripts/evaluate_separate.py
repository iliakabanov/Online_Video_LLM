"""Separate decision module (scripts/train_separate.py) on exactly the questions of a readiness head's predictions file:
its validation or test half (results/lora/<arch>/<id>/predictions_validation.json, predictions_test_best_tau<tau>.json),
with M_off's answer at every pair from the same file, so the module and the heads are compared question by question.
The target pair and the number of pairs of every question are checked against that file first.

One pass per question: p_k at every pair. Writes <run>/<name>_<adapter>.json (classification, latency / accuracy, online
metrics per threshold, the threshold chosen on validation, cost) and predictions_<name>_<adapter>.json in the format of
eval/thresholds.py, eval/readiness_plot.py and scripts/test_report.py (tau 0.5 is the native rule).

tau_val, the threshold of the largest --tau-criterion (default on_time_0_4: the answer comes 0..4 pairs after the target
pair, not before it), is chosen on --tau-from: by default the run's predictions_validation_epoch_<epoch of the adapter>,
which are already the head's validation questions; or an earlier --name head_validation output.

Cost (--time-questions n, default 20; 0: skip): trainable parameters, wall time of one decision through the KV cache
(from cached visual tokens: the vision encoder is not counted) and peak GPU memory, on the first n questions.

python scripts/evaluate_separate.py --run outputs/separate/sb3b --name head_validation \\
    --records-from results/lora/mlp/last_hidden_mlp_off10/predictions_validation.json
python scripts/evaluate_separate.py --run outputs/separate/sb3b --name head_test \\
    --records-from results/lora/mlp/last_hidden_mlp_off10/predictions_test_best_tau0.5.json \\
    --tau-from outputs/separate/sb3b/predictions_head_validation_best.json
"""
import argparse
import json
import os
from pathlib import Path

import torch

import _paths  # noqa: F401  (src/ on the path)
from lora.lora import ClevrerLoRASplitConfig
from lora.online import _best_threshold_line, check_same_questions, records_of_predictions
from online_vllm.data.schema import load_jsonl
from online_vllm.eval.metrics import THRESHOLD_WINDOWS
from online_vllm.eval.thresholds import sweep
from separate.decision import (DecisionObjective, answers_of_predictions, evaluate_decision, load_separate_run,
                               open_vision_cache, print_summary, time_decisions, trainable_parameters)

REPO = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="output directory of train_separate.py")
    parser.add_argument("--adapter", default="best", help="best, last or epoch_<n>")
    parser.add_argument("--records-from", required=True,
                        help="a readiness head's predictions file: its questions and M_off's answers")
    parser.add_argument("--name", required=True, help="output name: <run>/<name>_<adapter>.json")
    parser.add_argument("--records", help="dataset file of the questions, relative to --data-root "
                                          "(default: the validation file of the run, val_group1_features.jsonl)")
    parser.add_argument("--tau-from", help="validation predictions of this module to choose tau_val on (default: the "
                                           "run's predictions_validation_epoch_<epoch of the adapter>)")
    parser.add_argument("--tau-criterion", choices=tuple(THRESHOLD_WINDOWS), default="on_time_0_4")
    parser.add_argument("--taus", type=float, nargs="+", help="default: those of the run")
    parser.add_argument("--eval-batch-size", type=int, help="default: that of the run")
    parser.add_argument("--time-questions", type=int, default=20, help="questions to time streaming decisions on")
    parser.add_argument("--encoder-frames", type=int, default=32,
                        help="LLaVA-OV: frames per pass of the vision encoder when filling the cache (less: less memory)")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    run = Path(args.run)
    info = _read(run / "separate.json")
    split_info = _read(run / "split.json")
    data = dict(split_info["data"])
    data["train_frame_size"] = tuple(data["train_frame_size"])
    split = ClevrerLoRASplitConfig(**data)
    split.validate()
    taus = list(args.taus or info["taus"])
    epoch = _read(run / args.adapter / "sft_metadata.json")["epoch"]
    tau_from = Path(args.tau_from) if args.tau_from else run / f"predictions_validation_epoch_{epoch}.json"
    criterion = args.tau_criterion
    validation_best = sweep(_read(tau_from), [0.5])["best_thresholds"][criterion]
    if validation_best["tau"] not in taus:
        taus.append(validation_best["tau"])

    model, family, info = load_separate_run(run, args.adapter, local_files_only=args.local_files_only,
                                            device=args.device)
    reference = _read(Path(args.records_from))
    records_file = args.records or split.validation
    records = records_of_predictions(reference, load_jsonl(Path(args.data_root) / records_file))
    check_same_questions(reference, records, builder=family, split=split)
    metadata = {"run": str(run), "adapter": args.adapter, "family": model.family, "records_from": args.records_from,
                "records": records_file, "questions": len(records), "videos": len({record.video for record in records})}
    open_vision_cache(model, family, split, records, data_root=args.data_root, device=args.device,
                      settings=split_info.get("vision"), frames_per_pass=args.encoder_frames)
    summary, predictions = evaluate_decision(
        model, records, family=family, split=split,
        objective=DecisionObjective(**info["objective"]), device=args.device,
        answers=answers_of_predictions(reference), taus=taus,
        batch_size=args.eval_batch_size or info.get("eval_batch_size", 4), desc=f"{args.name}, adapter {args.adapter}")
    validation_tau = summary["online"][str(validation_best["tau"])]
    summary["validation_threshold"] = {"criterion": criterion, "epoch": epoch, "tau_from": str(tau_from),
                                       "validation": validation_best,
                                       f"{args.name}_{criterion}": validation_tau[criterion]}
    summary["cost"] = {"trainable_parameters": trainable_parameters(model),
                       "parameters": sum(p.numel() for p in model.parameters())}
    if args.time_questions > 0:
        summary["cost"]["timing"] = time_decisions(model, family, records[:args.time_questions], split=split)
    adapter_metadata = run / args.adapter / "sft_metadata.json"
    metadata["adapter_metadata"] = _read(adapter_metadata)
    output = run / f"{args.name}_{args.adapter}.json"
    output.write_text(json.dumps({"metadata": metadata, **summary}, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    predictions["metadata"].update(metadata)
    (run / f"predictions_{args.name}_{args.adapter}.json").write_text(
        json.dumps(predictions, allow_nan=False) + "\n", encoding="utf-8")

    print(f"{args.name}, adapter {args.adapter} (epoch {epoch}): {metadata['questions']} questions / "
          f"{metadata['videos']} videos, loss {summary['loss']:.4f}")
    print_summary(None, summary)
    print(f"  tau of the best validation {criterion} ({tau_from.name}): {validation_best['tau']:.6g} -> validation "
          f"{100 * validation_best[criterion]:.1f}%, {args.name} {100 * validation_tau[criterion]:.1f}%")
    print(f"  ({args.name} itself: " + _best_threshold_line(summary["best_thresholds"][criterion]) + ", an upper bound)")
    timing = summary["cost"].get("timing")
    if timing:
        peak = "-" if timing["peak_gpu_memory_gb"] is None else f"{timing['peak_gpu_memory_gb']:.2f} GB"
        print(f"  cost: {summary['cost']['trainable_parameters']:,} trainable / {summary['cost']['parameters']:,} "
              f"parameters; a decision {timing['ms_per_decision_mean']:.1f} ms mean, "
              f"{timing['ms_per_decision_median']:.1f} ms median ({timing['decisions']} decisions), peak {peak}")
    print(f"-> {output}")


if __name__ == "__main__":
    main()
