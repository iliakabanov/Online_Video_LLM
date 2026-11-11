"""Online SFT run (scripts/train_online_sft.py) on its test half (or validation half): the half of the LoRA validation
videos that neither training nor the choice of the best epoch saw (split.json of the run).

One pass per question with an answer branch after every pair: p_k = P(<speak>) / (P(<speak>) + P(<listen>)) and the
answer at every pair. Writes <run>/<part>_<adapter>.json (loss, answer accuracy at the start / target / s / end, online
metrics per threshold) and predictions_<part>_<adapter>.json in the format of eval/thresholds.py and
eval/readiness_plot.py (tau = 0.5 is the native rule P(<speak>) > P(<listen>)).

python scripts/evaluate_online_sft.py --run outputs/online_sft --adapter best
"""
import argparse
import json
import os
from pathlib import Path

import torch

import _paths  # noqa: F401  (src/ on the path)
from lora.lora import clevrer_lora_part_provider
from lora.online import (OnlineObjective, RopeSpec, _best_threshold_line, _classification_line, _latency_line,
                         evaluate_online, load_online_run)
from online_vllm.eval.thresholds import sweep
from lora.vision_cache import VisionCache, build_vision_cache

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="output directory of train_online_sft.py")
    parser.add_argument("--adapter", choices=("best", "last"), default="best")
    parser.add_argument("--part", choices=("test", "validation"), default="test")
    parser.add_argument("--taus", type=float, nargs="+", help="default: those of the run")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    run = Path(args.run)
    taus = list(args.taus or json.loads((run / "online.json").read_text(encoding="utf-8"))["taus"])
    # The threshold of the largest on_target_pm2 on the validation half at the epoch of this adapter: applied to the
    # test half it gives an honest number (the best threshold of the test half itself is an upper bound).
    epoch = json.loads((run / args.adapter / "sft_metadata.json").read_text(encoding="utf-8"))["epoch"]
    validation_predictions = run / f"predictions_validation_epoch_{epoch}.json"
    validation_best = sweep(json.loads(validation_predictions.read_text(encoding="utf-8")), [0.5])["best_threshold_pm2"]
    taus.append(validation_best["tau"])
    model, builder, tokens, info = load_online_run(run, args.adapter, local_files_only=args.local_files_only,
                                                   device=args.device)
    provider, split = clevrer_lora_part_provider(run, args.part, data_root=args.data_root, builder=builder)
    records = provider.records
    cache = VisionCache.for_model(split.vision_cache, model, frame_size=split.train_frame_size,
                                  frame_stride=split.frame_stride)
    build_vision_cache(model, builder, records, cache, data_root=args.data_root, frame_size=split.train_frame_size,
                       frame_stride=split.frame_stride, device=args.device, num_workers=split.num_workers)
    objective = OnlineObjective(**info["objective"])
    summary, predictions = evaluate_online(model, records, builder=builder, split=split, tokens=tokens,
                                           spec=RopeSpec.of(model), objective=objective, device=args.device,
                                           taus=taus, desc=f"{args.part}, adapter {args.adapter}")
    validation_tau = summary["online"][str(validation_best["tau"])]
    summary["validation_threshold_pm2"] = {"epoch": epoch, "validation": validation_best,
                                           f"{args.part}_on_target_pm2": validation_tau["on_target_pm2"]}
    metadata = {"run": str(run), "adapter": args.adapter, "split": args.part, "questions": len(records),
                "videos": len({record.video for record in records})}
    adapter_metadata = run / args.adapter / "sft_metadata.json"
    if adapter_metadata.exists():
        metadata["adapter_metadata"] = json.loads(adapter_metadata.read_text(encoding="utf-8"))
    output = run / f"{args.part}_{args.adapter}.json"
    output.write_text(json.dumps({"metadata": metadata, **summary}, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    predictions["metadata"].update(metadata)
    (run / f"predictions_{args.part}_{args.adapter}.json").write_text(
        json.dumps(predictions, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.part}, adapter {args.adapter}: {metadata['questions']} questions / {metadata['videos']} videos, "
          f"loss {summary['loss']:.4f}, answer at the end {100 * summary['end_accuracy']:.2f}%")
    print("  " + _latency_line(summary["latency_accuracy"]))
    print("  " + _classification_line(summary["classification"]))
    print(f"  tau of the best validation on_target_pm2 (epoch {epoch}): {validation_best['tau']:.6g} -> validation "
          f"{100 * validation_best['on_target_pm2']:.1f}%, {args.part} {100 * validation_tau['on_target_pm2']:.1f}%")
    print(f"  ({args.part} itself: " + _best_threshold_line(summary["best_threshold_pm2"]) + ", an upper bound)")
    for tau, values in summary["online"].items():
        print(f"  tau={tau}: on_target_pm2 {100 * values['on_target_pm2']:.1f}%, premature "
              f"{100 * values['premature_rate']:.1f}%, missed {100 * values['missed_rate']:.1f}%, answer "
              f"{100 * values['answer_accuracy']:.1f}%, timely_correct_pm2 {100 * values['timely_correct_pm2']:.1f}%")
    print(f"-> {output}")


if __name__ == "__main__":
    main()
