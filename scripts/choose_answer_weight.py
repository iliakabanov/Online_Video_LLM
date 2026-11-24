"""Выбор веса лосса ответа онлайн-SFT (scripts/train_online_sft_weight.py) по validation.

Эпоха каждого прогона — его best/ по bce_pm2 (best_epoch последней строки validation_metrics.jsonl). Вес — по
наименьшей средней задержке при доле преждевременных <= 5% и точности >= 90% (R0.05_A0.9) у этой эпохи. Для страховки
печатается та же задержка на каждой эпохе: насколько эпоха по bce_pm2 отстаёт от лучшей по задержке. Задержка
пересчитывается из predictions_validation_epoch_<n>.json без GPU, поэтому работает и для старых прогонов, которые
логировали только A 0.85 / 0.88.

python scripts/choose_answer_weight.py outputs/listen_sft_10ep outputs/listen_sft_10ep_aw0.5 \\
    outputs/listen_sft_10ep_aw0.25 outputs/listen_sft_10ep_aw0.1 --output outputs/answer_weight_choice.json
"""
import argparse
import json
import sys
from pathlib import Path

import _paths  # noqa: F401  (src/ on the path)
from online_vllm.eval.metrics import constrained_latency
from online_vllm.eval.thresholds import payload_arrays

KEYS = ("R0.05_A0.9", "R0.1_A0.9")


def latency(run: Path, epoch: int) -> dict:
    arrays = payload_arrays(json.loads((run / f"predictions_validation_epoch_{epoch}.json").read_text(encoding="utf-8")))
    if arrays["correct"] is None:
        sys.exit(f"{run}: в predictions_validation_epoch_{epoch}.json нет answer_correct")
    result = constrained_latency(arrays["probabilities"], arrays["target_pairs"], arrays["valid_mask"],
                                 arrays["correct"], budgets=(0.05, 0.10), levels=(0.90,), points=False)
    return {key: result[key] for key in KEYS}


def show(point: dict) -> str:
    if not point["feasible"]:
        return f"{'infeasible':>22s}"
    return f"{2 * point['latency_mean']:5.2f} fr (P {100 * point['premature_rate']:4.1f}%, A {100 * point['accuracy']:4.1f}%)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="JSON со сводкой")
    args = parser.parse_args()

    summary = {}
    for run in args.runs:
        rows = [json.loads(line) for line in (run / "validation_metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        weight = json.loads((run / "online.json").read_text(encoding="utf-8"))["objective"]["answer_weight"]
        best = rows[-1]["best_epoch"]
        epochs = {row["epoch"]: latency(run, row["epoch"]) for row in rows}
        print(f"\n{run}  (answer_weight {weight}, best/ по bce_pm2 — эпоха {best})")
        print(f"  {'epoch':>5s}  {'latency R0.05_A0.9':>34s}  {'latency R0.1_A0.9':>34s}")
        for epoch, points in epochs.items():
            mark = "  <- best/" if epoch == best else ""
            print(f"  {epoch:5d}  {show(points[KEYS[0]]):>34s}  {show(points[KEYS[1]]):>34s}{mark}")
        feasible = {e: p[KEYS[0]]["latency_mean"] for e, p in epochs.items() if p[KEYS[0]]["feasible"]}
        oracle = min(feasible, key=feasible.get) if feasible else None
        print(f"  лучшая эпоха по задержке R0.05_A0.9: {oracle}")
        summary[str(run)] = {"answer_weight": weight, "best_epoch": best, "oracle_epoch": oracle,
                             "best": epochs[best], "per_epoch": {str(e): p for e, p in epochs.items()}}

    ranked = sorted(summary.items(), key=lambda item: (not item[1]["best"][KEYS[0]]["feasible"],
                                                        item[1]["best"][KEYS[0]].get("latency_mean") or 0))
    print("\nВыбор веса (эпоха best/, R0.05_A0.9 на validation, задержка в кадрах):")
    for run, entry in ranked:
        print(f"  w = {entry['answer_weight']:<5g} эпоха {entry['best_epoch']:2d}: {show(entry['best'][KEYS[0]])}  {run}")
    print(f"Лучший: {ranked[0][0]}")
    if args.output:
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
