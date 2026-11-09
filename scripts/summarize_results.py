"""Сводная таблица запусков детекторов готовности: метрики validation и test при пороге, выбранном на validation.

Читает results/<mode>/<arch>/<id>/: selection.json (порог, experiment_launcher/evaluate_test.sh), results.json
(validation при каждом пороге) и test_best_tau<τ>.json (uhead.evaluate). Запуски без selection.json или без test
выводятся с прочерками. Ни признаки, ни веса, ни torch не нужны.

python scripts/summarize_results.py                      # все запуски results/, markdown-таблица
python scripts/summarize_results.py --mode lora --arch mlp
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (заголовок, ключ метрики онлайн-оценки при пороге, формат)
ONLINE = [("±2 пары", "on_target_pm2", "{:.3f}"), ("в цель", "on_target_rate", "{:.3f}"),
          ("раньше", "premature_rate", "{:.3f}"), ("позже", "late_rate", "{:.3f}"), ("пропуск", "missed_rate", "{:.3f}"),
          ("MAE, кадры", "timing_mae_frames", "{:.1f}"), ("ответ верен", "answer_accuracy", "{:.3f}"),
          ("верен и ±2", "timely_correct_pm2", "{:.3f}")]


def _format(value, fmt: str) -> str:
    return "—" if value is None else fmt.format(value)


def rows(results_root: Path, mode: str | None, arch: str | None):
    for result in sorted(results_root.glob("*/*/*/results.json")):
        run = result.parent
        run_mode, run_arch, run_id = run.relative_to(results_root).parts
        if mode and run_mode != mode or arch and run_arch != arch:
            continue
        validation = json.loads(result.read_text(encoding="utf-8"))["validation"]
        selection_path = run / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else None
        tau = selection["threshold"] if selection else None
        test_path = run / f"test_best_tau{tau}.json"
        test = json.loads(test_path.read_text(encoding="utf-8")) if tau is not None and test_path.exists() else None
        yield {"mode": run_mode, "arch": run_arch, "id": run_id, "tau": tau,
               "validation": validation["thresholds"].get(str(tau), {}) if tau is not None else {},
               "validation_auroc": validation["classification"]["auroc"],
               "test": test["online"] if test else {}, "test_auroc": test["classification"]["auroc"] if test else None,
               "test_questions": test["questions"] if test else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(REPO / "results"), help="корень results/<mode>/<arch>/<id>")
    parser.add_argument("--mode", choices=["original", "lora"])
    parser.add_argument("--arch", choices=["mlp", "transformer"])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # консоль Windows по умолчанию не печатает τ
    table =list(rows(Path(args.results), args.mode, args.arch))
    if not table:
        raise SystemExit(f"В {args.results} нет запусков")
    header = ["режим", "запуск", "τ (val)", "±2 пары, val", "AUROC, test"] + [f"{name}, test" for name, _, _ in ONLINE]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in table:
        cells = [row["mode"], f"{row['arch']}/{row['id']}", _format(row["tau"], "{:g}"),
                 _format(row["validation"].get("on_target_pm2"), "{:.3f}"), _format(row["test_auroc"], "{:.4f}")]
        cells += [_format(row["test"].get(key), fmt) for _, key, fmt in ONLINE]
        print("| " + " | ".join(cells) + " |")
    questions = {row["test_questions"] for row in table if row["test_questions"] is not None}
    if questions:
        print(f"\nВопросов в test: {', '.join(map(str, sorted(questions)))}")


if __name__ == "__main__":
    main()
