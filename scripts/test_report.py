"""Test-таблица нескольких детекторов готовности на одних вопросах: у каждого порог выбирается на его validation
(по --criterion) и применяется к test (online_vllm/eval/report.py). Перед таблицей проверяется, что у всех систем
test — одни и те же вопросы с теми же целевыми парами и числом пар.

    --system ИМЯ VALIDATION.json TEST.json   (повторяется; файлы предсказаний в формате eval/export.py)
    --pair ИМЯ and|or VAL_P TEST_P VAL_Q TEST_Q
        два детектора одной модели (p — первый, его ответы и разметка; q — второй) с двумя порогами: "and" — ответ,
        когда оба выше своих порогов, "or" — когда хотя бы один; пара порогов выбирается на validation
        (online_vllm/eval/report.py two_detector_report).

python scripts/test_report.py --output results/lora/test_report_off10.json \\
    --system "maxprob" results/lora/confidence/maxprob_conf_off10/predictions_validation.json \\
                       results/lora/confidence/maxprob_conf_off10_test/predictions_test.json \\
    --system "MLP last_hidden" results/lora/mlp/last_hidden_mlp_off10/predictions_validation.json \\
                               results/lora/mlp/last_hidden_mlp_off10/predictions_test_best_tau0.5.json
"""
import argparse
import json
from pathlib import Path

import _paths  # noqa: F401  (src/ on the path)
from online_vllm.eval.report import questions, test_report, two_detector_report


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _latency(point) -> str:
    if point is None or not point.get("feasible", True) or point.get("latency_mean") is None:
        return "-"
    return f"{point['latency_mean']:.2f}/{point['latency_median']:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", nargs=3, action="append", default=[], metavar=("NAME", "VALIDATION", "TEST"))
    parser.add_argument("--pair", nargs=6, action="append", default=[],
                        metavar=("NAME", "MODE", "VAL_P", "TEST_P", "VAL_Q", "TEST_Q"))
    parser.add_argument("--criterion", choices=("on_time_0_4", "on_target_pm2"), default="on_time_0_4")
    parser.add_argument("--output", help="JSON со всеми метриками")
    args = parser.parse_args()

    if not args.system and not args.pair:
        parser.error("нужен хотя бы один --system или --pair")
    reports, reference = {}, None

    def check(name, test_payload):
        nonlocal reference
        current = questions(test_payload)
        if reference is None:
            reference = (name, current)
        elif current != reference[1]:
            differ = len(set(current.items()) ^ set(reference[1].items()))
            raise SystemExit(f"Test «{name}» и «{reference[0]}» — разные вопросы или целевые пары ({differ} расхождений)")

    for name, validation, test in args.system:
        test_payload = _load(test)
        check(name, test_payload)
        reports[name] = test_report(_load(validation), test_payload, args.criterion)
    for name, mode, val_p, test_p, val_q, test_q in args.pair:
        test_first = _load(test_p)
        check(name, test_first)
        reports[name] = two_detector_report(_load(val_p), test_first, _load(val_q), _load(test_q), mode,
                                            args.criterion)
        print(f"{name}: τ1 = {reports[name]['tau1']:.4g}, τ2 = {reports[name]['tau']:.4g} "
              f"({args.criterion} на validation {reports[name]['validation_' + args.criterion]:.4f})")

    keys = list(next(iter(reports.values()))["constrained_latency"])
    width = max(14, *(len(name) for name in reports))
    print(f"Test: {len(reference[1])} вопросов; порог каждой системы — с её validation ({args.criterion}); "
          "задержка — пары после t*, среднее/медиана при τ с validation (в скобках — по кривой самого test)")
    print(f"{'система':{width}s} {'AUROC±2':>8s} {'auc_2':>6s} {'конец':>6s} {'on_time':>8s} {'ранних':>7s} "
          f"{'пропуск':>8s} {'вовр+верн':>9s} " + " ".join(f"{key:>20s}" for key in keys))
    for name, report in reports.items():
        at, latency = report["at_tau"], report["latency_accuracy"] or {}
        end = latency.get("end_accuracy")
        cells = [f"{_latency(e['test_at_validation_tau'])} ({_latency(e['test_curve'])})"
                 for e in report["constrained_latency"].values()]
        print(f"{name:{width}s} {report['classification']['auroc_pm2']:8.4f} {latency.get('auc_2', float('nan')):6.4f} "
              f"{end if end is None else f'{end:6.4f}'} {at['on_time_0_4']:8.4f} {100 * at['premature_rate']:6.1f}% "
              f"{100 * at['missed_rate']:7.2f}% {at.get('timely_correct_0_4', float('nan')):9.4f} "
              + " ".join(f"{cell:>20s}" for cell in cells))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8")
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
