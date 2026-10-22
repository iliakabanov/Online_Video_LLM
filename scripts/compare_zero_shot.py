"""Сравнение систем без дообучения по результатам scripts/zero_shot_probe.py (eval/ZERO_SHOT.md), без моделей.

Для каждой системы: метрики на замерах first / target / mid / end, прирост от first к target, AUROC уверенности,
разрезы. Попарно: совпадение ответов и McNemar на каждом замере. Кривые точности по сдвигу относительно целевой пары
у всех систем на одном графике. У онлайн-системы ещё «готовность» 1 − P(NO REPLY): predictions в формате
eval/export.py, перебор порогов (eval/thresholds.py) и кривые готовности (eval/readiness_plot.py).

python scripts/compare_zero_shot.py --runs results/zero_shot/qwen_offline results/zero_shot/mmduet2_offline \
    results/zero_shot/mmduet2_online --output-dir results/zero_shot
"""
import argparse
import json
from pathlib import Path

import _paths  # noqa: F401  (src/ on the path)
from online_vllm.eval.readiness_plot import load_traces, plot_readiness, readiness_curves
from online_vllm.eval.thresholds import sweep
from online_vllm.eval.zero_shot import (CHECKPOINT_NAMES, CHECKPOINTS, compare_systems, curves_of, plot_systems,
                                        readiness_predictions, system_metrics)


def load_run(path: Path):
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (path / "answers.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return meta, rows


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def report(summary: dict) -> str:
    """Короткий отчёт в markdown: главные таблицы summary.json."""
    lines = ["# Оценка без дообучения", ""]
    systems = summary["systems"]
    lines += ["## Точность на замерах (95% ДИ)", "",
              "| система | " + " | ".join(CHECKPOINT_NAMES[c] for c in CHECKPOINTS) + " | прирост target − first (p) |",
              "|---|" + "---|" * (len(CHECKPOINTS) + 1)]
    for name, m in systems.items():
        cells = [f"{fmt(m['checkpoints'][c]['accuracy'])} [{fmt(m['checkpoints'][c]['accuracy_ci95'][0], 2)}; "
                 f"{fmt(m['checkpoints'][c]['accuracy_ci95'][1], 2)}]" for c in CHECKPOINTS]
        gain = m["gain_target_vs_first"]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {gain['accuracy']:+.3f} ({gain['p_value']:.3g}) |")
    chance = next(iter(systems.values()))["checkpoints"]["first"]["chance"]
    lines += ["", f"Угадывание: {chance:.3f}.", ""]

    lines += ["## Уверенность", "", "| система | замер | P(правильный) | P(ответ) | отрыв лидера | энтропия | "
              "масса на буквах (медиана) |", "|---|---|---|---|---|---|---|"]
    for name, m in systems.items():
        for c in CHECKPOINTS:
            s = m["checkpoints"][c]
            lines.append(f"| {name} | {CHECKPOINT_NAMES[c]} | {fmt(s['p_correct'])} | {fmt(s['p_answer'])} | "
                         f"{fmt(s['margin'])} | {fmt(s['entropy'])} | {s['letter_mass_median']:.2e} |")
    lines += ["", "AUROC уверенности (P(ответ) на target против first): " + ", ".join(
        f"{name} {fmt(m['confidence_auroc_target_vs_first'])}" for name, m in systems.items()), ""]

    online = {name: m for name, m in systems.items() if "self_reply_rate" in m["checkpoints"]["first"]}
    if online:
        lines += ["## Собственное решение онлайн-модели", "",
                  "| система | замер | P(NO REPLY): квартили | сама начала бы отвечать | сама и верно |",
                  "|---|---|---|---|---|"]
        for name, m in online.items():
            for c in CHECKPOINTS:
                s = m["checkpoints"][c]
                q = ", ".join(fmt(v) for v in s["p_no_reply_quartiles"])
                lines.append(f"| {name} | {CHECKPOINT_NAMES[c]} | {q} | {fmt(s['self_reply_rate'])} | "
                             f"{fmt(s['self_reply_correct_rate'])} |")
        for name, values in summary["readiness"].items():
            c = values["thresholds"]["classification"]
            lines += ["", f"**{name}**, готовность = 1 − P(NO REPLY): AUROC «пара до / начиная с целевой» "
                          f"{fmt(c['auroc'])}, AUPRC {fmt(c['auprc'])}.", "",
                      "| τ | преждевр. | в цель | поздно | пропуск | ±2 пары | ответ верный |", "|---|---|---|---|---|---|---|"]
            for tau, r in values["thresholds"]["results"].items():
                lines.append(f"| {tau} | {fmt(r['premature_rate'])} | {fmt(r['on_target_rate'])} | "
                             f"{fmt(r['late_rate'])} | {fmt(r['missed_rate'])} | {fmt(r['on_target_pm2'])} | "
                             f"{fmt(r.get('answer_accuracy'))} |")
        lines.append("")

    lines += ["## Попарно", "", "| системы | замер | совпадение ответов | разность точности | p (McNemar) |",
              "|---|---|---|---|---|"]
    for pair, values in summary["comparisons"].items():
        for c in CHECKPOINTS:
            v = values[c]
            lines.append(f"| {pair} | {CHECKPOINT_NAMES[c]} | {fmt(v['agreement'])} | {v['accuracy_difference']:+.3f} | "
                         f"{v['p_value']:.3g} |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="каталоги запусков zero_shot_probe.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9], help="пороги готовности")
    parser.add_argument("--seed", type=int, default=0, help="бутстреп и примеры на графиках")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs, metas = {}, {}
    for path in map(Path, args.runs):
        meta, rows = load_run(path)
        name = meta["system"]
        if name in runs:
            parser.error(f"система {name} дважды")
        metas[name], runs[name] = meta, rows
    # Общие вопросы в порядке первой системы: у недосчитанного запуска сравниваются только посчитанные вопросы.
    common = set.intersection(*({r["id"] for r in rows} for rows in runs.values()))
    order = [r["id"] for r in next(iter(runs.values())) if r["id"] in common]
    for name, rows in runs.items():
        by_id = {r["id"]: r for r in rows}
        runs[name] = [by_id[i] for i in order]
        if len(rows) != len(order):
            print(f"{name}: {len(rows)} вопросов, в сравнении общие {len(order)}")

    summary = {"questions": len(order), "runs": {name: str(path) for name, path in zip(runs, args.runs)},
               "metadata": metas, "systems": {name: system_metrics(rows, args.seed) for name, rows in runs.items()},
               "comparisons": compare_systems(runs) if len(runs) > 1 else {},
               "curves": {name: curves_of(rows) for name, rows in runs.items()}, "readiness": {}}
    plot_systems(summary["curves"], out / "answer_curves.png", f"{len(order)} вопросов, ответ по буквам вариантов")

    for name, rows in runs.items():
        if "silence" not in rows[0]:
            continue
        payload = readiness_predictions(rows, metas[name])
        predictions = out / f"predictions_{name}.json"
        predictions.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        curves = readiness_curves(load_traces(payload), range(-30, 31), sorted(args.taus)[-4:], 0.5, 4, args.seed)
        (out / f"readiness_curves_{name}.json").write_text(
            json.dumps(curves, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        plot_readiness(curves, out / f"readiness_curves_{name}.png",
                       f"{name}: готовность = 1 − P(NO REPLY), {len(rows)} вопросов")
        summary["readiness"][name] = {"predictions": str(predictions), "thresholds": sweep(payload, args.taus)}

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    (out / "report.md").write_text(report(summary), encoding="utf-8")
    print(report(summary))


if __name__ == "__main__":
    main()
