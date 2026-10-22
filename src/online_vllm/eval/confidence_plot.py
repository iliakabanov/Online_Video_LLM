"""Уверенность головы готовности и точность ответа Qwen: по времени и друг против друга, без признаков и без Qwen.

Вход — predictions JSON в формате online_vllm.eval.export (predictions_validation.json из uhead.train,
predictions_test_*.json из uhead.evaluate): P(готов) и верность ответа Qwen по пробе на каждой паре каждого вопроса.

- time: по сдвигу пары относительно целевой — медиана, среднее и 10–90-й перцентили P(готов) и точность ответа Qwen
  на той же паре (одна пара на вопрос, поэтому каждый вопрос весит одинаково);
- bins: точки (вопрос, пара) по бинам P(готов) равной ширины. Точность, средняя уверенность и средний сдвиг бина
  считаются с весом 1 / (число пар вопроса в бине): каждый вопрос, попавший в бин, весит в нём ровно 1. Иначе бин
  заполняют вопросы, у которых в нём больше всего пар (поздняя цель — много пар до неё), а не типичный вопрос.
  Интервал — ±1,96·√(p(1−p)/n) по числу вопросов n.
- levels: угадывание (1 / число вариантов, нужны записи вопросов --records) и точность на последней паре видео.
- tradeoff: компромисс точности и задержки, не зависящий от порога. При пороге τ вопрос отвечается на первой паре
  f с P > τ, а если такой нет — на последней паре видео. Точка τ — точность ответа Qwen на f и задержка,
  усреднённые по вопросам, в двух вариантах:
    signed   — среднее f − t относительно целевой пары t: ранний ответ уменьшает задержку;
    absolute — среднее f, пары от начала видео (вопрос задан на паре 0). Та же кривая, что signed, сдвинутая
               на среднюю целевую пару, поэтому AUC и delta_auc у них совпадают.
  Опора — таймер: ответ на паре k для всех вопросов, k = 0 … L − 1. У обеих кривых общие концы (все отвечают на
  первой паре / на последней), поэтому площади сравнимы: AUC — площадь под кривой, делённая на ширину оси X,
  delta_auc = AUC головы − AUC таймера (> 0: при той же задержке голова в среднем точнее таймера). Для signed ещё
  точность головы и таймера при задержке 0.

PYTHONPATH=src python -m online_vllm.eval.confidence_plot --predictions results/lora/mlp/01_lm/predictions_validation.json \
    --output results/lora/mlp/01_lm/confidence_accuracy.json --plot results/lora/mlp/01_lm/confidence_accuracy.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .answers import GRID, INK, INK_2, MUTED, SERIES
from .readiness_plot import ReadinessTrace, _curve, _style, load_traces

# Диверсирующая шкала сдвига: синий — до целевой пары, красный — после, серый — на ней.
EARLY, NEUTRAL, LATE = "#2a78d6", "#f0efec", "#e34948"
DEFAULT_RECORDS = "processed/clevrer_proactive/val_group1_features.jsonl"


def chance_by_question(records_path: Path) -> dict[str, float]:
    """1 / число вариантов по id вопроса."""
    from ..data.schema import load_jsonl
    return {record.id: 1 / len(record.choices) for record in load_jsonl(records_path)}


def confidence_bins(traces: list[ReadinessTrace], bins: int = 10) -> list[dict]:
    """Бины P(готов) равной ширины; точность, уверенность и сдвиг с равным весом вопросов внутри бина."""
    if any(t.answer_correct is None for t in traces):
        raise ValueError("Нужна верность ответа Qwen (answer_correct) на каждой паре")
    probability = np.concatenate([t.probabilities for t in traces])
    correct = np.concatenate([t.answer_correct for t in traces]).astype(np.float64)
    offset = np.concatenate([np.arange(len(t.probabilities)) - t.target_pair for t in traces]).astype(np.float64)
    question = np.concatenate([np.full(len(t.probabilities), i) for i, t in enumerate(traces)])
    index = np.minimum((probability * bins).astype(int), bins - 1)
    result = []
    for b in range(bins):
        members = index == b
        if not members.any():
            result.append({"low": b / bins, "high": (b + 1) / bins, "questions": 0, "pairs": 0})
            continue
        _, inverse, counts = np.unique(question[members], return_inverse=True, return_counts=True)
        weights = 1 / counts[inverse]
        accuracy = float(np.average(correct[members], weights=weights))
        n = len(counts)
        result.append({"low": b / bins, "high": (b + 1) / bins, "questions": n, "pairs": int(members.sum()),
                       "confidence": float(np.average(probability[members], weights=weights)),
                       "accuracy": accuracy, "accuracy_ci": float(1.96 * np.sqrt(accuracy * (1 - accuracy) / n)),
                       "offset": float(np.average(offset[members], weights=weights))})
    return result


def _points(fired: np.ndarray, target: np.ndarray, correct: np.ndarray) -> dict:
    """Средние по вопросам при ответе на парах fired [Q]: точность, задержка от целевой пары и от начала видео."""
    return {"accuracy": float(correct[np.arange(len(fired)), fired].mean()),
            "signed": float((fired - target).mean()), "absolute": float(fired.mean())}


def _area(points: list[dict], axis: str, start: float, end: float) -> float:
    """Площадь под кривой точность(задержка) методом трапеций, делённая на ширину оси [start, end]."""
    x = np.array([p[axis] for p in points])
    y = np.array([p["accuracy"] for p in points])
    order = np.argsort(x, kind="stable")
    return float(np.trapezoid(y[order], x[order]) / (end - start))


def tradeoff(traces: list[ReadinessTrace], taus=None, marked=(0.3, 0.5, 0.7, 0.9)) -> dict:
    """Кривые точность — задержка головы (по порогам) и таймера (по паре ответа); AUC и прирост над таймером."""
    taus = np.round(np.linspace(0, 1, 1001), 3) if taus is None else np.asarray(taus)
    lengths = np.array([len(t.probabilities) for t in traces])
    width = lengths.max()
    # Пары за концом короткого видео: P = −1 никогда не превышает τ, верность не используется.
    probability = np.full((len(traces), width), -1.0)
    correct = np.zeros((len(traces), width), dtype=bool)
    for i, t in enumerate(traces):
        probability[i, :lengths[i]] = t.probabilities
        correct[i, :lengths[i]] = t.answer_correct
    target = np.array([t.target_pair for t in traces])
    last = lengths - 1
    first = _points(np.zeros(len(traces), dtype=int), target, correct)
    end = _points(last, target, correct)

    head = []
    for tau in taus:
        hits = probability > tau
        fired = np.where(hits.any(axis=1), hits.argmax(axis=1), last)
        head.append({"tau": float(tau), **_points(fired, target, correct)})
    timer = [{"pair": k, **_points(np.minimum(k, last), target, correct)} for k in range(width)]
    # Общие концы: все отвечают на первой паре / на последней.
    head_curve, timer_curve = [first, *head, end], [first, *timer, end]

    result = {"first_pair": first, "end_of_video": end, "mean_target_pair": float(target.mean()), "head": head,
              "timer": timer, "marked": [p for p in head if any(np.isclose(p["tau"], m) for m in marked)]}
    for axis in ("signed", "absolute"):
        head_auc = _area(head_curve, axis, first[axis], end[axis])
        timer_auc = _area(timer_curve, axis, first[axis], end[axis])
        result[axis] = {"head_auc": head_auc, "timer_auc": timer_auc, "delta_auc": head_auc - timer_auc}
    for name, curve in (("head", head_curve), ("timer", timer_curve)):
        x = np.array([p["signed"] for p in curve])
        order = np.argsort(x, kind="stable")
        result["signed"][f"{name}_accuracy_at_0"] = float(np.interp(0.0, x[order],
                                                                    [curve[i]["accuracy"] for i in order]))
    return result


def confidence_accuracy(traces: list[ReadinessTrace], offsets=range(-30, 31), bins: int = 10,
                        chance: dict[str, float] | None = None) -> dict:
    levels = {"end_of_video": float(np.mean([t.answer_correct[-1] for t in traces])), "chance": None}
    if chance is not None:
        missing = [t.record_id for t in traces if t.record_id not in chance]
        if missing:
            raise ValueError(f"Нет записей для {len(missing)} вопросов, например {missing[0]}")
        levels["chance"] = float(np.mean([chance[t.record_id] for t in traces]))
    return {"questions": len(traces), "levels": levels, "time": _curve(traces, list(offsets)),
            "bins": confidence_bins(traces, bins), "tradeoff": tradeoff(traces)}


def _level(ax, value: float | None, text: str, x: float, ha: str = "right"):
    if value is None:
        return
    ax.axhline(value, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(x, value, f" {text} ", color=INK_2, fontsize=9, va="bottom", ha=ha)


def _plot_tradeoff(ax, tradeoff: dict, axis: str, title: str, xlabel: str):
    """Кривая точность — задержка головы по порогам и таймера по паре ответа, с подписанными порогами."""
    from matplotlib.ticker import PercentFormatter
    first, end = tradeoff["first_pair"], tradeoff["end_of_video"]
    metrics = tradeoff[axis]
    for name, curve, style in (("timer", tradeoff["timer"], {"color": MUTED, "linestyle": (0, (4, 3))}),
                               ("head", tradeoff["head"], {"color": SERIES[0]})):
        points = sorted([first, *curve, end], key=lambda p: p[axis])
        label = ("таймер: ответ на паре k для всех" if name == "timer" else "голова: ответ при P > τ") + \
            f", AUC {metrics[f'{name}_auc']:.3f}"
        ax.plot([p[axis] for p in points], [p["accuracy"] for p in points], linewidth=2, label=label, zorder=3,
                **style)
    # Пороги лежат близко друг к другу: подписи лесенкой вниз с выноской, чтобы не налезали.
    for i, point in enumerate(tradeoff["marked"]):
        ax.plot(point[axis], point["accuracy"], "o", color=SERIES[0], markersize=7, markeredgecolor="white",
                markeredgewidth=1.5, zorder=4)
        ax.annotate(f"τ = {point['tau']:g}", (point[axis], point["accuracy"]), xytext=(18, -22 - 14 * i),
                    textcoords="offset points", color=INK_2, fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    x, text = (0, " ответ появился") if axis == "signed" else (tradeoff["mean_target_pair"], " средняя целевая пара")
    ax.axvline(x, color=MUTED, linewidth=1, zorder=1)
    ax.text(x, 1.0, text, color=INK_2, fontsize=9, va="top", ha="left", transform=ax.get_xaxis_transform())
    ax.plot([], [], " ", label=f"прирост головы над таймером ΔAUC {metrics['delta_auc']:+.3f}")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    _style(ax, title, xlabel)
    ax.set_ylabel("точность ответа Qwen", color=INK_2, fontsize=9)


def plot_confidence_accuracy(result: dict, path: str | Path, title: str | None = None):
    """PNG из четырёх графиков: P(готов) и точность по сдвигу от целевой пары; точность по бинам P(готов);
    компромисс точности и задержки относительно целевой пары и от начала видео, голова против таймера."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.ticker import PercentFormatter

    fig, ((time_ax, bins_ax), (signed_ax, absolute_ax)) = plt.subplots(2, 2, figsize=(14, 11.2),
                                                                     constrained_layout=True)
    levels = result["levels"]

    points = [row for row in result["time"] if row["count"]]
    x = [row["offset"] for row in points]
    time_ax.fill_between(x, [row["q10"] for row in points], [row["q90"] for row in points], color=SERIES[0],
                         alpha=0.18, linewidth=0, label="P(готов), 10–90-й перцентили", zorder=2)
    time_ax.plot(x, [row["median"] for row in points], color=SERIES[0], linewidth=2, label="P(готов), медиана",
                 zorder=3)
    time_ax.plot(x, [row["mean"] for row in points], color=SERIES[0], linewidth=1.2, linestyle=(0, (4, 3)),
                 label="P(готов), среднее", zorder=3)
    accuracy = [(row["offset"], row["answer_accuracy"]) for row in points if row["answer_accuracy"] is not None]
    time_ax.plot(*zip(*accuracy), color=SERIES[1], linewidth=2, label="точность ответа Qwen", zorder=3)
    _level(time_ax, levels["chance"], "угадывание", x[-1])
    time_ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
    time_ax.text(0, 1.0, " целевая пара", color=INK_2, fontsize=9, va="top", ha="left",
                 transform=time_ax.get_xaxis_transform())
    time_ax.set_ylim(0, 1.1)
    time_ax.set_yticks(np.linspace(0, 1, 6))
    time_ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    time_ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0, 0.92))
    _style(time_ax, "Уверенность головы и точность ответа во времени",
           "сдвиг относительно целевой пары, пары (1 пара = 0,08 с)")

    rows = [row for row in result["bins"] if row["questions"]]
    conf = np.array([row["confidence"] for row in rows])
    acc = np.array([row["accuracy"] for row in rows])
    bins_ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, zorder=1)
    # Подписи слева: справа вверху лежат точки самых уверенных бинов.
    _level(bins_ax, levels["chance"], "угадывание", 0.0, "left")
    _level(bins_ax, levels["end_of_video"], "точность в конце видео", 0.0, "left")
    bins_ax.errorbar(conf, acc, yerr=[row["accuracy_ci"] for row in rows], color=SERIES[1], linewidth=2,
                     elinewidth=1, capsize=0, zorder=3, label="точность ответа Qwen, вопросы с равным весом")
    span = max(1.0, max(abs(row["offset"]) for row in rows))
    cmap = LinearSegmentedColormap.from_list("offset", [EARLY, NEUTRAL, LATE])
    dots = bins_ax.scatter(conf, acc, c=[row["offset"] for row in rows], cmap=cmap,
                           norm=TwoSlopeNorm(0, -span, span), s=70, edgecolors=INK_2, linewidths=1, zorder=4)
    for row in rows:
        bins_ax.annotate(f"{row['questions']}", (row["confidence"], row["accuracy"]), xytext=(0, -14),
                         textcoords="offset points", ha="center", color=MUTED, fontsize=8)
    colorbar = fig.colorbar(dots, ax=bins_ax, pad=0.01, fraction=0.04)
    colorbar.set_label("средний сдвиг от целевой пары в бине, пары", color=INK_2, fontsize=9)
    colorbar.ax.tick_params(colors=INK_2, labelsize=8)
    colorbar.outline.set_visible(False)
    bins_ax.set_xlim(-0.02, 1.02)
    bins_ax.set_ylim(0, 1.05)
    for axis in (bins_ax.xaxis, bins_ax.yaxis):
        axis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    bins_ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    _style(bins_ax, f"Точность по бинам уверенности (ширина {rows[0]['high'] - rows[0]['low']:.2g}; "
                    "под точкой — число вопросов)", "уверенность головы P(готов), среднее в бине")

    tradeoff = result["tradeoff"]
    signed = tradeoff["signed"]
    _plot_tradeoff(signed_ax, tradeoff, "signed",
                   f"Точность и задержка со знаком; при задержке 0: голова {signed['head_accuracy_at_0']:.3f}, "
                   f"таймер {signed['timer_accuracy_at_0']:.3f}",
                   "средняя задержка ответа относительно целевой пары, пары (1 пара = 0,08 с; < 0 — раньше)")
    _plot_tradeoff(absolute_ax, tradeoff, "absolute", "Точность и задержка от начала видео (вопрос задан на паре 0)",
                   "средняя пара ответа от начала видео (1 пара = 0,08 с)")

    fig.suptitle(title or f"{result['questions']} вопросов", color=INK_2, fontsize=10, x=0.01, ha="left")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="predictions JSON (eval/export.py, uhead.train)")
    parser.add_argument("--output", required=True, help="JSON с кривыми и бинами")
    parser.add_argument("--plot", help="PNG с графиками")
    parser.add_argument("--offsets", default="-30:30", help="диапазон сдвигов относительно целевой пары, a:b")
    parser.add_argument("--bins", type=int, default=10, help="число бинов P(готов) равной ширины")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--records", default=DEFAULT_RECORDS,
                        help="записи вопросов (число вариантов для угадывания), относительно --data-root; "
                             "если файла нет, уровень угадывания не рисуется")
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins: не меньше 2")
    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    records = Path(args.data_root) / args.records
    chance = chance_by_question(records) if records.exists() else None
    if chance is None:
        print(f"Нет {records}: уровень угадывания не рисуется")
    start, stop = (int(v) for v in args.offsets.split(":"))
    result = {"predictions": str(args.predictions), "metadata": payload.get("metadata", {}),
              **confidence_accuracy(load_traces(payload), range(start, stop + 1), args.bins, chance)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in result["bins"]:
        if row["questions"]:
            print(f"P ∈ [{row['low']:.2f}, {row['high']:.2f}): вопросов {row['questions']}, "
                  f"точность {row['accuracy']:.3f} ± {row['accuracy_ci']:.3f}, сдвиг {row['offset']:+.1f}")
    tradeoff = result["tradeoff"]
    for axis, name in (("signed", "задержка от целевой пары"), ("absolute", "задержка от начала видео")):
        metrics = tradeoff[axis]
        print(f"{name}: AUC головы {metrics['head_auc']:.3f}, таймера {metrics['timer_auc']:.3f}, "
              f"ΔAUC {metrics['delta_auc']:+.3f}")
    print(f"точность при задержке 0: голова {tradeoff['signed']['head_accuracy_at_0']:.3f}, "
          f"таймер {tradeoff['signed']['timer_accuracy_at_0']:.3f}")
    if args.plot:
        metadata = result["metadata"]
        title = (f"{result['questions']} вопросов · {metadata.get('model', '')} · {metadata.get('tags', '')} · "
                 f"{metadata.get('split', '')} · {Path(args.predictions).parent.name}")
        plot_confidence_accuracy(result, args.plot, title)


if __name__ == "__main__":
    main()
