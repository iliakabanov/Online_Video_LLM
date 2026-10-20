"""Ответы замороженной Qwen по пробе «ответь сейчас» и их качество во времени, без загрузки Qwen.

Проба пары p (train/FEATURES.md) — состояние, из которого модель начала бы отвечать, если бы видео оборвалось на
паре p; probe_logprob хранит log P букв вариантов по всему словарю. Ответ модели на паре p — буква с наибольшей
вероятностью среди вариантов вопроса; вероятности нормируются по вариантам.

От головы готовности модуль не зависит: он даёт опорные уровни (случайное угадывание, ответ перед появлением
ответа в видео, на целевой паре, в конце видео) и кривые по сдвигу относительно целевой пары. Точность на паре
срабатывания головы по тем же ответам считает eval/metrics.py (evaluate_readiness, аргумент correct).

PYTHONPATH=src python -m online_vllm.eval.answers --features $DATA_ROOT/features/group1_s1_252x168/val_first2000 \
    --output results/answers/val_first2000.json --plot results/answers/val_first2000.png
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def choice_probabilities(raw: np.ndarray, num_choices: int) -> np.ndarray:
    """Распределение по вариантам [n, num_choices] из probe_logprob [n, max_choices + ...] (softmax log P букв)."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or not 2 <= num_choices <= raw.shape[1]:
        raise ValueError("probe_logprob должен иметь форму [n, >= num_choices], а вариантов должно быть хотя бы два")
    letters = raw[:, :num_choices].astype(np.float64)
    if not np.isfinite(letters).all():
        raise ValueError("log P букв вариантов должны быть конечными")
    weights = np.exp(letters - letters.max(1, keepdims=True))
    return weights / weights.sum(1, keepdims=True)


def correct_choice(record) -> int:
    """Номер правильного варианта: ответ первого (в группе 1 — единственного) интервала."""
    return record.choices.index(record.answers[0].answer)


@dataclass(frozen=True)
class AnswerTrace:
    """Ответы модели одного вопроса по парам: вероятности вариантов [n, k] и номер правильного варианта."""
    probabilities: np.ndarray
    correct: int
    target_pair: int = -1
    visible_pair: int = -1

    @property
    def chosen(self) -> np.ndarray:
        return self.probabilities.argmax(1)

    @property
    def is_correct(self) -> np.ndarray:
        return self.chosen == self.correct


def answer_trace(fs, index: int, n: int | None = None, target_pair: int = -1, visible_pair: int = -1) -> AnswerTrace:
    """Ответы модели на первых n парах вопроса index набора признаков fs (train/features.py FeatureSet)."""
    record = fs.records[index]
    n = fs.patches[index] if n is None else n
    raw = np.asarray(fs.question("probe_logprob", index)[:n])
    return AnswerTrace(choice_probabilities(raw, len(record.choices)), correct_choice(record), target_pair,
                       visible_pair)


def _stats(points) -> dict:
    """Точность, средние P(правильный) и P(текущий ответ) и уровень угадывания по парам (trace, pair)."""
    if not points:
        return {"count": 0, "accuracy": None, "p_correct": None, "p_answer": None, "chance": None}
    return {"count": len(points),
            "accuracy": float(np.mean([t.is_correct[p] for t, p in points])),
            "p_correct": float(np.mean([t.probabilities[p, t.correct] for t, p in points])),
            "p_answer": float(np.mean([t.probabilities[p].max() for t, p in points])),
            "chance": float(np.mean([1 / t.probabilities.shape[1] for t, _ in points]))}


def answer_curves(traces: list[AnswerTrace], offsets=range(-10, 21)) -> dict:
    """Опорные уровни и кривые по сдвигу пары относительно целевой; у каждого вопроса заданы target и visible."""
    if not traces:
        raise ValueError("Нужен хотя бы один вопрос")
    for t in traces:
        if not 0 <= t.visible_pair <= t.target_pair < len(t.probabilities):
            raise ValueError("Нужны 0 <= visible_pair <= target_pair < числа пар")
    levels = {
        "chance": float(np.mean([1 / t.probabilities.shape[1] for t in traces])),
        # Только первая пара видео (начальная расстановка, без движения); вопросы, где ответ виден уже на ней, не входят.
        "first_pair": _stats([(t, 0) for t in traces if t.visible_pair >= 1]),
        "before_visible": _stats([(t, t.visible_pair - 1) for t in traces if t.visible_pair >= 1]),
        "at_target": _stats([(t, t.target_pair) for t in traces]),
        "at_end": _stats([(t, len(t.probabilities) - 1) for t in traces]),
    }
    curve = [{"offset": int(d), **_stats([(t, t.target_pair + d) for t in traces
                                          if 0 <= t.target_pair + d < len(t.probabilities)])} for d in offsets]
    return {"questions": len(traces), "mean_visible_offset": float(np.mean([t.visible_pair - t.target_pair
                                                                            for t in traces])),
            "levels": levels, "curve": curve}


INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ("#2a78d6", "#eb6834")  # первые два слота категориальной палитры


def plot_curves(result: dict, path: str | Path, title: str | None = None):
    """PNG из двух графиков по сдвигу относительно целевой пары: точность и вероятности ответа."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    points = [row for row in result["curve"] if row["count"]]
    x = [row["offset"] for row in points]
    chance = result["levels"]["chance"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    panels = (("Точность ответа Qwen", [("accuracy", "доля верных ответов")]),
              ("Вероятности ответа Qwen", [("p_correct", "P(правильный ответ)"), ("p_answer", "P(текущий ответ)")]))
    for ax, (panel_title, series) in zip(axes, panels):
        for (key, label), color in zip(series, SERIES):
            ax.plot(x, [row[key] for row in points], color=color, linewidth=2, marker="o", markersize=4,
                    markeredgecolor="white", markeredgewidth=1, solid_capstyle="round", label=label, zorder=3)
        ax.axhline(chance, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.text(x[-1], chance, " угадывание", color=INK_2, fontsize=9, va="bottom", ha="right")
        ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
        ax.text(0, 1.0, " целевая пара", color=INK_2, fontsize=9, va="top", ha="left")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("сдвиг относительно целевой пары, пары (1 пара = 0,08 с)", color=INK_2, fontsize=9)
        ax.set_title(panel_title, color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK_2, labelsize=9)
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.suptitle(title or f"{result['questions']} вопросов", color=INK_2, fontsize=10, x=0.01, ha="left")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def traces_from_features(fs, config, videos: set[str] | None = None) -> list[AnswerTrace]:
    """Ответы по всем вопросам набора признаков с целевой и видимой парой по разметке конфига (data/labels.py)."""
    from ..data.labels import plan
    traces = []
    for index, record in enumerate(fs.records):
        if videos is not None and record.video not in videos:
            continue
        state_plan = plan(record, config.video.frame_stride, config.labels.speak_delay, fs.meta["temporal_patch"])
        if state_plan is None:
            continue
        target = len(state_plan.labels) - 1
        visible = next(j for j, label in enumerate(state_plan.labels) if label != "silent")
        traces.append(answer_trace(fs, index, None, target, visible))
    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, help="каталог FeatureSet (extract_features.py)")
    parser.add_argument("--train", default="configs/train/group1.yaml", help="конфиг с labels и video")
    parser.add_argument("--split", help="split.json запуска uhead.train, чтобы взять только одну половину eval")
    parser.add_argument("--part", choices=("validation", "test"), help="какую половину взять из --split")
    parser.add_argument("--offsets", default="-10:20", help="диапазон сдвигов относительно целевой пары, a:b")
    parser.add_argument("--output", required=True, help="JSON с уровнями и кривыми")
    parser.add_argument("--plot", help="PNG с графиками точности и вероятностей")
    args = parser.parse_args()
    if bool(args.split) != bool(args.part):
        parser.error("--split и --part задаются вместе")
    from ..train.config import TrainConfig
    from ..train.features import FeatureSet
    config = TrainConfig.load(args.train)
    fs = FeatureSet(args.features)
    fs.check(config.video)
    videos = None
    if args.split:
        videos = set(json.loads(Path(args.split).read_text(encoding="utf-8"))[f"{args.part}_videos"])
    start, stop = (int(v) for v in args.offsets.split(":"))
    result = {"features": str(args.features), "part": args.part or "all",
              "speak_delay": config.labels.speak_delay, "frame_stride": config.video.frame_stride,
              **answer_curves(traces_from_features(fs, config, videos), range(start, stop + 1))}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    levels = result["levels"]
    fmt = lambda level: "—" if level["accuracy"] is None else f"{level['accuracy']:.3f} (n = {level['count']})"
    print(f"{result['questions']} вопросов; угадывание {levels['chance']:.3f}; точность на первой паре "
          f"{fmt(levels['first_pair'])}, перед появлением ответа {fmt(levels['before_visible'])}, "
          f"на целевой паре {fmt(levels['at_target'])}, в конце видео {fmt(levels['at_end'])}")
    if args.plot:
        plot_curves(result, args.plot, f"{Path(args.features).name}: {result['questions']} вопросов, "
                                       f"speak_delay = {config.labels.speak_delay}")


if __name__ == "__main__":
    main()
