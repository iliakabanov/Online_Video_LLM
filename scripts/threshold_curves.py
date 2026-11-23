"""Как от порога τ детектора готовности зависят точность ответа, доля преждевременных ответов и задержка — три
отдельных графика по сохранённым предсказаниям одного чекпоинта (формат eval/export.py, нужен answer_correct).

Правило ответа то же, что в eval/metrics.py operating_point: вопрос отвечается на первой паре k с p_k > τ, а если
такой нет — принудительно на последней паре. Преждевременный ответ — k < t*. Задержка k − t* и точность
непреждевременных ответов считаются по ответам с k >= t* (принудительные входят). С --validation на графиках отмечены
пороги, выбранные на validation: по on_time_0_4 и с наименьшей задержкой при R <= 5 %, A >= 0.85, — и печатаются
метрики test при них.

python scripts/threshold_curves.py \\
    --test results/lora/mlp/lm_attention_entropy_vision_confidence_mlp_off10/predictions_test_best_tau0.8.json \\
    --validation results/lora/mlp/lm_attention_entropy_vision_confidence_mlp_off10/predictions_validation.json \\
    --output-dir results/lora/threshold_curves/all_groups_off10 --lang en --unit frames
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (src/ on the path)
from online_vllm.eval.metrics import best_window_threshold, constrained_latency, operating_point
from online_vllm.eval.thresholds import payload_arrays

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ("#2a78d6", "#eb6834")  # первые два слота категориальной палитры (как в eval/answers.py)
UNITS = ("pairs", "frames", "seconds")
MIN_TIMELY = 0.01  # доля непреждевременных ответов, ниже которой их точность и задержка не рисуются
TEXT = {
    "en": {"tau": "threshold τ", "accuracy": "Answer accuracy", "acc_timely": "non-premature answers",
           "acc_all": "all answers", "premature": "Premature answers (before t*)",
           "latency": "Delay of non-premature answers", "mean": "mean", "median": "median",
           "unit": {"pairs": "steps (1 step = 2 frames)", "frames": "frames", "seconds": "seconds"},
           "on_time": "τ on-time (val)", "budget": "τ R ≤ 5% (val)", "share": "share of questions",
           "level": "A = {:.0%}", "R": "R = {:.0%}"},
    "ru": {"tau": "порог τ", "accuracy": "Точность ответа", "acc_timely": "непреждевременные ответы",
           "acc_all": "все ответы", "premature": "Преждевременные ответы (до t*)",
           "latency": "Задержка непреждевременных ответов", "mean": "среднее", "median": "медиана",
           "unit": {"pairs": "пары (1 пара = 2 кадра)", "frames": "кадры", "seconds": "секунды"},
           "on_time": "τ вовремя (val)", "budget": "τ R ≤ 5% (val)", "share": "доля вопросов",
           "level": "A = {:.0%}", "R": "R = {:.0%}"},
}


def curves(arrays: dict, taus: np.ndarray) -> dict:
    """Метрики при каждом τ из taus: пара ответа вопроса — первая k с running max p_k > τ (searchsorted по
    неубывающему running max), иначе последняя пара."""
    probabilities, mask, targets = arrays["probabilities"], arrays["valid_mask"], np.asarray(arrays["target_pairs"])
    fired = np.empty((len(targets), len(taus)), dtype=np.int64)
    right = np.empty_like(fired, dtype=bool)
    forced = np.empty_like(right)
    for i, (p, valid, c) in enumerate(zip(probabilities, mask, arrays["correct"])):
        n = int(valid.sum())
        run = np.maximum.accumulate(p[:n])
        k = np.searchsorted(run, taus, side="right")
        forced[i] = k >= n
        fired[i] = np.minimum(k, n - 1)
        right[i] = np.asarray(c[:n], dtype=bool)[fired[i]]
    timely = fired >= targets[:, None]
    delay = np.where(timely, fired - targets[:, None], np.nan).astype(float)
    count = timely.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # τ, при которых все ответы преждевременные: NaN
        return {"tau": taus, "premature_rate": 1 - count / len(targets),
                "accuracy_timely": np.where(count > 0, (right & timely).sum(0) / count, np.nan),
                "accuracy_all": right.mean(0), "forced_rate": forced.mean(0),
                "latency_mean": np.nanmean(delay, 0), "latency_median": np.nanmedian(delay, 0)}


def grid(arrays: dict, points: int) -> np.ndarray:
    """τ, равномерные в логит-шкале от ~наименьшей до наибольшей оценки (детекторы прижимают p_k к 0 и 1)."""
    values = arrays["probabilities"][arrays["valid_mask"]]
    low = float(np.clip(np.quantile(values, 0.001), 1e-6, 0.5))
    high = float(np.clip(values.max(), 0.5, 1 - 1e-7))
    logit = lambda x: np.log(x / (1 - x))
    return 1 / (1 + np.exp(-np.linspace(logit(low), logit(high), points)))


def chosen_taus(validation: dict) -> dict:
    """Пороги, выбранные на validation, как в scripts/test_report.py."""
    arrays = payload_arrays(validation)
    p, targets, mask, correct = arrays["probabilities"], arrays["target_pairs"], arrays["valid_mask"], arrays["correct"]
    result = {"on_time": best_window_threshold(p, targets, mask, (0, 4))["tau"]}
    budget = constrained_latency(p, targets, mask, correct, budgets=(0.05,), levels=(0.85,))["R0.05_A0.85"]
    if budget["feasible"]:
        result["budget"] = budget["tau"]
    return result


def _axes(text: dict, title: str, ylabel: str, suptitle: str | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.4), constrained_layout=True)
    ax.set_xscale("logit")
    ax.set_xlabel(text["tau"], color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    if suptitle:
        fig.suptitle(suptitle, color=INK_2, fontsize=10, x=0.01, ha="left")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    return fig, ax


# Деления логит-шкалы: подписи матplotlib (1 − 10⁻ⁿ) налезают друг на друга.
TICKS = ((1e-6, "10⁻⁶"), (1e-4, "10⁻⁴"), (1e-2, "0.01"), (0.5, "0.5"), (1 - 1e-2, "0.99"), (1 - 1e-4, "0.9999"),
         (1 - 1e-6, "1 − 10⁻⁶"))


def _ticks(ax, taus: np.ndarray) -> None:
    from matplotlib.ticker import NullFormatter, NullLocator
    low, high = float(taus[0]), float(taus[-1])
    ax.set_xlim(low, high)
    ticks = [(value, label) for value, label in TICKS if low <= value <= high]
    ax.set_xticks([value for value, _ in ticks], [label for _, label in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def _reference(ax, value: float, label: str) -> None:
    """Горизонтальная опорная линия (ограничение R или A) с подписью справа."""
    ax.axhline(value, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(ax.get_xlim()[1], value, label + " ", color=INK_2, fontsize=8, va="bottom", ha="right")


def _markers(ax, marks: dict, text: dict, where: str = "top") -> None:
    """Вертикальные линии порогов с validation; подпись вдоль линии, справа от неё, сверху или снизу графика."""
    y, va = (0.98, "top") if where == "top" else (0.02, "bottom")
    for key, tau in marks.items():
        ax.axvline(tau, color=INK_2, linewidth=1, linestyle=(0, (1, 2)), zorder=1)
        ax.annotate(text[key], xy=(tau, y), xycoords=ax.get_xaxis_transform(), xytext=(3, 0),
                    textcoords="offset points", rotation=90, color=INK_2, fontsize=8, va=va, ha="left")


def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot(result: dict, marks: dict, out: Path, lang: str, unit: str, scale: float, suptitle: str | None) -> list:
    from matplotlib.ticker import PercentFormatter
    text = TEXT[lang]
    tau = result["tau"]
    # Метрики непреждевременных ответов — только там, где их хотя бы MIN_TIMELY вопросов: иначе это шум единиц ответов.
    sparse = 1 - result["premature_rate"] < MIN_TIMELY
    hide = lambda values: np.where(sparse, np.nan, values)
    paths = []

    fig, ax = _axes(text, text["accuracy"], text["share"], suptitle)
    for key, color in (("accuracy_timely", SERIES[0]), ("accuracy_all", SERIES[1])):
        values = hide(result[key]) if key == "accuracy_timely" else result[key]
        ax.plot(tau, values, color=color, linewidth=2, label=text[key.replace("accuracy_", "acc_")], zorder=3)
    _ticks(ax, tau)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _reference(ax, 0.85, text["level"].format(0.85))
    _markers(ax, marks, text, "bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower left")
    paths.append(out / "accuracy.png"); _save(fig, paths[-1])

    fig, ax = _axes(text, text["premature"], text["share"], suptitle)
    ax.plot(tau, result["premature_rate"], color=SERIES[0], linewidth=2, zorder=3)
    _ticks(ax, tau)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for budget in (0.05, 0.10):
        _reference(ax, budget, text["R"].format(budget))
    _markers(ax, marks, text)
    paths.append(out / "premature.png"); _save(fig, paths[-1])

    fig, ax = _axes(text, text["latency"], text["unit"][unit], suptitle)
    for key, color in (("latency_mean", SERIES[0]), ("latency_median", SERIES[1])):
        ax.plot(tau, scale * hide(result[key]), color=color, linewidth=2, label=text[key.replace("latency_", "")],
                zorder=3)
    _ticks(ax, tau)
    ax.set_ylim(0, None)
    _markers(ax, marks, text)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    paths.append(out / "latency.png"); _save(fig, paths[-1])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", required=True, help="предсказания test (predictions_test_*.json) с answer_correct")
    parser.add_argument("--validation", help="предсказания validation того же чекпоинта: отметить выбранные на ней τ")
    parser.add_argument("--output-dir", help="каталог PNG и curves.json; по умолчанию <каталог --test>/threshold_curves")
    parser.add_argument("--lang", choices=tuple(TEXT), default="en")
    parser.add_argument("--unit", choices=UNITS, default="frames", help="единицы задержки")
    parser.add_argument("--points", type=int, default=600, help="число порогов на кривой")
    parser.add_argument("--title", help="подпись над графиками (например, имя чекпоинта)")
    args = parser.parse_args()

    test = json.loads(Path(args.test).read_text(encoding="utf-8"))
    arrays = payload_arrays(test)
    if arrays["correct"] is None:
        parser.error("В предсказаниях --test нет answer_correct: точность и задержку не посчитать")
    metadata = test.get("metadata", {})
    frames_per_pair = int(metadata.get("temporal_patch", 2)) * int(metadata.get("frame_stride", 1))
    fps = float(np.mean([row.get("fps", 25.0) for row in test["predictions"]]))
    scale = {"pairs": 1.0, "frames": float(frames_per_pair), "seconds": frames_per_pair / fps}[args.unit]

    result = curves(arrays, grid(arrays, args.points))
    marks = chosen_taus(json.loads(Path(args.validation).read_text(encoding="utf-8"))) if args.validation else {}
    out = Path(args.output_dir) if args.output_dir else Path(args.test).parent / "threshold_curves"
    out.mkdir(parents=True, exist_ok=True)
    title = args.title or f"{Path(args.test).parent.name}, test: {len(arrays['target_pairs'])} questions"
    paths = plot(result, marks, out, args.lang, args.unit, scale, title)

    operating = {key: operating_point(arrays["probabilities"], arrays["target_pairs"], arrays["valid_mask"],
                                      arrays["correct"], tau) for key, tau in marks.items()}
    payload = {"test": str(args.test), "validation": args.validation, "questions": len(arrays["target_pairs"]),
               "latency_unit": "pairs", "frames_per_pair": frames_per_pair, "fps": fps,
               "curves": {key: [None if np.isnan(v) else float(v) for v in values] for key, values in result.items()},
               "validation_taus": marks, "test_at_validation_taus": operating}
    (out / "curves.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8")
    for key, point in operating.items():
        latency = "-" if point["latency_mean"] is None else f"{scale * point['latency_mean']:.2f}"
        print(f"{key}: τ = {point['tau']:.6g} -> test: преждевременных {100 * point['premature_rate']:.1f}%, "
              f"точность непреждевременных {100 * point['accuracy']:.1f}%, задержка {latency} {args.unit}, "
              f"принудительных {100 * point['forced_rate']:.1f}%")
    print("-> " + ", ".join(str(path) for path in paths) + f", {out / 'curves.json'}")


if __name__ == "__main__":
    main()
