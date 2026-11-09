"""Кривые вероятности готовности головы по сдвигу относительно целевой пары, без признаков и без Qwen.

Вход — predictions JSON в формате online_vllm.eval.export (predictions_validation.json из uhead.train,
predictions_test_*.json из uhead.evaluate): на каждый вопрос P(готов) на всех парах видео, target_pair, visible_pair
и, если есть, верность ответа замороженной Qwen на каждой паре (answer_correct).

Что считается:
- curve: P(готов) по сдвигу пары относительно целевой — среднее, медиана, 10-й и 90-й перцентили, и рядом точность
  ответа Qwen на той же паре;
- firing: при каждом τ доля вопросов, где P > τ впервые на данном сдвиге (как в eval/metrics.py), плюс исходы;
- background: до visible_pair при каждом τ — доля вопросов, где P > τ хоть раз (это premature_rate), и средняя доля
  таких пар. Большая первая при маленькой второй — одиночные всплески (поможет гистерезис), обе большие — высокий фон
  (поможет τ выше);
- examples: несколько вопросов с разными исходами при --example-tau;
- by_target: то же по моменту появления ответа (целевой паре) при --example-tau — кривые по квартилям целевой пары
  и сдвиг срабатывания в зависимости от неё. Голова, которая следует за событием, даёт одинаковые кривые и сдвиг
  около 0; голова, выучившая типичное время ответа, — сдвиг с наклоном −1 (поздно на ранних, рано на поздних).

PYTHONPATH=src python -m online_vllm.eval.readiness_plot --predictions results/transformer/19_lm/predictions_validation.json \
    --output results/transformer/19_lm/readiness_curves.json --plot results/transformer/19_lm/readiness_curves.png
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .answers import GRID, INK, INK_2, MUTED, SERIES

OUTCOMES = ("premature", "early", "on_target", "late", "missed")
# Пороги упорядочены, поэтому один оттенок от светлого к тёмному (синяя шкала палитры, шаги 250 / 400 / 550 / 700).
TAU_RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
# Исходы — категории: первые четыре слота категориальной палитры в фиксированном порядке.
OUTCOME_COLORS = {"on_target": "#2a78d6", "premature": "#eb6834", "late": "#1baf7a", "missed": "#eda100",
                  "early": "#e87ba4"}
OUTCOME_NAMES = {"on_target": "в цель", "premature": "преждевременно", "late": "поздно", "missed": "пропуск",
                 "early": "рано"}


@dataclass(frozen=True)
class ReadinessTrace:
    """P(готов) одного вопроса по парам [n], целевая и видимая пара, верность ответа Qwen по парам [n] или None."""
    probabilities: np.ndarray
    target_pair: int
    visible_pair: int
    answer_correct: np.ndarray | None = None
    record_id: str = ""

    def fired(self, tau: float) -> int | None:
        """Первая пара, где P > τ (то же правило, что в eval/metrics.py), или None."""
        hits = np.flatnonzero(self.probabilities > tau)
        return int(hits[0]) if hits.size else None

    def outcome(self, tau: float) -> str:
        fired = self.fired(tau)
        if fired is None:
            return "missed"
        if fired < self.visible_pair:
            return "premature"
        if fired < self.target_pair:
            return "early"
        return "on_target" if fired == self.target_pair else "late"


def load_traces(payload: dict) -> list[ReadinessTrace]:
    """Вопросы из predictions JSON формата eval/export.py."""
    if "predictions" not in payload:
        raise ValueError("Нужен predictions JSON формата online_vllm.eval.export (поле predictions)")
    traces = []
    for row in payload["predictions"]:
        probabilities = np.asarray(row["probabilities"], dtype=np.float64)
        correct = row.get("answer_correct")
        trace = ReadinessTrace(probabilities, int(row["target_pair"]), int(row["visible_pair"]),
                               None if correct is None else np.asarray(correct, dtype=bool), row.get("record_id", ""))
        if not 0 <= trace.visible_pair <= trace.target_pair < len(probabilities):
            raise ValueError(f"{trace.record_id}: нужны 0 <= visible_pair <= target_pair < числа пар")
        if trace.answer_correct is not None and len(trace.answer_correct) != len(probabilities):
            raise ValueError(f"{trace.record_id}: answer_correct и probabilities разной длины")
        traces.append(trace)
    if not traces:
        raise ValueError("Нужен хотя бы один вопрос")
    return traces


def _quantiles(values) -> dict:
    if not values:
        return {"count": 0, "mean": None, "q10": None, "median": None, "q90": None}
    values = np.asarray(values, dtype=np.float64)
    q10, median, q90 = np.quantile(values, [0.1, 0.5, 0.9])
    return {"count": len(values), "mean": float(values.mean()), "q10": float(q10), "median": float(median),
            "q90": float(q90)}


def _curve(traces: list[ReadinessTrace], offsets: list[int]) -> list[dict]:
    """P(готов) и точность ответа Qwen по сдвигу пары относительно целевой."""
    curve = []
    for d in offsets:
        at = [(t, t.target_pair + d) for t in traces if 0 <= t.target_pair + d < len(t.probabilities)]
        answered = [t.answer_correct[p] for t, p in at if t.answer_correct is not None]
        curve.append({"offset": d, **_quantiles([t.probabilities[p] for t, p in at]),
                      "answer_accuracy": float(np.mean(answered)) if answered else None})
    return curve


def by_target_time(traces: list[ReadinessTrace], offsets: list[int], tau: float, groups: int = 4,
                   bin_width: int = 4) -> dict:
    """Зависимость от момента появления ответа (целевой пары).

    groups — вопросы по квантилям целевой пары: кривая P(готов) и исходы при τ в каждой группе. Если голова ловит
    событие, кривые по сдвигу совпадают. shift — сдвиг срабатывания при τ в зависимости от целевой пары (корзины
    по bin_width пар): наклон 0 значит, что голова следует за событием, наклон −1 — что срабатывает в одно и то же
    время видео независимо от события."""
    targets = np.array([t.target_pair for t in traces])
    edges = np.unique(np.quantile(targets, np.linspace(0, 1, groups + 1)[1:-1]))
    group_of = np.searchsorted(edges, targets, side="right")
    result = {"tau": tau, "groups": [], "shift": []}
    for g in range(len(edges) + 1):
        members = [t for t, k in zip(traces, group_of) if k == g]
        if not members:
            continue
        outcomes = [t.outcome(tau) for t in members]
        shifts = [f - t.target_pair for t in members if (f := t.fired(tau)) is not None]
        result["groups"].append({"first_target": int(min(t.target_pair for t in members)),
                                 "last_target": int(max(t.target_pair for t in members)), "count": len(members),
                                 "outcomes": {o: outcomes.count(o) / len(members) for o in OUTCOMES},
                                 "shift": _quantiles(shifts), "curve": _curve(members, offsets)})
    for start in range(int(targets.min()) // bin_width * bin_width, int(targets.max()) + 1, bin_width):
        members = [t for t in traces if start <= t.target_pair < start + bin_width]
        if members:
            shifts = [f - t.target_pair for t in members if (f := t.fired(tau)) is not None]
            result["shift"].append({"first_target": start, "last_target": start + bin_width - 1,
                                    "center": float(np.mean([t.target_pair for t in members])),
                                    "missed": 1 - len(shifts) / len(members), **_quantiles(shifts)})
    return result


def readiness_curves(traces: list[ReadinessTrace], offsets=range(-30, 31), taus=(0.5, 0.7, 0.9),
                     example_tau: float = 0.5, examples: int = 4, seed: int = 0) -> dict:
    offsets = list(offsets)
    curve = _curve(traces, offsets)

    n = len(traces)
    firing = {}
    for tau in taus:
        shifts = np.array([f - t.target_pair for t in traces if (f := t.fired(tau)) is not None], dtype=int)
        outcomes = [t.outcome(tau) for t in traces]
        firing[str(tau)] = {"share": [float((shifts == d).sum() / n) for d in offsets],
                            "before_range": float((shifts < offsets[0]).sum() / n),
                            "after_range": float((shifts > offsets[-1]).sum() / n),
                            "outcomes": {o: outcomes.count(o) / n for o in OUTCOMES}}

    # До visible_pair ответа в видео ещё нет; вопросы, где он виден уже на первой паре, сюда не входят.
    before = [t.probabilities[:t.visible_pair] for t in traces if t.visible_pair >= 1]
    background = [{"tau": float(tau), "any_pair": float(np.mean([b.max() > tau for b in before])),
                   "pair_share": float(np.mean([(b > tau).mean() for b in before]))}
                  for tau in np.round(np.arange(0.05, 1.0, 0.05), 2)] if before else []

    rng = np.random.default_rng(seed)
    chosen = []
    for outcome in ("on_target", "premature", "late", "missed", "early"):
        pool = [t for t in traces if t.outcome(example_tau) == outcome]
        if pool and len(chosen) < examples:
            t = pool[rng.integers(len(pool))]
            chosen.append({"record_id": t.record_id, "outcome": outcome, "target_pair": t.target_pair,
                           "visible_pair": t.visible_pair, "probabilities": t.probabilities.tolist()})

    return {"questions": n, "mean_visible_offset": float(np.mean([t.visible_pair - t.target_pair for t in traces])),
            "max_before_visible": _quantiles([float(b.max()) for b in before]),
            "curve": curve, "firing": firing, "background": background,
            "example_tau": example_tau, "examples": chosen,
            "by_target": by_target_time(traces, offsets, example_tau)}


def _style(ax, title: str, xlabel: str):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)


def _target_lines(ax, visible_offset: float):
    ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
    ax.text(0, 1.0, " целевая пара", color=INK_2, fontsize=9, va="top", ha="left", transform=ax.get_xaxis_transform())
    if visible_offset < -0.5:
        ax.axvline(visible_offset, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(visible_offset, 1.0, "ответ виден ", color=INK_2, fontsize=9, va="top", ha="right",
                transform=ax.get_xaxis_transform())


def plot_readiness(result: dict, path: str | Path, title: str | None = None):
    """PNG из шести графиков: P(готов) по сдвигу, первое срабатывание при τ, фон до ответа, примеры вопросов,
    P(готов) по группам момента появления ответа и сдвиг срабатывания в зависимости от этого момента."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    xlabel = "сдвиг относительно целевой пары, пары (1 пара = 0,08 с)"
    visible_offset = result["mean_visible_offset"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12.9), constrained_layout=True)
    (curve_ax, firing_ax), (background_ax, examples_ax), (groups_ax, shift_ax) = axes

    points = [row for row in result["curve"] if row["count"]]
    x = [row["offset"] for row in points]
    curve_ax.fill_between(x, [row["q10"] for row in points], [row["q90"] for row in points], color=SERIES[0],
                          alpha=0.18, linewidth=0, label="P(готов), 10–90-й перцентили", zorder=2)
    curve_ax.plot(x, [row["median"] for row in points], color=SERIES[0], linewidth=2, label="P(готов), медиана",
                  zorder=3)
    curve_ax.plot(x, [row["mean"] for row in points], color=SERIES[0], linewidth=1.2, linestyle=(0, (4, 3)),
                  label="P(готов), среднее", zorder=3)
    accuracy = [(row["offset"], row["answer_accuracy"]) for row in points if row["answer_accuracy"] is not None]
    if accuracy:
        curve_ax.plot(*zip(*accuracy), color=SERIES[1], linewidth=2, label="точность ответа Qwen", zorder=3)
    _target_lines(curve_ax, visible_offset)
    # Запас над 100%, чтобы подписи вертикалей не ложились на кривые.
    curve_ax.set_ylim(0, 1.1)
    curve_ax.set_yticks(np.linspace(0, 1, 6))
    curve_ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    curve_ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0, 0.92))
    _style(curve_ax, "Вероятность готовности головы", xlabel)

    offsets = [row["offset"] for row in result["curve"]]
    for (tau, values), color in zip(result["firing"].items(), TAU_RAMP):
        outcomes = values["outcomes"]
        label = (f"τ = {tau}\nв цель {outcomes['on_target']:.0%}, рано {outcomes['early']:.0%}\n"
                 f"преждевр. {outcomes['premature']:.0%}\nпоздно {outcomes['late']:.0%}, "
                 f"пропуск {outcomes['missed']:.0%}")
        firing_ax.plot(offsets, values["share"], color=color, linewidth=2, label=label, zorder=3)
    _target_lines(firing_ax, visible_offset)
    top = max([max(values["share"]) for values in result["firing"].values()] + [0.01])
    firing_ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0 if top >= 0.1 else 1))
    firing_ax.set_ylim(0, top * 1.15)
    firing_ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0, 0.92))
    _style(firing_ax, "Где P впервые превышает τ (доля всех вопросов)", xlabel)

    if result["background"]:
        taus = [row["tau"] for row in result["background"]]
        background_ax.plot(taus, [row["any_pair"] for row in result["background"]], color=SERIES[1], linewidth=2,
                           marker="o", markersize=4, markeredgecolor="white", markeredgewidth=1,
                           label="вопросов, где P > τ хоть раз (= premature)", zorder=3)
        background_ax.plot(taus, [row["pair_share"] for row in result["background"]], color=SERIES[0],
                           linewidth=2, marker="o", markersize=4, markeredgecolor="white", markeredgewidth=1,
                           label="пар с P > τ, средняя доля по вопросам", zorder=3)
        background_ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    background_ax.set_ylim(0, 1.02)
    background_ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _style(background_ax, "До появления ответа: всплески или фон", "порог τ")

    for example in result["examples"]:
        p = np.asarray(example["probabilities"])
        shift = np.arange(len(p)) - example["target_pair"]
        color = OUTCOME_COLORS[example["outcome"]]
        examples_ax.plot(shift, p, color=color, linewidth=2, label=OUTCOME_NAMES[example["outcome"]], zorder=3)
        if example["visible_pair"] < example["target_pair"]:
            examples_ax.plot(example["visible_pair"] - example["target_pair"], p[example["visible_pair"]], "o",
                             color=color, markersize=8, markeredgecolor="white", markeredgewidth=2, zorder=4)
    examples_ax.axhline(result["example_tau"], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    examples_ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
    examples_ax.set_ylim(0, 1.02)
    examples_ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if result["examples"]:
        examples_ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    _style(examples_ax, f"Отдельные вопросы, исход при τ = {result['example_tau']} (пунктир)", xlabel)

    by_target = result["by_target"]
    # Группы упорядочены по времени, поэтому та же однотонная шкала, что у порогов.
    for group, color in zip(by_target["groups"], TAU_RAMP[-len(by_target["groups"]):]):
        points = [row for row in group["curve"] if row["count"]]
        outcomes = group["outcomes"]
        label = (f"цель на парах {group['first_target']}–{group['last_target']} (n = {group['count']})\n"
                 f"в цель {outcomes['on_target']:.0%}, преждевр. {outcomes['premature']:.0%}, "
                 f"поздно {outcomes['late']:.0%}")
        groups_ax.plot([row["offset"] for row in points], [row["median"] for row in points], color=color,
                       linewidth=2, label=label, zorder=3)
    _target_lines(groups_ax, visible_offset)
    groups_ax.set_ylim(0, 1.1)
    groups_ax.set_yticks(np.linspace(0, 1, 6))
    groups_ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    groups_ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0, 0.92))
    _style(groups_ax, f"Медиана P(готов) по моменту появления ответа; исходы при τ = {by_target['tau']}", xlabel)

    rows = [row for row in by_target["shift"] if row["count"]]
    if rows:
        x = [row["center"] for row in rows]
        shift_ax.fill_between(x, [row["q10"] for row in rows], [row["q90"] for row in rows], color=SERIES[0],
                              alpha=0.18, linewidth=0, label="10–90-й перцентили", zorder=2)
        shift_ax.plot(x, [row["median"] for row in rows], color=SERIES[0], linewidth=2, marker="o", markersize=5,
                      markeredgecolor="white", markeredgewidth=1, label="медиана", zorder=3)
        # Опора «срабатывает в одно и то же время видео»: наклон −1 через медиану по всем вопросам.
        all_x, all_median = np.average(x, weights=[row["count"] for row in rows]), np.median(
            [row["median"] for row in rows])
        shift_ax.plot(x, [all_median - (v - all_x) for v in x], color=MUTED, linewidth=1, linestyle=(0, (4, 3)),
                      label="если бы срабатывала в одно время видео", zorder=2)
        shift_ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    shift_ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
    _style(shift_ax, f"Сдвиг срабатывания (τ = {by_target['tau']}) от момента появления ответа",
           "целевая пара (момент появления ответа + speak_delay), пары от начала видео")
    shift_ax.set_ylabel("срабатывание − цель, пары", color=INK_2, fontsize=9)

    fig.suptitle(title or f"{result['questions']} вопросов", color=INK_2, fontsize=10, x=0.01, ha="left")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="predictions JSON (eval/export.py, uhead.train)")
    parser.add_argument("--output", required=True, help="JSON с кривыми")
    parser.add_argument("--plot", help="PNG с графиками")
    parser.add_argument("--offsets", default="-30:30", help="диапазон сдвигов относительно целевой пары, a:b")
    parser.add_argument("--taus", type=float, nargs="+", default=[0.5, 0.7, 0.9], help="до четырёх порогов")
    parser.add_argument("--example-tau", type=float, default=0.5)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="выбор примеров")
    args = parser.parse_args()
    if not 1 <= len(args.taus) <= len(TAU_RAMP) or any(not 0 < tau < 1 for tau in args.taus):
        parser.error(f"--taus: от 1 до {len(TAU_RAMP)} порогов в (0, 1)")
    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    start, stop = (int(v) for v in args.offsets.split(":"))
    result = {"predictions": str(args.predictions), "metadata": payload.get("metadata", {}),
              **readiness_curves(load_traces(payload), range(start, stop + 1), sorted(args.taus), args.example_tau,
                                 args.examples, args.seed)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for tau, values in result["firing"].items():
        print(f"τ = {tau}: " + ", ".join(f"{o} {share:.3f}" for o, share in values["outcomes"].items()))
    if args.plot:
        metadata = result["metadata"]
        parts = [metadata.get("model"), "+".join(metadata.get("feature_groups") or []), metadata.get("split"),
                 f"{result['questions']} вопросов"]
        plot_readiness(result, args.plot, f"{Path(args.predictions).parent.name}: " + ", ".join(p for p in parts if p))


if __name__ == "__main__":
    main()
