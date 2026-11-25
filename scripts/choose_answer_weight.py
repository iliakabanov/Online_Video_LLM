"""Выбор веса лосса ответа онлайн-SFT (scripts/train_online_sft_weight.py) по validation.

Эпоха каждого прогона — его best/ по bce_pm2 (best_epoch последней строки validation_metrics.jsonl). Вес — по
наименьшей средней задержке при доле преждевременных <= 5% и точности >= 90% (R0.05_A0.9) у этой эпохи. Для страховки
печатается та же задержка на каждой эпохе: насколько эпоха по bce_pm2 отстаёт от лучшей по задержке. Задержка
пересчитывается из predictions_validation_epoch_<n>.json без GPU, поэтому работает и для старых прогонов, которые
логировали только A 0.85 / 0.88.

Различимы ли веса: 95% bootstrap-интервал задержки эпохи best/ (как в scripts/test_bootstrap.py: видео validation с
возвращением, видео входит со всеми вопросами, порог τ фиксирован — выбранный на всей validation; перцентильный
интервал). Выборки видео одни и те же для всех прогонов, если у них одна validation.

python scripts/choose_answer_weight.py outputs/listen_sft_10ep outputs/listen_sft_10ep_aw0.5 \\
    outputs/listen_sft_10ep_aw0.25 outputs/listen_sft_10ep_aw0.1 --output outputs/answer_weight_choice.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (src/ on the path)
from online_vllm.eval.metrics import constrained_latency
from online_vllm.eval.thresholds import payload_arrays
from test_bootstrap import first_fire, video_of

KEYS = ("R0.05_A0.9", "R0.1_A0.9")


def load(run: Path, epoch: int) -> dict:
    return json.loads((run / f"predictions_validation_epoch_{epoch}.json").read_text(encoding="utf-8"))


def latency(run: Path, payload: dict, epoch: int) -> dict:
    arrays = payload_arrays(payload)
    if arrays["correct"] is None:
        sys.exit(f"{run}: в predictions_validation_epoch_{epoch}.json нет answer_correct")
    result = constrained_latency(arrays["probabilities"], arrays["target_pairs"], arrays["valid_mask"],
                                 arrays["correct"], budgets=(0.05, 0.10), levels=(0.90,), points=False)
    return {key: result[key] for key in KEYS}


def per_question(payload: dict, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """При пороге τ: непреждевременный ли ответ (первая пара с p_k > τ, иначе последняя) и его задержка k − t*."""
    timely, delay = [], []
    for row in payload["predictions"]:
        k, _ = first_fire(row["probabilities"], len(row["probabilities"]), tau)
        timely.append(k >= row["target_pair"])
        delay.append(max(k - row["target_pair"], 0))
    return np.asarray(timely, dtype=float), np.asarray(delay, dtype=float)


def show(point: dict) -> str:
    if not point["feasible"]:
        return f"{'infeasible':>22s}"
    return f"{2 * point['latency_mean']:5.2f} fr (P {100 * point['premature_rate']:4.1f}%, A {100 * point['accuracy']:4.1f}%)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="JSON со сводкой")
    parser.add_argument("--resamples", type=int, default=2000, help="выборок видео для интервалов (0 — без них)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary, best_payloads = {}, {}
    for run in args.runs:
        rows = [json.loads(line) for line in (run / "validation_metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        weight = json.loads((run / "online.json").read_text(encoding="utf-8"))["objective"]["answer_weight"]
        best = rows[-1]["best_epoch"]
        epochs = {}
        for row in rows:
            payload = load(run, row["epoch"])
            epochs[row["epoch"]] = latency(run, payload, row["epoch"])
            if row["epoch"] == best:
                best_payloads[str(run)] = payload
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

    if args.resamples > 0:
        bootstrap(summary, best_payloads, ranked, args.resamples, args.seed)
    if args.output:
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bootstrap(summary: dict, payloads: dict, ranked: list, resamples: int, seed: int) -> None:
    """Интервалы задержки best/ каждого прогона при его τ; пишет summary[run]["ci95"][key] (в кадрах)."""
    draws_by_videos = {}  # одна validation у всех прогонов — одни и те же выборки видео
    print(f"\n95% bootstrap-интервал задержки best/ на validation ({resamples} выборок видео с возвращением, "
          f"τ фиксирован), в кадрах:")
    print(f"  {'w':<6s} {'эпоха':>5s}  {'R0.05_A0.9':^24s}  {'R0.1_A0.9':^24s}")
    for run, entry in ranked:
        payload = payloads[run]
        video_ids = [video_of(row["record_id"]) for row in payload["predictions"]]
        videos, video_index = np.unique(video_ids, return_inverse=True)
        key = tuple(videos)
        if key not in draws_by_videos:
            if draws_by_videos:
                print(f"  ! {run}: другие видео validation, чем у предыдущих прогонов — свои выборки")
            rng = np.random.default_rng(seed)
            draws = rng.integers(0, len(videos), size=(resamples, len(videos)))
            counts = np.stack([np.bincount(draw, minlength=len(videos)) for draw in draws]).astype(float)
            draws_by_videos[key] = counts[:, video_index]  # вес вопроса = сколько раз выбрано его видео
        weights = draws_by_videos[key]
        entry["ci95"] = {}
        cells = []
        for name in KEYS:
            point = entry["best"][name]
            if not point["feasible"]:
                entry["ci95"][name] = None
                cells.append(f"{'infeasible':^24s}")
                continue
            timely, delay = per_question(payload, point["tau"])
            value = float(timely @ delay / timely.sum())
            if not np.isclose(value, point["latency_mean"], rtol=0, atol=1e-9):
                sys.exit(f"{run} {name}: задержка {value} при τ не совпадает с constrained_latency "
                         f"{point['latency_mean']}")
            samples = (weights @ (timely * delay)) / (weights @ timely)
            low, high = np.nanpercentile(samples, [2.5, 97.5])
            entry["ci95"][name] = {"value": 2 * value, "ci95": [2 * float(low), 2 * float(high)], "unit": "frames"}
            cells.append(f"{2 * value:5.2f} [{2 * low:5.2f}, {2 * high:5.2f}]")
        print(f"  {entry['answer_weight']:<6g} {entry['best_epoch']:5d}  {cells[0]:>24s}  {cells[1]:>24s}")
    winner = summary[ranked[0][0]]["ci95"].get(KEYS[0])
    if winner is None:
        return
    for run, entry in ranked[1:]:
        other = entry["ci95"].get(KEYS[0])
        if other is None:
            continue
        overlap = other["ci95"][0] <= winner["ci95"][1] and winner["ci95"][0] <= other["ci95"][1]
        print(f"  w = {ranked[0][1]['answer_weight']:g} против w = {entry['answer_weight']:g} (R0.05_A0.9): "
              f"{'интервалы перекрываются' if overlap else 'интервалы не перекрываются'}")


if __name__ == "__main__":
    main()
