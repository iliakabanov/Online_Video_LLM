"""Общие online- и классификационные метрики для StateHead и UHead."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch


def statehead_probabilities(logits: torch.Tensor | np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Берёт P(SPEAK) из logits StateHead [..., C] без предположения о порядке классов.

    Вход [B, P, C] даёт [B, P]; плоские logits позиций решений [N, C] (train/head.py, train/loop.py) дают [N].
    """
    if "speak" not in classes:
        raise ValueError("classes должен содержать speak")
    x = torch.as_tensor(logits)
    if x.ndim < 2 or x.shape[-1] != len(classes):
        raise ValueError("StateHead logits должны иметь форму [..., C]")
    return x.float().softmax(-1)[..., classes.index("speak")].cpu().numpy()


def uhead_probabilities(logits: torch.Tensor | np.ndarray) -> np.ndarray:
    """Превращает бинарные logits UHead [B,P] или [B,P,1] в вероятность готовности [B,P]."""
    x = torch.as_tensor(logits)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError("UHead logits должны иметь форму [B, P] или [B, P, 1]")
    return x.float().sigmoid().cpu().numpy()


def _binary_curves(target: np.ndarray, probability: np.ndarray):
    """Tie-aware AUROC и average precision; undefined при единственном классе.

    Группировка одинаковых score обязательна: произвольный порядок внутри tie
    иначе превращает константный классификатор в искусственно хороший/плохой.
    Формула AP совпадает с ``sklearn.metrics.average_precision_score``.
    """
    positives, negatives = int(target.sum()), int(len(target) - target.sum())
    if not positives or not negatives:
        return None, None
    order = np.argsort(-probability, kind="stable")
    y, score = target[order], probability[order]
    # Индексы последних элементов групп одинаковых вероятностей.
    ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score) - 1]
    tp, fp = np.cumsum(y)[ends], np.cumsum(1 - y)[ends]
    tpr, fpr = np.r_[0.0, tp / positives], np.r_[0.0, fp / negatives]
    auroc = float(np.trapezoid(tpr, fpr))
    precision = tp / (tp + fp)
    recall = tp / positives
    auprc = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
    return auroc, auprc


def flatten_readiness_metrics(result: dict) -> dict:
    """Преобразует общий JSON evaluator в плоские ключи training-логов.

    Старые training scripts продолжают писать JSONL, но все readiness-значения
    ниже происходят из единственного evaluator.
    """
    classification = result["classification"]
    metrics = {f"eval_{key}": value for key, value in classification.items() if key != "count"}
    metrics["eval_bce"] = classification["binary_cross_entropy"]
    for tau, values in result["thresholds"].items():
        for name, value in values.items():
            metrics[f"online_{tau}_{name}"] = value
        # Совместимость с прежним именем в head.py/uhead.train.
        metrics[f"online_{tau}_miss_rate"] = values["missed_rate"]
    return metrics


# Окна вокруг целевой пары (в парах) для классификации префиксов: далёкие от t* префиксы лёгкие и завышают AUROC по
# всем парам, окно |k - t*| <= w оставляет трудную границу (w пар до t* — отрицательные, t*..t*+w — положительные).
WINDOWS = (2, 5, 10)


def _pointwise(y: np.ndarray, score: np.ndarray, threshold: float, suffix: str = "") -> dict:
    pred = score >= threshold
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    auroc, auprc = _binary_curves(y, score)
    return {f"accuracy{suffix}": float((pred == y).mean()), f"precision{suffix}": precision,
            f"recall{suffix}": recall,
            f"f1{suffix}": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            f"auroc{suffix}": auroc, f"auprc{suffix}": auprc}


def _calibration(y: np.ndarray, score: np.ndarray, suffix: str = "") -> dict:
    clipped = np.clip(score, 1e-7, 1 - 1e-7)
    return {f"brier_score{suffix}": float(np.mean((score - y) ** 2)),
            f"binary_cross_entropy{suffix}": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))}


def _classification(probabilities, target_pairs, valid_mask, threshold):
    """Каждая пара каждого видео — префикс, метка 1 начиная с t*. Метрики по всем парам и, с суффиксом _pm<w>, только
    по парам |k - t*| <= w (WINDOWS), в том числе BCE и Brier."""
    labels, scores, offsets = [], [], []
    for p, target_pair, mask in zip(probabilities, target_pairs, valid_mask):
        n = int(mask.sum())
        if not 0 <= target_pair < n:
            raise ValueError("target_pair должен принадлежать valid части видео")
        labels.append(np.arange(n) >= target_pair)
        scores.append(p[:n])
        offsets.append(np.arange(n) - target_pair)
    y, score = np.concatenate(labels).astype(np.int64), np.concatenate(scores)
    offset = np.concatenate(offsets)
    result = {**_pointwise(y, score, threshold), **_calibration(y, score), "count": int(len(y))}
    for w in WINDOWS:
        near = np.abs(offset) <= w
        result.update(_pointwise(y[near], score[near], threshold, f"_pm{w}"))
        result.update(_calibration(y[near], score[near], f"_pm{w}"))
        result[f"count_pm{w}"] = int(near.sum())
    return result


# Бюджеты средней задержки (в парах): при них latency_accuracy сообщает точность (accuracy_at) и площадь (auc_<b>).
LATENCY_BUDGETS = (0, 1, 2, 5, 10)
AUC_BUDGETS = (2, 5, 10)


def latency_accuracy(probabilities, target_pairs, valid_mask, correct) -> dict:
    """Кривая «средняя задержка — точность» по всем порогам τ и площадь под ней.

    При пороге τ вопрос отвечается на первой паре k с p_k > τ, а без такой пары — принудительно на последней паре
    (ответ есть всегда, верность — correct на этой паре). Задержка вопроса — max(k − t*, 0) пар: ранний ответ не
    экономит время, его цена — в точности. Рост τ сдвигает каждый ответ только позже, поэтому кривая идёт от τ → 0
    (все отвечают на паре 0: задержка 0) до τ → 1 (все в конце: задержка L = mean(P − 1 − t*)); оба конца зависят
    только от данных, не от модели. A(ℓ) — лучшая точность при средней задержке ≤ ℓ (ступенчатая огибающая, без
    интерполяции). auc = ∫₀ᴸ A(ℓ)dℓ / L, auc_<b> — то же на [0, b] пар (AUC_BUDGETS), accuracy_at — A(ℓ) при
    LATENCY_BUDGETS. На CLEVRER L ≈ 33 пары, а кривая выходит на полку за ~1 пару, поэтому auc почти равна точности в
    конце видео; своевременность отражают короткие бюджеты. Кривая точная: события — пороги, на которых меняется пара
    ответа хотя бы одного вопроса."""
    late0 = right0 = 0.0
    taus, d_late, d_right = [], [], []
    for p, target, mask, c in zip(probabilities, target_pairs, valid_mask, correct):
        n = int(mask.sum())
        run = np.maximum.accumulate(np.asarray(p[:n], dtype=float))
        records = np.r_[0, np.flatnonzero(run[1:] > run[:-1]) + 1]  # пары, где растёт running max
        after = np.r_[records[1:], n - 1]  # куда сдвигается ответ, когда τ доходит до run[record]
        late = lambda k: np.maximum(k - target, 0)  # noqa: E731
        c = np.asarray(c, dtype=float)
        late0 += late(0); right0 += c[0]
        taus.append(run[records]); d_late.append(late(after) - late(records)); d_right.append(c[after] - c[records])
    videos = len(target_pairs)
    tau, d_late, d_right = np.concatenate(taus), np.concatenate(d_late), np.concatenate(d_right)
    order = np.argsort(tau, kind="stable")
    tau, x, y = tau[order], late0 + np.cumsum(d_late[order]), right0 + np.cumsum(d_right[order])
    last = np.r_[tau[1:] != tau[:-1], True]  # состояние после всех событий одного порога
    x, y = np.r_[late0, x[last]] / videos, np.r_[right0, y[last]] / videos
    best = np.maximum.accumulate(y)
    total = float(x[-1])

    def area(limit: float) -> float:
        edges = np.minimum(np.r_[x, max(limit, total)], limit)
        return float(np.sum(best * np.diff(edges)))

    def at(budget: float) -> float:
        return float(best[np.searchsorted(x, budget, side="right") - 1])

    return {"auc": area(total) / total if total > 0 else float(best[-1]),
            **{f"auc_{budget}": area(budget) / budget for budget in AUC_BUDGETS}, "max_lateness_pairs": total, "start_accuracy": float(y[0]), "end_accuracy": float(y[-1]),
            "accuracy_at": {str(b): at(b) for b in LATENCY_BUDGETS}, "points": int(len(x))}


# Окна срабатывания, по которым выбирается порог: метрика _threshold_metrics -> допустимый сдвиг k − t* в парах
# (включительно). on_target_pm2 — симметричное окно ±2; on_time_0_4 — не раньше целевой пары и не позже 4 пар после неё.
THRESHOLD_WINDOWS = {"on_target_pm2": (-2, 2), "on_time_0_4": (0, 4)}
# При каждом τ: доля ответов с задержкой 0..d пар после целевой пары (on_time_0_d) и из них верных (timely_correct_0_d).
ON_TIME_DELAYS = (4, 9)


def window_metric(window: tuple[int, int]) -> str:
    low, high = window
    return f"on_target_pm{high}" if low == -high else f"on_time_{low}_{high}"


def best_window_threshold(probabilities, target_pairs, valid_mask, window: int | tuple[int, int] = 2) -> dict:
    """Порог τ с наибольшей долей вопросов, ответ на которые пришёл в окне low <= k − t* <= high (window = (low, high);
    число w — окно (−w, w)). Это метрика window_metric(window) из _threshold_metrics: срабатывание на первой паре с
    p_k > τ, пропуск — мимо окна.

    Перебор точный: доля меняется только на порогах, где у какого-то вопроса сдвигается пара срабатывания (значения
    running max его p_k), поэтому она постоянна на интервалах [tau_low, tau_high). Из интервалов с максимумом берётся
    самый широкий, tau — его середина (внутри интервала любое τ даёт ту же долю)."""
    low_shift, high_shift = (-window, window) if isinstance(window, int) else tuple(window)
    if low_shift > high_shift:
        raise ValueError("Окно должно быть low <= high")
    hits0, taus, deltas = 0, [], []
    for p, target, mask in zip(probabilities, target_pairs, valid_mask):
        n = int(mask.sum())
        run = np.maximum.accumulate(np.asarray(p[:n], dtype=float))
        records = np.r_[0, np.flatnonzero(run[1:] > run[:-1]) + 1]  # пары срабатывания при росте τ
        shift = records - target
        inside = ((shift >= low_shift) & (shift <= high_shift)).astype(int)
        hits0 += inside[0]
        taus.append(run[records]); deltas.append(np.r_[inside[1:], 0] - inside)  # после последней — пропуск
    tau, delta = np.concatenate(taus), np.concatenate(deltas)
    order = np.argsort(tau, kind="stable")
    tau, counts = tau[order], hits0 + np.cumsum(delta[order])
    last = np.r_[tau[1:] != tau[:-1], True]
    values = np.r_[hits0, counts[last]]
    edges = np.clip(np.r_[-np.inf, tau[last], np.inf], 0.0, 1.0)  # интервал j: [edges[j], edges[j + 1])
    widths = np.diff(edges)
    # Пустой интервал (p_k ровно 0 или 1 в float32) порогом из (0, 1) не достигается.
    best = int(np.lexsort((-widths, -np.where(widths > 0, values, -1)))[0])
    low, high = float(edges[best]), float(edges[best + 1])
    metric = window_metric((low_shift, high_shift))
    return {"metric": metric, "window": [low_shift, high_shift], "tau": (low + high) / 2, "tau_low": low,
            "tau_high": high, metric: float(values[best] / len(target_pairs))}


# Задержка пары «модель ответов — детектор» при ограничениях: доля преждевременных ответов (раньше t*) не больше
# бюджета R и точность среди непреждевременных ответов не ниже уровня A (constrained_latency).
PREMATURE_BUDGETS = (0.05, 0.10)
ACCURACY_LEVELS = (0.85, 0.90)


def constraint_key(budget: float, level: float) -> str:
    return f"R{budget:g}_A{level:g}"


def operating_point(probabilities, target_pairs, valid_mask, correct, tau: float) -> dict:
    """Пара «модель — детектор» при пороге τ: ответ на первой паре k с p_k > τ, без такой пары — принудительно на
    последней (forced). Преждевременный ответ (k < t*) — ошибка при любой его верности; точность, задержка k − t* (в
    парах; среднее и медиана) — по непреждевременным ответам, включая принудительные."""
    target = np.asarray(target_pairs)
    fired, forced, right = [], [], []
    for p, mask, c in zip(probabilities, valid_mask, correct):
        n = int(np.asarray(mask).sum())
        hits = np.flatnonzero(np.asarray(p[:n], dtype=float) > tau)
        k = int(hits[0]) if len(hits) else n - 1
        fired.append(k); forced.append(not len(hits)); right.append(bool(c[k]))
    fired, forced, right = np.asarray(fired), np.asarray(forced), np.asarray(right)
    timely = fired >= target
    delay = (fired - target)[timely]
    return {"tau": float(tau), "premature_rate": float((~timely).mean()),
            "accuracy": float(right[timely].mean()) if timely.any() else None,
            "timely_correct": float((right & timely).mean()), "forced_rate": float(forced.mean()),
            "latency_mean": float(delay.mean()) if len(delay) else None,
            "latency_median": float(np.median(delay)) if len(delay) else None}


def constrained_latency(probabilities, target_pairs, valid_mask, correct, budgets: Sequence[float] = PREMATURE_BUDGETS,
                        levels: Sequence[float] = ACCURACY_LEVELS, points: bool = True) -> dict:
    """Для каждого бюджета R и уровня A: порог τ с наименьшей средней задержкой среди τ, где доля преждевременных
    ответов <= R и точность непреждевременных >= A (operating_point). Кривая по τ точная: доли меняются только на
    порогах, где у какого-то вопроса сдвигается пара ответа (значения running max p_k), поэтому всё постоянно на
    интервалах [tau_low, tau_high); tau — середина выбранного интервала (при равной задержке — самого широкого).
    Если ни один τ не проходит ограничения — feasible false и лучшая точность, достижимая при доле <= R."""
    count = len(target_pairs)
    start = np.zeros(3)  # преждевременные, сумма задержек и верные среди непреждевременных при τ ниже всех событий
    taus, deltas = [], []
    for p, target, mask, c in zip(probabilities, target_pairs, valid_mask, correct):
        n = int(np.asarray(mask).sum())
        run = np.maximum.accumulate(np.asarray(p[:n], dtype=float))
        records = np.r_[0, np.flatnonzero(run[1:] > run[:-1]) + 1]  # пара ответа до события
        after = np.r_[records[1:], n - 1]  # и после: τ дошёл до run[record]
        c = np.asarray(c[:n], dtype=bool)

        def state(k):
            timely = k >= target
            return np.stack([~timely, np.where(timely, k - target, 0), timely & c[k]], axis=1).astype(float)

        start += state(np.array([0]))[0]
        taus.append(run[records]); deltas.append(state(after) - state(records))
    tau, delta = np.concatenate(taus), np.concatenate(deltas)
    order = np.argsort(tau, kind="stable")
    tau, sums = tau[order], start + np.cumsum(delta[order], axis=0)
    last = np.r_[tau[1:] != tau[:-1], True]
    sums = np.vstack([start, sums[last]])  # состояние j — на интервале [edges[j], edges[j + 1])
    edges = np.clip(np.r_[-np.inf, tau[last], np.inf], 0.0, 1.0)
    widths = np.diff(edges)
    premature, timely = sums[:, 0] / count, count - sums[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        accuracy, latency = sums[:, 2] / timely, sums[:, 1] / timely
    result = {}
    for budget in budgets:
        allowed = (premature <= budget) & (timely > 0) & (widths > 0)
        for level in levels:
            feasible = allowed & (accuracy >= level)
            key = constraint_key(budget, level)
            if not feasible.any():
                result[key] = {"premature_budget": budget, "accuracy_level": level, "feasible": False,
                               "max_accuracy": float(accuracy[allowed].max()) if allowed.any() else None}
                continue
            best = int(np.lexsort((-widths, np.where(feasible, latency, np.inf)))[0])
            low, high = float(edges[best]), float(edges[best + 1])
            if not points:  # только выбор τ (перебор многих оценок): без медианы и доли принудительных ответов
                result[key] = {"premature_budget": budget, "accuracy_level": level, "feasible": True,
                               "tau_low": low, "tau_high": high, "tau": (low + high) / 2,
                               "premature_rate": float(premature[best]), "accuracy": float(accuracy[best]),
                               "latency_mean": float(latency[best])}
                continue
            point = operating_point(probabilities, target_pairs, valid_mask, correct, (low + high) / 2)
            result[key] = {"premature_budget": budget, "accuracy_level": level, "feasible": True,
                           "tau_low": low, "tau_high": high, **point}
    return result


def _threshold_metrics(probabilities, target_pairs, visible_pairs, valid_mask, tau, frames_per_pair, fps,
                       correct=None):
    kinds = {"premature": 0, "early": 0, "on_target": 0, "late": 0, "missed": 0}
    right = dict.fromkeys(kinds, 0)  # верные ответы Qwen на паре срабатывания, по исходам
    timely_right, on_time_right = 0, dict.fromkeys(ON_TIME_DELAYS, 0)
    errors, seconds = [], []
    for v, (p, target, visible, mask, one_fps) in enumerate(zip(probabilities, target_pairs, visible_pairs,
                                                                 valid_mask, fps)):
        n = int(mask.sum())
        if not 0 <= visible <= target < n:
            raise ValueError("Нужны 0 <= t_visible <= t* < длины видео")
        hits = np.flatnonzero(p[:n] > tau)
        if not len(hits):
            kinds["missed"] += 1
            continue
        fired = int(hits[0]); error = fired - target; errors.append(error * frames_per_pair)
        if one_fps is not None:
            if one_fps <= 0:
                raise ValueError("fps должен быть положительным")
            seconds.append(error * frames_per_pair / one_fps)
        kind = "premature" if fired < visible else "early" if fired < target else "on_target" if fired == target else "late"
        kinds[kind] += 1
        if correct is not None and correct[v][fired]:
            right[kind] += 1
            timely_right += abs(error) <= 2
            for delay in ON_TIME_DELAYS:
                on_time_right[delay] += 0 <= error <= delay
    videos = len(target_pairs)
    result = {f"{name}_rate": value / videos for name, value in kinds.items()}
    result["miss_rate"] = result["missed_rate"]  # старый JSON sweep до унификации
    result["false_alarm_rate"] = result["premature_rate"]
    fired_errors = np.asarray(errors, dtype=float)
    result.update({"on_target_pm1": sum(abs(x / frames_per_pair) <= 1 for x in fired_errors) / videos,
                   "on_target_pm2": sum(abs(x / frames_per_pair) <= 2 for x in fired_errors) / videos,
                   "on_target_pm5": sum(abs(x / frames_per_pair) <= 5 for x in fired_errors) / videos,
                   **{f"on_time_0_{delay}": sum(0 <= x / frames_per_pair <= delay for x in fired_errors) / videos
                      for delay in ON_TIME_DELAYS},
                   "timing_count": int(len(fired_errors))})
    for name, fn in (("mean", np.mean), ("std", np.std), ("mae", lambda x: np.mean(abs(x))),
                     ("rmse", lambda x: math.sqrt(np.mean(x * x))), ("median", np.median)):
        result[f"timing_{name}_frames"] = None if not len(fired_errors) else float(fn(fired_errors))
    second_errors = np.asarray(seconds, dtype=float)
    for name, fn in (("mean", np.mean), ("std", np.std), ("mae", lambda x: np.mean(abs(x))),
                     ("rmse", lambda x: math.sqrt(np.mean(x * x))), ("median", np.median)):
        result[f"timing_{name}_seconds"] = None if not len(second_errors) else float(fn(second_errors))
    if correct is not None:
        # Ответ Qwen на паре срабатывания головы (eval/answers.py); пропуск считается неверным ответом.
        answered = videos - kinds["missed"]
        result["answer_accuracy"] = sum(right.values()) / videos
        result["answer_accuracy_answered"] = sum(right.values()) / answered if answered else None
        result["timely_correct_pm2"] = timely_right / videos
        for delay in ON_TIME_DELAYS:
            result[f"timely_correct_0_{delay}"] = on_time_right[delay] / videos
        for kind in ("premature", "early", "on_target", "late"):
            result[f"answer_accuracy_{kind}"] = right[kind] / kinds[kind] if kinds[kind] else None
    return result


def evaluate_readiness(probabilities: np.ndarray | torch.Tensor, target_pairs: Sequence[int], visible_pairs: Sequence[int],
                       taus: Sequence[float], temporal_patch: int, frame_stride: int, valid_mask=None,
                       fps: Sequence[float | None] | None = None, classification_threshold: float = 0.5,
                       model: str | None = None, tags: str | None = None, correct=None) -> dict:
    """Возвращает JSON-совместимые метрики; все timing-значения в исходных кадрах и, при FPS, секундах.

    correct — по видео булев массив «ответ Qwen на этой паре верный» (eval/answers.py); с ним к метрикам каждого
    порога добавляются answer_accuracy, answer_accuracy_answered, timely_correct_pm2 и точность по исходам, а в
    latency_accuracy — кривая «задержка — точность» по всем порогам (без correct — None)."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or len(p) != len(target_pairs) or len(p) != len(visible_pairs):
        raise ValueError("probabilities должны иметь форму [B,P], согласованную с metadata")
    if np.any((p < 0) | (p > 1)) or temporal_patch < 1 or frame_stride < 1 or not 0 < classification_threshold < 1:
        raise ValueError("Некорректные вероятности или параметры времени/порога")
    mask = np.ones_like(p, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if mask.shape != p.shape or np.any(mask.sum(1) == 0):
        raise ValueError("valid_mask должен иметь форму probabilities и хотя бы одну позицию на видео")
    if np.any((~mask[:, :-1]) & mask[:, 1:]):
        raise ValueError("valid_mask должен описывать right-padding без пропусков внутри последовательности")
    if fps is None:
        fps = [None] * len(p)
    if len(fps) != len(p):
        raise ValueError("fps должен содержать одно значение на видео")
    if correct is not None:
        correct = [np.asarray(c, dtype=bool) for c in correct]
        if len(correct) != len(p) or any(len(c) < n for c, n in zip(correct, mask.sum(1))):
            raise ValueError("correct должен содержать по видео массив не короче его valid-части")
    output = {"model": model, "tags": tags, "num_videos": len(p),
              "classification": _classification(p, target_pairs, mask, classification_threshold), "thresholds": {},
              "latency_accuracy": None if correct is None else latency_accuracy(p, target_pairs, mask, correct),
              "best_thresholds": {metric: best_window_threshold(p, target_pairs, mask, window)
                                  for metric, window in THRESHOLD_WINDOWS.items()},
              "constrained_latency": None if correct is None else constrained_latency(p, target_pairs, mask, correct)}
    frames_per_pair = temporal_patch * frame_stride
    for tau in taus:
        if not 0 < tau < 1:
            raise ValueError("taus должны лежать в (0,1)")
        output["thresholds"][str(tau)] = _threshold_metrics(p, target_pairs, visible_pairs, mask, tau,
                                                              frames_per_pair, fps, correct)
    return output
