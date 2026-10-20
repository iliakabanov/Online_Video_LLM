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


def _classification(probabilities, target_pairs, valid_mask, threshold):
    labels, scores = [], []
    for p, target_pair, mask in zip(probabilities, target_pairs, valid_mask):
        n = int(mask.sum())
        if not 0 <= target_pair < n:
            raise ValueError("target_pair должен принадлежать valid части видео")
        labels.append(np.arange(n) >= target_pair)
        scores.append(p[:n])
    y, score = np.concatenate(labels).astype(np.int64), np.concatenate(scores)
    pred = score >= threshold
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    auroc, auprc = _binary_curves(y, score)
    score_clip = np.clip(score, 1e-7, 1 - 1e-7)
    return {"accuracy": float((pred == y).mean()), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "auroc": auroc, "auprc": auprc, "brier_score": float(np.mean((score - y) ** 2)),
            "binary_cross_entropy": float(-np.mean(y * np.log(score_clip) + (1 - y) * np.log(1 - score_clip))),
            "count": int(len(y))}


def _threshold_metrics(probabilities, target_pairs, visible_pairs, valid_mask, tau, frames_per_pair, fps):
    kinds = {"premature": 0, "early": 0, "on_target": 0, "late": 0, "missed": 0}
    errors, seconds = [], []
    for p, target, visible, mask, one_fps in zip(probabilities, target_pairs, visible_pairs, valid_mask, fps):
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
        kinds["premature" if fired < visible else "early" if fired < target else "on_target" if fired == target else "late"] += 1
    videos = len(target_pairs)
    result = {f"{name}_rate": value / videos for name, value in kinds.items()}
    result["miss_rate"] = result["missed_rate"]  # старый JSON sweep до унификации
    result["false_alarm_rate"] = result["premature_rate"]
    fired_errors = np.asarray(errors, dtype=float)
    result.update({"on_target_pm1": sum(abs(x / frames_per_pair) <= 1 for x in fired_errors) / videos,
                   "on_target_pm2": sum(abs(x / frames_per_pair) <= 2 for x in fired_errors) / videos,
                   "on_target_pm5": sum(abs(x / frames_per_pair) <= 5 for x in fired_errors) / videos,
                   "timing_count": int(len(fired_errors))})
    for name, fn in (("mean", np.mean), ("std", np.std), ("mae", lambda x: np.mean(abs(x))),
                     ("rmse", lambda x: math.sqrt(np.mean(x * x))), ("median", np.median)):
        result[f"timing_{name}_frames"] = None if not len(fired_errors) else float(fn(fired_errors))
    second_errors = np.asarray(seconds, dtype=float)
    for name, fn in (("mean", np.mean), ("std", np.std), ("mae", lambda x: np.mean(abs(x))),
                     ("rmse", lambda x: math.sqrt(np.mean(x * x))), ("median", np.median)):
        result[f"timing_{name}_seconds"] = None if not len(second_errors) else float(fn(second_errors))
    return result


def evaluate_readiness(probabilities: np.ndarray | torch.Tensor, target_pairs: Sequence[int], visible_pairs: Sequence[int],
                       taus: Sequence[float], temporal_patch: int, frame_stride: int, valid_mask=None,
                       fps: Sequence[float | None] | None = None, classification_threshold: float = 0.5,
                       model: str | None = None, tags: str | None = None) -> dict:
    """Возвращает JSON-совместимые метрики; все timing-значения в исходных кадрах и, при FPS, секундах."""
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
    output = {"model": model, "tags": tags, "num_videos": len(p),
              "classification": _classification(p, target_pairs, mask, classification_threshold), "thresholds": {}}
    frames_per_pair = temporal_patch * frame_stride
    for tau in taus:
        if not 0 < tau < 1:
            raise ValueError("taus должны лежать в (0,1)")
        output["thresholds"][str(tau)] = _threshold_metrics(p, target_pairs, visible_pairs, mask, tau,
                                                              frames_per_pair, fps)
    return output
