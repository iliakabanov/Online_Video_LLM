"""Test-отчёт детектора готовности по сохранённым предсказаниям (формат eval/export.py): порог выбирается на его
validation и применяется к test — тем же способом для любой системы (голова, собственный детектор онлайн-модели,
бейзлайн без обучения), так что системы сравниваются на одних вопросах одной процедурой.

    test_report(validation, test)  ->  метрики test без порога; при τ, выбранном на validation по criterion; задержка
                                       при ограничениях (τ каждого (R, A) с validation -> test, и кривая самого test).
    two_detector_report(...)  ->  то же для двух детекторов одной модели (p — первый, q — второй) с двумя порогами:
                                  "and" — ответ на первой паре, где p_k > τ1 и q_k > τ2; "or" — где p_k > τ1 или
                                  q_k > τ2. Пара порогов выбирается на validation для каждого критерия.
"""
from __future__ import annotations

import numpy as np

from .metrics import THRESHOLD_WINDOWS, best_window_threshold, constrained_latency, operating_point
from .thresholds import payload_arrays, sweep

# Крайние τ1, при которых правило сводится ко второму детектору: "and" с τ1 ниже любого p, "or" с τ1 выше любого p.
ALWAYS, NEVER = -1.0, 2.0


def _arrays(payload: dict):
    data = payload_arrays(payload)
    if data["correct"] is None:
        raise ValueError("В предсказаниях нет answer_correct: задержка при ограничениях требует верности ответа")
    return data["probabilities"], data["target_pairs"], data["valid_mask"], data["correct"]


def questions(payload: dict) -> dict:
    """record_id -> (целевая пара, число пар): чтобы проверить, что системы оценены на одних вопросах."""
    return {row["record_id"]: (row["target_pair"], len(row["probabilities"])) for row in payload["predictions"]}


def test_report(validation: dict, test: dict, criterion: str = "on_time_0_4") -> dict:
    tau = sweep(validation, [0.5])["best_thresholds"][criterion]["tau"]
    swept = sweep(test, [0.5, tau])
    val_arrays, test_arrays = _arrays(validation), _arrays(test)
    constrained = {}
    for key, chosen in constrained_latency(*val_arrays).items():
        constrained[key] = {"validation": chosen, "test_curve": swept["constrained_latency"][key],
                            "test_at_validation_tau": operating_point(*test_arrays, chosen["tau"])
                            if chosen["feasible"] else None}
    return {"criterion": criterion, "tau": tau, "questions": swept["num_videos"],
            "classification": swept["classification"], "latency_accuracy": swept["latency_accuracy"],
            "at_tau": swept["results"][str(tau)], "at_0.5": swept["results"]["0.5"], "constrained_latency": constrained}


def _aligned(first: dict, second: dict) -> list[np.ndarray]:
    """Оценки второго детектора в порядке вопросов первого; вопросы, целевые пары и число пар должны совпадать."""
    if questions(first) != questions(second):
        raise ValueError("У двух детекторов разные вопросы, целевые пары или число пар")
    by_id = {row["record_id"]: np.asarray(row["probabilities"], dtype=float) for row in second["predictions"]}
    return [by_id[row["record_id"]] for row in first["predictions"]]


def combined(first: dict, second_scores: list[np.ndarray], tau1: float, mode: str) -> dict:
    """Предсказания первого детектора (его ответы и разметка) с оценкой s_k правила двух детекторов при пороге τ1:
    первая пара с s_k > τ2 — та же, что по правилу. "and": s_k = q_k, если p_k > τ1, иначе 0; "or": s_k = 1, если
    p_k > τ1, иначе q_k."""
    if mode not in ("and", "or"):
        raise ValueError("mode: and или or")
    rows = []
    for row, q in zip(first["predictions"], second_scores):
        fired = np.asarray(row["probabilities"], dtype=float) > tau1
        score = np.where(fired, q, 0.0) if mode == "and" else np.where(fired, 1.0, q)
        rows.append({**row, "probabilities": score.tolist()})
    return {"metadata": first.get("metadata", {}), "predictions": rows}


def two_detector_report(val_first: dict, test_first: dict, val_second: dict, test_second: dict, mode: str,
                        criterion: str = "on_time_0_4", grid: int = 200) -> dict:
    """Правило двух детекторов (combined) с парой порогов, выбранной на validation: τ1 — по сетке квантилей p_k
    validation и крайнему значению (правило сводится ко второму детектору), τ2 — точный перебор при каждом τ1. Для
    criterion — пара с его наибольшей долей; для каждого (R, A) — с наименьшей средней задержкой. Результат в формате
    test_report: метрики test при выбранных порогах (classification и latency_accuracy — оценки s при τ1 criterion)."""
    val_q, test_q = _aligned(val_first, val_second), _aligned(test_first, test_second)
    values = np.concatenate([np.asarray(row["probabilities"], dtype=float) for row in val_first["predictions"]])
    taus1 = np.unique(np.quantile(values, np.linspace(0, 1, grid)))
    taus1 = np.r_[ALWAYS, taus1] if mode == "and" else np.r_[taus1, NEVER]
    window = THRESHOLD_WINDOWS[criterion]
    best_window, best_constrained = None, {}
    for tau1 in taus1:
        probabilities, targets, mask, correct = _arrays(combined(val_first, val_q, float(tau1), mode))
        window_best = best_window_threshold(probabilities, targets, mask, window)
        if best_window is None or window_best[criterion] > best_window[1][criterion]:
            best_window = (float(tau1), window_best)
        for key, point in constrained_latency(probabilities, targets, mask, correct, points=False).items():
            current = best_constrained.get(key)
            if point["feasible"] and (current is None or not current[1]["feasible"]
                                      or point["latency_mean"] < current[1]["latency_mean"]):
                best_constrained[key] = (float(tau1), point)
            elif current is None:
                best_constrained[key] = (float(tau1), point)

    tau1, chosen = best_window
    swept = sweep(combined(test_first, test_q, tau1, mode), [0.5, chosen["tau"]])
    constrained = {}
    for key, (key_tau1, point) in best_constrained.items():
        val_arrays = _arrays(combined(val_first, val_q, key_tau1, mode))
        test_arrays = _arrays(combined(test_first, test_q, key_tau1, mode))
        validation = constrained_latency(*val_arrays)[key]  # с медианой и долей принудительных ответов
        constrained[key] = {"validation": {**validation, "tau1": key_tau1}, "tau1": key_tau1,
                            "test_curve": constrained_latency(*test_arrays)[key],
                            "test_at_validation_tau": operating_point(*test_arrays, validation["tau"])
                            if validation["feasible"] else None}
    return {"criterion": criterion, "mode": mode, "tau1": tau1, "tau": chosen["tau"], "questions": swept["num_videos"],
            "validation_" + criterion: chosen[criterion], "classification": swept["classification"],
            "latency_accuracy": swept["latency_accuracy"], "at_tau": swept["results"][str(chosen["tau"])],
            "at_0.5": swept["results"]["0.5"], "constrained_latency": constrained}
