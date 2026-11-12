"""Test-отчёт детектора готовности по сохранённым предсказаниям (формат eval/export.py): порог выбирается на его
validation и применяется к test — тем же способом для любой системы (голова, собственный детектор онлайн-модели,
бейзлайн без обучения), так что системы сравниваются на одних вопросах одной процедурой.

    test_report(validation, test)  ->  метрики test без порога; при τ, выбранном на validation по criterion; задержка
                                       при ограничениях (τ каждого (R, A) с validation -> test, и кривая самого test).
"""
from __future__ import annotations

from .metrics import constrained_latency, operating_point
from .thresholds import payload_arrays, sweep


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
