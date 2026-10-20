"""Перебор порогов по заранее сохранённым вероятностям без повторного inference."""
import argparse
import json
from pathlib import Path

from .metrics import evaluate_readiness


def resolve_thresholds(values=None, start=None, stop=None, step=None):
    """Проверяет явный список или равномерную сетку, устраняя дубликаты floating-point чисел."""
    explicit = values is not None and len(values) > 0
    ranged = any(value is not None for value in (start, stop, step))
    if explicit == ranged:
        raise ValueError("Укажите либо --thresholds, либо полный диапазон --threshold-start/stop/step")
    if explicit:
        result = sorted({round(float(value), 12) for value in values})
    else:
        if start is None or stop is None or step is None or step <= 0 or start > stop:
            raise ValueError("Диапазон требует start <= stop и положительный step")
        result, index = [], 0
        while start + index * step <= stop + abs(step) * 1e-10:
            result.append(round(start + index * step, 12)); index += 1
    if not result or any(not 0 < value < 1 for value in result):
        raise ValueError("Все пороги должны лежать строго в (0, 1)")
    return result


def sweep(payload: dict, thresholds: list[float]) -> dict:
    """Вызывает общий evaluator один раз; patch-метрики вычисляются в нём независимо от числа порогов."""
    # Новый export хранит metadata отдельно и сохраняет каждую последовательность
    # исходной длины. Старый агрегированный формат остаётся рабочим.
    metadata = payload.get("metadata", {})
    if "predictions" in payload:
        rows = payload["predictions"]
        payload = {"model": metadata.get("model"), "tags": metadata.get("tags"), "split": metadata.get("split"),
                   "probabilities": [row["probabilities"] for row in rows],
                   "target_pairs": [row["target_pair"] for row in rows],
                   "visible_pairs": [row["visible_pair"] for row in rows],
                   "fps": [row.get("fps") for row in rows],
                   "temporal_patch": metadata.get("temporal_patch"), "frame_stride": metadata.get("frame_stride")}
    required = ("probabilities", "target_pairs", "visible_pairs", "temporal_patch", "frame_stride")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"В predictions JSON отсутствуют поля: {missing}")
    probabilities = payload["probabilities"]
    # evaluate_readiness принимает прямоугольный ndarray; JSON export намеренно
    # непаддированный, поэтому padding добавляется только внутри sweep.
    if probabilities and len({len(row) for row in probabilities}) > 1:
        width = max(map(len, probabilities))
        padded = [[0.0] * width for _ in probabilities]
        valid_mask = [[False] * width for _ in probabilities]
        for row, value, mask in zip(padded, probabilities, valid_mask):
            row[:len(value)] = value; mask[:len(value)] = [True] * len(value)
    else:
        padded, valid_mask = probabilities, payload.get("valid_mask")
    result = evaluate_readiness(padded, payload["target_pairs"], payload["visible_pairs"], thresholds,
                                payload["temporal_patch"], payload["frame_stride"], valid_mask,
                                payload.get("fps"), payload.get("classification_threshold", 0.5), payload.get("model"),
                                payload.get("tags"))
    return {"model": result["model"], "tags": result["tags"], "split": payload.get("split"),
            "metadata": metadata,
            "num_videos": result["num_videos"], "thresholds": thresholds,
            "classification": result["classification"], "results": result["thresholds"],
            "temporal_patch": payload["temporal_patch"], "frame_stride": payload["frame_stride"]}


def build_parser():
    """CLI sweep; его вход — JSON сохранённых вероятностей, а не checkpoint или модель."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="JSON с probabilities и временной metadata")
    parser.add_argument("--output", required=True, help="Путь итогового JSON")
    parser.add_argument("--thresholds", type=float, nargs="+")
    parser.add_argument("--threshold-start", type=float)
    parser.add_argument("--threshold-stop", type=float)
    parser.add_argument("--threshold-step", type=float)
    return parser


def main():
    args = build_parser().parse_args()
    thresholds = resolve_thresholds(args.thresholds, args.threshold_start, args.threshold_stop, args.threshold_step)
    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    result = sweep(payload, thresholds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
