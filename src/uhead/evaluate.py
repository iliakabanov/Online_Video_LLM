"""Оценка обученного детектора на test-половине eval с заданным порогом, без загрузки Qwen.

Test — видео из split.json каталога весов (uhead.train --output). Порог выбирается по метрикам validation в
results.json. Отчёт и вероятности пишутся в --results (по умолчанию в каталог весов).

PYTHONPATH=src python -m uhead.evaluate --run checkpoints/mlp/01_lm --results results/mlp/01_lm --threshold 0.7
"""
import argparse
import json
import logging
import os
from pathlib import Path

import torch

from .train import _readiness, _write_predictions, head_config, load_checkpoint, predict

logger = logging.getLogger(__name__)


def evaluate_test(run_dir: str | Path, threshold: float, data_root: str | Path, device="cpu",
                  weights: str = "best.pt", output: str | Path | None = None,
                  results_dir: str | Path | None = None) -> dict:
    """Метрики весов run_dir/weights на test-видео из split.json при пороге threshold; пишет JSON и вероятности
    (output, по умолчанию <results_dir или run_dir>/test_<weights>_tau<threshold>.json)."""
    from online_vllm.train.config import TrainConfig
    from online_vllm.train.features import FeatureSet
    from .data import ReadinessDataset, labelled
    if not 0 < threshold < 1:
        raise ValueError("threshold должен лежать в (0, 1)")
    run_dir = Path(run_dir)
    config = TrainConfig.load(run_dir / "train.yaml")
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    model = load_checkpoint(run_dir, device, weights).eval()
    tag_settings = model.tag_settings
    model_name = "statehead" if head_config(model).temporal_detector == "mlp" else "uhead"
    fs = FeatureSet(Path(data_root) / config.features.eval)
    fs.check(config.video)
    # Метки строятся так же, как при обучении: loss на test сопоставим с validation loss.
    items, _ = labelled(fs, config, tag_settings["tags"], tag_settings["step_window"],
                        tag_settings["sigmoid_slope"], tag_settings.get("gaussian_sigma", 2.0))
    test_videos = set(split["test_videos"])
    items = [item for item in items if fs.records[item.index].video in test_videos]
    if len(items) != split["test_questions"]:
        raise ValueError(f"В test {len(items)} вопросов, а split.json ожидает {split['test_questions']}: "
                         "признаки eval или разметка не совпадают с обучением")
    dataset = ReadinessDataset(fs, items, head_config(model).groups)
    loss, rows = predict(model, dataset, config, device, config.optim.batch_size)
    result = _readiness(rows, [threshold], fs.meta, threshold, model_name, tag_settings["tags"])
    summary = {"model": model_name, "feature_groups": head_config(model).groups, **tag_settings,
               "weights": str(run_dir / weights), "threshold": threshold, "questions": len(rows),
               "videos": len(split["test_videos"]), "loss": loss, "classification": result["classification"],
               "online": result["thresholds"][str(threshold)]}
    output = Path(output) if output else (Path(results_dir) if results_dir else run_dir) / \
        f"test_{Path(weights).stem}_tau{threshold}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_predictions(output.with_name(f"predictions_{output.stem}.json"), rows, fs.meta, model_name,
                       tag_settings["tags"], "test")
    logger.info("test: %s", {k: summary[k] for k in ("threshold", "questions", "loss")} | summary["online"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="каталог весов uhead.train (train.yaml, split.json, *.pt)")
    parser.add_argument("--results", help="каталог результатов этого запуска; по умолчанию --run")
    parser.add_argument("--threshold", type=float, required=True, help="порог P(готовность), выбранный на validation")
    parser.add_argument("--weights", default="best.pt", help="best.pt или last.pt")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="по умолчанию <results или run>/test_<weights>_tau<threshold>.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluate_test(args.run, args.threshold, args.data_root, args.device, args.weights, args.output, args.results)


if __name__ == "__main__":
    main()
