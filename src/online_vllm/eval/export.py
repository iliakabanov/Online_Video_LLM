"""Экспорт readiness-вероятностей из cached features без загрузки Qwen2.5-VL.

Пример:
``python -m online_vllm.eval.export --model uhead --checkpoint outputs/uhead/hard/final
--features data/features/val --split validation --output predictions.json``.
Полученный JSON напрямую принимается ``python -m online_vllm.eval.thresholds --predictions ...``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..data.labels import plan
from ..models.heads import StateHead
from ..train.config import TrainConfig
from ..train.features import FeatureSet
from .metrics import uhead_probabilities


def _items(fs: FeatureSet, config: TrainConfig):
    """Metadata evaluation всегда строится от plan(), а не от training tags."""
    result = []
    for index, record in enumerate(fs.records):
        state_plan = plan(record, config.video.frame_stride, config.labels.speak_delay, fs.meta["temporal_patch"])
        if state_plan is None:
            continue
        target_pair = len(state_plan.labels) - 1
        if target_pair >= fs.patches[index]:
            raise ValueError(f"{record.id}: target_pair отсутствует в cached features")
        visible_pair = next(i for i, label in enumerate(state_plan.labels) if label != "silent")
        result.append((index, target_pair, visible_pair))
    return result


@torch.no_grad()
def export_predictions(model_name: str, checkpoint: str | Path, features: str | Path, split: str,
                       output: str | Path, device: str = "cpu") -> dict:
    """Записывает непаддированные вероятности каждой записи в переносимый JSON."""
    checkpoint, output = Path(checkpoint), Path(output)
    config = TrainConfig.load(checkpoint / "train.yaml")
    fs = FeatureSet(features)
    fs.check(config.video)
    items = _items(fs, config)
    metadata = {"model": model_name, "seed": config.seed, "checkpoint_path": str(checkpoint), "split": split,
                "temporal_patch": fs.meta["temporal_patch"], "frame_stride": fs.meta["frame_stride"]}
    experiment_path = checkpoint / "experiment.json"
    if experiment_path.exists():
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        metadata.update({key: experiment[key] for key in ("architecture", "feature_groups", "training_gt",
                                                           "trainable_parameters") if key in experiment})
    rows = []
    if model_name == "statehead":
        # Новый StateHead хранится в том же компактном detector checkpoint, что и UHead.
        from uhead.config import UHeadConfig
        if not (checkpoint / "uhead.pt").exists() or not (checkpoint / "uhead.yaml").exists():
            raise ValueError("Трёхклассовый StateHead checkpoint несовместим с бинарным MLP readiness detector")
        state = torch.load(checkpoint / "uhead.pt", map_location="cpu", weights_only=True)
        uhead_config = UHeadConfig.load(checkpoint / "uhead.yaml")
        if uhead_config.temporal_detector != "mlp":
            raise ValueError("--model statehead требует checkpoint с temporal_detector: mlp")
        head = StateHead(uhead_config).to(device)
        head.load_state_dict(state["model_state"], strict=True); head.eval()
        metadata["tags"] = state["tag_settings"].get("tags")
        metadata["feature_groups"] = list(uhead_config.groups)
        from uhead.data import feature_rows
        for index, target, visible in items:
            n = fs.patches[index]
            values = feature_rows(fs, index, n, uhead_config.groups)
            features_tensor = {name: torch.from_numpy(value)[None].to(device) for name, value in values.items()}
            probability = uhead_probabilities(head(features_tensor).cpu())[0]
            record = fs.records[index]
            rows.append({"record_id": record.id, "probabilities": probability.tolist(), "target_pair": target,
                         "visible_pair": visible, "fps": record.fps})
    elif model_name == "uhead":
        from uhead.data import feature_rows
        from uhead.train import load_checkpoint
        model = load_checkpoint(checkpoint, device).eval()
        metadata["tags"] = model.tag_settings.get("tags")
        metadata["feature_groups"] = list(model.config.groups)
        for index, target, visible in items:
            n = fs.patches[index]
            values = feature_rows(fs, index, n, model.config.groups)
            features_tensor = {name: torch.from_numpy(value)[None].to(device) for name, value in values.items()}
            probability = uhead_probabilities(model(features_tensor).cpu())[0]
            record = fs.records[index]
            rows.append({"record_id": record.id, "probabilities": probability.tolist(), "target_pair": target,
                         "visible_pair": visible, "fps": record.fps})
    else:
        raise ValueError("model должен быть statehead или uhead")
    payload = {"metadata": metadata, "predictions": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("statehead", "uhead"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True, help="Каталог FeatureSet для validation или test")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    export_predictions(args.model, args.checkpoint, args.features, args.split, args.output, args.device)


if __name__ == "__main__":
    main()
