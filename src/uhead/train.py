"""Обучение бинарного causal UHead на кешированных признаках без загрузки Qwen."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import UHeadConfig
from .labels import TagMode
from .model import UHead
from online_vllm.models.heads import StateHead
from online_vllm.eval.metrics import evaluate_readiness, flatten_readiness_metrics, uhead_probabilities

logger = logging.getLogger(__name__)


def binary_loss(logits: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor,
                reduction: str = "patch_mean") -> torch.Tensor:
    """BCE-with-logits только по действительным temporal-позициям, без sigmoid до loss."""
    if logits.shape != targets.shape or padding_mask.shape != targets.shape:
        raise ValueError("logits, targets и padding_mask должны иметь одинаковую форму [B, P]")
    valid = ~padding_mask
    if not valid.any():
        raise ValueError("Батч не содержит действительных позиций")
    if reduction == "patch_mean":
        return F.binary_cross_entropy_with_logits(logits[valid], targets[valid])
    if reduction == "sequence_mean":
        per_patch = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        lengths = valid.sum(dim=1)
        return (per_patch.masked_fill(~valid, 0).sum(dim=1) / lengths).mean()
    raise ValueError("reduction должен быть patch_mean или sequence_mean")


def _pad_probabilities(probabilities: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Создаёт единый right-padded массив только на границе evaluator."""
    if not probabilities or any(value.ndim != 1 or not len(value) for value in probabilities):
        raise ValueError("Нужна непустая последовательность вероятностей для каждого видео")
    width = max(map(len, probabilities))
    padded = np.zeros((len(probabilities), width), dtype=np.float64)
    valid = np.zeros_like(padded, dtype=bool)
    for row, mask, value in zip(padded, valid, probabilities):
        row[:len(value)] = value
        mask[:len(value)] = True
    return padded, valid


def _preserve_rng(callable_):
    """Validation не меняет последовательность случайности последующего обучения."""
    state = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        return callable_()
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.random.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])


@torch.no_grad()
def evaluate(model, dataset, config, device, batch_size=64):
    """Единая hard-GT оценка; soft training BCE сохраняется отдельным ключом."""
    from .data import collate
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    total = count = 0
    probabilities, masks, target_pairs, visible_pairs, fps = [], [], [], [], []
    for features, targets, padding, items in loader:
        features = {name: value.to(device) for name, value in features.items()}
        targets, padding = targets.to(device), padding.to(device)
        logits = model(features, padding).float()
        valid = ~padding
        loss = binary_loss(logits, targets, padding, config.loss.binary_reduction)
        weight = len(items) if config.loss.binary_reduction == "sequence_mean" else valid.sum().item()
        total += loss.item() * weight
        count += weight
        probability = uhead_probabilities(logits)
        valid_cpu = valid.cpu().numpy()
        for b, item in enumerate(items):
            n = int(valid_cpu[b].sum())
            probabilities.append(probability[b, :n])
            target_pairs.append(item.target_pair)
            visible_pairs.append(item.visible_pair)
            fps.append(dataset.fs.records[item.index].fps)
    model.train(was_training)
    probabilities, masks = _pad_probabilities(probabilities)
    result = evaluate_readiness(probabilities, target_pairs, visible_pairs, config.eval.taus,
                                dataset.fs.meta["temporal_patch"], dataset.fs.meta["frame_stride"], masks,
                                fps, model="uhead", tags=getattr(model, "tag_settings", {}).get("tags"))
    return {"eval_loss": total / count, "eval_target_bce": total / count, **flatten_readiness_metrics(result)}


def save_checkpoint(model, uhead_config, train_config, tag_settings: dict | Path, path: Path | None = None,
                    metadata: dict | None = None):
    """Сохраняет веса, train-only normalizer buffers и выбранный режим формирования GT."""
    # Старый API передавал path четвёртым аргументом; поддерживаем его, не теряя metadata нового режима.
    if path is None:
        path, tag_settings = Path(tag_settings), {"tags": "hard", "step_window": 5, "sigmoid_slope": 0.65}
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "tag_settings": tag_settings,
                "temporal_detector": uhead_config.temporal_detector}, path / "uhead.pt")
    uhead_config.save(path / "uhead.yaml")
    train_config.save(path / "train.yaml")
    (path / "tags.json").write_text(json.dumps(tag_settings, indent=2) + "\n", encoding="utf-8")
    if metadata is not None:
        saved_metadata = {**metadata, "checkpoint_path": str(path)}
        (path / "experiment.json").write_text(json.dumps(saved_metadata, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    return path


def load_checkpoint(path: str | Path, device="cpu"):
    """Восстанавливает бинарную модель и metadata режима; статистики лежат в state_dict."""
    path = Path(path)
    state = torch.load(path / "uhead.pt", map_location=device, weights_only=True)
    config = UHeadConfig.load(path / "uhead.yaml")
    if state.get("classes") is not None:
        raise ValueError("Трёхклассовый UHead checkpoint несовместим с бинарным readiness detector")
    if state.get("temporal_detector", config.temporal_detector) != config.temporal_detector:
        raise ValueError("Checkpoint и uhead.yaml задают разные temporal_detector")
    model = StateHead(config) if config.temporal_detector == "mlp" else UHead(config)
    model = model.to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.tag_settings = state["tag_settings"]
    return model


def train(uhead_config, train_config, data_root: Path, output_dir: Path, device, tags: TagMode = "hard",
          step_window: int = 5, sigmoid_slope: float = 0.65, flat_output: bool = False,
          overwrite: bool = False) -> Path:
    """Обучает один из трёх бинарных режимов GT на полном наборе доступных пар кадров."""
    # Сохраняем работоспособность прежнего scripts/train_uhead.py, который передавал список классов третьим аргументом.
    if isinstance(data_root, list):
        _, data_root, output_dir, device, tags = data_root, output_dir, device, tags, "hard"
    if tags not in ("hard", "step", "sigmoid") or step_window < 0 or sigmoid_slope <= 0:
        raise ValueError("Некорректные параметры режима разметки")
    from online_vllm.train.features import FeatureSet
    from online_vllm.train.loop import lr_factor, seed_everything
    from .data import ReadinessDataset, collate, fit_normalization, labelled
    if train_config.loss.lm_weight:
        raise NotImplementedError("UHead обучается на кешированных признаках; loss.lm_weight должен быть 0")
    seed_everything(train_config.seed)
    tag_settings = {"tags": tags, "step_window": step_window, "sigmoid_slope": sigmoid_slope}
    train_fs = FeatureSet(Path(data_root) / train_config.features.train)
    eval_fs = FeatureSet(Path(data_root) / train_config.features.eval)
    for fs in (train_fs, eval_fs):
        fs.check(train_config.video)
    for key in ("model_id", "revision", "quantization", "temporal_patch"):
        if train_fs.meta[key] != eval_fs.meta[key]:
            raise ValueError(f"Train и eval features расходятся по {key}")
    train_items, train_dropped = labelled(train_fs, train_config, tags, step_window, sigmoid_slope)
    eval_items, eval_dropped = labelled(eval_fs, train_config, tags, step_window, sigmoid_slope)
    logger.info("train: %d записей (%d исключено), eval: %d (%d исключено)", len(train_items), train_dropped,
                len(eval_items), eval_dropped)
    train_data = ReadinessDataset(train_fs, train_items, uhead_config.groups)
    eval_all = ReadinessDataset(eval_fs, eval_items, uhead_config.groups)
    if train_config.data.eval_samples is not None and train_config.data.eval_samples < len(eval_items):
        sampled = random.Random(train_config.seed).sample(eval_items, train_config.data.eval_samples)
        eval_data = ReadinessDataset(eval_fs, sampled, uhead_config.groups)
    else:
        eval_data = eval_all
    model = (StateHead(uhead_config) if uhead_config.temporal_detector == "mlp" else UHead(uhead_config)).to(device)
    model.set_normalization(fit_normalization(train_data))
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Обучаемых параметров: %d", trainable_parameters)
    output_dir = Path(output_dir) if flat_output else Path(output_dir) / tags
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} уже содержит запуск; укажите --overwrite для явной перезаписи")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"architecture": uhead_config.temporal_detector, "feature_groups": uhead_config.groups,
                "training_gt": tags, "seed": train_config.seed, "trainable_parameters": trainable_parameters,
                "validation_split": str(train_config.features.eval),
                "train_config": asdict(train_config), "uhead_config": asdict(uhead_config)}
    o = train_config.optim
    loader = DataLoader(train_data, batch_size=o.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(train_config.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=o.lr, weight_decay=o.weight_decay)
    batches = math.ceil(len(loader) / o.grad_accum)
    total_steps = o.epochs * batches if o.max_steps is None else min(o.epochs * batches, o.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, o.warmup_steps, total_steps))
    log = open(output_dir / "metrics.jsonl", "w", encoding="utf-8")
    step, losses = 0, []

    def record(metrics):
        log.write(json.dumps({"step": step, **tag_settings, **metrics}) + "\n")
        log.flush()
        logger.info("step %d: %s", step, metrics)

    model.train()
    for epoch in range(o.epochs):
        for batch_index, (features, targets, padding, _) in enumerate(loader):
            features = {name: value.to(device) for name, value in features.items()}
            targets, padding = targets.to(device), padding.to(device)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                                enabled=torch.device(device).type == "cuda"):
                loss = binary_loss(model(features, padding), targets, padding, train_config.loss.binary_reduction)
            (loss / o.grad_accum).backward()
            losses.append(loss.item())
            if (batch_index + 1) % o.grad_accum and batch_index + 1 < len(loader):
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
            if step % o.log_every == 0:
                record({"epoch": epoch, "loss": sum(losses) / len(losses), "lr": scheduler.get_last_lr()[0]}); losses = []
            if step % o.eval_every == 0:
                record(_preserve_rng(lambda: evaluate(model, eval_data, train_config, device, o.batch_size)))
            if step % o.save_every == 0:
                save_checkpoint(model, uhead_config, train_config, tag_settings, output_dir / f"step_{step}", metadata)
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    record({"final": True, **_preserve_rng(lambda: evaluate(model, eval_all, train_config, device, o.batch_size))})
    log.close()
    return save_checkpoint(model, uhead_config, train_config, tag_settings, output_dir / "final", metadata)


def build_parser():
    """Создаёт CLI модуля; --tags обязателен, чтобы эксперименты были явно различимы."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", choices=("hard", "step", "sigmoid"), required=True)
    parser.add_argument("--step-window", type=int, default=5)
    parser.add_argument("--sigmoid-slope", type=float, default=0.65)
    parser.add_argument("--uhead", default="configs/uhead/default.yaml")
    parser.add_argument("--train", default="configs/train/group1.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--flat-output", action="store_true", help="Не добавлять каталог режима GT к --output")
    parser.add_argument("--overwrite", action="store_true", help="Разрешить запись в существующий каталог запуска")
    return parser


def main():
    """Точка запуска: python -m uhead.train --tags {hard,step,sigmoid} ..."""
    args = build_parser().parse_args()
    if args.step_window < 0 or args.sigmoid_slope <= 0:
        raise ValueError("step-window должен быть >= 0, sigmoid-slope — положительным")
    from online_vllm.train.config import TrainConfig
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = train(UHeadConfig.load(args.uhead), TrainConfig.load(args.train), Path(args.data_root), Path(args.output),
                   args.device, args.tags, args.step_window, args.sigmoid_slope, args.flat_output, args.overwrite)
    print("Финальный checkpoint:", result)


if __name__ == "__main__":
    main()
