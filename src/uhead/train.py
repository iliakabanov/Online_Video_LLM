"""Обучение бинарного causal UHead на кешированных признаках без загрузки Qwen.

Eval-признаки делятся по видео на две равные части: validation и test (split.json). После каждой эпохи считаются
средний train loss и метрики validation; сохраняются last.pt и best.pt (наименьший validation loss). После обучения
best.pt оценивается на validation при каждом пороге eval.thresholds (results.json). Test оценивается отдельно, с
выбранным порогом: python -m uhead.evaluate.
"""
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
from online_vllm.eval.answers import answer_trace
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
def predict(model, dataset, config, device, batch_size=64, with_rows=True) -> tuple[float, list[dict]]:
    """Loss по меткам режима обучения и по каждой записи dataset: непаддированные вероятности готовности и
    answer_correct — верен ли на каждой паре ответ Qwen по пробе (eval/answers.py). with_rows=False — только loss."""
    from .data import collate
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    total = count = 0
    rows = []
    for features, targets, padding, items in loader:
        features = {name: value.to(device) for name, value in features.items()}
        targets, padding = targets.to(device), padding.to(device)
        logits = model(features, padding).float()
        valid = ~padding
        loss = binary_loss(logits, targets, padding, config.loss.binary_reduction)
        weight = len(items) if config.loss.binary_reduction == "sequence_mean" else valid.sum().item()
        total += loss.item() * weight
        count += weight
        if not with_rows:
            continue
        probability = uhead_probabilities(logits)
        valid_cpu = valid.cpu().numpy()
        for b, item in enumerate(items):
            record = dataset.fs.records[item.index]
            n = int(valid_cpu[b].sum())
            rows.append({"record_id": record.id, "probabilities": probability[b, :n],
                         "target_pair": item.target_pair, "visible_pair": item.visible_pair, "fps": record.fps,
                         "answer_correct": answer_trace(dataset.fs, item.index, n).is_correct})
    model.train(was_training)
    return total / count, rows


def _readiness(rows, taus, meta, classification_threshold=0.5, model=None, tags=None) -> dict:
    probabilities, masks = _pad_probabilities([row["probabilities"] for row in rows])
    correct = [row["answer_correct"] for row in rows] if all("answer_correct" in row for row in rows) else None
    return evaluate_readiness(probabilities, [row["target_pair"] for row in rows],
                              [row["visible_pair"] for row in rows], taus, meta["temporal_patch"],
                              meta["frame_stride"], masks, [row["fps"] for row in rows], classification_threshold,
                              model=model, tags=tags, correct=correct)


def evaluate(model, dataset, config, device, batch_size=64):
    """Единая hard-GT оценка; soft training BCE сохраняется отдельным ключом."""
    loss, rows = predict(model, dataset, config, device, batch_size)
    result = _readiness(rows, config.eval.taus, dataset.fs.meta, model="uhead",
                        tags=getattr(model, "tag_settings", {}).get("tags"))
    return {"eval_loss": loss, "eval_target_bce": loss, **flatten_readiness_metrics(result)}


def split_by_video(fs, items, seed: int):
    """Делит записи eval на validation и test по видео, чтобы вопросы одного видео не попали в обе части."""
    by_video = {}
    for item in items:
        by_video.setdefault(fs.records[item.index].video, []).append(item)
    videos = list(by_video)
    if len(videos) < 2:
        raise ValueError("Для деления на validation и test нужно хотя бы два видео в eval")
    random.Random(seed).shuffle(videos)
    half = len(videos) // 2
    validation_videos, test_videos = sorted(videos[:half]), sorted(videos[half:])
    validation = [item for video in validation_videos for item in by_video[video]]
    test = [item for video in test_videos for item in by_video[video]]
    return validation, test, {"seed": seed, "validation_videos": validation_videos, "test_videos": test_videos,
                              "validation_questions": len(validation), "test_questions": len(test)}


def head_config(model) -> UHeadConfig:
    """UHeadConfig модели: у бинарного StateHead он хранится в readiness_config, у UHead — в config."""
    return model.readiness_config if isinstance(model, StateHead) else model.config


def _write_predictions(path: Path, rows, meta, model_name, tags, split):
    """Формат online_vllm.eval.export: его принимает python -m online_vllm.eval.thresholds."""
    payload = {"metadata": {"model": model_name, "tags": tags, "split": split,
                            "temporal_patch": meta["temporal_patch"], "frame_stride": meta["frame_stride"]},
               "predictions": [{**row, "probabilities": row["probabilities"].tolist(),
                                "answer_correct": np.asarray(row["answer_correct"], dtype=bool).tolist()}
                               for row in rows]}
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_checkpoint(model, uhead_config, train_config, tag_settings: dict | Path, path: Path | None = None,
                    metadata: dict | None = None, filename: str = "uhead.pt"):
    """Сохраняет веса (path/filename), train-only normalizer buffers и выбранный режим формирования GT."""
    # Старый API передавал path четвёртым аргументом; поддерживаем его, не теряя metadata нового режима.
    if path is None:
        path, tag_settings = Path(tag_settings), {"tags": "hard", "step_window": 5, "sigmoid_slope": 0.65,
                                                   "gaussian_sigma": 2.0}
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "tag_settings": tag_settings,
                "temporal_detector": uhead_config.temporal_detector}, path / filename)
    uhead_config.save(path / "uhead.yaml")
    train_config.save(path / "train.yaml")
    (path / "tags.json").write_text(json.dumps(tag_settings, indent=2) + "\n", encoding="utf-8")
    if metadata is not None:
        saved_metadata = {**metadata, "checkpoint_path": str(path)}
        (path / "experiment.json").write_text(json.dumps(saved_metadata, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    return path


def load_checkpoint(path: str | Path, device="cpu", filename: str | None = None):
    """Восстанавливает бинарную модель и metadata режима; статистики лежат в state_dict.

    path — каталог запуска или файл весов внутри него. В каталоге без filename берётся best.pt, если он есть,
    иначе uhead.pt (чекпоинты до разделения на best/last)."""
    path = Path(path)
    if path.is_file():
        path, filename = path.parent, path.name
    elif filename is None:
        filename = "best.pt" if (path / "best.pt").exists() else "uhead.pt"
    state = torch.load(path / filename, map_location=device, weights_only=True)
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
          overwrite: bool = False, results_dir: Path | None = None, gaussian_sigma: float = 2.0) -> Path:
    """Обучает один из трёх бинарных режимов GT; возвращает каталог запуска.

    В output_dir — то, что нужно для загрузки и оценки модели: best.pt, last.pt, конфиги, tags.json,
    experiment.json и split.json. В results_dir (по умолчанию тот же каталог) — результаты: metrics.jsonl,
    results.json (метрики best.pt на validation при каждом пороге) и predictions_validation.json."""
    # Сохраняем работоспособность прежнего scripts/train_uhead.py, который передавал список классов третьим аргументом.
    if isinstance(data_root, list):
        _, data_root, output_dir, device, tags = data_root, output_dir, device, tags, "hard"
    if tags not in ("hard", "step", "sigmoid", "gaussian") or step_window < 0 or sigmoid_slope <= 0 or gaussian_sigma <= 0:
        raise ValueError("Некорректные параметры режима разметки")
    from online_vllm.train.features import FeatureSet
    from online_vllm.train.loop import lr_factor, seed_everything
    from .data import ReadinessDataset, collate, fit_normalization, labelled
    if train_config.loss.lm_weight:
        raise NotImplementedError("UHead обучается на кешированных признаках; loss.lm_weight должен быть 0")
    seed_everything(train_config.seed)
    tag_settings = {"tags": tags, "step_window": step_window, "sigmoid_slope": sigmoid_slope,
                    "gaussian_sigma": gaussian_sigma}
    model_name = "statehead" if uhead_config.temporal_detector == "mlp" else "uhead"
    train_fs = FeatureSet(Path(data_root) / train_config.features.train)
    eval_fs = FeatureSet(Path(data_root) / train_config.features.eval)
    for fs in (train_fs, eval_fs):
        fs.check(train_config.video)
    for key in ("model_id", "revision", "quantization", "temporal_patch", "adapter"):
        if train_fs.meta.get(key) != eval_fs.meta.get(key):  # adapter: только у признаков модели с LoRA
            raise ValueError(f"Train и eval features расходятся по {key}")
    train_items, train_dropped = labelled(train_fs, train_config, tags, step_window, sigmoid_slope, gaussian_sigma)
    eval_items, eval_dropped = labelled(eval_fs, train_config, tags, step_window, sigmoid_slope, gaussian_sigma)
    validation_items, test_items, split = split_by_video(eval_fs, eval_items, train_config.seed)
    logger.info("train: %d записей (%d исключено); eval: %d (%d исключено) -> validation %d вопросов / %d видео, "
                "test %d / %d", len(train_items), train_dropped, len(eval_items), eval_dropped,
                len(validation_items), len(split["validation_videos"]), len(test_items), len(split["test_videos"]))
    train_data = ReadinessDataset(train_fs, train_items, uhead_config.groups)
    validation_data = ReadinessDataset(eval_fs, validation_items, uhead_config.groups)
    model = (StateHead(uhead_config) if uhead_config.temporal_detector == "mlp" else UHead(uhead_config)).to(device)
    model.tag_settings = tag_settings
    model.set_normalization(fit_normalization(train_data))
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Обучаемых параметров: %d", trainable_parameters)
    output_dir = Path(output_dir) if flat_output else Path(output_dir) / tags
    results_dir = output_dir if results_dir is None else Path(results_dir) if flat_output else Path(results_dir) / tags
    for directory in {output_dir, results_dir}:
        if directory.exists() and any(directory.iterdir()) and not overwrite:
            raise FileExistsError(f"{directory} уже содержит запуск; укажите --overwrite для явной перезаписи")
        directory.mkdir(parents=True, exist_ok=True)
    (output_dir / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, o.warmup_steps, total_steps, o.schedule))
    log = open(results_dir / "metrics.jsonl", "w", encoding="utf-8")
    step, best_loss, best_epoch = 0, math.inf, None

    def record(metrics):
        log.write(json.dumps({"step": step, **tag_settings, **metrics}) + "\n")
        log.flush()
        logger.info("step %d: %s", step, metrics)

    for epoch in range(1, o.epochs + 1):
        model.train()
        losses = []
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
            if step >= total_steps:
                break
        # После эпохи — только validation loss (по нему выбирается best.pt); метрики считаются в конце обучения.
        validation_loss = _preserve_rng(lambda: predict(model, validation_data, train_config, device, o.batch_size,
                                                        with_rows=False)[0])
        record({"epoch": epoch, "train_loss": sum(losses) / len(losses), "lr": scheduler.get_last_lr()[0],
                "eval_loss": validation_loss})
        save_checkpoint(model, uhead_config, train_config, tag_settings, output_dir, metadata, "last.pt")
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            save_checkpoint(model, uhead_config, train_config, tag_settings, output_dir, metadata, "best.pt")
        if step >= total_steps:
            break

    # best.pt на validation при каждом пороге; test оценивается отдельно (python -m uhead.evaluate).
    model = load_checkpoint(output_dir, device, "best.pt")
    validation_loss, validation_rows = predict(model, validation_data, train_config, device, o.batch_size)
    validation = _readiness(validation_rows, train_config.eval.thresholds, eval_fs.meta, model=model_name, tags=tags)
    results = {
        "model": model_name, "feature_groups": uhead_config.groups, **tag_settings,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "validation": {"questions": len(validation_rows), "videos": len(split["validation_videos"]),
                       "loss": validation_loss, "classification": validation["classification"],
                       "thresholds": validation["thresholds"]},
    }
    (results_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                              encoding="utf-8")
    _write_predictions(results_dir / "predictions_validation.json", validation_rows, eval_fs.meta, model_name, tags,
                       "validation")
    record({"final": True, "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "validation_auroc": validation["classification"]["auroc"],
            "validation_auprc": validation["classification"]["auprc"]})
    log.close()
    return output_dir


def build_parser():
    """Создаёт CLI модуля; --tags обязателен, чтобы эксперименты были явно различимы."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", choices=("hard", "step", "sigmoid", "gaussian"), required=True)
    parser.add_argument("--step-window", type=int, default=5)
    parser.add_argument("--sigmoid-slope", type=float, default=0.65)
    parser.add_argument("--gaussian-sigma", type=float, default=2.0)
    parser.add_argument("--uhead", default="configs/uhead/default.yaml")
    parser.add_argument("--train", default="configs/train/group1.yaml")
    parser.add_argument("--output", required=True, help="каталог весов и конфигов (best.pt, last.pt, split.json)")
    parser.add_argument("--results", help="каталог метрик и предсказаний; по умолчанию тот же, что --output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--flat-output", action="store_true", help="Не добавлять каталог режима GT к --output")
    parser.add_argument("--overwrite", action="store_true", help="Разрешить запись в существующий каталог запуска")
    return parser


def main():
    """Точка запуска: python -m uhead.train --tags {hard,step,sigmoid,gaussian} ..."""
    args = build_parser().parse_args()
    if args.step_window < 0 or args.sigmoid_slope <= 0 or args.gaussian_sigma <= 0:
        raise ValueError("step-window должен быть >= 0, sigmoid-slope и gaussian-sigma — положительными")
    from online_vllm.train.config import TrainConfig
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = train(UHeadConfig.load(args.uhead), TrainConfig.load(args.train), Path(args.data_root), Path(args.output),
                   args.device, args.tags, args.step_window, args.sigmoid_slope, args.flat_output, args.overwrite,
                   Path(args.results) if args.results else None, gaussian_sigma=args.gaussian_sigma)
    print("Каталог запуска:", result)


if __name__ == "__main__":
    main()
