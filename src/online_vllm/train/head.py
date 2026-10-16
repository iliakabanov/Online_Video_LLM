"""Training the state head alone on extracted features (train/features.py): no backbone in the loop.

Labels are made from the records with the current speak_delay, so the features serve any delay. Evaluation reports the
per-position metrics and an online simulation: with a threshold tau the model answers at the first patch where
P(speak) > tau, which is
    premature   before the answer is visible (a patch labelled silent)
    early       while the answer is visible but before the speak patch (prepare)
    on_target   at the speak patch
    late        after it
    missed      never in the video
"""
import json
import logging
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from ..data.labels import SILENT, SPEAK, plan
from ..models import ModelConfig, save_head_checkpoint
from ..models.heads import StateHead
from .config import TrainConfig
from .features import FeatureSet
from .loop import lr_factor, seed_everything
from .loss import class_weights, focal_loss

logger = logging.getLogger(__name__)


def labelled(fs: FeatureSet, config: TrainConfig, classes: list[str]):
    """(record index, class id per patch up to speak) for records whose speak patch fits; and the number dropped."""
    class_ids = {c: i for i, c in enumerate(classes)}
    items, dropped = [], 0
    for i, r in enumerate(fs.records):
        p = plan(r, config.video.frame_stride, config.labels.speak_delay, fs.meta["temporal_patch"])
        if p is None:
            dropped += 1
        else:
            items.append((i, np.array([class_ids[label] for label in p.labels])))
    return items, dropped


def gather(fs: FeatureSet, items, full=False):
    """Features of the batch (only up to the speak patch unless full) as float32, and the concatenated labels."""
    x = np.concatenate([fs.rows(i, None if full else len(y)) for i, y in items])
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(np.concatenate([y for _, y in items]))


@torch.no_grad()
def evaluate(head, fs: FeatureSet, items, config: TrainConfig, alpha, classes, device, batch_records=256) -> dict:
    training = head.training
    head.eval()
    speak = classes.index(SPEAK)
    total, count = 0.0, 0
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.long)
    outcomes = {tau: [] for tau in config.eval.taus}
    for b in range(0, len(items), batch_records):
        batch = items[b:b + batch_records]
        x, _ = gather(fs, batch, full=True)
        probs = head(x.to(device)).float().softmax(-1).cpu()
        start = 0
        for i, y in batch:
            p = probs[start:start + fs.patches[i]]
            start += fs.patches[i]
            logits = torch.log(p[:len(y)].clamp_min(1e-12))  # log-probabilities give the same loss as logits
            target = torch.from_numpy(y)
            total += focal_loss(logits, target, config.loss.focal_gamma, alpha).item() * len(y)
            count += len(y)
            confusion.index_put_((target, p[:len(y)].argmax(-1)), torch.tensor(1), accumulate=True)
            visible = int(np.argmax(y != classes.index(SILENT)))  # first patch at which the answer is visible
            target_patch = len(y) - 1
            for tau in config.eval.taus:
                fired = (p[:, speak] > tau).nonzero()
                outcomes[tau].append((None if len(fired) == 0 else int(fired[0]), visible, target_patch))
    head.train(training)
    metrics = {"eval_loss": total / count, "eval_accuracy": confusion.diag().sum().item() / count}
    for i, c in enumerate(classes):
        metrics[f"eval_recall_{c}"] = confusion[i, i].item() / max(1, confusion[i].sum().item())
    frames_per_patch = fs.meta["temporal_patch"] * fs.meta["frame_stride"]
    for tau, rows in outcomes.items():
        kinds = {"premature": 0, "early": 0, "on_target": 0, "late": 0, "missed": 0}
        delays = []
        for fired, visible, target in rows:
            if fired is None:
                kinds["missed"] += 1
                continue
            kinds["premature" if fired < visible else "early" if fired < target else
                  "on_target" if fired == target else "late"] += 1
            if fired >= visible:
                delays.append((fired - target) * frames_per_patch)
        for k, v in kinds.items():
            metrics[f"online_{tau}_{k}"] = v / len(rows)
        metrics[f"online_{tau}_median_delay_frames"] = float(np.median(delays)) if delays else None
    metrics["eval_confusion"] = confusion.tolist()
    return metrics


def train_head(model_config: ModelConfig, config: TrainConfig, data_root: Path, output_dir: Path, device) -> Path:
    """Train the state head on cached features; returns the directory of the final head-only checkpoint."""
    if config.loss.lm_weight > 0:
        raise NotImplementedError("The language loss needs the backbone; set loss.lm_weight: 0")
    seed_everything(config.seed)
    train_fs = FeatureSet(Path(data_root) / config.features.train)
    eval_fs = FeatureSet(Path(data_root) / config.features.eval)
    backbone = ("model_id", "revision", "quantization")
    for fs in (train_fs, eval_fs):
        fs.check(config.video)
        if [fs.meta[k] for k in backbone] != [train_fs.meta[k] for k in backbone]:
            raise ValueError("Train and eval features come from different backbones")
    # The head will run on the backbone of its checkpoint: it must be the one that made the features.
    made, named = train_fs.meta, model_config.backbone
    if (made["model_id"], made["quantization"]) != (named.model_id, named.quantization):
        raise ValueError(f"Features come from {made['model_id']} (quantization {made['quantization']}), the model "
                         f"config names {named.model_id} (quantization {named.quantization})")
    model_config = deepcopy(model_config)
    model_config.backbone.revision = train_fs.meta["revision"]
    model_config.validate()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "train.yaml")
    classes = model_config.head.classes
    alpha = class_weights(config.loss.class_alpha, classes)
    train_items, dropped = labelled(train_fs, config, classes)
    eval_all, eval_dropped = labelled(eval_fs, config, classes)
    logger.info("train: %d records (%d dropped), eval: %d (%d dropped)",
                len(train_items), dropped, len(eval_all), eval_dropped)
    eval_items = eval_all
    if config.data.eval_samples is not None and config.data.eval_samples < len(eval_all):
        eval_items = [eval_all[j] for j in sorted(random.Random(config.seed).sample(range(len(eval_all)),
                                                                                    config.data.eval_samples))]

    head = StateHead(train_fs.meta["hidden_size"], model_config.head).to(device)
    o = config.optim
    optimizer = torch.optim.AdamW(head.parameters(), lr=o.lr, weight_decay=o.weight_decay)
    batches = math.ceil(len(train_items) / o.batch_size)
    total = o.epochs * math.ceil(batches / o.grad_accum)
    if o.max_steps is not None:
        total = min(total, o.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, o.warmup_steps, total))
    rng = np.random.default_rng(config.seed)
    log = open(output_dir / "metrics.jsonl", "a", encoding="utf-8")
    step, running = 0, []

    def record(metrics):
        log.write(json.dumps({"step": step, **metrics}) + "\n")
        log.flush()
        logger.info("step %d: %s", step, {k: v for k, v in metrics.items() if k != "eval_confusion"})

    def save(name):
        path = output_dir / name
        save_head_checkpoint(model_config, head, path)
        config.save(path / "train.yaml")
        return path

    head.train()
    for epoch in range(o.epochs):
        order = rng.permutation(len(train_items))
        for b in range(batches):
            x, y = gather(train_fs, [train_items[j] for j in order[b * o.batch_size:(b + 1) * o.batch_size]])
            loss = focal_loss(head(x.to(device)), y.to(device), config.loss.focal_gamma, alpha)
            (loss / o.grad_accum).backward()
            running.append(loss.item())
            if (b + 1) % o.grad_accum and b + 1 < batches:
                continue
            torch.nn.utils.clip_grad_norm_(head.parameters(), o.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % o.log_every == 0:
                record({"epoch": epoch, "loss": sum(running) / len(running), "lr": scheduler.get_last_lr()[0]})
                running = []
            if step % o.eval_every == 0:
                record(evaluate(head, eval_fs, eval_items, config, alpha, classes, device))
            if step % o.save_every == 0:
                save(f"step_{step}")
            if step >= total:
                break
        if step >= total:
            break
    record({"final": True, **evaluate(head, eval_fs, eval_all, config, alpha, classes, device)})
    log.close()
    return save("final")
