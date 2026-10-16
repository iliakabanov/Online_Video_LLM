"""Training loop: one sequence per record, one forward pass, the state loss at every decision position.

Trains whatever requires grad: the state head alone on a frozen backbone (lora.enabled: false), or the head and LoRA.
The language loss on the answer is not implemented yet (loss.lm_weight must be 0).
"""
import json
import logging
import math
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..data.dataset import ProactiveDataset, collate
from ..data.inputs import IGNORE
from ..models import save_checkpoint
from .config import TrainConfig
from .loss import class_weights, focal_loss

logger = logging.getLogger(__name__)
LABELS = ("state_labels", "lm_labels")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup over `warmup` optimizer steps, then cosine decay to 0 at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def decision_loss(model, batch, config: TrainConfig, alpha):
    """Focal loss over all decision positions of the batch; also returns their logits and targets."""
    labels = batch["state_labels"]
    mask = labels != IGNORE
    inputs = {k: v for k, v in batch.items() if k not in LABELS}
    logits = model(inputs, mode="decisions", decision_mask=mask).state_logits
    target = labels[mask]
    return focal_loss(logits, target, config.loss.focal_gamma, alpha), logits, target


@torch.no_grad()
def evaluate(model, loader, config: TrainConfig, alpha, device) -> dict:
    """Loss, accuracy and per-class recall over the decision positions; confusion[true][predicted]."""
    classes = model.config.head.classes
    training = model.training
    model.eval()
    total, count = 0.0, 0
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.long)
    for batch in loader:
        loss, logits, target = decision_loss(model, {k: v.to(device) for k, v in batch.items()}, config, alpha)
        total += loss.item() * len(target)
        count += len(target)
        confusion.index_put_((target.cpu(), logits.argmax(-1).cpu()), torch.tensor(1), accumulate=True)
    model.train(training)
    metrics = {"eval_loss": total / count, "eval_accuracy": confusion.diag().sum().item() / count}
    for i, c in enumerate(classes):
        metrics[f"eval_recall_{c}"] = confusion[i, i].item() / max(1, confusion[i].sum().item())
    metrics["eval_confusion"] = confusion.tolist()
    return metrics


def train(model, builder, config: TrainConfig, data_root: Path, output_dir: Path, device) -> Path:
    """Run the training; returns the directory of the final checkpoint."""
    if config.loss.lm_weight > 0:
        raise NotImplementedError("The language loss is not implemented yet; set loss.lm_weight: 0")
    seed_everything(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "train.yaml")
    classes = model.config.head.classes
    alpha = class_weights(config.loss.class_alpha, classes)
    o = config.optim

    train_set = ProactiveDataset(Path(data_root) / config.data.train, data_root, builder, config, classes)
    eval_set = ProactiveDataset(Path(data_root) / config.data.eval, data_root, builder, config, classes)
    logger.info("train: %d sequences (%d dropped), eval: %d (%d dropped)",
                len(train_set), train_set.dropped, len(eval_set), eval_set.dropped)
    if config.data.eval_samples is not None and config.data.eval_samples < len(eval_set):
        eval_set = Subset(eval_set, sorted(random.Random(config.seed).sample(range(len(eval_set)),
                                                                             config.data.eval_samples)))
    collate_fn = partial(collate, pad_id=builder.processor.tokenizer.pad_token_id)
    train_loader = DataLoader(train_set, batch_size=o.batch_size, shuffle=True, num_workers=o.num_workers,
                              collate_fn=collate_fn, generator=torch.Generator().manual_seed(config.seed),
                              persistent_workers=o.num_workers > 0)
    eval_loader = DataLoader(eval_set, batch_size=o.batch_size, num_workers=o.num_workers, collate_fn=collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    logger.info("Trainable parameters: %d", sum(p.numel() for p in params))
    optimizer = torch.optim.AdamW(params, lr=o.lr, weight_decay=o.weight_decay)
    total = o.epochs * math.ceil(len(train_loader) / o.grad_accum)
    if o.max_steps is not None:
        total = min(total, o.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, o.warmup_steps, total))

    model.train()
    if not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()  # frozen: no dropout in the features the head learns from
    log = open(output_dir / "metrics.jsonl", "a", encoding="utf-8")
    step, running = 0, []

    def record(metrics):
        log.write(json.dumps({"step": step, **metrics}) + "\n")
        log.flush()
        logger.info("step %d: %s", step, {k: v for k, v in metrics.items() if k != "eval_confusion"})

    def save(name):
        path = output_dir / name
        save_checkpoint(model, path)
        config.save(path / "train.yaml")
        return path

    for epoch in range(o.epochs):
        for i, batch in enumerate(train_loader):
            loss, _, _ = decision_loss(model, {k: v.to(device) for k, v in batch.items()}, config, alpha)
            (loss / o.grad_accum).backward()
            running.append(loss.item())
            if (i + 1) % o.grad_accum and i + 1 < len(train_loader):
                continue
            torch.nn.utils.clip_grad_norm_(params, o.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % o.log_every == 0:
                record({"epoch": epoch, "loss": sum(running) / len(running), "lr": scheduler.get_last_lr()[0]})
                running = []
            if step % o.eval_every == 0:
                record(evaluate(model, eval_loader, config, alpha, device))
            if step % o.save_every == 0:
                save(f"step_{step}")
            if step >= total:
                break
        if step >= total:
            break
    record(evaluate(model, eval_loader, config, alpha, device))
    log.close()
    return save("final")
