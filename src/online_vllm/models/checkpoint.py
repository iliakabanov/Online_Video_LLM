"""Portable trainable weights; the original base is referenced, not duplicated."""
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from .builder import build_model
from .config import ModelConfig


def save_checkpoint(model, path: str | Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.config.save(path / "model.yaml")
    weights = {"head": model.state_head.state_dict()}
    if model.config.lora.enabled:
        weights["lora"] = get_peft_model_state_dict(model.backbone)
    torch.save(weights, path / "trainable.pt")


def save_head_checkpoint(config: ModelConfig, head, path: str | Path):
    """A head trained apart from the backbone (on cached features), in the same format as save_checkpoint."""
    if config.lora.enabled:
        raise ValueError("A head-only checkpoint must have lora.enabled: false")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    config.save(path / "model.yaml")
    torch.save({"head": head.state_dict()}, path / "trainable.pt")


def load_checkpoint(path: str | Path, *, local_files_only: bool = False, device: str | None = None):
    path = Path(path)
    model = build_model(ModelConfig.load(path / "model.yaml"), local_files_only=local_files_only, device=device)
    weights = torch.load(path / "trainable.pt", map_location="cpu", weights_only=True)
    model.state_head.load_state_dict(weights["head"], strict=True)
    if model.config.lora.enabled:
        expected = get_peft_model_state_dict(model.backbone)
        if expected.keys() != weights["lora"].keys():
            raise ValueError("Checkpoint adapter keys do not match the model configuration")
        set_peft_model_state_dict(model.backbone, weights["lora"])
    return model
