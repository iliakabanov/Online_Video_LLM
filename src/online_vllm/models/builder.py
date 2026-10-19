"""Load Qwen and resolve LoRA targets to exact qualified names before injection."""
from copy import deepcopy
import logging
import re

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from . import positions
from .config import ModelConfig
from .model import VideoStateModel

logger = logging.getLogger(__name__)


def resolve_targets(base, config):
    prefixes = {
        "language_model": "model.language_model.",
        "vision_encoder": "model.visual.blocks.",
        "merger": "model.visual.merger.",
    }
    prefix = prefixes[config.component]
    targets, matched_modules, matched_layers = [], set(), set()
    for name, module in base.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if not name.startswith(prefix) or leaf not in config.modules:
            continue
        match = re.search(r"\.(?:layers|blocks)\.(\d+)\.", name)
        layer = int(match[1]) if match else None
        if config.layers is not None and layer not in config.layers:
            continue
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"LoRA target is not linear: {name}")
        targets.append(name)
        matched_modules.add(leaf)
        matched_layers.add(layer)
    if not targets or set(config.modules) != matched_modules:
        raise ValueError(f"Unmatched LoRA modules: {set(config.modules) - matched_modules}")
    if config.layers is not None and set(config.layers) != matched_layers:
        raise ValueError(f"Unmatched LoRA layers: {set(config.layers) - matched_layers}")
    return targets


def build_model(config: ModelConfig, *, local_files_only: bool = False, device: str | None = None) -> VideoStateModel:
    """Load the backbone straight onto `device` (default: CPU), without a full copy of the weights in RAM first."""
    config = deepcopy(config)
    config.validate()
    quantization = None
    if config.backbone.quantization == "8bit":
        if device is None or torch.device(device).type != "cuda":
            raise ValueError("8-bit quantization needs a CUDA device")
        # The vision encoder stays in dtype: it is small, and its output is what the language model attends to.
        # Skip patterns match from the start of the full module name.
        quantization = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["model.visual", "lm_head"])
        # Every int8 layer warns on every call that its bfloat16 inputs are cast to float16 (the kernel's type).
        logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.backbone.model_id,
        revision=config.backbone.revision,
        dtype=getattr(torch, config.backbone.dtype),
        local_files_only=local_files_only,
        attn_implementation="sdpa",
        device_map=device,
        quantization_config=quantization,
    )
    positions.install(base.model)  # transformers 5.3 gives all frames of a video the same positions
    # Preserve the resolved Hub commit in checkpoints even if the caller used a branch name.
    revision = getattr(base.config, "_commit_hash", None)
    if revision:
        config.backbone.revision = revision
    base.requires_grad_(False)
    if config.lora.enabled:
        targets = resolve_targets(base, config.lora)
        logger.info("LoRA targets (%d): %s", len(targets), targets)
        base = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora.r, lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout, bias="none", target_modules=targets,
        ))
    if quantization is not None:
        torch.cuda.empty_cache()  # int8 conversion leaves ~2 GB of freed loading buffers reserved on the GPU
    model = VideoStateModel(base, config)
    logger.info("Trainable parameters: %d / %d",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
                sum(p.numel() for p in model.parameters()))
    return model
