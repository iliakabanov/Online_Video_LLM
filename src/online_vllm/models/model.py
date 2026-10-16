"""One shared Qwen; classification and language heads are mutually exclusive paths."""
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .config import ModelConfig
from .heads import StateHead


@dataclass
class StateOutput:
    state_logits: torch.Tensor


class VideoStateModel(nn.Module):
    def __init__(self, backbone, config: ModelConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config
        base = self.qwen
        self.state_head = StateHead(base.config.text_config.hidden_size, config.head)
        embedding = base.get_input_embeddings().weight
        self.state_head.to(device=embedding.device, dtype=embedding.dtype)

    @property
    def qwen(self):
        # PEFT injects adapters into the original modules; get_base_model does not disable them.
        return self.backbone.get_base_model() if self.config.lora.enabled else self.backbone

    def _fresh_inputs(self, inputs):
        if inputs.get("past_key_values") is not None:
            raise ValueError("Pass the full current sequence, not a cache from a previous call")
        # Qwen stores positional bookkeeping internally. Independent observations must not share it.
        self.qwen.model.rope_deltas = None
        return {**inputs, "use_cache": False, "return_dict": True}

    def forward(self, inputs: dict, *, mode: Literal["state", "language"] = "state"):
        if mode not in {"state", "language"}:
            raise ValueError(f"Unknown mode: {mode}")
        kwargs = self._fresh_inputs(inputs)
        if mode == "language":
            return self.backbone(**kwargs)
        if "labels" in inputs:
            raise ValueError("Classification targets belong in the external loss, not model inputs")
        ids = inputs["input_ids"]
        mask = inputs.get("attention_mask", torch.ones_like(ids))
        if mask.shape != ids.shape or mask.ndim != 2 or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("attention_mask must be a binary [B, S] mask matching input_ids")
        if ids.shape[1] == 0 or not mask.bool().any(dim=1).all():
            raise ValueError("Every input must contain at least one non-padding token")
        positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        last = positions.masked_fill(~mask.bool(), -1).amax(dim=1)
        # Calling Qwen.model avoids the vocabulary projection entirely.
        hidden = self.qwen.model(**kwargs).last_hidden_state
        features = hidden[torch.arange(ids.shape[0], device=ids.device), last]
        return StateOutput(self.state_head(features))

    @torch.inference_mode()
    def predict_state(self, inputs: dict) -> StateOutput:
        if self.training:
            raise RuntimeError("Call model.eval() before inference")
        return self(inputs, mode="state")

    @torch.inference_mode()
    def generate_answer(self, inputs: dict, generation_config=None, **kwargs) -> torch.Tensor:
        """Return new token IDs only. No classification head; no cache survives this call."""
        if self.training:
            raise RuntimeError("Call model.eval() before inference")
        self._fresh_inputs(inputs)
        if "labels" in inputs:
            raise ValueError("Generation input must not include labels")
        mask = inputs.get("attention_mask")
        if mask is not None and not mask[:, -1].bool().all():
            raise ValueError("Generation batches must use left padding")
        options = {**inputs, **kwargs, "return_dict_in_generate": False}
        if generation_config is not None:
            options["generation_config"] = generation_config
        # Default is full-prefix recomputation; a caller may enable cache within this answer.
        options.setdefault("use_cache", False)
        try:
            sequences = self.backbone.generate(**options)
            return sequences[:, inputs["input_ids"].shape[1]:]
        finally:
            self.qwen.model.rope_deltas = None
