"""One shared Qwen; classification and language heads are mutually exclusive paths.

The state head is kept in float32 whatever the backbone dtype: its features are cast up, so small head updates are not
lost to bfloat16 rounding.
"""
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
        self.state_head.to(device=base.get_input_embeddings().weight.device)

    @property
    def qwen(self):
        # PEFT injects adapters into the original modules; get_base_model does not disable them.
        return self.backbone.get_base_model() if self.config.lora.enabled else self.backbone

    def _fresh_inputs(self, inputs):
        if inputs.get("past_key_values") is not None:
            raise ValueError("Pass the full current sequence, not a cache from a previous call")
        if ("pixel_values_videos" in inputs or "pixel_values" in inputs) and inputs.get("mm_token_type_ids") is None:
            # Without them Qwen silently falls back to 1D positions for the visual tokens.
            raise ValueError("Visual inputs need the processor's mm_token_type_ids")
        # Qwen stores positional bookkeeping internally. Independent observations must not share it.
        self.qwen.model.rope_deltas = None
        return {**inputs, "use_cache": False, "return_dict": True}

    def _hidden(self, kwargs):
        # Calling Qwen.model avoids the vocabulary projection entirely. A frozen backbone keeps no activations:
        # the head's backward pass does not need them.
        trainable = torch.is_grad_enabled() and any(p.requires_grad for p in self.backbone.parameters())
        with torch.set_grad_enabled(trainable):
            return self.qwen.model(**kwargs).last_hidden_state

    def decision_features(self, inputs: dict, decision_mask: torch.Tensor) -> torch.Tensor:
        """Backbone hidden states at every True position of decision_mask [B, S], flattened in batch then position
        order [N, hidden]: exactly what the state head sees."""
        if decision_mask is None or decision_mask.shape != inputs["input_ids"].shape:
            raise ValueError("decision_mask must match input_ids")
        return self._hidden(self._fresh_inputs(inputs))[decision_mask]

    def _classify(self, features):
        return self.state_head(features.to(next(self.state_head.parameters()).dtype))

    def forward(self, inputs: dict, *, mode: Literal["state", "decisions", "language"] = "state",
                decision_mask: torch.Tensor | None = None):
        """state: logits at the last non-padding position [B, C]. decisions: logits at every True position of
        decision_mask [B, S] in one pass over the whole sequence, flattened in batch then position order [N, C]."""
        if mode not in {"state", "decisions", "language"}:
            raise ValueError(f"Unknown mode: {mode}")
        kwargs = self._fresh_inputs(inputs)
        if mode == "language":
            return self.backbone(**kwargs)
        if "labels" in inputs:
            raise ValueError("Classification targets belong in the external loss, not model inputs")
        if mode == "decisions":
            return StateOutput(self._classify(self.decision_features(inputs, decision_mask)))
        ids = inputs["input_ids"]
        mask = inputs.get("attention_mask", torch.ones_like(ids))
        if mask.shape != ids.shape or mask.ndim != 2 or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("attention_mask must be a binary [B, S] mask matching input_ids")
        if ids.shape[1] == 0 or not mask.bool().any(dim=1).all():
            raise ValueError("Every input must contain at least one non-padding token")
        positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        last = positions.masked_fill(~mask.bool(), -1).amax(dim=1)
        hidden = self._hidden(kwargs)
        return StateOutput(self._classify(hidden[torch.arange(ids.shape[0], device=ids.device), last]))

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
