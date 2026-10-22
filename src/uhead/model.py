"""Causal Transformer UHead, адаптированный для бинарной готовности к ответу."""
import torch
from torch import nn

from .config import UHeadConfig
from online_vllm.readiness import ReadinessFusion


class UHead(nn.Module):
    """Causal Transformer readiness detector, including the capacity-matched compact variant."""
    def __init__(self, config: UHeadConfig, classes: list[str] | None = None):
        super().__init__()
        config.validate()
        if config.temporal_detector not in {"transformer", "compact"}:
            raise ValueError("Для temporal_detector: mlp создавайте StateHead, а не UHead")
        if classes is not None:
            raise ValueError("Трёхклассовый UHead checkpoint несовместим с бинарным readiness detector")
        self.config, self.classes = config, None
        self.fusion_block = ReadinessFusion(config)
        if config.temporal_detector == "compact":
            self.temporal_adapter = nn.Linear(config.model_dim, config.compact_model_dim)
            encoder_dim, nhead = config.compact_model_dim, config.compact_nhead
            num_layers, ffn_dim = config.compact_num_layers, config.compact_ffn_dim
        else:
            self.temporal_adapter = nn.Identity()
            encoder_dim, nhead = config.model_dim, config.nhead
            num_layers, ffn_dim = config.num_layers, encoder_dim * 4
        layer = nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=nhead, dim_feedforward=ffn_dim,
                                           dropout=config.dropout, activation="gelu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(encoder_dim, 1)

    def set_normalization(self, stats: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        """Копирует сохранённые train-статистики в persistent buffers checkpoint-а."""
        self.fusion_block.set_normalization(stats)

    def forward(self, features: dict[str, torch.Tensor], padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Возвращает бинарные logits [B, P]; causal mask исключает доступ к будущим парам кадров."""
        out = self.temporal_adapter(self.fusion_block(features))
        batch, steps = out.shape[:2]
        causal_mask = torch.ones(steps, steps, device=out.device, dtype=torch.bool).triu_(diagonal=1)
        if padding_mask is not None and padding_mask.shape != (batch, steps):
            raise ValueError("padding_mask must have shape [B, P]")
        out = self.transformer_encoder(out, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.classifier(out).squeeze(-1)

    def forward_prefix(self, features: dict[str, torch.Tensor], prefix_length: int) -> torch.Tensor:
        """Запуск на произвольном непустом префиксе для online inference."""
        if prefix_length < 1:
            raise ValueError("prefix_length must be positive")
        return self({name: value[:, :prefix_length] for name, value in features.items()})
