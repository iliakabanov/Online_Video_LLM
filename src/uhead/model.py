"""Causal Transformer UHead, адаптированный для бинарной готовности к ответу."""
import torch
from torch import nn

from .config import UHeadConfig
from online_vllm.readiness import ReadinessFusion


class UHead(nn.Module):
    """Детектор готовности: одна temporal-позиция соответствует одной паре кадров."""
    def __init__(self, config: UHeadConfig, classes: list[str] | None = None):
        super().__init__()
        config.validate()
        if config.temporal_detector != "transformer":
            raise ValueError("Для temporal_detector: mlp создавайте StateHead, а не UHead")
        if classes is not None:
            raise ValueError("Трёхклассовый UHead checkpoint несовместим с бинарным readiness detector")
        self.config, self.classes = config, None
        self.fusion_block = ReadinessFusion(config)
        layer = nn.TransformerEncoderLayer(d_model=config.model_dim, nhead=config.nhead, dropout=config.dropout,
                                           activation="gelu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.classifier = nn.Linear(config.model_dim, 1)

    def set_normalization(self, stats: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        """Копирует сохранённые train-статистики в persistent buffers checkpoint-а."""
        self.fusion_block.set_normalization(stats)

    def forward(self, features: dict[str, torch.Tensor], padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Возвращает бинарные logits [B, P]; causal mask исключает доступ к будущим парам кадров."""
        out = self.fusion_block(features)
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
