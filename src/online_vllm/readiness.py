"""Общие входные слои бинарных readiness-детекторов.

Этот модуль намеренно не зависит ни от StateHead, ни от UHead: обе
архитектуры используют один и тот же набор параметров и преобразований.
"""
from __future__ import annotations

import torch
from torch import nn


# confidence = raw probe vector plus один индикатор доступности на каждую букву выбора.
FEATURE_DIMS = {"lm": 24576, "attention": 12672, "entropy": 1728, "vision": 5120, "confidence": 19}
FEATURE_OUTPUT_DIMS = {"lm": 256, "attention": 256, "entropy": 64, "vision": 128, "confidence": 32}


class Projection(nn.Sequential):
    """Проекция одной группы cached features."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU())


class ReadinessFusion(nn.Module):
    """Нормализация, независимые проекции и fusion, общие для MLP и Transformer."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projections = nn.ModuleDict({
            name: Projection(FEATURE_DIMS[name], config.projection_hidden_dims.get(name, config.projection_dims[name] * 2),
                             config.projection_dims[name], config.dropout)
            for name in config.groups
        })
        self.fusion = nn.Sequential(nn.Linear(config.fusion_dim, config.model_dim), nn.LayerNorm(config.model_dim))
        for name in config.groups:
            self.register_buffer(f"{name}_mean", torch.zeros(FEATURE_DIMS[name]), persistent=True)
            self.register_buffer(f"{name}_std", torch.ones(FEATURE_DIMS[name]), persistent=True)

    def set_normalization(self, stats: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        if set(stats) != set(self.config.groups):
            raise ValueError("Статистики нормализации не соответствуют группам признаков")
        for name, (mean, std) in stats.items():
            if mean.shape != (FEATURE_DIMS[name],) or std.shape != mean.shape:
                raise ValueError(f"Некорректная форма статистики для {name}")
            getattr(self, f"{name}_mean").copy_(mean.float())
            # Колонка, постоянная на train, только центрируется: деление на eps превратило бы малое отличие
            # на val в значения порядка 1/eps.
            std = std.float()
            getattr(self, f"{name}_std").copy_(torch.where(std > self.config.normalization_eps, std,
                                                           torch.ones_like(std)))

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if set(features) != set(self.config.groups):
            raise ValueError(f"Ожидались группы {self.config.groups}, получены {sorted(features)}")
        first = features[self.config.groups[0]]
        if first.ndim != 3:
            raise ValueError("Признаки должны иметь форму [B, P, D]")
        batch, steps = first.shape[:2]
        projected = []
        for name in self.config.groups:
            value = features[name]
            if value.shape != (batch, steps, FEATURE_DIMS[name]):
                raise ValueError(f"{name}: ожидалась форма {(batch, steps, FEATURE_DIMS[name])}")
            normalized = (value.float() - getattr(self, f"{name}_mean")) / getattr(self, f"{name}_std").clamp_min(
                self.config.normalization_eps)
            projected.append(self.projections[name](normalized))
        return self.fusion(torch.cat(projected, dim=-1))
