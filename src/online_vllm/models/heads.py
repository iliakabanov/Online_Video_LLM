import torch
from torch import nn

from ..readiness import ReadinessFusion

from .config import StateHeadConfig


class StateHead(nn.Module):
    """StateHead: legacy Qwen classifier либо бинарный MLP readiness-детектор.

    Вариант с ``input_dim, StateHeadConfig`` сохранён только для Qwen pipeline.
    Новый вариант принимает UHeadConfig и разделяет ReadinessFusion с UHead.
    """

    def __init__(self, input_dim_or_config, config=None):
        super().__init__()
        if config is None:
            config = input_dim_or_config
            if getattr(config, "temporal_detector", None) != "mlp":
                raise ValueError("Бинарный StateHead требует UHeadConfig с temporal_detector: mlp")
            self.readiness_config = config
            self.fusion_block = ReadinessFusion(config)
            self.classifier = nn.Sequential(nn.Linear(config.model_dim, config.model_dim), nn.GELU(),
                                            nn.Linear(config.model_dim, 1))
            self.legacy = False
            self.classes = None
            return
        input_dim = input_dim_or_config
        activation = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[config.activation]
        layers = []
        for width in config.hidden_dims:
            layers.extend([nn.Linear(input_dim, width), activation(), nn.Dropout(config.dropout)])
            input_dim = width
        layers.append(nn.Linear(input_dim, len(config.classes)))
        self.layers = nn.Sequential(*layers)
        self.legacy = True
        self.classes = list(config.classes)

    def set_normalization(self, stats):
        if self.legacy:
            raise RuntimeError("У legacy StateHead нет group-wise normalisation")
        self.fusion_block.set_normalization(stats)

    def forward(self, features, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self.legacy:
            if padding_mask is not None and padding_mask.shape != features[next(iter(features))].shape[:2]:
                raise ValueError("padding_mask должен иметь форму [B, P]")
            return self.classifier(self.fusion_block(features)).squeeze(-1)
        return self.layers(features)
