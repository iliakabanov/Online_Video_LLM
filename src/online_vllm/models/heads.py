import torch
from torch import nn

from .config import StateHeadConfig


class StateHead(nn.Module):
    """MLP returning unnormalized, mutually exclusive class scores."""

    def __init__(self, input_dim: int, config: StateHeadConfig):
        super().__init__()
        activation = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[config.activation]
        layers = []
        for width in config.hidden_dims:
            layers.extend([nn.Linear(input_dim, width), activation(), nn.Dropout(config.dropout)])
            input_dim = width
        layers.append(nn.Linear(input_dim, len(config.classes)))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)
