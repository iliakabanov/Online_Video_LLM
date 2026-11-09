"""Конфигурация компактной causal-архитектуры UHead."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from online_vllm.readiness import FEATURE_DIMS, FEATURE_OUTPUT_DIMS


@dataclass
class UHeadConfig:
    """Настройки проекций, fusion и causal Transformer; порядок groups определяет конкатенацию."""
    groups: list[str] = field(default_factory=lambda: list(FEATURE_DIMS))
    projection_dims: dict[str, int] = field(default_factory=lambda: dict(FEATURE_OUTPUT_DIMS))
    projection_hidden_dims: dict[str, int] = field(default_factory=dict)
    model_dim: int = 512
    nhead: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    normalization_eps: float = 1.0e-6
    temporal_detector: str = "transformer"
    compact_model_dim: int = 128
    compact_nhead: int = 8
    compact_num_layers: int = 1
    compact_ffn_dim: int = 508

    def validate(self):
        if not self.groups or len(set(self.groups)) != len(self.groups) or set(self.groups) - set(FEATURE_DIMS):
            raise ValueError(f"groups must be a nonempty unique subset of {sorted(FEATURE_DIMS)}")
        if set(self.projection_dims) != set(FEATURE_DIMS):
            raise ValueError("projection_dims must name every known feature group")
        if set(self.projection_hidden_dims) - set(FEATURE_DIMS):
            raise ValueError("projection_hidden_dims contains an unknown feature group")
        if any(v <= 0 for v in self.projection_dims.values()) or any(v <= 0 for v in self.projection_hidden_dims.values()):
            raise ValueError("projection dimensions must be positive")
        if self.model_dim <= 0 or self.nhead <= 0 or self.model_dim % self.nhead:
            raise ValueError("model_dim must be positive and divisible by nhead")
        if self.num_layers < 1 or not 0 <= self.dropout < 1 or self.normalization_eps <= 0:
            raise ValueError("invalid Transformer, dropout, or normalization settings")
        if self.temporal_detector not in {"mlp", "transformer", "compact"}:
            raise ValueError("temporal_detector must be mlp, transformer, or compact")
        if (self.compact_model_dim <= 0 or self.compact_nhead <= 0
                or self.compact_model_dim % self.compact_nhead or self.compact_num_layers < 1
                or self.compact_ffn_dim <= 0):
            raise ValueError("invalid compact Transformer settings")

    @property
    def fusion_dim(self):
        return sum(self.projection_dims[g] for g in self.groups)

    @classmethod
    def from_dict(cls, values):
        result = cls(**dict(values))
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path):
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")
