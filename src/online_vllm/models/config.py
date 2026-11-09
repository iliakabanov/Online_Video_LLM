"""Serializable model settings; optimization and target construction live elsewhere."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class BackboneConfig:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    revision: str | None = None
    dtype: str = "bfloat16"
    quantization: str | None = None  # "8bit": the language model's linear layers in int8 (CUDA only)


@dataclass
class LoRAConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    component: str = "language_model"
    modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    layers: list[int] | None = None


@dataclass
class StateHeadConfig:
    classes: list[str] = field(default_factory=lambda: ["silent", "prepare", "speak"])
    hidden_dims: list[int] = field(default_factory=lambda: [512])
    activation: str = "gelu"
    dropout: float = 0.1


@dataclass
class ModelConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    head: StateHeadConfig = field(default_factory=StateHeadConfig)

    def validate(self):
        if self.backbone.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported backbone dtype")
        if self.backbone.quantization not in {None, "8bit"}:
            raise ValueError("quantization must be null or 8bit")
        if not self.head.classes or len(set(self.head.classes)) != len(self.head.classes):
            raise ValueError("classes must be nonempty and unique")
        if any(not isinstance(c, str) or not c for c in self.head.classes):
            raise ValueError("class names must be nonempty strings")
        if any(d <= 0 for d in self.head.hidden_dims):
            raise ValueError("hidden_dims must be positive")
        if self.head.activation not in {"gelu", "relu", "silu"}:
            raise ValueError("Unsupported head activation")
        if not 0 <= self.head.dropout < 1 or not 0 <= self.lora.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.lora.r <= 0 or self.lora.alpha <= 0 or not self.lora.modules:
            raise ValueError("LoRA rank, alpha and modules must be nonempty/positive")
        if self.lora.component not in {"language_model", "vision_encoder", "merger"}:
            raise ValueError("Unsupported LoRA component")
        if self.lora.layers is not None:
            if not self.lora.layers or any(i < 0 for i in self.lora.layers):
                raise ValueError("layers must be null or nonempty nonnegative indices")
            if self.lora.component == "merger":
                raise ValueError("merger does not have numbered transformer layers")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        result = cls(
            backbone=BackboneConfig(**values.pop("backbone", {})),
            lora=LoRAConfig(**values.pop("lora", {})),
            head=StateHeadConfig(**values.pop("head", {})),
        )
        if values:
            raise ValueError(f"Unknown model settings: {sorted(values)}")
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path):
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")
