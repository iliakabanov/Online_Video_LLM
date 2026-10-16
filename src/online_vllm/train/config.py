"""Serializable training settings: which frames the model sees, how they are labelled, the loss."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class VideoConfig:
    frame_stride: int = 1
    frame_size: list[int] = field(default_factory=lambda: [252, 168])  # [width, height]


@dataclass
class LabelConfig:
    speak_delay: int = 10


@dataclass
class LossConfig:
    focal_gamma: float = 2.0
    class_alpha: dict[str, float] = field(default_factory=lambda: {"silent": 0.25, "prepare": 1.0, "speak": 1.0})
    lm_weight: float = 1.0


@dataclass
class TrainConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def validate(self):
        if self.video.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if len(self.video.frame_size) != 2 or any(s <= 0 for s in self.video.frame_size):
            raise ValueError("frame_size must be [width, height] with positive values")
        if self.labels.speak_delay < 0:
            raise ValueError("speak_delay must be >= 0")
        if self.loss.focal_gamma < 0 or self.loss.lm_weight < 0:
            raise ValueError("focal_gamma and lm_weight must be >= 0")
        if any(a < 0 for a in self.loss.class_alpha.values()):
            raise ValueError("class_alpha values must be >= 0")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        result = cls(
            video=VideoConfig(**values.pop("video", {})),
            labels=LabelConfig(**values.pop("labels", {})),
            loss=LossConfig(**values.pop("loss", {})),
        )
        if values:
            raise ValueError(f"Unknown training settings: {sorted(values)}")
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path):
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")
