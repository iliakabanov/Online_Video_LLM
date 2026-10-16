"""Serializable training settings: data, which frames the model sees, how they are labelled, the loss, optimization."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    train: str = "processed/clevrer_proactive/train_group1.jsonl"  # relative to DATA_ROOT
    eval: str = "processed/clevrer_proactive/val_group1.jsonl"
    eval_samples: int | None = 500  # a fixed random subset of eval used during training; null: all of it


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
    lm_weight: float = 0.0


@dataclass
class OptimConfig:
    lr: float = 1.0e-3
    weight_decay: float = 0.01
    warmup_steps: int = 50
    epochs: int = 1
    max_steps: int | None = None  # optimizer steps; stops before the end of the epochs if reached
    batch_size: int = 1
    grad_accum: int = 16
    grad_clip: float = 1.0
    num_workers: int = 2
    log_every: int = 10
    eval_every: int = 200
    save_every: int = 200


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    seed: int = 0

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
        o = self.optim
        if o.lr <= 0 or o.epochs < 1 or o.batch_size < 1 or o.grad_accum < 1 or o.warmup_steps < 0:
            raise ValueError("lr, epochs, batch_size, grad_accum must be positive and warmup_steps >= 0")
        if o.max_steps is not None and o.max_steps < 1:
            raise ValueError("max_steps must be null or positive")
        if min(o.log_every, o.eval_every, o.save_every) < 1:
            raise ValueError("log_every, eval_every, save_every must be positive")
        if self.data.eval_samples is not None and self.data.eval_samples < 1:
            raise ValueError("eval_samples must be null or positive")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        result = cls(
            data=DataConfig(**values.pop("data", {})),
            video=VideoConfig(**values.pop("video", {})),
            labels=LabelConfig(**values.pop("labels", {})),
            loss=LossConfig(**values.pop("loss", {})),
            optim=OptimConfig(**values.pop("optim", {})),
            seed=values.pop("seed", 0),
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
