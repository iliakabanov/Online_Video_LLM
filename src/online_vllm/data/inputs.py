"""Question-first, history-free Qwen inputs from uniformly sampled video prefixes."""
from dataclasses import dataclass

import numpy as np
from transformers import AutoProcessor


@dataclass(frozen=True)
class VideoPrefix:
    frames: np.ndarray  # [N, H, W, 3], uint8; uniformly sampled from time zero
    fps: float

    def until(self, t: float) -> "VideoPrefix":
        if not np.isfinite(t) or t < 0 or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("t must be finite/nonnegative and fps finite/positive")
        if self.frames.ndim != 4 or self.frames.shape[-1] != 3 or self.frames.dtype != np.uint8:
            raise ValueError("frames must be uint8 [N, H, W, 3]")
        selected = self.frames[np.arange(len(self.frames)) / self.fps <= t]
        if len(selected) == 0:
            raise ValueError("No observed frames")
        return VideoPrefix(selected.copy(), self.fps)


class InputBuilder:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"

    @classmethod
    def from_config(cls, config, *, local_files_only=False):
        return cls(AutoProcessor.from_pretrained(
            config.backbone.model_id, revision=config.backbone.revision,
            local_files_only=local_files_only,
        ))

    def build_observation(self, question: str, video: VideoPrefix, t: float):
        # Crop before visual encoding: no temporal patch can contain a future frame.
        prefix = video.until(t)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": question}, {"type": "video"},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        n = len(prefix.frames)
        return self.processor(
            text=[text], videos=[prefix.frames], return_tensors="pt",
            do_sample_frames=False,
            video_metadata=[{"total_num_frames": n, "fps": prefix.fps,
                             "frames_indices": list(range(n))}],
        )
