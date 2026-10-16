"""Question-first, history-free Qwen inputs from uniformly sampled video prefixes."""
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoProcessor

IGNORE = -100


def format_prompt(question: str, choices: list[str]) -> str:
    return f"{question}\nOptions: {', '.join(choices)}"


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

    @property
    def temporal_patch(self) -> int:
        return self.processor.video_processor.temporal_patch_size

    def _encode(self, prompt: str, frames: np.ndarray, fps: float):
        """Chat template (prompt, then video, then the generation prompt) and the frames as one video, not resized:
        the frames are expected at their final size, so training and inference see the same visual tokens."""
        vp = self.processor.video_processor
        factor = vp.patch_size * vp.merge_size
        if frames.shape[1] % factor or frames.shape[2] % factor:
            raise ValueError(f"Frame height and width must be multiples of {factor}, got {frames.shape[1:3]}")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt}, {"type": "video"},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        n = len(frames)
        return self.processor(
            text=[text], videos=[frames], return_tensors="pt",
            do_sample_frames=False, do_resize=False,
            video_metadata=[{"total_num_frames": n, "fps": fps,
                             "frames_indices": list(range(n))}],
        )

    def build_observation(self, question: str, video: VideoPrefix, t: float):
        # Crop before visual encoding: no temporal patch can contain a future frame.
        return self._encode(question, video.until(t).frames, video.fps)

    def build_sequence(self, prompt: str, frames: np.ndarray, fps: float, labels: list[int],
                       answer: str | None = None):
        """One training sequence: prompt, all frames, then the answer and EOS if given.

        `frames` must be a whole number of temporal patches, `labels` one class id per patch. Adds to the inputs
        `state_labels` (the class id on the last visual token of each patch) and, with an answer, `lm_labels` (the
        answer tokens, HF convention: label j is token j, the loss shifts); IGNORE elsewhere.
        """
        if len(frames) % self.temporal_patch or len(labels) != len(frames) // self.temporal_patch:
            raise ValueError("frames must be whole temporal patches with one label per patch")
        inputs = self._encode(prompt, frames, fps)
        ids = inputs["input_ids"][0]
        t, h, w = inputs["video_grid_thw"][0].tolist()
        per_patch = h * w // self.processor.video_processor.merge_size ** 2
        tokenizer = self.processor.tokenizer
        video = (ids == tokenizer.convert_tokens_to_ids(self.processor.video_token)).nonzero()[:, 0]
        if t != len(labels) or len(video) != t * per_patch:
            raise ValueError("Unexpected video token layout")
        state = torch.full_like(ids, IGNORE)
        state[video[per_patch - 1::per_patch]] = torch.tensor(labels)
        lm = torch.full_like(ids, IGNORE)
        if answer is not None:
            answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False,
                                   return_tensors="pt").input_ids[0]
            ids = torch.cat([ids, answer_ids])
            state = torch.cat([state, torch.full_like(answer_ids, IGNORE)])
            lm = torch.cat([lm, answer_ids])
        inputs["input_ids"] = ids[None]
        inputs["attention_mask"] = torch.ones_like(ids)[None]
        inputs["state_labels"] = state[None]
        inputs["lm_labels"] = lm[None]
        return inputs
