"""Training sequences from common-schema records: one record -> one sequence -> one forward pass."""
from pathlib import Path

import torch

from .inputs import IGNORE, InputBuilder, format_prompt
from .labels import plan
from .schema import load_jsonl
from .video import read_every


class ProactiveDataset(torch.utils.data.Dataset):
    """Group-1 records whose speak patch fits into the video (the rest are counted in `dropped`)."""

    def __init__(self, path: Path, data_root: Path, builder: InputBuilder, train_config, classes: list[str]):
        self.data_root = Path(data_root)
        self.builder = builder
        self.video = train_config.video
        self.with_answer = train_config.loss.lm_weight > 0
        self.class_ids = {c: i for i, c in enumerate(classes)}
        self.items, self.dropped = [], 0
        for r in load_jsonl(Path(path)):
            p = plan(r, self.video.frame_stride, train_config.labels.speak_delay, builder.temporal_patch)
            if p is None:
                self.dropped += 1
            else:
                self.items.append((r, p))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r, p = self.items[i]
        frames = read_every(self.data_root / r.video, self.video.frame_stride, tuple(self.video.frame_size))
        if len(frames) < p.num_frames:
            raise ValueError(f"{r.id}: decoded {len(frames)} frames, need {p.num_frames}")
        return self.builder.build_sequence(
            format_prompt(r.question, r.choices), frames[:p.num_frames], r.fps / self.video.frame_stride,
            [self.class_ids[label] for label in p.labels], p.answer if self.with_answer else None,
        )


def collate(examples, pad_id: int):
    """Right-pad the token tensors; videos are concatenated in example order, as the processor does for a batch."""
    length = max(e["input_ids"].shape[1] for e in examples)

    def pad(key, value):
        return torch.cat([torch.nn.functional.pad(e[key], (0, length - e[key].shape[1]), value=value)
                          for e in examples])

    return {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": pad("attention_mask", 0),
        "state_labels": pad("state_labels", IGNORE),
        "lm_labels": pad("lm_labels", IGNORE),
        "pixel_values_videos": torch.cat([e["pixel_values_videos"] for e in examples]),
        "video_grid_thw": torch.cat([e["video_grid_thw"] for e in examples]),
        "second_per_grid_ts": torch.cat([torch.as_tensor(e["second_per_grid_ts"]) for e in examples]),
    }
