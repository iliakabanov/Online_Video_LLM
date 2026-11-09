"""Cached vision-encoder outputs of whole videos, so that LoRA training and evaluation never run the encoder.

LoRA trains only the language model, so the vision encoder (and its merger) is frozen: its output for a video is the
same at every step. It is computed once per video and stored; a prefix of k temporal patches takes the first k
patches' tokens, since the encoder does not mix temporal patches (windowed and full attention both stay inside one).
The cached tokens then replace the video placeholders of the text embeddings, as Qwen's own forward does.

<dir>/meta.json                  model, revision, frame size, frame stride, dtype: a cache answers for these only
<dir>/<video path>.safetensors   "embeds" [patches * tokens per patch, hidden], "grid" [3] = (patches, h, w) of the whole
                                 video, "second_per_grid_ts" [1]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from online_vllm.data.inputs import InputBuilder, format_prompt
from online_vllm.data.schema import Record
from online_vllm.data.video import read_every
from online_vllm.models.model import attention_kernels


class VisionCache:
    def __init__(self, path: str | Path, settings: Mapping[str, Any] | None = None):
        """With ``settings``: create the cache, or check that an existing one holds the same. Without: open an existing
        cache as it is (DataLoader workers do; the training script has checked it with ``for_model``)."""
        self.path = Path(path)
        meta = self.path / "meta.json"
        if settings is None:
            if not meta.exists():
                raise FileNotFoundError(f"{self.path} is not a vision cache (no meta.json)")
            self.settings = json.loads(meta.read_text(encoding="utf-8"))
            return
        self.settings = dict(settings)
        self.path.mkdir(parents=True, exist_ok=True)
        if meta.exists():
            stored = json.loads(meta.read_text(encoding="utf-8"))
            if stored != self.settings:
                raise ValueError(f"{self.path} caches other settings: {stored}, not {self.settings}")
        else:
            meta.write_text(json.dumps(self.settings, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def for_model(cls, path: str | Path, model, *, frame_size: Sequence[int], frame_stride: int) -> "VisionCache":
        backbone = model.config.backbone
        return cls(path, {"model_id": backbone.model_id, "revision": backbone.revision,
                          "dtype": str(model.qwen.model.visual.dtype), "frame_size": list(frame_size),
                          "frame_stride": frame_stride})

    def file(self, video: str) -> Path:
        return self.path / f"{video}.safetensors"

    def has(self, video: str) -> bool:
        return self.file(video).exists()

    def save(self, video: str, embeds: torch.Tensor, grid: torch.Tensor, second_per_grid_ts: torch.Tensor) -> None:
        from safetensors.torch import save_file
        path = self.file(video)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        save_file({"embeds": embeds.contiguous(), "grid": grid.reshape(3).contiguous(),
                   "second_per_grid_ts": second_per_grid_ts.reshape(1).contiguous()}, str(partial))
        os.replace(partial, path)  # an interrupted write never leaves a truncated entry behind

    def load(self, video: str) -> dict[str, torch.Tensor]:
        """The whole video's entry: "embeds", "grid" [3] and "second_per_grid_ts" [1]."""
        from safetensors.torch import load_file
        return load_file(str(self.file(video)))

    @staticmethod
    def patches(entry: Mapping[str, torch.Tensor]) -> int:
        return int(entry["grid"][0])

    @staticmethod
    def encode(builder: InputBuilder, prompt: str, entry: Mapping[str, torch.Tensor],
               patches: int | None = None) -> dict[str, Any]:
        """Model inputs for the prompt and the first ``patches`` temporal patches of a loaded entry (all by default),
        with ``video_embeds`` from the cache in place of ``pixel_values_videos``."""
        total = VisionCache.patches(entry)
        patches = total if patches is None else patches
        if not 1 <= patches <= total:
            raise ValueError(f"{patches} temporal patches requested, {total} cached")
        grid = entry["grid"].clone()
        grid[0] = patches
        inputs = dict(builder._encode_tokens(prompt, grid, entry["second_per_grid_ts"]))
        inputs["video_embeds"] = entry["embeds"][:patches * (entry["embeds"].shape[0] // total)]
        return inputs


class _VideoPixels(torch.utils.data.Dataset):
    """Item i: the processor's inputs for the i-th video (with the prompt of one of its questions)."""

    def __init__(self, records: Sequence[Record], data_root: Path, builder: InputBuilder, frame_size, frame_stride):
        self.records, self.data_root, self.builder = records, data_root, builder
        self.frame_size, self.frame_stride = tuple(frame_size), frame_stride

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        frames = read_every(self.data_root / record.video, self.frame_stride, self.frame_size)
        frames = frames[:len(frames) // self.builder.temporal_patch * self.builder.temporal_patch]
        prompt = format_prompt(record.question, record.choices)
        return record, prompt, dict(self.builder._encode(prompt, frames, record.fps / self.frame_stride))


def _as_is(item):
    return item


@torch.inference_mode()
def build_vision_cache(model, builder: InputBuilder, records: Sequence[Record], cache: VisionCache, *,
                       data_root: str | Path, frame_size: Sequence[int], frame_stride: int, device,
                       num_workers: int = 2) -> int:
    """Encode every video of ``records`` that the cache does not hold yet; returns how many were added.

    The first video is also encoded without pixels (``InputBuilder._encode_tokens``) and compared with the processor's
    own tokens, so that cached inputs are known to match the processor in use."""
    first: dict[str, Record] = {}
    for record in records:
        if not cache.has(record.video):
            first.setdefault(record.video, record)
    if not first:
        return 0
    qwen = model.qwen.model
    loader = DataLoader(_VideoPixels(list(first.values()), Path(data_root), builder, frame_size, frame_stride),
                        batch_size=None, num_workers=num_workers, collate_fn=_as_is)
    checked = False
    for record, prompt, inputs in tqdm(loader, total=len(first), desc="Vision cache", unit="video", dynamic_ncols=True):
        if not checked:
            tokens = builder._encode_tokens(prompt, inputs["video_grid_thw"], inputs["second_per_grid_ts"])
            for key in tokens:
                if not torch.equal(tokens[key], inputs[key]):
                    raise RuntimeError(f"Inputs without pixels differ from the processor's in {key}")
            checked = True
        grid = inputs["video_grid_thw"].to(device)
        with attention_kernels():
            embeds = qwen.get_video_features(inputs["pixel_values_videos"].to(device), grid,
                                             return_dict=True).pooler_output[0]
        cache.save(record.video, embeds.cpu(), inputs["video_grid_thw"][0], inputs["second_per_grid_ts"])
    return len(first)


def embed_cached_video(model, inputs: Mapping[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    """Qwen language-model inputs from token ids with cached video tokens: the text embeddings with the video
    placeholders replaced by ``video_embeds`` (as Qwen's forward does with the encoder's output), and the 3D positions
    from the same get_rope_index the probe features use. Returns the keyword arguments and the positions."""
    qwen = model.qwen.model
    ids = inputs["input_ids"]
    embeds = qwen.get_input_embeddings()(ids)
    video = inputs["video_embeds"].to(embeds.device, embeds.dtype)
    placeholders = ids == model.qwen.config.video_token_id
    if int(placeholders.sum()) != video.shape[0]:
        raise ValueError(f"{int(placeholders.sum())} video placeholders, {video.shape[0]} cached video tokens")
    embeds = embeds.masked_scatter(placeholders[..., None].expand_as(embeds), video)
    mask = inputs.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(ids)
    positions, _ = qwen.get_rope_index(ids, inputs["mm_token_type_ids"], video_grid_thw=inputs["video_grid_thw"],
                                       second_per_grid_ts=inputs["second_per_grid_ts"], attention_mask=mask)
    qwen.rope_deltas = None
    return {"inputs_embeds": embeds, "position_ids": positions, "attention_mask": mask}, positions
