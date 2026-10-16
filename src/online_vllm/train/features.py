"""Backbone features at every decision position of the full video, computed once for training the state head.

The backbone is frozen while the head trains, so its features never change. One pass over the full video (question,
then every whole temporal patch) gives the feature at the end of each patch; the language model is causal and the vision
encoder does not mix patches, so the feature at patch p is the one the model has online right after patch p. Labels are
not stored: they depend on speak_delay and are made at training time (data/labels.py), and features past the speak
patch serve the online evaluation.

    <dir>/meta.json      backbone, frames the features were made from, hidden size, number of records and patches
    <dir>/index.jsonl    one line per record, in order: {"offset", "patches", "record"}
    <dir>/features.npy   float16 [total patches, hidden]; rows offset ... offset + patches - 1 belong to a record
    <dir>/done           records written so far; an interrupted extraction resumes from it
"""
import json
import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from ..data.dataset import collate
from ..data.inputs import IGNORE, InputBuilder, format_prompt
from ..data.labels import whole_patches
from ..data.schema import from_dict, load_jsonl, to_dict
from ..data.video import read_every

logger = logging.getLogger(__name__)
LABELS = ("state_labels", "lm_labels")


class FullVideoDataset(Dataset):
    """Each record as one sequence over all whole temporal patches of its video, without an answer."""

    def __init__(self, records, data_root: Path, builder: InputBuilder, video):
        self.records, self.data_root, self.builder, self.video = records, Path(data_root), builder, video

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        patches = whole_patches(r, self.video.frame_stride, self.builder.temporal_patch)
        n = patches * self.builder.temporal_patch
        frames = read_every(self.data_root / r.video, self.video.frame_stride, tuple(self.video.frame_size))
        if len(frames) < n:
            raise ValueError(f"{r.id}: decoded {len(frames)} frames, need {n}")
        # The labels only mark the decision positions (the last token of each patch).
        return self.builder.build_sequence(format_prompt(r.question, r.choices), frames[:n],
                                           r.fps / self.video.frame_stride, [0] * patches)


def _write_done(out: Path, n: int):
    tmp = out / "done.tmp"
    tmp.write_text(str(n))
    tmp.replace(out / "done")


def extract(model, builder: InputBuilder, records_path: Path, data_root: Path, video, out: Path, device,
            num_workers: int = 2, flush_every: int = 50, limit: int | None = None):
    """Write the features of every record of records_path to `out`; resumes an interrupted run."""
    out = Path(out)
    records = list(load_jsonl(Path(records_path)))[:limit]
    temporal_patch = builder.temporal_patch
    patches = [whole_patches(r, video.frame_stride, temporal_patch) for r in records]
    offsets = np.concatenate([[0], np.cumsum(patches)]).tolist()
    meta = {
        "model_id": model.config.backbone.model_id, "revision": model.config.backbone.revision,
        "dtype": model.config.backbone.dtype, "quantization": model.config.backbone.quantization,
        "source": str(records_path).replace("\\", "/"),
        "frame_stride": video.frame_stride, "frame_size": list(video.frame_size), "temporal_patch": temporal_patch,
        "hidden_size": model.qwen.config.text_config.hidden_size, "records": len(records), "patches": offsets[-1],
    }
    if (out / "meta.json").exists():
        if json.loads((out / "meta.json").read_text()) != meta:
            raise ValueError(f"{out} holds features of other settings; remove it or choose another directory")
        done = int((out / "done").read_text())
        features = np.load(out / "features.npy", mmap_mode="r+")
    else:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "index.jsonl", "w", encoding="utf-8") as f:
            for r, o, p in zip(records, offsets, patches):
                f.write(json.dumps({"offset": o, "patches": p, "record": to_dict(r)}, ensure_ascii=False) + "\n")
        features = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float16,
                                             shape=(offsets[-1], meta["hidden_size"]))
        _write_done(out, 0)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        done = 0
    logger.info("%s: %d / %d records done, %d patches in all", out, done, len(records), offsets[-1])
    if done == len(records):
        return out

    loader = DataLoader(Subset(FullVideoDataset(records, data_root, builder, video), range(done, len(records))),
                        batch_size=1, num_workers=num_workers,
                        collate_fn=partial(collate, pad_id=builder.processor.tokenizer.pad_token_id))
    model.eval()
    bar = tqdm(loader, desc=f"{out.parent.name}/{out.name}", unit="q", initial=done, total=len(records),
               dynamic_ncols=True)
    with torch.inference_mode():
        for i, batch in enumerate(bar, start=done):
            mask = (batch["state_labels"] != IGNORE).to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k not in LABELS}
            h = model.decision_features(inputs, mask)
            if len(h) != patches[i]:
                raise RuntimeError(f"{records[i].id}: {len(h)} decision positions, expected {patches[i]}")
            # int8 layers compute in float16: an activation beyond its range would turn into inf here.
            if not torch.isfinite(h).all():
                raise RuntimeError(f"{records[i].id} (record {i}): non-finite features")
            features[offsets[i]:offsets[i + 1]] = h.float().cpu().numpy().astype(np.float16)
            if (i + 1) % flush_every == 0 or i + 1 == len(records):
                features.flush()
                _write_done(out, i + 1)
                bar.set_postfix(saved=i + 1)
    return out


class FeatureSet:
    """Extracted features: the records in order, their row ranges, and the memory-mapped matrix."""

    def __init__(self, path: Path):
        path = Path(path)
        self.meta = json.loads((path / "meta.json").read_text())
        if int((path / "done").read_text()) != self.meta["records"]:
            raise ValueError(f"{path}: extraction is not finished")
        self.features = np.load(path / "features.npy", mmap_mode="r")
        self.records, self.offsets, self.patches = [], [], []
        with open(path / "index.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.records.append(from_dict(row["record"]))
                self.offsets.append(row["offset"])
                self.patches.append(row["patches"])

    def check(self, video):
        """The features must come from the frames the training config describes."""
        if (self.meta["frame_stride"], self.meta["frame_size"]) != (video.frame_stride, list(video.frame_size)):
            raise ValueError(f"Features made with frame_stride={self.meta['frame_stride']}, "
                             f"frame_size={self.meta['frame_size']}; the config has {video.frame_stride}, "
                             f"{list(video.frame_size)}")

    def rows(self, i: int, n: int | None = None) -> np.ndarray:
        """The first n (default: all) patch features of record i."""
        return self.features[self.offsets[i]:self.offsets[i] + (self.patches[i] if n is None else n)]
