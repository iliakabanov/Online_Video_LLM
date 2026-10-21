"""Backbone features for the state head, extracted once (the feature set and its sizes: FEATURES.md).

For every pair of frames p of a question there are two positions:
    decision  the last visual token of pair p;
    probe     the last token of the template tail ("<|vision_end|><|im_end|>\n<|im_start|>assistant\n") placed
              right after pair p: where the model would start answering if the video stopped at p.
The main pass runs once over the question and the whole video. The probe tails of all pairs then run in one small pass
on top of its key/value cache: the tail of pair p sees only the tokens up to the end of pair p (custom mask) and its
positions continue from there, as in the sequence cut after pair p. The language model is causal and the vision encoder
does not mix pairs, so every feature of pair p depends only on the question and the frames up to p.

Attention rows are never materialized as a whole: a wrapper around the attention function computes the rows of the
positions of interest (softmax of their scores over the visible keys) next to the normal fused attention, and keeps
only their sums over token groups and their entropy.

Questions are processed video by video (a video is decoded once) and written in shards of whole videos, each closed
once it reaches features.shard_gb; with features.upload a closed shard is copied to Google Drive and deleted locally
(train/remote.py).

    <dir>/meta.json             backbone, frames, layers, groups, array shapes, sizes, the shard plan
    <dir>/done                  questions written so far (always a whole number of videos); extraction resumes from it
    <dir>/uploaded.txt          shards whose Drive copy is complete (with features.upload)
    <dir>/shard_NNNNN/index.jsonl   one line per question: record, row offset and pair count, video row offset
    <dir>/shard_NNNNN/<group>.npy   one array per feature group; rows offset ... offset + pairs - 1 belong to a
                                    question (vis_mean: to a video)
    <dir>/shard_NNNNN/layout.npy    per question: first visual token, tokens per pair, question span, tail length
    <dir>/shard_NNNNN/complete      written when the shard is finished
"""
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention, repeat_kv

from ..data.dataset import collate
from ..data.inputs import LETTERS, InputBuilder, format_prompt
from ..data.labels import whole_patches
from ..data.schema import from_dict, load_jsonl, to_dict
from ..data.video import read_every
from ..models.model import attention_kernels

logger = logging.getLogger(__name__)
# Absolute key segments: the pairs follow from PAIR0 on, the probe tails come after the last pair.
SPECIAL, QUESTION, PAIR0 = 0, 1, 2
LM_GROUPS = ("lm_last", "lm_mean", "probe_hidden")
ATT_GROUPS = ("att_last", "att_mean", "att_probe")
QUESTION_ARRAYS = LM_GROUPS + ATT_GROUPS + ("att_entropy", "probe_logprob")


@dataclass
class Layout:
    num_pairs: int
    per_pair: int                 # visual tokens of one pair
    video_start: int              # first visual token; <|vision_start|> is right before it
    question: tuple[int, int]     # [start, end) of the prompt text
    tail: list[int]               # token ids after the video: the probe tail

    def decisions(self, device=None):
        return torch.arange(1, self.num_pairs + 1, device=device) * self.per_pair - 1 + self.video_start


def layout(x, prompt: str, tokenizer) -> Layout:
    """Where the question, the pairs and the tail are in a sequence of InputBuilder.build_sequence (no answer)."""
    ids, video = x["input_ids"][0], (x["mm_token_type_ids"][0] == 2).nonzero()[:, 0]
    pairs, h, w = x["video_grid_thw"][0].tolist()
    per_pair = len(video) // pairs
    start, end = int(video[0]), int(video[-1]) + 1
    if len(video) != pairs * per_pair or end - start != len(video):
        raise ValueError("Visual tokens are not one contiguous video")
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    q_end = start - 1
    q_start = q_end - len(prompt_ids)
    if ids[q_start:q_end].tolist() != prompt_ids:
        raise ValueError("The prompt is not found right before the video")
    return Layout(pairs, per_pair, start, (q_start, q_end), ids[end:].tolist())


class AttentionRecorder:
    """Group sums and normalized entropies of the attention rows of chosen query positions, per text layer.

    Groups of a row of pair p: special tokens, question, pair p, pairs p-1 ... p-prev_pairs, earlier pairs, and for the
    probe its own tail. The entropy is divided by log(number of visible keys), so it is in [0, 1] at any length."""

    def __init__(self, prev_pairs: int, chunk_rows: int = 1024):
        self.prev_pairs, self.chunk_rows = prev_pairs, chunk_rows
        self.tasks = None
        self.results = {}

    def set(self, tasks, tail: bool):
        """tasks: one (rows, row_pair, segments, num_pairs) per sequence of the batch; rows are query positions,
        segments the segment of every key position."""
        self.tasks, self.tail = tasks, tail
        self.results = {}

    def record(self, layer, query, key, mask, scaling):
        heads = query.shape[1]
        keys_all = repeat_kv(key, heads // key.shape[1])                      # [B, H, K, D]
        results = []
        for b, (rows_all, row_pair_all, segments, num_pairs) in enumerate(self.tasks):
            keys = keys_all[b].float()
            num_keys = keys.shape[1]
            onehot = F.one_hot(segments[:num_keys], PAIR0 + num_pairs + 1).float()
            groups, entropy = [], []
            for s in range(0, len(rows_all), self.chunk_rows):
                rows = rows_all[s:s + self.chunk_rows]
                allow = self._allowed(rows, num_keys, None if mask is None else mask[b])  # [R, K]
                scores = (query[b][:, rows].float() @ keys.transpose(1, 2)) * scaling
                probs = scores.masked_fill(~allow, float("-inf")).softmax(-1)  # [H, R, K]
                ent = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
                entropy.append(ent / allow.sum(-1).clamp_min(2).log())
                groups.append(self._groups(probs @ onehot, row_pair_all[s:s + self.chunk_rows], num_pairs))
            results.append((torch.cat(groups, 1), torch.cat(entropy, 1)))
        self.results[layer] = results

    @staticmethod
    def _allowed(rows, num_keys, mask):
        if mask is None:  # causal attention; right padding sits after every real token, so it is never visible
            return torch.arange(num_keys, device=rows.device)[None] <= rows[:, None]
        mask = mask[0][rows]
        return mask if mask.dtype == torch.bool else mask == 0

    def _groups(self, segment_sums, row_pair, num_pairs):
        pairs = segment_sums[..., PAIR0:PAIR0 + num_pairs]                    # [H, R, P]

        def at(values, index):  # values[h, r, index[r]], 0 where index < 0
            got = values.gather(-1, index.clamp_min(0).view(1, -1, 1).expand(values.shape[0], -1, 1))[..., 0]
            return got * (index >= 0)

        cols = [segment_sums[..., SPECIAL], segment_sums[..., QUESTION]]
        cols += [at(pairs, row_pair - d) for d in range(self.prev_pairs + 1)]
        cols.append(at(pairs.cumsum(-1), row_pair - self.prev_pairs - 1))  # pairs 0 .. p - prev_pairs - 1
        if self.tail:
            cols.append(segment_sums[..., PAIR0 + num_pairs])
        return torch.stack(cols, -1)                                         # [H, R, G]


@contextmanager
def recording(recorder: AttentionRecorder):
    """Route the language model's SDPA attention through the recorder (vision attention is left alone)."""
    local = ALL_ATTENTION_FUNCTIONS._local_mapping
    previous = local.get("sdpa")
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def attention(module, query, key, value, attention_mask, **kwargs):
        if recorder.tasks is not None and isinstance(module, Qwen2_5_VLAttention):
            scaling = kwargs.get("scaling")
            recorder.record(module.layer_idx, query, key, attention_mask,
                            module.scaling if scaling is None else scaling)
        return original(module, query, key, value, attention_mask, **kwargs)

    local["sdpa"] = attention
    try:
        yield
    finally:
        if previous is None:
            local.pop("sdpa", None)
        else:
            local["sdpa"] = previous


class FeatureExtractor:
    """All features of one question (FEATURES.md) from a sequence of InputBuilder.build_sequence without an answer."""

    def __init__(self, model, builder: InputBuilder, spec, max_choices: int):
        self.model, self.builder, self.spec = model, builder, spec
        self.qwen = model.qwen.model
        self.layers = list(self.qwen.language_model.layers)
        n = len(self.layers)
        if any(k > n for k in spec.lm_layers) or any(i >= len(self.qwen.visual.blocks) for i in spec.vis_layers):
            raise ValueError(f"The model has {n} language layers and {len(self.qwen.visual.blocks)} vision blocks")
        tokenizer = builder.processor.tokenizer
        letters = [tokenizer(c, add_special_tokens=False).input_ids for c in LETTERS[:max_choices]]
        if any(len(ids) != 1 for ids in letters):
            raise ValueError("Option letters must be single tokens")
        self.letters = torch.tensor([ids[0] for ids in letters])
        self.max_choices = max_choices
        self.attention = AttentionRecorder(spec.prev_pairs)
        self.pad_id = tokenizer.pad_token_id
        self._hidden, self._vision = {}, {}
        self._capture = None  # per sequence of the batch: (rows, (start, pairs, per_pair) or None)
        self._vision_split = None  # (patches per video, which videos to record, pairs) in the main pass
        self.hooks = [self.layers[k - 1].register_forward_hook(self._hidden_hook(k)) for k in spec.lm_layers if k < n]
        self.hooks += [self.qwen.visual.blocks[i].register_forward_hook(self._vision_hook(i)) for i in spec.vis_layers]

    def close(self):
        for h in self.hooks:
            h.remove()

    def _take(self, key, h):  # h: [B, S, D]
        self._hidden[key] = [
            (h[b, rows].float(), None if span is None else
             h[b, span[0]:span[0] + span[1] * span[2]].float().view(span[1], span[2], -1).mean(1))
            for b, (rows, span) in enumerate(self._capture)]

    def _hidden_hook(self, k):
        def hook(module, args, output):
            if self._capture is not None:
                self._take(k, output[0] if isinstance(output, tuple) else output)
        return hook

    def _vision_hook(self, i):
        def hook(module, args, output):
            if self._vision_split is not None:
                sizes, wanted, pairs = self._vision_split
                h = (output[0] if isinstance(output, tuple) else output).float()  # the batch's videos, one by one
                self._vision[i] = [part.view(pairs, -1, part.shape[-1]).mean(1) if want else None
                                   for part, want in zip(h.split(sizes), wanted)]
        return hook

    def _attention_arrays(self, b, mean_of, last_rows):
        """Sequence b: [P, layers * heads, G] for the last_rows (and the pair means), [P, layers * heads] entropies."""
        groups = torch.stack([self.attention.results[i][b][0] for i in range(len(self.layers))])  # [N, H, R, G]
        entropy = torch.stack([self.attention.results[i][b][1] for i in range(len(self.layers))])  # [N, H, R]
        n, h, _, g = groups.shape
        out = [groups[:, :, last_rows].permute(2, 0, 1, 3).reshape(-1, n * h, g),
               entropy[:, :, last_rows].permute(2, 0, 1).reshape(-1, n * h)]
        if mean_of is not None:
            pairs, per_pair = mean_of
            out += [groups.view(n, h, pairs, per_pair, g).mean(3).permute(2, 0, 1, 3).reshape(pairs, n * h, g),
                    entropy.view(n, h, pairs, per_pair).mean(3).permute(2, 0, 1).reshape(pairs, n * h)]
        return out

    def __call__(self, batch) -> list[dict[str, np.ndarray]]:
        """batch: (sequence of build_sequence without an answer, prompt, number of options, record the video) per
        question. Sequences with different numbers of pairs run as separate batches."""
        results = [None] * len(batch)
        by_pairs = {}
        for j, item in enumerate(batch):
            by_pairs.setdefault(item[0]["video_grid_thw"][0, 0].item(), []).append(j)
        for indices in by_pairs.values():
            for j, r in zip(indices, self._batch([batch[j] for j in indices])):
                results[j] = r
        return results

    @torch.inference_mode()
    def _batch(self, batch) -> list[dict[str, np.ndarray]]:
        device = self.qwen.get_input_embeddings().weight.device
        tokenizer = self.builder.processor.tokenizer
        lays = [layout(x, prompt, tokenizer) for x, prompt, _, _ in batch]
        pairs, per_pair, tail = lays[0].num_pairs, lays[0].per_pair, lays[0].tail
        if any(lay.per_pair != per_pair or lay.tail != tail for lay in lays):
            raise ValueError("Sequences of a batch must share the frame size and the template tail")
        size = len(batch)
        # Right padding: every pad comes after the real tokens, so plain causal attention is exact for them.
        x = collate([item[0] for item in batch], pad_id=self.pad_id)
        inputs = {k: x[k].to(device) for k in
                  ("input_ids", "mm_token_type_ids", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts")}
        length = inputs["input_ids"].shape[1]
        positions, _ = self.qwen.get_rope_index(
            inputs["input_ids"], inputs["mm_token_type_ids"], video_grid_thw=inputs["video_grid_thw"],
            second_per_grid_ts=inputs["second_per_grid_ts"], attention_mask=x["attention_mask"].to(device))
        decisions = torch.stack([lay.decisions(device) for lay in lays])            # [B, P]
        pair_of = torch.arange(pairs * per_pair, device=device) // per_pair
        segments, main_tasks = [], []
        for lay in lays:
            seg = torch.full((length,), SPECIAL, dtype=torch.long, device=device)
            seg[lay.question[0]:lay.question[1]] = QUESTION
            rows = torch.arange(lay.video_start, lay.video_start + pairs * per_pair, device=device)
            seg[rows] = PAIR0 + pair_of
            segments.append(seg)
            main_tasks.append((rows, pair_of, seg, pairs))

        # Main pass: questions and whole videos.
        self.attention.set(main_tasks, tail=False)
        self._hidden, self._vision = {}, {}
        self._capture = [(decisions[b], (lay.video_start, pairs, per_pair)) for b, lay in enumerate(lays)]
        self._vision_split = (inputs["video_grid_thw"].prod(-1).tolist(), [item[3] for item in batch], pairs)
        self.qwen.rope_deltas = None
        with recording(self.attention), attention_kernels():
            out = self.qwen(**inputs, position_ids=positions, use_cache=True, return_dict=True)
        self._vision_split = None
        n = len(self.layers)
        self._take(n, out.last_hidden_state)  # the last layer after the final norm
        lm = dict(self._hidden)
        last_rows = torch.arange(1, pairs + 1, device=device) * per_pair - 1
        main_att = [self._attention_arrays(b, (pairs, per_pair), last_rows) for b in range(size)]
        vision = dict(self._vision)

        # Probe pass: the tail after every pair, on top of the main pass's cache.
        t = len(tail)
        probe_of = torch.arange(pairs, device=device).repeat_interleave(t)
        step_of = torch.arange(t, device=device).repeat(pairs)
        reach = positions.max(0).values.cummax(-1).values                          # [B, S] largest position so far
        tail_positions = (reach.gather(1, decisions) + 1).repeat_interleave(t, 1) + step_of
        own_tail = (probe_of[:, None] == probe_of[None]) & (step_of[None] <= step_of[:, None])
        prefix = torch.arange(length, device=device)[None, None] <= decisions[:, probe_of][:, :, None]
        mask = torch.cat([prefix, own_tail.expand(size, -1, -1)], 2)[:, None]     # [B, 1, P * t, S + P * t]
        probe_rows = torch.arange(pairs, device=device) * t + t - 1
        self.attention.set([(probe_rows, torch.arange(pairs, device=device),
                             torch.cat([seg, torch.full((pairs * t,), PAIR0 + pairs, device=device)]), pairs)
                            for seg in segments], tail=True)
        self._hidden = {}
        self._capture = [(probe_rows, None)] * size
        with recording(self.attention), attention_kernels():
            probe = self.qwen(input_ids=torch.tensor(tail, device=device).repeat(pairs)[None].expand(size, -1),
                              position_ids=tail_positions[None].expand(3, -1, -1), attention_mask=mask,
                              past_key_values=out.past_key_values, use_cache=True, return_dict=True)
        self._capture = None
        self.attention.tasks = None
        final = probe.last_hidden_state[:, probe_rows]                              # [B, P, D]
        self._hidden[n] = [(final[b].float(), None) for b in range(size)]
        log_probs = self.model.qwen.lm_head(final).float().log_softmax(-1)          # [B, P, vocab]
        top = log_probs.topk(self.spec.top_tokens, -1).values
        vocab_entropy = -(log_probs.exp() * log_probs).sum(-1, keepdim=True)

        results = []
        for b, (lay, (_, _, num_choices, with_vision)) in enumerate(zip(lays, batch)):
            att_last, ent_last, att_mean, ent_mean = main_att[b]
            att_probe, ent_probe = self._attention_arrays(b, None, torch.arange(pairs, device=device))
            letters = torch.full((pairs, self.max_choices), float("nan"), device=device)
            letters[:, :num_choices] = log_probs[b][:, self.letters[:num_choices].to(device)]
            result = {
                "lm_last": torch.stack([lm[k][b][0] for k in self.spec.lm_layers], 1),
                "lm_mean": torch.stack([lm[k][b][1] for k in self.spec.lm_layers], 1),
                "probe_hidden": torch.stack([self._hidden[k][b][0] for k in self.spec.lm_layers], 1),
                "att_last": att_last, "att_mean": att_mean, "att_probe": att_probe,
                "att_entropy": torch.stack([ent_last, ent_mean, ent_probe], 1),
            }
            arrays = {k: v.cpu().numpy().astype(np.float16) for k, v in result.items()}
            arrays["probe_logprob"] = torch.cat([letters, top[b], vocab_entropy[b]], 1).cpu().numpy().astype(
                np.float32)
            if with_vision:
                arrays["vis_mean"] = torch.stack([vision[i][b] for i in self.spec.vis_layers], 1).cpu().numpy() \
                    .astype(np.float16)
            for k, v in arrays.items():
                finite = np.delete(v, np.s_[num_choices:self.max_choices], 1) if k == "probe_logprob" else v
                if not np.isfinite(finite).all():
                    raise RuntimeError(f"{k}: non-finite values (float16 overflow?)")
            arrays["layout"] = np.array([lay.video_start, per_pair, *lay.question, t], dtype=np.int32)
            results.append(arrays)
        return results


class VideoQuestions(Dataset):
    """Item: one video decoded once and the sequences of all its questions."""

    def __init__(self, records, groups, data_root: Path, builder: InputBuilder, video):
        self.records, self.groups, self.data_root, self.builder, self.video = records, groups, Path(data_root), \
            builder, video

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, g):
        first = self.records[self.groups[g][0]]
        frames = read_every(self.data_root / first.video, self.video.frame_stride, tuple(self.video.frame_size))
        items = []
        for i in self.groups[g]:
            r = self.records[i]
            n = whole_patches(r, self.video.frame_stride, self.builder.temporal_patch) * self.builder.temporal_patch
            if len(frames) < n:
                raise ValueError(f"{r.id}: decoded {len(frames)} frames, need {n}")
            prompt = format_prompt(r.question, r.choices)
            x = self.builder.build_sequence(prompt, frames[:n], r.fps / self.video.frame_stride,
                                            [0] * (n // self.builder.temporal_patch))
            items.append((i, prompt, x))
        return g, items


def _as_is(item):
    return item


def _write_done(out: Path, n: int):
    tmp = out / "done.tmp"
    tmp.write_text(str(n))
    tmp.replace(out / "done")


def _plan(groups, pairs, question_bytes, video_bytes, shard_bytes):
    """Consecutive videos grouped into shards, each closed once it reaches shard_bytes: lists of video indices."""
    shards, current, size = [], [], 0
    for g, group in enumerate(groups):
        current.append(g)
        size += sum(pairs[i] for i in group) * question_bytes + pairs[group[0]] * video_bytes
        if size >= shard_bytes:
            shards.append(current)
            current, size = [], 0
    if current:
        shards.append(current)
    return shards


def _shard_name(n: int) -> str:
    return f"shard_{n:05d}"


class _Shard:
    """The arrays of one shard: created empty, or reopened to continue an interrupted run."""

    def __init__(self, path: Path, videos, groups, records, pairs, shapes):
        self.path = Path(path)
        self.offsets, self.video_offsets, self.position = {}, {}, {}
        total = video_total = 0
        for g in videos:
            self.video_offsets[g], video_total = video_total, video_total + pairs[groups[g][0]]
            for i in groups[g]:
                self.position[i] = len(self.position)
                self.offsets[i], total = total, total + pairs[i]
        if (self.path / "layout.npy").exists():
            self.arrays = {k: np.load(self.path / f"{k}.npy", mmap_mode="r+") for k in list(shapes) + ["layout"]}
            return
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "index.jsonl", "w", encoding="utf-8") as f:
            for g in videos:
                for i in groups[g]:
                    f.write(json.dumps({"offset": self.offsets[i], "patches": pairs[i],
                                        "video_offset": self.video_offsets[g], "record": to_dict(records[i])},
                                       ensure_ascii=False) + "\n")
        self.arrays = {}
        for k, shape in shapes.items():
            rows = video_total if k == "vis_mean" else total
            dtype = np.float32 if k == "probe_logprob" else np.float16
            self.arrays[k] = np.lib.format.open_memmap(self.path / f"{k}.npy", mode="w+", dtype=dtype,
                                                       shape=(rows, *shape))
        self.arrays["layout"] = np.lib.format.open_memmap(self.path / "layout.npy", mode="w+", dtype=np.int32,
                                                          shape=(len(self.position), 5))

    def write(self, i, g, features, first_of_video, pairs):
        for name in QUESTION_ARRAYS:
            self.arrays[name][self.offsets[i]:self.offsets[i] + pairs] = features[name]
        if first_of_video:
            self.arrays["vis_mean"][self.video_offsets[g]:self.video_offsets[g] + pairs] = features["vis_mean"]
        self.arrays["layout"][self.position[i]] = features["layout"]

    def flush(self):
        for a in self.arrays.values():
            a.flush()

    def close(self):
        self.flush()
        self.arrays = None  # release the memory maps: a mapped file cannot be deleted on Windows
        (self.path / "complete").write_text("")


def extract(model, builder: InputBuilder, records_path: Path, data_root: Path, video, spec, out: Path, device,
            num_workers: int = 1, flush_every: int = 20, videos: int | None = None, uploader=None,
            adapter: dict | None = None) -> Path:
    """Write the features of every question of records_path to `out`, video by video, in shards of spec.shard_gb;
    resumes an interrupted run. With an uploader (train/remote.py) every closed shard is copied to Google Drive and
    deleted locally, in the background while the next shard is computed. `videos`: only the questions of the first
    videos of the file, in its order (every question of each). `adapter` describes the LoRA adapter loaded
    into the model (its path, epoch and file hash) and goes into meta.json: features of a fine-tuned model never mix
    with those of the base model or of another adapter."""
    out = Path(out)
    records = list(load_jsonl(Path(records_path)))
    if videos is not None:
        kept = set(list(dict.fromkeys(r.video for r in records))[:videos])
        records = [r for r in records if r.video in kept]
    by_video = {}
    for i, r in enumerate(records):
        by_video.setdefault(r.video, []).append(i)
    groups = list(by_video.values())
    order = [i for g in groups for i in g]
    temporal_patch = builder.temporal_patch
    pairs = {i: whole_patches(records[i], video.frame_stride, temporal_patch) for i in order}
    total = sum(pairs.values())
    video_total = sum(pairs[g[0]] for g in groups)

    qwen = model.qwen
    layers = len(qwen.model.language_model.layers)
    heads = qwen.config.text_config.num_attention_heads
    hidden = qwen.config.text_config.hidden_size
    max_choices = max(len(r.choices) for r in records)
    g_att = 4 + spec.prev_pairs
    shapes = {
        "lm_last": (len(spec.lm_layers), hidden), "lm_mean": (len(spec.lm_layers), hidden),
        "probe_hidden": (len(spec.lm_layers), hidden),
        "att_last": (layers * heads, g_att), "att_mean": (layers * heads, g_att), "att_probe": (layers * heads, g_att + 1),
        "att_entropy": (3, layers * heads), "probe_logprob": (max_choices + spec.top_tokens + 1,),
        "vis_mean": (len(spec.vis_layers), qwen.config.vision_config.hidden_size),
    }
    question_bytes = sum(int(np.prod(shapes[k])) * (4 if k == "probe_logprob" else 2) for k in QUESTION_ARRAYS)
    video_bytes = int(np.prod(shapes["vis_mean"])) * 2
    plan = _plan(groups, pairs, question_bytes, video_bytes, spec.shard_gb * 1e9)
    meta = {
        "model_id": model.config.backbone.model_id, "revision": model.config.backbone.revision,
        "dtype": model.config.backbone.dtype, "quantization": model.config.backbone.quantization,
        "source": str(records_path).replace("\\", "/"),
        "frame_stride": video.frame_stride, "frame_size": list(video.frame_size), "temporal_patch": temporal_patch,
        "lm_layers": list(spec.lm_layers), "vis_layers": list(spec.vis_layers), "layers": layers, "heads": heads,
        "prev_pairs": spec.prev_pairs, "top_tokens": spec.top_tokens, "max_choices": max_choices,
        "attention_groups": ["special", "question", "pair"] + [f"pair-{d}" for d in range(1, spec.prev_pairs + 1)]
                            + ["earlier", "tail (probe only)"],
        "attention_rows": "layer-major: column = (layer - 1) * heads + head",
        "att_entropy": ["decision", "pair mean", "probe"],
        "probe_logprob": f"log P of the letters A.. (NaN past the options), top-{spec.top_tokens} log P, entropy",
        "shapes": {k: list(v) for k, v in shapes.items()},
        "records": len(records), "videos": len(groups), "pairs": total, "video_pairs": video_total,
        "bytes_per_pair": question_bytes, "vis_bytes_per_pair": video_bytes,
        "bytes": total * question_bytes + video_total * video_bytes,
        "shard_gb": spec.shard_gb, "shards": [[s[0], s[-1] + 1] for s in plan],  # [first video, end video)
    }
    if adapter is not None:  # absent for the base model, so that its earlier extractions still resume
        meta["adapter"] = adapter
    if (out / "meta.json").exists():
        if json.loads((out / "meta.json").read_text()) != meta:
            raise ValueError(f"{out} holds features of other settings; remove it or choose another directory")
        done = int((out / "done").read_text())
    else:
        out.mkdir(parents=True, exist_ok=True)
        _write_done(out, 0)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        done = 0
    finished = np.cumsum([len(g) for g in groups]).tolist()
    if done and done not in finished:
        raise ValueError(f"{out}/done = {done} is not at a video boundary")
    first = finished.index(done) + 1 if done else 0
    logger.info("%s: %d / %d questions done, %d videos, %d pairs, %d shards, %.1f GB in all", out, done,
                len(records), len(groups), total, len(plan), meta["bytes"] / 1e9)
    shard_of = {g: n for n, s in enumerate(plan) for g in s}
    if uploader is not None:
        uploader.put(out / "meta.json")
        uploaded = uploader.uploaded()
        for n in range(len(plan)):  # shards finished earlier whose upload did not complete
            if (out / _shard_name(n) / "complete").exists() and _shard_name(n) not in uploaded:
                uploader.submit(out / _shard_name(n))

    if done < len(records):
        extractor = FeatureExtractor(model, builder, spec, max_choices)
        loader = DataLoader(VideoQuestions(records, groups, data_root, builder, video), batch_size=None,
                            sampler=range(first, len(groups)), num_workers=num_workers, collate_fn=_as_is)
        model.eval()
        shard = None
        pending, open_videos = [], []  # questions not computed yet; [video, questions left] in order

        def run(batch, bar):
            """Compute and write a batch, then finish every video whose questions are all written."""
            nonlocal shard, done
            for (g, k, i, _, _), f in zip(batch, extractor([(x, prompt, len(records[i].choices), k == 0)
                                                             for g, k, i, prompt, x in batch])):
                shard.write(i, g, f, k == 0, pairs[i])
                next(v for v in open_videos if v[0] == g)[1] -= 1
                bar.update(1)
            while open_videos and open_videos[0][1] == 0:
                g = open_videos.pop(0)[0]
                done = finished[g]
                if g == plan[shard_of[g]][-1]:
                    shard.close()
                    shard = None
                    _write_done(out, done)
                    if uploader is not None:
                        uploader.submit(out / _shard_name(shard_of[g]))
                elif (g + 1) % flush_every == 0:
                    shard.flush()
                    _write_done(out, done)
                else:
                    continue
                bar.set_postfix(saved=done)

        try:
            with tqdm(total=len(records), initial=done, unit="q", desc=f"{out.parent.name}/{out.name}",
                      dynamic_ncols=True) as bar:
                for g, items in loader:
                    n = shard_of[g]
                    if shard is None:
                        shard = _Shard(out / _shard_name(n), plan[n], groups, records, pairs, shapes)
                    open_videos.append([g, len(items)])
                    pending += [(g, k, i, prompt, x) for k, (i, prompt, x) in enumerate(items)]
                    # A batch never spans two shards: the last video of a shard flushes what is pending.
                    while len(pending) >= spec.batch_size or (g == plan[n][-1] and pending):
                        batch, pending = pending[:spec.batch_size], pending[spec.batch_size:]
                        run(batch, bar)
        finally:
            extractor.close()
    if uploader is not None:
        uploader.wait()
        uploader.put(out / "done")
    return out


class FeatureSet:
    """Extracted features: the questions in processing order, their rows, and the memory-mapped arrays.

    `rows` concatenates the chosen inputs into one float32 vector per pair. An input is a group name, optionally with
    a layer: "lm_last@18", "lm_mean", "probe_hidden@36", "att_last", "att_probe@12", "att_entropy", "vis_mean@23",
    and "probe" for the features derived from probe_logprob (they do not depend on the order of the options):
    the option distribution sorted, its maximum, the lead over the second, its entropy / log K, K / max K,
    the probability mass on the letters, the top log-probabilities of the vocabulary, the vocabulary entropy.
    Language and attention layers count from 1, vision blocks from 0."""

    def __init__(self, path: Path):
        path = Path(path)
        self.meta = json.loads((path / "meta.json").read_text())
        if int((path / "done").read_text()) != self.meta["records"]:
            raise ValueError(f"{path}: extraction is not finished")
        self.shards, self.records, self.patches, self._rows = [], [], [], []
        for n in range(len(self.meta["shards"])):
            shard = path / _shard_name(n)
            if not (shard / "complete").exists():
                raise ValueError(f"{shard} is missing; if it was uploaded to Google Drive, fetch it with "
                                 "python scripts/sync_features.py download")
            self.shards.append({k: np.load(shard / f"{k}.npy", mmap_mode="r") for k in self.meta["shapes"]})
            with open(shard / "index.jsonl", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.records.append(from_dict(row["record"]))
                    self.patches.append(row["patches"])
                    self._rows.append((n, row["offset"], row["video_offset"]))

    def question(self, group: str, i: int) -> np.ndarray:
        """All pair rows of question i in one stored group (memory-mapped)."""
        n, offset, _ = self._rows[i]
        return self.shards[n][group][offset:offset + self.patches[i]]

    def video(self, i: int) -> np.ndarray:
        """The vis_mean rows of the video of question i."""
        n, _, offset = self._rows[i]
        return self.shards[n]["vis_mean"][offset:offset + self.patches[i]]

    def check(self, video):
        """The features must come from the frames the training config describes."""
        if (self.meta["frame_stride"], self.meta["frame_size"]) != (video.frame_stride, list(video.frame_size)):
            raise ValueError(f"Features made with frame_stride={self.meta['frame_stride']}, "
                             f"frame_size={self.meta['frame_size']}; the config has {video.frame_stride}, "
                             f"{list(video.frame_size)}")

    def _input(self, name, i, n):
        group, _, layer = name.partition("@")
        layer = int(layer) if layer else None
        if group == "probe":
            raw = self.question("probe_logprob", i)[:n]
            return probe_features(raw, len(self.records[i].choices), self.meta["max_choices"])
        if group not in self.meta["shapes"]:
            raise ValueError(f"Unknown feature group: {group}")
        a = np.asarray((self.video(i) if group == "vis_mean" else self.question(group, i))[:n])
        if layer is not None:
            if group in LM_GROUPS or group == "vis_mean":
                layers = self.meta["lm_layers" if group in LM_GROUPS else "vis_layers"]
                if layer not in layers:
                    raise ValueError(f"{group} has layers {layers}, not {layer}")
                a = a[:, layers.index(layer)]
            elif group in ATT_GROUPS or group == "att_entropy":
                if not 1 <= layer <= self.meta["layers"]:
                    raise ValueError(f"{group}: layers are 1 .. {self.meta['layers']}")
                heads = self.meta["heads"]
                a = (a[:, (layer - 1) * heads:layer * heads] if group in ATT_GROUPS
                     else a[:, :, (layer - 1) * heads:layer * heads])
            else:
                raise ValueError(f"{group} has no layers")
        return a.reshape(n, -1).astype(np.float32)

    def rows(self, i: int, n: int | None = None, inputs=("lm_last",)) -> np.ndarray:
        """The chosen inputs of the first n (default: all) pairs of question i, concatenated: float32 [n, dim]."""
        n = self.patches[i] if n is None else n
        return np.concatenate([self._input(name, i, n) for name in inputs], 1)

    def dim(self, inputs) -> int:
        return self.rows(0, 1, inputs).shape[1]


def probe_features(raw: np.ndarray, k: int, max_choices: int) -> np.ndarray:
    """Order-free features of the probe from probe_logprob rows [n, max_choices + top + 1] of a question with k options."""
    letters = raw[:, :k].astype(np.float64)
    top = letters.max(1, keepdims=True)
    weights = np.exp(letters - top)
    mass = np.exp(top[:, 0]) * weights.sum(1)
    dist = -np.sort(-weights / weights.sum(1, keepdims=True), 1)
    lead = dist[:, 0] - (dist[:, 1] if k > 1 else 0)
    entropy = -(dist * np.log(np.clip(dist, 1e-30, None))).sum(1) / np.log(k) if k > 1 else np.zeros(len(raw))
    padded = np.zeros((len(raw), max_choices))
    padded[:, :k] = dist
    return np.concatenate([padded, dist[:, :1], lead[:, None], entropy[:, None], np.full((len(raw), 1), k / max_choices),
                           mass[:, None], raw[:, max_choices:]], 1).astype(np.float32)
