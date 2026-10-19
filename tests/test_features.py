"""Feature extraction (train/features.py, FEATURES.md) against an independent reference, resuming, and training the
head on the features. Reference: plain forward passes with eager attention that returns full weights; the probe of
pair p against the sequence cut after pair p with the real template tail."""
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention, eager_attention_forward

from online_vllm.data.inputs import IGNORE, answer_letter, format_prompt
from online_vllm.data.schema import Interval, Record, save_jsonl
from online_vllm.data.video import read_every
from online_vllm.models import build_model, load_checkpoint
from online_vllm.train.config import TrainConfig
from online_vllm.train.features import FeatureSet, extract, probe_features
from online_vllm.train.head import train_head
from online_vllm.train.remote import DriveUploader
from test_train import TinyBuilder, head_only, write_video

FRAMES, PAIRS, SIZE = 17, 8, (4, 8)  # 17 frames -> 8 whole pairs; frames 4 wide, 8 high -> 2 tokens per pair


@pytest.fixture
def setup(builder, tiny_base, tmp_path):
    data = tmp_path / "data"
    (data / "videos").mkdir(parents=True)
    for v in range(2):
        write_video(data / "videos" / f"v{v}.mp4", n=FRAMES)
    # Two questions on the first video, one with three options on the second.
    specs = [(0, 3, ["green", "question"]), (0, 6, ["question", "green"]), (1, 4, ["green", "question", "A"])]
    records = [Record(id=f"r{i}", dataset="toy", split="train", video=f"videos/v{v}.mp4", fps=25, num_frames=FRAMES,
                      question="question", choices=choices, ask_frame=0, answers=[Interval(s, FRAMES - 1, "green")])
               for i, (v, s, choices) in enumerate(specs)]
    save_jsonl(records, data / "train.jsonl")
    config = TrainConfig.from_dict({
        "data": {"train": "train.jsonl", "eval": "train.jsonl", "eval_samples": 2},
        "features": {"train": "features/train", "eval": "features/train", "lm_layers": [1, 2], "vis_layers": [0],
                     "prev_pairs": 1, "top_tokens": 3, "inputs": ["lm_last@2"]},
        "video": {"frame_stride": 1, "frame_size": list(SIZE)},
        "labels": {"speak_delay": 2},
        "optim": {"lr": 1e-2, "warmup_steps": 1, "epochs": 3, "batch_size": 2, "grad_accum": 1, "log_every": 1,
                  "eval_every": 2, "save_every": 100},
        "eval": {"taus": [0.5]},
    })
    model = build_model(head_only(tiny_base), local_files_only=True).eval()
    # Random small weights give near-uniform attention that hardly depends on positions; sharpen it so the
    # comparisons below would notice wrong positions or masks.
    with torch.no_grad():
        for layer in model.qwen.model.language_model.layers:
            layer.self_attn.q_proj.weight.mul_(6)
            layer.self_attn.k_proj.weight.mul_(6)
    return data, records, config, model, TinyBuilder(builder.processor)


def run_extract(setup, **kwargs):
    data, _, config, model, builder = setup
    return extract(model, builder, data / config.data.train, data, config.video, config.features,
                   data / config.features.train, "cpu", num_workers=0, flush_every=1, **kwargs)


@contextmanager
def eager_weights(store):
    """Language attention computed eagerly; its full weights [H, Q, K] stored by layer."""
    local = ALL_ATTENTION_FUNCTIONS._local_mapping
    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def attention(module, query, key, value, attention_mask, **kwargs):
        if not isinstance(module, Qwen2_5_VLAttention):
            return sdpa(module, query, key, value, attention_mask, **kwargs)
        q_len, k_len = query.shape[2], key.shape[2]
        if attention_mask is None:
            attention_mask = torch.ones(q_len, k_len, dtype=torch.bool).tril(k_len - q_len)[None, None]
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.zeros(attention_mask.shape).masked_fill(~attention_mask, float("-inf"))
        out, weights = eager_attention_forward(module, query, key, value, attention_mask, **kwargs)
        store[module.layer_idx] = weights[0].float()
        return out, weights

    local["sdpa"] = attention
    try:
        yield
    finally:
        local.pop("sdpa")


def reference(model, x):
    """Hidden states of every layer (index k: after layer k, last: final norm), attention weights, lm logits."""
    weights = {}
    inputs = {k: v for k, v in x.items() if k not in ("state_labels", "lm_labels")}
    model.qwen.model.rope_deltas = None
    with torch.no_grad(), eager_weights(weights):
        out = model.qwen.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        logits = model.qwen.lm_head(out.last_hidden_state[0]).float()
    return [h[0] for h in out.hidden_states], weights, logits


def segments(x, prompt, tokenizer, end):
    """Group label of every token up to `end`: special, question, (pair, j), tail."""
    ids, types = x["input_ids"][0].tolist(), x["mm_token_type_ids"][0]
    video = (types == 2).nonzero()[:, 0].tolist()
    per_pair = len(video) // x["video_grid_thw"][0, 0].item()
    q = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    labels = []
    for i in range(len(ids)):
        if video[0] - 1 - q <= i < video[0] - 1:
            labels.append("question")
        elif video[0] <= i <= video[-1]:
            labels.append(("pair", (i - video[0]) // per_pair))
        elif i > video[-1] and i >= end:
            labels.append("tail")
        else:
            labels.append("special")
    return labels, video[0], per_pair


def groups(row, labels, pair, prev, with_tail):
    g = [0.0] * (4 + prev + with_tail)
    for w, label in zip(row.tolist(), labels):
        if label in ("special", "question"):
            g[0 if label == "special" else 1] += w
        elif label == "tail":
            g[-1] += w
        else:
            d = pair - label[1]
            g[2 + d if d <= prev else 3 + prev] += w * (d >= 0)
    return g


def entropy(row, visible):
    row = row[:visible].clamp_min(1e-30)
    return float(-(row * row.log()).sum() / np.log(visible))


@pytest.mark.parametrize("batch_size", [1, 2, 3])  # 3: prompts of different lengths and two videos in one batch
def test_features_match_reference(setup, batch_size):
    data, records, config, model, builder = setup
    config.features.batch_size = batch_size
    out = run_extract(setup)
    fs = FeatureSet(out)
    meta = fs.meta
    assert meta["records"] == 3 and meta["videos"] == 2 and meta["pairs"] == 3 * PAIRS and meta["max_choices"] == 3
    shard = fs.shards[0]  # the default 20 GB holds everything in one shard
    assert len(fs.shards) == 1 and shard["att_last"].shape == (3 * PAIRS, 2 * 4, 5)
    assert shard["att_probe"].shape[-1] == 6 and shard["vis_mean"].shape == (2 * PAIRS, 1, 16)
    tokenizer = builder.processor.tokenizer
    prev, heads, layers = 1, 4, 2
    close = dict(rtol=1e-2, atol=1e-2)

    for i in (0, 2):
        r = records[i]
        prompt = format_prompt(r.question, r.choices)
        frames = read_every(data / r.video, 1, SIZE)[:2 * PAIRS]
        x = builder.build_sequence(prompt, frames, 25, [0] * PAIRS)
        hidden, weights, _ = reference(model, x)
        labels, start, per_pair = segments(x, prompt, tokenizer, len(x["input_ids"][0]))
        stored = {name: np.asarray(fs.question(name, i)) for name in fs.meta["shapes"] if name != "vis_mean"}
        for p in range(PAIRS):
            d = start + (p + 1) * per_pair - 1
            span = slice(start + p * per_pair, d + 1)
            for li, k in enumerate((1, 2)):
                torch.testing.assert_close(torch.from_numpy(stored["lm_last"][p, li].astype(np.float32)),
                                           hidden[k][d], **close)
                torch.testing.assert_close(torch.from_numpy(stored["lm_mean"][p, li].astype(np.float32)),
                                           hidden[k][span].mean(0), **close)
            for layer in range(layers):
                for h in range(heads):
                    col = layer * heads + h
                    w = weights[layer][h]
                    np.testing.assert_allclose(stored["att_last"][p, col],
                                               groups(w[d], labels, p, prev, False), atol=2e-3)
                    mean = np.mean([groups(w[t], labels, p, prev, False) for t in range(span.start, span.stop)], 0)
                    np.testing.assert_allclose(stored["att_mean"][p, col], mean, atol=2e-3)
                    np.testing.assert_allclose(stored["att_entropy"][p, 0, col], entropy(w[d], d + 1),
                                               atol=2e-3)

        # The probe of pair p is the model reading the sequence cut after pair p, with the template tail.
        for p in (0, 3, PAIRS - 1):
            cut = builder.build_sequence(prompt, frames[:2 * (p + 1)], 25, [0] * (p + 1))
            hidden, weights, logits = reference(model, cut)
            last = len(cut["input_ids"][0]) - 1
            labels, start, per_pair = segments(cut, prompt, tokenizer, start + (p + 1) * per_pair)
            for li, k in enumerate((1, 2)):
                torch.testing.assert_close(
                    torch.from_numpy(stored["probe_hidden"][p, li].astype(np.float32)), hidden[k][last],
                    **close)
            log_probs = logits[last].log_softmax(-1)
            letters = [tokenizer(c, add_special_tokens=False).input_ids[0] for c in "ABC"[:len(r.choices)]]
            raw = stored["probe_logprob"][p]
            np.testing.assert_allclose(raw[:len(r.choices)], log_probs[letters].numpy(), atol=1e-3)
            assert np.isnan(raw[len(r.choices):3]).all()
            np.testing.assert_allclose(raw[3:6], log_probs.topk(3).values.numpy(), atol=1e-3)
            np.testing.assert_allclose(raw[6], float(-(log_probs.exp() * log_probs).sum()), atol=1e-3)
            for layer in range(layers):
                for h in range(heads):
                    w = weights[layer][h][last]
                    np.testing.assert_allclose(stored["att_probe"][p, layer * heads + h],
                                               groups(w, labels, p, prev, True), atol=2e-3)
                    np.testing.assert_allclose(stored["att_entropy"][p, 2, layer * heads + h],
                                               entropy(w, last + 1), atol=2e-3)

    # Vision: one row block per video, the mean over the patches of each pair after block 0.
    blocks = []
    hook = model.qwen.model.visual.blocks[0].register_forward_hook(lambda m, a, o: blocks.append(o))
    x = builder.build_sequence(format_prompt("question", ["green", "question"]),
                               read_every(data / "videos/v1.mp4", 1, SIZE)[:2 * PAIRS], 25, [0] * PAIRS)
    reference(model, x)
    hook.remove()
    torch.testing.assert_close(torch.from_numpy(fs.video(2)[:, 0].astype(np.float32)),
                               blocks[0].float().view(PAIRS, -1, 16).mean(1), **close)
    np.testing.assert_array_equal(fs.video(0), fs.video(1))  # one copy per video
    np.testing.assert_allclose(shard["att_last"][:].astype(np.float32).sum(-1), 1, atol=2e-3)
    assert ((shard["att_entropy"][:] >= 0) & (shard["att_entropy"][:] <= 1.001)).all()
    assert (np.load(out / "shard_00000" / "layout.npy")[:, 1] == 2).all()


def test_extraction_resumes(setup):
    out = run_extract(setup)
    shard = out / "shard_00000"
    names = list(json.loads((out / "meta.json").read_text())["shapes"]) + ["layout"]
    full = {k: np.load(shard / f"{k}.npy").copy() for k in names}
    for k in names:  # as if interrupted after the first video (two questions), before the shard was finished
        a = np.load(shard / f"{k}.npy", mmap_mode="r+")
        a[2 * PAIRS if k not in ("vis_mean", "layout") else (PAIRS if k == "vis_mean" else 2):] = 0
        a.flush()
        del a
    (out / "done").write_text("2")
    (shard / "complete").unlink()
    run_extract(setup)
    assert (shard / "complete").exists()
    for k in names:
        np.testing.assert_array_equal(np.load(shard / f"{k}.npy"), full[k])
    (out / "done").write_text("1")
    with pytest.raises(ValueError, match="video boundary"):
        run_extract(setup)
    (out / "done").write_text("3")
    data, _, config, model, builder = setup
    config.features.top_tokens = 4
    with pytest.raises(ValueError, match="other settings"):
        run_extract(setup)


class FakeDrive(DriveUploader):
    """DriveUploader with rclone replaced by copies into a local folder standing for the Drive folder."""

    def __init__(self, root, split):
        self.remote = self.folder = None
        self.split, self.keep_local, self.job = Path(split), False, None
        self.root = Path(root) / self.split.parent.name / self.split.name

    def _command(self, verb, source, target):
        copy = ("import os, shutil, sys; s, t = sys.argv[1:]; os.makedirs(os.path.dirname(t), exist_ok=True); "
                "shutil.copytree(s, t, dirs_exist_ok=True) if os.path.isdir(s) else shutil.copy(s, t)")
        return [sys.executable, "-c", copy, str(source), f"{self.root}{target}"]


def test_shards_uploaded_and_removed(setup, tmp_path):
    data, records, config, model, builder = setup
    config.features.shard_gb = 1e-9  # every video closes its own shard
    out = data / config.features.train
    drive = FakeDrive(tmp_path / "drive", out)
    run_extract(setup, uploader=drive)
    remote = tmp_path / "drive" / "features" / "train"
    assert sorted(p.name for p in remote.iterdir()) == ["done", "meta.json", "shard_00000", "shard_00001"]
    assert not list(out.glob("shard_*")) and drive.uploaded() == {"shard_00000", "shard_00001"}
    with pytest.raises(ValueError, match="missing"):
        FeatureSet(out)

    # Fetched back, the shards read as one set, equal to a run that kept everything in one local shard.
    for shard in remote.glob("shard_*"):
        shutil.copytree(shard, out / shard.name)
    fs = FeatureSet(out)
    config.features.shard_gb = 20
    local = FeatureSet(extract(model, builder, data / config.data.train, data, config.video, config.features,
                               tmp_path / "local" / "train", "cpu", num_workers=0))
    assert len(fs.shards) == 2 and len(local.shards) == 1 and fs.records == local.records
    for i in range(3):
        for name in fs.meta["shapes"]:
            if name != "vis_mean":
                np.testing.assert_array_equal(fs.question(name, i), local.question(name, i))
        np.testing.assert_array_equal(fs.video(i), local.video(i))

    # Shards finished locally but never uploaded (e.g. an interrupted upload) go up on the next run.
    config.features.shard_gb = 1e-9
    kept = tmp_path / "kept" / "train"
    extract(model, builder, data / config.data.train, data, config.video, config.features, kept, "cpu",
            num_workers=0)
    assert len(list(kept.glob("shard_*"))) == 2
    drive = FakeDrive(tmp_path / "drive2", kept)
    extract(model, builder, data / config.data.train, data, config.video, config.features, kept, "cpu",
            num_workers=0, uploader=drive)
    assert not list(kept.glob("shard_*")) and drive.uploaded() == {"shard_00000", "shard_00001"}


def test_probe_features():
    raw = np.array([[np.log(0.2), np.log(0.6), np.nan, -1.0, -2.0, -3.0, 0.5]], dtype=np.float32)
    f = probe_features(raw, 2, 3)
    np.testing.assert_allclose(f[0, :3], [0.75, 0.25, 0], atol=1e-6)        # sorted, padded to max choices
    np.testing.assert_allclose(f[0, 3:8], [0.75, 0.5, -(0.75 * np.log(0.75) + 0.25 * np.log(0.25)) / np.log(2),
                                           2 / 3, 0.8], atol=1e-6)            # max, lead, entropy, K, letter mass
    np.testing.assert_allclose(f[0, 8:], [-1, -2, -3, 0.5])


def test_head_on_features(setup, tmp_path):
    data, records, config, model, builder = setup
    out = run_extract(setup)
    (out / "done").write_text("2")
    with pytest.raises(ValueError, match="not finished"):
        FeatureSet(out)
    (out / "done").write_text("3")

    final = train_head(model.config, config, data, tmp_path / "out", "cpu")
    lines = [json.loads(line) for line in (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in lines if "loss" in m] == [1, 2, 3, 4, 5, 6]  # 3 questions, batch 2, 3 epochs
    assert lines[-1]["final"] and np.isfinite(lines[-1]["eval_loss"])
    # With the last layer at the decision token as input, the checkpoint is the full model's state head.
    restored = load_checkpoint(final, local_files_only=True).eval()
    fs = FeatureSet(out)
    r = records[0]
    x = builder.build_sequence(format_prompt(r.question, r.choices), read_every(data / r.video, 1, SIZE)[:6], 25,
                               [0, 0, 0])
    inputs = {k: v for k, v in x.items() if k not in ("state_labels", "lm_labels")}
    with torch.no_grad():
        online = restored(inputs, mode="decisions", decision_mask=x["state_labels"] != IGNORE).state_logits[-1]
        cached = restored.state_head(torch.from_numpy(fs.rows(0, 3, ["lm_last@2"])[-1]))
    # The cached features are float16; the trained head amplifies their rounding to ~1e-2 on the logits.
    torch.testing.assert_close(online, cached, rtol=3e-2, atol=3e-2)

    # Any combination of groups trains.
    config.features.inputs = ["lm_mean@1", "probe_hidden", "att_probe@2", "att_entropy", "probe", "vis_mean@0"]
    assert fs.dim(config.features.inputs) == 32 + 64 + 4 * 6 + 3 * 8 + (3 + 5 + 3 + 1) + 16
    train_head(model.config, config, data, tmp_path / "rich", "cpu")
    last = json.loads((tmp_path / "rich" / "metrics.jsonl").read_text().splitlines()[-1])
    assert np.isfinite(last["eval_loss"])
    assert answer_letter(["x", "y", "z"], "z") == "C"
