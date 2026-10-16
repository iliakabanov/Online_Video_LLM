"""Feature extraction over the full video, resuming it, and training the head on the features (tiny Qwen)."""
import json

import numpy as np
import pytest
import torch

from online_vllm.data.inputs import IGNORE, format_prompt
from online_vllm.data.labels import plan
from online_vllm.data.schema import Interval, Record, save_jsonl
from online_vllm.data.video import read_every
from online_vllm.models import build_model, load_checkpoint
from online_vllm.train.config import TrainConfig
from online_vllm.train.features import FeatureSet, extract
from online_vllm.train.head import train_head
from test_train import TinyBuilder, head_only, write_video


@pytest.fixture
def setup(builder, tiny_base, tmp_path):
    data = tmp_path / "data"
    (data / "videos").mkdir(parents=True)
    records = []
    for i in range(4):
        write_video(data / "videos" / f"v{i}.mp4", n=17)  # 17 frames -> 8 whole patches, the last frame dropped
        records.append(Record(id=f"r{i}", dataset="toy", split="train", video=f"videos/v{i}.mp4", fps=25,
                              num_frames=17, question="question", choices=["green", "question"], ask_frame=0,
                              answers=[Interval(3 + i, 16, "green")]))
    save_jsonl(records, data / "train.jsonl")
    config = TrainConfig.from_dict({
        "data": {"train": "train.jsonl", "eval": "train.jsonl", "eval_samples": 2},
        "features": {"train": "features/train", "eval": "features/train"},
        "video": {"frame_stride": 1, "frame_size": [4, 8]},
        "labels": {"speak_delay": 2},
        "optim": {"lr": 1e-2, "warmup_steps": 1, "epochs": 3, "batch_size": 2, "grad_accum": 1, "log_every": 1,
                  "eval_every": 2,
                  "save_every": 100},
        "eval": {"taus": [0.5]},
    })
    model = build_model(head_only(tiny_base), local_files_only=True)
    return data, records, config, model, TinyBuilder(builder.processor)


def run_extract(setup, **kwargs):
    data, _, config, model, builder = setup
    return extract(model, builder, data / config.data.train, data, config.video, data / config.features.train, "cpu",
                   num_workers=0, flush_every=1, **kwargs)


def test_features_match_the_training_sequence(setup):
    data, records, config, model, builder = setup
    fs = FeatureSet(run_extract(setup))
    assert fs.patches == [8] * 4 and fs.features.shape == (32, 32)
    assert fs.meta["frame_size"] == [4, 8] and fs.meta["records"] == 4
    # The features of the full video, up to the speak patch, are those of the training sequence cut at speak.
    r = records[1]
    p = plan(r, 1, config.labels.speak_delay)
    frames = read_every(data / r.video, 1, (4, 8))[:p.num_frames]
    x = builder.build_sequence(format_prompt(r.question, r.choices), frames, 25, [0] * len(p.labels))
    with torch.no_grad():
        h = model.decision_features({k: v for k, v in x.items() if k not in ("state_labels", "lm_labels")},
                                    x["state_labels"] != IGNORE)
    torch.testing.assert_close(torch.from_numpy(fs.rows(1, len(p.labels)).astype(np.float32)), h,
                               rtol=1e-3, atol=1e-3)


def test_extraction_resumes(setup):
    out = run_extract(setup)
    full = np.load(out / "features.npy").copy()
    features = np.load(out / "features.npy", mmap_mode="r+")
    features[8:] = 0  # as if interrupted after the first record
    features.flush()
    del features
    (out / "done").write_text("1")
    run_extract(setup)
    np.testing.assert_array_equal(np.load(out / "features.npy"), full)
    data, _, config, model, builder = setup
    config.video.frame_size = [4, 4]
    with pytest.raises(ValueError, match="other settings"):
        extract(model, builder, data / config.data.train, data, config.video, out, "cpu", num_workers=0)


def test_head_on_features(setup, tmp_path):
    data, records, config, model, builder = setup
    out = run_extract(setup)
    (out / "done").write_text("3")
    with pytest.raises(ValueError, match="not finished"):
        FeatureSet(out)
    (out / "done").write_text("4")
    final = train_head(model.config, config, data, tmp_path / "out", "cpu")
    lines = [json.loads(line) for line in (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in lines if "loss" in m] == [1, 2, 3, 4, 5, 6]  # 4 records, batch 2, 3 epochs
    last = lines[-1]
    assert last["final"] and np.isfinite(last["eval_loss"])
    outcome = sum(last[f"online_0.5_{k}"] for k in ("premature", "early", "on_target", "late", "missed"))
    assert outcome == pytest.approx(1.0)
    # The head checkpoint loads as a full model and gives, on the first 3 patches only, what it gave on the features.
    restored = load_checkpoint(final, local_files_only=True).eval()
    fs = FeatureSet(data / config.features.train)
    r = records[0]
    frames = read_every(data / r.video, 1, (4, 8))[:6]
    x = builder.build_sequence(format_prompt(r.question, r.choices), frames, 25, [0, 0, 0])
    inputs = {k: v for k, v in x.items() if k not in ("state_labels", "lm_labels")}
    with torch.no_grad():
        online = restored(inputs, mode="decisions", decision_mask=x["state_labels"] != IGNORE).state_logits[-1]
        cached = restored.state_head(torch.from_numpy(fs.rows(0, 3)[-1].astype(np.float32)))
    torch.testing.assert_close(online, cached, rtol=1e-2, atol=1e-2)
