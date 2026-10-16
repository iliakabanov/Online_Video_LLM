"""State loss, the one-pass decisions mode and an end-to-end head-only training run on the tiny Qwen."""
import json

import imageio_ffmpeg
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from online_vllm.data.inputs import IGNORE, InputBuilder
from online_vllm.data.schema import Interval, Record, save_jsonl
from online_vllm.models import BackboneConfig, LoRAConfig, ModelConfig, StateHeadConfig, build_model, load_checkpoint
from online_vllm.train.config import TrainConfig
from online_vllm.train.loop import train
from online_vllm.train.loss import class_weights, focal_loss

TOY_TO_TINY = [(2, 60), (3, 61), (4, 62), (5, 63)]  # toy tokenizer special IDs -> tiny checkpoint vocabulary


def remap(x):
    ids = x["input_ids"].clone()
    for old, new in TOY_TO_TINY:
        x["input_ids"][ids == old] = new
    return x


class TinyBuilder(InputBuilder):
    def build_sequence(self, *args, **kwargs):
        return remap(super().build_sequence(*args, **kwargs))


def head_only(path):
    return ModelConfig(backbone=BackboneConfig(model_id=path, dtype="float32"), lora=LoRAConfig(enabled=False),
                       head=StateHeadConfig(hidden_dims=[16], dropout=0))


def test_focal_loss():
    logits, target = torch.randn(8, 3), torch.randint(0, 3, (8,))
    torch.testing.assert_close(focal_loss(logits, target, 0.0, torch.ones(3)), F.cross_entropy(logits, target))
    assert focal_loss(logits, target, 2.0, torch.ones(3)) < F.cross_entropy(logits, target)
    alpha = class_weights({"speak": 1.0, "silent": 0.25, "prepare": 0.5}, ["silent", "prepare", "speak"])
    assert alpha.tolist() == [0.25, 0.5, 1.0]
    with pytest.raises(ValueError, match="exactly the head classes"):
        class_weights({"silent": 1.0}, ["silent", "speak"])


def test_decisions_match_online_prefixes(builder, tiny_base):
    """Logits at every decision of one pass == logits of the model run online on the prefix ending there."""
    model = build_model(head_only(tiny_base), local_files_only=True).eval()
    frames = np.random.default_rng(0).integers(0, 256, (6, 8, 4, 3), dtype=np.uint8)
    x = remap(builder.build_sequence("question", frames, fps=25, labels=[0, 1, 2]))
    inputs = {k: v for k, v in x.items() if k not in ("state_labels", "lm_labels")}
    mask = x["state_labels"] != IGNORE
    with torch.no_grad():
        decisions = model(inputs, mode="decisions", decision_mask=mask).state_logits
    assert decisions.shape == (3, 3)
    _, h, w = x["video_grid_thw"][0].tolist()
    for p, pos in enumerate(mask[0].nonzero()[:, 0].tolist()):
        prefix = {
            "input_ids": x["input_ids"][:, :pos + 1], "attention_mask": x["attention_mask"][:, :pos + 1],
            "mm_token_type_ids": x["mm_token_type_ids"][:, :pos + 1],
            "pixel_values_videos": x["pixel_values_videos"][:(p + 1) * h * w],
            "video_grid_thw": torch.tensor([[p + 1, h, w]]), "second_per_grid_ts": x["second_per_grid_ts"],
        }
        torch.testing.assert_close(model.predict_state(prefix).state_logits[0], decisions[p])
    with pytest.raises(ValueError, match="mm_token_type_ids"):
        model.predict_state({k: v for k, v in inputs.items() if k != "mm_token_type_ids"})


def write_video(path, n=16, size=16):
    writer = imageio_ffmpeg.write_frames(str(path), (size, size), fps=25, macro_block_size=1)
    writer.send(None)
    for i in range(n):
        writer.send(np.full((size, size, 3), 15 * i, dtype=np.uint8).tobytes())
    writer.close()


def test_head_only_training_run(builder, tiny_base, tmp_path):
    data = tmp_path / "data"
    (data / "videos").mkdir(parents=True)
    records = []
    for i in range(4):
        write_video(data / "videos" / f"v{i}.mp4")
        records.append(Record(id=f"r{i}", dataset="toy", split="train", video=f"videos/v{i}.mp4", fps=25,
                              num_frames=16, question="question", choices=["green", "question"], ask_frame=0,
                              answers=[Interval(3 + i, 15, "green")]))
    records.append(Record(id="late", dataset="toy", split="train", video="videos/v0.mp4", fps=25, num_frames=16,
                          question="question", choices=["green"], ask_frame=0, answers=[Interval(14, 15, "green")]))
    save_jsonl(records, data / "train.jsonl")
    config = TrainConfig.from_dict({
        "data": {"train": "train.jsonl", "eval": "train.jsonl", "eval_samples": 2},
        "video": {"frame_stride": 1, "frame_size": [4, 8]},
        "labels": {"speak_delay": 2},
        "optim": {"lr": 1e-2, "warmup_steps": 1, "epochs": 3, "batch_size": 2, "grad_accum": 1, "num_workers": 0,
                  "log_every": 1, "eval_every": 2, "save_every": 100},
    })
    model = build_model(head_only(tiny_base), local_files_only=True)
    backbone = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    head = {k: v.clone() for k, v in model.state_head.state_dict().items()}
    tiny = TinyBuilder(builder.processor)

    final = train(model, tiny, config, data, tmp_path / "out", "cpu")

    for k, v in model.backbone.state_dict().items():
        torch.testing.assert_close(v, backbone[k])
    assert any(not torch.equal(v, head[k]) for k, v in model.state_head.state_dict().items())
    lines = [json.loads(line) for line in (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()]
    steps = [m for m in lines if "loss" in m]
    assert [m["step"] for m in steps] == [1, 2, 3, 4, 5, 6]  # 4 sequences ("late" is dropped), batch 2, 3 epochs
    evals = [m for m in lines if "eval_loss" in m]
    assert len(evals) == 4 and all(np.isfinite(m["eval_loss"]) for m in evals)
    assert sum(map(sum, evals[-1]["eval_confusion"])) > 0
    assert (final / "train.yaml").exists()
    restored = load_checkpoint(final, local_files_only=True)
    for k, v in restored.state_head.state_dict().items():
        torch.testing.assert_close(v, model.state_head.state_dict()[k])
