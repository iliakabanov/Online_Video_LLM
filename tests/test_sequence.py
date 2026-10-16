"""Training sequence layout on the real Qwen processor, and causality of the visual tokens in the tiny Qwen."""
import numpy as np
import pytest
import torch

from online_vllm.data.dataset import collate
from online_vllm.data.inputs import IGNORE
from online_vllm.models import BackboneConfig, ModelConfig, build_model


def frames(n, value=0):
    # 8 x 4 pixels, 2x2 patches, 2x2 merge -> 2 visual tokens per temporal patch.
    return np.full((n, 8, 4, 3), value, dtype=np.uint8)


def test_layout_with_answer(builder):
    tok = builder.processor.tokenizer.convert_tokens_to_ids
    x = builder.build_sequence("question", frames(6), fps=25, labels=[0, 1, 2], answer="green")
    ids = x["input_ids"][0].tolist()
    video = [tok("<|video_pad|>")] * 6
    assert ids == [tok("question"), tok("<|vision_start|>"), *video, tok("<|vision_end|>"), tok("assistant"),
                   tok("green"), tok("<|im_end|>")]
    assert x["video_grid_thw"].tolist() == [[3, 4, 2]]
    # The last visual token of each temporal patch carries its label.
    state = x["state_labels"][0].tolist()
    assert state == [IGNORE, IGNORE, IGNORE, 0, IGNORE, 1, IGNORE, 2, IGNORE, IGNORE, IGNORE, IGNORE]
    assert x["lm_labels"][0].tolist() == [IGNORE] * 10 + [tok("green"), tok("<|im_end|>")]
    assert x["attention_mask"].tolist() == [[1] * 12]


def test_layout_without_answer(builder):
    x = builder.build_sequence("question", frames(4), fps=25, labels=[0, 2])
    assert x["input_ids"].shape == (1, 8)
    assert (x["lm_labels"] == IGNORE).all()
    assert (x["state_labels"] != IGNORE).sum() == 2


def test_rejects_partial_patches_and_resizing(builder):
    with pytest.raises(ValueError, match="whole temporal patches"):
        builder.build_sequence("question", frames(5), fps=25, labels=[0, 0, 2])
    with pytest.raises(ValueError, match="multiples of 4"):
        builder.build_sequence("question", np.zeros((2, 6, 4, 3), np.uint8), fps=25, labels=[2])


def test_collate_pads_right(builder):
    a = builder.build_sequence("question", frames(6), fps=25, labels=[0, 1, 2], answer="green")
    b = builder.build_sequence("question", frames(2), fps=25, labels=[2], answer="green")
    batch = collate([a, b], pad_id=builder.processor.tokenizer.pad_token_id)
    assert batch["input_ids"].shape == (2, 12)
    assert batch["attention_mask"][1].tolist() == [1] * 8 + [0] * 4
    assert batch["state_labels"][1, 8:].eq(IGNORE).all() and batch["lm_labels"][1, 8:].eq(IGNORE).all()
    assert batch["video_grid_thw"].tolist() == [[3, 4, 2], [1, 4, 2]]
    assert batch["pixel_values_videos"].shape[0] == len(a["pixel_values_videos"]) + len(b["pixel_values_videos"])


def test_hidden_states_do_not_see_later_patches(builder, tiny_base):
    """One forward pass over the whole sequence: the state at the end of a patch must not depend on later frames,
    neither through the vision encoder nor through the language model."""
    model = build_model(ModelConfig(backbone=BackboneConfig(model_id=tiny_base, dtype="float32")),
                        local_files_only=True).eval()
    rng = np.random.default_rng(0)
    early = rng.integers(0, 256, (4, 8, 4, 3), dtype=np.uint8)
    hidden = []
    for late in (0, 255):
        x = builder.build_sequence("question", np.concatenate([early, frames(2, late)]), fps=25, labels=[0, 1, 2])
        ids = x["input_ids"].clone()
        for old, new in [(2, 60), (3, 61), (4, 62), (5, 63)]:  # toy special IDs -> tiny checkpoint vocabulary
            x["input_ids"][ids == old] = new
        inputs = {k: v for k, v in x.items() if k not in ("state_labels", "lm_labels")}
        model.qwen.model.rope_deltas = None
        with torch.no_grad():
            hidden.append(model.qwen.model(**inputs, use_cache=False).last_hidden_state[0])
        decisions = (x["state_labels"][0] != IGNORE).nonzero()[:, 0]
    # Tokens up to the end of patch 1 (the second decision) are identical; the last patch differs.
    end = decisions[1] + 1
    torch.testing.assert_close(hidden[0][:end], hidden[1][:end])
    assert not torch.allclose(hidden[0][decisions[2]], hidden[1][decisions[2]])
