"""Check actual processor packing and the boundary between observation and generation."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast, Qwen2_5_VLProcessor,
    Qwen2VLImageProcessor, Qwen2VLVideoProcessor,
)

from online_vllm.data.inputs import InputBuilder, VideoPrefix
from online_vllm.runtime import OnlineController
from online_vllm.models import BackboneConfig, ModelConfig, build_model


@pytest.fixture
def builder():
    tokens = ["<unk>", "<pad>", "<|image_pad|>", "<|video_pad|>",
              "<|vision_start|>", "<|vision_end|>", "question", "assistant"]
    tokenizer = Tokenizer(WordLevel({t: i for i, t in enumerate(tokens)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>",
        additional_special_tokens=tokens[2:6],
    )
    template = (
        "{{ messages[0]['content'][0]['text'] }} "
        "<|vision_start|><|video_pad|><|vision_end|>"
        "{% if add_generation_prompt %} assistant{% endif %}"
    )
    processor = Qwen2_5_VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(),
        video_processor=Qwen2VLVideoProcessor(
            size={"shortest_edge": 16, "longest_edge": 16},
            patch_size=2, temporal_patch_size=2, merge_size=2,
        ),
        chat_template=template,
    )
    return InputBuilder(processor)


def test_processor_excludes_future_and_packs_video(builder):
    frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
    video = VideoPrefix(frames, fps=2)
    first = builder.build_observation("question", video, t=0.5)
    frames[2:] = 255
    second = builder.build_observation("question", video, t=0.5)
    torch.testing.assert_close(first["pixel_values_videos"], second["pixel_values_videos"])
    assert first["video_grid_thw"].tolist() == [[1, 2, 2]]
    assert first["second_per_grid_ts"].tolist() == [1.0]
    assert first["input_ids"][0, -1].item() == builder.processor.tokenizer.convert_tokens_to_ids("assistant")
    frozen = video.until(0.5)
    frames[:2] = 255
    assert not frozen.frames.any()


def test_processed_video_through_loaded_model(builder, tiny_base):
    model = build_model(ModelConfig(backbone=BackboneConfig(
        model_id=tiny_base, dtype="float32",
    )), local_files_only=True).eval()
    inputs = builder.build_observation(
        "question", VideoPrefix(np.zeros((2, 4, 4, 3), dtype=np.uint8), 2), 0.5,
    )
    # Align toy tokenizer special IDs with the tiny checkpoint's vocabulary.
    ids = inputs["input_ids"].clone()
    for old, new in [(2, 60), (3, 61), (4, 62), (5, 63)]:
        inputs["input_ids"][ids == old] = new
    logits = model.predict_state(inputs).state_logits
    assert logits.shape == (1, 3) and torch.isfinite(logits).all()
    assert model.generate_answer(inputs, max_new_tokens=2).shape[0] == 1


@pytest.mark.parametrize("class_id", [0, 2])
def test_controller_routes_and_reuses_snapshot(builder, class_id):
    model = Mock()
    model.config.head.classes = ["silent", "prepare", "speak"]
    model.qwen.get_input_embeddings.return_value.weight = torch.zeros(1)
    logits = torch.zeros(1, 3)
    logits[0, class_id] = 1
    model.predict_state.return_value = SimpleNamespace(state_logits=logits)
    model.generate_answer.return_value = torch.tensor([[10, 11]])
    controller = OnlineController(model, builder)
    result = controller.observe("question", VideoPrefix(np.zeros((4, 4, 4, 3), dtype=np.uint8), 2), 0.5)
    assert result.class_id == class_id
    model.predict_state.assert_called_once()
    if class_id == 2:
        model.generate_answer.assert_called_once()
        assert model.generate_answer.call_args.args[0] is model.predict_state.call_args.args[0]
        assert result.answer_ids.tolist() == [[10, 11]]
    else:
        model.generate_answer.assert_not_called()
        assert result.answer_ids is None
