"""3D rotary positions as in the original Qwen2.5-VL: time grows with the temporal patch."""
import pytest
import torch

from online_vllm.models import BackboneConfig, LoRAConfig, ModelConfig, build_model

# 4 text tokens, a video of 3 temporal patches x (1 x 2) merged tokens, <|vision_end|>, 1 text token.
IDS = torch.tensor([[5, 6, 7, 62] + [61] * 6 + [63, 8]])
TYPES = (IDS == 61).long() * 2
GRID = torch.tensor([[3, 2, 4]])


@pytest.fixture(scope="module")
def qwen(tiny_base):
    config = ModelConfig(backbone=BackboneConfig(model_id=tiny_base, dtype="float32"), lora=LoRAConfig(enabled=False))
    return build_model(config, local_files_only=True).qwen.model


def test_time_grows_with_the_patch(qwen):
    # One time step per patch: t = 0, 1, 2 after the 4 text tokens; text resumes after the largest position.
    second = 1 / qwen.config.vision_config.tokens_per_second
    pos, delta = qwen.get_rope_index(IDS, TYPES, video_grid_thw=GRID, second_per_grid_ts=torch.tensor([second]))
    assert pos[0, 0].tolist() == [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8]
    assert pos[1, 0].tolist() == [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 7, 8]
    assert pos[2, 0].tolist() == [0, 1, 2, 3, 4, 5, 4, 5, 4, 5, 7, 8]
    assert delta.tolist() == [[8 + 1 - 12]]
    # 0.08 s per patch (2 frames at 25 fps): far below one time step, so these three patches share t.
    pos, _ = qwen.get_rope_index(IDS, TYPES, video_grid_thw=GRID, second_per_grid_ts=torch.tensor([0.08]))
    assert pos[0, 0].tolist() == [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 6, 7]


def test_images_keep_their_start_time(qwen):
    # MMDuet2's turns (data/turns.py): every frame is an image. All its tokens share t = the position reached so far,
    # as in the original Qwen2.5-VL (transformers 5.3 doubles it); text resumes after the largest position.
    ids = torch.tensor([[5, 6, 7, 62] + [60] * 2 + [63, 62] + [60] * 2 + [63, 8]])
    types = (ids == 60).long()
    grids = torch.tensor([[1, 2, 4], [1, 2, 4]])
    pos, _ = qwen.get_rope_index(ids, types, image_grid_thw=grids)
    assert pos[0, 0].tolist() == [0, 1, 2, 3, 4, 4, 6, 7, 8, 8, 10, 11]
    assert pos[1, 0].tolist() == [0, 1, 2, 3, 4, 4, 6, 7, 8, 8, 10, 11]
    assert pos[2, 0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_padding_is_skipped(qwen):
    ids = torch.cat([torch.zeros(1, 2, dtype=torch.long), IDS], 1)
    mask = torch.cat([torch.zeros(1, 2, dtype=torch.long), torch.ones_like(IDS)], 1)
    types = torch.cat([torch.zeros(1, 2, dtype=torch.long), TYPES], 1)
    padded, _ = qwen.get_rope_index(ids, types, video_grid_thw=GRID, second_per_grid_ts=torch.tensor([1.0]),
                                    attention_mask=mask)
    plain, _ = qwen.get_rope_index(IDS, TYPES, video_grid_thw=GRID, second_per_grid_ts=torch.tensor([1.0]))
    assert torch.equal(padded[:, :, 2:], plain)
