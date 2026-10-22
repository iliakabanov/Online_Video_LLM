"""3D rotary positions of Qwen2.5-VL as the model was trained with them.

In transformers 5.3 `Qwen2_5_VLModel.get_rope_index` gives every temporal patch of a video the same (t, h, w)
positions (the temporal index is a constant, and `int(second_per_grid_ts)` truncates 2 / 25 s to 0), so frames differ
only by their order in causal attention. The original Qwen2.5-VL indexes time as
    t = floor(patch index * seconds per patch * tokens_per_second)
(tokens_per_second = 2: one step per 0.5 s), h and w as the merged grid row and column, all offset by the position
reached so far; text after a visual block continues from its largest position + 1. An image has no time axis: all its
tokens share t = the position reached so far (transformers 5.3 doubles that t, which shifts every image of MMDuet2's
turns, data/turns.py). `install` puts this back on a loaded model, so every forward pass, generate included, uses it.
"""
import types

import torch


def vision_positions(start: int, grid_thw, merge: int, seconds_per_patch: float, tokens_per_second: float):
    """(3, T * H * W) positions of one image or video: t, h, w of the merged grid, offset by start."""
    t, h, w = (int(x) for x in grid_thw)
    h, w = h // merge, w // merge
    t_index = (torch.arange(t, dtype=torch.float64) * seconds_per_patch * tokens_per_second).long()
    t_index = t_index.view(-1, 1).expand(-1, h * w).flatten()
    h_index = torch.arange(h).view(1, -1, 1).expand(t, -1, w).flatten()
    w_index = torch.arange(w).view(1, 1, -1).expand(t, h, -1).flatten()
    return torch.stack([t_index, h_index, w_index]) + start


def rope_index(self, input_ids, mm_token_type_ids, image_grid_thw=None, video_grid_thw=None,
               second_per_grid_ts=None, attention_mask=None, **kwargs):
    """Drop-in for Qwen2_5_VLModel.get_rope_index: position ids [3, B, S] and rope deltas [B, 1]."""
    merge = self.config.vision_config.spatial_merge_size
    tokens_per_second = self.config.vision_config.tokens_per_second
    position_ids = torch.zeros(3, *input_ids.shape, dtype=input_ids.dtype, device=input_ids.device)
    grids = {1: iter(image_grid_thw if image_grid_thw is not None else []),
             2: iter(video_grid_thw if video_grid_thw is not None else [])}
    seconds = iter(second_per_grid_ts.tolist() if torch.is_tensor(second_per_grid_ts)
                   else second_per_grid_ts if second_per_grid_ts is not None else [])
    deltas = []
    for b in range(input_ids.shape[0]):
        keep = attention_mask[b].bool() if attention_mask is not None else torch.ones_like(input_ids[b], dtype=torch.bool)
        kinds = mm_token_type_ids[b][keep].tolist()
        parts, pos, i = [], 0, 0
        while i < len(kinds):
            j = i
            while j < len(kinds) and kinds[j] == kinds[i]:
                j += 1
            if kinds[i] == 0:
                parts.append(torch.arange(pos, pos + j - i).view(1, -1).expand(3, -1))
            else:
                # Images have no time axis; a video's patches are seconds_per_patch apart.
                step = next(seconds) if kinds[i] == 2 else 0.0
                parts.append(vision_positions(pos, next(grids[kinds[i]]), merge, step, tokens_per_second))
                if parts[-1].shape[1] != j - i:
                    raise ValueError("Visual tokens do not match the grid")
            pos = int(parts[-1].max()) + 1
            i = j
        positions = torch.cat(parts, dim=1) if parts else torch.zeros(3, 0, dtype=torch.long)
        position_ids[:, b, keep] = positions.to(position_ids)
        deltas.append(pos - len(kinds))
    return position_ids, torch.tensor(deltas, device=input_ids.device).unsqueeze(1)


def install(qwen_model):
    """Replace get_rope_index on a loaded Qwen2_5_VLModel instance (`Qwen2_5_VLForConditionalGeneration.model`)."""
    qwen_model.get_rope_index = types.MethodType(rope_index, qwen_model)
