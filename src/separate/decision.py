"""Separate decision module (StreamBridge-style, SB): a VLM of its own watches the stream and decides when the answering
model M_off should answer; it never answers itself.

One question is one sequence: the prompt, then every pair of the video followed by the decision token <act>:

    [prompt] [pair 0] <act> [pair 1] <act> ... [pair P-1] <act>
                      p_0            p_1                  p_{P-1}

p_k = sigmoid(head(h_k)), h_k the last hidden state at the <act> after pair k. The input of every slot is the same
token, so the history never depends on the earlier decisions and p_0 .. p_{P-1} of one pass are exactly what streaming
inference gives (DecisionSession); for any threshold the first p_k > tau is read off the recorded p_k. No answer: M_off
answers, and its answer at every pair comes with the questions (answer_correct of the readiness heads' predictions
files, eval/export.py). A pair is two source frames, 0.08 s, for every backbone: the target pair t* and the decision
points are those of the readiness heads and of the online model.

Two backbone families (``Family``: the sequence layout, the language-model pass and the streaming positions):
- QwenFamily, Qwen2.5-VL (SB-3B): the layout of the online SFT model (lora/online.py) with <listen> replaced by <act>
  and kept after the last pair as well: the prompt of the offline input, the video as one contiguous video (its visual
  tokens keep the 3D positions Qwen gives it, models/positions.py), the slot after pair k at the largest position so
  far + 1; visual tokens from the vision cache of the LoRA runs (lora/vision_cache.py).
- LlavaFamily, LLaVA-OneVision (SB-0.5B): SigLIP encodes every frame on its own; its features, pooled as LLaVA-OV
  pools (2 x 2: 196 per frame) and optionally further (``tokens_side``), are cached before the projector
  (build_llava_vision_cache), and the projector runs in the forward pass, so that it can train (``train_projector``:
  its whole copy trains with the LoRA). LLaVA-OV pools after the projector; pooling first keeps the cache small and
  is the same linear pooling on the other side of the projector's nonlinearity. A pair is both of its frames or only
  the last one (``frames_per_pair``). 1D positions. LLaVA-OV's chat template puts the video before the text, so the
  streaming prompt is its user turn with the question first (LLAVA_PROMPT), as for SB-3B; the image_newline that
  LLaVA-OV appends after a whole video is left out (a stream has no end).

Trainable: a fresh LoRA of the language model (model.yaml), the embedding row of <act> (PEFT trainable_token_indices),
LLaVA-OV's projector with ``train_projector`` (PEFT modules_to_save) and the head: StateHead with temporal_detector mlp on the group last_hidden, the architecture of the detector "MLP
last_hidden" on M_off's features (uhead), without the train-split normalization (the backbone is trained, so fixed
statistics would drift; the last hidden state is already RMS-normalized). Loss: BCE with logits over every slot of the
batch (the patch_mean of uhead.train), hard / step / sigmoid targets (uhead/labels.py).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import partial
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lora.lora import (
    EXPECTED_LORA_LEAVES,
    ClevrerLoRASplitConfig,
    LoRASFTConfig,
    _LengthGroupedBatches,
    _move_inputs,
    _prepare_output_dir,
    _records_for_first_videos,
    _set_epoch,
    _target_patch,
    _train_epoch,
    configure_gradient_checkpointing,
    load_lora_adapter,
)
from lora.online import (
    RopeSpec,
    Template,
    _best_threshold_line,
    _classification_line,
    _constrained_line,
    _latency_line,
    _question,
    online_hidden,
    template,
)
from lora.vision_cache import VisionCache
from online_vllm.data.inputs import InputBuilder, format_prompt
from online_vllm.data.labels import plan
from online_vllm.data.schema import Record
from online_vllm.data.video import read_every
from online_vllm.eval.thresholds import sweep
from online_vllm.models.config import ModelConfig
from online_vllm.models.heads import StateHead
from online_vllm.models.model import attention_kernels
from online_vllm.models.positions import vision_positions
from uhead.config import UHeadConfig

ACT = "<act>"
INIT_WORD = "ready"  # the new row starts at the mean embedding of this word
TAGS = ("hard", "step", "sigmoid")
QWEN, LLAVA = "qwen2_5_vl", "llava_onevision"  # model_type of the Hugging Face config
FAMILIES = (QWEN, LLAVA)
# The user turn of LLaVA-OV's own chat template ("<|im_start|>user <video>\n{text}<|im_end|>...", no system turn) with
# the question before the video, as the streaming input of every model here.
LLAVA_PROMPT = "<|im_start|>user {prompt}\n"
PAIR = 2  # source frames per pair: the unit of the target pair and of the decisions, for every backbone
# best/ on the validation half: "auroc_pm2" (default, as the readiness heads of offline_10ep) — AUROC of p_k against
# "ready" (k >= t*) over the pairs |k - t*| <= 2, higher is better; "bce_pm2" — BCE over the same pairs (the choice of
# the online SFT model), lower is better; "loss" — the validation loss, lower is better.
SELECTIONS = ("auroc_pm2", "bce_pm2", "loss")
SEPARATE_FORMAT = "online_video_llm_separate_v1"
RUN_FILE = "separate.json"
HEAD_FILE = "head.safetensors"
OPTIMIZER_FILE = "optimizer.pt"
RNG_FILE = "rng.pt"  # random generators' state in last/, so that a resumed run goes on as the uninterrupted one
COMPLETE_FILE = "complete"  # in a run directory: all its epochs are trained


def selection_score(summary: Mapping[str, Any], selection: str) -> float:
    """Higher is better for every choice; an undefined AUROC (one class in the window) is the worst."""
    if selection == "auroc_pm2":
        value = summary["classification"]["auroc_pm2"]
        return float("-inf") if value is None else float(value)
    if selection == "bce_pm2":
        return -summary["classification"]["binary_cross_entropy_pm2"]
    if selection == "loss":
        return -summary["loss"]
    raise ValueError(f"selection must be one of {SELECTIONS}")


@dataclass(frozen=True)
class DecisionObjective:
    """Slot targets (uhead/labels.py make_targets): hard 0/1 at the target pair, step and sigmoid soft before it."""

    tags: str = "hard"
    step_window: int = 5
    sigmoid_slope: float = 0.65

    def validate(self) -> None:
        if self.tags not in TAGS:
            raise ValueError(f"tags must be one of {TAGS}: from the target pair on every slot is 'ready'")
        if self.step_window < 0 or self.sigmoid_slope <= 0:
            raise ValueError("invalid step_window or sigmoid_slope")

    def targets(self, patches: int, target: int) -> torch.Tensor:
        from uhead.labels import make_targets
        return torch.from_numpy(make_targets(patches, target, self.tags, step_window=self.step_window,
                                             sigmoid_slope=self.sigmoid_slope)).float()


# ---------------------------------------------------------------------------------------------------------------- model

def add_act_token(tokenizer) -> int:
    """Add <act> as a special token (once) and return its id."""
    from transformers import AddedToken
    if ACT not in tokenizer.get_vocab():
        tokenizer.add_tokens([AddedToken(ACT, special=True, normalized=False)], special_tokens=True)
    act = tokenizer.convert_tokens_to_ids(ACT)
    if tokenizer(ACT, add_special_tokens=False).input_ids != [act]:
        raise RuntimeError(f"{ACT} is not a single token of its own")
    return act


def _tied(base) -> bool:
    return base.lm_head.weight.data_ptr() == base.get_input_embeddings().weight.data_ptr()


@torch.no_grad()
def init_act_row(base, tokenizer, act: int) -> None:
    """The input embedding row of <act> := the mean row of INIT_WORD. Deterministic, so a saved run rebuilds the same
    starting row (its adapter stores only the change)."""
    embed = base.get_input_embeddings().weight
    if act >= embed.shape[0]:
        raise ValueError(f"Token id {act} does not fit the {embed.shape[0]} embedding rows")
    words = tokenizer(INIT_WORD, add_special_tokens=False).input_ids
    embed[act] = embed[words].float().mean(0).to(embed.dtype)


def head_config_for(base: UHeadConfig, hidden_size: int) -> UHeadConfig:
    """The head of the detector "MLP last_hidden" (uhead): base (configs/uhead/default.yaml) on the group last_hidden."""
    config = replace(base, groups=["last_hidden"], temporal_detector="mlp", input_dims={"last_hidden": hidden_size})
    config.validate()
    return config


class SeparateModel(nn.Module):
    """A VLM with a LoRA, the trainable <act> row and, for LLaVA-OV with ``train_projector``, a trainable copy of its
    whole projector (``backbone``, PEFT) and the decision head (float32)."""

    def __init__(self, backbone, config: ModelConfig, head_config: UHeadConfig, act: int, family: str = QWEN,
                 train_projector: bool = False):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        self.backbone, self.config, self.head_config, self.act, self.family = backbone, config, head_config, act, family
        self.train_projector = train_projector
        self.head = StateHead(head_config)
        self.head.to(device=self.qwen.get_input_embeddings().weight.device, dtype=torch.float32)

    @property
    def qwen(self):
        """The Hugging Face model (Qwen2.5-VL or LLaVA-OV; the name is that of the shared LoRA code), adapters in it."""
        return self.backbone.get_base_model()

    def decision_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logits [N] of the hidden states [N, D] at <act> positions."""
        return self.head({"last_hidden": hidden.float()[None]})[0]


def model_family(config: ModelConfig, *, local_files_only: bool = False) -> str:
    from transformers import AutoConfig
    kind = AutoConfig.from_pretrained(config.backbone.model_id, revision=config.backbone.revision,
                                      local_files_only=local_files_only).model_type
    if kind not in FAMILIES:
        raise ValueError(f"{config.backbone.model_id}: model type {kind}, supported {FAMILIES}")
    return kind


def _load_llava(config: ModelConfig, *, local_files_only: bool, device: str | None):
    """LLaVA-OneVision (frozen) on ``device``; the resolved Hub commit goes into ``config``. Loaded into CPU memory and
    then moved: loading straight onto the GPU crashed in transformers' weight loader on Windows (a 0.5B model costs
    ~2 GB of RAM this way)."""
    from transformers import LlavaOnevisionForConditionalGeneration
    if config.backbone.quantization is not None:
        raise ValueError("Quantization is not supported for LLaVA-OneVision here")
    base = LlavaOnevisionForConditionalGeneration.from_pretrained(
        config.backbone.model_id, revision=config.backbone.revision, dtype=getattr(torch, config.backbone.dtype),
        local_files_only=local_files_only, attn_implementation="sdpa")
    if device is not None:
        base.to(device)
    revision = getattr(base.config, "_commit_hash", None)
    if revision:
        config.backbone.revision = revision
    return base.requires_grad_(False)


PROJECTOR = "multi_modal_projector"  # LLaVA-OV's projector (vision features -> language model), its "adapter"


def build_separate_model(config: ModelConfig, tokenizer, head_config: UHeadConfig, *, local_files_only: bool = False,
                         device: str | None = None, train_projector: bool = False) -> SeparateModel:
    """The backbone of config.backbone (Qwen2.5-VL without any adapter of M_off, or LLaVA-OneVision) with <act>, a
    fresh LoRA (config.lora, B = 0) and a fresh head (draws from the global torch seed). The token is added to
    ``tokenizer``. ``train_projector`` (LLaVA-OV only): its whole projector trains too, as a PEFT modules_to_save copy
    that starts from the pretrained one and is saved with the adapter."""
    from peft import LoraConfig, TaskType, get_peft_model
    from online_vllm.models import build_model
    from online_vllm.models.builder import resolve_targets

    if not config.lora.enabled or config.lora.component != "language_model":
        raise ValueError("The decision module needs a language_model LoRA config")
    family = model_family(config, local_files_only=local_files_only)
    if train_projector and family != LLAVA:
        raise ValueError("train_projector is for LLaVA-OneVision: Qwen's merger is part of its cached encoder output")
    config = deepcopy(config)
    if family == QWEN:
        plain = deepcopy(config)
        plain.lora.enabled = False
        built = build_model(plain, local_files_only=local_files_only, device=device)  # installs the 3D positions
        config = deepcopy(built.config)  # with the resolved revision
        config.lora.enabled = True
        base = built.backbone
    else:
        base = _load_llava(config, local_files_only=local_files_only, device=device)
    act = add_act_token(tokenizer)
    init_act_row(base, tokenizer, act)
    backbone = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config.lora.r, lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout, bias="none", target_modules=resolve_targets(base, config.lora),
        trainable_token_indices=[act] if _tied(base) else {"embed_tokens": [act]},
        modules_to_save=[PROJECTOR] if train_projector else None))
    return SeparateModel(backbone, config, head_config_for(head_config, base.config.text_config.hidden_size), act,
                         family, train_projector)


def expected_lora_targets(model: SeparateModel) -> int:
    lora = model.config.lora
    layers = model.qwen.config.text_config.num_hidden_layers if lora.layers is None else len(lora.layers)
    return layers * len(lora.modules)


def assert_separate_isolation(model: nn.Module, *, expected_targets: int | None = None) -> None:
    """Only the language-model attention LoRA, the <act> row, the head and, with model.train_projector, the projector
    copy may train."""
    targets, token_rows, head, projector = set(), 0, 0, 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head."):
            head += 1
            continue
        if "trainable_tokens_delta" in name:
            token_rows += 1
            continue
        if f"{PROJECTOR}.modules_to_save." in name:
            projector += 1
            continue
        if not name.startswith("backbone.") or ".lora_" not in name or "model.language_model.layers." not in name:
            raise ValueError(f"Trainable parameter outside the LM-attention LoRA, the <act> row and the head: {name}")
        module = name.split(".lora_", 1)[0]
        if module.rsplit(".", 1)[-1] not in EXPECTED_LORA_LEAVES:
            raise ValueError(f"LoRA is outside attention projections: {name}")
        targets.add(module)
    if not targets or not token_rows or not head:
        raise ValueError("The decision module needs a trainable LoRA, <act> row and head")
    if bool(projector) != bool(getattr(model, "train_projector", False)):
        raise ValueError(f"{projector} trainable projector tensors, train_projector={model.train_projector}")
    if expected_targets is not None and len(targets) != expected_targets:
        raise ValueError(f"Expected {expected_targets} LoRA targets, got {len(targets)}")


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_separate_adapter(model: SeparateModel, path: str | Path, *,
                          metadata: Mapping[str, Any] | None = None) -> Path:
    """The LoRA and the <act> row (PEFT, adapter_model.safetensors) and the head (head.safetensors)."""
    from safetensors.torch import save_file
    assert_separate_isolation(model)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path, safe_serialization=True)
    save_file({key: value.detach().contiguous().cpu() for key, value in model.head.state_dict().items()},
              str(path / HEAD_FILE))
    payload = {"format": SEPARATE_FORMAT, **(dict(metadata) if metadata else {})}
    (path / "sft_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_separate_adapter(model: SeparateModel, path: str | Path) -> SeparateModel:
    from safetensors.torch import load_file
    load_lora_adapter(model, path)
    model.head.load_state_dict(load_file(str(Path(path) / HEAD_FILE)))
    return model


def load_processor(config: ModelConfig, *, local_files_only: bool = False):
    """What the family needs besides the model: Qwen's InputBuilder, or LLaVA-OV's processor."""
    if model_family(config, local_files_only=local_files_only) == QWEN:
        return InputBuilder.from_config(config, local_files_only=local_files_only)
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(config.backbone.model_id, revision=config.backbone.revision,
                                         local_files_only=local_files_only)


def tokenizer_of(processor) -> Any:
    return processor.processor.tokenizer if isinstance(processor, InputBuilder) else processor.tokenizer


def load_separate_run(run_dir: str | Path, adapter: str = "best", *, local_files_only: bool = False,
                      device: str | None = None):
    """Rebuild a trained decision module from ``<run>/<adapter>``. Returns (model, family, info of separate.json)."""
    run_dir = Path(run_dir)
    info = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    config = ModelConfig.load(run_dir / "model.yaml")
    processor = load_processor(config, local_files_only=local_files_only)
    model = build_separate_model(config, tokenizer_of(processor), UHeadConfig.from_dict(info["head"]),
                                 local_files_only=local_files_only, device=device,
                                 train_projector=info.get("train_projector", False))
    if model.act != info["act"]:
        raise ValueError(f"Token id {model.act} differs from that of the run {info['act']}")
    load_separate_adapter(model, run_dir / adapter)
    return model.eval(), make_family(model, processor), info


# ----------------------------------------------------------------------------------------------------------- families

def build_decision_example(tpl: Template, entry: Mapping[str, torch.Tensor], *, spec: RopeSpec, act: int,
                           targets: torch.Tensor) -> dict[str, Any]:
    """Qwen: one question as one causal sequence, the prompt, then each of the first P = len(targets) pairs of the video
    followed by <act>. Tensors: input_ids, mm_token_type_ids [L]; position_ids [3, L]; video_embeds [P * per_pair, D]
    from the vision cache; act_index [P] (the <act> after pair k) with target [P]."""
    grid = entry["grid"].reshape(3)
    patches = len(targets)
    if not 1 <= patches <= int(grid[0]):
        raise ValueError(f"{patches} targets for a video of {int(grid[0])} pairs")
    n, per = len(tpl.prefix), tpl.per_pair
    video_pos = vision_positions(n, grid.tolist(), spec.merge, float(entry["second_per_grid_ts"].reshape(-1)[0]),
                                 spec.tokens_per_second)
    if video_pos.shape[1] != int(grid[0]) * per:
        raise ValueError("Visual positions do not match the template")
    ids, types = list(tpl.prefix), [0] * n
    positions = [torch.arange(n).expand(3, -1)]
    reach, act_index = n - 1, []
    for k in range(patches):
        chunk = video_pos[:, k * per:(k + 1) * per]
        ids += [tpl.video_token] * per
        types += [2] * per
        positions.append(chunk)
        reach = max(reach, int(chunk.max()))
        ids.append(act)
        types.append(0)
        positions.append(torch.full((3, 1), reach + 1))  # where the offline template tail after pair k starts
        act_index.append(len(ids) - 1)
    return {"input_ids": torch.tensor(ids), "mm_token_type_ids": torch.tensor(types),
            "position_ids": torch.cat(positions, 1).long(), "video_embeds": entry["embeds"][:patches * per],
            "act_index": torch.tensor(act_index), "target": targets.float()}


def _scatter_video(model: SeparateModel, embeds: torch.Tensor, ids: torch.Tensor, video: torch.Tensor,
                   video_token: int) -> torch.Tensor:
    placeholders = ids == video_token
    if int(placeholders.sum()) != video.shape[0]:
        raise ValueError(f"{int(placeholders.sum())} video placeholders, {video.shape[0]} cached video tokens")
    return embeds.masked_scatter(placeholders[..., None].expand_as(embeds), video.to(embeds.device, embeds.dtype))


class QwenFamily:
    """Qwen2.5-VL: the prompt and video layout of M_off, 3D positions (lora/online.py)."""

    name = QWEN

    def __init__(self, model: SeparateModel, builder: InputBuilder):
        self.builder, self.spec = builder, RopeSpec.of(model)
        self.temporal_patch = builder.temporal_patch
        self.pad_id = builder.processor.tokenizer.pad_token_id

    def example(self, record: Record, entry, *, act: int, objective: DecisionObjective,
                split: ClevrerLoRASplitConfig) -> tuple[int, dict[str, Any]]:
        question = _question(record, entry, self.builder, split)
        return question.target, build_decision_example(question.template, entry, spec=self.spec, act=act,
                                                       targets=objective.targets(question.patches, question.target))

    def hidden(self, model: SeparateModel, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return online_hidden(model, batch)

    def start(self, record: Record, entry) -> tuple[list[int], torch.Tensor, dict[str, Any]]:
        grid = entry["grid"].reshape(3).tolist()
        seconds = float(entry["second_per_grid_ts"].reshape(-1)[0])
        tpl = template(self.builder, format_prompt(record.question, record.choices), torch.tensor([1, *grid[1:]]),
                       torch.tensor([seconds]))
        n = len(tpl.prefix)
        return tpl.prefix, torch.arange(n).expand(3, -1), {"start": n, "reach": n - 1, "grid_hw": grid[1:],
                                                           "seconds": seconds, "per": tpl.per_pair}

    def advance(self, state: dict[str, Any], k: int) -> torch.Tensor:
        """Positions [3, per + 1] of pair k and its <act>."""
        per = state["per"]
        positions = vision_positions(state["start"], (k + 1, *state["grid_hw"]), self.spec.merge, state["seconds"],
                                     self.spec.tokens_per_second)[:, k * per:]
        state["reach"] = max(state["reach"], int(positions.max()))
        return torch.cat([positions, torch.full((3, 1), state["reach"] + 1)], 1)

    def project(self, model: SeparateModel, features: torch.Tensor) -> torch.Tensor:
        return features  # the cache already holds the language model's visual tokens (encoder and merger)

    def forward_cached(self, model: SeparateModel, embeds: torch.Tensor, positions: torch.Tensor, cache):
        qwen = model.qwen.model
        qwen.rope_deltas = None
        return qwen(inputs_embeds=embeds, position_ids=positions[:, None], past_key_values=cache, use_cache=True,
                    return_dict=True)


class LlavaFamily:
    """LLaVA-OneVision: LLAVA_PROMPT, then every pair's cached frame tokens and <act>; 1D positions."""

    name = LLAVA
    temporal_patch = PAIR

    def __init__(self, model: SeparateModel, tokenizer, video_processor=None):
        self.tokenizer, self.video_processor = tokenizer, video_processor  # the processor: to fill the vision cache
        self.video_token = int(model.qwen.config.video_token_id)
        pad = tokenizer.pad_token_id
        self.pad_id = tokenizer.eos_token_id if pad is None else pad

    def prefix(self, record: Record) -> list[int]:
        text = LLAVA_PROMPT.format(prompt=format_prompt(record.question, record.choices))
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def example(self, record: Record, entry, *, act: int, objective: DecisionObjective,
                split: ClevrerLoRASplitConfig) -> tuple[int, dict[str, Any]]:
        patches = VisionCache.patches(entry)
        target = _target_patch(record, builder=self, frame_stride=split.frame_stride, speak_delay=split.speak_delay,
                               patches=patches)
        per = entry["embeds"].shape[0] // patches
        ids = self.prefix(record)
        types, act_index = [0] * len(ids), []
        for _ in range(patches):
            ids += [self.video_token] * per + [act]
            types += [2] * per + [0]
            act_index.append(len(ids) - 1)
        return target, {"input_ids": torch.tensor(ids), "mm_token_type_ids": torch.tensor(types),
                        "position_ids": torch.arange(len(ids))[None], "video_embeds": entry["embeds"],
                        "act_index": torch.tensor(act_index), "target": objective.targets(patches, target)}

    def project(self, model: SeparateModel, features: torch.Tensor) -> torch.Tensor:
        """Language-model tokens of cached vision features [N, C]: LLaVA-OV's projector (with train_projector, its
        trainable copy: PEFT's wrapper routes the call)."""
        projector = getattr(model.qwen.model, PROJECTOR)
        weight = next(projector.parameters())
        return projector(features.to(weight.device, weight.dtype))

    def hidden(self, model: SeparateModel, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        llava = model.qwen.model
        ids = batch["input_ids"]
        video = self.project(model, batch["video_embeds"])
        embeds = _scatter_video(model, llava.get_input_embeddings()(ids), ids, video, self.video_token)
        with attention_kernels():
            return llava.language_model(inputs_embeds=embeds, position_ids=batch["position_ids"][0],
                                        attention_mask=batch["attention_mask"], use_cache=False,
                                        return_dict=True).last_hidden_state

    def start(self, record: Record, entry) -> tuple[list[int], torch.Tensor, dict[str, Any]]:
        ids = self.prefix(record)
        per = entry["embeds"].shape[0] // VisionCache.patches(entry)
        return ids, torch.arange(len(ids))[None], {"next": len(ids), "per": per}

    def advance(self, state: dict[str, Any], k: int) -> torch.Tensor:
        """Positions [1, per + 1] of pair k and its <act>."""
        start = state["next"]
        state["next"] += state["per"] + 1
        return torch.arange(start, state["next"])[None]

    def forward_cached(self, model: SeparateModel, embeds: torch.Tensor, positions: torch.Tensor, cache):
        return model.qwen.model.language_model(inputs_embeds=embeds, position_ids=positions, past_key_values=cache,
                                               use_cache=True, return_dict=True)


def make_family(model: SeparateModel, processor):
    """``processor``: Qwen's InputBuilder, or LLaVA-OV's processor (or its tokenizer)."""
    if model.family == QWEN:
        return QwenFamily(model, processor)
    return LlavaFamily(model, getattr(processor, "tokenizer", processor), getattr(processor, "video_processor", None))


def open_vision_cache(model: SeparateModel, family, split: ClevrerLoRASplitConfig, records: Sequence[Record], *,
                      data_root: str | Path, device, settings: Mapping[str, Any] | None = None,
                      frames_per_pass: int = 32) -> VisionCache:
    """The vision cache of ``split.vision_cache`` with every video of ``records`` in it (missing ones are encoded now).
    Qwen: the cache of the LoRA runs (lora/vision_cache.py). LLaVA-OV: ``settings`` (llava_cache_settings), its
    encoder taking ``frames_per_pass`` frames at a time."""
    from lora.vision_cache import build_vision_cache
    if family.name == QWEN:
        cache = VisionCache.for_model(split.vision_cache, model, frame_size=split.train_frame_size,
                                      frame_stride=split.frame_stride)
        added = build_vision_cache(model, family.builder, records, cache, data_root=data_root,
                                   frame_size=split.train_frame_size, frame_stride=split.frame_stride, device=device,
                                   num_workers=split.num_workers)
    else:
        if settings is None or family.video_processor is None:
            raise ValueError("A LLaVA-OV vision cache needs its settings and the video processor")
        cache = VisionCache(split.vision_cache, settings)
        added = build_llava_vision_cache(model, family.video_processor, records, cache, data_root=data_root,
                                         num_workers=split.num_workers, frames_per_pass=frames_per_pass)
    print(f"Vision cache {split.vision_cache}: {added} videos encoded now")
    return cache


def collate_decision(examples: Sequence[Mapping[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    """Right padding to the longest sequence, causal attention within a row (a padded position attends only to
    itself); position_ids [R, B, L] (R = 3 for Qwen's 3D positions, 1 for 1D); the <act> positions become (row,
    index) pairs of the batch."""
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    length = max(len(example["input_ids"]) for example in examples)
    batch, axes = len(examples), examples[0]["position_ids"].shape[0]
    ids = torch.full((batch, length), pad_id, dtype=torch.long)
    types = torch.zeros(batch, length, dtype=torch.long)
    positions = torch.zeros(axes, batch, length, dtype=torch.long)
    attention = torch.eye(length, dtype=torch.bool).repeat(batch, 1, 1, 1)
    for row, example in enumerate(examples):
        n = len(example["input_ids"])
        ids[row, :n], types[row, :n] = example["input_ids"], example["mm_token_type_ids"]
        positions[:, row, :n] = example["position_ids"]
        attention[row, 0, :n, :n] = torch.ones(n, n, dtype=torch.bool).tril()
    return {"input_ids": ids, "mm_token_type_ids": types, "position_ids": positions, "attention_mask": attention,
            "video_embeds": torch.cat([example["video_embeds"] for example in examples]),
            "act_row": torch.cat([torch.full((len(example["act_index"]),), row) for row, example in enumerate(examples)]),
            "act_index": torch.cat([example["act_index"] for example in examples]),
            "target": torch.cat([example["target"] for example in examples])}


def batch_logits(model: SeparateModel, batch: Mapping[str, torch.Tensor], family) -> torch.Tensor:
    """Decision logits [N] at every <act> of a collated batch, in batch then position order."""
    hidden = family.hidden(model, batch)
    return model.decision_logits(hidden[batch["act_row"], batch["act_index"]])


def decision_loss(model: SeparateModel, batch: Mapping[str, torch.Tensor], family) -> torch.Tensor:
    """BCE with logits, mean over every slot of the batch."""
    logits = batch_logits(model, batch, family)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, batch["target"].to(logits))
    if not torch.isfinite(loss):
        raise RuntimeError("The decision loss is not finite")
    return loss


# ------------------------------------------------------------------------------------------------- LLaVA vision cache

def llava_cache_settings(model: SeparateModel, *, frames_per_pair: int, tokens_side: int | None) -> dict[str, Any]:
    """What a LLaVA-OV vision cache answers for: the encoder, the frame size and the tokens of a pair. The cache holds
    vision features before the projector ("stage": "vision"), so that the projector can train."""
    vision = model.qwen.config.vision_config
    side = tokens_side or math.ceil(vision.image_size // vision.patch_size / 2)  # LLaVA-OV's own pooling
    if frames_per_pair not in (1, 2) or side < 1:
        raise ValueError("frames_per_pair must be 1 or 2 and tokens_side positive")
    backbone = model.config.backbone
    return {"model_id": backbone.model_id, "revision": backbone.revision,
            "dtype": str(model.qwen.model.vision_tower.dtype), "frame_size": [vision.image_size] * 2,
            "frame_stride": 1, "frames_per_pair": frames_per_pair, "tokens_side": side, "stage": "vision",
            "pooling": "before_projector",
            "vision_feature_layer": model.qwen.config.vision_feature_layer,
            "vision_feature_select_strategy": model.qwen.config.vision_feature_select_strategy}


class _LlavaPixels(torch.utils.data.Dataset):
    def __init__(self, records: Sequence[Record], data_root: Path, video_processor, size: int):
        self.records, self.data_root, self.video_processor, self.size = records, data_root, video_processor, size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        frames = read_every(self.data_root / record.video, 1, (self.size, self.size))
        frames = frames[:len(frames) // PAIR * PAIR]
        pixels = self.video_processor(videos=[frames], return_tensors="pt")["pixel_values_videos"]
        return record, pixels


def _first(items):
    return items[0]


def _pool(tokens: torch.Tensor, side: int) -> torch.Tensor:
    """Bilinear pooling of square token grids [F, s * s, C] to [F, side * side, C] (as LLaVA-OV's apply_pooling)."""
    current = math.isqrt(tokens.shape[1])
    if current * current != tokens.shape[1] or side > current:
        raise ValueError(f"{tokens.shape[1]} tokens per frame cannot be pooled to {side} x {side}")
    if side == current:
        return tokens
    grid = tokens.view(len(tokens), current, current, -1).permute(0, 3, 1, 2).float()
    grid = nn.functional.interpolate(grid, size=(side, side), mode="bilinear")
    return grid.permute(0, 2, 3, 1).reshape(len(tokens), side * side, -1).to(tokens.dtype)


@torch.inference_mode()
def llava_frame_features(model: SeparateModel, pixels: torch.Tensor, *, tokens_side: int,
                         chunk: int = 32) -> torch.Tensor:
    """Vision features [F, tokens_side ** 2, C] of the frames [1, F, 3, H, W], before the projector: LLaVA-OV's encoder
    frame by frame (no frame sees another; its feature layer and select strategy), then its own 2 x 2 pooling
    (27 x 27 -> 14 x 14 at 384) and bilinear pooling to tokens_side x tokens_side if smaller. LLaVA-OV pools after the
    projector; here pooling comes first, so that the cache stays small (49 x 1152 per frame instead of 729 x 1152) and
    the projector can train on it: the same linear pooling, on the other side of the projector's nonlinearity."""
    llava, config = model.qwen.model, model.qwen.config
    tower = llava.vision_tower
    layer, strategy = config.vision_feature_layer, config.vision_feature_select_strategy
    parts = []
    for start in range(0, pixels.shape[1], chunk):
        frames = pixels[0, start:start + chunk].to(tower.device, tower.dtype)
        with attention_kernels():
            states = tower(frames, output_hidden_states=True, return_dict=True).hidden_states
        features = states[layer] if isinstance(layer, int) else torch.cat([states[i] for i in layer], -1)
        if strategy == "default":
            features = features[:, 1:]
        parts.append(features)
    features = torch.cat(parts)
    native = math.ceil(math.isqrt(features.shape[1]) / 2)
    return _pool(_pool(features, native), tokens_side)


@torch.inference_mode()
def build_llava_vision_cache(model: SeparateModel, video_processor, records: Sequence[Record], cache: VisionCache, *,
                             data_root: str | Path, num_workers: int = 2, frames_per_pass: int = 32) -> int:
    """Encode every video of ``records`` that the cache does not hold yet (cache.settings: llava_cache_settings);
    returns how many were added. An entry: "embeds" [pairs * frames_per_pair * tokens_side ** 2, C], vision features
    before the projector (llava_frame_features; with one frame per pair, the last frame of each pair), "grid" [3] =
    (pairs, frames_per_pair, tokens per frame), "second_per_grid_ts" [1] = seconds per pair."""
    settings = cache.settings
    first: dict[str, Record] = {}
    for record in records:
        if not cache.has(record.video):
            first.setdefault(record.video, record)
    if not first:
        return 0
    loader = DataLoader(_LlavaPixels(list(first.values()), Path(data_root), video_processor, settings["frame_size"][0]),
                        batch_size=1, num_workers=num_workers, collate_fn=_first)
    for record, pixels in tqdm(loader, total=len(first), desc="LLaVA vision cache", unit="video", dynamic_ncols=True):
        tokens = llava_frame_features(model, pixels, tokens_side=settings["tokens_side"], chunk=frames_per_pass)
        if settings["frames_per_pair"] == 1:
            tokens = tokens[PAIR - 1::PAIR]  # the last frame of every pair: the one that defines the target pair
        pairs = len(tokens) // settings["frames_per_pair"]
        grid = torch.tensor([pairs, settings["frames_per_pair"], tokens.shape[1]])
        cache.save(record.video, tokens.reshape(-1, tokens.shape[-1]).cpu(), grid,
                   torch.tensor([PAIR / record.fps], dtype=torch.float32))
    return len(first)


# --------------------------------------------------------------------------------------------------------------- data

def decision_records(path: str | Path, videos: int | None, *, family, split: ClevrerLoRASplitConfig) -> list[Record]:
    """Every question of the first ``videos`` videos of a group-1 file by scene_index (None: all) that has a target
    pair: the training questions of the readiness heads (extract_features.py --videos, uhead/data.py drops questions
    without a target)."""
    return [record for record in _records_for_first_videos(Path(path), videos)
            if plan(record, split.frame_stride, split.speak_delay, family.temporal_patch) is not None]


class DecisionDataset(torch.utils.data.Dataset):
    """Training sequences: every question with its whole video, a slot after every pair."""

    def __init__(self, records: Sequence[Record], *, family, split: ClevrerLoRASplitConfig, act: int,
                 objective: DecisionObjective):
        if split.vision_cache is None:
            raise ValueError("The decision module reads the video from a vision cache")
        self.records, self.family, self.split, self.act, self.objective = list(records), family, split, act, objective
        self._cache = None

    @property
    def cache(self) -> VisionCache:
        if self._cache is None:  # opened in every DataLoader worker
            self._cache = VisionCache(self.split.vision_cache)
        return self._cache

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:  # the sequences do not depend on the epoch
        pass

    def prefix_patches(self, epoch: int) -> list[int]:
        """Pairs of every sequence, from the record alone, for _LengthGroupedBatches."""
        return [-(-record.num_frames // self.split.frame_stride) // self.family.temporal_patch
                for record in self.records]

    def item(self, index: int) -> tuple[Record, int, dict[str, Any]]:
        record = self.records[index]
        target, example = self.family.example(record, self.cache.load(record.video), act=self.act,
                                              objective=self.objective, split=self.split)
        return record, target, example

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.item(index)[2]


def build_decision_loader(records: Sequence[Record], *, family, split: ClevrerLoRASplitConfig, act: int,
                          objective: DecisionObjective, seed: int, batch_size: int) -> DataLoader:
    dataset = DecisionDataset(records, family=family, split=split, act=act, objective=objective)
    return DataLoader(dataset, batch_sampler=_LengthGroupedBatches(dataset, batch_size, seed),
                      num_workers=split.num_workers, collate_fn=partial(collate_decision, pad_id=family.pad_id))


def answers_of_predictions(predictions: Mapping[str, Any]) -> dict[str, list[bool]]:
    """answer_correct of M_off at every pair, per question, from a readiness head's predictions file."""
    rows = predictions["predictions"]
    if not rows or any("answer_correct" not in row for row in rows):
        raise ValueError("The predictions file has no answer_correct: the answers of M_off are needed")
    return {row["record_id"]: list(row["answer_correct"]) for row in rows}


# --------------------------------------------------------------------------------------------------------- evaluation

class _EvaluationItems(torch.utils.data.Dataset):
    def __init__(self, dataset: DecisionDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset.item(index)


def _as_list(items):
    return items


@torch.inference_mode()
def evaluate_decision(model: SeparateModel, records: Sequence[Record], *, family, split: ClevrerLoRASplitConfig,
                      objective: DecisionObjective, device, answers: Mapping[str, Sequence[bool]] | None,
                      taus: Sequence[float] = (0.5, 0.7, 0.9), batch_size: int = 4,
                      desc: str = "Decision evaluation") -> tuple[dict[str, Any], dict[str, Any]]:
    """One pass per question: p_k at every pair. Returns the summary and the predictions in the format of
    online_vllm/eval/export.py; answer_correct is that of M_off (``answers``, by question), so that the online metrics
    (eval/metrics.py) are those of the pair "this module decides, M_off answers". tau = 0.5 is the native rule.

    loss: the training objective, mean over every pair of every question."""
    if answers is not None:
        missing = [record.id for record in records if record.id not in answers]
        if missing:
            raise ValueError(f"{len(missing)} questions have no answers of M_off, e.g. {missing[:3]}")
    training = model.training
    model.eval()
    dataset = DecisionDataset(records, family=family, split=split, act=model.act, objective=objective)
    loader = DataLoader(_EvaluationItems(dataset), batch_size=batch_size, num_workers=split.num_workers,
                        collate_fn=_as_list)
    rows, losses = [], []
    try:
        with tqdm(total=len(records), desc=desc, unit="q", dynamic_ncols=True) as progress:
            for items in loader:
                batch = _move_inputs(collate_decision([example for _, _, example in items], pad_id=family.pad_id),
                                     device)
                logits = batch_logits(model, batch, family).float()
                target = batch["target"].to(logits)
                losses.append(nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none").cpu())
                probabilities, row_of = logits.sigmoid().cpu(), batch["act_row"].cpu()
                for row, (record, target_pair, example) in enumerate(items):
                    p = probabilities[row_of == row].tolist()
                    entry = {"record_id": record.id, "probabilities": p, "target_pair": target_pair,
                             "visible_pair": target_pair, "fps": record.fps, "family": record.meta.get("family")}
                    if answers is not None:
                        correct = list(answers[record.id])
                        if len(correct) != len(p):
                            raise ValueError(f"{record.id}: {len(p)} pairs, {len(correct)} answers of M_off")
                        entry["answer_correct"] = correct
                    rows.append(entry)
                progress.update(len(items))
    finally:
        model.train(training)
    payload = {"metadata": {"model": f"separate_{family.name}", "tags": objective.tags,
                            "temporal_patch": family.temporal_patch, "frame_stride": split.frame_stride},
               "predictions": rows}
    readiness = sweep(payload, list(taus))
    summary = {"count": len(rows), "loss": float(torch.cat(losses).mean()),
               "latency_accuracy": readiness["latency_accuracy"], "best_thresholds": readiness["best_thresholds"],
               "constrained_latency": readiness["constrained_latency"],
               "classification": readiness["classification"], "online": readiness["results"]}
    return summary, payload


def print_summary(epoch: int | None, summary: Mapping[str, Any], train_loss: float | None = None) -> None:
    if epoch is not None:
        loss = "-" if train_loss is None else f"{train_loss:.4f}"
        print(f"Epoch {epoch} | train_loss={loss} | val_loss={summary['loss']:.4f}")
    print("  " + _classification_line(summary["classification"]))
    if summary["latency_accuracy"] is not None:
        print("  " + _latency_line(summary["latency_accuracy"]))
    for best in summary["best_thresholds"].values():
        print("  " + _best_threshold_line(best))
    if summary["constrained_latency"] is not None:
        print("  " + _constrained_line(summary["constrained_latency"]))
    for tau, values in summary["online"].items():
        answer = "" if values.get("answer_accuracy") is None else (
            f" answer={100 * values['answer_accuracy']:.1f}% "
            f"timely_correct_0_4={100 * values['timely_correct_0_4']:.1f}%")
        print(f"  tau={tau}: on_time_0_4={100 * values['on_time_0_4']:.1f}% "
              f"on_target_pm2={100 * values['on_target_pm2']:.1f}% premature={100 * values['premature_rate']:.1f}% "
              f"missed={100 * values['missed_rate']:.1f}%" + answer)


# ----------------------------------------------------------------------------------------------------------- training

def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


@dataclass(frozen=True)
class ResumeState:
    """Where an interrupted run continues: the epoch of last/, the optimizer steps up to it, AdamW's and the random
    generators' state, and the best epoch so far by the run's selection."""

    epoch: int
    global_step: int
    optimizer_state: Mapping[str, Any]
    rng: Mapping[str, Any] | None
    best_epoch: int
    best_score: float


def read_resume_state(output_dir: str | Path, selection: str) -> ResumeState | None:
    """The state of ``<run>/last`` (None: no epoch was finished, the run starts anew). The best epoch is recomputed from
    validation_metrics.jsonl up to that epoch, as the uninterrupted run chose it (a tie keeps the earlier epoch)."""
    output_dir = Path(output_dir)
    last = output_dir / "last"
    if not (last / "sft_metadata.json").exists():
        return None
    metadata = json.loads((last / "sft_metadata.json").read_text(encoding="utf-8"))
    epoch = int(metadata["epoch"])
    best_epoch, best_score = None, None
    for line in _jsonl(output_dir / "validation_metrics.jsonl"):
        if line["epoch"] > epoch:
            continue
        score = selection_score(line["validation"], selection)
        if best_score is None or score > best_score:
            best_epoch, best_score = line["epoch"], score
    if best_epoch is None:
        raise ValueError(f"{output_dir}/validation_metrics.jsonl has no epoch up to {epoch}")
    rng = last / RNG_FILE
    return ResumeState(epoch, int(metadata["global_step"]),
                       torch.load(last / OPTIMIZER_FILE, map_location="cpu", weights_only=True),
                       torch.load(rng, weights_only=False) if rng.exists() else None, best_epoch, best_score)


def _discard_after(output_dir: Path, state: ResumeState) -> None:
    """Drop what an interrupted epoch after last/ left behind (its metrics, steps, predictions, adapter), and put back
    best/ if that epoch had replaced it."""
    _write_jsonl(output_dir / "validation_metrics.jsonl",
                 [line for line in _jsonl(output_dir / "validation_metrics.jsonl") if line["epoch"] <= state.epoch])
    _write_jsonl(output_dir / "train_steps.jsonl",
                 [line for line in _jsonl(output_dir / "train_steps.jsonl") if line["epoch"] <= state.epoch])
    for path in output_dir.glob("predictions_validation_epoch_*.json"):
        if int(path.stem.rsplit("_", 1)[1]) > state.epoch:
            path.unlink()
    for path in output_dir.glob("epoch_*"):
        if path.is_dir() and int(path.name.split("_", 1)[1]) > state.epoch:
            shutil.rmtree(path)
    best = output_dir / "best"
    saved = json.loads((best / "sft_metadata.json").read_text(encoding="utf-8"))["epoch"]
    if saved != state.best_epoch:
        source = output_dir / f"epoch_{state.best_epoch}"
        if not source.exists():
            raise ValueError(f"best/ holds epoch {saved}, after last/ (epoch {state.epoch}); the best epoch before it, "
                             f"{state.best_epoch}, has no epoch_{state.best_epoch}/ to restore best/ from")
        shutil.rmtree(best)
        shutil.copytree(source, best)
        metadata = json.loads((best / "sft_metadata.json").read_text(encoding="utf-8"))
        (best / "sft_metadata.json").write_text(json.dumps(
            {**metadata, "selection_score": state.best_score, "restored_from": source.name}, indent=2,
            sort_keys=True) + "\n", encoding="utf-8")


def _save_last(model: SeparateModel, output_dir: Path, metadata: Mapping[str, Any], optimizer) -> Path:
    """last/ with the optimizer and random state, written beside it first and then swapped in, so that an interruption
    never leaves a half-written last/ behind."""
    partial, last = output_dir / "last.partial", output_dir / "last"
    if partial.exists():
        shutil.rmtree(partial)
    save_separate_adapter(model, partial, metadata=metadata)
    torch.save(optimizer.state_dict(), partial / OPTIMIZER_FILE)
    torch.save(_rng_state(), partial / RNG_FILE)
    if last.exists():
        shutil.rmtree(last)
    partial.rename(last)
    return last


def train_separate(model: SeparateModel, train_loader: DataLoader, config: LoRASFTConfig, *, family,
                   output_dir: str | Path, device, objective: DecisionObjective, evaluate, run_info: Mapping[str, Any],
                   split_info: Mapping[str, Any] | None = None, selection: str = "auroc_pm2",
                   head_lr: float = 5.0e-4, head_weight_decay: float = 0.0, save_every_epoch: bool = True,
                   resume: bool = False) -> Path:
    """Epochs 1..``config.epochs`` over ``train_loader``; ``evaluate()`` -> (summary, predictions) before training
    (epoch 0) and after every epoch. AdamW (config.betas, grad_clip over all trainable weights), constant learning
    rates: the LoRA, the <act> row and a trained projector at config.learning_rate / weight_decay (those of M_off and
    M_on), the head at ``head_lr`` / ``head_weight_decay`` (those of the readiness heads). ``best`` is the adapter of
    the best validation ``selection`` (a tie keeps the earlier epoch), ``last`` that of the last epoch with the
    optimizer and random state, ``epoch_<n>`` every epoch (``save_every_epoch``); the file ``complete`` marks a run
    that has trained all its epochs.

    ``resume``: continue the interrupted run in ``output_dir`` from its last/ (the model must be built as the run's,
    model.yaml and separate.json): its adapter, AdamW and the random generators are restored, the epochs go on from
    there (the batches of an epoch are drawn by its number), and what an unfinished epoch left is discarded, so the run
    ends as an uninterrupted one would. Without last/ (no epoch finished) the run starts anew in the same directory.
    The settings files of the run are kept."""
    config.validate()
    objective.validate()
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}")
    if head_lr <= 0 or head_weight_decay < 0:
        raise ValueError("head_lr must be positive and head_weight_decay nonnegative")
    configure_gradient_checkpointing(model, enabled=config.gradient_checkpointing)
    assert_separate_isolation(model)
    output_dir = Path(output_dir)
    state = read_resume_state(output_dir, selection) if resume else None
    if state is None:
        if resume:
            print(f"{output_dir}: no finished epoch to resume from, the run starts anew")
        output_dir = _prepare_output_dir(output_dir, overwrite=config.overwrite or resume)
        (output_dir / "sft_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
        model.config.save(output_dir / "model.yaml")
        (output_dir / RUN_FILE).write_text(json.dumps(
            {**run_info, "family": model.family, "train_projector": model.train_projector, "act": model.act,
             "token": ACT, "init_word": INIT_WORD, "objective": asdict(objective), "head": asdict(model.head_config),
             "head_lr": head_lr, "head_weight_decay": head_weight_decay, "selection": selection,
             "trainable_parameters": trainable_parameters(model)}, indent=2) + "\n", encoding="utf-8")
        if split_info is not None:
            (output_dir / "split.json").write_text(json.dumps(split_info, indent=2) + "\n", encoding="utf-8")
    else:
        if (output_dir / COMPLETE_FILE).exists() and state.epoch >= config.epochs:
            print(f"{output_dir}: all {config.epochs} epochs are trained")
            return output_dir / "best"
        (output_dir / COMPLETE_FILE).unlink(missing_ok=True)  # more epochs than before
        _discard_after(output_dir, state)
        load_separate_adapter(model, output_dir / "last")
        print(f"Resuming {output_dir}: epoch {state.epoch} done ({state.global_step} optimizer steps), best epoch "
              f"{state.best_epoch}; training epochs {state.epoch + 1}..{config.epochs}")
    head = [p for p in model.head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    arguments: dict[str, Any] = {"betas": config.betas}
    cuda = torch.cuda.is_available() and torch.device(device).type == "cuda"
    if cuda:
        arguments["fused"] = True
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": config.learning_rate, "weight_decay": config.weight_decay},
        {"params": head, "lr": head_lr, "weight_decay": head_weight_decay}], **arguments)
    best_epoch, best_score, global_step, first_epoch = None, None, 0, 1
    if state is not None:
        optimizer.load_state_dict(state.optimizer_state)
        for group, lr, decay in zip(optimizer.param_groups, (config.learning_rate, head_lr),
                                    (config.weight_decay, head_weight_decay)):
            group.update(lr=lr, weight_decay=decay, betas=config.betas)
        best_epoch, best_score, global_step = state.best_epoch, state.best_score, state.global_step
        first_epoch = state.epoch + 1

    def evaluate_and_save(epoch: int, train_loss: float | None, steps: int, seconds: float, peak: float | None):
        nonlocal best_epoch, best_score
        summary, predictions = evaluate()
        score = selection_score(summary, selection)
        is_best = best_score is None or score > best_score
        if is_best:
            best_epoch, best_score = epoch, score
            save_separate_adapter(model, output_dir / "best", metadata={
                "epoch": epoch, "selection": selection, "score": score, "validation_loss": summary["loss"],
                "validation_auroc_pm2": summary["classification"]["auroc_pm2"],
                "validation_bce_pm2": summary["classification"]["binary_cross_entropy_pm2"]})
        record = {"epoch": epoch, "train_loss": train_loss, "optimizer_steps": steps, "epoch_duration_seconds": seconds,
                  "peak_gpu_memory_gb": peak, "best_epoch": best_epoch, "is_best_checkpoint": is_best,
                  "validation": summary}
        with (output_dir / "validation_metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")
        (output_dir / f"predictions_validation_epoch_{epoch}.json").write_text(
            json.dumps(predictions, allow_nan=False) + "\n", encoding="utf-8")
        print_summary(epoch, summary, train_loss)

    if state is None:
        evaluate_and_save(0, None, 0, 0.0, None)
    elif state.rng is not None:
        _set_rng_state(state.rng)  # as they were when the uninterrupted run saved last/
    loss_fn = partial(decision_loss, family=family)
    parameters = backbone + head
    for epoch in range(first_epoch, config.epochs + 1):
        _set_epoch(train_loader, epoch)
        model.train()
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        mean_loss, steps, step_records = _train_epoch(
            model, train_loader, optimizer=optimizer, gradient_accumulation=config.gradient_accumulation,
            grad_clip=config.grad_clip, parameters=parameters, device=device, epoch=epoch,
            global_step_start=global_step, loss_fn=loss_fn)
        global_step += steps
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if cuda else None
        with (output_dir / "train_steps.jsonl").open("a", encoding="utf-8") as file:
            for step in step_records:
                file.write(json.dumps(step, allow_nan=False) + "\n")
        evaluate_and_save(epoch, mean_loss, steps, time.perf_counter() - started, peak)
        metadata = {"epoch": epoch, "train_loss": mean_loss, "global_step": global_step}
        if save_every_epoch:
            save_separate_adapter(model, output_dir / f"epoch_{epoch}", metadata=metadata)
        _save_last(model, output_dir, metadata, optimizer)  # last: once it is there, the epoch counts as done
    (output_dir / COMPLETE_FILE).write_text(json.dumps({"epochs": config.epochs, "best_epoch": best_epoch}) + "\n",
                                            encoding="utf-8")
    return output_dir / "best"


# ---------------------------------------------------------------------------------------------------------- inference

class DecisionSession:
    """Streaming inference of one question with a KV cache: the prompt once, then per pair its visual tokens and <act>
    in one pass over the new tokens; p_k from the head at <act>.

        session = DecisionSession(model, family, record, entry)   # entry: the video's grid / tokens per pair
        for pair in pairs:                 # visual tokens of a pair: [per_pair, D]
            if session.observe(pair) > tau:
                break                      # M_off answers on the pairs so far
    """

    def __init__(self, model: SeparateModel, family, record: Record, entry: Mapping[str, torch.Tensor], device=None):
        from transformers import DynamicCache
        self.model, self.family = model, family
        self.device = device or model.qwen.get_input_embeddings().weight.device
        self.cache, self.pair = DynamicCache(), 0
        with torch.inference_mode():
            ids, positions, self.state = family.start(record, entry)
            embed = model.qwen.get_input_embeddings()
            self._forward(embed(torch.tensor([ids], device=self.device)), positions)

    @torch.inference_mode()
    def _forward(self, embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Not cuDNN (first in attention_kernels): it builds a plan for every new key length, which here grows by a
        # pair every step — ~5x slower per decision (measured on LLaVA-OV-0.5B). The cache is short and unmasked, so
        # the flash / memory-efficient kernels fit; bfloat16 p_k differ from the one-pass kernels by ~1e-2.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
                         set_priority=True):
            out = self.family.forward_cached(self.model, embeds.to(self.device), positions.to(self.device), self.cache)
        self.cache = out.past_key_values
        return out.last_hidden_state[0, -1]

    @torch.inference_mode()
    def observe(self, pair_embeds: torch.Tensor) -> float:
        """Append the visual tokens of the next pair and <act>; returns p_k."""
        if pair_embeds.shape[0] != self.state["per"]:
            raise ValueError(f"A pair has {self.state['per']} visual tokens, got {pair_embeds.shape[0]}")
        positions = self.family.advance(self.state, self.pair)
        act = self.model.qwen.get_input_embeddings()(torch.tensor([[self.model.act]], device=self.device))
        pair = self.family.project(self.model, pair_embeds.to(self.device))
        last = self._forward(torch.cat([pair[None].to(act.dtype), act], 1), positions)
        self.pair += 1
        return float(self.model.decision_logits(last[None]).sigmoid()[0])


def run_decision(session: DecisionSession, pairs, tau: float = 0.5) -> dict[str, Any]:
    """The inference loop: wake M_off at the first pair with p > tau; a video that ends without it is a miss (M_off
    then answers on the whole video)."""
    probabilities = []
    for pair in pairs:
        probabilities.append(session.observe(pair))
        if probabilities[-1] > tau:
            return {"fired_pair": len(probabilities) - 1, "missed": False, "probabilities": probabilities}
    return {"fired_pair": None, "missed": True, "probabilities": probabilities}


def time_decisions(model: SeparateModel, family, records: Sequence[Record], *,
                   split: ClevrerLoRASplitConfig) -> dict[str, Any]:
    """Wall time of one decision (a pair and <act> through the KV cache, the head) over every pair of ``records``,
    from cached visual tokens (the encoder is not counted); peak GPU memory of the sessions."""
    cache = VisionCache(split.vision_cache)
    device = model.qwen.get_input_embeddings().weight.device
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for record in records:
        entry = cache.load(record.video)
        patches = VisionCache.patches(entry)
        per = entry["embeds"].shape[0] // patches
        session = DecisionSession(model, family, record, entry)
        pairs = entry["embeds"].to(device)
        for k in range(patches):
            if cuda:
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            session.observe(pairs[k * per:(k + 1) * per])
            if cuda:
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - started)
    milliseconds = 1000 * np.asarray(times)
    return {"questions": len(records), "decisions": len(times), "ms_per_decision_mean": float(milliseconds.mean()),
            "ms_per_decision_median": float(np.median(milliseconds)),
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1024 ** 3 if cuda else None}
