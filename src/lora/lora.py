"""LoRA-only multi-prefix SFT infrastructure for Qwen2.5-VL-3B-Instruct.

This module reads the established CLEVRER-Proactive group-1 JSONL split and
uses its documented scene-index partitions. Other dataset formats can implement
``ValidationProvider`` without changing the SFT loop.

The only trainable weights accepted here are PEFT adapters in the language-model
attention projections.  StateHead is explicitly frozen and never receives a
forward call in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from hashlib import blake2b
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence

import torch
from torch.utils.data import DataLoader

from online_vllm.data.inputs import InputBuilder, answer_letter, format_prompt
from online_vllm.data.inputs import IGNORE
from online_vllm.data.labels import plan
from online_vllm.data.schema import Record, load_jsonl
from online_vllm.data.video import read_every
from online_vllm.models.config import ModelConfig
from online_vllm.models.model import attention_kernels


# Validation prefixes: the first patch only; up to the target patch; up to a patch drawn uniformly from the target to
# the end (fixed per question, like a training prefix); the whole video.
TIME_POINTS = ("start", "target", "answerable", "end")
VALIDATION_PREFIX_SEED = 0
# Fixed, unlike the training seed: every run is selected on the same validation half and tested on the same test half.
VALIDATION_SPLIT_SEED = 0
VIDEO_KEYS = ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts")
TOKEN_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids", "labels")
ANSWER_IDS_KEY = "answer_ids"
EXPECTED_LORA_LEAVES = frozenset(("q_proj", "k_proj", "v_proj", "o_proj"))
EXPECTED_LORA_TARGETS = 36 * len(EXPECTED_LORA_LEAVES)
# Mean accuracy over prefixes where the answer is already visible: what a good offline model for this task is for.
SELECTION_METRIC = "answerable_micro_accuracy"


@dataclass(frozen=True)
class LoRASFTConfig:
    """Optimization and checkpoint policy, independent of a future data schema."""

    learning_rate: float = 1.0e-4  # constant: no warmup, no decay
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    batch_size: int = 4  # right-padded sequences per forward pass
    gradient_accumulation: int = 1
    epochs: int = 3
    gradient_checkpointing: bool = True
    seed: int = 0
    overwrite: bool = False

    def validate(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate/grad_clip must be positive and weight_decay nonnegative")
        if len(self.betas) != 2 or any(not 0 <= beta < 1 for beta in self.betas):
            raise ValueError("betas must be two values in [0, 1)")
        if self.batch_size < 1 or self.gradient_accumulation < 1 or self.epochs < 1:
            raise ValueError("invalid batch_size, accumulation, or epochs")


@dataclass(frozen=True)
class PrefixSelection:
    """A temporal-patch endpoint selected without accessing any video frames."""

    last_patch: int
    mode: str  # after_target: uniform over [target, last patch]


@dataclass(frozen=True)
class ClevrerLoRASplitConfig:
    """The video-disjoint LoRA partitions documented in CLEVRER-Proactive README §11."""

    train: str = "processed/clevrer_proactive/train_group1.jsonl"
    validation: str = "processed/clevrer_proactive/val_group1.jsonl"
    train_videos: int = 5000
    validation_videos: int = 2000
    frame_stride: int = 1
    speak_delay: int = 0
    train_frame_size: tuple[int, int] = (252, 168)  # width, height
    num_workers: int = 2

    def validate(self) -> None:
        if self.train_videos < 1 or self.validation_videos < 1:
            raise ValueError("train_videos and validation_videos must be positive")
        if self.frame_stride < 1 or self.speak_delay < 0 or self.num_workers < 0:
            raise ValueError("invalid frame_stride, speak_delay, or num_workers")
        if len(self.train_frame_size) != 2 or any(side <= 0 or side % 28 for side in self.train_frame_size):
            raise ValueError("train_frame_size must be positive [width, height], multiples of 28")


def _stable_seed(seed: int, epoch: int, example_key: str) -> int:
    """Stable across Python processes, unlike ``hash(example_key)``."""
    payload = f"{seed}|{epoch}|{example_key}".encode("utf-8")
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big")


def sample_prefix(target_pair: int, num_temporal_patches: int, *, seed: int, epoch: int,
                  example_key: str) -> PrefixSelection:
    """Sample the prefix end uniformly from the target patch to the last patch of the video.

    ``last_patch`` is inclusive.  The caller must pass exactly
    ``(last_patch + 1) * temporal_patch`` decoded model frames to Qwen, which
    makes including later frames impossible at this boundary.
    """
    if num_temporal_patches < 1 or not 0 <= target_pair < num_temporal_patches:
        raise ValueError("target_pair must index an available temporal patch")
    rng = random.Random(_stable_seed(seed, epoch, example_key))
    return PrefixSelection(rng.randint(target_pair, num_temporal_patches - 1), "after_target")


def _records_for_first_videos(path: Path, video_limit: int) -> list[Record]:
    """Select every question from the first ``video_limit`` scene-indexed videos.

    The group-1 files are scene-index ordered, but sorting explicitly by the
    exported ``meta.scene_index`` makes the README partition invariant robust to
    question ordering inside a file.
    """
    records = list(load_jsonl(path))
    scene_by_video: dict[str, int] = {}
    for record in records:
        scene = record.meta.get("scene_index")
        if not isinstance(scene, int):
            raise ValueError(f"{record.id}: LoRA split needs integer meta.scene_index")
        previous = scene_by_video.setdefault(record.video, scene)
        if previous != scene:
            raise ValueError(f"{record.id}: one video has conflicting scene_index values")
    if len(scene_by_video) < video_limit:
        raise ValueError(f"{path}: requested {video_limit} videos, only {len(scene_by_video)} available")
    selected = {video for video, _ in sorted(scene_by_video.items(), key=lambda item: item[1])[:video_limit]}
    return [record for record in records if record.video in selected]


def _prefix_context(record: Record, *, data_root: Path, builder: InputBuilder, frame_stride: int,
                    speak_delay: int, frame_size: tuple[int, int]) -> tuple[Any, int, int]:
    """Decode once and return full sampled frames, target patch, and available patch count."""
    state_plan = plan(record, frame_stride, speak_delay, builder.temporal_patch)
    if state_plan is None:
        raise ValueError(f"{record.id}: target patch does not fit the schema video length")
    frames = read_every(data_root / record.video, frame_stride, frame_size)
    patches = len(frames) // builder.temporal_patch
    target = len(state_plan.labels) - 1
    if patches < 1 or target >= patches:
        raise ValueError(f"{record.id}: decoded video has {patches} temporal patches, needs target {target}")
    return frames[:patches * builder.temporal_patch], target, patches


def _build_sft_example(builder: InputBuilder, record: Record, frames: Any, *, fps: float) -> dict[str, Any]:
    """Use the existing Qwen processor/chat template and append the answer letter as the only supervised token.

    No EOS follows the letter: the answer is read as log P(letter) at the assistant position (as probe_logprob in the
    cached features), so the loss is exactly -log P(correct letter) and nothing else shares its weight."""
    inputs = builder._encode(format_prompt(record.question, record.choices), frames, fps)
    ids = inputs["input_ids"]
    tokenizer = builder.processor.tokenizer
    answer = answer_letter(record.choices, record.answers[0].answer)
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids[0]
    answer_ids = answer_ids.to(device=ids.device, dtype=ids.dtype)
    inputs["input_ids"] = torch.cat((ids, answer_ids[None]), dim=1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(answer_ids)[None]), dim=1)
    inputs["labels"] = torch.cat((torch.full_like(ids, IGNORE), answer_ids[None]), dim=1)
    inputs[ANSWER_IDS_KEY] = answer_ids
    return dict(inputs)


class ClevrerLoRASFTDataset(torch.utils.data.Dataset):
    """Multi-prefix SFT dataset for README §11's first LoRA training videos."""

    def __init__(self, path: Path, *, data_root: Path, builder: InputBuilder, split: ClevrerLoRASplitConfig,
                 seed: int):
        self.records = _records_for_first_videos(path, split.train_videos)
        self.data_root, self.builder, self.split, self.seed, self.epoch = Path(data_root), builder, split, seed, 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        frames, target, patches = _prefix_context(
            record, data_root=self.data_root, builder=self.builder, frame_stride=self.split.frame_stride,
            speak_delay=self.split.speak_delay, frame_size=self.split.train_frame_size)
        selection = sample_prefix(target, patches, seed=self.seed, epoch=self.epoch, example_key=record.id)
        count = (selection.last_patch + 1) * self.builder.temporal_patch
        return _build_sft_example(self.builder, record, frames[:count], fps=record.fps / self.split.frame_stride)


def _require_sft_model_config(config: ModelConfig) -> None:
    lora = config.lora
    if not lora.enabled or lora.component != "language_model":
        raise ValueError("SFT requires enabled language_model LoRA")
    if set(lora.modules) != EXPECTED_LORA_LEAVES or len(lora.modules) != len(EXPECTED_LORA_LEAVES):
        raise ValueError("SFT LoRA targets must be exactly q_proj, k_proj, v_proj, o_proj")
    if lora.layers is not None or (lora.r, lora.alpha, lora.dropout) != (32, 64, 0.05):
        raise ValueError("SFT requires all 36 layers with r=32, alpha=64, dropout=0.05")
    if config.backbone.dtype != "bfloat16" or config.backbone.quantization is not None:
        raise ValueError("SFT requires an unquantized bfloat16 backbone")


def assert_lora_isolation(model: torch.nn.Module, *, require_qwen_3b_target_count: bool = False) -> None:
    """Fail closed unless every trainable parameter is a LM-attention LoRA tensor."""
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable LoRA parameters")
    target_modules: set[str] = set()
    for name, _ in trainable:
        if not name.startswith("backbone.") or ".lora_" not in name:
            raise ValueError(f"Non-LoRA parameter is trainable: {name}")
        marker = "model.language_model.layers."
        if marker not in name:
            raise ValueError(f"LoRA is outside language_model: {name}")
        before_adapter = name.split(".lora_", 1)[0]
        leaf = before_adapter.rsplit(".", 1)[-1]
        if leaf not in EXPECTED_LORA_LEAVES:
            raise ValueError(f"LoRA is outside attention projections: {name}")
        target_modules.add(before_adapter)
    if require_qwen_3b_target_count and len(target_modules) != EXPECTED_LORA_TARGETS:
        raise ValueError(f"Expected {EXPECTED_LORA_TARGETS} Qwen2.5-VL-3B LoRA targets, got {len(target_modules)}")


def configure_gradient_checkpointing(model: torch.nn.Module, *, enabled: bool) -> None:
    """Apply and verify checkpointing before SFT, keeping frozen embeddings usable."""
    if not hasattr(model, "backbone"):
        raise TypeError("Expected a model with a PEFT backbone")
    backbone = model.backbone
    if enabled:
        if not hasattr(backbone, "gradient_checkpointing_enable") or not hasattr(backbone, "enable_input_require_grads"):
            raise RuntimeError("Installed PEFT/Transformers model lacks gradient-checkpointing support")
        backbone.gradient_checkpointing_enable()
        backbone.enable_input_require_grads()
        active = getattr(backbone, "is_gradient_checkpointing", None)
        if active is None and hasattr(backbone, "get_base_model"):
            active = getattr(backbone.get_base_model(), "is_gradient_checkpointing", None)
        if active is not True:
            raise RuntimeError("gradient_checkpointing=True but the backbone did not activate it")
    # Qwen's forward path is called with use_cache=False too, but the config
    # guard prevents accidental cache allocation from a future caller.
    if hasattr(backbone, "config"):
        backbone.config.use_cache = False


def prepare_lora_sft_model(model: torch.nn.Module, config: ModelConfig, *,
                           gradient_checkpointing: bool = True) -> torch.nn.Module:
    """Freeze StateHead and prepare an already-built project model for SFT.

    ``build_model`` remains the sole owner of Qwen/PEFT construction.  Calling
    ``enable_input_require_grads`` is necessary because checkpointed layers need
    a differentiable input although token embeddings are frozen.
    """
    _require_sft_model_config(config)
    if not hasattr(model, "backbone") or not hasattr(model, "state_head"):
        raise TypeError("Expected online_vllm.models.VideoStateModel")
    model.state_head.requires_grad_(False)
    configure_gradient_checkpointing(model, enabled=gradient_checkpointing)
    assert_lora_isolation(model, require_qwen_3b_target_count=True)
    return model


def _validate_example(example: Mapping[str, Any]) -> None:
    missing = set(TOKEN_KEYS + VIDEO_KEYS + (ANSWER_IDS_KEY,)) - set(example)
    if missing:
        raise KeyError(f"Prepared SFT example is missing keys: {sorted(missing)}")
    for key in TOKEN_KEYS:
        value = example[key]
        if not torch.is_tensor(value) or value.ndim != 2 or value.shape[0] != 1:
            raise ValueError(f"{key} must be a [1, sequence] tensor")
        if value.is_floating_point() or value.is_complex():
            raise ValueError(f"{key} must contain integer token IDs or labels")
    width = example["input_ids"].shape[1]
    if any(example[key].shape[1] != width for key in TOKEN_KEYS):
        raise ValueError("token tensors must have identical sequence length")
    if not torch.equal(example["attention_mask"].bool(), example["attention_mask"] == 1):
        raise ValueError("attention_mask must be binary")
    labels = example["labels"]
    if not (labels != IGNORE).any():
        raise ValueError("SFT example must supervise at least one answer token")
    if ((labels != IGNORE) & ~example["attention_mask"].bool()).any():
        raise ValueError("Supervised token is padded")
    answer_ids = example[ANSWER_IDS_KEY]
    if not torch.is_tensor(answer_ids) or answer_ids.ndim != 1 or answer_ids.numel() < 1:
        raise ValueError("answer_ids must be a nonempty 1D tensor")
    if answer_ids.is_floating_point() or answer_ids.is_complex() or answer_ids.dtype != example["input_ids"].dtype:
        raise ValueError("answer_ids must use the input_ids integer dtype")
    valid = int(example["attention_mask"].sum().item())
    if not example["attention_mask"][0, :valid].bool().all() or example["attention_mask"][0, valid:].bool().any():
        raise ValueError("Prepared SFT examples must use right padding only")
    supervised = (labels[0] != IGNORE).nonzero().flatten()
    expected = torch.arange(valid - answer_ids.numel(), valid, device=labels.device)
    if valid < answer_ids.numel() or not torch.equal(supervised, expected):
        raise ValueError("Only the contiguous terminal answer_ids tokens may be supervised")
    if not torch.equal(labels[0, supervised].to(answer_ids), answer_ids):
        raise ValueError("Supervised labels do not equal answer_ids")
    if not torch.equal(example["input_ids"][0, supervised].to(answer_ids), answer_ids):
        raise ValueError("answer_ids do not match input_ids at supervised positions")


def _validate_collated_sft_batch(batch: Mapping[str, Any]) -> None:
    """Recheck the answer-only contract immediately before the Qwen forward pass."""
    required = set(TOKEN_KEYS + (ANSWER_IDS_KEY,))
    if required - set(batch):
        raise KeyError(f"SFT batch is missing keys: {sorted(required - set(batch))}")
    ids, mask, labels, answers = (batch[key] for key in ("input_ids", "attention_mask", "labels", ANSWER_IDS_KEY))
    if not isinstance(answers, Sequence) or len(answers) != ids.shape[0]:
        raise ValueError("answer_ids must contain one tensor per batch row")
    for index, answer_ids in enumerate(answers):
        _validate_example({
            "input_ids": ids[index:index + 1], "attention_mask": mask[index:index + 1],
            "mm_token_type_ids": batch["mm_token_type_ids"][index:index + 1],
            "labels": labels[index:index + 1], "answer_ids": answer_ids,
            # These are not relevant after collation, but preserve the exact
            # processor-ready contract checked by _validate_example.
            "pixel_values_videos": torch.empty(0), "video_grid_thw": torch.empty(0),
            "second_per_grid_ts": torch.empty(0),
        })


def collate_sft(examples: Sequence[Mapping[str, Any]], pad_id: int) -> dict[str, Any]:
    """Collate processor-ready video SFT examples; labels stay -100 outside answers."""
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    for example in examples:
        _validate_example(example)
    length = max(example["input_ids"].shape[1] for example in examples)

    def pad(key: str, value: int) -> torch.Tensor:
        return torch.cat([torch.nn.functional.pad(example[key], (0, length - example[key].shape[1]), value=value)
                          for example in examples])

    return {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": pad("attention_mask", 0),
        "mm_token_type_ids": pad("mm_token_type_ids", 0),
        "labels": pad("labels", IGNORE),
        ANSWER_IDS_KEY: [example[ANSWER_IDS_KEY].clone() for example in examples],
        "pixel_values_videos": torch.cat([example["pixel_values_videos"] for example in examples]),
        "video_grid_thw": torch.cat([example["video_grid_thw"] for example in examples]),
        "second_per_grid_ts": torch.cat([torch.as_tensor(example["second_per_grid_ts"]) for example in examples]),
    }


def masked_sft_loss(model: torch.nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    """Qwen's shifted causal-LM loss on the answer tokens, with the vocabulary projection only where it is needed.

    Token j is predicted from position j - 1, so only the positions just before supervised tokens go through lm_head.
    The language-mode forward would project every position of every sequence instead: ~4 GB of bf16 logits for 4
    sequences of ~3500 tokens, plus their float32 copy and gradient in the loss. The value is the same.
    """
    _validate_collated_sft_batch(batch)
    labels = batch["labels"]
    inputs = {key: value for key, value in batch.items() if key not in (ANSWER_IDS_KEY, "labels")}
    supervised = labels != IGNORE
    predicting = torch.zeros_like(supervised)
    predicting[:, :-1] = supervised[:, 1:]
    # Both masks flatten in batch-then-position order, so features and targets stay paired.
    hidden = model.decision_features(inputs, predicting)
    logits = model.qwen.lm_head(hidden).float()
    loss = torch.nn.functional.cross_entropy(logits, labels[supervised])
    if not torch.isfinite(loss):
        raise RuntimeError("Qwen did not produce a finite language-model loss")
    return loss


@dataclass(frozen=True)
class ChoiceEvaluationExample:
    """Schema-free validation interface produced by a future dataset adapter.

    ``inputs_by_time`` has exactly the keys of ``TIME_POINTS``.
    Each value is a single, processor-ready prompt/video payload without answer
    tokens. ``choice_token_ids`` holds one or more token IDs for each option.
    """

    record_id: str
    answer_index: int
    choice_token_ids: Sequence[torch.Tensor]
    inputs_by_time: Mapping[str, Mapping[str, Any]]
    family: str | None = None


class ValidationProvider(Protocol):
    """Future JSON adapter contract; resolution is ``(width, height)``."""

    def iter_choice_examples(self, resolution: tuple[int, int]) -> Iterable[ChoiceEvaluationExample]:
        ...


def split_validation_videos(records: Sequence[Record], seed: int) -> tuple[list[Record], list[Record], dict[str, Any]]:
    """Halve the LoRA validation videos at random (seeded): one half selects the checkpoint during training, the other
    is the test, evaluated once afterwards. All questions of a video stay in one half."""
    by_video: dict[str, list[Record]] = {}
    for record in records:
        by_video.setdefault(record.video, []).append(record)
    videos = sorted(by_video)
    if len(videos) < 2:
        raise ValueError("Splitting into validation and test needs at least two videos")
    random.Random(seed).shuffle(videos)
    half = len(videos) // 2
    validation_videos, test_videos = sorted(videos[:half]), sorted(videos[half:])
    validation = [record for video in validation_videos for record in by_video[video]]
    test = [record for video in test_videos for record in by_video[video]]
    return validation, test, {"seed": seed, "validation_videos": validation_videos, "test_videos": test_videos,
                              "validation_questions": len(validation), "test_questions": len(test)}


class ClevrerLoRAValidationProvider:
    """Questions of one half of README §11's LoRA validation videos (the first 2,000 of ``val_group1``)."""

    def __init__(self, records: Sequence[Record], *, data_root: Path, builder: InputBuilder,
                 split: ClevrerLoRASplitConfig):
        self.records = list(records)
        self.data_root, self.builder, self.split = Path(data_root), builder, split

    def iter_choice_examples(self, resolution: tuple[int, int]) -> Iterable[ChoiceEvaluationExample]:
        if len(resolution) != 2 or any(side <= 0 or side % 28 for side in resolution):
            raise ValueError("evaluation resolution must be [width, height], multiples of 28")
        for record in self.records:
            frames, target, patches = _prefix_context(
                record, data_root=self.data_root, builder=self.builder, frame_stride=self.split.frame_stride,
                speak_delay=self.split.speak_delay, frame_size=resolution)
            choices = []
            for letter in "ABCDEFGH"[:len(record.choices)]:
                token_ids = self.builder.processor.tokenizer(letter, add_special_tokens=False).input_ids
                if not token_ids:
                    raise ValueError(f"{record.id}: empty tokenization for answer letter {letter}")
                choices.append(torch.tensor(token_ids, dtype=torch.long))
            answerable = sample_prefix(target, patches, seed=VALIDATION_PREFIX_SEED, epoch=0,
                                       example_key=record.id).last_patch
            prefixes = {"start": 0, "target": target, "answerable": answerable, "end": patches - 1}
            inputs = {
                point: dict(self.builder._encode(
                    format_prompt(record.question, record.choices),
                    frames[:(last_patch + 1) * self.builder.temporal_patch],
                    record.fps / self.split.frame_stride))
                for point, last_patch in prefixes.items()
            }
            yield ChoiceEvaluationExample(
                record_id=record.id, answer_index=record.choices.index(record.answers[0].answer),
                choice_token_ids=choices, inputs_by_time=inputs, family=record.meta.get("family"))


def build_clevrer_lora_data(*, data_root: str | Path, builder: InputBuilder, split: ClevrerLoRASplitConfig,
                            seed: int, batch_size: int = 1) \
        -> tuple[DataLoader, ClevrerLoRAValidationProvider, dict[str, Any]]:
    """Build the documented LoRA partitions: the training loader, the validation half of the LoRA validation videos,
    and split information (the data settings and both video lists) for ``split.json`` of the run."""
    split.validate()
    root = Path(data_root)
    train = ClevrerLoRASFTDataset(root / split.train, data_root=root, builder=builder, split=split, seed=seed)
    validation_records, _, info = split_validation_videos(
        _records_for_first_videos(root / split.validation, split.validation_videos), VALIDATION_SPLIT_SEED)
    validation = ClevrerLoRAValidationProvider(validation_records, data_root=root, builder=builder, split=split)
    loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=split.num_workers,
        collate_fn=partial(collate_sft, pad_id=builder.processor.tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(seed), persistent_workers=False)
    return loader, validation, {"data": asdict(split), **info}


def clevrer_lora_test_provider(run_dir: str | Path, *, data_root: str | Path,
                               builder: InputBuilder) -> tuple[ClevrerLoRAValidationProvider, ClevrerLoRASplitConfig]:
    """The test half recorded in ``split.json`` of a training run, with the data settings of that run."""
    info = json.loads((Path(run_dir) / "split.json").read_text(encoding="utf-8"))
    data = dict(info["data"])
    data["train_frame_size"] = tuple(data["train_frame_size"])
    split = ClevrerLoRASplitConfig(**data)
    split.validate()
    root = Path(data_root)
    test_videos = set(info["test_videos"])
    records = [record for record in _records_for_first_videos(root / split.validation, split.validation_videos)
               if record.video in test_videos]
    if len({record.video for record in records}) != len(test_videos) or len(records) != info["test_questions"]:
        raise ValueError(f"{root / split.validation} no longer matches split.json of {run_dir}: "
                         f"{len(records)} test questions, expected {info['test_questions']}")
    return ClevrerLoRAValidationProvider(records, data_root=root, builder=builder, split=split), split


def _move_inputs(inputs: Mapping[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def _candidate_score(model: torch.nn.Module, prompt: Mapping[str, Any], candidate: torch.Tensor,
                     device: torch.device | str) -> float:
    """Log P(candidate | prompt), including every candidate token."""
    if candidate.ndim != 1 or candidate.numel() < 1:
        raise ValueError("Each answer choice must be a nonempty 1D token tensor")
    inputs = _move_inputs(prompt, device)
    required = {"input_ids", "attention_mask", "mm_token_type_ids"}
    if required - set(inputs):
        raise KeyError(f"Validation prompt missing {sorted(required - set(inputs))}")
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("Choice scoring accepts one prompt at a time")
    candidate = candidate.to(device=device, dtype=inputs["input_ids"].dtype)[None]
    prefix = inputs["input_ids"].shape[1]
    inputs["input_ids"] = torch.cat((inputs["input_ids"], candidate), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(candidate)), dim=1)
    inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"],
                                               torch.zeros_like(candidate, dtype=inputs["mm_token_type_ids"].dtype)), dim=1)
    with torch.inference_mode():
        logits = model(inputs, mode="language").logits[0, prefix - 1:prefix - 1 + candidate.shape[1]].float()
    token_logprob = logits.log_softmax(-1).gather(1, candidate[0, :, None]).sum()
    return float(token_logprob.cpu())


def _choice_scores(model: torch.nn.Module, prompt: Mapping[str, Any], choices: Sequence[torch.Tensor], *,
                   device: torch.device | str) -> list[float]:
    """Score choices with one prompt pass when all answer choices are one token.

    The fast path is mathematically the first-token special case of
    :func:`_candidate_score`; multi-token candidates retain the full
    autoregressive fallback.
    """
    if not choices:
        raise ValueError("At least one answer choice is required")
    if not all(torch.is_tensor(choice) and choice.ndim == 1 and choice.numel() >= 1 for choice in choices):
        raise ValueError("Each answer choice must be a nonempty 1D token tensor")
    if not all(choice.numel() == 1 for choice in choices):
        return [_candidate_score(model, prompt, choice, device) for choice in choices]
    inputs = _move_inputs(prompt, device)
    required = {"input_ids", "attention_mask", "mm_token_type_ids"}
    if required - set(inputs):
        raise KeyError(f"Validation prompt missing {sorted(required - set(inputs))}")
    if inputs["input_ids"].shape[0] != 1 or not inputs["attention_mask"].bool().all():
        raise ValueError("One-token fast path requires one unpadded prompt")
    with torch.inference_mode():
        logits = model(inputs, mode="language").logits[0, -1].float()
    choice_ids = torch.stack([choice.to(device=logits.device, dtype=torch.long)[0] for choice in choices])
    return [float(value) for value in logits.log_softmax(-1)[choice_ids].cpu()]


@torch.inference_mode()
def evaluate_choice_examples(model: torch.nn.Module, examples: Iterable[ChoiceEvaluationExample], *,
                             resolution: tuple[int, int], device: torch.device | str) -> dict[str, Any]:
    """Exact multiple-choice conditional likelihood at start/target/end."""
    training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    try:
        for example in examples:
            if set(example.inputs_by_time) != set(TIME_POINTS):
                raise ValueError(f"inputs_by_time must contain exactly {', '.join(TIME_POINTS)}")
            if not 0 <= example.answer_index < len(example.choice_token_ids):
                raise ValueError(f"{example.record_id}: invalid answer_index")
            row = {"record_id": example.record_id, "family": example.family,
                   "answer_index": example.answer_index, "resolution": list(resolution), "predictions": {}}
            for point in TIME_POINTS:
                scores = _choice_scores(model, example.inputs_by_time[point], example.choice_token_ids, device=device)
                chosen = int(max(range(len(scores)), key=scores.__getitem__))
                row["predictions"][point] = {"chosen_index": chosen, "scores": scores,
                                              "correct": chosen == example.answer_index}
            rows.append(row)
    finally:
        model.train(training)

    metrics: dict[str, Any] = {"resolution": list(resolution), "count": len(rows), "predictions": rows}
    for point in TIME_POINTS:
        correct = [row["predictions"][point]["correct"] for row in rows]
        metrics[f"{point}_micro_accuracy"] = None if not correct else float(sum(correct) / len(correct))
        families: dict[str, list[bool]] = {}
        for row in rows:
            if row["family"] is not None:
                families.setdefault(row["family"], []).append(row["predictions"][point]["correct"])
        per_family = {family: float(sum(values) / len(values)) for family, values in sorted(families.items())}
        metrics[f"{point}_family_accuracy"] = per_family
        metrics[f"{point}_macro_accuracy"] = None if not per_family else float(sum(per_family.values()) / len(per_family))
    return metrics


def evaluate_validation(model: torch.nn.Module, provider: ValidationProvider, *,
                        resolutions: Sequence[tuple[int, int]], device: torch.device | str) -> dict[str, Any]:
    """Evaluate every requested resolution separately; results are never averaged."""
    result: dict[str, Any] = {}
    for width, height in resolutions:
        key = f"{width}x{height}"
        result[key] = evaluate_choice_examples(model, provider.iter_choice_examples((width, height)),
                                               resolution=(width, height), device=device)
    return result


def save_lora_adapter(model: torch.nn.Module, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> Path:
    """Persist only the PEFT adapter; no base Qwen or StateHead tensors are serialized."""
    assert_lora_isolation(model)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path, safe_serialization=True)
    payload = {"format": "online_video_llm_lora_sft_v1", **(dict(metadata) if metadata else {})}
    (path / "sft_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_lora_adapter(model: torch.nn.Module, path: str | Path) -> torch.nn.Module:
    """Load an adapter written by :func:`save_lora_adapter` into a model built with the same LoRA settings."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    weights = load_file(str(Path(path) / "adapter_model.safetensors"))
    expected = get_peft_model_state_dict(model.backbone)
    if expected.keys() != weights.keys():
        raise ValueError(f"Adapter {path} does not match the model's LoRA configuration")
    set_peft_model_state_dict(model.backbone, weights)
    return model


def _set_epoch(loader: DataLoader, epoch: int) -> None:
    dataset = loader.dataset
    setter = getattr(dataset, "set_epoch", None)
    if setter is not None:
        setter(epoch)


def _prepare_output_dir(path: str | Path, *, overwrite: bool) -> Path:
    """Create an empty result directory, never silently mixing experiments."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to write into nonempty output_dir {path}; pass overwrite=True for a new run")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_new_best(score: float | None, best_score: float | None) -> bool:
    """Strict comparison preserves the earlier checkpoint when accuracies tie."""
    return score is not None and (best_score is None or score > best_score)


def _accuracy_deltas(validation: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Add start/target/end improvements in percentage points without mixing resolutions."""
    result: dict[str, Any] = {}
    for resolution, values in validation.items():
        reference = baseline[resolution]
        enriched = dict(values)
        for point in TIME_POINTS:
            current, initial = values[f"{point}_micro_accuracy"], reference[f"{point}_micro_accuracy"]
            enriched[f"{point}_accuracy_delta_pp"] = None if current is None or initial is None \
                else 100.0 * (current - initial)
        result[resolution] = enriched
    return result


def _train_epoch(model: torch.nn.Module, train_loader: DataLoader, *, optimizer: torch.optim.Optimizer,
                 gradient_accumulation: int, grad_clip: float, parameters: Sequence[torch.nn.Parameter],
                 device: torch.device | str, epoch: int = 1, global_step_start: int = 0,
                 loss_fn=masked_sft_loss) -> tuple[float, int, list[dict[str, Any]]]:
    """Run one epoch, normalizing every accumulation window by its actual size."""
    losses: list[float] = []
    steps = 0
    step_records: list[dict[str, Any]] = []
    batches = len(train_loader)
    optimizer.zero_grad(set_to_none=True)
    window_losses: list[float] = []
    for batch_index, batch in enumerate(train_loader, 1):
        # For a short final window (e.g. 3 batches with accumulation 2), this
        # is 1 rather than the nominal accumulation factor 2.
        window_start = ((batch_index - 1) // gradient_accumulation) * gradient_accumulation
        window_size = min(gradient_accumulation, batches - window_start)
        moved = _move_inputs(batch, device)
        loss = loss_fn(model, moved)
        # Gradient checkpointing recomputes each layer during backward. The forward chose attention kernels under
        # attention_kernels(); the recomputation must choose the same ones, or it saves differently shaped tensors
        # (e.g. the math kernel's [B, heads, S, S] weights instead of the fused kernel's mask) and checkpoint fails.
        with attention_kernels():
            (loss / window_size).backward()
        value = float(loss.detach().cpu())
        losses.append(value)
        window_losses.append(value)
        if batch_index % gradient_accumulation and batch_index != batches:
            continue
        # clip_grad_norm_ returns the total norm before clipping.
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, grad_clip))
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        step_records.append({
            "global_step": global_step_start + steps,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": float(sum(window_losses) / len(window_losses)),
            "gradient_norm": gradient_norm,
            "effective_accumulation_steps": window_size,
        })
        window_losses = []
    return float(sum(losses) / len(losses)), steps, step_records


def _print_epoch_summary(epoch: int, total_epochs: int, metrics: Mapping[str, Any]) -> None:
    """One compact line block per epoch; optimizer-step records stay in JSONL only."""
    loss = "-" if metrics["train_loss"] is None else f"{metrics['train_loss']:.6f}"
    peak = "-" if metrics["peak_gpu_memory_gb"] is None else f"{metrics['peak_gpu_memory_gb']:.2f}GB"
    print(f"Epoch {epoch}/{total_epochs} | loss={loss} | lr={metrics['learning_rate']:.3e} | "
          f"steps={metrics['optimizer_steps']} | time={metrics['epoch_duration_seconds']:.1f}s | peak_vram={peak}")
    for resolution, values in metrics["validation"].items():
        segments = []
        for point in TIME_POINTS:
            accuracy, delta = values[f"{point}_micro_accuracy"], values[f"{point}_accuracy_delta_pp"]
            text = "-" if accuracy is None else f"{100 * accuracy:.2f}%"
            shift = "-" if delta is None else f"{delta:+.2f}pp"
            segments.append(f"{point}={text} ({shift})")
        print(f"{resolution} | " + " | ".join(segments))
    score = "-" if metrics["best_score"] is None else f"{100 * metrics['best_score']:.2f}%"
    print(f"Best | epoch={metrics['best_epoch']} | {metrics['selection_metric']}={score} | "
          f"resolution={metrics['selection_resolution']}")


def train_lora_sft(model: torch.nn.Module, train_loader: DataLoader, config: LoRASFTConfig, *,
                   output_dir: str | Path, device: torch.device | str,
                   validation_provider: ValidationProvider | None = None,
                   validation_resolution: tuple[int, int] = (252, 168),
                   split_info: Mapping[str, Any] | None = None) -> Path:
    """Train an externally supplied prepared dataset and checkpoint adapters.

    The future dataset owns loading, temporal conversion, and video preprocessing.
    It must yield examples accepted by :func:`collate_sft`; if it supports
    ``set_epoch(epoch)``, multi-prefix sampling remains reproducible per epoch.
    Validation runs before training and after every epoch at one resolution, ``(width, height)``.
    """
    config.validate()
    configure_gradient_checkpointing(model, enabled=config.gradient_checkpointing)
    assert_lora_isolation(model, require_qwen_3b_target_count=True)
    if len(train_loader) == 0:
        raise ValueError("train_loader must contain at least one prepared SFT batch")
    output_dir = _prepare_output_dir(output_dir, overwrite=config.overwrite)
    (output_dir / "sft_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    # What scripts/evaluate_lora_sft.py needs to rebuild the model and the test half of the run.
    if hasattr(model, "config") and hasattr(model.config, "save"):
        model.config.save(output_dir / "model.yaml")
    if split_info is not None:
        (output_dir / "split.json").write_text(json.dumps(split_info, indent=2) + "\n", encoding="utf-8")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_args: dict[str, Any] = {"lr": config.learning_rate, "betas": config.betas,
                                      "weight_decay": config.weight_decay}
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        optimizer_args["fused"] = True
    optimizer = torch.optim.AdamW(parameters, **optimizer_args)
    resolution = f"{validation_resolution[0]}x{validation_resolution[1]}"
    metrics_file = output_dir / "validation_metrics.jsonl"
    steps_file = output_dir / "train_steps.jsonl"
    best_epoch: int | None = None
    best_score: float | None = None
    baseline: dict[str, Any] | None = None
    global_step = 0

    def evaluate_and_save(epoch: int, *, train_loss: float | None, optimizer_steps: int,
                          epoch_duration_seconds: float, peak_gpu_memory_gb: float | None) -> None:
        nonlocal baseline, best_epoch, best_score
        if validation_provider is None:
            return
        validation = evaluate_validation(model, validation_provider, resolutions=(validation_resolution,),
                                         device=device)
        no_predictions = {key: {k: v for k, v in value.items() if k != "predictions"}
                          for key, value in validation.items()}
        if baseline is None:
            baseline = no_predictions
        validation_with_delta = _accuracy_deltas(no_predictions, baseline)
        score = validation_with_delta[resolution][SELECTION_METRIC]
        is_best = _is_new_best(score, best_score)
        if is_best:
            best_epoch, best_score = epoch, score
            save_lora_adapter(model, output_dir / "best", metadata={
                "epoch": epoch, "best_epoch": best_epoch, "best_score": best_score,
                "selection_metric": SELECTION_METRIC, "selection_resolution": resolution,
                "is_best_checkpoint": True,
            })
        metrics = {
            "epoch": epoch, "train_loss": train_loss, "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimizer_steps": optimizer_steps, "epoch_duration_seconds": epoch_duration_seconds,
            "peak_gpu_memory_gb": peak_gpu_memory_gb, "selection_metric": SELECTION_METRIC,
            "selection_resolution": resolution, "best_epoch": best_epoch,
            "best_score": best_score, "is_best_checkpoint": is_best, "validation": validation_with_delta,
        }
        with metrics_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, allow_nan=False) + "\n")
        (output_dir / f"validation_predictions_epoch_{epoch}.json").write_text(
            json.dumps(validation, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _print_epoch_summary(epoch, config.epochs, metrics)

    evaluate_and_save(0, train_loss=None, optimizer_steps=0, epoch_duration_seconds=0.0, peak_gpu_memory_gb=None)
    for epoch in range(1, config.epochs + 1):
        _set_epoch(train_loader, epoch)
        model.train()
        cuda = torch.cuda.is_available() and torch.device(device).type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        mean_loss, epoch_steps, step_records = _train_epoch(
            model, train_loader, optimizer=optimizer,
            gradient_accumulation=config.gradient_accumulation, grad_clip=config.grad_clip,
            parameters=parameters, device=device, epoch=epoch, global_step_start=global_step)
        global_step += epoch_steps
        peak = None if not cuda else torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        with steps_file.open("a", encoding="utf-8") as file:
            for record in step_records:
                file.write(json.dumps(record, allow_nan=False) + "\n")
        evaluate_and_save(epoch, train_loss=mean_loss, optimizer_steps=epoch_steps,
                          epoch_duration_seconds=time.perf_counter() - started, peak_gpu_memory_gb=peak)
        save_lora_adapter(model, output_dir / "last", metadata={"epoch": epoch,
                          "train_loss": mean_loss, "lr": config.learning_rate})
    return output_dir / "best" if (output_dir / "best").exists() else output_dir / "last"
