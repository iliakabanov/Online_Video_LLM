"""Online SFT: after every pair of frames Qwen writes a control token, <listen> (keep watching) or <speak> (answer now).

Starting point: the offline LoRA SFT adapter (lora/lora.py) merged into Qwen; on top of it a new LoRA of the same
settings and the embedding rows of the two new tokens are trained (PEFT trainable_token_indices; in Qwen2.5-VL-3B the
embedding and lm_head are tied, so one row per token serves both).

One question is one forward pass, in training and in evaluation:

    main      [prompt <|vision_start|>] [pair 0] ▢ [pair 1] ▢ ... [pair P-1] ▢
    branches  [<speak> <tail>]_s ... one per answer pair s

Every slot ▢ holds <listen> on the input: the history of a policy that has not answered yet. The target of slot k is
predicted from the last visual token of pair k (before the slot), so the slot's input never affects its own target:
<listen> before the target pair t* (the first pair whose last frame shows the answer), <speak> from t* on; with the
step and sigmoid tags a soft target y_k, loss -(y log P(<speak>) - (1 - y) log P(<listen>)) over the full vocabulary,
so that the two tokens also take the probability mass (hard tags: plain cross-entropy on the token).

A branch [<speak> <tail>] after pair s is what the model sees when it speaks there: <speak> and the template tail
(<|vision_end|><|im_end|>\\n<|im_start|>assistant\\n), then the answer letter, supervised at the last tail token. Through
the attention mask a branch sees the main sequence up to the last visual token of pair s, without the <listen> of
slot s, and its own tokens; the main sequence never sees a branch. So a branch equals a separate sequence that ends
with <speak> after pair s (tests/test_online_sft.py). Training uses two branches: s ~ U(t*, P - 1), drawn anew every
epoch, and the end of the video P - 1. Evaluation uses a branch after every pair.

3D positions are those of offline training: the visual tokens keep the positions of the contiguous video
(models/positions.py), and a slot after pair k stands at the largest position so far + 1, where the offline template
tail started. A branch continues from there: <speak> at that position, the tail after it. Before slot k the model sees
exactly the positions of the offline prefix up to pair k.

Inference (OnlineSession): the prompt goes into the KV cache, then for every pair its visual tokens are appended (one
pass over the new tokens) and p_k = softmax(logit <speak>, logit <listen>)[<speak>] is read at the last one. If
p_k > tau, <speak> and the tail are appended and the answer letter is read; otherwise <listen> is appended. A video
that ends without <speak> gets <speak> right after its last pair (no <listen> before it, as the end-of-video branch of
training) and counts as a miss.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from online_vllm.data.inputs import LETTERS, InputBuilder, answer_letter, format_prompt
from online_vllm.data.schema import Record
from online_vllm.eval.thresholds import sweep
from online_vllm.models.config import ModelConfig
from online_vllm.models.model import attention_kernels
from online_vllm.models.positions import vision_positions

from .lora import (
    EXPECTED_LORA_LEAVES,
    VALIDATION_PREFIX_SEED,
    ClevrerLoRASplitConfig,
    LoRASFTConfig,
    _LengthGroupedBatches,
    _move_inputs,
    _prepare_output_dir,
    _set_epoch,
    _target_patch,
    _train_epoch,
    configure_gradient_checkpointing,
    load_lora_adapter,
    sample_prefix,
)
from .vision_cache import VisionCache

LISTEN, SPEAK = "<listen>", "<speak>"
# The new rows start at the mean embedding of these words.
INIT_WORDS = {LISTEN: "listen", SPEAK: "speak"}
TAGS = ("hard", "step", "sigmoid")
SELECTION_METRIC = "loss"  # validation loss, lower is better
ONLINE_FORMAT = "online_video_llm_online_sft_v1"


@dataclass(frozen=True)
class ControlTokens:
    listen: int
    speak: int


@dataclass(frozen=True)
class OnlineObjective:
    """Slot targets (uhead/labels.py make_targets) and the weight of the answer loss."""

    tags: str = "hard"
    step_window: int = 5
    sigmoid_slope: float = 0.65
    answer_weight: float = 1.0

    def validate(self) -> None:
        if self.tags not in TAGS:
            raise ValueError(f"tags must be one of {TAGS}: after the target pair every slot should say <speak>")
        if self.step_window < 0 or self.sigmoid_slope <= 0 or self.answer_weight < 0:
            raise ValueError("invalid step_window, sigmoid_slope or answer_weight")

    def targets(self, patches: int, target: int) -> torch.Tensor:
        from uhead.labels import make_targets
        return torch.from_numpy(make_targets(patches, target, self.tags, step_window=self.step_window,
                                             sigmoid_slope=self.sigmoid_slope))


# ---------------------------------------------------------------------------------------------------------------- model

def add_control_tokens(tokenizer) -> ControlTokens:
    """Add <listen> and <speak> as special tokens (once) and return their ids."""
    from transformers import AddedToken
    vocab = tokenizer.get_vocab()
    new = [AddedToken(token, special=True, normalized=False) for token in (LISTEN, SPEAK) if token not in vocab]
    if new:
        tokenizer.add_tokens(new, special_tokens=True)
    ids = tokenizer.convert_tokens_to_ids([LISTEN, SPEAK])
    for token, value in zip((LISTEN, SPEAK), ids):
        if tokenizer(token, add_special_tokens=False).input_ids != [value]:
            raise RuntimeError(f"{token} is not a single token of its own")
    return ControlTokens(*ids)


def _tied(qwen) -> bool:
    return qwen.lm_head.weight.data_ptr() == qwen.get_input_embeddings().weight.data_ptr()


@torch.no_grad()
def init_control_rows(qwen, tokenizer, tokens: ControlTokens) -> None:
    """Set the rows of the new tokens to the mean rows of their INIT_WORDS: in the input embedding and, unless tied, in
    lm_head. Deterministic, so a saved run rebuilds the same starting rows (its adapter stores only the change)."""
    embed = qwen.get_input_embeddings().weight
    if max(tokens.listen, tokens.speak) >= embed.shape[0]:
        raise ValueError(f"Token ids {tokens} do not fit the {embed.shape[0]} embedding rows")
    matrices = [embed] if _tied(qwen) else [embed, qwen.lm_head.weight]
    for token, word in INIT_WORDS.items():
        index = tokens.listen if token == LISTEN else tokens.speak
        words = tokenizer(word, add_special_tokens=False).input_ids
        for matrix in matrices:
            matrix[index] = matrix[words].float().mean(0).to(matrix.dtype)


def build_online_model(config: ModelConfig, tokenizer, *, init_adapter: str | Path | None,
                       local_files_only: bool = False, device: str | None = None):
    """Qwen with the offline adapter merged in, the control tokens, and a new LoRA (``config.lora``) plus trainable
    rows of the two tokens. Returns the VideoStateModel (its StateHead frozen and unused) and the token ids; the tokens
    are added to ``tokenizer``."""
    from peft import LoraConfig, TaskType, get_peft_model
    from online_vllm.models import build_model
    from online_vllm.models.builder import resolve_targets
    from online_vllm.models.model import VideoStateModel

    if not config.lora.enabled or config.lora.component != "language_model":
        raise ValueError("Online SFT needs a language_model LoRA config (that of the offline adapter)")
    offline = build_model(config, local_files_only=local_files_only, device=device)
    if init_adapter is not None:
        load_lora_adapter(offline, init_adapter)
    merged = offline.backbone.merge_and_unload()  # without an adapter: LoRA B = 0, the base itself
    config = offline.config  # with the resolved revision
    tokens = add_control_tokens(tokenizer)
    init_control_rows(merged, tokenizer, tokens)
    rows = [tokens.listen, tokens.speak]
    backbone = get_peft_model(merged, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config.lora.r, lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout, bias="none", target_modules=resolve_targets(merged, config.lora),
        trainable_token_indices=rows if _tied(merged) else {"embed_tokens": rows, "lm_head": rows}))
    model = VideoStateModel(backbone, config)
    model.state_head.requires_grad_(False)
    return model, tokens


def expected_lora_targets(model) -> int:
    """LoRA modules that model.config.lora asks for: every chosen layer times every module."""
    lora = model.config.lora
    layers = model.qwen.config.text_config.num_hidden_layers if lora.layers is None else len(lora.layers)
    return layers * len(lora.modules)


def assert_online_isolation(model: torch.nn.Module, *, expected_targets: int | None = None) -> None:
    """Only the language-model attention LoRA and the control-token rows may train."""
    targets, token_rows = set(), 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "trainable_tokens_delta" in name:
            token_rows += 1
            continue
        if not name.startswith("backbone.") or ".lora_" not in name or "model.language_model.layers." not in name:
            raise ValueError(f"Trainable parameter outside the LM-attention LoRA and the token rows: {name}")
        module = name.split(".lora_", 1)[0]
        if module.rsplit(".", 1)[-1] not in EXPECTED_LORA_LEAVES:
            raise ValueError(f"LoRA is outside attention projections: {name}")
        targets.add(module)
    if not targets or not token_rows:
        raise ValueError("Online SFT needs trainable LoRA and trainable control-token rows")
    if expected_targets is not None and len(targets) != expected_targets:
        raise ValueError(f"Expected {expected_targets} LoRA targets, got {len(targets)}")


def save_online_adapter(model: torch.nn.Module, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> Path:
    """The new LoRA and the change of the two token rows (PEFT stores both in adapter_model.safetensors)."""
    assert_online_isolation(model)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path, safe_serialization=True)
    payload = {"format": ONLINE_FORMAT, **(dict(metadata) if metadata else {})}
    (path / "sft_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_online_run(run_dir: str | Path, adapter: str = "best", *, local_files_only: bool = False,
                    device: str | None = None):
    """Rebuild a trained online model: the offline adapter of the run merged in, the tokens, and ``<run>/<adapter>``.
    Returns (model, builder, tokens, info of online.json)."""
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "online.json").read_text(encoding="utf-8"))
    config = ModelConfig.load(run_dir / "model.yaml")
    builder = InputBuilder.from_config(config, local_files_only=local_files_only)
    model, tokens = build_online_model(config, builder.processor.tokenizer, init_adapter=info["init_adapter"],
                                       local_files_only=local_files_only, device=device)
    if asdict(tokens) != info["tokens"]:
        raise ValueError(f"Token ids {asdict(tokens)} differ from those of the run {info['tokens']}")
    load_lora_adapter(model, run_dir / adapter)
    return model.eval(), builder, tokens, info


# --------------------------------------------------------------------------------------------------- sequence layout

@dataclass(frozen=True)
class Template:
    """Token ids around the video of one prompt: everything before the first visual token (ends with
    <|vision_start|>) and the tail after the video; the video placeholder id; visual tokens per pair."""

    prefix: list[int]
    tail: list[int]
    video_token: int
    per_pair: int


@dataclass(frozen=True)
class RopeSpec:
    merge: int
    tokens_per_second: float

    @classmethod
    def of(cls, model) -> "RopeSpec":
        vision = model.qwen.config.vision_config
        return cls(vision.spatial_merge_size, float(vision.tokens_per_second))


def template(builder: InputBuilder, prompt: str, grid: torch.Tensor, second_per_grid_ts: torch.Tensor) -> Template:
    """The offline input of the prompt and a video of this grid, cut around its (contiguous) visual tokens."""
    x = builder._encode_tokens(prompt, grid, second_per_grid_ts)
    ids, types = x["input_ids"][0], x["mm_token_type_ids"][0]
    video = (types == 2).nonzero()[:, 0]
    start, end = int(video[0]), int(video[-1]) + 1
    patches = int(torch.as_tensor(grid).reshape(3)[0])
    if end - start != len(video) or len(video) % patches:
        raise ValueError("Visual tokens are not one contiguous video of whole pairs")
    return Template(ids[:start].tolist(), ids[end:].tolist(), int(ids[start]), len(video) // patches)


def build_example(tpl: Template, entry: Mapping[str, torch.Tensor], *, spec: RopeSpec, tokens: ControlTokens,
                  slot_targets: torch.Tensor, branch_pairs: Sequence[int], answer: int) -> dict[str, Any]:
    """One question as one sequence: main (prefix, pairs, a <listen> slot after each) and a branch per answer pair.

    Tensors: input_ids, mm_token_type_ids [L]; position_ids [3, L]; attention [L, L] (bool, True = may attend);
    video_embeds [P * per_pair, D] from the vision cache; slot_index [P] (where slot k's target is predicted: the last
    visual token of pair k) with slot_target [P] (P(<speak>) wanted); answer_index [B] (the last tail token of each
    branch) with answer_target [B] (the correct letter) and branch_pair [B]."""
    grid = entry["grid"].reshape(3)
    patches = int(grid[0])
    if len(slot_targets) != patches:
        raise ValueError(f"{len(slot_targets)} slot targets for {patches} pairs")
    if not branch_pairs or not all(0 <= s < patches for s in branch_pairs):
        raise ValueError("branch_pairs must be pairs of the video")
    n, per = len(tpl.prefix), tpl.per_pair
    video_pos = vision_positions(n, grid.tolist(), spec.merge, float(entry["second_per_grid_ts"].reshape(-1)[0]),
                                 spec.tokens_per_second)
    if video_pos.shape[1] != patches * per:
        raise ValueError("Visual positions do not match the template")
    ids, types = list(tpl.prefix), [0] * n
    positions = [torch.arange(n).expand(3, -1)]
    reach, reaches, last_visual = n - 1, [], []
    for k in range(patches):
        chunk = video_pos[:, k * per:(k + 1) * per]
        ids += [tpl.video_token] * per
        types += [2] * per
        positions.append(chunk)
        reach = max(reach, int(chunk.max()))
        reaches.append(reach)
        last_visual.append(len(ids) - 1)
        ids.append(tokens.listen)
        types.append(0)
        positions.append(torch.full((3, 1), reach + 1))
    main = len(ids)
    branch = [tokens.speak] + list(tpl.tail)
    spans, answer_index = [], []
    for s in branch_pairs:
        spans.append((len(ids), len(ids) + len(branch), s))
        ids += branch
        types += [0] * len(branch)
        positions.append(torch.arange(reaches[s] + 1, reaches[s] + 1 + len(branch)).expand(3, -1))
        answer_index.append(len(ids) - 1)
    length = len(ids)
    attention = torch.zeros(length, length, dtype=torch.bool)
    attention[:main, :main] = torch.ones(main, main, dtype=torch.bool).tril()
    own = torch.ones(len(branch), len(branch), dtype=torch.bool).tril()
    for start, end, s in spans:
        attention[start:end, :last_visual[s] + 1] = True  # up to pair s: not slot s, not later pairs
        attention[start:end, start:end] = own
    return {
        "input_ids": torch.tensor(ids), "mm_token_type_ids": torch.tensor(types),
        "position_ids": torch.cat(positions, 1).long(), "attention": attention,
        "video_embeds": entry["embeds"][:patches * per],
        "slot_index": torch.tensor(last_visual), "slot_target": slot_targets.float(),
        "answer_index": torch.tensor(answer_index), "answer_target": torch.full((len(branch_pairs),), answer),
        "branch_pair": torch.tensor(list(branch_pairs)),
    }


def collate_online(examples: Sequence[Mapping[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    """Right padding to the longest sequence; a padded row attends only to itself. Indices of slots and branches become
    (row, index) pairs of the batch."""
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    length = max(len(example["input_ids"]) for example in examples)
    batch = len(examples)
    ids = torch.full((batch, length), pad_id, dtype=torch.long)
    types = torch.zeros(batch, length, dtype=torch.long)
    positions = torch.zeros(3, batch, length, dtype=torch.long)
    attention = torch.eye(length, dtype=torch.bool).repeat(batch, 1, 1, 1)
    for row, example in enumerate(examples):
        n = len(example["input_ids"])
        ids[row, :n], types[row, :n] = example["input_ids"], example["mm_token_type_ids"]
        positions[:, row, :n] = example["position_ids"]
        attention[row, 0, :n, :n] = example["attention"]

    def rows(key: str) -> torch.Tensor:
        return torch.cat([torch.full((len(example[key]),), row) for row, example in enumerate(examples)])

    def cat(key: str) -> torch.Tensor:
        return torch.cat([example[key] for example in examples])

    return {"input_ids": ids, "mm_token_type_ids": types, "position_ids": positions, "attention_mask": attention,
            "video_embeds": cat("video_embeds"),
            "slot_row": rows("slot_index"), "slot_index": cat("slot_index"), "slot_target": cat("slot_target"),
            "answer_row": rows("answer_index"), "answer_index": cat("answer_index"),
            "answer_target": cat("answer_target"), "branch_pair": cat("branch_pair")}


def online_hidden(model, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Last hidden states [B, L, D] of a collated batch, the cached video tokens in place of the placeholders."""
    qwen = model.qwen.model
    ids = batch["input_ids"]
    embeds = qwen.get_input_embeddings()(ids)
    video = batch["video_embeds"].to(embeds.device, embeds.dtype)
    placeholders = ids == model.qwen.config.video_token_id
    if int(placeholders.sum()) != video.shape[0]:
        raise ValueError(f"{int(placeholders.sum())} video placeholders, {video.shape[0]} cached video tokens")
    embeds = embeds.masked_scatter(placeholders[..., None].expand_as(embeds), video)
    qwen.rope_deltas = None
    with attention_kernels():
        return qwen(inputs_embeds=embeds, position_ids=batch["position_ids"], attention_mask=batch["attention_mask"],
                    use_cache=False, return_dict=True).last_hidden_state


def _logprobs(model, hidden: torch.Tensor) -> torch.Tensor:
    return model.qwen.lm_head(hidden).float().log_softmax(-1)


def online_losses(model, batch: Mapping[str, torch.Tensor], tokens: ControlTokens) -> dict[str, torch.Tensor]:
    """Slot loss -(y log P(<speak>) + (1 - y) log P(<listen>)), mean over all slots of the batch, and the answer
    cross-entropy at the last tail token of every branch, mean over the branches."""
    hidden = online_hidden(model, batch)
    slots = _logprobs(model, hidden[batch["slot_row"], batch["slot_index"]])
    y = batch["slot_target"].to(slots)
    slot = -(y * slots[:, tokens.speak] + (1 - y) * slots[:, tokens.listen]).mean()
    answers = _logprobs(model, hidden[batch["answer_row"], batch["answer_index"]])
    answer = torch.nn.functional.nll_loss(answers, batch["answer_target"])
    return {"slot": slot, "answer": answer, "slots": slots, "answers": answers}


def online_loss(model, batch: Mapping[str, torch.Tensor], *, tokens: ControlTokens, answer_weight: float) -> torch.Tensor:
    parts = online_losses(model, batch, tokens)
    loss = parts["slot"] + answer_weight * parts["answer"]
    if not torch.isfinite(loss):
        raise RuntimeError("The online loss is not finite")
    return loss


# --------------------------------------------------------------------------------------------------------------- data

def online_training_records(path: str | Path, *, skip_videos: int, videos: int | None) -> list[Record]:
    """Training questions of an online run: the videos of a LoRA training file in scene_index order, the first
    ``skip_videos`` left out (those the offline adapter learned: it knows their answers, so the slots would learn its
    memory instead of what the frames show), then the first ``videos`` of the rest (None: all of them)."""
    from .lora import _records_for_first_videos
    if skip_videos < 0 or (videos is not None and videos < 1):
        raise ValueError("skip_videos must be nonnegative and videos positive or None")
    records = _records_for_first_videos(Path(path), None)
    scenes = {record.video: record.meta["scene_index"] for record in records}
    order = sorted(scenes, key=scenes.__getitem__)
    rest = order[skip_videos:]
    if not rest:
        raise ValueError(f"{path}: all {len(order)} videos are among the {skip_videos} of the offline adapter")
    if videos is not None and videos > len(rest):
        raise ValueError(f"{path}: {videos} videos requested, {len(rest)} left after the first {skip_videos}")
    chosen = set(rest if videos is None else rest[:videos])
    return [record for record in records if record.video in chosen]


def _letter_ids(tokenizer, count: int) -> list[int]:
    ids = [tokenizer(letter, add_special_tokens=False).input_ids for letter in LETTERS[:count]]
    if any(len(value) != 1 for value in ids):
        raise ValueError("Option letters must be single tokens")
    return [value[0] for value in ids]


@dataclass(frozen=True)
class _Question:
    """Everything a sequence of one question needs that does not depend on the epoch."""

    record: Record
    template: Template
    target: int
    patches: int
    letters: list[int]


def _question(record: Record, entry, builder: InputBuilder, split: ClevrerLoRASplitConfig) -> _Question:
    patches = VisionCache.patches(entry)
    target = _target_patch(record, builder=builder, frame_stride=split.frame_stride, speak_delay=split.speak_delay,
                           patches=patches)
    tpl = template(builder, format_prompt(record.question, record.choices), entry["grid"], entry["second_per_grid_ts"])
    return _Question(record, tpl, target, patches, _letter_ids(builder.processor.tokenizer, len(record.choices)))


def _answer_id(question: _Question) -> int:
    record = question.record
    return question.letters[LETTERS.index(answer_letter(record.choices, record.answers[0].answer))]


class OnlineSFTDataset(torch.utils.data.Dataset):
    """Training sequences: the whole video with a slot after every pair, and two answer branches, s ~ U(t*, P - 1)
    (drawn per epoch as the offline prefixes, lora.sample_prefix) and the end of the video."""

    def __init__(self, records: Sequence[Record], *, builder: InputBuilder, split: ClevrerLoRASplitConfig,
                 tokens: ControlTokens, spec: RopeSpec, objective: OnlineObjective, seed: int):
        if split.vision_cache is None:
            raise ValueError("Online SFT reads the video from a vision cache (lora/vision_cache.py)")
        self.records, self.builder, self.split = list(records), builder, split
        self.tokens, self.spec, self.objective, self.seed, self.epoch = tokens, spec, objective, seed, 0
        self._cache = None

    @property
    def cache(self) -> VisionCache:
        if self._cache is None:  # opened in every DataLoader worker
            self._cache = VisionCache(self.split.vision_cache)
        return self._cache

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def prefix_patches(self, epoch: int) -> list[int]:
        """Pairs of every sequence (the whole video), for _LengthGroupedBatches."""
        temporal = self.builder.temporal_patch
        return [-(-record.num_frames // self.split.frame_stride) // temporal for record in self.records]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        entry = self.cache.load(record.video)
        question = _question(record, entry, self.builder, self.split)
        s = sample_prefix(question.target, question.patches, seed=self.seed, epoch=self.epoch,
                          example_key=record.id).last_patch
        # Always two branches, also when s is the last pair: every question weighs the same in the answer loss.
        return build_example(question.template, entry, spec=self.spec, tokens=self.tokens,
                             slot_targets=self.objective.targets(question.patches, question.target),
                             branch_pairs=[s, question.patches - 1], answer=_answer_id(question))


def build_online_loader(records: Sequence[Record], *, builder: InputBuilder, split: ClevrerLoRASplitConfig,
                        tokens: ControlTokens, spec: RopeSpec, objective: OnlineObjective, seed: int,
                        batch_size: int) -> DataLoader:
    dataset = OnlineSFTDataset(records, builder=builder, split=split, tokens=tokens, spec=spec, objective=objective,
                               seed=seed)
    return DataLoader(dataset, batch_sampler=_LengthGroupedBatches(dataset, batch_size, seed),
                      num_workers=split.num_workers,
                      collate_fn=partial(collate_online, pad_id=builder.processor.tokenizer.pad_token_id))


# --------------------------------------------------------------------------------------------------------- evaluation

class _EvaluationExamples(torch.utils.data.Dataset):
    """Item i: the evaluation sequence of question i, with a branch after every pair."""

    def __init__(self, records, builder, split, tokens, spec, objective):
        self.records, self.builder, self.split = records, builder, split
        self.tokens, self.spec, self.objective = tokens, spec, objective
        self._cache = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        if self._cache is None:
            self._cache = VisionCache(self.split.vision_cache)
        record = self.records[index]
        entry = self._cache.load(record.video)
        question = _question(record, entry, self.builder, self.split)
        example = build_example(question.template, entry, spec=self.spec, tokens=self.tokens,
                                slot_targets=self.objective.targets(question.patches, question.target),
                                branch_pairs=range(question.patches), answer=_answer_id(question))
        # The answer pair of the validation, fixed per question (as the offline "answerable" point).
        s = sample_prefix(question.target, question.patches, seed=VALIDATION_PREFIX_SEED, epoch=0,
                          example_key=record.id).last_patch
        return record, question.target, question.letters, s, example


def _as_is(item):
    return item


@torch.inference_mode()
def evaluate_online(model, records: Sequence[Record], *, builder: InputBuilder, split: ClevrerLoRASplitConfig,
                    tokens: ControlTokens, spec: RopeSpec, objective: OnlineObjective, device,
                    taus: Sequence[float] = (0.5, 0.7, 0.9), desc: str = "Online evaluation") \
        -> tuple[dict[str, Any], dict[str, Any]]:
    """One pass per question with a branch after every pair. Returns the summary and the predictions in the format of
    online_vllm/eval/export.py (probabilities = p_k, answer_correct = the branch answer of pair k is right), which
    eval/thresholds.py and eval/readiness_plot.py read; tau = 0.5 is the native rule <speak> more likely than <listen>.

    Loss: the training objective with the branches s (VALIDATION_PREFIX_SEED) and the last pair."""
    training = model.training
    model.eval()
    loader = DataLoader(_EvaluationExamples(list(records), builder, split, tokens, spec, objective), batch_size=None,
                        num_workers=split.num_workers, collate_fn=_as_is)
    rows, slot_losses, answer_losses, mass = [], [], [], []
    at = {"start": [], "target": [], "answerable": [], "end": []}
    try:
        for record, target, letters, s, example in tqdm(loader, total=len(records), desc=desc, unit="q",
                                                        dynamic_ncols=True):
            batch = _move_inputs(collate_online([example], pad_id=builder.processor.tokenizer.pad_token_id), device)
            parts = online_losses(model, batch, tokens)
            slots, answers = parts["slots"], parts["answers"]
            slot_losses.append(float(parts["slot"]))
            last = len(example["branch_pair"]) - 1
            answer_losses.append(float(-(answers[s, batch["answer_target"][s]]
                                         + answers[last, batch["answer_target"][last]]) / 2))
            control = slots[:, [tokens.speak, tokens.listen]]
            mass.append(float(control.exp().sum(-1).mean()))
            p_speak = control.softmax(-1)[:, 0]
            letter_scores = answers[:, letters]                        # [P, K], one branch per pair
            correct = LETTERS.index(answer_letter(record.choices, record.answers[0].answer))
            right = (letter_scores.argmax(-1) == correct).cpu()
            for name, pair in (("start", 0), ("target", target), ("answerable", s), ("end", last)):
                at[name].append(bool(right[pair]))
            rows.append({"record_id": record.id, "probabilities": p_speak.cpu().tolist(), "target_pair": target,
                         "visible_pair": target, "fps": record.fps, "answer_correct": right.tolist(),
                         "answer_probabilities": letter_scores.softmax(-1).cpu().tolist(),
                         "family": record.meta.get("family")})
    finally:
        model.train(training)
    payload = {"metadata": {"model": "online_sft", "tags": objective.tags, "temporal_patch": builder.temporal_patch,
                            "frame_stride": split.frame_stride}, "predictions": rows}
    slot_loss, answer_loss = float(np.mean(slot_losses)), float(np.mean(answer_losses))
    summary = {"count": len(rows), "loss": slot_loss + objective.answer_weight * answer_loss, "slot_loss": slot_loss,
               "answer_loss": answer_loss, "control_mass": float(np.mean(mass)),
               **{f"{name}_accuracy": float(np.mean(values)) for name, values in at.items()},
               "online": sweep(payload, list(taus))["results"]}
    return summary, payload


def _print_summary(epoch: int, summary: Mapping[str, Any], train_loss: float | None) -> None:
    loss = "-" if train_loss is None else f"{train_loss:.4f}"
    print(f"Epoch {epoch} | train_loss={loss} | val_loss={summary['loss']:.4f} (slots {summary['slot_loss']:.4f}, "
          f"answer {summary['answer_loss']:.4f}) | P(<speak>)+P(<listen>)={summary['control_mass']:.3f}")
    print("  answer accuracy: " + " | ".join(f"{name}={100 * summary[f'{name}_accuracy']:.1f}%"
                                             for name in ("start", "target", "answerable", "end")))
    for tau, values in summary["online"].items():
        print(f"  tau={tau}: on_target_pm2={100 * values['on_target_pm2']:.1f}% "
              f"premature={100 * values['premature_rate']:.1f}% missed={100 * values['missed_rate']:.1f}% "
              f"answer={100 * values['answer_accuracy']:.1f}% timely_correct_pm2={100 * values['timely_correct_pm2']:.1f}%")


# ----------------------------------------------------------------------------------------------------------- training

def train_online_sft(model, train_loader: DataLoader, config: LoRASFTConfig, *, output_dir: str | Path, device,
                     tokens: ControlTokens, objective: OnlineObjective, evaluate, run_info: Mapping[str, Any],
                     split_info: Mapping[str, Any] | None = None) -> Path:
    """Epochs over ``train_loader``; ``evaluate()`` -> (summary, predictions) before training and after every epoch.
    ``best`` is the adapter of the lowest validation loss, ``last`` that of the last epoch."""
    config.validate()
    objective.validate()
    configure_gradient_checkpointing(model, enabled=config.gradient_checkpointing)
    assert_online_isolation(model)
    output_dir = _prepare_output_dir(output_dir, overwrite=config.overwrite)
    (output_dir / "sft_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    model.config.save(output_dir / "model.yaml")
    (output_dir / "online.json").write_text(json.dumps(
        {**run_info, "tokens": asdict(tokens), "token_strings": {"listen": LISTEN, "speak": SPEAK},
         "init_words": INIT_WORDS, "objective": asdict(objective)}, indent=2) + "\n", encoding="utf-8")
    if split_info is not None:
        (output_dir / "split.json").write_text(json.dumps(split_info, indent=2) + "\n", encoding="utf-8")
    parameters = [p for p in model.parameters() if p.requires_grad]
    arguments: dict[str, Any] = {"lr": config.learning_rate, "betas": config.betas, "weight_decay": config.weight_decay}
    cuda = torch.cuda.is_available() and torch.device(device).type == "cuda"
    if cuda:
        arguments["fused"] = True
    optimizer = torch.optim.AdamW(parameters, **arguments)
    loss_fn = partial(online_loss, tokens=tokens, answer_weight=objective.answer_weight)
    best_epoch, best_loss, global_step = None, None, 0

    def evaluate_and_save(epoch: int, train_loss: float | None, steps: int, seconds: float, peak: float | None):
        nonlocal best_epoch, best_loss
        summary, predictions = evaluate()
        is_best = best_loss is None or summary[SELECTION_METRIC] < best_loss
        if is_best:
            best_epoch, best_loss = epoch, summary[SELECTION_METRIC]
            save_online_adapter(model, output_dir / "best", metadata={"epoch": epoch, "selection_metric": SELECTION_METRIC,
                                                                      "validation_loss": best_loss})
        record = {"epoch": epoch, "train_loss": train_loss, "optimizer_steps": steps, "epoch_duration_seconds": seconds,
                  "peak_gpu_memory_gb": peak, "best_epoch": best_epoch, "is_best_checkpoint": is_best,
                  "validation": summary}
        with (output_dir / "validation_metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")
        (output_dir / f"predictions_validation_epoch_{epoch}.json").write_text(
            json.dumps(predictions, allow_nan=False) + "\n", encoding="utf-8")
        _print_summary(epoch, summary, train_loss)

    evaluate_and_save(0, None, 0, 0.0, None)
    for epoch in range(1, config.epochs + 1):
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
        save_online_adapter(model, output_dir / "last", metadata={"epoch": epoch, "train_loss": mean_loss})
    return output_dir / "best"


# ---------------------------------------------------------------------------------------------------------- inference

class OnlineSession:
    """Streaming inference of one question with a KV cache, as in training: the prompt once, then per pair its visual
    tokens (one pass over them), p_k at the last one, and <listen> (one token) or <speak> + tail + answer.

        session = OnlineSession(model, builder, tokens, prompt, grid_hw=(h, w), seconds_per_pair=2 / 25, n_options=3)
        for pair in pairs:                 # visual tokens of a pair: [per_pair, D] (cache slice or encode_pair)
            if session.observe(pair) > tau:
                break
            session.listen()
        letters = session.speak()          # log P of the option letters, normalized over the options
    """

    def __init__(self, model, builder: InputBuilder, tokens: ControlTokens, prompt: str, *, grid_hw: Sequence[int],
                 seconds_per_pair: float, n_options: int, device=None):
        from transformers import DynamicCache
        self.model, self.tokens = model, tokens
        self.device = device or model.qwen.get_input_embeddings().weight.device
        self.spec = RopeSpec.of(model)
        self.grid_hw, self.seconds = tuple(int(x) for x in grid_hw), float(seconds_per_pair)
        self.template = template(builder, prompt, torch.tensor([1, *self.grid_hw]), torch.tensor([self.seconds]))
        self.letters = torch.tensor(_letter_ids(builder.processor.tokenizer, n_options), device=self.device)
        self.cache, self.pair, self.spoken = DynamicCache(), 0, False
        n = len(self.template.prefix)
        self.start = n
        self.reach = n - 1
        self._last = self._forward(ids=self.template.prefix, positions=torch.arange(n).expand(3, -1))

    @torch.inference_mode()
    def _forward(self, *, positions: torch.Tensor, ids: Sequence[int] | None = None,
                 embeds: torch.Tensor | None = None) -> torch.Tensor:
        qwen = self.model.qwen.model
        if embeds is None:
            embeds = qwen.get_input_embeddings()(torch.tensor([list(ids)], device=self.device))
        qwen.rope_deltas = None
        with attention_kernels():
            out = qwen(inputs_embeds=embeds.to(self.device), position_ids=positions[:, None].to(self.device),
                       past_key_values=self.cache, use_cache=True, return_dict=True)
        self.cache = out.past_key_values
        return out.last_hidden_state[0, -1]

    @torch.inference_mode()
    def observe(self, pair_embeds: torch.Tensor) -> float:
        """Append the visual tokens of the next pair; returns p = P(<speak>) / (P(<speak>) + P(<listen>))."""
        if self.spoken:
            raise RuntimeError("The session has answered")
        k, per = self.pair, self.template.per_pair
        if pair_embeds.shape[0] != per:
            raise ValueError(f"A pair has {per} visual tokens, got {pair_embeds.shape[0]}")
        positions = vision_positions(self.start, (k + 1, *self.grid_hw), self.spec.merge, self.seconds,
                                     self.spec.tokens_per_second)[:, k * per:]
        dtype = self.model.qwen.get_input_embeddings().weight.dtype
        self._last = self._forward(embeds=pair_embeds[None].to(self.device, dtype), positions=positions)
        self.reach = max(self.reach, int(positions.max()))
        self.pair += 1
        logits = self.model.qwen.lm_head(self._last).float()
        return float(logits[[self.tokens.speak, self.tokens.listen]].softmax(-1)[0])

    def listen(self) -> None:
        self._forward(ids=[self.tokens.listen], positions=torch.full((3, 1), self.reach + 1))

    @torch.inference_mode()
    def speak(self) -> torch.Tensor:
        """Append <speak> and the template tail; log P of the option letters (normalized over the options)."""
        branch = [self.tokens.speak] + self.template.tail
        last = self._forward(ids=branch, positions=torch.arange(self.reach + 1, self.reach + 1 + len(branch))
                             .expand(3, -1))
        self.spoken = True
        return self.model.qwen.lm_head(last).float().log_softmax(-1)[self.letters].log_softmax(-1)


def run_online(session: OnlineSession, pairs: Iterable[torch.Tensor], tau: float = 0.5) -> dict[str, Any]:
    """The inference loop: answer at the first pair with p > tau; at the end of the video without it, answer anyway
    (a miss). <listen> is written only when another pair follows: the forced answer comes right after the last pair,
    as the end-of-video branch of training."""
    probabilities = []
    pairs = iter(pairs)
    pair = next(pairs, None)
    while pair is not None:
        probabilities.append(session.observe(pair))
        if probabilities[-1] > tau:
            return {"fired_pair": session.pair - 1, "missed": False, "probabilities": probabilities,
                    "letter_logprobs": session.speak().cpu().tolist()}
        pair = next(pairs, None)
        if pair is not None:
            session.listen()
    return {"fired_pair": None, "missed": True, "probabilities": probabilities,
            "letter_logprobs": session.speak().cpu().tolist()}


@torch.inference_mode()
def encode_pair(model, builder: InputBuilder, frames: np.ndarray, fps: float) -> torch.Tensor:
    """Visual tokens [per_pair, D] of one pair of frames (uint8 [2, H, W, 3] at their final size), from the vision
    encoder. Equal to the pair's slice of a vision cache entry: the encoder does not mix temporal patches."""
    if len(frames) != builder.temporal_patch:
        raise ValueError(f"A pair is {builder.temporal_patch} frames")
    x = builder._encode("", frames, fps)
    device = model.qwen.get_input_embeddings().weight.device
    with attention_kernels():
        return model.qwen.model.get_video_features(x["pixel_values_videos"].to(device), x["video_grid_thw"].to(device),
                                                   return_dict=True).pooler_output[0]
