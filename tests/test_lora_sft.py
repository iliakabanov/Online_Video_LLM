"""Unit tests for schema-free LoRA SFT infrastructure."""
import collections
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from lora.lora import (
    ClevrerLoRASplitConfig,
    IGNORE,
    SELECTION_METRIC,
    _accuracy_deltas,
    _candidate_score,
    _choice_scores,
    _prepare_output_dir,
    _is_new_best,
    _train_epoch,
    _records_for_first_videos,
    assert_lora_isolation,
    collate_sft,
    configure_gradient_checkpointing,
    sample_prefix,
)
from online_vllm.data.schema import Interval, Record, save_jsonl


def _example(length=4):
    ids = torch.cat((torch.arange(1, length)[None], torch.tensor([[7]])), dim=1)
    labels = torch.full_like(ids, IGNORE)
    labels[0, -1] = 7
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": torch.zeros_like(ids), "labels": labels,
            "answer_ids": torch.tensor([7]),
            "pixel_values_videos": torch.ones(2, 3), "video_grid_thw": torch.tensor([[1, 1, 1]]),
            "second_per_grid_ts": torch.tensor([0.08])}


def test_multiprefix_is_deterministic_and_never_precedes_target():
    values = [sample_prefix(4, 20, seed=9, epoch=3, example_key=str(i)) for i in range(500)]
    again = [sample_prefix(4, 20, seed=9, epoch=3, example_key=str(i)) for i in range(500)]
    assert values == again
    assert all(4 <= value.last_patch < 20 for value in values)
    counts = collections.Counter(value.mode for value in values)
    # Statistical guard against accidentally dropping a branch; exact ratios are
    # guaranteed only over an infinite number of independently keyed examples.
    assert set(counts) == {"target", "near_target", "after_target", "full"}


def test_multiprefix_distribution_remains_40_30_20_10():
    counts = collections.Counter(
        sample_prefix(4, 20, seed=123, epoch=2, example_key=str(i)).mode for i in range(20_000))
    expected = {"target": .40, "near_target": .30, "after_target": .20, "full": .10}
    for mode, probability in expected.items():
        assert abs(counts[mode] / 20_000 - probability) < .015


class _EpochPrefixDataset(Dataset):
    def __init__(self):
        self.epoch = 0

    def __len__(self):
        return 32

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        return sample_prefix(4, 20, seed=41, epoch=self.epoch, example_key=str(index)).last_patch


def test_nonpersistent_dataloader_workers_observe_new_epoch():
    dataset = _EpochPrefixDataset()
    loader = DataLoader(dataset, batch_size=8, num_workers=1, persistent_workers=False)
    dataset.set_epoch(3)
    first = torch.cat(list(loader)).tolist()
    dataset.set_epoch(4)
    second = torch.cat(list(loader)).tolist()
    assert first != second
    dataset.set_epoch(3)
    assert first == torch.cat(list(loader)).tolist()


def test_sft_collator_masks_everything_except_answer():
    batch = collate_sft([_example(4), _example(2)], pad_id=99)
    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].tolist() == [[IGNORE, IGNORE, IGNORE, 7], [IGNORE, 7, IGNORE, IGNORE]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]


@pytest.mark.parametrize("mutate", ["prompt_label", "wrong_answer"])
def test_sft_collator_rejects_non_answer_supervision(mutate):
    example = _example()
    if mutate == "prompt_label":
        example["labels"][0, 0] = example["input_ids"][0, 0]
    else:
        example["labels"][0, -1] = 6
    with pytest.raises(ValueError):
        collate_sft([example], pad_id=99)


def test_lora_isolation_rejects_trainable_state_head():
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.register_parameter(
                "weight", torch.nn.Parameter(torch.ones(1)))
            self.state_head = torch.nn.Linear(1, 1)

        def named_parameters(self, prefix="", recurse=True):
            yield "backbone.base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight", self.backbone.weight
            yield "state_head.weight", self.state_head.weight

    model = Module()
    try:
        assert_lora_isolation(model)
    except ValueError as error:
        assert "Non-LoRA" in str(error)
    else:
        raise AssertionError("trainable StateHead must be rejected")


def test_lora_isolation_accepts_language_attention_adapter_only():
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.ones(1))

        def named_parameters(self, prefix="", recurse=True):
            yield "backbone.base_model.model.model.language_model.layers.0.self_attn.v_proj.lora_B.default.weight", self.adapter

    assert_lora_isolation(Module())


class _LossResult:
    def __init__(self, loss):
        self.loss = loss


class _TinySFTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs, *, mode):
        assert mode == "language"
        return _LossResult((self.weight - inputs["input_ids"][0, 0].float()).square())


def test_final_gradient_accumulation_window_uses_actual_size():
    model = _TinySFTModel()
    batches = [collate_sft([_example(2)], pad_id=0) for _ in range(3)]
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    _, steps, records = _train_epoch(model, batches, optimizer=optimizer, scheduler=scheduler,
                                     gradient_accumulation=2, grad_clip=100.0,
                                     parameters=list(model.parameters()), device="cpu", epoch=7,
                                     global_step_start=10)
    # First two examples average to one SGD step from 0 to 2.  The final,
    # one-example window must use its own denominator (1), returning to 0.
    assert steps == 2
    torch.testing.assert_close(model.weight.detach(), torch.tensor(0.0))
    assert [record["effective_accumulation_steps"] for record in records] == [2, 1]
    assert [record["global_step"] for record in records] == [11, 12]
    assert all(record["epoch"] == 7 and record["gradient_norm"] >= 0 for record in records)


class _LogitResult:
    def __init__(self, logits):
        self.logits = logits


class _ChoiceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, inputs, *, mode):
        assert mode == "language"
        self.calls += 1
        vocabulary = 11
        logits = torch.arange(vocabulary, dtype=torch.float32).view(1, 1, vocabulary)
        return _LogitResult(logits.expand(1, inputs["input_ids"].shape[1], vocabulary).clone())


def test_one_token_choice_fast_path_matches_full_likelihood_fallback():
    model = _ChoiceModel()
    prompt = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
              "mm_token_type_ids": torch.zeros(1, 2, dtype=torch.long)}
    choices = [torch.tensor([3]), torch.tensor([5]), torch.tensor([7]), torch.tensor([9])]
    fast = _choice_scores(model, prompt, choices, device="cpu")
    assert model.calls == 1
    fallback = [_candidate_score(model, prompt, choice, "cpu") for choice in choices]
    assert fast == fallback
    assert max(range(len(fast)), key=fast.__getitem__) == max(range(len(fallback)), key=fallback.__getitem__)


def test_nonempty_output_dir_requires_explicit_overwrite(tmp_path):
    output = _prepare_output_dir(tmp_path / "experiment", overwrite=False)
    (output / "old_result.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        _prepare_output_dir(output, overwrite=False)
    replacement = _prepare_output_dir(output, overwrite=True)
    assert replacement == output and not list(output.iterdir())


def test_gradient_checkpointing_is_activated_and_disables_cache():
    class Backbone:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True)
            self.is_gradient_checkpointing = False
            self.enabled_inputs = False

        def gradient_checkpointing_enable(self):
            self.is_gradient_checkpointing = True

        def enable_input_require_grads(self):
            self.enabled_inputs = True

    model = SimpleNamespace(backbone=Backbone())
    configure_gradient_checkpointing(model, enabled=True)
    assert model.backbone.is_gradient_checkpointing
    assert model.backbone.enabled_inputs
    assert not model.backbone.config.use_cache


def test_clevrer_lora_partition_is_first_scene_indexed_videos(tmp_path):
    records = [
        Record("late-a", "x", "train", "late.mp4", 25, 128, "q", ["a"], 0, [Interval(25, 127, "a")],
               {"scene_index": 5}),
        Record("early", "x", "train", "early.mp4", 25, 128, "q", ["a"], 0, [Interval(25, 127, "a")],
               {"scene_index": 1}),
        Record("late-b", "x", "train", "late.mp4", 25, 128, "q", ["a"], 0, [Interval(25, 127, "a")],
               {"scene_index": 5}),
        Record("middle", "x", "train", "middle.mp4", 25, 128, "q", ["a"], 0, [Interval(25, 127, "a")],
               {"scene_index": 3}),
    ]
    path = tmp_path / "split.jsonl"
    save_jsonl(records, path)
    selected = _records_for_first_videos(path, 2)
    assert {record.video for record in selected} == {"early.mp4", "middle.mp4"}


def test_lora_checkpoint_selection_metric_is_end_accuracy():
    assert SELECTION_METRIC == "end_micro_accuracy"
    assert ClevrerLoRASplitConfig().train_videos == 5000
    assert ClevrerLoRASplitConfig().validation_videos == 2000


def test_accuracy_deltas_are_percentage_points_and_resolution_local():
    baseline = {
        "252x168": {"start_micro_accuracy": .2, "target_micro_accuracy": .3, "end_micro_accuracy": .4},
        "504x336": {"start_micro_accuracy": .5, "target_micro_accuracy": .6, "end_micro_accuracy": .7},
    }
    current = {
        "252x168": {"start_micro_accuracy": .25, "target_micro_accuracy": .3, "end_micro_accuracy": .45},
        "504x336": {"start_micro_accuracy": .4, "target_micro_accuracy": .65, "end_micro_accuracy": .7},
    }
    delta = _accuracy_deltas(current, baseline)
    assert delta["252x168"]["start_accuracy_delta_pp"] == pytest.approx(5.0)
    assert delta["252x168"]["target_accuracy_delta_pp"] == pytest.approx(0.0)
    assert delta["504x336"]["start_accuracy_delta_pp"] == pytest.approx(-10.0)
    assert delta["504x336"]["end_accuracy_delta_pp"] == pytest.approx(0.0)


def test_checkpoint_selection_keeps_epoch_zero_and_earlier_ties():
    score = .61
    assert _is_new_best(score, None)  # epoch 0 may be the best model
    assert not _is_new_best(score, score)  # equal later epoch cannot replace it
    assert not _is_new_best(.60, score)
    assert _is_new_best(.62, score)
