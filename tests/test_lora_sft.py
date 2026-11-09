"""Unit tests for schema-free LoRA SFT infrastructure."""
import collections
import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from lora.lora import (
    ClevrerLoRASplitConfig,
    IGNORE,
    SELECTION_METRIC,
    _accuracy_deltas,
    _build_sft_example,
    _prepare_output_dir,
    _is_new_best,
    _train_epoch,
    _records_for_first_videos,
    assert_lora_isolation,
    collate_sft,
    configure_gradient_checkpointing,
    sample_prefix,
)
from online_vllm.data.inputs import InputBuilder
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


def test_sft_example_supervises_only_the_answer_letter():
    vocab = {"A": 11, "B": 12, "<|im_end|>": 99}

    class Tokenizer:
        eos_token = "<|im_end|>"

        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            ids = [vocab[text]] if text in vocab else [vocab["B"], vocab["<|im_end|>"]]
            return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors == "pt" else ids)

    prompt = torch.tensor([[1, 2, 3]])
    builder = SimpleNamespace(
        processor=SimpleNamespace(tokenizer=Tokenizer()),
        _encode=lambda text, frames, fps: {"input_ids": prompt.clone(), "attention_mask": torch.ones_like(prompt),
                                           "mm_token_type_ids": torch.zeros_like(prompt)})
    record = SimpleNamespace(question="Q?", choices=["red", "blue"], answers=[SimpleNamespace(answer="blue")])
    example = _build_sft_example(builder, record, frames=None, fps=25.0)
    assert example["input_ids"].tolist() == [[1, 2, 3, 12]]  # the letter B and no EOS after it
    assert example["labels"].tolist() == [[IGNORE, IGNORE, IGNORE, 12]]
    assert example["answer_ids"].tolist() == [12]


def test_multiprefix_is_deterministic_and_never_precedes_target():
    values = [sample_prefix(4, 20, seed=9, epoch=3, example_key=str(i)) for i in range(500)]
    again = [sample_prefix(4, 20, seed=9, epoch=3, example_key=str(i)) for i in range(500)]
    assert values == again
    assert all(4 <= value.last_patch < 20 for value in values)
    assert {value.mode for value in values} == {"after_target"}


def test_prefix_end_is_uniform_from_target_to_video_end():
    counts = collections.Counter(
        sample_prefix(4, 20, seed=123, epoch=2, example_key=str(i)).last_patch for i in range(16_000))
    # 16 equally likely ends, patches 4..19 inclusive: 1000 each in expectation.
    assert set(counts) == set(range(4, 20))
    assert all(abs(count / 16_000 - 1 / 16) < .01 for count in counts.values())


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
    _, steps, records = _train_epoch(model, batches, optimizer=optimizer,
                                     gradient_accumulation=2, grad_clip=100.0,
                                     parameters=list(model.parameters()), device="cpu", epoch=7,
                                     global_step_start=10,
                                     loss_fn=lambda m, batch: m(batch, mode="language").loss)
    # First two examples average to one SGD step from 0 to 2.  The final,
    # one-example window must use its own denominator (1), returning to 0.
    assert steps == 2
    torch.testing.assert_close(model.weight.detach(), torch.tensor(0.0))
    assert [record["effective_accumulation_steps"] for record in records] == [2, 1]
    assert [record["global_step"] for record in records] == [11, 12]
    assert all(record["epoch"] == 7 and record["gradient_norm"] >= 0 for record in records)


def _tiny_sft_examples(builder):
    import numpy as np
    rng = np.random.default_rng(0)
    examples = []
    for n_frames, answer in ((4, "green"), (6, "red")):
        record = SimpleNamespace(question="question", choices=["green", "red"], answers=[SimpleNamespace(answer=answer)])
        example = _build_sft_example(builder, record, rng.integers(0, 256, (n_frames, 8, 4, 3), dtype=np.uint8), fps=25)
        ids = example["input_ids"].clone()
        for old, new in ((2, 60), (3, 61), (4, 62), (5, 63)):  # toy special tokens -> tiny Qwen vocabulary
            example["input_ids"][ids == old] = new
        examples.append(example)
    assert examples[0]["input_ids"].shape[1] != examples[1]["input_ids"].shape[1]
    return examples


def _tiny_model(tiny_base, lora):
    from online_vllm.models import BackboneConfig, LoRAConfig, ModelConfig, StateHeadConfig, build_model
    model = build_model(ModelConfig(backbone=BackboneConfig(model_id=tiny_base, dtype="float32"),
                                    lora=LoRAConfig(enabled=lora, r=4, alpha=8, dropout=0),
                                    head=StateHeadConfig(hidden_dims=[16], dropout=0)), local_files_only=True)
    model.state_head.requires_grad_(False)
    return model


def test_right_padded_video_batch_loss_equals_mean_of_single_losses(builder, tiny_base):
    """Batch of 2 videos of different length (right padding) == the two examples one by one, on the tiny Qwen."""
    from lora.lora import masked_sft_loss
    model, examples = _tiny_model(tiny_base, lora=False).eval(), _tiny_sft_examples(builder)
    with torch.no_grad():
        single = [masked_sft_loss(model, collate_sft([example], pad_id=1)) for example in examples]
        batch = masked_sft_loss(model, collate_sft(examples, pad_id=1))
    # One supervised token per example, so the batch loss is the mean of the two.
    torch.testing.assert_close(batch, torch.stack(single).mean(), rtol=1e-4, atol=1e-4)


def test_answer_position_loss_equals_qwen_causal_lm_loss_and_reaches_lora(builder, tiny_base):
    """lm_head only before the answer token == Qwen's own shifted loss over the full logits; LoRA gets gradients."""
    from lora.lora import ANSWER_IDS_KEY, masked_sft_loss
    model, examples = _tiny_model(tiny_base, lora=True), _tiny_sft_examples(builder)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_B" in name:
                parameter.normal_(std=0.1)  # B starts at zero; make the adapter change the output
    batch = collate_sft(examples, pad_id=1)
    reference = model({k: v for k, v in batch.items() if k != ANSWER_IDS_KEY}, mode="language").loss
    loss = masked_sft_loss(model, batch)
    torch.testing.assert_close(loss, reference, rtol=1e-4, atol=1e-4)
    loss.backward()
    grads = [p.grad for name, p in model.named_parameters() if ".lora_A" in name]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)


TOY_TO_TINY = ((2, 60), (3, 61), (4, 62), (5, 63))  # toy tokenizer special IDs -> tiny Qwen vocabulary


class _RemapBuilder(InputBuilder):
    @staticmethod
    def _remap(x):
        ids = x["input_ids"].clone()
        for old, new in TOY_TO_TINY:
            x["input_ids"][ids == old] = new
        return x

    def _encode(self, prompt, frames, fps):
        return self._remap(super()._encode(prompt, frames, fps))

    def _encode_tokens(self, prompt, video_grid_thw, second_per_grid_ts):
        return self._remap(super()._encode_tokens(prompt, video_grid_thw, second_per_grid_ts))


def _toy_videos(tmp_path, n_videos=2, per_video=2):
    """Videos of 8 frames 28 x 28 (4 temporal patches) and their questions, in a group-1 style JSONL."""
    from test_train import write_video
    data = tmp_path / "data"
    (data / "videos").mkdir(parents=True)
    for v in range(n_videos):
        write_video(data / "videos" / f"v{v}.mp4", n=8, size=28)
    records = [Record(f"v{v}-q{q}", "toy", "train", f"videos/v{v}.mp4", 25, 8, "question", ["green", "red"], 0,
                      [Interval(2 + 2 * q, 7, "green")], {"scene_index": v, "family": "toy"})
               for v in range(n_videos) for q in range(per_video)]
    save_jsonl(records, data / "toy.jsonl")
    return data, records


def _toy_cache(model, tiny, data, records, tmp_path):
    from lora.vision_cache import VisionCache, build_vision_cache
    cache = VisionCache.for_model(tmp_path / "cache", model, frame_size=(28, 28), frame_stride=1)
    added = build_vision_cache(model, tiny, records, cache, data_root=data, frame_size=(28, 28), frame_stride=1,
                               device="cpu", num_workers=0)
    assert added == len({r.video for r in records})
    assert build_vision_cache(model, tiny, records, cache, data_root=data, frame_size=(28, 28), frame_stride=1,
                              device="cpu", num_workers=0) == 0  # nothing left to encode
    return cache


def test_cached_vision_training_equals_encoding_the_pixels(builder, tiny_base, tmp_path):
    """The cached encoder output, cut to the prefix, gives the same inputs and the same loss as the pixels."""
    from dataclasses import replace
    from lora.lora import ClevrerLoRASFTDataset, masked_sft_loss

    data, records = _toy_videos(tmp_path)
    tiny, model = _RemapBuilder(builder.processor), _tiny_model(tiny_base, lora=False).eval()
    cache = _toy_cache(model, tiny, data, records, tmp_path)
    split = ClevrerLoRASplitConfig(train="toy.jsonl", train_videos=2, train_frame_size=(28, 28), num_workers=0)
    plain = ClevrerLoRASFTDataset(data / "toy.jsonl", data_root=data, builder=tiny, split=split, seed=3)
    cached = ClevrerLoRASFTDataset(data / "toy.jsonl", data_root=data, builder=tiny,
                                   split=replace(split, vision_cache=str(cache.path)), seed=3)
    shorter = 0
    for epoch in (1, 2):
        plain.set_epoch(epoch)
        cached.set_epoch(epoch)
        for index in range(len(records)):
            a, b = plain[index], cached[index]
            assert "pixel_values_videos" in a and "video_embeds" in b
            for key in ("input_ids", "labels", "mm_token_type_ids", "video_grid_thw", "second_per_grid_ts"):
                torch.testing.assert_close(b[key], a[key])
            shorter += int(b["video_grid_thw"][0, 0]) < 4
            with torch.no_grad():
                torch.testing.assert_close(masked_sft_loss(model, collate_sft([b], pad_id=1)),
                                           masked_sft_loss(model, collate_sft([a], pad_id=1)), rtol=1e-4, atol=1e-4)
    assert shorter  # some prefixes end before the last patch: the cached tokens were cut, not taken whole


def test_cached_vision_validation_equals_encoding_the_pixels(builder, tiny_base, tmp_path):
    from dataclasses import replace
    from lora.lora import ClevrerLoRAValidationProvider, evaluate_choice_examples

    data, records = _toy_videos(tmp_path)
    tiny, model = _RemapBuilder(builder.processor), _tiny_model(tiny_base, lora=False).eval()
    cache = _toy_cache(model, tiny, data, records, tmp_path)
    split = ClevrerLoRASplitConfig(num_workers=0)
    results = [evaluate_choice_examples(model, list(ClevrerLoRAValidationProvider(
        records, data_root=data, builder=tiny, split=s).iter_choice_examples((28, 28))), resolution=(28, 28),
        device="cpu") for s in (split, replace(split, vision_cache=str(cache.path)))]
    for plain, cached in zip(*(r["predictions"] for r in results)):
        for point in plain["predictions"]:
            torch.testing.assert_close(torch.tensor(cached["predictions"][point]["scores"]),
                                       torch.tensor(plain["predictions"][point]["scores"]), rtol=1e-4, atol=1e-4)


def test_answer_traces_agree_with_validation_and_disabled_adapter_is_the_base(builder, tiny_base, tmp_path):
    import numpy as np
    from lora.lora import ClevrerLoRAValidationProvider, answer_traces, evaluate_choice_examples

    data, records = _toy_videos(tmp_path)
    tiny = _RemapBuilder(builder.processor)
    provider = ClevrerLoRAValidationProvider(records, data_root=data, builder=tiny,
                                             split=ClevrerLoRASplitConfig(num_workers=0))
    model = _tiny_model(tiny_base, lora=True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_B" in name:
                parameter.normal_(std=0.5)  # an adapter that changes the answers
    traces = answer_traces(model, provider, (28, 28), device="cpu")
    rows = evaluate_choice_examples(model, list(provider.iter_choice_examples((28, 28))), resolution=(28, 28),
                                    device="cpu")["predictions"]
    for trace, row in zip(traces, rows):
        assert trace.probabilities.shape == (4, 2) and trace.target_pair == row["predictions"]["target"]["last_pair"]
        for point in ("start", "target", "end"):
            prediction = row["predictions"][point]
            expected = torch.tensor(prediction["scores"]).softmax(-1).numpy()
            np.testing.assert_allclose(trace.probabilities[prediction["last_pair"]], expected, rtol=1e-4, atol=1e-5)
    with model.backbone.disable_adapter():
        disabled = answer_traces(model, provider, (28, 28), device="cpu")
    base = answer_traces(_tiny_model(tiny_base, lora=False), provider, (28, 28), device="cpu")
    for a, b, adapted in zip(disabled, base, traces):
        np.testing.assert_allclose(a.probabilities, b.probabilities, rtol=1e-4, atol=1e-6)
    assert any(not np.allclose(a.probabilities, t.probabilities, atol=1e-4) for a, t in zip(disabled, traces))


def test_answer_curve_plot(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    import numpy as np
    from online_vllm.eval.answers import AnswerTrace, answer_curves

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("lora_answer_curves", scripts / "lora_answer_curves.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    rng = np.random.default_rng(0)
    results = {}
    for name in ("none", "best"):
        traces = [AnswerTrace(rng.dirichlet(np.ones(3), 64), int(rng.integers(3)), t, t) for t in rng.integers(5, 60, 20)]
        results[name] = {**answer_curves(traces, range(-30, 41)), "by_pair": module.pair_curve(traces),
                         "mean_target_pair": float(np.mean([t.target_pair for t in traces]))}
    assert [row["pair"] for row in results["best"]["by_pair"]] == list(range(64))
    module.plot(results, tmp_path / "curves.png", "test")
    assert (tmp_path / "curves.png").stat().st_size > 0


def test_length_grouped_batches_cover_every_example_once_with_similar_lengths():
    from lora.lora import _LengthGroupedBatches

    class Prefixes:
        def __init__(self):
            self.n = 103

        def __len__(self):
            return self.n

        def prefix_patches(self, epoch):
            rng = np.random.default_rng(epoch)
            return rng.integers(10, 65, self.n).tolist()

    import numpy as np
    dataset = Prefixes()
    sampler = _LengthGroupedBatches(dataset, batch_size=2, seed=0, window=8)
    sampler.set_epoch(1)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(dataset.n))
    assert all(len(batch) <= 2 for batch in batches)
    lengths = dataset.prefix_patches(1)
    grouped = np.mean([abs(lengths[a] - lengths[b]) for a, b in (b for b in batches if len(b) == 2)])
    random_pairs = np.mean([abs(lengths[i] - lengths[i + 1]) for i in range(0, dataset.n - 1, 2)])
    assert grouped < random_pairs / 3  # neighbours in length share a batch
    assert list(sampler) == batches  # the same epoch gives the same batches
    sampler.set_epoch(2)
    assert list(sampler) != batches


def test_validation_decodes_each_video_once_in_workers(builder, tiny_base, tmp_path):
    from lora.lora import TIME_POINTS, ClevrerLoRAValidationProvider, _decode, evaluate_choice_examples
    from online_vllm.data.inputs import format_prompt
    from test_train import write_video

    data = tmp_path / "data"
    (data / "videos").mkdir(parents=True)
    for v in range(2):
        write_video(data / "videos" / f"v{v}.mp4", n=8, size=28)
    records = [Record(f"v{v}-q{q}", "toy", "validation", f"videos/v{v}.mp4", 25, 8, "question", ["green", "red"], 0,
                      [Interval(2 + 2 * q, 7, "green")], {"scene_index": v, "family": "toy"})
               for v in range(2) for q in range(2)]
    tiny = _RemapBuilder(builder.processor)
    provider = ClevrerLoRAValidationProvider(records, data_root=data, builder=tiny,
                                             split=ClevrerLoRASplitConfig(num_workers=1))
    examples = list(provider.iter_choice_examples((28, 28)))
    assert [e.record_id for e in examples] == [r.id for r in records]
    assert examples[0].inputs["pixel_values_videos"] is examples[1].inputs["pixel_values_videos"]
    assert examples[1].inputs["pixel_values_videos"] is not examples[2].inputs["pixel_values_videos"]
    for example, record in zip(examples, records):  # the same as preparing every question on its own
        frames, _ = _decode(record, data_root=data, builder=tiny, frame_stride=1, frame_size=(28, 28))
        reference = tiny._encode(format_prompt(record.question, record.choices), frames, 25)
        for key in ("input_ids", "pixel_values_videos", "video_grid_thw"):
            torch.testing.assert_close(example.inputs[key], reference[key])
    assert examples[0].pair_by_time["target"] < examples[1].pair_by_time["target"]
    metrics = evaluate_choice_examples(_tiny_model(tiny_base, lora=False), examples, resolution=(28, 28), device="cpu")
    assert metrics["count"] == 4 and all(0 <= metrics[f"{point}_micro_accuracy"] <= 1 for point in TIME_POINTS)


def test_cached_probes_equal_separate_forwards_on_truncated_videos(builder, tiny_base):
    """One pass over the whole video plus probe tails == one full forward per truncated video, at every pair."""
    import numpy as np
    from lora.lora import probe_letter_logprobs
    from online_vllm.data.inputs import format_prompt
    from online_vllm.train.features import layout

    model = _tiny_model(tiny_base, lora=False).eval()
    # Random small weights give near-uniform attention; sharpen it so wrong masks or positions would show.
    with torch.no_grad():
        for layer in model.qwen.model.language_model.layers:
            layer.self_attn.q_proj.weight.mul_(6)
            layer.self_attn.k_proj.weight.mul_(6)
    frames = np.random.default_rng(1).integers(0, 256, (8, 8, 4, 3), dtype=np.uint8)  # 4 pairs
    prompt = format_prompt("question", ["green", "red", "question"])
    letters = torch.tensor([10, 11, 12])  # A, B, C in the toy vocabulary

    def encode(n_frames):
        x = dict(builder._encode(prompt, frames[:n_frames], 25))
        ids = x["input_ids"].clone()
        for old, new in ((2, 60), (3, 61), (4, 62), (5, 63)):  # toy special tokens -> tiny Qwen vocabulary
            x["input_ids"][ids == old] = new
        return x

    full = encode(8)
    lay = layout(full, prompt, builder.processor.tokenizer)
    assert lay.num_pairs == 4
    pairs = [0, 2, 3]
    cached = probe_letter_logprobs(model, full, lay, pairs, letters, device="cpu")
    with torch.no_grad():
        for row, pair in enumerate(pairs):
            logits = model(encode(2 * (pair + 1)), mode="language").logits[0, -1].float()
            torch.testing.assert_close(cached[row], logits.log_softmax(-1)[letters], rtol=1e-4, atol=1e-4)


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


def _scene_records(n_videos, per_video=3):
    return [Record(f"v{v}-q{q}", "x", "validation", f"v{v}.mp4", 25, 128, "q", ["a"], 0, [Interval(25, 127, "a")],
                   {"scene_index": v}) for v in range(n_videos) for q in range(per_video)]


def test_validation_videos_are_halved_by_video_and_deterministically():
    from lora.lora import split_validation_videos
    records = _scene_records(9)
    validation, test, info = split_validation_videos(records, seed=0)
    assert info["validation_videos"] == sorted(info["validation_videos"])
    assert len(info["validation_videos"]) == 4 and len(info["test_videos"]) == 5
    assert not set(info["validation_videos"]) & set(info["test_videos"])
    assert {r.video for r in validation} == set(info["validation_videos"])
    assert {r.video for r in test} == set(info["test_videos"])
    assert len(validation) + len(test) == len(records) and info["test_questions"] == len(test)
    assert split_validation_videos(list(reversed(records)), seed=0)[2] == info  # order of the file does not matter
    assert split_validation_videos(records, seed=1)[2]["test_videos"] != info["test_videos"]


def test_test_half_is_rebuilt_from_split_json(tmp_path):
    from dataclasses import asdict
    from lora.lora import clevrer_lora_test_provider, split_validation_videos
    records = _scene_records(10)
    save_jsonl(records, tmp_path / "processed" / "val.jsonl")
    split = ClevrerLoRASplitConfig(validation="processed/val.jsonl", validation_videos=8)
    _, test, info = split_validation_videos(records[:8 * 3], seed=0)  # the first 8 videos are the LoRA part
    run = tmp_path / "run"
    run.mkdir()
    (run / "split.json").write_text(json.dumps({"data": asdict(split), **info}), encoding="utf-8")
    provider, loaded = clevrer_lora_test_provider(run, data_root=tmp_path, builder=None)
    assert [r.id for r in provider.records] == [r.id for r in test]
    assert loaded == split
    save_jsonl(records[3:], tmp_path / "processed" / "val.jsonl")  # the file changed since training
    with pytest.raises(ValueError, match="no longer matches"):
        clevrer_lora_test_provider(run, data_root=tmp_path, builder=None)


def test_saved_adapter_loads_into_a_fresh_model(tiny_base, tmp_path):
    from lora.lora import load_lora_adapter, save_lora_adapter
    from online_vllm.models import BackboneConfig, LoRAConfig, ModelConfig, StateHeadConfig, build_model
    from peft import get_peft_model_state_dict

    config = ModelConfig(backbone=BackboneConfig(model_id=tiny_base, dtype="float32"),
                         lora=LoRAConfig(r=4, alpha=8), head=StateHeadConfig(hidden_dims=[16], dropout=0))
    trained = build_model(config, local_files_only=True)
    trained.state_head.requires_grad_(False)  # as prepare_lora_sft_model does
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if parameter.requires_grad and ".lora_" in name:
                parameter.normal_()
    save_lora_adapter(trained, tmp_path / "best")
    fresh = load_lora_adapter(build_model(config, local_files_only=True), tmp_path / "best")
    saved, loaded = get_peft_model_state_dict(trained.backbone), get_peft_model_state_dict(fresh.backbone)
    assert saved.keys() == loaded.keys()
    for key in saved:
        torch.testing.assert_close(loaded[key], saved[key])


def test_lora_checkpoint_selection_metric_is_answerable_prefix_accuracy():
    assert SELECTION_METRIC == "answerable_micro_accuracy"
    split = ClevrerLoRASplitConfig()  # the LoRA halves of README §11, whole
    assert split.train.endswith("train_group1_lora.jsonl") and split.validation.endswith("val_group1_lora.jsonl")
    assert split.train_videos is None and split.validation_videos is None


def test_no_video_limit_takes_the_whole_file(tmp_path):
    records = _scene_records(5)
    save_jsonl(list(reversed(records)), tmp_path / "all.jsonl")
    assert len(_records_for_first_videos(tmp_path / "all.jsonl", None)) == len(records)
    assert {r.video for r in _records_for_first_videos(tmp_path / "all.jsonl", 2)} == {"v0.mp4", "v1.mp4"}


def test_accuracy_deltas_are_percentage_points_and_resolution_local():
    baseline = {
        "252x168": {"start_micro_accuracy": .2, "target_micro_accuracy": .3, "answerable_micro_accuracy": .35,
                    "end_micro_accuracy": .4},
        "504x336": {"start_micro_accuracy": .5, "target_micro_accuracy": .6, "answerable_micro_accuracy": .65,
                    "end_micro_accuracy": .7},
    }
    current = {
        "252x168": {"start_micro_accuracy": .25, "target_micro_accuracy": .3, "answerable_micro_accuracy": .4,
                    "end_micro_accuracy": .45},
        "504x336": {"start_micro_accuracy": .4, "target_micro_accuracy": .65, "answerable_micro_accuracy": .65,
                    "end_micro_accuracy": .7},
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
