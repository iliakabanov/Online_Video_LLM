"""Separate decision module (separate/decision.py) on the tiny Qwen2.5-VL and the tiny LLaVA-OneVision."""
import json

import pytest
import torch

from lora.lora import ClevrerLoRASplitConfig, LoRASFTConfig
from lora.online import template
from lora.vision_cache import VisionCache
from online_vllm.data.inputs import format_prompt
from online_vllm.models import BackboneConfig, LoRAConfig, ModelConfig, StateHeadConfig
from separate.decision import (
    ACT,
    LLAVA,
    QWEN,
    DecisionDataset,
    DecisionObjective,
    DecisionSession,
    LlavaFamily,
    QwenFamily,
    answers_of_predictions,
    assert_separate_isolation,
    batch_logits,
    build_decision_example,
    build_decision_loader,
    build_llava_vision_cache,
    build_separate_model,
    collate_decision,
    decision_loss,
    evaluate_decision,
    expected_lora_targets,
    llava_cache_settings,
    llava_frame_features,
    load_separate_adapter,
    run_decision,
    save_separate_adapter,
    time_decisions,
    train_separate,
)
from test_lora_sft import _RemapBuilder, _toy_cache, _toy_videos
from uhead.config import UHeadConfig


def _config(path):
    return ModelConfig(backbone=BackboneConfig(model_id=path, dtype="float32"),
                       lora=LoRAConfig(r=4, alpha=8, dropout=0), head=StateHeadConfig(hidden_dims=[16], dropout=0))


def _sharpen(model):
    """Sharper attention, so that a wrong mask or position would show."""
    with torch.no_grad():
        for layer in model.qwen.model.language_model.layers:
            layer.self_attn.q_proj.base_layer.weight.mul_(6)
            layer.self_attn.k_proj.base_layer.weight.mul_(6)


def _perturb(model, scale=0.1):
    """B of LoRA and the <act> delta start at zero: make both change the output."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_B" in name or "trainable_tokens_delta" in name:
                parameter.normal_(std=scale)
    _sharpen(model)


def _build(path, tokenizer, train_projector=False):
    torch.manual_seed(0)
    return build_separate_model(_config(path), tokenizer, UHeadConfig(dropout=0.0), local_files_only=True,
                                train_projector=train_projector)


@pytest.fixture(scope="session")
def tiny_llava(tmp_path_factory):
    from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration
    torch.manual_seed(11)
    config = LlavaOnevisionConfig(
        text_config=dict(model_type="qwen2", vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                         tie_word_embeddings=True),
        vision_config=dict(model_type="siglip_vision_model", hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, image_size=56, patch_size=14),
        image_token_index=60, video_token_index=61, vision_feature_layer=-1, vision_feature_select_strategy="full",
        image_grid_pinpoints=[[56, 56]], tie_word_embeddings=True)
    path = tmp_path_factory.mktemp("tiny_llava")
    LlavaOnevisionForConditionalGeneration(config).save_pretrained(path)
    return str(path)


def _video_processor():
    from transformers import LlavaOnevisionVideoProcessor
    return LlavaOnevisionVideoProcessor(size={"height": 56, "width": 56})


@pytest.fixture(params=[QWEN, LLAVA, "llava_projector"])
def separate(request, builder, tiny_base, tiny_llava, tmp_path):
    """Decision module of either family, LLaVA-OV also with its projector trained (LoRA and <act> perturbed), its
    family, toy data with a vision cache, the split settings."""
    data, records = _toy_videos(tmp_path)
    if request.param == QWEN:
        tiny = _RemapBuilder(builder.processor)
        model = _build(tiny_base, tiny.processor.tokenizer)
        family = QwenFamily(model, tiny)
        cache = _toy_cache(model, tiny, data, records, tmp_path)
    else:
        model = _build(tiny_llava, builder.processor.tokenizer, train_projector=request.param != LLAVA)
        family = LlavaFamily(model, builder.processor.tokenizer)
        cache = VisionCache(tmp_path / "llava_cache", llava_cache_settings(model, frames_per_pair=2, tokens_side=None))
        assert build_llava_vision_cache(model, _video_processor(), records, cache, data_root=data, num_workers=0) == 2
    _perturb(model)
    model.eval()
    split = ClevrerLoRASplitConfig(num_workers=0, vision_cache=str(cache.path), train_frame_size=(28, 28))
    return model, family, data, records, cache, split


def _dataset(model, family, records, split, objective=DecisionObjective()):
    return DecisionDataset(records, family=family, split=split, act=model.act, objective=objective)


def _logits(model, family, examples):
    with torch.no_grad():
        return batch_logits(model, collate_decision(examples, pad_id=1), family)


def _cut(entry, pairs):
    """The cache entry of the video's first ``pairs`` pairs."""
    total = VisionCache.patches(entry)
    grid = entry["grid"].clone()
    grid[0] = pairs
    return {**entry, "grid": grid, "embeds": entry["embeds"][:pairs * (entry["embeds"].shape[0] // total)]}


def test_act_is_one_token_and_only_lora_the_act_row_and_the_head_train(separate):
    model, family, *_ = separate
    tokenizer = family.builder.processor.tokenizer if isinstance(family, QwenFamily) else family.tokenizer
    assert tokenizer.convert_tokens_to_ids(ACT) == model.act
    assert tokenizer(f"question {ACT}", add_special_tokens=False).input_ids[-1] == model.act
    assert_separate_isolation(model, expected_targets=expected_lora_targets(model))
    assert expected_lora_targets(model) == 2 * 4  # the tiny language models: 2 layers x q, k, v, o
    lora = [name for name, p in model.named_parameters() if p.requires_grad and ".lora_" in name]
    assert all("vision" not in name for name in lora)  # LLaVA's SigLIP has q_proj too: it stays frozen
    other = [name for name, p in model.named_parameters() if p.requires_grad and ".lora_" not in name
             and not name.startswith("head.") and "multi_modal_projector" not in name]
    assert other and all("trainable_tokens_delta" in name for name in other)
    projector = [name for name, p in model.named_parameters() if p.requires_grad and "multi_modal_projector" in name]
    assert bool(projector) == model.train_projector
    assert all(".modules_to_save." in name for name in projector)  # a trainable copy; the original stays frozen
    assert model.head_config.groups == ["last_hidden"] and model.head_config.temporal_detector == "mlp"
    assert model.head_config.input_dims == {"last_hidden": model.qwen.config.text_config.hidden_size}


def test_targets():
    assert DecisionObjective().targets(4, 1).tolist() == [0, 1, 1, 1]
    assert DecisionObjective(tags="step", step_window=1).targets(4, 2).tolist() == [0, 0.5, 1, 1]
    with pytest.raises(ValueError):
        DecisionObjective(tags="gaussian").validate()


def test_qwen_layout_and_positions_are_those_of_the_offline_input(builder, tiny_base, tmp_path):
    """<act> after every pair; the prompt and the visual tokens keep the positions Qwen gives the contiguous video, a
    slot after pair k stands at the largest position so far + 1."""
    tiny = _RemapBuilder(builder.processor)
    model = _build(tiny_base, tiny.processor.tokenizer)
    data, records = _toy_videos(tmp_path)
    cache = _toy_cache(model, tiny, data, records, tmp_path)
    split = ClevrerLoRASplitConfig(num_workers=0, vision_cache=str(cache.path), train_frame_size=(28, 28))
    record = records[0]
    example = _dataset(model, QwenFamily(model, tiny), records, split)[0]
    entry = cache.load(record.video)
    patches = VisionCache.patches(entry)
    ids, positions = example["input_ids"], example["position_ids"]
    assert (ids == model.act).sum() == patches == len(example["target"])
    assert (ids[example["act_index"]] == model.act).all() and example["act_index"][-1] == len(ids) - 1
    offline = VisionCache.encode(tiny, format_prompt(record.question, record.choices), entry)
    reference, _ = model.qwen.model.get_rope_index(offline["input_ids"], offline["mm_token_type_ids"],
                                                   video_grid_thw=offline["video_grid_thw"],
                                                   second_per_grid_ts=offline["second_per_grid_ts"],
                                                   attention_mask=torch.ones_like(offline["input_ids"]))
    not_act = ids != model.act
    tpl = template(tiny, format_prompt(record.question, record.choices), entry["grid"], entry["second_per_grid_ts"])
    video_end = len(tpl.prefix) + patches * tpl.per_pair
    torch.testing.assert_close(positions[:, not_act], reference[:, 0, :video_end])
    for index in example["act_index"].tolist():
        assert (positions[:, index] == positions[:, :index][:, not_act[:index]].max() + 1).all()


def test_llava_cache_and_layout(builder, tiny_llava, tmp_path):
    """The cache holds SigLIP's features of every frame before the projector, pooled as LLaVA-OV pools (2 x 2; one frame
    per pair: the last; pooled further); the sequence is the prompt with the question, then per pair its frame tokens
    and <act>, 1D positions."""
    model = _build(tiny_llava, builder.processor.tokenizer)
    data, records = _toy_videos(tmp_path)
    caches = {}
    for frames, side in ((2, None), (1, None), (2, 1)):
        cache = VisionCache(tmp_path / f"cache_{frames}_{side}",
                            llava_cache_settings(model, frames_per_pair=frames, tokens_side=side))
        build_llava_vision_cache(model, _video_processor(), records, cache, data_root=data, num_workers=0)
        caches[frames, side] = cache.load(records[0].video)
    from online_vllm.data.video import read_every
    pixels = _video_processor()(videos=[read_every(data / records[0].video, 1, (56, 56))],
                                return_tensors="pt")["pixel_values_videos"]
    frames = llava_frame_features(model, pixels, tokens_side=2)                     # [8 frames, 2 x 2, C]
    # An independent reference: the encoder's last layer (the tiny config's feature layer, strategy "full"), 4 x 4
    # patches pooled bilinearly to LLaVA-OV's 2 x 2, before the projector (C = the encoder's width, not the LM's).
    with torch.no_grad():
        states = model.qwen.model.vision_tower(pixels[0], output_hidden_states=True).hidden_states[-1]
    grid = states.view(8, 4, 4, -1).permute(0, 3, 1, 2)
    reference = torch.nn.functional.interpolate(grid, size=(2, 2), mode="bilinear").permute(0, 2, 3, 1).reshape(8, 4, -1)
    torch.testing.assert_close(frames, reference, rtol=1e-4, atol=1e-5)
    assert frames.shape[-1] == model.qwen.config.vision_config.hidden_size != model.qwen.config.text_config.hidden_size
    both, last, pooled = caches[2, None], caches[1, None], caches[2, 1]
    assert both["grid"].tolist() == [4, 2, 4] and last["grid"].tolist() == [4, 1, 4] and pooled["grid"].tolist() == [4, 2, 1]
    torch.testing.assert_close(both["embeds"], frames.reshape(-1, frames.shape[-1]))
    torch.testing.assert_close(last["embeds"], frames[1::2].reshape(-1, frames.shape[-1]))
    torch.testing.assert_close(pooled["embeds"], frames.mean(1), rtol=1e-4, atol=1e-5)  # 2 x 2 -> 1: the mean
    assert float(both["second_per_grid_ts"]) == pytest.approx(2 / 25)

    family = LlavaFamily(model, builder.processor.tokenizer)
    split = ClevrerLoRASplitConfig(num_workers=0, vision_cache=str(tmp_path / "cache_2_None"))
    example = _dataset(model, family, records, split)[0]
    ids = example["input_ids"]
    prefix = family.prefix(records[0])
    assert ids[:len(prefix)].tolist() == prefix and (ids == 61).sum() == 4 * 2 * 4
    assert example["act_index"].tolist() == [len(prefix) + 9 * (k + 1) - 1 for k in range(4)]
    assert torch.equal(example["position_ids"], torch.arange(len(ids))[None])


def test_one_pass_equals_the_video_cut_after_every_pair(separate):
    """p_k of the whole sequence == p_k of the sequence that ends after pair k: no slot sees a later pair."""
    model, family, data, records, cache, split = separate
    record = records[0]  # target pair 1
    entry = cache.load(record.video)
    objective = DecisionObjective()
    full = _logits(model, family, [family.example(record, entry, act=model.act, objective=objective, split=split)[1]])
    for k in range(1, 4):
        cut = family.example(record, _cut(entry, k + 1), act=model.act, objective=objective, split=split)[1]
        torch.testing.assert_close(_logits(model, family, [cut]), full[:k + 1], rtol=1e-4, atol=1e-5)
    changed = dict(entry)
    per = entry["embeds"].shape[0] // 4
    changed["embeds"] = entry["embeds"].clone()
    changed["embeds"][2 * per:] += 1.0
    other = _logits(model, family, [family.example(record, changed, act=model.act, objective=objective,
                                                   split=split)[1]])
    torch.testing.assert_close(other[:2], full[:2], rtol=1e-4, atol=1e-5)
    assert not torch.allclose(other[2:], full[2:])


def test_padding_and_loss(separate):
    model, family, data, records, cache, split = separate
    a = _dataset(model, family, records, split)[1]
    b = family.example(records[0], _cut(cache.load(records[0].video), 2), act=model.act,
                       objective=DecisionObjective(tags="sigmoid"), split=split)[1]  # shorter: padded
    assert len(b["input_ids"]) < len(a["input_ids"])
    both = _logits(model, family, [a, b])
    torch.testing.assert_close(both, torch.cat([_logits(model, family, [a]), _logits(model, family, [b])]),
                               rtol=1e-4, atol=1e-5)
    with torch.no_grad():
        loss = decision_loss(model, collate_decision([a, b], pad_id=1), family)
    targets = torch.cat([a["target"], b["target"]])
    torch.testing.assert_close(loss, torch.nn.functional.binary_cross_entropy_with_logits(both, targets))


def test_gradients_reach_only_lora_the_act_row_and_the_head(separate):
    model, family, data, records, cache, split = separate
    model.train()
    embed = model.qwen.get_input_embeddings()
    rows = torch.arange(model.act + 1)
    before = embed(rows).detach().clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    decision_loss(model, collate_decision([_dataset(model, family, records, split)[0]], pad_id=1), family).backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
    assert all(g is not None for g in grads.values())
    parts = ["trainable_tokens_delta", ".lora_A", "head."] + (["multi_modal_projector"] if model.train_projector else [])
    for part in parts:
        assert any(g.abs().sum() > 0 for name, g in grads.items() if part in name), part
    assert model.train_projector == any("multi_modal_projector" in name for name in grads)
    optimizer.step()
    after = embed(rows).detach()
    torch.testing.assert_close(after[:model.act], before[:model.act])  # every other row stays
    assert not torch.allclose(after[model.act], before[model.act])


def test_streaming_session_equals_one_pass(separate):
    model, family, data, records, cache, split = separate
    record = records[1]
    entry = cache.load(record.video)
    one_pass = _logits(model, family, [_dataset(model, family, [record], split)[0]]).sigmoid()
    per = entry["embeds"].shape[0] // 4
    pairs = [entry["embeds"][k * per:(k + 1) * per] for k in range(4)]
    streamed = DecisionSession(model, family, record, entry)
    for k in range(4):
        assert streamed.observe(pairs[k]) == pytest.approx(float(one_pass[k]), abs=1e-5)
    fired = run_decision(DecisionSession(model, family, record, entry), pairs, tau=float(one_pass.max()) - 1e-4)
    assert fired["fired_pair"] == int(one_pass.argmax()) and not fired["missed"]
    missed = run_decision(DecisionSession(model, family, record, entry), pairs, tau=1.0)
    assert missed["missed"] and missed["fired_pair"] is None and len(missed["probabilities"]) == 4
    timing = time_decisions(model, family, records[:2], split=split)
    assert timing["decisions"] == 8 and timing["ms_per_decision_mean"] > 0


def test_answers_of_m_off_come_with_the_questions(separate):
    model, family, data, records, cache, split = separate
    answers = {record.id: [False, True, True, i % 2 == 0] for i, record in enumerate(records)}
    summary, payload = evaluate_decision(model, records, family=family, split=split, objective=DecisionObjective(),
                                         device="cpu", answers=answers, taus=(0.5,), batch_size=3)
    assert [row["record_id"] for row in payload["predictions"]] == [record.id for record in records]
    assert all(row["answer_correct"] == answers[row["record_id"]] for row in payload["predictions"])
    one = _logits(model, family, [_dataset(model, family, [records[3]], split)[0]]).sigmoid()
    torch.testing.assert_close(torch.tensor(payload["predictions"][3]["probabilities"]), one, rtol=1e-4, atol=1e-5)
    assert summary["count"] == 4 and summary["latency_accuracy"] is not None
    assert summary["classification"]["count"] == 4 * 4
    assert answers_of_predictions(payload) == answers
    with pytest.raises(ValueError, match="no answers of M_off"):
        evaluate_decision(model, records, family=family, split=split, objective=DecisionObjective(), device="cpu",
                          answers={}, taus=(0.5,))
    with pytest.raises(ValueError, match="no answer_correct"):
        answers_of_predictions({"predictions": [{"record_id": "x", "probabilities": [0.1]}]})


def test_tied_embeddings_as_in_qwen_3b(builder, tiny_base, tmp_path):
    """Qwen2.5-VL-3B ties the embedding and lm_head: <act> is one trainable row, and a saved adapter reloads."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(tiny_base)
    base.config.tie_word_embeddings = base.config.text_config.tie_word_embeddings = True
    base.tie_weights()
    base.save_pretrained(tmp_path / "tied")
    reloaded = Qwen2_5_VLForConditionalGeneration.from_pretrained(tmp_path / "tied")
    assert reloaded.lm_head.weight.data_ptr() == reloaded.get_input_embeddings().weight.data_ptr()
    tiny = _RemapBuilder(builder.processor)
    model = _build(str(tmp_path / "tied"), tiny.processor.tokenizer)
    family = QwenFamily(model, tiny)
    _perturb(model)
    assert_separate_isolation(model, expected_targets=expected_lora_targets(model))
    data, records = _toy_videos(tmp_path)
    cache = _toy_cache(model, tiny, data, records, tmp_path)
    split = ClevrerLoRASplitConfig(num_workers=0, vision_cache=str(cache.path), train_frame_size=(28, 28))
    example = _dataset(model, family, records[:1], split)[0]
    model.train()
    decision_loss(model, collate_decision([example], pad_id=1), family).backward()
    deltas = [p.grad for name, p in model.named_parameters() if "trainable_tokens_delta" in name]
    assert deltas and any(g.abs().sum() > 0 for g in deltas)
    model.eval()
    save_separate_adapter(model, tmp_path / "adapter")
    fresh = _build(str(tmp_path / "tied"), _RemapBuilder(builder.processor).processor.tokenizer)
    _sharpen(fresh)
    load_separate_adapter(fresh, tmp_path / "adapter")
    torch.testing.assert_close(_logits(fresh.eval(), family, [example]), _logits(model, family, [example]),
                               rtol=1e-4, atol=1e-5)


def test_training_run_writes_adapters_metrics_and_predictions_and_reloads(separate, builder, tiny_base, tiny_llava,
                                                                          tmp_path):
    model, family, data, records, cache, split = separate
    model.train()
    objective = DecisionObjective()
    answers = {record.id: [False, True, True, True] for record in records}
    loader = build_decision_loader(records, family=family, split=split, act=model.act, objective=objective, seed=0,
                                   batch_size=2)

    def evaluate():
        return evaluate_decision(model, records, family=family, split=split, objective=objective, device="cpu",
                                 answers=answers, taus=(0.5, 0.9))

    out = tmp_path / "run"
    best = train_separate(model, loader, LoRASFTConfig(epochs=2, batch_size=2, gradient_accumulation=1),
                          family=family, output_dir=out, device="cpu", objective=objective, evaluate=evaluate,
                          run_info={"note": "test"}, head_lr=1e-3)
    assert best == out / "best"
    for name in ("best", "last", "epoch_1", "epoch_2"):
        assert (out / name / "adapter_model.safetensors").exists() and (out / name / "head.safetensors").exists()
    assert (out / "last" / "optimizer.pt").exists() and not (out / "epoch_1" / "optimizer.pt").exists()
    lines = [json.loads(line) for line in (out / "validation_metrics.jsonl").read_text().splitlines()]
    assert [line["epoch"] for line in lines] == [0, 1, 2]
    scores = [line["validation"]["classification"]["auroc_pm2"] for line in lines]
    best_info = json.loads((out / "best" / "sft_metadata.json").read_text())
    assert best_info["selection"] == "auroc_pm2" and best_info["epoch"] == scores.index(max(scores))
    info = json.loads((out / "separate.json").read_text())
    assert info["family"] == model.family and info["act"] == model.act and info["head_lr"] == 1e-3
    predictions = json.loads((out / "predictions_validation_epoch_2.json").read_text())["predictions"]
    assert all(row["answer_correct"] == answers[row["record_id"]] for row in predictions)

    # The last adapter (LoRA, <act> row, head) loads into a fresh module and gives the same decisions.
    model.eval()
    example = _dataset(model, family, records[:1], split)[0]
    trained = _logits(model, family, [example])
    if model.family == QWEN:
        fresh = _build(tiny_base, _RemapBuilder(builder.processor).processor.tokenizer)
    else:
        fresh = _build(tiny_llava, builder.processor.tokenizer, train_projector=model.train_projector)
    assert fresh.act == model.act and fresh.family == model.family
    assert info["train_projector"] == model.train_projector
    _sharpen(fresh)  # the base attention that _perturb sharpened in the fixture
    load_separate_adapter(fresh, out / "last")
    torch.testing.assert_close(_logits(fresh.eval(), family, [example]), trained, rtol=1e-4, atol=1e-5)


def test_an_interrupted_run_resumes_as_one_run(separate, builder, tiny_base, tiny_llava, tmp_path):
    """3 epochs at once == 1 epoch, an interruption during epoch 2 (its metrics line, steps, predictions and epoch_2/
    already written), then resume in the same directory: the same weights (LoRA, <act>, projector, head), metrics,
    steps and best epoch; what the interrupted epoch left is replaced; a complete run is left as it is."""
    import shutil
    model, family, data, records, cache, split = separate
    objective = DecisionObjective()
    answers = {record.id: [False, True, True, True] for record in records}

    def run(m, out, epochs, resume=False):
        m.train()
        loader = build_decision_loader(records, family=family, split=split, act=m.act, objective=objective, seed=0,
                                       batch_size=2)

        def evaluate():
            return evaluate_decision(m, records[:2], family=family, split=split, objective=objective, device="cpu",
                                     answers=answers, taus=(0.5,))

        return train_separate(m, loader, LoRASFTConfig(epochs=epochs, batch_size=2, gradient_accumulation=1),
                              family=family, output_dir=out, device="cpu", objective=objective, evaluate=evaluate,
                              run_info={}, resume=resume)

    def trainable(m):
        return {name: p.detach().clone() for name, p in m.named_parameters() if p.requires_grad}

    start = trainable(model)
    run(model, tmp_path / "whole", 3)
    whole = trainable(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in start:
                parameter.copy_(start[name])
    out = tmp_path / "cut"
    run(model, out, 1)
    assert (out / "complete").exists()
    (out / "complete").unlink()  # interrupted during epoch 2, after its evaluation, before last/ was replaced
    shutil.copytree(out / "last", out / "epoch_2")
    shutil.copy(out / "predictions_validation_epoch_1.json", out / "predictions_validation_epoch_2.json")
    lines = (out / "validation_metrics.jsonl").read_text().splitlines()
    fake = {**json.loads(lines[-1]), "epoch": 2}
    (out / "validation_metrics.jsonl").write_text("\n".join(lines + [json.dumps(fake)]) + "\n")
    with (out / "train_steps.jsonl").open("a") as file:
        file.write(json.dumps({"global_step": 99, "epoch": 2}) + "\n")

    if model.family == QWEN:
        fresh = _build(tiny_base, _RemapBuilder(builder.processor).processor.tokenizer)
    else:
        fresh = _build(tiny_llava, builder.processor.tokenizer, train_projector=model.train_projector)
    _sharpen(fresh)  # the base attention that _perturb sharpened in the fixture; the rest comes from last/
    run(fresh, out, 3, resume=True)
    resumed = trainable(fresh)
    assert resumed.keys() == whole.keys()
    for name, value in resumed.items():
        torch.testing.assert_close(value, whole[name], rtol=1e-5, atol=1e-6)

    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    a, b = read(tmp_path / "whole" / "validation_metrics.jsonl"), read(out / "validation_metrics.jsonl")
    assert [line["epoch"] for line in b] == [0, 1, 2, 3]
    assert [line["best_epoch"] for line in b] == [line["best_epoch"] for line in a]
    for x, y in zip(a, b):
        assert x["validation"]["loss"] == pytest.approx(y["validation"]["loss"], rel=1e-5, abs=1e-6)
    steps = [(s["epoch"], s["global_step"]) for s in read(out / "train_steps.jsonl")]
    assert steps == [(s["epoch"], s["global_step"]) for s in read(tmp_path / "whole" / "train_steps.jsonl")]
    best = [json.loads((d / "best" / "sft_metadata.json").read_text())["epoch"] for d in (tmp_path / "whole", out)]
    assert best[0] == best[1] and (out / "complete").exists()
    assert json.loads((out / "epoch_2" / "sft_metadata.json").read_text())["epoch"] == 2

    before = (out / "validation_metrics.jsonl").read_text()
    run(fresh, out, 3, resume=True)  # complete: nothing to do
    assert (out / "validation_metrics.jsonl").read_text() == before
