"""Online SFT with <listen>/<speak> slots (lora/online.py) on the tiny Qwen."""
import json

import numpy as np
import pytest
import torch

from lora.lora import ClevrerLoRASplitConfig, LoRASFTConfig, load_lora_adapter, save_lora_adapter
from lora.online import (
    LISTEN,
    SPEAK,
    OnlineObjective,
    OnlineSession,
    RopeSpec,
    assert_online_isolation,
    build_example,
    build_online_loader,
    build_online_model,
    collate_online,
    encode_pair,
    evaluate_online,
    expected_lora_targets,
    online_hidden,
    online_losses,
    run_online,
    save_online_adapter,
    template,
    train_online_sft,
)
from lora.vision_cache import VisionCache
from online_vllm.data.inputs import format_prompt
from online_vllm.models import BackboneConfig, LoRAConfig, ModelConfig, StateHeadConfig
from test_lora_sft import _RemapBuilder, _toy_cache, _toy_videos


def _config(path):
    return ModelConfig(backbone=BackboneConfig(model_id=path, dtype="float32"),
                       lora=LoRAConfig(r=4, alpha=8, dropout=0), head=StateHeadConfig(hidden_dims=[16], dropout=0))


def _perturb(model, scale=0.1):
    """B of LoRA starts at zero and the token deltas too: make both change the output. Sharpen attention, so that a
    wrong mask or position would show."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_B" in name or "trainable_tokens_delta" in name:
                parameter.normal_(std=scale)
        for layer in model.qwen.model.language_model.layers:
            layer.self_attn.q_proj.base_layer.weight.mul_(6)
            layer.self_attn.k_proj.base_layer.weight.mul_(6)


@pytest.fixture
def online(builder, tiny_base, tmp_path):
    """Online model (fresh LoRA perturbed), its builder and tokens, toy data with a vision cache."""
    tiny = _RemapBuilder(builder.processor)
    model, tokens = build_online_model(_config(tiny_base), tiny.processor.tokenizer, init_adapter=None,
                                       local_files_only=True)
    _perturb(model)
    model.eval()
    data, records = _toy_videos(tmp_path)
    cache = _toy_cache(model, tiny, data, records, tmp_path)
    return model, tiny, tokens, data, records, cache


def _example(model, tiny, tokens, cache, record, branches, objective=OnlineObjective()):
    entry = cache.load(record.video)
    tpl = template(tiny, format_prompt(record.question, record.choices), entry["grid"], entry["second_per_grid_ts"])
    patches = VisionCache.patches(entry)
    return entry, tpl, build_example(tpl, entry, spec=RopeSpec.of(model), tokens=tokens,
                                     slot_targets=objective.targets(patches, 1), branch_pairs=branches, answer=10)


def _plain_hidden(model, ids, types, positions, video):
    """A separate causal sequence, no branches."""
    n = len(ids)
    batch = {"input_ids": torch.tensor([ids]), "mm_token_type_ids": torch.tensor([types]),
             "position_ids": positions[:, None], "attention_mask": torch.ones(1, 1, n, n, dtype=torch.bool).tril(),
             "video_embeds": video}
    with torch.no_grad():
        return online_hidden(model, batch)[0]


def test_tokens_are_added_once_and_only_lora_and_token_rows_train(online, builder):
    model, tiny, tokens, *_ = online
    tokenizer = tiny.processor.tokenizer
    assert tokenizer.convert_tokens_to_ids([LISTEN, SPEAK]) == [tokens.listen, tokens.speak]
    assert tokenizer(f"question {SPEAK}", add_special_tokens=False).input_ids[-1] == tokens.speak
    assert_online_isolation(model, expected_targets=expected_lora_targets(model))
    assert expected_lora_targets(model) == 2 * 4  # the tiny Qwen: 2 layers x q, k, v, o
    names = [name for name, p in model.named_parameters() if p.requires_grad and ".lora_" not in name]
    assert names and all("trainable_tokens_delta" in name for name in names)


def test_positions_are_those_of_the_offline_input(online):
    """Prefix and visual tokens keep the positions Qwen gives the contiguous video; a slot after pair k and the branch
    of pair k start where the offline template tail after pair k would."""
    model, tiny, tokens, data, records, cache = online
    entry, tpl, example = _example(model, tiny, tokens, cache, records[0], branches=[1, 3])
    offline = VisionCache.encode(tiny, format_prompt(records[0].question, records[0].choices), entry)
    qwen = model.qwen.model
    reference, _ = qwen.get_rope_index(offline["input_ids"], offline["mm_token_type_ids"],
                                       video_grid_thw=offline["video_grid_thw"],
                                       second_per_grid_ts=offline["second_per_grid_ts"],
                                       attention_mask=torch.ones_like(offline["input_ids"]))
    reference = reference[:, 0]
    ids, positions = example["input_ids"], example["position_ids"]
    not_slots = ids[:len(ids) - 2 * (1 + len(tpl.tail))] != tokens.listen  # main without slots
    main_positions = positions[:, :not_slots.numel()][:, not_slots]
    video_end = len(tpl.prefix) + 4 * tpl.per_pair
    torch.testing.assert_close(main_positions, reference[:, :video_end])
    tail_start = reference[0, video_end]  # the offline tail after the whole video
    slot_last = example["slot_index"][-1] + 1
    assert (positions[:, slot_last] == tail_start).all()
    branch_end = positions[:, -(1 + len(tpl.tail)):]
    assert (branch_end[:, 0] == tail_start).all()  # <speak> of the last-pair branch: where the tail would start


def test_branch_equals_a_separate_sequence_without_its_own_listen(online):
    """A branch after pair s == the sequence prefix, pairs 0..s with <listen> after pairs < s, then <speak> + tail.
    Slot logits of pair k == the main sequence cut after pair k (nothing later is seen)."""
    model, tiny, tokens, data, records, cache = online
    entry, tpl, example = _example(model, tiny, tokens, cache, records[0], branches=[0, 2, 3])
    with torch.no_grad():
        hidden = online_hidden(model, collate_online([example], pad_id=1))[0]
    ids, types, positions = example["input_ids"], example["mm_token_type_ids"], example["position_ids"]
    per, branch = tpl.per_pair, 1 + len(tpl.tail)
    for row, s in enumerate([0, 2, 3]):
        cut = int(example["slot_index"][s]) + 1  # through the last visual token of pair s, not slot s
        start = int(example["answer_index"][row]) - branch + 1
        assert ids[start] == tokens.speak and ids[cut] == tokens.listen  # slot s is left out of the reference
        plain = _plain_hidden(model, ids[:cut].tolist() + ids[start:start + branch].tolist(),
                              types[:cut].tolist() + [0] * branch,
                              torch.cat([positions[:, :cut], positions[:, start:start + branch]], 1),
                              entry["embeds"][:(s + 1) * per])
        torch.testing.assert_close(hidden[start:start + branch], plain[cut:], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(hidden[cut - 1], plain[cut - 1], rtol=1e-4, atol=1e-4)  # the slot-s prediction


def test_the_input_of_slot_s_does_not_reach_branch_s(online):
    model, tiny, tokens, data, records, cache = online
    _, _, example = _example(model, tiny, tokens, cache, records[0], branches=[1])
    changed = dict(example)
    changed["input_ids"] = example["input_ids"].clone()
    slot = int(example["slot_index"][1]) + 1
    changed["input_ids"][slot] = tokens.speak
    with torch.no_grad():
        a = online_hidden(model, collate_online([example], pad_id=1))[0]
        b = online_hidden(model, collate_online([changed], pad_id=1))[0]
    branch = slice(-(1 + 2), None)  # <speak> + the toy tail (<|vision_end|>, assistant)
    torch.testing.assert_close(a[branch], b[branch])
    assert not torch.allclose(a[slot + 1:slot + 10], b[slot + 1:slot + 10])  # later pairs do see it


def test_targets_loss_and_padding(online):
    model, tiny, tokens, data, records, cache = online
    hard = OnlineObjective()
    assert hard.targets(4, 1).tolist() == [0, 1, 1, 1]
    assert OnlineObjective(tags="step", step_window=1).targets(4, 2).tolist() == [0, 0.5, 1, 1]
    with pytest.raises(ValueError):
        OnlineObjective(tags="gaussian").validate()
    _, _, a = _example(model, tiny, tokens, cache, records[0], branches=[1, 3])
    # One branch: a shorter sequence, right-padded in the batch.
    _, _, b = _example(model, tiny, tokens, cache, records[2], branches=[3], objective=OnlineObjective(tags="sigmoid"))
    assert len(b["input_ids"]) < len(a["input_ids"])
    with torch.no_grad():
        both = online_losses(model, collate_online([a, b], pad_id=1), tokens)
        single = [online_losses(model, collate_online([x], pad_id=1), tokens) for x in (a, b)]
    # 4 slots each: the mean of the two; the answer loss is the mean over the 3 branches of the batch.
    torch.testing.assert_close(both["slot"], (single[0]["slot"] + single[1]["slot"]) / 2, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(both["answer"], (2 * single[0]["answer"] + single[1]["answer"]) / 3,
                               rtol=1e-4, atol=1e-5)
    slots, y = single[0]["slots"], a["slot_target"]
    manual = -(y * slots[:, tokens.speak] + (1 - y) * slots[:, tokens.listen]).mean()
    torch.testing.assert_close(single[0]["slot"], manual)


def test_gradients_reach_only_lora_and_the_two_token_rows(online):
    model, tiny, tokens, data, records, cache = online
    model.train()
    _, _, example = _example(model, tiny, tokens, cache, records[0], branches=[1, 3])
    embed = model.qwen.get_input_embeddings()
    before = embed(torch.arange(tokens.listen + 1)).detach().clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    parts = online_losses(model, collate_online([example], pad_id=1), tokens)
    (parts["slot"] + parts["answer"]).backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
    assert all(g is not None for g in grads.values())
    assert any(g.abs().sum() > 0 for name, g in grads.items() if "trainable_tokens_delta" in name)
    optimizer.step()
    after = embed(torch.arange(tokens.listen + 1)).detach()
    torch.testing.assert_close(after[:tokens.listen], before[:tokens.listen])  # every other row stays
    assert not torch.allclose(after[tokens.listen], before[tokens.listen])


def test_streaming_session_equals_one_pass(online):
    """KV-cache inference, pair by pair with <listen>, gives the p_k of the one-pass evaluation, and <speak> after pair
    j gives the letters of branch j."""
    model, tiny, tokens, data, records, cache = online
    record = records[1]
    entry, tpl, example = _example(model, tiny, tokens, cache, record, branches=range(4))
    with torch.no_grad():
        parts = online_losses(model, collate_online([example], pad_id=1), tokens)
    p_one_pass = parts["slots"][:, [tokens.speak, tokens.listen]].softmax(-1)[:, 0]
    letters = [10, 11]
    expected = parts["answers"][:, letters].log_softmax(-1)
    prompt = format_prompt(record.question, record.choices)
    grid, per = entry["grid"].tolist(), tpl.per_pair
    pairs = [entry["embeds"][k * per:(k + 1) * per] for k in range(4)]
    for j in range(4):
        session = OnlineSession(model, tiny, tokens, prompt, grid_hw=grid[1:],
                                seconds_per_pair=float(entry["second_per_grid_ts"]), n_options=2)
        for k in range(j + 1):
            assert session.observe(pairs[k]) == pytest.approx(float(p_one_pass[k]), abs=1e-4)
            if k < j:
                session.listen()
        torch.testing.assert_close(session.speak(), expected[j], rtol=1e-4, atol=1e-4)
    result = run_online(OnlineSession(model, tiny, tokens, prompt, grid_hw=grid[1:],
                                      seconds_per_pair=float(entry["second_per_grid_ts"]), n_options=2),
                        pairs, tau=float(p_one_pass.max()) - 1e-3)
    assert result["fired_pair"] == int(p_one_pass.argmax())
    missed = run_online(OnlineSession(model, tiny, tokens, prompt, grid_hw=grid[1:],
                                      seconds_per_pair=float(entry["second_per_grid_ts"]), n_options=2), pairs, tau=1.0)
    assert missed["missed"] and len(missed["probabilities"]) == 4
    torch.testing.assert_close(torch.tensor(missed["letter_logprobs"]), expected[3], rtol=1e-4, atol=1e-4)


def test_encode_pair_equals_the_cache_slice(online):
    from online_vllm.data.video import read_every
    model, tiny, tokens, data, records, cache = online
    entry = cache.load(records[0].video)
    frames = read_every(data / records[0].video, 1, (28, 28))
    per = entry["embeds"].shape[0] // VisionCache.patches(entry)
    for k in (0, 3):
        torch.testing.assert_close(encode_pair(model, tiny, frames[2 * k:2 * k + 2], 25.0),
                                   entry["embeds"][k * per:(k + 1) * per], rtol=1e-4, atol=1e-4)


@pytest.fixture(params=[False, True], ids=["untied", "tied"])
def base_path(request, tiny_base, tmp_path_factory):
    if not request.param:
        return tiny_base
    from transformers import Qwen2_5_VLForConditionalGeneration
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(tiny_base)
    base.config.tie_word_embeddings = base.config.text_config.tie_word_embeddings = True
    base.tie_weights()
    path = tmp_path_factory.mktemp("tiny_tied")
    base.save_pretrained(path)
    return str(path)


def test_offline_adapter_is_merged_and_the_run_reloads(builder, base_path, tmp_path):
    """The online model without training answers as the offline model with its adapter; a saved online adapter (LoRA
    and token rows) loads into a freshly built online model."""
    from online_vllm.models import build_model
    offline = build_model(_config(base_path), local_files_only=True).eval()
    offline.state_head.requires_grad_(False)  # as prepare_lora_sft_model does
    with torch.no_grad():
        for name, parameter in offline.named_parameters():
            if ".lora_B" in name:
                parameter.normal_(std=0.2)
    save_lora_adapter(offline, tmp_path / "offline")
    tiny = _RemapBuilder(builder.processor)
    model, tokens = build_online_model(_config(base_path), tiny.processor.tokenizer, init_adapter=tmp_path / "offline",
                                       local_files_only=True)
    model.eval()
    ids = torch.tensor([[1, 6, 9, 10, 7]])
    old = [i for i in range(offline.qwen.config.text_config.vocab_size) if i not in (tokens.listen, tokens.speak)]
    with torch.no_grad():  # the rows of the two new tokens were set from INIT_WORDS: every other logit is the same
        torch.testing.assert_close(model.qwen(input_ids=ids).logits[..., old],
                                   offline.qwen(input_ids=ids).logits[..., old], rtol=1e-4, atol=1e-4)
    _perturb(model, scale=0.3)
    ids = torch.tensor([[1, 6, tokens.listen, 9, tokens.speak, 7]])
    with torch.no_grad():
        trained = model.qwen(input_ids=ids).logits
    save_online_adapter(model, tmp_path / "online")
    fresh, again = build_online_model(_config(base_path), _RemapBuilder(builder.processor).processor.tokenizer,
                                      init_adapter=tmp_path / "offline", local_files_only=True)
    assert again == tokens
    # _perturb sharpened the base attention too: do the same to the fresh base, then load the adapter.
    with torch.no_grad():
        for layer in fresh.qwen.model.language_model.layers:
            layer.self_attn.q_proj.base_layer.weight.mul_(6)
            layer.self_attn.k_proj.base_layer.weight.mul_(6)
    load_lora_adapter(fresh, tmp_path / "online")
    with torch.no_grad():
        torch.testing.assert_close(fresh.eval().qwen(input_ids=ids).logits, trained, rtol=1e-4, atol=1e-4)


def test_training_run_writes_adapters_metrics_and_predictions(online, tmp_path):
    model, tiny, tokens, data, records, cache = online
    model.train()
    split = ClevrerLoRASplitConfig(num_workers=0, vision_cache=str(cache.path))
    objective = OnlineObjective(tags="sigmoid")
    loader = build_online_loader(records, builder=tiny, split=split, tokens=tokens, spec=RopeSpec.of(model),
                                 objective=objective, seed=0, batch_size=2)

    def evaluate():
        return evaluate_online(model, records[:2], builder=tiny, split=split, tokens=tokens, spec=RopeSpec.of(model),
                               objective=objective, device="cpu", taus=(0.5, 0.9))

    out = tmp_path / "run"
    best = train_online_sft(model, loader, LoRASFTConfig(epochs=1, batch_size=2, gradient_accumulation=1),
                            output_dir=out, device="cpu", tokens=tokens, objective=objective, evaluate=evaluate,
                            run_info={"init_adapter": None})
    assert best == out / "best" and (out / "last" / "adapter_model.safetensors").exists()
    lines = [json.loads(line) for line in (out / "validation_metrics.jsonl").read_text().splitlines()]
    assert [line["epoch"] for line in lines] == [0, 1]
    summary = lines[1]["validation"]
    assert summary["count"] == 2 and set(summary["online"]) == {"0.5", "0.9"}
    assert 0 < summary["control_mass"] <= 1
    predictions = json.loads((out / "predictions_validation_epoch_1.json").read_text())["predictions"]
    assert all(len(row["probabilities"]) == len(row["answer_correct"]) == 4 for row in predictions)
    info = json.loads((out / "online.json").read_text())
    assert info["tokens"] == {"listen": tokens.listen, "speak": tokens.speak}
    assert np.isfinite([line["train_loss"] for line in
                        map(json.loads, (out / "train_steps.jsonl").read_text().splitlines())]).all()
