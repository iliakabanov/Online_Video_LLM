"""Оценка без дообучения (eval/zero_shot.py, data/turns.py): вход MMDuet2, один проход = онлайн-работа, выборка,
статистика и формат предсказаний."""
import json
import random

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor

from online_vllm.data.schema import Interval, Record
from online_vllm.data.turns import NO_REPLY, TurnsBuilder
from online_vllm.eval.readiness_plot import load_traces
from online_vllm.eval.thresholds import sweep
from online_vllm.eval.zero_shot import (CHECKPOINTS, checkpoint_pairs, compare_systems, letter_token_ids, mcnemar,
                                        online_pass, readiness_predictions, sample_questions, system_metrics)
from online_vllm.models import build_model
from test_train import head_only, remap

# Шаблон чата MMDuet2 (chat_template.json чекпойнта; совпадает с Qwen2.5-VL).
TEMPLATE = (
    "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}"
    "<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in "
    "content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}<|vision_start|>"
    "<|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}<|vision_start|>"
    "<|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n"
    "{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
FRAMES = 6  # три хода по два кадра; кадры 8 x 4 -> 2 токена на кадр


@pytest.fixture(scope="module")
def processor():
    """Настоящий Qwen processor со словарём из нескольких слов; служебные токены 2..5 — как в test_train (remap)."""
    tokens = ["<unk>", "<pad>", "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>",
              "<|im_start|>", "<|im_end|>", "system", "user", "assistant", "NO", "REPLY", "question", "A", "B", '"']
    tokenizer = Tokenizer(WordLevel({t: i for i, t in enumerate(tokens)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>",
                                        eos_token="<|im_end|>", additional_special_tokens=tokens[2:8])
    return Qwen2_5_VLProcessor(
        tokenizer=tokenizer, chat_template=TEMPLATE,
        image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2),
        video_processor=Qwen2VLVideoProcessor(patch_size=2, temporal_patch_size=2, merge_size=2))


class TinyTurns(TurnsBuilder):
    def build(self, *args, **kwargs):
        turns = super().build(*args, **kwargs)
        remap(turns.inputs)
        return turns


def frames(n=FRAMES, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, 8, 4, 3), dtype=np.uint8)


def test_turns_layout(processor):
    turns = TurnsBuilder(processor).build("question", frames())
    ids, tok = turns.inputs["input_ids"][0], processor.tokenizer
    assert len(turns.decisions) == FRAMES // 2
    # «NO REPLY» в кавычках есть и в системном промпте; решения — только в ходах ассистента после кадров.
    assert turns.silence == tok(NO_REPLY, add_special_tokens=False).input_ids
    for d in turns.decisions.tolist():
        assert ids[d].item() == tok.convert_tokens_to_ids("assistant")
        assert ids[d + 1:d + 1 + len(turns.silence)].tolist() == turns.silence
    # Каждый кадр — отдельная картинка: 2 токена на кадр, 4 на ход, кадры хода k идут до решения хода k.
    images = (turns.inputs["mm_token_type_ids"][0] == 1).nonzero()[:, 0]
    assert len(images) == FRAMES * 2 and turns.inputs["image_grid_thw"].shape == (FRAMES, 3)
    for k, d in enumerate(turns.decisions.tolist()):
        assert (images < d).sum() == 4 * (k + 1)
    with pytest.raises(ValueError, match="whole turns"):
        TurnsBuilder(processor).build("question", frames(5))


def test_one_pass_is_online(processor, tiny_base):
    """Решение каждого хода в одном проходе == решение модели, которой подали только кадры до этого хода."""
    model = build_model(head_only(tiny_base), local_files_only=True).eval()
    with torch.no_grad():  # острое внимание, чтобы сравнение заметило лишние кадры или неверные позиции
        for layer in model.qwen.model.language_model.layers:
            layer.self_attn.q_proj.weight.mul_(6)
            layer.self_attn.k_proj.weight.mul_(6)
    builder, letters, video = TinyTurns(processor), letter_token_ids(processor.tokenizer, 2), frames()
    full = online_pass(model, builder, "question", video, letters, "cpu")
    assert full["letters"].shape == (3, 2) and full["silence"].shape == (3,) and full["no_top1"].shape == (3,)
    assert (full["silence"] < 0).all()
    for k in range(3):
        part = online_pass(model, builder, "question", video[:2 * (k + 1)], letters, "cpu")
        np.testing.assert_allclose(part["silence"][-1], full["silence"][k], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(part["letters"][-1], full["letters"][k], rtol=1e-4, atol=1e-5)
    # Последние кадры меняют только последнее решение.
    other = video.copy()
    other[4:] = frames(2, seed=1)
    changed = online_pass(model, builder, "question", other, letters, "cpu")
    np.testing.assert_allclose(changed["silence"][:2], full["silence"][:2], rtol=1e-4, atol=1e-5)
    assert not np.allclose(changed["silence"][2], full["silence"][2])


def record(i, video, start):
    return Record(id=f"q{i}", dataset="toy", split="validation", video=video, fps=25, num_frames=128,
                  question="question", choices=["a", "b"], ask_frame=0, answers=[Interval(start, 127, "a")],
                  meta={"family": "toy", "answer_type": "color"})


def test_checkpoints_and_sample():
    assert checkpoint_pairs(record(0, "v", 39), 64, 0)["target"] == 19   # пара (38, 39) — первая с кадром >= 39
    assert checkpoint_pairs(record(0, "v", 40), 64, 0)["target"] == 20
    assert checkpoint_pairs(record(0, "v", 125), 64, 0) is not None      # target 62, после него ещё пара 63
    assert checkpoint_pairs(record(0, "v", 126), 64, 0) is None          # target 63 — последняя пара
    pairs = [checkpoint_pairs(record(i, "v", 40), 64, 0) for i in range(50)]
    assert all(p["first"] == 0 and 20 < p["mid"] <= 63 and p["end"] == 63 for p in pairs)
    assert len({p["mid"] for p in pairs}) > 5  # mid свой у каждого вопроса
    records = [record(i, f"v{i % 7}", 30 + i) for i in range(28)] + [record(99, "late", 126)]
    picked = sample_questions(records, 5, seed=3)
    assert len({r.video for r in picked}) == 5 and "late" not in {r.video for r in picked}
    shuffled = records[:]
    random.Random(1).shuffle(shuffled)
    assert [r.id for r in sample_questions(shuffled, 5, seed=3)] == [r.id for r in picked]
    with pytest.raises(ValueError, match="Подходящих видео"):
        sample_questions(records, 8, seed=3)


def test_mcnemar():
    assert mcnemar([1, 1, 0, 0], [0, 0, 0, 1]) == {"only_first": 2, "only_second": 1, "p_value": 1.0}
    assert mcnemar([1] * 5 + [0], [0] * 6)["p_value"] == pytest.approx(2 / 32)


def rows(online: bool):
    """Синтетические ответы: до целевой пары модель уверенно ошибается, начиная с неё — уверенно права."""
    out = []
    for i in range(20):
        target = 10 + i
        letters = np.full((64, 2), -5.0)
        letters[:target, 1] = -0.5      # неверный вариант (правильный — 0)
        letters[target:, 0] = -0.5
        row = {"id": f"q{i}", "family": "toy", "answer_type": "color" if i % 2 else "yesno", "choices": 2,
               "correct": 0, "start_frame": 2 * target, "fps": 25, "num_pairs": 64,
               "pairs": {"first": 0, "target": target, "mid": target + 3, "end": 63}, "letters": letters.tolist()}
        if online:
            silence = np.where(np.arange(64) < target, np.log(0.9), np.log(0.1))
            row.update({"silence": silence.tolist(), "no_top1": (np.arange(64) < target).tolist()})
        out.append(row)
    return out


def test_system_metrics_and_predictions():
    m = system_metrics(rows(True))
    assert m["checkpoints"]["first"]["accuracy"] == 0 and m["checkpoints"]["target"]["accuracy"] == 1
    assert m["gain_target_vs_first"]["accuracy"] == 1 and m["gain_target_vs_first"]["only_first"] == 20
    assert m["checkpoints"]["first"]["self_reply_rate"] == 0 and m["checkpoints"]["target"]["self_reply_rate"] == 1
    assert m["checkpoints"]["target"]["self_reply_correct_rate"] == 1
    assert m["checkpoints"]["end"]["p_no_reply_quartiles"][1] == pytest.approx(0.1)
    assert set(m["breakdown"]["answer_type"]) == {"color", "yesno"}
    assert set(m["breakdown"]["start_quartile"]) == {"Q1", "Q2", "Q3", "Q4"}
    offline = system_metrics(rows(False))
    assert "self_reply_rate" not in offline["checkpoints"]["first"]
    comparison = compare_systems({"a": rows(False), "b": rows(True)})["a vs b"]
    assert all(comparison[c]["agreement"] == 1 and comparison[c]["p_value"] == 1 for c in CHECKPOINTS)

    payload = readiness_predictions(rows(True), {"model": "toy", "frames_per_pair": 2, "stride": 1})
    json.dumps(payload, allow_nan=False)
    traces = load_traces(payload)
    assert traces[0].fired(0.5) == traces[0].target_pair
    result = sweep(payload, [0.5])
    assert result["results"]["0.5"]["on_target_rate"] == 1 and result["classification"]["auroc"] == 1
    assert result["results"]["0.5"]["answer_accuracy"] == 1
    with pytest.raises(ValueError, match="не онлайн"):
        readiness_predictions(rows(False), {"frames_per_pair": 2, "stride": 1})
