"""Точность ответа Qwen2.5-VL без дообучения на выборке вопросов: модель и размер кадров задаются флагами.

Для каждого вопроса модель видит вопрос с вариантами и видео (целиком или до целевой пары), и ответ берётся из
вероятностей букв вариантов в позиции, с которой ассистент начал бы отвечать (та же проба, что в признаках,
train/FEATURES.md). Признаки и голова не нужны. Выборка вопросов фиксирована --seed, поэтому разные модели и
разрешения сравниваются на одних и тех же вопросах.

python scripts/zero_shot_answers.py --model Qwen/Qwen2.5-VL-3B-Instruct --size 504x336 --n 300 \
    --output results/zero_shot/3b_504x336.json
"""
import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import _paths  # noqa: F401  (src/ on the path)
from online_vllm.data.inputs import LETTERS, InputBuilder, format_prompt
from online_vllm.data.labels import plan, whole_patches
from online_vllm.data.schema import load_jsonl
from online_vllm.data.video import read_every
from online_vllm.eval.answers import choice_probabilities
from online_vllm.models import positions

REPO = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def sample_records(path: Path, n: int, seed: int, stride: int, delay: int, temporal_patch: int):
    """n случайных вопросов группы 1 с размеченной целевой парой; порядок файла не важен."""
    records = [r for r in load_jsonl(path) if plan(r, stride, delay, temporal_patch) is not None]
    random.Random(seed).shuffle(records)
    return records[:n]


@torch.no_grad()
def answer(model, builder, record, frames: np.ndarray, fps: float, letter_ids: torch.Tensor, device) -> dict:
    """Ответ модели одним проходом: вероятности вариантов и то, была бы буква первым токеном при жадной генерации."""
    inputs = builder._encode(format_prompt(record.question, record.choices), frames, fps).to(device)
    logits = model(**inputs, logits_to_keep=1).logits[0, -1].float()
    logprob = logits.log_softmax(-1)
    k = len(record.choices)
    raw = logprob[letter_ids[:k]].cpu().numpy()[None]
    probs = choice_probabilities(raw, k)[0]
    correct = record.choices.index(record.answers[0].answer)
    chosen = int(probs.argmax())
    return {"id": record.id, "choices": k, "correct": correct, "chosen": chosen, "is_correct": chosen == correct,
            "p_correct": float(probs[correct]), "p_answer": float(probs.max()),
            "letter_is_top1": bool(int(logprob.argmax()) == int(letter_ids[chosen])), "tokens": int(inputs["input_ids"].shape[1])}


def summary(rows: list[dict]) -> dict:
    if not rows:
        return {"questions": 0}
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    accuracy = mean("is_correct")
    return {"questions": len(rows), "accuracy": accuracy,
            # биномиальная стандартная ошибка: при n = 300 и точности около 0.5 это около ±2.9 п.п.
            "accuracy_se": float(np.sqrt(accuracy * (1 - accuracy) / len(rows))),
            "chance": float(np.mean([1 / r["choices"] for r in rows])), "p_correct": mean("p_correct"),
            "p_answer": mean("p_answer"), "letter_is_top1": mean("letter_is_top1"),
            "mean_tokens": mean("tokens")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="любая Qwen2.5-VL с Hugging Face")
    parser.add_argument("--size", default="252x168", help="ширина x высота кадров, обе кратны 28")
    parser.add_argument("--stride", type=int, default=1, help="модель видит каждый k-й кадр")
    parser.add_argument("--at", choices=("end", "target", "both"), default="end",
                        help="где спрашивать: на всём видео, на видео до целевой пары или в обоих местах")
    parser.add_argument("--speak-delay", type=int, default=0, help="задержка целевой пары в кадрах модели (для --at target)")
    parser.add_argument("--questions", default="processed/clevrer_proactive/val_group1.jsonl",
                        help="jsonl вопросов; относительный путь — от DATA_ROOT, как в конфигах")
    parser.add_argument("--n", type=int, default=300, help="сколько вопросов")
    parser.add_argument("--seed", type=int, default=0, help="выборка вопросов; одинаковый seed — одни и те же вопросы")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", required=True, help="JSON со сводкой и ответами по вопросам")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    width, height = (int(v) for v in args.size.lower().split("x"))
    if width % 28 or height % 28:
        parser.error("--size: ширина и высота должны быть кратны 28 (патч Qwen 14 x слияние 2)")
    data_root = Path(args.data_root)
    questions = data_root / args.questions  # абсолютный путь в --questions заменяет data_root целиком

    processor = AutoProcessor.from_pretrained(args.model)
    builder = InputBuilder(processor)
    temporal_patch = builder.temporal_patch
    letter_ids = []
    for letter in LETTERS:
        ids = processor.tokenizer(letter, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"Буква {letter} — не один токен у {args.model}")
        letter_ids.append(ids[0])
    letter_ids = torch.tensor(letter_ids)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="sdpa", device_map=args.device).eval()
    positions.install(model.model)  # та же правка позиций кадров, что в models/builder.py

    records = sample_records(questions, args.n, args.seed, args.stride, args.speak_delay, temporal_patch)
    places = ("end", "target") if args.at == "both" else (args.at,)
    rows = {place: [] for place in places}
    started = time.time()
    for i, record in enumerate(records, 1):
        frames = read_every(data_root / record.video, args.stride, (width, height))
        fps = record.fps / args.stride
        for place in places:
            if place == "end":
                n_frames = whole_patches(record, args.stride, temporal_patch) * temporal_patch
            else:
                n_frames = plan(record, args.stride, args.speak_delay, temporal_patch).num_frames
            rows[place].append(answer(model, builder, record, frames[:n_frames], fps, letter_ids, args.device))
        if i % 25 == 0 or i == len(records):
            logger.info("%d / %d вопросов, %.1f с на вопрос; точность сейчас: %s", i, len(records),
                        (time.time() - started) / i,
                        {p: round(summary(r)["accuracy"], 3) for p, r in rows.items()})

    result = {"model": args.model, "size": [width, height], "stride": args.stride, "speak_delay": args.speak_delay,
              "questions_file": str(questions), "seed": args.seed,
              "summary": {place: summary(r) for place, r in rows.items()}, "answers": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for place, s in result["summary"].items():
        print(f"{args.model} {width}x{height}, {place}: точность {s['accuracy']:.3f} ± {s['accuracy_se']:.3f} "
              f"(угадывание {s['chance']:.3f}), P(правильный) {s['p_correct']:.3f}, P(ответ) {s['p_answer']:.3f}, "
              f"буква — первый токен в {s['letter_is_top1']:.0%}, токенов в среднем {s['mean_tokens']:.0f}")


if __name__ == "__main__":
    main()
