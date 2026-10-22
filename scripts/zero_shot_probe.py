"""Ответы одной системы без дообучения на всех парах видео каждого вопроса выборки (eval/ZERO_SHOT.md).

Системы: qwen_offline, mmduet2_offline (вопрос с вариантами, затем видео одним входом; проба после каждой пары) и
mmduet2_online (формат MMDuet2, data/turns.py: ход — пара кадров, в ходах ассистента «NO REPLY»; на каждом ходе
log P букв и P(NO REPLY)). Один проход на вопрос. Строки дописываются в answers.jsonl по мере счёта, поэтому
прерванный запуск продолжается той же командой.

python scripts/zero_shot_probe.py --system qwen_offline --device cuda:0
python scripts/zero_shot_probe.py --system mmduet2_online --device cuda:1
python scripts/zero_shot_probe.py --system mmduet2_online --ids validation_10000_clevrer_query_material_1 \
    --questions processed/clevrer_proactive/val_group1.jsonl --output-dir results/zero_shot/check   # один вопрос
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

import _paths  # noqa: F401  (src/ on the path)
from online_vllm.data.inputs import InputBuilder, format_prompt
from online_vllm.data.labels import whole_patches
from online_vllm.data.schema import load_jsonl
from online_vllm.data.turns import TurnsBuilder
from online_vllm.data.video import read_every
from online_vllm.eval.zero_shot import SYSTEMS, checkpoint_pairs, letter_token_ids, offline_pass, online_pass
from online_vllm.models import BackboneConfig, LoRAConfig, ModelConfig, build_model

REPO = Path(__file__).resolve().parents[1]
FRAMES_PER_PAIR = 2  # временной патч Qwen2.5-VL офлайн и ход из двух кадров онлайн: решения на одних и тех же парах
logger = logging.getLogger(__name__)


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--model", help="по умолчанию — модель системы (eval/zero_shot.py SYSTEMS)")
    parser.add_argument("--revision", help="коммит модели на Hub; выбранный записывается в meta.json")
    parser.add_argument("--processor", help="откуда взять processor, по умолчанию — из --model")
    parser.add_argument("--questions", default="processed/clevrer_proactive/val_group1_features_eval500.jsonl",
                        help="jsonl вопросов (scripts/sample_eval_questions.py); относительный путь — от DATA_ROOT")
    parser.add_argument("--ids", nargs="+", help="только эти вопросы")
    parser.add_argument("--limit", type=int, help="только первые N вопросов (пилот)")
    parser.add_argument("--size", default="252x168", help="ширина x высота кадров, обе кратны 28")
    parser.add_argument("--stride", type=int, default=1, help="модель видит каждый k-й кадр")
    parser.add_argument("--seed", type=int, default=0, help="пара замера mid; тот же, что у выборки")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", help="по умолчанию results/zero_shot/<system>")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    width, height = (int(v) for v in args.size.lower().split("x"))
    if width % 28 or height % 28:
        parser.error("--size: ширина и высота должны быть кратны 28 (патч Qwen 14 x слияние 2)")
    model_id = args.model or SYSTEMS[args.system]
    data_root = Path(args.data_root)
    out = Path(args.output_dir or REPO / "results" / "zero_shot" / args.system)
    out.mkdir(parents=True, exist_ok=True)
    answers_path = out / "answers.jsonl"

    records = list(load_jsonl(data_root / args.questions))
    if args.ids:
        wanted = set(args.ids)
        records = [r for r in records if r.id in wanted]
        if len(records) != len(wanted):
            parser.error(f"не найдены вопросы: {sorted(wanted - {r.id for r in records})}")
    records = records[:args.limit]
    done = done_ids(answers_path)
    todo = [r for r in records if r.id not in done]
    logger.info("%s: %d вопросов, %d уже посчитано", args.system, len(records), len(records) - len(todo))

    processor = AutoProcessor.from_pretrained(args.processor or model_id, revision=None if args.processor else
                                              args.revision, local_files_only=args.local_files_only)
    online = args.system.endswith("online")
    builder = TurnsBuilder(processor, FRAMES_PER_PAIR) if online else InputBuilder(processor)
    letters = letter_token_ids(processor.tokenizer)
    config = ModelConfig(backbone=BackboneConfig(model_id=model_id, revision=args.revision, dtype=args.dtype),
                         lora=LoRAConfig(enabled=False))
    model = build_model(config, local_files_only=args.local_files_only, device=args.device).eval()
    meta = {"system": args.system, "model": model_id, "revision": getattr(model.qwen.config, "_commit_hash", None),
            "processor": args.processor or model_id, "dtype": args.dtype, "size": [width, height],
            "stride": args.stride, "frames_per_pair": FRAMES_PER_PAIR, "seed": args.seed,
            "questions_file": args.questions, "questions": len(records)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    started = time.time()
    with open(answers_path, "a", encoding="utf-8") as f:
        for i, record in enumerate(todo, 1):
            pairs_total = whole_patches(record, args.stride, FRAMES_PER_PAIR)
            frames = read_every(data_root / record.video, args.stride, (width, height))[:pairs_total * FRAMES_PER_PAIR]
            if len(frames) != pairs_total * FRAMES_PER_PAIR:
                raise ValueError(f"{record.id}: декодировано {len(frames)} кадров, нужно {pairs_total * FRAMES_PER_PAIR}")
            pairs = checkpoint_pairs(record, pairs_total, args.seed, args.stride, FRAMES_PER_PAIR)
            if pairs is None:
                raise ValueError(f"{record.id}: нет пары после появления ответа (выборка — scripts/sample_eval_questions.py)")
            k = len(record.choices)
            prompt = format_prompt(record.question, record.choices)
            if online:
                result = online_pass(model, builder, prompt, frames, letters[:k], args.device)
            else:
                result = offline_pass(model, builder, prompt, frames, record.fps / args.stride, letters[:k],
                                      args.device)
            row = {"id": record.id, "video": record.video, "family": record.meta.get("family"),
                   "answer_type": record.meta.get("answer_type"), "choices": k,
                   "correct": record.choices.index(record.answers[0].answer),
                   "start_frame": record.answers[0].start_frame, "fps": record.fps, "num_pairs": pairs_total,
                   "pairs": pairs, "letters": np.round(result["letters"], 5).tolist()}
            if online:
                row["silence"] = np.round(result["silence"], 5).tolist()
                row["no_top1"] = result["no_top1"].tolist()
            if len(row["letters"]) != pairs_total:
                raise RuntimeError(f"{record.id}: ответов {len(row['letters'])}, пар {pairs_total}")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                logger.info("%d / %d, %.2f с на вопрос", i, len(todo), (time.time() - started) / i)
    print(f"{args.system}: {len(done_ids(answers_path))} вопросов в {answers_path}")


if __name__ == "__main__":
    main()
