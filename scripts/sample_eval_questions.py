"""Выборка вопросов для оценки без дообучения (eval/ZERO_SHOT.md): по одному вопросу группы 1 на видео из N случайных
видео, у каждого вопроса после пары, где ответ стал виден, есть ещё хотя бы одна пара. Выборка детерминирована --seed.

python scripts/sample_eval_questions.py --questions processed/clevrer_proactive/val_group1_features.jsonl \
    --videos 500 --output processed/clevrer_proactive/val_group1_features_eval500.jsonl
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import _paths  # noqa: F401  (src/ on the path)
from online_vllm.data.schema import load_jsonl, save_jsonl
from online_vllm.eval.zero_shot import sample_questions

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default="processed/clevrer_proactive/val_group1_features.jsonl",
                        help="jsonl вопросов; относительный путь — от DATA_ROOT")
    parser.add_argument("--videos", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1, help="модель видит каждый k-й кадр")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--output", required=True, help="jsonl выборки; относительный путь — от DATA_ROOT")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    records = sample_questions(load_jsonl(data_root / args.questions), args.videos, args.seed, args.stride)
    output = data_root / args.output
    save_jsonl(records, output)
    stats = {"source": args.questions, "videos": args.videos, "seed": args.seed, "stride": args.stride,
             "family": dict(Counter(r.meta["family"] for r in records).most_common()),
             "answer_type": dict(Counter(r.meta["answer_type"] for r in records).most_common()),
             "choices": dict(sorted(Counter(len(r.choices) for r in records).items())),
             "ids": [r.id for r in records]}
    stats_path = output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(records)} вопросов из {len({r.video for r in records})} видео -> {output}")
    print("семейства:", stats["family"])
    print("типы ответа:", stats["answer_type"], "| вариантов:", stats["choices"])


if __name__ == "__main__":
    main()
