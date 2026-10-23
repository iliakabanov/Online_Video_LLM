"""Онлайн-ответы обученной головы готовности на вопросы одного видео: когда голова сработала и что ответила Qwen.

Вопросы задаются id записей (по умолчанию два вопроса к первому видео validation, video_10000). Скрипт:
1. извлекает признаки этих вопросов той же моделью, что признаки обучения головы (база с адаптером LoRA из meta.json
   признаков train головы, или --adapter / --no-adapter), с настройками кадров и слоёв из train.yaml головы;
2. считает голову (best.pt) на каждой паре кадров: P(готов);
3. отвечает на первой паре, где P > τ (τ — --threshold, иначе из results/.../selection.json запуска, иначе 0.5):
   ответ — вариант с наибольшей вероятностью в пробе «ответь сейчас» на этой паре. Если голова не сработала — ответ
   на последней паре видео (как в нашей метрике).

Пишет JSON: по каждому вопросу момент срабатывания (пара, последний кадр пары, секунды), данный ответ, верен ли он,
момент появления ответа в видео, а также P(готов) и ответ модели на каждой паре.

python scripts/answer_video.py --head checkpoints/lora/mlp/04_lm_attention_mlp_hard --device cuda \
    --output outputs/presentation/video_10000_answers.json
"""
import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import torch

import _paths  # noqa: F401  (src/ on the path)
from lora import load_lora_adapter
from online_vllm.data.inputs import InputBuilder
from online_vllm.eval.answers import answer_trace
from online_vllm.models import ModelConfig, build_model
from online_vllm.train.config import TrainConfig
from online_vllm.train.features import FeatureSet, extract
from uhead.data import ReadinessDataset, labelled
from uhead.train import head_config, load_checkpoint, predict

REPO = Path(__file__).resolve().parents[1]
DEFAULT_IDS = ["validation_10000_clevrer_query_shape_0", "validation_10000_first_hit_6"]


def adapter_of(train_config: TrainConfig, data_root: Path) -> str | None:
    """Адаптер, которым сняты признаки обучения головы (meta.json), или None для исходной модели."""
    meta = json.loads((data_root / train_config.features.train / "meta.json").read_text(encoding="utf-8"))
    return meta["adapter"]["path"] if meta.get("adapter") else None


def default_threshold(head: Path) -> float:
    """τ, выбранный на validation (experiment_launcher/evaluate_test.sh): results/... рядом с checkpoints/...."""
    parts = head.resolve().parts
    if "checkpoints" in parts:
        i = len(parts) - 1 - parts[::-1].index("checkpoints")
        selection = Path(*parts[:i], "results", *parts[i + 1:]) / "selection.json"
        if selection.exists():
            return float(json.loads(selection.read_text(encoding="utf-8"))["threshold"])
    return 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", required=True, help="каталог весов головы (uhead.train --output): best.pt, train.yaml")
    parser.add_argument("--weights", default="best.pt")
    parser.add_argument("--ids", nargs="+", default=DEFAULT_IDS, help="id вопросов; по умолчанию два вопроса video_10000")
    parser.add_argument("--records", default="processed/clevrer_proactive/val_group1.jsonl",
                        help="записи вопросов относительно --data-root")
    parser.add_argument("--threshold", type=float, help="τ; по умолчанию из selection.json запуска, иначе 0.5")
    parser.add_argument("--adapter", help="адаптер LoRA вместо того, что записан в meta.json признаков головы")
    parser.add_argument("--no-adapter", action="store_true", help="исходная Qwen, без LoRA")
    parser.add_argument("--model", default="configs/model/qwen_3b_head.yaml", help="модель без адаптера")
    parser.add_argument("--output", required=True, help="JSON с ответами; рядом каталог <output>_features")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    head_dir, data_root, output = Path(args.head), Path(args.data_root), Path(args.output)
    train_config = TrainConfig.load(head_dir / "train.yaml")
    threshold = args.threshold if args.threshold is not None else default_threshold(head_dir)
    if not 0 < threshold < 1:
        parser.error("τ должен лежать в (0, 1)")

    # Записи вопросов во временный файл: extract берёт все вопросы файла.
    wanted, records = set(args.ids), {}
    with open(data_root / args.records, encoding="utf-8") as f:
        for line in f:
            if (record := json.loads(line))["id"] in wanted:
                records[record["id"]] = line
    missing = [i for i in args.ids if i not in records]
    if missing:
        parser.error(f"нет записей {missing} в {data_root / args.records}")
    features_dir = output.with_name(f"{output.stem}_features")
    shutil.rmtree(features_dir, ignore_errors=True)  # extract продолжает прерванный запуск: начинаем с чистого
    features_dir.mkdir(parents=True)
    records_path = features_dir.parent / f"{output.stem}_records.jsonl"
    records_path.write_text("".join(records[i] for i in args.ids), encoding="utf-8")

    adapter = None if args.no_adapter else (args.adapter or adapter_of(train_config, data_root))
    model_config = ModelConfig.load(Path(adapter).parent / "model.yaml" if adapter else args.model)
    if args.device == "cpu":
        model_config.backbone.dtype = "float32"
    model = build_model(model_config, local_files_only=args.local_files_only, device=args.device)
    if adapter:
        load_lora_adapter(model, Path(adapter))
    logging.info("модель: %s", f"LoRA {adapter}" if adapter else "исходная Qwen")
    builder = InputBuilder.from_config(model.config, local_files_only=args.local_files_only)
    extract(model, builder, records_path, data_root, train_config.video, train_config.features, features_dir,
            args.device, num_workers=0, adapter={"path": str(adapter)} if adapter else None)
    del model
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()

    fs = FeatureSet(features_dir)
    head = load_checkpoint(head_dir, args.device, args.weights).eval()
    tags = head.tag_settings
    items, _ = labelled(fs, train_config, tags["tags"], tags["step_window"], tags["sigmoid_slope"],
                        tags.get("gaussian_sigma", 2.0))
    _, rows = predict(head, ReadinessDataset(fs, items, head_config(head).groups), train_config, args.device)

    patch, stride = fs.meta["temporal_patch"], fs.meta["frame_stride"]
    last_frame = lambda pair: (pair * patch + patch - 1) * stride  # noqa: E731 — решение принимается после пары
    answers = []
    for row, item in zip(rows, items):
        record = fs.records[item.index]
        trace = answer_trace(fs, item.index, len(row["probabilities"]))
        above = [p for p, value in enumerate(row["probabilities"]) if value > threshold]
        fired = above[0] if above else len(row["probabilities"]) - 1
        chosen = record.choices[int(trace.chosen[fired])]
        answers.append({
            "id": record.id, "question": record.question, "choices": record.choices,
            "true_answer": record.answers[0].answer, "answer_visible_frame": record.answers[0].start_frame,
            "target_pair": item.target_pair, "fired": bool(above), "answer_pair": fired,
            "answer_frame": last_frame(fired), "answer_seconds": last_frame(fired) / record.fps,
            "answer": chosen, "correct": chosen == record.answers[0].answer,
            "delay_frames": last_frame(fired) - last_frame(item.target_pair),
            "per_pair": [{"pair": p, "frame": last_frame(p), "p_ready": float(value),
                          "answer": record.choices[int(trace.chosen[p])],
                          "answer_probability": float(trace.probabilities[p].max())}
                         for p, value in enumerate(row["probabilities"])]})
        print(f"{record.question}\n  ответ «{chosen}» ({'верно' if answers[-1]['correct'] else 'неверно'}, "
              f"правильный «{record.answers[0].answer}») на кадре {last_frame(fired)} "
              f"({last_frame(fired) / record.fps:.2f} с){'' if above else ', голова не сработала: конец видео'}; "
              f"ответ виден с кадра {record.answers[0].start_frame}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"head": str(head_dir / args.weights), "threshold": threshold,
                                  "adapter": adapter, "questions": answers}, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"τ = {threshold} -> {output}")


if __name__ == "__main__":
    main()
