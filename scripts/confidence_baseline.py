"""Детектор готовности без обучения: уверенность ответа модели на паре k как оценка готовности p_k.

Ответ на паре k — распределение по вариантам в пробе «ответь сейчас» из признаков (probe_logprob: у офлайн-адаптера —
его проба, у онлайн-модели — ветка <speak>; online_vllm/eval/answers.py answer_trace, те же ответы, что видят головы).
Три оценки, все в [0, 1] и сравнимы между вопросами с разным числом вариантов K:
    maxprob   (max_i p_i − 1/K) / (1 − 1/K)  — наибольшая вероятность варианта, нормированная на её минимум 1/K;
    margin    p_(1) − p_(2)                   — отрыв самого вероятного варианта от второго;
    entropy   1 − H(p) / log K                — уверенность как отсутствие неопределённости.
Модель отвечает на первой паре с оценкой > τ; метрики — те же, что у голов (online_vllm/eval/metrics.py), на той же
половине eval-признаков (validation или test из split.json головы: те же видео и вопросы).

Для каждой оценки <output-root>/<оценка>_conf_<suffix>/ (с --part test — <...>_test/): results.json (классификация,
метрики при порогах eval.thresholds, кривая «задержка — точность», лучшие τ, задержка при ограничениях) и
predictions_<part>.json. Validation попадает в сводку experiment_launcher/offline_10ep/summary.py рядом с головами.

python scripts/confidence_baseline.py --features data/features/group1_half_s1_252x168/val_offline_10ep_best_videos1000 \\
    --split checkpoints/lora/mlp/attention_mlp_off10/split.json --suffix off10
"""
import argparse
import json
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (src/ on the path)
from online_vllm.eval.answers import answer_trace
from online_vllm.train.config import TrainConfig
from online_vllm.train.features import FeatureSet
from uhead.data import labelled
from uhead.train import THRESHOLD_FREE_KEYS, _readiness, _write_predictions

REPO = Path(__file__).resolve().parents[1]
SCORES = ("maxprob", "margin", "entropy")


def confidence_scores(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Оценки уверенности по распределению ответа [P, K] на каждой паре (K >= 2)."""
    p = np.asarray(probabilities, dtype=np.float64)
    k = p.shape[1]
    top = -np.sort(-p, 1)
    entropy = -(p * np.log(np.clip(p, 1e-30, None))).sum(1) / np.log(k)
    scores = {"maxprob": (top[:, 0] - 1 / k) / (1 - 1 / k), "margin": top[:, 0] - top[:, 1], "entropy": 1 - entropy}
    return {name: np.clip(value, 0.0, 1.0) for name, value in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, help="каталог eval-признаков (extract_features.py --split eval)")
    parser.add_argument("--split", required=True, help="split.json головы на этих признаках: её validation и test")
    parser.add_argument("--part", choices=("validation", "test"), default="validation")
    parser.add_argument("--train", default=str(REPO / "configs/train/group1.yaml"),
                        help="разметка (speak_delay, frame_stride) и пороги eval.thresholds, как у голов")
    parser.add_argument("--output-root", default=str(REPO / "results/lora/confidence"))
    parser.add_argument("--suffix", required=True, help="как у голов на тех же признаках: off10, on10")
    args = parser.parse_args()

    config = TrainConfig.load(args.train)
    fs = FeatureSet(Path(args.features))
    fs.check(config.video)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    videos = set(split[f"{args.part}_videos"])
    items, _ = labelled(fs, config, "hard", 5, 0.65)
    items = [item for item in items if fs.records[item.index].video in videos]
    if len(items) != split[f"{args.part}_questions"]:
        raise ValueError(f"{len(items)} вопросов {args.part}, а split.json ожидает {split[f'{args.part}_questions']}: "
                         "признаки или разметка не те, что у головы")

    rows = {name: [] for name in SCORES}
    for item in items:
        record, n = fs.records[item.index], len(item.target)
        trace = answer_trace(fs, item.index, n)
        for name, score in confidence_scores(trace.probabilities).items():
            rows[name].append({"record_id": record.id, "probabilities": score, "target_pair": item.target_pair,
                               "visible_pair": item.visible_pair, "fps": record.fps,
                               "answer_correct": trace.is_correct})
    for name in SCORES:
        # test — в отдельный каталог: сводка summary.py читает только validation.
        out = Path(args.output_root) / f"{name}_conf_{args.suffix}{'' if args.part == 'validation' else '_test'}"
        out.mkdir(parents=True, exist_ok=True)
        result = _readiness(rows[name], config.eval.thresholds, fs.meta, model=f"confidence_{name}", tags="hard")
        summary = {"model": f"confidence_{name}", "score": name, "features": str(args.features),
                   "split": str(args.split), "part": args.part, "best_epoch": None, "selection": None,
                   args.part: {"questions": len(items), "videos": len(videos),
                               "classification": result["classification"], "thresholds": result["thresholds"],
                               **{key: result[key] for key in THRESHOLD_FREE_KEYS}}}
        (out / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
        _write_predictions(out / f"predictions_{args.part}.json", rows[name], fs.meta, f"confidence_{name}", "hard",
                           args.part)
        classification = result["classification"]
        best = result["best_thresholds"]["on_time_0_4"]
        print(f"{name:8s} {args.part}: AUROC ±2 {classification['auroc_pm2']:.4f}, on_time_0_4 {best['on_time_0_4']:.4f} "
              f"(τ {best['tau']:.3f}) -> {out}")


if __name__ == "__main__":
    main()
