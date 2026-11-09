#!/usr/bin/env python3
"""Diagnose whether a cached-feature readiness detector is question-conditioned.

The script never loads Qwen or writes under a checkpoint directory.  It runs the
binary MLP detector independently for every eligible validation question, then
compares readiness curves belonging to the same source video.

Example:
    py tests/analyze_multi_question_readiness.py \
        --checkpoint checkpoints/mlp/01_lm \
        --features /path/to/validation/features --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from online_vllm.data.labels import plan
from online_vllm.train.config import TrainConfig
from online_vllm.train.features import FeatureSet
from online_vllm.eval.metrics import uhead_probabilities
from uhead.data import feature_rows
from uhead.train import head_config, load_checkpoint


THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
TOLERANCE_PATCHES = 1


def native(value: Any) -> Any:
    """Convert NumPy/PyTorch values recursively to strict JSON-native values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [native(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(x) for x in value]
    return value


def first_fire(probabilities: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(probabilities >= threshold)
    return int(indices[0]) if len(indices) else None


def firing_outcome(fire_pair: int | None, target_pair: int) -> str:
    """Four-way timing outcome relative to the readiness target itself."""
    if fire_pair is None:
        return "missed"
    if fire_pair < target_pair:
        return "premature"
    if fire_pair == target_pair:
        return "on_target"
    return "late"


def pearson(left: list[float], right: list[float]) -> float | None:
    """Correlation over the shared unpadded prefix; None for degenerate curves."""
    n = min(len(left), len(right))
    if n < 2:
        return None
    x, y = np.asarray(left[:n], dtype=float), np.asarray(right[:n], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def shift_summary(values: list[int]) -> dict[str, float | None]:
    """Population summaries, or JSON nulls when no pair has two predictions."""
    if not values:
        return {"mean": None, "median": None, "std": None}
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "median": float(np.median(array)), "std": float(array.std())}


def select_video_groups(groups: dict[str, list[dict]], max_videos: int, seed: int) -> list[tuple[str, list[dict]]]:
    """Seeded selection, prioritising videos whose questions have distinct targets."""
    eligible = [(video, rows) for video, rows in groups.items() if len(rows) >= 2]
    varied = [(video, rows) for video, rows in eligible if len({row["target_pair"] for row in rows}) >= 2]
    tied = [(video, rows) for video, rows in eligible if len({row["target_pair"] for row in rows}) == 1]
    rng = random.Random(seed)
    rng.shuffle(varied)
    rng.shuffle(tied)
    return (varied + tied)[:max_videos]


def question_statistics(record, target_pair: int, probabilities: np.ndarray) -> dict:
    target_frame = int(record.answers[0].start_frame)
    target_seconds = float(target_frame / record.fps) if record.fps else None
    fires = {str(tau): first_fire(probabilities, tau) for tau in THRESHOLDS}
    return {
        "question_id": record.id,
        "question_text": record.question,
        "target_pair": int(target_pair),
        # This is the original CLEVRER event/answer-availability frame; target_pair
        # additionally includes the configured speak delay and patch discretisation.
        "target_frame": target_frame,
        "target_seconds": target_seconds,
        "num_pairs": int(len(probabilities)),
        "probabilities": [float(p) for p in probabilities],
        "predicted_peak_pair": int(np.argmax(probabilities)),
        "predicted_peak_probability": float(np.max(probabilities)),
        "first_fire_pair": fires,
        "timing_error_pairs": {key: None if fire is None else int(fire - target_pair) for key, fire in fires.items()},
        "outcome": {key: firing_outcome(fire, target_pair) for key, fire in fires.items()},
    }


def cross_question_statistics(questions: list[dict]) -> tuple[dict, dict[str, int]]:
    """Return per-video pair comparisons and aggregate-rate numerators/denominators.

    A same-fire match is |first_fire_0.5(q1)-first_fire_0.5(q2)| <= 1,
    evaluated only for questions with different target_pair values and two observed
    fires.  A first-target-bias match is a later-target question firing within one
    patch of the earlier-target question's target_pair; it excludes missed later
    predictions.  Thus the two rates have explicit, different denominators.
    """
    pairs, counts = [], {"eligible_target_pairs": 0, "same_fire_compared": 0, "same_fire_matches": 0,
                         "same_fire_missing": 0, "first_target_bias_compared": 0,
                         "first_target_bias_matches": 0, "first_target_bias_missing": 0,
                         "wrong_event_preference_compared": 0, "wrong_event_preference_matches": 0,
                         "wrong_event_preference_missing": 0, "shift_compared": 0, "shift_missing": 0}
    for left, right in combinations(questions, 2):
        lf, rf = left["first_fire_pair"]["0.5"], right["first_fire_pair"]["0.5"]
        different_targets = left["target_pair"] != right["target_pair"]
        pair = {
            "question_ids": [left["question_id"], right["question_id"]],
            "target_pair_difference": int(right["target_pair"] - left["target_pair"]),
            "target_frame_difference": int(right["target_frame"] - left["target_frame"]),
            "target_seconds_difference": None if left["target_seconds"] is None or right["target_seconds"] is None
            else float(right["target_seconds"] - left["target_seconds"]),
            "first_fire_pair_difference_0.5": None if lf is None or rf is None else int(rf - lf),
            "pearson_correlation": pearson(left["probabilities"], right["probabilities"]),
            "different_target_pairs": different_targets,
            "same_fire_within_1_patch": None,
            "later_question_fires_near_earliest_target": None,
            "wrong_event_preference": None,
            "target_shift_pairs": None,
            "prediction_shift_pairs": None,
        }
        if different_targets:
            counts["eligible_target_pairs"] += 1
            if lf is None or rf is None:
                counts["same_fire_missing"] += 1
            else:
                matched = abs(lf - rf) <= TOLERANCE_PATCHES
                pair["same_fire_within_1_patch"] = matched
                counts["same_fire_compared"] += 1
                counts["same_fire_matches"] += int(matched)
            earlier, later = (left, right) if left["target_pair"] < right["target_pair"] else (right, left)
            later_fire = later["first_fire_pair"]["0.5"]
            pair["target_shift_pairs"] = int(later["target_pair"] - earlier["target_pair"])
            if earlier["first_fire_pair"]["0.5"] is None or later_fire is None:
                counts["shift_missing"] += 1
            else:
                pair["prediction_shift_pairs"] = int(later_fire - earlier["first_fire_pair"]["0.5"])
                counts["shift_compared"] += 1
            if later_fire is None:
                counts["first_target_bias_missing"] += 1
                counts["wrong_event_preference_missing"] += 1
            else:
                biased = abs(later_fire - earlier["target_pair"]) <= TOLERANCE_PATCHES
                pair["later_question_fires_near_earliest_target"] = biased
                counts["first_target_bias_compared"] += 1
                counts["first_target_bias_matches"] += int(biased)
                wrong_event = abs(later_fire - earlier["target_pair"]) < abs(later_fire - later["target_pair"])
                pair["wrong_event_preference"] = wrong_event
                counts["wrong_event_preference_compared"] += 1
                counts["wrong_event_preference_matches"] += int(wrong_event)
        pairs.append(pair)
    return {"matching_rule": "same fire means absolute first-fire difference at tau=0.5 is <= 1 patch",
            "first_target_bias_rule": "a later-target question fires within +/-1 patch of the earlier target_pair",
            "missing_predictions_excluded_from_timing_comparisons": True,
            "pairs": pairs, **counts}, counts


def resolve_checkpoint(value: Path) -> Path:
    """Resolve an exact run dir, or the canonical 01_lm baseline below checkpoints/mlp."""
    value = value.resolve()
    if value.is_file():
        return value.parent
    if (value / "best.pt").exists() and (value / "uhead.yaml").exists():
        return value
    baseline = value / "01_lm"
    if (baseline / "best.pt").exists() and (baseline / "uhead.yaml").exists():
        return baseline
    candidates = sorted(path.parent for path in value.rglob("best.pt") if (path.parent / "uhead.yaml").exists())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No binary MLP run (best.pt and uhead.yaml) under {value}")
    raise ValueError("Several detector runs found; pass --checkpoint for one run: " + ", ".join(map(str, candidates)))


@torch.inference_mode()
def analyze(checkpoint: Path, features: Path, output: Path, max_videos: int, seed: int,
            device: str, overwrite: bool) -> dict:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; rerun with --overwrite")
    checkpoint = resolve_checkpoint(checkpoint)
    config = TrainConfig.load(checkpoint / "train.yaml")
    fs = FeatureSet(features)
    fs.check(config.video)
    model = load_checkpoint(checkpoint, device).eval()
    cfg = head_config(model)
    if cfg.temporal_detector != "mlp":
        raise ValueError(f"Expected an MLP StateHead checkpoint, got {cfg.temporal_detector}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for index, record in enumerate(fs.records):
        state_plan = plan(record, config.video.frame_stride, config.labels.speak_delay, fs.meta["temporal_patch"])
        if state_plan is None:
            continue
        target_pair = len(state_plan.labels) - 1
        if target_pair >= fs.patches[index]:
            raise ValueError(f"{record.id}: target_pair is absent from cached features")
        grouped[record.video].append({"index": index, "record": record, "target_pair": target_pair})
    selected = select_video_groups(grouped, max_videos, seed)

    videos, total_counts = [], defaultdict(int)
    for video_id, rows in selected:
        questions = []
        for row in rows:  # deliberately independent inference per question
            index, record, target_pair = row["index"], row["record"], row["target_pair"]
            values = feature_rows(fs, index, fs.patches[index], cfg.groups)
            inputs = {name: torch.from_numpy(value)[None].to(device) for name, value in values.items()}
            probabilities = uhead_probabilities(model(inputs))[0]
            questions.append(question_statistics(record, target_pair, probabilities))
        statistics, counts = cross_question_statistics(questions)
        for key, value in counts.items():
            total_counts[key] += value
        videos.append({"video_id": video_id, "questions": questions, "cross_question_statistics": statistics})

    same_denominator = total_counts["same_fire_compared"]
    bias_denominator = total_counts["first_target_bias_compared"]
    wrong_event_denominator = total_counts["wrong_event_preference_compared"]
    target_shifts, prediction_shifts = [], []
    for video in videos:
        for pair in video["cross_question_statistics"]["pairs"]:
            if pair["prediction_shift_pairs"] is not None:
                target_shifts.append(pair["target_shift_pairs"])
                prediction_shifts.append(pair["prediction_shift_pairs"])
    payload = {
        "metadata": {"checkpoint": str(checkpoint), "split": "validation", "features": str(features),
                     "num_videos": len(videos), "num_questions": sum(len(v["questions"]) for v in videos),
                     "thresholds": list(THRESHOLDS), "seed": seed, "tolerance_patches": TOLERANCE_PATCHES,
                     "temporal_patch": int(fs.meta["temporal_patch"]), "frame_stride": int(fs.meta["frame_stride"])},
        "summary": {"same_fire_rate": None if not same_denominator else total_counts["same_fire_matches"] / same_denominator,
                    "first_target_bias_rate": None if not bias_denominator else total_counts["first_target_bias_matches"] / bias_denominator,
                    "wrong_event_preference_rate": None if not wrong_event_denominator
                    else total_counts["wrong_event_preference_matches"] / wrong_event_denominator,
                    "target_shift_pairs": shift_summary(target_shifts),
                    "prediction_shift_pairs": shift_summary(prediction_shifts),
                    **dict(total_counts)},
        "videos": videos,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(native(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints/mlp")
    parser.add_argument("--features", type=Path, required=True, help="Completed FeatureSet for CLEVRER validation")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "tests/results/multi_question_readiness.json")
    parser.add_argument("--max-videos", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_videos < 1:
        raise ValueError("--max-videos must be positive")
    payload = analyze(args.checkpoint, args.features, args.output, args.max_videos, args.seed, args.device, args.overwrite)
    summary, metadata = payload["summary"], payload["metadata"]
    print(f"selected videos={metadata['num_videos']}, questions={metadata['num_questions']}, "
          f"analyzed target-different pairs={summary['eligible_target_pairs']}, output={args.output}")


if __name__ == "__main__":
    main()
