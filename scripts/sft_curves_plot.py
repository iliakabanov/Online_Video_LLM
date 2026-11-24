"""Online SFT (M_on) figure for the "Training Setup" slide.

Top: AUROC±5 of the native <speak> score of M_on by epoch (validation, classification.auroc_pm5 over pairs
|k - t*| <= 5); the selected epoch is marked with a dashed line.
Bottom: answer accuracy on test vs. the pair k at which the video is cut, relative to t* (answer_correct of the test
predictions; the same 3229 questions for both models). Band: 95% CI, video-level bootstrap (as in
scripts/test_bootstrap.py: test videos resampled with replacement, percentile interval).

    python scripts/sft_curves_plot.py results/lora/sft_curves/listen_sft_10ep.jsonl \
        --test-off results/lora/mlp/attention_mlp_off10/predictions_test_best_tau0.7.json \
        --test-on results/lora/mlp/attention_mlp_on10/predictions_test_best_tau0.6.json \
        --output results/lora/sft_curves/online_sft_curves
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"  # as in scripts/depth_plot.py
OFF = ("$M_{\\mathrm{off}}$", "#2a78d6", "o")
ON = ("$M_{\\mathrm{on}}$", "#eb6834", "s")
VIDEO = re.compile(r"^([a-z]+_\d+)_")  # as in scripts/test_bootstrap.py


def rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=10)


def line(ax, x, values, spec, dy=0, markersize=6, text=None) -> None:
    label, color, marker = spec
    ax.plot(x, values, color=color, linewidth=2, marker=marker, markersize=markersize, markeredgecolor="white",
            markeredgewidth=1.0, solid_capstyle="round", label=label, zorder=3)
    ax.annotate(text or f"{label} {values[-1]:.3f}", (x[-1], values[-1]), xytext=(7, dy),
                textcoords="offset points", color=INK_2, fontsize=10, va="center")


def accuracy_by_offset(path: str, offsets: np.ndarray, order: list[str] | None):
    """Per-video sums and counts of answer_correct at pair k = t* + offset: arrays [videos, offsets]."""
    predictions = {p["record_id"]: p for p in json.loads(Path(path).read_text(encoding="utf-8"))["predictions"]}
    order = order or sorted(predictions)
    if set(order) != set(predictions):
        raise ValueError(f"{path}: other questions than in the first file")
    videos, index = np.unique([VIDEO.match(r).group(1) for r in order], return_inverse=True)
    hits = np.zeros((len(videos), len(offsets)))
    counts = np.zeros_like(hits)
    for record_id, v in zip(order, index):
        p = predictions[record_id]
        for j, k in enumerate(offsets + p["target_pair"]):
            if 0 <= k < len(p["answer_correct"]) and p["answer_correct"][k] is not None:
                hits[v, j] += p["answer_correct"][k]
                counts[v, j] += 1
    return hits, counts, order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("online", help="validation_metrics.jsonl of the online SFT run")
    parser.add_argument("--test-off", required=True, help="test predictions with answer_correct of M_off")
    parser.add_argument("--test-on", required=True, help="test predictions with answer_correct of M_on")
    parser.add_argument("--selected", type=int, default=10, help="selected M_on epoch")
    parser.add_argument("--range", type=int, nargs=2, default=(-10, 20), help="offsets k - t* to plot")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, help="path without extension: writes .png, .pdf and .json")
    args = parser.parse_args()

    on = rows(args.online)
    epochs = [r["epoch"] for r in on]
    auroc = [r["validation"]["classification"]["auroc_pm5"] for r in on]

    offsets = np.arange(args.range[0], args.range[1] + 1)
    hits_off, counts_off, order = accuracy_by_offset(args.test_off, offsets, None)
    hits_on, counts_on, _ = accuracy_by_offset(args.test_on, offsets, order)
    rng = np.random.default_rng(args.seed)
    n_videos = len(hits_off)
    weights = np.stack([np.bincount(rng.integers(0, n_videos, n_videos), minlength=n_videos)
                        for _ in range(args.bootstrap)])  # same video resamples for both models
    curves = {}
    for name, hits, counts in (("M_off", hits_off, counts_off), ("M_on", hits_on, counts_on)):
        boot = (weights @ hits) / (weights @ counts)
        curves[name] = {"accuracy": hits.sum(0) / counts.sum(0), "ci95": np.percentile(boot, [2.5, 97.5], axis=0),
                        "questions": counts.sum(0)}

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(4.6, 4.5), constrained_layout=True)

    line(top, epochs, auroc, ON)
    top.axvline(args.selected, color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
    top.annotate(f"selected: epoch {args.selected}", (args.selected, 0.42), xytext=(-5, 0),
                 textcoords="offset points", ha="right", va="bottom", color=INK_2, fontsize=9)
    top.set_title("Readiness: AUROC±5 of p(<speak>) ↑  (validation)", color=INK, fontsize=11, loc="left")
    top.set_xlabel("epoch", color=INK_2, fontsize=10)
    top.set_xticks(epochs)
    top.set_xlim(-0.4, epochs[-1] + 2.6)  # room for the direct label
    top.set_ylim(0.4, 1.0)

    for (name, spec), dy in zip((("M_off", OFF), ("M_on", ON)), (-8, 8)):
        c = curves[name]
        bottom.fill_between(offsets, *c["ci95"], color=spec[1], alpha=0.18, linewidth=0, zorder=2)
        line(bottom, offsets, c["accuracy"], spec, dy=dy, markersize=0, text=f"{spec[0]} {c['accuracy'][-1]:.3f}")
    bottom.axvline(0, color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
    bottom.annotate("$t^*$: answer visible", (0, 0.52), xytext=(5, 0), textcoords="offset points",
                    ha="left", va="bottom", color=INK_2, fontsize=9)
    bottom.set_title("Answer accuracy vs. answer time ↑  (test)", color=INK, fontsize=11, loc="left")
    bottom.set_xlabel("video cut at pair $k$, relative to $t^*$ ($k - t^*$)", color=INK_2, fontsize=10)
    bottom.set_xlim(offsets[0] - 0.5, offsets[-1] + 7)  # room for the direct labels
    bottom.set_xticks([o for o in offsets if o % 5 == 0])
    bottom.set_ylim(0.5, 1.0)
    for ax in (top, bottom):
        style(ax)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=250)
    summary = {"offsets": offsets.tolist(), "bootstrap": args.bootstrap, "seed": args.seed,
               "videos": n_videos, "test_off": args.test_off, "test_on": args.test_on,
               **{name: {k: np.round(v, 5).tolist() for k, v in c.items()} for name, c in curves.items()}}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
