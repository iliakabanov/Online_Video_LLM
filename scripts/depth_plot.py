"""Test metrics of the readiness detector on the first k language-model layers (experiment_launcher/depth/run.sh) with
95% bootstrap intervals (scripts/test_bootstrap.py), for the offline and the online model.

Systems in the bootstrap JSON are named "<off10|on10> k=<k>". Latency is converted from frame pairs to frames
(1 pair = 2 frames), as in the report tables.

python scripts/depth_plot.py results/lora/depth/lm_attention_entropy_mlp_test_bootstrap.json \\
    --output results/lora/depth/lm_attention_entropy_mlp_test_ci
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"  # as in online_vllm/eval/answers.py
MODELS = {"off10": ("$M_{\\mathrm{off}}$", "#2a78d6", "o"), "on10": ("$M_{\\mathrm{on}}$", "#eb6834", "s")}
FRAMES_PER_PAIR = 2
# key, title, scale, y label
PANELS = (("auroc", "AUROC ↑", 1, None),
          ("auroc_pm5", "AUROC±5 ↑", 1, None),
          ("on_time_0_4", "On-time ↑", 100, "%"),
          ("latency_R5_A90", "Latency @ A ≥ 90%, P ≤ 5% ↓", FRAMES_PER_PAIR, "frames"),
          ("latency_R10_A90", "Latency @ A ≥ 90%, P ≤ 10% ↓", FRAMES_PER_PAIR, "frames"))
NAME = re.compile(r"^(off10|on10) k=(\d+)$")


def series(systems: dict) -> dict:
    """model -> sorted [(k, entry)]."""
    result = {}
    for name, entry in systems.items():
        match = NAME.match(name)
        if match:
            result.setdefault(match.group(1), []).append((int(match.group(2)), entry))
    return {model: sorted(rows, key=lambda row: row[0]) for model, rows in result.items()}


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bootstrap", help="JSON of scripts/test_bootstrap.py")
    parser.add_argument("--output", required=True, help="path without extension: writes .png and .pdf")
    args = parser.parse_args()

    data = json.loads(Path(args.bootstrap).read_text(encoding="utf-8"))
    by_model = series(data["systems"])
    fig = plt.figure(figsize=(13, 7.4), constrained_layout=True)
    grid = GridSpec(2, 6, figure=fig)
    axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]), fig.add_subplot(grid[0, 4:6]),
            fig.add_subplot(grid[1, 1:3]), fig.add_subplot(grid[1, 3:5])]
    for ax, (key, title, scale, unit) in zip(axes, PANELS):
        for model, rows in by_model.items():
            label, color, marker = MODELS[model]
            ks = [k for k, _ in rows]
            values = [entry[key]["value"] * scale for _, entry in rows]
            low = [entry[key]["ci95"][0] * scale for _, entry in rows]
            high = [entry[key]["ci95"][1] * scale for _, entry in rows]
            ax.fill_between(ks, low, high, color=color, alpha=0.15, linewidth=0, zorder=2)
            ax.plot(ks, values, color=color, linewidth=2, marker=marker, markersize=6, markeredgecolor="white",
                    markeredgewidth=1.5, solid_capstyle="round", label=label, zorder=3)
            ax.annotate(label, (ks[-1], values[-1]), xytext=(6, 0), textcoords="offset points", color=INK_2,
                        fontsize=9, va="center")
        ks = sorted({k for rows in by_model.values() for k, _ in rows})
        ax.set_xticks(ks)
        ax.set_xlim(ks[0] - 1, ks[-1] + 4)  # room for the direct labels
        ax.set_xlabel("k: first k of 36 layers", color=INK_2, fontsize=9)
        if unit:
            ax.set_ylabel(unit, color=INK_2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=10, labelcolor=INK_2, ncol=2)
    fig.suptitle(f"MLP on lm + attention + entropy from the first k layers; test, {data['questions']} questions; "
                 f"shaded: 95% video-level bootstrap CI", color=INK_2, fontsize=10, x=0.01, ha="left")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=150, facecolor="white")
    plt.close(fig)
    print(f"-> {output.with_suffix('.png')}, {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
