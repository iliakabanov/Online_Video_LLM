"""Answer accuracy of a LoRA SFT run along the video: by the shift from the target pair (the pair where the answer
becomes visible) and by the pair from the start of the video, for several adapters on the same questions.

The answer at pair p is the option letter the model rates highest as if the video ended after pair p, the probe of
the validation (one pass per question, then the template tail after every pair). Levels and curves are those of
eval/answers.py (chance, first pair, the pair before the answer appears, the target pair, the end of the video).

By default on the test half of the run (split.json): videos that neither training nor the choice of the best epoch
saw, so the curve of the best adapter is not flattered by its selection.

python scripts/lora_answer_curves.py --run outputs/lora_200_fast --adapters none best [--part test|validation]
"""
import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

import _paths  # noqa: F401,E402  (src/ on the path)
from lora import load_lora_adapter  # noqa: E402
from lora.lora import answer_traces, clevrer_lora_part_provider  # noqa: E402
from lora.vision_cache import VisionCache, build_vision_cache  # noqa: E402
from online_vllm.data.inputs import InputBuilder  # noqa: E402
from online_vllm.eval.answers import GRID, INK, INK_2, MUTED, _stats, answer_curves  # noqa: E402
from online_vllm.models import ModelConfig, build_model  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# One fixed categorical slot per adapter, so a color always means the same model.
COLORS = {"none": "#2a78d6", "best": "#eb6834", "last": "#1baf7a"}
NAMES = {"none": "исходная Qwen", "best": "LoRA, лучшая эпоха", "last": "LoRA, последняя эпоха"}


def pair_curve(traces) -> list[dict]:
    """Accuracy and probabilities at every pair from the start of the video."""
    pairs = max(len(t.probabilities) for t in traces)
    return [{"pair": p, **_stats([(t, p) for t in traces if p < len(t.probabilities)])} for p in range(pairs)]


def plot(results: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, (shift_ax, pair_ax) = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    chance = next(iter(results.values()))["levels"]["chance"]
    mean_target = sum(r["mean_target_pair"] for r in results.values()) / len(results)
    for name, result in results.items():
        shift = [row for row in result["curve"] if row["count"]]
        shift_ax.plot([row["offset"] for row in shift], [row["accuracy"] for row in shift], color=COLORS[name],
                      linewidth=2, label=NAMES[name], zorder=3)
        pairs = [row for row in result["by_pair"] if row["count"]]
        pair_ax.plot([row["pair"] for row in pairs], [row["accuracy"] for row in pairs], color=COLORS[name],
                     linewidth=2, label=NAMES[name], zorder=3)
    for ax, xlabel in ((shift_ax, "сдвиг относительно целевой пары, пары (1 пара = 0,08 с)"),
                       (pair_ax, "пара кадров от начала видео (1 пара = 0,08 с)")):
        ax.axhline(chance, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.text(ax.get_xlim()[1], chance, "угадывание ", color=INK_2, fontsize=9, va="bottom", ha="right")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    shift_ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
    shift_ax.text(0, 1.0, " ответ виден", color=INK_2, fontsize=9, va="top", ha="left")
    shift_ax.set_title("Точность ответа по сдвигу от момента появления ответа", color=INK, fontsize=11, loc="left")
    pair_ax.axvline(mean_target, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    pair_ax.text(mean_target, 1.0, " средний момент ответа", color=INK_2, fontsize=9, va="top", ha="left")
    pair_ax.set_title("Точность ответа от начала видео", color=INK, fontsize=11, loc="left")
    fig.suptitle(title, color=INK_2, fontsize=10, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="output directory of train_lora_sft.py")
    parser.add_argument("--adapters", nargs="+", choices=tuple(COLORS), default=["none", "best"])
    parser.add_argument("--part", choices=("validation", "test"), default="test",
                        help="half of the LoRA validation videos: test (default) is untouched by training and by the "
                             "choice of the best epoch; validation is the half that chose it")
    parser.add_argument("--offsets", default="-30:40", help="range of shifts from the target pair, a:b")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="JSON; default <run>/answer_curves_<part>.json, the PNG next to it")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    run = Path(args.run)
    model = build_model(ModelConfig.load(run / "model.yaml"), local_files_only=args.local_files_only,
                        device=args.device)
    builder = InputBuilder.from_config(model.config, local_files_only=args.local_files_only)
    provider, split = clevrer_lora_part_provider(run, args.part, data_root=args.data_root, builder=builder)
    if split.vision_cache is not None:
        cache = VisionCache.for_model(split.vision_cache, model, frame_size=split.train_frame_size,
                                      frame_stride=split.frame_stride)
        build_vision_cache(model, builder, provider.records, cache, data_root=args.data_root,
                           frame_size=split.train_frame_size, frame_stride=split.frame_stride, device=args.device,
                           num_workers=split.num_workers)
    start, stop = (int(v) for v in args.offsets.split(":"))
    results = {}
    for name in args.adapters:
        if name != "none":
            load_lora_adapter(model, run / name)
        # The base model is the same backbone with every LoRA layer switched off.
        with model.backbone.disable_adapter() if name == "none" else nullcontext():
            traces = answer_traces(model, provider, split.train_frame_size, device=args.device, desc=NAMES[name])
        results[name] = {**answer_curves(traces, range(start, stop + 1)), "by_pair": pair_curve(traces),
                         "mean_target_pair": sum(t.target_pair for t in traces) / len(traces)}
        levels = results[name]["levels"]
        print(f"{NAMES[name]}: " + ", ".join(f"{key} {levels[key]['accuracy']:.3f}"
                                             for key in ("first_pair", "before_visible", "at_target", "at_end")))

    output = Path(args.output) if args.output else run / f"answer_curves_{args.part}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"run": str(run), "part": args.part, "questions": len(provider.records),
                                  "adapters": results}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    plot(results, output.with_suffix(".png"), f"{run.name}, {args.part}: {len(provider.records)} вопросов")
    print(f"-> {output}, {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
