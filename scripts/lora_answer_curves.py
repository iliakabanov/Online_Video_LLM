"""Answer accuracy of a LoRA SFT run along the video: by the shift from the target pair (the pair where the answer
becomes visible) and by the pair from the start of the video, for several adapters on the same questions.

The answer at pair p is the option letter the model rates highest as if the video ended after pair p, the probe of
the validation (one pass per question, then the template tail after every pair). Levels and curves are those of
eval/answers.py (chance, first pair, the pair before the answer appears, the target pair, the end of the video).

By default on the test half of the run (split.json): videos that neither training nor the choice of the best epoch
saw, so the curve of the best adapter is not flattered by its selection.

python scripts/lora_answer_curves.py --run outputs/lora_200_fast --adapters none best [--part test|validation]

Redraw a saved result without the model, e.g. in English and only the panel by the shift:
python scripts/lora_answer_curves.py --replot outputs/lora_1000/answer_curves_test.json --lang en --panels shift \
    --plot answer_curves_test_en.png
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
TEXT = {
    "ru": {"names": NAMES, "questions": "вопросов", "chance": "угадывание ", "visible": " ответ виден",
           "mean_target": " средний момент ответа",
           "shift_title": "Точность ответа по сдвигу от момента появления ответа",
           "pair_title": "Точность ответа от начала видео",
           "shift_x": "сдвиг относительно целевой пары, пары (1 пара = 0,08 с)",
           "pair_x": "пара кадров от начала видео (1 пара = 0,08 с)"},
    "en": {"names": {"none": "original Qwen", "best": "LoRA, best epoch", "last": "LoRA, last epoch"},
           "questions": "questions", "chance": "chance ", "visible": " answer visible",
           "mean_target": " mean answer time",
           "shift_title": "Answer accuracy by the shift from the moment the answer appears",
           "pair_title": "Answer accuracy from the start of the video",
           "shift_x": "shift from the target pair, pairs (1 pair = 0.08 s)",
           "pair_x": "frame pair from the start of the video (1 pair = 0.08 s)"},
}


def pair_curve(traces) -> list[dict]:
    """Accuracy and probabilities at every pair from the start of the video."""
    pairs = max(len(t.probabilities) for t in traces)
    return [{"pair": p, **_stats([(t, p) for t in traces if p < len(t.probabilities)])} for p in range(pairs)]


def plot(results: dict, path: Path, title: str, lang: str = "ru", panels: str = "both") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    text = TEXT[lang]
    if panels == "both":
        fig, (shift_ax, pair_ax) = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)
        shift_ax, pair_ax = (ax, None) if panels == "shift" else (None, ax)
    chance = next(iter(results.values()))["levels"]["chance"]
    mean_target = sum(r["mean_target_pair"] for r in results.values()) / len(results)
    for name, result in results.items():
        if shift_ax is not None:
            shift = [row for row in result["curve"] if row["count"]]
            shift_ax.plot([row["offset"] for row in shift], [row["accuracy"] for row in shift], color=COLORS[name],
                          linewidth=2, label=text["names"][name], zorder=3)
        if pair_ax is not None:
            pairs = [row for row in result["by_pair"] if row["count"]]
            pair_ax.plot([row["pair"] for row in pairs], [row["accuracy"] for row in pairs], color=COLORS[name],
                         linewidth=2, label=text["names"][name], zorder=3)
    for ax, xlabel in ((shift_ax, text["shift_x"]), (pair_ax, text["pair_x"])):
        if ax is None:
            continue
        ax.axhline(chance, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.text(ax.get_xlim()[1], chance, text["chance"], color=INK_2, fontsize=9, va="bottom", ha="right")
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
    if shift_ax is not None:
        shift_ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
        shift_ax.text(0, 1.0, text["visible"], color=INK_2, fontsize=9, va="top", ha="left")
        shift_ax.set_title(text["shift_title"], color=INK, fontsize=11, loc="left")
    if pair_ax is not None:
        pair_ax.axvline(mean_target, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        pair_ax.text(mean_target, 1.0, text["mean_target"], color=INK_2, fontsize=9, va="top", ha="left")
        pair_ax.set_title(text["pair_title"], color=INK, fontsize=11, loc="left")
    fig.suptitle(title, color=INK_2, fontsize=10, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="output directory of train_lora_sft.py")
    parser.add_argument("--replot", help="answer_curves_*.json of an earlier run: only redraw it, no model")
    parser.add_argument("--lang", choices=tuple(TEXT), default="ru", help="language of the plot")
    parser.add_argument("--panels", choices=("both", "shift", "pair"), default="both",
                        help="both panels, or only the one by the shift / by the pair from the start")
    parser.add_argument("--plot", help="PNG; default the JSON path with .png (with --replot: _<lang>.png)")
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
    if args.replot:
        saved = json.loads(Path(args.replot).read_text(encoding="utf-8"))
        path = Path(args.plot) if args.plot else Path(args.replot).with_name(f"{Path(args.replot).stem}_{args.lang}.png")
        plot(saved["adapters"], path, f"{Path(saved['run']).name}, {saved['part']}: {saved['questions']} "
                                      f"{TEXT[args.lang]['questions']}", args.lang, args.panels)
        print(f"-> {path}")
        return
    if not args.run:
        parser.error("--run is required unless --replot is given")

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
    path = Path(args.plot) if args.plot else output.with_suffix(".png")
    plot(results, path, f"{run.name}, {args.part}: {len(provider.records)} {TEXT[args.lang]['questions']}",
         args.lang, args.panels)
    print(f"-> {output}, {path}")


if __name__ == "__main__":
    main()
