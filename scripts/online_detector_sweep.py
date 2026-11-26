"""MLP-детекторы готовности на разных наборах групп признаков онлайн-модели и их сравнение на validation — как
experiment_launcher/online_10ep/mlp.sh и combos.sh у outputs/listen_sft_10ep/epoch_10, для любого адаптера онлайн-SFT.

Признаки и собственный детектор онлайн-модели (native <speak>) на validation-половине головы уже должны быть посчитаны:
scripts/online_model_detector.py --steps features head native. Каждая голова — как в нём (StateHead, метки hard,
10 эпох, seed 0, best.pt по AUROC в окне ±2): checkpoints|results/lora/mlp/<группы через _>_mlp_<suffix>; готовые
пропускаются, упавшая не останавливает остальные.

Сравнение — только validation (test не смотрим, пока не выбран детектор), всё из сохранённых вероятностей, как в
scripts/test_bootstrap.py, только «test» — та же validation: τ по on_time_0_4 и τ ограничений R ≤ 5% / 10%, A ≥ 90%
выбираются на ней же, 95% интервалы — по видео validation с возвращением. Native — первая строка, головы — по
задержке при P ≤ 5%. -> results/lora/validation_sweep_<suffix>.json

python scripts/online_detector_sweep.py --run outputs/listen_sft_10ep_aw0.25
python scripts/online_detector_sweep.py --run outputs/listen_sft_10ep_aw0.25 --sets "attention" "lm attention" --summary-only
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (src/ on the path)
from online_model_detector import REPO, feature_dirs, native_file, split_name, step_head
from online_vllm.eval.report import questions, test_report
from test_bootstrap import LATENCY_KEYS, System, video_of

# Как mlp.sh (группы по одной), combos.sh (с собственным детектором модели) и база абляции глубины.
DEFAULT_SETS = (
    "attention", "lm", "entropy", "vision", "confidence", "listen_token_logit",
    "listen_token_logit attention", "listen_token_logit attention entropy",
    "listen_token_logit attention entropy lm", "lm attention entropy",
    "lm attention entropy vision confidence listen_token_logit",
)


def train_heads(args, sets: list[tuple[str, ...]], suffix: str) -> dict:
    """Обучает головы, которых ещё нет; возвращает {набор: каталог результатов} обученных."""
    dirs = feature_dirs(args.data_root, args.run, args.adapter)
    done, failed = {}, []
    for groups in sets:
        experiment = f"{'_'.join(groups)}_mlp_{suffix}"
        checkpoint, results = REPO / "checkpoints/lora/mlp" / experiment, REPO / "results/lora/mlp" / experiment
        if not args.summary_only:
            print(f"\n==== {' + '.join(groups)} -> {experiment}", flush=True)
            try:
                step_head(args, dirs, checkpoint, results, groups)
            except (subprocess.CalledProcessError, SystemExit) as error:
                print(f"!! {experiment}: {error}", flush=True)
                failed.append(experiment)
                continue
        if (results / "predictions_validation.json").exists():
            done[groups] = (checkpoint, results)
        else:
            print(f"-- {experiment}: нет предсказаний validation, в сводку не входит")
    if failed:
        print(f"\nУпали: {', '.join(failed)}")
    return done


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="каталог scripts/train_online_sft.py")
    parser.add_argument("--adapter", default="best")
    parser.add_argument("--suffix", help="как в online_model_detector.py (по умолчанию on_<run>_<adapter>)")
    parser.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS), help="наборы групп, группы через пробел")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--summary-only", action="store_true", help="не обучать, только сводка по готовым головам")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", REPO / "data")))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.overwrite = False  # step_head: готовые головы не переобучаются
    args.run = args.run if args.run.is_absolute() else REPO / args.run
    args.data_root = args.data_root.resolve()
    suffix = args.suffix or f"on_{args.run.name}_{args.adapter}"
    sets = [tuple(item.split()) for item in args.sets]

    heads = train_heads(args, sets, suffix)
    if not heads:
        sys.exit("Нет ни одной обученной головы")

    # Native на validation-половине головы; у всех голов одни и те же половины (split.json, seed 0).
    names = {split_name(checkpoint) for checkpoint, _ in heads.values() if (checkpoint / "split.json").exists()}
    if len(names) > 1:
        sys.exit(f"У голов разные половины validation/test: {sorted(names)}")
    native_path = native_file(args, names.pop(), "validation") if names else None
    systems_in = []
    if native_path is not None and native_path.exists():
        systems_in.append((f"native <speak> ({args.run.name}/{args.adapter})", load(native_path)))
    else:
        missing = native_path or "split.json голов (checkpoints/lora/mlp/*/split.json)"
        print(f"!! нет {missing} (online_model_detector.py --steps head native): сводка без native")
    for groups, (_, results) in heads.items():
        systems_in.append((" + ".join(groups), load(results / "predictions_validation.json")))

    reference = questions(systems_in[0][1])
    order = [row["record_id"] for row in systems_in[0][1]["predictions"]]
    systems = []
    for name, payload in systems_in:
        if questions(payload) != reference:
            sys.exit(f"«{name}»: другие вопросы validation, чем у «{systems_in[0][0]}»")
        # validation как «test»: τ выбираются на ней же, метрики — на ней же.
        systems.append(System(name, test_report(payload, payload, "on_time_0_4"), payload, order))

    videos, video_index = np.unique([video_of(record_id) for record_id in order], return_inverse=True)
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(videos), size=(args.resamples, len(videos)))
    weights = np.stack([np.bincount(draw, minlength=len(videos)) for draw in draws]).astype(float)[:, video_index]
    keys = ["auroc_pm2", "auroc_pm5", "on_time_0_4", *LATENCY_KEYS]
    output = {"run": str(args.run), "adapter": args.adapter, "suffix": suffix, "videos": int(len(videos)),
              "questions": len(order), "resamples": args.resamples, "seed": args.seed, "systems": {}}
    for system in systems:
        point = system.metrics(np.ones(len(order)))
        samples = [system.metrics(w) for w in weights]
        entry = {}
        for key in keys:
            if point[key] is None:
                entry[key] = None
                continue
            low, high = np.nanpercentile([s[key] for s in samples], [2.5, 97.5])
            entry[key] = {"value": point[key], "ci95": [float(low), float(high)]}
        output["systems"][system.name] = entry

    def cell(entry, scale, digits):
        if entry is None:
            return f"{'недостижимо':^21s}"
        v, (lo, hi) = entry["value"] * scale, entry["ci95"]
        return f"{v:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"

    def rank(item):
        latency = item[1]["latency_R5_A90"]
        return float("inf") if latency is None else latency["value"]

    rows = list(output["systems"].items())
    with_native = native_path is not None and native_path.exists()
    rows = rows[:1] + sorted(rows[1:], key=rank) if with_native else sorted(rows, key=rank)
    width = max(len(name) for name, _ in rows)
    print(f"\nValidation: {len(videos)} видео, {len(order)} вопросов; 95% интервал по {args.resamples} выборкам видео; "
          f"задержка в кадрах (2 кадра на пару)")
    print(f"{'детектор':{width}s} | {'AUROC±2':^21s} | {'AUROC±5':^21s} | {'On-time, %':^18s} | "
          f"{'Задержка P≤5%':^21s} | {'Задержка P≤10%':^21s}")
    for name, e in rows:
        print(f"{name:{width}s} | {cell(e['auroc_pm2'], 1, 3)} | {cell(e['auroc_pm5'], 1, 3)} | "
              f"{cell(e['on_time_0_4'], 100, 1)} | "
              f"{cell(e['latency_R5_A90'], 2, 2)} | {cell(e['latency_R10_A90'], 2, 2)}")
    path = REPO / "results/lora" / f"validation_sweep_{suffix}.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"-> {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
