"""Сводка ablation голов на признаках нового офлайн-адаптера (mlp.sh, transformer.sh, compact.sh, combos.sh) — только
validation.

Из results.json каждого запуска results/lora/<arch>/<группы через _>_<arch>_<suffix> (метрики best.pt на validation):
- выбор best.pt: лучшая эпоха и AUROC классификации префиксов в окне |k − t*| <= 2 (критерий выбора), BCE там же;
- on_time_0_4 при лучшем τ validation (ответ через 0..4 пары после целевой);
- задержка (пары после t*; среднее / медиана) при доле преждевременных ответов <= R и точности непреждевременных >= A.

python experiment_launcher/offline_10ep/summary.py [--suffix off10] [--output results/lora/summary_off10.json]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = {"mlp": "mlp", "transformer": "base", "compact": "compact"}  # каталог результатов -> метка в id


def latency(point) -> str:
    if point is None or not point.get("feasible"):
        return "-"
    return f"{point['latency_mean']:.2f}/{point['latency_median']:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suffix", default="off10")
    parser.add_argument("--output", help="JSON сводки; по умолчанию results/lora/summary_<suffix>.json")
    args = parser.parse_args()

    rows, keys = [], None
    for architecture, tag in ARCHITECTURES.items():
        ending = f"_{tag}_{args.suffix}"
        for path in sorted((ROOT / "results" / "lora" / architecture).glob(f"*{ending}/results.json")):
            group = "+".join(path.parent.name[:-len(ending)].split("_"))
            result = json.loads(path.read_text(encoding="utf-8"))
            validation = result["validation"]
            constrained = validation.get("constrained_latency") or {}
            keys = keys or list(constrained)
            best = (validation.get("best_thresholds") or {}).get("on_time_0_4") or {}
            rows.append({"architecture": architecture, "group": group, "best_epoch": result["best_epoch"],
                         "selection": result.get("selection"), "questions": validation["questions"],
                         "auroc_pm2": validation["classification"]["auroc_pm2"],
                         "bce_pm2": validation["classification"]["binary_cross_entropy_pm2"],
                         "on_time_0_4": best.get("on_time_0_4"), "on_time_0_4_tau": best.get("tau"),
                         "constrained_latency": constrained})
    if not rows:
        raise SystemExit(f"Нет results.json запусков *_{args.suffix} в results/lora/")

    keys = keys or []
    width = max(11, *(len(row["group"]) for row in rows))
    header = f"{'модель':12s} {'группы':{width}s} {'эп.':>3s} {'AUROC±2':>8s} {'BCE±2':>7s} {'on_time':>8s} {'τ':>6s} " + \
        " ".join(f"{key:>12s}" for key in keys)
    print(f"Validation, {rows[0]['questions']} вопросов; задержка — пары после t*, среднее/медиана")
    print(header)
    for row in sorted(rows, key=lambda r: -r["auroc_pm2"]):
        on_time = "-" if row["on_time_0_4"] is None else f"{row['on_time_0_4']:.4f}"
        tau = "-" if row["on_time_0_4_tau"] is None else f"{row['on_time_0_4_tau']:.3f}"
        print(f"{row['architecture']:12s} {row['group']:{width}s} {row['best_epoch']:>3} {row['auroc_pm2']:8.4f} "
              f"{row['bce_pm2']:7.4f} {on_time:>8s} {tau:>6s} " +
              " ".join(f"{latency(row['constrained_latency'].get(key)):>12s}" for key in keys))
    output = Path(args.output) if args.output else ROOT / "results" / "lora" / f"summary_{args.suffix}.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"-> {output}")


if __name__ == "__main__":
    main()
