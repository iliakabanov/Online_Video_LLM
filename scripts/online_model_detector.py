"""Детектор готовности поверх онлайн-модели (Part II) для одного адаптера онлайн-SFT — та же цепочка, что была у
outputs/listen_sft_10ep/epoch_10 (experiment_launcher/online_10ep, evaluate_test.sh, listen/compare_head.sh), одной
командой. Шаги по порядку (--steps — часть из них):

  features   признаки онлайн-модели на её собственном входе (scripts/extract_features.py --online-run --videos 1000
             --no-upload) для train- и eval-половины; прерванное извлечение та же команда продолжает
  head       MLP на внимании (группа attention, StateHead): метки hard, 10 эпох, seed 0, best.pt по AUROC в окне ±2
             -> checkpoints/lora/mlp/attention_mlp_<suffix>, results/lora/mlp/attention_mlp_<suffix>
  head_test  test головы при τ с наибольшим on_target_pm2 на validation (uhead.evaluate, как evaluate_test.sh)
  native     собственный детектор онлайн-модели (<speak>) на validation- и test-половинах головы (те же вопросы),
             τ по on_time_0_4 на validation (scripts/evaluate_online_sft.py) -> <run>/{,predictions_}head_<хэш>_*
  bootstrap  95% интервалы на test (scripts/test_bootstrap.py): свой детектор и MLP, плюс --system из других таблиц
             -> results/lora/attention_mlp_<suffix>_test_bootstrap.json

Готовые шаги head, head_test и native пропускаются (--overwrite — пересчитать). GPU — CUDA_VISIBLE_DEVICES.

python scripts/online_model_detector.py --run outputs/listen_sft_10ep_aw0.25
python scripts/online_model_detector.py --run outputs/listen_sft_10ep_aw0.25 --steps bootstrap \\
    --system "MLP attention (M_off)" results/lora/mlp/attention_mlp_off10/predictions_validation.json \\
                                     results/lora/mlp/attention_mlp_off10/predictions_test_best_tau<τ>.json
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
STEPS = ("features", "head", "head_test", "native", "bootstrap")
VIDEOS = 1000


def run_command(command: list, env: dict | None = None) -> None:
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=REPO, env=env, check=True)


def python_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def feature_dirs(data_root: Path, run: Path, adapter: str) -> dict:
    """Каталоги, куда extract_features.py пишет признаки онлайн-модели: <features.train|eval>_<run>_<adapter>_videos1000."""
    features = yaml.safe_load((REPO / "configs/train/group1.yaml").read_text(encoding="utf-8"))["features"]
    return {split: data_root / f"{features[split]}_{run.name}_{adapter}_videos{VIDEOS}" for split in ("train", "eval")}


def check_features(path: Path, run: Path, adapter: str) -> None:
    """Признаки именно этой онлайн-модели и 1000 видео половины для признаков (как experiment_launcher/online_10ep/setup.sh)."""
    meta_path = path / "meta.json"
    if not meta_path.exists():
        sys.exit(f"Нет {meta_path}: сначала шаг features")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    found = meta.get("adapter") or {}
    problems = []
    if "online" not in meta:
        problems.append("это признаки не онлайн-модели (нет поля online)")
    if (found.get("run"), found.get("name")) != (run.name, adapter):
        problems.append(f"адаптер {found.get('run')}/{found.get('name')}, ожидался {run.name}/{adapter}")
    if meta.get("videos") != VIDEOS:
        problems.append(f"{meta.get('videos')} видео, ожидалось {VIDEOS}")
    if not str(meta.get("source", "")).endswith("_group1_features.jsonl"):
        problems.append(f"источник {meta.get('source')} — не половина для признаков")
    if problems:
        sys.exit(f"{meta_path}: " + "; ".join(problems))
    print(f"{meta_path}: онлайн-модель {found['run']}/{found['name']} (эпоха {found.get('epoch')}), "
          f"{meta['videos']} видео, {meta['records']} вопросов")


def step_features(args, dirs: dict) -> None:
    for split in ("train", "eval"):
        run_command([sys.executable, "scripts/extract_features.py", "--split", split, "--videos", VIDEOS, "--no-upload",
                     "--online-run", args.run, "--online-adapter", args.adapter, "--device", args.device,
                     "--data-root", args.data_root])


def step_head(args, dirs: dict, checkpoint: Path, results: Path) -> None:
    if (checkpoint / "best.pt").exists() and (results / "results.json").exists() and not args.overwrite:
        print(f"\n-- голова уже обучена: {checkpoint} (--overwrite — заново)")
        return
    for path in (checkpoint, results):
        if path.exists() and any(path.iterdir()):
            if not args.overwrite:
                sys.exit(f"{path} не пуст, но голова не дообучена: --overwrite — удалить и обучить заново")
            shutil.rmtree(path)
    for split in ("train", "eval"):
        check_features(dirs[split], args.run, args.adapter)
    checkpoint.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    # Конфиги как у common.sh run_experiment: group1.yaml с путями этих признаков, uhead default.yaml с группой attention.
    train = yaml.safe_load((REPO / "configs/train/group1.yaml").read_text(encoding="utf-8"))
    train["features"]["train"], train["features"]["eval"] = str(dirs["train"]), str(dirs["eval"])
    train["seed"] = 0
    train["optim"]["epochs"] = args.epochs
    uhead = yaml.safe_load((REPO / "configs/uhead/default.yaml").read_text(encoding="utf-8"))
    uhead["groups"], uhead["temporal_detector"] = ["attention"], "mlp"
    with tempfile.TemporaryDirectory() as tmp:
        train_config, uhead_config = Path(tmp) / "train.yaml", Path(tmp) / "uhead.yaml"
        train_config.write_text(yaml.safe_dump(train, sort_keys=False), encoding="utf-8")
        uhead_config.write_text(yaml.safe_dump(uhead, sort_keys=False), encoding="utf-8")
        run_command([sys.executable, "-m", "uhead.train", "--tags", "hard", "--uhead", uhead_config,
                     "--train", train_config, "--data-root", args.data_root, "--output", checkpoint,
                     "--results", results, "--device", args.device, "--selection", "auroc_pm2", "--flat-output",
                     "--overwrite"], env=python_env())

    config = json.loads((checkpoint / "experiment.json").read_text(encoding="utf-8"))
    best_epoch = json.loads((results / "results.json").read_text(encoding="utf-8"))["best_epoch"]
    config.update({"mode": "lora", "architecture": "mlp", "experiment_id": checkpoint.name,
                   "feature_groups": ["attention"], "training_gt": "hard", "seed": 0,
                   "checkpoint_path": str(checkpoint / "best.pt"), "best_epoch": best_epoch,
                   "max_layer": config.get("uhead_config", {}).get("max_layer"),
                   "online_run": str(args.run), "online_adapter": args.adapter})
    (results / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def head_test_file(results: Path) -> Path | None:
    found = sorted(results.glob("predictions_test_best_tau*.json"))
    return found[0] if found else None


def step_head_test(args, checkpoint: Path, results: Path) -> None:
    if head_test_file(results) and not args.overwrite:
        print(f"\n-- test головы уже есть: {head_test_file(results)}")
        return
    if not (checkpoint / "best.pt").exists():
        sys.exit(f"Нет {checkpoint / 'best.pt'}: сначала шаг head")
    # τ с наибольшим on_target_pm2 на validation, при равенстве — больший (как evaluate_test.sh).
    thresholds = json.loads((results / "results.json").read_text(encoding="utf-8"))["validation"]["thresholds"]
    scores = {tau: values["on_target_pm2"] for tau, values in thresholds.items()}
    tau = max(scores, key=lambda t: (scores[t], float(t)))
    (results / "selection.json").write_text(json.dumps(
        {"metric": "on_target_pm2", "threshold": float(tau), "validation_value": scores[tau],
         "validation_by_threshold": scores}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n== τ головы = {tau} (on_target_pm2 на validation)")
    run_command([sys.executable, "-m", "uhead.evaluate", "--run", checkpoint, "--results", results, "--threshold", tau,
                 "--data-root", args.data_root, "--device", args.device], env=python_env())


def split_name(checkpoint: Path) -> str:
    """head_<хэш половин головы>, как в listen/compare_head.sh: у голов с одним split.json — одни и те же оценки."""
    split = json.loads((checkpoint / "split.json").read_text(encoding="utf-8"))
    digest = hashlib.sha1(json.dumps([sorted(split["validation_videos"]),
                                      sorted(split["test_videos"])]).encode()).hexdigest()[:8]
    return f"head_{digest}"


def native_file(args, name: str, part: str) -> Path:
    return args.run / f"predictions_{name}_{part}_{args.adapter}.json"


def step_native(args, checkpoint: Path) -> None:
    if not (checkpoint / "split.json").exists():
        sys.exit(f"Нет {checkpoint / 'split.json'}: сначала шаг head (половины головы)")
    name = split_name(checkpoint)
    for part in ("validation", "test"):
        if native_file(args, name, part).exists() and not args.overwrite:
            print(f"\n-- {native_file(args, name, part)} уже есть")
            continue
        command = [sys.executable, "scripts/evaluate_online_sft.py", "--run", args.run, "--adapter", args.adapter,
                   "--name", f"{name}_{part}", "--records-from", checkpoint / "split.json", "--records-part", part,
                   "--tau-criterion", "on_time_0_4", "--data-root", args.data_root, "--device", args.device]
        if part == "test":
            command += ["--tau-from", native_file(args, name, "validation")]
        run_command(command)


def step_bootstrap(args, checkpoint: Path, results: Path) -> None:
    name, head_test = split_name(checkpoint), head_test_file(results)
    if head_test is None:
        sys.exit(f"Нет test головы в {results}: сначала шаг head_test")
    label = f"{args.run.name}/{args.adapter}"
    command = [sys.executable, "scripts/test_bootstrap.py",
               "--output", REPO / "results/lora" / f"{checkpoint.name}_test_bootstrap.json",
               "--system", f"свой детектор ({label})", native_file(args, name, "validation"),
               native_file(args, name, "test"),
               "--system", f"MLP attention ({label})", results / "predictions_validation.json", head_test]
    for system in args.system:
        command += ["--system", *system]
    run_command(command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="каталог scripts/train_online_sft.py")
    parser.add_argument("--adapter", default="best", help="best, last или epoch_<n>")
    parser.add_argument("--suffix", help="суффикс id головы (по умолчанию on_<run>_<adapter>)")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--epochs", type=int, default=10, help="эпох обучения головы")
    parser.add_argument("--system", nargs=3, action="append", default=[], metavar=("NAME", "VALIDATION", "TEST"),
                        help="ещё система для таблицы bootstrap (как в test_bootstrap.py)")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", REPO / "data")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true", help="пересчитать готовые шаги head, head_test, native")
    args = parser.parse_args()

    args.run = args.run if args.run.is_absolute() else REPO / args.run
    if not (args.run / "online.json").exists():
        parser.error(f"{args.run}: нет online.json — это не каталог train_online_sft.py")
    if not (args.run / args.adapter / "adapter_model.safetensors").exists():
        parser.error(f"нет адаптера {args.run / args.adapter}")
    args.data_root = args.data_root.resolve()
    experiment = f"attention_mlp_{args.suffix or f'on_{args.run.name}_{args.adapter}'}"
    checkpoint, results = REPO / "checkpoints/lora/mlp" / experiment, REPO / "results/lora/mlp" / experiment
    dirs = feature_dirs(args.data_root, args.run, args.adapter)
    print(f"Онлайн-модель {args.run.name}/{args.adapter}; признаки {dirs['train'].name}, {dirs['eval'].name}; "
          f"голова {experiment}; шаги {' '.join(s for s in STEPS if s in args.steps)}")

    if "features" in args.steps:
        step_features(args, dirs)
    if "head" in args.steps:
        step_head(args, dirs, checkpoint, results)
    if "head_test" in args.steps:
        step_head_test(args, checkpoint, results)
    if "native" in args.steps:
        step_native(args, checkpoint)
    if "bootstrap" in args.steps:
        step_bootstrap(args, checkpoint, results)


if __name__ == "__main__":
    main()
