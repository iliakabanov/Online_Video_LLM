#!/usr/bin/env bash
# Общий запуск: обучение cached-feature detector и метрики на validation-половине VAL_FEATURES при каждом пороге
# (eval.thresholds в configs/train/group1.yaml). MODE — чьи признаки: original (исходная Qwen) или lora (Qwen с
# адаптером); сверяется с meta.json признаков. В корне репозитория:
#   checkpoints/<mode>/<arch>/<id>/  веса и всё для их загрузки и оценки: best.pt, last.pt, конфиги, split.json (не в git)
#   results/<mode>/<arch>/<id>/      метрики и предсказания: results.json, metrics.jsonl, predictions_validation.json,
#                                    run_config.json
# Test-половина оценивается отдельно, с выбранным порогом:
#   PYTHONPATH=src python -m uhead.evaluate --run checkpoints/<mode>/<arch>/<id> --results results/<mode>/<arch>/<id> \
#     --threshold <tau> --data-root "$DATA_ROOT"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/PATH/TO/DATA}"
TRAIN_FEATURES="${TRAIN_FEATURES:-/PATH/TO/TRAIN_FEATURES}"
VAL_FEATURES="${VAL_FEATURES:-/PATH/TO/VAL_FEATURES}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-0}"
TAG_MODE="${TAG_MODE:-hard}"
MODE="${MODE:-}"
UHEAD_CONFIG_PATH="${UHEAD_CONFIG_PATH:-$ROOT/configs/uhead/default.yaml}"

run_experiment() {
  local experiment_id="$1" architecture="$2"; shift 2
  local groups=("$@") detector model_name
  case "$TAG_MODE" in hard|step|sigmoid|gaussian) ;; *) echo "Unknown tag mode: $TAG_MODE" >&2; return 2 ;; esac
  case "$MODE" in original|lora) ;; *) echo "Задайте MODE=original или MODE=lora (сейчас: '$MODE')." >&2; return 2 ;; esac
  case "$architecture" in
    mlp) detector="mlp"; model_name="statehead" ;;
    transformer) detector="transformer"; model_name="uhead" ;;
    compact) detector="compact"; model_name="uhead" ;;
    *) echo "Неизвестная архитектура: $architecture" >&2; return 2 ;;
  esac
  [[ -f "$UHEAD_CONFIG_PATH" ]] || { echo "UHead config not found: $UHEAD_CONFIG_PATH" >&2; return 2; }
  [[ "$DATA_ROOT" != "/PATH/TO/DATA" && "$TRAIN_FEATURES" != "/PATH/TO/TRAIN_FEATURES" && "$VAL_FEATURES" != "/PATH/TO/VAL_FEATURES" ]] || {
    echo "Замените DATA_ROOT, TRAIN_FEATURES и VAL_FEATURES перед запуском." >&2; return 2; }
  # adapter есть в meta.json только у признаков модели с LoRA; train и eval сверяет между собой uhead.train.
  local features
  for features in "$TRAIN_FEATURES" "$VAL_FEATURES"; do
    MODE="$MODE" META="$features/meta.json" python - <<'PY' || return 2
import json, os, sys
mode, meta = os.environ["MODE"], os.environ["META"]
found = "lora" if "adapter" in json.load(open(meta, encoding="utf-8")) else "original"
if found != mode:
    sys.exit(f"MODE={mode}, но признаки {meta} сняты моделью {found}.")
PY
  done

  local checkpoint_dir="$ROOT/checkpoints/$MODE/$architecture/$experiment_id"
  local result_dir="$ROOT/results/$MODE/$architecture/$experiment_id"
  local train_config uhead_config
  if { [[ -d "$checkpoint_dir" ]] && [[ -n "$(find "$checkpoint_dir" -mindepth 1 -print -quit)" ]]; } || \
     { [[ -d "$result_dir" ]] && [[ -n "$(find "$result_dir" -mindepth 1 -print -quit)" ]]; }; then
    [[ "$OVERWRITE" == "1" ]] || { echo "Эксперимент $experiment_id уже существует; задайте OVERWRITE=1 для перезаписи." >&2; return 2; }
    rm -rf "$checkpoint_dir" "$result_dir"
  fi
  mkdir -p "$checkpoint_dir" "$result_dir"
  train_config="$(mktemp)"; uhead_config="$(mktemp)"
  trap 'rm -f "$train_config" "$uhead_config"' RETURN
  ROOT="$ROOT" TRAIN_FEATURES="$TRAIN_FEATURES" VAL_FEATURES="$VAL_FEATURES" \
    FEATURE_GROUPS="${groups[*]}" DETECTOR="$detector" TRAIN_CONFIG="$train_config" UHEAD_CONFIG="$uhead_config" \
    UHEAD_CONFIG_PATH="$UHEAD_CONFIG_PATH" python - <<'PY'
import os
from pathlib import Path
import yaml

root = Path(os.environ["ROOT"])
train = yaml.safe_load((root / "configs/train/group1.yaml").read_text(encoding="utf-8"))
train["features"]["train"] = os.environ["TRAIN_FEATURES"]
train["features"]["eval"] = os.environ["VAL_FEATURES"]
train["seed"] = 0
uhead = yaml.safe_load(Path(os.environ["UHEAD_CONFIG_PATH"]).read_text(encoding="utf-8"))
uhead["groups"] = os.environ["FEATURE_GROUPS"].split()
uhead["temporal_detector"] = os.environ["DETECTOR"]
Path(os.environ["TRAIN_CONFIG"]).write_text(yaml.safe_dump(train, sort_keys=False), encoding="utf-8")
Path(os.environ["UHEAD_CONFIG"]).write_text(yaml.safe_dump(uhead, sort_keys=False), encoding="utf-8")
PY
  # Обучение по эпохам (best.pt / last.pt) и метрики best.pt на validation-половине при каждом пороге.
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m uhead.train --tags "$TAG_MODE" --uhead "$uhead_config" \
    --train "$train_config" --data-root "$DATA_ROOT" --output "$checkpoint_dir" --results "$result_dir" \
    --device "$DEVICE" --flat-output --overwrite
  ROOT="$ROOT" ARCHITECTURE="$architecture" EXPERIMENT_ID="$experiment_id" FEATURE_GROUPS="${groups[*]}" TAG_MODE="$TAG_MODE" \
    MODE="$MODE" CHECKPOINT="$checkpoint_dir" RESULT="$result_dir" python - <<'PY'
import json, os
from pathlib import Path
checkpoint = Path(os.environ["CHECKPOINT"])
config = json.loads(checkpoint.joinpath("experiment.json").read_text(encoding="utf-8"))
results = json.loads(Path(os.environ["RESULT"]).joinpath("results.json").read_text(encoding="utf-8"))
config.update({
    "mode": os.environ["MODE"], "architecture": os.environ["ARCHITECTURE"], "experiment_id": os.environ["EXPERIMENT_ID"],
    "feature_groups": os.environ["FEATURE_GROUPS"].split(), "training_gt": os.environ["TAG_MODE"], "seed": 0,
    "checkpoint_path": str(checkpoint / "best.pt"), "best_epoch": results["best_epoch"],
})
Path(os.environ["RESULT"]).joinpath("run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}
