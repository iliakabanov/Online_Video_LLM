#!/usr/bin/env bash
# Общий запуск: обучение cached-feature detector, один validation inference и sweep.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/PATH/TO/DATA}"
TRAIN_FEATURES="${TRAIN_FEATURES:-/PATH/TO/TRAIN_FEATURES}"
VAL_FEATURES="${VAL_FEATURES:-/PATH/TO/VAL_FEATURES}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-0}"
THRESHOLDS=(0.5 0.6 0.7 0.8 0.85 0.9 0.95 0.99)

run_experiment() {
  local experiment_id="$1" architecture="$2"; shift 2
  local groups=("$@") detector model_name
  case "$architecture" in
    mlp) detector="mlp"; model_name="statehead" ;;
    transformer) detector="transformer"; model_name="uhead" ;;
    *) echo "Неизвестная архитектура: $architecture" >&2; return 2 ;;
  esac
  [[ "$DATA_ROOT" != "/PATH/TO/DATA" && "$TRAIN_FEATURES" != "/PATH/TO/TRAIN_FEATURES" && "$VAL_FEATURES" != "/PATH/TO/VAL_FEATURES" ]] || {
    echo "Замените DATA_ROOT, TRAIN_FEATURES и VAL_FEATURES перед запуском." >&2; return 2; }

  local checkpoint_dir="$ROOT/experiment_launcher/checkpoints/$architecture/$experiment_id"
  local result_dir="$ROOT/experiment_launcher/results/$architecture/$experiment_id"
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
    GROUPS="${groups[*]}" DETECTOR="$detector" TRAIN_CONFIG="$train_config" UHEAD_CONFIG="$uhead_config" python - <<'PY'
import os
from pathlib import Path
import yaml

root = Path(os.environ["ROOT"])
train = yaml.safe_load((root / "configs/train/group1.yaml").read_text(encoding="utf-8"))
train["features"]["train"] = os.environ["TRAIN_FEATURES"]
train["features"]["eval"] = os.environ["VAL_FEATURES"]
train["seed"] = 0
uhead = yaml.safe_load((root / "configs/uhead/default.yaml").read_text(encoding="utf-8"))
uhead["groups"] = os.environ["GROUPS"].split()
uhead["temporal_detector"] = os.environ["DETECTOR"]
Path(os.environ["TRAIN_CONFIG"]).write_text(yaml.safe_dump(train, sort_keys=False), encoding="utf-8")
Path(os.environ["UHEAD_CONFIG"]).write_text(yaml.safe_dump(uhead, sort_keys=False), encoding="utf-8")
PY
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m uhead.train --tags hard --uhead "$uhead_config" \
    --train "$train_config" --data-root "$DATA_ROOT" --output "$checkpoint_dir" --device "$DEVICE" --flat-output --overwrite
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m online_vllm.eval.export --model "$model_name" \
    --checkpoint "$checkpoint_dir/final" --features "$VAL_FEATURES" --split validation \
    --output "$result_dir/predictions_val.json" --device "$DEVICE"
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m online_vllm.eval.thresholds \
    --predictions "$result_dir/predictions_val.json" --output "$result_dir/thresholds_val.json" \
    --thresholds "${THRESHOLDS[@]}"
  ROOT="$ROOT" ARCHITECTURE="$architecture" EXPERIMENT_ID="$experiment_id" GROUPS="${groups[*]}" \
    CHECKPOINT="$checkpoint_dir/final" RESULT="$result_dir" python - <<'PY'
import json, os
from pathlib import Path
checkpoint = Path(os.environ["CHECKPOINT"])
config = json.loads(checkpoint.joinpath("experiment.json").read_text(encoding="utf-8"))
config.update({
    "architecture": os.environ["ARCHITECTURE"], "experiment_id": os.environ["EXPERIMENT_ID"],
    "feature_groups": os.environ["GROUPS"].split(), "training_gt": "hard", "seed": 0,
    "checkpoint_path": os.environ["CHECKPOINT"], "split": "validation",
    "thresholds": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99],
})
Path(os.environ["RESULT"]).joinpath("run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}
