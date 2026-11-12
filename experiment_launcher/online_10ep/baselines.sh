#!/usr/bin/env bash
# Бейзлайны на признаках онлайн-модели (настройки — setup.sh):
#   1. MLP только на скрытом состоянии последнего слоя в последнем визуальном токене пары (группа last_hidden):
#      results/lora/mlp/last_hidden_mlp_on10;
#   2. без обучения: уверенность ответа ветки <speak> на паре как оценка готовности — maxprob, margin, entropy
#      (scripts/confidence_baseline.py) на той же validation-половине: results/lora/confidence/<оценка>_conf_on10.
# Сводка вместе с головами: python experiment_launcher/offline_10ep/summary.py --suffix on10
#   CUDA_VISIBLE_DEVICES=1 bash experiment_launcher/online_10ep/baselines.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup.sh"
echo "== mlp: last_hidden"
run_experiment "last_hidden_mlp_${SUFFIX}" mlp last_hidden
echo "== уверенность ответа (без обучения)"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python "$ROOT/scripts/confidence_baseline.py" \
  --features "$VAL_FEATURES" --split "$ROOT/checkpoints/lora/mlp/last_hidden_mlp_${SUFFIX}/split.json" --suffix "$SUFFIX"
