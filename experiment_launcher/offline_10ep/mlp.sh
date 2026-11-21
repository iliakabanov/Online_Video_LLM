#!/usr/bin/env bash
# MLP-голова (StateHead) на каждой группе признаков нового офлайн-адаптера по отдельности: lm, attention, entropy,
# vision, confidence — 5 запусков, results/lora/mlp/<группа>_mlp_off10. Настройки — setup.sh.
#   CUDA_VISIBLE_DEVICES=1 bash experiment_launcher/offline_10ep/mlp.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup.sh"
for group in "${GROUPS_TO_RUN[@]}"; do
  echo "== mlp: $group"
  run_experiment "${group}_mlp_${SUFFIX}" mlp "$group"
done
