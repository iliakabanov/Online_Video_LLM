#!/usr/bin/env bash
# Компактная causal Transformer-голова (configs/uhead/compact.yaml) на каждой группе признаков нового офлайн-адаптера
# по отдельности: lm, attention, entropy, vision, confidence — 5 запусков, results/lora/compact/<группа>_compact_off10.
# Настройки — setup.sh.
#   CUDA_VISIBLE_DEVICES=1 bash experiment_launcher/offline_10ep/compact.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup.sh"
for group in "${GROUPS_TO_RUN[@]}"; do
  echo "== compact: $group"
  UHEAD_CONFIG_PATH="$ROOT/configs/uhead/compact.yaml" run_experiment "${group}_compact_${SUFFIX}" compact "$group"
done
