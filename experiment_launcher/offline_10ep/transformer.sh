#!/usr/bin/env bash
# Causal Transformer-голова (UHead, configs/uhead/default.yaml) на каждой группе признаков нового офлайн-адаптера по
# отдельности: lm, attention, entropy, vision, confidence — 5 запусков, results/lora/transformer/<группа>_base_off10.
# Настройки — setup.sh.
#   CUDA_VISIBLE_DEVICES=1 bash experiment_launcher/offline_10ep/transformer.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup.sh"
for group in "${GROUPS_TO_RUN[@]}"; do
  echo "== transformer: $group"
  UHEAD_CONFIG_PATH="$ROOT/configs/uhead/default.yaml" run_experiment "${group}_base_${SUFFIX}" transformer "$group"
done
