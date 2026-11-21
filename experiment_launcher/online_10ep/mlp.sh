#!/usr/bin/env bash
# MLP-голова (StateHead) на каждой группе признаков онлайн-модели по отдельности: lm, attention, entropy, vision,
# confidence и listen_token_logit (собственный детектор модели: голова на нём одном — его перекалибровка, AUROC в окне
# ±2 должен быть около её собственного) — 6 запусков, results/lora/mlp/<группа>_mlp_on10. Настройки — setup.sh.
#   CUDA_VISIBLE_DEVICES=1 bash experiment_launcher/online_10ep/mlp.sh
#   python experiment_launcher/offline_10ep/summary.py --suffix on10
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup.sh"
for group in lm attention entropy vision confidence listen_token_logit; do
  echo "== mlp: $group"
  run_experiment "${group}_mlp_${SUFFIX}" mlp "$group"
done
