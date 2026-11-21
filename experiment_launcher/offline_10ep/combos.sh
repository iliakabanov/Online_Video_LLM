#!/usr/bin/env bash
# MLP-голова (StateHead) на комбинациях групп признаков нового офлайн-адаптера: две сильные группы (lm, attention),
# с entropy и все пять — 4 запуска, results/lora/mlp/<группы через _>_mlp_off10. На одиночных группах архитектуры
# (mlp, transformer, compact) не различались, поэтому комбинации — только на MLP. Настройки — setup.sh.
#   CUDA_VISIBLE_DEVICES=1 bash experiment_launcher/offline_10ep/combos.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup.sh"
COMBOS=(
  "lm attention"
  "lm attention entropy"
  "attention entropy"
  "lm attention entropy vision confidence"
)
for combo in "${COMBOS[@]}"; do
  read -r -a groups <<< "$combo"
  name="$(IFS=_; echo "${groups[*]}")"
  echo "== mlp: ${groups[*]}"
  run_experiment "${name}_mlp_${SUFFIX}" mlp "${groups[@]}"
done
