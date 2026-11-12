#!/usr/bin/env bash
# MLP-голова (StateHead) на комбинациях групп признаков онлайн-модели: собственный детектор модели (listen_token_logit)
# вместе с вниманием и далее — 4 запуска, results/lora/mlp/<группы через _>_mlp_on10. Настройки — setup.sh.
#   1. все группы: lm, attention, entropy, vision, confidence, listen_token_logit
#   2. listen_token_logit + attention
#   3. listen_token_logit + attention + entropy
#   4. listen_token_logit + attention + entropy + lm
#   CUDA_VISIBLE_DEVICES=1 bash experiment_launcher/online_10ep/combos.sh
#   python experiment_launcher/offline_10ep/summary.py --suffix on10
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup.sh"
COMBOS=(
  "lm attention entropy vision confidence listen_token_logit"
  "listen_token_logit attention"
  "listen_token_logit attention entropy"
  "listen_token_logit attention entropy lm"
)
for combo in "${COMBOS[@]}"; do
  read -r -a groups <<< "$combo"
  name="$(IFS=_; echo "${groups[*]}")"
  echo "== mlp: ${groups[*]}"
  run_experiment "${name}_mlp_${SUFFIX}" mlp "${groups[@]}"
done
