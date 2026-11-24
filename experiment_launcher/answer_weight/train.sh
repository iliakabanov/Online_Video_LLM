#!/usr/bin/env bash
# Онлайн-SFT с другим весом лосса ответа: loss = слоты + W × ответ. Всё остальное — как у outputs/listen_sft_10ep
# (online.json, sft_config.json): свежая LoRA с базовой Qwen (--init-adapter none), split и model.yaml офлайн-прогона
# outputs/offline_10ep, те же 1000 train-видео, метки hard, 10 эпох, lr 1e-4, batch 2 × accumulation 2, seed 0,
# best/ — по bce_pm2. Выбор веса — experiment_launcher/answer_weight/choose_weight.py.
#
#   CUDA_VISIBLE_DEVICES=0 bash experiment_launcher/answer_weight/train.sh 0.5
#   CUDA_VISIBLE_DEVICES=1 bash experiment_launcher/answer_weight/train.sh 0.25
#   CUDA_VISIBLE_DEVICES=2 bash experiment_launcher/answer_weight/train.sh 0.1
set -euo pipefail
W="${1:?Укажите вес лосса ответа, например 0.5}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
EPOCHS="${EPOCHS:-10}"
OUTPUT="outputs/listen_sft_10ep_aw$W"

python scripts/train_online_sft.py \
  --init-run outputs/offline_10ep --init-adapter none \
  --tags hard --answer-weight "$W" --selection bce_pm2 \
  --epochs "$EPOCHS" --lr 1e-4 --batch-size 2 --gradient-accumulation 2 --no-gradient-checkpointing --seed 0 \
  --output "$OUTPUT" 2>&1 | tee -a "outputs/listen_sft_10ep_aw$W.log"
