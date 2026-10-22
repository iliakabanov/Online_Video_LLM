#!/usr/bin/env bash
# Оценка без дообучения (src/online_vllm/eval/ZERO_SHOT.md) на двух GPU: три системы на выборке prepare.sh, затем сводка.
#   GPU_OFFLINE: qwen_offline, затем mmduet2_offline (~3,5 тыс. токенов на вопрос)
#   GPU_ONLINE:  mmduet2_online (~7,9 тыс. токенов на вопрос)
# Результаты: $OUT/<система>/{answers.jsonl,meta.json}, сводка в $OUT (report.md, summary.json, графики), логи в
# $OUT/logs. Прерванный запуск продолжается той же командой: посчитанные вопросы пропускаются.
#   DATA_ROOT=/data bash experiment_launcher/zero_shot/run.sh
#   LIMIT=20 OUT=results/zero_shot_pilot DATA_ROOT=/data bash experiment_launcher/zero_shot/run.sh   # пилот
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/PATH/TO/DATA}"
QUESTIONS="${QUESTIONS:-processed/clevrer_proactive/val_group1_features_eval500.jsonl}"
SEED="${SEED:-0}"
OUT="${OUT:-$ROOT/results/zero_shot}"
GPU_OFFLINE="${GPU_OFFLINE:-cuda:0}"
GPU_ONLINE="${GPU_ONLINE:-cuda:1}"
LIMIT="${LIMIT:-}"
[[ "$DATA_ROOT" != "/PATH/TO/DATA" ]] || { echo "Задайте DATA_ROOT." >&2; exit 2; }
[[ -f "$DATA_ROOT/$QUESTIONS" ]] || { echo "Нет выборки $DATA_ROOT/$QUESTIONS: сначала prepare.sh." >&2; exit 2; }
export DATA_ROOT
cd "$ROOT"

common=(--questions "$QUESTIONS" --seed "$SEED")
[[ -z "$LIMIT" ]] || common+=(--limit "$LIMIT")
mkdir -p "$OUT/logs"

{
  python scripts/zero_shot_probe.py --system qwen_offline --device "$GPU_OFFLINE" --output-dir "$OUT/qwen_offline" "${common[@]}"
  python scripts/zero_shot_probe.py --system mmduet2_offline --device "$GPU_OFFLINE" --output-dir "$OUT/mmduet2_offline" "${common[@]}"
} > "$OUT/logs/offline.log" 2>&1 &
offline=$!
python scripts/zero_shot_probe.py --system mmduet2_online --device "$GPU_ONLINE" --output-dir "$OUT/mmduet2_online" \
  "${common[@]}" > "$OUT/logs/online.log" 2>&1 &
online=$!
echo "Идёт счёт; логи: $OUT/logs/offline.log, $OUT/logs/online.log"
status=0
wait "$offline" || { echo "Офлайн-системы упали, см. $OUT/logs/offline.log" >&2; status=1; }
wait "$online" || { echo "Онлайн-система упала, см. $OUT/logs/online.log" >&2; status=1; }
[[ "$status" == 0 ]] || exit 1

python scripts/compare_zero_shot.py --runs "$OUT/qwen_offline" "$OUT/mmduet2_offline" "$OUT/mmduet2_online" \
  --output-dir "$OUT" --seed "$SEED"
