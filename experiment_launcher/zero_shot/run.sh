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
OUT="$(cd "$OUT" && pwd)"
SYSTEMS=(qwen_offline mmduet2_offline mmduet2_online)
total="$(grep -c . "$DATA_ROOT/$QUESTIONS")"
[[ -z "$LIMIT" ]] || (( LIMIT >= total )) || total="$LIMIT"

echo "Вопросов: $total. Результаты (по строке на вопрос, дописываются по мере счёта):"
for system in "${SYSTEMS[@]}"; do echo "  $OUT/$system/answers.jsonl"; done
echo "Логи (там же полоса прогресса каждой системы): $OUT/logs/offline.log, $OUT/logs/online.log"
echo "Сводка после счёта: $OUT/report.md, $OUT/summary.json, графики *.png там же"

progress() {  # посчитано вопросов у каждой системы: число строк её answers.jsonl
  local line="" system n
  for system in "${SYSTEMS[@]}"; do
    n=0
    if [[ -f "$OUT/$system/answers.jsonl" ]]; then n="$(grep -c . "$OUT/$system/answers.jsonl" || true)"; fi
    line+="$system $n/$total   "
  done
  echo "$(date +%H:%M:%S)  $line"
}

{
  python scripts/zero_shot_probe.py --system qwen_offline --device "$GPU_OFFLINE" --output-dir "$OUT/qwen_offline" "${common[@]}"
  python scripts/zero_shot_probe.py --system mmduet2_offline --device "$GPU_OFFLINE" --output-dir "$OUT/mmduet2_offline" "${common[@]}"
} > "$OUT/logs/offline.log" 2>&1 &
offline=$!
python scripts/zero_shot_probe.py --system mmduet2_online --device "$GPU_ONLINE" --output-dir "$OUT/mmduet2_online" \
  "${common[@]}" > "$OUT/logs/online.log" 2>&1 &
online=$!
# Прогресс, пока идёт счёт: в терминале строка обновляется на месте, при выводе в файл — новая строка раз в минуту.
while kill -0 "$offline" 2>/dev/null || kill -0 "$online" 2>/dev/null; do
  if [[ -t 1 ]]; then printf '\r%s' "$(progress)"; sleep "${PROGRESS_EVERY:-10}"; else progress; sleep "${PROGRESS_EVERY:-60}"; fi
done
if [[ -t 1 ]]; then echo; fi
progress
status=0
wait "$offline" || { echo "Офлайн-системы упали, см. $OUT/logs/offline.log" >&2; status=1; }
wait "$online" || { echo "Онлайн-система упала, см. $OUT/logs/online.log" >&2; status=1; }
[[ "$status" == 0 ]] || exit 1

python scripts/compare_zero_shot.py --runs "$OUT/qwen_offline" "$OUT/mmduet2_offline" "$OUT/mmduet2_online" \
  --output-dir "$OUT" --seed "$SEED" > "$OUT/logs/compare.log" 2>&1 \
  || { echo "Сводка упала, см. $OUT/logs/compare.log" >&2; exit 1; }
cat "$OUT/report.md"
echo
echo "Готово. Сводка: $OUT/report.md, все метрики: $OUT/summary.json"
echo "Графики: $OUT/answer_curves.png, $OUT/readiness_curves_mmduet2_online.png"
echo "Ответы систем: $OUT/<система>/answers.jsonl"
