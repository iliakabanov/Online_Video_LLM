#!/usr/bin/env bash
# Оценка обученного отдельного модуля решения (обучение — отдельно: scripts/train_separate.py). Вопросы и ответы M_off —
# те же, что у голов офлайн-адаптера: validation и test — половины головы MLP last_hidden
# (results/lora/mlp/last_hidden_mlp_off10), на validation-половине модуль уже оценён при обучении.
#
#   1. порог τ на validation: эпоха адаптера, критерий CRITERION (on_time_0_4: ответ через 0..4 пары после целевой,
#      не раньше; или on_target_pm2) -> $RUN/thresholds_validation_<adapter>.json
#   2. test-половина головы при этом τ и при τ = 0.5 (scripts/evaluate_separate.py) -> $RUN/head_test_<adapter>.json
#   3. метрики и предсказания без весов -> results/separate/<имя прогона>/
#   4. test-таблица: бейзлайны по уверенности, головы офлайн-адаптера, онлайн-модель и все прогоны из
#      results/separate/ (scripts/test_report.py) -> results/separate/test_report.json
#
#   CUDA_VISIBLE_DEVICES=0 bash experiment_launcher/separate/run.sh outputs/separate/sb05b
#   ADAPTER=epoch_7 bash experiment_launcher/separate/run.sh outputs/separate/sb05b_proj
# Переменные: ADAPTER (best | last | epoch_<n>), CRITERION, ONLINE_RUN / ONLINE_ADAPTER (онлайн-модель для таблицы),
# EVAL_ARGS (флаги evaluate_separate.py, например "--encoder-frames 8 --eval-batch-size 2").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
DEVICE="${DEVICE:-cuda}"
ADAPTER="${ADAPTER:-best}"
CRITERION="${CRITERION:-on_time_0_4}"
ONLINE_RUN="${ONLINE_RUN:-$ROOT/outputs/listen_sft_10ep}"
ONLINE_ADAPTER="${ONLINE_ADAPTER:-epoch_10}"
R="$ROOT/results/lora"
REFERENCE="$R/mlp/last_hidden_mlp_off10"
read -ra eval_args <<< "${EVAL_ARGS:-}"

[[ $# -eq 1 ]] || { echo "Укажите каталог прогона train_separate.py, например outputs/separate/sb05b" >&2; exit 2; }
RUN="$1"
case "$CRITERION" in on_time_0_4|on_target_pm2) ;; *) echo "CRITERION: on_time_0_4 или on_target_pm2" >&2; exit 2 ;; esac
[[ -f "$RUN/$ADAPTER/head.safetensors" ]] || { echo "Нет $RUN/$ADAPTER: прогон не обучен (train_separate.py)" >&2; exit 2; }
if [[ ! -f "$RUN/complete" ]]; then
  echo "!! $RUN обучен не до конца (нет complete): продолжить — python scripts/train_separate.py --output $RUN --resume" >&2
  [[ "$ADAPTER" == "best" ]] && { echo "   best/ ещё может смениться; оценить промежуточную эпоху — ADAPTER=epoch_<n>" >&2; exit 2; }
fi
OUT="$ROOT/results/separate/$(basename "$RUN")"
test_predictions() {  # предсказания best.pt головы на test (evaluate_test.sh), первый найденный файл
  ls "$R/mlp/$1"/predictions_test_best_tau*.json 2>/dev/null | head -n 1 || true
}
REFERENCE_TEST="$(test_predictions last_hidden_mlp_off10)"
[[ -n "$REFERENCE_TEST" ]] || { echo "Нет test головы mlp/last_hidden_mlp_off10 (evaluate_test.sh)" >&2; exit 2; }
EPOCH="$(FILE="$RUN/$ADAPTER/sft_metadata.json" python -c 'import json, os; print(json.load(open(os.environ["FILE"]))["epoch"])')"
VALIDATION="$RUN/predictions_validation_epoch_${EPOCH}.json"  # validation-половина головы, эпоха адаптера

echo "== 1. порог на validation ($RUN/$ADAPTER, эпоха $EPOCH, критерий $CRITERION)"
python -m online_vllm.eval.thresholds --predictions "$VALIDATION" \
  --output "$RUN/thresholds_validation_${ADAPTER}.json" --thresholds 0.5
TAU="$(FILE="$RUN/thresholds_validation_${ADAPTER}.json" CRITERION="$CRITERION" python - <<'PY'
import json, os
best = json.load(open(os.environ["FILE"], encoding="utf-8"))["best_thresholds"][os.environ["CRITERION"]]
print(f"{best['tau']:.6g} (validation {os.environ['CRITERION']} = {100 * best[os.environ['CRITERION']]:.1f}%)")
PY
)"
echo "-- τ = $TAU"

echo "== 2. test-половина головы: $(basename "$REFERENCE_TEST")"
python "$ROOT/scripts/evaluate_separate.py" --run "$RUN" --adapter "$ADAPTER" --name head_test \
  --records-from "$REFERENCE_TEST" --tau-from "$VALIDATION" --tau-criterion "$CRITERION" \
  --data-root "$DATA_ROOT" --device "$DEVICE" ${eval_args[@]+"${eval_args[@]}"}

echo "== 3. метрики без весов -> $OUT"
mkdir -p "$OUT"
for file in separate.json split.json sft_config.json model.yaml validation_metrics.jsonl train_steps.jsonl; do
  cp "$RUN/$file" "$OUT/"
done
cp "$RUN/$ADAPTER/sft_metadata.json" "$OUT/${ADAPTER}_metadata.json"
cp "$VALIDATION" "$OUT/predictions_validation.json"
cp "$RUN/thresholds_validation_${ADAPTER}.json" "$OUT/thresholds_validation.json"
cp "$RUN/head_test_${ADAPTER}.json" "$OUT/test_${ADAPTER}.json"
cp "$RUN/predictions_head_test_${ADAPTER}.json" "$OUT/predictions_test_${ADAPTER}.json"

echo "== 4. test-таблица"
systems=()
for score in maxprob margin entropy; do
  systems+=(--system "уверенность: $score" "$R/confidence/${score}_conf_off10/predictions_validation.json"
            "$R/confidence/${score}_conf_off10_test/predictions_test.json")
done
add_head() {  # $1 — подпись, $2 — id головы в results/lora/mlp
  local test; test="$(test_predictions "$2")"
  if [[ -z "$test" ]]; then echo "-- нет test для mlp/$2: пропущена" >&2; return; fi
  systems+=(--system "$1" "$R/mlp/$2/predictions_validation.json" "$test")
}
add_head "A: MLP last_hidden (M_off)" last_hidden_mlp_off10
add_head "A: MLP все группы (M_off)" lm_attention_entropy_vision_confidence_mlp_off10
# Онлайн-модель B: её оценки на половинах голов (compare_head.sh, имя — хэш половин из split.json головы).
SPLIT="$ROOT/checkpoints/lora/mlp/last_hidden_mlp_off10/split.json"
if [[ -f "$SPLIT" ]]; then
  NAME="head_$(FILE="$SPLIT" python - <<'PY'
import hashlib, json, os
split = json.load(open(os.environ["FILE"], encoding="utf-8"))
print(hashlib.sha1(json.dumps([sorted(split["validation_videos"]), sorted(split["test_videos"])]).encode()).hexdigest()[:8])
PY
)"
  ONLINE_VAL="$ONLINE_RUN/predictions_${NAME}_validation_${ONLINE_ADAPTER}.json"
  ONLINE_TEST="$ONLINE_RUN/predictions_${NAME}_test_${ONLINE_ADAPTER}.json"
  if [[ -f "$ONLINE_VAL" && -f "$ONLINE_TEST" ]]; then
    systems+=(--system "B: <listen>/<speak> (M_on)" "$ONLINE_VAL" "$ONLINE_TEST")
  else
    echo "-- нет оценок онлайн-модели $ONLINE_RUN/$ONLINE_ADAPTER на половинах $NAME (compare_head.sh): пропущена" >&2
  fi
else
  echo "-- нет $SPLIT: онлайн-модель пропущена" >&2
fi
for dir in "$ROOT"/results/separate/*/; do  # все оценённые прогоны отдельного модуля (SB-0.5B, SB-3B, ...)
  name="$(basename "$dir")"
  if [[ -f "$dir/predictions_validation.json" && -f "$dir/predictions_test_${ADAPTER}.json" ]]; then
    systems+=(--system "SB: $name (отвечает M_off)" "$dir/predictions_validation.json" "$dir/predictions_test_${ADAPTER}.json")
  fi
done
python "$ROOT/scripts/test_report.py" --criterion "$CRITERION" --output "$ROOT/results/separate/test_report.json" \
  "${systems[@]}"
