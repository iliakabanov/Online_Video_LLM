#!/usr/bin/env bash
# Отдельный модуль решения (separate/decision.py): своя VLM, свежая LoRA, токен <act> после каждой пары и бинарная
# голова; отвечает M_off. MODEL=llava — SB-0.5B (LLaVA-OneVision-0.5B), MODEL=qwen — SB-3B (Qwen2.5-VL-3B без адаптера
# M_off). Данные, половины validation/test и ответы M_off — те же, что у голов офлайн-адаптера
# (experiment_launcher/offline_10ep): train — первые 1000 видео половины для признаков, validation и test — вопросы
# файлов предсказаний головы MLP last_hidden (results/lora/mlp/last_hidden_mlp_off10).
#
#   1. обучение (scripts/train_separate.py, 10 эпох, best/ по AUROC в окне ±2) -> $RUN (веса, не в git); прерванный
#      прогон (в $RUN есть separate.json, нет complete) тот же запуск продолжает с последней эпохи (--resume)
#   2. test-половина головы при τ с validation (scripts/evaluate_separate.py) -> $RUN/head_test_<adapter>.json
#   3. метрики и предсказания без весов -> results/separate/<имя прогона>/
#   4. test-таблица: бейзлайны по уверенности, головы офлайн-адаптера, онлайн-модель и все прогоны из
#      results/separate/ (scripts/test_report.py) -> results/separate/test_report.json
#
#   CUDA_VISIBLE_DEVICES=0 MODEL=llava bash experiment_launcher/separate/run.sh
#   CUDA_VISIBLE_DEVICES=0 MODEL=qwen TRAIN_ARGS="--gradient-checkpointing" bash experiment_launcher/separate/run.sh
# Переменные: MODEL (llava | qwen), RUN (каталог прогона; по умолчанию outputs/separate/sb05b или sb3b), ADAPTER
# (best | last | epoch_<n>), OVERWRITE=1 (обучить заново), TRAIN_ARGS (флаги train_separate.py, например
# "--frames-per-pair 1 --tokens-side 7" или "--tags sigmoid --seed 1"), ONLINE_RUN / ONLINE_ADAPTER (онлайн-модель
# для таблицы).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-llava}"
# LLaVA-OV по умолчанию: оба кадра пары по 7 x 7 = 49 токенов (98 на пару), <act> после пары — те же кадры, что у
# SB-3B (его токен склеивает два кадра пары, 54 на пару при 252x168), разрешение токена как у него. Другие варианты —
# TRAIN_ARGS="--frames-per-pair 1" (только последний кадр, 49 на пару) или "--tokens-side 14" (родные 196 на кадр);
# флаги TRAIN_ARGS идут позже и перекрывают эти. Проектор LLaVA-OV по умолчанию заморожен, обучить его целиком —
# TRAIN_ARGS="--train-projector" (лучше с отдельным RUN, например outputs/separate/sb05b_proj).
case "$MODEL" in
  llava) MODEL_CONFIG="$ROOT/configs/model/llava_ov_0.5b_separate.yaml"; DEFAULT_RUN=sb05b
         DEFAULT_ARGS=(--frames-per-pair 2 --tokens-side 7) ;;
  qwen) MODEL_CONFIG="$ROOT/configs/model/qwen_3b_lora_sft.yaml"; DEFAULT_RUN=sb3b; DEFAULT_ARGS=() ;;
  *) echo "MODEL: llava или qwen" >&2; exit 2 ;;
esac
RUN="${RUN:-$ROOT/outputs/separate/$DEFAULT_RUN}"
ADAPTER="${ADAPTER:-best}"
OVERWRITE="${OVERWRITE:-0}"
ONLINE_RUN="${ONLINE_RUN:-$ROOT/outputs/listen_sft_10ep}"
ONLINE_ADAPTER="${ONLINE_ADAPTER:-epoch_10}"
R="$ROOT/results/lora"
REFERENCE="$R/mlp/last_hidden_mlp_off10"
OUT="$ROOT/results/separate/$(basename "$RUN")"
read -ra train_args <<< "${TRAIN_ARGS:-}"

test_predictions() {  # предсказания best.pt головы на test (evaluate_test.sh), первый найденный файл
  ls "$R/mlp/$1"/predictions_test_best_tau*.json 2>/dev/null | head -n 1 || true
}
REFERENCE_TEST="$(test_predictions last_hidden_mlp_off10)"
[[ -n "$REFERENCE_TEST" ]] || { echo "Нет test головы mlp/last_hidden_mlp_off10 (evaluate_test.sh)" >&2; exit 2; }

echo "== 1. обучение ($MODEL) -> $RUN"
if [[ -f "$RUN/complete" && "$OVERWRITE" != "1" ]]; then
  echo "-- $RUN обучен полностью (OVERWRITE=1 — обучить заново)"
elif [[ -f "$RUN/separate.json" && "$OVERWRITE" != "1" ]]; then
  # Прерванный прогон: продолжить с last/ с его же настройками (TRAIN_ARGS здесь не действуют, кроме --epochs).
  echo "-- $RUN прерван: продолжение с последней сохранённой эпохи"
  python "$ROOT/scripts/train_separate.py" --output "$RUN" --resume --data-root "$DATA_ROOT" --device "$DEVICE" \
    ${train_args[@]+"${train_args[@]}"}
else
  extra=()
  [[ "$OVERWRITE" == "1" ]] && extra+=(--overwrite)
  python "$ROOT/scripts/train_separate.py" --model-config "$MODEL_CONFIG" --output "$RUN" --data-root "$DATA_ROOT" \
    --device "$DEVICE" --validation-from "$REFERENCE/predictions_validation.json" ${extra[@]+"${extra[@]}"} \
    ${DEFAULT_ARGS[@]+"${DEFAULT_ARGS[@]}"} ${train_args[@]+"${train_args[@]}"}
fi

echo "== 2. test-половина головы: $(basename "$REFERENCE_TEST")"
python "$ROOT/scripts/evaluate_separate.py" --run "$RUN" --adapter "$ADAPTER" --name head_test \
  --records-from "$REFERENCE_TEST" --data-root "$DATA_ROOT" --device "$DEVICE"

echo "== 3. метрики без весов -> $OUT"
EPOCH="$(FILE="$RUN/$ADAPTER/sft_metadata.json" python -c 'import json, os; print(json.load(open(os.environ["FILE"]))["epoch"])')"
mkdir -p "$OUT"
for file in separate.json split.json sft_config.json model.yaml validation_metrics.jsonl train_steps.jsonl; do
  cp "$RUN/$file" "$OUT/"
done
cp "$RUN/$ADAPTER/sft_metadata.json" "$OUT/${ADAPTER}_metadata.json"
cp "$RUN/predictions_validation_epoch_${EPOCH}.json" "$OUT/predictions_validation.json"  # эпоха адаптера
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
for dir in "$ROOT"/results/separate/*/; do  # все прогоны отдельного модуля (SB-0.5B, SB-3B, ...)
  name="$(basename "$dir")"
  if [[ -f "$dir/predictions_validation.json" && -f "$dir/predictions_test_${ADAPTER}.json" ]]; then
    systems+=(--system "SB: $name (отвечает M_off)" "$dir/predictions_validation.json" "$dir/predictions_test_${ADAPTER}.json")
  fi
done
python "$ROOT/scripts/test_report.py" --output "$ROOT/results/separate/test_report.json" "${systems[@]}"
