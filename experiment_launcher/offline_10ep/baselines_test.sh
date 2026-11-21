#!/usr/bin/env bash
# Test бейзлайнов и итоговая таблица всех систем на одних test-вопросах (test-половина голов: 1000 eval-видео
# половины для признаков, 500 видео в test). Порог каждой системы — с её validation по on_time_0_4
# (scripts/test_report.py). Шаги:
#   1. test MLP на last_hidden (evaluate_test.sh: предсказания best.pt на test);
#   2. test бейзлайна по уверенности (confidence_baseline.py --part test: results/lora/confidence/<оценка>_conf_off10_test);
#   3. таблица: maxprob, margin, entropy, MLP last_hidden, лучшая голова на признаках офлайн-адаптера (все группы),
#      собственный детектор онлайн-модели (если в ONLINE_RUN есть её оценки на половинах голов — compare_head.sh)
#      и голова на внимании онлайн-модели (mlp/attention_mlp_on10, если для неё посчитан test).
#   CUDA_VISIBLE_DEVICES=1 bash experiment_launcher/offline_10ep/baselines_test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
VAL_FEATURES="${VAL_FEATURES:-$DATA_ROOT/features/group1_half_s1_252x168/val_offline_10ep_best_videos1000}"
ONLINE_RUN="${ONLINE_RUN:-$ROOT/outputs/listen_sft_10ep}"
ONLINE_ADAPTER="${ONLINE_ADAPTER:-epoch_10}"
R="$ROOT/results/lora"
SPLIT="$ROOT/checkpoints/lora/mlp/last_hidden_mlp_off10/split.json"

test_predictions() {  # предсказания best.pt головы на test (evaluate_test.sh), первый найденный файл
  ls "$R/mlp/$1"/predictions_test_best_tau*.json 2>/dev/null | head -n 1 || true
}

echo "== 1. test MLP last_hidden"
[[ -n "$(test_predictions last_hidden_mlp_off10)" ]] || MODE=lora bash "$ROOT/experiment_launcher/evaluate_test.sh" mlp/last_hidden_mlp_off10
echo "== 2. test уверенности ответа"
python "$ROOT/scripts/confidence_baseline.py" --features "$VAL_FEATURES" --split "$SPLIT" --part test --suffix off10

echo "== 3. таблица"
systems=()
for score in maxprob margin entropy; do
  systems+=(--system "уверенность: $score" "$R/confidence/${score}_conf_off10/predictions_validation.json"
            "$R/confidence/${score}_conf_off10_test/predictions_test.json")
done
add_head() {  # $1 — подпись, $2 — id головы в results/lora/mlp
  local test; test="$(test_predictions "$2")"
  if [[ -z "$test" ]]; then
    echo "-- нет test для mlp/$2 (MODE=lora bash experiment_launcher/evaluate_test.sh mlp/$2): пропущена" >&2; return
  fi
  systems+=(--system "$1" "$R/mlp/$2/predictions_validation.json" "$test")
}
add_head "офлайн: MLP last_hidden" last_hidden_mlp_off10
add_head "офлайн: MLP все группы" lm_attention_entropy_vision_confidence_mlp_off10
# Собственный детектор онлайн-модели: её оценки на половинах голов (compare_head.sh, имя — хэш половин).
NAME="head_$(FILE="$SPLIT" python - <<'PY'
import hashlib, json, os
split = json.load(open(os.environ["FILE"], encoding="utf-8"))
print(hashlib.sha1(json.dumps([sorted(split["validation_videos"]), sorted(split["test_videos"])]).encode()).hexdigest()[:8])
PY
)"
ONLINE_VAL="$ONLINE_RUN/predictions_${NAME}_validation_${ONLINE_ADAPTER}.json"
ONLINE_TEST="$ONLINE_RUN/predictions_${NAME}_test_${ONLINE_ADAPTER}.json"
if [[ -f "$ONLINE_VAL" && -f "$ONLINE_TEST" ]]; then
  systems+=(--system "онлайн: свой детектор" "$ONLINE_VAL" "$ONLINE_TEST")
else
  echo "-- нет оценок онлайн-модели $ONLINE_RUN/$ONLINE_ADAPTER на половинах $NAME (compare_head.sh): пропущена" >&2
fi
add_head "онлайн: MLP attention" attention_mlp_on10

python "$ROOT/scripts/test_report.py" --output "$R/test_report_off10.json" "${systems[@]}"
