#!/usr/bin/env bash
# Онлайн-модель с двумя детекторами: собственный (p_k = P(<speak>) против P(<listen>)) и MLP на её внимании
# (mlp/attention_mlp_on10, q_k), два порога. "И" — ответ на первой паре, где p_k > τ1 и q_k > τ2; "ИЛИ" — где хотя бы
# один выше своего порога. Пара порогов выбирается на validation голов (для on_time_0_4 и для каждого бюджета R),
# метрики — на их test (scripts/test_report.py --pair). В таблице рядом — каждый детектор по отдельности.
#
# Нужны: оценки онлайн-модели на половинах голов (compare_head.sh: $ONLINE_RUN/predictions_head_<хэш>_*_<адаптер>.json)
# и test головы на внимании (evaluate_test.sh mlp/attention_mlp_on10). Без GPU, ~1–2 минуты.
#   bash experiment_launcher/online_10ep/two_detectors_test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ONLINE_RUN="${ONLINE_RUN:-$ROOT/outputs/listen_sft_10ep}"
ONLINE_ADAPTER="${ONLINE_ADAPTER:-epoch_10}"
HEAD="${HEAD:-attention_mlp_on10}"
R="$ROOT/results/lora/mlp/$HEAD"
SPLIT="$ROOT/checkpoints/lora/mlp/$HEAD/split.json"

HEAD_TEST="$(ls "$R"/predictions_test_best_tau*.json 2>/dev/null | head -n 1 || true)"
[[ -n "$HEAD_TEST" ]] || { echo "Нет test головы: MODE=lora bash experiment_launcher/evaluate_test.sh mlp/$HEAD" >&2; exit 2; }
NAME="head_$(FILE="$SPLIT" python - <<'PY'
import hashlib, json, os
split = json.load(open(os.environ["FILE"], encoding="utf-8"))
print(hashlib.sha1(json.dumps([sorted(split["validation_videos"]), sorted(split["test_videos"])]).encode()).hexdigest()[:8])
PY
)"
ONLINE_VAL="$ONLINE_RUN/predictions_${NAME}_validation_${ONLINE_ADAPTER}.json"
ONLINE_TEST="$ONLINE_RUN/predictions_${NAME}_test_${ONLINE_ADAPTER}.json"
for f in "$ONLINE_VAL" "$ONLINE_TEST"; do
  [[ -f "$f" ]] || { echo "Нет $f: ONLINE_RUN=$ONLINE_RUN ADAPTER=$ONLINE_ADAPTER bash experiment_launcher/listen/compare_head.sh mlp/$HEAD" >&2; exit 2; }
done

python "$ROOT/scripts/test_report.py" --output "$ROOT/results/lora/test_report_two_detectors.json" \
  --system "онлайн: свой детектор" "$ONLINE_VAL" "$ONLINE_TEST" \
  --system "онлайн: MLP attention" "$R/predictions_validation.json" "$HEAD_TEST" \
  --pair "онлайн: И(свой, MLP)" and "$ONLINE_VAL" "$ONLINE_TEST" "$R/predictions_validation.json" "$HEAD_TEST" \
  --pair "онлайн: ИЛИ(свой, MLP)" or "$ONLINE_VAL" "$ONLINE_TEST" "$R/predictions_validation.json" "$HEAD_TEST"
