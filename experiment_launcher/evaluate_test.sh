#!/usr/bin/env bash
# Test-половина обученных детекторов при пороге, выбранном на validation. Для каждого запуска results/<mode>/<arch>/<id>:
#   1. по results.json (метрики best.pt на validation при каждом пороге eval.thresholds) выбирается τ с наибольшим
#      SELECT_METRIC (по умолчанию on_target_pm2: срабатывание не дальше ±2 пар от целевой); выбор пишется
#      в selection.json;
#   2. uhead.evaluate оценивает best.pt из checkpoints/<mode>/<arch>/<id> на test-видео из split.json при этом τ:
#      test_best_tau<τ>.json и predictions_test_best_tau<τ>.json в каталоге результатов.
# Сводная таблица validation / test: python scripts/summarize_results.py
#
#   MODE=lora bash experiment_launcher/evaluate_test.sh                      # все запуски results/lora/
#   MODE=lora bash experiment_launcher/evaluate_test.sh mlp/01_lm mlp/02_attention
#   SELECT_METRIC=timely_correct_pm2 MODE=original bash experiment_launcher/evaluate_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"   # пути признаков в train.yaml запусков абсолютные
DEVICE="${DEVICE:-cuda}"
MODE="${MODE:-}"
SELECT_METRIC="${SELECT_METRIC:-on_target_pm2}"
case "$MODE" in original|lora) ;; *) echo "Задайте MODE=original или MODE=lora (сейчас: '$MODE')." >&2; exit 2 ;; esac

runs=("$@")
if [[ ${#runs[@]} -eq 0 ]]; then
  for path in "$ROOT/results/$MODE"/*/*/results.json; do
    [[ -e "$path" ]] || continue
    run="${path%/results.json}"; runs+=("${run#"$ROOT/results/$MODE/"}")
  done
fi
[[ ${#runs[@]} -gt 0 ]] || { echo "В results/$MODE нет запусков." >&2; exit 2; }

for run in "${runs[@]}"; do
  result_dir="$ROOT/results/$MODE/$run" checkpoint_dir="$ROOT/checkpoints/$MODE/$run"
  [[ -f "$result_dir/results.json" ]] || { echo "Нет $result_dir/results.json" >&2; exit 2; }
  [[ -f "$checkpoint_dir/best.pt" ]] || { echo "Нет $checkpoint_dir/best.pt" >&2; exit 2; }
  tau="$(RESULT="$result_dir" METRIC="$SELECT_METRIC" python - <<'PY'
import json, os, sys
from pathlib import Path
result = Path(os.environ["RESULT"])
metric = os.environ["METRIC"]
thresholds = json.loads((result / "results.json").read_text(encoding="utf-8"))["validation"]["thresholds"]
scores = {tau: values.get(metric) for tau, values in thresholds.items()}
if any(value is None for value in scores.values()):
    sys.exit(f"{metric} нет среди метрик порогов results.json")
# При равенстве — больший τ: меньше преждевременных ответов при той же метрике.
tau = max(scores, key=lambda t: (scores[t], float(t)))
(result / "selection.json").write_text(json.dumps(
    {"metric": metric, "threshold": float(tau), "validation_value": scores[tau],
     "validation_by_threshold": scores}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(tau)
PY
)"
  echo "== $MODE/$run: τ = $tau ($SELECT_METRIC на validation)"
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m uhead.evaluate --run "$checkpoint_dir" \
    --results "$result_dir" --threshold "$tau" --data-root "$DATA_ROOT" --device "$DEVICE"
done
