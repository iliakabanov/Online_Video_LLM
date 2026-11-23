#!/usr/bin/env bash
# Детектор готовности по первым k слоям языковой модели: сколько слоёв нужно прогнать, чтобы решить, пора ли отвечать.
# Для каждого k из START..END с шагом STEP (END входит всегда) — голова на признаках только слоёв 1..k (uhead
# max_layer): lm — извлечённые слои не глубже k (12, 18, 24, 36), attention и entropy — головы слоёв 1..k; группы с
# выхода модели (confidence, listen_token_logit, last_hidden) есть только при k = последнему слою, vision от k не
# зависит. Группа, которой при данном k нет (lm при k < 12), выпадает из набора этого k.
#
#   CUDA_VISIBLE_DEVICES=0 bash experiment_launcher/depth/run.sh START END STEP
#   CUDA_VISIBLE_DEVICES=0 bash experiment_launcher/depth/run.sh 4 36 4                 # k = 4, 8, ..., 36
#   MODEL=online TEST=1 bash experiment_launcher/depth/run.sh 2 36 2
#
# Переменные:
#   MODEL         offline (признаки офлайн-адаптера, offline_10ep/setup.sh) | online (онлайн-модели, online_10ep/setup.sh)
#   DEPTH_GROUPS  группы признаков, по умолчанию "lm attention entropy"
#   ARCH          mlp | transformer | compact (по умолчанию mlp)
#   TEST=1        ещё и test: каждая голова на test-половине (evaluate_test.sh) и общая test-таблица по всем k
#                 (scripts/test_report.py, порог — с validation по on_time_0_4)
#   OVERWRITE=1   переобучить уже обученные k (иначе они пропускаются: прерванный перебор можно запустить заново)
#   а также переменные setup.sh / common.sh (TRAIN_FEATURES, VAL_FEATURES, EPOCHS, SELECTION, DEVICE, ...).
#
# Запуски: results/lora/<ARCH>/depth<k>_<группы>_<ARCH>_<off10|on10> (веса — checkpoints/lora/...).
# Сводка validation (и test при TEST=1): results/lora/depth/<группы>_<ARCH>_<off10|on10>{,_test}.json.
set -euo pipefail

[[ $# -eq 3 ]] || { echo "Использование: bash $0 START END STEP (k — слои 1..k языковой модели)" >&2; exit 2; }
START="$1" END="$2" STEP="$3"
for value in "$START" "$END" "$STEP"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "START, END и STEP — целые числа >= 1, получено '$value'" >&2; exit 2; }
done
(( START <= END )) || { echo "START ($START) больше END ($END)" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-offline}"
case "$MODEL" in
  offline) source "$HERE/../offline_10ep/setup.sh" ;;
  online) source "$HERE/../online_10ep/setup.sh" ;;
  *) echo "MODEL: offline или online (сейчас '$MODEL')" >&2; exit 2 ;;
esac
ARCH="${ARCH:-mlp}"
TEST="${TEST:-0}"
read -r -a requested <<< "${DEPTH_GROUPS:-lm attention entropy}"
NAME="$(IFS=_; echo "${requested[*]}")_${ARCH}_${SUFFIX}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Число слоёв языковой модели — из признаков; END не глубже него.
LAYERS="$(META="$TRAIN_FEATURES/meta.json" python -c 'import json, os; print(json.load(open(os.environ["META"]))["layers"])')"
(( END <= LAYERS )) || { echo "END ($END) больше числа слоёв языковой модели ($LAYERS)" >&2; exit 2; }
depths=()
for (( k = START; k <= END; k += STEP )); do depths+=("$k"); done
[[ "${depths[-1]}" == "$END" ]] || depths+=("$END")
echo "== $MODEL ($SUFFIX), $ARCH, группы: ${requested[*]}; k: ${depths[*]} из $LAYERS слоёв"

run_ids=() run_depths=()
for k in "${depths[@]}"; do
  # Группы, которые есть при слоях 1..k (uhead.data.feature_dims).
  available="$(META="$TRAIN_FEATURES/meta.json" K="$k" GROUPS_REQUESTED="${requested[*]}" python - <<'PY'
import json, os
from uhead.data import feature_dims
dims = feature_dims(json.load(open(os.environ["META"], encoding="utf-8")), int(os.environ["K"]))
print(" ".join(group for group in os.environ["GROUPS_REQUESTED"].split() if group in dims))
PY
)"
  read -r -a groups <<< "$available"
  if [[ ${#groups[@]} -eq 0 ]]; then
    echo "-- k = $k: ни одной из групп ${requested[*]} при слоях 1..$k нет, пропущено" >&2
    continue
  fi
  [[ "${groups[*]}" == "${requested[*]}" ]] || echo "-- k = $k: только ${groups[*]} (остальных групп при слоях 1..$k нет)"
  id="$(printf 'depth%02d' "$k")_${NAME}"
  if [[ -f "$ROOT/results/$MODE/$ARCH/$id/results.json" && "$OVERWRITE" != "1" ]]; then
    echo "== k = $k: $id уже обучен, пропущен (OVERWRITE=1 — переобучить)"
  else
    echo "== k = $k: $ARCH на ${groups[*]}, слои 1..$k -> $id"
    MAX_LAYER="$k" run_experiment "$id" "$ARCH" "${groups[@]}"
  fi
  run_ids+=("$id") run_depths+=("$k")
done
[[ ${#run_ids[@]} -gt 0 ]] || { echo "Ни одного запуска" >&2; exit 2; }

OUT="$ROOT/results/$MODE/depth"
mkdir -p "$OUT"
echo "== сводка validation -> $OUT/$NAME.json"
RESULTS="$ROOT/results/$MODE/$ARCH" IDS="${run_ids[*]}" DEPTHS="${run_depths[*]}" LAYERS="$LAYERS" \
  OUTPUT="$OUT/$NAME.json" python - <<'PY'
import json, os
from pathlib import Path
rows = []
for run, k in zip(os.environ["IDS"].split(), map(int, os.environ["DEPTHS"].split())):
    path = Path(os.environ["RESULTS"]) / run
    results = json.loads((path / "results.json").read_text(encoding="utf-8"))
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    validation = results["validation"]
    on_time = validation["best_thresholds"]["on_time_0_4"]
    latency = (validation.get("constrained_latency") or {}).get("R0.05_A0.85") or {}
    rows.append({"k": k, "layers_share": k / int(os.environ["LAYERS"]), "run": run, "groups": config["feature_groups"],
                 "best_epoch": results["best_epoch"], "auroc_pm2": results["best_validation_auroc_pm2"],
                 "on_time_0_4": on_time["on_time_0_4"], "tau": on_time["tau"],
                 "latency_mean_R0.05_A0.85": latency.get("latency_mean") if latency.get("feasible") else None})
Path(os.environ["OUTPUT"]).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("validation (порог — на ней же, по on_time_0_4); задержка — в парах, R <= 5%, точность >= 0.85")
print(f"{'k':>3s} {'доля слоёв':>10s} {'AUROC±2':>8s} {'вовремя':>8s} {'задержка':>9s} {'эпоха':>6s}  группы")
for row in rows:
    latency = row["latency_mean_R0.05_A0.85"]
    print(f"{row['k']:3d} {row['layers_share']:10.2f} {row['auroc_pm2']:8.4f} {100 * row['on_time_0_4']:7.1f}% "
          f"{'-' if latency is None else f'{latency:.2f}':>9s} {row['best_epoch']:6d}  {' '.join(row['groups'])}")
PY

if [[ "$TEST" == "1" ]]; then
  test_runs=()
  for id in "${run_ids[@]}"; do
    if ls "$ROOT/results/$MODE/$ARCH/$id"/predictions_test_best_tau*.json >/dev/null 2>&1 && [[ "$OVERWRITE" != "1" ]]; then
      echo "== test: $id уже оценён"
    else
      test_runs+=("$ARCH/$id")
    fi
  done
  [[ ${#test_runs[@]} -eq 0 ]] || DATA_ROOT="$DATA_ROOT" MODE="$MODE" bash "$ROOT/experiment_launcher/evaluate_test.sh" "${test_runs[@]}"
  systems=()
  for i in "${!run_ids[@]}"; do
    dir="$ROOT/results/$MODE/$ARCH/${run_ids[$i]}"
    systems+=(--system "k=${run_depths[$i]}" "$dir/predictions_validation.json"
              "$(ls "$dir"/predictions_test_best_tau*.json | head -n 1)")
  done
  echo "== test-таблица -> $OUT/${NAME}_test.json"
  python "$ROOT/scripts/test_report.py" --output "$OUT/${NAME}_test.json" "${systems[@]}"
fi
