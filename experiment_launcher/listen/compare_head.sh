#!/usr/bin/env bash
# Онлайн-модель (<listen>/<speak>, scripts/train_online_sft.py) против головы готовности на признаках офлайн-адаптера
# (checkpoints|results/<mode>/<arch>/<id>) — на одних и тех же вопросах: validation- и test-половинах головы
# (split.json головы, вопросы из val_group1_features.jsonl). Онлайн-модель этих видео не видела. Порог каждой модели
# выбирается на validation головы одной процедурой: точный максимум CRITERION (on_time_0_4: ответ через 0..4 пары
# после целевой, не раньше; или on_target_pm2); test оценивается один раз при этом пороге и при τ = 0.5.
#
#   1. онлайн-модель на validation головы   -> $ONLINE_RUN/{,predictions_}head_<split>_validation_<adapter>.json
#   2. онлайн-модель на test головы при τ из шага 1 -> $ONLINE_RUN/{,predictions_}head_<split>_test_<adapter>.json
#   3. τ головы по её predictions_validation.json   -> results/listen/<mode>_<arch>_<id>/head_validation.json
#   4. test головы при этом τ (вероятности уже сохранены) -> results/listen/<mode>_<arch>_<id>/head_test.json
#   5. сверка вопросов по одному и сводная таблица   -> results/listen/<mode>_<arch>_<id>/comparison.json
#
# ONLINE_ONLY=1 — только шаги 1–2 (GPU): нужен лишь split.json головы, не её test. Потом тот же запуск без ONLINE_ONLY
# досчитает шаги 3–5 (готовые оценки онлайн-модели не пересчитываются без OVERWRITE=1); для них нужен test головы:
# MODE=lora bash experiment_launcher/evaluate_test.sh <arch>/<id>. <split> — короткий хэш половин головы: у голов с
# одним split.json онлайн-модель оценивается один раз.
#
#   CUDA_VISIBLE_DEVICES=1 ONLINE_RUN=outputs/online_sft ONLINE_ONLY=1 bash experiment_launcher/listen/compare_head.sh mlp/01_lm
#   ONLINE_RUN=outputs/online_sft bash experiment_launcher/listen/compare_head.sh mlp/01_lm
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
DEVICE="${DEVICE:-cuda}"
MODE="${MODE:-lora}"
ADAPTER="${ADAPTER:-best}"
CRITERION="${CRITERION:-on_time_0_4}"
ONLINE_ONLY="${ONLINE_ONLY:-0}"
OVERWRITE="${OVERWRITE:-0}"
ONLINE_RUN="${ONLINE_RUN:-}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

[[ $# -eq 1 ]] || { echo "Укажите голову: <arch>/<id> из results/$MODE/ (например mlp/01_lm)." >&2; exit 2; }
HEAD="$1"
case "$CRITERION" in on_time_0_4|on_target_pm2) ;; *) echo "CRITERION: on_time_0_4 или on_target_pm2" >&2; exit 2 ;; esac
[[ -n "$ONLINE_RUN" && -f "$ONLINE_RUN/online.json" ]] || { echo "Задайте ONLINE_RUN — каталог train_online_sft.py." >&2; exit 2; }
HEAD_SPLIT="$ROOT/checkpoints/$MODE/$HEAD/split.json"
[[ -f "$HEAD_SPLIT" ]] || { echo "Нет $HEAD_SPLIT (половины головы)" >&2; exit 2; }

SPLIT="$(FILE="$HEAD_SPLIT" python - <<'PY'
import hashlib, json, os
split = json.load(open(os.environ["FILE"], encoding="utf-8"))
print(hashlib.sha1(json.dumps([sorted(split["validation_videos"]), sorted(split["test_videos"])]).encode()).hexdigest()[:8])
PY
)"
NAME="head_$SPLIT"
echo "== Голова $MODE/$HEAD, половины $NAME, онлайн-модель $ONLINE_RUN/$ADAPTER, критерий порога $CRITERION"

online_eval() {  # $1 — половина головы (validation|test), остальное — флаги evaluate_online_sft.py
  local part="$1"; shift
  local output="$ONLINE_RUN/${NAME}_${part}_${ADAPTER}.json"
  if [[ -f "$output" && "$OVERWRITE" != "1" ]]; then
    echo "-- $output уже есть (OVERWRITE=1 — пересчитать)"; return
  fi
  python "$ROOT/scripts/evaluate_online_sft.py" --run "$ONLINE_RUN" --adapter "$ADAPTER" --name "${NAME}_${part}" \
    --records-from "$HEAD_SPLIT" --records-part "$part" --tau-criterion "$CRITERION" --data-root "$DATA_ROOT" \
    --device "$DEVICE" "$@"
}

# 1–2. Онлайн-модель на вопросах головы; её τ — по её предсказаниям на validation головы.
online_eval validation
online_eval test --tau-from "$ONLINE_RUN/predictions_${NAME}_validation_${ADAPTER}.json"
if [[ "$ONLINE_ONLY" == "1" ]]; then
  echo "== ONLINE_ONLY=1: онлайн-модель оценена; сравнение — тот же запуск без ONLINE_ONLY."
  exit 0
fi

HEAD_DIR="$ROOT/results/$MODE/$HEAD"
HEAD_VALIDATION="$HEAD_DIR/predictions_validation.json"
[[ -f "$HEAD_VALIDATION" ]] || { echo "Нет $HEAD_VALIDATION" >&2; exit 2; }
HEAD_TEST="$(ls "$HEAD_DIR"/predictions_test_best_tau*.json 2>/dev/null | head -n 1 || true)"
[[ -n "$HEAD_TEST" ]] || {
  echo "Нет test головы: MODE=$MODE bash experiment_launcher/evaluate_test.sh $HEAD (порог там не важен: берутся вероятности)." >&2
  exit 2; }
OUT="$ROOT/results/listen/${MODE}_${HEAD//\//_}"
mkdir -p "$OUT"

# 3–4. Порог головы той же процедурой по её validation, затем её test при этом τ и при 0.5.
python -m online_vllm.eval.thresholds --predictions "$HEAD_VALIDATION" --output "$OUT/head_validation.json" \
  --thresholds 0.5
HEAD_TAU="$(FILE="$OUT/head_validation.json" CRITERION="$CRITERION" python - <<'PY'
import json, os
print(repr(json.load(open(os.environ["FILE"], encoding="utf-8"))["best_thresholds"][os.environ["CRITERION"]]["tau"]))
PY
)"
python -m online_vllm.eval.thresholds --predictions "$HEAD_TEST" --output "$OUT/head_test.json" \
  --thresholds 0.5 "$HEAD_TAU"

# 5. Те же вопросы с теми же целевыми парами и числом пар, затем сводка на test: метрики без порога и при пороге с
# validation (у обеих моделей — на validation головы).
ONLINE_TEST="$ONLINE_RUN/${NAME}_test_${ADAPTER}.json" ONLINE_PREDICTIONS="$ONLINE_RUN/predictions_${NAME}_test_${ADAPTER}.json" \
  ONLINE_VALIDATION="$ONLINE_RUN/predictions_${NAME}_validation_${ADAPTER}.json" HEAD_VALIDATION="$HEAD_VALIDATION" \
  HEAD_TEST_JSON="$OUT/head_test.json" HEAD_PREDICTIONS="$HEAD_TEST" HEAD_TAU="$HEAD_TAU" CRITERION="$CRITERION" \
  OUTPUT="$OUT/comparison.json" HEAD="$MODE/$HEAD" ONLINE="$ONLINE_RUN/$ADAPTER" python - <<'PY'
import json, os

def load(key):
    return json.load(open(os.environ[key], encoding="utf-8"))

def at(results, tau):  # ключи порогов — строки; у thresholds.py они округлены
    key = min(results, key=lambda k: abs(float(k) - tau))
    if abs(float(key) - tau) > 1e-9:
        raise SystemExit(f"Нет порога {tau} среди {list(results)}")
    return results[key]

questions = [{row["record_id"]: (row["target_pair"], len(row["probabilities"])) for row in load(key)["predictions"]}
             for key in ("ONLINE_PREDICTIONS", "HEAD_PREDICTIONS")]
if questions[0].keys() != questions[1].keys():
    raise SystemExit(f"Разные вопросы: у онлайн-модели {len(questions[0])}, у головы {len(questions[1])}")
differ = [qid for qid in questions[0] if questions[0][qid] != questions[1][qid]]
if differ:
    raise SystemExit(f"{len(differ)} вопросов с разной целевой парой или числом пар, например {differ[0]}: "
                     f"{questions[0][differ[0]]} против {questions[1][differ[0]]}")

criterion = os.environ["CRITERION"]
online, head = load("ONLINE_TEST"), load("HEAD_TEST_JSON")
online_tau = online["validation_threshold"]["validation"]["tau"]
head_tau = float(os.environ["HEAD_TAU"])
models = {
    "online": {"classification": online["classification"], "latency": online["latency_accuracy"], "tau": online_tau,
               "at_tau": at(online["online"], online_tau), "at_0.5": at(online["online"], 0.5)},
    "head": {"classification": head["classification"], "latency": head["latency_accuracy"], "tau": head_tau,
             "at_tau": at(head["results"], head_tau), "at_0.5": at(head["results"], 0.5)},
}
rows = [
    ("без порога", None, None),
    ("AUROC, окно ±2", "classification", "auroc_pm2"),
    ("BCE, окно ±2", "classification", "binary_cross_entropy_pm2"),
    ("AUROC, все пары", "classification", "auroc"),
    ("auc (задержка — точность)", "latency", "auc"),
    ("auc_2", "latency", "auc_2"),
    ("auc_5", "latency", "auc_5"),
    ("точность ответа в начале", "latency", "start_accuracy"),
    ("точность ответа в конце", "latency", "end_accuracy"),
    (f"при τ с validation ({criterion})", None, None),
    ("τ", "tau", None),
]
for metric in ("on_time_0_4", "on_time_0_9", "on_target_pm2", "premature_rate", "late_rate", "missed_rate",
               "answer_accuracy", "timely_correct_0_4", "timely_correct_0_9", "timely_correct_pm2"):
    rows.append((metric, "at_tau", metric))
rows.append(("при τ = 0.5", None, None))
for metric in ("on_time_0_4", "on_time_0_9", "premature_rate", "missed_rate", "timely_correct_0_4"):
    rows.append((metric, "at_0.5", metric))

def value(model, part, key):  # latency нет, если в предсказаниях нет answer_correct
    data = models[model][part]
    return data if key is None or data is None else data.get(key)

table = {}
print(f"\nTest: {len(questions[0])} вопросов. online = {os.environ['ONLINE']}, head = {os.environ['HEAD']}")
print(f"{'метрика':34s} {'online':>10s} {'head':>10s}")
for label, part, key in rows:
    if part is None:
        print(f"-- {label}"); continue
    pair = {model: value(model, part, key) for model in models}
    table[f"{part}:{key or label}"] = pair
    print(f"{label:34s} " + " ".join("         -" if v is None else f"{v:10.4f}" for v in pair.values()))
# Задержка пары «модель ответов — детектор» при доле преждевременных <= R и точности непреждевременных >= A
# (metrics.constrained_latency): τ выбран на validation и применён к test; рядом — та же точка кривой самого test.
from online_vllm.eval.metrics import constrained_latency, operating_point
from online_vllm.eval.thresholds import payload_arrays

def arrays(key):
    data = payload_arrays(load(key))
    return data["probabilities"], data["target_pairs"], data["valid_mask"], data["correct"]

constrained = {}
pairs = {"online": ("ONLINE_VALIDATION", "ONLINE_PREDICTIONS"), "head": ("HEAD_VALIDATION", "HEAD_PREDICTIONS")}
for model, (val_key, test_key) in pairs.items():
    val, test = arrays(val_key), arrays(test_key)
    test_curve = constrained_latency(*test)
    constrained[model] = {}
    for key, chosen in constrained_latency(*val).items():
        at_val_tau = operating_point(*test, chosen["tau"]) if chosen["feasible"] else None
        constrained[model][key] = {"validation": chosen, "test_at_validation_tau": at_val_tau, "test_curve": test_curve[key]}

def show(point):
    if point is None or not point.get("feasible", True):
        return f"{'недостижимо':>31s}"
    if point["accuracy"] is None:  # при этом τ все ответы преждевременные
        return f"{'-':>6s} {'-':>5s} {100 * point['premature_rate']:6.1f}% {'-':>7s} {point['tau']:6.3f}"
    return (f"{point['latency_mean']:6.2f} {point['latency_median']:5g} {100 * point['premature_rate']:6.1f}% "
            f"{100 * point['accuracy']:6.1f}% {point['tau']:6.3f}")

print("\nЗадержка (пары после t*) при доле преждевременных <= R и точности непреждевременных >= A")
print(f"{'':12s} {'':7s} {'τ с validation → test':^31s} | {'кривая самого test':^31s}")
print(f"{'R, A':12s} {'модель':7s} {'mean':>6s} {'med':>5s} {'premat':>7s} {'acc':>7s} {'τ':>6s} | "
      f"{'mean':>6s} {'med':>5s} {'premat':>7s} {'acc':>7s} {'τ':>6s}")
for key in constrained["online"]:
    for model in pairs:
        entry = constrained[model][key]
        print(f"{key:12s} {model:7s} {show(entry['test_at_validation_tau'])} | {show(entry['test_curve'])}")

json.dump({"online": os.environ["ONLINE"], "head": os.environ["HEAD"], "criterion": criterion,
           "questions": len(questions[0]), "metrics": table, "constrained_latency": constrained},
          open(os.environ["OUTPUT"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"-> {os.environ['OUTPUT']}")
PY
