#!/usr/bin/env bash
# Общие настройки голов на признаках онлайн-модели (outputs/listen_sft_10ep/epoch_10, scripts/extract_features.py
# --online-run): те же 1000 train- и 1000 eval-видео половины для признаков, что у голов офлайн-адаптера
# (experiment_launcher/offline_10ep), MODE=lora, метки hard, 10 эпох, best.pt — по наибольшему AUROC в окне
# |k − t*| <= 2 на validation. Сравнение — только на validation (offline_10ep/summary.py --suffix on10).
#
# Признаки онлайн-модели: внимание с группой <listen> и долями пар по визуальным токенам, группа listen_token_logit
# (logit(<speak>) − logit(<listen>), log P обоих). Пути по умолчанию — те, что пишет extract_features.py
# (--videos 1000 --online-run outputs/listen_sft_10ep --online-adapter epoch_10); другие — TRAIN_FEATURES / VAL_FEATURES.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
ONLINE_RUN="${ONLINE_RUN:-listen_sft_10ep}"
ONLINE_ADAPTER="${ONLINE_ADAPTER:-epoch_10}"
export TRAIN_FEATURES="${TRAIN_FEATURES:-$DATA_ROOT/features/group1_half_s1_252x168/train_${ONLINE_RUN}_${ONLINE_ADAPTER}_videos1000}"
export VAL_FEATURES="${VAL_FEATURES:-$DATA_ROOT/features/group1_half_s1_252x168/val_${ONLINE_RUN}_${ONLINE_ADAPTER}_videos1000}"
export MODE=lora TAG_MODE=hard EPOCHS="${EPOCHS:-10}" SELECTION="${SELECTION:-auroc_pm2}"
# Суффикс id запусков: results/lora/<arch>/<группы через _>_<arch>_<SUFFIX>; у голов офлайн-адаптера — off10.
SUFFIX="${SUFFIX:-on10}"

for features in "$TRAIN_FEATURES" "$VAL_FEATURES"; do
  [[ -f "$features/meta.json" ]] || {
    echo "Нет $features/meta.json: задайте TRAIN_FEATURES / VAL_FEATURES (каталоги extract_features.py --online-run)." >&2
    exit 2; }
  # Признаки именно этой онлайн-модели и 1000 видео половины для признаков.
  META="$features/meta.json" RUN="$ONLINE_RUN" NAME="$ONLINE_ADAPTER" python - <<'PY' || exit 2
import json, os, sys
meta = json.load(open(os.environ["META"], encoding="utf-8"))
adapter = meta.get("adapter") or {}
problems = []
if "online" not in meta:
    problems.append("это признаки не онлайн-модели (нет поля online)")
if (adapter.get("run"), adapter.get("name")) != (os.environ["RUN"], os.environ["NAME"]):
    problems.append(f"адаптер {adapter.get('run')}/{adapter.get('name')}, ожидался {os.environ['RUN']}/{os.environ['NAME']}")
if meta.get("videos") != 1000:
    problems.append(f"{meta.get('videos')} видео, ожидалось 1000")
if not str(meta.get("source", "")).endswith("_group1_features.jsonl"):
    problems.append(f"источник {meta.get('source')} — не половина для признаков")
if problems:
    sys.exit(f"{os.environ['META']}: " + "; ".join(problems))
print(f"{os.environ['META']}: онлайн-модель {adapter['run']}/{adapter['name']} (эпоха {adapter.get('epoch')}), "
      f"{meta['videos']} видео, {meta['records']} вопросов")
PY
done

source "$ROOT/experiment_launcher/common.sh"
