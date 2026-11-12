#!/usr/bin/env bash
# Общие настройки ablation голов на признаках нового офлайн-адаптера (outputs/offline_10ep/best, эпоха 10):
# 1000 train-видео и 1000 eval-видео половины для признаков (*_group1_features.jsonl), MODE=lora, метки hard,
# 10 эпох, best.pt — по наибольшему AUROC в окне |k − t*| <= 2 на validation. Подключается из mlp.sh,
# transformer.sh, compact.sh; сравнение запусков — только на validation (summary.py), test не трогается.
#
# Пути к признакам по умолчанию — те, что пишет scripts/extract_features.py (--videos 1000, --adapter
# outputs/offline_10ep/best); другие — через TRAIN_FEATURES / VAL_FEATURES.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
ADAPTER_RUN="${ADAPTER_RUN:-offline_10ep}"
export TRAIN_FEATURES="${TRAIN_FEATURES:-$DATA_ROOT/features/group1_half_s1_252x168/train_${ADAPTER_RUN}_best_videos1000}"
export VAL_FEATURES="${VAL_FEATURES:-$DATA_ROOT/features/group1_half_s1_252x168/val_${ADAPTER_RUN}_best_videos1000}"
export MODE=lora TAG_MODE=hard EPOCHS="${EPOCHS:-10}" SELECTION="${SELECTION:-auroc_pm2}"
# Суффикс id запусков: results/lora/<arch>/<группа>_<arch>_<SUFFIX>, не пересекается с прежними головами.
SUFFIX="${SUFFIX:-off10}"
GROUPS_TO_RUN=(lm attention entropy vision confidence)

for features in "$TRAIN_FEATURES" "$VAL_FEATURES"; do
  [[ -f "$features/meta.json" ]] || {
    echo "Нет $features/meta.json: задайте TRAIN_FEATURES / VAL_FEATURES (каталоги extract_features.py)." >&2; exit 2; }
  # Признаки именно этого адаптера и 1000 видео половины для признаков.
  META="$features/meta.json" RUN="$ADAPTER_RUN" python - <<'PY' || exit 2
import json, os, sys
meta = json.load(open(os.environ["META"], encoding="utf-8"))
adapter = meta.get("adapter") or {}
problems = []
if adapter.get("run") != os.environ["RUN"]:
    problems.append(f"адаптер {adapter.get('run')!r}, ожидался {os.environ['RUN']!r}")
if meta.get("videos") != 1000:
    problems.append(f"{meta.get('videos')} видео, ожидалось 1000")
if not str(meta.get("source", "")).endswith("_group1_features.jsonl"):
    problems.append(f"источник {meta.get('source')} — не половина для признаков")
if problems:
    sys.exit(f"{os.environ['META']}: " + "; ".join(problems))
print(f"{os.environ['META']}: адаптер {adapter['run']}/{adapter['name']} (эпоха {adapter.get('epoch')}), "
      f"{meta['videos']} видео, {meta['records']} вопросов")
PY
done

source "$ROOT/experiment_launcher/common.sh"
