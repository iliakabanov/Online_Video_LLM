#!/usr/bin/env bash
# Подготовка сервера к оценке без дообучения (src/online_vllm/eval/ZERO_SHOT.md, «Запуск на сервере»):
#   1) половины val_group1 по видео, если их ещё нет (dataset_builders/clevrer_proactive/subset.py);
#   2) выборка: 500 видео, по одному вопросу (scripts/sample_eval_questions.py), если её ещё нет;
#   3) проверка, что видео выборки и проверочного вопроса (check.sh) на диске;
#   4) веса Qwen2.5-VL-3B и MMDuet2 в кэш Hugging Face ($HF_HOME), чтобы дальше считать без сети.
#   DATA_ROOT=/data HF_HOME=/big/disk/hf bash experiment_launcher/zero_shot/prepare.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/PATH/TO/DATA}"
SOURCE="${SOURCE:-processed/clevrer_proactive/val_group1_features.jsonl}"
QUESTIONS="${QUESTIONS:-processed/clevrer_proactive/val_group1_features_eval500.jsonl}"
VIDEOS="${VIDEOS:-500}"
SEED="${SEED:-0}"
[[ "$DATA_ROOT" != "/PATH/TO/DATA" ]] || { echo "Задайте DATA_ROOT." >&2; exit 2; }
export DATA_ROOT
cd "$ROOT"

if [[ ! -f "$DATA_ROOT/$SOURCE" ]]; then
  echo "Нет $DATA_ROOT/$SOURCE: делю train_group1.jsonl и val_group1.jsonl на половины по видео"
  python -m dataset_builders.clevrer_proactive.subset --halves-only
fi
if [[ -f "$DATA_ROOT/$QUESTIONS" ]]; then
  echo "Выборка уже есть: $DATA_ROOT/$QUESTIONS (оставлена как есть: по ней идут запуски)"
else
  python scripts/sample_eval_questions.py --questions "$SOURCE" --videos "$VIDEOS" --seed "$SEED" --output "$QUESTIONS"
fi

QUESTIONS="$QUESTIONS" python - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, "src")
from online_vllm.data.schema import load_jsonl
root = Path(os.environ["DATA_ROOT"])
records = list(load_jsonl(root / os.environ["QUESTIONS"]))
check = [r for r in load_jsonl(root / "processed/clevrer_proactive/val_group1.jsonl")
         if r.id == "validation_10000_clevrer_query_material_1"]
missing = sorted({r.video for r in records + check if not (root / r.video).exists()})
print(f"Выборка: {len(records)} вопросов из {len({r.video for r in records})} видео; видео нет на диске: {len(missing)}")
if missing:
    print("Например:", *missing[:5], sep="\n  ")
    sys.exit("Нужны видео validation: VIDEOS=all bash scripts/prepare_clevrer_proactive.sh")
PY

python - <<'PY'
from huggingface_hub import snapshot_download
for model in ("Qwen/Qwen2.5-VL-3B-Instruct", "wangyueqian/MMDuet2"):
    print(model, "->", snapshot_download(model))
PY
echo "Готово. Дальше: bash experiment_launcher/zero_shot/check.sh, затем пилот и run.sh."
