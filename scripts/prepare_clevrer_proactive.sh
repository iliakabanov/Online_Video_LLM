#!/usr/bin/env bash
# Build CLEVRER-Proactive from scratch into $DATA_ROOT (default ./data). About 10 minutes without videos.
#   bash scripts/prepare_clevrer_proactive.sh            # questions + annotations, generate, balance, export
#   VIDEOS=all bash scripts/prepare_clevrer_proactive.sh # also the 18.6 GB of train + validation videos
set -euo pipefail
cd "$(dirname "$0")/.."
M=dataset_builders.clevrer_proactive

python -m $M.download --videos "${VIDEOS:-none}"
for split in train validation; do
    python -m $M.generate --split $split            # -> processed/clevrer_proactive/<split>.json
    python -m $M.balance $split.json                # -> <split>_balanced.json
    python -m $M.export --split $split              # -> <split>.jsonl (common schema)
done
python -m pytest -q tests/test_clevrer_proactive.py
