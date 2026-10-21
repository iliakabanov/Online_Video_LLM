#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/absolute/path/to/data}"
TRAIN_FEATURES="${TRAIN_FEATURES:-/absolute/path/to/train_features}"
VAL_FEATURES="${VAL_FEATURES:-/absolute/path/to/val_features}"
DEVICE="${DEVICE:-cuda}"

TAG_MODE=gaussian
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
run_experiment "exp_1_gaussian_labels" mlp lm