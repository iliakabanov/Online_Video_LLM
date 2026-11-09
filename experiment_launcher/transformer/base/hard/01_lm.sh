#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/common.sh"
UHEAD_CONFIG_PATH="$ROOT/configs/uhead/default.yaml" TAG_MODE=hard run_experiment "01_lm_base_hard" transformer lm
