#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/common.sh"
UHEAD_CONFIG_PATH="$ROOT/configs/uhead/default.yaml" TAG_MODE=sigmoid run_experiment "04_lm_attention_base_sigmoid" transformer lm attention
