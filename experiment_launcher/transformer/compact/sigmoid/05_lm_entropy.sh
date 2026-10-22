#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/common.sh"
UHEAD_CONFIG_PATH="$ROOT/configs/uhead/compact.yaml" TAG_MODE=sigmoid run_experiment "05_lm_entropy_compact_sigmoid" compact lm entropy
