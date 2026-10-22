#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/common.sh"
TAG_MODE=step run_experiment "01_lm_mlp_step" mlp lm
