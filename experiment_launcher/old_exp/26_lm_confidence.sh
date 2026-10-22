#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
run_experiment "26_lm_confidence" transformer lm confidence
