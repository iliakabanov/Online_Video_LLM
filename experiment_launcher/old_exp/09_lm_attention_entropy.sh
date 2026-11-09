#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
run_experiment "09_lm_attention_entropy" mlp lm attention entropy
