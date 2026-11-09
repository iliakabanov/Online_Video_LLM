#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/common.sh"
TAG_MODE=sigmoid run_experiment "04_lm_attention_mlp_sigmoid" mlp lm attention
