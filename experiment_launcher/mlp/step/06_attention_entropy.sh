#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/common.sh"
TAG_MODE=step run_experiment "06_attention_entropy_mlp_step" mlp attention entropy
