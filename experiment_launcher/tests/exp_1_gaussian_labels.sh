#!/usr/bin/env bash
set -euo pipefail
TAG_MODE=gaussian
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
run_experiment "exp_1_gaussian_labels" mlp lm
