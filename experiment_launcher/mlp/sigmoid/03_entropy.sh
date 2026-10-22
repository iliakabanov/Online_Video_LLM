#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/common.sh"
TAG_MODE=sigmoid run_experiment "03_entropy_mlp_sigmoid" mlp entropy
