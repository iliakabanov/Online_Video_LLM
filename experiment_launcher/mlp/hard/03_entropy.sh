#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/common.sh"
TAG_MODE=hard run_experiment "03_entropy_mlp_hard" mlp entropy
