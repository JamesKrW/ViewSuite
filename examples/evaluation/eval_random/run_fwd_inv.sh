#!/usr/bin/env bash
set -euo pipefail

# Random baseline evaluation for forward & inverse dynamics.
# No external server needed — responses are sampled locally.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config_fwd_inv.yaml}"
shift 2>/dev/null || true

python -m vagen.evaluate.run_eval --config "$CONFIG" "$@" \
  2>&1 | tee "${SCRIPT_DIR}/run_fwd_inv.log"
