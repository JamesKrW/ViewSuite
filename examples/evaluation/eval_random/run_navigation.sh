#!/usr/bin/env bash
set -euo pipefail

# Random baseline evaluation for active exploration (navigation).
# Requires the rendering server to be running:
#   see client_url.txt for the expected endpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config_navigation.yaml}"
shift 2>/dev/null || true

python -m vagen.evaluate.run_eval --config "$CONFIG" "$@" \
  2>&1 | tee "${SCRIPT_DIR}/run_navigation.log"
