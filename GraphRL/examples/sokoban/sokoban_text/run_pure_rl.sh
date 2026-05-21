#!/usr/bin/env bash
# =============================================================================
# Sokoban (text) — Pure-RL Baseline (1 iteration, ~401 steps, no SFT)
# =============================================================================
#
# Usage:
#   bash run_pure_rl.sh
#   bash run_pure_rl.sh general_overrides.rl.training_steps=200
#
# experiment_dir is computed by pipeline_pure_rl.yaml as
#   exps/sokoban/sokoban_text_pure_rl/
# resolved relative to the current working directory.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="${PWD}/exps/sokoban/sokoban_text_pure_rl"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

python -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline_pure_rl \
    general_overrides.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train.yaml" \
    general_overrides.rl.hydra_overrides.data.val_files="${SCRIPT_DIR}/val.yaml" \
    +iteration_overrides.iter0.rl.hydra_overrides.huggingface_hub.hf_save_freq=200 \
    "$@" 2>&1 | tee "${LOG_FILE}"
