# export RENDER_WS_URL=ws://x.x.x.x:x/render
# export ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fileroot="${fileroot:-${VIEWSUITE_ROOT}}"

LOG_FILE="${SCRIPT_DIR}/run.log"

python -m vagen.evaluate.run_eval \
  --config "${SCRIPT_DIR}/config.yaml" \
  run.backend=openai \
  fileroot="${fileroot}" \
  2>&1 | tee "${LOG_FILE}"

