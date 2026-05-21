#!/usr/bin/env bash
# Unified install script for the merged ViewSuite monorepo.
#
# Usage:
#   conda create -n viewsuite python=3.12 -y && conda activate viewsuite
#   bash scripts/install.sh
#
# Environment knobs:
#   SKIP_VERL_BOOTSTRAP=1   skip verl's install_vllm_sglang_mcore.sh
#   USE_MEGATRON=0          (default) skip megatron build
#   USE_SGLANG=1            (default) install sglang

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export VIEWSUITE_ROOT="${VIEWSUITE_ROOT:-${REPO_ROOT}}"

GRAPHRL_DIR="${REPO_ROOT}/GraphRL"
VAGEN_DIR="${GRAPHRL_DIR}/VAGEN"
VERL_DIR="${VAGEN_DIR}/verl"
LF_DIR="${GRAPHRL_DIR}/LLaMA-Factory"

cat > "${REPO_ROOT}/.env" <<EOF
export VIEWSUITE_ROOT="${VIEWSUITE_ROOT}"
EOF

if [ "${SKIP_VERL_BOOTSTRAP:-0}" != "1" ]; then
    (cd "${VERL_DIR}" && USE_MEGATRON="${USE_MEGATRON:-0}" USE_SGLANG="${USE_SGLANG:-1}" \
        bash scripts/install_vllm_sglang_mcore.sh)
fi

pip install --no-deps -e "${VERL_DIR}"
pip install "trl==0.26.2"
pip install "huggingface-hub>=0.34.0,<1.0"

pip install -e "${VAGEN_DIR}"

pip install -e "${LF_DIR}"
pip install -r "${LF_DIR}/requirements/metrics.txt"
pip install -r "${LF_DIR}/requirements/deepspeed.txt"

pip install -e "${GRAPHRL_DIR}"
pip install -e "${REPO_ROOT}"

pip install transformers==4.57.1
pip install "sglang[all]==0.5.3.post3"

echo ""
echo "============================================================"
echo "Done. Quick sanity check:"
echo "============================================================"
python - <<'PY'
import importlib, sys
mods = ["view_suite", "graphrl", "vagen", "verl", "llamafactory", "transformers", "sglang"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "n/a")
        print(f"  OK  {m:<14} {ver}")
    except Exception as e:
        print(f"  FAIL {m:<14} {type(e).__name__}: {e}")
        sys.exit(1)
PY
