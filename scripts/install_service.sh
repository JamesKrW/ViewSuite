#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export VIEWSUITE_ROOT="${VIEWSUITE_ROOT:-${REPO_ROOT}}"

cat > "${REPO_ROOT}/.env" <<EOF
export VIEWSUITE_ROOT="${VIEWSUITE_ROOT}"
EOF

pip install -e "${REPO_ROOT}"
pip install "huggingface-hub>=0.34.0,<1.0"

echo ""
echo "============================================================"
echo "Done. Quick sanity check:"
echo "============================================================"
python - <<'PY'
import importlib, sys
mods = ["view_suite"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "n/a")
        print(f"  OK  {m:<14} {ver}")
    except Exception as e:
        print(f"  FAIL {m:<14} {type(e).__name__}: {e}")
        sys.exit(1)
PY
