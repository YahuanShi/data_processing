#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# setup_env.sh — Check and install data_processing dependencies
#
# Usage:
#   bash setup_env.sh          # check + install missing packages
#   bash setup_env.sh --check  # check only, no install
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC}  $1"; }
fail() { echo -e "  ${RED}✗${NC}  $1"; }
warn() { echo -e "  ${YELLOW}!${NC}  $1"; }

echo
echo "════════════════════════════════════════════════════════"
echo "  data_processing — environment check"
echo "════════════════════════════════════════════════════════"
echo

# ── Python version ────────────────────────────────────────────────────────────
PY=$(python3 --version 2>&1)
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [[ "$PY_MINOR" -ge 10 ]]; then
    ok "Python: $PY"
else
    fail "Python 3.10+ required, found: $PY"
    exit 1
fi

# ── Package checks ────────────────────────────────────────────────────────────
MISSING=()

check_pkg() {
    local import_name="$1"
    local display_name="$2"
    if python3 -c "import $import_name" 2>/dev/null; then
        VER=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', 'ok'))" 2>/dev/null || echo "ok")
        ok "$display_name  ($VER)"
    else
        fail "$display_name  — NOT FOUND"
        MISSING+=("$display_name")
    fi
}

check_pkg numpy      "numpy"
check_pkg scipy      "scipy"
check_pkg h5py       "h5py"
check_pkg cv2        "opencv-python"
check_pkg matplotlib "matplotlib"

echo

# ── Result ────────────────────────────────────────────────────────────────────
if [[ ${#MISSING[@]} -eq 0 ]]; then
    echo -e "  ${GREEN}All dependencies satisfied.${NC}"
    echo
    exit 0
fi

warn "Missing: ${MISSING[*]}"
echo

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "  Run without --check to install."
    echo
    exit 1
fi

# ── Install ───────────────────────────────────────────────────────────────────
echo "  Installing via pip ..."
echo
pip install -r "$SCRIPT_DIR/requirements.txt"
echo
echo -e "  ${GREEN}Done. Re-run with --check to verify.${NC}"
echo
