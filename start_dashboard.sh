#!/usr/bin/env bash
# ============================================================
# start_dashboard.sh — Bear Risk Dashboard Launcher (Mac/Linux)
# ============================================================
# Usage: ./start_dashboard.sh
# Stop:  Ctrl+C  (cleanly kills scheduler and Streamlit)
# ============================================================

set -euo pipefail

# ── Resolve project directory ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
AMBER='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${CYAN}║       AI BEAR RISK DASHBOARD  v1.0           ║${RESET}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════╝${RESET}"
    echo ""
}

print_section() {
    echo -e "${BOLD}▸ $1${RESET}"
}

# ── Cleanup on exit ────────────────────────────────────────────────────────────
SCHEDULER_PID=""
cleanup() {
    echo ""
    echo -e "${AMBER}Shutting down...${RESET}"
    if [ -n "$SCHEDULER_PID" ] && kill -0 "$SCHEDULER_PID" 2>/dev/null; then
        kill "$SCHEDULER_PID"
        echo -e "${GREEN}✓ Scheduler stopped (PID $SCHEDULER_PID)${RESET}"
    fi
    # Kill Streamlit by port (8501 is default)
    pkill -f "streamlit run app.py" 2>/dev/null || true
    echo -e "${GREEN}✓ Dashboard stopped${RESET}"
    echo ""
    exit 0
}
trap cleanup INT TERM

print_header

# ── Step 1: Virtual environment ────────────────────────────────────────────────
print_section "Activating virtual environment"
VENV_PATHS=("venv" ".venv" "env" ".env_venv")
VENV_FOUND=false
for venv in "${VENV_PATHS[@]}"; do
    if [ -f "$venv/bin/activate" ]; then
        # shellcheck source=/dev/null
        source "$venv/bin/activate"
        echo -e "  ${GREEN}✓ Activated: $venv${RESET}"
        VENV_FOUND=true
        break
    fi
done
if [ "$VENV_FOUND" = false ]; then
    echo -e "  ${AMBER}⚠ No virtual environment found — using system Python${RESET}"
    echo -e "  ${AMBER}  Tip: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt${RESET}"
fi

# ── Step 2: FRED API key check ─────────────────────────────────────────────────
print_section "Checking environment"
if [ -z "${FRED_API_KEY:-}" ]; then
    if [ -f ".env" ]; then
        # shellcheck source=/dev/null
        set -a; source .env; set +a
    fi
fi
if [ -z "${FRED_API_KEY:-}" ]; then
    echo -e "  ${RED}✗ FRED_API_KEY not set!${RESET}"
    echo -e "  ${AMBER}  Get a free key at: https://fred.stlouisfed.org/${RESET}"
    echo -e "  ${AMBER}  Then run: export FRED_API_KEY='your_key_here'${RESET}"
    echo -e "  ${AMBER}  Or add it to a .env file in this directory.${RESET}"
    read -r -p "  Continue anyway? (data will load from cache if available) [y/N]: " resp
    if [[ ! "$resp" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "  ${GREEN}✓ FRED_API_KEY set${RESET}"
fi

# ── Step 3: Health check ───────────────────────────────────────────────────────
print_section "Running startup health check"
python3 - <<'PYCHECK'
import sys, json
from pathlib import Path
from datetime import datetime

cache_dir = Path(".cache")
if not cache_dir.exists():
    print("  ⚠  Cache directory not found — first run will fetch all data (~3 min)")
    sys.exit(0)

# Check key cache files and their ages
checks = [
    ("fred_T10Y2Y.parquet",         "Yield Curve",        1),
    ("fred_BAMLH0A0HYM2.parquet",   "HY Spreads",         1),
    ("shiller_ie_data.parquet",     "Shiller CAPE",       30),
    ("aaii_sentiment.parquet",      "AAII Sentiment",     7),
    ("fred_NFCI.parquet",           "NFCI",               7),
]

any_missing = False
for filename, label, max_days in checks:
    path = cache_dir / filename
    if not path.exists():
        print(f"  ⚠  {label}: NOT CACHED (will fetch on launch)")
        any_missing = True
    else:
        age = (datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)).days
        status = "✓" if age <= max_days else "⚠ STALE"
        color = "" if age <= max_days else ""
        print(f"  {status} {label}: {age}d old")

# Check for trained models
model_files = list((cache_dir / "models").glob("stack_h*.joblib")) if (cache_dir / "models").exists() else []
if model_files:
    latest = max(model_files, key=lambda p: p.stat().st_mtime)
    age_days = (datetime.utcnow() - datetime.utcfromtimestamp(latest.stat().st_mtime)).days
    print(f"  ✓  ML models: {len(model_files)} found, latest {age_days}d old")
else:
    print("  ⚠  ML models: NOT TRAINED (run: python ml_model.py)")

# Quick composite score from cache
try:
    import sys; sys.path.insert(0, ".")
    from bear_risk_api import get_current_risk_score
    result = get_current_risk_score()
    if result:
        score = result["score"]
        band  = result["band"]
        stars = "██████████"[:int(score/10)] + "░░░░░░░░░░"[int(score/10):]
        print(f"\n  Current Bear Risk Score: {score:.0f}/100 [{stars}] {band}")
    else:
        print("\n  Current Bear Risk Score: N/A (data not yet loaded)")
except Exception as e:
    print(f"\n  Could not compute score: {e}")
PYCHECK

echo ""

# ── Step 4: Start scheduler in background ─────────────────────────────────────
print_section "Starting background scheduler"
python3 scheduler.py &
SCHEDULER_PID=$!
sleep 1
if kill -0 "$SCHEDULER_PID" 2>/dev/null; then
    echo -e "  ${GREEN}✓ Scheduler running (PID $SCHEDULER_PID)${RESET}"
else
    echo -e "  ${AMBER}⚠ Scheduler failed to start — refresh will be manual${RESET}"
    SCHEDULER_PID=""
fi

# ── Step 5: Run alert check ────────────────────────────────────────────────────
print_section "Running alert check"
python3 personal_alerts.py 2>&1 | grep -E "ALERT|score=|No new" | head -5 || true
echo ""

# ── Step 6: Launch Streamlit ───────────────────────────────────────────────────
print_section "Launching dashboard"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
echo -e "  Opening: ${CYAN}http://localhost:$STREAMLIT_PORT${RESET}"
echo -e "  ${AMBER}Press Ctrl+C to stop${RESET}"
echo ""

# Open browser after a short delay
(sleep 3 && open "http://localhost:$STREAMLIT_PORT" 2>/dev/null || xdg-open "http://localhost:$STREAMLIT_PORT" 2>/dev/null) &

streamlit run app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.base dark \
    --theme.backgroundColor "#0a0e1a" \
    --theme.secondaryBackgroundColor "#0f1525" \
    --theme.textColor "#e2e8f4"
