# 🐻 AI Bear Risk Dashboard

A personal morning dashboard for macro bear market risk — open it, get a 60-second read, make better investment decisions.

> **Not financial advice.** Educational tool only. Model trained on ~8 historical bear market samples. Use judgment.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [FRED API Key Setup](#3-fred-api-key-setup)
4. [First Run Walkthrough](#4-first-run-walkthrough)
5. [Daily Use Workflow](#5-daily-use-workflow)
6. [Troubleshooting](#6-troubleshooting)
7. [Monthly Maintenance Checklist](#7-monthly-maintenance-checklist)
8. [When to Retrain the Model](#8-when-to-retrain-the-model)

---

## 1. Prerequisites

You need two things installed on your computer before starting:

**Python 3.11**
- Mac: download from [python.org/downloads](https://www.python.org/downloads/) — pick version 3.11.x
- Windows: same link — during install, **check "Add Python to PATH"** before clicking Install

**Git** (to download the code)
- Mac: open Terminal, type `git --version`. If it asks you to install, click Install.
- Windows: download from [git-scm.com](https://git-scm.com/download/win) and install with defaults

To verify both are installed, open Terminal (Mac) or Command Prompt (Windows) and run:
```
python --version
git --version
```
Both should print version numbers without errors.

---

## 2. Installation

Open Terminal (Mac) or Command Prompt (Windows). Run these commands **one line at a time**, copying each exactly:

```bash
# 1. Download the project
git clone https://github.com/YOUR_USERNAME/bear-risk-dashboard.git
cd bear-risk-dashboard

# 2. Create an isolated Python environment
python3 -m venv venv          # Mac
# python -m venv venv         # Windows (use "python" not "python3")

# 3. Activate it
source venv/bin/activate       # Mac
# venv\Scripts\activate        # Windows

# 4. Install all dependencies
pip install -r requirements.txt
```

**Expected output for step 4:** You will see a long list of packages being downloaded and installed. This takes 2–5 minutes depending on your internet. The last line should say something like `Successfully installed ...`.

> If you see `pip: command not found`, try `pip3 install -r requirements.txt`.

---

## 3. FRED API Key Setup

The dashboard pulls economic data from the St. Louis Federal Reserve (FRED). Their API is free — you just need to register.

**Step 1 — Create a free account:**
Go to [fred.stlouisfed.org](https://fred.stlouisfed.org/), click "My Account" → "Create Account". Takes 2 minutes.

**Step 2 — Generate your API key:**
Log in → click your name (top right) → "API Keys" → "Request API Key". Copy the 32-character key.

**Step 3 — Set the key on your computer:**

*Mac (permanent — survives restarts):*
```bash
echo 'export FRED_API_KEY="paste_your_key_here"' >> ~/.zshrc
source ~/.zshrc
```

*Windows (permanent — survives restarts):*
```cmd
setx FRED_API_KEY "paste_your_key_here"
```
Then **close and reopen** Command Prompt for the change to take effect.

*Alternative for both platforms — create a `.env` file:*
In the project folder, create a file called `.env` (no other extension) containing:
```
FRED_API_KEY=paste_your_key_here
```

**Verify it worked:**
```bash
echo $FRED_API_KEY    # Mac — should print your key
# echo %FRED_API_KEY% # Windows
```

---

## 4. First Run Walkthrough

With your virtual environment active and API key set, run:

```bash
# Mac
./start_dashboard.sh

# Windows
start_dashboard.bat
```

**What happens and how long it takes:**

| Stage | What you see | Time |
|-------|-------------|------|
| Health check | Cache status table in terminal | ~5 sec |
| Data fetch | "Fetching market data…" in browser | 2–4 min |
| Indicator compute | "Computing indicators…" | 30 sec |
| Dashboard ready | Tab 1 appears with risk gauge | — |

> **First run takes 3–5 minutes total** — everything is cached after that. Subsequent launches take under 30 seconds.

**Expected terminal output on first run:**
```
╔══════════════════════════════════════════════╗
║       AI BEAR RISK DASHBOARD  v1.0           ║
╚══════════════════════════════════════════════╝

▸ Activating virtual environment
  ✓ Activated: venv

▸ Checking environment
  ✓ FRED_API_KEY is set

▸ Running startup health check
  ⚠ Yield Curve: NOT CACHED (will fetch on launch)
  ⚠ HY Spreads: NOT CACHED (will fetch on launch)
  ...
  Current Bear Risk Score: N/A (data not yet loaded)

▸ Starting background scheduler
  ✓ Scheduler running (PID 12345)

Launching dashboard at http://localhost:8501
```

**The browser opens automatically.** If it doesn't, open [http://localhost:8501](http://localhost:8501) manually.

**To train the ML models** (optional but recommended for the probability estimates):
```bash
# In a separate terminal, with venv active:
python ml_model.py
```
This takes 5–15 minutes. The dashboard works without it — you just won't have the probability numbers until training completes.

---

## 5. Daily Use Workflow

Three steps, every morning:

**Step 1 — Launch:**
```bash
# Mac
./start_dashboard.sh

# Windows
start_dashboard.bat
```

**Step 2 — Read Tab 1 (Overview):**
- Gauge shows composite score (0–100)
- Three probability numbers show ML bear risk estimates
- Top 3 risk drivers tell you *why* the score is where it is
- Entire read takes under 60 seconds

**Step 3 — Act on what you see:**

| Score | Band | Action |
|-------|------|--------|
| 0–24 | 🟢 LOW | Normal allocation. Nothing to do. |
| 25–44 | 🔵 GUARDED | Review equity allocation. Don't act on one reading. |
| 45–59 | 🟡 ELEVATED | Consider reducing equity 10–20%. |
| 60–74 | 🟠 HIGH | Reduce equity 20–35%. Add cash/short-duration. |
| 75–100 | 🔴 SEVERE | Maximum defensive posture. Review all positions. |

**Automatic alerts:** If you leave the scheduler running, desktop notifications fire automatically when the score crosses 50 or 70, or when any indicator turns red. No manual checks needed.

---

## 6. Troubleshooting

### Error: `ModuleNotFoundError: No module named 'streamlit'`
**Fix:** Your virtual environment is not activated.
```bash
source venv/bin/activate    # Mac
venv\Scripts\activate       # Windows
```
Then try again.

---

### Error: `EnvironmentError: FRED_API_KEY is not set`
**Fix:** Follow [Section 3](#3-fred-api-key-setup) to set your key. Quick workaround for this session only:
```bash
export FRED_API_KEY="your_key_here"    # Mac
set FRED_API_KEY=your_key_here         # Windows
```

---

### Dashboard shows "N/A" everywhere / all scores are blank
**Cause:** Cache is empty (first run or cache was deleted) and live fetch failed.
**Fix:**
1. Check your internet connection
2. Verify your FRED API key works: `python -c "from data_fetcher import fetch_fred_series; print(fetch_fred_series('T10Y2Y'))"`
3. Try a manual force-refresh: `python scheduler.py --force-refresh`
4. Check the log: `cat .cache/scheduler.log`

---

### Error: `Port 8501 is already in use`
**Cause:** A previous Streamlit session is still running.
**Fix (Mac):** `pkill -f "streamlit"`
**Fix (Windows):** Open Task Manager → find Python processes → end them
**Alternative:** Run on a different port: `streamlit run app.py --server.port 8502`

---

### AAII sentiment data fails to load
**Cause:** AAII occasionally changes their website structure.
**Fix:** The dashboard degrades gracefully — AAII indicator will show N/A. Indicator 16 weight redistributes to others automatically. Check `.cache/scheduler.log` for details.

---

### `yfinance` returns empty data for S&P 500 constituent prices
**Cause:** yfinance rate limits batch downloads of 500 tickers.
**Fix:** This is normal on first run. Wait 10 minutes and retry: `python scheduler.py --force-refresh --tier weekly`. The breadth calculation uses whatever tickers successfully downloaded.

---

## 7. Monthly Maintenance Checklist

Do this on the first weekend of each month (takes ~10 minutes):

- [ ] **1. Force-refresh all data:** `python scheduler.py --force-refresh`
  This ensures quarterly indicators (GDP, corporate debt, profit margins) are current.

- [ ] **2. Check the scheduler log for errors:**
  ```bash
  tail -50 .cache/scheduler.log | grep -i "error\|fail\|warning"
  ```
  Investigate any repeated failures.

- [ ] **3. Update S&P 500 constituent list:**
  ```bash
  rm .cache/sp500_tickers.txt
  python scheduler.py --force-refresh --tier weekly
  ```
  Index constituents change quarterly — stale lists affect market breadth accuracy.

- [ ] **4. Check model age — retrain if >6 months old:**
  ```bash
  ls -la .cache/models/
  ```
  If the newest `stack_h*.joblib` file is older than 6 months, run:
  ```bash
  python ml_model.py
  ```

- [ ] **5. Review Tab 5 (Model Diagnostics) in the dashboard:**
  Check that PR-AUC is ≥ 0.40 and ECE is < 0.05. If either threshold fails, the dashboard shows a warning — see Section 8.

---

## 8. When to Retrain the Model

The ML model should be retrained when any of the following occur:

**Scheduled trigger (every 6 months):**
Training data grows — new months of indicators and market outcomes improve the model.

**Performance drift signals (check Tab 5 monthly):**
- PR-AUC drops below 0.35 in the backtest report
- ECE (calibration error) rises above 0.08
- The backtest report shows "MODEL PERFORMANCE BELOW THRESHOLD" banner

**Structural change triggers:**
- A new NBER recession has been officially declared and ended (adds new positive training samples)
- The Fed implements a policy regime shift (e.g., resumes QE, adopts yield curve control) — the regime weights need relearning
- Any FRED series ID changes and you've updated `config.py` accordingly

**How to retrain:**
```bash
# Make sure your venv is active and data is fresh first
python scheduler.py --force-refresh
# Then train (takes 5–15 minutes)
python ml_model.py
```

New model files are saved to `.cache/models/` with a timestamp and PR-AUC in the filename. The old models are kept — you can delete files older than 2 versions to save disk space.

---

## Project File Tree

```
bear-risk-dashboard/
│
├── app.py                    # Streamlit UI — 5 tabs
├── config.py                 # All settings, thresholds, indicator metadata
├── data_fetcher.py           # FRED / yfinance / Shiller / AAII fetchers
├── indicator_calculator.py   # 20 indicator computation functions
├── ml_model.py               # XGBoost / RF / LR ensemble + SHAP
├── scheduler.py              # Background data refresh daemon
├── bear_risk_api.py          # StockMind integration bridge (read-only)
├── personal_alerts.py        # Threshold crossing alerts (desktop + email)
├── start_dashboard.sh        # Mac/Linux one-click launcher
├── start_dashboard.bat       # Windows one-click launcher
├── requirements.txt          # All Python dependencies (pinned)
├── README.md                 # This file
│
└── .cache/                   # Auto-created — local data cache
    ├── fred_T10Y2Y.parquet
    ├── fred_BAMLH0A0HYM2.parquet
    ├── shiller_ie_data.parquet
    ├── aaii_sentiment.parquet
    ├── sp500_tickers.txt
    ├── sp500_breadth_prices.parquet
    ├── scheduler.log
    ├── alert_state.json
    ├── latest_prediction.json
    └── models/
        ├── stack_h3m_20250601_0845_pr0p412.joblib
        ├── stack_h6m_20250601_0845_pr0p445.joblib
        └── stack_h12m_20250601_0845_pr0p481.joblib
```

### Module relationships

```
app.py
  ├── data_fetcher.py        (data acquisition, caching)
  ├── indicator_calculator.py (20 indicator functions)
  ├── ml_model.py             (training, prediction, SHAP)
  └── config.py               (all settings)

scheduler.py
  ├── data_fetcher.py        (called to refresh cache)
  └── config.py              (refresh times, paths)

bear_risk_api.py             (StockMind bridge — reads cache directly)
  └── config.py

personal_alerts.py
  ├── bear_risk_api.py       (reads current scores)
  └── config.py              (thresholds, email settings)

start_dashboard.sh / .bat
  ├── scheduler.py           (starts in background)
  ├── personal_alerts.py     (runs once at launch)
  └── app.py                 (starts Streamlit)
```

---

## StockMind Integration

Import the API bridge into your StockMind codebase with zero modifications:

```python
from bear_risk_api import (
    get_current_risk_score,
    get_ml_probabilities,
    get_top_risk_drivers,
    get_macro_context_summary,
    get_indicator_status,
)

# Example: macro overlay in stock analysis panel
risk = get_current_risk_score()
summary = get_macro_context_summary()

if risk and risk["band"] in ("HIGH", "SEVERE"):
    show_warning(f"⚠ Macro risk elevated: {risk['score']:.0f}/100 — {summary}")
```

All five functions read from local cache only — no network calls, no API keys required at integration time.

---

*Bear Risk Dashboard · v1.0 · For personal use and StockMind integration*
