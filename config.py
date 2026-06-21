"""
config.py — Single source of truth for the AI Bear Risk Dashboard.

All API keys, date ranges, caching rules, indicator metadata, and risk-band
definitions live here.  No other module may hardcode these values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── Load .env (silently ignored if file absent) ────────────────────────────────
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# API KEYS
# ═══════════════════════════════════════════════════════════════════════════════

def get_fred_api_key() -> str:
    """Return FRED API key from environment; raise with a clear message if absent."""
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY is not set.\n"
            "  1. Create a free account at https://fred.stlouisfed.org/\n"
            "  2. Generate an API key under My Account → API Keys\n"
            "  3. Set it via:\n"
            "       export FRED_API_KEY='your_key_here'     # shell\n"
            "     or add FRED_API_KEY=your_key_here to a .env file\n"
            "       in the project root."
        )
    return key


# ═══════════════════════════════════════════════════════════════════════════════
# DATE RANGES
# ═══════════════════════════════════════════════════════════════════════════════

START_DATE: date = date(1985, 1, 1)   # Earliest date to pull (some series shorter)
END_DATE:   date = date.today()        # Always pull through today


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_DIR: Path = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Maximum age (days) before a cached file triggers a live API refresh.
# The dashboard still works offline if the cache exists — age check is
# bypassed gracefully when internet is unavailable.
CACHE_EXPIRY_DAYS: dict[str, int] = {
    "daily":     1,
    "weekly":    7,
    "monthly":  30,
    "quarterly": 90,
}


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR METADATA — single source of truth
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IndicatorMeta:
    """All metadata needed to fetch, score, and display a single indicator."""

    id: int                          # 1-based index matching framework doc
    name: str                        # Human-readable label
    category: str                    # One of the 7 framework categories
    weight_pct: float                # Baseline weight (0-100, sums to 100)
    frequency: str                   # "daily" | "weekly" | "monthly" | "quarterly"
    publication_lag_days: int        # Expected delay between reference period and release
    scoring_method: str              # "A" (percentile) | "B" (threshold) | "C" (momentum)
    direction: str                   # "higher_worse" | "lower_worse" | "bidirectional"
    fred_series: list[str] = field(default_factory=list)   # FRED series IDs required
    yf_tickers:  list[str] = field(default_factory=list)   # yfinance tickers required
    special_source: Optional[str]    = None                # "shiller" | "aaii" | None

    # Thresholds for Method B piecewise linear scoring
    # [x_min, green_boundary, amber_boundary, x_max]
    thresholds: Optional[list[float]] = None

    # Percentile-method rolling window in periods (months or quarters)
    percentile_window: Optional[int] = None

    # Regime weight overrides: {regime: weight_pct}
    regime_weights: dict[str, float] = field(default_factory=dict)


INDICATORS: dict[int, IndicatorMeta] = {

    # ── Category 1: Valuations ─────────────────────────────────────────────────

    1: IndicatorMeta(
        id=1, name="Shiller CAPE Ratio",
        category="Valuations", weight_pct=7.0,
        frequency="monthly", publication_lag_days=5,
        scoring_method="A", direction="higher_worse",
        fred_series=["DFII10"],          # needed for ERP cross-check
        special_source="shiller",
        percentile_window=480,           # 40-year window
        regime_weights={"A": 5.0, "B": 7.0, "C": 4.0},
    ),
    2: IndicatorMeta(
        id=2, name="Price-to-Sales Ratio",
        category="Valuations", weight_pct=4.0,
        frequency="daily", publication_lag_days=1,
        scoring_method="A", direction="higher_worse",
        yf_tickers=["SPY"],
        percentile_window=360,
        regime_weights={"A": 4.0, "B": 4.0, "C": 2.0},
    ),
    3: IndicatorMeta(
        id=3, name="Equity Risk Premium",
        category="Valuations", weight_pct=6.0,
        frequency="daily", publication_lag_days=1,
        scoring_method="A", direction="lower_worse",
        fred_series=["DFII10"],
        special_source="shiller",        # CAPE denominator comes from Shiller
        percentile_window=360,
        regime_weights={"A": 3.0, "B": 9.0, "C": 6.0},
    ),

    # ── Category 2: Macro & Policy ─────────────────────────────────────────────

    4: IndicatorMeta(
        id=4, name="Yield Curve (10Y-2Y)",
        category="Macro & Policy", weight_pct=9.0,
        frequency="daily", publication_lag_days=1,
        scoring_method="A", direction="lower_worse",
        fred_series=["T10Y2Y"],
        percentile_window=600,
        regime_weights={"A": 6.0, "B": 11.0, "C": 9.0},
    ),
    5: IndicatorMeta(
        id=5, name="Real Federal Funds Rate",
        category="Macro & Policy", weight_pct=7.0,
        frequency="monthly", publication_lag_days=30,
        scoring_method="B", direction="higher_worse",
        fred_series=["FEDFUNDS", "PCEPILFE"],
        thresholds=[-5.0, 0.0, 2.0, 6.0],
        regime_weights={"A": 7.0, "B": 10.0, "C": 7.0},
    ),
    6: IndicatorMeta(
        id=6, name="ISM Manufacturing PMI",
        category="Macro & Policy", weight_pct=4.0,
        frequency="monthly", publication_lag_days=3,
        scoring_method="B", direction="lower_worse",
        # FRED permanently removed ALL ISM data in June 2016 — no ISM series ID works.
        # Replacement: IPMAN (Industrial Production: Manufacturing, NAICS) — FRED 1972-present.
        # Correlates ~0.85 with ISM PMI historically. Thresholds recalibrated for index scale.
        fred_series=["IPMAN"],
        thresholds=[70.0, 103.0, 97.0, 130.0],
        regime_weights={"A": 4.0, "B": 2.0, "C": 4.0},
    ),
    7: IndicatorMeta(
        id=7, name="Sahm Rule Indicator",
        category="Macro & Policy", weight_pct=5.0,
        frequency="monthly", publication_lag_days=7,
        scoring_method="B", direction="higher_worse",
        fred_series=["SAHMREALTIME"],
        thresholds=[0.0, 0.2, 0.5, 2.0],
        regime_weights={"A": 5.0, "B": 3.0, "C": 8.0},
    ),

    # ── Category 3: Credit & Liquidity ─────────────────────────────────────────

    8: IndicatorMeta(
        id=8, name="HY Credit Spread (OAS)",
        category="Credit & Liquidity", weight_pct=8.0,
        frequency="daily", publication_lag_days=1,
        scoring_method="A", direction="higher_worse",
        fred_series=["BAMLH0A0HYM2"],
        percentile_window=336,
        regime_weights={"A": 11.0, "B": 8.0, "C": 11.0},
    ),
    9: IndicatorMeta(
        id=9, name="IG Credit Spread (OAS)",
        category="Credit & Liquidity", weight_pct=5.0,
        frequency="daily", publication_lag_days=1,
        scoring_method="A", direction="higher_worse",
        fred_series=["BAMLC0A0CM"],
        percentile_window=324,
        regime_weights={"A": 7.0, "B": 5.0, "C": 5.0},
    ),
    10: IndicatorMeta(
        id=10, name="Chicago Fed NFCI",
        category="Credit & Liquidity", weight_pct=5.0,
        frequency="weekly", publication_lag_days=5,
        scoring_method="A", direction="higher_worse",
        fred_series=["NFCI"],
        percentile_window=600,
        regime_weights={"A": 5.0, "B": 5.0, "C": 9.0},
    ),

    # ── Category 4: Corporate Behavior & Financing ─────────────────────────────

    11: IndicatorMeta(
        id=11, name="Corporate Debt/GDP",
        category="Corporate Behavior", weight_pct=4.0,
        frequency="quarterly", publication_lag_days=60,
        scoring_method="A", direction="higher_worse",
        fred_series=["BCNSDODNS", "GDP"],
        percentile_window=120,
        regime_weights={"A": 4.0, "B": 4.0, "C": 4.0},
    ),
    12: IndicatorMeta(
        id=12, name="Net Equity Issuance",
        category="Corporate Behavior", weight_pct=3.0,
        frequency="quarterly", publication_lag_days=90,
        scoring_method="C", direction="higher_worse",
        fred_series=["NCBEILQ027S"],
        regime_weights={"A": 3.0, "B": 3.0, "C": 3.0},
    ),
    13: IndicatorMeta(
        id=13, name="Earnings Revision Breadth",
        category="Corporate Behavior", weight_pct=5.0,
        frequency="weekly", publication_lag_days=0,
        scoring_method="C", direction="lower_worse",
        yf_tickers=["XLF", "XLK", "XLY", "XLI", "XLP", "XLV", "XLE", "XLU", "XLB", "XLRE", "XLC"],
        regime_weights={"A": 7.0, "B": 5.0, "C": 5.0},
    ),

    # ── Category 5: Fundamentals & Profitability ───────────────────────────────

    14: IndicatorMeta(
        id=14, name="S&P 500 Profit Margin",
        category="Fundamentals", weight_pct=4.0,
        frequency="quarterly", publication_lag_days=60,
        scoring_method="B", direction="lower_worse",
        fred_series=["CPATAX", "GDP"],
        thresholds=[4.0, 10.0, 8.0, 14.0],
        regime_weights={"A": 4.0, "B": 4.0, "C": 4.0},
    ),
    15: IndicatorMeta(
        id=15, name="Interest Coverage Ratio",
        category="Fundamentals", weight_pct=4.0,
        frequency="quarterly", publication_lag_days=90,
        scoring_method="B", direction="lower_worse",
        # BOGZ1FL103070005Q (Level series) was retired.
        # Replacement: BOGZ1FA106130001Q (Flow/Transactions series) — FRED Q4 1946-present.
        # "FA" = Financial Accounts transactions flow; "FL" = level/stock. Use FA series.
        fred_series=["CPATAX", "BOGZ1FA106130001Q"],
        thresholds=[1.0, 5.0, 3.0, 10.0],
        regime_weights={"A": 4.0, "B": 6.0, "C": 4.0},
    ),

    # ── Category 6: Sentiment & Positioning ───────────────────────────────────

    16: IndicatorMeta(
        id=16, name="AAII Bull-Bear Spread",
        category="Sentiment", weight_pct=4.0,
        frequency="weekly", publication_lag_days=7,
        scoring_method="B", direction="bidirectional",
        special_source="aaii",
        thresholds=[-60.0, 30.0, -20.0, 60.0],   # [x_min, G_high, G_low, x_max]
        regime_weights={"A": 4.0, "B": 2.0, "C": 4.0},
    ),
    17: IndicatorMeta(
        id=17, name="VIX Level & Trend",
        category="Sentiment", weight_pct=5.0,
        frequency="daily", publication_lag_days=0,
        scoring_method="A", direction="higher_worse",
        fred_series=["VIXCLS"],
        yf_tickers=["^VIX"],
        percentile_window=408,
        regime_weights={"A": 5.0, "B": 5.0, "C": 7.0},
    ),
    18: IndicatorMeta(
        id=18, name="Conference Board LEI",
        category="Sentiment", weight_pct=5.0,
        frequency="monthly", publication_lag_days=21,
        scoring_method="B", direction="lower_worse",
        fred_series=["USSLIND"],
        thresholds=[-15.0, 1.5, -2.0, 8.0],
        regime_weights={"A": 5.0, "B": 5.0, "C": 3.0},
    ),

    # ── Category 7: Technical & Regime ────────────────────────────────────────

    19: IndicatorMeta(
        id=19, name="Market Breadth (% > 200D MA)",
        category="Technical & Regime", weight_pct=6.0,
        frequency="daily", publication_lag_days=0,
        scoring_method="C", direction="lower_worse",
        yf_tickers=["^GSPC"],    # constituent list built at runtime
        regime_weights={"A": 8.0, "B": 6.0, "C": 6.0},
    ),
    20: IndicatorMeta(
        id=20, name="S&P 500 Price Momentum",
        category="Technical & Regime", weight_pct=6.0,
        frequency="daily", publication_lag_days=0,
        scoring_method="C", direction="lower_worse",
        yf_tickers=["^GSPC"],
        regime_weights={"A": 6.0, "B": 6.0, "C": 6.0},
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# RISK BAND DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RiskBand:
    label: str
    color: str          # CSS / Plotly color name
    low: float          # inclusive lower bound
    high: float         # inclusive upper bound
    action: str


RISK_BANDS: list[RiskBand] = [
    RiskBand("LOW",      "green",  0,   24,  "No defensive adjustment. Normal allocation. Review monthly."),
    RiskBand("GUARDED",  "teal",  25,   44,  "Review equity allocation. Consider adding bond duration. Do not act on single-band entry."),
    RiskBand("ELEVATED", "gold",  45,   59,  "Reduce equity 10–20%. Add quality/defensive tilt. Review weekly."),
    RiskBand("HIGH",     "orange",60,   74,  "Reduce equity 20–35%. Increase cash and short-duration. Begin trailing stops."),
    RiskBand("SEVERE",   "red",   75,  100,  "Maximum defensive posture. Consider 40–60% equity reduction. Review hedges daily."),
]


def get_risk_band(score: float) -> RiskBand:
    """Return the RiskBand for a composite score in [0, 100]."""
    for band in RISK_BANDS:
        if band.low <= score <= band.high:
            return band
    # Clamp edge cases
    return RISK_BANDS[-1] if score > 100 else RISK_BANDS[0]


# ═══════════════════════════════════════════════════════════════════════════════
# SHILLER DATA URL
# ═══════════════════════════════════════════════════════════════════════════════

SHILLER_DATA_URL: str = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
SHILLER_CACHE_FILE: Path = CACHE_DIR / "shiller_ie_data.xls"

# ═══════════════════════════════════════════════════════════════════════════════
# AAII DATA URL
# ═══════════════════════════════════════════════════════════════════════════════

AAII_CSV_URL: str = "https://www.aaii.com/sentimentsurvey/sent_results"
AAII_CACHE_FILE: Path = CACHE_DIR / "aaii_sentiment.csv"

# ═══════════════════════════════════════════════════════════════════════════════
# S&P 500 CONSTITUENT LIST (Wikipedia, updated quarterly)
# ═══════════════════════════════════════════════════════════════════════════════

SP500_CONSTITUENTS_URL: str = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)
SP500_TICKERS_CACHE: Path = CACHE_DIR / "sp500_tickers.txt"

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING FORMAT
# ═══════════════════════════════════════════════════════════════════════════════

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ═══════════════════════════════════════════════════════════════════════════════
# PERSONAL ALERT THRESHOLDS
# Edit these values to customize when you receive desktop/email notifications.
# ═══════════════════════════════════════════════════════════════════════════════

# Composite score thresholds (0–100).  Alert fires when score crosses these
# levels in an upward direction since the last check.
ALERT_SCORE_ELEVATED: float = 50.0   # Composite crosses into ELEVATED band
ALERT_SCORE_HIGH: float = 70.0       # Composite crosses into HIGH band

# Single-indicator threshold: alert fires when any indicator's score exceeds
# this value (indicator scores are 0–100 just like the composite).
ALERT_INDICATOR_RED_THRESHOLD: float = 66.0   # Above this = indicator is "red"

# Email alert settings (leave EMAIL_ENABLED = False to use desktop-only).
# Uses Gmail SMTP with an App Password — never your main account password.
# Generate an App Password at: https://myaccount.google.com/apppasswords
EMAIL_ENABLED: bool = False
EMAIL_SENDER: str = ""          # e.g. "yourname@gmail.com"
EMAIL_PASSWORD: str = ""        # Gmail App Password (16-char, no spaces)
EMAIL_RECIPIENT: str = ""       # Who gets the alert (can be same as sender)
EMAIL_SMTP_HOST: str = "smtp.gmail.com"
EMAIL_SMTP_PORT: int = 587

# State file: stores last-seen score to detect threshold crossings.
ALERT_STATE_FILE: Path = CACHE_DIR / "alert_state.json"

# Scheduler: time-of-day for each refresh job (24-hour local time).
SCHEDULE_DAILY_TIME: str = "06:30"    # Daily indicators
SCHEDULE_WEEKLY_TIME: str = "07:00"   # Weekly indicators (Monday)
SCHEDULE_MONTHLY_TIME: str = "07:00"  # Monthly indicators (1st of month)
