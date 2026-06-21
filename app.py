"""
app.py — AI Bear Risk Dashboard
================================
Morning dashboard: 60-second read on macro bear market risk.

HOW TO RUN
----------
1. Install dependencies:
       pip install -r requirements.txt

2. Set your FRED API key (one-time):
       export FRED_API_KEY="your_key_here"
   or add to a .env file in this directory.

3. Launch:
       streamlit run app.py

4. On first launch, data is fetched from FRED / Yahoo Finance (~2-4 min).
   Subsequent launches use the local cache (<10 sec).

ARCHITECTURE
------------
app.py  ──calls──▶  data_fetcher.fetch_all_data()
                    indicator_calculator.calculate_all_indicators()
                    scorer.compute_composite_score()   [inline below]
                    ml_model.predict_current()
                    ml_model.run_backtest()

All heavy computation is cached via @st.cache_data / @st.cache_resource.
Tab switching never triggers re-computation.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import logging
from datetime import datetime, timedelta
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Local modules ─────────────────────────────────────────────────────────────
import config
from config import INDICATORS, get_risk_band, RISK_BANDS
from data_fetcher import fetch_all_data
from indicator_calculator import calculate_all_indicators
import ml_model

def _fmt_score(val) -> str:
    """Safely format a score float; returns - for NaN/None."""
    try:
        import pandas as pd
        if val is None or pd.isna(float(val)):
            return "-"
        return f"{float(val):.0f}"
    except Exception:
        return "-"


log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bear Risk Dashboard",
    page_icon="🐻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — dark terminal aesthetic
# ═══════════════════════════════════════════════════════════════════════════════
DARK_CSS = """
<style>
/* ── Fonts ─────────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ────────────────────────────────────────────────────────── */
:root {
    --bg:          #0a0e1a;
    --bg-panel:    #0f1525;
    --bg-card:     #141928;
    --border:      #1e2a42;
    --border-bright: #2a3a5c;
    --text-primary:  #e2e8f4;
    --text-muted:    #6b7fa8;
    --text-dim:      #3d4f72;

    --green:   #00c853;
    --amber:   #ffb300;
    --orange:  #ff6d00;
    --red:     #d50000;
    --severe:  #7b0000;
    --teal:    #00bcd4;

    --accent:  #4f8ef7;
    --accent2: #7c6af7;

    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
}

/* ── Global resets ───────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
    font-family: var(--sans) !important;
}

/* Main content area */
.main .block-container {
    background: var(--bg) !important;
    padding-top: 1rem !important;
    max-width: 1400px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton button {
    background: transparent !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text-primary) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.06em !important;
    padding: 6px 16px !important;
    border-radius: 3px !important;
    transition: border-color 0.15s, color 0.15s !important;
}
.stButton button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Metrics ─────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--mono) !important;
    font-size: 11px !important;
}

/* ── DataFrames ──────────────────────────────────────────────────────────── */
.stDataFrame {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
}

/* ── Sliders ─────────────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] {
    padding: 0 !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.06em !important;
    color: var(--text-muted) !important;
}

/* ── Score gauge card ────────────────────────────────────────────────────── */
.gauge-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
}

/* ── Risk pills ──────────────────────────────────────────────────────────── */
.pill {
    display: inline-block;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 3px 10px;
    border-radius: 12px;
    margin: 2px;
}
.pill-green  { background: rgba(0,200,83,0.15);  color: #00c853; border: 1px solid #00c853; }
.pill-amber  { background: rgba(255,179,0,0.15); color: #ffb300; border: 1px solid #ffb300; }
.pill-orange { background: rgba(255,109,0,0.15); color: #ff6d00; border: 1px solid #ff6d00; }
.pill-red    { background: rgba(213,0,0,0.15);   color: #d50000; border: 1px solid #d50000; }
.pill-teal   { background: rgba(0,188,212,0.15); color: #00bcd4; border: 1px solid #00bcd4; }

/* ── Driver cards ────────────────────────────────────────────────────────── */
.driver-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 14px;
    margin: 4px 0;
    font-family: var(--mono);
    font-size: 12px;
}
.driver-up   { border-left: 3px solid var(--red) !important; }
.driver-down { border-left: 3px solid var(--green) !important; }
.driver-flat { border-left: 3px solid var(--text-dim) !important; }

/* ── Score number ────────────────────────────────────────────────────────── */
.score-display {
    font-family: var(--mono);
    font-size: 72px;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.02em;
}
.band-label {
    font-family: var(--mono);
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Footer disclaimer ───────────────────────────────────────────────────── */
.disclaimer {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 0.04em;
    border-top: 1px solid var(--border);
    padding-top: 8px;
    margin-top: 16px;
}

/* ── Freshness badges ────────────────────────────────────────────────────── */
.fresh  { color: var(--green); }
.stale  { color: var(--amber); }
.overdue { color: var(--red); }

/* ── Mono label ──────────────────────────────────────────────────────────── */
.mono-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.mono-value {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text-primary);
}

/* ── Section heading ─────────────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
</style>
"""

# Risk colour map keyed by band label
BAND_COLORS = {
    "LOW":      "#00c853",
    "GUARDED":  "#00bcd4",
    "ELEVATED": "#ffb300",
    "HIGH":     "#ff6d00",
    "SEVERE":   "#d50000",
}
BAND_CSS = {
    "LOW":      "pill-green",
    "GUARDED":  "pill-teal",
    "ELEVATED": "pill-amber",
    "HIGH":     "pill-orange",
    "SEVERE":   "pill-red",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f1525",
        font=dict(family="IBM Plex Mono", color="#e2e8f4", size=11),
        xaxis=dict(gridcolor="#1e2a42", linecolor="#1e2a42", zerolinecolor="#1e2a42"),
        yaxis=dict(gridcolor="#1e2a42", linecolor="#1e2a42", zerolinecolor="#1e2a42"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2a42"),
        margin=dict(l=50, r=20, t=40, b=40),
    )
)

DISCLAIMER = (
    "⚠ Educational tool only. Not financial advice. "
    "Model trained on ~8 structural bear market samples. "
    "Treat probabilities as directional signals, not precise forecasts. "
    "Past performance does not predict future results."
)

# ═══════════════════════════════════════════════════════════════════════════════
# CACHED DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Fetching market data…")
def load_raw_data() -> dict:
    """Fetch all raw data. Cached 1 hour; uses local cache files offline."""
    return fetch_all_data()


@st.cache_data(ttl=3600, show_spinner="Computing indicators…")
def load_indicators(raw_hash: str) -> pd.DataFrame:  # raw_hash used to bust cache
    """Calculate all 20 indicators from raw data."""
    raw = load_raw_data()
    return calculate_all_indicators(raw)


@st.cache_resource(show_spinner="Loading ML models…")
def load_ml_models() -> Optional[dict]:
    """Load pre-trained model stack from disk."""
    models = {}
    for h in ml_model.HORIZONS:
        m = ml_model.load_latest_model(h)
        if m is not None:
            models[h] = m
    return models if models else None


@st.cache_data(ttl=3600, show_spinner="Generating predictions…")
def load_predictions(indicator_hash: str) -> Optional[dict]:
    """Run predict_current() and cache result."""
    indicators = load_indicators(indicator_hash)
    models = load_ml_models()
    composite = compute_composite_series(indicators)
    return ml_model.predict_current(indicators, composite, models)


@st.cache_data(ttl=86400, show_spinner="Running backtest…")
def load_backtest(indicator_hash: str) -> Optional[pd.DataFrame]:
    """Run backtesting evaluation. Cached 24 hours (expensive)."""
    models = load_ml_models()
    if models is None:
        return None
    indicators = load_indicators(indicator_hash)
    raw = load_raw_data()
    # Build targets
    sp500 = raw.get("sp500_price")
    if sp500 is not None and not isinstance(sp500, pd.DataFrame):
        sp500_s = sp500
    elif sp500 is not None:
        sp500_s = sp500.iloc[:, 0]
    else:
        return None
    usrec_df = raw.get("USREC")
    usrec = usrec_df.iloc[:, 0] if usrec_df is not None else None
    targets = ml_model.build_targets(sp500_s, usrec)
    features = ml_model.engineer_features(indicators)
    return ml_model.run_backtest(features, targets, sp500_s, models)


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING HELPERS  (inline — no extra import)
# ═══════════════════════════════════════════════════════════════════════════════

def score_indicator_percentile(
    value: float,
    history: pd.Series,
    window: int,
    invert: bool = False,
) -> float:
    """Method A: percentile rank within rolling window → [0, 100]."""
    if pd.isna(value) or len(history.dropna()) < 12:
        return np.nan
    h = history.dropna().iloc[-window:]
    p01, p99 = h.quantile(0.01), h.quantile(0.99)
    val = np.clip(value, p01, p99)
    rank = (h <= val).sum() / len(h)
    score = (1 - rank if invert else rank) * 100
    return float(np.clip(score, 0, 100))


def score_indicator_threshold(
    value: float,
    x_min: float,
    green: float,
    amber: float,
    x_max: float,
    direction: str = "higher_worse",
) -> float:
    """Method B: piecewise linear interpolation → [0, 100]."""
    if pd.isna(value):
        return np.nan
    if direction == "higher_worse":
        if value <= green:
            s = 33.0 * max(0, value - x_min) / max(green - x_min, 1e-9)
        elif value <= amber:
            s = 33.0 + 34.0 * (value - green) / max(amber - green, 1e-9)
        else:
            s = 66.0 + 34.0 * min(value - amber, x_max - amber) / max(x_max - amber, 1e-9)
    else:  # lower_worse: green > amber
        if value >= green:
            s = 33.0 - 33.0 * min(value - green, x_max - green) / max(x_max - green, 1e-9)
        elif value >= amber:
            s = 66.0 - 33.0 * (value - amber) / max(green - amber, 1e-9)
        else:
            s = 100.0 - 34.0 * max(value - x_min, 0) / max(amber - x_min, 1e-9)
    return float(np.clip(s, 0, 100))


def score_aaii(value: float) -> float:
    """Special bidirectional AAII scoring."""
    if pd.isna(value):
        return np.nan
    if value < -20:
        s = np.clip(33.0 + 33.0 * (value + 20) / 20.0, 0, 33)
    elif value <= 30:
        s = 33.0 + 33.0 * (value + 20) / 50.0
    else:
        s = 66.0 + 34.0 * (value - 30) / 30.0
    return float(np.clip(s, 0, 100))


def compute_indicator_scores(indicators: pd.DataFrame) -> pd.DataFrame:
    """
    Score all 20 indicators for the current (latest) row.
    Returns a DataFrame with columns [name, raw_value, score, direction, staleness_days].
    """
    rows = []
    latest_date = indicators.index[-1]

    for ind_id, meta in INDICATORS.items():
        col = meta.name
        if col not in indicators.columns:
            rows.append({
                "id": ind_id, "name": meta.name, "category": meta.category,
                "raw_value": np.nan, "score": np.nan,
                "direction": "→", "staleness_days": np.nan,
                "weight": meta.weight_pct,
            })
            continue

        series = indicators[col].dropna()
        if series.empty:
            rows.append({
                "id": ind_id, "name": meta.name, "category": meta.category,
                "raw_value": np.nan, "score": np.nan,
                "direction": "→", "staleness_days": np.nan,
                "weight": meta.weight_pct,
            })
            continue

        raw = float(series.iloc[-1])
        last_date = series.index[-1]
        staleness = (latest_date - last_date).days

        # Score
        method = meta.scoring_method
        direction_str = meta.direction

        if method == "A":
            window = meta.percentile_window or 360
            invert = direction_str == "lower_worse"
            score = score_indicator_percentile(raw, series, window, invert)
        elif method == "B":
            if meta.direction == "bidirectional":
                score = score_aaii(raw)
            else:
                thresholds = meta.thresholds or [-5, 0, 2, 6]
                x_min, g, a, x_max = thresholds
                score = score_indicator_threshold(raw, x_min, g, a, x_max, direction_str)
        else:  # Method C: level + momentum 50/50
            thresholds = meta.thresholds
            if thresholds:
                x_min, g, a, x_max = thresholds
                s_level = score_indicator_threshold(raw, x_min, g, a, x_max, direction_str)
            else:
                s_level = 50.0
            # Momentum component
            if len(series) >= 14:
                chg = series.diff(13)
                recent_chg = float(chg.iloc[-1]) if not pd.isna(chg.iloc[-1]) else 0
                mu, sigma = chg.mean(), chg.std()
                if sigma > 0.001:
                    z = (recent_chg - mu) / sigma
                    if direction_str == "lower_worse":
                        s_mom = float(np.clip(50 - 10 * z, 0, 100))
                    else:
                        s_mom = float(np.clip(50 + 10 * z, 0, 100))
                else:
                    s_mom = 50.0
            else:
                s_mom = 50.0
            score = 0.5 * (s_level or 50) + 0.5 * s_mom

        # Trend arrow
        if len(series) >= 3:
            delta = float(series.iloc[-1]) - float(series.iloc[-3])
            if abs(delta) < 1e-6:
                arrow = "→"
            elif direction_str == "lower_worse":
                arrow = "↓ improving" if delta > 0 else "↑ worsening"
            elif direction_str == "bidirectional":
                arrow = "→"
            else:
                arrow = "↑ worsening" if delta > 0 else "↓ improving"
        else:
            arrow = "→"

        rows.append({
            "id": ind_id, "name": meta.name, "category": meta.category,
            "raw_value": raw, "score": score,
            "direction": arrow, "staleness_days": staleness,
            "weight": meta.weight_pct,
        })

    return pd.DataFrame(rows).set_index("id")


def compute_composite_series(indicators: pd.DataFrame) -> pd.Series:
    """
    Compute the full historical composite score series.
    Simplified version (no regime weighting) for the historical chart.
    Uses baseline weights with staleness decay applied.
    """
    scores_list = []
    for date in indicators.index[-240:]:  # last 20 years for performance
        row = indicators.loc[:date]
        score_df = compute_indicator_scores(row)
        valid = score_df[score_df["score"].notna()]
        if len(valid) < 5:
            scores_list.append((date, np.nan))
            continue
        w = valid["weight"].values / valid["weight"].sum()
        s = valid["score"].values
        # Amplifier
        w_mean = float((w * s).sum())
        w_std = float(np.sqrt((w * (s - w_mean) ** 2).sum()))
        amplifier = 1 + 0.15 * max(0, (w_std - 20) / 30)
        composite = min(100, w_mean * amplifier)
        scores_list.append((date, composite))

    if not scores_list:
        return pd.Series(dtype=float)
    idx, vals = zip(*scores_list)
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name="CompositeScore")


def _hash_df(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame for cache busting."""
    try:
        return str(hash(str(df.shape) + str(df.index[-1]) + str(df.iloc[-1].sum())))
    except Exception:
        return "default"


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_gauge(score: float, band_label: str, band_color: str) -> go.Figure:
    """Large composite score gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"family": "IBM Plex Mono", "size": 48, "color": band_color},
                "suffix": ""},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#2a3a5c", "nticks": 6,
                     "tickfont": {"family": "IBM Plex Mono", "size": 10, "color": "#6b7fa8"}},
            "bar": {"color": band_color, "thickness": 0.25},
            "bgcolor": "#0f1525",
            "borderwidth": 1,
            "bordercolor": "#1e2a42",
            "steps": [
                {"range": [0, 24],  "color": "rgba(0,200,83,0.08)"},
                {"range": [24, 44], "color": "rgba(0,188,212,0.08)"},
                {"range": [44, 59], "color": "rgba(255,179,0,0.08)"},
                {"range": [59, 74], "color": "rgba(255,109,0,0.08)"},
                {"range": [74, 100],"color": "rgba(213,0,0,0.08)"},
            ],
            "threshold": {
                "line": {"color": band_color, "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
        title={"text": band_label,
               "font": {"family": "IBM Plex Mono", "size": 13, "color": band_color}},
    ))
    fig.update_layout(
        paper_bgcolor="#141928",
        font={"family": "IBM Plex Mono", "color": "#e2e8f4"},
        height=240,
        margin=dict(l=20, r=20, t=20, b=10),
    )
    return fig


def make_heatmap_cell_chart(series: pd.Series, meta) -> go.Figure:
    """Tiny sparkline for a single indicator within expander."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines",
        line=dict(color="#4f8ef7", width=1.5),
        name=meta.name,
    ))
    # Threshold zones if available
    if meta.thresholds:
        x_min, g, a, x_max = meta.thresholds
        fig.add_hrect(y0=x_min, y1=g, fillcolor="rgba(0,200,83,0.06)",
                      line_width=0, annotation_text="Green zone")
        fig.add_hrect(y0=g, y1=a, fillcolor="rgba(255,179,0,0.06)", line_width=0)
        fig.add_hrect(y0=a, y1=x_max, fillcolor="rgba(213,0,0,0.06)", line_width=0)
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=200, showlegend=False,
        title=dict(text=meta.name, font=dict(size=11)),
        margin=dict(l=40, r=10, t=30, b=30),
    )
    return fig


def make_composite_history_chart(
    composite: pd.Series,
    category_scores: Optional[dict[str, pd.Series]] = None,
    show_categories: bool = False,
    usrec: Optional[pd.Series] = None,
    bear_dates: Optional[list[tuple]] = None,
) -> go.Figure:
    """Full historical composite risk score chart with recession shading."""
    fig = go.Figure()

    # ── Recession shading ──────────────────────────────────────────────────────
    if usrec is not None:
        in_recession = False
        rec_start = None
        for date, val in usrec.items():
            if val == 1 and not in_recession:
                in_recession = True
                rec_start = date
            elif val == 0 and in_recession:
                in_recession = False
                fig.add_vrect(
                    x0=rec_start, x1=date,
                    fillcolor="rgba(100,100,150,0.12)",
                    line_width=0,
                    annotation_text="Recession",
                    annotation_font_size=9,
                    annotation_font_color="#6b7fa8",
                )

    # ── Bear market vertical lines ─────────────────────────────────────────────
    bear_peaks = [
        ("2000 Dot-com peak", "2000-03-24"),
        ("2007 GFC peak", "2007-10-09"),
        ("2020 COVID crash", "2020-02-19"),
        ("2022 Rate shock", "2022-01-03"),
    ]
    for label, date_str in bear_peaks:
        fig.add_vline(
            x=pd.Timestamp(date_str).timestamp() * 1000,
            line=dict(color="#d50000", width=1, dash="dot"),
            annotation_text=label.split(" ")[0],
            annotation_font_size=9,
            annotation_font_color="#d50000",
        )

    # ── Category overlays (optional) ──────────────────────────────────────────
    cat_colors = {
        "Valuations": "#7c6af7",
        "Macro & Policy": "#4f8ef7",
        "Credit & Liquidity": "#00bcd4",
        "Corporate Behavior": "#ffb300",
        "Fundamentals": "#ff6d00",
        "Sentiment": "#ab47bc",
        "Technical & Regime": "#26a69a",
    }
    if show_categories and category_scores:
        for cat, series in category_scores.items():
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values,
                mode="lines",
                name=cat,
                line=dict(color=cat_colors.get(cat, "#888"), width=1, dash="dot"),
                opacity=0.5,
            ))

    # ── Risk band fills ────────────────────────────────────────────────────────
    for low, high, color in [(75, 100, "rgba(213,0,0,0.04)"),
                              (60, 75, "rgba(255,109,0,0.04)"),
                              (45, 60, "rgba(255,179,0,0.04)")]:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, line_width=0)

    # ── Composite line ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=composite.index, y=composite.values,
        mode="lines",
        name="Composite Score",
        line=dict(color="#e2e8f4", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(79,142,247,0.06)",
    ))

    # ── Threshold annotations ──────────────────────────────────────────────────
    for y, text, color in [(60, "HIGH", "#ff6d00"), (75, "SEVERE", "#d50000")]:
        fig.add_hline(y=y, line=dict(color=color, width=0.8, dash="dash"),
                      annotation_text=text, annotation_font_size=9,
                      annotation_font_color=color, annotation_position="right")

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=360,
        title=dict(text="Bear Risk Composite Score — Historical", font=dict(size=13)),
        yaxis=dict(range=[0, 105], title="Risk Score"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def make_shap_bar(drivers: list[dict], title: str = "Top Risk Drivers") -> go.Figure:
    """Horizontal bar chart of SHAP top drivers."""
    if not drivers:
        return go.Figure()
    names = [d["feature"][:28] for d in drivers]
    vals  = [d["shap_value"] for d in drivers]
    colors = ["#d50000" if v > 0 else "#00c853" for v in vals]
    fig = go.Figure(go.Bar(
        y=names, x=vals,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10),
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=max(180, len(names) * 36),
        title=dict(text=title, font=dict(size=12)),
        xaxis=dict(title="SHAP value (positive = increases bear risk)"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=180, r=60, t=40, b=40),
    )
    return fig


def make_calibration_plot(y_true: np.ndarray, y_prob: np.ndarray, title: str) -> go.Figure:
    """Reliability diagram for calibration visualization."""
    from sklearn.calibration import calibration_curve
    n_bins = min(10, max(3, int(y_true.sum())))
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    except Exception:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="#2a3a5c", dash="dot"), name="Perfect calibration"))
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers",
                             line=dict(color="#4f8ef7", width=2),
                             marker=dict(size=6), name="Model"))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=280,
        title=dict(text=title, font=dict(size=12)),
        xaxis=dict(title="Mean predicted probability", range=[0, 1]),
        yaxis=dict(title="Fraction of positives", range=[0, 1]),
    )
    return fig


def score_to_css_class(score: float) -> str:
    if pd.isna(score): return "pill-teal"
    if score < 33: return "pill-green"
    if score < 66: return "pill-amber"
    return "pill-red"


def score_to_bar_color(score: float) -> str:
    if pd.isna(score): return "#2a3a5c"
    if score < 33: return "#00c853"
    if score < 66: return "#ffb300"
    return "#d50000"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    # ── Header bar ─────────────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'IBM Plex Mono\';font-size:11px;'
        'letter-spacing:0.15em;color:#6b7fa8;text-transform:uppercase;'
        'margin-bottom:4px;">AI Bear Risk Dashboard · StockMind</p>',
        unsafe_allow_html=True,
    )

    # ── Session state init ─────────────────────────────────────────────────────
    if "force_refresh" not in st.session_state:
        st.session_state.force_refresh = False
    if "show_categories" not in st.session_state:
        st.session_state.show_categories = False
    if "selected_indicator" not in st.session_state:
        st.session_state.selected_indicator = None

    # ── Data load ──────────────────────────────────────────────────────────────
    raw = load_raw_data()
    indicators = load_indicators(_hash_df(pd.DataFrame({"t": [datetime.utcnow().date()]})))
    indicator_hash = _hash_df(indicators)

    # Current indicator scores
    scores_df = compute_indicator_scores(indicators)
    composite_series = compute_composite_series(indicators)
    current_score = float(composite_series.dropna().iloc[-1]) if not composite_series.dropna().empty else 50.0
    band = get_risk_band(current_score)
    band_color = BAND_COLORS.get(band.label, "#e2e8f4")

    # Predictions
    predictions = load_predictions(indicator_hash)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "01 · OVERVIEW",
        "02 · INDICATOR HEATMAP",
        "03 · HISTORICAL RISK",
        "04 · WHAT-IF SCENARIO",
        "05 · MODEL DIAGNOSTICS",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW (Morning Dashboard, above-the-fold)
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        # Row 1: Gauge + ML probabilities + risk band
        col_gauge, col_probs, col_band = st.columns([2, 2, 2])

        with col_gauge:
            st.plotly_chart(
                make_gauge(current_score, band.label, band_color),
                use_container_width=True, config={"displayModeBar": False},
            )

        with col_probs:
            st.markdown('<p class="mono-label">ML Bear Probability</p>', unsafe_allow_html=True)
            p3  = predictions.get("bear_3m_prob",  np.nan) if predictions else np.nan
            p6  = predictions.get("bear_6m_prob",  np.nan) if predictions else np.nan
            p12 = predictions.get("bear_12m_prob", np.nan) if predictions else np.nan
            ci3  = predictions.get("bear_3m_ci",  (np.nan, np.nan)) if predictions else (np.nan, np.nan)
            ci6  = predictions.get("bear_6m_ci",  (np.nan, np.nan)) if predictions else (np.nan, np.nan)
            ci12 = predictions.get("bear_12m_ci", (np.nan, np.nan)) if predictions else (np.nan, np.nan)

            def _pct(v): return f"{v*100:.0f}%" if not pd.isna(v) else "N/A"
            def _ci(c):
                if pd.isna(c[0]): return ""
                return f"[{c[0]*100:.0f}–{c[1]*100:.0f}%]"

            p3_color  = score_to_bar_color((p3 or 0) * 100)
            p6_color  = score_to_bar_color((p6 or 0) * 100)
            p12_color = score_to_bar_color((p12 or 0) * 100)

            for label, p, ci, color in [
                ("Bear in 3 months", p3, ci3, p3_color),
                ("Bear in 6 months", p6, ci6, p6_color),
                ("Bear in 12 months", p12, ci12, p12_color),
            ]:
                st.markdown(
                    f'<div style="background:#141928;border:1px solid #1e2a42;'
                    f'border-radius:4px;padding:10px 14px;margin:4px 0;">'
                    f'<span class="mono-label">{label}</span><br>'
                    f'<span style="font-family:\'IBM Plex Mono\';font-size:28px;'
                    f'font-weight:700;color:{color};">{_pct(p)}</span>'
                    f'<span style="font-family:\'IBM Plex Mono\';font-size:11px;'
                    f'color:#6b7fa8;margin-left:8px;">{_ci(ci)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if predictions is None:
                st.caption("⚠ ML models not trained yet. Run training first.")

        with col_band:
            st.markdown('<p class="mono-label">Current Regime</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:#141928;border:1px solid {band_color}22;'
                f'border-left:4px solid {band_color};border-radius:4px;'
                f'padding:14px 16px;margin:4px 0;">'
                f'<span style="font-family:\'IBM Plex Mono\';font-size:20px;'
                f'font-weight:700;color:{band_color};">{band.label}</span><br>'
                f'<span style="font-family:\'IBM Plex Mono\';font-size:11px;'
                f'color:#a0b0d0;margin-top:6px;display:block;">{band.action}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Last updated + freshness
            last_date = indicators.index[-1]
            days_since = (datetime.utcnow().date() - last_date.date()).days
            freshness_class = "fresh" if days_since <= 1 else ("stale" if days_since <= 7 else "overdue")
            st.markdown(
                f'<div style="margin-top:8px;">'
                f'<span class="mono-label">Data as of </span>'
                f'<span class="mono-value">{last_date.strftime("%Y-%m-%d")}</span>'
                f'<span class="{freshness_class}" style="font-family:\'IBM Plex Mono\';'
                f'font-size:10px;margin-left:8px;">({days_since}d ago)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Refresh button
            if st.button("↺  Refresh Data", key="refresh_btn"):
                st.cache_data.clear()
                st.rerun()

        st.divider()

        # Row 2: Top 3 SHAP drivers + summary
        col_drivers, col_summary = st.columns([2, 3])

        with col_drivers:
            st.markdown('<p class="mono-label">Top Risk Drivers (SHAP · 12M Model)</p>', unsafe_allow_html=True)
            drivers_12m = []
            if predictions and "top_drivers" in predictions:
                td = predictions["top_drivers"]
                if isinstance(td, dict):
                    drivers_12m = td.get("bear_12m", td.get(12, []))
                elif isinstance(td, list):
                    drivers_12m = td

            if drivers_12m:
                for d in drivers_12m[:3]:
                    name = d.get("feature", "?")[:32]
                    shap_val = d.get("shap_value", 0)
                    direction_str = d.get("direction", "→")
                    arrow = "↑" if shap_val > 0 else "↓"
                    driver_color = "#d50000" if shap_val > 0 else "#00c853"
                    css_class = "driver-up" if shap_val > 0 else "driver-down"
                    st.markdown(
                        f'<div class="driver-card {css_class}">'
                        f'<span style="color:{driver_color};font-weight:700;">{arrow} </span>'
                        f'<span style="color:#e2e8f4;">{name}</span>'
                        f'<span style="float:right;color:{driver_color};">{shap_val:+.3f}</span>'
                        f'<br><span style="color:#6b7fa8;font-size:10px;">{direction_str}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Train ML models to see SHAP drivers.")

            # Also show top 3 indicators by score as fallback context
            st.markdown('<p class="mono-label" style="margin-top:12px;">Highest-Score Indicators</p>', unsafe_allow_html=True)
            top3 = scores_df.dropna(subset=["score"]).nlargest(3, "score")
            for _, row in top3.iterrows():
                s = row["score"]
                css = score_to_css_class(s)
                st.markdown(
                    f'<span class="pill {css}">{row["name"][:20]} {s:.0f}</span> ',
                    unsafe_allow_html=True,
                )

        with col_summary:
            st.markdown('<p class="mono-label">Situation Summary</p>', unsafe_allow_html=True)
            if predictions and "summary" in predictions:
                summary_text = predictions["summary"]
            else:
                # Fallback summary from deterministic score
                n_red   = int((scores_df["score"] >= 66).sum())
                n_amber = int(((scores_df["score"] >= 33) & (scores_df["score"] < 66)).sum())
                summary_text = (
                    f"Composite score is {current_score:.0f}/100 ({band.label}). "
                    f"{n_red} indicators in red zone, {n_amber} in amber. "
                    f"{band.action}"
                )
            st.markdown(
                f'<div style="background:#141928;border:1px solid #1e2a42;'
                f'border-radius:4px;padding:14px 16px;">'
                f'<p style="font-family:\'IBM Plex Sans\';font-size:14px;'
                f'line-height:1.6;color:#c8d4e8;margin:0;">{summary_text}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Staleness warnings
            stale_indicators = scores_df[
                scores_df["staleness_days"] > scores_df["staleness_days"].median() * 2
            ]
            if len(stale_indicators) > 0:
                stale_names = ", ".join(stale_indicators["name"].tolist()[:3])
                st.warning(f"⚠ Stale data: {stale_names}. Consider refreshing.")

        # Disclaimer footer
        st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — INDICATOR HEATMAP
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<p class="mono-label" style="margin-bottom:8px;">All 20 Indicators — Current Score</p>', unsafe_allow_html=True)
        st.caption("Click any row in the table below to expand the full indicator history.")

        # Build heatmap as a Plotly grid
        cats = [meta.category for meta in INDICATORS.values()]
        unique_cats = list(dict.fromkeys(cats))

        for cat in unique_cats:
            cat_rows = scores_df[scores_df["category"] == cat]
            if cat_rows.empty:
                continue
            st.markdown(
                f'<p style="font-family:\'IBM Plex Mono\';font-size:10px;'
                f'letter-spacing:0.1em;text-transform:uppercase;color:#3d4f72;'
                f'margin:12px 0 4px 0;">{cat}</p>',
                unsafe_allow_html=True,
            )
            cols = st.columns(min(len(cat_rows), 4))
            for i, (ind_id, row) in enumerate(cat_rows.iterrows()):
                with cols[i % 4]:
                    score = row["score"]
                    css = score_to_css_class(score)
                    bg_color = {"pill-green": "rgba(0,200,83,0.08)",
                                "pill-amber": "rgba(255,179,0,0.08)",
                                "pill-red":   "rgba(213,0,0,0.08)",
                                "pill-teal":  "rgba(0,188,212,0.08)"}.get(css, "transparent")
                    bar_color = score_to_bar_color(score)
                    bar_w = int(score) if not pd.isna(score) else 0
                    raw_display = f"{row['raw_value']:.2f}" if not pd.isna(row['raw_value']) else "N/A"
                    direction = row["direction"]

                    _score_str = (f"{_fmt_score(score)}" if (score is not None and not __import__('pandas').isna(score)) else "-")
        with st.expander(f"{row['name'][:22]}  ·  {_score_str}"):
                        # Progress bar
                        st.markdown(
                            f'<div style="background:#1e2a42;border-radius:3px;height:6px;margin-bottom:8px;">'
                            f'<div style="background:{bar_color};width:{bar_w}%;height:6px;border-radius:3px;"></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        # Metadata
                        meta = INDICATORS.get(ind_id)
                        freq_str = meta.frequency if meta else ""
                        lag_str  = f"{meta.publication_lag_days}d" if meta else ""
                        staleness = row["staleness_days"]
                        st.markdown(
                            f'<span class="mono-label">Raw:</span> '
                            f'<span class="mono-value">{raw_display}</span>&nbsp;&nbsp;'
                            f'<span class="mono-label">Trend:</span> '
                            f'<span class="mono-value">{direction}</span><br>'
                            f'<span class="mono-label">Freq:</span> '
                            f'<span class="mono-value">{freq_str}</span>&nbsp;&nbsp;'
                            f'<span class="mono-label">Lag:</span> '
                            f'<span class="mono-value">{lag_str}</span>&nbsp;&nbsp;'
                            f'<span class="mono-label">Staleness:</span> '
                            f'<span class="mono-value">{int(staleness) if not pd.isna(staleness) else "?"} days</span>',
                            unsafe_allow_html=True,
                        )
                        # Historical chart
                        if meta and meta.name in indicators.columns:
                            series = indicators[meta.name].dropna()
                            if len(series) >= 12:
                                st.plotly_chart(
                                    make_heatmap_cell_chart(series, meta),
                                    use_container_width=True,
                                    config={"displayModeBar": False},
                                )

        st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — HISTORICAL RISK SCORE
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        ctrl_col, _ = st.columns([2, 3])
        with ctrl_col:
            show_cats = st.checkbox("Show category overlays", value=st.session_state.show_categories,
                                    key="cat_toggle")
            st.session_state.show_categories = show_cats

        # Recession shading data
        usrec_df = raw.get("USREC")
        usrec_series = usrec_df.iloc[:, 0] if usrec_df is not None else None

        composite_chart = make_composite_history_chart(
            composite_series,
            show_categories=show_cats,
            usrec=usrec_series,
        )
        st.plotly_chart(composite_chart, use_container_width=True, config={"displayModeBar": True})

        # ML probability overlay
        if predictions and "shap_history" in predictions:
            shap_hist = predictions["shap_history"]
            if isinstance(shap_hist, dict):
                ml_hist = shap_hist.get("bear_12m") or shap_hist.get(12)
            else:
                ml_hist = None

            if ml_hist is not None and hasattr(ml_hist, "index"):
                # Build probability series from shap history (approximation)
                st.markdown('<p class="mono-label" style="margin-top:4px;">ML Bear Probability · 12M Horizon</p>', unsafe_allow_html=True)
                # The shap history is a DataFrame; we don't have probability series here
                # so skip this secondary chart unless predictions include it
                pass

        # Performance summary box
        st.markdown('<p class="mono-label" style="margin-top:8px;">Historical Signal Performance</p>', unsafe_allow_html=True)
        perf_data = [
            {"Bear Market": "2000 Dot-com",   "Peak": "Mar 2000", "Est. Score @ Peak": "72–78", "Band": "HIGH",     "Lead Time": "~12 months"},
            {"Bear Market": "2007–08 GFC",    "Peak": "Oct 2007", "Est. Score @ Peak": "58–64", "Band": "ELEVATED", "Lead Time": "~12–15 months"},
            {"Bear Market": "2020 COVID",      "Peak": "Feb 2020", "Est. Score @ Peak": "35–42", "Band": "GUARDED",  "Lead Time": "None (exogenous)"},
            {"Bear Market": "2022 Rate Shock", "Peak": "Jan 2022", "Est. Score @ Peak": "68–74", "Band": "HIGH",     "Lead Time": "Concurrent"},
        ]
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        st.caption("⚠ 2020 COVID excluded from primary validation — system designed for structural/cyclical bears only.")
        st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4 — WHAT-IF SCENARIO ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab4:
        # Identify 6 highest-impact indicators by average SHAP or by weight
        top_indicators_by_weight = (
            scores_df.nlargest(6, "weight")
            if "weight" in scores_df.columns
            else scores_df.head(6)
        )

        st.markdown(
            '<p class="mono-label">Adjust the 6 highest-weight indicators to model hypothetical scenarios.</p>',
            unsafe_allow_html=True,
        )

        if "scenario_values" not in st.session_state:
            st.session_state.scenario_values = {}

        col_sliders, col_result = st.columns([2, 1])

        with col_sliders:
            slider_values = {}
            for ind_id, row in top_indicators_by_weight.iterrows():
                meta = INDICATORS.get(ind_id)
                if meta is None or meta.name not in indicators.columns:
                    continue
                series = indicators[meta.name].dropna()
                if series.empty:
                    continue

                current_val = float(series.iloc[-1])
                s_min = float(series.min())
                s_max = float(series.max())
                s_range = s_max - s_min
                if s_range < 1e-6:
                    continue

                # Use stored scenario value or default to current
                default_val = st.session_state.scenario_values.get(str(ind_id), current_val)
                default_val = np.clip(default_val, s_min, s_max)

                new_val = st.slider(
                    label=f"{row['name']}  (current: {current_val:.2f})",
                    min_value=float(s_min),
                    max_value=float(s_max),
                    value=float(default_val),
                    step=float(s_range / 100),
                    key=f"scenario_{ind_id}",
                )
                slider_values[ind_id] = new_val
                st.session_state.scenario_values[str(ind_id)] = new_val

        # Compute scenario score
        scenario_indicators = indicators.copy()
        for ind_id, new_val in slider_values.items():
            meta = INDICATORS.get(ind_id)
            if meta and meta.name in scenario_indicators.columns:
                scenario_indicators.iloc[-1, scenario_indicators.columns.get_loc(meta.name)] = new_val

        scenario_scores_df = compute_indicator_scores(scenario_indicators)

        def _composite_from_scores(sdf: pd.DataFrame) -> float:
            valid = sdf.dropna(subset=["score"])
            if valid.empty:
                return 50.0
            w = valid["weight"].values / valid["weight"].sum()
            s = valid["score"].values
            c_linear = float((w * s).sum())
            w_std = float(np.sqrt((w * (s - c_linear) ** 2).sum()))
            amp = 1 + 0.15 * max(0, (w_std - 20) / 30)
            return float(min(100, c_linear * amp))

        current_composite = _composite_from_scores(scores_df)
        scenario_composite = _composite_from_scores(scenario_scores_df)
        current_band  = get_risk_band(current_composite)
        scenario_band = get_risk_band(scenario_composite)

        with col_result:
            st.markdown('<p class="mono-label">Scenario vs. Current</p>', unsafe_allow_html=True)

            cc = BAND_COLORS.get(current_band.label, "#e2e8f4")
            sc = BAND_COLORS.get(scenario_band.label, "#e2e8f4")
            delta = scenario_composite - current_composite
            delta_color = "#d50000" if delta > 0 else "#00c853"

            st.markdown(
                f'<div style="background:#141928;border:1px solid #1e2a42;'
                f'border-radius:4px;padding:14px 16px;margin-bottom:8px;">'
                f'<p class="mono-label">Current Score</p>'
                f'<p style="font-family:\'IBM Plex Mono\';font-size:36px;'
                f'font-weight:700;color:{cc};margin:0;">{current_composite:.0f}'
                f'<span style="font-size:14px;margin-left:8px;">{current_band.label}</span></p>'
                f'</div>'
                f'<div style="background:#141928;border:1px solid {sc}44;'
                f'border-left:4px solid {sc};border-radius:4px;padding:14px 16px;">'
                f'<p class="mono-label">Scenario Score</p>'
                f'<p style="font-family:\'IBM Plex Mono\';font-size:36px;'
                f'font-weight:700;color:{sc};margin:0;">{scenario_composite:.0f}'
                f'<span style="font-size:14px;margin-left:8px;">{scenario_band.label}</span></p>'
                f'<p style="font-family:\'IBM Plex Mono\';font-size:14px;'
                f'color:{delta_color};margin:4px 0 0 0;">{delta:+.1f} points</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if current_band.label != scenario_band.label:
                st.warning(f"⚡ Band change: {current_band.label} → {scenario_band.label}")
            else:
                st.success(f"Same band: {current_band.label}")

            if st.button("↺  Reset to current values", key="reset_scenario"):
                st.session_state.scenario_values = {}
                st.rerun()

        st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5 — MODEL DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab5:
        diag_tabs = st.tabs(["Backtest Results", "SHAP Summary", "Calibration", "Data Coverage", "Model Metadata"])

        # ── Backtest ───────────────────────────────────────────────────────────
        with diag_tabs[0]:
            st.markdown('<p class="mono-label">Walk-Forward Backtest Performance</p>', unsafe_allow_html=True)
            backtest_df = load_backtest(indicator_hash)
            if backtest_df is not None and not backtest_df.empty:
                # Colour code PR-AUC column
                def _highlight_pr(val):
                    if pd.isna(val): return ""
                    if val >= 0.40: return "color: #00c853"
                    return "color: #d50000"
                st.dataframe(
                    backtest_df.style.applymap(_highlight_pr, subset=["pr_auc"] if "pr_auc" in backtest_df.columns else []),
                    use_container_width=True,
                    height=300,
                )

                # Check thresholds
                if "pr_auc" in backtest_df.columns:
                    overall_pr = backtest_df[backtest_df.get("period", backtest_df.get("decade", backtest_df.columns[0])) == "ALL"]["pr_auc"].values
                    if len(overall_pr) > 0 and not pd.isna(overall_pr[0]) and float(overall_pr[0]) < 0.40:
                        st.error("⚠ MODEL PERFORMANCE BELOW THRESHOLD — PR-AUC < 0.40. Use with extreme caution.")
            else:
                st.info("No backtest results available. Train models first via: python ml_model.py")

        # ── SHAP Summary ───────────────────────────────────────────────────────
        with diag_tabs[1]:
            st.markdown('<p class="mono-label">SHAP Feature Importance</p>', unsafe_allow_html=True)
            if predictions and "top_drivers" in predictions:
                td = predictions["top_drivers"]
                for horizon_key in ["bear_12m", 12, "bear_6m", 6, "bear_3m", 3]:
                    drivers = td.get(horizon_key) if isinstance(td, dict) else None
                    if drivers:
                        label = str(horizon_key).replace("bear_", "").replace("m", "")
                        if label.isdigit():
                            label = f"{label}-Month Horizon"
                        st.plotly_chart(
                            make_shap_bar(drivers[:10], f"SHAP Drivers — {label}"),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
            else:
                st.info("SHAP data unavailable. Train ML models to enable explainability.")

        # ── Calibration ────────────────────────────────────────────────────────
        with diag_tabs[2]:
            st.markdown('<p class="mono-label">Model Calibration (Reliability Diagrams)</p>', unsafe_allow_html=True)
            models_loaded = load_ml_models()
            if models_loaded:
                for horizon_key, model in models_loaded.items():
                    ece = getattr(model, "ece", np.nan)
                    calibrated = getattr(model, "meta_calibrated", None)
                    label = f"{horizon_key}-Month Horizon"
                    col_c, col_e = st.columns([3, 1])
                    with col_c:
                        if calibrated is not None and hasattr(model, "calibration_y_true"):
                            try:
                                fig_cal = make_calibration_plot(
                                    model.calibration_y_true,
                                    model.calibration_y_prob,
                                    f"Calibration — {label}",
                                )
                                st.plotly_chart(fig_cal, use_container_width=True,
                                                config={"displayModeBar": False})
                            except Exception:
                                st.caption(f"Calibration plot unavailable for {label}")
                        else:
                            st.caption(f"No calibration data stored for {label}")
                    with col_e:
                        ece_color = "#00c853" if (not pd.isna(ece) and ece < 0.05) else "#d50000"
                        st.markdown(
                            f'<div style="padding:12px;background:#141928;border:1px solid #1e2a42;border-radius:4px;">'
                            f'<p class="mono-label">ECE</p>'
                            f'<p style="font-family:\'IBM Plex Mono\';font-size:24px;font-weight:700;'
                            f'color:{ece_color};margin:0;">'
                            f'{"N/A" if pd.isna(ece) else f"{ece:.4f}"}</p>'
                            f'<p style="font-family:\'IBM Plex Mono\';font-size:10px;color:#6b7fa8;">'
                            f'threshold &lt; 0.05</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.info("Models not loaded. Train models first.")

        # ── Data Coverage ──────────────────────────────────────────────────────
        with diag_tabs[3]:
            st.markdown('<p class="mono-label">Indicator Data Coverage</p>', unsafe_allow_html=True)
            coverage_rows = []
            for ind_id, meta in INDICATORS.items():
                if meta.name in indicators.columns:
                    series = indicators[meta.name].dropna()
                    start = series.index.min().strftime("%Y-%m") if not series.empty else "—"
                    end   = series.index.max().strftime("%Y-%m") if not series.empty else "—"
                    n_obs = len(series)
                    post90 = series.loc["1990":]
                    coverage_pct = f"{post90.notna().mean()*100:.0f}%"
                else:
                    start, end, n_obs, coverage_pct = "—", "—", 0, "0%"

                coverage_rows.append({
                    "#": ind_id,
                    "Indicator": meta.name,
                    "Category": meta.category,
                    "Frequency": meta.frequency,
                    "Lag (days)": meta.publication_lag_days,
                    "Start": start,
                    "End": end,
                    "N Obs": n_obs,
                    "Post-1990 Coverage": coverage_pct,
                })

            cov_df = pd.DataFrame(coverage_rows)
            st.dataframe(cov_df, use_container_width=True, height=450, hide_index=True)

        # ── Model Metadata ─────────────────────────────────────────────────────
        with diag_tabs[4]:
            st.markdown('<p class="mono-label">Model Metadata</p>', unsafe_allow_html=True)
            models_loaded = load_ml_models()
            if models_loaded:
                for horizon_key, model in models_loaded.items():
                    train_end = getattr(model, "train_end_date", "Unknown")
                    n_features = len(getattr(model, "feature_names", []))
                    pr_auc = getattr(model, "cv_pr_auc", np.nan)
                    brier  = getattr(model, "cv_brier",  np.nan)
                    ece    = getattr(model, "ece", np.nan)
                    calib_flag = "✓ PASS" if (not pd.isna(ece) and ece < 0.05) else "✗ FAIL"
                    pr_flag    = "✓ PASS" if (not pd.isna(pr_auc) and pr_auc >= 0.40) else "✗ FAIL"

                    st.markdown(
                        f'<div style="background:#141928;border:1px solid #1e2a42;'
                        f'border-radius:4px;padding:14px 16px;margin:6px 0;">'
                        f'<p style="font-family:\'IBM Plex Mono\';font-size:12px;'
                        f'font-weight:600;color:#4f8ef7;margin:0 0 8px 0;">'
                        f'HORIZON: {horizon_key} MONTHS</p>'
                        f'<table style="font-family:\'IBM Plex Mono\';font-size:11px;'
                        f'color:#a0b0d0;width:100%;">'
                        f'<tr><td style="color:#6b7fa8;">Trained through</td><td>{train_end}</td></tr>'
                        f'<tr><td style="color:#6b7fa8;">Feature count</td><td>{n_features}</td></tr>'
                        f'<tr><td style="color:#6b7fa8;">CV PR-AUC</td>'
                        f'<td>{"N/A" if pd.isna(pr_auc) else f"{pr_auc:.3f}"} {pr_flag}</td></tr>'
                        f'<tr><td style="color:#6b7fa8;">CV Brier Score</td>'
                        f'<td>{"N/A" if pd.isna(brier) else f"{brier:.3f}"}</td></tr>'
                        f'<tr><td style="color:#6b7fa8;">Calibration ECE</td>'
                        f'<td>{"N/A" if pd.isna(ece) else f"{ece:.4f}"} {calib_flag}</td></tr>'
                        f'</table></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No trained models found in .cache/models/. Run: python ml_model.py")

            # Framework version / data sources
            st.markdown('<p class="mono-label" style="margin-top:16px;">System Info</p>', unsafe_allow_html=True)
            sys_data = {
                "Dashboard version": "1.0.0",
                "Framework": "20-indicator composite + XGBoost/RF/LR ensemble",
                "Data sources": "FRED API, Shiller XLS, yfinance, AAII",
                "Bear definition": "≥20% peak-to-trough drawdown",
                "Training horizon": "1990–present",
                "Scope": "Structural/cyclical bears only (not exogenous shocks)",
                "Known gap": "Global contagion (BIS/ECB data not included)",
            }
            for k, v in sys_data.items():
                st.markdown(
                    f'<div style="display:flex;gap:16px;padding:4px 0;border-bottom:1px solid #1e2a42;">'
                    f'<span style="font-family:\'IBM Plex Mono\';font-size:11px;color:#6b7fa8;'
                    f'min-width:200px;">{k}</span>'
                    f'<span style="font-family:\'IBM Plex Mono\';font-size:11px;color:#c8d4e8;">{v}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__" or True:
    main()
