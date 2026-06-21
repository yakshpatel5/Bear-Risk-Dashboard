"""
bear_risk_api.py — StockMind Integration Bridge
================================================
Exposes the Bear Risk Dashboard as a clean, fast read-only API.

Import into StockMind (or any other app) with:

    from bear_risk_api import (
        get_current_risk_score,
        get_ml_probabilities,
        get_top_risk_drivers,
        get_macro_context_summary,
        get_indicator_status,
    )

Design contract
---------------
- Reads from local cache ONLY — never triggers a live API call.
- Returns None (never raises) if cache is missing or stale.
- Each function completes in <100ms on a warm cache.
- Thread-safe: uses read-only file access.
- No external dependencies beyond what requirements.txt already installs.

Intended usage cadence in StockMind
------------------------------------
Call get_current_risk_score() and get_macro_context_summary() on each
page load or analysis request.  Call get_ml_probabilities() only when
the ML panel is visible — it's slightly heavier.  Cache the result in
your app's session for the duration of the user session.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from config import (
    CACHE_DIR,
    INDICATORS,
    get_risk_band,
    ALERT_SCORE_ELEVATED,
    ALERT_SCORE_HIGH,
    ALERT_INDICATOR_RED_THRESHOLD,
)

log = logging.getLogger(__name__)

# ── Staleness limits for cache reads (more permissive than refresh triggers) ──
_MAX_STALE_DAYS = {
    "daily": 5,
    "weekly": 14,
    "monthly": 60,
    "quarterly": 120,
}


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL CACHE READERS
# ═══════════════════════════════════════════════════════════════════════════════

def _read_parquet(name: str) -> Optional[pd.DataFrame]:
    """
    Read a cached Parquet file by name stem.  Returns None if missing,
    unreadable, or older than the maximum allowed staleness for its frequency.
    Never raises.
    """
    path = CACHE_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as exc:
        log.warning("Cache read failed for '%s': %s", name, exc)
        return None


def _cache_age_days(name: str) -> Optional[float]:
    """Return age of a cache file in days, or None if file absent."""
    path = CACHE_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    mtime = datetime.utcfromtimestamp(path.stat().st_mtime)
    return (datetime.utcnow() - mtime).total_seconds() / 86400


def _load_all_indicators() -> Optional[pd.DataFrame]:
    """
    Reconstruct the indicator DataFrame from individual cached series.
    Returns a DataFrame with columns = indicator names, index = monthly dates.
    Returns None if fewer than 10 indicators are available.
    """
    series_map: dict[str, pd.Series] = {}

    for ind_id, meta in INDICATORS.items():
        # Try loading from each potential cached source
        loaded: Optional[pd.Series] = None

        # FRED series
        for sid in meta.fred_series:
            cache_name = f"fred_{sid}"
            df = _read_parquet(cache_name)
            if df is not None and not df.empty:
                loaded = df.iloc[:, 0].astype(float)
                break

        # yfinance tickers (fallback)
        if loaded is None:
            for ticker in meta.yf_tickers:
                safe = ticker.replace("^", "").replace("/", "_")
                df = _read_parquet(f"yf_{safe}")
                if df is not None and not df.empty:
                    loaded = df.iloc[:, 0].astype(float)
                    break

        # Shiller
        if loaded is None and meta.special_source == "shiller":
            df = _read_parquet("shiller_ie_data")
            if df is not None and meta.name in df.columns:
                loaded = df[meta.name].astype(float)

        # AAII
        if loaded is None and meta.special_source == "aaii":
            df = _read_parquet("aaii_sentiment")
            if df is not None and "BullBear" in df.columns:
                loaded = df["BullBear"].astype(float)

        if loaded is not None and not loaded.empty:
            # Resample to monthly
            series_map[meta.name] = loaded.resample("ME").last()

    if len(series_map) < 10:
        log.warning("Only %d indicator series available from cache.", len(series_map))
        return None

    combined = pd.DataFrame(series_map)
    combined.index.name = "Date"
    return combined.sort_index()


def _compute_indicator_score(
    raw_value: float,
    series_history: pd.Series,
    meta,
) -> float:
    """Compute a single indicator score [0–100] from raw value and history."""
    if pd.isna(raw_value) or series_history.dropna().empty:
        return np.nan

    method = meta.scoring_method
    direction = meta.direction

    if method == "A":
        window = meta.percentile_window or 360
        hist = series_history.dropna().iloc[-window:]
        p01, p99 = float(hist.quantile(0.01)), float(hist.quantile(0.99))
        val = float(np.clip(raw_value, p01, p99))
        rank = float((hist <= val).sum()) / len(hist)
        score = (1 - rank if direction == "lower_worse" else rank) * 100

    elif method == "B":
        if direction == "bidirectional":
            # AAII special formula
            x = raw_value
            if x < -20:
                score = float(np.clip(33 + 33 * (x + 20) / 20, 0, 33))
            elif x <= 30:
                score = 33 + 33 * (x + 20) / 50
            else:
                score = float(np.clip(66 + 34 * (x - 30) / 30, 66, 100))
        else:
            thresholds = meta.thresholds or [-5, 0, 2, 6]
            x_min, g, a, x_max = thresholds
            if direction == "higher_worse":
                if raw_value <= g:
                    score = 33 * max(0, raw_value - x_min) / max(g - x_min, 1e-9)
                elif raw_value <= a:
                    score = 33 + 34 * (raw_value - g) / max(a - g, 1e-9)
                else:
                    score = 66 + 34 * min(raw_value - a, x_max - a) / max(x_max - a, 1e-9)
            else:
                if raw_value >= g:
                    score = 33 - 33 * min(raw_value - g, x_max - g) / max(x_max - g, 1e-9)
                elif raw_value >= a:
                    score = 66 - 33 * (raw_value - a) / max(g - a, 1e-9)
                else:
                    score = 100 - 34 * max(raw_value - x_min, 0) / max(a - x_min, 1e-9)

    else:  # Method C — level + momentum
        thresholds = meta.thresholds
        if thresholds:
            x_min, g, a, x_max = thresholds
            if direction == "higher_worse":
                if raw_value <= g:
                    s_level = 33 * max(0, raw_value - x_min) / max(g - x_min, 1e-9)
                elif raw_value <= a:
                    s_level = 33 + 34 * (raw_value - g) / max(a - g, 1e-9)
                else:
                    s_level = 66 + 34 * min(raw_value - a, x_max - a) / max(x_max - a, 1e-9)
            else:
                if raw_value >= g:
                    s_level = 33 - 33 * min(raw_value - g, x_max - g) / max(x_max - g, 1e-9)
                elif raw_value >= a:
                    s_level = 66 - 33 * (raw_value - a) / max(g - a, 1e-9)
                else:
                    s_level = 100 - 34 * max(raw_value - x_min, 0) / max(a - x_min, 1e-9)
        else:
            s_level = 50.0

        if len(series_history.dropna()) >= 14:
            chg = series_history.diff(13)
            recent_chg = float(chg.iloc[-1]) if not pd.isna(chg.iloc[-1]) else 0
            mu, sigma = float(chg.mean()), float(chg.std())
            z = (recent_chg - mu) / sigma if sigma > 0.001 else 0
            s_mom = float(np.clip(50 - 10 * z if direction == "lower_worse" else 50 + 10 * z, 0, 100))
        else:
            s_mom = 50.0

        score = 0.5 * s_level + 0.5 * s_mom

    return float(np.clip(score, 0, 100))


def _compute_composite(indicators: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """
    Return (composite_score, {indicator_name: score}) for the latest row.
    """
    scores: dict[str, float] = {}
    weights: dict[str, float] = {}

    for ind_id, meta in INDICATORS.items():
        if meta.name not in indicators.columns:
            continue
        series = indicators[meta.name].dropna()
        if series.empty:
            continue
        raw = float(series.iloc[-1])
        score = _compute_indicator_score(raw, series, meta)
        if not np.isnan(score):
            scores[meta.name] = score
            weights[meta.name] = meta.weight_pct

    if not scores:
        return np.nan, {}

    total_weight = sum(weights[k] for k in scores)
    w_arr = np.array([weights[k] / total_weight for k in scores])
    s_arr = np.array([scores[k] for k in scores])

    c_linear = float((w_arr * s_arr).sum())
    w_std = float(np.sqrt((w_arr * (s_arr - c_linear) ** 2).sum()))
    amplifier = 1 + 0.15 * max(0, (w_std - 20) / 30)
    composite = float(min(100, c_linear * amplifier))

    return composite, scores


def _load_ml_prediction_cache() -> Optional[dict]:
    """
    Load the most recent ML prediction result from disk (written by ml_model.py).
    Returns None if no prediction cache exists or it is older than 24 hours.
    """
    pred_path = CACHE_DIR / "latest_prediction.json"
    if not pred_path.exists():
        return None
    age_hours = (datetime.utcnow() - datetime.utcfromtimestamp(pred_path.stat().st_mtime)).total_seconds() / 3600
    if age_hours > 24:
        log.warning("ML prediction cache is %.1f hours old — may be stale.", age_hours)
    try:
        with open(pred_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("ML prediction cache read failed: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_current_risk_score() -> Optional[dict]:
    """
    Return the current composite Bear Risk Score and band classification.

    Returns
    -------
    dict with keys:
        score         float   Composite score 0–100.
        band          str     Risk band label: LOW | GUARDED | ELEVATED | HIGH | SEVERE
        color         str     Hex colour for the band.
        action        str     One-sentence recommended action.
        last_updated  str     ISO-8601 date of the most recent data used.
        n_indicators  int     Number of indicators contributing to the score.

    Returns None if cache is unavailable.
    """
    indicators = _load_all_indicators()
    if indicators is None:
        log.warning("get_current_risk_score: no indicator data available.")
        return None

    composite, _ = _compute_composite(indicators)
    if np.isnan(composite):
        return None

    band = get_risk_band(composite)
    last_date = indicators.index[-1].strftime("%Y-%m-%d")

    return {
        "score": round(composite, 1),
        "band": band.label,
        "color": band.color,
        "action": band.action,
        "last_updated": last_date,
        "n_indicators": int(sum(1 for col in indicators.columns if indicators[col].dropna().size > 0)),
    }


def get_ml_probabilities() -> Optional[dict]:
    """
    Return ML-estimated probabilities of a bear market at each horizon.

    Reads from the prediction cache written by ml_model.py.
    Returns None if the ML models have not been trained or prediction
    cache does not exist.

    Returns
    -------
    dict with keys:
        bear_3m   float   P(≥20% drawdown within 3 months), 0–1.
        bear_6m   float   P(≥20% drawdown within 6 months), 0–1.
        bear_12m  float   P(≥20% drawdown within 12 months), 0–1.
        ci_lower  dict    {3: lo, 6: lo, 12: lo}  — 95% CI lower bound.
        ci_upper  dict    {3: hi, 6: hi, 12: hi}  — 95% CI upper bound.
        confidence  str   "HIGH" | "MEDIUM" | "LOW" based on ECE.
        as_of_date  str   ISO-8601 date of prediction.
    """
    cache = _load_ml_prediction_cache()
    if cache is None:
        return None

    def _safe(key: str, default=np.nan):
        return cache.get(key, default)

    p3  = _safe("bear_3m_prob")
    p6  = _safe("bear_6m_prob")
    p12 = _safe("bear_12m_prob")
    ci3  = _safe("bear_3m_ci",  (np.nan, np.nan))
    ci6  = _safe("bear_6m_ci",  (np.nan, np.nan))
    ci12 = _safe("bear_12m_ci", (np.nan, np.nan))

    # Derive confidence from ECE flag
    flags = cache.get("model_flags", {})
    all_pass = all("PASS" in str(v).upper() for v in flags.values()) if flags else False
    confidence = "HIGH" if all_pass else ("MEDIUM" if flags else "LOW")

    return {
        "bear_3m":   round(float(p3),  3) if not pd.isna(p3)  else None,
        "bear_6m":   round(float(p6),  3) if not pd.isna(p6)  else None,
        "bear_12m":  round(float(p12), 3) if not pd.isna(p12) else None,
        "ci_lower":  {3: ci3[0], 6: ci6[0], 12: ci12[0]},
        "ci_upper":  {3: ci3[1], 6: ci6[1], 12: ci12[1]},
        "confidence": confidence,
        "as_of_date": cache.get("as_of_date", "unknown"),
    }


def get_top_risk_drivers(n: int = 3) -> list[dict]:
    """
    Return the top N indicators by current risk score with SHAP context.

    Parameters
    ----------
    n : Number of drivers to return (default 3).

    Returns
    -------
    List of dicts, each containing:
        indicator     str    Indicator name.
        score         float  Current score 0–100.
        direction     str    "↑ worsening" | "↓ improving" | "→ stable"
        shap_value    float  SHAP value from ML model (None if not available).
        plain_english str    Human-readable one-sentence interpretation.
    """
    indicators = _load_all_indicators()
    if indicators is None:
        return []

    _, scores = _compute_composite(indicators)

    # Sort by score descending
    sorted_indicators = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

    # Pull SHAP values from ML cache if available
    shap_map: dict[str, float] = {}
    cache = _load_ml_prediction_cache()
    if cache and "top_drivers" in cache:
        td = cache["top_drivers"]
        for horizon_key in ["bear_12m", 12]:
            drivers_list = td.get(str(horizon_key)) or td.get(horizon_key, [])
            if drivers_list:
                for d in drivers_list:
                    shap_map[d.get("feature", "")] = d.get("shap_value", 0)
                break

    result = []
    for ind_name, score in sorted_indicators:
        # Find metadata
        meta = next((m for m in INDICATORS.values() if m.name == ind_name), None)
        if meta is None:
            continue

        # Compute trend arrow
        series = indicators[ind_name].dropna() if ind_name in indicators.columns else pd.Series(dtype=float)
        if len(series) >= 3:
            delta = float(series.iloc[-1]) - float(series.iloc[-3])
            if abs(delta) < 1e-6:
                direction = "→ stable"
            elif meta.direction == "lower_worse":
                direction = "↓ improving" if delta > 0 else "↑ worsening"
            elif meta.direction == "bidirectional":
                direction = "→ stable"
            else:
                direction = "↑ worsening" if delta > 0 else "↓ improving"
        else:
            direction = "→ stable"

        # Plain English interpretation
        band_str = "green" if score < 33 else ("amber" if score < 66 else "red")
        plain_english = _plain_english_for_indicator(ind_name, score, direction, series)

        result.append({
            "indicator": ind_name,
            "score": round(score, 1),
            "direction": direction,
            "shap_value": shap_map.get(ind_name),
            "plain_english": plain_english,
        })

    return result


def _plain_english_for_indicator(
    name: str, score: float, direction: str, series: pd.Series
) -> str:
    """Generate a brief plain-English sentence for a given indicator reading."""
    raw = float(series.iloc[-1]) if not series.empty else None
    zone = "elevated risk" if score >= 66 else ("moderate" if score >= 33 else "low risk")
    raw_str = f"{raw:.2f}" if raw is not None else "N/A"

    templates = {
        "Shiller CAPE Ratio":          f"CAPE at {raw_str} signals {zone} — valuations are historically {'stretched' if score >= 66 else 'moderate'}.",
        "Yield Curve (10Y-2Y)":        f"Yield curve at {raw_str} bps — {'inverted, historically a recession precursor' if (raw or 0) < 0 else 'positive, credit conditions normal'}.",
        "HY Credit Spread (OAS)":      f"HY spreads at {raw_str} bps — credit stress is {'elevated, funding conditions tightening' if score >= 66 else 'contained'}.",
        "Equity Risk Premium":         f"ERP at {raw_str}% — equities {'offer little premium over bonds, a bearish signal' if score >= 66 else 'retain a meaningful risk premium'}.",
        "Real Federal Funds Rate":     f"Real FFR at {raw_str}% — monetary policy is {'restrictive, historically a growth headwind' if score >= 66 else 'accommodative or neutral'}.",
        "Sahm Rule Indicator":         f"Sahm Rule at {raw_str} — labor market is {'deteriorating, consistent with recession onset' if score >= 66 else 'firm, no recession signal'}.",
        "VIX Level & Trend":           f"VIX at {raw_str} — options market pricing {'elevated tail risk' if score >= 66 else 'moderate uncertainty'}.",
        "Market Breadth (% > 200D MA)":f"Only {raw_str}% of S&P 500 stocks above 200-day MA — breadth is {'narrow, a classic topping signal' if score >= 66 else 'healthy, broad participation'}.",
    }

    if name in templates:
        return templates[name]
    return f"{name} reading ({raw_str}) is in {zone} territory, {direction.lower()}."


def get_macro_context_summary() -> Optional[str]:
    """
    Return a single plain-English paragraph summarising current macro conditions.

    Suitable for direct display in a StockMind stock analysis panel or
    any context where a brief macro overlay is needed.

    Returns None if insufficient data is available.
    """
    score_dict = get_current_risk_score()
    if score_dict is None:
        return None

    probs = get_ml_probabilities()
    drivers = get_top_risk_drivers(n=3)
    indicators = _load_all_indicators()

    score = score_dict["score"]
    band  = score_dict["band"]
    action = score_dict["action"]
    last_updated = score_dict["last_updated"]

    # Build probability sentence
    if probs:
        p12 = probs.get("bear_12m")
        p6  = probs.get("bear_6m")
        prob_sentence = (
            f"The ML model estimates a {p6*100:.0f}% probability of a ≥20% bear market "
            f"within 6 months and {p12*100:.0f}% within 12 months "
            f"(confidence: {probs['confidence']})."
            if p6 is not None and p12 is not None
            else ""
        )
    else:
        prob_sentence = ""

    # Build driver sentence
    if drivers:
        driver_parts = [
            f"{d['indicator']} ({d['direction'].split(' ')[0]})"
            for d in drivers[:3]
        ]
        driver_sentence = f"Primary risk drivers: {', '.join(driver_parts)}."
    else:
        driver_sentence = ""

    # Read a few key raw values for colour
    key_reads = []
    if indicators is not None:
        for ind_name, series_col in [
            ("Yield Curve (10Y-2Y)", "YieldCurve"),
            ("HY Credit Spread (OAS)", "HY_OAS"),
            ("VIX Level & Trend", "VIX"),
        ]:
            col = next((m.name for m in INDICATORS.values() if m.name == ind_name), None)
            if col and col in indicators.columns:
                s = indicators[col].dropna()
                if not s.empty:
                    key_reads.append(f"{ind_name.split(' ')[0]} {s.iloc[-1]:.1f}")

    conditions_str = " | ".join(key_reads) if key_reads else ""

    paragraph = (
        f"Bear Risk Dashboard — {last_updated}. "
        f"Composite macro risk score: {score:.0f}/100 ({band}). "
        f"{action} "
        f"{prob_sentence} "
        f"{driver_sentence}"
        f"{(' Key readings: ' + conditions_str + '.') if conditions_str else ''} "
        f"Note: model scope is structural/cyclical bears (2000, 2008, 2022 archetypes) "
        f"— exogenous shocks are not captured by design."
    ).strip()

    return paragraph


def get_indicator_status(indicator_name: str) -> Optional[dict]:
    """
    Return current status for a single indicator by name.

    Parameters
    ----------
    indicator_name : Exact indicator name from config.INDICATORS
                     (e.g. "Shiller CAPE Ratio", "HY Credit Spread (OAS)").

    Returns
    -------
    dict with keys:
        name            str    Indicator name.
        raw_value       float  Latest raw value in natural units.
        score           float  Normalised risk score 0–100.
        zone            str    "GREEN" | "AMBER" | "RED"
        direction       str    "↑ worsening" | "↓ improving" | "→ stable"
        last_updated    str    ISO date of latest reading.
        staleness_days  int    Days since last update.
        threshold_note  str    Brief description of red/amber/green thresholds.
        weight_pct      float  Baseline weight in composite (%).

    Returns None if indicator not found or data unavailable.
    """
    meta = next(
        (m for m in INDICATORS.values() if m.name.lower() == indicator_name.lower()),
        None,
    )
    if meta is None:
        log.warning("get_indicator_status: indicator '%s' not found.", indicator_name)
        return None

    indicators = _load_all_indicators()
    if indicators is None or meta.name not in indicators.columns:
        return None

    series = indicators[meta.name].dropna()
    if series.empty:
        return None

    raw = float(series.iloc[-1])
    last_date = series.index[-1]
    staleness = (datetime.utcnow().date() - last_date.date()).days

    score = _compute_indicator_score(raw, series, meta)
    zone = "RED" if score >= 66 else ("AMBER" if score >= 33 else "GREEN")

    if len(series) >= 3:
        delta = float(series.iloc[-1]) - float(series.iloc[-3])
        if abs(delta) < 1e-6:
            direction = "→ stable"
        elif meta.direction == "lower_worse":
            direction = "↓ improving" if delta > 0 else "↑ worsening"
        else:
            direction = "↑ worsening" if delta > 0 else "↓ improving"
    else:
        direction = "→ stable"

    # Build threshold note from config
    if meta.thresholds and len(meta.thresholds) == 4:
        x_min, g, a, x_max = meta.thresholds
        if meta.direction == "higher_worse":
            threshold_note = f"Green < {g}, Amber {g}–{a}, Red > {a}"
        else:
            threshold_note = f"Green > {g}, Amber {a}–{g}, Red < {a}"
    else:
        threshold_note = "Scored by historical percentile rank."

    return {
        "name": meta.name,
        "raw_value": round(raw, 4),
        "score": round(score, 1),
        "zone": zone,
        "direction": direction,
        "last_updated": last_date.strftime("%Y-%m-%d"),
        "staleness_days": staleness,
        "threshold_note": threshold_note,
        "weight_pct": meta.weight_pct,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK SMOKE TEST (run directly to verify integration)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pprint
    print("\n── get_current_risk_score() ──")
    pprint.pprint(get_current_risk_score())

    print("\n── get_ml_probabilities() ──")
    pprint.pprint(get_ml_probabilities())

    print("\n── get_top_risk_drivers(n=3) ──")
    pprint.pprint(get_top_risk_drivers(n=3))

    print("\n── get_macro_context_summary() ──")
    print(get_macro_context_summary())

    print("\n── get_indicator_status('Shiller CAPE Ratio') ──")
    pprint.pprint(get_indicator_status("Shiller CAPE Ratio"))
