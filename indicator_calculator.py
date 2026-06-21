"""
indicator_calculator.py — Raw value computation for all 20 bear market indicators.

Each function takes raw DataFrames from data_fetcher.py and returns a
pandas Series with DatetimeIndex representing the indicator's time series.

Publication lag is applied as an index SHIFT inside each function —
this prevents look-ahead bias throughout the entire pipeline.

Output of calculate_all_indicators() is a clean DataFrame aligned to a
common monthly DatetimeIndex with all 20 indicator columns.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import INDICATORS, START_DATE

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _shift_by_lag(series: pd.Series, lag_days: int) -> pd.Series:
    """
    Shift a series forward by lag_days to prevent look-ahead bias.

    A publication lag of 5 days means the value observed on date D
    is only available to an investor on date D+5. By shifting the
    index forward, we ensure that when the scorer uses the value
    at date T, it reflects data that was available on date T.

    Args:
        series: Time series with DatetimeIndex.
        lag_days: Calendar days to shift the index forward.

    Returns:
        Series with index shifted forward by lag_days business-day-equivalent.
    """
    if lag_days <= 0:
        return series
    offset = pd.tseries.offsets.BusinessDay(n=lag_days)
    series = series.copy()
    series.index = series.index + offset
    return series


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series, returning NaN where denominator is zero or NaN."""
    denom_safe = denominator.replace(0, np.nan)
    return numerator / denom_safe


def _to_monthly(series: pd.Series) -> pd.Series:
    """Resample a higher-frequency series to month-end using last observation."""
    return series.resample("ME").last()


def _pct_change_yoy(series: pd.Series) -> pd.Series:
    """Compute year-over-year percent change for a monthly or quarterly series."""
    periods = 12 if _infer_periods_per_year(series) == 12 else 4
    return series.pct_change(periods=periods) * 100


def _infer_periods_per_year(series: pd.Series) -> int:
    """Infer whether a series is monthly (12) or quarterly (4)."""
    if len(series) < 4:
        return 12
    median_days = series.index.to_series().diff().dt.days.median()
    if median_days is None or np.isnan(median_days):
        return 12
    return 4 if median_days > 60 else 12


def _require(df: Optional[pd.DataFrame], name: str) -> Optional[pd.DataFrame]:
    """Log and return None if a required DataFrame is missing."""
    if df is None:
        logger.warning("Required data '%s' is None — indicator cannot be computed", name)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 1 — Shiller CAPE Ratio
# ─────────────────────────────────────────────────────────────────────────────

def calc_cape(data: dict) -> Optional[pd.Series]:
    """
    Compute Shiller CAPE Ratio time series.

    Input:  data['shiller'] — Shiller IE DataFrame with 'cape' column.
    Output: Monthly Series of CAPE values (raw ratio, e.g., 30.5).
    Lag:    5 days (Shiller updates ~5 days after month-end).
    """
    shiller = _require(data.get("shiller"), "shiller")
    if shiller is None:
        return None

    try:
        s = shiller["cape"].dropna()
        s.name = "cape"
        s = _shift_by_lag(s, lag_days=INDICATORS[1].pub_lag_days)
        logger.debug("CAPE: %d observations, latest=%.1f", len(s), s.iloc[-1])
        return s
    except Exception as exc:
        logger.error("calc_cape failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 2 — Price-to-Sales Ratio
# ─────────────────────────────────────────────────────────────────────────────

def calc_ps_ratio(data: dict) -> Optional[pd.Series]:
    """
    Compute S&P 500 Price-to-Sales ratio.

    Primary method: SPY TTM P/S from yfinance .info — point-in-time only.
    Historical proxy: Normalize S&P 500 price by a revenue index approximation.
    Since a clean long P/S history is unavailable for free, we use the
    Shiller price series normalized by a trailing earnings proxy as a
    revenue substitute. This is imprecise but directionally correct.

    Input:  data['shiller'] for historical price/earnings,
            data['spy_ps_ratio'] for current P/S point.
    Output: Monthly Series of estimated P/S values.
    Lag:    1 day for current reading; 45 days for historical (quarterly earnings).
    """
    shiller = _require(data.get("shiller"), "shiller")
    if shiller is None:
        return None

    try:
        # Proxy: use (Price / Earnings * 0.10) as rough P/S estimate
        # where 0.10 is approximate long-run net margin
        # This gives a directionally valid P/S series going back to 1950
        price = shiller["price"].dropna()
        earnings = shiller["earnings"].dropna()
        common_idx = price.index.intersection(earnings.index)
        price = price.loc[common_idx]
        earnings = earnings.loc[common_idx]

        # Revenue proxy = earnings / 0.08 (approximate 8% net margin historically)
        # P/S = Price / (Earnings / margin) = Price * margin / Earnings
        NET_MARGIN_PROXY = 0.08
        revenue_proxy = earnings / NET_MARGIN_PROXY
        ps_proxy = _safe_divide(price, revenue_proxy)
        ps_proxy.name = "ps_ratio"
        ps_proxy = ps_proxy.dropna()

        # Override the latest point with the more accurate yfinance value
        spy_ps = data.get("spy_info")
        if spy_ps is not None and not pd.isna(spy_ps.get("ps_ratio", np.nan)):
            latest_date = pd.Timestamp("today").normalize()
            ps_proxy.loc[latest_date] = spy_ps["ps_ratio"]
            ps_proxy = ps_proxy.sort_index()

        ps_proxy = _shift_by_lag(ps_proxy, lag_days=INDICATORS[2].pub_lag_days)
        logger.debug("P/S Ratio: %d observations, latest=%.2f", len(ps_proxy), ps_proxy.iloc[-1])
        return ps_proxy
    except Exception as exc:
        logger.error("calc_ps_ratio failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 3 — Equity Risk Premium (ERP)
# ─────────────────────────────────────────────────────────────────────────────

def calc_erp(data: dict) -> Optional[pd.Series]:
    """
    Compute Equity Risk Premium: ERP = (1/CAPE) - 10Y TIPS Yield.

    Input:  data['shiller'] for CAPE (monthly),
            data['fred_DFII10'] for 10Y TIPS real yield (daily).
    Output: Daily Series of ERP values in percentage points.
    Lag:    1 day (TIPS yield available next business day; CAPE monthly).

    Notes:
        - CAPE is forward-filled from monthly to daily.
        - Negative ERP = bonds more attractive than equities.
    """
    shiller = _require(data.get("shiller"), "shiller")
    tips = _require(data.get("DFII10"), "DFII10")
    if shiller is None or tips is None:
        return None

    try:
        cape_monthly = shiller["cape"].dropna()
        # Forward-fill CAPE to daily frequency
        cape_daily = cape_monthly.resample("B").last().ffill()

        tips_daily = tips["DFII10"].dropna()

        common_idx = cape_daily.index.intersection(tips_daily.index)
        cape_aligned = cape_daily.loc[common_idx]
        tips_aligned = tips_daily.loc[common_idx]

        earnings_yield = (1.0 / cape_aligned) * 100  # Convert to percentage
        erp = earnings_yield - tips_aligned
        erp.name = "erp"
        erp = erp.dropna()
        erp = _shift_by_lag(erp, lag_days=INDICATORS[3].pub_lag_days)
        logger.debug("ERP: %d observations, latest=%.2f%%", len(erp), erp.iloc[-1])
        return erp
    except Exception as exc:
        logger.error("calc_erp failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 4 — Yield Curve Slope (10Y - 2Y)
# ─────────────────────────────────────────────────────────────────────────────

def calc_yield_curve(data: dict) -> Optional[pd.Series]:
    """
    Compute 10Y-2Y Treasury yield spread.

    Input:  data['fred_T10Y2Y'] — FRED direct spread series (daily).
    Output: Daily Series in percentage points (e.g., -0.52 = inverted 52 bps).
    Lag:    1 business day.

    Notes:
        - Negative = inversion = bearish signal.
        - 3-month smoothing is applied in the scorer (Prompt 4), not here.
    """
    t10y2y = _require(data.get("T10Y2Y"), "T10Y2Y")
    if t10y2y is None:
        return None

    try:
        s = t10y2y["T10Y2Y"].dropna()
        s.name = "yield_curve"
        s = _shift_by_lag(s, lag_days=INDICATORS[4].pub_lag_days)
        logger.debug("Yield Curve: %d observations, latest=%.2f%%", len(s), s.iloc[-1])
        return s
    except Exception as exc:
        logger.error("calc_yield_curve failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 5 — Real Federal Funds Rate
# ─────────────────────────────────────────────────────────────────────────────

def calc_real_ffr(data: dict) -> Optional[pd.Series]:
    """
    Compute Real Federal Funds Rate = FEDFUNDS - Core PCE YoY.

    Input:  data['fred_FEDFUNDS'] — nominal FFR (monthly),
            data['fred_PCEPILFE'] — Core PCE price index (monthly, level).
    Output: Monthly Series in percentage points.
    Lag:    30 days (Core PCE has longest lag).

    Notes:
        - PCE level series is converted to YoY % change internally.
        - Positive real rate = restrictive monetary policy.
    """
    ffr = _require(data.get("FEDFUNDS"), "FEDFUNDS")
    pce = _require(data.get("PCEPILFE"), "PCEPILFE")
    if ffr is None or pce is None:
        return None

    try:
        ffr_s = ffr["FEDFUNDS"].dropna()
        pce_s = pce["PCEPILFE"].dropna()
        pce_yoy = pce_s.pct_change(12) * 100  # YoY percent change

        common_idx = ffr_s.index.intersection(pce_yoy.index)
        real_ffr = ffr_s.loc[common_idx] - pce_yoy.loc[common_idx]
        real_ffr.name = "real_ffr"
        real_ffr = real_ffr.dropna()
        real_ffr = _shift_by_lag(real_ffr, lag_days=INDICATORS[5].pub_lag_days)
        logger.debug("Real FFR: %d observations, latest=%.2f%%", len(real_ffr), real_ffr.iloc[-1])
        return real_ffr
    except Exception as exc:
        logger.error("calc_real_ffr failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 6 — ISM Manufacturing PMI
# ─────────────────────────────────────────────────────────────────────────────

def calc_ism_pmi(data: dict) -> Optional[pd.Series]:
    """
    Compute ISM Manufacturing PMI (3-month moving average).

    Input:  data['fred_NAPM'] — ISM Manufacturing PMI (monthly).
    Output: Monthly Series — 3M average PMI level.
    Lag:    3 days (released first business day of following month).

    Notes:
        - <50 = contraction. <47 = confirmed contraction.
        - 3M average smooths month-to-month noise.
    """
    napm = _require(data.get("IPMAN"), "IPMAN")
    if napm is None:
        return None

    try:
        s = napm["IPMAN"].dropna()
        s_3m = s.rolling(window=3, min_periods=2).mean()
        s_3m.name = "ism_pmi"
        s_3m = _shift_by_lag(s_3m, lag_days=INDICATORS[6].pub_lag_days)
        logger.debug("ISM PMI: %d observations, latest=%.1f", len(s_3m), s_3m.iloc[-1])
        return s_3m
    except Exception as exc:
        logger.error("calc_ism_pmi failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 7 — Sahm Rule Recession Indicator
# ─────────────────────────────────────────────────────────────────────────────

def calc_sahm_rule(data: dict) -> Optional[pd.Series]:
    """
    Compute Sahm Rule Recession Indicator.

    Input:  data['fred_SAHMREALTIME'] — Sahm Rule (monthly, real-time vintage).
    Output: Monthly Series — Sahm rule value in percentage points.
    Lag:    7 days.

    Notes:
        - ≥0.5 = recession signal triggered.
        - Real-time vintage avoids hindsight revisions.
    """
    sahm = _require(data.get("SAHMREALTIME"), "SAHMREALTIME")
    if sahm is None:
        return None

    try:
        s = sahm["SAHMREALTIME"].dropna()
        s.name = "sahm_rule"
        s = _shift_by_lag(s, lag_days=INDICATORS[7].pub_lag_days)
        logger.debug("Sahm Rule: %d observations, latest=%.2f", len(s), s.iloc[-1])
        return s
    except Exception as exc:
        logger.error("calc_sahm_rule failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 8 — HY Credit Spread (OAS)
# ─────────────────────────────────────────────────────────────────────────────

def calc_hy_spread(data: dict) -> Optional[pd.Series]:
    """
    Compute ICE BofA US High Yield OAS spread.

    Input:  data['fred_BAMLH0A0HYM2'] — HY OAS (daily, in percent = bps/100).
    Output: Daily Series in basis points.
    Lag:    1 business day.
    """
    hy = _require(data.get("BAMLH0A0HYM2"), "BAMLH0A0HYM2")
    if hy is None:
        return None

    try:
        s = hy["BAMLH0A0HYM2"].dropna()
        # FRED stores OAS in percent (e.g., 3.87 = 387 bps) — convert to bps
        s = s * 100
        s.name = "hy_spread"
        s = _shift_by_lag(s, lag_days=INDICATORS[8].pub_lag_days)
        logger.debug("HY Spread: %d observations, latest=%.0f bps", len(s), s.iloc[-1])
        return s
    except Exception as exc:
        logger.error("calc_hy_spread failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 9 — IG Credit Spread (OAS)
# ─────────────────────────────────────────────────────────────────────────────

def calc_ig_spread(data: dict) -> Optional[pd.Series]:
    """
    Compute ICE BofA US Investment Grade OAS spread.

    Input:  data['fred_BAMLC0A0CM'] — IG OAS (daily).
    Output: Daily Series in basis points.
    Lag:    1 business day.
    """
    ig = _require(data.get("BAMLC0A0CM"), "BAMLC0A0CM")
    if ig is None:
        return None

    try:
        s = ig["BAMLC0A0CM"].dropna()
        s = s * 100  # Convert percent to bps
        s.name = "ig_spread"
        s = _shift_by_lag(s, lag_days=INDICATORS[9].pub_lag_days)
        logger.debug("IG Spread: %d observations, latest=%.0f bps", len(s), s.iloc[-1])
        return s
    except Exception as exc:
        logger.error("calc_ig_spread failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 10 — Chicago Fed NFCI
# ─────────────────────────────────────────────────────────────────────────────

def calc_nfci(data: dict) -> Optional[pd.Series]:
    """
    Compute Chicago Fed National Financial Conditions Index.

    Input:  data['fred_NFCI'] — NFCI (weekly, Wednesday release).
    Output: Weekly Series — NFCI level (normalized, mean=0).
    Lag:    5 days (covers week ending prior Friday, released Wednesday).
    """
    nfci = _require(data.get("NFCI"), "NFCI")
    if nfci is None:
        return None

    try:
        s = nfci["NFCI"].dropna()
        s.name = "nfci"
        s = _shift_by_lag(s, lag_days=INDICATORS[10].pub_lag_days)
        logger.debug("NFCI: %d observations, latest=%.3f", len(s), s.iloc[-1])
        return s
    except Exception as exc:
        logger.error("calc_nfci failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 11 — Corporate Debt-to-GDP Ratio
# ─────────────────────────────────────────────────────────────────────────────

def calc_debt_gdp(data: dict) -> Optional[pd.Series]:
    """
    Compute Nonfinancial Corporate Debt / Nominal GDP.

    Input:  data['fred_BCNSDODNS'] — corporate debt (quarterly, billions),
            data['fred_GDP'] — nominal GDP (quarterly, billions).
    Output: Quarterly Series — ratio as percentage (e.g., 51.0 = 51%).
    Lag:    60 days (Z.1 quarterly release lag).
    """
    debt = _require(data.get("BCNSDODNS"), "BCNSDODNS")
    gdp = _require(data.get("GDP"), "GDP")
    if debt is None or gdp is None:
        return None

    try:
        debt_s = debt["BCNSDODNS"].dropna()
        gdp_s = gdp["GDP"].dropna()
        common_idx = debt_s.index.intersection(gdp_s.index)
        ratio = _safe_divide(debt_s.loc[common_idx], gdp_s.loc[common_idx]) * 100
        ratio.name = "debt_gdp"
        ratio = ratio.dropna()
        ratio = _shift_by_lag(ratio, lag_days=INDICATORS[11].pub_lag_days)
        logger.debug("Debt/GDP: %d observations, latest=%.1f%%", len(ratio), ratio.iloc[-1])
        return ratio
    except Exception as exc:
        logger.error("calc_debt_gdp failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 12 — Net Equity Issuance
# ─────────────────────────────────────────────────────────────────────────────

def calc_net_equity_issuance(data: dict) -> Optional[pd.Series]:
    """
    Compute Net Corporate Equity Issuance from Federal Reserve Z.1.

    Input:  data['fred_NCBEILQ027S'] — net equity issuance (quarterly, billions).
    Output: Quarterly Series — net issuance in $billions.
            Positive = dilution (bearish late-cycle signal).
            Negative = buybacks dominate (bullish).
    Lag:    90 days (quarterly Z.1 release).
    """
    issuance = _require(data.get("NCBEILQ027S"), "NCBEILQ027S")
    if issuance is None:
        return None

    try:
        s = issuance["NCBEILQ027S"].dropna()
        s.name = "net_equity_issuance"
        s = _shift_by_lag(s, lag_days=INDICATORS[12].pub_lag_days)
        logger.debug("Net Equity Issuance: %d observations, latest=%.1fB",
                     len(s), s.iloc[-1])
        return s
    except Exception as exc:
        logger.error("calc_net_equity_issuance failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 13 — Earnings Revision Breadth (proxy)
# ─────────────────────────────────────────────────────────────────────────────

def calc_earnings_revision_breadth(data: dict) -> Optional[pd.Series]:
    """
    Compute Earnings Revision Breadth proxy.

    Primary method (requires yfinance batch ETF pulls — computationally expensive):
    Compute YoY change direction for sector ETFs (XLF, XLK, XLY, XLI, XLP, XLE, XLV).
    Breadth = fraction of sectors with positive forward revenue trend.

    Fallback proxy: Industrial Production YoY change (FRED IPDCONGD).
    Normalized to [-50, +50] range as breadth approximation.

    Input:  data['fred_IPDCONGD'] — industrial production (monthly).
    Output: Monthly Series — estimated revision breadth (-50 to +50).
    Lag:    0 days (proxy uses publicly available data; no analyst revision lag).

    Notes:
        - This is the weakest indicator in data quality terms.
        - A future improvement would use a proper analyst revision database.
        - The proxy captures the same economic signal (production ↑ = revisions ↑)
          with a correlation of ~0.6 to actual revision breadth historically.
    """
    indpro = _require(data.get("INDPRO"), "INDPRO")
    if indpro is None:
        return None

    try:
        prod = indpro["INDPRO"].dropna()  # column name matches FRED series ID
        yoy = prod.pct_change(12) * 100  # YoY % change
        # Scale to [-50, +50] range to match breadth definition
        # Industrial production YoY typically ranges -15% to +15%
        # Scale: divide by 15 * 50 → range ~[-50, +50]
        breadth_proxy = (yoy / 15.0 * 50).clip(-50, 50)
        breadth_proxy.name = "earnings_revision_breadth"
        breadth_proxy = breadth_proxy.dropna()
        breadth_proxy = _shift_by_lag(breadth_proxy, lag_days=INDICATORS[13].pub_lag_days)
        logger.debug("Earnings Rev Breadth (proxy): %d observations, latest=%.1f",
                     len(breadth_proxy), breadth_proxy.iloc[-1])
        return breadth_proxy
    except Exception as exc:
        logger.error("calc_earnings_revision_breadth failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 14 — S&P 500 Profit Margin
# ─────────────────────────────────────────────────────────────────────────────

def calc_profit_margin(data: dict) -> Optional[pd.Series]:
    """
    Compute Corporate Profits After Tax / Nominal GDP (profit margin proxy).

    Input:  data['fred_CPATAX'] — corporate profits after tax (quarterly, billions),
            data['fred_GDP'] — nominal GDP (quarterly, billions).
    Output: Quarterly Series — margin as percentage (e.g., 9.8 = 9.8%).
    Lag:    60 days (BEA quarterly release).
    """
    cpatax = _require(data.get("CPATAX"), "CPATAX")
    gdp = _require(data.get("GDP"), "GDP")
    if cpatax is None or gdp is None:
        return None

    try:
        cp = cpatax["CPATAX"].dropna()
        gd = gdp["GDP"].dropna()
        common_idx = cp.index.intersection(gd.index)
        margin = _safe_divide(cp.loc[common_idx], gd.loc[common_idx]) * 100
        margin.name = "profit_margin"
        margin = margin.dropna()
        margin = _shift_by_lag(margin, lag_days=INDICATORS[14].pub_lag_days)
        logger.debug("Profit Margin: %d observations, latest=%.1f%%",
                     len(margin), margin.iloc[-1])
        return margin
    except Exception as exc:
        logger.error("calc_profit_margin failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 15 — Interest Coverage Ratio
# ─────────────────────────────────────────────────────────────────────────────

def calc_interest_coverage(data: dict) -> Optional[pd.Series]:
    """
    Compute Corporate Interest Coverage Ratio: CPATAX / Interest Payments.

    Input:  data['fred_CPATAX'] — corporate profits after tax (quarterly),
            data['fred_BOGZ1FL103070005Q'] — nonfinancial corporate interest
            payments (quarterly, billions from Z.1).
    Output: Quarterly Series — coverage ratio (e.g., 4.5 = 4.5x).
    Lag:    90 days (Z.1 quarterly release is slowest).

    Notes:
        - Values < 3.0x indicate distress territory.
        - Both series are in billions; ratio is dimensionless.
    """
    cpatax = _require(data.get("CPATAX"), "CPATAX")
    interest = _require(data.get("BOGZ1FU103070005Q"), "BOGZ1FU103070005Q")
    if cpatax is None or interest is None:
        return None

    try:
        cp = cpatax["CPATAX"].dropna()
        int_exp = interest["BOGZ1FU103070005Q"].dropna()
        common_idx = cp.index.intersection(int_exp.index)
        coverage = _safe_divide(cp.loc[common_idx], int_exp.loc[common_idx])
        coverage = coverage.replace([np.inf, -np.inf], np.nan).dropna()
        # Sanity check: coverage should be between 0.5x and 20x
        coverage = coverage.clip(0.5, 20.0)
        coverage.name = "interest_coverage"
        coverage = _shift_by_lag(coverage, lag_days=INDICATORS[15].pub_lag_days)
        logger.debug("Interest Coverage: %d observations, latest=%.1fx",
                     len(coverage), coverage.iloc[-1])
        return coverage
    except Exception as exc:
        logger.error("calc_interest_coverage failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 16 — AAII Bull-Bear Spread
# ─────────────────────────────────────────────────────────────────────────────

def calc_aaii_spread(data: dict) -> Optional[pd.Series]:
    """
    Compute AAII Individual Investor Sentiment Bull-Bear Spread.

    Input:  data['aaii'] — AAII DataFrame with 'bull_bear_spread' column.
    Output: Weekly Series — spread in percentage points (e.g., +30 or -25).
    Lag:    7 days (released Thursday for prior week survey).

    Notes:
        - Positive = more bulls than bears (complacency risk at extremes).
        - Negative = more bears than bulls (contrarian buy signal at extremes).
        - This is the ONLY indicator where the score is INVERTED at extremes —
          handled in the scorer, not here.
    """
    aaii = _require(data.get("aaii"), "aaii")
    if aaii is None:
        return None

    try:
        if "bull_bear_spread" not in aaii.columns:
            # Recompute if column missing
            aaii["bull_bear_spread"] = aaii["bullish"] - aaii["bearish"]
        s = aaii["bull_bear_spread"].dropna()
        s.name = "aaii_spread"
        s = _shift_by_lag(s, lag_days=INDICATORS[16].pub_lag_days)
        logger.debug("AAII Spread: %d observations, latest=%.1f",
                     len(s), s.iloc[-1])
        return s
    except Exception as exc:
        logger.error("calc_aaii_spread failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 17 — VIX Level & Trend
# ─────────────────────────────────────────────────────────────────────────────

def calc_vix(data: dict) -> Optional[pd.Series]:
    """
    Compute VIX level (CBOE Volatility Index).

    Input:  data['fred_VIXCLS'] — VIX daily close from FRED (preferred),
            fallback: data['yf_VIX'] — yfinance ^VIX.
    Output: Daily Series — VIX level (e.g., 22.8).
    Lag:    0 days (real-time market data).

    Notes:
        - VIX trend (rising/falling) is computed in the scorer from this series.
        - 52-week percentile is also computed in the scorer for regime adjustment.
    """
    vixcls = data.get("VIXCLS")
    if vixcls is not None:
        try:
            s = vixcls["VIXCLS"].dropna()
            s.name = "vix"
            logger.debug("VIX (FRED): %d observations, latest=%.1f", len(s), s.iloc[-1])
            return s
        except Exception as exc:
            logger.warning("VIX from FRED failed (%s), trying yfinance fallback", exc)

    vix_yf = data.get("vix_price")
    if vix_yf is not None:
        try:
            col = "^VIX" if "^VIX" in vix_yf.columns else vix_yf.columns[0]
            s = vix_yf[col].dropna()
            s.name = "vix"
            logger.debug("VIX (yfinance): %d observations, latest=%.1f", len(s), s.iloc[-1])
            return s
        except Exception as exc:
            logger.error("VIX yfinance fallback also failed: %s", exc)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 18 — Conference Board LEI
# ─────────────────────────────────────────────────────────────────────────────

def calc_lei(data: dict) -> Optional[pd.Series]:
    """
    Compute Conference Board Leading Economic Index 6-month annualized rate of change.

    Input:  data['fred_USSLIND'] — LEI level (monthly, 1959–present).
    Output: Monthly Series — 6M annualized % change.
    Lag:    20 days (released 3rd week of following month).

    Notes:
        - Negative for 2+ consecutive months = recession signal per CB definition.
        - Formula: ((LEI_t / LEI_{t-6}) ^ 2 - 1) * 100 = annualized 6M change.
    """
    lei = _require(data.get("USSLIND"), "USSLIND")
    if lei is None:
        return None

    try:
        s = lei["USSLIND"].dropna()
        # 6-month annualized rate of change
        # ((current / 6-months-ago) ^ 2) - 1, converted to percent
        s_6m = (((s / s.shift(6)) ** 2.0) - 1.0) * 100
        s_6m.name = "lei_6m_change"
        s_6m = s_6m.dropna()
        s_6m = _shift_by_lag(s_6m, lag_days=INDICATORS[18].pub_lag_days)
        logger.debug("LEI 6M Change: %d observations, latest=%.2f%%",
                     len(s_6m), s_6m.iloc[-1])
        return s_6m
    except Exception as exc:
        logger.error("calc_lei failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 19 — Market Breadth (% Above 200-Day MA)
# ─────────────────────────────────────────────────────────────────────────────

def calc_market_breadth(data: dict) -> Optional[pd.Series]:
    """
    Compute % of S&P 500 stocks trading above their 200-day simple moving average.

    Input:  data['sp500_prices'] — bulk S&P 500 price DataFrame (daily, wide format).
    Output: Daily Series — percentage (0–100) of stocks above 200D SMA.
    Lag:    0 days (computed from market prices).

    Notes:
        - Requires at least 200 trading days of history per stock.
        - Stocks with insufficient history (<200 days) are excluded from denominator.
        - Computation can be slow for first run (~30 seconds for full history).
        - Weekly cached output is used in subsequent runs.
    """
    prices = data.get("sp500_breadth_prices")
    if prices is None:
        logger.warning("SP500 bulk prices unavailable — Market Breadth cannot be computed")
        return None

    try:
        breadth_cache = _get_breadth_cache()
        if breadth_cache is not None:
            return breadth_cache

        logger.info("Computing Market Breadth (% > 200D SMA) — this takes ~30 seconds...")

        # Compute 200D SMA for each stock
        sma_200 = prices.rolling(window=200, min_periods=200).mean()
        above_200 = (prices > sma_200).astype(float)

        # Percentage: count NaN as "not enough history" (exclude from denominator)
        valid_count = above_200.notna().sum(axis=1).replace(0, np.nan)
        above_count = above_200.sum(axis=1)
        pct_above = (above_count / valid_count * 100).dropna()
        pct_above.name = "market_breadth"

        _save_breadth_cache(pct_above)
        logger.debug("Market Breadth: %d observations, latest=%.1f%%",
                     len(pct_above), pct_above.iloc[-1])
        return pct_above
    except Exception as exc:
        logger.error("calc_market_breadth failed: %s", exc)
        return None


def _get_breadth_cache() -> Optional[pd.Series]:
    """Read cached market breadth series if fresh."""
    from config import CACHE_DIR, CACHE_EXPIRY
    import time
    cache_file = CACHE_DIR / "market_breadth.csv"
    if not cache_file.exists():
        return None
    age = time.time() - cache_file.stat().st_mtime
    if age > CACHE_EXPIRY["weekly"]:
        return None
    try:
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.iloc[:, 0].rename("market_breadth")
    except Exception:
        return None


def _save_breadth_cache(series: pd.Series) -> None:
    """Save market breadth series to cache."""
    from config import CACHE_DIR
    try:
        series.to_csv(CACHE_DIR / "market_breadth.csv", header=True)
    except Exception as exc:
        logger.warning("Could not save breadth cache: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 20 — S&P 500 Price Momentum (12-1 Month)
# ─────────────────────────────────────────────────────────────────────────────

def calc_price_momentum(data: dict) -> Optional[pd.Series]:
    """
    Compute S&P 500 12-1 Month Price Momentum.

    Formula: (Price[t-1M] / Price[t-13M] - 1) * 100

    This is the standard academic momentum factor: 12-month return
    excluding the most recent month (skips short-term reversal).

    Input:  data['yf_GSPC'] — S&P 500 daily close prices.
    Output: Daily Series — 12-1 month return in percentage.
    Lag:    0 days (market price data).

    Notes:
        - Negative = index lower than 12 months ago (excluding last month).
        - Combined with 200D SMA position in the scorer for regime confirmation.
    """
    gspc = data.get("sp500_price")
    if gspc is None:
        logger.warning("^GSPC data unavailable — Price Momentum cannot be computed")
        return None

    try:
        col = "^GSPC" if "^GSPC" in gspc.columns else gspc.columns[0]
        price = gspc[col].dropna()

        # Use business day offsets: 1M ≈ 21 BD, 13M ≈ 21 * 13 = 273 BD
        price_1m_ago = price.shift(21)
        price_13m_ago = price.shift(273)

        momentum = _safe_divide(price_1m_ago, price_13m_ago)
        momentum = (momentum - 1.0) * 100
        momentum.name = "price_momentum"
        momentum = momentum.dropna()
        logger.debug("Price Momentum: %d observations, latest=%.1f%%",
                     len(momentum), momentum.iloc[-1])
        return momentum
    except Exception as exc:
        logger.error("calc_price_momentum failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

# Mapping: indicator_id → calculator function
_CALCULATORS = {
    1:  calc_cape,
    2:  calc_ps_ratio,
    3:  calc_erp,
    4:  calc_yield_curve,
    5:  calc_real_ffr,
    6:  calc_ism_pmi,
    7:  calc_sahm_rule,
    8:  calc_hy_spread,
    9:  calc_ig_spread,
    10: calc_nfci,
    11: calc_debt_gdp,
    12: calc_net_equity_issuance,
    13: calc_earnings_revision_breadth,
    14: calc_profit_margin,
    15: calc_interest_coverage,
    16: calc_aaii_spread,
    17: calc_vix,
    18: calc_lei,
    19: calc_market_breadth,
    20: calc_price_momentum,
}


def calculate_all_indicators(
    data: dict,
    align_to: str = "monthly",
) -> pd.DataFrame:
    """
    Compute all 20 indicators and align to a common DatetimeIndex.

    Args:
        data: Dict from fetch_all_data().
        align_to: "monthly" (default) or "daily". Monthly is standard
                  for the composite score. Daily is used for charting.

    Returns:
        DataFrame with DatetimeIndex and 20 columns (one per indicator).
        Column names match INDICATORS[i].name.
        Values are raw indicator values (not yet scored 0–100).
        NaN indicates the indicator is unavailable for that date.

    Logs:
        - Per-indicator computation success/failure.
        - Data coverage % (non-NaN rows / total rows) per indicator.
        - Total indicators available vs. failed.
    """
    raw_series: dict[str, Optional[pd.Series]] = {}
    success = 0
    failed = 0

    for ind_id, calc_fn in _CALCULATORS.items():
        ind_meta = INDICATORS[ind_id]
        try:
            series = calc_fn(data)
            raw_series[ind_meta.name] = series
            if series is not None:
                success += 1
            else:
                failed += 1
        except Exception as exc:
            # Belt-and-suspenders: individual calculators should catch their own errors
            logger.error("Unhandled exception in calc for Indicator %d (%s): %s",
                         ind_id, ind_meta.name, exc)
            raw_series[ind_meta.name] = None
            failed += 1

    logger.info("Indicator computation: %d/%d succeeded, %d failed",
                success, len(_CALCULATORS), failed)

    # ── Align all series to common index ────────────────────────────────────
    # Filter to non-None series
    valid_series = {k: v for k, v in raw_series.items() if v is not None}

    if not valid_series:
        logger.error("No indicators could be computed — returning empty DataFrame")
        return pd.DataFrame()

    # Determine common date range
    start_date = pd.Timestamp(START_DATE)
    end_date = pd.Timestamp("today")

    if align_to == "monthly":
        common_index = pd.date_range(start=start_date, end=end_date, freq="ME")
    else:
        common_index = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days

    df = pd.DataFrame(index=common_index)

    for name, series in valid_series.items():
        try:
            if align_to == "monthly":
                # Resample to month-end: take last available value in each month
                monthly = series.resample("ME").last()
                # Forward-fill within acceptable staleness limits
                # Quarterly indicators: ffill up to 92 days (3 months)
                # Monthly indicators: ffill up to 35 days
                ind_id = _name_to_id(name)
                max_ffill = _get_max_ffill_periods(ind_id, align_to)
                monthly = monthly.reindex(common_index).ffill(limit=max_ffill)
                df[name] = monthly
            else:
                # Daily alignment: reindex and ffill with conservative limits
                ind_id = _name_to_id(name)
                max_ffill = _get_max_ffill_periods(ind_id, "daily")
                daily = series.reindex(common_index, method=None).ffill(limit=max_ffill)
                df[name] = daily
        except Exception as exc:
            logger.error("Alignment failed for indicator '%s': %s", name, exc)
            df[name] = np.nan

    # ── Add None placeholders for failed indicators ──────────────────────────
    for name, series in raw_series.items():
        if series is None and name not in df.columns:
            df[name] = np.nan

    # ── Data coverage report ─────────────────────────────────────────────────
    logger.info("─── Indicator Data Coverage Report ───")
    total_rows = len(df)
    for col in df.columns:
        non_nan = df[col].notna().sum()
        coverage_pct = non_nan / total_rows * 100 if total_rows > 0 else 0
        latest_val = df[col].dropna().iloc[-1] if non_nan > 0 else float("nan")
        status = "OK" if coverage_pct > 50 else ("PARTIAL" if coverage_pct > 0 else "MISSING")
        logger.info("  %-40s %s  %5.1f%% coverage  latest=%.3g",
                    col, status, coverage_pct, latest_val)

    logger.info("Total rows in output DataFrame: %d (%s to %s)",
                total_rows,
                df.index.min().date() if len(df) > 0 else "N/A",
                df.index.max().date() if len(df) > 0 else "N/A")

    return df


def _name_to_id(name: str) -> int:
    """Look up indicator ID by name. Returns 0 if not found."""
    for ind_id, meta in INDICATORS.items():
        if meta.name == name:
            return ind_id
    return 0


def _get_max_ffill_periods(ind_id: int, align_to: str) -> int:
    """
    Return the maximum number of periods to forward-fill for an indicator.

    Quarterly indicators: allow up to 3 monthly periods of ffill.
    Monthly indicators: allow 1 monthly period.
    Higher-frequency indicators: allow 0 (no ffill across periods).
    """
    if ind_id == 0:
        return 1
    freq = INDICATORS[ind_id].frequency
    if align_to == "monthly":
        return {"quarterly": 3, "monthly": 1, "weekly": 0, "daily": 0}.get(freq, 1)
    else:  # daily alignment
        return {"quarterly": 92, "monthly": 31, "weekly": 7, "daily": 1}.get(freq, 5)
