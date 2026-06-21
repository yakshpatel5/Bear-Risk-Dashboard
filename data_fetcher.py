"""
data_fetcher.py — Data acquisition layer for the AI Bear Risk Dashboard.

Provides five fetchers (FRED, Yahoo Finance, Shiller, AAII, S&P constituents)
and a master fetch_all_data() that aggregates them fault-tolerantly.

Caching strategy
----------------
Each fetcher writes to a Parquet file (or CSV for non-tabular text) in
CACHE_DIR.  On the next run:
  1. If the cache file is younger than CACHE_EXPIRY_DAYS[frequency], return
     cached data immediately (works offline).
  2. If stale or absent, attempt a live fetch.  On network failure, fall back
     to the stale cache with a logged warning rather than crashing.

Look-ahead bias
---------------
Raw data is returned with the natural publication date as the index.
Publication lag shifts are applied in indicator_calculator.py — NOT here.
"""
from __future__ import annotations

import io
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

import config
from config import (
    AAII_CACHE_FILE,
    AAII_CSV_URL,
    CACHE_DIR,
    CACHE_EXPIRY_DAYS,
    END_DATE,
    SHILLER_CACHE_FILE,
    SHILLER_DATA_URL,
    SP500_TICKERS_CACHE,
    SP500_CONSTITUENTS_URL,
    START_DATE,
    get_fred_api_key,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# Lazy import of fredapi so the module is importable even if the key is absent
# (unit testing without a live key is still possible for non-FRED fetchers).
try:
    from fredapi import Fred as _FredClient  # type: ignore
    _FRED_AVAILABLE = True
except ImportError:
    _FRED_AVAILABLE = False
    log.warning("fredapi not installed — FRED fetching disabled.")


# ── yfinance session initialisation ───────────────────────────────────────────
# yfinance 1.x requires a valid session/crumb from Yahoo Finance before
# downloading price data. Without it, Yahoo returns an empty response body
# causing the YFTzMissingError ("possibly delisted; No timezone found").
# We pre-warm the session once at import time so all subsequent downloads work.

def _init_yfinance_session() -> None:
    """
    Pre-warm the yfinance session by fetching the Yahoo Finance crumb.
    This prevents YFTzMissingError on first download calls.
    Called once at module import; failures are silently swallowed.
    """
    try:
        # yfinance 0.2.x exposes a _base.py session; newer versions use YfData
        # The cleanest cross-version approach is to make one small Ticker call
        # which forces cookie/crumb negotiation before any download() call.
        t = yf.Ticker("SPY")
        # Just accessing .history for 1 day is enough to prime the session
        _ = t.history(period="1d", auto_adjust=True)
        log.debug("yfinance session initialised successfully.")
    except Exception as e:
        log.debug("yfinance session pre-warm failed (non-fatal): %s", e)

_init_yfinance_session()


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_path(name: str) -> Path:
    """Return a Parquet cache path for a given series/ticker name."""
    safe_name = name.replace("^", "").replace("/", "_").replace(" ", "_")
    return CACHE_DIR / f"{safe_name}.parquet"


def _is_stale(path: Path, frequency: str) -> bool:
    """Return True if the cache file is older than the configured expiry."""
    if not path.exists():
        return True
    age_days = (datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)).days
    return age_days > CACHE_EXPIRY_DAYS.get(frequency, 1)


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    """Load a cached Parquet DataFrame; return None on any error."""
    try:
        df = pd.read_parquet(path)
        # Ensure timezone-naive UTC index
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df
    except Exception as exc:
        log.warning("Cache read failed for %s: %s", path, exc)
        return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame to Parquet; log but do not raise on failure."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    except Exception as exc:
        log.warning("Cache write failed for %s: %s", path, exc)


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the DataFrame index to a timezone-naive DatetimeIndex."""
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index()
    return df


def _fetch_with_retry(
    fetch_fn,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs,
):
    """
    Call fetch_fn(*args, **kwargs) with exponential back-off on failure.

    Parameters
    ----------
    fetch_fn    : Callable that performs the network request.
    max_retries : Maximum number of attempts.
    base_delay  : Initial wait in seconds; doubles each retry.

    Returns
    -------
    The return value of fetch_fn on success, or raises the last exception.
    """
    delay = base_delay
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                log.warning(
                    "Attempt %d/%d failed (%s). Retrying in %.1fs…",
                    attempt, max_retries, exc, delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                log.error("All %d attempts failed: %s", max_retries, exc)
    raise last_exc


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3a — FRED FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_fred_series(
    series_id: str,
    frequency: str = "daily",
    start: date = START_DATE,
    end: date = END_DATE,
) -> Optional[pd.DataFrame]:
    """
    Fetch a single FRED series, using a local Parquet cache.

    Parameters
    ----------
    series_id : FRED series identifier (e.g. "T10Y2Y").
    frequency : Data update frequency — controls cache expiry.
    start     : Earliest date to request.
    end       : Latest date to request (defaults to today).

    Returns
    -------
    DataFrame with a DatetimeIndex and one column named `series_id`,
    or None if fetching fails and no cache is available.

    Notes
    -----
    - Cache is checked first; live fetch occurs only when stale.
    - Rate limiting: exponential backoff with up to 3 retries.
    - Publication lag is NOT applied here; see indicator_calculator.py.
    """
    cache_file = _cache_path(f"fred_{series_id}")

    # ── Cache hit ──────────────────────────────────────────────────────────────
    if not _is_stale(cache_file, frequency):
        cached = _load_cache(cache_file)
        if cached is not None:
            log.debug("FRED %s: loaded from cache.", series_id)
            return cached

    # ── Live fetch ─────────────────────────────────────────────────────────────
    if not _FRED_AVAILABLE:
        log.error("FRED fetch skipped — fredapi not installed.")
        return _load_cache(cache_file)  # stale fallback

    try:
        fred_key = get_fred_api_key()
    except EnvironmentError as exc:
        log.error("%s", exc)
        return _load_cache(cache_file)

    def _do_fetch() -> pd.DataFrame:
        fred = _FredClient(fred_key)
        series = fred.get_series(
            series_id,
            observation_start=start.isoformat(),
            observation_end=end.isoformat(),
        )
        df = series.to_frame(name=series_id)
        return df

    try:
        df = _fetch_with_retry(_do_fetch, max_retries=3, base_delay=2.0)
        df = _normalize_index(df)
        _save_cache(df, cache_file)
        log.info("FRED %s: fetched %d rows (%s – %s).",
                 series_id, len(df), df.index.min().date(), df.index.max().date())
        return df
    except Exception as exc:
        log.error("FRED %s: live fetch failed — %s. Trying stale cache.", series_id, exc)
        stale = _load_cache(cache_file)
        if stale is not None:
            log.warning("FRED %s: using stale cache (may be outdated).", series_id)
        return stale


def fetch_multiple_fred_series(
    series_ids: list[str],
    frequency: str = "daily",
) -> dict[str, Optional[pd.DataFrame]]:
    """
    Fetch multiple FRED series, returning a dict keyed by series ID.

    Each series is fetched independently so a failure in one does not
    block the others.
    """
    return {sid: fetch_fred_series(sid, frequency=frequency) for sid in series_ids}


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3b — YAHOO FINANCE FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_yfinance(
    ticker: str,
    frequency: str = "daily",
    start: date = START_DATE,
    end: date = END_DATE,
    column: str = "Close",
) -> Optional[pd.DataFrame]:
    """
    Fetch historical price data from Yahoo Finance with local caching.

    Parameters
    ----------
    ticker    : Yahoo Finance ticker symbol (e.g. "^GSPC", "SPY", "^VIX").
    frequency : Controls cache expiry.
    start     : Earliest date to fetch.
    end       : Latest date to fetch.
    column    : Which OHLCV column to return. Defaults to "Close".
                Ticker.history(auto_adjust=True) returns "Close" (adjusted).
                Pass "all" to return the full OHLCV frame.

    Returns
    -------
    DataFrame with DatetimeIndex.  If `column` is not "all", the frame has
    a single column named `ticker`.  Returns None on total failure.

    Notes
    -----
    - Adjusted close prices are used to handle splits and dividends correctly.
    - The cache stores the full OHLCV frame; column selection happens on load.
    """
    cache_file = _cache_path(f"yf_{ticker}")

    def _return_column(df: pd.DataFrame) -> pd.DataFrame:
        if column == "all":
            return df
        # yfinance >= 0.2.38 returns MultiIndex (Price, Ticker) for single tickers
        # Flatten it: keep only column names (price types like Close, Open, etc.)
        if isinstance(df.columns, pd.MultiIndex):
            # Level 0 contains price types (Close, Open, High, Low, Volume)
            # Level 1 contains the ticker symbol
            level0 = df.columns.get_level_values(0).tolist()
            level1 = df.columns.get_level_values(1).tolist()
            price_types = {"Close", "Open", "High", "Low", "Volume", "Adj Close"}
            if set(level0) & price_types:
                # Format: (PriceType, Ticker) — drop ticker level
                df.columns = df.columns.get_level_values(0)
            else:
                # Format: (Ticker, PriceType) — drop ticker level
                df.columns = df.columns.get_level_values(1)
        if column in df.columns:
            return df[[column]].rename(columns={column: ticker})
        # Fallback: always try "Close" since auto_adjust=True renames Adj Close → Close
        if "Close" in df.columns:
            if column != "Close":
                log.warning("yfinance %s: '%s' not found; using 'Close'.", ticker, column)
            return df[["Close"]].rename(columns={"Close": ticker})
        log.error("yfinance %s: column '%s' not in %s.", ticker, column, df.columns.tolist())
        return df.iloc[:, :1]

    # ── Cache hit ──────────────────────────────────────────────────────────────
    if not _is_stale(cache_file, frequency):
        cached = _load_cache(cache_file)
        if cached is not None:
            log.debug("yfinance %s: loaded from cache.", ticker)
            return _return_column(cached)

    # ── Live fetch ─────────────────────────────────────────────────────────────
    def _do_fetch() -> pd.DataFrame:
        # Use Ticker.history() instead of yf.download() — it handles Yahoo Finance
        # cookie/crumb authentication more reliably and avoids YFTzMissingError.
        # yf.download() is a thin wrapper that has auth issues on some IPs in 2024-2026.
        t = yf.Ticker(ticker)
        df = t.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            raise_errors=False,
        )
        if df is None or df.empty:
            raise ValueError(f"yfinance returned empty DataFrame for {ticker}")
        # Rename columns to standard names if needed
        df = df.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low",
            "Close": "Close", "Volume": "Volume"
        })
        return df

    try:
        df = _fetch_with_retry(_do_fetch, max_retries=3, base_delay=2.0)
        df = _normalize_index(df)
        _save_cache(df, cache_file)
        log.info("yfinance %s: fetched %d rows (%s – %s).",
                 ticker, len(df), df.index.min().date(), df.index.max().date())
        return _return_column(df)
    except Exception as exc:
        log.error("yfinance %s: live fetch failed — %s. Trying stale cache.", ticker, exc)
        stale = _load_cache(cache_file)
        if stale is not None:
            log.warning("yfinance %s: using stale cache.", ticker)
            return _return_column(stale)
        return None


def fetch_yfinance_info(ticker: str) -> dict:
    """
    Fetch the `.info` dict from yfinance for fundamental ratios (e.g. P/S ratio).

    Returns an empty dict on failure (never raises).
    Retries up to 3 times with backoff to handle Yahoo Finance 429 rate limits.
    """
    for attempt in range(1, 4):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if info and isinstance(info, dict) and len(info) > 5:
                # Validate we got real data (not an empty throttle response)
                return info
            log.warning("yfinance info for %s: empty/minimal dict on attempt %d.", ticker, attempt)
        except Exception as exc:
            log.error("yfinance info fetch for %s failed (attempt %d): %s", ticker, attempt, exc)
        if attempt < 3:
            time.sleep(5 * attempt)  # 5s, 10s backoff
    log.error("yfinance info for %s: all attempts failed, returning empty dict.", ticker)
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3c — SHILLER DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_shiller_data() -> Optional[pd.DataFrame]:
    """
    Download and parse Robert Shiller's monthly S&P 500 data workbook.

    Source: http://www.econ.yale.edu/~shiller/data/ie_data.xls

    Returns
    -------
    DataFrame with a monthly DatetimeIndex and columns:
        Price      – S&P 500 nominal price index
        Dividend   – S&P 500 dividends (annual, divided by 12 for monthly)
        Earnings   – S&P 500 earnings (annual, divided by 12)
        CPI        – Consumer Price Index
        GS10       – 10-year government bond yield
        RealPrice  – CPI-adjusted price
        RealDividend
        RealEarnings
        CAPE       – Shiller Cyclically Adjusted P/E ratio

    Notes
    -----
    - The workbook has a header block in the first ~7 rows; data starts row 8.
    - The "Date" column is formatted as YYYY.MM (float like 1881.01).
    - Missing CAPE values for the first 10 years (insufficient earnings history)
      are left as NaN.
    - Cache expiry uses the "monthly" rule.
    """
    cache_file = _cache_path("shiller_ie_data")

    if not _is_stale(cache_file, "monthly"):
        cached = _load_cache(cache_file)
        if cached is not None:
            log.debug("Shiller data: loaded from cache.")
            return cached

    def _do_fetch() -> pd.DataFrame:
        response = requests.get(SHILLER_DATA_URL, timeout=30)
        response.raise_for_status()

        # Save raw XLS for local cache (also used as fallback)
        SHILLER_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SHILLER_CACHE_FILE.write_bytes(response.content)

        return _parse_shiller_xls(io.BytesIO(response.content))

    try:
        df = _fetch_with_retry(_do_fetch, max_retries=3, base_delay=5.0)
        _save_cache(df, cache_file)
        log.info("Shiller data: fetched %d rows (%s – %s).",
                 len(df), df.index.min().date(), df.index.max().date())
        return df
    except Exception as exc:
        log.error("Shiller live fetch failed: %s. Trying stale cache.", exc)
        # Try cached parquet first
        stale = _load_cache(cache_file)
        if stale is not None:
            log.warning("Shiller: using stale parquet cache.")
            return stale
        # Try raw XLS file
        if SHILLER_CACHE_FILE.exists():
            try:
                log.warning("Shiller: trying raw XLS fallback.")
                return _parse_shiller_xls(SHILLER_CACHE_FILE)
            except Exception as parse_exc:
                log.error("Shiller XLS fallback parse failed: %s", parse_exc)
        return None


def _parse_shiller_xls(source) -> pd.DataFrame:
    """
    Parse the Shiller ie_data.xls workbook from a file path or BytesIO.

    The workbook structure:
    - Sheet "Data"
    - Row 1–7: descriptive header (skipped)
    - Row 8: column names
    - Row 9+: data

    The Date column contains floats like 1881.01, 1881.1, 2000.1 meaning
    year=1881, month=January.  Month 10 appears as 1881.1 (not 1881.10).

    Shiller updated the workbook layout in 2024 — this parser handles both
    the old format (header row 7) and new format (header row varies) by
    scanning for the first row where column 0 looks like a 4-digit year.
    """
    # ── Try reading with multiple header rows ─────────────────────────────────
    raw = None
    last_error = None
    for engine in ("xlrd", "openpyxl"):
        for header_row in (7, 6, 8, 5):
            try:
                raw = pd.read_excel(
                    source,
                    sheet_name="Data",
                    header=header_row,
                    engine=engine,
                )
                # Validate: first column should be a date-like value (year ~1800–2100)
                first_valid = raw.iloc[:, 0].dropna().iloc[0] if not raw.empty else None
                if first_valid is None:
                    raw = None
                    continue
                # Accept if value looks like YYYY.MM (float between 1800 and 2200)
                try:
                    fv = float(first_valid)
                    if not (1800 <= fv <= 2200):
                        raw = None
                        continue
                except (ValueError, TypeError):
                    # Could be a string like "1881.01" — still acceptable
                    pass
                break   # good read
            except Exception as e:
                last_error = e
                raw = None
        if raw is not None:
            break

    if raw is None:
        raise ValueError(f"Could not parse Shiller XLS with any known format. Last error: {last_error}")

    # ── Drop fully-empty rows ─────────────────────────────────────────────────
    raw = raw.dropna(how="all").reset_index(drop=True)
    raw = raw.dropna(subset=[raw.columns[0]])

    # ── Map columns by name where possible, fall back to position ─────────────
    col_lower = {str(c).strip().lower(): c for c in raw.columns}

    def _find_col(candidates: list[str], position: int):
        for cname in candidates:
            if cname in col_lower:
                return col_lower[cname]
        if position < len(raw.columns):
            return raw.columns[position]
        return None

    date_col     = _find_col(["date", "year", "yr"], 0)
    price_col    = _find_col(["price", "s&p comp", "comp. price", "p"], 1)
    div_col      = _find_col(["dividend", "dividends", "d"], 2)
    earn_col     = _find_col(["earnings", "e"], 3)
    cpi_col      = _find_col(["cpi", "consumer price index"], 4)
    gs10_col     = _find_col(["gs10", "interest rate", "long interest rate"], 5)
    real_p_col   = _find_col(["real price", "price.1"], 6)
    real_div_col = _find_col(["real dividend", "dividend.1"], 7)

    # CAPE: search by name first, then try column 10 (new layout) or 8 (old)
    cape_col = None
    for c in raw.columns:
        cs = str(c).strip().upper()
        if any(kw in cs for kw in ("CAPE", "P/E10", "PE10", "CYCLICALLY")):
            cape_col = c
            break
    if cape_col is None:
        # Try positions 10, 8, 9 in order (Shiller has shifted this column)
        for pos in (10, 8, 9, 11):
            if pos < len(raw.columns):
                cape_col = raw.columns[pos]
                break

    # ── Build clean DataFrame ─────────────────────────────────────────────────
    col_map = {}
    if date_col:     col_map[date_col]     = "DateRaw"
    if price_col:    col_map[price_col]    = "Price"
    if div_col:      col_map[div_col]      = "Dividend"
    if earn_col:     col_map[earn_col]     = "Earnings"
    if cpi_col:      col_map[cpi_col]      = "CPI"
    if gs10_col:     col_map[gs10_col]     = "GS10"
    if real_p_col:   col_map[real_p_col]   = "RealPrice"
    if real_div_col: col_map[real_div_col] = "RealDividend"
    if cape_col:     col_map[cape_col]     = "CAPE"

    raw = raw.rename(columns=col_map)

    # ── Parse DateRaw — handles both float (1881.01) and string ("1881.01") ───
    def _parse_shiller_date(d) -> pd.Timestamp:
        try:
            d_float = float(str(d).strip())
        except (ValueError, TypeError):
            return pd.NaT
        year = int(d_float)
        if not (1800 <= year <= 2200):
            return pd.NaT
        month_frac = round((d_float - year) * 100)
        month = max(1, min(12, month_frac if month_frac > 0 else 1))
        try:
            return pd.Timestamp(year=year, month=month, day=1)
        except Exception:
            return pd.NaT

    if "DateRaw" not in raw.columns:
        raise ValueError("Could not identify Date column in Shiller workbook.")

    raw["Date"] = raw["DateRaw"].apply(_parse_shiller_date)
    raw = raw.dropna(subset=["Date"])
    raw = raw.set_index("Date").drop(columns=["DateRaw"], errors="ignore")

    # ── Coerce all columns to numeric ─────────────────────────────────────────
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # ── Filter to requested date range ────────────────────────────────────────
    raw = raw.loc[
        (raw.index >= pd.Timestamp(START_DATE))
        & (raw.index <= pd.Timestamp(END_DATE))
    ]

    return raw.sort_index()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3d — AAII SENTIMENT FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_aaii_xls(path) -> Optional[pd.DataFrame]:
    """
    Parse AAII sentiment data from their XLS file.

    The XLS structure has changed over versions but typically:
    - Sheet 0 or "Sentiment"
    - First few rows are headers/metadata
    - Data columns include Date, Bullish %, Neutral %, Bearish %
    """
    try:
        # Try reading with openpyxl (xlsx) first, then xlrd (xls)
        for engine in ["openpyxl", "xlrd", None]:
            try:
                kwargs = {"engine": engine} if engine else {}
                dfs = pd.read_excel(path, sheet_name=None, header=None, **kwargs)
                break
            except Exception:
                continue
        else:
            return None

        for sheet_name, raw in dfs.items():
            if raw is None or raw.empty:
                continue
            # Find the row containing the header (has "Date" and "Bullish")
            for i, row in raw.iterrows():
                row_str = " ".join(str(v).lower() for v in row.values if pd.notna(v))
                if "date" in row_str and "bull" in row_str:
                    # Use this row as header
                    df = raw.iloc[i + 1:].copy()
                    df.columns = raw.iloc[i].values
                    df = df.reset_index(drop=True)
                    return _normalize_aaii_table(df)
    except Exception as e:
        log.debug("_parse_aaii_xls failed: %s", e)
    return None


def _normalize_aaii_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Normalize a raw AAII table (from XLS or HTML) to the standard format.

    Returns DataFrame with columns: Bullish, Bearish, Neutral, BullBear
    and a DatetimeIndex. Returns None if normalization fails.
    """
    try:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        # Find and rename key columns
        rename_map = {}
        date_col = None
        for col in df.columns:
            lower = col.lower().replace("-", "").replace(" ", "").replace("%", "")
            if lower == "date" or lower == "weekendingdate":
                date_col = col
                rename_map[col] = "Date"
            elif "bull" in lower and "bear" not in lower and "spread" not in lower:
                rename_map[col] = "Bullish"
            elif "bear" in lower and "spread" not in lower:
                rename_map[col] = "Bearish"
            elif "neutral" in lower:
                rename_map[col] = "Neutral"

        if date_col is None:
            return None

        df = df.rename(columns=rename_map)

        required = ["Date", "Bullish", "Bearish"]
        if not all(c in df.columns for c in required):
            return None

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        if df.empty:
            return None

        df = df.set_index("Date").sort_index()

        for col in ["Bullish", "Bearish", "Neutral"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Normalize to 0–1 if in percentage form (e.g. 38.23)
                valid = df[col].dropna()
                if not valid.empty and valid.abs().max() > 1.5:
                    df[col] = df[col] / 100.0

        if "Neutral" not in df.columns:
            df["Neutral"] = 1.0 - df["Bullish"].fillna(0) - df["Bearish"].fillna(0)

        df["BullBear"] = (df["Bullish"] - df["Bearish"]) * 100.0
        result = df[["Bullish", "Bearish", "Neutral", "BullBear"]].dropna(how="all")
        return result if not result.empty else None
    except Exception as e:
        log.debug("_normalize_aaii_table failed: %s", e)
        return None


def fetch_aaii_sentiment() -> Optional[pd.DataFrame]:
    """
    Fetch AAII Individual Investor Sentiment Survey data.

    Primary source: https://www.aaii.com/sentimentsurvey/sent_results
    The page returns a CSV download when accessed directly.

    Returns
    -------
    DataFrame with a weekly DatetimeIndex and columns:
        Bullish   – Fraction of bulls (0–1)
        Neutral   – Fraction neutral (0–1)
        Bearish   – Fraction bears (0–1)
        BullBear  – Bull % minus Bear % (percentage points, e.g. 15.3)

    Notes
    -----
    - AAII does not provide a formal API. The URL above returns a CSV for
      direct download as of 2024. If AAII changes the format, a fallback
      cached CSV (aaii_sentiment.csv in CACHE_DIR) is used.
    - Publication lag: survey closes Wednesday, released Thursday.
      Raw dates in the CSV represent the survey week-ending date.
    - If live fetch fails and no cache exists, returns None; indicator 16
      will be excluded from the composite with weight redistributed.
    """
    cache_file = _cache_path("aaii_sentiment")

    if not _is_stale(cache_file, "weekly"):
        cached = _load_cache(cache_file)
        if cached is not None:
            log.debug("AAII: loaded from cache.")
            return cached

    def _do_fetch() -> pd.DataFrame:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.aaii.com/",
        }

        # AAII provides data via their sentiment page which requires session cookies.
        # Strategy 1: Try the XLS file (most reliable direct download)
        # Strategy 2: Try the CSV URL with session cookies
        # Strategy 3: Use pandas read_html to scrape the table from the page

        xls_url = "https://www.aaii.com/files/surveys/sentiment.xls"
        csv_url  = AAII_CSV_URL  # https://www.aaii.com/sentimentsurvey/sent_results

        last_exc = None

        # ── Attempt 1: XLS direct download ───────────────────────────────────
        try:
            resp_xls = requests.get(xls_url, headers=headers, timeout=30)
            if resp_xls.status_code == 200 and len(resp_xls.content) > 5000:
                # Save raw XLS
                xls_path = CACHE_DIR / "aaii_raw.xls"
                xls_path.write_bytes(resp_xls.content)
                df = _parse_aaii_xls(xls_path)
                if df is not None and not df.empty:
                    log.info("AAII: successfully fetched via XLS download.")
                    AAII_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                    return df
        except Exception as e:
            last_exc = e
            log.debug("AAII XLS attempt failed: %s", e)

        # ── Attempt 2: CSV URL with session ──────────────────────────────────
        try:
            session = requests.Session()
            # Prime session with a GET to the main page first
            session.get("https://www.aaii.com/", headers=headers, timeout=15)
            resp = session.get(csv_url, headers=headers, timeout=30)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            # Reject if we got HTML back instead of CSV
            if "html" in content_type or resp.text.strip().startswith("<!DOCTYPE"):
                raise ValueError("AAII URL returned HTML page instead of CSV data")
            AAII_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            AAII_CACHE_FILE.write_text(resp.text, encoding="utf-8")
            df = _parse_aaii_csv(io.StringIO(resp.text))
            if df is not None and not df.empty:
                log.info("AAII: successfully fetched via CSV URL.")
                return df
        except Exception as e:
            last_exc = e
            log.debug("AAII CSV URL attempt failed: %s", e)

        # ── Attempt 3: pandas read_html scrape ───────────────────────────────
        try:
            tables = pd.read_html("https://www.aaii.com/sentimentsurvey/sent_results",
                                   storage_options={"User-Agent": headers["User-Agent"]})
            for tbl in tables:
                cols = [str(c).lower() for c in tbl.columns]
                if any("bull" in c for c in cols) and any("bear" in c for c in cols):
                    tbl.columns = [str(c).strip() for c in tbl.columns]
                    df = _normalize_aaii_table(tbl)
                    if df is not None and not df.empty:
                        log.info("AAII: successfully fetched via HTML table scrape.")
                        return df
        except Exception as e:
            last_exc = e
            log.debug("AAII HTML scrape attempt failed: %s", e)

        raise ValueError(f"All AAII fetch strategies failed. Last error: {last_exc}")

    try:
        df = _fetch_with_retry(_do_fetch, max_retries=3, base_delay=5.0)
        _save_cache(df, cache_file)
        log.info("AAII: fetched %d rows (%s – %s).",
                 len(df), df.index.min().date(), df.index.max().date())
        return df
    except Exception as exc:
        log.error("AAII live fetch failed: %s. Trying stale cache.", exc)
        stale = _load_cache(cache_file)
        if stale is not None:
            log.warning("AAII: using stale parquet cache.")
            return stale
        # Last resort: raw CSV fallback
        if AAII_CACHE_FILE.exists():
            try:
                log.warning("AAII: parsing raw CSV fallback.")
                with open(AAII_CACHE_FILE, encoding="utf-8") as f:
                    return _parse_aaii_csv(f)
            except Exception as parse_exc:
                log.error("AAII CSV fallback parse failed: %s", parse_exc)
        log.error("AAII: no data available — indicator 16 will be excluded.")
        return None


def _parse_aaii_csv(source) -> pd.DataFrame:
    """
    Parse AAII sentiment CSV.

    AAII's CSV format has changed over time and uses a multi-line preamble.
    Current format (2024+):
        Line 1: "AAII Investor Sentiment Survey"
        Line 2: blank
        Line 3: "Date,Bullish,Neutral,Bearish,Total,Bull-Bear Spread,..."
        Line 4+: data rows  e.g.  "6/5/2026,0.3823,0.2941,0.3235,..."

    This parser scans all lines to find the header row (contains "Date")
    then reads data from that point forward, regardless of preamble length.
    """
    # ── Read raw text and find the header line ────────────────────────────────
    if hasattr(source, "read"):
        text = source.read()
    else:
        text = str(source)

    lines = text.splitlines()
    header_line_idx = None
    for i, line in enumerate(lines):
        # Header row contains "Date" and at least one of the sentiment columns
        lower = line.lower()
        if "date" in lower and ("bull" in lower or "bear" in lower):
            header_line_idx = i
            break

    if header_line_idx is None:
        # Last-resort fallback: try treating first non-blank line as header
        for i, line in enumerate(lines):
            if line.strip():
                header_line_idx = i
                break

    if header_line_idx is None:
        raise ValueError("AAII CSV: could not find header row containing 'Date'")

    # ── Parse from header line onward ─────────────────────────────────────────
    data_text = "\n".join(lines[header_line_idx:])
    try:
        raw = pd.read_csv(
            io.StringIO(data_text),
            skipinitialspace=True,
            on_bad_lines="skip",   # skip malformed rows silently
        )
    except TypeError:
        # on_bad_lines added in pandas 1.3 — fallback for older versions
        raw = pd.read_csv(
            io.StringIO(data_text),
            skipinitialspace=True,
            error_bad_lines=False,
            warn_bad_lines=False,
        )

    if raw.empty:
        raise ValueError("AAII CSV: no data rows found after header")

    # ── Normalize column names ────────────────────────────────────────────────
    raw.columns = [str(c).strip() for c in raw.columns]
    rename_map: dict[str, str] = {}
    for col in raw.columns:
        lower = col.lower().replace("-", "").replace(" ", "")
        if "bull" in lower and "bear" not in lower and "spread" not in lower:
            rename_map[col] = "Bullish"
        elif "bear" in lower and "spread" not in lower:
            rename_map[col] = "Bearish"
        elif "neutral" in lower:
            rename_map[col] = "Neutral"
        elif lower == "date":
            rename_map[col] = "Date"
    raw = raw.rename(columns=rename_map)

    required = ["Date", "Bullish", "Bearish"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            f"AAII CSV missing required columns: {missing}. "
            f"Available: {raw.columns.tolist()}. "
            f"Header line was: {lines[header_line_idx]!r}"
        )

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"])
    raw = raw.set_index("Date")

    for col in ["Bullish", "Bearish", "Neutral"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
            # Normalize to 0–1 if expressed as whole percentages (e.g. 38.23)
            if raw[col].dropna().empty:
                continue
            if raw[col].dropna().abs().max() > 1.5:
                raw[col] = raw[col] / 100.0

    raw["BullBear"] = (raw["Bullish"] - raw["Bearish"]) * 100.0  # percentage points
    raw = raw.sort_index()

    result = raw[["Bullish", "Bearish", "Neutral", "BullBear"]].dropna(how="all")
    if result.empty:
        raise ValueError("AAII CSV: parsed successfully but all data rows are NaN")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# S&P 500 CONSTITUENT FETCHER (for Indicator 19 — Market Breadth)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_sp500_tickers() -> list[str]:
    """
    Fetch the current list of S&P 500 constituent tickers from Wikipedia.

    Returns
    -------
    List of ticker symbols (e.g. ["AAPL", "MSFT", ...]).
    Falls back to cached list if Wikipedia is unavailable.

    Notes
    -----
    - The constituent list changes quarterly.  Cache expiry is 30 days.
    - Some Wikipedia tickers differ from Yahoo Finance (e.g. BRK.B vs BRK-B).
      The function normalizes dots to hyphens for yfinance compatibility.
    """
    if SP500_TICKERS_CACHE.exists():
        age_days = (datetime.utcnow() - datetime.utcfromtimestamp(
            SP500_TICKERS_CACHE.stat().st_mtime)).days
        if age_days <= CACHE_EXPIRY_DAYS["monthly"]:
            tickers = SP500_TICKERS_CACHE.read_text().splitlines()
            if tickers:
                log.debug("S&P 500 tickers: loaded %d from cache.", len(tickers))
                return tickers

    try:
        tables = pd.read_html(SP500_CONSTITUENTS_URL)
        # The first table on the page is the constituent list
        df = tables[0]
        # Column is usually "Symbol" or "Ticker"
        ticker_col = next(
            (c for c in df.columns if "symbol" in str(c).lower() or "ticker" in str(c).lower()),
            df.columns[0],
        )
        tickers = df[ticker_col].astype(str).str.strip().tolist()
        # Normalize BRK.B → BRK-B for yfinance
        tickers = [t.replace(".", "-") for t in tickers]
        SP500_TICKERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SP500_TICKERS_CACHE.write_text("\n".join(tickers))
        log.info("S&P 500 tickers: fetched %d constituents from Wikipedia.", len(tickers))
        return tickers
    except Exception as exc:
        log.error("S&P 500 ticker fetch failed: %s. Using stale cache if available.", exc)
        if SP500_TICKERS_CACHE.exists():
            tickers = SP500_TICKERS_CACHE.read_text().splitlines()
            log.warning("S&P 500 tickers: using stale cache (%d tickers).", len(tickers))
            return tickers
        # Hard fallback: a representative subset of large-caps
        log.error("S&P 500 tickers: no cache — using minimal fallback list (breadth calc impaired).")
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM",
                "JNJ", "UNH", "XOM", "V", "MA", "PG", "HD", "CVX", "MRK", "ABBV"]


def fetch_sp500_prices_for_breadth(
    tickers: list[str],
    start: date = START_DATE,
) -> Optional[pd.DataFrame]:
    """
    Batch-fetch adjusted close prices for S&P 500 constituents.

    Fetches all tickers in a single yfinance call to minimize API round-trips.
    Results are cached as a single wide Parquet file.

    Parameters
    ----------
    tickers : List of ticker symbols (from fetch_sp500_tickers()).
    start   : Earliest date needed (typically 300 trading days before today
              to compute 200-day MAs with history).

    Returns
    -------
    Wide DataFrame: DatetimeIndex × tickers, values = Adj Close prices.
    Returns None if fetch fails and no cache exists.
    """
    cache_file = _cache_path("sp500_breadth_prices")

    if not _is_stale(cache_file, "daily"):
        cached = _load_cache(cache_file)
        if cached is not None:
            log.debug("SP500 breadth prices: loaded from cache (%d tickers).", cached.shape[1])
            return cached

    def _do_fetch() -> pd.DataFrame:
        # yfinance batch download (show_errors param removed in >= 0.2.38 — do not add it)
        df = yf.download(
            tickers,
            start=start.isoformat(),
            end=(END_DATE + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        if df is None or df.empty:
            raise ValueError("yfinance batch download returned empty DataFrame")
        # Extract Close prices — column structure varies by yfinance version
        # yfinance >= 0.2.38: MultiIndex is (PriceType, Ticker)
        # yfinance < 0.2.38:  MultiIndex is (Ticker, PriceType)
        if isinstance(df.columns, pd.MultiIndex):
            level0 = df.columns.get_level_values(0).unique().tolist()
            level1 = df.columns.get_level_values(1).unique().tolist()
            price_types = {"Close", "Open", "High", "Low", "Volume", "Adj Close"}

            if set(level0) & price_types:
                # Format: (PriceType, Ticker) — new yfinance
                if "Close" in level0:
                    close_df = df["Close"]
                elif "Adj Close" in level0:
                    close_df = df["Adj Close"]
                else:
                    close_df = df[level0[0]]  # take whatever first price column there is
            else:
                # Format: (Ticker, PriceType) — old yfinance
                if "Close" in level1:
                    close_df = df.xs("Close", axis=1, level=1)
                elif "Adj Close" in level1:
                    close_df = df.xs("Adj Close", axis=1, level=1)
                else:
                    close_df = df.xs(level1[0], axis=1, level=1)
        elif "Close" in df.columns:
            close_df = df[["Close"]]
        elif "Adj Close" in df.columns:
            close_df = df[["Adj Close"]]
        else:
            close_df = df.iloc[:, :1]  # just take first column

        # Drop any all-NaN ticker columns (failed downloads)
        close_df = close_df.dropna(axis=1, how="all")
        return close_df

    try:
        df = _fetch_with_retry(_do_fetch, max_retries=2, base_delay=5.0)
        df = _normalize_index(df)
        _save_cache(df, cache_file)
        log.info("SP500 breadth prices: fetched %d tickers × %d rows.", df.shape[1], df.shape[0])
        return df
    except Exception as exc:
        log.error("SP500 breadth prices fetch failed: %s. Trying stale cache.", exc)
        stale = _load_cache(cache_file)
        if stale is not None:
            log.warning("SP500 breadth prices: using stale cache.")
        return stale


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3e — MASTER FETCH_ALL_DATA
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_all_data() -> dict[str, Optional[pd.DataFrame | dict | list]]:
    """
    Orchestrate all data fetches and return a unified dictionary.

    Returns
    -------
    Dict with the following keys (values are None if that fetch failed):

    FRED series (keyed by FRED ID):
        "T10Y2Y", "FEDFUNDS", "PCEPILFE", "IPMAN", "SAHMREALTIME",
        "BAMLH0A0HYM2", "BAMLC0A0CM", "NFCI", "BCNSDODNS", "GDP",
        "NCBEILQ027S", "CPATAX", "BOGZ1FA106130001Q", "USSLIND",
        "VIXCLS", "DFII10"

    Computed / external:
        "shiller"           – Shiller monthly DataFrame
        "aaii"              – AAII weekly DataFrame
        "spy_info"          – yfinance SPY info dict (for P/S ratio)
        "sp500_price"       – ^GSPC daily prices (for momentum/breadth)
        "vix_price"         – ^VIX daily prices
        "sp500_breadth_prices" – Wide DataFrame of constituent prices
        "sp500_tickers"     – List of current S&P 500 ticker strings

    Logs a summary table of loaded vs. failed sources.
    Never raises — individual failures are logged and returned as None.
    """
    results: dict[str, object] = {}
    successes: list[str] = []
    failures:  list[str] = []

    def _record(key: str, fetcher, *args, **kwargs):
        """Run a fetcher and record success/failure."""
        try:
            val = fetcher(*args, **kwargs)
            results[key] = val
            if val is not None and (not isinstance(val, (pd.DataFrame, list)) or len(val) > 0):
                successes.append(key)
            else:
                failures.append(f"{key} (empty)")
        except Exception as exc:
            log.error("fetch_all_data: unexpected error for '%s': %s", key, exc)
            results[key] = None
            failures.append(key)

    # ── FRED series ────────────────────────────────────────────────────────────
    fred_daily = [
        "T10Y2Y", "BAMLH0A0HYM2", "BAMLC0A0CM", "VIXCLS", "DFII10",
    ]
    fred_weekly   = ["NFCI"]
    fred_monthly  = [
        "FEDFUNDS", "PCEPILFE",
        "IPMAN",          # Industrial Production: Manufacturing (NAICS) — FRED series
                          # available 1972-present. ISM PMI was permanently removed from
                          # FRED in 2016. IPMAN is the best available free-API proxy:
                          # it measures actual manufacturing output (not survey), correlates
                          # ~0.85 with ISM PMI over full history, and has no API restriction.
        "SAHMREALTIME", "USSLIND",
    ]
    fred_quarterly = [
        "BCNSDODNS", "GDP", "NCBEILQ027S", "CPATAX",
        "BOGZ1FA106130001Q",  # Nonfinancial Corporate Business; Interest Paid, Transactions
                               # FRED: Q4 1946 – present (quarterly, seasonally adjusted)
                               # This is the CURRENT series — distinct from the retired
                               # BOGZ1FL103070005Q which was a Level (stock) series.
                               # FA = Flow of Funds (transactions); FL = Level. Use FA.
    ]

    for sid in fred_daily:
        _record(sid, fetch_fred_series, sid, frequency="daily")
    for sid in fred_weekly:
        _record(sid, fetch_fred_series, sid, frequency="weekly")
    for sid in fred_monthly:
        _record(sid, fetch_fred_series, sid, frequency="monthly")
    for sid in fred_quarterly:
        _record(sid, fetch_fred_series, sid, frequency="quarterly")

    # ── Shiller ────────────────────────────────────────────────────────────────
    _record("shiller", fetch_shiller_data)

    # ── AAII ──────────────────────────────────────────────────────────────────
    _record("aaii", fetch_aaii_sentiment)

    # ── yfinance — individual tickers ─────────────────────────────────────────
    # Longer delays between yfinance calls — Yahoo Finance enforces strict rate limits.
    # The YFTzMissingError / "Expecting value" errors are caused by Yahoo returning
    # an empty body when the session/crumb has expired. Spacing calls out reduces this.
    _record("sp500_price", fetch_yfinance, "^GSPC", "daily")
    time.sleep(5)
    _record("vix_price",   fetch_yfinance, "^VIX",  "daily")
    time.sleep(5)
    _record("spy_price",   fetch_yfinance, "SPY",   "daily")
    time.sleep(8)
    _record("spy_info",    fetch_yfinance_info, "SPY")

    # ── S&P 500 constituents + batch breadth prices ────────────────────────────
    _record("sp500_tickers", fetch_sp500_tickers)

    tickers = results.get("sp500_tickers") or []
    if isinstance(tickers, list) and tickers:
        breadth_start = date(START_DATE.year, START_DATE.month, START_DATE.day)
        _record(
            "sp500_breadth_prices",
            fetch_sp500_prices_for_breadth,
            tickers,
            breadth_start,
        )
    else:
        results["sp500_breadth_prices"] = None
        failures.append("sp500_breadth_prices (no tickers)")

    # ── Summary log ───────────────────────────────────────────────────────────
    log.info(
        "fetch_all_data complete: %d loaded, %d failed.\n"
        "  ✓ Loaded : %s\n"
        "  ✗ Failed : %s",
        len(successes), len(failures),
        ", ".join(successes) if successes else "(none)",
        ", ".join(failures) if failures else "(none)",
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — run directly to fetch all data
# Usage: python data_fetcher.py
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Force all logging to stdout so Windows Command Prompt shows output
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers and add a fresh stdout handler
    root_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt=config.LOG_FORMAT,
            datefmt=config.LOG_DATE_FORMAT,
        )
    )
    root_logger.addHandler(handler)

    print("=" * 60)
    print("  Bear Risk Dashboard — Data Fetcher")
    print("  Fetching all 20 indicator data sources...")
    print("  This takes 3-5 minutes on first run.")
    print("=" * 60)
    print()

    results = fetch_all_data()

    # Print a human-readable summary
    print()
    print("=" * 60)
    loaded  = [k for k, v in results.items() if v is not None]
    failed  = [k for k, v in results.items() if v is None]
    print(f"  Done. {len(loaded)} sources loaded, {len(failed)} failed.")
    if failed:
        print(f"  Failed sources: {', '.join(failed)}")
        print()
        print("  A few failures are normal (AAII, breadth prices).")
        print("  Core FRED series must all succeed for the dashboard to work.")
    else:
        print("  All sources loaded successfully.")
    print()
    print(f"  Cache location: {CACHE_DIR}")
    print("=" * 60)
