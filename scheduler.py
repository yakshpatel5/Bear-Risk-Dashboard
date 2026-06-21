"""
scheduler.py — Background Data Refresh Scheduler
=================================================
Runs continuously in the background while the dashboard is open.
Keeps the local cache fresh without manual intervention.

Usage
-----
    # Normal background mode (start once, leave running):
    python scheduler.py

    # Force-refresh everything right now and exit:
    python scheduler.py --force-refresh

    # Refresh a specific frequency tier only:
    python scheduler.py --force-refresh --tier daily
    python scheduler.py --force-refresh --tier weekly
    python scheduler.py --force-refresh --tier monthly

The scheduler reads timing from config.py — change SCHEDULE_* constants
there to adjust when each refresh fires.  No edits to this file required.

Dependencies: pip install schedule plyer
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import schedule

# ── Local imports ─────────────────────────────────────────────────────────────
import config
from config import (
    CACHE_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    SCHEDULE_DAILY_TIME,
    SCHEDULE_WEEKLY_TIME,
    SCHEDULE_MONTHLY_TIME,
)
from data_fetcher import (
    fetch_fred_series,
    fetch_shiller_data,
    fetch_aaii_sentiment,
    fetch_sp500_tickers,
    fetch_sp500_prices_for_breadth,
    fetch_yfinance,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CACHE_DIR / "scheduler.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("scheduler")

# ── Optional: desktop notifications ───────────────────────────────────────────
try:
    from plyer import notification as _plyer_notification
    _PLYER_AVAILABLE = True
except ImportError:
    _PLYER_AVAILABLE = False
    log.warning("plyer not installed — desktop notifications disabled.  Run: pip install plyer")


def _notify(title: str, message: str) -> None:
    """Send a desktop notification; silently skip if plyer is unavailable."""
    if not _PLYER_AVAILABLE:
        return
    try:
        _plyer_notification.notify(
            title=title,
            message=message,
            app_name="Bear Risk Dashboard",
            timeout=8,
        )
    except Exception as exc:
        log.debug("Desktop notification failed: %s", exc)


def _log_refresh_summary(label: str, successes: list[str], failures: list[str]) -> None:
    """Log a structured summary after a refresh job completes."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    if failures:
        log.warning(
            "[%s] %s refresh: %d OK, %d FAILED — failed: %s",
            ts, label, len(successes), len(failures), ", ".join(failures),
        )
        _notify(
            "Bear Risk — Refresh Warning",
            f"{label} refresh: {len(failures)} source(s) failed.\n"
            f"Failed: {', '.join(failures[:3])}",
        )
    else:
        log.info(
            "[%s] %s refresh: all %d sources updated successfully.",
            ts, label, len(successes),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# REFRESH JOBS
# ═══════════════════════════════════════════════════════════════════════════════

# FRED series grouped by update frequency.
# Adding a new series? Just add its ID to the appropriate list.
_DAILY_FRED_SERIES = [
    "T10Y2Y", "BAMLH0A0HYM2", "BAMLC0A0CM", "VIXCLS", "DFII10",
]
_WEEKLY_FRED_SERIES = ["NFCI"]
_MONTHLY_FRED_SERIES = [
    "FEDFUNDS", "PCEPILFE", "NAPM", "SAHMREALTIME", "USSLIND",
]
_QUARTERLY_FRED_SERIES = [
    "BCNSDODNS", "GDP", "NCBEILQ027S", "CPATAX", "BOGZ1FL103070005Q",
]
_DAILY_YF_TICKERS = ["^GSPC", "^VIX", "SPY"]


def refresh_daily() -> None:
    """
    Refresh all daily-frequency data sources.
    Runs every weekday at SCHEDULE_DAILY_TIME.
    """
    log.info("=== Daily refresh started ===")
    successes, failures = [], []

    # FRED daily series
    for sid in _DAILY_FRED_SERIES:
        try:
            df = fetch_fred_series(sid, frequency="daily")
            if df is not None and not df.empty:
                successes.append(sid)
            else:
                failures.append(f"{sid}(empty)")
        except Exception as exc:
            log.error("Daily FRED %s failed: %s", sid, exc)
            failures.append(sid)

    # Yahoo Finance tickers
    for ticker in _DAILY_YF_TICKERS:
        try:
            df = fetch_yfinance(ticker, frequency="daily")
            if df is not None and not df.empty:
                successes.append(ticker)
            else:
                failures.append(f"{ticker}(empty)")
        except Exception as exc:
            log.error("Daily yfinance %s failed: %s", ticker, exc)
            failures.append(ticker)

    _log_refresh_summary("Daily", successes, failures)


def refresh_weekly() -> None:
    """
    Refresh weekly-frequency data sources.
    Runs every Monday at SCHEDULE_WEEKLY_TIME.
    """
    log.info("=== Weekly refresh started ===")
    successes, failures = [], []

    # FRED weekly series
    for sid in _WEEKLY_FRED_SERIES:
        try:
            df = fetch_fred_series(sid, frequency="weekly")
            if df is not None and not df.empty:
                successes.append(sid)
            else:
                failures.append(f"{sid}(empty)")
        except Exception as exc:
            log.error("Weekly FRED %s failed: %s", sid, exc)
            failures.append(sid)

    # AAII sentiment
    try:
        df = fetch_aaii_sentiment()
        if df is not None and not df.empty:
            successes.append("AAII")
        else:
            failures.append("AAII(empty)")
    except Exception as exc:
        log.error("AAII fetch failed: %s", exc)
        failures.append("AAII")

    # S&P 500 constituent breadth prices (batched weekly)
    try:
        tickers = fetch_sp500_tickers()
        if tickers:
            from config import START_DATE
            df = fetch_sp500_prices_for_breadth(tickers, START_DATE)
            if df is not None and not df.empty:
                successes.append("SP500_breadth")
            else:
                failures.append("SP500_breadth(empty)")
    except Exception as exc:
        log.error("SP500 breadth fetch failed: %s", exc)
        failures.append("SP500_breadth")

    _log_refresh_summary("Weekly", successes, failures)


def refresh_monthly() -> None:
    """
    Refresh monthly- and quarterly-frequency data sources.
    Runs on the 1st of each month at SCHEDULE_MONTHLY_TIME.
    """
    log.info("=== Monthly refresh started ===")
    successes, failures = [], []

    # Monthly FRED
    for sid in _MONTHLY_FRED_SERIES:
        try:
            df = fetch_fred_series(sid, frequency="monthly")
            if df is not None and not df.empty:
                successes.append(sid)
            else:
                failures.append(f"{sid}(empty)")
        except Exception as exc:
            log.error("Monthly FRED %s failed: %s", sid, exc)
            failures.append(sid)

    # Quarterly FRED
    for sid in _QUARTERLY_FRED_SERIES:
        try:
            df = fetch_fred_series(sid, frequency="quarterly")
            if df is not None and not df.empty:
                successes.append(sid)
            else:
                failures.append(f"{sid}(empty)")
        except Exception as exc:
            log.error("Quarterly FRED %s failed: %s", sid, exc)
            failures.append(sid)

    # Shiller XLS
    try:
        df = fetch_shiller_data()
        if df is not None and not df.empty:
            successes.append("Shiller")
        else:
            failures.append("Shiller(empty)")
    except Exception as exc:
        log.error("Shiller fetch failed: %s", exc)
        failures.append("Shiller")

    # S&P 500 constituent list (refreshed monthly)
    try:
        tickers = fetch_sp500_tickers()
        if tickers:
            successes.append("SP500_tickers")
        else:
            failures.append("SP500_tickers(empty)")
    except Exception as exc:
        log.error("SP500 tickers fetch failed: %s", exc)
        failures.append("SP500_tickers")

    _log_refresh_summary("Monthly", successes, failures)


def refresh_all() -> None:
    """Force-refresh everything regardless of frequency tier."""
    log.info("=== FULL FORCE REFRESH ===")
    refresh_daily()
    refresh_weekly()
    refresh_monthly()
    log.info("=== Force refresh complete ===")
    _notify("Bear Risk Dashboard", "Force refresh complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULER SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_schedule() -> None:
    """Register all recurring jobs with the schedule library."""
    # Daily indicators: every weekday at configured time
    schedule.every().monday.at(SCHEDULE_DAILY_TIME).do(refresh_daily)
    schedule.every().tuesday.at(SCHEDULE_DAILY_TIME).do(refresh_daily)
    schedule.every().wednesday.at(SCHEDULE_DAILY_TIME).do(refresh_daily)
    schedule.every().thursday.at(SCHEDULE_DAILY_TIME).do(refresh_daily)
    schedule.every().friday.at(SCHEDULE_DAILY_TIME).do(refresh_daily)

    # Weekly indicators: every Monday
    schedule.every().monday.at(SCHEDULE_WEEKLY_TIME).do(refresh_weekly)

    # Monthly indicators: every day, but the job self-checks for the 1st
    def _monthly_guard():
        if datetime.now().day == 1:
            refresh_monthly()

    schedule.every().day.at(SCHEDULE_MONTHLY_TIME).do(_monthly_guard)

    log.info(
        "Schedule registered: daily@%s (weekdays), weekly@%s (Mon), monthly@%s (1st)",
        SCHEDULE_DAILY_TIME, SCHEDULE_WEEKLY_TIME, SCHEDULE_MONTHLY_TIME,
    )


def run_scheduler() -> None:
    """Main loop — runs until keyboard interrupt."""
    setup_schedule()
    log.info("Scheduler running. Press Ctrl+C to stop.")
    _notify("Bear Risk Dashboard", "Scheduler started.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        log.info("Scheduler stopped by user.")
        _notify("Bear Risk Dashboard", "Scheduler stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bear Risk Dashboard Data Scheduler")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refresh all data immediately and exit.",
    )
    parser.add_argument(
        "--tier",
        choices=["daily", "weekly", "monthly"],
        default=None,
        help="With --force-refresh: limit to one frequency tier.",
    )
    args = parser.parse_args()

    if args.force_refresh:
        if args.tier == "daily":
            refresh_daily()
        elif args.tier == "weekly":
            refresh_weekly()
        elif args.tier == "monthly":
            refresh_monthly()
        else:
            refresh_all()
        sys.exit(0)
    else:
        run_scheduler()
