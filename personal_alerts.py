"""
personal_alerts.py — Personal Threshold Crossing Alerts
=========================================================
Checks whether the Bear Risk Score or any individual indicator has
crossed a configured threshold since the last check, and delivers
alerts via desktop notification and/or email.

Run on startup or as part of the scheduler:

    python personal_alerts.py

All thresholds and email settings are in config.py — do NOT hardcode
them here.  This module reads state from a small JSON file
(config.ALERT_STATE_FILE) to detect crossings, not just levels.

Dependencies: pip install plyer  (email uses stdlib smtplib)
"""

from __future__ import annotations

import json
import logging
import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import config
from config import (
    ALERT_SCORE_ELEVATED,
    ALERT_SCORE_HIGH,
    ALERT_INDICATOR_RED_THRESHOLD,
    ALERT_STATE_FILE,
    EMAIL_ENABLED,
    EMAIL_SENDER,
    EMAIL_PASSWORD,
    EMAIL_RECIPIENT,
    EMAIL_SMTP_HOST,
    EMAIL_SMTP_PORT,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)
from bear_risk_api import (
    get_current_risk_score,
    get_top_risk_drivers,
    get_indicator_status,
    get_macro_context_summary,
)
from config import INDICATORS

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
log = logging.getLogger("personal_alerts")

# ── Optional desktop notifications ────────────────────────────────────────────
try:
    from plyer import notification as _plyer_notification
    _PLYER_AVAILABLE = True
except ImportError:
    _PLYER_AVAILABLE = False
    log.warning("plyer not installed — desktop notifications disabled.")


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _load_state() -> dict:
    """
    Load persisted alert state from disk.
    State tracks the last-seen composite score and which thresholds
    have already been notified (to avoid repeated alerts).
    """
    if not ALERT_STATE_FILE.exists():
        return {
            "last_score": None,
            "thresholds_crossed": [],
            "last_check": None,
        }
    try:
        with open(ALERT_STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Could not read alert state: %s — starting fresh.", exc)
        return {"last_score": None, "thresholds_crossed": [], "last_check": None}


def _save_state(state: dict) -> None:
    """Persist alert state to disk."""
    try:
        ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ALERT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as exc:
        log.error("Could not save alert state: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTIFICATION DELIVERY
# ═══════════════════════════════════════════════════════════════════════════════

def _send_desktop(title: str, message: str) -> None:
    """Send a desktop notification via plyer."""
    if not _PLYER_AVAILABLE:
        log.info("Desktop alert (plyer unavailable): %s — %s", title, message)
        return
    try:
        _plyer_notification.notify(
            title=title,
            message=message,
            app_name="Bear Risk Dashboard",
            timeout=15,
        )
        log.info("Desktop notification sent: %s", title)
    except Exception as exc:
        log.warning("Desktop notification failed: %s", exc)


def _send_email(subject: str, body: str) -> None:
    """
    Send an alert email via Gmail SMTP (or any SMTP server in config.py).

    Requires EMAIL_ENABLED = True and valid credentials in config.py.
    Uses TLS — never sends credentials in plain text.
    """
    if not EMAIL_ENABLED:
        return
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECIPIENT:
        log.warning("Email not configured — skipping email alert. "
                    "Set EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT in config.py.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECIPIENT

    # Plain text body
    text_part = MIMEText(body, "plain")
    msg.attach(text_part)

    # Simple HTML version
    html_body = (
        f"<html><body style='font-family:monospace;'>"
        f"<h2 style='color:#d50000;'>🐻 Bear Risk Alert</h2>"
        f"<pre>{body}</pre>"
        f"<hr><small>Bear Risk Dashboard · Educational tool only</small>"
        f"</body></html>"
    )
    html_part = MIMEText(html_body, "html")
    msg.attach(html_part)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
        log.info("Email alert sent to %s: %s", EMAIL_RECIPIENT, subject)
    except smtplib.SMTPAuthenticationError:
        log.error(
            "Email authentication failed. Use a Gmail App Password, not your main password. "
            "Generate one at: https://myaccount.google.com/apppasswords"
        )
    except Exception as exc:
        log.error("Email send failed: %s", exc)


def _alert(title: str, message: str, email_subject: Optional[str] = None) -> None:
    """Deliver an alert via all configured channels."""
    _send_desktop(title, message)
    _send_email(email_subject or title, message)
    log.warning("ALERT: %s — %s", title, message)


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD CROSSING DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_composite_crossings(
    current_score: float,
    last_score: Optional[float],
    already_crossed: list[str],
) -> list[tuple[str, str]]:
    """
    Detect upward crossings of the configured composite thresholds.

    Returns list of (threshold_name, alert_message) for each new crossing.
    A crossing is only reported once until the score drops back below
    the threshold and re-crosses (reset on downward crossing).
    """
    crossings = []
    thresholds = [
        ("elevated", ALERT_SCORE_ELEVATED, "ELEVATED"),
        ("high",     ALERT_SCORE_HIGH,     "HIGH"),
    ]

    for key, level, band_name in thresholds:
        crossed_key = f"composite_{key}"
        just_crossed = (
            current_score >= level and
            (last_score is None or last_score < level) and
            crossed_key not in already_crossed
        )
        just_uncrossed = current_score < level - 5  # 5-point hysteresis

        if just_crossed:
            msg = (
                f"Bear Risk Score is now {current_score:.0f} — crossed into {band_name} zone "
                f"(threshold: {level:.0f}). Review your portfolio allocation."
            )
            crossings.append((crossed_key, msg))
        elif just_uncrossed and crossed_key in already_crossed:
            # Reset so crossing can fire again in the future
            already_crossed.remove(crossed_key)

    return crossings


def _detect_indicator_red(
    already_crossed: list[str],
) -> list[tuple[str, str]]:
    """
    Detect any individual indicator newly entering red zone.
    Returns list of (crossing_key, alert_message).
    """
    crossings = []
    for ind_id, meta in INDICATORS.items():
        status = get_indicator_status(meta.name)
        if status is None:
            continue

        score = status.get("score", 0)
        key = f"indicator_red_{ind_id}"

        just_red = score >= ALERT_INDICATOR_RED_THRESHOLD and key not in already_crossed
        no_longer_red = score < ALERT_INDICATOR_RED_THRESHOLD - 5

        if just_red:
            msg = (
                f"{meta.name} entered RED zone (score: {score:.0f}/100). "
                f"Current value: {status.get('raw_value', 'N/A')}. "
                f"Trend: {status.get('direction', '?')}. "
                f"Threshold: {status.get('threshold_note', '')}."
            )
            crossings.append((key, msg))
        elif no_longer_red and key in already_crossed:
            already_crossed.remove(key)

    return crossings


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHECK FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_alert_check() -> None:
    """
    Run one complete alert check cycle.

    - Loads current scores from bear_risk_api (cache-only, <100ms).
    - Compares against last-seen state.
    - Fires desktop + email alerts for any new threshold crossings.
    - Saves updated state to disk.
    """
    state = _load_state()
    last_score    = state.get("last_score")
    already_crossed = state.get("thresholds_crossed", [])
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Fetch current scores ───────────────────────────────────────────────────
    score_dict = get_current_risk_score()
    if score_dict is None:
        log.warning("Alert check: no data available — skipping.")
        return

    current_score = score_dict["score"]
    band = score_dict["band"]
    log.info("Alert check at %s: score=%.1f band=%s", now_str, current_score, band)

    # ── Composite threshold crossings ──────────────────────────────────────────
    comp_crossings = _detect_composite_crossings(current_score, last_score, already_crossed)
    for key, msg in comp_crossings:
        already_crossed.append(key)
        summary = get_macro_context_summary() or ""
        full_msg = f"{msg}\n\n{summary}" if summary else msg
        _alert(
            title=f"🐻 Bear Risk: {band} ({current_score:.0f}/100)",
            message=full_msg,
            email_subject=f"Bear Risk Alert: Score Crossed {band} Threshold",
        )

    # ── Indicator red crossings ────────────────────────────────────────────────
    ind_crossings = _detect_indicator_red(already_crossed)
    for key, msg in ind_crossings:
        already_crossed.append(key)
        _alert(
            title="🐻 Bear Risk: Indicator Alert",
            message=msg,
            email_subject="Bear Risk: Indicator Entered Red Zone",
        )

    # ── If already in SEVERE with no crossing (reminder every 7 days) ─────────
    if band == "SEVERE":
        last_severe_alert = state.get("last_severe_reminder")
        if last_severe_alert is None:
            days_since = 8  # trigger on first check in SEVERE
        else:
            last_dt = datetime.fromisoformat(last_severe_alert)
            days_since = (datetime.now() - last_dt).days
        if days_since >= 7:
            drivers = get_top_risk_drivers(3)
            driver_str = "\n".join(
                f"  • {d['indicator']}: {d['plain_english']}" for d in drivers
            )
            _alert(
                title="🐻 Bear Risk SEVERE — Weekly Reminder",
                message=f"Score remains at {current_score:.0f}/100 (SEVERE).\n\n"
                        f"Top drivers:\n{driver_str}\n\nReview hedge ratios.",
                email_subject="Bear Risk: Weekly SEVERE Reminder",
            )
            state["last_severe_reminder"] = datetime.now().isoformat()

    # ── Persist updated state ──────────────────────────────────────────────────
    state["last_score"] = current_score
    state["thresholds_crossed"] = already_crossed
    state["last_check"] = now_str
    _save_state(state)

    if not comp_crossings and not ind_crossings:
        log.info("No new threshold crossings detected.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_alert_check()
