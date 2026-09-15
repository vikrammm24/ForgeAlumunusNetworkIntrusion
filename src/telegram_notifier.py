"""
src/telegram_notifier.py
------------------------
Telegram bot notifications for NIDA alerts.

Configuration (via .env):
  TELEGRAM_BOT_TOKEN  - Bot token from BotFather
  TELEGRAM_CHAT_ID    - Destination chat/group ID (see below)
  TELEGRAM_MIN_RISK   - Minimum risk score to notify (default 70)
  TELEGRAM_ENABLED    - Set to "false" to disable without removing token

How to get your CHAT_ID:
  1. Start a conversation with your bot at t.me/NIDAII_bot
  2. Send any message (e.g. /start)
  3. Open https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
  4. Copy the "chat" -> "id" value from the response into TELEGRAM_CHAT_ID

The notifier sends one message per high-severity alert using the
sendMessage Bot API endpoint. It runs in a background thread so it
never blocks the Flask request. Failed sends are logged and silently
dropped -- notification is best-effort, never mission-critical.
"""

import logging
import os
import threading
from typing import Optional

import requests

log = logging.getLogger("telegram_notifier")

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_config() -> dict:
    """Read Telegram config from environment at call time (supports .env reload)."""
    return {
        "token": os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        "min_risk": int(os.getenv("TELEGRAM_MIN_RISK", "70")),
        "enabled": os.getenv("TELEGRAM_ENABLED", "true").lower() not in ("false", "0", "no"),
    }


def is_configured() -> bool:
    """Return True if the notifier has a token and chat_id configured."""
    cfg = _get_config()
    return bool(cfg["token"] and cfg["chat_id"] and cfg["enabled"])


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------

def _severity_emoji(risk_score: float) -> str:
    if risk_score >= 90:
        return "🔴"
    if risk_score >= 75:
        return "🟠"
    if risk_score >= 60:
        return "🟡"
    return "🟢"


def _format_alert_message(alert: dict) -> str:
    """
    Format an alert dict into a Telegram message.
    Uses MarkdownV2 escaping for special characters.
    """
    risk = alert.get("risk_score", 0)
    category = (alert.get("attack_category") or "UNKNOWN").upper()
    label = alert.get("attack_label") or category
    technique_id = alert.get("technique_id") or ""
    technique_name = alert.get("technique_name") or ""
    ts = alert.get("timestamp", "")[:19]  # trim microseconds
    summary = alert.get("incident_summary") or ""
    reason = alert.get("reason_string") or ""

    emoji = _severity_emoji(risk)
    mitre_line = f"🛡 MITRE: `{technique_id}` {technique_name}" if technique_id else "🛡 MITRE: Unclassified anomaly"

    # Plain text (HTML parse mode is safer than MarkdownV2 for dynamic content)
    lines = [
        f"{emoji} <b>NIDA Alert — {category}</b>",
        f"",
        f"🎯 <b>Attack:</b> {label}",
        f"⚠️ <b>Risk Score:</b> {risk:.0f}/100",
        f"🕐 <b>Time:</b> {ts}",
        mitre_line,
    ]

    if reason:
        # Trim long reason strings
        short_reason = reason[:200] + "..." if len(reason) > 200 else reason
        lines.append(f"")
        lines.append(f"🔍 <b>Reason:</b> {short_reason}")

    if summary:
        short_summary = summary[:300] + "..." if len(summary) > 300 else summary
        lines.append(f"")
        lines.append(f"📋 <b>Summary:</b> {short_summary}")

    lines.append(f"")
    lines.append(f"<i>NIDA — Network Intrusion Detection Agent</i>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Send logic
# ---------------------------------------------------------------------------

def _send_message(token: str, chat_id: str, text: str) -> bool:
    """
    POST to Telegram sendMessage API. Returns True on success.
    Uses HTML parse mode for safe formatting.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        log.warning("Telegram sendMessage failed: %s %s", resp.status_code, resp.text[:200])
        return False
    except requests.RequestException as exc:
        log.warning("Telegram sendMessage exception: %s", exc)
        return False


def _send_in_background(token: str, chat_id: str, text: str):
    """Fire-and-forget: send in a daemon thread so Flask never blocks."""
    t = threading.Thread(target=_send_message, args=(token, chat_id, text), daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def notify_alert(alert: dict) -> bool:
    """
    Send a Telegram notification for a new alert, if configured and
    the risk score meets the minimum threshold.

    Parameters
    ----------
    alert : dict
        Alert record (same shape as Alert.to_dict()).

    Returns
    -------
    bool
        True if a notification was dispatched (fire-and-forget).
        False if skipped (not configured, below threshold, disabled).
    """
    cfg = _get_config()

    if not cfg["enabled"]:
        return False
    if not cfg["token"] or not cfg["chat_id"]:
        return False

    risk = float(alert.get("risk_score") or 0)
    if risk < cfg["min_risk"]:
        return False

    text = _format_alert_message(alert)
    _send_in_background(cfg["token"], cfg["chat_id"], text)
    log.info("Telegram notification dispatched for alert risk=%.0f category=%s",
             risk, alert.get("attack_category"))
    return True


def notify_batch_summary(new_alerts: int, critical_count: int, top_category: str):
    """
    Send a batch-summary notification after a detection run.
    Only fires if there were alerts -- keeps noise low.
    """
    cfg = _get_config()
    if not cfg["enabled"] or not cfg["token"] or not cfg["chat_id"]:
        return
    if new_alerts == 0:
        return

    text = (
        f"📊 <b>NIDA Detection Run Complete</b>\n\n"
        f"🔔 New alerts: <b>{new_alerts}</b>\n"
        f"🔴 Critical (≥90): <b>{critical_count}</b>\n"
        f"📂 Top category: <b>{top_category or 'N/A'}</b>\n\n"
        f"<i>Open the dashboard to review.</i>"
    )
    _send_in_background(cfg["token"], cfg["chat_id"], text)


def send_test_message() -> tuple[bool, str]:
    """
    Send a test message to verify the bot is configured correctly.
    Called from the /api/telegram/test endpoint.
    Returns (success, message).
    """
    cfg = _get_config()
    if not cfg["token"]:
        return False, "TELEGRAM_BOT_TOKEN not set in .env"
    if not cfg["chat_id"]:
        return False, "TELEGRAM_CHAT_ID not set in .env — send /start to the bot, then check getUpdates"

    text = (
        "✅ <b>NIDA Bot Connected!</b>\n\n"
        "Your Network Intrusion Detection Agent will now send alerts here "
        f"for events with risk score ≥ {cfg['min_risk']}.\n\n"
        "<i>NIDA — Network Intrusion Detection Agent</i>"
    )
    ok = _send_message(cfg["token"], cfg["chat_id"], text)
    if ok:
        return True, "Test message sent successfully"
    return False, "Failed to send — check token and chat_id (see logs)"
