"""
src/utils/telegram.py
Telegram notifications with retries + message types (proven on the Alpaca edition).
"""
import time
import requests
from config.settings import config
from src.utils.logger import setup_logger

logger = setup_logger("Telegram", "logs/telegram.log")

_MAX_ATTEMPTS = 3
_TIMEOUT = 10
_BACKOFF = 3
_session = requests.Session()

ICONS = {
    'entry': '🟢', 'exit': '✅', 'partial': '🔹', 'warning': '⚠️',
    'critical': '🚨', 'report': '📊', 'heartbeat': '💓', 'target': '🎯',
    'stop': '🛑', 'info': 'ℹ️', 'brain': '🧠',
}


def send_telegram(message, kind: str = 'info'):
    icon = ICONS.get(kind, '')
    text = f"{icon} {message}" if icon and not message.startswith(icon) else message

    if not (config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID):
        logger.info(f"[TG-OFFLINE] {text}")
        return False

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": config.TELEGRAM_CHAT_ID, "text": text}

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = _session.post(url, json=payload, timeout=_TIMEOUT)
            if resp.status_code == 200:
                return True
            logger.warning(f"Telegram failed ({attempt}/{_MAX_ATTEMPTS}): {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Telegram error ({attempt}/{_MAX_ATTEMPTS}): {e}")
        if attempt < _MAX_ATTEMPTS:
            time.sleep(_BACKOFF * attempt)

    logger.error(f"Telegram gave up. Message: {text}")
    return False
