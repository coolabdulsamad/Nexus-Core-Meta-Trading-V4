"""
src/live/risk_engine.py
================================================================
The portfolio risk engine - V4's big new shield vs the Alpaca edition.

Forex pairs share currencies: long EURUSD + long GBPUSD is a doubled
USD-short. The Alpaca edition sized every position in isolation; V4 tracks
NET RISK PER CURRENCY across the whole book and refuses entries that push
a currency past its cap, plus a hard cap on TOTAL open risk, a max position
count, daily loss/profit guards and an account drawdown circuit breaker.

Everything here is PURE MATH over plain dicts (no MT5 / DB imports) so the
whole engine is unit-testable offline. The caller (live_trader) supplies
account info, open positions and symbol specs.

Risk convention: a position's open risk = what it loses if its stop fills,
in account currency, computed from the broker's tick value / tick size
(exact, not the backtester's cross-rate approximation).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from config.settings import config
from src.mt5_client.connector import MT5Connector
from src.utils.logger import setup_logger

logger = setup_logger("RiskEngine", "logs/live.log")


def position_risk_usd(volume_lots: float, stop_dist_price: float,
                      specs: dict) -> float:
    """Account-currency loss if the stop fills: lots x ticks x tick_value."""
    if volume_lots <= 0 or stop_dist_price <= 0 or not specs:
        return 0.0
    tick_size = specs.get("tick_size") or 0.0
    tick_value = specs.get("tick_value") or 0.0
    if tick_size <= 0 or tick_value <= 0:
        return 0.0
    return volume_lots * (stop_dist_price / tick_size) * tick_value


def stop_distance_for(pos: dict) -> float:
    """Stop distance in price units. Uses the live SL when the broker has
    one; falls back to the configured ATR distance from entry (adopted
    positions may carry no SL)."""
    entry = float(pos.get("entry_price") or 0.0)
    sl = float(pos.get("sl") or 0.0)
    if entry > 0 and sl > 0:
        return abs(entry - sl)
    atr = float(pos.get("atr") or 0.0)
    return config.STOP_ATR_MULT * atr if atr > 0 else 0.0


def currency_exposure(positions: list[dict]) -> dict[str, float]:
    """Net open risk per CURRENCY (account currency units), signed.
    LONG EURUSD = +risk on EUR, -risk on USD. Metals/crypto bases count as
    their own currency (XAU, XAG, BTC...). Pairs that don't split (indices)
    are skipped."""
    net: dict[str, float] = {}
    for pos in positions:
        pair = MT5Connector.split_pair(pos.get("symbol", ""))
        if not pair:
            continue
        base, quote = pair
        risk = float(pos.get("risk_usd") or 0.0)
        if risk <= 0:
            continue
        sign = 1.0 if pos.get("side") == "LONG" else -1.0
        net[base] = net.get(base, 0.0) + sign * risk
        net[quote] = net.get(quote, 0.0) - sign * risk
    return net


class RiskEngine:
    """Portfolio-level entry vetoes. Construct once per cycle with the
    current account + open book; then ask it about candidate trades."""

    def __init__(self, account: dict, open_positions: list[dict]):
        self.account = account or {}
        self.equity = float(self.account.get("equity") or 0.0)
        self.open_positions = open_positions  # each: symbol/side/volume/sl/entry_price/atr/risk_usd
        self.total_open_risk = sum(float(p.get("risk_usd") or 0.0)
                                   for p in open_positions)
        self.exposure = currency_exposure(open_positions)

    # ------------------------------------------------------------------
    def check_entry(self, symbol: str, side: str, lots: float,
                    sl: float, entry_price: float, specs: dict,
                    atr: float) -> tuple[bool, str]:
        """All portfolio gates for ONE candidate trade. Returns (ok, why)."""
        if self.equity <= 0:
            return False, "no equity reading"

        if len(self.open_positions) >= config.MAX_POSITIONS:
            return False, f"max positions ({config.MAX_POSITIONS}) reached"

        stop_dist = abs(entry_price - sl) if sl > 0 else config.STOP_ATR_MULT * atr
        new_risk = position_risk_usd(lots, stop_dist, specs)
        if new_risk <= 0:
            return False, "zero risk sizing (lots too small?)"

        # total open risk cap
        cap_total = config.TOTAL_OPEN_RISK_PCT * self.equity
        if self.total_open_risk + new_risk > cap_total:
            return False, (f"total open risk {self.total_open_risk + new_risk:.0f} "
                           f"> cap {cap_total:.0f}")

        # per-currency net risk cap
        if config.CORRELATION_CAP_ENABLED:
            pair = MT5Connector.split_pair(symbol)
            if pair:
                base, quote = pair
                sign = 1.0 if side == "LONG" else -1.0
                cap_ccy = config.CURRENCY_RISK_CAP_PCT * self.equity
                for ccy, delta in ((base, sign * new_risk),
                                   (quote, -sign * new_risk)):
                    net_after = self.exposure.get(ccy, 0.0) + delta
                    if abs(net_after) > cap_ccy:
                        return False, (f"currency cap: {ccy} net risk would be "
                                       f"{net_after:+.0f} (cap {cap_ccy:.0f})")
        return True, "ok"


# ---------------------------------------------------------------------------
# Daily guards + drawdown breaker (account-wide; mutate the state dict)
# ---------------------------------------------------------------------------
def refresh_daily_guards(state: dict, account: dict, now: datetime) -> list[str]:
    """Roll the daily anchors at a new UTC day, track peak equity, and
    evaluate the three account-level guards. Returns a list of events that
    fired (for logging/Telegram)."""
    events: list[str] = []
    equity = float(account.get("equity") or 0.0)
    balance = float(account.get("balance") or 0.0)
    if equity <= 0:
        return events

    today = now.date().isoformat()
    day = state["day"]
    if day.get("date") != today:
        state["day"] = {
            "date": today, "start_balance": balance, "start_equity": equity,
            "closed_count": 0, "realized_pnl": 0.0,
            "halted_loss": False, "profit_lock": False,
        }
        state["halted_drawdown"] = False   # breaker re-arms each new UTC day
        events.append("new_day")
        day = state["day"]

    peak = state.get("peak_equity")
    if peak is None or equity > peak:
        state["peak_equity"] = equity
        peak = equity

    # daily loss limit (vs day-start balance)
    start_balance = float(day.get("start_balance") or 0.0)
    if start_balance > 0 and not day.get("halted_loss"):
        day_ret = (equity - start_balance) / start_balance
        if day_ret <= -config.DAILY_LOSS_LIMIT_PCT:
            day["halted_loss"] = True
            events.append("daily_loss_limit")

    # daily profit target -> stop opening (optional breakeven lock)
    if (config.DAILY_PROFIT_TARGET_PCT > 0 and start_balance > 0
            and not day.get("profit_lock")):
        day_ret = (equity - start_balance) / start_balance
        if day_ret >= config.DAILY_PROFIT_TARGET_PCT:
            day["profit_lock"] = True
            events.append("daily_profit_target")

    # drawdown circuit breaker (latches until the next UTC day)
    if peak and equity < peak * (1.0 - config.MAX_DRAWDOWN_PCT) \
            and not state.get("halted_drawdown"):
        state["halted_drawdown"] = True
        events.append("drawdown_breaker")

    return events


def entries_allowed(state: dict) -> tuple[bool, str]:
    """Single question the entry pipeline asks before ANY new trade."""
    if state.get("halted_drawdown"):
        return False, "drawdown circuit breaker latched"
    day = state.get("day", {})
    if day.get("halted_loss"):
        return False, "daily loss limit hit"
    if day.get("profit_lock"):
        return False, "daily profit target locked"
    return True, "ok"
