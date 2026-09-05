"""
src/live/reconciler.py
================================================================
Broker truth vs local state, plus the ONE shared close workflow.

The broker is the only truth about positions. Every cycle (and at
startup) we compare the broker's open positions against state["positions"]:

- broker has a position with OUR magic that state doesn't know
    -> ADOPT it (restart/crash recovery). Peak/stage history is lost, so
       adopted positions start conservative: peak = entry, no scale-outs
       done, ATR re-estimated from the live SL distance or fresh bars.
- state tracks a REAL ticket the broker no longer has
    -> it closed while we weren't looking (SL/TP fill, manual intervention).
       Pull the exact exit price/time/pnl from deal history and finalize.
- VIRTUAL tickets (< 0, DRY_RUN) are skipped: the broker never sees them.

finalize_close() is the single path every close flows through (manager-
initiated or broker-detected): journal -> loss cooldowns -> day counters
-> Telegram -> state removal. Partial closes never come here.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Optional

from config.settings import config
from src.live.journal import journal_close
from src.live.position_manager import ManagedPosition
from src.live.risk_engine import position_risk_usd, stop_distance_for
from src.live.state import iso, parse_iso
from src.utils.logger import setup_logger
from src.utils.telegram import send_telegram

logger = setup_logger("Reconciler", "logs/live.log")

_CLOSE_ICONS = {"stop_loss": "stop", "take_profit": "target",
                "scale_out_1": "partial", "scale_out_2": "partial",
                "time_partial": "partial"}


def estimate_pnl(pos: ManagedPosition, exit_price: float,
                 specs: Optional[dict], lots: float = None) -> float:
    """Account-ccy pnl for `lots` closed at exit_price, via broker tick
    math. Falls back to 0 when specs are missing (never crash a close)."""
    lots = pos.volume if lots is None else lots
    if not specs or specs.get("tick_size", 0) <= 0:
        return 0.0
    ticks = pos.sign * (exit_price - pos.entry_price) / specs["tick_size"]
    return lots * ticks * specs.get("tick_value", 0.0)


def finalize_close(state: dict, pos: ManagedPosition, *, exit_price: float,
                   exit_time: datetime, reason: str, pnl: float) -> None:
    """The single close workflow. Idempotent-ish: safe to call once per
    position removal; journal_close no-ops when the row is already closed."""
    r_multiple = pnl / pos.risk_usd if pos.risk_usd > 0 else 0.0
    journal_close(symbol=pos.symbol, ticket=pos.ticket, exit_time=exit_time,
                  exit_price=exit_price, pnl=round(pnl, 2),
                  r_multiple=round(r_multiple, 3), exit_reason=reason)

    # ---- loss cooldown bookkeeping (mirrors the backtest engine) -------
    sym = pos.symbol
    until = exit_time + timedelta(hours=config.COOLDOWN_AFTER_CLOSE_BARS)
    if reason == "stop_loss":
        until = max(until, exit_time + timedelta(hours=config.LOSS_COOLDOWN_HOURS))
        stops = state["stop_times"].setdefault(sym, [])
        stops.append(iso(exit_time))
        cutoff = exit_time - timedelta(days=config.REPEAT_LOSS_WINDOW_DAYS)
        recent = [t for t in (parse_iso(x) for x in stops)
                  if t is not None and t >= cutoff]
        state["stop_times"][sym] = [iso(t) for t in recent]
        if len(recent) >= 2:
            until = max(until, exit_time + timedelta(
                hours=config.REPEAT_LOSS_COOLDOWN_HOURS))
            send_telegram(f"{sym}: repeat stop-outs - symbol banned "
                          f"{config.REPEAT_LOSS_COOLDOWN_HOURS}h", "warning")
    prev = parse_iso(state["no_entry_until"].get(sym))
    if prev is None or until > prev:
        state["no_entry_until"][sym] = iso(until)

    # ---- day counters ---------------------------------------------------
    day = state.get("day", {})
    day["closed_count"] = int(day.get("closed_count", 0)) + 1
    day["realized_pnl"] = float(day.get("realized_pnl", 0.0)) + pnl

    # ---- notify + forget -------------------------------------------------
    kind = _CLOSE_ICONS.get(reason, "exit")
    send_telegram(
        f"{pos.symbol} {pos.side} closed ({reason})\n"
        f"exit {exit_price} | pnl {pnl:+.2f} ({r_multiple:+.2f}R)"
        + (" | DRY_RUN" if pos.dry_run else ""), kind)
    state["positions"].pop(str(pos.ticket), None)
    logger.info(f"closed {pos.symbol} ticket {pos.ticket}: {reason} "
                f"pnl={pnl:+.2f} r={r_multiple:+.2f}")


def adopt_position(state: dict, broker_pos: dict, canonical: str,
                   asset_class: str, atr: float, specs: Optional[dict]) -> None:
    """Build a conservative ManagedPosition from broker truth."""
    entry = float(broker_pos["entry_price"])
    sl = float(broker_pos.get("sl") or 0.0)
    if atr <= 0 and sl > 0 and entry > 0:
        atr = abs(entry - sl) / config.STOP_ATR_MULT   # infer from live stop
    pos = ManagedPosition(
        ticket=int(broker_pos["ticket"]), symbol=canonical,
        asset_class=asset_class, side=broker_pos["side"],
        entry_price=entry, entry_time=iso(broker_pos["time"]),
        initial_volume=float(broker_pos["volume"]),
        volume=float(broker_pos["volume"]),
        atr=atr, sl=sl, tp=float(broker_pos.get("tp") or 0.0),
        peak_price=entry, trough_price=entry,
        risk_usd=0.0, dry_run=False,
    )
    pos.risk_usd = position_risk_usd(pos.volume, stop_distance_for(
        {"entry_price": entry, "sl": sl, "atr": atr}), specs or {})
    state["positions"][str(pos.ticket)] = pos.to_dict()
    logger.info(f"adopted broker position: {canonical} {pos.side} "
                f"{pos.volume} lots ticket {pos.ticket}")
    send_telegram(f"Adopted open position at startup: {canonical} "
                  f"{pos.side} {pos.volume} lots (ticket {pos.ticket})", "info")


def reconcile(state: dict, connector, *, atr_for: Callable[[str], float],
              dry_run: bool) -> list[str]:
    """One reconciliation pass. Returns events that fired (for logging)."""
    events: list[str] = []
    if dry_run:
        # The virtual book is authoritative; we only REPORT real positions
        # carrying our magic (never adopt/manage them in DRY_RUN).
        try:
            real = connector.positions(ours_only=True)
        except Exception as exc:
            logger.warning(f"reconcile: broker read failed ({exc})")
            return events
        if real:
            logger.info(f"reconcile (DRY_RUN): {len(real)} real positions "
                        f"with our magic at broker - left untouched")
        return events

    try:
        broker_positions = connector.positions(ours_only=True)
    except Exception as exc:
        logger.warning(f"reconcile: broker read failed ({exc}) - keeping state")
        return events
    broker_by_ticket = {int(p["ticket"]): p for p in broker_positions}
    reverse_map = {v: k for k, v in connector._symbol_map.items()}

    # ---- 1. state positions the broker no longer has -> closed away -----
    for ticket_s, pos_d in list(state["positions"].items()):
        pos = ManagedPosition.from_dict(pos_d)
        if pos.ticket < 0:
            continue                      # virtual: broker never had it
        if pos.ticket in broker_by_ticket:
            continue
        summary = None
        try:
            summary = connector.closed_position_summary(pos.ticket)
        except Exception as exc:
            logger.warning(f"history lookup failed for {pos.ticket}: {exc}")
        if summary:
            exit_price = summary["exit_price"]
            exit_time = summary["exit_time"]
            pnl = summary["profit"]
        else:                             # history unreachable: estimate
            exit_price = pos.sl or pos.entry_price
            exit_time = datetime.now().astimezone()
            specs = None
            try:
                specs = connector.symbol_specs(pos.symbol)
            except Exception:
                pass
            pnl = estimate_pnl(pos, exit_price, specs)
        # SL or TP? whichever the exit price is closer to
        reason = "closed_away"
        if pos.sl > 0 and pos.tp > 0:
            reason = ("stop_loss" if abs(exit_price - pos.sl)
                      <= abs(exit_price - pos.tp) else "take_profit")
        finalize_close(state, pos, exit_price=exit_price,
                       exit_time=exit_time, reason=reason, pnl=pnl)
        events.append(f"closed_away:{pos.symbol}")

    # ---- 2. broker positions state doesn't know -> adopt -----------------
    if config.ADOPT_BROKER_POSITIONS:
        for ticket, bp in broker_by_ticket.items():
            if str(ticket) in state["positions"]:
                continue
            canonical = reverse_map.get(bp["symbol"], bp["symbol"])
            asset_class = connector.classify_asset(canonical)
            atr = 0.0
            try:
                atr = float(atr_for(canonical) or 0.0)
            except Exception:
                pass
            specs = None
            try:
                specs = connector.symbol_specs(canonical)
            except Exception:
                pass
            adopt_position(state, bp, canonical, asset_class, atr, specs)
            events.append(f"adopted:{canonical}")

    return events
