"""
src/live/journal.py
================================================================
Trade journal writes to the Postgres `trades` table.

Standing rule: nothing is traded that wasn't journaled. Every entry writes
a row at fill time; every exit updates that row with the broker's truth
(exit price / time / pnl from deal history, never our estimates).

Rows are matched open->close by the ticket embedded in notes
("ticket:<n> ..."), which works in hedging mode where one symbol can have
several concurrent positions (symbol+magic is NOT unique there).

Journal failures never crash the trader: they are logged and reported via
the return value; the state file remains the working record and the next
reconcile pass re-attempts the close write.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.storage.db import get_conn
from src.utils.logger import setup_logger

logger = setup_logger("Journal", "logs/live.log")

_OPEN_SQL = """
INSERT INTO trades (magic, symbol, asset_class, side, volume_lots,
                    entry_time, entry_price, sl_initial, tp_initial,
                    quality, eff_quality, regime, sentiment, memory_n,
                    spread_pct_at_entry, atr_at_entry, notes)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
RETURNING id
"""

_CLOSE_SQL = """
UPDATE trades
SET exit_time = %s, exit_price = %s, pnl = %s, r_multiple = %s,
    exit_reason = %s,
    notes = notes || %s
WHERE id = (
    SELECT id FROM trades
    WHERE symbol = %s AND exit_time IS NULL AND notes LIKE %s
    ORDER BY entry_time DESC LIMIT 1
)
"""


def journal_open(*, magic: int, symbol: str, asset_class: str, side: str,
                 volume_lots: float, entry_time: datetime, entry_price: float,
                 sl: float, tp: float, quality: float, eff_quality: float,
                 regime: str, sentiment: float, memory_n: int,
                 spread_pct: float, atr: float, ticket: int,
                 dry_run: bool) -> Optional[int]:
    """Insert the entry row. Returns the row id, or None on failure."""
    notes = f"ticket:{ticket}" + (" | DRY_RUN" if dry_run else "")
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_OPEN_SQL, (
                    magic, symbol, asset_class, side, volume_lots,
                    entry_time, entry_price, sl, tp,
                    quality, eff_quality, regime, sentiment, memory_n,
                    spread_pct, atr, notes))
                row_id = cur.fetchone()[0]
        logger.info(f"journal open #{row_id}: {symbol} {side} {volume_lots} "
                    f"@ {entry_price} (ticket {ticket})")
        return row_id
    except Exception as exc:
        logger.error(f"journal_open FAILED for {symbol} ticket {ticket}: {exc}")
        return None


def journal_close(*, symbol: str, ticket: int, exit_time: datetime,
                  exit_price: float, pnl: float, r_multiple: float,
                  exit_reason: str, extra_note: str = "") -> bool:
    """Close the open row for this ticket. Returns True when a row was
    actually updated (False = row missing or DB error - the reconciler
    will retry on its next pass)."""
    note_suffix = f" | closed:{exit_reason}" + (f" | {extra_note}" if extra_note else "")
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_CLOSE_SQL, (
                    exit_time, exit_price, pnl, r_multiple, exit_reason,
                    note_suffix, symbol, f"ticket:{ticket}%"))
                updated = cur.rowcount
        if updated:
            logger.info(f"journal close: {symbol} ticket {ticket} "
                        f"{exit_reason} pnl={pnl:+.2f}")
        else:
            logger.warning(f"journal_close: no open row for {symbol} "
                           f"ticket {ticket} (already closed?)")
        return bool(updated)
    except Exception as exc:
        logger.error(f"journal_close FAILED for {symbol} ticket {ticket}: {exc}")
        return False
