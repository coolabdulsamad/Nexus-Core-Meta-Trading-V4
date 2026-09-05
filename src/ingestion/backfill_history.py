"""
Historical H1 backfill: MT5 -> market_data_1h.

Walks backwards from "now" in ~monthly chunks using copy_rates_range
(the only reliable way to reach deep history on MT5) and upserts
OHLCV + spread into TimescaleDB. Idempotent: safe to re-run; the
ON CONFLICT clause makes repeats a no-op.

Usage (on the Windows MT5 machine):
    python -m src.ingestion.backfill_history                # all universe symbols, 3 years
    python -m src.ingestion.backfill_history --years 5
    python -m src.ingestion.backfill_history --symbols EURUSD XAUUSD
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from config.settings import config
from src.utils.logger import setup_logger
from src.mt5_client.connector import MT5Connector
from src.storage.db import get_conn, execute_values_upsert

logger = setup_logger("backfill", "logs/backfill.log")

MARKET_DATA_COLUMNS = (
    "symbol", "time_bucket", "open", "high", "low", "close",
    "volume", "spread_points",
)


def _chunk_ranges(start: datetime, end: datetime) -> List[tuple]:
    """Month-sized (chunk_start, chunk_end) windows, oldest first."""
    ranges = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=31), end)
        ranges.append((cur, nxt))
        cur = nxt
    return ranges


def bars_to_rows(df, canonical: str) -> List[tuple]:
    """DataFrame from MT5Connector.get_bars_range -> DB rows."""
    return [
        (
            canonical,
            r.timestamp.to_pydatetime(),
            float(r.open), float(r.high), float(r.low), float(r.close),
            float(r.volume), int(r.spread_points),
        )
        for r in df.itertuples()
    ]


def backfill_symbol(
    client: MT5Connector,
    conn,
    canonical: str,
    start: datetime,
    end: datetime,
) -> int:
    """Backfill one symbol. Returns total rows written."""
    total = 0
    for chunk_start, chunk_end in _chunk_ranges(start, end):
        df = client.get_bars_range(canonical, chunk_start, chunk_end)
        if df is None or df.empty:
            continue
        total += execute_values_upsert(
            conn, "market_data_1h", MARKET_DATA_COLUMNS,
            bars_to_rows(df, canonical),
        )
    logger.info("Backfilled %s: %d bars", canonical, total)
    return total


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill H1 history from MT5")
    parser.add_argument("--years", type=float, default=3.0,
                        help="history depth (default 3)")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="canonical symbols (default: all enabled pools)")
    args = parser.parse_args(argv)

    symbols = args.symbols or (
        config.FOREX_POOL + config.METALS_POOL
        + config.CRYPTO_POOL + config.INDICES_POOL
    )

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(args.years * 365.25))
    logger.info("Backfill: %d symbols, %s -> %s", len(symbols),
                start.date(), end.date())

    client = MT5Connector()
    client.connect()

    grand_total = 0
    failed = []
    with get_conn() as conn:
        for sym in symbols:
            try:
                grand_total += backfill_symbol(client, conn, sym, start, end)
            except Exception as exc:  # keep going; report at the end
                logger.error("Backfill failed for %s: %s", sym, exc)
                failed.append(sym)

    logger.info("Backfill complete: %d bars total across %d symbols",
                grand_total, len(symbols) - len(failed))
    if failed:
        logger.warning("Failed symbols: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
