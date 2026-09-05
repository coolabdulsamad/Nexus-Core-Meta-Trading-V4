"""
Single entry point for reading bars back OUT of TimescaleDB.

Everything downstream (feature pump, brain memory build, backtester,
audits) loads bars through here so the column contract lives in one
place: timestamp (UTC), open, high, low, close, volume, spread_points.
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.storage.db import get_conn


def load_bars(
    symbol: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    conn=None,
) -> pd.DataFrame:
    """Load H1 bars for one canonical symbol, oldest first."""
    sql = (
        "SELECT time_bucket AS timestamp, open, high, low, close, volume, "
        "spread_points FROM market_data_1h WHERE symbol = %s"
    )
    params: list = [symbol]
    if start is not None:
        sql += " AND time_bucket >= %s"
        params.append(start)
    if end is not None:
        sql += " AND time_bucket <= %s"
        params.append(end)
    sql += " ORDER BY time_bucket ASC"

    if conn is not None:
        return pd.read_sql(sql, conn, params=params)
    with get_conn() as owned:
        return pd.read_sql(sql, owned, params=params)


def load_all_bars(symbols: List[str], **kwargs) -> pd.DataFrame:
    """Concatenated frame with a `symbol` column (brain memory build)."""
    frames = []
    with get_conn() as conn:
        for sym in symbols:
            df = load_bars(sym, conn=conn, **kwargs)
            if not df.empty:
                df["symbol"] = sym
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def last_bar_time(symbol: str, conn=None) -> Optional[datetime]:
    """Most recent bar we hold for the symbol (None if never backfilled)."""
    sql = "SELECT MAX(time_bucket) FROM market_data_1h WHERE symbol = %s"
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol,))
            return cur.fetchone()[0]
    with get_conn() as owned:
        with owned.cursor() as cur:
            cur.execute(sql, (symbol,))
            return cur.fetchone()[0]


def symbols_in_db(conn=None) -> List[str]:
    """Canonical symbols that have any bars stored."""
    sql = "SELECT DISTINCT symbol FROM market_data_1h ORDER BY symbol"
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [r[0] for r in cur.fetchall()]
    with get_conn() as owned:
        with owned.cursor() as cur:
            cur.execute(sql)
            return [r[0] for r in cur.fetchall()]
