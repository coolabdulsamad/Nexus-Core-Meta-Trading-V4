"""
Hourly data pump: MT5 -> market_data_1h -> feature_cache_1h.

Per universe symbol, each run:
1. TOP UP raw bars since the newest stored bar (auto-backfills 3y when
   the symbol has no history yet).
2. RECOMPUTE features on the symbol's FULL history window - never on a
   slice. (The Alpaca edition's v3.2 bug: indicators computed on a short
   slice produce different warm-up values than on the full frame, which
   silently killed 100% of entries. Full-frame recompute is the fix.)
3. UPSERT feature_cache_1h, including forward_return_1h/4h. The newest
   rows have NULL outcomes; later runs fill them in once the future has
   happened - this is also the brain's outcome-sync mechanism.
4. REFRESH the symbols registry (canonical <-> broker name, asset class).

Usage (on the Windows MT5 machine):
    python -m src.ingestion.run_pump                 # all universe symbols
    python -m src.ingestion.run_pump --symbols EURUSD GOLD...
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import config
from src.utils.logger import setup_logger
from src.mt5_client.connector import MT5Connector
from src.storage.db import get_conn, execute_values_upsert
from src.ingestion.loader import load_bars, last_bar_time
from src.ingestion.backfill_history import (
    MARKET_DATA_COLUMNS, bars_to_rows, backfill_symbol,
)
from src.ingestion.indicator_calculator import calculate_all_indicators

logger = setup_logger("pump", "logs/pump.log")

# feature_cache_1h columns the pump writes (sentiment_score stays NULL -
# the live trader overlays real-time sentiment, history has none).
FEATURE_COLUMNS = (
    "symbol", "time_bucket",
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_pct_b", "bb_width",
    "atr_14", "atr_pct", "volume_profile_ratio", "vol_z",
    "ret_1", "ret_3", "ret_12", "adx_14",
    "dist_sma50", "dist_sma200", "dist_vwap",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "spread_pct", "forward_return_1h", "forward_return_4h",
    "regime_label",
)


def add_forward_returns(features: pd.DataFrame) -> pd.DataFrame:
    """
    Prediction target: close-to-close return N bars ahead. Bar-count
    based (a 4-bar shift over a weekend spans more wall time) - this
    matches the brain's training definition exactly, which is what
    matters for consistency.
    """
    close = features["close"]
    features["forward_return_1h"] = close.shift(-1) / close - 1.0
    features["forward_return_4h"] = (
        close.shift(-config.FORWARD_HORIZON_HOURS) / close - 1.0
    )
    return features


def features_to_rows(symbol: str, features: pd.DataFrame) -> List[tuple]:
    """Feature DataFrame -> DB rows; NaN -> None, ordered by FEATURE_COLUMNS."""
    frame = features.replace([np.inf, -np.inf], np.nan)
    rows = []
    for r in frame.itertuples():
        values: Dict[str, object] = {"symbol": symbol}
        for col in FEATURE_COLUMNS:
            if col in ("symbol",):
                continue
            if col == "time_bucket":
                values[col] = r.timestamp.to_pydatetime()
                continue
            v = getattr(r, col, None)
            if col == "regime_label":
                values[col] = str(v) if v is not None else "unknown"
            else:
                values[col] = None if v is None or pd.isna(v) else float(v)
        rows.append(tuple(values[c] for c in FEATURE_COLUMNS))
    return rows


def top_up_symbol(client: MT5Connector, conn, symbol: str,
                  point_cache: Dict[str, float]) -> int:
    """Insert bars newer than what we hold. Returns rows written."""
    last = last_bar_time(symbol, conn=conn)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    if last is None:
        logger.info("%s: no history - running full %sy backfill first",
                    symbol, config.HISTORY_YEARS)
        start = now - timedelta(days=int(config.HISTORY_YEARS * 365.25))
        return backfill_symbol(client, conn, symbol, start, now)

    gap_hours = int((now - last).total_seconds() // 3600)
    if gap_hours <= 0:
        return 0
    # +2 headroom; cap defensively (a symbol offline for weeks gets a
    # ranged fetch instead of a giant count).
    if gap_hours <= 500:
        df = client.get_bars(symbol, count=gap_hours + 2)
        if df is None or df.empty:
            return 0
        df = df[df["timestamp"] > pd.Timestamp(last)]
    else:
        df = client.get_bars_range(symbol, last, now)
        if df is None or df.empty:
            return 0
        df = df[df["timestamp"] > pd.Timestamp(last)]
    if df.empty:
        return 0
    return execute_values_upsert(
        conn, "market_data_1h", MARKET_DATA_COLUMNS, bars_to_rows(df, symbol),
    )


def refresh_features(conn, symbol: str, point: Optional[float]) -> int:
    """Full-window recompute + upsert. Returns feature rows written."""
    bars = load_bars(symbol, conn=conn)
    if bars.empty:
        return 0
    features = calculate_all_indicators(bars, point=point)
    if features.empty:
        return 0
    features = add_forward_returns(features)
    return execute_values_upsert(
        conn, "feature_cache_1h", FEATURE_COLUMNS,
        features_to_rows(symbol, features),
    )


def refresh_symbols_registry(client: MT5Connector, conn, symbols: List[str]) -> None:
    """Seed/update the canonical <-> broker mapping table."""
    rows = []
    for sym in symbols:
        broker = client.resolve_symbol(sym)
        if not broker:
            continue
        rows.append((sym, broker, client.classify_asset(sym), True))
    if not rows:
        return
    with conn.cursor() as cur:
        psycopg2_upsert_symbols(cur, rows)
    logger.info("Symbols registry: %d verified mappings", len(rows))


def psycopg2_upsert_symbols(cur, rows: List[Tuple]) -> None:
    from psycopg2.extras import execute_values
    execute_values(
        cur,
        """
        INSERT INTO symbols (symbol, broker_symbol, asset_class, verified_broker)
        VALUES %s
        ON CONFLICT (symbol, asset_class) DO UPDATE SET
            broker_symbol = EXCLUDED.broker_symbol,
            verified_broker = EXCLUDED.verified_broker
        """,
        rows,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hourly MT5 -> DB feature pump")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="canonical symbols (default: full pools)")
    args = parser.parse_args(argv)

    symbols = args.symbols or (
        config.FOREX_POOL + config.METALS_POOL
        + config.CRYPTO_POOL + config.INDICES_POOL
    )

    client = MT5Connector()
    client.connect()

    point_cache: Dict[str, float] = {}
    failures: List[str] = []
    t0 = datetime.now(timezone.utc)

    with get_conn() as conn:
        refresh_symbols_registry(client, conn, symbols)
        for sym in symbols:
            try:
                specs = client.symbol_specs(sym)
                point = specs["point"] if specs else None
                point_cache[sym] = point
                new_bars = top_up_symbol(client, conn, sym, point_cache)
                n_feat = refresh_features(conn, sym, point)
                logger.info("Pump %s: +%d bars, %d feature rows",
                            sym, new_bars, n_feat)
            except Exception as exc:
                logger.error("Pump failed for %s: %s", sym, exc)
                failures.append(sym)

    dt = (datetime.now(timezone.utc) - t0).total_seconds()
    logger.info("Pump complete in %.1fs: %d/%d symbols OK",
                dt, len(symbols) - len(failures), len(symbols))
    if failures:
        logger.warning("Failed: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
