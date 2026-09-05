"""
Data-quality audit for the V4 data layer (DB-only - runs anywhere).

Per symbol with stored bars:
- coverage: first/last bar, row count, completeness vs expected
  weekday hours (forex/metals trade ~Mon-Fri; crypto trades 24/7)
- gaps: weekday holes longer than 3 consecutive hours
- outliers: bars whose range exceeds 8x ATR(14) (bad ticks)
- staleness: last bar older than 3 days
- spread regime: p50/p75/p95/max of spread_pct from feature_cache_1h,
  written to reports/spread_stats.json -> the Phase-4 backtester's
  cost model uses these per-symbol distributions.

Usage:
    python scripts/data_audit.py
    python scripts/data_audit.py --min-completeness 0.995
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.db import get_conn  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger("audit", "logs/audit.log")

CRYPTO_24_7 = {"BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD",
               "LTCUSD", "BCHUSD", "ADAUSD", "DOGEUSD"}


def _expected_hours(first: pd.Timestamp, last: pd.Timestamp, crypto: bool) -> int:
    """Expected bar count between first and last (inclusive)."""
    hours = pd.date_range(first, last, freq="h", tz="UTC")
    if crypto:
        return len(hours)
    # forex/metals: weekday hours only, minus the Friday-21:00 -> Sunday-21:00
    # close is approximated by weekday filtering (close enough for an audit)
    return int((hours.dayofweek < 5).sum())


def audit_symbol(conn, symbol: str) -> Dict:
    bars = pd.read_sql(
        "SELECT time_bucket AS timestamp, open, high, low, close, volume "
        "FROM market_data_1h WHERE symbol = %s ORDER BY time_bucket",
        conn, params=(symbol,),
    )
    if bars.empty:
        return {"symbol": symbol, "status": "EMPTY"}

    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    first, last = bars["timestamp"].iloc[0], bars["timestamp"].iloc[-1]
    crypto = symbol in CRYPTO_24_7

    expected = _expected_hours(first, last, crypto)
    completeness = len(bars) / expected if expected else 1.0

    # --- gaps: holes > 3h inside trading time ---
    diffs = bars["timestamp"].diff().dt.total_seconds().div(3600)
    gap_mask = diffs > 3
    if not crypto:
        # ignore the natural weekend hole (Fri->Mon ~ 72h)
        gap_mask &= bars["timestamp"].dt.dayofweek != 6  # hole ENDING on Sunday
        gap_mask &= ~((bars["timestamp"].dt.dayofweek == 0)
                      & (diffs <= 72))                   # normal Mon open
    gaps = int(gap_mask.sum())

    # --- outliers: range > 8x ATR(14) ---
    prev_close = bars["close"].shift(1)
    tr = pd.concat([(bars["high"] - bars["low"]),
                    (bars["high"] - prev_close).abs(),
                    (bars["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    outliers = int(((bars["high"] - bars["low"]) > 8 * atr).sum())

    # --- staleness ---
    age_h = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 3600
    stale = bool(age_h > 72)

    # --- spread distribution from the feature cache (scale-free) ---
    spread = pd.read_sql(
        "SELECT spread_pct FROM feature_cache_1h "
        "WHERE symbol = %s AND spread_pct IS NOT NULL AND spread_pct > 0",
        conn, params=(symbol,),
    )["spread_pct"].astype(float)
    spread_stats = {}
    if not spread.empty:
        spread_stats = {
            "p50": float(spread.quantile(0.50)),
            "p75": float(spread.quantile(0.75)),
            "p95": float(spread.quantile(0.95)),
            "max": float(spread.max()),
            "n": int(len(spread)),
        }

    return {
        "symbol": symbol,
        "status": "STALE" if stale else "OK",
        "first": str(first), "last": str(last),
        "bars": int(len(bars)), "expected": int(expected),
        "completeness": round(float(completeness), 4),
        "gaps_over_3h": gaps, "outlier_bars": outliers,
        "age_hours": round(age_h, 1),
        "spread_pct": spread_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V4 data-quality audit")
    parser.add_argument("--min-completeness", type=float, default=0.95)
    args = parser.parse_args()

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM market_data_1h ORDER BY symbol")
            symbols = [r[0] for r in cur.fetchall()]
        if not symbols:
            logger.error("market_data_1h is empty - run backfill_history first")
            return 1
        reports = [audit_symbol(conn, s) for s in symbols]

    # --- console table ---
    hdr = (f"{'symbol':<9} {'bars':>7} {'compl':>6} {'gaps':>5} "
           f"{'outl':>5} {'age_h':>7}  status")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    problems: List[str] = []
    spread_out: Dict[str, Dict] = {}
    for r in reports:
        if r["status"] == "EMPTY":
            print(f"{r['symbol']:<9} EMPTY")
            problems.append(r["symbol"])
            continue
        print(f"{r['symbol']:<9} {r['bars']:>7} {r['completeness']:>6.1%} "
              f"{r['gaps_over_3h']:>5} {r['outlier_bars']:>5} "
              f"{r['age_hours']:>7}  {r['status']}")
        if (r["completeness"] < args.min_completeness
                or r["status"] != "OK" or r["gaps_over_3h"] > 10):
            problems.append(r["symbol"])
        if r["spread_pct"]:
            spread_out[r["symbol"]] = r["spread_pct"]

    # --- spread stats for the Phase-4 cost model ---
    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "spread_stats.json")
    with open(out_path, "w") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(),
                   "symbols": spread_out}, f, indent=2)
    logger.info("Spread stats for %d symbols -> %s", len(spread_out), out_path)

    ok = len(reports) - len(problems)
    print(f"\nAudit: {ok}/{len(reports)} symbols clean "
          f"(min completeness {args.min_completeness:.1%})")
    if problems:
        print("Attention needed: " + ", ".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
