#!/usr/bin/env python3
"""
Out-of-sample stability check (post-Phase-4c).

Phase 4c left exactly one bright spot: XAUUSD, PF 1.23 over 230 trades
(l24_tight variant) - and it was also positive in the independent 4b sweep
(1.18 / 222). Before anyone utters the word "profit" again, that edge must
survive a TIME split: replay the same variant on the FIRST half vs the
SECOND half of the cached 24-month window. An edge that only existed in
one year is not an edge; it is a memory of one.

Method (reuses the sweep machinery, ZERO new Qdrant traffic):
- load the symbol's cached verdicts + DB frame (same window as the sweep)
- split the frame positionally at the midpoint
- run the engine TWICE on the full frame: once with only first-half
  verdicts visible, once with only second-half verdicts visible
- a trade is counted in the half where it ENTERED (its exit may spill a
  few bars past the midpoint - documented, immaterial at 1h bars)

Usage (repo root, venv active, DB up):
  python scripts/oos_split_check.py                          # all cached symbols
  python scripts/oos_split_check.py --symbols XAUUSD
  python scripts/oos_split_check.py --variants l24_tight,baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import config                              # noqa: E402
from scripts.sweep_configs import (VARIANTS, _cache_path,        # noqa: E402
                                   _load_cache, _window, config_overrides)
from src.backtester.engine import SymbolBacktester               # noqa: E402
from src.backtester.report import perf_stats, trades_to_frame    # noqa: E402
from src.backtester.run_backtest import CLASS_POOLS, load_symbol_frame  # noqa: E402
from src.storage.db import get_conn                              # noqa: E402


def half_verdicts(verdicts: dict, split_pos: int, half: str) -> dict:
    first = half == "first"
    return {p: v for p, v in verdicts.items() if (p < split_pos) == first}


def main() -> int:
    ap = argparse.ArgumentParser(description="out-of-sample half-split check")
    ap.add_argument("--symbols", default="",
                    help="comma-separated; default = every cached symbol")
    ap.add_argument("--variants", default="l24_tight",
                    help="comma-separated variant names (default l24_tight)")
    ap.add_argument("--months", type=int, default=24)
    args = ap.parse_args()

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        ap.error(f"unknown variants: {unknown} (choices: {list(VARIANTS)})")

    sym_class = {s: cls for cls, pool in CLASS_POOLS.items() for s in pool}
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        cache_dir = _cache_path("XAUUSD", args.months).parent
        symbols = sorted(p.name.split("_")[0]
                         for p in cache_dir.glob(f"*_{args.months}m.pkl"))
    if not symbols:
        print("no cached symbols found - run sweep_configs.py --collect first")
        return 1

    start, end = _window(args.months)
    loaded = []
    with get_conn() as conn:
        for symbol in symbols:
            path = _cache_path(symbol, args.months)
            if not path.exists():
                print(f"[skip] {symbol}: no cache")
                continue
            df = load_symbol_frame(conn, symbol, start, end)
            if len(df) < 500:
                print(f"[skip] {symbol}: only {len(df)} rows")
                continue
            loaded.append((symbol, sym_class.get(symbol, "forex"), df,
                           _load_cache(path)))
    print(f"loaded {len(loaded)} symbols "
          f"({start.date()} -> {end.date()}, split at midpoint)\n")

    for name in names:
        overrides = VARIANTS[name]
        label = overrides.get("BRAIN_LABEL_HORIZON", config.BRAIN_LABEL_HORIZON)
        print(f"=== variant: {name} (label {label}) ===")
        print(f"{'symbol':<8} {'half':<7} {'trades':>6} {'win':>6} "
              f"{'PF':>7} {'expR':>8} {'pnl$':>10}")
        rows = []
        with config_overrides(overrides):
            for symbol, cls, df, vbs in loaded:
                verdicts = vbs.get(label)
                if not verdicts:
                    print(f"{symbol:<8} (no '{label}' label cached)")
                    continue
                split = len(df) // 2
                rec = {"symbol": symbol}
                for half in ("first", "second"):
                    bt = SymbolBacktester(
                        symbol, cls,
                        verdict_override=half_verdicts(verdicts, split, half))
                    res = bt.run(df)
                    stats = perf_stats(trades_to_frame(res.trades),
                                       config.BACKTEST_INITIAL_CAPITAL)
                    rec[half] = stats
                    print(f"{symbol:<8} {half:<7} {stats['trades']:>6} "
                          f"{stats['win_rate']:>5.0%} "
                          f"{stats['profit_factor']:>7.2f} "
                          f"{stats['expectancy_R']:>+8.4f} "
                          f"{stats['total_pnl_usd']:>10,.0f}")
                rows.append(rec)
        # verdict line per symbol
        print("\nstability verdicts:")
        for rec in rows:
            f, s = rec.get("first"), rec.get("second")
            if not f or not s or f["trades"] < 20 or s["trades"] < 20:
                print(f"  {rec['symbol']:<8} INCONCLUSIVE "
                      f"(too few trades in a half)")
                continue
            stable = f["profit_factor"] > 1.0 and s["profit_factor"] > 1.0
            print(f"  {rec['symbol']:<8} "
                  f"{'STABLE  ' if stable else 'UNSTABLE'} "
                  f"(PF {f['profit_factor']:.2f} -> {s['profit_factor']:.2f})")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
