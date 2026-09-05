"""Phase 4 backtest driver.

Usage:
    python -m src.backtester.run_backtest                      # all symbols, 24 months
    python -m src.backtester.run_backtest --symbols EURUSD BTCUSD
    python -m src.backtester.run_backtest --months 12 --class forex

Reads bars+features from TimescaleDB, runs the brain-in-the-loop engine,
and writes:
    reports/backtest_trades.csv    every simulated trade
    reports/backtest_summary.json  headline stats, funnels, per-symbol table
Prints the decision funnel and per-class / per-bucket performance so the
quality gates can be re-derived from this brain's own distribution.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")

from config.settings import config
from src.backtester.engine import Funnel, SymbolBacktester
from src.backtester.report import (exit_reason_table, perf_stats,
                                   quality_buckets, side_table,
                                   trades_to_frame)
from src.memory.memory_store import get_qdrant
from src.memory.vector_encoder import VectorEncoder
from src.storage.db import get_conn
from src.utils.logger import setup_logger

logger = setup_logger("backtest.run", "logs/backtest.log")

CLASS_POOLS = {
    "forex": list(config.FOREX_POOL),
    "metal": list(config.METALS_POOL),
    "crypto": list(config.CRYPTO_POOL),
}

FEATURE_SQL = """
SELECT f.time_bucket AS timestamp, m.open, m.high, m.low, m.close,
       f.atr_14, f.adx_14, f.spread_pct,
       f.rsi_14, f.bb_pct_b, f.bb_width, f.atr_pct, f.volume_profile_ratio,
       f.vol_z, f.ret_1, f.ret_3, f.ret_12, f.dist_sma50, f.dist_sma200,
       f.dist_vwap, f.hour_sin, f.hour_cos, f.dow_sin, f.dow_cos
FROM feature_cache_1h f
JOIN market_data_1h m
  ON m.symbol = f.symbol AND m.time_bucket = f.time_bucket
WHERE f.symbol = %s AND f.time_bucket >= %s AND f.time_bucket <= %s
ORDER BY f.time_bucket
"""


def load_symbol_frame(conn, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return pd.read_sql(FEATURE_SQL, conn, params=(symbol, start, end),
                       parse_dates=["timestamp"])


def print_funnel(name: str, f: Funnel):
    d = f.as_dict()
    print(f"\n  Funnel [{name}]")
    stages = ["bars", "flat_bars", "cooldown_skips", "session_ok", "spread_ok",
              "brain_called", "prob_directional", "agreement_ok", "quality_ok",
              "adx_ok", "tape_confirm_ok", "no_chase_ok", "tp_worth_spread",
              "entries"]
    for s in stages:
        print(f"    {s:<18} {d[s]:>10,}")


def main():
    p = argparse.ArgumentParser(description="Nexus V4 honest backtester")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--class", dest="asset_class", default=None,
                   choices=["forex", "metal", "crypto"])
    p.add_argument("--months", type=int, default=24,
                   help="trailing window to test (default 24)")
    p.add_argument("--out", default="reports")
    args = p.parse_args()

    symbols = args.symbols
    if symbols is None:
        symbols = []
        for cls, pool in CLASS_POOLS.items():
            if args.asset_class and cls != args.asset_class:
                continue
            symbols.extend(pool)
    sym_class = {s: cls for cls, pool in CLASS_POOLS.items() for s in pool}
    if args.symbols:
        sym_class.update({s: sym_class.get(s, "forex") for s in args.symbols})

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.DateOffset(months=args.months)
    print(f"Backtest window: {start.date()} -> {end.date()}  ({len(symbols)} symbols)")

    qc = get_qdrant()
    encoders = {}

    all_trades = []
    funnels: dict[str, Funnel] = {}
    t0 = time.time()

    with get_conn() as conn:
        for k, sym in enumerate(symbols, 1):
            cls = sym_class.get(sym, "forex")
            if cls not in encoders:
                encoders[cls] = VectorEncoder.load(VectorEncoder.path_for(cls))
            df = load_symbol_frame(conn, sym, start, end)
            if len(df) < 500:
                print(f"[{k}/{len(symbols)}] {sym}: only {len(df)} rows - skipped")
                continue
            print(f"[{k}/{len(symbols)}] {sym} ({cls}) {len(df):,} bars ...",
                  end=" ", flush=True)
            bt = SymbolBacktester(sym, cls, encoders[cls], qc)
            res = bt.run(df)
            for t in res.trades:
                t.asset_class = cls
            all_trades.extend(res.trades)
            funnels[sym] = res.funnel
            print(f"{len(res.trades)} trades")

    os.makedirs(args.out, exist_ok=True)
    df_tr = trades_to_frame(all_trades)
    trades_path = os.path.join(args.out, "backtest_trades.csv")
    if not df_tr.empty:
        df_tr.to_csv(trades_path, index=False)

    print(f"\n{'='*70}\nBACKTEST COMPLETE in {time.time()-t0:.0f}s\n{'='*70}")

    stats = perf_stats(df_tr, config.BACKTEST_INITIAL_CAPITAL)
    print("\n-- HEADLINE ----------------------------------------------------")
    for k_, v_ in stats.items():
        print(f"  {k_:<22} {v_:,.3f}" if isinstance(v_, float) else f"  {k_:<22} {v_:,}")

    print("\n-- PER CLASS ----------------------------------------------------")
    if not df_tr.empty:
        for cls, g in df_tr.groupby("asset_class"):
            s = perf_stats(g, config.BACKTEST_INITIAL_CAPITAL)
            print(f"  {cls:<7} trades={s['trades']:<5} PF={s.get('profit_factor', 0):.2f} "
                  f"win={s.get('win_rate', 0):.1%} exp={s.get('expectancy_R', 0):+.3f}R "
                  f"pnl=${s.get('total_pnl_usd', 0):,.0f}")

    print("\n-- QUALITY BUCKETS (gate calibration input) -------------------")
    qb = quality_buckets(df_tr)
    print(qb.to_string(index=False) if not qb.empty else "  (no trades)")

    print("\n-- EXIT REASONS -------------------------------------------------")
    er = exit_reason_table(df_tr)
    print(er.to_string(index=False) if not er.empty else "  (no trades)")

    print("\n-- SIDES --------------------------------------------------------")
    st = side_table(df_tr)
    print(st.to_string(index=False) if not st.empty else "  (no trades)")

    total_funnel = Funnel()
    for f in funnels.values():
        for k_, v_ in f.as_dict().items():
            setattr(total_funnel, k_, getattr(total_funnel, k_) + v_)
    print_funnel("TOTAL", total_funnel)

    per_symbol = []
    for sym, f in funnels.items():
        g = df_tr[df_tr["symbol"] == sym] if not df_tr.empty else pd.DataFrame()
        s = perf_stats(g, config.BACKTEST_INITIAL_CAPITAL)
        per_symbol.append({"symbol": sym, "asset_class": sym_class.get(sym, "forex"),
                           "stats": s, "funnel": f.as_dict()})

    summary = {
        "window": {"start": str(start.date()), "end": str(end.date()),
                   "months": args.months},
        "initial_capital": config.BACKTEST_INITIAL_CAPITAL,
        "headline": stats,
        "per_symbol": per_symbol,
        "quality_buckets": qb.to_dict("records") if not qb.empty else [],
        "exit_reasons": er.to_dict("records") if not er.empty else [],
        "notes": [
            "Decisions on bar close; fills at next bar open + full costs.",
            "Spread: real per-bar stored spread, paid at ask-crossing side.",
            "Swap: per-class notional %/day, tripled Wednesday (fx/metal).",
            "Same-bar SL/TP ambiguity resolves to the stop (conservative).",
            "Portfolio caps not simulated: trade counts are an upper bound.",
            "Encoder z-scaler fitted on full history (second-order leakage).",
        ],
    }
    summary_path = os.path.join(args.out, "backtest_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\nWrote {trades_path} and {summary_path}")


if __name__ == "__main__":
    main()
