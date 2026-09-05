#!/usr/bin/env python3
"""
Phase 4b — parameter sweep over CACHED brain verdicts.

The expensive part of a backtest is asking the brain (Qdrant) for a verdict
at every candidate bar. Those verdicts do NOT depend on the trading config
(stops, horizons, gates) — only on the brain and the data. So we:

  --collect : query the brain ONCE per symbol at every candidate bar and
              cache the verdicts to reports/verdict_cache/*.pkl
              (slow — same order of time as the Phase 4 backtest;
               resumable: symbols with an existing cache are skipped)
  --eval    : replay many trading-config variants against the cached
              verdicts (fast — seconds per variant, ZERO Qdrant traffic)

Sweepable knobs are the gates DOWNSTREAM of candidacy (conviction, quality,
ADX, exits, sizing, horizon) plus the brain LABEL horizon (4h/12h/24h) once
the memory stores multi-horizon labels (Phase 4c). Session/spread
prefilters decide WHICH bars get a verdict at collect time, so variants
cannot change those.

The cache stores verdicts per label horizon; collect is incremental — it
only queries horizons missing from the cache (a 4b-era cache counts as
having "4h"). Each horizon is queried with its OWN look-ahead guard
((label_bars + 1) hours), so 24h verdicts never see fresher-than-knowable
neighbors.

Usage (from repo root, venv active, DB+Qdrant up):
  python scripts/sweep_configs.py --collect --cls forex --months 24
  python scripts/sweep_configs.py --collect --months 24        (rest)
  python scripts/sweep_configs.py --eval --months 24
  python scripts/sweep_configs.py --eval --variants baseline,horizon6
"""
from __future__ import annotations

import argparse
import contextlib
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import config
from src.backtester.engine import SymbolBacktester, prepare_frame, candidate_positions
from src.backtester.report import trades_to_frame, perf_stats, exit_reason_table
from src.backtester.run_backtest import CLASS_POOLS, FEATURE_SQL, load_symbol_frame
from src.memory.memory_store import get_qdrant
from src.memory.vector_encoder import VectorEncoder
from src.storage.db import get_conn

CACHE_DIR = ROOT / "reports" / "verdict_cache"
RESULTS_CSV = ROOT / "reports" / "sweep_results.csv"

# ---------------------------------------------------------------------------
# Variants: name -> {config attribute overrides}
# ---------------------------------------------------------------------------
VARIANTS = {
    "baseline": {},
    # horizon mismatch: brain predicts 4h ahead; holding 16 bars dilutes it
    "horizon6": {"TIME_LIMIT_BARS": 6, "TIME_PARTIAL_BARS": 4},
    "horizon4": {"TIME_LIMIT_BARS": 4, "TIME_PARTIAL_BARS": 3},
    # exits: no scale-outs / ratchet / trailing / retracement / time-partial
    "no_early_exits": {
        "SCALE_OUT_ENABLED": False,
        "ENABLE_PROFIT_DRAWDOWN_PROTECTION": False,
        "ENABLE_TIME_PARTIAL": False,
    },
    "wide_tp": {"REWARD_RISK_RATIO": 2.5},
    "tight_stop": {"STOP_ATR_MULT": 1.5},
    # stricter gates
    "conviction_hi": {"ENTRY_CONVICTION_MARGIN": 0.04},
    "quality_055": {"MIN_SIGNAL_QUALITY": 0.55, "CRYPTO_MIN_SIGNAL_QUALITY": 0.45},
    # combos
    "h6_conviction": {"TIME_LIMIT_BARS": 6, "TIME_PARTIAL_BARS": 4,
                      "ENTRY_CONVICTION_MARGIN": 0.04},
    "h6_noearly": {"TIME_LIMIT_BARS": 6, "TIME_PARTIAL_BARS": 4,
                   "SCALE_OUT_ENABLED": False,
                   "ENABLE_PROFIT_DRAWDOWN_PROTECTION": False,
                   "ENABLE_TIME_PARTIAL": False},
    "h4_wide_tp": {"TIME_LIMIT_BARS": 4, "TIME_PARTIAL_BARS": 3,
                   "REWARD_RISK_RATIO": 2.5},
    "h6_q055": {"TIME_LIMIT_BARS": 6, "TIME_PARTIAL_BARS": 4,
                "MIN_SIGNAL_QUALITY": 0.55, "CRYPTO_MIN_SIGNAL_QUALITY": 0.45},
    # ---- Phase 4c: label-horizon axis ----------------------------------
    # The 4b sweep showed longer holds lose LESS (16 > 6 > 4 bars), i.e.
    # the brain's 4h label may simply be the wrong horizon. These variants
    # replay the SAME bars with verdicts computed from 12h / 24h outcomes.
    # Needs a multi-horizon memory + collect (see SETUP 11b/11c).
    "label12h": {"BRAIN_LABEL_HORIZON": "12h"},
    "label24h": {"BRAIN_LABEL_HORIZON": "24h"},
    # label/hold alignment: hold roughly as long as the label predicts
    "l12_h12": {"BRAIN_LABEL_HORIZON": "12h",
                "TIME_LIMIT_BARS": 12, "TIME_PARTIAL_BARS": 8},
    "l24_h24": {"BRAIN_LABEL_HORIZON": "24h",
                "TIME_LIMIT_BARS": 24, "TIME_PARTIAL_BARS": 16},
    # label x best 4b variant
    "l12_tight": {"BRAIN_LABEL_HORIZON": "12h", "STOP_ATR_MULT": 1.5},
    "l24_tight": {"BRAIN_LABEL_HORIZON": "24h", "STOP_ATR_MULT": 1.5},
}

_STAT_KEYS = ("trades", "win_rate", "profit_factor", "expectancy_R",
              "total_pnl_usd", "total_return_pct", "max_drawdown_pct")


def _cache_path(symbol: str, months: int) -> Path:
    return CACHE_DIR / f"{symbol}_{months}m.pkl"


def _window(months: int):
    end = pd.Timestamp.now(tz="UTC")
    return end - pd.DateOffset(months=months), end


def _build_pairs(args) -> list[tuple[str, str]]:
    """[(symbol, asset_class)] exactly like run_backtest resolves them."""
    sym_class = {s: cls for cls, pool in CLASS_POOLS.items() for s in pool}
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        return [(s, sym_class.get(s, "forex")) for s in symbols]
    pools = CLASS_POOLS if args.cls == "all" else {args.cls: CLASS_POOLS[args.cls]}
    return [(s, cls) for cls, pool in pools.items() for s in pool]


def _load_cache(path: Path) -> dict:
    """Cache file -> {horizon: {pos: verdict}}. Back-compatible with the
    Phase 4b single-horizon format ({"verdicts": ...} == 4h)."""
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    if "verdicts_by_horizon" in blob:
        return dict(blob["verdicts_by_horizon"])
    if "verdicts" in blob:           # 4b format: 4h only
        return {"4h": blob["verdicts"]}
    return {}


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------
def collect(pairs, months: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    start, end = _window(months)
    horizons = tuple(config.BRAIN_LABEL_HORIZONS)
    qc = get_qdrant()
    encoders = {}
    with get_conn() as conn:
        for k, (symbol, cls) in enumerate(pairs, 1):
            out = _cache_path(symbol, months)
            existing = _load_cache(out) if out.exists() else {}
            missing = [h for h in horizons if h not in existing]
            if not missing:
                print(f"[collect {k}/{len(pairs)}] {symbol}: cache complete "
                      f"({', '.join(horizons)}), skipping", flush=True)
                continue
            df = load_symbol_frame(conn, symbol, start, end)
            if len(df) < 500:
                print(f"[collect {k}/{len(pairs)}] {symbol}: only {len(df)} rows, skipping",
                      flush=True)
                continue
            if cls not in encoders:
                encoders[cls] = VectorEncoder.load(VectorEncoder.path_for(cls))
            d, ts_idx = prepare_frame(df)
            bt = SymbolBacktester(symbol, cls, encoder=encoders[cls], qdrant_client=qc)
            # match the engine: candidates never include the final bar
            cands = candidate_positions(
                symbol, 0, len(d) - 1,
                d["atr_14"].to_numpy(),
                d["spread_price"].to_numpy(),
                d["spread_med20"].to_numpy(),
                ts_idx,
            )
            t0 = time.time()
            if cands:
                result = bt._batch_verdicts(d, cands, label_horizons=tuple(missing))
                # single missing horizon comes back flat; normalize to nested
                if len(missing) == 1:
                    result = {missing[0]: result}
            else:
                result = {h: {} for h in missing}
            merged = {**existing, **result}
            with open(out, "wb") as fh:
                pickle.dump({"symbol": symbol, "asset_class": cls,
                             "months": months, "n_bars": len(d),
                             "verdicts_by_horizon": merged}, fh)
            counts = ", ".join(f"{h}:{len(merged.get(h, {}))}" for h in horizons)
            print(f"[collect {k}/{len(pairs)}] {symbol}: {len(cands)} candidates, "
                  f"queried {','.join(missing)} in {time.time() - t0:.0f}s ({counts})",
                  flush=True)
    print(f"[collect] done -> {CACHE_DIR}")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def config_overrides(overrides: dict):
    """Temporarily patch the shared config object (all modules read config.X
    at runtime, so this one object is the only thing to patch)."""
    saved = {k: getattr(config, k) for k in overrides}
    for k, v in overrides.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def evaluate(pairs, months: int, variant_names) -> pd.DataFrame:
    start, end = _window(months)
    loaded = []  # (symbol, cls, df, {horizon: {pos: verdict}})
    with get_conn() as conn:
        for symbol, cls in pairs:
            path = _cache_path(symbol, months)
            if not path.exists():
                print(f"[eval] {symbol}: no cache (run --collect first), skipping",
                      flush=True)
                continue
            df = load_symbol_frame(conn, symbol, start, end)
            if len(df) < 500:
                print(f"[eval] {symbol}: only {len(df)} rows, skipping", flush=True)
                continue
            vbs = _load_cache(path)
            if not vbs:
                print(f"[eval] {symbol}: cache unreadable, skipping", flush=True)
                continue
            loaded.append((symbol, cls, df, vbs))
    if not loaded:
        print("[eval] nothing cached — nothing to do")
        return pd.DataFrame()
    horizons_avail = sorted({h for *_, vbs in loaded for h in vbs})
    print(f"[eval] {len(loaded)} symbols cached; horizons available: "
          f"{', '.join(horizons_avail)}")

    rows = []
    details = {}
    for name in variant_names:
        overrides = VARIANTS[name]
        label = overrides.get("BRAIN_LABEL_HORIZON", config.BRAIN_LABEL_HORIZON)
        t0 = time.time()
        results = []
        skipped = 0
        with config_overrides(overrides):
            for symbol, cls, df, vbs in loaded:
                verdicts = vbs.get(label)
                if verdicts is None:
                    skipped += 1
                    continue
                bt = SymbolBacktester(symbol, cls, verdict_override=verdicts)
                res = bt.run(df)
                for tr in res.trades:
                    tr.asset_class = cls
                results.append(res)
        if skipped:
            print(f"[eval] {name}: {skipped} symbols lack the '{label}' label "
                  f"(run --collect with the new code first)", flush=True)
        all_trades = [t for r in results for t in r.trades]
        df_tr = trades_to_frame(all_trades)
        stats = perf_stats(df_tr, config.BACKTEST_INITIAL_CAPITAL)
        row = {"variant": name}
        row.update({k: stats.get(k, 0) for k in _STAT_KEYS})
        rows.append(row)
        details[name] = (df_tr, stats, results)
        print(f"[eval] {name:<15} trades={row['trades']:>5}  "
              f"PF={row['profit_factor']:.3f}  exp={row['expectancy_R']:+.4f}R  "
              f"ret={row['total_return_pct']:+.1f}%  ({time.time() - t0:.0f}s)",
              flush=True)

    table = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {RESULTS_CSV}")

    # per-symbol detail for the best variant by profit factor
    valid = table[table["trades"] >= 30]
    best_name = (valid.sort_values("profit_factor", ascending=False).iloc[0]["variant"]
                 if not valid.empty else table.iloc[0]["variant"])
    df_best, stats_best, results_best = details[best_name]
    print(f"\n=== Best variant: {best_name} ===")
    print(f"PF={stats_best.get('profit_factor', 0):.3f}  "
          f"win={stats_best.get('win_rate', 0):.1%}  "
          f"exp={stats_best.get('expectancy_R', 0):+.4f}R  "
          f"maxDD={stats_best.get('max_drawdown_pct', 0):.1f}%")
    if not df_best.empty:
        print("\nPer class:")
        for cls, g in df_best.groupby("asset_class"):
            s = perf_stats(g, config.BACKTEST_INITIAL_CAPITAL)
            print(f"  {cls:<7} trades={s['trades']:<5} PF={s['profit_factor']:.2f} "
                  f"win={s['win_rate']:.1%} exp={s['expectancy_R']:+.3f}R "
                  f"pnl=${s['total_pnl_usd']:,.0f}")
    per_symbol = []
    for r in results_best:
        d = trades_to_frame(r.trades)
        s = perf_stats(d, config.BACKTEST_INITIAL_CAPITAL)
        per_symbol.append({"symbol": r.symbol, "trades": s.get("trades", 0),
                           "profit_factor": s.get("profit_factor", 0),
                           "expectancy_R": s.get("expectancy_R", 0),
                           "total_pnl_usd": s.get("total_pnl_usd", 0)})
    ps = pd.DataFrame(per_symbol).sort_values("profit_factor", ascending=False)
    ps_path = RESULTS_CSV.parent / f"sweep_best_{best_name}_per_symbol.csv"
    ps.to_csv(ps_path, index=False)
    print(f"\n{ps.to_string(index=False)}")
    print(f"\nSaved {ps_path}")
    er = exit_reason_table(df_best)
    print("\nExit reasons (best variant):")
    print(er.to_string(index=False) if not er.empty else "  (no trades)")
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4b cached-verdict parameter sweep")
    ap.add_argument("--collect", action="store_true",
                    help="query brain and cache verdicts (slow, once, resumable)")
    ap.add_argument("--eval", action="store_true",
                    help="replay config variants on cached verdicts (fast)")
    ap.add_argument("--symbols", default="", help="comma-separated; default = class pools")
    ap.add_argument("--cls", "--class", dest="cls",
                    choices=list(CLASS_POOLS) + ["all"], default="all")
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--variants", default="",
                    help="comma-separated variant names; default = all")
    args = ap.parse_args()

    if not args.collect and not args.eval:
        ap.error("pass --collect and/or --eval")

    pairs = _build_pairs(args)
    if args.collect:
        collect(pairs, args.months)
    if args.eval:
        names = [v.strip() for v in args.variants.split(",") if v.strip()] or list(VARIANTS)
        unknown = [n for n in names if n not in VARIANTS]
        if unknown:
            ap.error(f"unknown variants: {unknown} (choices: {list(VARIANTS)})")
        evaluate(pairs, args.months, names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
