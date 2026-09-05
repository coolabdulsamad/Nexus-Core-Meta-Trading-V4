#!/usr/bin/env python3
"""
Frozen-brain holdout test (post-Phase-4c, post-split-check).

The half-split check (oos_split_check.py) answered "is the XAUUSD l24_tight
edge stable across TIME?" — both halves of the 24-month window were
profitable. But two in-sample contaminations remain:

1. VARIANT SELECTION BIAS. l24_tight was crowned best of 18 variants x 38
   symbols on the FULL window — the same window the split check re-measured.
   Of course the winner looks good on the data that crowned it.
2. ENCODER/MEMORY LEAKAGE (documented in engine.py): the z-scaler that
   embeds every query and every memory point was fitted on FULL history,
   including the period under test. Unsupervised and second-order, but the
   future still shaped past decisions.

This script removes BOTH:

  Stage A — selection replay (zero Qdrant traffic):
    Re-rank all 18 variants using ONLY pre-cutoff cached verdicts. If
    l24_tight's promise is invisible without the holdout months, it was
    crowned BY those months — selection bias confirmed.

  Stage B — frozen brain (true holdout):
    Rebuild the class memory + encoder from states at/before the cutoff
    ONLY, into a SEPARATE collection (market_memory_60m_metal_holdout) and
    encoder (models/encoder_metal_holdout.pkl) — production memory
    untouched. Then collect verdicts for post-cutoff bars against that
    frozen brain and trade them. The brain never saw a bar, a scaling
    statistic, or an outcome from the holdout period.

Interpretation, honestly:
  - HOLDOUT PASS (PF > 1.0 on >= 30 trades the brain never saw) is
    necessary but NOT sufficient: still one symbol, still a few dozen
    trades. The next validation is forward (the DRY_RUN live engine),
    never a shortcut to live money.
  - HOLDOUT FAIL closes the XAUUSD question for good.

NOTE on Stage A alignment: cached verdicts are keyed by frame position and
the frame is anchored to "now" — run this soon after the sweep collect.
The script checks the frame length against the cache and skips Stage A for
a symbol on drift. Stage B computes positions fresh in-process; it cannot
misalign.

Usage (repo root, venv active, DB+Qdrant up):
  python scripts/holdout_check.py                          # XAUUSD, 6-month holdout
  python scripts/holdout_check.py --holdout-months 4
  python scripts/holdout_check.py --symbols XAUUSD,XAGUSD
  python scripts/holdout_check.py --rebuild                # force fresh frozen brain
  python scripts/holdout_check.py --cleanup                # delete holdout artifacts after
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client.models import (Distance, PayloadSchemaType, PointStruct,
                                  VectorParams)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import config                                    # noqa: E402
from scripts.sweep_configs import (VARIANTS, _cache_path,              # noqa: E402
                                   _load_cache, _window, config_overrides)
from src.backtester.engine import (SymbolBacktester,                   # noqa: E402
                                   candidate_positions, prepare_frame)
from src.backtester.report import perf_stats, trades_to_frame          # noqa: E402
from src.backtester.run_backtest import CLASS_POOLS, load_symbol_frame  # noqa: E402
from src.memory.build_memory import _point_id                          # noqa: E402
from src.memory.memory_store import get_qdrant                         # noqa: E402
from src.memory.vector_encoder import MODELS_DIR, VectorEncoder        # noqa: E402
from src.storage.db import get_conn                                    # noqa: E402

KEYS = ("trades", "win_rate", "profit_factor", "expectancy_R",
        "total_pnl_usd", "max_drawdown_pct")
MIN_TRADES = 30   # below this a profit factor is a coin flip wearing a suit
_BATCH = 2000


def holdout_collection(asset_class: str) -> str:
    return f"market_memory_{config.BAR_MINUTES}m_{asset_class}_holdout"


def holdout_encoder_path(asset_class: str) -> str:
    return os.path.join(MODELS_DIR, f"encoder_{asset_class}_holdout.pkl")


# ---------------------------------------------------------------------------
# Stage A: pre-cutoff variant selection replay (cached verdicts only)
# ---------------------------------------------------------------------------
def stage_a_selection_replay(loaded: list) -> pd.DataFrame:
    """Re-rank ALL variants on pre-cutoff data only.

    loaded: [(symbol, cls, df_pre, {horizon: {pos: verdict}})] — frames and
    verdict keys already truncated to the pre-cutoff prefix."""
    rows = []
    for name, overrides in VARIANTS.items():
        label = overrides.get("BRAIN_LABEL_HORIZON", config.BRAIN_LABEL_HORIZON)
        trades = []
        with config_overrides(overrides):
            for symbol, cls, df_pre, vbs_pre in loaded:
                verdicts = vbs_pre.get(label)
                if not verdicts:
                    continue
                bt = SymbolBacktester(symbol, cls, verdict_override=verdicts)
                trades.extend(bt.run(df_pre).trades)
        stats = perf_stats(trades_to_frame(trades),
                           config.BACKTEST_INITIAL_CAPITAL)
        rows.append({"variant": name, "label": label,
                     **{k: stats.get(k, 0) for k in KEYS}})
    table = pd.DataFrame(rows).sort_values(
        "profit_factor", ascending=False).reset_index(drop=True)
    ranks, r = {}, 0
    for _, row in table.iterrows():
        if row["trades"] >= MIN_TRADES:
            r += 1
            ranks[row["variant"]] = r
    table.insert(0, "rank", table["variant"].map(ranks))
    return table


# ---------------------------------------------------------------------------
# Stage B: frozen memory + encoder, states <= cutoff ONLY
# ---------------------------------------------------------------------------
def build_frozen_brain(asset_class: str, symbols: list[str],
                       cutoff: pd.Timestamp) -> None:
    feature_cols = ", ".join(VectorEncoder.VECTOR_FEATURES)
    sql = (f"SELECT symbol, time_bucket, {feature_cols}, forward_return_4h,"
           " forward_return_12h, forward_return_24h, regime_label"
           " FROM feature_cache_1h"
           " WHERE symbol = ANY(%s) AND forward_return_4h IS NOT NULL"
           " AND time_bucket <= %s"
           " ORDER BY time_bucket")
    with get_conn() as conn:
        df = pd.read_sql(sql, conn, params=(symbols, cutoff.to_pydatetime()))
    df = df.dropna(subset=list(VectorEncoder.VECTOR_FEATURES))
    print(f"[build] {asset_class}: {len(df)} states at/before "
          f"{cutoff:%Y-%m-%d %H:%M} UTC", flush=True)
    if len(df) < 500:
        raise SystemExit(f"[build] {asset_class}: only {len(df)} states — "
                         f"too few to freeze a brain")

    encoder = VectorEncoder()
    vectors = encoder.fit_transform(df)
    encoder.save(holdout_encoder_path(asset_class))

    client = get_qdrant()
    name = holdout_collection(asset_class)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=encoder.output_dim_,
                                    distance=Distance.COSINE))
    client.create_payload_index(collection_name=name, field_name="ts",
                                field_schema=PayloadSchemaType.INTEGER)

    ts_epoch = (pd.to_datetime(df["time_bucket"], utc=True)
                  .astype("int64").to_numpy() // 10**9)
    symbols_arr = df["symbol"].to_numpy()
    fwds = df["forward_return_4h"].astype(float).to_numpy()
    fwds_12 = pd.to_numeric(df["forward_return_12h"],
                            errors="coerce").to_numpy(dtype="float64")
    fwds_24 = pd.to_numeric(df["forward_return_24h"],
                            errors="coerce").to_numpy(dtype="float64")
    regimes = df["regime_label"].astype(str).to_numpy()

    written = 0
    batch: list[PointStruct] = []
    for i in range(len(df)):
        payload = {"symbol": str(symbols_arr[i]), "ts": int(ts_epoch[i]),
                   "fwd": float(fwds[i]), "regime": regimes[i]}
        if np.isfinite(fwds_12[i]):
            payload["fwd_12h"] = float(fwds_12[i])
        if np.isfinite(fwds_24[i]):
            payload["fwd_24h"] = float(fwds_24[i])
        batch.append(PointStruct(
            id=_point_id(symbols_arr[i], int(ts_epoch[i])),
            vector=vectors[i].tolist(), payload=payload))
        if len(batch) >= _BATCH:
            client.upsert(collection_name=name, points=batch)
            written += len(batch)
            batch = []
    if batch:
        client.upsert(collection_name=name, points=batch)
        written += len(batch)
    print(f"[build] {asset_class}: {written} points -> {name} "
          f"(dim={encoder.output_dim_})", flush=True)


def stage_b_frozen_holdout(symbol: str, cls: str, df: pd.DataFrame,
                           cutoff: pd.Timestamp, names: list[str]) -> dict:
    """Collect post-cutoff verdicts from the FROZEN brain, then trade them."""
    d, ts_idx = prepare_frame(df)
    encoder = VectorEncoder.load(holdout_encoder_path(cls))
    bt = SymbolBacktester(symbol, cls, encoder=encoder,
                          qdrant_client=get_qdrant())
    bt.collection = holdout_collection(cls)   # ask the frozen brain
    cands = candidate_positions(
        symbol, 0, len(d) - 1,
        d["atr_14"].to_numpy(), d["spread_price"].to_numpy(),
        d["spread_med20"].to_numpy(), ts_idx)
    cands = [p for p in cands if ts_idx[p] >= cutoff]
    horizons = tuple(config.BRAIN_LABEL_HORIZONS)
    t0 = time.time()
    if cands:
        by_horizon = bt._batch_verdicts(d, cands, label_horizons=horizons)
        if len(horizons) == 1:                     # flat return — normalize
            by_horizon = {horizons[0]: by_horizon}
    else:
        by_horizon = {h: {} for h in horizons}
    print(f"[holdout] {symbol}: {len(cands)} post-cutoff candidates, "
          f"frozen-brain verdicts ({', '.join(horizons)}) in "
          f"{time.time() - t0:.0f}s", flush=True)

    out = {}
    for name in names:
        overrides = VARIANTS[name]
        label = overrides.get("BRAIN_LABEL_HORIZON", config.BRAIN_LABEL_HORIZON)
        with config_overrides(overrides):
            replay = SymbolBacktester(
                symbol, cls, verdict_override=by_horizon.get(label, {}))
            res = replay.run(df)
        stats = perf_stats(trades_to_frame(res.trades),
                           config.BACKTEST_INITIAL_CAPITAL)
        out[name] = stats
        print(f"[holdout] {symbol} {name:<12} "
              f"trades={stats.get('trades', 0):>4}  "
              f"win={stats.get('win_rate', 0):.0%}  "
              f"PF={stats.get('profit_factor', 0):.2f}  "
              f"exp={stats.get('expectancy_R', 0):+.4f}R  "
              f"pnl=${stats.get('total_pnl_usd', 0):,.0f}  "
              f"maxDD={stats.get('max_drawdown_pct', 0):.1f}%", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="frozen-brain holdout test")
    ap.add_argument("--symbols", default="XAUUSD",
                    help="comma-separated (default XAUUSD)")
    ap.add_argument("--variants", default="l24_tight,baseline",
                    help="comma-separated variant names for the holdout replay")
    ap.add_argument("--months", type=int, default=24,
                    help="total window — must match the sweep cache")
    ap.add_argument("--holdout-months", type=int, default=6,
                    help="trailing months the frozen brain never sees")
    ap.add_argument("--rebuild", action="store_true",
                    help="force-rebuild the frozen memory/encoder")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete holdout collections + encoders at the end")
    args = ap.parse_args()

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        ap.error(f"unknown variants: {unknown} (choices: {list(VARIANTS)})")

    sym_class = {s: cls for cls, pool in CLASS_POOLS.items() for s in pool}
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start, end = _window(args.months)
    cutoff = end - pd.DateOffset(months=args.holdout_months)

    print("=== frozen-brain holdout test ===")
    print(f"window {start:%Y-%m-%d} -> {end:%Y-%m-%d} | "
          f"cutoff {cutoff:%Y-%m-%d} "
          f"({args.holdout_months} months the frozen brain never sees)")
    print("stage A removes variant-selection bias | "
          "stage B removes encoder/memory leakage\n")

    # ---- load frames; align stage A caches -------------------------------
    loaded_a, frames = [], {}
    with get_conn() as conn:
        for symbol in symbols:
            cls = sym_class.get(symbol, "forex")
            df = load_symbol_frame(conn, symbol, start, end)
            if len(df) < 500:
                print(f"[skip] {symbol}: only {len(df)} rows")
                continue
            frames[symbol] = (cls, df)
            _, ts_idx = prepare_frame(df)
            pre_len = int((ts_idx < cutoff).sum())
            if pre_len < 500:
                print(f"[stage A] {symbol}: only {pre_len} pre-cutoff rows, "
                      f"skipping stage A")
                continue
            path = _cache_path(symbol, args.months)
            if not path.exists():
                print(f"[stage A] {symbol}: no sweep cache — skipped "
                      f"(run sweep_configs.py --collect first)")
                continue
            with open(path, "rb") as fh:
                cached_n = pickle.load(fh).get("n_bars")
            if cached_n and cached_n != len(df):
                print(f"[stage A] {symbol}: frame drifted since collect "
                      f"({len(df)} rows vs cached {cached_n}) — verdict "
                      f"positions would misalign; skipping stage A "
                      f"(re-collect, or run this the same day as collect)")
                continue
            vbs_pre = {h: {p: v for p, v in vs.items() if p < pre_len}
                       for h, vs in _load_cache(path).items()}
            loaded_a.append((symbol, cls, df.iloc[:pre_len], vbs_pre))
            print(f"[stage A] {symbol}: {pre_len} pre-cutoff rows, "
                  f"cache aligned ({cached_n} bars)")

    # ---- stage A ----------------------------------------------------------
    print("\n--- stage A: pre-cutoff variant selection replay ---")
    if loaded_a:
        table = stage_a_selection_replay(loaded_a)
        print(f"{'rank':>4} {'variant':<15} {'label':<6} {'trades':>6} "
              f"{'win':>5} {'PF':>6} {'expR':>9}")
        for _, r in table.iterrows():
            rank = str(int(r["rank"])) if pd.notna(r["rank"]) else "-"
            print(f"{rank:>4} {r['variant']:<15} {r['label']:<6} "
                  f"{int(r['trades']):>6} {r['win_rate']:>4.0%} "
                  f"{r['profit_factor']:>6.2f} {r['expectancy_R']:>+9.4f}")
        tgt = table[table["variant"] == "l24_tight"].iloc[0]
        n_ranked = int(table["rank"].notna().sum())
        if pd.notna(tgt["rank"]):
            print(f"\nl24_tight pre-cutoff: rank {int(tgt['rank'])}/"
                  f"{n_ranked} (PF {tgt['profit_factor']:.2f}, "
                  f"{int(tgt['trades'])} trades)")
        else:
            print(f"\nl24_tight pre-cutoff: INELIGIBLE "
                  f"({int(tgt['trades'])} trades < {MIN_TRADES}) — it would "
                  f"never have been picked without the holdout months")
    else:
        print("(no aligned caches — stage A skipped)")

    # ---- stage B ----------------------------------------------------------
    print("\n--- stage B: frozen-brain holdout ---")
    client = get_qdrant()
    needed = sorted({cls for cls, _ in frames.values()})
    for cls in needed:
        have = (client.collection_exists(holdout_collection(cls))
                and os.path.exists(holdout_encoder_path(cls)))
        if args.rebuild or not have:
            build_frozen_brain(cls, CLASS_POOLS.get(cls, []), cutoff)
        else:
            print(f"[build] {cls}: frozen brain exists "
                  f"(use --rebuild to refresh)")

    results = {}
    for symbol, (cls, df) in frames.items():
        results[symbol] = stage_b_frozen_holdout(symbol, cls, df, cutoff, names)

    # ---- verdict ----------------------------------------------------------
    print("\n=== verdicts ===")
    for symbol, by_variant in results.items():
        for name, stats in by_variant.items():
            n_tr = stats.get("trades", 0)
            pf = stats.get("profit_factor", 0)
            ok = n_tr >= MIN_TRADES and pf > 1.0
            print(f"  {symbol:<8} {name:<12} "
                  f"{'HOLDOUT PASS' if ok else 'HOLDOUT FAIL'} "
                  f"(trades={n_tr}, PF={pf:.2f})")
    print("\nA PASS is one symbol and a few dozen trades on data the brain "
          "never saw —\nit justifies a focused research phase and the "
          "DRY_RUN forward test, never live money.")

    if args.cleanup:
        for cls in needed:
            cname = holdout_collection(cls)
            if client.collection_exists(cname):
                client.delete_collection(cname)
                print(f"[cleanup] deleted collection {cname}")
            epath = holdout_encoder_path(cls)
            if os.path.exists(epath):
                os.remove(epath)
                print(f"[cleanup] deleted {epath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
