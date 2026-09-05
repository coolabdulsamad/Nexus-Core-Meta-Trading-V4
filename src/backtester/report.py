"""Backtest metrics and reporting.

Turns lists of TradeResult objects into the numbers that decide whether
this brain has a tradeable edge: profit factor, expectancy, drawdown,
Sharpe, exit-reason anatomy, and the quality-bucket table that Phase 7
uses to re-derive the quality gates from THIS brain's distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trades_to_frame(trades: list) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({
            "symbol": t.symbol, "asset_class": getattr(t, "asset_class", ""),
            "side": t.side, "entry_time": t.entry_time,
            "exit_idx": t.exit_idx, "exit_reason": t.exit_reason,
            "pnl_usd": t.pnl_usd, "r_multiple": t.r_multiple,
            "bars_held": t.bars_held, "mae_atr": t.mae_atr, "mfe_atr": t.mfe_atr,
            "swap_paid": t.swap_paid, "commission_paid": t.commission_paid,
            "scale_outs": t.scale_outs, "quality": t.quality,
            "prob": t.prob, "agreement": t.agreement,
        })
    return pd.DataFrame(rows)


def perf_stats(df: pd.DataFrame, initial_capital: float) -> dict:
    if df.empty:
        return {"trades": 0}
    pnl = df["pnl_usd"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    eq = initial_capital + df.sort_values("entry_time")["pnl_usd"].cumsum()
    peak = eq.cummax()
    max_dd = float((peak - eq).max())
    max_dd_pct = float(((peak - eq) / peak).max() * 100) if len(eq) else 0.0

    # hourly-bar Sharpe approximation from per-trade returns on risk
    r = df["r_multiple"]
    sharpe_r = float(r.mean() / r.std() * np.sqrt(252 * 6)) if r.std() > 0 else 0.0

    return {
        "trades": int(len(df)),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(pf),
        "expectancy_usd": float(pnl.mean()),
        "expectancy_R": float(r.mean()),
        "avg_win_R": float(df.loc[pnl > 0, "r_multiple"].mean()) if len(wins) else 0.0,
        "avg_loss_R": float(df.loc[pnl < 0, "r_multiple"].mean()) if len(losses) else 0.0,
        "total_pnl_usd": float(pnl.sum()),
        "total_return_pct": float(pnl.sum() / initial_capital * 100),
        "max_drawdown_usd": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_per_trade_R": sharpe_r,
        "avg_bars_held": float(df["bars_held"].mean()),
        "swap_paid_total": float(df["swap_paid"].sum()),
        "commission_paid_total": float(df["commission_paid"].sum()),
    }


def quality_buckets(df: pd.DataFrame, n_buckets: int = 8) -> pd.DataFrame:
    """PF / expectancy per quality decile — the Phase 7 gate-calibration input."""
    if df.empty:
        return pd.DataFrame()
    q = df.copy()
    q["bucket"] = pd.qcut(q["quality"], q=min(n_buckets, q["quality"].nunique()),
                          duplicates="drop")
    out = q.groupby("bucket", observed=True).apply(
        lambda g: pd.Series({
            "trades": len(g),
            "win_rate": (g["pnl_usd"] > 0).mean(),
            "profit_factor": (g.loc[g.pnl_usd > 0, "pnl_usd"].sum()
                              / max(1e-9, -g.loc[g.pnl_usd < 0, "pnl_usd"].sum())),
            "expectancy_R": g["r_multiple"].mean(),
        }), include_groups=False).reset_index()
    return out


def exit_reason_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby("exit_reason").apply(
        lambda g: pd.Series({
            "trades": len(g),
            "share": len(g) / len(df),
            "win_rate": (g["pnl_usd"] > 0).mean(),
            "avg_R": g["r_multiple"].mean(),
            "total_pnl": g["pnl_usd"].sum(),
        }), include_groups=False).reset_index()


def side_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby("side").apply(
        lambda g: pd.Series({
            "trades": len(g),
            "win_rate": (g["pnl_usd"] > 0).mean(),
            "profit_factor": (g.loc[g.pnl_usd > 0, "pnl_usd"].sum()
                              / max(1e-9, -g.loc[g.pnl_usd < 0, "pnl_usd"].sum())),
            "avg_R": g["r_multiple"].mean(),
        }), include_groups=False).reset_index()
