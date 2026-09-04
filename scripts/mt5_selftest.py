"""
scripts/mt5_selftest.py
================================================================
Phase-1 acceptance test. Run on the Windows machine that hosts the
MT5 terminal, from the project root:

    python scripts/mt5_selftest.py

Checks, in order:
  1. MetaTrader5 package importable (Windows only)
  2. Terminal connection + account login
  3. Account snapshot (balance/equity/currency, netting vs hedging)
  4. Universe discovery: which of our configured symbols exist at
     this broker, and under which broker-side name (suffixes etc.)
  5. Per-symbol specs sanity: digits, tick value, volume step,
     stops level, swap, live spread
  6. Bar download sanity: 300 H1 bars per resolved symbol
  7. Indicator engine smoke test on one symbol

Exit code 0 = all critical checks passed. Warnings never fail the run.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mt5_client.connector import MT5Connector, MT5_AVAILABLE  # noqa: E402
from config.settings import config                                # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results = []


def report(status, check, detail=""):
    results.append((status, check, detail))
    line = f"[{status}] {check}"
    if detail:
        line += f"  ->  {detail}"
    print(line)


def main():
    print("=" * 72)
    print("Nexus Core Meta Trading V4 — MT5 self-test")
    print("=" * 72)

    # --- 1. package -------------------------------------------------
    if not MT5_AVAILABLE:
        report(FAIL, "MetaTrader5 package import",
               "not available — install on Windows: pip install MetaTrader5")
        return _summary(1)
    report(PASS, "MetaTrader5 package import")

    # --- 2. connect --------------------------------------------------
    conn = MT5Connector()
    if not conn.connect():
        report(FAIL, "terminal connection",
               "is the MT5 terminal running and logged in?")
        return _summary(1)
    report(PASS, "terminal connection")

    # --- 3. account --------------------------------------------------
    acct = conn.account()
    if not acct:
        report(FAIL, "account info")
        return _summary(1)
    report(PASS, "account info",
           f"login={acct['login']} {acct['currency']} "
           f"balance={acct['balance']:.2f} equity={acct['equity']:.2f} "
           f"mode={'HEDGING' if acct['hedging'] else 'NETTING'}")
    if acct["currency"] != "USD":
        report(WARN, "account currency",
               f"{acct['currency']} — P&L math works, but risk % is computed "
               f"in account currency")
    if acct.get("trade_allowed") is False:
        report(WARN, "algo trading",
               "terminal reports algo trading DISABLED — enable the "
               "'Algo Trading' button before going live")

    # --- 4. universe discovery ---------------------------------------
    universe = conn.discover_universe()
    pools = {
        "forex": config.FOREX_POOL,
        "metal": config.METALS_POOL,
        "crypto": config.CRYPTO_POOL,
    }
    resolved = {}
    for klass, pool in pools.items():
        found, missing = [], []
        for sym in pool:
            if sym in universe:
                found.append(f"{sym}({universe[sym]['broker_symbol']})")
                resolved[sym] = universe[sym]
            else:
                missing.append(sym)
        status = PASS if found else FAIL
        report(status, f"universe[{klass}]",
               f"{len(found)}/{len(pool)} available: {', '.join(found[:8])}"
               + (" ..." if len(found) > 8 else ""))
        if missing:
            report(WARN, f"universe[{klass}] missing",
                   ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))

    if not resolved:
        report(FAIL, "universe", "no configured symbol exists at this broker")
        return _summary(1)

    # --- 5+6. specs & bars per resolved symbol ------------------------
    worst_spread = []
    for sym in sorted(resolved):
        specs = conn.symbol_specs(sym)
        if not specs:
            report(WARN, f"specs[{sym}]", "symbol_specs failed")
            continue
        bars = conn.get_bars(sym, count=300)
        n = 0 if bars is None else len(bars)
        if n < 200:
            report(WARN, f"bars[{sym}]",
                   f"only {n} H1 bars — broker history short; "
                   f"backfill will fetch by date range")
        sp = conn.spread_pct(sym)
        worst_spread.append((sp or 0, sym))
        base = (f"{n} bars | digits={specs['digits']} "
                f"tick_val={specs['tick_value']:.2f} "
                f"vol_step={specs['volume_step']} "
                f"stops_lvl={specs['stops_level_points']} "
                f"swap(L/S)={specs['swap_long']:.2f}/{specs['swap_short']:.2f}")
        report(PASS, f"symbol[{sym}]",
               base + (f" spread={sp*1e4:.1f}bp" if sp else " spread n/a"))
        time.sleep(0.05)   # be polite to the terminal API

    if worst_spread:
        worst = sorted(worst_spread, reverse=True)[:3]
        report(WARN, "widest spreads right now",
               ", ".join(f"{s} {v*1e4:.1f}bp" for v, s in worst))

    # --- 7. indicator smoke test --------------------------------------
    try:
        from src.ingestion.indicator_calculator import calculate_all_indicators
        probe = next(iter(sorted(resolved)))
        df = conn.get_bars(probe, count=300)
        point = conn.symbol_specs(probe)["point"]
        feats = calculate_all_indicators(df, point=point)
        need = ["rsi_14", "atr_14", "adx_14", "atr_pct", "dist_vwap",
                "hour_sin", "dow_sin", "spread_pct", "regime_label"]
        missing = [c for c in need if c not in feats.columns]
        if missing:
            report(FAIL, "indicator engine", f"missing columns: {missing}")
        else:
            last = feats.iloc[-1]
            report(PASS, "indicator engine",
                   f"{probe}: rsi={last['rsi_14']:.1f} adx={last['adx_14']:.1f} "
                   f"atr%={last['atr_pct']:.4f} regime={last['regime_label']}")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, "indicator engine", repr(exc))

    conn.shutdown()
    return _summary(0)


def _summary(code):
    print("-" * 72)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    n_pass = sum(1 for s, _, _ in results if s == PASS)
    print(f"self-test complete: {n_pass} passed, {n_warn} warnings, "
          f"{n_fail} failed")
    if n_fail:
        print("Fix the FAIL items above before moving to Phase 2.")
        code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
