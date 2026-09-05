#!/usr/bin/env python3
"""
Phase 5 acceptance: demo smoke test for the live engine.

Run on the Windows machine with the MT5 terminal OPEN and logged into the
demo account (from the repo root, venv active):

    python scripts/demo_smoke.py                # full check, ZERO real orders
    python scripts/demo_smoke.py --real-order   # + ONE minimum-size real
                                                #   order opened & closed
                                                #   (needs DRY_RUN=false)

The default mode never sends an order: the lifecycle check runs through
the REAL engine code (state -> exit stack -> close workflow -> journal)
with a virtual DRY_RUN position, so every layer is exercised with zero
broker writes. Two journal rows appear (flagged "DRY_RUN") - that is the
journal doing its job, not an error.

Exit code 0 = every check passed.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import config          # noqa: E402

_RESULTS = []


def check(name: str, ok: bool, detail: str = "", warn: bool = False) -> bool:
    tag = "WARN" if (warn and not ok) else ("PASS" if ok else "FAIL")
    _RESULTS.append(tag != "FAIL")
    print(f"[{tag}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 demo smoke test")
    ap.add_argument("--real-order", action="store_true",
                    help="place and close ONE minimum-size real order "
                         "(requires DRY_RUN=false in .env)")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Nexus V4 Phase 5 smoke test | DRY_RUN={config.DRY_RUN}")
    print("=" * 70)

    # ---- 1. MT5 connection + account ------------------------------------
    from src.mt5_client.connector import MT5Connector
    connector = MT5Connector()
    try:
        connected = connector.connect()
    except Exception as exc:
        check("MT5 connect", False, str(exc))
        return _finish()
    check("MT5 connect", connected)
    if not connected:
        return _finish()
    account = connector.account()
    check("account info", account is not None,
          f"login={account.get('login')} server={account.get('server')} "
          f"equity={account.get('equity'):.2f} {account.get('currency')} "
          f"mode={'hedging' if account.get('hedging') else 'netting'}")
    check("terminal Algo Trading enabled",
          bool(account.get("trade_allowed")),
          "orders will be REJECTED until the toolbar button is green "
          "(DRY_RUN checks below still work)", warn=True)

    # ---- 2. universe discovery -------------------------------------------
    universe = connector.discover_universe()
    check("universe discovery", len(universe) >= 30,
          f"{len(universe)} pool symbols resolve at this broker")

    # ---- 3. encoders + Qdrant memory --------------------------------------
    from src.memory.vector_encoder import VectorEncoder
    for cls in ("forex", "metal", "crypto"):
        path = Path(VectorEncoder.path_for(cls))
        ok = path.exists()
        check(f"encoder [{cls}]", ok,
              str(path) if ok else f"{path} missing - run "
              f"python -m src.memory.build_memory first")
    from src.memory.memory_store import collection_size, get_qdrant
    try:
        qc = get_qdrant()
        for cls in ("forex", "metal", "crypto"):
            n = collection_size(qc, cls)
            check(f"memory [{cls}]", n > 0, f"{n:,} points")
    except Exception as exc:
        check("Qdrant reachable", False, str(exc))

    # ---- 4. DB + journal table ---------------------------------------------
    from src.storage.db import get_conn
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM trades")
                n_trades = cur.fetchone()[0]
        check("trades journal table", True, f"{n_trades} rows")
    except Exception as exc:
        check("trades journal table", False, str(exc))

    # ---- 5. brain answers on live bars -------------------------------------
    from src.ingestion.indicator_calculator import calculate_all_indicators
    from src.live.live_trader import LiveTrader
    from src.memory.meta_learner import Brain
    brains = {}
    for symbol in ("EURUSD", "XAUUSD", "BTCUSD"):
        if symbol not in universe:
            check(f"brain live read [{symbol}]", False, "not in universe",
                  warn=True)
            continue
        try:
            bars = connector.get_bars(symbol, config.LIVE_BARS_WINDOW)
            specs = connector.symbol_specs(symbol)
            feats = calculate_all_indicators(
                bars.iloc[:-1], point=specs["point"] if specs else None)
            cls = universe[symbol]["asset_class"]
            if cls not in brains:
                brains[cls] = Brain(cls)
            v = brains[cls].predict(feats.iloc[-1])
            # reaching this line without an exception IS the pass condition;
            # None ("no neighbors") is a valid brain answer, not a failure
            check(f"brain live read [{symbol}]", True,
                  f"prob={v['prob']:.3f} q={v['quality']:.2f} "
                  f"n={v['n_kept']}" if v else "no neighbors (valid HOLD)")
        except Exception as exc:
            check(f"brain live read [{symbol}]", False, str(exc))

    # ---- 6. virtual lifecycle through the REAL engine code -----------------
    # A DRY_RUN position opened 20h ago must hit the time-stop this cycle:
    # exercises evaluate() -> CLOSE_ALL -> estimate_pnl -> finalize_close
    # -> journal open+close -> state removal. Zero broker writes.
    print("-" * 70)
    print("virtual lifecycle (DRY_RUN code path, real engine):")
    lifecycle_ok = False
    tmp_state = Path(tempfile.gettempdir()) / "nexus_v4_smoke_state.json"
    if tmp_state.exists():
        tmp_state.unlink()
    saved_path = config.LIVE_STATE_PATH
    config.LIVE_STATE_PATH = str(tmp_state)
    try:
        trader = LiveTrader()
        trader.universe = universe
        # pick the tightest-spread liquid symbol for the virtual trade
        cands = [s for s in ("EURUSD", "XAUUSD", "BTCUSD") if s in universe]
        symbol = min(cands, key=lambda s: connector.spread_pct(s) or 1e9)
        cls = universe[symbol]["asset_class"]
        atr = trader._atr_for(symbol)
        tick = connector.get_tick(symbol)
        specs = connector.symbol_specs(symbol)
        entry_ref = float(tick.ask)
        from src.live.position_manager import ManagedPosition
        from src.live.state import iso, now_utc, save_state
        from src.live.journal import journal_open
        now = now_utc()
        stop_dist = config.STOP_ATR_MULT * atr
        lots = connector.volume_for_risk(symbol, 50.0, stop_dist) or \
            (specs["volume_min"] if specs else 0.01)
        ticket = -900001
        pos = ManagedPosition(
            ticket=ticket, symbol=symbol, asset_class=cls, side="LONG",
            entry_price=entry_ref,
            entry_time=iso(now - timedelta(hours=20)),   # forces time_stop
            initial_volume=lots, volume=lots, atr=atr,
            sl=entry_ref - 50 * atr, tp=entry_ref + 50 * atr,  # never touch
            peak_price=entry_ref, trough_price=entry_ref,
            quality=0.40, prob=0.55, agreement=0.60, memory_n=100,
            regime="smoke", risk_usd=50.0, dry_run=True)
        trader.state["positions"][str(ticket)] = pos.to_dict()
        save_state(config.LIVE_STATE_PATH, trader.state)
        row_id = journal_open(
            magic=connector.magic_for(symbol), symbol=symbol, asset_class=cls,
            side="LONG", volume_lots=lots, entry_time=now - timedelta(hours=20),
            entry_price=entry_ref, sl=pos.sl, tp=pos.tp, quality=0.40,
            eff_quality=0.40, regime="smoke", sentiment=0.0, memory_n=100,
            spread_pct=0.0, atr=atr, ticket=ticket, dry_run=True)
        check("journal open (virtual)", row_id is not None, f"row #{row_id}")

        trader._manage(now)     # the real management path: time_stop fires
        gone = str(ticket) not in trader.state["positions"]
        check("exit stack closes stale position", gone,
              "time_stop fired through evaluate()->finalize_close()")

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT exit_reason, pnl IS NOT NULL FROM trades "
                    "WHERE notes LIKE %s ORDER BY id DESC LIMIT 1",
                    (f"ticket:{ticket}%",))
                row = cur.fetchone()
        check("journal close (virtual)",
              bool(row) and row[0] == "time_stop" and row[1],
              f"exit_reason={row[0] if row else 'MISSING'}")
        lifecycle_ok = gone and bool(row)
    except Exception as exc:
        check("virtual lifecycle", False, repr(exc))
    finally:
        config.LIVE_STATE_PATH = saved_path
        if tmp_state.exists():
            tmp_state.unlink()

    # ---- 7. optional: ONE real order round-trip -----------------------------
    if args.real_order:
        print("-" * 70)
        print("real-order round trip (explicit --real-order):")
        if config.DRY_RUN:
            check("real order", False,
                  "DRY_RUN is still true in .env - set DRY_RUN=false first")
        elif not account.get("trade_allowed"):
            check("real order", False,
                  "terminal Algo Trading is disabled (toolbar button)")
        else:
            try:
                symbol = min((s for s in ("EURUSD", "XAUUSD", "BTCUSD")
                              if s in universe),
                             key=lambda s: connector.spread_pct(s) or 1e9)
                specs = connector.symbol_specs(symbol)
                bars = connector.get_bars(symbol, config.LIVE_BARS_WINDOW)
                feats = calculate_all_indicators(
                    bars.iloc[:-1], point=specs["point"])
                atr = float(feats["atr_14"].iloc[-1])
                if not (atr == atr and atr > 0):
                    atr = specs["point"] * 100      # fallback: 100 points
                tick = connector.get_tick(symbol)
                entry_ref = float(tick.ask)
                stop_dist = config.STOP_ATR_MULT * atr
                lots = specs["volume_min"]
                res = connector.place_market_order(
                    symbol, "LONG", lots,
                    sl=entry_ref - stop_dist, tp=entry_ref + stop_dist,
                    comment="smoke-real")
                check("real order placed", res is not None,
                      f"ticket {res['ticket']} @ {res['price']}" if res else "")
                if res:
                    time.sleep(5)
                    ok = connector.close_position(symbol, ticket=res["ticket"],
                                                  comment="smoke-close")
                    summary = connector.closed_position_summary(res["ticket"])
                    check("real order closed", ok,
                          f"pnl {summary['profit']:+.2f}" if summary else "")
            except Exception as exc:
                check("real order round trip", False, repr(exc))

    return _finish()


def _finish() -> int:
    print("=" * 70)
    if all(_RESULTS):
        print("SMOKE TEST: ALL CHECKS PASSED (warnings, if any, are listed above)")
        return 0
    print("SMOKE TEST: FAILURES PRESENT - fix the FAIL lines above and re-run")
    return 1


if __name__ == "__main__":
    sys.exit(main())
