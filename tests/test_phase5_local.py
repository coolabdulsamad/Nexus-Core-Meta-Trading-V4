"""Phase 5 local test — the whole live stack, fully offline.

Stubs MetaTrader5 / qdrant_client / psycopg2 / pythonjsonlogger (sandbox
has no MT5, no DB, no PyPI). Run: python tests/test_phase5_local.py
"""
import os, sys, types, logging, json, tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/mnt/agents/work/mt5v4")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- stubs ---
if "pythonjsonlogger" not in sys.modules:
    pkg = types.ModuleType("pythonjsonlogger")
    sub = types.ModuleType("pythonjsonlogger.jsonlogger")
    class JsonFormatter(logging.Formatter): pass
    sub.JsonFormatter = JsonFormatter
    pkg.jsonlogger = sub
    sys.modules["pythonjsonlogger"] = pkg
    sys.modules["pythonjsonlogger.jsonlogger"] = sub

if "qdrant_client" not in sys.modules:
    qc_pkg = types.ModuleType("qdrant_client")
    qc_models = types.ModuleType("qdrant_client.models")
    class _Bag:
        def __init__(self, **kw): self.__dict__.update(kw)
    for name in ("FieldCondition", "Filter", "QueryRequest", "Range",
                 "Distance", "PayloadSchemaType", "PointStruct", "VectorParams"):
        setattr(qc_models, name, type(name, (_Bag,), {}))
    class QdrantClient:
        def __init__(self, **kw): pass
    qc_pkg.QdrantClient = QdrantClient
    qc_pkg.models = qc_models
    sys.modules["qdrant_client"] = qc_pkg
    sys.modules["qdrant_client.models"] = qc_models

if "psycopg2" not in sys.modules:
    pg = types.ModuleType("psycopg2")
    pg_extras = types.ModuleType("psycopg2.extras")
    pg.extras = pg_extras
    def _no_connect(*a, **k): raise RuntimeError("stub: no DB in sandbox")
    pg.connect = _no_connect
    sys.modules["psycopg2"] = pg
    sys.modules["psycopg2.extras"] = pg_extras

if "MetaTrader5" not in sys.modules:
    mt5 = types.ModuleType("MetaTrader5")
    mt5.TIMEFRAME_H1 = 16385; mt5.TIMEFRAME_M5 = 5
    mt5.POSITION_TYPE_BUY = 0; mt5.POSITION_TYPE_SELL = 1
    mt5.ORDER_TYPE_BUY = 0; mt5.ORDER_TYPE_SELL = 1
    mt5.TRADE_ACTION_DEAL = 1; mt5.TRADE_ACTION_SLTP = 6
    mt5.ORDER_FILLING_IOC = 2; mt5.ORDER_FILLING_FOK = 1
    mt5.ORDER_FILLING_RETURN = 0
    mt5.ORDER_TIME_GTC = 0
    mt5.TRADE_RETCODE_DONE = 10009; mt5.TRADE_RETCODE_PLACED = 10008
    mt5.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 2
    mt5.DEAL_ENTRY_OUT = 1; mt5.DEAL_ENTRY_INOUT = 2
    sys.modules["MetaTrader5"] = mt5

import numpy as np
import pandas as pd

from config.settings import config

PASS, FAIL = 0, 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")

def section(t):
    print(f"\n=== {t} ===")

TMP = Path(tempfile.mkdtemp(prefix="nexus_p5_"))
config.LIVE_STATE_PATH = str(TMP / "live_state.json")

UTC = timezone.utc
NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)   # a Wednesday

SPECS = {"broker_symbol": "EURUSD.pro", "asset_class": "forex", "digits": 5,
         "point": 0.00001, "spread_points": 2, "tick_size": 0.00001,
         "tick_value": 1.0, "contract_size": 100000, "volume_min": 0.01,
         "volume_step": 0.01, "volume_max": 100.0, "stops_level_points": 0}

# ===========================================================================
section("1. state: defaults / roundtrip / corrupt tolerance")
from src.live.state import (default_state, load_state, save_state, iso,
                            parse_iso)

st = default_state()
st["positions"]["-1"] = {"ticket": -1, "symbol": "EURUSD"}
st["virt_ticket_seq"] = -2
save_state(config.LIVE_STATE_PATH, st)
back = load_state(config.LIVE_STATE_PATH)
check("roundtrip positions", back["positions"].get("-1", {}).get("symbol") == "EURUSD")
check("roundtrip seq", back["virt_ticket_seq"] == -2)
check("roundtrip defaults merged", "day" in back and "peak_equity" in back)
# atomicity: file is valid JSON and tmp file is gone
json.load(open(config.LIVE_STATE_PATH))
check("no tmp residue", not list(TMP.glob("*.tmp")))
# corrupt file -> defaults
with open(config.LIVE_STATE_PATH, "w") as fh:
    fh.write("{not json")
check("corrupt tolerated", load_state(config.LIVE_STATE_PATH) == default_state())
check("parse_iso None-safe", parse_iso(None) is None and parse_iso("junk") is None)

# ===========================================================================
section("2. risk_engine: math, caps, guards")
from src.live.risk_engine import (RiskEngine, currency_exposure,
                                  entries_allowed, position_risk_usd,
                                  refresh_daily_guards, stop_distance_for)

r = position_risk_usd(0.10, 0.0020, SPECS)     # 0.1 lot x 200 ticks x $1
check("risk math", abs(r - 20.0) < 1e-9, f"got {r}")
check("risk zero-guards", position_risk_usd(0, 1, SPECS) == 0
      and position_risk_usd(1, 0, SPECS) == 0)

exp = currency_exposure([
    {"symbol": "EURUSD", "side": "LONG", "risk_usd": 100.0},
    {"symbol": "GBPUSD", "side": "LONG", "risk_usd": 100.0},
])
check("currency net: EUR +100", abs(exp.get("EUR", 0) - 100) < 1e-9)
check("currency net: USD -200", abs(exp.get("USD", 0) + 200) < 1e-9)

acct = {"equity": 100000.0, "balance": 100000.0}
re_eng = RiskEngine(acct, [])
ok, why = re_eng.check_entry("EURUSD", "LONG", 0.10, 1.0980, 1.1000, SPECS, 0.0010)
check("clean entry allowed", ok, why)

# max positions
full = [{"symbol": f"S{i}", "side": "LONG", "risk_usd": 10.0}
        for i in range(config.MAX_POSITIONS)]
ok, why = RiskEngine(acct, full).check_entry("EURUSD", "LONG", 0.1, 1.098, 1.1, SPECS, 0.001)
check("max positions veto", not ok and "max positions" in why)

# total open risk cap: 5% of 100k = 5000; existing 4900 + new 200 -> veto
book = [{"symbol": "USDJPY", "side": "LONG", "risk_usd": 4900.0}]
ok, why = RiskEngine(acct, book).check_entry("EURUSD", "LONG", 1.0, 1.098, 1.1, SPECS, 0.001)
check("total risk cap veto", not ok and "total open risk" in why, why)

# currency cap: 2.5% = 2500; book already LONG EURUSD (USD -2400); another
# LONG EURUSD pushes USD net to -2400-2000 -> veto
book = [{"symbol": "EURUSD", "side": "LONG", "risk_usd": 2400.0}]
ok, why = RiskEngine(acct, book).check_entry("EURUSD", "LONG", 1.0, 1.098, 1.1, SPECS, 0.001)
check("currency cap veto", not ok and "currency cap" in why, why)
# opposite direction reduces net -> allowed
ok, why = RiskEngine(acct, book).check_entry("EURUSD", "SHORT", 1.0, 1.102, 1.1, SPECS, 0.001)
check("hedged entry allowed", ok, why)

# daily guards
state = default_state()
ev = refresh_daily_guards(state, acct, NOW)
check("new day event", "new_day" in ev)
allowed, _ = entries_allowed(state)
check("entries allowed on fresh day", allowed)
# -6% day -> loss latch
ev = refresh_daily_guards(state, {"equity": 94000.0, "balance": 100000.0}, NOW)
check("daily loss latch", "daily_loss_limit" in ev)
allowed, why = entries_allowed(state)
check("entries blocked after loss limit", not allowed and "loss" in why)
# next UTC day resets
ev = refresh_daily_guards(state, acct, NOW + timedelta(days=1))
allowed, _ = entries_allowed(state)
check("new day re-arms", allowed)
# drawdown breaker: peak 100k -> equity 89k (< 90% of peak)
state2 = default_state()
refresh_daily_guards(state2, acct, NOW)
ev = refresh_daily_guards(state2, {"equity": 89000.0, "balance": 100000.0}, NOW)
check("drawdown breaker", "drawdown_breaker" in ev)
allowed, why = entries_allowed(state2)
check("dd breaker blocks", not allowed and "drawdown" in why)
# profit target: +2.5% day
state3 = default_state()
refresh_daily_guards(state3, acct, NOW)
ev = refresh_daily_guards(state3, {"equity": 102500.0, "balance": 100000.0}, NOW)
check("profit target latch", "daily_profit_target" in ev)

# ===========================================================================
section("3. position_manager.evaluate: the exit stack")
from src.live.position_manager import (CLOSE_ALL, CLOSE_VOLUME, SET_SL,
                                       ManagedPosition, evaluate, bars_held)

def mkpos(**kw):
    base = dict(ticket=-1, symbol="EURUSD", asset_class="forex", side="LONG",
                entry_price=1.1000, entry_time=iso(NOW), initial_volume=0.30,
                volume=0.30, atr=0.0010, sl=1.0980, tp=1.1030,
                peak_price=1.1000, trough_price=1.1000, risk_usd=60.0,
                dry_run=True)
    base.update(kw)
    return ManagedPosition(**base)

def kinds(actions):
    return [(a[0], a[2]) for a in actions]

# virtual SL / TP
a = evaluate(mkpos(), 1.0970, NOW)
check("virtual stop fires", (CLOSE_ALL, "stop_loss") in kinds(a))
a = evaluate(mkpos(), 1.1035, NOW)
check("virtual tp fires", (CLOSE_ALL, "take_profit") in kinds(a))
# real positions: NO virtual sl/tp (broker enforces)
a = evaluate(mkpos(dry_run=False), 1.0970, NOW)
check("real pos: no virtual stop", (CLOSE_ALL, "stop_loss") not in kinds(a))

# scale-outs (prices sit clearly PAST the thresholds: 1.1010-1.1000 is
# 0.00099999... in binary, which would test float noise, not the rule)
a = evaluate(mkpos(), 1.1011, NOW)   # +1.1 ATR
cv = [x for x in a if x[0] == CLOSE_VOLUME and x[2] == "scale_out_1"]
check("scale_out_1 at +1 ATR", len(cv) == 1 and abs(cv[0][1] - 0.30 * config.SCALE_OUT_PCT) < 1e-9)
check("scale_out_2 needs scaled_1", not any(x[2] == "scale_out_2" for x in a))
a = evaluate(mkpos(scaled_1=True), 1.1021, NOW)  # +2.1 ATR
check("scale_out_2 at +2 ATR", any(x[2] == "scale_out_2" for x in a))

# breakeven lock (v2): price +1.05 ATR -> stop to entry + 0.10 ATR
a = evaluate(mkpos(), 1.10105, NOW)
be = [x for x in a if x[0] == SET_SL and x[2] == "breakeven_lock"]
check("breakeven lock at +1 ATR", len(be) == 1
      and abs(be[0][1] - (1.1000 + 0.10 * 0.0010)) < 1e-9)

# ratchet rung 1
a = evaluate(mkpos(), 1.1016, NOW)   # +1.6 ATR peak
sl_actions = [x for x in a if x[0] == SET_SL and x[2] == "ratchet"]
check("ratchet locks +0.5 ATR", len(sl_actions) == 1
      and abs(sl_actions[0][1] - (1.1000 + 0.5 * 0.0010)) < 1e-9)

# ratchet rung 2 (v2): +2.1 ATR -> lock +1.0 ATR
a = evaluate(mkpos(scaled_1=True, scaled_2=True), 1.1021, NOW)
r2 = [x for x in a if x[0] == SET_SL and x[2] == "ratchet2"]
check("ratchet2 locks +1.0 ATR", len(r2) == 1
      and abs(r2[0][1] - (1.1000 + 1.0 * 0.0010)) < 1e-9)

# trailing (v2: arms at +1.75 ATR, trails 0.75 ATR behind the peak; price
# +2.9 ATR, just under the TP so the virtual bracket doesn't fire first)
a = evaluate(mkpos(scaled_1=True, scaled_2=True), 1.1029, NOW)
tr = [x for x in a if x[0] == SET_SL and x[2] == "trailing"]
check("trailing engages and trails 0.75 ATR", len(tr) == 1
      and abs(tr[0][1] - (1.1029 - 0.75 * 0.0010)) < 1e-6)
check("no retracement at peak", (CLOSE_ALL, "retracement") not in kinds(a))

# trailing beats ratchet2 once the peak runs (peak +2.9 -> trail +2.15 > +1.0)
check("trailing tighter than ratchet2 at +2.9",
      tr[0][1] > 1.1000 + 1.0 * 0.0010)

# retracement: peak +3 ATR (from state), now back to +1.5 ATR (< 0.6*3)
p = mkpos(scaled_1=True, scaled_2=True, peak_price=1.1030)
a = evaluate(p, 1.1015, NOW)
check("retracement exit", (CLOSE_ALL, "retracement") in kinds(a))

# time partial: 12h held, +0.2 ATR only
p = mkpos(entry_time=iso(NOW - timedelta(hours=12)))
a = evaluate(p, 1.1002, NOW)
tp_ = [x for x in a if x[0] == CLOSE_VOLUME and x[2] == "time_partial"]
check("time partial at 12 bars", len(tp_) == 1 and abs(tp_[0][1] - 0.15) < 1e-9)

# time stop: 20h held
p = mkpos(entry_time=iso(NOW - timedelta(hours=20)))
a = evaluate(p, 1.1002, NOW)
check("time stop at 16 bars", (CLOSE_ALL, "time_stop") in kinds(a))
check("bars_held math", bars_held(p, NOW) == 20)

# friday flatten (find a Friday; entry must be RECENT vs the Friday,
# else the 16-bar time stop correctly fires first)
fri = NOW
while fri.weekday() != 4:
    fri += timedelta(days=1)
fri = fri.replace(hour=21)
p = mkpos(asset_class="forex", entry_time=iso(fri - timedelta(hours=5)))
a = evaluate(p, 1.1002, fri)
check("friday flatten forex", (CLOSE_ALL, "friday_flatten") in kinds(a))
p = mkpos(asset_class="crypto", symbol="BTCUSD",
          entry_time=iso(fri - timedelta(hours=5)))
a = evaluate(p, 1.1002, fri)
check("crypto NOT friday-flattened", (CLOSE_ALL, "friday_flatten") not in kinds(a))

# flip exit: opposing verdict + profit >= 0.5 ATR
flip = {"prob": 0.40}
a = evaluate(mkpos(), 1.1006, NOW, flip_verdict=flip)
check("flip exit in profit", (CLOSE_ALL, "flip_exit") in kinds(a))
# flip tighten underwater (v2: tightens to FLIP_TIGHTEN_STOP_ATR = 0.75 ATR)
a = evaluate(mkpos(), 1.0995, NOW, flip_verdict=flip)
ft = [x for x in a if x[0] == SET_SL and x[2] == "flip_tighten"]
check("flip tighten underwater", len(ft) == 1
      and abs(ft[0][1] - (1.1000 - 0.75 * 0.0010)) < 1e-9)
# non-opposing verdict: nothing
a = evaluate(mkpos(), 1.1006, NOW, flip_verdict={"prob": 0.55})
check("friendly verdict: no flip", not any(x[2].startswith("flip") for x in a))

# ===========================================================================
section("4. journal: SQL shapes + return values")
import src.live.journal as J

class FakeCursor:
    def __init__(self, conn): self.conn = conn; self.rowcount = 1
    def execute(self, sql, params=None):
        self.conn.queries.append((sql, params))
    def fetchone(self): return (42,)
    def __enter__(self): return self
    def __exit__(self, *a): return False

class FakeConn:
    def __init__(self): self.queries = []
    def cursor(self): return FakeCursor(self)
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass

conns = []
@contextmanager
def fake_get_conn():
    c = FakeConn(); conns.append(c); yield c

J.get_conn = fake_get_conn
row_id = J.journal_open(magic=860001, symbol="EURUSD", asset_class="forex",
                        side="LONG", volume_lots=0.1, entry_time=NOW,
                        entry_price=1.1, sl=1.098, tp=1.103,
                        quality=0.4, eff_quality=0.4, regime="trend_up",
                        sentiment=0.0, memory_n=100, spread_pct=0.00002,
                        atr=0.001, ticket=-1, dry_run=True)
check("journal_open returns id", row_id == 42)
sql, params = conns[0].queries[0]
check("open inserts ticket note", "ticket:-1" in params[-1] and "DRY_RUN" in params[-1])

ok = J.journal_close(symbol="EURUSD", ticket=-1, exit_time=NOW, exit_price=1.101,
                     pnl=10.0, r_multiple=0.17, exit_reason="time_stop")
check("journal_close true on rowcount", ok is True)
sql, params = conns[1].queries[0]
check("close matches by ticket note", params[-1] == "ticket:-1%")
check("close carries reason", "time_stop" in params[4])

class ZeroCursor(FakeCursor):
    def __init__(self, conn):
        super().__init__(conn)
        self.rowcount = 0          # instance attr must override the parent's 1
class ZeroConn(FakeConn):
    def cursor(self): return ZeroCursor(self)
@contextmanager
def zero_get_conn():
    c = ZeroConn(); conns.append(c); yield c
J.get_conn = zero_get_conn
ok = J.journal_close(symbol="EURUSD", ticket=-1, exit_time=NOW, exit_price=1.101,
                     pnl=10.0, r_multiple=0.17, exit_reason="time_stop")
check("journal_close false when no row", ok is False)

# ===========================================================================
section("5. reconciler: finalize_close + adopt + closed-away")
import src.live.reconciler as R

sent_tg, closed_j = [], []
R.send_telegram = lambda msg, kind="info": sent_tg.append((kind, msg))
R.journal_close = lambda **kw: closed_j.append(kw) or True

state = default_state()
state["day"] = {"date": NOW.date().isoformat(), "start_balance": 100000.0,
                "start_equity": 100000.0, "closed_count": 0,
                "realized_pnl": 0.0, "halted_loss": False, "profit_lock": False}
pos = mkpos(ticket=555, dry_run=False)
state["positions"]["555"] = pos.to_dict()
R.finalize_close(state, pos, exit_price=1.0980, exit_time=NOW,
                 reason="stop_loss", pnl=-60.0)
check("finalize removes position", "555" not in state["positions"])
check("finalize journals", len(closed_j) == 1 and closed_j[0]["exit_reason"] == "stop_loss")
until = parse_iso(state["no_entry_until"].get("EURUSD"))
check("stop -> >=24h cooldown", until is not None
      and until >= NOW + timedelta(hours=config.LOSS_COOLDOWN_HOURS))
check("day counters", state["day"]["closed_count"] == 1
      and abs(state["day"]["realized_pnl"] + 60.0) < 1e-9)
# repeat loss -> 72h
pos2 = mkpos(ticket=556, dry_run=False)
state["positions"]["556"] = pos2.to_dict()
R.finalize_close(state, pos2, exit_price=1.0980, exit_time=NOW + timedelta(hours=1),
                 reason="stop_loss", pnl=-60.0)
until = parse_iso(state["no_entry_until"].get("EURUSD"))
check("repeat stop -> 72h ban", until >= NOW + timedelta(hours=1)
      + timedelta(hours=config.REPEAT_LOSS_COOLDOWN_HOURS))
check("repeat-loss telegram", any(k == "warning" for k, _ in sent_tg))

class FakeConnector:
    def __init__(self, positions, summary=None):
        self._positions = positions; self._summary = summary
        self._symbol_map = {"EURUSD": "EURUSD.pro"}
    def positions(self, ours_only=True): return self._positions
    def closed_position_summary(self, ticket): return self._summary
    def symbol_specs(self, s): return SPECS
    def classify_asset(self, s): return "forex"
    def magic_for(self, s): return 860001

# closed-away detection: state ticket 777 gone from broker
state = default_state()
p777 = mkpos(ticket=777, dry_run=False)
state["positions"]["777"] = p777.to_dict()
summary = {"exit_price": 1.0981, "exit_time": NOW, "profit": -57.0,
           "volume_closed": 0.30, "comment": ""}
fc = FakeConnector([], summary)
ev = R.reconcile(state, fc, atr_for=lambda s: 0.001, dry_run=False)
check("closed-away detected", "777" not in state["positions"]
      and any("closed_away" in e for e in ev))
check("closed-away reason = stop_loss", closed_j[-1]["exit_reason"] == "stop_loss")
check("closed-away pnl from history", closed_j[-1]["pnl"] == -57.0)

# adoption: broker has ticket 888 unknown to state
bp = {"ticket": 888, "symbol": "EURUSD.pro", "side": "LONG", "volume": 0.20,
      "entry_price": 1.1050, "sl": 1.1030, "tp": 1.1110, "magic": 860001,
      "profit": 0.0, "swap": 0.0, "time": NOW, "comment": ""}
fc = FakeConnector([bp])
ev = R.reconcile(state, fc, atr_for=lambda s: 0.001, dry_run=False)
adopted = state["positions"].get("888")
check("adoption", adopted is not None and any("adopted" in e for e in ev))
check("adopted canonical symbol", adopted["symbol"] == "EURUSD")
check("adopted risk computed", adopted["risk_usd"] > 0)

# DRY_RUN reconcile: reports but adopts nothing
state = default_state()
fc = FakeConnector([bp])
ev = R.reconcile(state, fc, atr_for=lambda s: 0.001, dry_run=True)
check("dry reconcile adopts nothing", not state["positions"] and ev == [])

# estimate_pnl tick math
from src.live.reconciler import estimate_pnl
p = mkpos(volume=0.30)
check("estimate_pnl long", abs(estimate_pnl(p, 1.1010, SPECS) - 30.0) < 1e-9)
p_short = mkpos(side="SHORT", volume=0.30)
check("estimate_pnl short", abs(estimate_pnl(p_short, 1.1010, SPECS) + 30.0) < 1e-9)

# ===========================================================================
section("6. live_trader: entry gate chain (mirrors engine)")
import src.live.live_trader as LT

GATE_COLS = {"atr_14": 0.0010, "spread_price": 0.00002, "spread_med20": 0.00002,
             "adx_14": 25.0, "open": 1.1000, "close": 1.1002,
             "dist_vwap": 0.1, "bar_range": 0.0008, "spread_pct": 0.000018,
             "regime_label": "trend_up", "ret_12": 0.002, "dist_sma200": 0.5}

def frame(**over):
    row = dict(GATE_COLS); row.update(over)
    return pd.DataFrame([row])

VERDICT = {"prob": 0.56, "direction": "LONG", "quality": 0.46,
           "agreement": 0.60, "n_kept": 100}

def trader_with(verdict):
    t = LT.LiveTrader.__new__(LT.LiveTrader)   # skip __init__ (no state/connector)
    t._brains = {}
    class FB:
        def predict(self, row, asof=None): return verdict
    t._brain = lambda cls: FB()
    calls = []
    t._execute_entry = lambda *a, **k: calls.append((a, k))
    return t, calls

def run_gate(d, verdict=VERDICT, cls="forex", symbol="EURUSD"):
    t, calls = trader_with(verdict)
    acct = {"equity": 100000.0, "balance": 100000.0}
    re_eng = RiskEngine(acct, [])
    t._evaluate_entry(symbol, cls, d, NOW, acct, re_eng)
    return calls

check("all gates green -> execute", len(run_gate(frame())) == 1)
check("HOLD prob blocked", len(run_gate(frame(), verdict={**VERDICT, "prob": 0.51})) == 0)
check("low agreement blocked", len(run_gate(frame(), verdict={**VERDICT, "agreement": 0.50})) == 0)
check("low quality blocked", len(run_gate(frame(), verdict={**VERDICT, "quality": 0.30})) == 0)
check("no verdict blocked", len(run_gate(frame(), verdict=None)) == 0)
check("ADX fail blocked", len(run_gate(frame(adx_14=10.0), verdict={**VERDICT, "quality": 0.40})) == 0)
check("ADX fail but STRONG passes", len(run_gate(frame(adx_14=10.0))) == 1)
check("bar-confirm blocks long", len(run_gate(frame(close=1.1000 - 0.6 * 0.0010))) == 0)
check("vwap-confirm blocks long", len(run_gate(frame(dist_vwap=-0.6))) == 0)
check("no-chase blocks", len(run_gate(frame(bar_range=2.0 * 0.0010))) == 0)
check("spread filter blocks", len(run_gate(frame(spread_price=3 * 0.00002))) == 0)
check("NaN atr blocked", len(run_gate(frame(atr_14=float("nan")))) == 0)

# tp-worth-spread (isolate: disable the spread filter first)
saved = config.SPREAD_FILTER_ENABLED
config.SPREAD_FILTER_ENABLED = False
try:
    check("tp-worth-spread blocks", len(run_gate(frame(spread_price=0.0006))) == 0)
finally:
    config.SPREAD_FILTER_ENABLED = saved

# crypto momentum gate (live-only, config-documented)
check("crypto momentum blocks long below sma200",
      len(run_gate(frame(dist_sma200=-0.1), cls="crypto", symbol="BTCUSD")) == 0)
check("crypto momentum passes above",
      len(run_gate(frame(), cls="crypto", symbol="BTCUSD")) == 1)
check("forex unaffected by momentum gate",
      len(run_gate(frame(dist_sma200=-0.1))) == 1)

# short side passes the mirror gates (dist_vwap must be on the SHORT side
# too - a positive dist_vwap rightly blocks shorts at the VWAP gate)
short_v = {"prob": 0.44, "direction": "SHORT", "quality": 0.46,
           "agreement": 0.60, "n_kept": 100}
calls = run_gate(frame(close=1.0998, dist_vwap=-0.1), verdict=short_v)
check("short signal executes", len(calls) == 1 and calls[0][0][2] == "SHORT")

# ===========================================================================
section("7. live_trader._apply_actions: virtual book")
trader = LT.LiveTrader.__new__(LT.LiveTrader)
trader.state = default_state()
trader.connector = SimpleNamespace(
    symbol_specs=lambda s: SPECS,
    modify_sltp=lambda *a, **k: True,
    close_position=lambda *a, **k: True)

finalized = []
LT.finalize_close = lambda state, pos, **kw: finalized.append((pos.ticket, kw))

pos = mkpos(ticket=-7)
trader.state["positions"]["-7"] = pos.to_dict()

# SET_SL (ratchet) on virtual book
trader._apply_actions(pos, 1.1016, [(SET_SL, 1.1005, "ratchet")], NOW)
check("virtual SET_SL applied", trader.state["positions"]["-7"]["sl"] == 1.1005)

# CLOSE_VOLUME (scale_out_1)
trader._apply_actions(pos, 1.1010,
                      [(CLOSE_VOLUME, 0.10, "scale_out_1")], NOW)
after = trader.state["positions"]["-7"]
check("partial reduces volume", abs(after["volume"] - 0.20) < 1e-9)
check("partial books realized pnl", after["realized_pnl"] > 0)
check("scale flag marked", after["scaled_1"] is True)

# CLOSE_ALL -> finalize_close called with time_stop
trader._apply_actions(pos, 1.1005, [(CLOSE_ALL, None, "time_stop")], NOW)
check("close_all finalizes", len(finalized) == 1
      and finalized[0][0] == -7 and finalized[0][1]["reason"] == "time_stop")
check("close_all pnl includes partials",
      finalized[0][1]["pnl"] != 0)

# real-mode SET_SL goes through modify_sltp
pos_real = mkpos(ticket=99, dry_run=False)
trader.state["positions"]["99"] = pos_real.to_dict()
trader._apply_actions(pos_real, 1.1016, [(SET_SL, 1.1005, "ratchet")], NOW)
check("real SET_SL via broker", trader.state["positions"]["99"]["sl"] == 1.1005)

# failed broker close -> flags NOT marked (retry next cycle)
trader.connector.close_position = lambda *a, **k: False
pos_fail = mkpos(ticket=98, dry_run=False)
trader.state["positions"]["98"] = pos_fail.to_dict()
trader._apply_actions(pos_fail, 1.1010, [(CLOSE_VOLUME, 0.10, "scale_out_1")], NOW)
check("failed partial: flag unmarked",
      trader.state["positions"]["98"]["scaled_1"] is False)

# ===========================================================================
section("8. oos_split_check.half_verdicts")
from scripts.oos_split_check import half_verdicts
v = {0: "a", 1: "b", 2: "c", 3: "d"}
check("first half", half_verdicts(v, 2, "first") == {0: "a", 1: "b"})
check("second half", half_verdicts(v, 2, "second") == {2: "c", 3: "d"})

# ===========================================================================
print(f"\n{'=' * 60}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
