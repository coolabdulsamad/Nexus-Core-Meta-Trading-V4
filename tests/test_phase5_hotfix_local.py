"""Phase 5 hotfix tests (2026-09-07) — offline.

Covers the three fixes shipped after the first real demo run:
1. Broker-time frame: MT5 bar/tick timestamps are broker-server wall time
   (XM: UTC+2/+3). The entry cycle must match target_bar in THAT frame, or
   no entry can ever fire (observed: 10h, 10 entry cycles, 0 entries).
2. save_state: transient WinError 5 on os.replace (AV/indexer locks) must be
   retried, with a direct-write fallback; never raise into the trade cycle.
3. Telegram: non-final retry attempts log at WARNING, not ERROR.

Run: python tests/test_phase5_hotfix_local.py
"""
import os, sys, types, logging, json, tempfile
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
    mt5.ORDER_FILLING_IOC = 2; mt5.ORDER_FILLING_RETURN = 0
    mt5.ORDER_TIME_GTC = 0
    mt5.TRADE_RETCODE_DONE = 10009
    mt5.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 2
    mt5.DEAL_ENTRY_IN = 0; mt5.DEAL_ENTRY_OUT = 1; mt5.DEAL_ENTRY_INOUT = 2
    sys.modules["MetaTrader5"] = mt5

import numpy as np
import pandas as pd

from config.settings import config
import src.live.live_trader as LT
from src.live import state as state_mod
from src.live.state import default_state, save_state, load_state

PASS, FAIL = 0, 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")

NOW = datetime(2026, 9, 7, 12, 0, 25, tzinfo=timezone.utc)

print("\n=== 1. broker-time offset ===")
t = LT.LiveTrader.__new__(LT.LiveTrader)
t._broker_offset_h = 0.0
t.universe = {"EURUSD": {"asset_class": "forex"},
              "BTCUSD": {"asset_class": "crypto"}}

class ConnFresh:
    def get_tick(self, sym):
        # XM server wall clock = UTC+3, tick stamped in that frame
        return SimpleNamespace(time=(NOW + timedelta(hours=3)).timestamp())
t.connector = ConnFresh()
t._refresh_broker_offset(NOW)
check("fresh tick -> +3.0h offset", t._broker_offset_h == 3.0)

class ConnStaleThenFresh:
    def get_tick(self, sym):
        if sym == "EURUSD":   # weekend-stale forex tick (~50h old)
            return SimpleNamespace(time=(NOW - timedelta(hours=50)).timestamp())
        return SimpleNamespace(time=(NOW + timedelta(hours=3)).timestamp())
t._broker_offset_h = 0.0
t.connector = ConnStaleThenFresh()
t._refresh_broker_offset(NOW)
check("stale tick skipped, crypto tick used", t._broker_offset_h == 3.0)

class ConnAllStale:
    def get_tick(self, sym):
        return SimpleNamespace(time=(NOW - timedelta(hours=50)).timestamp())
t._broker_offset_h = 2.0
t.connector = ConnAllStale()
t._refresh_broker_offset(NOW)
check("all stale -> previous offset kept", t._broker_offset_h == 2.0)

class ConnNone:
    def get_tick(self, sym): return None
t._broker_offset_h = 2.0
t.connector = ConnNone()
t._refresh_broker_offset(NOW)
check("no ticks -> previous offset kept", t._broker_offset_h == 2.0)

print("\n=== 2. entry-cycle target bar uses broker frame ===")
# Build a trader whose frames end at the broker-labeled bar 14:00 (== the bar
# that just closed at true 12:00 UTC with a +3h server clock). Pre-fix this
# never matched target_bar (11:00) and every symbol stayed "not_ready".
t = LT.LiveTrader.__new__(LT.LiveTrader)
t.universe = {"BTCUSD": {"asset_class": "crypto"}}
t._brains = {}
t._frame_cache = {}
t._retry_symbols = set()
t._maintenance_proc = None
t._broker_offset_h = 3.0
t._last_scan = ""
t.state = default_state()

broker_now = NOW + timedelta(hours=3)            # 15:00:25 server time
last_bar = broker_now.replace(minute=0, second=0, microsecond=0) \
           - timedelta(hours=1)                  # 14:00 labeled bar

frame_d = pd.DataFrame({
    "timestamp": [last_bar],
    "atr_14": [1.0], "adx_14": [30.0],
    "spread_price": [0.1], "spread_med20": [0.1],
    "open": [100.0], "close": [100.1],
    "dist_vwap": [0.0], "bar_range": [0.5], "spread_pct": [0.001],
    "dist_sma200": [0.01], "ret_12": [0.01], "regime_label": ["trend"],
})
frame_ts = pd.DatetimeIndex(frame_d["timestamp"])

evaluated = []
t._fresh_frame = lambda symbol, now: (frame_d, frame_ts)
t._evaluate_entry = lambda *a, **k: evaluated.append(a[0]) or "conviction"
t._refresh_broker_offset = lambda now: None      # keep the +3.0 set above

class ConnAcct:
    def account(self):
        return {"equity": 100000.0, "balance": 100000.0}
    def symbol_specs(self, sym):
        return {"point": 0.01, "contract_size": 1.0}
t.connector = ConnAcct()
t._entry_cycle(NOW, LT._floor_hour(NOW))
check("bar matched target_bar (symbol evaluated)", evaluated == ["BTCUSD"])
check("nothing left in retry set", t._retry_symbols == set())
check("scan summary recorded", "0/1 passed" in t._last_scan)

# and pre-fix behaviour would have been: frame at 14:00 vs target 11:00 UTC
t2 = LT.LiveTrader.__new__(LT.LiveTrader)
t2.universe = t.universe; t2._brains = {}; t2._frame_cache = {}
t2._retry_symbols = set(); t2._maintenance_proc = None
t2._broker_offset_h = 0.0                        # the bug: true-UTC frame
t2._last_scan = ""
t2.state = default_state()
t2._fresh_frame = t._fresh_frame
t2._evaluate_entry = lambda *a, **k: 1/0         # must never be reached
t2._refresh_broker_offset = lambda now: None
t2.connector = ConnAcct()
t2._entry_cycle(NOW, LT._floor_hour(NOW))
check("offset 0 -> bar never matches (reproduces the bug)",
      t2._retry_symbols == {"BTCUSD"})

print("\n=== 3. save_state resilience ===")
tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "live_state.json")

orig_replace = os.replace
calls = {"n": 0}
def flaky_replace(a, b):
    calls["n"] += 1
    if calls["n"] <= 2:
        raise PermissionError(5, "Access is denied")
    return orig_replace(a, b)
os.replace = flaky_replace
try:
    ok = save_state(path, default_state())
finally:
    os.replace = orig_replace
check("transient WinError 5 retried -> save succeeds", ok and calls["n"] == 3)
check("state readable after retry", load_state(path)["version"] == 1)

def always_denied(a, b):
    raise PermissionError(5, "Access is denied")
os.replace = always_denied
try:
    ok = save_state(path, default_state())
finally:
    os.replace = orig_replace
check("persistent lock -> direct-write fallback succeeds", ok)
check("fallback content readable", load_state(path)["version"] == 1)

# destination locked for ALL writes -> returns False, never raises
orig_open = state_mod.open if hasattr(state_mod, "open") else open
import builtins
real_open = builtins.open
def denied_open(p, *a, **k):
    if str(p).startswith(tmpdir):
        raise PermissionError(13, "Access is denied")
    return real_open(p, *a, **k)
builtins.open = denied_open
try:
    ok = save_state(path, default_state())
finally:
    builtins.open = real_open
check("total lock -> returns False (no exception into cycle)", ok is False)

print("\n=== 4. telegram retry levels ===")
import src.utils.telegram as tg
records = []
class RecHandler(logging.Handler):
    def emit(self, r): records.append(r)
tg.logger.addHandler(RecHandler())

class FlakySession:
    def __init__(self): self.n = 0
    def post(self, *a, **k):
        self.n += 1
        if self.n < 3:
            raise ConnectionResetError(10054, "reset")
        return SimpleNamespace(status_code=200)
old_session, old_tok, old_chat = tg._session, config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID
old_backoff = tg._BACKOFF
tg._session = FlakySession()
tg._BACKOFF = 0
config.TELEGRAM_BOT_TOKEN = "x"; config.TELEGRAM_CHAT_ID = "y"
records.clear()
try:
    ok = tg.send_telegram("hi", "info")
finally:
    tg._session = old_session; tg._BACKOFF = old_backoff
    config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID = old_tok, old_chat
levels = [r.levelno for r in records if "Telegram error" in r.getMessage()]
check("retry succeeds after 2 resets", ok)
check("non-final retries are WARNING not ERROR",
      levels == [logging.WARNING, logging.WARNING])

class DeadSession:
    def post(self, *a, **k): raise ConnectionResetError(10054, "reset")
tg._session = DeadSession()
tg._BACKOFF = 0
config.TELEGRAM_BOT_TOKEN = "x"; config.TELEGRAM_CHAT_ID = "y"
records.clear()
try:
    ok = tg.send_telegram("hi", "info")
finally:
    tg._session = old_session
    config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID = old_tok, old_chat
levels = [r.levelno for r in records if "Telegram error" in r.getMessage()]
gave_up = [r for r in records if "gave up" in r.getMessage()]
check("all retries fail -> returns False", ok is False)
check("final attempt still ERROR + gave-up line",
      levels == [logging.WARNING, logging.WARNING, logging.ERROR]
      and len(gave_up) == 1 and gave_up[0].levelno == logging.ERROR)

print("\n============================================================")
print(f"RESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
