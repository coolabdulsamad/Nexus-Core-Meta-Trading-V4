"""
config/settings.py
Single source of truth for Nexus Core Meta (V4) - the MetaTrader 5 edition.
Secrets come ONLY from environment / .env - never hardcode credentials here.

Lineage
-------
V4 carries over every evidence-calibrated constant from the Alpaca edition
(Nexus Core v3.6.5): the 1-hour brain, the measured quality distribution
(max q ever observed live = 0.480, so STRONG = 0.45), the memory-depth
scaling, the tolerance-mode confirmations, the rebuilt exit stack and the
loss cooldowns. What changes is the broker layer (MetaTrader 5) and the
asset universe (forex + crypto + metals instead of US stocks).

What V4 adds on top of the Alpaca edition
------------------------------------------
1. PORTFOLIO RISK ENGINE. Forex pairs share currencies: long EURUSD +
   long GBPUSD is a doubled USD-short. V4 tracks NET RISK PER CURRENCY
   across all open positions and refuses entries that push a currency
   past its cap, plus a hard cap on TOTAL open risk.
2. COST-AWARE EDGES. MT5 gives us the real spread on every bar. Entries
   are refused when the spread is abnormally wide, and a trade is only
   allowed when the TP distance is many multiples of the current spread
   (the Alpaca edition proved an edge the size of your costs is a
   guaranteed loss). Backtests charge spread + commission + swap.
3. SESSION INTELLIGENCE. Forex has no single "market open": there are
   Tokyo / London / New York sessions, a daily rollover (spread spikes,
   swap charged), and a weekend gap. V4 blackouts the rollover window,
   can refuse the dead hours per pair class, and optionally flattens
   forex before the weekend.
4. PER-ASSET-CLASS MEMORY. Forex, crypto and metals get separate Qdrant
   collections - a JPY carry unwind looks nothing like a BTC breakout,
   so the brain never recalls crypto states to judge a forex setup.
5. TRADE JOURNAL. Every position is journalled to Postgres with its
   full "why" (quality, regime, sentiment, memory depth) so weekly
   calibration (the v3.6.3 method) is a SQL query, not log archaeology.
6. NO TA-LIB. Indicators are pure pandas/numpy - Windows install is
   `pip install -r requirements.txt` and nothing else.
"""
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def _database_url() -> str:
    """DATABASE_URL wins if set; otherwise compose from DB_* parts so the
    .env file matches docker-compose.yml field for field."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    user = os.getenv("DB_USER", "nexus")
    pwd = os.getenv("DB_PASSWORD", "change_me_strong_password")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5544")  # 5432 belongs to the Alpaca edition
    name = os.getenv("DB_NAME", "nexus_mt5")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{name}"


class DatabaseSettings(BaseModel):
    url: str = _database_url()


class QdrantSettings(BaseModel):
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6644"))  # 6333 belongs to the Alpaca edition
    url: str = os.getenv("QDRANT_URL", "")          # set for remote/cloud Qdrant
    api_key: str = os.getenv("QDRANT_API_KEY", "")


class MT5Settings(BaseModel):
    # Leave login/password/server empty to use the account already logged
    # in inside the running MT5 terminal (recommended on your own machine).
    login: int = int(os.getenv("MT5_LOGIN", "0") or 0)
    password: str = os.getenv("MT5_PASSWORD", "")
    server: str = os.getenv("MT5_SERVER", "")
    terminal_path: str = os.getenv("MT5_TERMINAL_PATH", "")  # optional explicit terminal64.exe path


class GlobalConfig:
    database = DatabaseSettings()
    qdrant = QdrantSettings()
    mt5 = MT5Settings()

    # ----- Data / timeframe -----
    # V4 keeps the 1-hour brain (the Alpaca edition proved 5-min recall has
    # no edge after costs - PF 0.41-0.74 - and hourly bars fixed it).
    BAR_MINUTES = 60
    BAR_SUFFIX = "_1h"
    LIVE_BARS_WINDOW = 300            # bars fetched per cycle (200 SMA + indicator warm-up)
    HISTORY_YEARS = 3                 # default backfill depth (MT5 gives deep H1 history for free)

    # ----- Universe (canonical names; broker suffixes auto-resolved) -----
    # MT5 brokers rename symbols (EURUSD.pro, EURUSDm, XAUUSD.a ...). The
    # connector discovers the real broker names at startup - you only ever
    # edit these canonical lists.
    FOREX_POOL = [
        # majors
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
        # JPY crosses
        "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY",
        # EUR/GBP crosses
        "EURGBP", "EURAUD", "EURCAD", "EURNZD", "EURCHF",
        "GBPAUD", "GBPCAD", "GBPNZD", "GBPCHF",
        # commodity-bloc crosses
        "AUDCAD", "AUDCHF", "AUDNZD", "CADCHF", "NZDCAD", "NZDCHF",
    ]
    METALS_POOL = ["XAUUSD", "XAGUSD"]
    CRYPTO_POOL = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD",
                   "LTCUSD", "BCHUSD", "ADAUSD", "DOGEUSD"]
    INDICES_POOL: list = []           # e.g. ["US30", "NAS100", "SPX500"] - enable deliberately

    # ----- Universe management -----
    UNIVERSE_MODE = os.getenv("UNIVERSE_MODE", "manual")  # 'manual' = watch ALL active | 'auto' = daily top-N
    TOP_N_SYMBOLS = 6
    SELECTOR_ATR_PCT_MIN = 0.0008     # forex is less volatile than stocks; wider acceptance band
    SELECTOR_ATR_PCT_MAX = 0.030

    # ----- Brain (case-based memory) -----
    FORWARD_HORIZON_HOURS = 4         # prediction target: 4 x 1h bars ahead
    MEMORY_NEIGHBORS = 100            # k nearest states retrieved
    MEMORY_MIN_AGE_MINUTES = (FORWARD_HORIZON_HOURS + 1) * 60  # look-ahead guard
    MIN_NEIGHBOR_SIMILARITY = 0.50    # cosine floor
    MIN_NEIGHBOR_AGREEMENT = 0.55     # weighted majority (0.5 = coin flip)
    REGIME_FILTER_ENABLED = True
    BUY_THRESHOLD = 0.52              # HOLD zone between 0.48 and 0.52
    SELL_THRESHOLD = 0.48
    PCA_COMPONENTS = 64
    # V4: one memory collection per asset class (market_memory_60m_forex,
    # market_memory_60m_crypto, ...). A BTC breakout is not evidence for EURUSD.
    MEMORY_PER_ASSET_CLASS = True

    # ----- Sentiment -----
    USE_REAL_SENTIMENT = True         # FinBERT on real headlines (live only)
    SENTIMENT_BIAS = 0.05
    SENTIMENT_IN_BACKTEST = False     # keep bias OFF in backtests (honesty)
    # Symbols whose sentiment NewsAPI can actually find (crypto majors, gold).
    # Forex pairs rarely produce usable headlines -> sentiment stays 0.0 and
    # the sentiment vetoes simply don't engage on them (by design).
    SENTIMENT_KEYWORDS = {            # canonical symbol -> news query
        "BTCUSD": "bitcoin", "ETHUSD": "ethereum", "SOLUSD": "solana",
        "XRPUSD": "ripple xrp", "XAUUSD": "gold price", "XAGUSD": "silver price",
    }

    # ----- Entry quality gate (v3.6.3-calibrated on 39,570 live readings) -----
    ENTRY_CONVICTION_MARGIN = 0.015   # min |prob - 0.5|
    MIN_SIGNAL_QUALITY = 0.35         # the 0.20-0.24 "weak" band had no edge
    QUALITY_MEMORY_REF_N = 80         # quality scaled by min(1, n/this)
    QUALITY_STRONG = 0.45             # ~p95 of the observed live range
    QUALITY_MEDIUM = 0.35
    CRYPTO_MIN_SIGNAL_QUALITY = 0.25  # crypto quality runs structurally lower

    # ----- Signal-strength position sizing -----
    RISK_PCT_STRONG = 0.020           # % of per-symbol slice risked to the stop
    RISK_PCT_MEDIUM = 0.010
    RISK_PCT_WEAK = 0.005
    NOTIONAL_CAP_PCT = 0.75           # max notional per position, % of slice
    NOTIONAL_CAP_ABS = 75000          # absolute $ cap per position

    # ----- Entry analysis gates (the Alpaca edition's post-mortems) -----
    SENTIMENT_VETO_LONG = -0.60       # sent <= this -> no LONG (extreme fear)
    SENTIMENT_VETO_SHORT = 0.60       # sent >= this -> no SHORT (extreme euphoria)
    TOXIC_REGIME_SENT = -0.30         # trend_down AND sent <= this -> veto LONG at ANY quality
    TREND_REGIME_MIN_QUALITY = 0.45   # counter-regime entries must be STRONG
    CRYPTO_MOMENTUM_GATE = True       # crypto LONG needs price > sma200 AND 24h return > 0

    # ----- Entry confirmation layer (tape must agree; tolerance mode) -----
    ENTRY_BAR_CONFIRM_ENABLED = True
    ENTRY_BAR_CONFIRM_TOLERANCE_ATR = 0.5   # only a STRONG counter-bar blocks
    ENTRY_VWAP_CONFIRM_ENABLED = True
    ENTRY_VWAP_CONFIRM_TOLERANCE_ATR = 0.5  # only block when DEEPLY wrong side of VWAP
    ENTRY_NO_CHASE_ENABLED = True
    ENTRY_NO_CHASE_MAX_RANGE_ATR = 1.5
    ENTRY_ADX_MIN = 20.0              # with-trend entries need a real trend (0 = off)

    # ----- Cost guards (NEW - forex edge lives or dies here) -----
    SPREAD_FILTER_ENABLED = True
    SPREAD_MAX_MULT_OF_MEDIAN = 2.5   # refuse entries while spread > 2.5x its own 20-bar median
    MIN_TP_TO_SPREAD_MULT = 6.0       # TP distance must be >= 6x current spread, else skip
    DEFAULT_COMMISSION_PER_LOT = 0.0  # per side, account currency; set to your broker's (e.g. 3.5)
    SWAP_MODEL_IN_BACKTEST = True     # charge triple-swap Wednesday, approximated from config below
    ASSUMED_SWAP_POINTS_PER_DAY = -2.0  # conservative default when the broker spec is unavailable

    # ----- Forex session intelligence (NEW; all hours UTC) -----
    SESSION_FILTER_ENABLED = True
    # Dead zone: after NY close, before Tokyo wakes. Spreads widen, moves fade.
    DEAD_HOURS_UTC = (21, 24)         # no new entries 21:00-24:00 UTC
    ASIAN_THIN_HOURS_UTC = (0, 7)     # non-Asia pairs (no JPY/AUD/NZD) skip 00:00-07:00 UTC
    ROLLOVER_BLACKOUT_MINUTES = 20    # around 21:00 UTC (5pm ET): spread spike + swap
    WEEKEND_FLAT_ENABLED = True       # forex/metals: no fresh entries late Friday ...
    FRIDAY_NO_ENTRY_UTC_HOUR = 17     # ... after this hour Friday ...
    FRIDAY_FLATTEN_UTC_HOUR = 20      # ... and open forex/metal positions are closed at this hour
    SUNDAY_OPEN_BLACKOUT_MINUTES = 30 # spread chaos at the weekly reopen

    # ----- Loss cooldowns (the ETH 3-stops-in-7h fix) -----
    COOLDOWN_AFTER_CLOSE_BARS = 3
    LOSS_COOLDOWN_HOURS = 24          # after a STOP_LOSS, symbol banned this long
    REPEAT_LOSS_WINDOW_DAYS = 7       # 2 stop-outs inside this window ...
    REPEAT_LOSS_COOLDOWN_HOURS = 72   # ... bans the symbol this long

    # ----- Trade structure (1h recalibration) -----
    STOP_ATR_MULT = 2.0               # stop distance = 2 x ATR(1h)
    REWARD_RISK_RATIO = 1.5           # TP = 1.5R (3 x ATR)
    SLIPPAGE_BPS = 0.0005
    TIME_LIMIT_BARS = 16              # TP needs 3 ATR - give it 16h
    COOLDOWN_BARS = 3
    ORDER_FILL_RETRIES = 3            # MT5 requotes happen; retry with fresh price
    DEVIATION_POINTS = 20             # max slippage on market orders

    # ----- SMA exit (DISABLED - structurally guaranteed-loss) -----
    SMA_EXIT_ENABLED = False
    SMA_EXIT_BUFFER_ATR = 0.25
    SMA_EXIT_CONFIRM_BARS = 2

    # ----- Profit locking (the rebuilt v3.6 stack) -----
    ENABLE_PROFIT_DRAWDOWN_PROTECTION = True
    RETRACEMENT_ARM_ATR = 2.0         # arm the lock only after +2 ATR peak
    RETRACEMENT_KEEP_PCT = 0.60       # exit if profit falls to 60% of peak
    PROFIT_RATCHET_ATR = 1.5          # at +1.5 ATR the stop ratchets up ...
    RATCHET_LOCK_ATR = 0.50           # ... to entry + 0.5 ATR (locks real money)
    TRAILING_STOP_ACTIVATE_ATR = 2.5  # hard trailing starts at +2.5 ATR
    TRAILING_STOP_DISTANCE_ATR = 2.5  # trail 2.5 ATR behind the peak
    SCALE_OUT_ENABLED = True          # sell 1/3 at +1 ATR and 1/3 at +2 ATR, trail the rest
    SCALE_OUT_1_ATR = 1.0
    SCALE_OUT_2_ATR = 2.0
    SCALE_OUT_PCT = 0.33
    # In-trade re-analysis: the brain re-judges every open position each cycle
    FLIP_EXIT_PROFIT_ATR = 0.5        # brain flips against + profit >= 0.5 ATR -> exit NOW
    FLIP_TIGHTEN_UNDERWATER = True    # brain flips against while underwater -> tighten stop to 1 ATR
    ENABLE_TIME_PARTIAL = True
    TIME_PARTIAL_BARS = 12
    TIME_PARTIAL_PROFIT_ATR = 0.5

    # ----- Portfolio risk engine (NEW) -----
    MAX_POSITIONS = 5
    SLICE_PCT_OF_EQUITY = 0.20        # per-symbol slice = equity x this (5 slots x 0.20 = 100%)
    CAPITAL_PER_SYMBOL = 100000.0     # absolute slice cap
    BUYING_POWER_USAGE_CAP = 0.95
    TOTAL_OPEN_RISK_PCT = 0.05        # sum of (qty x stop distance) over ALL positions <= 5% equity
    CURRENCY_RISK_CAP_PCT = 0.025     # net risk per single CURRENCY (EUR, USD, JPY...) <= 2.5% equity
    CORRELATION_CAP_ENABLED = True    # block entries that stack same-direction correlated risk

    # ----- Position adoption & honest sizing -----
    ADOPT_BROKER_POSITIONS = True     # adopt positions found at startup/restart (by magic OR orphan)
    ADOPTED_TIME_LIMIT_ENABLED = True
    MAGIC_BASE = 860001               # our orders carry magic MAGIC_BASE + stable symbol code
    ADOPT_FOREIGN_POSITIONS = False   # positions with magic=0/manual: alert, don't adopt (default)

    # ----- Daily guards (account-wide, not per-symbol - forex book is one book) -----
    DAILY_LOSS_LIMIT_PCT = 0.05       # stop opening after -5% day
    DAILY_PROFIT_TARGET_PCT = 0.02    # +2% day -> stop opening new trades (0 = disabled)
    DAILY_TARGET_LOCK_BREAKEVEN = True

    # ----- Circuit breakers -----
    MAX_DRAWDOWN_PCT = 0.10
    VOLATILITY_FILTER_ENABLED = True
    VOLATILITY_RATIO_MAX = 3.0        # ATR vs its 50-bar average

    # ----- Live observability -----
    POSITION_HEARTBEAT_BARS = 4       # Telegram status card every N bars per open trade
    TRAIL_ALERT_STEP_ATR = 0.25       # alert each time the trail ratchets meaningfully
    TELEGRAM_HEARTBEAT_CYCLES = 6     # equity heartbeat every N cycles (~30 min at 300s)
    TELEGRAM_EOD_REPORT = True

    # ----- Data freshness -----
    MAX_DATA_AGE_SECONDS = 7200       # 1h bars: accept up to 2h old
    DATA_FETCH_TIMEOUT = 30

    # ----- Daily self-maintenance -----
    DAILY_MAINTENANCE_ENABLED = True  # trader auto-runs pump + Qdrant outcome sync
    DAILY_MAINTENANCE_UTC_HOUR = 22   # after NY close / rollover, before Asia
    MAINTENANCE_SYNC_DAYS = 10        # incremental outcome sync window
    MAINTENANCE_TIMEOUT_SECONDS = 3600

    # ----- Backtesting -----
    BACKTEST_INITIAL_CAPITAL = 10000.0
    WALK_FORWARD_ENABLED = True       # memory is rebuilt as-of each fold (no future leakage)

    # ----- Notifications -----
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    # ----- Misc -----
    WIN_RATE_WINDOW = 20


config = GlobalConfig()
