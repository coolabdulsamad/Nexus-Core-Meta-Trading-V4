"""
src/mt5_client/connector.py
================================================================
Everything the system ever asks MetaTrader 5, in one audited place.

Design rules (learned the hard way on the Alpaca edition):
- The broker is the ONLY truth about positions. Every read is fresh.
- Close direction is derived from the position's actual side, never
  from tracked state.
- Volume is computed from RISK (account currency) via the symbol's
  tick value / tick size, then snapped to the broker's volume step.
- SL/TP respect the broker's trade_stops_level (min distance).
- Filling mode is per-symbol (FOK / IOC / RETURN) with fallback retry.
- Works in BOTH netting and hedging account modes; our positions are
  identified by magic number, and in hedging mode several tickets per
  symbol are supported.

IMPORTANT: the MetaTrader5 python package ONLY works on Windows with a
running MT5 terminal (or under Wine on Linux). Every public method fails
loudly but cleanly when MT5 is unavailable, so backtests/tools that only
need the DB still run anywhere.
"""
import os
import math
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:                     # non-Windows / not installed
    mt5 = None
    MT5_AVAILABLE = False

from config.settings import config
from src.utils.logger import setup_logger

logger = setup_logger("MT5Connector", "logs/mt5.log")

FX_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}
METALS = {"XAU", "XAG", "XPD", "XPT"}
CRYPTO_BASES = {"BTC", "ETH", "SOL", "XRP", "LTC", "BCH", "ADA", "DOGE",
                "DOT", "LINK", "AVAX", "MATIC", "BNB", "TRX", "XLM"}

# Common MT5 trade-server retcodes -> human text (log readability)
RETCODE_TEXT = {
    10004: "requote", 10006: "request rejected", 10008: "order placed",
    10009: "done", 10010: "partially done", 10013: "invalid request",
    10014: "invalid volume", 10015: "invalid price", 10016: "invalid stops",
    10019: "not enough money", 10021: "prices changed", 10024: "too many requests",
    10027: "autotrading disabled in terminal", 10028: "request locked",
    10030: "unsupported filling mode", 10031: "no connection",
    10036: "position closed already",
}


def _tf():
    return mt5.TIMEFRAME_H1 if config.BAR_MINUTES == 60 else mt5.TIMEFRAME_M5


class MT5Connector:
    def __init__(self):
        self.connected = False
        self._symbol_map: Dict[str, str] = {}      # canonical -> broker name
        self._filling_cache: Dict[str, int] = {}
        self._hedging: Optional[bool] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self, retries: int = 3, wait_s: int = 5) -> bool:
        if not MT5_AVAILABLE:
            raise RuntimeError(
                "MetaTrader5 package unavailable (Windows + MT5 terminal required). "
                "See docs/SETUP.md.")
        kwargs = {}
        if config.mt5.terminal_path:
            kwargs["path"] = config.mt5.terminal_path
        if config.mt5.login:
            kwargs.update(login=config.mt5.login,
                          password=config.mt5.password,
                          server=config.mt5.server)
        for attempt in range(1, retries + 1):
            if mt5.initialize(**kwargs):
                acc = mt5.account_info()
                if acc is None:
                    logger.warning("MT5 initialized but no account info yet...")
                self.connected = True
                self._hedging = bool(acc and acc.margin_mode ==
                                     mt5.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
                term = mt5.terminal_info()
                logger.info(f"MT5 connected: account={getattr(acc, 'login', '?')} "
                            f"server={getattr(acc, 'server', '?')} "
                            f"mode={'hedging' if self._hedging else 'netting'} "
                            f"build={getattr(term, 'build', '?')}")
                if not getattr(term, "trade_allowed", True):
                    logger.error("Algo trading is DISABLED in the terminal "
                                 "(toolbar 'Algo Trading' button must be green).")
                return True
            err = mt5.last_error()
            logger.error(f"MT5 initialize failed ({attempt}/{retries}): {err}")
            time.sleep(wait_s)
        self.connected = False
        return False

    def shutdown(self):
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False

    def _require(self):
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 unavailable on this machine")
        if not self.connected:
            self.connect()
        return mt5

    # ------------------------------------------------------------------
    # Symbol resolution + classification
    # ------------------------------------------------------------------
    def resolve_symbol(self, canonical: str) -> Optional[str]:
        """Map a canonical name (EURUSD) to the broker's real symbol
        (EURUSD.pro, EURUSDm, EUR/USD ...). Cached after first success."""
        if canonical in self._symbol_map:
            return self._symbol_map[canonical]
        m = self._require()
        exact = m.symbol_info(canonical)
        if exact is not None:
            self._symbol_map[canonical] = canonical
            return canonical
        flat = canonical.replace("/", "").replace("-", "").upper()
        for s in m.symbols_get():
            name = s.name.upper()
            stem = name.replace("/", "").replace("-", "")
            for suffix in (".PRO", ".A", ".M", "M", ".STD", ".RAW", ".ECN", "_", "."):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            if stem == flat:
                self._symbol_map[canonical] = s.name
                logger.info(f"Symbol resolved: {canonical} -> {s.name}")
                return s.name
        logger.warning(f"Symbol {canonical} not found at this broker")
        return None

    def discover_universe(self) -> Dict[str, dict]:
        """Resolve every pool symbol against the broker. Returns
        {canonical: {'broker_symbol', 'asset_class'}} for the ones that exist."""
        out = {}
        pools = (list(config.FOREX_POOL) + list(config.METALS_POOL)
                 + list(config.CRYPTO_POOL) + list(config.INDICES_POOL))
        for canonical in pools:
            broker_name = self.resolve_symbol(canonical)
            if broker_name:
                out[canonical] = {"broker_symbol": broker_name,
                                  "asset_class": self.classify_asset(canonical)}
        logger.info(f"Universe discovered: {len(out)}/{len(pools)} symbols tradable")
        return out

    @staticmethod
    def classify_asset(symbol: str) -> str:
        """canonical symbol -> 'forex' | 'metal' | 'crypto' | 'index'."""
        s = symbol.replace("/", "").replace("-", "").upper()
        for suf in (".PRO", ".A", ".M", ".STD", ".RAW", ".ECN"):
            if s.endswith(suf):
                s = s[: -len(suf)]
        if s[:3] in METALS:
            return "metal"
        if s[:3] in CRYPTO_BASES or s[:4] in CRYPTO_BASES:
            return "crypto"
        if len(s) == 6 and s[:3] in FX_CURRENCIES and s[3:] in FX_CURRENCIES:
            return "forex"
        return "index"

    @staticmethod
    def split_pair(symbol: str) -> Optional[Tuple[str, str]]:
        """Forex/metal/crypto canonical -> (base, quote) for the
        currency-exposure engine. None when it isn't a XXXYYY pair."""
        s = symbol.replace("/", "").upper()
        for suf in (".PRO", ".A", ".M", ".STD", ".RAW", ".ECN"):
            if s.endswith(suf):
                s = s[: -len(suf)]
        if len(s) >= 6:
            return s[:3], s[3:6] if len(s) == 6 else s[3:]
        return None

    def ensure_visible(self, broker_symbol: str) -> bool:
        m = self._require()
        info = m.symbol_info(broker_symbol)
        if info is None:
            return False
        if not info.visible:
            return bool(m.symbol_select(broker_symbol, True))
        return True

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def get_bars(self, symbol: str, count: int = None) -> Optional[pd.DataFrame]:
        """Latest `count` bars, oldest first.
        Columns: timestamp(UTC), open, high, low, close, volume, spread_points.
        NOTE: MT5 timestamps are broker-server time; we treat them as UTC
        consistently everywhere (features, memory, gates) so the brain only
        ever compares like with like."""
        m = self._require()
        broker = self.resolve_symbol(symbol)
        if not broker or not self.ensure_visible(broker):
            return None
        count = count or config.LIVE_BARS_WINDOW
        rates = m.copy_rates_from_pos(broker, _tf(), 0, count)
        if rates is None or len(rates) == 0:
            logger.error(f"{symbol}: no bars ({m.last_error()})")
            return None
        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume", "spread": "spread_points"})
        keep = ["timestamp", "open", "high", "low", "close", "volume", "spread_points"]
        return df[keep].sort_values("timestamp").reset_index(drop=True)

    def get_bars_range(self, symbol: str, start, end) -> Optional[pd.DataFrame]:
        """Historical window for backfills (datetime -> datetime, UTC)."""
        m = self._require()
        broker = self.resolve_symbol(symbol)
        if not broker or not self.ensure_visible(broker):
            return None
        rates = m.copy_rates_range(broker, _tf(), start, end)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume", "spread": "spread_points"})
        keep = ["timestamp", "open", "high", "low", "close", "volume", "spread_points"]
        return df[keep].sort_values("timestamp").reset_index(drop=True)

    def get_tick(self, symbol: str):
        m = self._require()
        broker = self.resolve_symbol(symbol)
        if not broker:
            return None
        return m.symbol_info_tick(broker)

    def get_latest_price(self, symbol: str, side: str = "LONG") -> Optional[float]:
        t = self.get_tick(symbol)
        if t is None:
            return None
        # LONGs are marked at bid (that's what you sell at), shorts at ask
        return float(t.bid if side == "LONG" else t.ask)

    # ------------------------------------------------------------------
    # Symbol specs / costs
    # ------------------------------------------------------------------
    def symbol_specs(self, symbol: str) -> Optional[dict]:
        m = self._require()
        broker = self.resolve_symbol(symbol)
        if not broker:
            return None
        info = m.symbol_info(broker)
        if info is None:
            return None
        return {
            "broker_symbol": broker,
            "asset_class": self.classify_asset(symbol),
            "digits": info.digits,
            "point": info.point,
            "spread_points": info.spread,
            "tick_size": info.trade_tick_size,
            "tick_value": info.trade_tick_value,        # per 1.0 lot, account currency
            "contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_step": info.volume_step,
            "volume_max": info.volume_max,
            "stops_level_points": info.trade_stops_level,
            "swap_long": info.swap_long,                # points per night
            "swap_short": info.swap_short,
            "swap_triple_day": info.swap_mode,          # informational
            "trade_mode": info.trade_mode,
            "currency_profit": info.currency_profit,
            "currency_margin": info.currency_margin,
        }

    def spread_pct(self, symbol: str) -> Optional[float]:
        """Current spread as a fraction of price (e.g. 0.0002 = 2 bps)."""
        specs = self.symbol_specs(symbol)
        t = self.get_tick(symbol)
        if not specs or t is None or t.ask <= 0:
            return None
        return (specs["spread_points"] * specs["point"]) / float(t.ask)

    # ------------------------------------------------------------------
    # Account / positions
    # ------------------------------------------------------------------
    def account(self) -> Optional[dict]:
        m = self._require()
        a = m.account_info()
        if a is None:
            return None
        return {"login": a.login, "server": a.server, "currency": a.currency,
                "balance": float(a.balance), "equity": float(a.equity),
                "margin_free": float(a.margin_free), "margin": float(a.margin),
                "leverage": a.leverage, "hedging": self._hedging,
                "trade_allowed": bool(a.trade_allowed)}

    def positions(self, ours_only: bool = True) -> List[dict]:
        """Open positions. ours_only filters by magic range (our bot only)."""
        m = self._require()
        raw = m.positions_get()
        out = []
        for p in raw or []:
            if ours_only and not (config.MAGIC_BASE <= p.magic < config.MAGIC_BASE + 1000):
                continue
            out.append({
                "ticket": p.ticket, "symbol": p.symbol,
                "side": "LONG" if p.type == mt5.POSITION_TYPE_BUY else "SHORT",
                "volume": float(p.volume), "entry_price": float(p.price_open),
                "sl": float(p.sl), "tp": float(p.tp), "magic": p.magic,
                "profit": float(p.profit), "swap": float(p.swap),
                "time": pd.to_datetime(p.time, unit="s", utc=True),
                "comment": p.comment,
            })
        return out

    _POOL_ORDER: Optional[list] = None

    @classmethod
    def _pool_order(cls) -> list:
        if cls._POOL_ORDER is None:
            cls._POOL_ORDER = list(dict.fromkeys(
                list(config.FOREX_POOL) + list(config.METALS_POOL)
                + list(config.CRYPTO_POOL) + list(config.INDICES_POOL)))
        return cls._POOL_ORDER

    def magic_for(self, symbol: str) -> int:
        """Stable, collision-free per-symbol magic: pool members get
        MAGIC_BASE + pool index (deterministic from config order — never
        Python's hash(), which is salted per process; crc32 mod-900 was
        dropped because EURUSD/GBPUSD collided in testing). Symbols outside
        the pools get MAGIC_BASE + 500 + crc32 % 400."""
        pool = self._pool_order()
        s = symbol.upper()
        if s in pool:
            return config.MAGIC_BASE + pool.index(s)
        import zlib
        return config.MAGIC_BASE + 500 + (zlib.crc32(s.encode()) % 400)

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------
    def volume_for_risk(self, symbol: str, risk_amount: float,
                        stop_distance_price: float) -> float:
        """Lots such that (stop hit) loses ~risk_amount of account currency."""
        specs = self.symbol_specs(symbol)
        if not specs or stop_distance_price <= 0 or risk_amount <= 0:
            return 0.0
        ticks = stop_distance_price / specs["tick_size"]
        loss_per_lot = ticks * specs["tick_value"]
        if loss_per_lot <= 0:
            return 0.0
        lots = risk_amount / loss_per_lot
        return self._round_volume(specs, lots)

    @staticmethod
    def _round_volume(specs: dict, lots: float) -> float:
        step = specs["volume_step"] or 0.01
        lots = math.floor(lots / step) * step
        lots = max(0.0, min(lots, specs["volume_max"]))
        if lots < specs["volume_min"]:
            return 0.0
        return round(lots, 8)

    def _enforce_stops_level(self, specs: dict, side: str, price: float,
                             sl: float, tp: float) -> Tuple[float, float]:
        """Broker requires SL/TP at least trade_stops_level away; nudge out."""
        min_dist = (specs["stops_level_points"] + 1) * specs["point"]
        d = specs["digits"]
        if side == "LONG":
            sl = min(sl, price - min_dist)
            tp = max(tp, price + min_dist)
        else:
            sl = max(sl, price + min_dist)
            tp = min(tp, price - min_dist)
        return round(sl, d), round(tp, d)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def _filling_mode(self, broker_symbol: str) -> int:
        if broker_symbol in self._filling_cache:
            return self._filling_cache[broker_symbol]
        m = self._require()
        info = m.symbol_info(broker_symbol)
        mode = getattr(info, "filling_mode", 0) if info else 0
        # bitmask: 1 = FOK, 2 = IOC; exchanges need RETURN
        if mode & 2:
            chosen = m.ORDER_FILLING_IOC
        elif mode & 1:
            chosen = m.ORDER_FILLING_FOK
        else:
            chosen = m.ORDER_FILLING_RETURN
        self._filling_cache[broker_symbol] = chosen
        return chosen

    def place_market_order(self, symbol: str, side: str, lots: float,
                           sl: float, tp: float, comment: str = "") -> Optional[dict]:
        """Open a position with an attached SL/TP bracket.
        Retries on requote/price-change with a fresh tick (ORDER_FILL_RETRIES).
        Returns {'ticket', 'price', 'volume'} or None."""
        m = self._require()
        broker = self.resolve_symbol(symbol)
        specs = self.symbol_specs(symbol)
        if not broker or not specs or lots <= 0:
            return None
        self.ensure_visible(broker)
        filling = self._filling_mode(broker)

        for attempt in range(1, config.ORDER_FILL_RETRIES + 1):
            tick = m.symbol_info_tick(broker)
            if tick is None:
                return None
            price = float(tick.ask if side == "LONG" else tick.bid)
            sl_r, tp_r = self._enforce_stops_level(specs, side, price, sl, tp)
            request = {
                "action": m.TRADE_ACTION_DEAL,
                "symbol": broker,
                "volume": float(lots),
                "type": m.ORDER_TYPE_BUY if side == "LONG" else m.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl_r, "tp": tp_r,
                "deviation": config.DEVIATION_POINTS,
                "magic": self.magic_for(symbol),
                "comment": (comment or "nexus-v4")[:31],
                "type_time": m.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            result = m.order_send(request)
            if result is None:
                logger.error(f"{symbol} order_send returned None ({m.last_error()})")
                return None
            if result.retcode in (m.TRADE_RETCODE_DONE, m.TRADE_RETCODE_PLACED):
                logger.info(f"{symbol} {side} {lots} lots filled @ {result.price} "
                            f"(ticket {result.order})")
                return {"ticket": result.order, "price": float(result.price),
                        "volume": float(lots)}
            txt = RETCODE_TEXT.get(result.retcode, str(result.retcode))
            logger.error(f"{symbol} order failed ({attempt}/{config.ORDER_FILL_RETRIES}): "
                         f"{txt} ({result.retcode}) {result.comment}")
            if result.retcode == 10030:      # filling not supported: rotate mode
                filling = {m.ORDER_FILLING_IOC: m.ORDER_FILLING_FOK,
                           m.ORDER_FILLING_FOK: m.ORDER_FILLING_RETURN,
                           m.ORDER_FILLING_RETURN: m.ORDER_FILLING_IOC}[filling]
                self._filling_cache[broker] = filling
            if result.retcode in (10019, 10027, 10013, 10014):
                break                         # money / disabled / invalid: retrying won't help
            time.sleep(1)
        return None

    def modify_sltp(self, position_ticket: int, symbol: str,
                    sl: float, tp: float) -> bool:
        """Move a position's SL/TP (ratchet / trailing / breakeven live here)."""
        m = self._require()
        broker = self.resolve_symbol(symbol)
        specs = self.symbol_specs(symbol)
        if not specs:
            return False
        d = specs["digits"]
        request = {
            "action": m.TRADE_ACTION_SLTP,
            "position": position_ticket,
            "symbol": broker,
            "sl": round(sl, d), "tp": round(tp, d),
            "magic": self.magic_for(symbol),
        }
        result = m.order_send(request)
        ok = bool(result and result.retcode in (m.TRADE_RETCODE_DONE, m.TRADE_RETCODE_PLACED))
        if not ok:
            txt = RETCODE_TEXT.get(getattr(result, "retcode", 0), "?")
            logger.error(f"{symbol} SLTP modify failed: {txt} ({getattr(result, 'comment', '')})")
        return ok

    def close_position(self, symbol: str, lots: float = None,
                       ticket: int = None, comment: str = "close") -> bool:
        """Close (fully or partially) with the direction derived from the
        BROKER's position side - never from tracked state. In netting mode
        the opposite deal on the symbol; in hedging mode against the ticket."""
        m = self._require()
        broker = self.resolve_symbol(symbol)
        if not broker:
            return False
        open_pos = [p for p in self.positions(ours_only=False) if p["symbol"] == broker]
        if ticket:
            open_pos = [p for p in open_pos if p["ticket"] == ticket]
        if not open_pos:
            logger.info(f"{symbol}: nothing to close at broker")
            return False
        filling = self._filling_mode(broker)
        all_ok = True
        for pos in open_pos:
            vol = min(lots, pos["volume"]) if lots else pos["volume"]
            vol = self._round_volume(self.symbol_specs(symbol), vol) or pos["volume"]
            tick = m.symbol_info_tick(broker)
            if tick is None:
                return False
            close_type = m.ORDER_TYPE_SELL if pos["side"] == "LONG" else m.ORDER_TYPE_BUY
            price = float(tick.bid if pos["side"] == "LONG" else tick.ask)
            request = {
                "action": m.TRADE_ACTION_DEAL,
                "symbol": broker,
                "volume": float(vol),
                "type": close_type,
                "price": price,
                "deviation": config.DEVIATION_POINTS,
                "magic": self.magic_for(symbol),
                "comment": comment[:31],
                "type_time": m.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            if self._hedging:
                request["position"] = pos["ticket"]
            result = m.order_send(request)
            ok = bool(result and result.retcode in (m.TRADE_RETCODE_DONE, m.TRADE_RETCODE_PLACED))
            if ok:
                logger.info(f"{symbol} closed {vol} lots ({comment}) @ {result.price}")
            else:
                all_ok = False
                txt = RETCODE_TEXT.get(getattr(result, "retcode", 0), "?")
                logger.error(f"{symbol} close failed: {txt} ({getattr(result, 'comment', '')})")
        return all_ok

    # ------------------------------------------------------------------
    def ping(self) -> bool:
        try:
            self._require()
            return mt5.terminal_info() is not None
        except Exception:
            return False
