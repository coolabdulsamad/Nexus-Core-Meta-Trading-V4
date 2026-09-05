"""Backtest engine: one symbol at a time, brain-in-the-loop.

Methodology notes (deliberately honest):
  - Decisions are made on bar-close features; fills happen at the NEXT bar's
    open plus costs. No same-bar entry.
  - Brain recalls use the memory's look-ahead guard (neighbor ts <= decision
    time - MEMORY_MIN_AGE_MINUTES), so every verdict is computable from
    information that genuinely existed at decision time. The encoder's
    z-scaler is fitted on full history (unsupervised, second-order leakage;
    documented).
  - Brain queries are batched per chunk (encode locally, query Qdrant in
    groups of 256) so a multi-year run takes minutes, not days. Some chunk
    queries are wasted on bars that turn out to be inside an open trade —
    harmless and much faster than per-bar round trips.
  - Loss cooldowns (post-stop ban, repeat-loss ban, flat-bars-after-close)
    ARE simulated: they shape live trade flow and must shape the backtest too.
  - Portfolio-level caps (currency risk, max positions) are NOT simulated;
    they only remove trades, so per-symbol trade counts are an upper bound.
    Per-trade economics are intact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qdrant_client.models import FieldCondition, Filter, QueryRequest, Range

from config.settings import config
from src.backtester.costs import CostModel
from src.backtester.trade_sim import simulate_trade
from src.memory.memory_store import collection_name
from src.memory.meta_learner import verdict_from_neighbors
from src.memory.vector_encoder import VectorEncoder
from src.utils.logger import setup_logger

logger = setup_logger("backtest.engine", "logs/backtest.log")

BATCH = 256
CHUNK = 2000  # bars per batching window
ASIA_CCYS = ("AUD", "NZD", "JPY")


@dataclass
class Funnel:
    bars: int = 0
    flat_bars: int = 0
    cooldown_skips: int = 0
    session_ok: int = 0
    spread_ok: int = 0
    brain_called: int = 0
    prob_directional: int = 0
    agreement_ok: int = 0
    quality_ok: int = 0
    adx_ok: int = 0
    tape_confirm_ok: int = 0
    no_chase_ok: int = 0
    tp_worth_spread: int = 0
    entries: int = 0

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class SymbolResult:
    symbol: str
    asset_class: str
    trades: list = field(default_factory=list)
    funnel: Funnel = field(default_factory=Funnel)


def _session_ok(t: pd.Timestamp, symbol: str) -> bool:
    """Trading-session gates from config (timestamps are broker server time,
    treated as UTC throughout the project)."""
    if not config.SESSION_FILTER_ENABLED:
        return True
    h = t.hour + t.minute / 60.0
    if config.DEAD_HOURS_UTC[0] <= h < config.DEAD_HOURS_UTC[1]:
        return False
    minutes_to_21 = 21 * 60 - (t.hour * 60 + t.minute)
    if 0 < minutes_to_21 <= config.ROLLOVER_BLACKOUT_MINUTES:
        return False
    if t.weekday() == 4 and h >= config.FRIDAY_NO_ENTRY_UTC_HOUR:
        return False
    if t.weekday() == 6 and h < 21 + config.SUNDAY_OPEN_BLACKOUT_MINUTES / 60.0:
        return False
    is_asia = symbol[:3] in ASIA_CCYS or symbol[3:] in ASIA_CCYS
    if not is_asia and config.ASIAN_THIN_HOURS_UTC[0] <= h < config.ASIAN_THIN_HOURS_UTC[1]:
        return False
    return True


def _quality_floor(asset_class: str) -> float:
    if asset_class == "crypto":
        return config.CRYPTO_MIN_SIGNAL_QUALITY
    return config.MIN_SIGNAL_QUALITY


def _risk_pct(quality: float) -> float:
    if quality >= config.QUALITY_STRONG:
        return config.RISK_PCT_STRONG
    if quality >= config.QUALITY_MEDIUM:
        return config.RISK_PCT_MEDIUM
    return config.RISK_PCT_WEAK


class SymbolBacktester:
    def __init__(self, symbol: str, asset_class: str, encoder: VectorEncoder,
                 qdrant_client, equity: float = None):
        self.symbol = symbol
        self.asset_class = asset_class
        self.encoder = encoder
        self.qc = qdrant_client
        self.collection = collection_name(asset_class)
        self.equity = equity if equity is not None else config.BACKTEST_INITIAL_CAPITAL
        self.costs = CostModel(asset_class)

    def _batch_verdicts(self, d: pd.DataFrame, positions: list[int]) -> dict:
        """Vectorized brain for a list of positional indices. Returns
        {position: verdict_dict}."""
        rows = d.iloc[positions]
        vecs = self.encoder.transform(rows[list(VectorEncoder.VECTOR_FEATURES)])
        out = {}
        for start in range(0, len(rows), BATCH):
            sub = rows.iloc[start:start + BATCH]
            vsub = vecs[start:start + BATCH]
            reqs = []
            for (_, row), v in zip(sub.iterrows(), vsub):
                cutoff = int((row["timestamp"] - pd.Timedelta(
                    minutes=config.MEMORY_MIN_AGE_MINUTES)).timestamp())
                reqs.append(QueryRequest(
                    query=v.tolist(), limit=config.MEMORY_NEIGHBORS,
                    filter=Filter(must=[FieldCondition(key="ts", range=Range(lte=cutoff))]),
                    with_payload=True))
            responses = self.qc.query_batch_points(
                collection_name=self.collection, requests=reqs)
            for idx, resp in zip(sub.index, responses):
                neigh = [(p.score, p.payload.get("fwd")) for p in resp.points]
                out[idx] = verdict_from_neighbors(neigh)
        return out

    def run(self, data: pd.DataFrame) -> SymbolResult:
        """data: merged bars+features sorted by timestamp with columns
        open/high/low/close, spread_pct, atr_14, adx_14 + VECTOR_FEATURES."""
        res = SymbolResult(self.symbol, self.asset_class)
        funnel = res.funnel
        d = data.reset_index(drop=True)
        d["spread_price"] = d["spread_pct"].clip(lower=0) * d["close"]
        d["spread_med20"] = d["spread_price"].rolling(20, min_periods=5).median()
        d["bar_range"] = d["high"] - d["low"]

        ts_idx = pd.DatetimeIndex(d["timestamp"])
        opens = d["open"].to_numpy(); highs = d["high"].to_numpy()
        lows = d["low"].to_numpy(); closes = d["close"].to_numpy()
        sprs = d["spread_price"].to_numpy()
        atrs = d["atr_14"].to_numpy(); adxs = d["adx_14"].to_numpy()
        vwaps = d["dist_vwap"].to_numpy()
        med20 = d["spread_med20"].to_numpy()
        bar_range = d["bar_range"].to_numpy()
        n = len(d)

        flat_until = 0
        no_entry_until = pd.Timestamp.min   # loss cooldown horizon
        stop_times: list[pd.Timestamp] = []  # recent stop-outs (repeat-loss rule)
        verdicts: dict[int, dict] = {}
        i = 0
        while i < n - 1:
            # refill verdict cache for the window ahead (cheap gates only)
            if i not in verdicts:
                c_end = min(i + CHUNK, n - 1)
                cands = [
                    j for j in range(i, c_end)
                    if np.isfinite(atrs[j]) and atrs[j] > 0
                    and _session_ok(ts_idx[j], self.symbol)
                    and not (np.isfinite(med20[j]) and med20[j] > 0
                             and sprs[j] > config.SPREAD_MAX_MULT_OF_MEDIAN * med20[j])
                ]
                if cands:
                    verdicts.update(self._batch_verdicts(d, cands))
                    funnel.brain_called += len(cands)

            funnel.bars += 1
            if i < flat_until:
                i += 1
                continue
            funnel.flat_bars += 1
            t = ts_idx[i]

            # ---- loss cooldowns ------------------------------------------
            if t < no_entry_until:
                funnel.cooldown_skips += 1
                i += 1
                continue

            if not _session_ok(t, self.symbol):
                i += 1
                continue
            funnel.session_ok += 1
            if not np.isfinite(atrs[i]) or atrs[i] <= 0:
                i += 1
                continue
            if config.SPREAD_FILTER_ENABLED and np.isfinite(med20[i]) \
                    and med20[i] > 0 and sprs[i] > config.SPREAD_MAX_MULT_OF_MEDIAN * med20[i]:
                i += 1
                continue
            funnel.spread_ok += 1

            verdict = verdicts.get(i)
            if verdict is None:
                i += 1
                continue
            prob = verdict["prob"]
            long_sig = prob > config.BUY_THRESHOLD + config.ENTRY_CONVICTION_MARGIN
            short_sig = prob < config.SELL_THRESHOLD - config.ENTRY_CONVICTION_MARGIN
            if not (long_sig or short_sig):
                i += 1
                continue
            funnel.prob_directional += 1
            if verdict["agreement"] < config.MIN_NEIGHBOR_AGREEMENT:
                i += 1
                continue
            funnel.agreement_ok += 1
            quality = verdict["quality"]
            if quality < _quality_floor(self.asset_class):
                i += 1
                continue
            funnel.quality_ok += 1
            # with-trend entries need a real trend; counter moves need STRONG quality
            if config.ENTRY_ADX_MIN > 0 and adxs[i] < config.ENTRY_ADX_MIN \
                    and quality < config.QUALITY_STRONG:
                i += 1
                continue
            funnel.adx_ok += 1

            atr = atrs[i]
            # tape confirmation (tolerance mode: only a STRONG counter-signal blocks)
            if config.ENTRY_BAR_CONFIRM_ENABLED:
                body = closes[i] - opens[i]
                if long_sig and body < -config.ENTRY_BAR_CONFIRM_TOLERANCE_ATR * atr:
                    i += 1
                    continue
                if short_sig and body > config.ENTRY_BAR_CONFIRM_TOLERANCE_ATR * atr:
                    i += 1
                    continue
            if config.ENTRY_VWAP_CONFIRM_ENABLED and np.isfinite(vwaps[i]):
                vwap_atr = vwaps[i] * closes[i] / atr
                if long_sig and vwap_atr < -config.ENTRY_VWAP_CONFIRM_TOLERANCE_ATR:
                    i += 1
                    continue
                if short_sig and vwap_atr > config.ENTRY_VWAP_CONFIRM_TOLERANCE_ATR:
                    i += 1
                    continue
            funnel.tape_confirm_ok += 1

            # no-chase: refuse to enter after an already-huge bar
            if config.ENTRY_NO_CHASE_ENABLED \
                    and bar_range[i] > config.ENTRY_NO_CHASE_MAX_RANGE_ATR * atr:
                i += 1
                continue
            funnel.no_chase_ok += 1

            # the trade must be worth the spread
            tp_dist = config.STOP_ATR_MULT * config.REWARD_RISK_RATIO * atr
            if config.MIN_TP_TO_SPREAD_MULT > 0 \
                    and tp_dist < config.MIN_TP_TO_SPREAD_MULT * sprs[i]:
                i += 1
                continue
            funnel.tp_worth_spread += 1

            # ---- entry at NEXT bar open ----------------------------------
            side = 1 if long_sig else -1
            entry_px = self.costs.fill_price(opens[i + 1], side, "entry", sprs[i + 1])
            slice_cap = min(config.CAPITAL_PER_SYMBOL, self.equity * config.SLICE_PCT_OF_EQUITY)
            risk_usd = slice_cap * _risk_pct(quality)
            stop_dist = config.STOP_ATR_MULT * atr
            notional = risk_usd * entry_px / stop_dist
            # notional caps (live has them, so does the backtest)
            notional = min(notional, slice_cap * config.NOTIONAL_CAP_PCT,
                           config.NOTIONAL_CAP_ABS)
            if notional <= 0 or not np.isfinite(notional):
                i += 1
                continue

            trade = simulate_trade(
                ts=ts_idx, opens=opens, highs=highs, lows=lows, closes=closes,
                spreads=sprs, entry_idx=i + 1, side=side, entry_price=entry_px,
                notional=notional, risk_usd=risk_usd, atr=atr,
                asset_class=self.asset_class, costs=self.costs)
            trade.symbol = self.symbol
            trade.side = "LONG" if side > 0 else "SHORT"
            trade.entry_time = ts_idx[i + 1]
            trade.quality = quality
            trade.prob = prob
            trade.agreement = verdict["agreement"]
            res.trades.append(trade)
            funnel.entries += 1

            # ---- cooldown bookkeeping ------------------------------------
            exit_time = ts_idx[min(trade.exit_idx, n - 1)]
            flat_until = trade.exit_idx + 1 + config.COOLDOWN_AFTER_CLOSE_BARS
            if trade.exit_reason == "stop_loss":
                stop_times.append(exit_time)
                no_entry_until = max(no_entry_until,
                                     exit_time + pd.Timedelta(hours=config.LOSS_COOLDOWN_HOURS))
                recent = [x for x in stop_times
                          if x >= exit_time - pd.Timedelta(days=config.REPEAT_LOSS_WINDOW_DAYS)]
                if len(recent) >= 2:
                    no_entry_until = max(no_entry_until,
                                         exit_time + pd.Timedelta(
                                             hours=config.REPEAT_LOSS_COOLDOWN_HOURS))
            i = trade.exit_idx + 1

        logger.info(f"{self.symbol}: {funnel.entries} trades / "
                    f"{funnel.brain_called} brain calls")
        return res
