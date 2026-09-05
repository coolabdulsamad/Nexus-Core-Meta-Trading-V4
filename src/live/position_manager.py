"""
src/live/position_manager.py
================================================================
The live exit stack - a faithful port of the backtester's trade_sim.py
exit rules, driven by tick prices instead of bar high/lows.

Pure core: evaluate() takes a ManagedPosition + current price + time and
returns a list of ACTIONS. No MT5, no DB, no network - the whole exit
stack is unit-testable offline. live_trader applies the actions through
the connector (real) or the virtual book (DRY_RUN).

Rule order mirrors trade_sim.py exactly (order matters):
  0. virtual SL/TP (DRY_RUN only - real positions have broker-side brackets)
  1. scale-outs: 1/3 of INITIAL volume at +SCALE_OUT_1_ATR, another at +2
  2. ratchet: peak >= PROFIT_RATCHET_ATR -> stop locks entry + RATCHET_LOCK_ATR
  3. trailing: peak >= TRAILING_STOP_ACTIVATE_ATR -> stop trails the peak
  4. retracement exit: armed at RETRACEMENT_ARM_ATR, keeps RETRACEMENT_KEEP_PCT
  5. time partial: after TIME_PARTIAL_BARS bars with < TIME_PARTIAL_PROFIT_ATR
  6. time stop: hard exit at TIME_LIMIT_BARS
  7. Friday flatten: forex/metal at >= FRIDAY_FLATTEN_UTC_HOUR
  8. flip exit / flip tighten: the brain re-judges the position on each new
     bar (passed in as flip_verdict; None on non-decision cycles)

Live vs backtest differences (documented, deliberate):
- peaks are tracked on TICKS, not bar extremes -> scale-outs/ratchets fire
  at the real touch, not up to an hour late.
- bars_held counts WHOLE hours since entry (1h bars).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

from config.settings import config
from src.live.state import parse_iso

# action kinds
CLOSE_ALL = "close_all"
CLOSE_VOLUME = "close_volume"     # payload: lots
SET_SL = "set_sl"                 # payload: new stop price


@dataclass
class ManagedPosition:
    ticket: int                   # broker ticket; NEGATIVE = DRY_RUN virtual
    symbol: str                   # canonical
    asset_class: str
    side: str                     # LONG | SHORT
    entry_price: float
    entry_time: str               # iso
    initial_volume: float
    volume: float                 # current open volume
    atr: float                    # ATR at entry (the R yardstick)
    sl: float
    tp: float
    peak_price: float
    trough_price: float
    scaled_1: bool = False
    scaled_2: bool = False
    time_partialed: bool = False
    flip_tightened: bool = False
    quality: float = 0.0
    prob: float = 0.5
    agreement: float = 0.0
    memory_n: int = 0
    regime: str = "unknown"
    spread_pct_at_entry: float = 0.0
    risk_usd: float = 0.0         # account-ccy loss if the initial stop fills
    realized_pnl: float = 0.0     # DRY_RUN: accumulated partial-close pnl
    dry_run: bool = True

    @property
    def sign(self) -> int:
        return 1 if self.side == "LONG" else -1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ManagedPosition":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def bars_held(pos: ManagedPosition, now: datetime) -> int:
    entry = parse_iso(pos.entry_time)
    if entry is None:
        return 0
    return max(0, int((now - entry).total_seconds() // 3600))


def evaluate(pos: ManagedPosition, price: float, now: datetime,
             flip_verdict: Optional[dict] = None) -> list[tuple]:
    """The exit stack. Returns [(kind, payload, reason), ...] in firing
    order. Mutates NOTHING - the caller applies actions and updates pos."""
    actions: list[tuple] = []
    if pos.atr <= 0 or price <= 0:
        return actions

    s = pos.sign
    atr = pos.atr
    # tick-based extremes
    pos_peak = max(pos.peak_price, price) if s > 0 else min(pos.peak_price, price)
    peak_atr = s * (pos_peak - pos.entry_price) / atr
    open_profit_atr = s * (price - pos.entry_price) / atr
    held = bars_held(pos, now)
    pdp = config.ENABLE_PROFIT_DRAWDOWN_PROTECTION

    def improves(new_sl: float) -> bool:
        return (new_sl > pos.sl) if s > 0 else (pos.sl == 0.0 or new_sl < pos.sl)

    # ---- 0. virtual SL/TP (DRY_RUN book only) ---------------------------
    if pos.dry_run:
        if s > 0:
            if pos.sl > 0 and price <= pos.sl:
                return [(CLOSE_ALL, None, "stop_loss")]
            if pos.tp > 0 and price >= pos.tp:
                return [(CLOSE_ALL, None, "take_profit")]
        else:
            if pos.sl > 0 and price >= pos.sl:
                return [(CLOSE_ALL, None, "stop_loss")]
            if pos.tp > 0 and price <= pos.tp:
                return [(CLOSE_ALL, None, "take_profit")]

    # ---- 1. scale-outs ---------------------------------------------------
    if config.SCALE_OUT_ENABLED:
        if not pos.scaled_1 and peak_atr >= config.SCALE_OUT_1_ATR:
            actions.append((CLOSE_VOLUME,
                            config.SCALE_OUT_PCT * pos.initial_volume,
                            "scale_out_1"))
        if pos.scaled_1 and not pos.scaled_2 \
                and peak_atr >= config.SCALE_OUT_2_ATR:
            actions.append((CLOSE_VOLUME,
                            min(config.SCALE_OUT_PCT * pos.initial_volume,
                                pos.volume),
                            "scale_out_2"))

    # ---- 2. ratchet ------------------------------------------------------
    if pdp and peak_atr >= config.PROFIT_RATCHET_ATR:
        lock = pos.entry_price + s * config.RATCHET_LOCK_ATR * atr
        if improves(lock):
            actions.append((SET_SL, lock, "ratchet"))

    # ---- 3. trailing ------------------------------------------------------
    if pdp and peak_atr >= config.TRAILING_STOP_ACTIVATE_ATR:
        trail = pos_peak - s * config.TRAILING_STOP_DISTANCE_ATR * atr
        if improves(trail):
            actions.append((SET_SL, trail, "trailing"))

    # ---- 4. retracement exit ---------------------------------------------
    if pdp and peak_atr >= config.RETRACEMENT_ARM_ATR \
            and open_profit_atr < config.RETRACEMENT_KEEP_PCT * peak_atr:
        return actions + [(CLOSE_ALL, None, "retracement")]

    # ---- 5. time partial ---------------------------------------------------
    if config.ENABLE_TIME_PARTIAL and not pos.time_partialed \
            and held >= config.TIME_PARTIAL_BARS \
            and open_profit_atr < config.TIME_PARTIAL_PROFIT_ATR:
        actions.append((CLOSE_VOLUME, pos.volume / 2.0, "time_partial"))

    # ---- 6. hard time stop -------------------------------------------------
    if held >= config.TIME_LIMIT_BARS:
        return actions + [(CLOSE_ALL, None, "time_stop")]

    # ---- 7. Friday flatten -------------------------------------------------
    if config.WEEKEND_FLAT_ENABLED and pos.asset_class in ("forex", "metal") \
            and now.weekday() == 4 and now.hour >= config.FRIDAY_FLATTEN_UTC_HOUR:
        return actions + [(CLOSE_ALL, None, "friday_flatten")]

    # ---- 8. flip exit / tighten (brain re-judged on a new bar) ------------
    if flip_verdict is not None:
        prob = float(flip_verdict.get("prob", 0.5))
        margin = config.ENTRY_CONVICTION_MARGIN
        opposes = (s > 0 and prob < config.SELL_THRESHOLD - margin) or \
                  (s < 0 and prob > config.BUY_THRESHOLD + margin)
        if opposes:
            if open_profit_atr >= config.FLIP_EXIT_PROFIT_ATR:
                return actions + [(CLOSE_ALL, None, "flip_exit")]
            if config.FLIP_TIGHTEN_UNDERWATER and not pos.flip_tightened:
                tight = pos.entry_price - s * 1.0 * atr
                if improves(tight):
                    actions.append((SET_SL, tight, "flip_tighten"))

    return actions
