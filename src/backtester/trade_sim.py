"""Single-trade simulation with the full v3.6 exit stack.

Pure function over numpy arrays — no DB, no Qdrant — so it is unit-testable
in isolation. Called by the engine once an entry decision is made; the
entry fill has ALREADY happened (entry_idx bar's open).

Exit stack (v3.6, order matters — checked every bar after entry):
  1. hard SL / TP, intrabar via high/low; if both touched in one bar the
     STOP is assumed hit first (conservative)
  2. scale-outs: 1/3 at +SCALE_OUT_1_ATR, another 1/3 at +SCALE_OUT_2_ATR
  3. ratchet: once peak >= PROFIT_RATCHET_ATR, stop locks entry + RATCHET_LOCK_ATR
  4. trailing: once peak >= TRAILING_STOP_ACTIVATE_ATR, stop trails the peak
  5. retracement exit: armed at RETRACEMENT_ARM_ATR, keeps RETRACEMENT_KEEP_PCT
  6. time partial: after TIME_PARTIAL_BARS with < TIME_PARTIAL_PROFIT_ATR
     open profit, close half
  7. time stop: hard exit at TIME_LIMIT_BARS
  8. Friday flatten: forex/metal positions closed at >= FRIDAY_FLATTEN_UTC_HOUR
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import config
from src.backtester.costs import CostModel, ROLLOVER_HOUR_UTC


@dataclass
class TradeResult:
    exit_idx: int
    exit_price: float
    exit_reason: str
    pnl_usd: float
    r_multiple: float
    bars_held: int
    mae_atr: float
    mfe_atr: float
    swap_paid: float
    commission_paid: float
    scale_outs: int = 0
    extras: dict = field(default_factory=dict)


def simulate_trade(ts: pd.DatetimeIndex, opens: np.ndarray, highs: np.ndarray,
                   lows: np.ndarray, closes: np.ndarray, spreads: np.ndarray,
                   entry_idx: int, side: int, entry_price: float,
                   notional: float, risk_usd: float, atr: float,
                   asset_class: str, costs: CostModel) -> TradeResult:
    """Walk bars forward from entry_idx (fill ALREADY done at entry_price)
    until an exit rule fires."""
    n = len(closes)
    i = entry_idx
    s = side  # +1 long, -1 short

    sl = entry_price - s * config.STOP_ATR_MULT * atr
    tp = entry_price + s * config.STOP_ATR_MULT * config.REWARD_RISK_RATIO * atr

    frac = 1.0                      # remaining position fraction
    realized = 0.0                  # realized pnl from partials, USD
    scale_outs = 0
    scaled_1 = scaled_2 = False
    time_partialed = False
    swap_paid = 0.0                 # positive = cost paid
    peak_price = entry_price        # best favorable extreme
    trough_price = entry_price      # worst adverse extreme
    last_rollover_date = None
    commission_paid = costs.commission_usd(notional, n_sides=2)
    pdp = config.ENABLE_PROFIT_DRAWDOWN_PROTECTION

    def frac_pnl(px: float, f: float) -> float:
        return f * notional * (px - entry_price) * s / entry_price

    exit_idx, exit_price, exit_reason = n - 1, closes[-1], "end_of_data"

    j = i + 1
    while j < n:
        h, l, c = highs[j], lows[j], closes[j]
        t = ts[j]
        spr = spreads[j]            # spread in PRICE units for this bar

        if s > 0:
            peak_price = max(peak_price, h)
            trough_price = min(trough_price, l)
        else:
            peak_price = min(peak_price, l)
            trough_price = max(trough_price, h)
        peak_atr = s * (peak_price - entry_price) / atr

        # ---- swap at rollover (21:00 UTC) --------------------------------
        if config.SWAP_MODEL_IN_BACKTEST and t.hour >= ROLLOVER_HOUR_UTC \
                and t.date() != last_rollover_date:
            last_rollover_date = t.date()
            swap_paid -= costs.swap_for_day(notional * frac, t.weekday())
            # swap_for_day returns negative (a cost); swap_paid accumulates
            # the positive amount paid

        # ---- 1. hard SL / TP (stop loses ties) ---------------------------
        hit_sl = (l <= sl) if s > 0 else (h >= sl)
        hit_tp = (h >= tp) if s > 0 else (l <= tp)
        if hit_sl or hit_tp:
            if hit_sl:
                px = costs.fill_price(sl, s, "exit", spr, is_stop=True)
                exit_idx, exit_price, exit_reason = j, px, "stop_loss"
            else:
                px = costs.fill_price(tp, s, "exit", spr, is_stop=False)
                exit_idx, exit_price, exit_reason = j, px, "take_profit"
            realized += frac_pnl(px, frac)
            frac = 0.0
            break

        # ---- 2. scale-outs ------------------------------------------------
        if config.SCALE_OUT_ENABLED:
            if not scaled_1 and peak_atr >= config.SCALE_OUT_1_ATR:
                px = costs.fill_price(entry_price + s * config.SCALE_OUT_1_ATR * atr,
                                      s, "exit", spr)
                realized += frac_pnl(px, config.SCALE_OUT_PCT)
                frac -= config.SCALE_OUT_PCT
                scaled_1 = True
                scale_outs += 1
            if scaled_1 and not scaled_2 and peak_atr >= config.SCALE_OUT_2_ATR:
                f2 = min(config.SCALE_OUT_PCT, frac)
                px = costs.fill_price(entry_price + s * config.SCALE_OUT_2_ATR * atr,
                                      s, "exit", spr)
                realized += frac_pnl(px, f2)
                frac -= f2
                scaled_2 = True
                scale_outs += 1

        # ---- 3. ratchet ---------------------------------------------------
        if pdp and peak_atr >= config.PROFIT_RATCHET_ATR:
            lock = entry_price + s * config.RATCHET_LOCK_ATR * atr
            sl = max(sl, lock) if s > 0 else min(sl, lock)

        # ---- 4. trailing ---------------------------------------------------
        if pdp and peak_atr >= config.TRAILING_STOP_ACTIVATE_ATR:
            trail = peak_price - s * config.TRAILING_STOP_DISTANCE_ATR * atr
            sl = max(sl, trail) if s > 0 else min(sl, trail)

        # ---- 5. retracement exit ------------------------------------------
        if pdp and peak_atr >= config.RETRACEMENT_ARM_ATR:
            open_profit_atr = s * (c - entry_price) / atr
            if open_profit_atr < config.RETRACEMENT_KEEP_PCT * peak_atr:
                px = costs.fill_price(c, s, "exit", spr)
                realized += frac_pnl(px, frac)
                exit_idx, exit_price, exit_reason = j, px, "retracement"
                frac = 0.0
                break

        # ---- 6. time partial ----------------------------------------------
        bars_held = j - i
        if config.ENABLE_TIME_PARTIAL and not time_partialed \
                and bars_held >= config.TIME_PARTIAL_BARS:
            open_profit_atr = s * (c - entry_price) / atr
            if open_profit_atr < config.TIME_PARTIAL_PROFIT_ATR:
                px = costs.fill_price(c, s, "exit", spr)
                realized += frac_pnl(px, frac / 2.0)
                frac /= 2.0
                time_partialed = True

        # ---- 7. hard time stop ---------------------------------------------
        if bars_held >= config.TIME_LIMIT_BARS:
            px = costs.fill_price(c, s, "exit", spr)
            realized += frac_pnl(px, frac)
            exit_idx, exit_price, exit_reason = j, px, "time_stop"
            frac = 0.0
            break

        # ---- 8. Friday flatten ---------------------------------------------
        if config.WEEKEND_FLAT_ENABLED and asset_class in ("forex", "metal") \
                and t.weekday() == 4 and t.hour >= config.FRIDAY_FLATTEN_UTC_HOUR:
            px = costs.fill_price(c, s, "exit", spr)
            realized += frac_pnl(px, frac)
            exit_idx, exit_price, exit_reason = j, px, "friday_flatten"
            frac = 0.0
            break

        j += 1
    else:
        px = costs.fill_price(closes[-1], s, "exit", spreads[-1])
        realized += frac_pnl(px, frac)
        exit_price = px
        frac = 0.0

    if frac > 0:  # defensive: always leave flat
        px = costs.fill_price(closes[exit_idx], s, "exit", spreads[exit_idx])
        realized += frac_pnl(px, frac)

    pnl = realized - swap_paid - commission_paid
    bars_held = exit_idx - i
    mae = max(0.0, -s * (trough_price - entry_price) / atr)
    mfe = max(0.0, s * (peak_price - entry_price) / atr)
    return TradeResult(
        exit_idx=exit_idx, exit_price=exit_price, exit_reason=exit_reason,
        pnl_usd=pnl, r_multiple=pnl / risk_usd if risk_usd > 0 else 0.0,
        bars_held=bars_held, mae_atr=float(mae), mfe_atr=float(mfe),
        swap_paid=swap_paid, commission_paid=commission_paid,
        scale_outs=scale_outs,
    )
