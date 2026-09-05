"""Execution cost model for the backtester.

Deliberately pessimistic. MT5 bars are bid-based, so:

  - LONG entry crosses the ask  -> pays the full bar spread once at entry.
  - LONG exit  sells at bid     -> no spread (already embedded), slippage only.
  - SHORT entry sells at bid    -> no spread at entry.
  - SHORT exit  crosses the ask -> pays the full bar spread at exit.

Slippage (SLIPPAGE_BPS) is charged on EVERY fill, entries and exits, and
doubles on stop-loss fills (stops slip more in fast tape).

Swap is charged on notional at each 21:00 UTC rollover while a position is
open, tripled on Wednesdays for forex/metals (industry convention), daily
for broker crypto CFDs. Rates are conservative per-class fractions of
notional, calibrated from the XM symbols captured in Phase 1 self-test
(e.g. BTCUSD ~ -3235 points/day ~ -0.03%/day of notional).
"""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import config

# Fraction of notional charged per rollover day, both sides (negative = cost).
SWAP_DAILY_PCT = {
    "forex": -0.00005,   # ~ -6 pts/day on EURUSD
    "metal": -0.00010,
    "crypto": -0.00030,  # XM BTCUSD runs about -0.03%/day
}

ROLLOVER_HOUR_UTC = 21
TRIPLE_SWAP_WEEKDAY = 2  # Wednesday


@dataclass
class Fill:
    price: float
    cost_usd: float  # spread+slippage+commission already in account ccy terms


class CostModel:
    """Per-symbol cost engine. Prices in quote ccy, PnL approximated in
    account ccy (documented approximation: cross-rate conversion ignored,
    <5% error on majors; live sizing uses exact broker tick values)."""

    def __init__(self, asset_class: str):
        self.asset_class = asset_class
        self.slip = config.SLIPPAGE_BPS / 10000.0
        self.commission_per_lot = config.DEFAULT_COMMISSION_PER_LOT
        self.swap_daily = SWAP_DAILY_PCT.get(asset_class, -0.0001)

    # ---- fills -----------------------------------------------------------
    def fill_price(self, mid: float, side: int, action: str,
                   spread_price: float, is_stop: bool = False) -> float:
        """side: +1 long, -1 short. action: 'entry' | 'exit'."""
        slip = self.slip * (2.0 if is_stop else 1.0)
        if side > 0:  # long
            if action == "entry":
                return mid + spread_price + mid * slip          # buy at ask
            return mid - mid * slip                             # sell at bid
        else:       # short
            if action == "entry":
                return mid - mid * slip                         # sell at bid
            return mid + spread_price + mid * slip              # buy at ask

    def commission_usd(self, notional_usd: float, n_sides: int = 2) -> float:
        if not self.commission_per_lot:
            return 0.0
        lots = notional_usd / 100000.0  # approx: 1 standard lot
        return lots * self.commission_per_lot * n_sides

    # ---- swap ------------------------------------------------------------
    def swap_for_day(self, notional_usd: float, weekday: int) -> float:
        """Cost (negative USD) for holding through one rollover."""
        rate = self.swap_daily
        if self.asset_class in ("forex", "metal") and weekday == TRIPLE_SWAP_WEEKDAY:
            rate *= 3.0
        return notional_usd * rate
