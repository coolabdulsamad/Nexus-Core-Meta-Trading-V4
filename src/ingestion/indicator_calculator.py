"""
src/ingestion/indicator_calculator.py
Feature Engineering v3 (V4 edition) - pure pandas/numpy, NO TA-Lib.

Same state description the Alpaca edition's brain traded on, plus two V4
features that matter for forex/crypto:
- dow_sin / dow_cos: day-of-week encoding (forex has a weekly rhythm -
  Monday gaps, Wednesday triple swap, Friday fade)
- spread_pct: the bar's spread as a fraction of price (cost regime state)

All vector features are SCALE-FREE (ratios, %, ATR units, z-scores) so a
EURUSD state and a BTCUSD state are comparable in memory.

Wilder smoothing is used for RSI/ATR/ADX (platform-standard definitions).
The full-frame dropna() warm-up behaviour of the Alpaca edition is kept:
callers must pass the FULL bar window, never a slice (that was the v3.2
"killed 100% of entries" bug).
"""
import numpy as np
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("IndicatorCalculator", "logs/ingestion.log")


# ---------------------------------------------------------------------------
# Wilder-smoothed primitives
# ---------------------------------------------------------------------------
def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), n)
    loss = _wilder(-delta.clip(upper=0), n)
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high, low, close, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return _wilder(tr, n)


def _adx(high, low, close, n: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _atr(high, low, close, n).replace(0, np.nan)
    plus_di = 100 * _wilder(pd.Series(plus_dm, index=high.index), n) / atr
    minus_di = 100 * _wilder(pd.Series(minus_dm, index=high.index), n) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx, n)


def _regime(row) -> str:
    adx = row['adx_14']
    if pd.isna(adx) or pd.isna(row['dist_sma50']):
        return 'unknown'
    if adx >= 25:
        return 'trend_up' if row['dist_sma50'] > 0 else 'trend_down'
    if adx < 20:
        return 'range'
    return 'transition'


# ---------------------------------------------------------------------------
def calculate_all_indicators(df: pd.DataFrame, point: float = None) -> pd.DataFrame:
    """
    Input columns: timestamp, open, high, low, close, volume
                   (+ optional spread_points - MT5 bars carry it)
    `point`: the symbol's point size from broker specs (e.g. 0.00001 for
    EURUSD, 0.001 for USDJPY, 0.01 for XAUUSD). Required for spread_pct to
    be meaningful; without it spread_pct is 0.0.
    Returns the full feature frame (warm-up rows dropped).
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to indicator calculator.")
        return df

    df = df.sort_values('timestamp').reset_index(drop=True)
    close, high, low, vol = df['close'], df['high'], df['low'], df['volume']

    # --- Classic set ---
    df['rsi_14'] = _rsi(close, 14)
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_upper'], df['bb_lower'] = sma20 + 2 * std20, sma20 - 2 * std20
    df['atr_14'] = _atr(high, low, close, 14)

    vol_ma20 = vol.rolling(20).mean()
    df['volume_profile_ratio'] = vol / vol_ma20

    # --- Scale-free set ---
    band = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    df['bb_pct_b'] = (close - df['bb_lower']) / band
    df['bb_width'] = band / close
    df['atr_pct'] = df['atr_14'] / close

    df['ret_1'] = close.pct_change(1)
    df['ret_3'] = close.pct_change(3)
    df['ret_12'] = close.pct_change(12)

    vol_std20 = vol.rolling(20).std().replace(0, np.nan)
    df['vol_z'] = (vol - vol_ma20) / vol_std20

    df['adx_14'] = _adx(high, low, close, 14)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    atr_safe = df['atr_14'].replace(0, np.nan)
    df['dist_sma50'] = (close - sma50) / atr_safe
    df['dist_sma200'] = (close - sma200) / atr_safe

    # --- Time encodings (hour-of-day + V4 day-of-week) ---
    ts = pd.to_datetime(df['timestamp'])
    minutes = ts.dt.hour * 60 + ts.dt.minute
    df['hour_sin'] = np.sin(2 * np.pi * minutes / 1440)
    df['hour_cos'] = np.cos(2 * np.pi * minutes / 1440)
    dow = ts.dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # --- VWAP (resets daily; on tick volume for forex) ---
    df['date'] = ts.dt.date
    cum_vp = (vol * close).groupby(df['date']).cumsum()
    cum_v = vol.groupby(df['date']).cumsum()
    df['vwap'] = cum_vp / cum_v.replace(0, np.nan)
    df['dist_vwap'] = (close - df['vwap']) / atr_safe

    # --- V4: spread regime (0 when the feed has no spread data) ---
    if 'spread_points' in df.columns and point:
        df['spread_pct'] = (df['spread_points'].fillna(0) * point) / close.replace(0, np.nan)
    else:
        df['spread_pct'] = 0.0

    # --- Regime label ---
    df['regime_label'] = df.apply(_regime, axis=1)

    df = df.drop(columns=['date'])
    df = df.replace([np.inf, -np.inf], np.nan)
    # spread_pct may be legitimately 0; never drop a row for it
    drop_cols = [c for c in df.columns if c not in ('spread_points', 'spread_pct', 'sentiment_score')]
    df = df.dropna(subset=drop_cols)

    logger.info(f"Indicators v3 calculated. {len(df)} rows after warm-up drop.")
    return df
