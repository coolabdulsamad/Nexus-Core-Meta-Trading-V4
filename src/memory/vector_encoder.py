"""
Vector encoder: feature rows -> fixed-width, L2-normalized memory vectors.

Pipeline (per asset class, fitted on that class's own history):
    18 scale-free features -> z-score (StandardScaler) -> clip to +/-5
    -> PCA(n) if we ever grow past n features -> L2 normalize

Why these choices:
- ONLY scale-free features enter the vector. Raw MACD values, price
  levels, ATR in price units etc. would make EURUSD (1.10) and USDJPY
  (150.00) live in different universes. Everything below is a ratio,
  a percentage, an ATR multiple, a z-score or a cycle encoding.
- Z-scoring is fitted per asset class so crypto's volatility regime
  doesn't squash forex's (a 0.3% daily move is noise for BTC, a big
  day for EURCHF).
- Clipping at +/-5 sigma keeps flash-crash bars from bending the
  whole space.
- PCA kicks in only when the feature count exceeds PCA_COMPONENTS
  (it was built for the Alpaca edition's 60+ features; with 18 inputs
  it is a no-op passthrough and the vector stays 18-wide - which is
  exactly what the Qdrant collection is sized from).

Encoders are persisted to models/encoder_<class>.pkl. The LIVE trader
and the backtester must load the SAME encoder the memory was built
with - never refit on a different window.
"""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.settings import config
from src.utils.logger import setup_logger

logger = setup_logger("VectorEncoder", "logs/memory.log")

MODELS_DIR = "models"


class VectorEncoder:
    VECTOR_FEATURES = (
        "rsi_14", "bb_pct_b", "bb_width", "atr_pct",
        "volume_profile_ratio", "vol_z",
        "ret_1", "ret_3", "ret_12", "adx_14",
        "dist_sma50", "dist_sma200", "dist_vwap",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "spread_pct",
    )
    ZSCORE_CLIP = 5.0

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components or config.PCA_COMPONENTS
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.output_dim_: Optional[int] = None

    # ------------------------------------------------------------------
    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.VECTOR_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"feature frame missing columns: {missing}")
        return df[list(self.VECTOR_FEATURES)].astype(float).to_numpy()

    def fit(self, df: pd.DataFrame) -> "VectorEncoder":
        X = self.scaler.fit_transform(self._matrix(df))
        X = np.clip(X, -self.ZSCORE_CLIP, self.ZSCORE_CLIP)
        if X.shape[1] >= self.n_components and len(X) > self.n_components:
            self.pca = PCA(n_components=self.n_components, random_state=42)
            self.pca.fit(X)
            logger.info("PCA fitted: %d -> %d dims (%.1f%% variance kept)",
                        X.shape[1], self.n_components,
                        100 * float(np.sum(self.pca.explained_variance_ratio_)))
        else:
            logger.info("PCA skipped: %d features < %d components - "
                        "vector stays %d-wide", X.shape[1],
                        self.n_components, X.shape[1])
        self.output_dim_ = (self.n_components if self.pca is not None
                            else X.shape[1])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.output_dim_ is None:
            raise RuntimeError("encoder not fitted/loaded")
        X = self.scaler.transform(self._matrix(df))
        X = np.clip(X, -self.ZSCORE_CLIP, self.ZSCORE_CLIP)
        if self.pca is not None:
            X = self.pca.transform(X)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (X / norms).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        logger.info("Encoder saved -> %s (dim=%d)", path, self.output_dim_)

    @classmethod
    def load(cls, path: str) -> "VectorEncoder":
        enc = joblib.load(path)
        if enc.output_dim_ is None:
            raise RuntimeError(f"encoder at {path} was never fitted")
        return enc

    @classmethod
    def path_for(cls, asset_class: str) -> str:
        return os.path.join(MODELS_DIR, f"encoder_{asset_class}.pkl")
