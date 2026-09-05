"""
The CBR brain's judge: neighbors -> a trading verdict.

Port of the Alpaca v3.6.5 decision core, kept deliberately simple and
FULLY testable without Qdrant (verdict_from_neighbors is pure math):

1. Keep only neighbors with cosine similarity >= MIN_NEIGHBOR_SIMILARITY.
2. Weight each by similarity^2 (close analogies count much more).
3. prob_up = sigmoid(0.5 * t), where t is the weighted mean forward
   return divided by its standard error. The t-stat form makes the
   probability SCALE-FREE: it auto-calibrates across forex (returns in
   0.1%) and crypto (returns in %) without a hand-tuned gain.
4. agreement = similarity-weighted fraction of neighbors whose outcome
   sign matches the verdict's direction.
5. quality = agreement x depth x sim_boost
   - depth     = min(1, n_kept / QUALITY_MEMORY_REF_N)  (a young memory
                 is a weak memory - the v3.6 memory-depth scaling)
   - sim_boost = 0.5 + 0.5 * (mean_sim mapped 0.50..0.95 -> 0..1)
   The 0.35/0.45 gates and the 0.48-0.52 HOLD zone are applied by the
   CALLER (trader/backtester), exactly like v3.6.

Verdict dict: prob, direction (LONG/SHORT/HOLD), quality, agreement,
n_kept, mean_sim. HOLD is decided here only for "no neighbors".
"""

from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.settings import config
from src.utils.logger import setup_logger
from src.memory.memory_store import get_qdrant, recall
from src.memory.vector_encoder import VectorEncoder

logger = setup_logger("MetaLearner", "logs/memory.log")

_SIGMOID_GAIN = 0.5        # t-stat -> probability; Phase 7 calibrates this
_SIM_MAP_LO, _SIM_MAP_HI = 0.50, 0.95  # mean-sim -> sim_boost range


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -60.0, 60.0))   # exp() overflow guard
    return 1.0 / (1.0 + np.exp(-x))


def verdict_from_neighbors(
    neighbors: Sequence[Tuple[float, float, int, str]],
) -> Optional[Dict]:
    """
    neighbors: [(similarity, forward_return, ts, symbol), ...]
        (forward_return = the active label; None entries are skipped)
    Returns the verdict dict, or None when nothing clears the floor.
    """
    keep = [(s, r) for s, r, _, _ in neighbors
            if s >= config.MIN_NEIGHBOR_SIMILARITY
            and r is not None and np.isfinite(r)]
    if not keep:
        return None

    sims = np.array([k[0] for k in keep])
    rets = np.array([k[1] for k in keep])
    w = sims ** 2
    w_sum = float(w.sum())
    if w_sum <= 0:
        return None

    wmean = float((w * rets).sum() / w_sum)
    wvar = float((w * (rets - wmean) ** 2).sum() / w_sum)
    wstd = float(np.sqrt(max(wvar, 0.0)))

    # effective sample size (Kish) - 100 near-duplicates are not 100 cases
    n_eff = float(w_sum ** 2 / max(float((w ** 2).sum()), 1e-12))

    t_stat = wmean / (wstd / np.sqrt(n_eff) + 1e-12)
    prob = float(_sigmoid(_SIGMOID_GAIN * t_stat))

    direction = "LONG" if prob > 0.5 else "SHORT"
    agree_sign = 1.0 if wmean >= 0 else -1.0
    agreement = float((w * (np.sign(rets) == agree_sign)).sum() / w_sum)

    mean_sim = float(sims.mean())
    sim01 = float(np.clip((mean_sim - _SIM_MAP_LO)
                          / (_SIM_MAP_HI - _SIM_MAP_LO), 0.0, 1.0))
    sim_boost = 0.5 + 0.5 * sim01
    depth = min(1.0, len(keep) / config.QUALITY_MEMORY_REF_N)
    quality = agreement * depth * sim_boost

    return {
        "prob": prob,
        "direction": direction,
        "quality": float(quality),
        "agreement": agreement,
        "n_kept": len(keep),
        "n_eff": round(n_eff, 1),
        "mean_sim": mean_sim,
        "wmean_fwd": wmean,
        "t_stat": float(t_stat),
    }


class Brain:
    """
    Live/historical interface: feature row -> verdict.

    Loads the encoder the memory was BUILT with (never refits) and
    recalls only look-ahead-safe neighbors.
    """

    def __init__(self, asset_class: str):
        self.asset_class = asset_class
        self.encoder = VectorEncoder.load(VectorEncoder.path_for(asset_class))
        self.client = get_qdrant()

    def predict(self, feature_row: pd.Series,
                asof: Optional[datetime] = None) -> Optional[Dict]:
        df = feature_row.to_frame().T
        vector = self.encoder.transform(df)[0]
        neighbors = recall(self.client, self.asset_class, vector, asof=asof)
        v = verdict_from_neighbors(neighbors)
        if v is None:
            logger.debug("%s: no neighbors cleared the similarity floor",
                         self.asset_class)
        return v
