"""
Brain sanity harness (Phase 3 acceptance test).

Runs against the REAL built memory, no MT5 needed. Per asset class:

1. Memory exists and holds a meaningful number of states.
2. Recall works: the 500 most recent eligible feature rows each get
   k=100 look-ahead-safe neighbors.
3. LOOK-AHEAD GUARD VERIFIED: no recalled neighbor is younger than
   query_time - (FORWARD_HORIZON_HOURS + 1h). One violation = FAIL.
4. The probability distribution is not degenerate (std > 0.005 -
   a brain that outputs ~0.5 for everything has no signal).
5. Quality distribution is printed (p50/p95/max). Reference from the
   Alpaca edition: live max quality ever observed was 0.480, so a max
   in the ~0.35-0.55 band is healthy; a max near 1.0 would mean the
   calibration math is broken.

Usage:
    python scripts/brain_sanity.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config  # noqa: E402
from src.storage.db import get_conn  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402
from src.memory.vector_encoder import VectorEncoder  # noqa: E402
from src.memory.memory_store import (  # noqa: E402
    collection_name, collection_size, get_qdrant, recall,
)
from src.memory.meta_learner import verdict_from_neighbors  # noqa: E402
from src.memory.build_memory import CLASS_POOLS  # noqa: E402

logger = setup_logger("brain_sanity", "logs/memory.log")

N_QUERIES = 500


def check_class(client, asset_class: str, symbols: List[str]) -> Dict:
    result = {"class": asset_class, "passed": [], "failed": []}

    def check(name, ok, detail=""):
        (result["passed"] if ok else result["failed"]).append(
            f"{name} {detail}".strip())
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    # --- 1. memory exists ---
    n_points = collection_size(client, asset_class)
    check("memory populated", n_points > 10000, f"({n_points:,} points)")
    if n_points == 0:
        return result

    # --- 2. recent feature rows as queries ---
    feature_cols = ", ".join(VectorEncoder.VECTOR_FEATURES)
    sql = f"""
        SELECT DISTINCT ON (symbol, time_bucket)
               symbol, time_bucket, {feature_cols}
        FROM feature_cache_1h
        WHERE symbol = ANY(%s)
        ORDER BY symbol, time_bucket DESC
    """
    with get_conn() as conn:
        df = pd.read_sql(sql, conn, params=(symbols,))
    # take the N most recent rows across the class (they have no outcomes
    # yet - perfect: the guard must still find only OLDER neighbors)
    df = df.sort_values("time_bucket").tail(N_QUERIES).reset_index(drop=True)
    check("query rows loaded", len(df) >= 100, f"({len(df)} rows)")

    encoder = VectorEncoder.load(VectorEncoder.path_for(asset_class))
    vectors = encoder.transform(df)
    check("encoder loaded", vectors.shape[1] == encoder.output_dim_,
          f"(dim={encoder.output_dim_})")

    min_age_s = config.MEMORY_MIN_AGE_MINUTES * 60
    probs, qualities, nbr_counts = [], [], []
    guard_violations = 0
    no_recall = 0

    for i in range(len(df)):
        asof = pd.Timestamp(df["time_bucket"].iloc[i]).to_pydatetime()
        neighbors = recall(client, asset_class, vectors[i], asof=asof)
        if not neighbors:
            no_recall += 1
            continue
        cutoff = int(asof.timestamp()) - min_age_s
        guard_violations += sum(1 for _, _, ts, _ in neighbors if ts > cutoff)
        v = verdict_from_neighbors(neighbors)
        if v:
            probs.append(v["prob"])
            qualities.append(v["quality"])
            nbr_counts.append(v["n_kept"])

    check("look-ahead guard", guard_violations == 0,
          f"({guard_violations} violations)")
    check("recall coverage", no_recall <= len(df) * 0.05,
          f"({no_recall}/{len(df)} queries had zero neighbors)")

    if probs:
        probs, qualities = np.array(probs), np.array(qualities)
        check("probability not degenerate", probs.std() > 0.005,
              f"(std={probs.std():.4f}, mean={probs.mean():.3f})")
        check("quality sane", qualities.max() < 0.9,
              f"(p50={np.percentile(qualities, 50):.3f} "
              f"p95={np.percentile(qualities, 95):.3f} "
              f"max={qualities.max():.3f})")
        print(f"  [info] neighbors kept per query: "
              f"median {int(np.median(nbr_counts))}, "
              f"min {int(np.min(nbr_counts))}, max {int(np.max(nbr_counts))}")
        print(f"  [info] prob band: p5={np.percentile(probs, 5):.3f} "
              f"p95={np.percentile(probs, 95):.3f} | "
              f"HOLD zone share: "
              f"{np.mean((probs > config.SELL_THRESHOLD) & (probs < config.BUY_THRESHOLD)):.0%}")
    return result


def main() -> int:
    print("=" * 64)
    print("BRAIN SANITY (Phase 3 acceptance)")
    print("=" * 64)
    client = get_qdrant()
    failures = 0
    for cls, symbols in CLASS_POOLS.items():
        if not symbols:
            continue
        print(f"\n--- {cls.upper()} ({len(symbols)} symbols, collection "
              f"{collection_name(cls)}) ---")
        try:
            r = check_class(client, cls, symbols)
            failures += len(r["failed"])
        except Exception as exc:
            print(f"  [FAIL] class crashed: {exc}")
            failures += 1

    print("\n" + "=" * 64)
    total_pass = "ALL CHECKS PASSED" if failures == 0 else f"{failures} FAILURES"
    print(f"brain sanity complete: {total_pass}")
    print("=" * 64)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
