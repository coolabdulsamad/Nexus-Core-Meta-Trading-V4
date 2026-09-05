"""
Build (or rebuild) the brain's case memory: feature_cache_1h -> Qdrant.

Per asset class:
1. Load every feature row WITH a known outcome (forward_return_4h).
2. Fit the class encoder (z-score [+ PCA]) and save it to models/.
3. Recreate the class collection sized to the encoder's output dim.
4. Upsert every state as a point with a DETERMINISTIC id (blake2b of
   symbol|timestamp) so re-runs overwrite instead of duplicating.

Usage (on any machine that can reach the DB + Qdrant - no MT5 needed):
    python -m src.memory.build_memory                 # all classes
    python -m src.memory.build_memory --classes forex
"""

import argparse
import hashlib
import sys
from typing import Dict, List

import pandas as pd
from qdrant_client.models import PointStruct

from config.settings import config
from src.utils.logger import setup_logger
from src.storage.db import get_conn
from src.memory.vector_encoder import VectorEncoder
from src.memory.memory_store import (
    ASSET_CLASSES, collection_name, ensure_collection, get_qdrant,
    upsert_points, collection_size,
)

logger = setup_logger("build_memory", "logs/memory.log")

_BATCH = 2000

CLASS_POOLS: Dict[str, List[str]] = {
    "forex": config.FOREX_POOL,
    "metal": config.METALS_POOL,
    "crypto": config.CRYPTO_POOL,
    "index": config.INDICES_POOL,
}

_FETCH_SQL = """
SELECT symbol, time_bucket, {features}, forward_return_4h, regime_label
FROM feature_cache_1h
WHERE symbol = ANY(%s) AND forward_return_4h IS NOT NULL
ORDER BY time_bucket
"""


def _point_id(symbol: str, ts_epoch: int) -> int:
    digest = hashlib.blake2b(f"{symbol}|{ts_epoch}".encode(),
                             digest_size=8).digest()
    return int.from_bytes(digest, "little")


def build_class(asset_class: str, symbols: List[str], recreate: bool = True) -> int:
    if not symbols:
        logger.info("Class %s: pool empty, skipping", asset_class)
        return 0

    feature_cols = ", ".join(VectorEncoder.VECTOR_FEATURES)
    with get_conn() as conn:
        df = pd.read_sql(_FETCH_SQL.format(features=feature_cols),
                         conn, params=(symbols,))

    if len(df) < 500:
        logger.warning("Class %s: only %d outcome rows - too few, skipping",
                       asset_class, len(df))
        return 0
    df = df.dropna(subset=list(VectorEncoder.VECTOR_FEATURES))
    logger.info("Class %s: %d states with known outcomes",
                asset_class, len(df))

    encoder = VectorEncoder()
    vectors = encoder.fit_transform(df)
    encoder.save(VectorEncoder.path_for(asset_class))

    client = get_qdrant()
    name = ensure_collection(client, asset_class,
                             vector_size=encoder.output_dim_, recreate=recreate)

    ts_epoch = (pd.to_datetime(df["time_bucket"], utc=True)
                  .astype("int64").to_numpy() // 10**9)
    symbols_arr = df["symbol"].to_numpy()
    fwds = df["forward_return_4h"].astype(float).to_numpy()
    regimes = df["regime_label"].astype(str).to_numpy()

    written = 0
    batch: List[PointStruct] = []
    for i in range(len(df)):
        batch.append(PointStruct(
            id=_point_id(symbols_arr[i], int(ts_epoch[i])),
            vector=vectors[i].tolist(),
            payload={
                "symbol": str(symbols_arr[i]),
                "ts": int(ts_epoch[i]),
                "fwd": float(fwds[i]),
                "regime": regimes[i],
            },
        ))
        if len(batch) >= _BATCH:
            written += upsert_points(client, asset_class, batch)
            batch = []
    written += upsert_points(client, asset_class, batch)

    logger.info("Class %s: %d points in %s (dim=%d)",
                asset_class, collection_size(client, asset_class),
                name, encoder.output_dim_)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build CBR memory in Qdrant")
    parser.add_argument("--classes", nargs="*", default=None,
                        choices=list(CLASS_POOLS),
                        help="asset classes to build (default: all non-empty)")
    parser.add_argument("--no-recreate", action="store_true",
                        help="upsert into existing collections instead of rebuilding")
    args = parser.parse_args(argv)

    classes = args.classes or [c for c in ASSET_CLASSES if CLASS_POOLS.get(c)]
    total = 0
    for cls in classes:
        try:
            total += build_class(cls, CLASS_POOLS[cls],
                                 recreate=not args.no_recreate)
        except Exception as exc:
            logger.error("Memory build failed for %s: %s", cls, exc)
            return 1
    logger.info("Memory build complete: %d total points", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
