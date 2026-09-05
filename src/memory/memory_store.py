"""
Qdrant memory store: one collection per asset class.

Collections: market_memory_60m_forex / _metal / _crypto / _index
(CBR memory is per-class: a BTC breakout is not evidence for EURUSD).

Point payload:
    symbol  str   canonical symbol
    ts      int   bar open time, unix seconds (the look-ahead guard filters on it)
    fwd     float realized forward_return_4h (the OUTCOME the brain recalls)
    regime  str   trend_up | trend_down | range | transition

The look-ahead guard lives HERE, at retrieval time, not at build time:
a query "as of T" only ever sees points with ts <= T - MEMORY_MIN_AGE,
so no neighbor's known outcome could have been known at decision time.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, PayloadSchemaType, PointStruct, Range,
    VectorParams,
)

from config.settings import config
from src.utils.logger import setup_logger

logger = setup_logger("MemoryStore", "logs/memory.log")

ASSET_CLASSES = ("forex", "metal", "crypto", "index")


def collection_name(asset_class: str) -> str:
    return f"market_memory_{config.BAR_MINUTES}m_{asset_class}"


def get_qdrant() -> QdrantClient:
    # timeout: filtered batch queries over ~1M points on Docker Desktop can
    # take far longer than the 5s library default - give them real room.
    q = config.qdrant
    if q.url:
        return QdrantClient(url=q.url, api_key=q.api_key or None, timeout=120)
    return QdrantClient(host=q.host, port=q.port, timeout=120)


def ensure_collection(client: QdrantClient, asset_class: str,
                      vector_size: int, recreate: bool = False) -> str:
    """Create (or recreate) the class collection + ts payload index."""
    name = collection_name(asset_class)
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created collection %s (dim=%d, cosine)", name, vector_size)
    # payload index makes the look-ahead range filter cheap at 1M+ points
    try:
        client.create_payload_index(
            collection_name=name, field_name="ts",
            field_schema=PayloadSchemaType.INTEGER,
        )
    except Exception:
        pass  # index already exists
    return name


def upsert_points(client: QdrantClient, asset_class: str,
                  points: List[PointStruct]) -> int:
    if not points:
        return 0
    client.upsert(collection_name=collection_name(asset_class), points=points)
    return len(points)


def recall(client: QdrantClient, asset_class: str, vector,
           asof: Optional[datetime] = None,
           k: Optional[int] = None,
           label_horizon: Optional[str] = None) -> List[Tuple[float, float, int, str]]:
    """
    k nearest memory states visible `asof` a decision time.

    Returns [(similarity, forward_return, ts, symbol), ...], best first,
    where forward_return is the ACTIVE label (config.BRAIN_LABEL_HORIZON).
    The look-ahead guard excludes every state whose outcome for that label
    was not yet knowable at `asof` ((label_bars + 1) hours).
    """
    asof = asof or datetime.now(timezone.utc)
    h = label_horizon or config.BRAIN_LABEL_HORIZON
    key = config.BRAIN_LABEL_PAYLOAD_KEY[h]
    guard_min = (config.BRAIN_LABEL_BARS[h] + 1) * 60
    cutoff = int((asof - timedelta(minutes=guard_min))
                 .timestamp())
    result = client.query_points(
        collection_name=collection_name(asset_class),
        query=list(map(float, vector)),
        limit=k or config.MEMORY_NEIGHBORS,
        query_filter=Filter(must=[
            FieldCondition(key="ts", range=Range(lte=cutoff)),
        ]),
        with_payload=True,
    )
    out = []
    for p in result.points:
        payload = p.payload or {}
        fwd = payload.get(key)
        if fwd is None:
            continue
        out.append((float(p.score), float(fwd),
                    int(payload.get("ts", 0)), str(payload.get("symbol", ""))))
    return out


def collection_size(client: QdrantClient, asset_class: str) -> int:
    name = collection_name(asset_class)
    if not client.collection_exists(name):
        return 0
    return client.get_collection(name).points_count
