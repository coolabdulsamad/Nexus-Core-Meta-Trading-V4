"""
PostgreSQL / TimescaleDB access helpers.

Single connection factory + batched write helpers shared by the
backfill, pump, audit and (later) live-trader modules.
"""

from contextlib import contextmanager
from typing import Iterable, List, Sequence

import psycopg2
import psycopg2.extras

from config.settings import config


@contextmanager
def get_conn():
    """Yield an autocommit-off connection; commit on clean exit."""
    conn = psycopg2.connect(config.database.url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute_values_upsert(
    conn,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence],
    conflict_cols: Sequence[str] = ("symbol", "time_bucket"),
    page_size: int = 1000,
) -> int:
    """
    Bulk INSERT ... ON CONFLICT (conflict_cols) DO UPDATE.

    Returns number of rows written. Empty input -> 0, no query issued.
    """
    rows: List[Sequence] = list(rows)
    if not rows:
        return 0

    cols = ", ".join(columns)
    conflict = ", ".join(conflict_cols)
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in columns if c not in conflict_cols
    )
    sql = (
        f"INSERT INTO {table} ({cols}) VALUES %s "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
    )
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=page_size)
    return len(rows)
