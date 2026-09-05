"""
src/live/state.py
================================================================
Persistent local state for the live trader (state/live_state.json).

Why a file, not the DB: this state is read and written every management
cycle (~60s) and must survive restarts instantly, even with the DB down.
The DB journal (trades table) is the audited record; this file is the
working memory. Both are written for every trade event.

Safety properties:
- ATOMIC writes (tmp file + os.replace): a crash mid-write can never
  leave a half-written state file.
- Schema-versioned: unknown versions load as defaults rather than crash.
- Everything is plain JSON (no pickle) so you can read it in Notepad.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

STATE_VERSION = 1


def default_state() -> dict:
    return {
        "version": STATE_VERSION,
        # str(ticket) -> position dict (position_manager.ManagedPosition).
        # DRY_RUN virtual tickets are NEGATIVE and count down from -1, so
        # they can never collide with a real broker ticket.
        "positions": {},
        "virt_ticket_seq": -1,
        # ---- loss cooldowns (per symbol) ----
        "no_entry_until": {},   # symbol -> iso ts: no new entry before this
        "stop_times": {},       # symbol -> [iso ts] of recent stop-outs
        # ---- daily guards (refreshed at 00:00 UTC) ----
        "day": {
            "date": None,             # "YYYY-MM-DD" (UTC) the anchors belong to
            "start_balance": None,
            "start_equity": None,
            "closed_count": 0,
            "realized_pnl": 0.0,
            "halted_loss": False,     # daily loss limit hit -> no new entries today
            "profit_lock": False,     # daily target hit -> no new entries today
        },
        # ---- account-level circuit breaker ----
        "peak_equity": None,          # all-time peak equity (drawdown breaker)
        "halted_drawdown": False,     # latched until the next UTC day
        # ---- scheduler bookkeeping ----
        "last_entry_hour": None,      # iso ts of the last bar-close entry cycle
        "last_heartbeat": None,
        "last_maintenance_date": None,
        "last_eod_date": None,
    }


def _merge(base: dict, override: dict) -> dict:
    """Deep-merge override onto base, keeping base keys the file lacks."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_state(path: str) -> dict:
    """Load state, tolerating a missing/corrupt/old file (returns defaults
    merged over whatever was readable)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
            return default_state()
        return _merge(default_state(), raw)
    except Exception:
        return default_state()


def save_state(path: str, state: dict) -> None:
    """Atomic write: serialize fully, then os.replace over the old file."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# small time helpers shared by the live package
# ---------------------------------------------------------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def parse_iso(s) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None
