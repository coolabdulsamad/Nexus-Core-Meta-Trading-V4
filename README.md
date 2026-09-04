# Nexus Core — Meta Trading V4 (MT5 Edition)

A case-based-reasoning (CBR) trading system for **MetaTrader 5** brokers —
forex, metals and crypto CFDs — evolved from the Nexus Core (Alpaca) edition,
rebuilt around the realities of MT5: broker symbol suffixes, filling modes,
netting vs hedging, tick-value lot sizing, spreads, swaps, sessions and
weekend gaps.

> Status: **Phase 1 — Foundation** (see `docs/PHASES.md` for the full roadmap)

---

## The edge, in one paragraph

Every hour the system encodes the current market state of each watched symbol
into a 17+ scale-free feature vector (z-scored, PCA-compressed to 64 dims) and
asks the vector memory: *"the last 100 times the market looked like this on
this asset class, what happened 4 hours later?"* It trades only when the
weighted, agreement-filtered answer clears calibrated quality gates — then a
cost-aware risk engine sizes the position from real tick values, a forex
session layer refuses dead-hour entries, and a layered exit stack (scale-outs,
ATR ratchet, retracement guard, time stop) manages the trade. Every decision is
journaled to Postgres so the memory keeps learning from its own outcomes.

## What V4 adds over the Alpaca edition

| Area | V4 improvement |
|---|---|
| **Costs** | Live spread gate (>2.5× median spread blocks entry), TP must be ≥6× spread, spread+swap modelled in backtests |
| **Sessions** | Forex session intelligence: dead hours 21–24 UTC, Asian thin hours, rollover blackout, Friday flatten, Sunday-open blackout |
| **Portfolio risk** | Per-currency net exposure cap (2.5% equity) and total open risk cap (5%) — stops EURUSD+EURJPY+EURGBP triple-counting the same bet |
| **Memory** | One Qdrant collection per asset class (`market_memory_60m_forex` / `_metal` / `_crypto`) so forex neighbors aren't polluted by crypto regimes |
| **Sizing** | True tick-value/tick-size lot sizing snapped to broker `volume_step`, min-distance SL/TP enforcement |
| **Ownership** | Magic-number position identification + Postgres trade journal — survives restarts, no orphaned-state bugs |
| **Tooling** | TA-Lib removed (pure-pandas indicators — no Windows build pain), MT5 self-test script, docker-compose infra |

## Architecture

```
MT5 terminal (Windows)                Docker
┌──────────────────────┐     ┌─────────────────────────────┐
│ src/mt5_client/      │     │ TimescaleDB                 │
│  connector.py  ──────┼────►│  market_data_1h (hypertable)│
│  orders, specs, bars │     │  feature_cache_1h           │
└─────────┬────────────┘     │  symbols / daily_selection  │
          │                  │  trades (journal)           │
          ▼                  └──────┬──────────────────────┘
┌──────────────────────┐            │        ┌──────────────┐
│ src/ingestion/       │────────────┘        │ Qdrant       │
│  pump, backfill,     │                     │ vector memory│
│  indicators (pandas) │                     │ per class    │
└─────────┬────────────┘                     └──────▲───────┘
          ▼                                         │
┌──────────────────────┐     CBR query (k=100,      │
│ src/models/          │───── similarity-weighted ──┘
│  meta_learner (CBR)  │     4h-forward outcomes)
│  vector encoder      │
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ src/live/            │  entries (quality gates, session filter,
│  live_trader         │  cost guard) → portfolio risk engine →
│  exit stack          │  exit stack (scale-outs, ratchet, trail,
└─────────┬────────────┘  retracement guard, time stop)
          ▼
   Telegram alerts + Postgres journal + daily maintenance
```

## Quick start (30-second version)

Full guide: **`docs/SETUP.md`**

```bash
git clone https://github.com/coolabdulsamad/Nexus-Core-Meta-Trading-V4.git
cd Nexus-Core-Meta-Trading-V4
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env                                 # fill in values
docker compose up -d                                   # TimescaleDB + Qdrant
bash scripts/setup_db.sh                               # apply schema
python scripts/mt5_selftest.py                         # MT5 must be running!
```

`DRY_RUN=true` in `.env` keeps the system from ever sending an order —
leave it on until Phase 4 backtests and a demo-account run are green.

## Project layout

```
config/settings.py            every knob, calibrated constants, pools
database/schema.sql           TimescaleDB hypertables + trade journal
docs/PHASES.md                the 7-phase build roadmap
docs/SETUP.md                 Windows + MT5 + VSCode setup, step by step
scripts/mt5_selftest.py       Phase-1 acceptance test
scripts/setup_db.sh           schema applier
src/mt5_client/connector.py   the ONLY file that talks to MT5
src/ingestion/                bars → DB → features (pure pandas)
src/memory/                   vector encoder, Qdrant memory, outcome sync
src/models/                   CBR meta-learner (the brain)
src/backtester/               honest backtester (costs, walk-forward)
src/live/                     live trader + exit stack + portfolio engine
src/news_engine/              FinBERT/NewsAPI sentiment (Phase 6)
src/utils/                    logger, telegram
tests/
```

## Safety posture

- `DRY_RUN=true` by default; order functions refuse to send while set.
- Daily account guards: halt at −5% day loss / +2% day profit lock.
- Loss cooldowns per symbol (24h/72h), toxic-combo and sentiment vetoes.
- The broker is the only source of truth for positions; the bot reconciles
  on every start instead of trusting local state.
