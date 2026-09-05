# Nexus Core Meta Trading V4 — Build Phases

The whole project, start to finish. Each phase ends with a concrete
acceptance test you can run yourself. We do **not** move to the next phase
until the current one passes — a trading system is only as strong as its
weakest layer.

---

## Phase 1 — Project setup & foundation ✅ (this phase)

**Goal:** a clean repo that installs, connects to MT5, and passes its
self-test on your Windows machine.

Steps:
1. Create repo on GitHub → clone to your PC → open in VSCode
   (`docs/SETUP.md` walks every click).
2. Python virtualenv + `requirements.txt` (MetaTrader5 is Windows-only and
   marked as such, so the rest installs anywhere).
3. `.env.example` → `.env` with your DB / Telegram / MT5 credentials.
4. `docker-compose.yml` — TimescaleDB + Qdrant in one command.
5. `database/schema.sql` — hypertables for bars & features, symbol map,
   trade journal.
6. `config/settings.py` — every constant in one place, including the
   calibrated gates carried over from the Alpaca edition and the new
   V4 cost/session/portfolio guards.
7. `src/mt5_client/connector.py` — the single, audited MT5 layer
   (symbol suffix discovery, filling modes, netting+hedging, tick-value
   sizing, magic-number ownership).
8. Pure-pandas indicator engine (TA-Lib removed — no C compiler needed).
9. `scripts/mt5_selftest.py` — the acceptance test.

**Acceptance:** `python scripts/mt5_selftest.py` → all PASS on the
Windows machine with MT5 running.

---

## Phase 2 — Data layer

**Goal:** 3 years of H1 bars for the whole universe in TimescaleDB, plus a
reliable hourly pump that keeps features fresh.

Steps:
1. `src/ingestion/backfill_history.py` — walk `copy_rates_range` backwards
   month by month per symbol (MT5 caps bars per call); store OHLCV +
   `spread_points` + tick volume; upsert so re-runs are safe.
2. `src/ingestion/loader.py` — single `load_bars(symbol)` that everything
   else uses.
3. `src/ingestion/indicator_calculator.py` ✅ (already written in Phase 1).
4. `src/ingestion/run_pump.py` — hourly job: fetch latest bars → recompute
   features on a full window (never a slice — that was Alpaca bug v3.2) →
   upsert into `feature_cache_1h`.
5. Data-quality audit: gap report per symbol, outlier bar detection
   (wick > 8×ATR vs broker feed), spread distribution per symbol saved for
   the Phase-4 cost model.

**Acceptance:** every universe symbol has ≥2.5 years of continuous H1
features; audit report shows <0.5% gaps (excluding weekends for forex).

---

## Phase 3 — The brain (CBR memory)

**Goal:** vector memory that answers "what happened the last 100 times the
market looked like this?" — per asset class, leakage-free.

Steps:
1. `src/memory/vector_encoder.py` — 17+ scale-free features → z-score →
   PCA(64). PCA is **fit on a training window only**, then frozen (leakage).
2. `src/memory/build_memory.py` — encode history → Qdrant collections
   `market_memory_60m_forex` / `_metal` / `_crypto`, payload = realized
   4h-forward return + regime + quality metadata.
3. `src/memory/backfill_forward_returns.py` — compute outcomes, enforce
   the look-ahead guard (only states older than horizon + 1h are usable).
4. `src/models/meta_learner.py` — port the Alpaca CBR decision core:
   similarity²-weighted mean of neighbor outcomes → sigmoid → LONG/SHORT/
   HOLD with the 0.48–0.52 dead zone, agreement ≥55%, similarity ≥0.50,
   memory-depth scaling (q × min(1, n/80)).
5. Sanity harness: synthetic trending series must produce LONG bias;
   synthetic mean-reverting must produce HOLD/SHORT — proves the brain
   actually reads the memory.

**Acceptance:** memory built for all symbols; brain outputs are
directionally sane on synthetic series and match a hand-computed example.

---

## Phase 4 — Honest backtester

**Goal:** numbers we can trust before a single real cent is risked.

Steps:
1. `src/backtester/engine.py` — bar-by-bar replay using ONLY data that
   existed at that timestamp (walk-forward: memory rebuilt as-of each fold).
2. `src/backtester/broker_simulator.py` — MT5-realistic fills:
   - entry at next bar open **+ spread**, exit at bid/ask correctly,
   - swap charged per day held (per symbol, from specs),
   - optional commission per lot, slippage in points,
   - margin check so impossible positions are rejected.
3. Full decision funnel instrumentation (like Alpaca v3.6): how many
   signals died at each gate and why.
4. Metrics: net P&L, win rate, profit factor, max drawdown, expectancy in
   R, per-symbol and per-session breakdowns, cost share of gross P&L.
5. Calibration: if quality gates are too tight/loose, adjust using the
   funnel stats — the same evidence-driven way the Alpaca edition reached
   v3.6.5.

**Acceptance:** walk-forward backtest over ≥2 years, positive expectancy
**after costs**, funnel report showing every gate's kill rate. If it's not
positive, we tune in Phase 7 before ever discussing live money.

---

## Phase 5 — Live MT5 trader ✅ (built; demo-only by verdict)

**Goal:** the real-time engine. First on a **demo account**, `DRY_RUN=false`.

> **Status note (post-4c):** Phases 4/4b/4c found no tradable edge after
> costs (best variant PF 0.78). Phase 5 therefore runs as **plumbing
> validation on demo** — the execution stack must earn trust in the real
> world regardless of which brain eventually drives it. No live money
> until an edge is proven (Phase 7).

Steps:
1. `src/live/live_trader.py` — main loop on the hourly bar close:
   refresh features → brain query → entry gate chain (quality, session,
   spread, cooldown, sentiment) → portfolio risk engine → size & send. ✅
2. Portfolio risk engine — per-currency net risk cap 2.5%, total open risk
   cap 5%, max 5 positions. This is the big new shield vs the Alpaca edition. ✅
3. Exit stack (ported + MT5-native, via `TRADE_ACTION_SLTP` and partial
   closes): scale-outs ⅓ @ +1/+2 ATR, SL ratchet to lock +0.5 ATR,
   trailing stop, retracement guard, flip-exit, 16-bar time stop. ✅
4. Reconcile-on-start: adopt positions the broker already has (magic match),
   never duplicate, never fight manual trades (different magic). ✅
5. Friday flatten (20:00 UTC) and rollover blackout. ✅

**Acceptance:** `python scripts/demo_smoke.py` all-PASS (see SETUP 12b),
then ≥2 weeks on demo with Telegram heartbeats, zero reconcile errors,
live behavior matching backtest expectations within noise.

---

## Phase 6 — Operations & intelligence

**Goal:** run it like infrastructure, not like a script.

Steps:
1. Telegram command layer: `/status`, `/positions`, `/pause`, `/resume`,
   `/equity`, error alerts, EOD report.
2. `src/maintenance.py` — daily 22:00 UTC: pump catch-up, Qdrant outcome
   sync, journal integrity check, DB health.
3. `src/news_engine/` — FinBERT on NewsAPI headlines per symbol/asset
   (crypto keywords, gold keywords); sentiment vetoes at ±0.60 exactly like
   the Alpaca edition.
4. `daily_selection` — auto universe mode: rank pools by spread/liquidity,
   pick top N per class each day.
5. Optional dashboard (streamlit) reading the trade journal.

**Acceptance:** a full unattended week: maintenance runs logged, sentiment
wired into the gate chain, EOD reports arrive.

---

## Phase 7 — Calibration & go-live review

**Goal:** earn the right to trade real money — or honestly conclude not yet.

Steps:
1. Tune gates on backtest + demo evidence: quality thresholds, memory
   depth, session windows, cost floors, per-class parameter overrides.
2. A/B shadow: run two gate profiles side by side on demo, keep the winner.
3. Risk sign-off checklist: max drawdown tolerable at planned size? cost
   share acceptable? worst day survivable?
4. Go-live at **minimum viable size** (0.01-lot scale risk), scale only
   after a profitable month.

**Acceptance:** written go-live checklist, all boxes ticked, size ramp plan.

---

## Standing rules for every phase

- The broker is the only truth about positions and prices.
- Nothing is traded that wasn't journaled; nothing is journaled that
  wasn't decided by the gate chain.
- `DRY_RUN=true` until Phase 4 is green **and** Phase 5 demo is green.
- Every constant lives in `config/settings.py` with a comment saying why.
