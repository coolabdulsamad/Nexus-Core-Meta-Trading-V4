# Nexus Core Meta Trading V4 — Setup Guide

From zero to a passing self-test. Windows 10/11 assumed (the MetaTrader5
Python package only works on Windows, next to a running MT5 terminal).

---

## 0. What you need installed

| Software | Why | Get it |
|---|---|---|
| MetaTrader 5 terminal | the broker connection itself | your broker's site (use THEIR installer so the server is preconfigured) |
| Python 3.11 or 3.12 (64-bit) | runtime | python.org — tick **"Add python.exe to PATH"** during install |
| Git | version control | git-scm.com |
| VSCode | editor | code.visualstudio.com (+ the official **Python** extension) |
| Docker Desktop | TimescaleDB + Qdrant | docker.com — enable WSL2 when it asks |

Open an MT5 **demo account** with your broker first (File → Open an
Account). Everything is tested on demo; real money is a Phase-7 decision.

---

## 1. GitHub → your PC

```powershell
# pick a home for the project, e.g. Documents\trading
cd $HOME\Documents
mkdir trading; cd trading

git clone https://github.com/coolabdulsamad/Nexus-Core-Meta-Trading-V4.git
cd Nexus-Core-Meta-Trading-V4
```

## 2. Open in VSCode

```powershell
code .
```

In VSCode: install the Python extension when prompted, then
`Ctrl+Shift+P` → *Python: Select Interpreter* → pick the `.venv` we create
next (it appears after step 3; you can re-run this then).

## 3. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# if activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

python -m pip install --upgrade pip
pip install -r requirements.txt
```

> The `MetaTrader5` package installs only on Windows (it's marked that way
> in requirements.txt), so this same file also works on Linux/macOS for the
> backtest-only tools.

## 4. Configure `.env`

```powershell
copy .env.example .env
notepad .env
```

Fill in:

- **MT5_LOGIN / MT5_PASSWORD / MT5_SERVER** — *leave empty* to reuse the
  account already logged in inside your running terminal (easiest).
  Fill them only if you want the bot to log in itself.
- **DB_PASSWORD** — pick a strong one; it must match docker-compose.
- **TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID** — create a bot via
  @BotFather, message it once, then get your chat id via
  `https://api.telegram.org/bot<TOKEN>/getUpdates`.
- **NEWSAPI_KEY** — free key at newsapi.org (Phase 6; can stay empty now).
- **DRY_RUN=true** — leave it. Seriously.

## 5. Start the infrastructure

```powershell
docker compose up -d
docker ps        # expect nexus_v4_db and nexus_v4_qdrant, both healthy
```

> **Runs side-by-side with the Alpaca edition.** V4 deliberately uses host
> ports **5544** (database) and **6644/6645** (Qdrant) because the old
> system's `nexus_timescaledb` / `nexus_qdrant` containers already own
> 5432 / 6333 / 6334. Both systems can run at the same time without
> touching each other — separate containers, separate volumes, separate
> databases, separate vector collections.

## 6. Apply the database schema

> Note: the very **first** `docker compose up -d` applies `schema.sql`
> automatically (it's mounted into the DB's init folder). You only need
> this step to re-apply the schema or run later migrations.

PowerShell (native, recommended on Windows):

```powershell
.\scripts\setup_db.ps1
```

Git Bash — right-click the repo folder → *Git Bash Here*:

```bash
bash scripts/setup_db.sh
```

⚠️ Two Windows gotchas, already fixed in the repo (just `git pull`):

- **`set: pipefail: invalid option name`** in Git Bash → the `.sh` file was
  checked out with CRLF line endings. The repo's `.gitattributes` now pins
  LF. After `git pull`, run `git add --renormalize .` once if it persists.
- **`<` operator is reserved** → PowerShell does not support `<` input
  redirection (that was CMD syntax). The correct PowerShell form is:
  `Get-Content database/schema.sql -Raw | docker exec -i nexus_v4_db psql -U nexus -d nexus_mt5 -v ON_ERROR_STOP=1`

## 7. Prepare the MT5 terminal

1. Start MT5 and log in to your **demo** account.
2. Tools → Options → Expert Advisors:
   - ✅ *Allow algorithmic trading*
   - ✅ *Allow DLL imports* (not strictly needed, but some brokers require it)
3. The big **Algo Trading** button in the toolbar must be green.
4. Right-click Market Watch → *Show All* so every broker symbol is visible
   (the bot also auto-enables the ones it needs).

## 8. Run the self-test

```powershell
python scripts/mt5_selftest.py
```

Expected: PASS on package import, connection, account info, universe
discovery, per-symbol specs/bars, and the indicator engine. Warnings about
short broker history or wide spreads are fine — Phase 2's backfill handles
history depth.

```
self-test complete: 40+ passed, a few warnings, 0 failed
```

**That's Phase 1 done.** Next: Phase 2 (data backfill) — see
`docs/PHASES.md`.

---

## 9. Phase 2 — build the data layer

Run these on the Windows MT5 machine, with the terminal open and logged in:

```powershell
# 1) One-time deep backfill: ~3 years of H1 bars per symbol -> TimescaleDB
#    (walks history month by month; safe to re-run, it resumes/ignores dupes)
python -m src.ingestion.backfill_history

# optional: deeper history or a subset while testing
python -m src.ingestion.backfill_history --years 5
python -m src.ingestion.backfill_history --symbols EURUSD XAUUSD BTCUSD

# 2) Feature pump: tops up new bars, recomputes features on the FULL
#    window, fills forward returns, refreshes the symbol registry.
#    Run it once now to build feature_cache_1h ...
python -m src.ingestion.run_pump

# 3) Audit what we stored (runs anywhere, DB only)
python scripts/data_audit.py
```

Expected: the audit prints every symbol at ~99%+ completeness with few
gaps and outliers, and writes `reports\spread_stats.json` (the per-symbol
spread distributions the Phase-4 backtester uses for honest costs).

After that, schedule the pump hourly (Task Scheduler, run at e.g. :05 past
the hour) so the brain's memory stays fresh:

```powershell
# Task Scheduler action (adjust paths):
#   Program: C:\...\Nexus-Core-Meta-Trading-V4\.venv\Scripts\python.exe
#   Args:    -m src.ingestion.run_pump
#   Start in: C:\...\Nexus-Core-Meta-Trading-V4
```

---

## 10. Phase 3 — build the brain's memory

No MT5 needed for this — it reads TimescaleDB and writes Qdrant:

```powershell
# 1) Build one vector memory per asset class from the feature cache
#    (only states with a KNOWN 4h outcome become cases)
python -m src.memory.build_memory

# 2) Acceptance: recall works, look-ahead guard holds, distributions sane
python scripts/brain_sanity.py
```

Expected: per class, memory populated (hundreds of thousands of states),
`0 look-ahead violations`, probability std > 0.005, and a smooth quality
distribution. NOTE: V4's quality scale runs HOTTER than the Alpaca
edition's (whose live max was 0.480) — with ~1.2M states, top-of-book
similarity is structurally higher, so a V4 max around 0.6-0.75 is normal.
The 0.35/0.45 gates are re-derived from the Phase 4 quality-bucket table
in Phase 7; do not "fix" the scale by hand.

---

## 11. Phase 4 — the honest backtest

No MT5 needed — it reads TimescaleDB + Qdrant and simulates trades bar by
bar with real costs (your stored per-bar spread, Wednesday triple swap,
slippage on every fill). Decisions are made on bar close and filled at the
NEXT bar's open; if a bar touches both stop and target, the stop loses.

```powershell
# Full run: every symbol, trailing 24 months (minutes, not hours —
# brain queries are batched; expect ~10-20 min for all 38 symbols)
python -m src.backtester.run_backtest

# Faster while iterating / single-class runs
python -m src.backtester.run_backtest --months 12 --class forex
python -m src.backtester.run_backtest --symbols EURUSD XAUUSD BTCUSD --months 24
```

Outputs:

- `reports\backtest_trades.csv` — every simulated trade with entry
  quality/prob/agreement, exit reason, MAE/MFE, swap paid
- `reports\backtest_summary.json` — headline stats, per-symbol results,
  and the decision funnel (where signals die)
- Console: headline PF / expectancy, per-class table, **quality-bucket
  table** (this is the Phase 7 input that re-derives the quality gates
  from THIS brain's distribution), exit-reason anatomy, total funnel

How to read the result:

| Signal | Meaning |
|---|---|
| overall PF > 1.0 after costs | the brain has a raw edge worth pursuing |
| PF rising across quality buckets | the quality score ranks trades correctly -> gates can be tightened to where PF turns > 1 |
| PF < 1 everywhere, all buckets | honest no-edge verdict — we recalibrate features/gates, not fudge costs |
| exits dominated by `time_stop` | horizon/targets mis-set; dominated by `stop_loss` means gates too loose |

Caveats printed in every summary: portfolio-level caps (max positions,
currency risk) are NOT simulated — they only remove trades, so the trade
count is an upper bound; per-trade economics are intact.

---

## 11b. Phase 4b — parameter sweep over cached verdicts

The brain's verdict at a bar does not depend on the trading config (stops,
horizons, gates) — only on the brain and the data. So we ask the brain
ONCE per symbol, cache the verdicts, then replay many config variants in
seconds each with zero Qdrant traffic.

```powershell
# Step 1 — collect (slow, ONCE; same order of time as the Phase 4 run).
# Resumable: a symbol with an existing cache file is skipped, so if it
# stops halfway just run the same command again.
python scripts/sweep_configs.py --collect --months 24

# Step 2 — evaluate all 12 variants (fast: seconds per variant)
python scripts/sweep_configs.py --eval --months 24

# Iterate on a subset while exploring
python scripts/sweep_configs.py --eval --variants baseline,horizon6,h6_noearly
python scripts/sweep_configs.py --collect --cls forex          # one class only
```

Cache lives in `reports\verdict_cache\*.pkl` (git-ignored, regenerable).
If you ever re-run Phase 2/3 (new bars, new brain), delete that folder and
collect again — stale verdicts would silently describe the old brain.

Variants tested (see `VARIANTS` in the script — add your own there):

| Group | Variants | Question answered |
|---|---|---|
| horizon | `horizon6`, `horizon4` | brain predicts 4h ahead — does a shorter hold keep more of the edge? |
| exits | `no_early_exits` | do scale-outs/ratchet/trailing help or clip winners? |
| risk shape | `wide_tp`, `tight_stop` | is the 1.5R target / 2xATR stop the right asymmetry? |
| gates | `conviction_hi`, `quality_055` | does trading less, but better, turn PF positive? |
| combos | `h6_conviction`, `h6_noearly`, `h4_wide_tp`, `h6_q055` | interactions of the above |

Outputs:

- `reports\sweep_results.csv` — one row per variant: trades, win rate,
  profit factor, expectancy (R), return %, max drawdown %
- `reports\sweep_best_<variant>_per_symbol.csv` — per-symbol breakdown of
  the best variant (which symbols carry the edge, which destroy it)
- Console: full ranking + per-class and exit-reason anatomy for the winner

The decision rule for Phase 5 is unchanged: a variant must show
**PF > 1.0 after costs** on the full universe, ideally with the per-symbol
table showing broad (not one-lucky-symbol) contribution.

Note: variants can only change gates DOWNSTREAM of candidacy (conviction,
quality, ADX, exits, sizing, horizon). Session/spread prefilters decide
WHICH bars get a verdict at collect time, so the sweep cannot vary them —
a session/spread experiment needs a fresh `--collect`.

---

## 11c. Phase 4c — multi-horizon brain labels

Phase 4b proved every variant of the 4h brain loses after costs — and that
longer holds lose LESS (PF: 16 bars 0.78 > 6 bars 0.67 > 4 bars 0.65).
So the label horizon itself is the prime suspect: the brain predicts 4h
ahead, but whatever weak predictability exists lives further out. Phase 4c
rebuilds the memory with **three outcome labels per state** (4h / 12h /
24h) and lets the sweep trade on any of them.

One-time upgrade (your existing 4b verdict caches are kept and reused):

```powershell
git pull

# 1) Add the two label columns to the database (one command)
Get-Content database/migrations/001_multi_horizon_labels.sql -Raw | docker exec -i nexus_v4_db psql -U nexus -d nexus_mt5 -v ON_ERROR_STOP=1

# 2) Refill the labels for ALL history (pump recomputes the full window)
#    Needs the MT5 terminal running. Minutes, not hours.
python -m src.ingestion.run_pump

# 3) Rebuild the memory so every state carries fwd_4h / fwd_12h / fwd_24h
#    (recreates the Qdrant collections; same order of time as Phase 3)
python -m src.memory.build_memory

# 4) Collect verdicts for the two NEW horizons only — your existing 4h
#    caches are reused untouched. One query per candidate per horizon
#    (each with its own look-ahead guard), so expect roughly 2x the
#    Phase 4b collect time. Resumable, as before.
python scripts/sweep_configs.py --collect --months 24

# 5) Sweep — the six new label variants plus the old twelve
python scripts/sweep_configs.py --eval --months 24
```

New variants: `label12h`, `label24h` (same trading config, brain trades on
longer outcomes), `l12_h12` / `l24_h24` (hold aligned to the label), and
`l12_tight` / `l24_tight` (label x the best 4b variant).

Honesty notes baked into the code:

- Each horizon is recalled with its OWN look-ahead guard — a 24h outcome
  is only knowable 25h later, so 24h verdicts never peek at fresher
  neighbors. This is stricter than most published "multi-horizon" work.
- The slippage units bug from 4b is fixed (0.5bp per fill was effectively
  0 before). Baseline numbers in the next sweep will look a few percent
  WORSE than the 4b table — that is the truth getting sharper, not a
  regression.

Decision rule is unchanged: **PF > 1.0 after costs**, broad across
symbols, before Phase 5 goes beyond demo.

---

## 12. Phase 5 — live engine (DEMO ONLY)

**Read this first.** Phases 4 / 4b / 4c all returned the same verdict:
the CBR brain has **no proven edge after costs** on this universe (best
sweep variant PF 0.78). Phase 5 therefore ships as what it was sanctioned
as — **execution plumbing validation on the demo account**, not a profit
deployment. The orders, risk engine, exit stack, reconciliation and
journaling must prove themselves in the real world no matter which brain
eventually drives them.

What was built:

- `src/live/live_trader.py` — the engine: hourly bar-close entries through
  the exact backtest gate chain, 60-second position management, daily
  guards, circuit breakers, Telegram heartbeats/EOD, daily maintenance.
- `src/live/position_manager.py` — the full exit stack live: scale-outs,
  ratchet, trailing, retracement, time partial/stop, flip exits, Friday
  flatten. Real positions carry broker-side SL/TP from second zero.
- `src/live/risk_engine.py` — per-currency net risk cap (2.5%), total open
  risk cap (5%), max 5 positions, daily loss/profit guards, drawdown breaker.
- `src/live/reconciler.py` — the broker is the only truth: adopts positions
  found at startup, journals closes that happened while the engine was down
  (exact pnl from deal history).
- `src/live/journal.py` + `src/live/state.py` — every decision journaled to
  the `trades` table; working state in `state/live_state.json` (atomic,
  human-readable).
- `scripts/demo_smoke.py` — the acceptance test below.
- `scripts/oos_split_check.py` — the XAUUSD out-of-sample stability check.

### 12a. Prerequisite: enable algo trading

In the MT5 terminal: **Tools → Options → Expert Advisors → "Allow
algorithmic trading"** ✓, and the **"Algo Trading" toolbar button must be
green**. The pump logs showed it disabled — data work doesn't care, but
every order will be rejected (retcode 10027) until it is on.

### 12b. Acceptance test (zero real orders)

```powershell
git pull
python scripts/demo_smoke.py
```

Expected: `SMOKE TEST: ALL CHECKS PASSED`. The test connects, verifies the
universe/encoders/memory/journal, asks the brain for live verdicts, and
then runs a **virtual trade through the real engine code** (entry → exit
stack → time-stop close → journal). Two rows appear in the `trades` table
flagged `DRY_RUN` — that is the journal doing its job, not an error.

### 12c. Run the engine in DRY_RUN

```powershell
python -m src.live.live_trader            # Ctrl+C stops cleanly
```

`DRY_RUN=true` in `.env` means **no order ever reaches the broker**:
entries fill virtually at the live tick and are managed by the same exit
stack. Leave it running for at least a week. Watch for: Telegram
heartbeats (~30 min), entry/exit cards, `state/live_state.json` updating,
and `SELECT * FROM trades ORDER BY id DESC;` showing the journal.

### 12d. Only after a clean DRY_RUN week — real orders on DEMO

```powershell
# .env:  DRY_RUN=false
python scripts/demo_smoke.py --real-order   # one min-size order, opened+closed
python -m src.live.live_trader
```

This is still the **demo account**. "Real orders" here means real broker
plumbing (fills, SL/TP, partial closes), not real money.

### 12e. The XAUUSD question

XAUUSD was the only symbol positive in BOTH independent sweeps (PF 1.18 →
1.23). Before anyone designs around it, check time-stability (seconds,
reuses the sweep caches):

```powershell
python scripts/oos_split_check.py --symbols XAUUSD --variants l24_tight,baseline
```

`STABLE` (PF > 1.0 in BOTH halves) keeps the conversation alive;
`UNSTABLE` closes it. Even a STABLE result is a post-hoc, single-symbol
find with a wide confidence interval — it would justify a focused
research phase, never a shortcut to live money.

---

## Daily use (later phases)

```powershell
python -m src.ingestion.backfill_history   # Phase 2: one-time, ~hours
python -m src.ingestion.run_pump           # Phase 2: hourly (task scheduler)
python -m src.memory.build_memory          # Phase 3
python -m src.backtester.run_backtest      # Phase 4
python scripts/sweep_configs.py --collect --months 24   # Phase 4b: once
python scripts/sweep_configs.py --eval --months 24      # Phase 4b: anytime
python -m src.live.live_trader             # Phase 5: the actual bot
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: MetaTrader5` | you're not on Windows, or the venv is 32-bit — reinstall Python 64-bit |
| `initialize() failed` | MT5 terminal not running, or a second terminal path — set `MT5_TERMINAL_PATH` in `.env` |
| self-test finds 0 symbols | Market Watch → Show All; also check your broker's suffixes (the connector tries `.PRO .A .M .STD .RAW .ECN`) |
| retcode 10027 when live | *Allow algorithmic trading* + green Algo Trading button |
| docker won't start | Docker Desktop running? WSL2 enabled? |
| psql: connection refused | `docker ps` — is `nexus_v4_db` healthy? connect with `-p 5544` (V4's port), not the default 5432 |
| `set: pipefail` in Git Bash | CRLF checkout — `git pull` (`.gitattributes` pins LF), then `git add --renormalize .` |
