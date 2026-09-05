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

Git Bash (installed with Git) — right-click the repo folder →
*Git Bash Here*:

```bash
bash scripts/setup_db.sh
```

Or pure PowerShell:

```powershell
docker exec -i nexus_v4_db psql -U nexus -d nexus_mt5 -v ON_ERROR_STOP=1 < database/schema.sql
```

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

## Daily use (later phases)

```powershell
python -m src.ingestion.backfill_history   # Phase 2: one-time, ~hours
python -m src.ingestion.run_pump           # Phase 2: hourly (task scheduler)
python -m src.memory.build_memory          # Phase 3
python -m src.backtester.run_backtest      # Phase 4
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
