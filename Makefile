# Nexus Core Meta Trading V4 — convenience commands
# (On Windows use the equivalent commands from docs/SETUP.md,
#  or run via Git Bash / WSL.)

.PHONY: infra db selftest backfill pump brain backtest live test fmt

infra:          ## Start TimescaleDB + Qdrant
	docker compose up -d

db:             ## Apply schema to a running database
	bash scripts/setup_db.sh

selftest:       ## Verify MT5 connectivity + discover broker universe
	python scripts/mt5_selftest.py

backfill:       ## Phase 2: download 3y of H1 history into TimescaleDB
	python -m src.ingestion.backfill_history

pump:           ## Phase 2: run the hourly feature pump once
	python -m src.ingestion.run_pump

brain:          ## Phase 3: build/refresh vector memory
	python -m src.memory.build_memory

backtest:       ## Phase 4: honest backtest with costs + walk-forward
	python -m src.backtester.run_backtest

live:           ## Phase 5: live/dry-run trader (respects DRY_RUN in .env)
	python -m src.live.live_trader

test:
	pytest -q
