-- Phase 4c: multi-horizon brain labels.
-- Adds 12h / 24h forward-return columns to feature_cache_1h.
-- Existing DBs: run once, then re-run the pump to fill history
-- (python -m src.ingestion.run_pump), then rebuild the memory
-- (python -m src.memory.build_memory).
-- Apply from the repo root:
--   Get-Content database/migrations/001_multi_horizon_labels.sql -Raw | docker exec -i nexus_v4_db psql -U nexus -d nexus_mt5 -v ON_ERROR_STOP=1

ALTER TABLE feature_cache_1h
    ADD COLUMN IF NOT EXISTS forward_return_12h DOUBLE PRECISION;
ALTER TABLE feature_cache_1h
    ADD COLUMN IF NOT EXISTS forward_return_24h DOUBLE PRECISION;
