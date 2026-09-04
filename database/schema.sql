-- Nexus Core Meta (V4) schema - fresh installs
-- TimescaleDB: bars + features hypertables, universe, selection, trade journal
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 1. Raw bars (1h layer is the ACTIVE one; MT5 gives us spread per bar)
CREATE TABLE IF NOT EXISTS market_data_1h (
    symbol VARCHAR(20) NOT NULL,          -- canonical (EURUSD), not broker name
    time_bucket TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 4) NOT NULL,       -- tick volume (forex has no real volume)
    spread_points INTEGER,                -- broker spread at bar time (cost memory!)
    vwap DECIMAL(20, 8),
    UNIQUE (symbol, time_bucket)
);
SELECT create_hypertable('market_data_1h', 'time_bucket', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- 2. Feature cache (brain state description; all scale-free)
CREATE TABLE IF NOT EXISTS feature_cache_1h (
    symbol VARCHAR(20) NOT NULL,
    time_bucket TIMESTAMPTZ NOT NULL,
    rsi_14 DECIMAL(12, 6),
    macd_line DECIMAL(14, 6),
    macd_signal DECIMAL(14, 6),
    macd_hist DECIMAL(14, 6),
    bb_upper DECIMAL(20, 8),
    bb_lower DECIMAL(20, 8),
    bb_pct_b DECIMAL(10, 4),
    bb_width DECIMAL(12, 8),
    atr_14 DECIMAL(20, 8),
    atr_pct DECIMAL(12, 8),
    volume_profile_ratio DECIMAL(12, 4),
    vol_z DECIMAL(10, 4),
    ret_1 DECIMAL(12, 8),
    ret_3 DECIMAL(12, 8),
    ret_12 DECIMAL(12, 8),
    adx_14 DECIMAL(10, 4),
    dist_sma50 DECIMAL(12, 4),
    dist_sma200 DECIMAL(12, 4),
    dist_vwap DECIMAL(12, 4),
    hour_sin DECIMAL(8, 6),
    hour_cos DECIMAL(8, 6),
    dow_sin DECIMAL(8, 6),                -- V4: day-of-week (forex weekly rhythm)
    dow_cos DECIMAL(8, 6),
    spread_pct DECIMAL(12, 8),            -- V4: spread as % of price (cost regime)
    sentiment_score DECIMAL(6, 4),
    forward_return_1h DECIMAL(12, 8),
    forward_return_4h DOUBLE PRECISION,
    regime_label VARCHAR(20) DEFAULT 'unknown',
    PRIMARY KEY (symbol, time_bucket)
);
SELECT create_hypertable('feature_cache_1h', 'time_bucket', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- 3. Symbol universe (canonical <-> broker mapping, verified at seed time)
CREATE TABLE IF NOT EXISTS symbols (
    symbol VARCHAR(20) NOT NULL,          -- canonical
    broker_symbol VARCHAR(40),            -- actual MT5 name (EURUSD.pro ...)
    asset_class VARCHAR(10) NOT NULL DEFAULT 'forex',  -- forex | metal | crypto | index
    active BOOLEAN NOT NULL DEFAULT TRUE,
    verified_broker BOOLEAN NOT NULL DEFAULT FALSE,
    added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT,
    PRIMARY KEY (symbol, asset_class)
);

-- 4. Daily top-N selection (auto mode) with the "why"
CREATE TABLE IF NOT EXISTS daily_selection (
    selection_date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    asset_class VARCHAR(10) NOT NULL DEFAULT 'forex',
    rank INTEGER NOT NULL,
    score DECIMAL(8, 4) NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (selection_date, symbol)
);

-- 5. Trade journal (NEW in V4): every position with its full context,
-- so weekly gate calibration is a SQL query instead of log archaeology.
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    magic BIGINT,
    symbol VARCHAR(20) NOT NULL,
    asset_class VARCHAR(10),
    side VARCHAR(5) NOT NULL,             -- LONG | SHORT
    volume_lots DECIMAL(14, 6),
    entry_time TIMESTAMPTZ NOT NULL,
    entry_price DECIMAL(20, 8),
    sl_initial DECIMAL(20, 8),
    tp_initial DECIMAL(20, 8),
    exit_time TIMESTAMPTZ,
    exit_price DECIMAL(20, 8),
    pnl DECIMAL(16, 2),
    r_multiple DECIMAL(8, 3),
    exit_reason VARCHAR(40),
    quality DECIMAL(6, 4),                -- brain quality at entry
    eff_quality DECIMAL(6, 4),            -- after memory-depth scaling / penalties
    regime VARCHAR(20),
    sentiment DECIMAL(6, 4),
    memory_n INTEGER,                     -- neighbors behind the entry decision
    spread_pct_at_entry DECIMAL(12, 8),
    atr_at_entry DECIMAL(20, 8),
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_open ON trades (symbol) WHERE exit_time IS NULL;

-- 6. Indexes
CREATE INDEX IF NOT EXISTS idx_md1h_symbol_time ON market_data_1h (symbol, time_bucket DESC);
CREATE INDEX IF NOT EXISTS idx_fc1h_symbol_time ON feature_cache_1h (symbol, time_bucket DESC);
