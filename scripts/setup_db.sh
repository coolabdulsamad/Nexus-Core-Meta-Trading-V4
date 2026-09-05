#!/usr/bin/env bash
# Apply database/schema.sql (+ any migrations) to the configured database.
# Works against the docker-compose TimescaleDB or any external Postgres.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5544}"   # 5432 is the Alpaca edition's database
DB_NAME="${DB_NAME:-nexus_mt5}"
DB_USER="${DB_USER:-nexus}"

echo "==> Applying schema to ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^nexus_v4_db$'; then
  docker exec -i nexus_v4_db psql -U "${DB_USER}" -d "${DB_NAME}" -v ON_ERROR_STOP=1 < database/schema.sql
  for f in database/migrations/*.sql; do
    [ -e "$f" ] || continue
    echo "==> migration: $f"
    docker exec -i nexus_v4_db psql -U "${DB_USER}" -d "${DB_NAME}" -v ON_ERROR_STOP=1 < "$f"
  done
else
  PGPASSWORD="${DB_PASSWORD:-}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -v ON_ERROR_STOP=1 -f database/schema.sql
  for f in database/migrations/*.sql; do
    [ -e "$f" ] || continue
    echo "==> migration: $f"
    PGPASSWORD="${DB_PASSWORD:-}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -v ON_ERROR_STOP=1 -f "$f"
  done
fi

echo "==> Done."
