# scripts/setup_db.ps1
# Apply database/schema.sql (+ any migrations) to the V4 TimescaleDB.
# Native PowerShell version — no bash needed on Windows.
# Usage (from the repo root):  .\scripts\setup_db.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# Load .env into the process environment (simple KEY=VALUE parser)
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*#' -or $_ -notmatch '=') { return }
        $k, $v = $_ -split '=', 2
        [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), 'Process')
    }
}

$DbHost = if ($env:DB_HOST) { $env:DB_HOST } else { 'localhost' }
$DbPort = if ($env:DB_PORT) { $env:DB_PORT } else { '5544' }   # 5432 = Alpaca edition
$DbName = if ($env:DB_NAME) { $env:DB_NAME } else { 'nexus_mt5' }
$DbUser = if ($env:DB_USER) { $env:DB_USER } else { 'nexus' }

Write-Host "==> Target: ${DbUser}@${DbHost}:${DbPort}/${DbName}"

$running = docker ps --format '{{.Names}}' 2>$null | Select-String -SimpleMatch 'nexus_v4_db'
if ($running) {
    Write-Host "==> Applying schema inside container nexus_v4_db"
    Get-Content database/schema.sql -Raw | docker exec -i nexus_v4_db psql -U $DbUser -d $DbName -v ON_ERROR_STOP=1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Get-ChildItem database/migrations/*.sql -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "==> migration: $($_.Name)"
        Get-Content $_.FullName -Raw | docker exec -i nexus_v4_db psql -U $DbUser -d $DbName -v ON_ERROR_STOP=1
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
else {
    Write-Host "==> Container not found; trying local psql on ${DbHost}:${DbPort}"
    $env:PGPASSWORD = $env:DB_PASSWORD
    Get-Content database/schema.sql -Raw | psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -v ON_ERROR_STOP=1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Get-ChildItem database/migrations/*.sql -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "==> migration: $($_.Name)"
        Get-Content $_.FullName -Raw | psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -v ON_ERROR_STOP=1
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

Write-Host "==> Done."
