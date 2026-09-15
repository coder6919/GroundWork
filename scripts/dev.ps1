<#
  Windows dev helper - mirrors the Makefile for hosts without `make`.
  Usage:  .\scripts\dev.ps1 <command>
  Commands: up | down | logs | ps | health | test | test-integration | verify | clean
#>
param(
  [Parameter(Position = 0)]
  [ValidateSet('up', 'down', 'logs', 'ps', 'health', 'test', 'test-integration', 'verify', 'clean', 'help')]
  [string]$Command = 'help'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Invoke-Checked($cmd) {
  Write-Host "==> $cmd" -ForegroundColor Cyan
  Invoke-Expression $cmd
  if ($LASTEXITCODE -ne 0) { throw "command failed ($LASTEXITCODE): $cmd" }
}

switch ($Command) {
  'up'   { Invoke-Checked 'docker compose up --build -d' }
  'down' { Invoke-Checked 'docker compose down' }
  'logs' { Invoke-Checked 'docker compose logs -f --tail=100' }
  'ps'   { Invoke-Checked 'docker compose ps' }
  'health' { Invoke-Checked 'bash ./scripts/healthcheck.sh' }
  'test' {
    Invoke-Checked 'docker compose run --rm --no-deps rag-core uv run --frozen pytest -v -m "not integration"'
    Invoke-Checked 'docker compose run --rm --no-deps api npm test'
  }
  'test-integration' {
    Invoke-Checked 'docker compose run --rm --no-deps -e RUN_INTEGRATION=1 rag-core uv run --frozen pytest -v -m integration'
    Invoke-Checked 'docker compose run --rm --no-deps -e RUN_INTEGRATION=1 api npm test'
  }
  'verify' {
    Invoke-Checked 'docker compose up --build -d'
    Invoke-Checked 'bash ./scripts/wait-for-health.sh'
    Invoke-Checked 'bash ./scripts/healthcheck.sh'
    & $PSCommandPath test
    & $PSCommandPath test-integration
  }
  'clean' { Invoke-Checked 'docker compose down -v --remove-orphans' }
  default {
    Write-Host 'Commands: up | down | logs | ps | health | test | test-integration | verify | clean'
  }
}
