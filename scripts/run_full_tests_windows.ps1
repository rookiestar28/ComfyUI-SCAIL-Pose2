$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    python -m venv .venv
}

& $Python -c "import pre_commit" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $Python -m pip install pre-commit
}

& $Python --version
& $Python -m pre_commit run detect-secrets --all-files
& $Python -m pre_commit run --all-files --show-diff-on-failure
& $Python -m unittest discover -s tests -p "test_*.py"

Write-Host "Frontend/npm/Playwright E2E: not applicable; no tracked package.json or frontend harness."
