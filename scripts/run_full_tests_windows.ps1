$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Invoke-CheckedExternal {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock] $Command
    )
    & $Command
    # CRITICAL: PowerShell ErrorActionPreference does not fail on native
    # process exit codes, so every gate command must be checked explicitly.
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    python -m venv .venv
}

& $Python -c "import pre_commit" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $Python -m pip install pre-commit
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Invoke-CheckedExternal { & $Python --version }
Invoke-CheckedExternal { & $Python -m pre_commit run detect-secrets --all-files }
Invoke-CheckedExternal { & $Python -m pre_commit run --all-files --show-diff-on-failure }
Invoke-CheckedExternal { & $Python -m unittest discover -s tests -p "test_*.py" }

Write-Host "Frontend/npm/Playwright E2E: not applicable; no tracked package.json or frontend harness."
