#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="$repo_root/.venv-wsl/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python3 -m venv .venv-wsl
fi

if ! "$python_bin" -c "import pre_commit" >/dev/null 2>&1; then
  "$python_bin" -m pip install pre-commit
fi

"$python_bin" --version
"$python_bin" -m pre_commit run detect-secrets --all-files
"$python_bin" -m pre_commit run --all-files --show-diff-on-failure
"$python_bin" -m unittest discover -s tests -p "test_*.py"

echo "Frontend/npm/Playwright E2E: not applicable; no tracked package.json or frontend harness."
