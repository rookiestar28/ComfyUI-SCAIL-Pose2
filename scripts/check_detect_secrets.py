from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


EXCLUDE_FILES = (
    r"(^|[\\/])("
    r"\.git|\.planning|\.sessions|reference|REFERENCE|\.reference|"
    r"\.venv|\.venv-wsl|\.venv-[^\\/]+|venv|ENV|"
    r"__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|"
    r"node_modules|dist|build|tmp|temp|\.tmp"
    r")([\\/]|$)"
)


def main() -> int:
    if not Path(".git").exists():
        print("detect-secrets check must be run from the repository root", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        "-m",
        "detect_secrets",
        "scan",
        "--all-files",
        "--slim",
        "--exclude-files",
        EXCLUDE_FILES,
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        sys.stdout.write(completed.stdout)
        return completed.returncode

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        print("detect-secrets produced non-JSON output", file=sys.stderr)
        sys.stdout.write(completed.stdout)
        return 2

    results = payload.get("results", {})
    findings = {path: values for path, values in results.items() if values}
    if findings:
        print("Potential secrets detected. Review before committing:")
        for path, values in sorted(findings.items()):
            print(f"- {path}: {len(values)} finding(s)")
        return 1

    print("No secrets detected in non-internal repository files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
