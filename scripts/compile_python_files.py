from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path


EXCLUDED_PREFIXES = (
    "reference/",
    "REFERENCE/",
    ".reference/",
    ".planning/",
    ".sessions/",
)


def git_files(args: list[str]) -> list[str]:
    completed = subprocess.run(["git", *args], text=True, capture_output=True, check=True)
    return [line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()]


def main() -> int:
    if not Path(".git").exists():
        print("compile check must be run from the repository root", file=sys.stderr)
        return 2

    candidates = set(git_files(["ls-files", "*.py"]))
    candidates.update(git_files(["ls-files", "--others", "--exclude-standard", "*.py"]))
    files = [
        path
        for path in sorted(candidates)
        if not path.startswith(EXCLUDED_PREFIXES) and Path(path).is_file()
    ]

    if not files:
        print("No tracked or candidate Python files to compile.")
        return 0

    failures: list[tuple[str, Exception]] = []
    for path in files:
        try:
            py_compile.compile(path, doraise=True)
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append((path, exc))

    if failures:
        for path, exc in failures:
            print(f"{path}: {exc}", file=sys.stderr)
        return 1

    print(f"Compiled {len(files)} Python file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
