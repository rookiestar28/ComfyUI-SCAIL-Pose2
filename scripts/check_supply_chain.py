from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


KNOWN_COMPROMISED_PACKAGES = {
    "@tanstack/setup",
    "guardrails-ai",
    "lightning",
    "mistralai",
}

SKIP_DIRS = {
    ".git",
    "." + "planning",
    "." + "sessions",
    ".venv",
    ".venv-wsl",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "ref" + "erence",
}


@dataclass(frozen=True)
class Finding:
    rule_id: str
    path: str
    message: str


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def add(findings: list[Finding], rule_id: str, path: Path, root: Path, message: str) -> None:
    findings.append(Finding(rule_id=rule_id, path=rel(path, root), message=message))


def iter_files(root: Path, include_install_trees: bool) -> list[Path]:
    files: list[Path] = []
    install_trees = {"node_modules", ".venv", ".venv-wsl", "venv"}
    for current, dirnames, filenames in os.walk(root):
        filtered = []
        for dirname in dirnames:
            if dirname in SKIP_DIRS:
                continue
            if not include_install_trees and dirname in install_trees:
                continue
            filtered.append(dirname)
        dirnames[:] = filtered
        current_path = Path(current)
        files.extend(current_path / filename for filename in filenames)
    return files


def scan_dependency_text(path: Path, root: Path, findings: list[Finding]) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return

    for package_name in KNOWN_COMPROMISED_PACKAGES:
        if re.search(rf"(^|[^a-z0-9_.@/-]){re.escape(package_name)}([^a-z0-9_.-]|$)", text):
            add(
                findings,
                "dependency.known-compromised-name",
                path,
                root,
                f"Manifest references known compromised package name {package_name!r}.",
            )


def scan_workflow(path: Path, root: Path, findings: list[Finding]) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return

    if re.search(r"(?m)^\s*pull_request_target\s*:", text):
        add(
            findings,
            "workflow.pull-request-target",
            path,
            root,
            "Workflow uses pull_request_target; keep registry publishing on trusted events only.",
        )

    for match in re.finditer(r"(?im)^\s*uses:\s*([^\s#]+)", text):
        action_ref = match.group(1)
        if action_ref.startswith("./") or "@" not in action_ref:
            continue
        action_name, ref = action_ref.rsplit("@", 1)
        if action_name.lower().startswith("actions/"):
            continue
        if not re.fullmatch(r"[a-f0-9]{40}", ref):
            add(
                findings,
                "workflow.mutable-third-party-action",
                path,
                root,
                f"Third-party action {action_name!r} is not pinned to a full commit SHA.",
            )


def scan_repository(root: Path, include_install_trees: bool) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []

    for path in iter_files(root, include_install_trees=include_install_trees):
        relative = rel(path, root)
        if path.name in {"pyproject.toml", "requirements.txt", "requirements-dev.txt"}:
            scan_dependency_text(path, root, findings)
        if relative.startswith(".github/workflows/") and path.suffix.lower() in {".yml", ".yaml"}:
            scan_workflow(path, root, findings)

    return sorted(findings, key=lambda finding: (finding.rule_id, finding.path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static supply-chain gate for release workflows.")
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument("--skip-install-trees", action="store_true", help="Skip local install/cache trees.")
    args = parser.parse_args(argv)

    findings = scan_repository(Path(args.root), include_install_trees=not args.skip_install_trees)
    if findings:
        print(f"Supply-chain gate found {len(findings)} issue(s):")
        for finding in findings:
            print(f"- {finding.rule_id} {finding.path}: {finding.message}")
        return 1

    print("PASS supply-chain gate found no blocking indicators.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
