from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_module(path: str):
    module_path = ROOT / path
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GovernanceBaselineTests(unittest.TestCase):
    def test_test_sops_are_scail_pose2_specific(self) -> None:
        combined = "\n".join(
            [
                read_text("tests/TEST_SOP.md"),
                read_text("tests/E2E_TESTING_NOTICE.md"),
                read_text("tests/E2E_TESTING_SOP.md"),
            ]
        )

        self.assertIn("ComfyUI-SCAIL-Pose2", combined)
        self.assertIn("SCAIL2_CONDITION", combined)
        self.assertIn("WanVideoWrapper", combined)

        stale_terms = [
            "ComfyUI Text Processor",
            "advanced_text_filter",
            "text_scraper",
            "RookieUI",
        ]
        for term in stale_terms:
            self.assertNotIn(term, combined)

    def test_pre_commit_hooks_are_repo_local(self) -> None:
        config = read_text(".pre-commit-config.yaml")

        self.assertIn("repo: local", config)
        self.assertIn("id: detect-secrets", config)
        self.assertIn("scripts/check_detect_secrets.py", config)
        self.assertIn("id: python-compile", config)
        self.assertIn("scripts/compile_python_files.py", config)

    def test_governance_scripts_exclude_internal_reference_paths(self) -> None:
        secret_check = load_module("scripts/check_detect_secrets.py")
        compile_check = load_module("scripts/compile_python_files.py")

        for term in ("reference", ".planning", ".sessions"):
            self.assertIn(term, secret_check.EXCLUDE_FILES)

        for prefix in ("reference/", ".planning/", ".sessions/"):
            self.assertIn(prefix, compile_check.EXCLUDED_PREFIXES)


if __name__ == "__main__":
    unittest.main()
