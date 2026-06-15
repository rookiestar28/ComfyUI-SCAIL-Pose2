from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class ValidationHarnessTests(unittest.TestCase):
    def test_full_test_scripts_exist(self) -> None:
        self.assertTrue((ROOT / "scripts" / "run_full_tests_windows.ps1").exists())
        self.assertTrue((ROOT / "scripts" / "run_full_tests_linux.sh").exists())

    def test_windows_script_runs_required_gate_with_repo_local_venv(self) -> None:
        script = read_text("scripts/run_full_tests_windows.ps1")

        self.assertIn(".venv\\Scripts\\python.exe", script)
        self.assertIn("detect-secrets --all-files", script)
        self.assertIn("--show-diff-on-failure", script)
        self.assertIn("unittest discover -s tests", script)
        self.assertIn("not applicable", script)

    def test_linux_script_runs_required_gate_with_repo_local_venv(self) -> None:
        script = read_text("scripts/run_full_tests_linux.sh")

        self.assertIn(".venv-wsl/bin/python", script)
        self.assertIn("detect-secrets --all-files", script)
        self.assertIn("--show-diff-on-failure", script)
        self.assertIn("unittest discover -s tests", script)
        self.assertIn("not applicable", script)

    def test_test_sop_points_to_present_full_test_scripts(self) -> None:
        sop = read_text("tests/TEST_SOP.md")

        self.assertIn("scripts/run_full_tests_windows.ps1", sop)
        self.assertIn("scripts/run_full_tests_linux.sh", sop)
        self.assertIn("repo-local virtual environments", sop)
        self.assertIn("Frontend / npm / Playwright Policy", sop)


if __name__ == "__main__":
    unittest.main()
