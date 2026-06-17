from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def dependency_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower().replace("_", "-")


class PackagingPublicDocsTests(unittest.TestCase):
    def test_pyproject_metadata_is_valid_for_pose2_package(self) -> None:
        data = tomllib.loads(read_text("pyproject.toml"))
        project = data["project"]
        comfy = data["tool"]["comfy"]

        self.assertEqual(project["name"], "comfyui-scail-pose2")
        self.assertEqual(project["version"], "0.1.0")
        self.assertEqual(project["authors"], [{"name": "rookiestar28"}])
        self.assertEqual(project["readme"], "readme.md")
        self.assertIn("SCAIL-2", project["description"])
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["license-files"], ["LICENSE"])
        self.assertEqual(
            data["project"]["urls"]["Repository"],
            "https://github.com/rookiestar28/ComfyUI-SCAIL-Pose2",
        )
        self.assertEqual(comfy["PublisherId"], "rookiestar")
        self.assertEqual(comfy["DisplayName"], "ComfyUI-SCAIL-Pose2")

    def test_requirements_match_hard_pyproject_dependencies(self) -> None:
        data = tomllib.loads(read_text("pyproject.toml"))
        pyproject_deps = data["project"]["dependencies"]
        requirement_deps = [
            line.strip()
            for line in read_text("requirements.txt").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        self.assertEqual(requirement_deps, pyproject_deps)

    def test_heavy_optional_runtime_packages_are_not_hard_dependencies(self) -> None:
        data = tomllib.loads(read_text("pyproject.toml"))
        dependency_names = {dependency_name(dep) for dep in data["project"]["dependencies"]}
        forbidden = {
            "comfyui",
            "comfyui-wanvideowrapper",
            "torch",
            "torchvision",
            "taichi",
            "ultralytics",
            "sam-2",
            "sam3",
        }

        self.assertTrue(dependency_names.isdisjoint(forbidden))

    def test_comfyignore_excludes_development_only_paths(self) -> None:
        comfyignore = {
            line.strip()
            for line in read_text(".comfyignore").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        required_patterns = {
            ".github/",
            ".gitattributes",
            ".pre-commit-config.yaml",
            "tests/",
            "scripts/",
        }

        self.assertTrue(required_patterns.issubset(comfyignore))


if __name__ == "__main__":
    unittest.main()
