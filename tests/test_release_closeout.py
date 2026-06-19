from __future__ import annotations

import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_RELEASE_NODE_KEYS = {
    "NLFModelLoader",
    "NLFPredictPoses",
    "PoseDetectionVitPoseToDWPose",
    "ConvertOpenPoseKeypointsToDWPose",
    "RenderNLFPoses",
    "SaveNLFPosesAs3D",
    "SCAIL2SAM3DependencyCheck",
    "SCAILPose2ColoredMask",
    "SCAILPose2SCAIL2Condition",
    "SCAILPose2WanVideoSCAIL2Adapter",
}


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_root_package():
    module_name = "ComfyUI_SCAIL_Pose2_release_closeout"
    package_init = ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_init,
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load root package spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


class ReleaseCloseoutTests(unittest.TestCase):
    def test_required_public_release_files_exist(self) -> None:
        required_paths = [
            "pyproject.toml",
            "requirements.txt",
            "readme.md",
            ".comfyignore",
            "__init__.py",
            "nodes.py",
            "nodes_sam3_preprocessing.py",
            "scripts/run_full_tests_windows.ps1",
            "scripts/run_full_tests_linux.sh",
            "scripts/check_supply_chain.py",
            ".github/workflows/publish.yml",
            "workflow_skeletons/wan_scail_v1_pose_control.json",
            "workflow_skeletons/scail2_condition_builder.json",
            "workflow_skeletons/wanvideo_native_scail2.json",
            "workflow_skeletons/wananimate_fallback.json",
        ]

        for path in required_paths:
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).exists(), path)

    def test_release_metadata_points_to_pose2_package(self) -> None:
        project = tomllib.loads(read_text("pyproject.toml"))["project"]

        self.assertEqual(project["name"], "comfyui-scail-pose2")
        self.assertRegex(project["version"], r"^\d+\.\d+\.\d+$")
        self.assertEqual(project["readme"], "readme.md")

    def test_root_package_exposes_expected_release_nodes(self) -> None:
        module = load_root_package()

        self.assertTrue(EXPECTED_RELEASE_NODE_KEYS.issubset(module.NODE_CLASS_MAPPINGS))
        self.assertTrue(EXPECTED_RELEASE_NODE_KEYS.issubset(module.NODE_DISPLAY_NAME_MAPPINGS))

    def test_test_sop_release_facts_are_current(self) -> None:
        sop = read_text("tests/TEST_SOP.md")

        self.assertNotIn("no restored product node package yet", sop)
        self.assertIn("restored v1 pose nodes", sop)
        self.assertIn("SCAIL-2 helper modules", sop)
        self.assertIn("release validation tests", sop)

    def test_readme_lists_final_scail2_nodes_and_boundaries(self) -> None:
        readme = read_text("readme.md")

        for node_key in (
            "SCAILPose2ColoredMask",
            "SCAILPose2SCAIL2Condition",
            "SCAILPose2WanVideoSCAIL2Adapter",
        ):
            with self.subTest(node_key=node_key):
                self.assertIn(node_key, readme)

        self.assertIn("SCAIL-2 adapter payload", readme)
        self.assertNotIn("SCAILPose2WanSCAILImages", readme)
        self.assertIn("WanVideoAddSCAIL2ConditionEmbeds", readme)
        self.assertIn("lossy v1 fallback", readme)
        self.assertNotIn("reference/docs", readme)
        self.assertNotIn(".planning", readme)

    def test_publish_workflow_matches_registry_release_boundary(self) -> None:
        workflow = read_text(".github/workflows/publish.yml")

        self.assertIn("workflow_dispatch", workflow)
        self.assertIn("pyproject.toml", workflow)
        self.assertIn("github.repository_owner == 'rookiestar28'", workflow)
        self.assertIn("python scripts/check_supply_chain.py --skip-install-trees", workflow)
        self.assertIn("secrets.REGISTRY_ACCESS_TOKEN", workflow)
        self.assertRegex(
            workflow,
            r"Comfy-Org/publish-node-action@[a-f0-9]{40}",
        )
        self.assertNotIn("pull_request_target", workflow)


if __name__ == "__main__":
    unittest.main()
