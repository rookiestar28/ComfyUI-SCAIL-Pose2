from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "ComfyUI_SCAIL_Pose2_TestPackage"

V1_NODE_DISPLAY_NAMES = {
    "PoseDetectionVitPoseToDWPose": "Pose Detection VitPose to DWPose",
    "RenderNLFPoses": "Render NLF Poses",
    "ConvertOpenPoseKeypointsToDWPose": "Convert OpenPose Keypoints to DWPose",
    "SaveNLFPosesAs3D": "Save NLF Poses as 3D Animation",
    "NLFModelLoader": "NLF Model Loader",
    "NLFPredictPoses": "NLF Predict Poses",
}


def import_root_package():
    for name in list(sys.modules):
        if name == PACKAGE_NAME or name.startswith(f"{PACKAGE_NAME}."):
            del sys.modules[name]

    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V1PosePipelineTests(unittest.TestCase):
    def test_root_package_imports_and_preserves_node_keys(self) -> None:
        package = import_root_package()

        expected_keys = set(V1_NODE_DISPLAY_NAMES)
        self.assertTrue(expected_keys.issubset(package.NODE_CLASS_MAPPINGS))
        self.assertNotIn("SCAILPose2WanSCAILImages", package.NODE_CLASS_MAPPINGS)
        self.assertFalse(
            any(name.startswith("WanVideoWrapper") for name in sys.modules),
            "base package import must not import WanVideoWrapper",
        )

    def test_v1_display_names_are_unchanged(self) -> None:
        package = import_root_package()

        for node_key, display_name in V1_NODE_DISPLAY_NAMES.items():
            with self.subTest(node_key=node_key):
                self.assertEqual(display_name, package.NODE_DISPLAY_NAME_MAPPINGS[node_key])

    def test_render_nlf_poses_output_metadata_is_preserved(self) -> None:
        package = import_root_package()
        render_node = package.NODE_CLASS_MAPPINGS["RenderNLFPoses"]

        self.assertEqual(("IMAGE", "MASK"), render_node.RETURN_TYPES)
        self.assertEqual(("image", "mask"), render_node.RETURN_NAMES)


if __name__ == "__main__":
    unittest.main()
