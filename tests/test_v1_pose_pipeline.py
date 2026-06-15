from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

from scail2 import wanvideo_contracts


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


@dataclass(frozen=True)
class FakeImageBatch:
    shape: tuple[int, int, int, int]


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

        expected_keys = set(V1_NODE_DISPLAY_NAMES) | {"SCAILPose2WanSCAILImages"}
        self.assertTrue(expected_keys.issubset(package.NODE_CLASS_MAPPINGS))
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

    def test_wan_scail_images_adapter_passes_through_valid_payload(self) -> None:
        package = import_root_package()
        adapter_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2WanSCAILImages"]
        ref_image = FakeImageBatch((1, 64, 96, 3))
        pose_images = FakeImageBatch((4, 64, 96, 3))

        payload, ref_out, pose_out, clip_out, width, height, frames = adapter_cls().build(
            ref_image=ref_image,
            pose_images=pose_images,
            width=96,
            height=64,
            num_frames=4,
        )

        self.assertIs(ref_image, ref_out)
        self.assertIs(pose_images, pose_out)
        self.assertIs(ref_image, clip_out)
        self.assertEqual((96, 64, 4), (width, height, frames))
        self.assertEqual("wan_scail_v1_images", payload["kind"])
        self.assertEqual(4, payload["image_shapes"]["pose_images"].frames)
        self.assertIn(
            "mask_latents_28_channel",
            payload["unsupported_wrapper_features"],
        )

    def test_wan_scail_images_adapter_rejects_invalid_payloads(self) -> None:
        package = import_root_package()
        adapter_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2WanSCAILImages"]
        ref_image = FakeImageBatch((1, 64, 96, 3))
        pose_images = FakeImageBatch((4, 64, 96, 3))

        with self.assertRaisesRegex(ValueError, "width must be a positive integer"):
            adapter_cls().build(ref_image, pose_images, 0, 64, 4)

        with self.assertRaisesRegex(ValueError, "pose_images frame count"):
            adapter_cls().build(ref_image, pose_images, 96, 64, 5)

        with self.assertRaisesRegex(ValueError, "ref_image width"):
            adapter_cls().build(FakeImageBatch((1, 64, 95, 3)), pose_images, 96, 64, 4)

    def test_adapter_payload_maps_to_mocked_wanvideo_sockets(self) -> None:
        package = import_root_package()
        adapter_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2WanSCAILImages"]
        ref_image = FakeImageBatch((1, 32, 32, 3))
        pose_images = FakeImageBatch((2, 32, 32, 3))

        payload = adapter_cls().build(ref_image, pose_images, 32, 32, 2)[0]
        pose_socket = payload["socket_map"]["pose_images"]

        self.assertEqual(
            wanvideo_contracts.NODE_WAN_ADD_SCAIL_POSE,
            pose_socket["wrapper_node"],
        )
        self.assertEqual("pose_images", pose_socket["wrapper_socket"])
        self.assertEqual(wanvideo_contracts.TYPE_IMAGE, pose_socket["comfy_type"])


if __name__ == "__main__":
    unittest.main()
