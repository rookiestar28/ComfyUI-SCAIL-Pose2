from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

from scail2.geometry import frame_bboxes


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

    def test_render_nlf_poses_exposes_optional_pose_video_mask_alignment(self) -> None:
        package = import_root_package()
        render_node = package.NODE_CLASS_MAPPINGS["RenderNLFPoses"]
        optional = render_node.INPUT_TYPES()["optional"]

        self.assertIn("pose_video_mask", optional)
        self.assertEqual("IMAGE", optional["pose_video_mask"][0])
        self.assertIn("bboxes", optional)
        self.assertEqual("BBOX", optional["bboxes"][0])
        self.assertIn("render_width", optional)
        self.assertEqual("INT", optional["render_width"][0])
        self.assertIn("render_height", optional)
        self.assertEqual("INT", optional["render_height"][0])

    def test_render_nlf_poses_applies_optional_bbox_alignment(self) -> None:
        try:
            import numpy as np
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"render dependencies unavailable: {exc}")

        package = import_root_package()
        render_node = package.NODE_CLASS_MAPPINGS["RenderNLFPoses"]()
        package_prefix = PACKAGE_NAME
        nlf_package_name = f"{package_prefix}.NLFPoseExtract"
        render_module_name = f"{nlf_package_name}.nlf_render"
        align3d_module_name = f"{nlf_package_name}.align3d"

        nlf_package = types.ModuleType(nlf_package_name)
        render_module = types.ModuleType(render_module_name)
        align3d_module = types.ModuleType(align3d_module_name)

        def intrinsic_matrix_from_field_of_view(_shape):
            return np.eye(3, dtype=np.float32)

        def render_nlf_as_images(_pose_input, _dw_pose_input, height, width, *_args, **_kwargs):
            frame = np.zeros((height, width, 4), dtype=np.uint8)
            frame[0:2, 0:2, 2] = 255
            frame[0:2, 0:2, 3] = 255
            return [frame]

        render_module.intrinsic_matrix_from_field_of_view = intrinsic_matrix_from_field_of_view
        render_module.render_nlf_as_images = render_nlf_as_images
        render_module.render_multi_nlf_as_images = render_nlf_as_images
        render_module.shift_dwpose_according_to_nlf = lambda *_args, **_kwargs: None
        render_module.process_data_to_COCO_format = lambda value: value
        align3d_module.solve_new_camera_params_central = lambda *_args, **_kwargs: (np.eye(3), 1.0, 1.0)
        align3d_module.solve_new_camera_params_down = lambda *_args, **_kwargs: (np.eye(3), 1.0, 1.0)

        original_modules = {
            name: sys.modules.get(name)
            for name in (nlf_package_name, render_module_name, align3d_module_name)
        }
        try:
            sys.modules[nlf_package_name] = nlf_package
            sys.modules[render_module_name] = render_module
            sys.modules[align3d_module_name] = align3d_module

            pose_input = [torch.zeros((1, 1, 3), dtype=torch.float32)]

            aligned_image, aligned_mask = render_node.predict(
                pose_input,
                8,
                8,
                bboxes=[[4, 4, 8, 8]],
                render_backend="torch",
            )

            self.assertEqual(
                (4.0, 4.0, 8.0, 8.0),
                frame_bboxes(aligned_image, kind="pose_image")[0].to_tuple(),
            )
            self.assertEqual(
                (4.0, 4.0, 8.0, 8.0),
                frame_bboxes(aligned_mask, kind="mask")[0].to_tuple(),
            )
        finally:
            for name, original_module in original_modules.items():
                if original_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original_module

    def test_render_nlf_poses_can_render_full_size_then_downsample(self) -> None:
        try:
            import numpy as np
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"render dependencies unavailable: {exc}")

        package = import_root_package()
        render_node = package.NODE_CLASS_MAPPINGS["RenderNLFPoses"]()
        package_prefix = PACKAGE_NAME
        nlf_package_name = f"{package_prefix}.NLFPoseExtract"
        render_module_name = f"{nlf_package_name}.nlf_render"
        align3d_module_name = f"{nlf_package_name}.align3d"

        nlf_package = types.ModuleType(nlf_package_name)
        render_module = types.ModuleType(render_module_name)
        align3d_module = types.ModuleType(align3d_module_name)
        recorded_shapes = []

        def intrinsic_matrix_from_field_of_view(shape):
            recorded_shapes.append(tuple(shape))
            return np.eye(3, dtype=np.float32)

        def render_nlf_as_images(_pose_input, _dw_pose_input, height, width, *_args, **_kwargs):
            frame = np.zeros((height, width, 4), dtype=np.uint8)
            frame[0:4, 0:4, 2] = 255
            frame[0:4, 0:4, 3] = 255
            return [frame]

        render_module.intrinsic_matrix_from_field_of_view = intrinsic_matrix_from_field_of_view
        render_module.render_nlf_as_images = render_nlf_as_images
        render_module.render_multi_nlf_as_images = render_nlf_as_images
        render_module.shift_dwpose_according_to_nlf = lambda *_args, **_kwargs: None
        render_module.process_data_to_COCO_format = lambda value: value
        align3d_module.solve_new_camera_params_central = lambda *_args, **_kwargs: (np.eye(3), 1.0, 1.0)
        align3d_module.solve_new_camera_params_down = lambda *_args, **_kwargs: (np.eye(3), 1.0, 1.0)

        original_modules = {
            name: sys.modules.get(name)
            for name in (nlf_package_name, render_module_name, align3d_module_name)
        }
        try:
            sys.modules[nlf_package_name] = nlf_package
            sys.modules[render_module_name] = render_module
            sys.modules[align3d_module_name] = align3d_module

            pose_input = [torch.zeros((1, 1, 3), dtype=torch.float32)]

            image, mask = render_node.predict(
                pose_input,
                8,
                8,
                render_width=16,
                render_height=16,
                render_backend="torch",
            )

            self.assertEqual((16, 16), recorded_shapes[0])
            self.assertEqual((1, 8, 8, 3), tuple(image.shape))
            self.assertEqual((1, 8, 8), tuple(mask.shape))
        finally:
            for name, original_module in original_modules.items():
                if original_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original_module

    def test_render_nlf_poses_applies_optional_pose_video_mask_alignment(self) -> None:
        try:
            import numpy as np
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"render dependencies unavailable: {exc}")

        package = import_root_package()
        render_node = package.NODE_CLASS_MAPPINGS["RenderNLFPoses"]()
        package_prefix = PACKAGE_NAME
        nlf_package_name = f"{package_prefix}.NLFPoseExtract"
        render_module_name = f"{nlf_package_name}.nlf_render"
        align3d_module_name = f"{nlf_package_name}.align3d"

        nlf_package = types.ModuleType(nlf_package_name)
        render_module = types.ModuleType(render_module_name)
        align3d_module = types.ModuleType(align3d_module_name)

        def intrinsic_matrix_from_field_of_view(_shape):
            return np.eye(3, dtype=np.float32)

        def render_nlf_as_images(_pose_input, _dw_pose_input, height, width, *_args, **_kwargs):
            frame = np.zeros((height, width, 4), dtype=np.uint8)
            frame[0:2, 0:2, 2] = 255
            frame[0:2, 0:2, 3] = 255
            return [frame]

        render_module.intrinsic_matrix_from_field_of_view = intrinsic_matrix_from_field_of_view
        render_module.render_nlf_as_images = render_nlf_as_images
        render_module.render_multi_nlf_as_images = render_nlf_as_images
        render_module.shift_dwpose_according_to_nlf = lambda *_args, **_kwargs: None
        render_module.process_data_to_COCO_format = lambda value: value
        align3d_module.solve_new_camera_params_central = lambda *_args, **_kwargs: (np.eye(3), 1.0, 1.0)
        align3d_module.solve_new_camera_params_down = lambda *_args, **_kwargs: (np.eye(3), 1.0, 1.0)

        original_modules = {
            name: sys.modules.get(name)
            for name in (nlf_package_name, render_module_name, align3d_module_name)
        }
        try:
            sys.modules[nlf_package_name] = nlf_package
            sys.modules[render_module_name] = render_module
            sys.modules[align3d_module_name] = align3d_module

            pose_input = [torch.zeros((1, 1, 3), dtype=torch.float32)]
            driving_mask = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
            driving_mask[0, 4:8, 4:8, 2] = 1.0

            raw_image, raw_mask = render_node.predict(pose_input, 8, 8, render_backend="torch")
            aligned_image, aligned_mask = render_node.predict(
                pose_input,
                8,
                8,
                pose_video_mask=driving_mask,
                render_backend="torch",
            )

            self.assertEqual((0.0, 0.0, 2.0, 2.0), frame_bboxes(raw_image, kind="pose_image")[0].to_tuple())
            self.assertEqual((4.0, 4.0, 8.0, 8.0), frame_bboxes(aligned_image, kind="pose_image")[0].to_tuple())
            self.assertEqual((0.0, 0.0, 2.0, 2.0), frame_bboxes(raw_mask, kind="mask")[0].to_tuple())
            self.assertEqual((4.0, 4.0, 8.0, 8.0), frame_bboxes(aligned_mask, kind="mask")[0].to_tuple())
        finally:
            for name, original_module in original_modules.items():
                if original_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original_module


if __name__ == "__main__":
    unittest.main()
