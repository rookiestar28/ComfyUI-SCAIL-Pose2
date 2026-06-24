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

    def test_nlf_predict_bbox_formatter_preserves_multi_person_candidates(self) -> None:
        import_root_package()
        nodes_module = sys.modules[f"{PACKAGE_NAME}.nodes"]

        formatted = nodes_module._format_nlf_detected_boxes(
            [
                [[1, 2, 3, 4], [10, 20, 5, 6]],
                [[7, 8, 1, 2]],
                [],
            ]
        )

        self.assertEqual(
            [[1.0, 2.0, 4.0, 6.0], [10.0, 20.0, 15.0, 26.0]],
            formatted[0],
        )
        self.assertEqual([7.0, 8.0, 8.0, 10.0], formatted[1])
        self.assertEqual([0.0, 0.0, 0.0, 0.0], formatted[2])

    def test_nlf_predict_prefers_filtered_result_boxes_over_detector_boxes(self) -> None:
        import_root_package()
        nodes_module = sys.modules[f"{PACKAGE_NAME}.nodes"]
        detector_boxes = [[[1, 2, 3, 4, 0.9], [10, 20, 5, 6, 0.8]]]
        result_boxes = [[[10, 20, 5, 6, 0.8]]]

        selected = nodes_module._nlf_result_boxes_or_detector_boxes(
            {"boxes": result_boxes},
            detector_boxes,
        )

        self.assertIs(result_boxes, selected)

    def test_nlf_predict_falls_back_to_detector_boxes_for_legacy_results(self) -> None:
        import_root_package()
        nodes_module = sys.modules[f"{PACKAGE_NAME}.nodes"]
        detector_boxes = [[[1, 2, 3, 4, 0.9]]]

        selected = nodes_module._nlf_result_boxes_or_detector_boxes(
            {"poses3d": []},
            detector_boxes,
        )

        self.assertIs(detector_boxes, selected)

    def test_render_device_gpu_prefers_cuda_when_available(self) -> None:
        import_root_package()
        nodes_module = sys.modules[f"{PACKAGE_NAME}.nodes"]
        expected = (
            "cuda"
            if nodes_module.torch is not None and nodes_module.torch.cuda.is_available()
            else "gpu"
        )

        self.assertEqual(
            expected,
            nodes_module._resolve_taichi_render_device_key("gpu"),
        )
        self.assertEqual(
            "cuda",
            nodes_module._resolve_taichi_render_device_key("cuda"),
        )

    def test_render_nlf_poses_exposes_half_output_render_dimensions(self) -> None:
        package = import_root_package()
        render_node = package.NODE_CLASS_MAPPINGS["RenderNLFPoses"]
        required = render_node.INPUT_TYPES()["required"]
        optional = render_node.INPUT_TYPES()["optional"]

        self.assertEqual(("nlf_poses", "render_width", "render_height"), tuple(required))
        self.assertNotIn("width", required)
        self.assertNotIn("height", required)
        self.assertEqual("INT", required["render_width"][0])
        self.assertEqual("INT", required["render_height"][0])
        self.assertIn("pose_video_mask", optional)
        self.assertEqual("IMAGE", optional["pose_video_mask"][0])
        self.assertIn("bboxes", optional)
        self.assertEqual("BBOX", optional["bboxes"][0])
        self.assertNotIn("render_width", optional)
        self.assertNotIn("render_height", optional)

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
                (2.0, 2.0, 4.0, 4.0),
                frame_bboxes(aligned_image, kind="pose_image")[0].to_tuple(),
            )
            self.assertEqual(
                (2.0, 2.0, 4.0, 4.0),
                frame_bboxes(aligned_mask, kind="mask")[0].to_tuple(),
            )
        finally:
            for name, original_module in original_modules.items():
                if original_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original_module

    def test_render_nlf_poses_applies_dwpose_overlay_after_half_size_repair(self) -> None:
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
        pose_draw_package_name = f"{package_prefix}.pose_draw"
        draw_module_name = f"{pose_draw_package_name}.draw_pose_utils"

        nlf_package = types.ModuleType(nlf_package_name)
        render_module = types.ModuleType(render_module_name)
        align3d_module = types.ModuleType(align3d_module_name)
        pose_draw_package = types.ModuleType(pose_draw_package_name)
        pose_draw_package.__path__ = []
        draw_module = types.ModuleType(draw_module_name)
        draw_calls = []

        def intrinsic_matrix_from_field_of_view(_shape):
            return np.eye(3, dtype=np.float32)

        def render_nlf_as_images(_pose_input, dw_pose_input, height, width, *_args, **_kwargs):
            self.assertIsNone(dw_pose_input, "raw NLF render must stay body-only")
            frame = np.zeros((height, width, 4), dtype=np.uint8)
            frame[0:2, 0:2, 2] = 255
            frame[0:2, 0:2, 3] = 255
            return [frame]

        def draw_pose_to_canvas_np(_poses, **kwargs):
            draw_calls.append(
                (
                    kwargs["H"],
                    kwargs["W"],
                    kwargs["show_face_flag"],
                    kwargs["show_hand_flag"],
                )
            )
            canvas = np.zeros((kwargs["H"], kwargs["W"], 3), dtype=np.uint8)
            canvas[0, kwargs["W"] - 1, 0] = 255
            return [canvas]

        render_module.intrinsic_matrix_from_field_of_view = intrinsic_matrix_from_field_of_view
        render_module.render_nlf_as_images = render_nlf_as_images
        render_module.render_multi_nlf_as_images = render_nlf_as_images
        render_module.shift_dwpose_according_to_nlf = lambda *_args, **_kwargs: None
        render_module.process_data_to_COCO_format = lambda value: value
        align3d_module.solve_new_camera_params_central = lambda *_args, **_kwargs: (np.eye(3), 1.0, 1.0)
        align3d_module.solve_new_camera_params_down = lambda *_args, **_kwargs: (np.eye(3), 1.0, 1.0)
        draw_module.draw_pose_to_canvas_np = draw_pose_to_canvas_np

        original_modules = {
            name: sys.modules.get(name)
            for name in (
                nlf_package_name,
                render_module_name,
                align3d_module_name,
                pose_draw_package_name,
                draw_module_name,
            )
        }
        try:
            sys.modules[nlf_package_name] = nlf_package
            sys.modules[render_module_name] = render_module
            sys.modules[align3d_module_name] = align3d_module
            sys.modules[pose_draw_package_name] = pose_draw_package
            sys.modules[draw_module_name] = draw_module

            pose_input = [torch.zeros((1, 1, 3), dtype=torch.float32)]
            dw_poses = {
                "poses": [
                    {
                        "bodies": {
                            "candidate": np.zeros((1, 18, 2), dtype=np.float32),
                        },
                    }
                ],
                "swap_hands": False,
            }

            image, mask = render_node.predict(
                pose_input,
                8,
                8,
                dw_poses=dw_poses,
                bboxes=[[4, 4, 8, 8]],
                render_backend="torch",
            )

            self.assertEqual([(4, 4, True, True)], draw_calls)
            self.assertEqual(1.0, float(image[0, 0, 3, 0].item()))
            self.assertEqual(1.0, float(mask[0, 0, 3].item()))
            self.assertTrue(bool((image[0, 2:4, 2:4, 2] > 0.5).all().item()))
        finally:
            for name, original_module in original_modules.items():
                if original_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original_module

    def test_dwpose_overlay_skips_multi_identity_person_count_mismatch(self) -> None:
        try:
            import numpy as np
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"render dependencies unavailable: {exc}")

        import_root_package()
        nodes_module = sys.modules[f"{PACKAGE_NAME}.nodes"]
        pose_draw_package_name = f"{PACKAGE_NAME}.pose_draw"
        draw_module_name = f"{pose_draw_package_name}.draw_pose_utils"

        pose_draw_package = types.ModuleType(pose_draw_package_name)
        pose_draw_package.__path__ = []
        draw_module = types.ModuleType(draw_module_name)

        def draw_pose_to_canvas_np(*_args, **_kwargs):
            raise AssertionError("mismatched DWPose overlay must be skipped")

        draw_module.draw_pose_to_canvas_np = draw_pose_to_canvas_np
        original_modules = {
            name: sys.modules.get(name)
            for name in (pose_draw_package_name, draw_module_name)
        }
        try:
            sys.modules[pose_draw_package_name] = pose_draw_package
            sys.modules[draw_module_name] = draw_module

            frames = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
            mask = torch.zeros((1, 4, 4), dtype=torch.float32)
            dw_pose_input = [
                {
                    "bodies": {
                        "candidate": np.zeros((1, 18, 2), dtype=np.float32),
                    },
                }
            ]

            output_frames, output_mask = nodes_module._overlay_dwpose_2d_on_frames(
                frames_tensor=frames,
                mask=mask,
                dw_pose_input=dw_pose_input,
                draw_face=True,
                draw_hands=True,
                identity_count=2,
            )

            self.assertTrue(torch.equal(frames, output_frames))
            self.assertTrue(torch.equal(mask, output_mask))
        finally:
            for name, original_module in original_modules.items():
                if original_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original_module

    def test_render_nlf_poses_renders_source_size_then_emits_half_output(self) -> None:
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
                16,
                16,
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

            self.assertEqual((0.0, 0.0, 1.0, 1.0), frame_bboxes(raw_image, kind="pose_image")[0].to_tuple())
            self.assertEqual((2.0, 2.0, 4.0, 4.0), frame_bboxes(aligned_image, kind="pose_image")[0].to_tuple())
            self.assertEqual((0.0, 0.0, 1.0, 1.0), frame_bboxes(raw_mask, kind="mask")[0].to_tuple())
            self.assertEqual((2.0, 2.0, 4.0, 4.0), frame_bboxes(aligned_mask, kind="mask")[0].to_tuple())
        finally:
            for name, original_module in original_modules.items():
                if original_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original_module

    def test_render_nlf_poses_composes_multi_person_identities_independently(self) -> None:
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
        render_calls = []

        def intrinsic_matrix_from_field_of_view(_shape):
            return np.eye(3, dtype=np.float32)

        def render_nlf_as_images(pose_stream, _dw_pose_input, height, width, *_args, **_kwargs):
            marker = int(float(pose_stream[0][0, 0, 0].item()))
            render_calls.append(marker)
            frame = np.zeros((height, width, 4), dtype=np.uint8)
            if marker == 0:
                frame[0:2, 0:2, 2] = 255
                frame[0:2, 0:2, 3] = 255
            else:
                frame[0:2, 6:8, 1] = 255
                frame[0:2, 6:8, 3] = 255
            return [frame]

        def render_multi_nlf_as_images(*_args, **_kwargs):
            raise AssertionError("multi-person semantic masks must use per-identity rendering")

        render_module.intrinsic_matrix_from_field_of_view = intrinsic_matrix_from_field_of_view
        render_module.render_nlf_as_images = render_nlf_as_images
        render_module.render_multi_nlf_as_images = render_multi_nlf_as_images
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

            pose_frame = torch.zeros((2, 1, 3), dtype=torch.float32)
            pose_frame[1, 0, 0] = 1.0
            pose_input = [pose_frame]
            pose_video_mask = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
            pose_video_mask[0, 0:2, 6:8, 2] = 1.0
            pose_video_mask[0, 0:2, 0:2, 0] = 1.0

            image, mask = render_node.predict(
                pose_input,
                8,
                8,
                bboxes=[[[0, 0, 2, 2], [6, 0, 8, 2]]],
                pose_video_mask=pose_video_mask,
                render_backend="torch",
            )

            self.assertEqual([1, 0], render_calls)
            self.assertEqual((1, 4, 4, 3), tuple(image.shape))
            self.assertEqual((1, 4, 4), tuple(mask.shape))
            self.assertGreater(float(mask[0, 0, 0].item()), 0.5)
            self.assertGreater(float(mask[0, 0, 3].item()), 0.5)
            self.assertEqual((0.0, 0.0, 4.0, 1.0), frame_bboxes(mask, kind="mask")[0].to_tuple())
        finally:
            for name, original_module in original_modules.items():
                if original_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original_module


if __name__ == "__main__":
    unittest.main()
