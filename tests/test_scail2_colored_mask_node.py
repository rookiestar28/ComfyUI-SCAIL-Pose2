from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

from scail2.colored_masks import (
    BLACK_RGB_FLOAT,
    BLUE_RGB_FLOAT,
    RED_RGB_FLOAT,
    WHITE_RGB_FLOAT,
    materialize_comfy_image,
    render_scail2_colored_mask_pair,
)
from scail2.observability import safe_value_summary


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "ComfyUI_SCAIL_Pose2_ColoredMaskTestPackage"


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


def track_data(frames):
    height = len(frames[0][0])
    width = len(frames[0][0][0])
    return {
        "masks": frames,
        "orig_size": (height, width),
        "n_frames": len(frames),
    }


def pixel(image, frame=0, row=0, col=0):
    return tuple(image[frame][row][col])


class FakeTensorLike:
    shape = (5, 8, 8, 3)
    dtype = "float16"
    device = "cuda:0"

    def __init__(self):
        self.to_kwargs = None

    def detach(self):
        return self

    def to(self, **kwargs):
        self.to_kwargs = kwargs
        return self


@contextmanager
def fake_sam3_unpack_masks(unpacked_masks):
    module_names = (
        "comfy",
        "comfy.ldm",
        "comfy.ldm.sam3",
        "comfy.ldm.sam3.tracker",
    )
    previous_modules = {name: sys.modules.get(name) for name in module_names}

    comfy_module = types.ModuleType("comfy")
    ldm_module = types.ModuleType("comfy.ldm")
    sam3_module = types.ModuleType("comfy.ldm.sam3")
    tracker_module = types.ModuleType("comfy.ldm.sam3.tracker")
    tracker_module.unpack_masks = lambda _packed: unpacked_masks
    comfy_module.ldm = ldm_module
    ldm_module.sam3 = sam3_module
    sam3_module.tracker = tracker_module

    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.ldm"] = ldm_module
    sys.modules["comfy.ldm.sam3"] = sam3_module
    sys.modules["comfy.ldm.sam3.tracker"] = tracker_module
    try:
        yield
    finally:
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class Scail2ColoredMaskNodeTests(unittest.TestCase):
    def test_materialize_preserves_tensor_like_image_without_list_roundtrip(self) -> None:
        tensor = FakeTensorLike()

        materialized = materialize_comfy_image(tensor)

        self.assertIs(tensor, materialized)

    def test_observability_summary_is_shape_only_and_value_safe(self) -> None:
        tensor = FakeTensorLike()

        summary = safe_value_summary(tensor)

        self.assertEqual("FakeTensorLike", summary["type"])
        self.assertEqual([5, 8, 8, 3], summary["shape"])
        self.assertEqual("float16", summary["dtype"])
        self.assertEqual("cuda:0", summary["device"])
        self.assertNotIn("data", summary)

    def test_packed_masks_are_resized_to_orig_size_before_rendering(self) -> None:
        track = {
            "packed_masks": object(),
            "orig_size": (1, 1),
            "n_frames": 1,
        }

        with fake_sam3_unpack_masks([[[[True, False], [False, False]]]]):
            result = render_scail2_colored_mask_pair(
                track,
                object_indices="",
                sort_by="none",
                replacement_mode=False,
            )

        self.assertEqual(BLUE_RGB_FLOAT, pixel(result.pose_video_mask))
        self.assertEqual(WHITE_RGB_FLOAT, pixel(result.reference_image_mask))

    def test_packed_masks_none_renders_solid_backgrounds(self) -> None:
        result = render_scail2_colored_mask_pair(
            {
                "packed_masks": None,
                "orig_size": (2, 2),
                "n_frames": 2,
            },
            object_indices="",
            sort_by="none",
            replacement_mode=False,
        )

        self.assertEqual(BLACK_RGB_FLOAT, pixel(result.pose_video_mask, frame=0, row=0, col=0))
        self.assertEqual(BLACK_RGB_FLOAT, pixel(result.pose_video_mask, frame=1, row=1, col=1))
        self.assertEqual(WHITE_RGB_FLOAT, pixel(result.reference_image_mask, frame=0, row=1, col=1))

    def test_missing_reference_inputs_render_solid_reference_mask(self) -> None:
        result = render_scail2_colored_mask_pair(
            track_data([[[[True, False]]]]),
            object_indices="",
            sort_by="none",
            replacement_mode=False,
        )

        self.assertEqual(BLUE_RGB_FLOAT, pixel(result.pose_video_mask, col=0))
        self.assertEqual(BLACK_RGB_FLOAT, pixel(result.pose_video_mask, col=1))
        self.assertEqual(WHITE_RGB_FLOAT, pixel(result.reference_image_mask, col=0))
        self.assertEqual(WHITE_RGB_FLOAT, pixel(result.reference_image_mask, col=1))

    def test_rejects_simultaneous_reference_inputs(self) -> None:
        driving = track_data([[[[True]]]])

        with self.assertRaisesRegex(ValueError, "either ref_track_data or ref_mask"):
            render_scail2_colored_mask_pair(
                driving,
                ref_track_data=driving,
                ref_mask=[[True]],
                object_indices="",
                sort_by="none",
                replacement_mode=False,
            )

    def test_track_shape_errors_include_source_and_shape_context(self) -> None:
        track = {
            "masks": [[[[True], [False]]]],
            "orig_size": (1, 1),
            "n_frames": 1,
        }

        with self.assertRaisesRegex(ValueError, "source=masks") as error_context:
            render_scail2_colored_mask_pair(
                track,
                object_indices="",
                sort_by="none",
                replacement_mode=False,
            )

        message = str(error_context.exception)
        self.assertIn("orig_size=(1, 1)", message)
        self.assertIn("actual_shape=(2, 1)", message)
        self.assertIn("frame=0", message)
        self.assertIn("object=0", message)

    def test_colored_mask_node_reference_inputs_have_tooltips(self) -> None:
        package = import_root_package()
        node_class = package.NODE_CLASS_MAPPINGS["SCAILPose2ColoredMask"]

        input_types = node_class.INPUT_TYPES()
        ref_track_config = input_types["optional"]["ref_track_data"][1]
        ref_mask_config = input_types["optional"]["ref_mask"][1]

        self.assertIn("SAM3 track", ref_track_config["tooltip"])
        self.assertIn("plain MASK", ref_mask_config["tooltip"])

    def test_shared_left_to_right_sort_keeps_reference_and_driving_colors(self) -> None:
        driving = track_data(
            [
                [
                    [[False, False, True]],
                    [[True, False, False]],
                ]
            ]
        )
        reference = track_data(
            [
                [
                    [[False, False, True]],
                    [[True, False, False]],
                ]
            ]
        )

        result = render_scail2_colored_mask_pair(
            driving,
            ref_track_data=reference,
            object_indices="",
            sort_by="left_to_right",
            replacement_mode=False,
        )

        self.assertEqual((1, 0), result.object_order)
        self.assertEqual(BLUE_RGB_FLOAT, pixel(result.pose_video_mask, col=0))
        self.assertEqual(BLACK_RGB_FLOAT, pixel(result.pose_video_mask, col=1))
        self.assertEqual(RED_RGB_FLOAT, pixel(result.pose_video_mask, col=2))
        self.assertEqual(BLUE_RGB_FLOAT, pixel(result.reference_image_mask, col=0))
        self.assertEqual(WHITE_RGB_FLOAT, pixel(result.reference_image_mask, col=1))
        self.assertEqual(RED_RGB_FLOAT, pixel(result.reference_image_mask, col=2))

    def test_area_sort_filter_and_replacement_backgrounds(self) -> None:
        driving = track_data(
            [
                [
                    [[True, False, False]],
                    [[False, True, True]],
                ]
            ]
        )

        result = render_scail2_colored_mask_pair(
            driving,
            object_indices="0",
            sort_by="area",
            replacement_mode=True,
        )

        self.assertEqual((1,), result.object_order)
        self.assertEqual(WHITE_RGB_FLOAT, pixel(result.pose_video_mask, col=0))
        self.assertEqual(BLUE_RGB_FLOAT, pixel(result.pose_video_mask, col=1))
        self.assertEqual(BLUE_RGB_FLOAT, pixel(result.pose_video_mask, col=2))
        self.assertEqual(BLACK_RGB_FLOAT, pixel(result.reference_image_mask, col=0))

    def test_plain_reference_mask_uses_first_identity_color(self) -> None:
        driving = track_data([[[[True, False]]]])

        result = render_scail2_colored_mask_pair(
            driving,
            ref_mask=[[True, False]],
            object_indices="",
            sort_by="none",
            replacement_mode=False,
        )

        self.assertEqual(BLUE_RGB_FLOAT, pixel(result.pose_video_mask, col=0))
        self.assertEqual(BLUE_RGB_FLOAT, pixel(result.reference_image_mask, col=0))
        self.assertEqual(WHITE_RGB_FLOAT, pixel(result.reference_image_mask, col=1))

    def test_root_package_registers_colored_mask_node_without_sam3_import(self) -> None:
        package = import_root_package()

        self.assertIn("SCAILPose2ColoredMask", package.NODE_CLASS_MAPPINGS)
        self.assertIn("SCAILPose2ColoredMask", package.NODE_DISPLAY_NAME_MAPPINGS)
        self.assertFalse(any(name.startswith("ultralytics") for name in sys.modules))

        node = package.NODE_CLASS_MAPPINGS["SCAILPose2ColoredMask"]()
        self.assertEqual(("IMAGE", "IMAGE"), node.RETURN_TYPES)
        self.assertEqual(("pose_video_mask", "reference_image_mask"), node.RETURN_NAMES)
        pose_mask, reference_mask = node.build(
            track_data([[[[True]]]]),
            object_indices="",
            sort_by="none",
            replacement_mode=False,
        )

        self.assertEqual(BLUE_RGB_FLOAT, pixel(pose_mask))
        self.assertEqual(WHITE_RGB_FLOAT, pixel(reference_mask))


if __name__ == "__main__":
    unittest.main()
