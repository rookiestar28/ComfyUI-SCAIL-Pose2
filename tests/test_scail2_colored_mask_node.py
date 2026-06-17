from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from scail2.colored_masks import (
    BLACK_RGB_FLOAT,
    BLUE_RGB_FLOAT,
    RED_RGB_FLOAT,
    WHITE_RGB_FLOAT,
    render_scail2_colored_mask_pair,
)


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


class Scail2ColoredMaskNodeTests(unittest.TestCase):
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
