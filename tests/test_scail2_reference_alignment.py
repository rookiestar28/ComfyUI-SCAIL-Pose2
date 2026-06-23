from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from scail2.geometry import frame_bboxes
from scail2.reference_alignment import (
    SCAIL_POSE2_REFERENCE_GEOMETRY_ALIGNED_ATTR,
    align_reference_image_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "ComfyUI_SCAIL_Pose2_ReferenceAlignmentTestPackage"


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


class Scail2ReferenceAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch unavailable: {exc}")
        self.torch = torch

    def _reference_inputs(self):
        torch = self.torch
        ref_image = torch.zeros((1, 10, 6, 3), dtype=torch.float32)
        ref_mask = torch.ones((1, 10, 6, 3), dtype=torch.float32)
        pose_video_mask = torch.zeros((3, 8, 8, 3), dtype=torch.float32)

        ref_image[0, 1:9, 2:4, 0] = 0.8
        ref_image[0, 1:9, 2:4, 1] = 0.2
        ref_mask[0, 1:9, 2:4, :] = self.torch.tensor([0.0, 0.0, 1.0])
        pose_video_mask[0, 2:8, 3:5, 2] = 1.0
        pose_video_mask[1, 1:7, 4:6, 2] = 1.0
        pose_video_mask[2, 2:8, 3:5, 2] = 1.0
        return ref_image, ref_mask, pose_video_mask

    def test_alignment_places_reference_mask_on_target_bbox_without_canvas_size_widgets(self) -> None:
        ref_image, ref_mask, pose_video_mask = self._reference_inputs()

        result = align_reference_image_geometry(
            ref_image=ref_image,
            ref_mask=ref_mask,
            pose_video_mask=pose_video_mask,
            fit_mode="contain",
            anchor="bottom_center",
            target_frame_policy="median_bbox",
        )

        self.assertEqual((1, 8, 8, 3), tuple(result.ref_image.shape))
        self.assertEqual((1, 8, 8, 3), tuple(result.ref_mask.shape))
        self.assertTrue(
            getattr(result.ref_image, SCAIL_POSE2_REFERENCE_GEOMETRY_ALIGNED_ATTR)
        )
        self.assertEqual(
            (3.0, 2.0, 5.0, 8.0),
            frame_bboxes(result.ref_mask, kind="semantic_rgb_mask")[0].to_tuple(),
        )
        self.assertIn("reference_geometry_alignment", result.summary)
        self.assertIn("target_size=8x8", result.summary)

    def test_alignment_rejects_empty_reference_mask(self) -> None:
        torch = self.torch
        ref_image = torch.zeros((1, 10, 6, 3), dtype=torch.float32)
        ref_mask = torch.ones((1, 10, 6, 3), dtype=torch.float32)
        pose_video_mask = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        pose_video_mask[0, 2:8, 3:5, 2] = 1.0

        with self.assertRaisesRegex(ValueError, "ref_mask foreground area ratio"):
            align_reference_image_geometry(
                ref_image=ref_image,
                ref_mask=ref_mask,
                pose_video_mask=pose_video_mask,
            )

    def test_root_package_registers_reference_alignment_node(self) -> None:
        package = import_root_package()

        self.assertIn("SCAILPose2ReferenceImageGeometryAlign", package.NODE_CLASS_MAPPINGS)
        node_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2ReferenceImageGeometryAlign"]
        self.assertEqual(("IMAGE", "IMAGE", "STRING"), node_cls.RETURN_TYPES)
        self.assertEqual(("ref_image", "ref_mask", "summary"), node_cls.RETURN_NAMES)
        required = node_cls.INPUT_TYPES()["required"]
        self.assertEqual(
            (
                "ref_image",
                "ref_mask",
                "pose_video_mask",
                "fit_mode",
                "anchor",
                "target_frame_policy",
                "bbox_margin",
                "max_scale",
                "min_mask_area_ratio",
            ),
            tuple(required),
        )
        self.assertNotIn("width", required)
        self.assertNotIn("height", required)
        self.assertEqual("contain", required["fit_mode"][1]["default"])
        self.assertEqual("bottom_center", required["anchor"][1]["default"])
        self.assertEqual("median_bbox", required["target_frame_policy"][1]["default"])

    def test_reference_alignment_node_executes_and_marks_condition_source_kind(self) -> None:
        package = import_root_package()
        align_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2ReferenceImageGeometryAlign"]
        condition_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2SCAIL2Condition"]
        ref_image, ref_mask, pose_video_mask = self._reference_inputs()

        aligned_ref, aligned_mask, summary = align_cls().align(
            ref_image,
            ref_mask,
            pose_video_mask,
        )
        condition, = condition_cls().build(
            pose_video_mask=pose_video_mask[:1],
            ref_image=aligned_ref,
            ref_mask=aligned_mask,
            mode="replacement",
            width=8,
            height=8,
            num_frames=1,
            driving_video=self.torch.zeros((1, 8, 8, 3), dtype=self.torch.float32),
        )

        self.assertIn("reference_geometry_alignment", summary)
        self.assertEqual("replacement", condition.mode)
        self.assertTrue(condition.source_kind.endswith(":reference_geometry_aligned"))


if __name__ == "__main__":
    unittest.main()
