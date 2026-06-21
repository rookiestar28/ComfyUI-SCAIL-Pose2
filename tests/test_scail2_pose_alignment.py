from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from scail2.geometry import diagnose_pose_mask_geometry, frame_bboxes
from scail2.pose_alignment import align_pose_video_to_mask


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "ComfyUI_SCAIL_Pose2_PoseAlignmentTestPackage"

BLACK = (0.0, 0.0, 0.0)
BLUE = (0.0, 0.0, 1.0)


def image_frame(width: int, height: int, fill=BLACK):
    return [[fill for _x in range(width)] for _y in range(height)]


def paint_rect(frame, *, x0: int, y0: int, x1: int, y1: int, color) -> None:
    for y in range(y0, y1):
        for x in range(x0, x1):
            frame[y][x] = color


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


class Scail2PoseAlignmentTests(unittest.TestCase):
    def test_alignment_improves_misaligned_pose_bbox(self) -> None:
        pose = image_frame(8, 8)
        mask = image_frame(8, 8)
        paint_rect(pose, x0=0, y0=0, x1=2, y1=2, color=BLUE)
        paint_rect(mask, x0=4, y0=4, x1=8, y1=8, color=BLUE)

        before = diagnose_pose_mask_geometry(
            pose_video=[pose],
            pose_video_mask=[mask],
            target_width=8,
            target_height=8,
        )
        result = align_pose_video_to_mask(
            pose_video=[pose],
            pose_video_mask=[mask],
        )

        self.assertEqual(0.0, before.mean_iou)
        self.assertEqual(1.0, result.after.mean_iou)
        self.assertGreater(result.after.mean_iou, result.before.mean_iou)
        self.assertIn("pose_mask_alignment", result.summary)
        self.assertEqual((4.0, 4.0, 8.0, 8.0), frame_bboxes(result.pose_video, kind="pose_image")[0].to_tuple())

    def test_alignment_handles_half_resolution_pose(self) -> None:
        pose = image_frame(4, 4)
        mask = image_frame(8, 8)
        paint_rect(pose, x0=0, y0=0, x1=1, y1=1, color=BLUE)
        paint_rect(mask, x0=4, y0=4, x1=8, y1=8, color=BLUE)

        result = align_pose_video_to_mask(
            pose_video=[pose],
            pose_video_mask=[mask],
            target_width=8,
            target_height=8,
        )

        self.assertEqual(1.0, result.after.mean_iou)
        self.assertEqual((2.0, 2.0, 4.0, 4.0), frame_bboxes(result.pose_video, kind="pose_image")[0].to_tuple())

    def test_empty_foreground_frame_is_left_unchanged_and_reported(self) -> None:
        pose = image_frame(4, 4)
        mask = image_frame(4, 4)
        paint_rect(mask, x0=1, y0=1, x1=3, y1=3, color=BLUE)

        result = align_pose_video_to_mask(
            pose_video=[pose],
            pose_video_mask=[mask],
            target_width=4,
            target_height=4,
        )

        self.assertEqual("empty_pose", result.after.status)
        self.assertEqual(tuple(tuple(row) for row in pose), result.pose_video[0])

    def test_root_package_registers_pose_alignment_node(self) -> None:
        package = import_root_package()

        self.assertIn("SCAILPose2PoseMaskGeometryAlign", package.NODE_CLASS_MAPPINGS)
        node_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2PoseMaskGeometryAlign"]
        self.assertEqual(("IMAGE", "STRING"), node_cls.RETURN_TYPES)
        self.assertEqual(("pose_video", "summary"), node_cls.RETURN_NAMES)
        required = node_cls.INPUT_TYPES()["required"]
        self.assertEqual(("pose_video", "pose_video_mask"), tuple(required))

    def test_pose_alignment_node_executes_without_manual_dimensions(self) -> None:
        package = import_root_package()
        node_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2PoseMaskGeometryAlign"]
        node = node_cls()
        pose = image_frame(8, 8)
        mask = image_frame(8, 8)
        paint_rect(pose, x0=0, y0=0, x1=2, y1=2, color=BLUE)
        paint_rect(mask, x0=4, y0=4, x1=8, y1=8, color=BLUE)

        aligned, summary = node.align([pose], [mask])

        self.assertIn("pose_mask_alignment", summary)
        self.assertEqual((4.0, 4.0, 8.0, 8.0), frame_bboxes(aligned, kind="pose_image")[0].to_tuple())

    def test_tensor_alignment_preserves_tensor_contract_when_torch_is_available(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch unavailable: {exc}")

        pose = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        mask = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        pose[0, 0:2, 0:2, 2] = 1.0
        mask[0, 4:8, 4:8, 2] = 1.0

        result = align_pose_video_to_mask(
            pose_video=pose,
            pose_video_mask=mask,
            target_width=8,
            target_height=8,
        )

        self.assertTrue(torch.is_tensor(result.pose_video))
        self.assertEqual(tuple(pose.shape), tuple(result.pose_video.shape))
        self.assertEqual(1.0, result.after.mean_iou)


if __name__ == "__main__":
    unittest.main()
