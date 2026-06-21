from __future__ import annotations

import unittest

from scail2.geometry import (
    BoundingBox,
    bbox_iou,
    diagnose_pose_mask_geometry,
    frame_bboxes,
    frame_size,
    replacement_geometry_issues,
)


BLACK = (0.0, 0.0, 0.0)
WHITE = (1.0, 1.0, 1.0)
BLUE = (0.0, 0.0, 1.0)
RED = (1.0, 0.0, 0.0)


def image_frame(width: int, height: int, fill=BLACK):
    return [[fill for _x in range(width)] for _y in range(height)]


def paint_rect(frame, *, x0: int, y0: int, x1: int, y1: int, color) -> None:
    for y in range(y0, y1):
        for x in range(x0, x1):
            frame[y][x] = color


class Scail2GeometryTests(unittest.TestCase):
    def test_pose_image_bbox_from_nested_list(self) -> None:
        frame = image_frame(8, 6)
        paint_rect(frame, x0=2, y0=1, x1=6, y1=5, color=BLUE)

        boxes = frame_bboxes([frame], kind="pose_image")

        self.assertEqual((2.0, 1.0, 6.0, 5.0), boxes[0].to_tuple())
        self.assertEqual((6, 8), frame_size([frame], kind="pose_image"))

    def test_semantic_rgb_mask_ignores_black_and_white_backgrounds(self) -> None:
        black_bg = image_frame(5, 4, fill=BLACK)
        white_bg = image_frame(5, 4, fill=WHITE)
        paint_rect(black_bg, x0=1, y0=1, x1=3, y1=3, color=BLUE)
        paint_rect(white_bg, x0=2, y0=0, x1=5, y1=2, color=RED)

        boxes = frame_bboxes([black_bg, white_bg], kind="semantic_rgb_mask")

        self.assertEqual((1.0, 1.0, 3.0, 3.0), boxes[0].to_tuple())
        self.assertEqual((2.0, 0.0, 5.0, 2.0), boxes[1].to_tuple())

    def test_empty_foreground_returns_none(self) -> None:
        boxes = frame_bboxes([image_frame(3, 2)], kind="pose_image")

        self.assertIsNone(boxes[0])

    def test_mask_tensor_bbox(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch unavailable: {exc}")

        mask = torch.zeros((2, 5, 6), dtype=torch.float32)
        mask[0, 1:4, 2:5] = 1.0
        mask[1, 0:2, 0:2] = 1.0

        boxes = frame_bboxes(mask, kind="mask")

        self.assertEqual((2.0, 1.0, 5.0, 4.0), boxes[0].to_tuple())
        self.assertEqual((0.0, 0.0, 2.0, 2.0), boxes[1].to_tuple())
        self.assertEqual((5, 6), frame_size(mask, kind="mask"))

    def test_bbox_iou(self) -> None:
        first = BoundingBox(0.0, 0.0, 4.0, 4.0)
        second = BoundingBox(2.0, 2.0, 6.0, 6.0)

        self.assertAlmostEqual(4.0 / 28.0, bbox_iou(first, second))

    def test_diagnostic_scales_half_resolution_pose_to_target(self) -> None:
        pose = image_frame(4, 4)
        mask = image_frame(8, 8)
        paint_rect(pose, x0=1, y0=1, x1=3, y1=3, color=BLUE)
        paint_rect(mask, x0=2, y0=2, x1=6, y1=6, color=BLUE)

        diagnostic = diagnose_pose_mask_geometry(
            pose_video=[pose],
            pose_video_mask=[mask],
            target_width=8,
            target_height=8,
        )

        self.assertEqual("ok", diagnostic.status)
        self.assertEqual(1, diagnostic.compared_frames)
        self.assertEqual((4, 4), diagnostic.pose_size)
        self.assertEqual((8, 8), diagnostic.mask_size)
        self.assertEqual(1.0, diagnostic.mean_iou)
        self.assertEqual(0.0, diagnostic.mean_center_delta_px)
        self.assertEqual(1.0, diagnostic.mean_width_ratio)
        self.assertEqual(1.0, diagnostic.mean_height_ratio)
        summary = diagnostic.to_summary()
        self.assertEqual("ok", summary["status"])
        self.assertNotIn("pixels", summary)

    def test_diagnostic_reports_missing_pose_foreground(self) -> None:
        pose = image_frame(4, 4)
        mask = image_frame(4, 4)
        paint_rect(mask, x0=1, y0=1, x1=3, y1=3, color=BLUE)

        diagnostic = diagnose_pose_mask_geometry(
            pose_video=[pose],
            pose_video_mask=[mask],
            target_width=4,
            target_height=4,
        )

        self.assertEqual("empty_pose", diagnostic.status)
        self.assertEqual(0, diagnostic.compared_frames)
        self.assertEqual(1, diagnostic.missing_pose_frames)
        self.assertEqual(0, diagnostic.missing_mask_frames)

    def test_diagnostic_exposes_misalignment_metrics_without_policy(self) -> None:
        pose = image_frame(8, 8)
        mask = image_frame(8, 8)
        paint_rect(pose, x0=0, y0=0, x1=2, y1=2, color=BLUE)
        paint_rect(mask, x0=4, y0=4, x1=8, y1=8, color=BLUE)

        diagnostic = diagnose_pose_mask_geometry(
            pose_video=[pose],
            pose_video_mask=[mask],
            target_width=8,
            target_height=8,
        )

        self.assertEqual("ok", diagnostic.status)
        self.assertEqual(0.0, diagnostic.mean_iou)
        self.assertGreater(diagnostic.mean_center_delta_px, 4.0)
        self.assertEqual(0.5, diagnostic.mean_width_ratio)
        self.assertEqual(0.5, diagnostic.mean_height_ratio)

    def test_replacement_geometry_issues_identify_worst_frame(self) -> None:
        aligned_pose = image_frame(8, 8)
        aligned_mask = image_frame(8, 8)
        drift_pose = image_frame(8, 8)
        drift_mask = image_frame(8, 8)
        paint_rect(aligned_pose, x0=2, y0=2, x1=6, y1=6, color=BLUE)
        paint_rect(aligned_mask, x0=2, y0=2, x1=6, y1=6, color=BLUE)
        paint_rect(drift_pose, x0=1, y0=2, x1=5, y1=6, color=BLUE)
        paint_rect(drift_mask, x0=3, y0=2, x1=7, y1=6, color=BLUE)

        diagnostic = diagnose_pose_mask_geometry(
            pose_video=[aligned_pose, drift_pose],
            pose_video_mask=[aligned_mask, drift_mask],
            target_width=8,
            target_height=8,
        )
        issues = replacement_geometry_issues(
            diagnostic,
            target_width=8,
            target_height=8,
            min_iou=0.50,
            max_center_delta_ratio=0.15,
            min_size_ratio=0.50,
            max_size_ratio=2.0,
        )

        self.assertEqual(1, diagnostic.worst_iou_frame_index)
        self.assertEqual(1, diagnostic.worst_center_delta_frame_index)
        self.assertEqual(["min_iou", "center_delta_ratio"], [issue.code for issue in issues])
        self.assertEqual(1, issues[0].frame_index)
        self.assertIn("frame=1", issues[0].format())


if __name__ == "__main__":
    unittest.main()
