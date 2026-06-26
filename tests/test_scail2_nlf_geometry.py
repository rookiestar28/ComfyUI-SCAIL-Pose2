from __future__ import annotations

import unittest

from scail2.geometry import frame_bboxes
from scail2.nlf_geometry import (
    align_pose_video_to_bboxes,
    bbox_payload_is_safe_for_render_repair,
    format_nlf_render_bbox_diagnostics,
    format_nlf_source_canvas_diagnostics,
    normalize_nlf_bboxes,
    pose_mask_alignment_is_safe_for_render_repair,
    select_nlf_bboxes_for_identity,
    validate_ref_dwpose_camera_solve,
)
from scail2.pose_alignment import align_pose_video_to_mask


BLACK = (0.0, 0.0, 0.0)
BLUE = (0.0, 0.0, 1.0)


def image_frame(width: int, height: int, fill=BLACK):
    return [[fill for _x in range(width)] for _y in range(height)]


def paint_rect(frame, *, x0: int, y0: int, x1: int, y1: int, color) -> None:
    for y in range(y0, y1):
        for x in range(x0, x1):
            frame[y][x] = color


class Scail2NLFGeometryTests(unittest.TestCase):
    def test_normalizes_core_xyxy_bbox_list(self) -> None:
        normalized = normalize_nlf_bboxes(
            [[1, 2, 5, 8], [0, 0, 0, 0]],
            frame_count=2,
        )

        self.assertEqual(2, normalized.frame_count)
        self.assertEqual(1, normalized.valid_count)
        self.assertEqual((1.0, 2.0, 5.0, 8.0), normalized.boxes[0].to_tuple())
        self.assertIsNone(normalized.boxes[1])
        self.assertFalse(normalized.ambiguous)

    def test_normalizes_tensor_bbox_payload_when_torch_is_available(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch unavailable: {exc}")

        payload = torch.tensor([[1, 1, 3, 4], [2, 2, 6, 7]], dtype=torch.float32)

        normalized = normalize_nlf_bboxes(payload, frame_count=2)

        self.assertEqual(2, normalized.valid_count)
        self.assertEqual((2.0, 2.0, 6.0, 7.0), normalized.boxes[1].to_tuple())

    def test_marks_multi_person_bbox_payload_as_ambiguous(self) -> None:
        normalized = normalize_nlf_bboxes(
            [
                [[1, 1, 3, 3], [2, 2, 8, 8]],
                [[0, 0, 2, 2]],
            ],
            frame_count=2,
        )

        self.assertTrue(normalized.ambiguous)
        self.assertEqual(2, normalized.max_person_count)
        self.assertTrue(normalized.is_multi_person)
        self.assertEqual((1.0, 1.0, 3.0, 3.0), normalized.candidates[0][0].to_tuple())
        self.assertEqual((2.0, 2.0, 8.0, 8.0), normalized.candidates[0][1].to_tuple())
        self.assertIn("multi_person_bbox_payload", normalized.warnings)
        safe, reason = bbox_payload_is_safe_for_render_repair(
            normalized,
            width=8,
            height=8,
        )
        self.assertFalse(safe)
        self.assertEqual("ambiguous_multi_person_bboxes", reason)

    def test_selects_identity_bbox_stream_from_multi_person_payload(self) -> None:
        normalized = normalize_nlf_bboxes(
            [
                [[1, 1, 3, 3], [4, 4, 7, 7]],
                [[2, 2, 4, 4]],
            ],
            frame_count=2,
        )

        selected = select_nlf_bboxes_for_identity(normalized, identity_index=1)

        self.assertFalse(selected.ambiguous)
        self.assertEqual("list.identity_1", selected.source)
        self.assertEqual((4.0, 4.0, 7.0, 7.0), selected.boxes[0].to_tuple())
        self.assertIsNone(selected.boxes[1])
        self.assertIn("identity_bbox_missing_frames", selected.warnings)

    def test_rejects_bbox_coordinate_space_mismatch_for_render_repair(self) -> None:
        normalized = normalize_nlf_bboxes([[10, 10, 30, 30]], frame_count=1)

        safe, reason = bbox_payload_is_safe_for_render_repair(
            normalized,
            width=8,
            height=8,
        )

        self.assertFalse(safe)
        self.assertEqual("bbox_coordinate_space_mismatch", reason)

    def test_aligns_pose_video_to_render_space_bboxes(self) -> None:
        pose = image_frame(8, 8)
        paint_rect(pose, x0=0, y0=0, x1=2, y1=2, color=BLUE)
        normalized = normalize_nlf_bboxes([[4, 4, 8, 8]], frame_count=1)

        result = align_pose_video_to_bboxes(
            pose_video=[pose],
            bboxes=normalized.boxes,
        )

        self.assertEqual(1.0, result.after.mean_iou)
        self.assertEqual(
            (4.0, 4.0, 8.0, 8.0),
            frame_bboxes(result.pose_video, kind="pose_image")[0].to_tuple(),
        )

    def test_formats_render_bbox_diagnostics_for_logs(self) -> None:
        pose = image_frame(8, 8)
        paint_rect(pose, x0=0, y0=0, x1=2, y1=2, color=BLUE)
        normalized = normalize_nlf_bboxes([[4, 4, 8, 8]], frame_count=1)

        summary = format_nlf_render_bbox_diagnostics(
            pose_video=[pose],
            target_bboxes=normalized.boxes,
            target_source="nlf_bboxes",
            width=8,
            height=8,
            fallback_reason="none",
        )

        self.assertIn("target_source=nlf_bboxes", summary)
        self.assertIn("mean_pose_coverage=0.062500", summary)
        self.assertIn("mean_target_coverage=0.250000", summary)
        self.assertIn("mean_center_delta_px=", summary)
        self.assertIn("fallback_reason=none", summary)

    def test_formats_source_canvas_diagnostics_for_matching_inputs(self) -> None:
        mask = image_frame(8, 6)
        normalized = normalize_nlf_bboxes([[1, 1, 5, 5]], frame_count=1)

        summary = format_nlf_source_canvas_diagnostics(
            render_width=8,
            render_height=6,
            output_width=4,
            output_height=3,
            pose_video_mask=[mask],
            normalized_bboxes=normalized,
            bboxes_connected=True,
            dw_pose_input=None,
        )

        self.assertIn("render_size=8x6", summary)
        self.assertIn("output_size=4x3", summary)
        self.assertIn("mask_size=8x6", summary)
        self.assertIn("mask_matches_render=True", summary)
        self.assertIn("bbox_safe=True", summary)
        self.assertIn("bbox_reason=ok", summary)
        self.assertIn("source_canvas_mismatch=False", summary)

    def test_formats_source_canvas_diagnostics_for_mask_size_mismatch(self) -> None:
        mask = image_frame(4, 6)
        normalized = normalize_nlf_bboxes([[10, 10, 20, 20]], frame_count=1)

        summary = format_nlf_source_canvas_diagnostics(
            render_width=8,
            render_height=6,
            output_width=4,
            output_height=3,
            pose_video_mask=[mask],
            normalized_bboxes=normalized,
            bboxes_connected=True,
            dw_pose_input=None,
        )

        self.assertIn("mask_size=4x6", summary)
        self.assertIn("mask_matches_render=False", summary)
        self.assertIn("bbox_safe=False", summary)
        self.assertIn("bbox_reason=bbox_coordinate_space_mismatch", summary)
        self.assertIn("source_canvas_mismatch=True", summary)

    def test_formats_source_canvas_diagnostics_with_dwpose_bounds(self) -> None:
        dw_pose_input = [
            {
                "bodies": {
                    "candidate": [
                        [
                            [0.1, 0.2],
                            [0.9, 0.8],
                        ]
                    ],
                },
                "faces": [[[0.3, 0.4]]],
                "hands": [
                    [[0.5, 0.6]],
                    [[0.7, 0.75]],
                ],
            }
        ]

        summary = format_nlf_source_canvas_diagnostics(
            render_width=8,
            render_height=6,
            output_width=4,
            output_height=3,
            pose_video_mask=None,
            normalized_bboxes=normalize_nlf_bboxes(None, frame_count=1),
            bboxes_connected=False,
            dw_pose_input=dw_pose_input,
        )

        self.assertIn("dwpose_bounds=x=0.100000..0.900000,y=0.200000..0.800000", summary)
        self.assertIn("dwpose_normalized=True", summary)
        self.assertIn("source_canvas_mismatch=False", summary)

    def test_mask_alignment_arbitration_accepts_valid_mask(self) -> None:
        pose = image_frame(8, 8)
        mask = image_frame(8, 8)
        paint_rect(pose, x0=0, y0=0, x1=2, y1=2, color=BLUE)
        paint_rect(mask, x0=4, y0=4, x1=8, y1=8, color=BLUE)
        alignment = align_pose_video_to_mask(pose_video=[pose], pose_video_mask=[mask])
        normalized = normalize_nlf_bboxes([[4, 4, 8, 8]], frame_count=1)

        safe, reason = pose_mask_alignment_is_safe_for_render_repair(
            alignment,
            normalized_bboxes=normalized,
            bboxes_connected=True,
            width=8,
            height=8,
        )

        self.assertTrue(safe)
        self.assertEqual("ok", reason)

    def test_mask_alignment_arbitration_rejects_empty_mask(self) -> None:
        pose = image_frame(8, 8)
        mask = image_frame(8, 8)
        paint_rect(pose, x0=0, y0=0, x1=2, y1=2, color=BLUE)
        alignment = align_pose_video_to_mask(pose_video=[pose], pose_video_mask=[mask])
        normalized = normalize_nlf_bboxes([[4, 4, 8, 8]], frame_count=1)

        safe, reason = pose_mask_alignment_is_safe_for_render_repair(
            alignment,
            normalized_bboxes=normalized,
            bboxes_connected=True,
            width=8,
            height=8,
        )

        self.assertFalse(safe)
        self.assertEqual("mask_empty_mask", reason)

    def test_mask_alignment_arbitration_rejects_bbox_target_conflict(self) -> None:
        pose = image_frame(8, 8)
        mask = image_frame(8, 8)
        paint_rect(pose, x0=0, y0=0, x1=2, y1=2, color=BLUE)
        paint_rect(mask, x0=6, y0=0, x1=8, y1=2, color=BLUE)
        alignment = align_pose_video_to_mask(pose_video=[pose], pose_video_mask=[mask])
        normalized = normalize_nlf_bboxes([[4, 4, 8, 8]], frame_count=1)

        safe, reason = pose_mask_alignment_is_safe_for_render_repair(
            alignment,
            normalized_bboxes=normalized,
            bboxes_connected=True,
            width=8,
            height=8,
        )

        self.assertFalse(safe)
        self.assertEqual("mask_bbox_target_mismatch", reason)

    def test_mask_alignment_arbitration_preserves_unsafe_bbox_guard(self) -> None:
        pose = image_frame(8, 8)
        mask = image_frame(8, 8)
        paint_rect(pose, x0=0, y0=0, x1=2, y1=2, color=BLUE)
        paint_rect(mask, x0=6, y0=0, x1=8, y1=2, color=BLUE)
        alignment = align_pose_video_to_mask(pose_video=[pose], pose_video_mask=[mask])
        normalized = normalize_nlf_bboxes(
            [[[6, 0, 8, 2], [4, 4, 8, 8]]],
            frame_count=1,
        )

        safe, reason = pose_mask_alignment_is_safe_for_render_repair(
            alignment,
            normalized_bboxes=normalized,
            bboxes_connected=True,
            width=8,
            height=8,
        )

        self.assertTrue(safe)
        self.assertEqual("ok", reason)

    def test_camera_solve_validation_accepts_safe_projection(self) -> None:
        points_3d = [
            [0.0, 0.0, 10.0],
            [0.1, 0.0, 10.0],
            [0.0, 0.1, 10.0],
            [0.1, 0.1, 10.0],
        ]
        target_2d = [
            [4.0, 4.0],
            [5.0, 4.0],
            [4.0, 5.0],
            [5.0, 5.0],
        ]

        result = validate_ref_dwpose_camera_solve(
            camera_intrinsics=[
                [100.0, 0.0, 4.0],
                [0.0, 100.0, 4.0],
                [0.0, 0.0, 1.0],
            ],
            scale_m=1.0,
            scale_s=1.0,
            points_3d=points_3d,
            target_points_2d=target_2d,
            width=8,
            height=8,
            solve_mode="central",
        )

        self.assertTrue(result.safe)
        self.assertEqual("ok", result.reason)
        self.assertIn("solve_mode=central", result.summary)
        self.assertIn("valid_points=4", result.summary)
        self.assertIn("mean_reprojection_error_px=0.000000", result.summary)

    def test_camera_solve_validation_rejects_extreme_principal_shift(self) -> None:
        result = validate_ref_dwpose_camera_solve(
            camera_intrinsics=[
                [100.0, 0.0, 20.0],
                [0.0, 100.0, 4.0],
                [0.0, 0.0, 1.0],
            ],
            scale_m=1.0,
            scale_s=1.0,
            points_3d=[[0.0, 0.0, 10.0], [0.1, 0.0, 10.0], [0.0, 0.1, 10.0], [0.1, 0.1, 10.0]],
            target_points_2d=[[4.0, 4.0], [5.0, 4.0], [4.0, 5.0], [5.0, 5.0]],
            width=8,
            height=8,
            solve_mode="central",
        )

        self.assertFalse(result.safe)
        self.assertEqual("extreme_principal_shift", result.reason)

    def test_camera_solve_validation_rejects_high_residual(self) -> None:
        result = validate_ref_dwpose_camera_solve(
            camera_intrinsics=[
                [100.0, 0.0, 4.0],
                [0.0, 100.0, 4.0],
                [0.0, 0.0, 1.0],
            ],
            scale_m=1.0,
            scale_s=1.0,
            points_3d=[[0.0, 0.0, 10.0], [0.1, 0.0, 10.0], [0.0, 0.1, 10.0], [0.1, 0.1, 10.0]],
            target_points_2d=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            width=8,
            height=8,
            solve_mode="central",
        )

        self.assertFalse(result.safe)
        self.assertEqual("high_reprojection_error", result.reason)

    def test_camera_solve_validation_rejects_off_canvas_projection(self) -> None:
        points_3d = [
            [0.0, 0.0, 10.0],
            [0.1, 0.0, 10.0],
            [0.0, 0.1, 10.0],
            [0.1, 0.1, 10.0],
        ]
        target_2d = [
            [4.0, 4.0],
            [14.0, 4.0],
            [4.0, 14.0],
            [14.0, 14.0],
        ]

        result = validate_ref_dwpose_camera_solve(
            camera_intrinsics=[
                [1000.0, 0.0, 4.0],
                [0.0, 1000.0, 4.0],
                [0.0, 0.0, 1.0],
            ],
            scale_m=1.0,
            scale_s=1.0,
            points_3d=points_3d,
            target_points_2d=target_2d,
            width=8,
            height=8,
            solve_mode="central",
        )

        self.assertFalse(result.safe)
        self.assertEqual("off_canvas_projection", result.reason)


    def test_camera_solve_validation_rejects_non_finite_intrinsics(self) -> None:
        result = validate_ref_dwpose_camera_solve(
            camera_intrinsics=[
                [float("nan"), 0.0, 4.0],
                [0.0, 100.0, 4.0],
                [0.0, 0.0, 1.0],
            ],
            scale_m=1.0,
            scale_s=1.0,
            points_3d=[[0.0, 0.0, 10.0], [0.1, 0.0, 10.0], [0.0, 0.1, 10.0], [0.1, 0.1, 10.0]],
            target_points_2d=[[4.0, 4.0], [5.0, 4.0], [4.0, 5.0], [5.0, 5.0]],
            width=8,
            height=8,
            solve_mode="central",
        )

        self.assertFalse(result.safe)
        self.assertEqual("non_finite_intrinsics", result.reason)


if __name__ == "__main__":
    unittest.main()
