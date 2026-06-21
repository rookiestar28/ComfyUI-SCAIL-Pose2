"""ComfyUI node wrapper for SCAIL-Pose2 pose/mask geometry alignment."""

from __future__ import annotations

from .scail2.observability import (
    elapsed_ms,
    get_logger,
    make_progress,
    perf_counter_ms,
    safe_value_summary,
)
from .scail2.pose_alignment import align_pose_video_to_mask

LOGGER = get_logger(__name__)


class SCAILPose2PoseMaskGeometryAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_video": ("IMAGE",),
                "pose_video_mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pose_video", "summary")
    FUNCTION = "align"
    CATEGORY = "SCAIL-Pose2/SCAIL-2"
    DESCRIPTION = (
        "Scale and translate rendered pose foregrounds so their bbox matches the "
        "SAM3-derived SCAIL-2 pose_video_mask bbox."
    )

    def align(self, pose_video, pose_video_mask):
        progress = make_progress(2)
        started_ms = perf_counter_ms()
        LOGGER.info(
            "SCAIL-Pose2 Pose Mask Geometry Align start: pose=%s pose_mask=%s",
            safe_value_summary(pose_video),
            safe_value_summary(pose_video_mask),
        )
        progress.update()
        result = align_pose_video_to_mask(
            pose_video=pose_video,
            pose_video_mask=pose_video_mask,
        )
        progress.update()
        LOGGER.info(
            "SCAIL-Pose2 Pose Mask Geometry Align done: %s elapsed_ms=%.2f",
            result.summary,
            elapsed_ms(started_ms),
        )
        return (result.pose_video, result.summary)


NODE_CLASS_MAPPINGS = {
    "SCAILPose2PoseMaskGeometryAlign": SCAILPose2PoseMaskGeometryAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILPose2PoseMaskGeometryAlign": "SCAIL-Pose2 Pose Mask Geometry Align",
}
