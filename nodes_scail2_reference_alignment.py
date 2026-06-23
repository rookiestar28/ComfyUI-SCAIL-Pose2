"""ComfyUI node wrapper for replacement reference geometry alignment."""

from __future__ import annotations

from .scail2.observability import (
    elapsed_ms,
    get_logger,
    make_progress,
    perf_counter_ms,
    safe_value_summary,
)
from .scail2.reference_alignment import align_reference_image_geometry

LOGGER = get_logger(__name__)


class SCAILPose2ReferenceImageGeometryAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "ref_mask": ("IMAGE",),
                "pose_video_mask": ("IMAGE",),
                "fit_mode": (
                    ["contain", "cover", "fit_height", "fit_width"],
                    {"default": "contain"},
                ),
                "anchor": (
                    ["bottom_center", "center"],
                    {"default": "bottom_center"},
                ),
                "target_frame_policy": (
                    ["median_bbox", "first_valid", "largest"],
                    {"default": "median_bbox"},
                ),
                "bbox_margin": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "max_scale": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "min_mask_area_ratio": (
                    "FLOAT",
                    {"default": 0.0005, "min": 0.0, "max": 1.0, "step": 0.0001},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("ref_image", "ref_mask", "summary")
    FUNCTION = "align"
    CATEGORY = "SCAIL-Pose2/SCAIL-2"
    DESCRIPTION = (
        "Align a replacement reference image and reference mask to the "
        "driving mask canvas before building a SCAIL-2 condition."
    )

    def align(
        self,
        ref_image,
        ref_mask,
        pose_video_mask,
        fit_mode="contain",
        anchor="bottom_center",
        target_frame_policy="median_bbox",
        bbox_margin=0,
        max_scale=2.0,
        min_mask_area_ratio=0.0005,
    ):
        progress = make_progress(2)
        started_ms = perf_counter_ms()
        LOGGER.info(
            "SCAIL-Pose2 Reference Image Geometry Align start: ref_image=%s ref_mask=%s pose_video_mask=%s fit_mode=%s anchor=%s target_frame_policy=%s",
            safe_value_summary(ref_image),
            safe_value_summary(ref_mask),
            safe_value_summary(pose_video_mask),
            str(fit_mode),
            str(anchor),
            str(target_frame_policy),
        )
        progress.update()
        result = align_reference_image_geometry(
            ref_image=ref_image,
            ref_mask=ref_mask,
            pose_video_mask=pose_video_mask,
            fit_mode=fit_mode,
            anchor=anchor,
            target_frame_policy=target_frame_policy,
            bbox_margin=bbox_margin,
            max_scale=max_scale,
            min_mask_area_ratio=min_mask_area_ratio,
        )
        progress.update()
        LOGGER.info(
            "SCAIL-Pose2 Reference Image Geometry Align done: %s elapsed_ms=%.2f",
            result.summary,
            elapsed_ms(started_ms),
        )
        return (result.ref_image, result.ref_mask, result.summary)


NODE_CLASS_MAPPINGS = {
    "SCAILPose2ReferenceImageGeometryAlign": SCAILPose2ReferenceImageGeometryAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILPose2ReferenceImageGeometryAlign": "SCAIL-Pose2 Reference Image Geometry Align",
}
