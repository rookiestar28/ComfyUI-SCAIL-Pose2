"""ComfyUI node wrapper for replacement-mode sampler denoise masks."""

from __future__ import annotations

from .scail2.observability import (
    elapsed_ms,
    get_logger,
    make_progress,
    perf_counter_ms,
    safe_value_summary,
)
from .scail2.replacement_mask import build_replacement_denoise_mask

LOGGER = get_logger(__name__)


class SCAILPose2ReplacementDenoiseMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("SCAIL2_CONDITION",),
                "pose_video_mask": ("IMAGE",),
                "grow_pixels": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1}),
                "blur_pixels": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "strict_replacement_mode": ("BOOLEAN", {"default": True}),
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "summary")
    FUNCTION = "build"
    CATEGORY = "SCAIL-Pose2/SCAIL-2"
    DESCRIPTION = (
        "Build a replacement denoise MASK for WanVideoEncode.mask. "
        "Subject pixels are 1.0 and background pixels are 0.0."
    )

    def build(
        self,
        condition,
        pose_video_mask,
        grow_pixels=8,
        blur_pixels=0,
        strict_replacement_mode=True,
        invert=False,
    ):
        progress = make_progress(2)
        started_ms = perf_counter_ms()
        LOGGER.info(
            "SCAIL-Pose2 Replacement Denoise Mask start: condition=%s pose_video_mask=%s grow=%s blur=%s strict=%s invert=%s",
            safe_value_summary(condition),
            safe_value_summary(pose_video_mask),
            int(grow_pixels),
            int(blur_pixels),
            bool(strict_replacement_mode),
            bool(invert),
        )
        progress.update()
        result = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_video_mask,
            grow_pixels=grow_pixels,
            blur_pixels=blur_pixels,
            strict_replacement_mode=bool(strict_replacement_mode),
            invert=bool(invert),
        )
        progress.update()
        LOGGER.info(
            "SCAIL-Pose2 Replacement Denoise Mask done: %s elapsed_ms=%.2f",
            result.summary,
            elapsed_ms(started_ms),
        )
        return (result.mask, result.summary)


NODE_CLASS_MAPPINGS = {
    "SCAILPose2ReplacementDenoiseMask": SCAILPose2ReplacementDenoiseMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILPose2ReplacementDenoiseMask": "SCAIL-Pose2 Replacement Denoise Mask",
}
