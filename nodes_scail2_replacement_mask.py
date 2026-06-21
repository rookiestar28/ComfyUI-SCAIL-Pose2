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
from .scail2.replacement_presets import MASK_PRESETS

LOGGER = get_logger(__name__)


class SCAILPose2ReplacementDenoiseMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("SCAIL2_CONDITION",),
                "pose_video_mask": ("IMAGE",),
                "mask_preset": (list(MASK_PRESETS), {"default": "custom"}),
                "grow_pixels": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1}),
                "blur_pixels": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "summary")
    FUNCTION = "build"
    CATEGORY = "SCAIL-Pose2/SCAIL-2"
    DESCRIPTION = (
        "Build a subject denoise / background preserve MASK for WanVideoEncode.mask. "
        "Replacement mode uses subject=1.0/background=0.0; non-replacement "
        "modes emit a full-denoise passthrough mask."
    )

    def build(
        self,
        condition,
        pose_video_mask,
        mask_preset="custom",
        grow_pixels=8,
        blur_pixels=0,
    ):
        progress = make_progress(2)
        started_ms = perf_counter_ms()
        LOGGER.info(
            "SCAIL-Pose2 Replacement Denoise Mask start: condition=%s pose_video_mask=%s preset=%s grow=%s blur=%s",
            safe_value_summary(condition),
            safe_value_summary(pose_video_mask),
            str(mask_preset),
            int(grow_pixels),
            int(blur_pixels),
        )
        progress.update()
        result = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_video_mask,
            mask_preset=mask_preset,
            grow_pixels=grow_pixels,
            blur_pixels=blur_pixels,
            strict_replacement_mode=False,
            invert=False,
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
