"""ComfyUI node wrapper for replacement condition-video subject suppression."""

from __future__ import annotations

from .scail2.observability import (
    elapsed_ms,
    get_logger,
    make_progress,
    perf_counter_ms,
    safe_value_summary,
)
from .scail2.replacement_condition_video import (
    MASK_PRESETS,
    SUPPRESSION_MODES,
    build_replacement_condition_video,
)

LOGGER = get_logger(__name__)


class SCAILPose2ReplacementConditionVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "driving_video": ("IMAGE",),
                "pose_video_mask": ("IMAGE",),
                "mask_preset": (
                    list(MASK_PRESETS),
                    {"default": "custom"},
                ),
                "grow_pixels": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1}),
                "blur_pixels": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "suppression_mode": (
                    list(SUPPRESSION_MODES),
                    {"default": "blur_fill"},
                ),
                "suppression_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("driving_video_condition", "summary")
    FUNCTION = "build"
    CATEGORY = "SCAIL-Pose2/SCAIL-2"
    DESCRIPTION = (
        "Suppress original subject pixels in the replacement driving video before "
        "feeding it to SCAIL-Pose2 SCAIL-2 Condition.driving_video."
    )

    def build(
        self,
        driving_video,
        pose_video_mask,
        mask_preset="custom",
        grow_pixels=8,
        blur_pixels=0,
        suppression_mode="blur_fill",
        suppression_strength=1.0,
        noise_seed=0,
    ):
        progress = make_progress(2)
        started_ms = perf_counter_ms()
        LOGGER.info(
            "SCAIL-Pose2 Replacement Condition Video start: driving=%s pose_video_mask=%s preset=%s grow=%s blur=%s mode=%s strength=%.3f",
            safe_value_summary(driving_video),
            safe_value_summary(pose_video_mask),
            str(mask_preset),
            int(grow_pixels),
            int(blur_pixels),
            str(suppression_mode),
            float(suppression_strength),
        )
        progress.update()
        result = build_replacement_condition_video(
            driving_video=driving_video,
            pose_video_mask=pose_video_mask,
            mask_preset=mask_preset,
            grow_pixels=grow_pixels,
            blur_pixels=blur_pixels,
            suppression_mode=suppression_mode,
            suppression_strength=suppression_strength,
            noise_seed=noise_seed,
        )
        progress.update()
        LOGGER.info(
            "SCAIL-Pose2 Replacement Condition Video done: %s elapsed_ms=%.2f",
            result.summary,
            elapsed_ms(started_ms),
        )
        return (result.driving_video_condition, result.summary)


NODE_CLASS_MAPPINGS = {
    "SCAILPose2ReplacementConditionVideo": SCAILPose2ReplacementConditionVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILPose2ReplacementConditionVideo": "SCAIL-Pose2 Replacement Condition Video",
}
