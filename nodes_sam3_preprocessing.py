"""ComfyUI node wrappers for optional SAM3 preprocessing paths."""

from __future__ import annotations

from .scail2.colored_masks import (
    materialize_comfy_image,
    render_scail2_colored_mask_pair,
)
from .scail2.observability import (
    elapsed_ms,
    get_logger,
    make_progress,
    perf_counter_ms,
    safe_value_summary,
)
from .scail2.sam3_preprocessing import require_sam3_predictors

LOGGER = get_logger(__name__)


class SCAIL2SAM3DependencyCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "check"
    CATEGORY = "SCAIL-Pose2/SAM3"
    DESCRIPTION = "Check whether optional SAM3 preprocessing dependencies are installed."

    def check(self):
        require_sam3_predictors()
        return ("SAM3 preprocessing dependencies are available.",)


class SCAILPose2ColoredMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "driving_track_data": (
                    "SAM3_TRACK_DATA",
                    {"tooltip": "SAM3 track of the driving pose video."},
                ),
                "object_indices": ("STRING", {"default": ""}),
                "sort_by": (["left_to_right", "area", "none"], {"default": "left_to_right"}),
                "replacement_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "ref_track_data": (
                    "SAM3_TRACK_DATA",
                    {
                        "tooltip": (
                            "Optional SAM3 track for reference identities. "
                            "Do not connect when using ref_mask."
                        )
                    },
                ),
                "ref_mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Optional plain MASK for the reference subject. "
                            "Do not connect when using ref_track_data."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("pose_video_mask", "reference_image_mask")
    FUNCTION = "build"
    CATEGORY = "SCAIL-Pose2/SAM3"
    DESCRIPTION = "Render SAM3 tracks into SCAIL-2 colored masks with shared identity sorting."

    def build(
        self,
        driving_track_data,
        object_indices="",
        sort_by="left_to_right",
        replacement_mode=False,
        ref_track_data=None,
        ref_mask=None,
    ):
        progress = make_progress(3)
        started_ms = perf_counter_ms()
        LOGGER.info(
            "SCAIL-Pose2 Colored Mask start: driving=%s ref_track=%s ref_mask=%s",
            safe_value_summary(driving_track_data.get("masks"))
            if isinstance(driving_track_data, dict)
            else safe_value_summary(driving_track_data),
            safe_value_summary(ref_track_data.get("masks"))
            if isinstance(ref_track_data, dict)
            else safe_value_summary(ref_track_data),
            safe_value_summary(ref_mask),
        )
        result = render_scail2_colored_mask_pair(
            driving_track_data,
            ref_track_data=ref_track_data,
            ref_mask=ref_mask,
            object_indices=object_indices,
            sort_by=sort_by,
            replacement_mode=bool(replacement_mode),
        )
        progress.update()
        pose_mask = materialize_comfy_image(result.pose_video_mask)
        progress.update()
        reference_mask = materialize_comfy_image(result.reference_image_mask)
        progress.update()
        LOGGER.info(
            "SCAIL-Pose2 Colored Mask done: pose=%s reference=%s objects=%s elapsed_ms=%.2f",
            safe_value_summary(pose_mask),
            safe_value_summary(reference_mask),
            len(result.object_order),
            elapsed_ms(started_ms),
        )
        return (
            pose_mask,
            reference_mask,
        )


NODE_CLASS_MAPPINGS = {
    "SCAIL2SAM3DependencyCheck": SCAIL2SAM3DependencyCheck,
    "SCAILPose2ColoredMask": SCAILPose2ColoredMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAIL2SAM3DependencyCheck": "SCAIL-2 SAM3 Dependency Check",
    "SCAILPose2ColoredMask": "SCAIL-Pose2 Colored Mask",
}
