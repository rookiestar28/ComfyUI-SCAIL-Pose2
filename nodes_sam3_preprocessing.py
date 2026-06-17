"""ComfyUI node wrappers for optional SAM3 preprocessing paths."""

from __future__ import annotations

from .scail2.colored_masks import (
    materialize_comfy_image,
    render_scail2_colored_mask_pair,
)
from .scail2.sam3_preprocessing import require_sam3_predictors


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
                "driving_track_data": ("SAM3_TRACK_DATA",),
                "object_indices": ("STRING", {"default": ""}),
                "sort_by": (["left_to_right", "area", "none"], {"default": "left_to_right"}),
                "replacement_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "ref_track_data": ("SAM3_TRACK_DATA",),
                "ref_mask": ("MASK",),
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
        result = render_scail2_colored_mask_pair(
            driving_track_data,
            ref_track_data=ref_track_data,
            ref_mask=ref_mask,
            object_indices=object_indices,
            sort_by=sort_by,
            replacement_mode=bool(replacement_mode),
        )
        return (
            materialize_comfy_image(result.pose_video_mask),
            materialize_comfy_image(result.reference_image_mask),
        )


NODE_CLASS_MAPPINGS = {
    "SCAIL2SAM3DependencyCheck": SCAIL2SAM3DependencyCheck,
    "SCAILPose2ColoredMask": SCAILPose2ColoredMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAIL2SAM3DependencyCheck": "SCAIL-2 SAM3 Dependency Check",
    "SCAILPose2ColoredMask": "SCAIL-Pose2 Colored Mask",
}
