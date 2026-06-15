"""ComfyUI node wrappers for optional SAM3 preprocessing paths."""

from __future__ import annotations

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


NODE_CLASS_MAPPINGS = {
    "SCAIL2SAM3DependencyCheck": SCAIL2SAM3DependencyCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAIL2SAM3DependencyCheck": "SCAIL-2 SAM3 Dependency Check",
}
