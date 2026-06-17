"""ComfyUI node wrapper for the SCAIL-2 WanVideoWrapper adapter contract."""

from __future__ import annotations

from .scail2.wanvideo_scail2_adapter import (
    build_wanvideo_scail2_adapter_payload,
    summarize_wanvideo_scail2_payload,
)


class SCAILPose2WanVideoSCAIL2Adapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("SCAIL2_CONDITION",),
                "degrade_to_v1": ("BOOLEAN", {"default": False}),
                "allow_degradation": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("SCAIL2_WANVIDEO_PAYLOAD", "STRING")
    RETURN_NAMES = ("adapter_payload", "summary")
    FUNCTION = "build"
    CATEGORY = "SCAIL-Pose2/WanVideoWrapper"
    DESCRIPTION = "Build a versioned SCAIL-2 adapter payload for WanVideoWrapper workflows."

    def build(
        self,
        condition,
        degrade_to_v1=False,
        allow_degradation=False,
    ):
        payload = build_wanvideo_scail2_adapter_payload(
            condition,
            degrade_to_v1=bool(degrade_to_v1),
            allow_degradation=bool(allow_degradation),
        )
        return payload, summarize_wanvideo_scail2_payload(payload)


NODE_CLASS_MAPPINGS = {
    "SCAILPose2WanVideoSCAIL2Adapter": SCAILPose2WanVideoSCAIL2Adapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILPose2WanVideoSCAIL2Adapter": "SCAIL-Pose2 WanVideo SCAIL-2 Adapter",
}
