"""ComfyUI node wrapper for the SCAIL-2 WanVideoWrapper adapter contract."""

from __future__ import annotations

from .scail2.observability import (
    elapsed_ms,
    get_logger,
    make_progress,
    perf_counter_ms,
    safe_value_summary,
)
from .scail2.wanvideo_scail2_adapter import build_wanvideo_scail2_adapter_payload

LOGGER = get_logger(__name__)


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

    RETURN_TYPES = ("SCAIL2_WANVIDEO_PAYLOAD",)
    RETURN_NAMES = ("condition",)
    FUNCTION = "build"
    CATEGORY = "SCAIL-Pose2/WanVideoWrapper"
    DESCRIPTION = "Build a versioned SCAIL-2 adapter payload for WanVideoWrapper workflows."

    def build(
        self,
        condition,
        degrade_to_v1=False,
        allow_degradation=False,
    ):
        progress = make_progress(2)
        started_ms = perf_counter_ms()
        LOGGER.info(
            "SCAIL-Pose2 WanVideo SCAIL-2 Adapter start: condition=%s degrade_to_v1=%s allow_degradation=%s",
            safe_value_summary(condition),
            bool(degrade_to_v1),
            bool(allow_degradation),
        )
        payload = build_wanvideo_scail2_adapter_payload(
            condition,
            degrade_to_v1=bool(degrade_to_v1),
            allow_degradation=bool(allow_degradation),
        )
        progress.update()
        LOGGER.info(
            "SCAIL-Pose2 WanVideo SCAIL-2 Adapter done: kind=%s degraded=%s runtime_masks=%s elapsed_ms=%.2f",
            payload.get("kind"),
            bool(payload.get("degraded")),
            sorted(payload.get("runtime_masks", {}).keys()),
            elapsed_ms(started_ms),
        )
        progress.update()
        return (payload,)


NODE_CLASS_MAPPINGS = {
    "SCAILPose2WanVideoSCAIL2Adapter": SCAILPose2WanVideoSCAIL2Adapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILPose2WanVideoSCAIL2Adapter": "SCAIL-Pose2 WanVideo SCAIL-2 Adapter",
}
