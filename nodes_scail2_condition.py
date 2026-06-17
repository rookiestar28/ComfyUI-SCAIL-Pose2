"""ComfyUI node wrappers for SCAIL-2 condition payloads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .scail2.preprocessing import build_user_mask_condition


def _as_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


def _looks_like_hwc_image(value: Any) -> bool:
    try:
        first_channel = value[0][0][0]
    except (IndexError, TypeError):
        return False
    return _is_number(first_channel)


def _normalize_image_frames(value: Any, *, name: str) -> Sequence[Any]:
    raw = _as_list(value)
    if not raw:
        raise ValueError(f"{name} must not be empty")
    if _looks_like_hwc_image(raw):
        return (raw,)
    return raw


def _split_additional_images(value: Any | None) -> Sequence[Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return (value,)
    raw = _as_list(value)
    if not isinstance(raw, Sequence):
        return (raw,)
    return tuple(raw)


def _split_additional_masks(value: Any | None) -> Sequence[Sequence[Any]] | None:
    if value is None:
        return None
    frames = _normalize_image_frames(value, name="additional_ref_masks")
    return tuple((frame,) for frame in frames)


class SCAILPose2SCAIL2Condition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "ref_mask": ("IMAGE",),
                "pose_video": ("IMAGE",),
                "driving_mask": ("IMAGE",),
                "mode": (["animation", "replacement", "pose_driven"], {"default": "animation"}),
                "width": ("INT", {"default": 512, "min": 1, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "step": 1}),
                "num_frames": ("INT", {"default": 81, "min": 1, "step": 1}),
                "segment_len": ("INT", {"default": 81, "min": 1, "step": 1}),
                "segment_overlap": ("INT", {"default": 5, "min": 0, "step": 1}),
                "previous_frame_count": ("INT", {"default": 0, "min": 0, "step": 1}),
                "video_frame_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
            "optional": {
                "additional_ref_images": ("IMAGE",),
                "additional_ref_masks": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("SCAIL2_CONDITION",)
    RETURN_NAMES = ("condition",)
    FUNCTION = "build"
    CATEGORY = "SCAIL-Pose2/SCAIL-2"
    DESCRIPTION = "Build a validated SCAIL-2 condition payload from RGB semantic masks."

    def build(
        self,
        ref_image,
        ref_mask,
        pose_video,
        driving_mask,
        mode,
        width,
        height,
        num_frames,
        segment_len,
        segment_overlap,
        previous_frame_count=0,
        video_frame_offset=0,
        additional_ref_images=None,
        additional_ref_masks=None,
    ):
        condition = build_user_mask_condition(
            mode=mode,
            ref_image=ref_image,
            ref_mask_frames=_normalize_image_frames(ref_mask, name="ref_mask"),
            pose_video=pose_video,
            pose_frame_count=num_frames,
            driving_mask_frames=_normalize_image_frames(
                driving_mask,
                name="driving_mask",
            ),
            width=width,
            height=height,
            segment_len=segment_len,
            segment_overlap=segment_overlap,
            additional_ref_images=_split_additional_images(additional_ref_images),
            additional_ref_masks=_split_additional_masks(additional_ref_masks),
            source_kind="comfy_node:SCAILPose2SCAIL2Condition",
            previous_frame_count=previous_frame_count,
            video_frame_offset=video_frame_offset,
        )
        return (condition,)


NODE_CLASS_MAPPINGS = {
    "SCAILPose2SCAIL2Condition": SCAILPose2SCAIL2Condition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILPose2SCAIL2Condition": "SCAIL-Pose2 SCAIL-2 Condition",
}
