"""ComfyUI node wrappers for SCAIL-2 condition payloads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .scail2.preprocessing import build_user_mask_condition
from .scail2.observability import (
    elapsed_ms,
    get_logger,
    make_progress,
    perf_counter_ms,
    safe_value_summary,
)
from .scail2.geometry import frame_bboxes, frame_size
from .scail2.reference_alignment import (
    reference_geometry_is_aligned,
    reference_geometry_summary,
)

LOGGER = get_logger(__name__)


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


def _is_tensor_like_image(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and hasattr(value, "detach") and len(tuple(shape)) in {3, 4}


def _normalize_image_frames(value: Any, *, name: str) -> Sequence[Any]:
    if _is_tensor_like_image(value):
        return value
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


def _split_additional_masks(
    value: Any | None,
    *,
    name: str = "additional_ref_mask",
) -> Sequence[Sequence[Any]] | None:
    if value is None:
        return None
    frames = _normalize_image_frames(value, name=name)
    return tuple((frame,) for frame in frames)


def _normalize_additional_inputs(
    additional_ref_image: Any | None,
    additional_ref_mask: Any | None,
) -> tuple[Sequence[Any] | None, Sequence[Sequence[Any]] | None]:
    additional_images = _split_additional_images(additional_ref_image)
    additional_masks = _split_additional_masks(additional_ref_mask)
    if additional_images and not additional_masks:
        raise ValueError("additional_ref_mask is required with additional_ref_image")
    if additional_masks and not additional_images:
        raise ValueError("additional_ref_image is required with additional_ref_mask")
    if (
        additional_images
        and additional_masks
        and len(additional_images) != len(additional_masks)
    ):
        raise ValueError(
            "additional_ref_image and additional_ref_mask must have same length"
        )
    return additional_images, additional_masks


def _select_mode_video_source(
    *,
    mode: str,
    pose_video: Any,
    driving_video: Any | None,
) -> Any:
    if mode == "replacement":
        if driving_video is None:
            raise ValueError(
                "driving_video is required when mode is replacement; connect the "
                "original driving video to SCAIL-Pose2 SCAIL-2 Condition.driving_video"
            )
        return driving_video
    if mode == "animation" and pose_video is None:
        raise ValueError(
            "pose_video is required when mode is animation; connect rendered poses "
            "to SCAIL-Pose2 SCAIL-2 Condition.pose_video"
        )
    return pose_video


def _source_kind_for_reference_geometry(ref_image: Any, ref_mask: Any) -> str:
    base = "comfy_node:SCAILPose2SCAIL2Condition"
    if reference_geometry_is_aligned(ref_image, ref_mask):
        return f"{base}:reference_geometry_aligned"
    return base


def _first_bbox(value: Any):
    for bbox in frame_bboxes(value, kind="semantic_rgb_mask"):
        if bbox is not None and bbox.area > 0:
            return bbox
    return None


def _log_replacement_reference_geometry(
    *,
    ref_image: Any,
    ref_mask: Any,
    pose_video_mask: Any,
) -> None:
    if reference_geometry_is_aligned(ref_image, ref_mask):
        LOGGER.info(
            "SCAIL-Pose2 replacement reference geometry aligned: %s",
            reference_geometry_summary(ref_image, ref_mask) or "summary=unavailable",
        )
        return

    try:
        ref_image_size = frame_size(ref_image, kind="pose_image")
        ref_mask_size = frame_size(ref_mask, kind="semantic_rgb_mask")
        target_size = frame_size(pose_video_mask, kind="semantic_rgb_mask")
    except Exception as exc:
        LOGGER.info(
            "SCAIL-Pose2 replacement reference geometry diagnostic unavailable: %s",
            exc,
        )
        return

    if ref_image_size != target_size or ref_mask_size != target_size:
        LOGGER.warning(
            "SCAIL-Pose2 replacement reference geometry is not target-canvas aligned: "
            "ref_image_size=%s ref_mask_size=%s target_mask_size=%s. "
            "Use SCAIL-Pose2 Reference Image Geometry Align before Condition when "
            "reference crop/aspect differs from the driving subject.",
            ref_image_size,
            ref_mask_size,
            target_size,
        )

    try:
        ref_bbox = _first_bbox(ref_mask)
        target_bbox = _first_bbox(pose_video_mask)
    except Exception as exc:
        LOGGER.info(
            "SCAIL-Pose2 replacement reference bbox diagnostic unavailable: %s",
            exc,
        )
        return
    if ref_bbox is None or target_bbox is None:
        return
    ref_h, ref_w = ref_mask_size
    target_h, target_w = target_size
    ref_cx = ref_bbox.center_x / max(ref_w, 1)
    ref_cy = ref_bbox.center_y / max(ref_h, 1)
    target_cx = target_bbox.center_x / max(target_w, 1)
    target_cy = target_bbox.center_y / max(target_h, 1)
    center_delta = ((ref_cx - target_cx) ** 2 + (ref_cy - target_cy) ** 2) ** 0.5
    ref_height_ratio = ref_bbox.height / max(ref_h, 1)
    target_height_ratio = target_bbox.height / max(target_h, 1)
    height_ratio = (
        ref_height_ratio / target_height_ratio
        if target_height_ratio > 0.0
        else 0.0
    )
    if center_delta > 0.15 or height_ratio < 0.50 or height_ratio > 2.0:
        LOGGER.warning(
            "SCAIL-Pose2 replacement reference bbox differs from driving bbox: "
            "ref_bbox=%s target_bbox=%s normalized_center_delta=%.4f "
            "normalized_height_ratio=%.4f",
            ref_bbox.to_tuple(),
            target_bbox.to_tuple(),
            center_delta,
            height_ratio,
        )


class SCAILPose2SCAIL2Condition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_video_mask": ("IMAGE",),
                "ref_image": ("IMAGE",),
                "ref_mask": ("IMAGE",),
                "mode": (["animation", "replacement"], {"default": "animation"}),
                "width": ("INT", {"default": 512, "min": 1, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "step": 1}),
                "num_frames": ("INT", {"default": 81, "min": 1, "step": 1}),
            },
            "optional": {
                "pose_video": ("IMAGE",),
                "driving_video": ("IMAGE",),
                "additional_ref_image": ("IMAGE",),
                "additional_ref_mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("SCAIL2_CONDITION",)
    RETURN_NAMES = ("condition",)
    FUNCTION = "build"
    CATEGORY = "SCAIL-Pose2/SCAIL-2"
    DESCRIPTION = "Build a validated SCAIL-2 condition payload from RGB semantic masks."

    def build(
        self,
        pose_video_mask,
        ref_image,
        ref_mask,
        mode,
        width,
        height,
        num_frames,
        pose_video=None,
        driving_video=None,
        additional_ref_image=None,
        additional_ref_mask=None,
    ):
        progress = make_progress(3)
        started_ms = perf_counter_ms()
        LOGGER.info(
            "SCAIL-Pose2 SCAIL-2 Condition start: pose=%s driving=%s pose_mask=%s ref=%s ref_mask=%s",
            safe_value_summary(pose_video),
            safe_value_summary(driving_video),
            safe_value_summary(pose_video_mask),
            safe_value_summary(ref_image),
            safe_value_summary(ref_mask),
        )
        progress.update()
        additional_images, additional_masks = _normalize_additional_inputs(
            additional_ref_image,
            additional_ref_mask,
        )
        progress.update()
        effective_pose_video = _select_mode_video_source(
            mode=mode,
            pose_video=pose_video,
            driving_video=driving_video,
        )
        if mode == "replacement":
            LOGGER.info(
                "SCAIL-Pose2 replacement condition uses driving_video as the "
                "condition video source; sparse NLF skeleton-to-mask bbox "
                "validation is not applied."
            )
            _log_replacement_reference_geometry(
                ref_image=ref_image,
                ref_mask=ref_mask,
                pose_video_mask=pose_video_mask,
            )
        condition = build_user_mask_condition(
            mode=mode,
            ref_image=ref_image,
            ref_mask_frames=_normalize_image_frames(ref_mask, name="ref_mask"),
            pose_video=effective_pose_video,
            pose_frame_count=num_frames,
            driving_mask_frames=_normalize_image_frames(
                pose_video_mask,
                name="pose_video_mask",
            ),
            width=width,
            height=height,
            additional_ref_images=additional_images,
            additional_ref_masks=additional_masks,
            source_kind=_source_kind_for_reference_geometry(ref_image, ref_mask),
        )
        progress.update()
        LOGGER.info(
            "SCAIL-Pose2 SCAIL-2 Condition done: mode=%s frames=%s size=%sx%s elapsed_ms=%.2f",
            condition.mode,
            condition.num_frames,
            condition.width,
            condition.height,
            elapsed_ms(started_ms),
        )
        return (condition,)


NODE_CLASS_MAPPINGS = {
    "SCAILPose2SCAIL2Condition": SCAILPose2SCAIL2Condition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILPose2SCAIL2Condition": "SCAIL-Pose2 SCAIL-2 Condition",
}
