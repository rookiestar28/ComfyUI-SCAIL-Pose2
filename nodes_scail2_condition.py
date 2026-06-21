"""ComfyUI node wrappers for SCAIL-2 condition payloads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .scail2.preprocessing import build_user_mask_condition
from .scail2.geometry import diagnose_pose_mask_geometry
from .scail2.observability import (
    elapsed_ms,
    get_logger,
    make_progress,
    perf_counter_ms,
    safe_value_summary,
)

LOGGER = get_logger(__name__)

REPLACEMENT_GEOMETRY_MIN_IOU = 0.05
REPLACEMENT_GEOMETRY_MAX_CENTER_DELTA_RATIO = 0.35
REPLACEMENT_GEOMETRY_MIN_SIZE_RATIO = 0.25
REPLACEMENT_GEOMETRY_MAX_SIZE_RATIO = 4.0


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


def _validate_replacement_geometry(
    *,
    mode: str,
    pose_video: Any,
    pose_video_mask: Any,
    width: Any,
    height: Any,
) -> None:
    if mode != "replacement":
        return
    if not (_is_tensor_like_image(pose_video) or not isinstance(pose_video, (str, bytes))):
        return

    try:
        diagnostic = diagnose_pose_mask_geometry(
            pose_video=pose_video,
            pose_video_mask=pose_video_mask,
            target_width=int(width),
            target_height=int(height),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        LOGGER.info("SCAIL-Pose2 replacement geometry diagnostic skipped: %s", exc)
        return

    issues: list[str] = []
    if diagnostic.status != "ok" or diagnostic.compared_frames <= 0:
        issues.append(f"status={diagnostic.status}")
    if diagnostic.min_iou is not None and diagnostic.min_iou < REPLACEMENT_GEOMETRY_MIN_IOU:
        issues.append(f"min_iou={diagnostic.min_iou:.4f}")
    target_diagonal = (int(width) * int(width) + int(height) * int(height)) ** 0.5
    max_center_delta = (
        diagnostic.max_center_delta_px / target_diagonal
        if diagnostic.max_center_delta_px is not None and target_diagonal > 0
        else None
    )
    if (
        max_center_delta is not None
        and max_center_delta > REPLACEMENT_GEOMETRY_MAX_CENTER_DELTA_RATIO
    ):
        issues.append(f"center_delta_ratio={max_center_delta:.4f}")
    for label, ratio in (
        ("width_ratio", diagnostic.mean_width_ratio),
        ("height_ratio", diagnostic.mean_height_ratio),
    ):
        if ratio is None:
            continue
        if (
            ratio < REPLACEMENT_GEOMETRY_MIN_SIZE_RATIO
            or ratio > REPLACEMENT_GEOMETRY_MAX_SIZE_RATIO
        ):
            issues.append(f"{label}={ratio:.4f}")

    if issues:
        summary = diagnostic.to_summary()
        raise ValueError(
            "replacement pose_video geometry is not aligned with pose_video_mask; "
            "connect SCAIL-Pose2 Pose Mask Geometry Align before "
            "SCAIL-Pose2 SCAIL-2 Condition. "
            f"issues={', '.join(issues)} summary={summary}"
        )

    LOGGER.info(
        "SCAIL-Pose2 replacement geometry accepted: %s",
        diagnostic.to_summary(),
    )


class SCAILPose2SCAIL2Condition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_video": ("IMAGE",),
                "pose_video_mask": ("IMAGE",),
                "ref_image": ("IMAGE",),
                "ref_mask": ("IMAGE",),
                "mode": (["animation", "replacement"], {"default": "animation"}),
                "width": ("INT", {"default": 512, "min": 1, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "step": 1}),
                "num_frames": ("INT", {"default": 81, "min": 1, "step": 1}),
            },
            "optional": {
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
        pose_video,
        pose_video_mask,
        ref_image,
        ref_mask,
        mode,
        width,
        height,
        num_frames,
        additional_ref_image=None,
        additional_ref_mask=None,
    ):
        progress = make_progress(3)
        started_ms = perf_counter_ms()
        LOGGER.info(
            "SCAIL-Pose2 SCAIL-2 Condition start: pose=%s pose_mask=%s ref=%s ref_mask=%s",
            safe_value_summary(pose_video),
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
        _validate_replacement_geometry(
            mode=mode,
            pose_video=pose_video,
            pose_video_mask=pose_video_mask,
            width=width,
            height=height,
        )
        condition = build_user_mask_condition(
            mode=mode,
            ref_image=ref_image,
            ref_mask_frames=_normalize_image_frames(ref_mask, name="ref_mask"),
            pose_video=pose_video,
            pose_frame_count=num_frames,
            driving_mask_frames=_normalize_image_frames(
                pose_video_mask,
                name="pose_video_mask",
            ),
            width=width,
            height=height,
            additional_ref_images=additional_images,
            additional_ref_masks=additional_masks,
            source_kind="comfy_node:SCAILPose2SCAIL2Condition",
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
