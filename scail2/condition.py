"""Typed SCAIL-2 condition bundles and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from .masks import (
    SEMANTIC_MASK_COLOR_NAMES,
    mask_indices_shape,
    semantic_mask_indices,
)
from .wanvideo_contracts import UNSUPPORTED_CURRENT_WAN_SCAIL2_FEATURES


TYPE_SCAIL2_CONDITION = "SCAIL2_CONDITION"
SCAIL2Mode = Literal["animation", "replacement", "pose_driven"]
SCAIL2_MODES: tuple[str, ...] = ("animation", "replacement", "pose_driven")


@dataclass(frozen=True)
class AdditionalReference:
    image: Any
    mask_indices: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class SCAIL2Condition:
    type_name: str
    mode: SCAIL2Mode
    replace_flag: bool
    width: int
    height: int
    num_frames: int
    ref_image: Any
    ref_mask_indices: tuple[tuple[tuple[int, ...], ...], ...]
    pose_video: Any
    driving_mask_indices: tuple[tuple[tuple[int, ...], ...], ...]
    segment_len: int
    segment_overlap: int
    additional_references: tuple[AdditionalReference, ...]
    mask_palette: tuple[str, ...] = SEMANTIC_MASK_COLOR_NAMES
    unsupported_wrapper_features: tuple[
        str, ...
    ] = UNSUPPORTED_CURRENT_WAN_SCAIL2_FEATURES


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _normalize_mask_indices(
    value: Sequence[Any],
    *,
    mask_name: str,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    try:
        first_pixel = value[0][0][0]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"{mask_name} must be a non-empty mask sequence") from exc

    if isinstance(first_pixel, int):
        shape = mask_indices_shape(value)
        return tuple(
            tuple(tuple(int(item) for item in row) for row in frame)
            for frame in value
        )
    indices = semantic_mask_indices(value)
    mask_indices_shape(indices)
    return indices


def _validate_mask_dimensions(
    mask_name: str,
    indices: Sequence[Sequence[Sequence[int]]],
    *,
    width: int,
    height: int,
) -> int:
    shape = mask_indices_shape(indices)
    if shape.width != width:
        raise ValueError(f"{mask_name} width must match condition width")
    if shape.height != height:
        raise ValueError(f"{mask_name} height must match condition height")
    return shape.frames


def _normalize_additional_references(
    additional_ref_images: Sequence[Any] | None,
    additional_ref_masks: Sequence[Sequence[Any]] | None,
    *,
    width: int,
    height: int,
) -> tuple[AdditionalReference, ...]:
    images = tuple(additional_ref_images or ())
    masks = tuple(additional_ref_masks or ())
    if images and not masks:
        raise ValueError("additional_ref_masks is required with additional_ref_images")
    if masks and not images:
        raise ValueError("additional_ref_images is required with additional_ref_masks")
    if len(images) != len(masks):
        raise ValueError("additional reference images and masks must have same length")

    additional = []
    for index, (image, mask) in enumerate(zip(images, masks)):
        mask_indices = _normalize_mask_indices(
            mask,
            mask_name=f"additional_ref_masks[{index}]",
        )
        frame_count = _validate_mask_dimensions(
            f"additional_ref_masks[{index}]",
            mask_indices,
            width=width,
            height=height,
        )
        if frame_count != 1:
            raise ValueError("additional reference masks must contain exactly one frame")
        additional.append(AdditionalReference(image=image, mask_indices=mask_indices))
    return tuple(additional)


def build_scail2_condition(
    *,
    mode: SCAIL2Mode,
    ref_image: Any,
    ref_mask_frames: Sequence[Any],
    pose_video: Any,
    pose_frame_count: Any,
    driving_mask_frames: Sequence[Any],
    width: Any,
    height: Any,
    segment_len: Any = 81,
    segment_overlap: Any = 5,
    additional_ref_images: Sequence[Any] | None = None,
    additional_ref_masks: Sequence[Sequence[Any]] | None = None,
) -> SCAIL2Condition:
    if mode not in SCAIL2_MODES:
        raise ValueError(f"mode must be one of {', '.join(SCAIL2_MODES)}")
    width_value = _positive_int("width", width)
    height_value = _positive_int("height", height)
    pose_frames = _positive_int("pose_frame_count", pose_frame_count)
    segment_len_value = _positive_int("segment_len", segment_len)
    segment_overlap_value = _positive_int("segment_overlap", segment_overlap)
    if segment_overlap_value >= segment_len_value:
        raise ValueError("segment_overlap must be smaller than segment_len")

    ref_mask_indices = _normalize_mask_indices(
        ref_mask_frames,
        mask_name="ref_mask_frames",
    )
    ref_mask_frame_count = _validate_mask_dimensions(
        "ref_mask_frames",
        ref_mask_indices,
        width=width_value,
        height=height_value,
    )
    if ref_mask_frame_count != 1:
        raise ValueError("ref_mask_frames must contain exactly one frame")

    driving_mask_indices = _normalize_mask_indices(
        driving_mask_frames,
        mask_name="driving_mask_frames",
    )
    driving_mask_frame_count = _validate_mask_dimensions(
        "driving_mask_frames",
        driving_mask_indices,
        width=width_value,
        height=height_value,
    )
    if driving_mask_frame_count != pose_frames:
        raise ValueError("pose_video and driving_mask_frames frame counts must match")

    additional_references = _normalize_additional_references(
        additional_ref_images,
        additional_ref_masks,
        width=width_value,
        height=height_value,
    )

    return SCAIL2Condition(
        type_name=TYPE_SCAIL2_CONDITION,
        mode=mode,
        replace_flag=mode == "replacement",
        width=width_value,
        height=height_value,
        num_frames=driving_mask_frame_count,
        ref_image=ref_image,
        ref_mask_indices=ref_mask_indices,
        pose_video=pose_video,
        driving_mask_indices=driving_mask_indices,
        segment_len=segment_len_value,
        segment_overlap=segment_overlap_value,
        additional_references=additional_references,
    )
