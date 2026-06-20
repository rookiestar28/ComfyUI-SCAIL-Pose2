"""Typed SCAIL-2 condition bundles and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from .masks import (
    BACKGROUND_INDEX,
    MASK_OFF_THRESHOLD,
    MASK_ON_THRESHOLD,
    SEMANTIC_MASK_COLOR_NAMES,
    _to_raw_rgb,
    mask_indices_shape,
    semantic_mask_indices,
    semantic_mask_indices_tensor_raw,
)
from .wanvideo_contracts import UNSUPPORTED_CURRENT_WAN_SCAIL2_FEATURES


TYPE_SCAIL2_CONDITION = "SCAIL2_CONDITION"
SCAIL2Mode = Literal["animation", "replacement"]
SCAIL2_MODES: tuple[str, ...] = ("animation", "replacement")
MaskRole = Literal["driving", "reference"]


@dataclass(frozen=True)
class AdditionalReference:
    image: Any
    mask_indices: Any


@dataclass(frozen=True)
class SCAIL2Condition:
    type_name: str
    mode: SCAIL2Mode
    replace_flag: bool
    width: int
    height: int
    num_frames: int
    ref_image: Any
    ref_mask_indices: Any
    pose_video: Any
    driving_mask_indices: Any
    additional_references: tuple[AdditionalReference, ...]
    source_kind: str = "user_rgb_masks"
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


def _is_black_rgb(raw: tuple[int, int, int]) -> bool:
    return all(channel <= MASK_OFF_THRESHOLD for channel in raw)


def _is_white_rgb(raw: tuple[int, int, int]) -> bool:
    return all(channel >= MASK_ON_THRESHOLD for channel in raw)


def _replacement_pixel(pixel: Sequence[Any], rgb: tuple[int, int, int]) -> tuple[Any, ...]:
    channels = tuple(pixel)
    normalized = all(
        isinstance(channel, (int, float)) and 0.0 <= float(channel) <= 1.0
        for channel in channels[:3]
    )
    replacement = tuple(channel / 255.0 if normalized else channel for channel in rgb)
    return replacement + channels[3:]


def _sequence_has_black_rgb(value: Any) -> bool:
    for frame in value:
        for row in frame:
            for pixel in row:
                if _is_black_rgb(_to_raw_rgb(pixel)):
                    return True
    return False


def _normalize_rgb_mask_sequence_for_mode(
    value: Any,
    *,
    mode: SCAIL2Mode,
    role: MaskRole,
) -> Any:
    if mode != "replacement":
        return value

    # IMPORTANT: Colored Mask no longer exposes mode. Condition owns the
    # official replacement polarity bridge for Colored Mask's canonical output.
    if role == "reference" and _sequence_has_black_rgb(value):
        return value

    converted_frames = []
    for frame in value:
        converted_rows = []
        for row in frame:
            converted_pixels = []
            for pixel in row:
                raw = _to_raw_rgb(pixel)
                if role == "driving" and _is_black_rgb(raw):
                    converted_pixels.append(_replacement_pixel(pixel, (255, 255, 255)))
                elif role == "reference" and _is_white_rgb(raw):
                    converted_pixels.append(_replacement_pixel(pixel, (0, 0, 0)))
                else:
                    converted_pixels.append(pixel)
            converted_rows.append(tuple(converted_pixels))
        converted_frames.append(tuple(converted_rows))
    return tuple(converted_frames)


def _normalize_rgb_tensor_for_mode(
    value: Any,
    *,
    mode: SCAIL2Mode,
    role: MaskRole,
) -> Any:
    if mode != "replacement":
        return value

    try:
        import torch
    except ModuleNotFoundError:
        return value

    tensor = value.detach() if hasattr(value, "detach") else torch.as_tensor(value)
    view = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
    if view.ndim != 4 or view.shape[-1] < 3:
        return value

    rgb = view[..., :3].to(dtype=torch.float32)
    normalized = bool(torch.logical_and(rgb >= 0.0, rgb <= 1.0).all().item())
    off_threshold = MASK_OFF_THRESHOLD / 255.0 if normalized else MASK_OFF_THRESHOLD
    on_threshold = MASK_ON_THRESHOLD / 255.0 if normalized else MASK_ON_THRESHOLD

    black_pixels = (rgb <= off_threshold).all(dim=-1)
    white_pixels = (rgb >= on_threshold).all(dim=-1)
    if role == "reference" and bool(black_pixels.any().item()):
        return value

    replace_pixels = black_pixels if role == "driving" else white_pixels
    if not bool(replace_pixels.any().item()):
        return value

    output = tensor.clone()
    output_view = output.unsqueeze(0) if output.ndim == 3 else output
    replacement_value = 1.0 if normalized else 255.0
    if role == "driving":
        replacement = torch.full(
            (3,),
            replacement_value,
            dtype=output_view.dtype,
            device=output_view.device,
        )
    else:
        replacement = torch.zeros((3,), dtype=output_view.dtype, device=output_view.device)
    output_rgb = output_view[..., :3]
    output_rgb[replace_pixels] = replacement
    return output


def _normalize_mask_indices(
    value: Any,
    *,
    mask_name: str,
    mode: SCAIL2Mode,
    role: MaskRole,
) -> Any:
    if hasattr(value, "shape") and hasattr(value, "detach"):
        indices = semantic_mask_indices_tensor_raw(
            _normalize_rgb_tensor_for_mode(value, mode=mode, role=role)
        )
        mask_indices_shape(indices)
        return indices

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
    indices = semantic_mask_indices(
        _normalize_rgb_mask_sequence_for_mode(value, mode=mode, role=role)
    )
    mask_indices_shape(indices)
    return indices


def _mask_indices_has_foreground(indices: Any) -> bool:
    if hasattr(indices, "detach"):
        try:
            return bool((indices != BACKGROUND_INDEX).any().item())
        except AttributeError:
            return bool((indices != BACKGROUND_INDEX).any())

    for frame in indices:
        for row in frame:
            for item in row:
                if int(item) != BACKGROUND_INDEX:
                    return True
    return False


def _full_frame_reference_indices(indices: Any) -> Any:
    if hasattr(indices, "detach"):
        try:
            import torch
        except ModuleNotFoundError:
            return indices
        return torch.zeros_like(indices)

    return tuple(
        tuple(tuple(0 for _item in row) for row in frame)
        for frame in indices
    )


def _replacement_reference_indices(indices: Any) -> Any:
    if _mask_indices_has_foreground(indices):
        return indices

    # IMPORTANT: an empty Load Image alpha mask would otherwise clear the
    # replacement reference latent; fall back to the official no-mask behavior.
    return _full_frame_reference_indices(indices)


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
    mode: SCAIL2Mode,
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
            mode=mode,
            role="reference",
        )
        frame_count = _validate_mask_dimensions(
            f"additional_ref_masks[{index}]",
            mask_indices,
            width=width,
            height=height,
        )
        if frame_count != 1:
            raise ValueError("additional reference masks must contain exactly one frame")
        if mode == "replacement":
            mask_indices = _replacement_reference_indices(mask_indices)
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
    additional_ref_images: Sequence[Any] | None = None,
    additional_ref_masks: Sequence[Sequence[Any]] | None = None,
    source_kind: Any = "user_rgb_masks",
) -> SCAIL2Condition:
    if mode not in SCAIL2_MODES:
        raise ValueError(f"mode must be one of {', '.join(SCAIL2_MODES)}")
    width_value = _positive_int("width", width)
    height_value = _positive_int("height", height)
    pose_frames = _positive_int("pose_frame_count", pose_frame_count)
    source_kind_value = str(source_kind).strip()
    if not source_kind_value:
        raise ValueError("source_kind must not be empty")

    ref_mask_indices = _normalize_mask_indices(
        ref_mask_frames,
        mask_name="ref_mask_frames",
        mode=mode,
        role="reference",
    )
    ref_mask_frame_count = _validate_mask_dimensions(
        "ref_mask_frames",
        ref_mask_indices,
        width=width_value,
        height=height_value,
    )
    if ref_mask_frame_count != 1:
        raise ValueError("ref_mask_frames must contain exactly one frame")
    if mode == "replacement":
        ref_mask_indices = _replacement_reference_indices(ref_mask_indices)

    driving_mask_indices = _normalize_mask_indices(
        driving_mask_frames,
        mask_name="driving_mask_frames",
        mode=mode,
        role="driving",
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
        mode=mode,
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
        additional_references=additional_references,
        source_kind=source_kind_value,
    )
