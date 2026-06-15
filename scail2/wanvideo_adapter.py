"""WanVideo SCAIL v1-style image adapter validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .wanvideo_contracts import (
    ADAPTER_FIELD_TO_WRAPPER_SOCKET,
    TYPE_IMAGE,
    UNSUPPORTED_CURRENT_WAN_SCAIL2_FEATURES,
    validate_required_adapter_fields,
)


@dataclass(frozen=True)
class ImageBatchShape:
    frames: int
    height: int
    width: int
    channels: int


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got boolean")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def image_batch_shape(value: Any, field_name: str) -> ImageBatchShape:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"{field_name} must expose a BHWC shape")
    try:
        dims = tuple(int(part) for part in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} has a non-integer shape") from exc
    if len(dims) != 4:
        raise ValueError(f"{field_name} must be a BHWC IMAGE batch")
    frames, height, width, channels = dims
    if frames <= 0:
        raise ValueError(f"{field_name} must contain at least one frame")
    if height <= 0 or width <= 0:
        raise ValueError(f"{field_name} height and width must be positive")
    if channels < 3:
        raise ValueError(f"{field_name} must have at least 3 channels")
    return ImageBatchShape(
        frames=frames,
        height=height,
        width=width,
        channels=channels,
    )


def validate_image_batch(
    value: Any,
    field_name: str,
    *,
    width: int,
    height: int,
    expected_frames: int | None = None,
) -> ImageBatchShape:
    shape = image_batch_shape(value, field_name)
    if shape.width != width:
        raise ValueError(f"{field_name} width must match target width")
    if shape.height != height:
        raise ValueError(f"{field_name} height must match target height")
    if expected_frames is not None and shape.frames != expected_frames:
        raise ValueError(f"{field_name} frame count must match num_frames")
    return shape


def build_wan_scail_images_payload(
    *,
    ref_image: Any,
    pose_images: Any,
    width: Any,
    height: Any,
    num_frames: Any,
    clip_ref_image: Any | None = None,
) -> dict[str, Any]:
    width_value = _positive_int("width", width)
    height_value = _positive_int("height", height)
    frame_count = _positive_int("num_frames", num_frames)
    clip_image = ref_image if clip_ref_image is None else clip_ref_image

    validate_required_adapter_fields(
        ("width", "height", "num_frames", "ref_image", "pose_images")
    )
    ref_shape = validate_image_batch(
        ref_image,
        "ref_image",
        width=width_value,
        height=height_value,
    )
    pose_shape = validate_image_batch(
        pose_images,
        "pose_images",
        width=width_value,
        height=height_value,
        expected_frames=frame_count,
    )
    clip_shape = validate_image_batch(
        clip_image,
        "clip_ref_image",
        width=width_value,
        height=height_value,
    )

    socket_map = {
        field_name: {
            "wrapper_node": contract.wrapper_node,
            "wrapper_socket": contract.wrapper_socket,
            "comfy_type": contract.comfy_type,
        }
        for field_name, contract in ADAPTER_FIELD_TO_WRAPPER_SOCKET.items()
        if contract.ownership == "scail_pose2"
    }

    return {
        "kind": "wan_scail_v1_images",
        "width": width_value,
        "height": height_value,
        "num_frames": frame_count,
        "ref_image": ref_image,
        "pose_images": pose_images,
        "clip_ref_image": clip_image,
        "image_shapes": {
            "ref_image": ref_shape,
            "pose_images": pose_shape,
            "clip_ref_image": clip_shape,
        },
        "adapter_fields": {
            "ref_image": TYPE_IMAGE,
            "pose_images": TYPE_IMAGE,
            "clip_ref_image": TYPE_IMAGE,
            "width": "INT",
            "height": "INT",
            "num_frames": "INT",
        },
        "socket_map": socket_map,
        "unsupported_wrapper_features": UNSUPPORTED_CURRENT_WAN_SCAIL2_FEATURES,
    }
