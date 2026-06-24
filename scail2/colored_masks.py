"""SCAIL-2 colored mask rendering helpers for SAM3 track data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Sequence

from .identity import IdentityContractDiagnostics, build_identity_slots
from .observability import safe_value_summary

BLACK_RGB_FLOAT = (0.0, 0.0, 0.0)
WHITE_RGB_FLOAT = (1.0, 1.0, 1.0)
BLUE_RGB_FLOAT = (0.0, 0.0, 1.0)
RED_RGB_FLOAT = (1.0, 0.0, 0.0)
GREEN_RGB_FLOAT = (0.0, 1.0, 0.0)
MAGENTA_RGB_FLOAT = (1.0, 0.0, 1.0)
CYAN_RGB_FLOAT = (0.0, 1.0, 1.0)
YELLOW_RGB_FLOAT = (1.0, 1.0, 0.0)

SCAIL2_IDENTITY_PALETTE_FLOAT: tuple[tuple[float, float, float], ...] = (
    BLUE_RGB_FLOAT,
    RED_RGB_FLOAT,
    GREEN_RGB_FLOAT,
    MAGENTA_RGB_FLOAT,
    CYAN_RGB_FLOAT,
    YELLOW_RGB_FLOAT,
)


@dataclass(frozen=True)
class NormalizedTrackMasks:
    frames: tuple[tuple[tuple[tuple[bool, ...], ...], ...], ...]
    frame_count: int
    object_count: int
    height: int
    width: int


@dataclass(frozen=True)
class TensorTrackMasks:
    frames: Any
    frame_count: int
    object_count: int
    height: int
    width: int
    source_field: str


@dataclass(frozen=True)
class ColoredMaskRenderResult:
    pose_video_mask: Any
    reference_image_mask: Any
    object_order: tuple[int, ...]
    sort_by: str
    replacement_mode: bool
    driving_background: tuple[float, float, float]
    reference_background: tuple[float, float, float]
    identity: IdentityContractDiagnostics = field(
        default_factory=lambda: IdentityContractDiagnostics(slots=())
    )


ProgressCallback = Callable[[str], None]


def summarize_sam3_track_data(track_data: Any) -> dict[str, Any]:
    if not isinstance(track_data, dict):
        return {"source": "unknown", "value": safe_value_summary(track_data)}

    masks = track_data.get("masks")
    packed_masks = track_data.get("packed_masks")
    if masks is not None:
        source = "masks"
    elif packed_masks is not None:
        source = "packed_masks"
    else:
        source = "empty"

    summary: dict[str, Any] = {
        "source": source,
        "masks": safe_value_summary(masks),
        "packed_masks": safe_value_summary(packed_masks),
    }
    if "orig_size" in track_data:
        try:
            summary["orig_size"] = list(_orig_size(track_data))
        except ValueError:
            summary["orig_size"] = "invalid"
    if "n_frames" in track_data:
        try:
            summary["n_frames"] = int(track_data["n_frames"])
        except (TypeError, ValueError):
            summary["n_frames"] = "invalid"
    return summary


def _as_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return float(value) > 0.5
    except (TypeError, ValueError):
        return bool(value)


def _orig_size(track_data: dict[str, Any]) -> tuple[int, int]:
    if "orig_size" not in track_data:
        raise ValueError("SAM3 track data must include orig_size")
    size = tuple(int(part) for part in track_data["orig_size"])
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError("SAM3 track data orig_size must be (height, width)")
    return size


def _unpack_packed_masks(track_data: dict[str, Any]) -> Any:
    packed = track_data.get("packed_masks")
    if packed is None:
        return None
    try:
        from comfy.ldm.sam3.tracker import unpack_masks
    except Exception as exc:  # pragma: no cover - requires ComfyUI runtime
        raise RuntimeError(
            "Packed SAM3 masks require ComfyUI's SAM3 unpack_masks helper. "
            "Provide expanded track_data['masks'] in tests or run inside ComfyUI."
        ) from exc
    return unpack_masks(packed)


def _torch_or_none() -> Any | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def _is_torch_tensor(value: Any, torch: Any | None = None) -> bool:
    torch_module = torch if torch is not None else _torch_or_none()
    return bool(torch_module is not None and isinstance(value, torch_module.Tensor))


def _preferred_tensor_device(tensor: Any) -> Any:
    try:
        import comfy.model_management

        return comfy.model_management.intermediate_device()
    except Exception:
        return tensor.device


def _normalize_track_data_tensor(track_data: dict[str, Any]) -> TensorTrackMasks | None:
    torch = _torch_or_none()
    if torch is None:
        return None

    height, width = _orig_size(track_data)
    raw_masks = track_data.get("masks")
    source_field = "masks"
    masks = None
    device = None

    if _is_torch_tensor(raw_masks, torch):
        masks = raw_masks
        device = _preferred_tensor_device(raw_masks)
    elif raw_masks is None:
        packed = track_data.get("packed_masks")
        if not _is_torch_tensor(packed, torch):
            return None
        source_field = "packed_masks"
        device = _preferred_tensor_device(packed)
        if packed.shape[1] == 0:
            frame_count = int(packed.shape[0])
            masks = torch.zeros(
                (frame_count, 0, height, width),
                dtype=torch.bool,
                device=device,
            )
        else:
            try:
                from comfy.ldm.sam3.tracker import unpack_masks
            except Exception as exc:  # pragma: no cover - requires ComfyUI runtime
                raise RuntimeError(
                    "Packed SAM3 masks require ComfyUI's SAM3 unpack_masks helper. "
                    "Provide expanded track_data['masks'] in tests or run inside ComfyUI."
                ) from exc
            masks = unpack_masks(packed.to(device))
    else:
        return None

    if masks.ndim != 4:
        raise ValueError(
            f"SAM3 tensor {source_field} must have shape [frames, objects, height, width]"
        )
    if masks.shape[0] <= 0:
        raise ValueError("SAM3 track masks must contain at least one frame")
    if masks.shape[2] <= 0 or masks.shape[3] <= 0:
        raise ValueError("SAM3 tensor mask spatial dimensions must be positive")

    frame_count = int(masks.shape[0])
    object_count = int(masks.shape[1])
    masks_bool = masks.to(device=device)
    if masks_bool.dtype != torch.bool:
        masks_bool = masks_bool > 0.5

    mask_height = int(masks_bool.shape[2])
    mask_width = int(masks_bool.shape[3])
    if (mask_height, mask_width) != (height, width):
        if object_count == 0:
            masks_bool = torch.zeros(
                (frame_count, 0, height, width),
                dtype=torch.bool,
                device=device,
            )
        else:
            import torch.nn.functional as F

            resized = F.interpolate(
                masks_bool.float().view(frame_count * object_count, 1, mask_height, mask_width),
                size=(height, width),
                mode="nearest",
            )
            masks_bool = resized.view(frame_count, object_count, height, width) > 0.5

    return TensorTrackMasks(
        frames=masks_bool.contiguous(),
        frame_count=frame_count,
        object_count=object_count,
        height=height,
        width=width,
        source_field=source_field,
    )


def _nearest_source_index(target_index: int, source_size: int, target_size: int) -> int:
    return min(int(target_index * source_size / target_size), source_size - 1)


def _resize_binary_mask_to_size(
    mask: Any,
    *,
    height: int,
    width: int,
) -> tuple[tuple[Any, ...], ...]:
    source_height = len(mask)
    if source_height <= 0:
        raise ValueError("SAM3 packed object masks must not be empty")
    source_width = len(mask[0])
    if source_width <= 0:
        raise ValueError("SAM3 packed object mask rows must not be empty")
    for row in mask:
        if len(row) != source_width:
            raise ValueError("SAM3 packed object mask rows must have consistent width")

    if source_height == height and source_width == width:
        return tuple(tuple(row) for row in mask)

    # Official ComfyUI SCAIL resizes unpacked SAM3 packed masks to orig_size.
    resized_rows = []
    for row_index in range(height):
        source_row = mask[_nearest_source_index(row_index, source_height, height)]
        resized_rows.append(
            tuple(
                source_row[_nearest_source_index(col_index, source_width, width)]
                for col_index in range(width)
            )
        )
    return tuple(resized_rows)


def _resize_track_masks_to_orig_size(
    raw_masks: Any,
    *,
    height: int,
    width: int,
) -> tuple[tuple[tuple[tuple[Any, ...], ...], ...], ...]:
    raw_frames = _as_list(raw_masks)
    return tuple(
        tuple(
            _resize_binary_mask_to_size(object_mask, height=height, width=width)
            for object_mask in frame
        )
        for frame in raw_frames
    )


def _mask_shape(mask: Any) -> tuple[Any, Any]:
    try:
        mask_height = len(mask)
    except TypeError:
        return ("unknown", "unknown")
    if mask_height <= 0:
        return (mask_height, 0)
    try:
        mask_width = len(mask[0])
    except TypeError:
        mask_width = "unknown"
    return (mask_height, mask_width)


def _track_shape_error(
    *,
    source_field: str,
    frame_index: int,
    object_index: int,
    expected_size: tuple[int, int],
    actual_shape: tuple[Any, Any],
) -> ValueError:
    return ValueError(
        "SAM3 track mask shape mismatch: "
        f"source={source_field} "
        f"frame={frame_index} "
        f"object={object_index} "
        f"orig_size={expected_size} "
        f"actual_shape={actual_shape}"
    )


def _normalize_track_data(track_data: dict[str, Any]) -> NormalizedTrackMasks:
    height, width = _orig_size(track_data)
    raw_masks = track_data.get("masks")
    source_field = "masks"
    if raw_masks is None:
        raw_masks = _unpack_packed_masks(track_data)
        source_field = "packed_masks"
        if raw_masks is not None:
            raw_masks = _resize_track_masks_to_orig_size(
                raw_masks,
                height=height,
                width=width,
            )

    if raw_masks is None:
        frame_count = int(track_data.get("n_frames", 1))
        if frame_count <= 0:
            raise ValueError("SAM3 track data n_frames must be positive")
        return NormalizedTrackMasks(
            frames=tuple(tuple() for _frame in range(frame_count)),
            frame_count=frame_count,
            object_count=0,
            height=height,
            width=width,
        )

    raw_frames = _as_list(raw_masks)
    if not raw_frames:
        raise ValueError("SAM3 track masks must contain at least one frame")

    normalized_frames = []
    expected_objects: int | None = None
    for frame_index, frame in enumerate(raw_frames):
        if expected_objects is None:
            expected_objects = len(frame)
        elif len(frame) != expected_objects:
            raise ValueError("SAM3 track masks must have consistent object count")

        normalized_objects = []
        for object_index, object_mask in enumerate(frame):
            if len(object_mask) != height:
                raise _track_shape_error(
                    source_field=source_field,
                    frame_index=frame_index,
                    object_index=object_index,
                    expected_size=(height, width),
                    actual_shape=_mask_shape(object_mask),
                )
            normalized_rows = []
            for row in object_mask:
                if len(row) != width:
                    raise _track_shape_error(
                        source_field=source_field,
                        frame_index=frame_index,
                        object_index=object_index,
                        expected_size=(height, width),
                        actual_shape=_mask_shape(object_mask),
                    )
                normalized_rows.append(tuple(_as_bool(value) for value in row))
            normalized_objects.append(tuple(normalized_rows))
        normalized_frames.append(tuple(normalized_objects))

    return NormalizedTrackMasks(
        frames=tuple(normalized_frames),
        frame_count=len(normalized_frames),
        object_count=expected_objects or 0,
        height=height,
        width=width,
    )


def _object_stats(
    masks: NormalizedTrackMasks,
) -> tuple[tuple[int, float, float], ...]:
    stats = []
    total_pixels = masks.height * masks.width
    for object_index in range(masks.object_count):
        first_frame = masks.frame_count
        centroid_x = 1.0
        area_ratio = 0.0
        for frame_index, frame in enumerate(masks.frames):
            active_cols = []
            active_count = 0
            for row in frame[object_index]:
                for col_index, active in enumerate(row):
                    if active:
                        active_cols.append(col_index)
                        active_count += 1
            if active_count:
                first_frame = frame_index
                centroid_x = sum(active_cols) / active_count / max(masks.width, 1)
                area_ratio = active_count / total_pixels
                break
        stats.append((first_frame, centroid_x, area_ratio))
    return tuple(stats)


def _object_stats_tensor(
    masks: TensorTrackMasks,
) -> tuple[tuple[int, float, float], ...]:
    if masks.object_count == 0:
        return ()

    torch = _torch_or_none()
    if torch is None:  # pragma: no cover - tensor path requires torch
        raise RuntimeError("torch is required for tensor SAM3 masks")

    tensor = masks.frames.float()
    total_pixels = masks.height * masks.width
    frame_indices = torch.arange(
        masks.frame_count,
        device=tensor.device,
        dtype=torch.long,
    ).unsqueeze(1).expand(masks.frame_count, masks.object_count)
    area_t = tensor.sum(dim=(-1, -2))
    present = area_t > 0
    missing_frame = torch.full_like(frame_indices, masks.frame_count)
    first_t = torch.where(present, frame_indices, missing_frame).amin(dim=0)

    grid_x = torch.arange(
        masks.width,
        device=tensor.device,
        dtype=tensor.dtype,
    ).view(1, 1, 1, masks.width)
    cx_t = (tensor * grid_x).sum(dim=(-1, -2)) / area_t.clamp(min=1)
    selected_t = first_t.clamp(max=max(masks.frame_count - 1, 0)).unsqueeze(0)
    cx = cx_t.gather(0, selected_t).squeeze(0) / max(masks.width, 1)
    area = area_t.gather(0, selected_t).squeeze(0) / total_pixels
    never_present = first_t >= masks.frame_count
    cx = torch.where(never_present, torch.ones_like(cx), cx)
    area = torch.where(never_present, torch.zeros_like(area), area)

    return tuple(
        (int(first_t[index].item()), float(cx[index].item()), float(area[index].item()))
        for index in range(masks.object_count)
    )


def _parse_object_indices(value: str) -> tuple[int, ...]:
    indices = []
    for raw_item in (value or "").split(","):
        item = raw_item.strip()
        if item.isdigit():
            indices.append(int(item))
    return tuple(indices)


def _sorted_object_order(
    masks: NormalizedTrackMasks,
    *,
    sort_by: str,
) -> tuple[int, ...]:
    if sort_by not in {"none", "left_to_right", "area"}:
        raise ValueError("sort_by must be one of none, left_to_right, area")
    order = tuple(range(masks.object_count))
    if sort_by == "none" or masks.object_count == 0:
        return order

    stats = _object_stats(masks)
    if sort_by == "left_to_right":
        return tuple(sorted(order, key=lambda index: (stats[index][0], stats[index][1], index)))
    return tuple(sorted(order, key=lambda index: (stats[index][0], -stats[index][2], index)))


def _sorted_object_order_tensor(
    masks: TensorTrackMasks,
    *,
    sort_by: str,
) -> tuple[int, ...]:
    if sort_by not in {"none", "left_to_right", "area"}:
        raise ValueError("sort_by must be one of none, left_to_right, area")
    order = tuple(range(masks.object_count))
    if sort_by == "none" or masks.object_count == 0:
        return order

    stats = _object_stats_tensor(masks)
    if sort_by == "left_to_right":
        return tuple(sorted(order, key=lambda index: (stats[index][0], stats[index][1], index)))
    return tuple(sorted(order, key=lambda index: (stats[index][0], -stats[index][2], index)))


def _filtered_order(order: tuple[int, ...], object_indices: str) -> tuple[int, ...]:
    requested = _parse_object_indices(object_indices)
    if not requested:
        return order
    return tuple(order[index] for index in requested if 0 <= index < len(order))


def _bg_tensor(
    color: tuple[float, float, float],
    *,
    device: Any,
    dtype: Any,
) -> Any:
    torch = _torch_or_none()
    if torch is None:  # pragma: no cover - tensor path requires torch
        raise RuntimeError("torch is required for tensor SAM3 masks")
    return torch.tensor(color, device=device, dtype=dtype)


def _solid_image(
    *,
    frame_count: int,
    height: int,
    width: int,
    color: tuple[float, float, float],
) -> tuple[tuple[tuple[tuple[float, float, float], ...], ...], ...]:
    return tuple(
        tuple(
            tuple(color for _col in range(width))
            for _row in range(height)
        )
        for _frame in range(frame_count)
    )


def _solid_tensor_image(
    *,
    frame_count: int,
    height: int,
    width: int,
    color: tuple[float, float, float],
    device: Any,
) -> Any:
    torch = _torch_or_none()
    if torch is None:  # pragma: no cover - tensor path requires torch
        raise RuntimeError("torch is required for tensor SAM3 masks")
    bg = _bg_tensor(color, device=device, dtype=torch.float32)
    return bg.view(1, 1, 1, 3).expand(frame_count, height, width, 3).clone()


def _report_tensor_render_progress(
    *,
    frame_count: int,
    width: int,
    height: int,
    object_count: int,
    progress: ProgressCallback | None,
    label: str,
) -> None:
    if progress is None:
        return
    frame_log_interval = max(1, frame_count // 10)
    for frame_index in range(frame_count):
        if (
            frame_index == 0
            or frame_index == frame_count - 1
            or frame_index % frame_log_interval == 0
        ):
            progress(
                f"render {label} frame {frame_index + 1}/{frame_count} "
                f"objects={object_count} size={width}x{height}"
            )


def _render_tensor_track_masks(
    masks: TensorTrackMasks,
    *,
    order: Sequence[int],
    background: tuple[float, float, float],
    progress: ProgressCallback | None = None,
    label: str = "track",
) -> Any:
    torch = _torch_or_none()
    if torch is None:  # pragma: no cover - tensor path requires torch
        raise RuntimeError("torch is required for tensor SAM3 masks")

    _report_tensor_render_progress(
        frame_count=masks.frame_count,
        width=masks.width,
        height=masks.height,
        object_count=len(order),
        progress=progress,
        label=label,
    )
    output = _solid_tensor_image(
        frame_count=masks.frame_count,
        height=masks.height,
        width=masks.width,
        color=background,
        device=masks.frames.device,
    )
    valid_pairs = [
        (palette_index, object_index)
        for palette_index, object_index in enumerate(order)
        if 0 <= object_index < masks.object_count
    ]
    if not valid_pairs:
        return output

    object_indices = [object_index for _palette_index, object_index in valid_pairs]
    palette = [
        SCAIL2_IDENTITY_PALETTE_FLOAT[
            palette_index % len(SCAIL2_IDENTITY_PALETTE_FLOAT)
        ]
        for palette_index, _object_index in valid_pairs
    ]
    selected = masks.frames[:, object_indices, :, :]
    any_mask = selected.any(dim=1)
    first_selected = selected.to(torch.uint8).argmax(dim=1).long()
    colors = torch.tensor(palette, device=masks.frames.device, dtype=torch.float32)
    color_overlay = colors[first_selected]
    return torch.where(any_mask.unsqueeze(-1), color_overlay, output)


def _render_track_masks(
    masks: NormalizedTrackMasks,
    *,
    order: Sequence[int],
    background: tuple[float, float, float],
    progress: ProgressCallback | None = None,
    label: str = "track",
) -> tuple[tuple[tuple[tuple[float, float, float], ...], ...], ...]:
    rendered = []
    frame_log_interval = max(1, masks.frame_count // 10)
    for frame_index, frame in enumerate(masks.frames):
        if progress is not None and (
            frame_index == 0
            or frame_index == masks.frame_count - 1
            or frame_index % frame_log_interval == 0
        ):
            progress(
                f"render {label} frame {frame_index + 1}/{masks.frame_count} "
                f"objects={len(order)} size={masks.width}x{masks.height}"
            )
        rendered_rows = []
        for row_index in range(masks.height):
            rendered_row = []
            for col_index in range(masks.width):
                color = background
                for palette_index, object_index in enumerate(order):
                    if object_index >= masks.object_count:
                        continue
                    if frame[object_index][row_index][col_index]:
                        color = SCAIL2_IDENTITY_PALETTE_FLOAT[
                            palette_index % len(SCAIL2_IDENTITY_PALETTE_FLOAT)
                        ]
                        break
                rendered_row.append(color)
            rendered_rows.append(tuple(rendered_row))
        rendered.append(tuple(rendered_rows))
    return tuple(rendered)


def _normalize_plain_mask(
    mask: Any,
) -> tuple[tuple[tuple[bool, ...], ...], ...]:
    raw_mask = _as_list(mask)
    if not raw_mask:
        raise ValueError("plain reference mask must not be empty")
    if raw_mask and raw_mask[0] and not isinstance(raw_mask[0][0], (list, tuple)):
        raw_mask = [raw_mask]

    frames = []
    expected_height: int | None = None
    expected_width: int | None = None
    for frame in raw_mask:
        if not frame:
            raise ValueError("plain reference mask frames must not be empty")
        if expected_height is None:
            expected_height = len(frame)
            expected_width = len(frame[0])
        elif len(frame) != expected_height:
            raise ValueError("plain reference mask frames must have consistent height")
        rows = []
        for row in frame:
            if len(row) != expected_width:
                raise ValueError("plain reference mask rows must have consistent width")
            rows.append(tuple(_as_bool(value) for value in row))
        frames.append(tuple(rows))
    return tuple(frames)


def _render_plain_mask(
    mask: Any,
    *,
    background: tuple[float, float, float],
    progress: ProgressCallback | None = None,
) -> tuple[tuple[tuple[tuple[float, float, float], ...], ...], ...]:
    normalized = _normalize_plain_mask(mask)
    if progress is not None:
        progress(f"render plain reference mask frames={len(normalized)}")
    rendered = []
    for frame in normalized:
        rows = []
        for row in frame:
            rows.append(
                tuple(
                    SCAIL2_IDENTITY_PALETTE_FLOAT[0] if active else background
                    for active in row
                )
            )
        rendered.append(tuple(rows))
    return tuple(rendered)


def _render_tensor_plain_mask(
    mask: Any,
    *,
    background: tuple[float, float, float],
    progress: ProgressCallback | None = None,
) -> Any | None:
    torch = _torch_or_none()
    if not _is_torch_tensor(mask, torch):
        return None
    tensor = mask.detach()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError("plain reference tensor mask must have shape [frames, height, width] or [height, width]")
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
        raise ValueError("plain reference tensor mask must be non-empty")
    if progress is not None:
        progress(f"render plain reference mask frames={int(tensor.shape[0])}")

    device = _preferred_tensor_device(tensor)
    mask_bool = tensor.to(device=device)
    if mask_bool.dtype != torch.bool:
        mask_bool = mask_bool > 0.5
    frame_count, height, width = (int(part) for part in mask_bool.shape)
    output = _solid_tensor_image(
        frame_count=frame_count,
        height=height,
        width=width,
        color=background,
        device=mask_bool.device,
    )
    color = _bg_tensor(
        SCAIL2_IDENTITY_PALETTE_FLOAT[0],
        device=mask_bool.device,
        dtype=torch.float32,
    )
    color_overlay = color.view(1, 1, 1, 3).expand(frame_count, height, width, 3)
    return torch.where(mask_bool.unsqueeze(-1), color_overlay, output)


def render_scail2_colored_mask_pair(
    driving_track_data: dict[str, Any],
    *,
    ref_track_data: dict[str, Any] | None = None,
    ref_mask: Any | None = None,
    object_indices: str = "",
    sort_by: str = "left_to_right",
    replacement_mode: bool = False,
    progress: ProgressCallback | None = None,
) -> ColoredMaskRenderResult:
    if ref_track_data is not None and ref_mask is not None:
        raise ValueError("Provide either ref_track_data or ref_mask, not both")

    if progress is not None:
        progress(f"normalize driving track {summarize_sam3_track_data(driving_track_data)}")
    tensor_driving = _normalize_track_data_tensor(driving_track_data)
    if tensor_driving is not None:
        if progress is not None:
            progress(
                f"driving normalized frames={tensor_driving.frame_count} "
                f"objects={tensor_driving.object_count} size={tensor_driving.width}x{tensor_driving.height}"
            )
        sorted_order = _sorted_object_order_tensor(tensor_driving, sort_by=sort_by)
        order = _filtered_order(sorted_order, object_indices)
        identity = build_identity_slots(
            object_order=order,
            object_stats=_object_stats_tensor(tensor_driving),
            frame_count=tensor_driving.frame_count,
            object_count=tensor_driving.object_count,
        )
        if progress is not None:
            progress(f"object order sort_by={sort_by} selected={list(order)}")
            progress(identity.summary())

        driving_background = WHITE_RGB_FLOAT if replacement_mode else BLACK_RGB_FLOAT
        reference_background = BLACK_RGB_FLOAT if replacement_mode else WHITE_RGB_FLOAT
        pose_video_mask = _render_tensor_track_masks(
            tensor_driving,
            order=order,
            background=driving_background,
            progress=progress,
            label="driving",
        )

        if ref_mask is not None:
            reference_image_mask = _render_tensor_plain_mask(
                ref_mask,
                background=reference_background,
                progress=progress,
            )
            if reference_image_mask is None:
                reference_image_mask = _render_plain_mask(
                    ref_mask,
                    background=reference_background,
                    progress=progress,
                )
        elif ref_track_data is not None:
            if progress is not None:
                progress(f"normalize reference track {summarize_sam3_track_data(ref_track_data)}")
            tensor_reference = _normalize_track_data_tensor(ref_track_data)
            if tensor_reference is not None:
                if progress is not None:
                    progress(
                        f"reference normalized frames={tensor_reference.frame_count} "
                        f"objects={tensor_reference.object_count} size={tensor_reference.width}x{tensor_reference.height}"
                    )
                reference_image_mask = _render_tensor_track_masks(
                    tensor_reference,
                    order=order,
                    background=reference_background,
                    progress=progress,
                    label="reference",
                )
            else:
                reference = _normalize_track_data(ref_track_data)
                if progress is not None:
                    progress(
                        f"reference normalized frames={reference.frame_count} "
                        f"objects={reference.object_count} size={reference.width}x{reference.height}"
                    )
                reference_image_mask = _render_track_masks(
                    reference,
                    order=order,
                    background=reference_background,
                    progress=progress,
                    label="reference",
                )
        else:
            if progress is not None:
                progress(
                    f"render solid reference mask frames=1 size={tensor_driving.width}x{tensor_driving.height}"
                )
            reference_image_mask = _solid_tensor_image(
                frame_count=1,
                height=tensor_driving.height,
                width=tensor_driving.width,
                color=reference_background,
                device=tensor_driving.frames.device,
            )

        if progress is not None:
            progress(
                f"render complete driving_frames={tensor_driving.frame_count} "
                f"reference_frames={len(reference_image_mask)} selected_objects={len(order)}"
            )
        return ColoredMaskRenderResult(
            pose_video_mask=pose_video_mask,
            reference_image_mask=reference_image_mask,
            object_order=order,
            sort_by=sort_by,
            replacement_mode=bool(replacement_mode),
            driving_background=driving_background,
            reference_background=reference_background,
            identity=identity,
        )

    driving = _normalize_track_data(driving_track_data)
    if progress is not None:
        progress(
            f"driving normalized frames={driving.frame_count} "
            f"objects={driving.object_count} size={driving.width}x{driving.height}"
        )
    sorted_order = _sorted_object_order(driving, sort_by=sort_by)
    order = _filtered_order(sorted_order, object_indices)
    identity = build_identity_slots(
        object_order=order,
        object_stats=_object_stats(driving),
        frame_count=driving.frame_count,
        object_count=driving.object_count,
    )
    if progress is not None:
        progress(f"object order sort_by={sort_by} selected={list(order)}")
        progress(identity.summary())

    driving_background = WHITE_RGB_FLOAT if replacement_mode else BLACK_RGB_FLOAT
    reference_background = BLACK_RGB_FLOAT if replacement_mode else WHITE_RGB_FLOAT
    pose_video_mask = _render_track_masks(
        driving,
        order=order,
        background=driving_background,
        progress=progress,
        label="driving",
    )

    if ref_mask is not None:
        reference_image_mask = _render_plain_mask(
            ref_mask,
            background=reference_background,
            progress=progress,
        )
    elif ref_track_data is not None:
        if progress is not None:
            progress(f"normalize reference track {summarize_sam3_track_data(ref_track_data)}")
        reference = _normalize_track_data(ref_track_data)
        if progress is not None:
            progress(
                f"reference normalized frames={reference.frame_count} "
                f"objects={reference.object_count} size={reference.width}x{reference.height}"
            )
        reference_image_mask = _render_track_masks(
            reference,
            order=order,
            background=reference_background,
            progress=progress,
            label="reference",
        )
    else:
        if progress is not None:
            progress(
                f"render solid reference mask frames=1 size={driving.width}x{driving.height}"
            )
        reference_image_mask = _solid_image(
            frame_count=1,
            height=driving.height,
            width=driving.width,
            color=reference_background,
        )

    if progress is not None:
        progress(
            f"render complete driving_frames={driving.frame_count} "
            f"reference_frames={len(reference_image_mask)} selected_objects={len(order)}"
        )
    return ColoredMaskRenderResult(
        pose_video_mask=pose_video_mask,
        reference_image_mask=reference_image_mask,
        object_order=order,
        sort_by=sort_by,
        replacement_mode=replacement_mode,
        driving_background=driving_background,
        reference_background=reference_background,
        identity=identity,
    )


def materialize_comfy_image(image_frames: Any) -> Any:
    if hasattr(image_frames, "detach") and hasattr(image_frames, "to"):
        try:
            import torch
        except ModuleNotFoundError:
            return image_frames
        if isinstance(image_frames, torch.Tensor):
            # IMPORTANT: file/preview nodes expect canonical ComfyUI IMAGE tensors.
            return (
                image_frames.detach()
                .to(device="cpu", dtype=torch.float32)
                .clamp(0.0, 1.0)
                .contiguous()
            )
        return image_frames.to(dtype=torch.float32)
    try:
        import torch
    except ModuleNotFoundError:
        return image_frames
    return torch.tensor(image_frames, dtype=torch.float32).clamp(0.0, 1.0).contiguous()
