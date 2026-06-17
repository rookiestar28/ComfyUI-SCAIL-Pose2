"""SCAIL-2 colored mask rendering helpers for SAM3 track data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


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
class ColoredMaskRenderResult:
    pose_video_mask: tuple[tuple[tuple[tuple[float, float, float], ...], ...], ...]
    reference_image_mask: tuple[tuple[tuple[tuple[float, float, float], ...], ...], ...]
    object_order: tuple[int, ...]
    sort_by: str
    replacement_mode: bool
    driving_background: tuple[float, float, float]
    reference_background: tuple[float, float, float]


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


def _normalize_track_data(track_data: dict[str, Any]) -> NormalizedTrackMasks:
    height, width = _orig_size(track_data)
    raw_masks = track_data.get("masks")
    if raw_masks is None:
        raw_masks = _unpack_packed_masks(track_data)

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
                raise ValueError(
                    f"SAM3 object {object_index} frame {frame_index} height mismatch"
                )
            normalized_rows = []
            for row in object_mask:
                if len(row) != width:
                    raise ValueError("SAM3 object masks must match orig_size width")
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


def _filtered_order(order: tuple[int, ...], object_indices: str) -> tuple[int, ...]:
    requested = _parse_object_indices(object_indices)
    if not requested:
        return order
    return tuple(order[index] for index in requested if 0 <= index < len(order))


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


def _render_track_masks(
    masks: NormalizedTrackMasks,
    *,
    order: Sequence[int],
    background: tuple[float, float, float],
) -> tuple[tuple[tuple[tuple[float, float, float], ...], ...], ...]:
    rendered = []
    for frame in masks.frames:
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
) -> tuple[tuple[tuple[tuple[float, float, float], ...], ...], ...]:
    normalized = _normalize_plain_mask(mask)
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


def render_scail2_colored_mask_pair(
    driving_track_data: dict[str, Any],
    *,
    ref_track_data: dict[str, Any] | None = None,
    ref_mask: Any | None = None,
    object_indices: str = "",
    sort_by: str = "left_to_right",
    replacement_mode: bool = False,
) -> ColoredMaskRenderResult:
    if ref_track_data is not None and ref_mask is not None:
        raise ValueError("Provide either ref_track_data or ref_mask, not both")

    driving = _normalize_track_data(driving_track_data)
    sorted_order = _sorted_object_order(driving, sort_by=sort_by)
    order = _filtered_order(sorted_order, object_indices)

    driving_background = WHITE_RGB_FLOAT if replacement_mode else BLACK_RGB_FLOAT
    reference_background = BLACK_RGB_FLOAT if replacement_mode else WHITE_RGB_FLOAT
    pose_video_mask = _render_track_masks(
        driving,
        order=order,
        background=driving_background,
    )

    if ref_mask is not None:
        reference_image_mask = _render_plain_mask(
            ref_mask,
            background=reference_background,
        )
    elif ref_track_data is not None:
        reference = _normalize_track_data(ref_track_data)
        reference_image_mask = _render_track_masks(
            reference,
            order=order,
            background=reference_background,
        )
    else:
        reference_image_mask = _solid_image(
            frame_count=1,
            height=driving.height,
            width=driving.width,
            color=reference_background,
        )

    return ColoredMaskRenderResult(
        pose_video_mask=pose_video_mask,
        reference_image_mask=reference_image_mask,
        object_order=order,
        sort_by=sort_by,
        replacement_mode=replacement_mode,
        driving_background=driving_background,
        reference_background=reference_background,
    )


def materialize_comfy_image(image_frames: Any) -> Any:
    try:
        import torch
    except ModuleNotFoundError:
        return image_frames
    return torch.tensor(image_frames, dtype=torch.float32)
