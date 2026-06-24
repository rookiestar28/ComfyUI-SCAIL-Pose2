"""Driving-mask anchored pose image alignment helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .geometry import (
    BoundingBox,
    GeometryDiagnostic,
    diagnose_pose_mask_geometry,
    frame_bboxes,
    frame_size,
)


@dataclass(frozen=True)
class PoseMaskAlignmentResult:
    pose_video: Any
    summary: str
    before: GeometryDiagnostic
    after: GeometryDiagnostic
    temporal: "AlignmentTemporalDiagnostic | None" = None


@dataclass(frozen=True)
class _AlignmentFrameMap:
    mode: str
    mask_indices: tuple[int, ...]
    pose_frame_count: int
    mask_frame_count: int


@dataclass(frozen=True)
class AlignmentTransform:
    frame_index: int
    pose_bbox: BoundingBox | None
    target_bbox: BoundingBox | None
    scale_x: float | None
    scale_y: float | None
    translate_x: float | None
    translate_y: float | None
    reason: str

    @property
    def valid(self) -> bool:
        return (
            self.pose_bbox is not None
            and self.target_bbox is not None
            and self.scale_x is not None
            and self.scale_y is not None
            and self.translate_x is not None
            and self.translate_y is not None
        )


@dataclass(frozen=True)
class AlignmentTemporalDiagnostic:
    frame_count: int
    valid_transform_count: int
    invalid_transform_count: int
    max_center_jump_px: float | None
    worst_center_jump_frame_index: int | None
    max_center_impulse_px: float | None
    worst_center_impulse_frame_index: int | None
    max_scale_jump_ratio: float | None
    worst_scale_jump_frame_index: int | None
    max_scale_impulse_ratio: float | None
    worst_scale_impulse_frame_index: int | None
    transforms: tuple[AlignmentTransform, ...]

    def to_summary(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "valid_transform_count": self.valid_transform_count,
            "invalid_transform_count": self.invalid_transform_count,
            "max_center_jump_px": self.max_center_jump_px,
            "worst_center_jump_frame_index": self.worst_center_jump_frame_index,
            "max_center_impulse_px": self.max_center_impulse_px,
            "worst_center_impulse_frame_index": self.worst_center_impulse_frame_index,
            "max_scale_jump_ratio": self.max_scale_jump_ratio,
            "worst_scale_jump_frame_index": self.worst_scale_jump_frame_index,
            "max_scale_impulse_ratio": self.max_scale_impulse_ratio,
            "worst_scale_impulse_frame_index": self.worst_scale_impulse_frame_index,
        }


def _torch_or_none() -> Any | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def _is_torch_tensor(value: Any) -> bool:
    torch = _torch_or_none()
    return bool(torch is not None and isinstance(value, torch.Tensor))


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


def _looks_like_hwc_image(frame: Any) -> bool:
    try:
        pixel = frame[0][0]
    except (IndexError, TypeError):
        return False
    return isinstance(pixel, (list, tuple)) and len(pixel) >= 3 and _is_number(pixel[0])


def _split_image_frames(value: Any) -> tuple[Any, ...]:
    raw = _as_list(value)
    if not raw:
        raise ValueError("pose_video must not be empty")
    if _looks_like_hwc_image(raw):
        return (raw,)
    if not _looks_like_hwc_image(raw[0]):
        raise ValueError("pose_video must have shape [H, W, C] or [T, H, W, C]")
    return tuple(raw)


def _target_mask_bbox_in_pose_space(
    *,
    mask_bbox: BoundingBox,
    mask_size: tuple[int, int],
    pose_size: tuple[int, int],
) -> BoundingBox:
    mask_height, mask_width = mask_size
    pose_height, pose_width = pose_size
    return mask_bbox.scale(
        scale_x=pose_width / mask_width,
        scale_y=pose_height / mask_height,
    )


def _int_bbox(
    bbox: BoundingBox,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x_min = max(0, min(width - 1, int(math.floor(bbox.x_min))))
    y_min = max(0, min(height - 1, int(math.floor(bbox.y_min))))
    x_max = max(x_min + 1, min(width, int(math.ceil(bbox.x_max))))
    y_max = max(y_min + 1, min(height, int(math.ceil(bbox.y_max))))
    return x_min, y_min, x_max, y_max


def _resize_crop_nearest(crop: Any, *, width: int, height: int) -> tuple[tuple[Any, ...], ...]:
    source_height = len(crop)
    source_width = len(crop[0]) if source_height else 0
    if source_height <= 0 or source_width <= 0:
        raise ValueError("pose crop must have positive dimensions")
    rows = []
    for y in range(height):
        source_y = min(source_height - 1, int(y * source_height / height))
        row = []
        for x in range(width):
            source_x = min(source_width - 1, int(x * source_width / width))
            row.append(tuple(crop[source_y][source_x]))
        rows.append(tuple(row))
    return tuple(rows)


def _black_python_frame(*, width: int, height: int, channels: int) -> list[list[tuple[float, ...]]]:
    pixel = tuple(0.0 for _ in range(channels))
    return [[pixel for _x in range(width)] for _y in range(height)]


def _freeze_python_frame(frame: Any) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    return tuple(tuple(tuple(pixel) for pixel in row) for row in frame)


def _build_alignment_frame_map(
    *,
    pose_frame_count: int,
    mask_frame_count: int,
) -> _AlignmentFrameMap:
    if pose_frame_count <= 0 or mask_frame_count <= 0:
        raise ValueError("pose_video and pose_video_mask frame counts must be positive")
    if pose_frame_count == mask_frame_count:
        return _AlignmentFrameMap(
            mode="exact",
            mask_indices=tuple(range(pose_frame_count)),
            pose_frame_count=pose_frame_count,
            mask_frame_count=mask_frame_count,
        )
    if mask_frame_count == 1:
        return _AlignmentFrameMap(
            mode="broadcast_mask",
            mask_indices=tuple(0 for _frame in range(pose_frame_count)),
            pose_frame_count=pose_frame_count,
            mask_frame_count=mask_frame_count,
        )
    raise ValueError(
        "pose_video and pose_video_mask frame counts must match, or "
        "pose_video_mask must contain exactly one broadcast frame; "
        f"got pose_frames={pose_frame_count} mask_frames={mask_frame_count}"
    )


def _alignment_transform(
    *,
    frame_index: int,
    pose_bbox: BoundingBox | None,
    mask_bbox: BoundingBox | None,
    pose_size: tuple[int, int],
    mask_size: tuple[int, int],
) -> AlignmentTransform:
    if pose_bbox is None:
        return AlignmentTransform(
            frame_index=frame_index,
            pose_bbox=None,
            target_bbox=None,
            scale_x=None,
            scale_y=None,
            translate_x=None,
            translate_y=None,
            reason="missing_pose",
        )
    if mask_bbox is None:
        return AlignmentTransform(
            frame_index=frame_index,
            pose_bbox=pose_bbox,
            target_bbox=None,
            scale_x=None,
            scale_y=None,
            translate_x=None,
            translate_y=None,
            reason="missing_mask",
        )
    if pose_bbox.width <= 0.0 or pose_bbox.height <= 0.0:
        return AlignmentTransform(
            frame_index=frame_index,
            pose_bbox=pose_bbox,
            target_bbox=None,
            scale_x=None,
            scale_y=None,
            translate_x=None,
            translate_y=None,
            reason="invalid_pose_bbox",
        )
    target_bbox = _target_mask_bbox_in_pose_space(
        mask_bbox=mask_bbox,
        mask_size=mask_size,
        pose_size=pose_size,
    )
    if target_bbox.width <= 0.0 or target_bbox.height <= 0.0:
        return AlignmentTransform(
            frame_index=frame_index,
            pose_bbox=pose_bbox,
            target_bbox=target_bbox,
            scale_x=None,
            scale_y=None,
            translate_x=None,
            translate_y=None,
            reason="invalid_target_bbox",
        )
    scale_x = target_bbox.width / pose_bbox.width
    scale_y = target_bbox.height / pose_bbox.height
    return AlignmentTransform(
        frame_index=frame_index,
        pose_bbox=pose_bbox,
        target_bbox=target_bbox,
        scale_x=scale_x,
        scale_y=scale_y,
        translate_x=target_bbox.x_min - pose_bbox.x_min * scale_x,
        translate_y=target_bbox.y_min - pose_bbox.y_min * scale_y,
        reason="ok",
    )


def _build_alignment_transforms(
    *,
    pose_boxes: tuple[BoundingBox | None, ...],
    mask_boxes: tuple[BoundingBox | None, ...],
    pose_size: tuple[int, int],
    mask_size: tuple[int, int],
    frame_map: _AlignmentFrameMap,
) -> tuple[AlignmentTransform, ...]:
    transforms = []
    for frame_index in range(frame_map.pose_frame_count):
        pose_bbox = pose_boxes[frame_index]
        mask_bbox = mask_boxes[frame_map.mask_indices[frame_index]]
        transforms.append(
            _alignment_transform(
                frame_index=frame_index,
                pose_bbox=pose_bbox,
                mask_bbox=mask_bbox,
                pose_size=pose_size,
                mask_size=mask_size,
            )
        )
    return tuple(transforms)


def _center_distance(a: BoundingBox, b: BoundingBox) -> float:
    dx = a.center_x - b.center_x
    dy = a.center_y - b.center_y
    return (dx * dx + dy * dy) ** 0.5


def _ratio_distance(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a <= 0.0 or b <= 0.0:
        return None
    ratio = a / b
    return ratio if ratio >= 1.0 else 1.0 / ratio


def _scale_distance(a: AlignmentTransform, b: AlignmentTransform) -> float | None:
    ratio_x = _ratio_distance(a.scale_x, b.scale_x)
    ratio_y = _ratio_distance(a.scale_y, b.scale_y)
    values = [value for value in (ratio_x, ratio_y) if value is not None]
    return max(values) if values else None


def _max_metric(values: list[tuple[int, float]]) -> tuple[float | None, int | None]:
    if not values:
        return None, None
    frame_index, value = max(values, key=lambda item: item[1])
    return value, frame_index


def _diagnose_transform_jitter(
    transforms: tuple[AlignmentTransform, ...],
) -> AlignmentTemporalDiagnostic:
    valid_count = sum(1 for transform in transforms if transform.valid)
    center_jumps = []
    scale_jumps = []
    for previous, current in zip(transforms, transforms[1:]):
        if not previous.valid or not current.valid:
            continue
        center_jumps.append(
            (
                current.frame_index,
                _center_distance(previous.target_bbox, current.target_bbox),
            )
        )
        scale_jump = _scale_distance(previous, current)
        if scale_jump is not None:
            scale_jumps.append((current.frame_index, scale_jump))

    center_impulses = []
    scale_impulses = []
    for previous, current, following in zip(transforms, transforms[1:], transforms[2:]):
        if not previous.valid or not current.valid or not following.valid:
            continue
        expected_center = BoundingBox(
            x_min=(previous.target_bbox.center_x + following.target_bbox.center_x) / 2.0,
            y_min=(previous.target_bbox.center_y + following.target_bbox.center_y) / 2.0,
            x_max=(previous.target_bbox.center_x + following.target_bbox.center_x) / 2.0,
            y_max=(previous.target_bbox.center_y + following.target_bbox.center_y) / 2.0,
        )
        center_impulses.append(
            (
                current.frame_index,
                _center_distance(current.target_bbox, expected_center),
            )
        )
        expected_scale_x = (previous.scale_x + following.scale_x) / 2.0
        expected_scale_y = (previous.scale_y + following.scale_y) / 2.0
        scale_x_impulse = _ratio_distance(current.scale_x, expected_scale_x)
        scale_y_impulse = _ratio_distance(current.scale_y, expected_scale_y)
        scale_values = [
            value
            for value in (scale_x_impulse, scale_y_impulse)
            if value is not None
        ]
        if scale_values:
            scale_impulses.append((current.frame_index, max(scale_values)))

    max_center_jump, worst_center_jump = _max_metric(center_jumps)
    max_center_impulse, worst_center_impulse = _max_metric(center_impulses)
    max_scale_jump, worst_scale_jump = _max_metric(scale_jumps)
    max_scale_impulse, worst_scale_impulse = _max_metric(scale_impulses)
    return AlignmentTemporalDiagnostic(
        frame_count=len(transforms),
        valid_transform_count=valid_count,
        invalid_transform_count=len(transforms) - valid_count,
        max_center_jump_px=max_center_jump,
        worst_center_jump_frame_index=worst_center_jump,
        max_center_impulse_px=max_center_impulse,
        worst_center_impulse_frame_index=worst_center_impulse,
        max_scale_jump_ratio=max_scale_jump,
        worst_scale_jump_frame_index=worst_scale_jump,
        max_scale_impulse_ratio=max_scale_impulse,
        worst_scale_impulse_frame_index=worst_scale_impulse,
        transforms=transforms,
    )


def diagnose_alignment_temporal_jitter(
    *,
    pose_video: Any,
    pose_video_mask: Any,
    target_width: Any | None = None,
    target_height: Any | None = None,
) -> AlignmentTemporalDiagnostic:
    pose_size = frame_size(pose_video, kind="pose_image")
    mask_size = frame_size(pose_video_mask, kind="semantic_rgb_mask")
    if (target_width is None) != (target_height is None):
        raise ValueError("target_width and target_height must be provided together")
    if target_width is not None:
        if int(target_width) <= 0 or int(target_height) <= 0:
            raise ValueError("target_width and target_height must be positive")
    pose_boxes = frame_bboxes(pose_video, kind="pose_image")
    mask_boxes = frame_bboxes(pose_video_mask, kind="semantic_rgb_mask")
    frame_map = _build_alignment_frame_map(
        pose_frame_count=len(pose_boxes),
        mask_frame_count=len(mask_boxes),
    )
    transforms = _build_alignment_transforms(
        pose_boxes=pose_boxes,
        mask_boxes=mask_boxes,
        pose_size=pose_size,
        mask_size=mask_size,
        frame_map=frame_map,
    )
    return _diagnose_transform_jitter(transforms)


def _align_python_pose_video(
    *,
    pose_video: Any,
    pose_boxes: tuple[BoundingBox | None, ...],
    mask_boxes: tuple[BoundingBox | None, ...],
    pose_size: tuple[int, int],
    mask_size: tuple[int, int],
    frame_map: _AlignmentFrameMap,
) -> tuple[Any, ...]:
    frames = _split_image_frames(pose_video)
    pose_height, pose_width = pose_size
    output_frames = []
    for frame_index, frame in enumerate(frames):
        pose_bbox = pose_boxes[frame_index]
        mask_bbox = mask_boxes[frame_map.mask_indices[frame_index]]
        if pose_bbox is None or mask_bbox is None:
            output_frames.append(_freeze_python_frame(frame))
            continue
        sx0, sy0, sx1, sy1 = _int_bbox(pose_bbox, width=pose_width, height=pose_height)
        target_bbox = _target_mask_bbox_in_pose_space(
            mask_bbox=mask_bbox,
            mask_size=mask_size,
            pose_size=pose_size,
        )
        dx0, dy0, dx1, dy1 = _int_bbox(target_bbox, width=pose_width, height=pose_height)
        crop = tuple(tuple(row[sx0:sx1]) for row in frame[sy0:sy1])
        resized = _resize_crop_nearest(crop, width=dx1 - dx0, height=dy1 - dy0)
        channels = len(frame[0][0])
        aligned = _black_python_frame(width=pose_width, height=pose_height, channels=channels)
        for y, row in enumerate(resized):
            for x, pixel in enumerate(row):
                aligned[dy0 + y][dx0 + x] = tuple(pixel)
        output_frames.append(tuple(tuple(row) for row in aligned))
    return tuple(output_frames)


def _align_tensor_pose_video(
    *,
    pose_video: Any,
    pose_boxes: tuple[BoundingBox | None, ...],
    mask_boxes: tuple[BoundingBox | None, ...],
    pose_size: tuple[int, int],
    mask_size: tuple[int, int],
    frame_map: _AlignmentFrameMap,
) -> Any:
    torch = _torch_or_none()
    if torch is None:
        raise RuntimeError("torch is required for tensor pose alignment")
    import torch.nn.functional as F

    single_frame = pose_video.ndim == 3
    view = pose_video.unsqueeze(0) if single_frame else pose_video
    if view.ndim != 4 or view.shape[-1] < 3:
        raise ValueError("pose_video tensor must have shape [H, W, C] or [T, H, W, C]")
    pose_height, pose_width = pose_size
    output = torch.zeros_like(view)
    for frame_index, frame in enumerate(view):
        pose_bbox = pose_boxes[frame_index]
        mask_bbox = mask_boxes[frame_map.mask_indices[frame_index]]
        if pose_bbox is None or mask_bbox is None:
            output[frame_index] = frame
            continue
        sx0, sy0, sx1, sy1 = _int_bbox(pose_bbox, width=pose_width, height=pose_height)
        target_bbox = _target_mask_bbox_in_pose_space(
            mask_bbox=mask_bbox,
            mask_size=mask_size,
            pose_size=pose_size,
        )
        dx0, dy0, dx1, dy1 = _int_bbox(target_bbox, width=pose_width, height=pose_height)
        crop = frame[sy0:sy1, sx0:sx1, :]
        resized = F.interpolate(
            crop.permute(2, 0, 1).unsqueeze(0),
            size=(dy1 - dy0, dx1 - dx0),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        output[frame_index, dy0:dy1, dx0:dx1, :] = resized.to(dtype=output.dtype)
    return output.squeeze(0) if single_frame else output


def _summary(
    before: GeometryDiagnostic,
    after: GeometryDiagnostic,
    frame_map: _AlignmentFrameMap,
    temporal: AlignmentTemporalDiagnostic | None,
) -> str:
    def fmt(value: float | None) -> str:
        return "None" if value is None else str(float(value))

    return (
        "pose_mask_alignment "
        f"before_status={before.status} after_status={after.status} "
        f"frame_map={frame_map.mode} "
        f"pose_frames={frame_map.pose_frame_count} mask_frames={frame_map.mask_frame_count} "
        f"frames={after.frame_count} compared={after.compared_frames} "
        f"missing_pose={after.missing_pose_frames} missing_mask={after.missing_mask_frames} "
        f"before_mean_iou={before.mean_iou} after_mean_iou={after.mean_iou} "
        f"after_center_delta_px={after.mean_center_delta_px} "
        f"before_center_path_error_px={before.mean_center_path_error_px} "
        f"after_center_path_error_px={after.mean_center_path_error_px} "
        f"temporal_valid_transforms={temporal.valid_transform_count if temporal else None} "
        f"temporal_invalid_transforms={temporal.invalid_transform_count if temporal else None} "
        f"max_center_jump_px={fmt(temporal.max_center_jump_px if temporal else None)} "
        f"worst_center_jump_frame={temporal.worst_center_jump_frame_index if temporal else None} "
        f"max_center_impulse_px={fmt(temporal.max_center_impulse_px if temporal else None)} "
        f"worst_center_impulse_frame={temporal.worst_center_impulse_frame_index if temporal else None} "
        f"max_scale_jump_ratio={fmt(temporal.max_scale_jump_ratio if temporal else None)} "
        f"worst_scale_jump_frame={temporal.worst_scale_jump_frame_index if temporal else None} "
        f"max_scale_impulse_ratio={fmt(temporal.max_scale_impulse_ratio if temporal else None)} "
        f"worst_scale_impulse_frame={temporal.worst_scale_impulse_frame_index if temporal else None}"
    )


def align_pose_video_to_mask(
    *,
    pose_video: Any,
    pose_video_mask: Any,
    target_width: Any | None = None,
    target_height: Any | None = None,
) -> PoseMaskAlignmentResult:
    pose_size = frame_size(pose_video, kind="pose_image")
    mask_size = frame_size(pose_video_mask, kind="semantic_rgb_mask")
    if target_width is None and target_height is None:
        height, width = mask_size
    elif target_width is None or target_height is None:
        raise ValueError("target_width and target_height must be provided together")
    else:
        width = int(target_width)
        height = int(target_height)
        if width <= 0 or height <= 0:
            raise ValueError("target_width and target_height must be positive")
    pose_boxes = frame_bboxes(pose_video, kind="pose_image")
    mask_boxes = frame_bboxes(pose_video_mask, kind="semantic_rgb_mask")
    frame_map = _build_alignment_frame_map(
        pose_frame_count=len(pose_boxes),
        mask_frame_count=len(mask_boxes),
    )
    transforms = _build_alignment_transforms(
        pose_boxes=pose_boxes,
        mask_boxes=mask_boxes,
        pose_size=pose_size,
        mask_size=mask_size,
        frame_map=frame_map,
    )
    temporal = _diagnose_transform_jitter(transforms)
    before = diagnose_pose_mask_geometry(
        pose_video=pose_video,
        pose_video_mask=pose_video_mask,
        target_width=width,
        target_height=height,
    )

    if _is_torch_tensor(pose_video):
        aligned = _align_tensor_pose_video(
            pose_video=pose_video,
            pose_boxes=pose_boxes,
            mask_boxes=mask_boxes,
            pose_size=pose_size,
            mask_size=mask_size,
            frame_map=frame_map,
        )
    else:
        aligned = _align_python_pose_video(
            pose_video=pose_video,
            pose_boxes=pose_boxes,
            mask_boxes=mask_boxes,
            pose_size=pose_size,
            mask_size=mask_size,
            frame_map=frame_map,
        )

    after = diagnose_pose_mask_geometry(
        pose_video=aligned,
        pose_video_mask=pose_video_mask,
        target_width=width,
        target_height=height,
    )
    return PoseMaskAlignmentResult(
        pose_video=aligned,
        summary=_summary(before, after, frame_map, temporal),
        before=before,
        after=after,
        temporal=temporal,
    )
