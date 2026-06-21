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


@dataclass(frozen=True)
class _AlignmentFrameMap:
    mode: str
    mask_indices: tuple[int, ...]
    pose_frame_count: int
    mask_frame_count: int


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
) -> str:
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
        f"after_center_path_error_px={after.mean_center_path_error_px}"
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
        summary=_summary(before, after, frame_map),
        before=before,
        after=after,
    )
