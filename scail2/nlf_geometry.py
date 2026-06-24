"""NLF bbox and render-geometry helpers for SCAIL-Pose2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .geometry import BoundingBox, frame_bboxes, frame_size
from .pose_alignment import PoseMaskAlignmentResult, align_pose_video_to_mask


@dataclass(frozen=True)
class NormalizedNLFBBoxes:
    boxes: tuple[BoundingBox | None, ...]
    source: str
    warnings: tuple[str, ...] = ()
    ambiguous: bool = False
    candidates: tuple[tuple[BoundingBox, ...], ...] = ()

    @property
    def frame_count(self) -> int:
        return len(self.boxes)

    @property
    def valid_count(self) -> int:
        return sum(1 for box in self.boxes if box is not None)

    @property
    def max_person_count(self) -> int:
        return max((len(frame) for frame in self.candidates), default=0)

    @property
    def is_multi_person(self) -> bool:
        return self.max_person_count > 1

    def summary(self) -> str:
        warning_text = ",".join(self.warnings) if self.warnings else "none"
        return (
            "nlf_bboxes "
            f"source={self.source} frames={self.frame_count} "
            f"valid={self.valid_count} max_persons={self.max_person_count} "
            f"ambiguous={self.ambiguous} "
            f"warnings={warning_text}"
        )


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


def _is_box_like(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 4 and all(
        _is_number(item) for item in value[:4]
    )


def _box_from_xyxy(value: Any) -> BoundingBox | None:
    if not _is_box_like(value):
        return None
    x0, y0, x1, y1 = (float(value[index]) for index in range(4))
    if x1 <= x0 or y1 <= y0:
        return None
    return BoundingBox(x0, y0, x1, y1)


def _normalize_frame_boxes(
    frame_value: Any,
) -> tuple[tuple[BoundingBox, ...], BoundingBox | None, bool, str | None]:
    if _is_box_like(frame_value):
        box = _box_from_xyxy(frame_value)
        return ((box,) if box is not None else ()), box, False, None
    if not isinstance(frame_value, (list, tuple)) or not frame_value:
        return (), None, False, None

    candidates = [_box_from_xyxy(candidate) for candidate in frame_value]
    valid = [box for box in candidates if box is not None]
    if not valid:
        return (), None, False, "empty_frame_boxes"
    if len(valid) > 1:
        # Multi-person bbox payloads cannot be safely paired to multi-person NLF
        # skeletons at this render seam without an explicit identity contract.
        return tuple(valid), max(valid, key=lambda box: box.area), True, "multi_person_bbox_payload"
    return tuple(valid), valid[0], False, None


def _empty_result(
    *,
    frame_count: int | None,
    source: str,
    warnings: tuple[str, ...],
) -> NormalizedNLFBBoxes:
    count = max(int(frame_count or 0), 0)
    return NormalizedNLFBBoxes(
        boxes=tuple(None for _frame in range(count)),
        candidates=tuple(tuple() for _frame in range(count)),
        source=source,
        warnings=warnings,
    )


def normalize_nlf_bboxes(
    bboxes: Any,
    *,
    frame_count: int | None = None,
) -> NormalizedNLFBBoxes:
    """Normalize known NLF bbox payloads into per-frame xyxy bounding boxes."""

    if bboxes is None:
        return _empty_result(
            frame_count=frame_count,
            source="none",
            warnings=("missing_bboxes",),
        )

    source = type(bboxes).__name__
    raw = _as_list(bboxes)
    if isinstance(raw, dict):
        for key in ("bboxes", "boxes", "bbox"):
            if key in raw:
                source = f"dict.{key}"
                raw = _as_list(raw[key])
                break

    if _is_box_like(raw):
        raw_frames = [raw]
    elif isinstance(raw, (list, tuple)):
        raw_frames = list(raw)
    else:
        return _empty_result(
            frame_count=frame_count,
            source=source,
            warnings=("unsupported_bbox_payload",),
        )

    warnings: list[str] = []
    boxes: list[BoundingBox | None] = []
    candidates: list[tuple[BoundingBox, ...]] = []
    ambiguous = False
    for frame_value in raw_frames:
        frame_candidates, box, frame_ambiguous, warning = _normalize_frame_boxes(frame_value)
        candidates.append(frame_candidates)
        boxes.append(box)
        ambiguous = ambiguous or frame_ambiguous
        if warning is not None and warning not in warnings:
            warnings.append(warning)

    if frame_count is not None:
        expected = max(int(frame_count), 0)
        if len(boxes) == 1 and expected > 1:
            boxes = boxes * expected
            candidates = candidates * expected
            warnings.append("broadcast_single_bbox")
        elif len(boxes) < expected:
            boxes.extend(None for _frame in range(expected - len(boxes)))
            candidates.extend(tuple() for _frame in range(expected - len(candidates)))
            warnings.append("padded_missing_bboxes")
        elif len(boxes) > expected:
            boxes = boxes[:expected]
            candidates = candidates[:expected]
            warnings.append("truncated_extra_bboxes")

    return NormalizedNLFBBoxes(
        boxes=tuple(boxes),
        candidates=tuple(candidates),
        source=source,
        warnings=tuple(dict.fromkeys(warnings)),
        ambiguous=ambiguous,
    )


def select_nlf_bboxes_for_identity(
    normalized: NormalizedNLFBBoxes,
    *,
    identity_index: int,
) -> NormalizedNLFBBoxes:
    """Select a stable per-person bbox stream from a normalized multi-person payload."""

    index = int(identity_index)
    warnings = list(normalized.warnings)
    if index < 0:
        raise ValueError("identity_index must be non-negative")
    selected: list[BoundingBox | None] = []
    for frame_candidates in normalized.candidates:
        if index < len(frame_candidates):
            selected.append(frame_candidates[index])
        else:
            selected.append(None)
    if any(box is None for box in selected):
        warnings.append("identity_bbox_missing_frames")
    return NormalizedNLFBBoxes(
        boxes=tuple(selected),
        candidates=tuple((box,) if box is not None else tuple() for box in selected),
        source=f"{normalized.source}.identity_{index}",
        warnings=tuple(dict.fromkeys(warnings)),
        ambiguous=False,
    )


def bbox_payload_is_safe_for_render_repair(
    normalized: NormalizedNLFBBoxes,
    *,
    width: int,
    height: int,
    tolerance_ratio: float = 0.05,
) -> tuple[bool, str]:
    if normalized.ambiguous:
        return False, "ambiguous_multi_person_bboxes"
    if normalized.valid_count <= 0:
        return False, "no_valid_bboxes"
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        return False, "invalid_canvas"
    tolerance_x = width * float(tolerance_ratio)
    tolerance_y = height * float(tolerance_ratio)
    for box in normalized.boxes:
        if box is None:
            continue
        if box.area <= 0.0:
            return False, "invalid_bbox_area"
        if (
            box.x_min < -tolerance_x
            or box.y_min < -tolerance_y
            or box.x_max > width + tolerance_x
            or box.y_max > height + tolerance_y
        ):
            return False, "bbox_coordinate_space_mismatch"
    return True, "ok"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_optional_float(value: float | None) -> str:
    return "none" if value is None else f"{value:.6f}"


def format_nlf_render_bbox_diagnostics(
    *,
    pose_video: Any,
    target_bboxes: tuple[BoundingBox | None, ...],
    target_source: str,
    width: int,
    height: int,
    fallback_reason: str = "none",
) -> str:
    """Build a compact bbox diagnostic string for node logs and command evidence."""

    width = int(width)
    height = int(height)
    canvas_area = max(float(width * height), 1.0)
    pose_boxes = frame_bboxes(pose_video, kind="pose_image")
    frame_count = min(len(pose_boxes), len(target_bboxes))
    compared = 0
    pose_coverages: list[float] = []
    target_coverages: list[float] = []
    center_offsets: list[float] = []
    missing_pose = 0
    missing_target = 0
    for index in range(frame_count):
        pose_box = pose_boxes[index]
        target_box = target_bboxes[index]
        if pose_box is None:
            missing_pose += 1
        if target_box is None:
            missing_target += 1
        if pose_box is None or target_box is None:
            continue
        compared += 1
        pose_coverages.append(pose_box.area / canvas_area)
        target_coverages.append(target_box.area / canvas_area)
        dx = pose_box.center_x - target_box.center_x
        dy = pose_box.center_y - target_box.center_y
        center_offsets.append((dx * dx + dy * dy) ** 0.5)

    return (
        "nlf_render_bbox_geometry "
        f"target_source={target_source} frames={frame_count} compared={compared} "
        f"missing_pose={missing_pose} missing_target={missing_target} "
        f"render_size={width}x{height} "
        f"mean_pose_coverage={_format_optional_float(_mean(pose_coverages))} "
        f"mean_target_coverage={_format_optional_float(_mean(target_coverages))} "
        f"mean_center_delta_px={_format_optional_float(_mean(center_offsets))} "
        f"fallback_reason={fallback_reason}"
    )


def _int_bbox(box: BoundingBox, *, width: int, height: int) -> tuple[int, int, int, int]:
    import math

    x0 = max(0, min(width - 1, int(math.floor(box.x_min))))
    y0 = max(0, min(height - 1, int(math.floor(box.y_min))))
    x1 = max(x0 + 1, min(width, int(math.ceil(box.x_max))))
    y1 = max(y0 + 1, min(height, int(math.ceil(box.y_max))))
    return x0, y0, x1, y1


def _build_python_bbox_mask(
    boxes: tuple[BoundingBox | None, ...],
    *,
    width: int,
    height: int,
) -> tuple[tuple[tuple[tuple[float, float, float], ...], ...], ...]:
    frames = []
    for box in boxes:
        frame = [[(0.0, 0.0, 0.0) for _x in range(width)] for _y in range(height)]
        if box is not None:
            x0, y0, x1, y1 = _int_bbox(box, width=width, height=height)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    frame[y][x] = (0.0, 0.0, 1.0)
        frames.append(tuple(tuple(row) for row in frame))
    return tuple(frames)


def _build_tensor_bbox_mask(
    pose_video: Any,
    boxes: tuple[BoundingBox | None, ...],
    *,
    width: int,
    height: int,
) -> Any:
    torch = _torch_or_none()
    if torch is None:
        raise RuntimeError("torch is required for tensor bbox masks")
    frame_count = len(boxes)
    device = pose_video.device if hasattr(pose_video, "device") else None
    dtype = pose_video.dtype if hasattr(pose_video, "dtype") else torch.float32
    mask = torch.zeros((frame_count, height, width, 3), dtype=dtype, device=device)
    for frame_index, box in enumerate(boxes):
        if box is None:
            continue
        x0, y0, x1, y1 = _int_bbox(box, width=width, height=height)
        mask[frame_index, y0:y1, x0:x1, 2] = 1.0
    return mask


def align_pose_video_to_bboxes(
    *,
    pose_video: Any,
    bboxes: tuple[BoundingBox | None, ...],
) -> PoseMaskAlignmentResult:
    """Align a rendered pose video to already-normalized render-space bboxes."""

    height, width = frame_size(pose_video, kind="pose_image")
    pose_boxes = frame_bboxes(pose_video, kind="pose_image")
    if len(bboxes) == 1 and len(pose_boxes) > 1:
        boxes = bboxes * len(pose_boxes)
    else:
        boxes = bboxes
    if _is_torch_tensor(pose_video):
        target_mask = _build_tensor_bbox_mask(
            pose_video,
            boxes,
            width=width,
            height=height,
        )
    else:
        target_mask = _build_python_bbox_mask(boxes, width=width, height=height)
    return align_pose_video_to_mask(
        pose_video=pose_video,
        pose_video_mask=target_mask,
        target_width=width,
        target_height=height,
    )


def resize_bhwc_video(
    video: Any,
    *,
    width: int,
    height: int,
) -> Any:
    torch = _torch_or_none()
    if torch is None or not _is_torch_tensor(video):
        raise RuntimeError("resize_bhwc_video requires a torch tensor")
    import torch.nn.functional as F

    single = video.ndim == 3
    view = video.unsqueeze(0) if single else video
    if view.ndim != 4:
        raise ValueError("video tensor must have shape [H, W, C] or [T, H, W, C]")
    resized = F.interpolate(
        view.permute(0, 3, 1, 2),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    return resized.squeeze(0) if single else resized


def resize_mask_video(
    mask: Any,
    *,
    width: int,
    height: int,
) -> Any:
    torch = _torch_or_none()
    if torch is None or not _is_torch_tensor(mask):
        raise RuntimeError("resize_mask_video requires a torch tensor")
    import torch.nn.functional as F

    single = mask.ndim == 2
    view = mask.unsqueeze(0) if single else mask
    if view.ndim != 3:
        raise ValueError("mask tensor must have shape [H, W] or [T, H, W]")
    resized = F.interpolate(
        view.unsqueeze(1).float(),
        size=(int(height), int(width)),
        mode="nearest",
    ).squeeze(1)
    return resized.squeeze(0) if single else resized.to(dtype=mask.dtype)
