"""Pose/mask geometry diagnostics for SCAIL-Pose2 workflows."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Literal


GeometryKind = Literal["pose_image", "semantic_rgb_mask", "mask"]


@dataclass(frozen=True)
class BoundingBox:
    """Pixel bbox using half-open bounds: x_min/y_min inclusive, x_max/y_max exclusive."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    def scale(self, *, scale_x: float, scale_y: float) -> "BoundingBox":
        return BoundingBox(
            x_min=self.x_min * scale_x,
            y_min=self.y_min * scale_y,
            x_max=self.x_max * scale_x,
            y_max=self.y_max * scale_y,
        )

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


@dataclass(frozen=True)
class FrameGeometryComparison:
    frame_index: int
    pose_bbox: BoundingBox
    mask_bbox: BoundingBox
    iou: float
    center_delta_px: float
    width_ratio: float
    height_ratio: float


@dataclass(frozen=True)
class GeometryIssue:
    code: str
    frame_index: int | None
    value: float | str
    threshold: float | str

    def to_summary(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "frame_index": self.frame_index,
            "value": self.value,
            "threshold": self.threshold,
        }

    def format(self) -> str:
        value = f"{self.value:.4f}" if isinstance(self.value, float) else str(self.value)
        threshold = (
            f"{self.threshold:.4f}"
            if isinstance(self.threshold, float)
            else str(self.threshold)
        )
        return (
            f"{self.code}(frame={self.frame_index}, "
            f"value={value}, threshold={threshold})"
        )


@dataclass(frozen=True)
class GeometryDiagnostic:
    status: str
    frame_count: int
    compared_frames: int
    missing_pose_frames: int
    missing_mask_frames: int
    pose_size: tuple[int, int] | None
    mask_size: tuple[int, int] | None
    target_size: tuple[int, int]
    comparisons: tuple[FrameGeometryComparison, ...]

    @property
    def mean_iou(self) -> float | None:
        return _mean_or_none(comparison.iou for comparison in self.comparisons)

    @property
    def min_iou(self) -> float | None:
        values = [comparison.iou for comparison in self.comparisons]
        return min(values) if values else None

    @property
    def worst_iou_frame_index(self) -> int | None:
        comparison = _comparison_with_min(self.comparisons, "iou")
        return comparison.frame_index if comparison is not None else None

    @property
    def mean_center_delta_px(self) -> float | None:
        return _mean_or_none(
            comparison.center_delta_px for comparison in self.comparisons
        )

    @property
    def max_center_delta_px(self) -> float | None:
        values = [comparison.center_delta_px for comparison in self.comparisons]
        return max(values) if values else None

    @property
    def worst_center_delta_frame_index(self) -> int | None:
        comparison = _comparison_with_max(self.comparisons, "center_delta_px")
        return comparison.frame_index if comparison is not None else None

    @property
    def mean_width_ratio(self) -> float | None:
        return _mean_or_none(comparison.width_ratio for comparison in self.comparisons)

    @property
    def min_width_ratio(self) -> float | None:
        values = [comparison.width_ratio for comparison in self.comparisons]
        return min(values) if values else None

    @property
    def max_width_ratio(self) -> float | None:
        values = [comparison.width_ratio for comparison in self.comparisons]
        return max(values) if values else None

    @property
    def mean_height_ratio(self) -> float | None:
        return _mean_or_none(comparison.height_ratio for comparison in self.comparisons)

    @property
    def min_height_ratio(self) -> float | None:
        values = [comparison.height_ratio for comparison in self.comparisons]
        return min(values) if values else None

    @property
    def max_height_ratio(self) -> float | None:
        values = [comparison.height_ratio for comparison in self.comparisons]
        return max(values) if values else None

    @property
    def mean_center_path_error_px(self) -> float | None:
        return _mean_or_none(_center_path_errors(self.comparisons))

    @property
    def max_center_path_error_px(self) -> float | None:
        values = _center_path_errors(self.comparisons)
        return max(values) if values else None

    def to_summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "frame_count": self.frame_count,
            "compared_frames": self.compared_frames,
            "missing_pose_frames": self.missing_pose_frames,
            "missing_mask_frames": self.missing_mask_frames,
            "pose_size": list(self.pose_size) if self.pose_size is not None else None,
            "mask_size": list(self.mask_size) if self.mask_size is not None else None,
            "target_size": list(self.target_size),
            "mean_iou": self.mean_iou,
            "min_iou": self.min_iou,
            "worst_iou_frame_index": self.worst_iou_frame_index,
            "mean_center_delta_px": self.mean_center_delta_px,
            "max_center_delta_px": self.max_center_delta_px,
            "worst_center_delta_frame_index": self.worst_center_delta_frame_index,
            "mean_width_ratio": self.mean_width_ratio,
            "min_width_ratio": self.min_width_ratio,
            "max_width_ratio": self.max_width_ratio,
            "mean_height_ratio": self.mean_height_ratio,
            "min_height_ratio": self.min_height_ratio,
            "max_height_ratio": self.max_height_ratio,
            "mean_center_path_error_px": self.mean_center_path_error_px,
            "max_center_path_error_px": self.max_center_path_error_px,
        }


def _comparison_with_min(
    comparisons: tuple[FrameGeometryComparison, ...],
    field: str,
) -> FrameGeometryComparison | None:
    if not comparisons:
        return None
    return min(comparisons, key=lambda comparison: float(getattr(comparison, field)))


def _comparison_with_max(
    comparisons: tuple[FrameGeometryComparison, ...],
    field: str,
) -> FrameGeometryComparison | None:
    if not comparisons:
        return None
    return max(comparisons, key=lambda comparison: float(getattr(comparison, field)))


def _mean_or_none(values: Any) -> float | None:
    materialized = [float(value) for value in values]
    return mean(materialized) if materialized else None


def _center_path_errors(
    comparisons: tuple[FrameGeometryComparison, ...],
) -> list[float]:
    if len(comparisons) < 2:
        return []
    errors = []
    previous = comparisons[0]
    for current in comparisons[1:]:
        pose_dx = current.pose_bbox.center_x - previous.pose_bbox.center_x
        pose_dy = current.pose_bbox.center_y - previous.pose_bbox.center_y
        mask_dx = current.mask_bbox.center_x - previous.mask_bbox.center_x
        mask_dy = current.mask_bbox.center_y - previous.mask_bbox.center_y
        delta_x = pose_dx - mask_dx
        delta_y = pose_dy - mask_dy
        errors.append((delta_x * delta_x + delta_y * delta_y) ** 0.5)
        previous = current
    return errors


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


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _looks_like_hwc_image(frame: Any) -> bool:
    try:
        pixel = frame[0][0]
    except (IndexError, TypeError):
        return False
    return _is_sequence(pixel) and len(pixel) >= 3 and _is_number(pixel[0])


def _split_image_frames(value: Any) -> tuple[Any, ...]:
    raw = _as_list(value)
    if not raw:
        raise ValueError("image sequence must not be empty")
    if _looks_like_hwc_image(raw):
        return (raw,)
    if not _looks_like_hwc_image(raw[0]):
        raise ValueError("image frames must have shape [H, W, C] or [T, H, W, C]")
    return tuple(raw)


def _split_mask_frames(value: Any) -> tuple[Any, ...]:
    raw = _as_list(value)
    if not raw:
        raise ValueError("mask sequence must not be empty")
    try:
        first_cell = raw[0][0]
    except (IndexError, TypeError) as exc:
        raise ValueError("mask frames must have shape [H, W] or [T, H, W]") from exc
    if _is_number(first_cell):
        return (raw,)
    return tuple(raw)


def _tensor_frame_size(value: Any, *, image: bool) -> tuple[int, int]:
    if image:
        if value.ndim == 3:
            return (int(value.shape[0]), int(value.shape[1]))
        if value.ndim == 4:
            return (int(value.shape[1]), int(value.shape[2]))
    else:
        if value.ndim == 2:
            return (int(value.shape[0]), int(value.shape[1]))
        if value.ndim == 3:
            return (int(value.shape[1]), int(value.shape[2]))
    raise ValueError("unsupported tensor rank for geometry extraction")


def frame_size(value: Any, *, kind: GeometryKind) -> tuple[int, int]:
    """Return `(height, width)` for a geometry input."""

    if _is_torch_tensor(value):
        return _tensor_frame_size(value, image=kind != "mask")
    if kind == "mask":
        frame = _split_mask_frames(value)[0]
    else:
        frame = _split_image_frames(value)[0]
    height = len(frame)
    width = len(frame[0]) if height else 0
    if height <= 0 or width <= 0:
        raise ValueError("geometry frames must have positive spatial dimensions")
    return (height, width)


def _bbox_from_torch_mask(active_mask: Any) -> BoundingBox | None:
    torch = _torch_or_none()
    if torch is None:
        raise RuntimeError("torch is required for tensor geometry extraction")
    indices = torch.nonzero(active_mask, as_tuple=False)
    if indices.numel() == 0:
        return None
    y_min = int(indices[:, 0].min().item())
    y_max = int(indices[:, 0].max().item()) + 1
    x_min = int(indices[:, 1].min().item())
    x_max = int(indices[:, 1].max().item()) + 1
    return BoundingBox(float(x_min), float(y_min), float(x_max), float(y_max))


def _bbox_from_python_mask(active_mask: Any) -> BoundingBox | None:
    x_min: int | None = None
    y_min: int | None = None
    x_max: int | None = None
    y_max: int | None = None
    for row_index, row in enumerate(active_mask):
        for col_index, active in enumerate(row):
            if not active:
                continue
            x_min = col_index if x_min is None else min(x_min, col_index)
            y_min = row_index if y_min is None else min(y_min, row_index)
            x_max = col_index + 1 if x_max is None else max(x_max, col_index + 1)
            y_max = row_index + 1 if y_max is None else max(y_max, row_index + 1)
    if x_min is None or y_min is None or x_max is None or y_max is None:
        return None
    return BoundingBox(float(x_min), float(y_min), float(x_max), float(y_max))


def _normalize_rgb_value(value: Any) -> float:
    numeric = float(value)
    if 0.0 <= numeric <= 1.0:
        return numeric
    return numeric / 255.0


def _is_black_pixel(pixel: Any, *, threshold: float) -> bool:
    return all(_normalize_rgb_value(channel) <= threshold for channel in pixel[:3])


def _is_white_pixel(pixel: Any, *, threshold: float) -> bool:
    return all(_normalize_rgb_value(channel) >= threshold for channel in pixel[:3])


def _pose_active_torch(frame: Any, *, threshold: float) -> Any:
    rgb = frame[..., :3].float()
    if bool((rgb > 1.0).any().item()):
        rgb = rgb / 255.0
    return (rgb > threshold).any(dim=-1)


def _semantic_active_torch(
    frame: Any,
    *,
    black_threshold: float,
    white_threshold: float,
) -> Any:
    rgb = frame[..., :3].float()
    if bool((rgb > 1.0).any().item()):
        rgb = rgb / 255.0
    black = (rgb <= black_threshold).all(dim=-1)
    white = (rgb >= white_threshold).all(dim=-1)
    return ~(black | white)


def _mask_active_torch(frame: Any, *, threshold: float) -> Any:
    return frame.float() > threshold


def _pose_active_python(frame: Any, *, threshold: float) -> tuple[tuple[bool, ...], ...]:
    return tuple(
        tuple(
            any(_normalize_rgb_value(channel) > threshold for channel in pixel[:3])
            for pixel in row
        )
        for row in frame
    )


def _semantic_active_python(
    frame: Any,
    *,
    black_threshold: float,
    white_threshold: float,
) -> tuple[tuple[bool, ...], ...]:
    return tuple(
        tuple(
            not (
                _is_black_pixel(pixel, threshold=black_threshold)
                or _is_white_pixel(pixel, threshold=white_threshold)
            )
            for pixel in row
        )
        for row in frame
    )


def _mask_active_python(frame: Any, *, threshold: float) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple(float(value) > threshold for value in row) for row in frame)


def frame_bboxes(
    value: Any,
    *,
    kind: GeometryKind,
    pose_threshold: float = 0.01,
    mask_threshold: float = 0.5,
    black_threshold: float = 0.01,
    white_threshold: float = 0.99,
) -> tuple[BoundingBox | None, ...]:
    """Extract per-frame foreground bboxes from pose, semantic RGB, or mask input."""

    if _is_torch_tensor(value):
        tensor = value.detach()
        if kind == "mask":
            view = tensor.unsqueeze(0) if tensor.ndim == 2 else tensor
        else:
            view = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
        if kind != "mask" and (view.ndim != 4 or view.shape[-1] < 3):
            raise ValueError("image tensor must have shape [H, W, C] or [T, H, W, C]")
        if kind == "mask" and view.ndim != 3:
            raise ValueError("mask tensor must have shape [H, W] or [T, H, W]")
        boxes = []
        for frame in view:
            if kind == "pose_image":
                active = _pose_active_torch(frame, threshold=pose_threshold)
            elif kind == "semantic_rgb_mask":
                active = _semantic_active_torch(
                    frame,
                    black_threshold=black_threshold,
                    white_threshold=white_threshold,
                )
            else:
                active = _mask_active_torch(frame, threshold=mask_threshold)
            boxes.append(_bbox_from_torch_mask(active))
        return tuple(boxes)

    if kind == "mask":
        frames = _split_mask_frames(value)
    else:
        frames = _split_image_frames(value)
    boxes = []
    for frame in frames:
        if kind == "pose_image":
            active = _pose_active_python(frame, threshold=pose_threshold)
        elif kind == "semantic_rgb_mask":
            active = _semantic_active_python(
                frame,
                black_threshold=black_threshold,
                white_threshold=white_threshold,
            )
        else:
            active = _mask_active_python(frame, threshold=mask_threshold)
        boxes.append(_bbox_from_python_mask(active))
    return tuple(boxes)


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    x_min = max(a.x_min, b.x_min)
    y_min = max(a.y_min, b.y_min)
    x_max = min(a.x_max, b.x_max)
    y_max = min(a.y_max, b.y_max)
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    union = a.area + b.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def compare_bboxes(
    *,
    frame_index: int,
    pose_bbox: BoundingBox,
    mask_bbox: BoundingBox,
) -> FrameGeometryComparison:
    center_dx = pose_bbox.center_x - mask_bbox.center_x
    center_dy = pose_bbox.center_y - mask_bbox.center_y
    width_ratio = pose_bbox.width / mask_bbox.width if mask_bbox.width > 0 else 0.0
    height_ratio = pose_bbox.height / mask_bbox.height if mask_bbox.height > 0 else 0.0
    return FrameGeometryComparison(
        frame_index=frame_index,
        pose_bbox=pose_bbox,
        mask_bbox=mask_bbox,
        iou=bbox_iou(pose_bbox, mask_bbox),
        center_delta_px=(center_dx * center_dx + center_dy * center_dy) ** 0.5,
        width_ratio=width_ratio,
        height_ratio=height_ratio,
    )


def diagnose_pose_mask_geometry(
    *,
    pose_video: Any,
    pose_video_mask: Any,
    target_width: int,
    target_height: int,
    pose_kind: GeometryKind = "pose_image",
    mask_kind: GeometryKind = "semantic_rgb_mask",
) -> GeometryDiagnostic:
    """Compare pose foreground and driving-mask foreground in target coordinates."""

    if int(target_width) <= 0 or int(target_height) <= 0:
        raise ValueError("target_width and target_height must be positive")
    pose_size = frame_size(pose_video, kind=pose_kind)
    mask_size = frame_size(pose_video_mask, kind=mask_kind)
    pose_scale_x = int(target_width) / pose_size[1]
    pose_scale_y = int(target_height) / pose_size[0]
    mask_scale_x = int(target_width) / mask_size[1]
    mask_scale_y = int(target_height) / mask_size[0]

    pose_boxes = frame_bboxes(pose_video, kind=pose_kind)
    mask_boxes = frame_bboxes(pose_video_mask, kind=mask_kind)
    frame_count = min(len(pose_boxes), len(mask_boxes))
    comparisons = []
    missing_pose = 0
    missing_mask = 0
    for index in range(frame_count):
        pose_bbox = pose_boxes[index]
        mask_bbox = mask_boxes[index]
        if pose_bbox is None:
            missing_pose += 1
        if mask_bbox is None:
            missing_mask += 1
        if pose_bbox is None or mask_bbox is None:
            continue
        comparisons.append(
            compare_bboxes(
                frame_index=index,
                pose_bbox=pose_bbox.scale(scale_x=pose_scale_x, scale_y=pose_scale_y),
                mask_bbox=mask_bbox.scale(scale_x=mask_scale_x, scale_y=mask_scale_y),
            )
        )

    if comparisons:
        status = "ok"
    elif missing_pose and missing_mask:
        status = "empty_pose_and_mask"
    elif missing_pose:
        status = "empty_pose"
    elif missing_mask:
        status = "empty_mask"
    else:
        status = "no_comparable_frames"

    return GeometryDiagnostic(
        status=status,
        frame_count=frame_count,
        compared_frames=len(comparisons),
        missing_pose_frames=missing_pose,
        missing_mask_frames=missing_mask,
        pose_size=pose_size,
        mask_size=mask_size,
        target_size=(int(target_height), int(target_width)),
        comparisons=tuple(comparisons),
    )


def _worst_ratio_issue(
    comparisons: tuple[FrameGeometryComparison, ...],
    *,
    field: str,
    min_ratio: float,
    max_ratio: float,
) -> FrameGeometryComparison | None:
    def distance(comparison: FrameGeometryComparison) -> float:
        value = float(getattr(comparison, field))
        if value < min_ratio:
            return min_ratio - value
        if value > max_ratio:
            return value - max_ratio
        return 0.0

    candidates = [comparison for comparison in comparisons if distance(comparison) > 0.0]
    if not candidates:
        return None
    return max(candidates, key=distance)


def replacement_geometry_issues(
    diagnostic: GeometryDiagnostic,
    *,
    target_width: int,
    target_height: int,
    min_iou: float,
    max_center_delta_ratio: float,
    min_size_ratio: float,
    max_size_ratio: float,
) -> tuple[GeometryIssue, ...]:
    """Return replacement-mode geometry policy issues for a diagnostic."""

    issues: list[GeometryIssue] = []
    if diagnostic.status != "ok" or diagnostic.compared_frames <= 0:
        issues.append(
            GeometryIssue(
                code="geometry_status",
                frame_index=None,
                value=diagnostic.status,
                threshold="ok",
            )
        )
        return tuple(issues)

    worst_iou = _comparison_with_min(diagnostic.comparisons, "iou")
    if worst_iou is not None and worst_iou.iou < min_iou:
        issues.append(
            GeometryIssue(
                code="min_iou",
                frame_index=worst_iou.frame_index,
                value=worst_iou.iou,
                threshold=min_iou,
            )
        )

    target_diagonal = (int(target_width) * int(target_width) + int(target_height) * int(target_height)) ** 0.5
    worst_center = _comparison_with_max(diagnostic.comparisons, "center_delta_px")
    if worst_center is not None and target_diagonal > 0:
        center_ratio = worst_center.center_delta_px / target_diagonal
        if center_ratio > max_center_delta_ratio:
            issues.append(
                GeometryIssue(
                    code="center_delta_ratio",
                    frame_index=worst_center.frame_index,
                    value=center_ratio,
                    threshold=max_center_delta_ratio,
                )
            )

    ratio_threshold = f"{min_size_ratio:.4f}..{max_size_ratio:.4f}"
    for code, field in (
        ("width_ratio_out_of_range", "width_ratio"),
        ("height_ratio_out_of_range", "height_ratio"),
    ):
        worst_ratio = _worst_ratio_issue(
            diagnostic.comparisons,
            field=field,
            min_ratio=min_size_ratio,
            max_ratio=max_size_ratio,
        )
        if worst_ratio is None:
            continue
        issues.append(
            GeometryIssue(
                code=code,
                frame_index=worst_ratio.frame_index,
                value=float(getattr(worst_ratio, field)),
                threshold=ratio_threshold,
            )
        )

    return tuple(issues)
