"""Reference-image geometry alignment helpers for replacement workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Literal

from .geometry import BoundingBox, frame_bboxes, frame_size


FitMode = Literal["contain", "cover", "fit_height", "fit_width"]
AnchorMode = Literal["bottom_center", "center"]
TargetFramePolicy = Literal["median_bbox", "first_valid", "largest"]

SCAIL_POSE2_REFERENCE_GEOMETRY_ALIGNED_ATTR = (
    "scail_pose2_reference_geometry_aligned"
)
SCAIL_POSE2_REFERENCE_GEOMETRY_SUMMARY_ATTR = (
    "scail_pose2_reference_geometry_summary"
)


@dataclass(frozen=True)
class ReferenceGeometryAlignmentResult:
    ref_image: Any
    ref_mask: Any
    summary: str
    source_bbox: BoundingBox
    target_bbox: BoundingBox
    placed_bbox: BoundingBox
    scale: float


def _torch_required() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError("torch is required for reference geometry alignment") from exc
    return torch


def _as_bhwc_tensor(value: Any, *, name: str) -> Any:
    torch = _torch_required()
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or int(tensor.shape[-1]) < 3:
        raise ValueError(f"{name} must have shape [H, W, C] or [T, H, W, C]")
    if int(tensor.shape[0]) <= 0 or int(tensor.shape[1]) <= 0 or int(tensor.shape[2]) <= 0:
        raise ValueError(f"{name} must be non-empty")
    tensor = tensor[..., :3].to(dtype=torch.float32)
    if bool((tensor > 1.0).any().item()):
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0).contiguous()


def _foreground_mask_from_rgb(rgb: Any) -> Any:
    torch = _torch_required()
    if bool((rgb > 1.0).any().item()):
        rgb = rgb / 255.0
    black = (rgb <= 0.01).all(dim=-1)
    white = (rgb >= 0.99).all(dim=-1)
    return ~(black | white)


def _foreground_area_ratio(value: Any, *, kind: str) -> float:
    tensor = _as_bhwc_tensor(value, name=kind)
    foreground = _foreground_mask_from_rgb(tensor[..., :3])
    return float(foreground.to(dtype=tensor.dtype).mean().item())


def _require_valid_mask_area(
    value: Any,
    *,
    name: str,
    min_mask_area_ratio: float,
) -> None:
    if min_mask_area_ratio <= 0.0:
        return
    area_ratio = _foreground_area_ratio(value, kind=name)
    if area_ratio < float(min_mask_area_ratio):
        raise ValueError(
            f"{name} foreground area ratio {area_ratio:.6f} is below "
            f"min_mask_area_ratio {float(min_mask_area_ratio):.6f}"
        )


def _valid_bboxes(value: Any, *, name: str) -> tuple[BoundingBox, ...]:
    boxes = tuple(
        box
        for box in frame_bboxes(value, kind="semantic_rgb_mask")
        if box is not None and box.area > 0.0
    )
    if not boxes:
        raise ValueError(f"{name} contains no foreground pixels")
    return boxes


def _select_target_bbox(
    boxes: tuple[BoundingBox, ...],
    *,
    policy: TargetFramePolicy,
) -> BoundingBox:
    if policy == "first_valid":
        return boxes[0]
    if policy == "largest":
        return max(boxes, key=lambda box: box.area)
    if policy != "median_bbox":
        raise ValueError("target_frame_policy must be one of median_bbox, first_valid, largest")
    return BoundingBox(
        x_min=float(median(box.x_min for box in boxes)),
        y_min=float(median(box.y_min for box in boxes)),
        x_max=float(median(box.x_max for box in boxes)),
        y_max=float(median(box.y_max for box in boxes)),
    )


def _expand_bbox(
    bbox: BoundingBox,
    *,
    margin: int,
    width: int,
    height: int,
) -> BoundingBox:
    margin_value = max(int(margin), 0)
    return BoundingBox(
        x_min=max(0.0, bbox.x_min - margin_value),
        y_min=max(0.0, bbox.y_min - margin_value),
        x_max=min(float(width), bbox.x_max + margin_value),
        y_max=min(float(height), bbox.y_max + margin_value),
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


def _fit_scale(
    *,
    source_bbox: BoundingBox,
    target_bbox: BoundingBox,
    fit_mode: FitMode,
    max_scale: float,
) -> float:
    if source_bbox.width <= 0.0 or source_bbox.height <= 0.0:
        raise ValueError("source bbox must have positive area")
    scale_x = target_bbox.width / source_bbox.width
    scale_y = target_bbox.height / source_bbox.height
    if fit_mode == "contain":
        scale = min(scale_x, scale_y)
    elif fit_mode == "cover":
        scale = max(scale_x, scale_y)
    elif fit_mode == "fit_height":
        scale = scale_y
    elif fit_mode == "fit_width":
        scale = scale_x
    else:
        raise ValueError("fit_mode must be one of contain, cover, fit_height, fit_width")
    if max_scale <= 0.0:
        raise ValueError("max_scale must be positive")
    return max(min(float(scale), float(max_scale)), 1e-6)


def _placement_origin(
    *,
    target_bbox: BoundingBox,
    scaled_width: int,
    scaled_height: int,
    anchor: AnchorMode,
) -> tuple[int, int]:
    if anchor == "bottom_center":
        x0 = int(round(target_bbox.center_x - scaled_width / 2.0))
        y0 = int(round(target_bbox.y_max - scaled_height))
        return x0, y0
    if anchor == "center":
        x0 = int(round(target_bbox.center_x - scaled_width / 2.0))
        y0 = int(round(target_bbox.center_y - scaled_height / 2.0))
        return x0, y0
    raise ValueError("anchor must be one of bottom_center, center")


def _background_rgb(mask: Any) -> Any:
    torch = _torch_required()
    rgb = mask[..., :3]
    black = (rgb <= 0.01).all(dim=-1)
    white = (rgb >= 0.99).all(dim=-1)
    white_count = int(white.sum().item())
    black_count = int(black.sum().item())
    value = 1.0 if white_count >= black_count else 0.0
    return torch.full((3,), value, dtype=mask.dtype, device=mask.device)


def _paste(
    canvas: Any,
    crop: Any,
    *,
    x0: int,
    y0: int,
) -> tuple[Any, BoundingBox]:
    crop_h = int(crop.shape[0])
    crop_w = int(crop.shape[1])
    canvas_h = int(canvas.shape[0])
    canvas_w = int(canvas.shape[1])
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(canvas_w, x0 + crop_w)
    dst_y1 = min(canvas_h, y0 + crop_h)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        raise ValueError("aligned reference crop falls outside target canvas")
    src_x0 = dst_x0 - x0
    src_y0 = dst_y0 - y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1, :] = crop[src_y0:src_y1, src_x0:src_x1, :]
    return canvas, BoundingBox(float(dst_x0), float(dst_y0), float(dst_x1), float(dst_y1))


def _resize_crop(crop: Any, *, height: int, width: int, mode: str) -> Any:
    import torch.nn.functional as F

    kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return (
        F.interpolate(
            crop.permute(2, 0, 1).unsqueeze(0),
            size=(height, width),
            **kwargs,
        )
        .squeeze(0)
        .permute(1, 2, 0)
        .contiguous()
    )


def _attach_metadata(value: Any, *, summary: str) -> None:
    try:
        setattr(value, SCAIL_POSE2_REFERENCE_GEOMETRY_ALIGNED_ATTR, True)
        setattr(value, SCAIL_POSE2_REFERENCE_GEOMETRY_SUMMARY_ATTR, summary)
    except Exception:
        return


def reference_geometry_is_aligned(*values: Any) -> bool:
    return any(
        bool(getattr(value, SCAIL_POSE2_REFERENCE_GEOMETRY_ALIGNED_ATTR, False))
        for value in values
    )


def reference_geometry_summary(*values: Any) -> str | None:
    for value in values:
        summary = getattr(value, SCAIL_POSE2_REFERENCE_GEOMETRY_SUMMARY_ATTR, None)
        if summary:
            return str(summary)
    return None


def align_reference_image_geometry(
    *,
    ref_image: Any,
    ref_mask: Any,
    pose_video_mask: Any,
    fit_mode: FitMode = "contain",
    anchor: AnchorMode = "bottom_center",
    target_frame_policy: TargetFramePolicy = "median_bbox",
    bbox_margin: int = 0,
    max_scale: float = 2.0,
    min_mask_area_ratio: float = 0.0005,
) -> ReferenceGeometryAlignmentResult:
    """Align reference subject geometry to the driving mask canvas.

    The output is a target-canvas `IMAGE` pair suitable for feeding into
    `SCAIL-Pose2 SCAIL-2 Condition.ref_image/ref_mask`.
    """

    _require_valid_mask_area(
        ref_mask,
        name="ref_mask",
        min_mask_area_ratio=float(min_mask_area_ratio),
    )
    _require_valid_mask_area(
        pose_video_mask,
        name="pose_video_mask",
        min_mask_area_ratio=float(min_mask_area_ratio),
    )
    ref_image_tensor = _as_bhwc_tensor(ref_image, name="ref_image")
    ref_mask_tensor = _as_bhwc_tensor(ref_mask, name="ref_mask")
    target_height, target_width = frame_size(
        pose_video_mask,
        kind="semantic_rgb_mask",
    )
    ref_mask_height, ref_mask_width = frame_size(ref_mask, kind="semantic_rgb_mask")
    image_height = int(ref_image_tensor.shape[1])
    image_width = int(ref_image_tensor.shape[2])

    source_mask_bbox = _valid_bboxes(ref_mask, name="ref_mask")[0]
    target_bbox = _expand_bbox(
        _select_target_bbox(
            _valid_bboxes(pose_video_mask, name="pose_video_mask"),
            policy=target_frame_policy,
        ),
        margin=max(int(bbox_margin), 0),
        width=target_width,
        height=target_height,
    )
    image_bbox = source_mask_bbox.scale(
        scale_x=image_width / ref_mask_width,
        scale_y=image_height / ref_mask_height,
    )
    ix0, iy0, ix1, iy1 = _int_bbox(
        image_bbox,
        width=image_width,
        height=image_height,
    )
    mx0, my0, mx1, my1 = _int_bbox(
        source_mask_bbox,
        width=ref_mask_width,
        height=ref_mask_height,
    )

    scale = _fit_scale(
        source_bbox=source_mask_bbox,
        target_bbox=target_bbox,
        fit_mode=fit_mode,
        max_scale=float(max_scale),
    )
    scaled_width = max(1, int(round(source_mask_bbox.width * scale)))
    scaled_height = max(1, int(round(source_mask_bbox.height * scale)))
    paste_x0, paste_y0 = _placement_origin(
        target_bbox=target_bbox,
        scaled_width=scaled_width,
        scaled_height=scaled_height,
        anchor=anchor,
    )

    ref_mask_crop = ref_mask_tensor[0, my0:my1, mx0:mx1, :]
    resized_mask = _resize_crop(
        ref_mask_crop,
        height=scaled_height,
        width=scaled_width,
        mode="nearest",
    )
    alpha = _foreground_mask_from_rgb(resized_mask).to(dtype=ref_image_tensor.dtype)
    alpha = alpha.unsqueeze(-1)

    aligned_images = []
    placed_bbox: BoundingBox | None = None
    for frame in ref_image_tensor:
        image_crop = frame[iy0:iy1, ix0:ix1, :]
        resized_image = _resize_crop(
            image_crop,
            height=scaled_height,
            width=scaled_width,
            mode="bilinear",
        )
        resized_image = resized_image * alpha
        canvas = ref_image_tensor.new_zeros((target_height, target_width, 3))
        canvas, placed = _paste(canvas, resized_image, x0=paste_x0, y0=paste_y0)
        placed_bbox = placed
        aligned_images.append(canvas)

    mask_bg = _background_rgb(ref_mask_tensor[0])
    mask_canvas = mask_bg.view(1, 1, 3).expand(target_height, target_width, 3).clone()
    mask_canvas, placed = _paste(mask_canvas, resized_mask, x0=paste_x0, y0=paste_y0)
    placed_bbox = placed_bbox or placed
    aligned_image = _torch_required().stack(aligned_images, dim=0).clamp(0.0, 1.0).contiguous()
    aligned_mask = mask_canvas.unsqueeze(0).clamp(0.0, 1.0).contiguous()
    summary = (
        "reference_geometry_alignment "
        f"fit_mode={fit_mode} anchor={anchor} "
        f"target_frame_policy={target_frame_policy} "
        f"target_size={target_width}x{target_height} "
        f"ref_image_size={image_width}x{image_height} "
        f"ref_mask_size={ref_mask_width}x{ref_mask_height} "
        f"source_bbox={source_mask_bbox.to_tuple()} "
        f"target_bbox={target_bbox.to_tuple()} "
        f"placed_bbox={placed_bbox.to_tuple()} "
        f"scale={scale:.6f} "
        "fallback=none"
    )
    _attach_metadata(aligned_image, summary=summary)
    _attach_metadata(aligned_mask, summary=summary)
    return ReferenceGeometryAlignmentResult(
        ref_image=aligned_image,
        ref_mask=aligned_mask,
        summary=summary,
        source_bbox=source_mask_bbox,
        target_bbox=target_bbox,
        placed_bbox=placed_bbox,
        scale=scale,
    )
