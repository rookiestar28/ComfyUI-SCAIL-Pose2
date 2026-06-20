"""Replacement-mode denoise mask helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .condition import SCAIL2Condition, TYPE_SCAIL2_CONDITION
from .masks import (
    BACKGROUND_INDEX,
    mask_indices_shape,
    semantic_mask_indices,
    semantic_mask_indices_tensor_raw,
)


@dataclass(frozen=True)
class ReplacementDenoiseMaskResult:
    mask: Any
    summary: str
    frame_count: int
    height: int
    width: int
    subject_ratio: float


def _torch_required() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("torch is required to build ComfyUI MASK tensors") from exc
    return torch


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return isinstance(value, torch.Tensor)


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


def _looks_like_hwc_image(value: Any) -> bool:
    try:
        first_channel = value[0][0][0]
    except (IndexError, TypeError):
        return False
    return _is_number(first_channel)


def _normalize_pose_mask_indices(pose_video_mask: Any) -> Any:
    if _is_torch_tensor(pose_video_mask):
        return semantic_mask_indices_tensor_raw(pose_video_mask)

    raw = _as_list(pose_video_mask)
    if not raw:
        raise ValueError("pose_video_mask must not be empty")
    frames = (raw,) if _looks_like_hwc_image(raw) else raw
    return semantic_mask_indices(frames)


def _validate_condition(condition: Any) -> SCAIL2Condition:
    if not isinstance(condition, SCAIL2Condition):
        raise ValueError("condition must be a SCAIL2Condition")
    if condition.type_name != TYPE_SCAIL2_CONDITION:
        raise ValueError("condition must be a SCAIL2_CONDITION payload")
    return condition


def _positive_or_zero_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _indices_to_subject_tensor(indices: Any) -> Any:
    torch = _torch_required()
    if isinstance(indices, torch.Tensor):
        return (indices != BACKGROUND_INDEX).to(dtype=torch.float32)
    return (torch.tensor(indices, dtype=torch.int16) != BACKGROUND_INDEX).to(
        dtype=torch.float32
    )


def _grow_mask(mask: Any, pixels: int) -> Any:
    if pixels <= 0:
        return mask
    import torch.nn.functional as F

    kernel_size = pixels * 2 + 1
    grown = F.max_pool2d(
        mask.unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=pixels,
    )
    return grown[:, 0]


def _blur_mask(mask: Any, pixels: int) -> Any:
    if pixels <= 0:
        return mask
    import torch.nn.functional as F

    kernel_size = pixels * 2 + 1
    blurred = F.avg_pool2d(
        mask.unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=pixels,
    )
    return blurred[:, 0].clamp(0.0, 1.0)


def _validate_shape(
    *,
    condition: SCAIL2Condition,
    indices: Any,
) -> None:
    shape = mask_indices_shape(indices)
    if shape.frames != condition.num_frames:
        raise ValueError("pose_video_mask frame count must match condition num_frames")
    if shape.width != condition.width:
        raise ValueError("pose_video_mask width must match condition width")
    if shape.height != condition.height:
        raise ValueError("pose_video_mask height must match condition height")


def build_replacement_denoise_mask(
    *,
    condition: Any,
    pose_video_mask: Any,
    grow_pixels: Any = 0,
    blur_pixels: Any = 0,
    strict_replacement_mode: bool = True,
    invert: bool = False,
) -> ReplacementDenoiseMaskResult:
    """Build a ComfyUI MASK for sampler-side replacement denoising.

    The mask polarity follows ComfyUI/WanVideoWrapper inpaint convention:
    subject pixels are 1.0 and may be denoised, while background pixels are 0.0
    and should be preserved by the downstream sampler `samples/noise_mask` path.
    """

    valid_condition = _validate_condition(condition)
    if strict_replacement_mode and valid_condition.mode != "replacement":
        raise ValueError("replacement denoise mask requires replacement mode")

    grow = _positive_or_zero_int("grow_pixels", grow_pixels)
    blur = _positive_or_zero_int("blur_pixels", blur_pixels)
    indices = _normalize_pose_mask_indices(pose_video_mask)
    _validate_shape(condition=valid_condition, indices=indices)

    mask = _indices_to_subject_tensor(indices)
    raw_subject_pixels = float(mask.sum().item())
    if raw_subject_pixels <= 0:
        raise ValueError("pose_video_mask contains no subject pixels")

    mask = _grow_mask(mask, grow)
    mask = _blur_mask(mask, blur)
    if invert:
        mask = 1.0 - mask
    mask = mask.clamp(0.0, 1.0).detach().to(device="cpu", dtype=_torch_required().float32)
    mask = mask.contiguous()

    frame_count, height, width = (int(part) for part in mask.shape)
    subject_ratio = float(mask.mean().item())
    summary = (
        "replacement_denoise_mask "
        f"mode={valid_condition.mode} "
        f"frames={frame_count} "
        f"size={width}x{height} "
        f"subject_ratio={subject_ratio:.6f} "
        f"grow_pixels={grow} "
        f"blur_pixels={blur} "
        f"invert={bool(invert)}"
    )
    return ReplacementDenoiseMaskResult(
        mask=mask,
        summary=summary,
        frame_count=frame_count,
        height=height,
        width=width,
        subject_ratio=subject_ratio,
    )
