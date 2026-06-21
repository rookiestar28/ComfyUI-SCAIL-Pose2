"""Replacement-mode denoise mask helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .condition import SCAIL2Condition, TYPE_SCAIL2_CONDITION
from .masks import (
    BACKGROUND_INDEX,
    MASK_OFF_THRESHOLD,
    mask_indices_shape,
    semantic_mask_indices,
    semantic_mask_indices_tensor_raw,
)


TENSOR_FAST_PATH_CHUNK_FRAMES = 16
SCAIL_POSE2_DISABLE_SAMPLES_ATTR = "scail_pose2_disable_samples"
SCAIL_POSE2_DISABLE_SAMPLES_REASON_ATTR = "scail_pose2_disable_samples_reason"
SCAIL_POSE2_CONDITION_MODE_ATTR = "scail_pose2_condition_mode"
SCAIL_POSE2_MASK_ROLE_ATTR = "scail_pose2_mask_role"
SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE = "replacement_denoise_mask"


@dataclass(frozen=True)
class ReplacementDenoiseMaskResult:
    mask: Any
    summary: str
    frame_count: int
    height: int
    width: int
    subject_ratio: float
    fast_path: str = "semantic_indices"
    input_device: str = "unknown"
    work_device: str = "cpu"
    output_device: str = "cpu"


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


def _tensor_image_frames(value: Any) -> Any:
    tensor = value.detach()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[-1] < 3:
        raise ValueError(
            "pose_video_mask tensor must have shape [frames, height, width, channels]"
        )
    if int(tensor.shape[0]) <= 0 or int(tensor.shape[1]) <= 0 or int(tensor.shape[2]) <= 0:
        raise ValueError("pose_video_mask tensor must be non-empty")
    return tensor


def _validate_tensor_shape(
    *,
    condition: SCAIL2Condition,
    tensor: Any,
) -> tuple[int, int, int]:
    frames, height, width = (int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.shape[2]))
    if frames != condition.num_frames:
        raise ValueError("pose_video_mask frame count must match condition num_frames")
    if width != condition.width:
        raise ValueError("pose_video_mask width must match condition width")
    if height != condition.height:
        raise ValueError("pose_video_mask height must match condition height")
    return frames, height, width


def _comfy_intermediate_device(torch: Any) -> Any | None:
    try:
        from comfy import model_management
    except Exception:
        return None
    try:
        device = torch.device(model_management.intermediate_device())
    except Exception:
        return None
    return device if device.type == "cuda" else None


def _preferred_work_device(torch: Any, tensor: Any) -> Any:
    tensor_device = getattr(tensor, "device", torch.device("cpu"))
    if getattr(tensor_device, "type", None) == "cuda":
        return tensor_device
    if torch.cuda.is_available():
        return _comfy_intermediate_device(torch) or torch.device("cuda")
    return torch.device("cpu")


def _off_threshold_for_rgb_chunk(torch: Any, rgb: Any) -> float:
    normalized = bool(torch.logical_and(rgb >= 0.0, rgb <= 1.0).all().item())
    if normalized:
        return MASK_OFF_THRESHOLD / 255.0
    return float(MASK_OFF_THRESHOLD)


def _subject_mask_chunk_from_rgb(torch: Any, rgb: Any) -> Any:
    off_threshold = _off_threshold_for_rgb_chunk(torch, rgb)
    return (rgb > off_threshold).any(dim=-1).to(dtype=torch.float32)


def _attach_scail_pose2_mask_metadata(
    mask: Any,
    *,
    condition: SCAIL2Condition,
    role: str,
) -> None:
    setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, condition.mode)
    setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, role)


def _build_tensor_subject_mask(
    pose_video_mask: Any,
    *,
    condition: SCAIL2Condition,
    grow_pixels: int,
    blur_pixels: int,
    invert: bool,
    work_device: Any | None = None,
) -> ReplacementDenoiseMaskResult:
    torch = _torch_required()
    tensor = _tensor_image_frames(pose_video_mask)
    frames, height, width = _validate_tensor_shape(condition=condition, tensor=tensor)
    input_device = str(tensor.device)
    selected_device = torch.device(work_device) if work_device is not None else _preferred_work_device(torch, tensor)

    output_chunks = []
    raw_subject_pixels = 0.0
    chunk_frames = max(1, min(TENSOR_FAST_PATH_CHUNK_FRAMES, frames))
    for start in range(0, frames, chunk_frames):
        end = min(start + chunk_frames, frames)
        rgb = tensor[start:end, :, :, :3].to(
            device=selected_device,
            dtype=torch.float32,
            non_blocking=True,
        )
        mask_chunk = _subject_mask_chunk_from_rgb(torch, rgb)
        raw_subject_pixels += float(mask_chunk.sum().item())
        mask_chunk = _grow_mask(mask_chunk, grow_pixels)
        mask_chunk = _blur_mask(mask_chunk, blur_pixels)
        if invert:
            mask_chunk = 1.0 - mask_chunk
        output_chunks.append(
            mask_chunk.clamp(0.0, 1.0).detach().to(device="cpu", dtype=torch.float32)
        )

    if raw_subject_pixels <= 0:
        raise ValueError("pose_video_mask contains no subject pixels")

    mask = torch.cat(output_chunks, dim=0).contiguous()
    _attach_scail_pose2_mask_metadata(
        mask,
        condition=condition,
        role=SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE,
    )
    subject_ratio = float(mask.mean().item())
    summary = _replacement_summary(
        condition=condition,
        frame_count=frames,
        height=height,
        width=width,
        subject_ratio=subject_ratio,
        grow_pixels=grow_pixels,
        blur_pixels=blur_pixels,
        invert=invert,
        fast_path="tensor_subject_mask",
        input_device=input_device,
        work_device=str(selected_device),
        output_device=str(mask.device),
    )
    return ReplacementDenoiseMaskResult(
        mask=mask,
        summary=summary,
        frame_count=frames,
        height=height,
        width=width,
        subject_ratio=subject_ratio,
        fast_path="tensor_subject_mask",
        input_device=input_device,
        work_device=str(selected_device),
        output_device=str(mask.device),
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


def _replacement_summary(
    *,
    condition: SCAIL2Condition,
    frame_count: int,
    height: int,
    width: int,
    subject_ratio: float,
    grow_pixels: int,
    blur_pixels: int,
    invert: bool,
    fast_path: str,
    input_device: str,
    work_device: str,
    output_device: str,
) -> str:
    return (
        "replacement_denoise_mask "
        f"mode={condition.mode} "
        f"frames={frame_count} "
        f"size={width}x{height} "
        f"subject_ratio={subject_ratio:.6f} "
        f"grow_pixels={grow_pixels} "
        f"blur_pixels={blur_pixels} "
        f"invert={bool(invert)} "
        f"fast_path={fast_path} "
        f"input_device={input_device} "
        f"work_device={work_device} "
        f"output_device={output_device}"
    )


def _build_mode_passthrough_mask(
    condition: SCAIL2Condition,
    *,
    grow_pixels: int,
    blur_pixels: int,
) -> ReplacementDenoiseMaskResult:
    torch = _torch_required()
    frame_count = int(condition.num_frames)
    height = int(condition.height)
    width = int(condition.width)
    if frame_count <= 0 or height <= 0 or width <= 0:
        raise ValueError("condition dimensions must be positive")

    mask = torch.ones((frame_count, height, width), dtype=torch.float32).contiguous()
    setattr(mask, SCAIL_POSE2_DISABLE_SAMPLES_ATTR, True)
    setattr(mask, SCAIL_POSE2_DISABLE_SAMPLES_REASON_ATTR, "non_replacement_mode")
    _attach_scail_pose2_mask_metadata(
        mask,
        condition=condition,
        role=SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE,
    )
    subject_ratio = float(mask.mean().item())
    summary = (
        _replacement_summary(
            condition=condition,
            frame_count=frame_count,
            height=height,
            width=width,
            subject_ratio=subject_ratio,
            grow_pixels=grow_pixels,
            blur_pixels=blur_pixels,
            invert=False,
            fast_path="mode_passthrough",
            input_device="condition",
            work_device=str(mask.device),
            output_device=str(mask.device),
        )
        + " background_lock=disabled samples_path=disabled reason=non_replacement_mode"
    )
    return ReplacementDenoiseMaskResult(
        mask=mask,
        summary=summary,
        frame_count=frame_count,
        height=height,
        width=width,
        subject_ratio=subject_ratio,
        fast_path="mode_passthrough",
        input_device="condition",
        work_device=str(mask.device),
        output_device=str(mask.device),
    )


def build_replacement_denoise_mask(
    *,
    condition: Any,
    pose_video_mask: Any,
    grow_pixels: Any = 0,
    blur_pixels: Any = 0,
    strict_replacement_mode: bool = False,
    invert: bool = False,
) -> ReplacementDenoiseMaskResult:
    """Build a ComfyUI MASK for sampler-side replacement denoising.

    The mask polarity follows ComfyUI/WanVideoWrapper inpaint convention:
    subject pixels are 1.0 and may be denoised, while background pixels are 0.0
    and should be preserved by the downstream sampler `samples/noise_mask` path.
    For non-replacement SCAIL-2 modes, the already-connected mask path is kept
    valid but converted to a full-denoise passthrough mask so animation workflows
    do not accidentally preserve the original video background.
    """

    valid_condition = _validate_condition(condition)
    if strict_replacement_mode and valid_condition.mode != "replacement":
        raise ValueError("replacement denoise mask requires replacement mode")

    grow = _positive_or_zero_int("grow_pixels", grow_pixels)
    blur = _positive_or_zero_int("blur_pixels", blur_pixels)
    if valid_condition.mode != "replacement":
        return _build_mode_passthrough_mask(
            valid_condition,
            grow_pixels=grow,
            blur_pixels=blur,
        )
    if _is_torch_tensor(pose_video_mask):
        try:
            return _build_tensor_subject_mask(
                pose_video_mask,
                condition=valid_condition,
                grow_pixels=grow,
                blur_pixels=blur,
                invert=bool(invert),
            )
        except RuntimeError as exc:
            if "cuda" not in str(exc).lower():
                raise
            return _build_tensor_subject_mask(
                pose_video_mask,
                condition=valid_condition,
                grow_pixels=grow,
                blur_pixels=blur,
                invert=bool(invert),
                work_device="cpu",
            )

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
    _attach_scail_pose2_mask_metadata(
        mask,
        condition=valid_condition,
        role=SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE,
    )

    frame_count, height, width = (int(part) for part in mask.shape)
    subject_ratio = float(mask.mean().item())
    summary = _replacement_summary(
        condition=valid_condition,
        frame_count=frame_count,
        height=height,
        width=width,
        subject_ratio=subject_ratio,
        grow_pixels=grow,
        blur_pixels=blur,
        invert=bool(invert),
        fast_path="semantic_indices",
        input_device="python",
        work_device=str(mask.device),
        output_device=str(mask.device),
    )
    return ReplacementDenoiseMaskResult(
        mask=mask,
        summary=summary,
        frame_count=frame_count,
        height=height,
        width=width,
        subject_ratio=subject_ratio,
        fast_path="semantic_indices",
        input_device="python",
        work_device=str(mask.device),
        output_device=str(mask.device),
    )
