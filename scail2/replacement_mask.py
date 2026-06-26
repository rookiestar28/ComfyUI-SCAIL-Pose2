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
from .replacement_diagnostics import (
    MaskDiagnostics,
    diagnostics_summary_fragment,
    mask_diagnostics,
)
from .replacement_presets import resolve_mask_preset


TENSOR_FAST_PATH_CHUNK_FRAMES = 16
DEFAULT_LOWER_CONTACT_REFINE = True
DEFAULT_LOWER_CONTACT_GROW_PIXELS = 8
DEFAULT_LOWER_CONTACT_BAND_RATIO = 0.30
DEFAULT_LOWER_CONTACT_AREA_CAP_RATIO = 0.25
SCAIL_POSE2_DISABLE_SAMPLES_ATTR = "scail_pose2_disable_samples"
SCAIL_POSE2_DISABLE_SAMPLES_REASON_ATTR = "scail_pose2_disable_samples_reason"
SCAIL_POSE2_CONDITION_MODE_ATTR = "scail_pose2_condition_mode"
SCAIL_POSE2_MASK_ROLE_ATTR = "scail_pose2_mask_role"
SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE = "replacement_denoise_mask"
LOW_SUBJECT_COVERAGE_RATIO = 0.02


@dataclass(frozen=True)
class LowerContactRefineStats:
    enabled: bool
    requested_grow_pixels: int
    band_ratio: float
    area_cap_ratio: float
    candidate_frames: int = 0
    refined_frames: int = 0
    max_applied_grow_pixels: int = 0
    base_pixels: float = 0.0
    added_pixels: float = 0.0
    max_frame_area_delta_ratio: float = 0.0

    @property
    def area_delta_ratio(self) -> float:
        if self.base_pixels <= 0.0:
            return 0.0
        return self.added_pixels / self.base_pixels


@dataclass(frozen=True)
class ReplacementCoverageStats:
    raw_subject_ratio: float
    coverage_warning: str
    empty_frame_count: int
    sparse_frame_count: int
    longest_sparse_streak: int


@dataclass(frozen=True)
class ReplacementDenoiseMaskResult:
    mask: Any
    summary: str
    frame_count: int
    height: int
    width: int
    subject_ratio: float
    raw_subject_ratio: float = 0.0
    coverage_warning: str = "none"
    coverage_empty_frame_count: int = 0
    coverage_sparse_frame_count: int = 0
    coverage_longest_sparse_streak: int = 0
    diagnostics: MaskDiagnostics | None = None
    mask_preset: str = "custom"
    fast_path: str = "semantic_indices"
    input_device: str = "unknown"
    work_device: str = "cpu"
    output_device: str = "cpu"
    lower_contact_refine: LowerContactRefineStats | None = None


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


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _float_in_range(name: str, value: Any, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


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


def _longest_true_streak(values: tuple[bool, ...]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _frame_subject_ratios(mask: Any) -> tuple[float, ...]:
    frame_count = int(mask.shape[0])
    if frame_count <= 0:
        return ()
    total_pixels = max(int(mask.shape[1]) * int(mask.shape[2]), 1)
    ratios = mask.detach().to(dtype=_torch_required().float32).reshape(frame_count, -1)
    ratios = ratios.sum(dim=1) / total_pixels
    return tuple(float(value) for value in ratios.cpu().tolist())


def _coverage_stats_from_ratios(ratios: tuple[float, ...]) -> ReplacementCoverageStats:
    if not ratios:
        return ReplacementCoverageStats(
            raw_subject_ratio=0.0,
            coverage_warning="none",
            empty_frame_count=0,
            sparse_frame_count=0,
            longest_sparse_streak=0,
        )
    raw_subject_ratio = sum(ratios) / len(ratios)
    empty = tuple(ratio <= 0.0 for ratio in ratios)
    sparse = tuple(ratio < LOW_SUBJECT_COVERAGE_RATIO for ratio in ratios)
    # IMPORTANT: grow/blur can expand existing foreground only; missing subject
    # regions stay preserved and can leak the driving person into replacement.
    warning = (
        "low_subject_coverage"
        if 0.0 < raw_subject_ratio < LOW_SUBJECT_COVERAGE_RATIO
        else "none"
    )
    return ReplacementCoverageStats(
        raw_subject_ratio=raw_subject_ratio,
        coverage_warning=warning,
        empty_frame_count=sum(1 for value in empty if value),
        sparse_frame_count=sum(1 for value in sparse if value),
        longest_sparse_streak=_longest_true_streak(sparse),
    )


def _attach_scail_pose2_mask_metadata(
    mask: Any,
    *,
    condition: SCAIL2Condition,
    role: str,
) -> None:
    setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, condition.mode)
    setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, role)


def _disabled_lower_contact_stats(
    *,
    enabled: bool,
    grow_pixels: int,
    band_ratio: float,
    area_cap_ratio: float,
) -> LowerContactRefineStats:
    return LowerContactRefineStats(
        enabled=enabled,
        requested_grow_pixels=grow_pixels,
        band_ratio=band_ratio,
        area_cap_ratio=area_cap_ratio,
    )


def _merge_lower_contact_stats(
    stats: list[LowerContactRefineStats],
) -> LowerContactRefineStats | None:
    if not stats:
        return None
    first = stats[0]
    return LowerContactRefineStats(
        enabled=any(item.enabled for item in stats),
        requested_grow_pixels=first.requested_grow_pixels,
        band_ratio=first.band_ratio,
        area_cap_ratio=first.area_cap_ratio,
        candidate_frames=sum(item.candidate_frames for item in stats),
        refined_frames=sum(item.refined_frames for item in stats),
        max_applied_grow_pixels=max(item.max_applied_grow_pixels for item in stats),
        base_pixels=sum(item.base_pixels for item in stats),
        added_pixels=sum(item.added_pixels for item in stats),
        max_frame_area_delta_ratio=max(
            item.max_frame_area_delta_ratio for item in stats
        ),
    )


def _lower_contact_summary_fragment(stats: LowerContactRefineStats | None) -> str:
    if stats is None:
        return (
            "lower_contact_refine=False "
            "lower_contact_grow_pixels=0 "
            "lower_contact_band_ratio=0.000000 "
            "lower_contact_area_cap_ratio=0.000000 "
            "lower_contact_candidate_frames=0 "
            "lower_contact_refined_frames=0 "
            "lower_contact_area_delta_ratio=0.000000 "
            "lower_contact_max_area_delta_ratio=0.000000"
        )
    return (
        f"lower_contact_refine={stats.enabled} "
        f"lower_contact_grow_pixels={stats.requested_grow_pixels} "
        f"lower_contact_band_ratio={stats.band_ratio:.6f} "
        f"lower_contact_area_cap_ratio={stats.area_cap_ratio:.6f} "
        f"lower_contact_candidate_frames={stats.candidate_frames} "
        f"lower_contact_refined_frames={stats.refined_frames} "
        f"lower_contact_area_delta_ratio={stats.area_delta_ratio:.6f} "
        f"lower_contact_max_area_delta_ratio={stats.max_frame_area_delta_ratio:.6f}"
    )


def _refine_lower_contact_mask(
    mask: Any,
    *,
    enabled: bool,
    grow_pixels: int,
    band_ratio: float,
    area_cap_ratio: float,
) -> tuple[Any, LowerContactRefineStats]:
    if not enabled or grow_pixels <= 0:
        return mask, _disabled_lower_contact_stats(
            enabled=enabled,
            grow_pixels=grow_pixels,
            band_ratio=band_ratio,
            area_cap_ratio=area_cap_ratio,
        )
    if mask.ndim != 3:
        raise ValueError(
            "lower-contact refinement requires mask shape [frames, height, width]"
        )

    torch = _torch_required()
    frames, height, _width = (int(part) for part in mask.shape)
    active = mask > 0.5
    base_area = active.flatten(1).sum(dim=1).to(dtype=torch.float32)
    has_subject = base_area > 0
    if not bool(has_subject.any().item()):
        return mask, _disabled_lower_contact_stats(
            enabled=enabled,
            grow_pixels=grow_pixels,
            band_ratio=band_ratio,
            area_cap_ratio=area_cap_ratio,
        )

    rows_present = active.any(dim=2)
    row_values = torch.arange(height, device=mask.device, dtype=torch.long).view(
        1,
        height,
    )
    row_values = row_values.expand(frames, height)
    y0 = (
        torch.where(rows_present, row_values, torch.full_like(row_values, height))
        .min(dim=1)
        .values
    )
    y1 = (
        torch.where(rows_present, row_values, torch.full_like(row_values, -1))
        .max(dim=1)
        .values
    )
    bbox_height = (y1 - y0 + 1).clamp(min=1)
    band_offset = torch.floor(bbox_height.to(dtype=torch.float32) * (1.0 - band_ratio))
    band_start = (y0 + band_offset.to(dtype=torch.long)).clamp(
        min=0,
        max=max(height - 1, 0),
    )

    y_grid = torch.arange(height, device=mask.device, dtype=torch.long).view(
        1,
        height,
        1,
    )
    band_mask = y_grid >= band_start.view(frames, 1, 1)
    lower_subject = active & band_mask & has_subject.view(frames, 1, 1)
    candidate_frames = lower_subject.flatten(1).any(dim=1)
    grown_lower = _grow_mask(lower_subject.to(dtype=mask.dtype), grow_pixels) > 0.5
    candidate_extra = grown_lower & band_mask & ~active
    added_area = candidate_extra.flatten(1).sum(dim=1).to(dtype=torch.float32)
    delta_ratio = torch.zeros_like(base_area)
    valid_area = base_area > 0
    delta_ratio[valid_area] = added_area[valid_area] / base_area[valid_area]

    accepted = candidate_frames & (added_area > 0)
    accepted = accepted & (delta_ratio <= area_cap_ratio)
    accepted_extra = candidate_extra & accepted.view(frames, 1, 1)
    refined = torch.maximum(mask, accepted_extra.to(dtype=mask.dtype))

    refined_frames = int(accepted.sum().item())
    added_pixels = float(accepted_extra.sum().item())
    max_delta = float(delta_ratio[accepted].max().item()) if refined_frames else 0.0
    stats = LowerContactRefineStats(
        enabled=True,
        requested_grow_pixels=grow_pixels,
        band_ratio=band_ratio,
        area_cap_ratio=area_cap_ratio,
        candidate_frames=int(candidate_frames.sum().item()),
        refined_frames=refined_frames,
        max_applied_grow_pixels=grow_pixels if refined_frames else 0,
        base_pixels=float(base_area[has_subject].sum().item()),
        added_pixels=added_pixels,
        max_frame_area_delta_ratio=max_delta,
    )
    return refined, stats


def _build_tensor_subject_mask(
    pose_video_mask: Any,
    *,
    condition: SCAIL2Condition,
    mask_preset: str,
    grow_pixels: int,
    blur_pixels: int,
    invert: bool,
    lower_contact_refine: bool,
    lower_contact_grow_pixels: int,
    lower_contact_band_ratio: float,
    lower_contact_area_cap_ratio: float,
    work_device: Any | None = None,
) -> ReplacementDenoiseMaskResult:
    torch = _torch_required()
    tensor = _tensor_image_frames(pose_video_mask)
    frames, height, width = _validate_tensor_shape(condition=condition, tensor=tensor)
    input_device = str(tensor.device)
    selected_device = torch.device(work_device) if work_device is not None else _preferred_work_device(torch, tensor)

    output_chunks = []
    lower_contact_stats = []
    raw_subject_pixels = 0.0
    raw_frame_ratios: list[float] = []
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
        raw_frame_ratios.extend(_frame_subject_ratios(mask_chunk))
        mask_chunk = _grow_mask(mask_chunk, grow_pixels)
        mask_chunk, refine_stats = _refine_lower_contact_mask(
            mask_chunk,
            enabled=lower_contact_refine,
            grow_pixels=lower_contact_grow_pixels,
            band_ratio=lower_contact_band_ratio,
            area_cap_ratio=lower_contact_area_cap_ratio,
        )
        lower_contact_stats.append(refine_stats)
        mask_chunk = _blur_mask(mask_chunk, blur_pixels)
        if invert:
            mask_chunk = 1.0 - mask_chunk
        output_chunks.append(
            mask_chunk.clamp(0.0, 1.0).detach().to(device="cpu", dtype=torch.float32)
        )

    if raw_subject_pixels <= 0:
        raise ValueError("pose_video_mask contains no subject pixels")

    mask = torch.cat(output_chunks, dim=0).contiguous()
    coverage_stats = _coverage_stats_from_ratios(tuple(raw_frame_ratios))
    merged_lower_contact_stats = _merge_lower_contact_stats(lower_contact_stats)
    _attach_scail_pose2_mask_metadata(
        mask,
        condition=condition,
        role=SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE,
    )
    subject_ratio = float(mask.mean().item())
    diagnostics = mask_diagnostics(mask)
    summary = _replacement_summary(
        condition=condition,
        frame_count=frames,
        height=height,
        width=width,
        subject_ratio=subject_ratio,
        coverage_stats=coverage_stats,
        mask_preset=mask_preset,
        diagnostics=diagnostics,
        grow_pixels=grow_pixels,
        blur_pixels=blur_pixels,
        invert=invert,
        fast_path="tensor_subject_mask",
        input_device=input_device,
        work_device=str(selected_device),
        output_device=str(mask.device),
        lower_contact_stats=merged_lower_contact_stats,
    )
    return ReplacementDenoiseMaskResult(
        mask=mask,
        summary=summary,
        frame_count=frames,
        height=height,
        width=width,
        subject_ratio=subject_ratio,
        raw_subject_ratio=coverage_stats.raw_subject_ratio,
        coverage_warning=coverage_stats.coverage_warning,
        coverage_empty_frame_count=coverage_stats.empty_frame_count,
        coverage_sparse_frame_count=coverage_stats.sparse_frame_count,
        coverage_longest_sparse_streak=coverage_stats.longest_sparse_streak,
        diagnostics=diagnostics,
        mask_preset=mask_preset,
        fast_path="tensor_subject_mask",
        input_device=input_device,
        work_device=str(selected_device),
        output_device=str(mask.device),
        lower_contact_refine=merged_lower_contact_stats,
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
    coverage_stats: ReplacementCoverageStats,
    mask_preset: str,
    diagnostics: MaskDiagnostics | None,
    grow_pixels: int,
    blur_pixels: int,
    invert: bool,
    fast_path: str,
    input_device: str,
    work_device: str,
    output_device: str,
    lower_contact_stats: LowerContactRefineStats | None,
) -> str:
    return (
        "replacement_denoise_mask "
        f"mode={condition.mode} "
        f"frames={frame_count} "
        f"size={width}x{height} "
        f"subject_ratio={subject_ratio:.6f} "
        f"raw_subject_ratio={coverage_stats.raw_subject_ratio:.6f} "
        f"final_subject_ratio={subject_ratio:.6f} "
        f"coverage_warning={coverage_stats.coverage_warning} "
        f"coverage_empty_frames={coverage_stats.empty_frame_count} "
        f"coverage_sparse_frames={coverage_stats.sparse_frame_count} "
        f"coverage_longest_sparse_streak={coverage_stats.longest_sparse_streak} "
        # IMPORTANT: keep this explicit so logs do not imply mask grow/blur can
        # recover subject regions that SAM3 never covered.
        f"coverage_limitation=missing_regions_not_recovered_by_grow_blur "
        f"mask_preset={mask_preset} "
        f"grow_pixels={grow_pixels} "
        f"blur_pixels={blur_pixels} "
        f"invert={bool(invert)} "
        f"fast_path={fast_path} "
        f"input_device={input_device} "
        f"work_device={work_device} "
        f"output_device={output_device}"
        f" {_lower_contact_summary_fragment(lower_contact_stats)}"
        + (
            f" {diagnostics_summary_fragment(diagnostics)}"
            if diagnostics is not None
            else ""
        )
    )


def _build_mode_passthrough_mask(
    condition: SCAIL2Condition,
    *,
    mask_preset: str,
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
    diagnostics = mask_diagnostics(mask)
    coverage_stats = _coverage_stats_from_ratios((1.0,) * frame_count)
    summary = (
        _replacement_summary(
            condition=condition,
            frame_count=frame_count,
            height=height,
            width=width,
            subject_ratio=subject_ratio,
            coverage_stats=coverage_stats,
            mask_preset=mask_preset,
            diagnostics=diagnostics,
            grow_pixels=grow_pixels,
            blur_pixels=blur_pixels,
            invert=False,
            fast_path="mode_passthrough",
            input_device="condition",
            work_device=str(mask.device),
            output_device=str(mask.device),
            lower_contact_stats=_disabled_lower_contact_stats(
                enabled=False,
                grow_pixels=0,
                band_ratio=0.0,
                area_cap_ratio=0.0,
            ),
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
        raw_subject_ratio=coverage_stats.raw_subject_ratio,
        coverage_warning=coverage_stats.coverage_warning,
        coverage_empty_frame_count=coverage_stats.empty_frame_count,
        coverage_sparse_frame_count=coverage_stats.sparse_frame_count,
        coverage_longest_sparse_streak=coverage_stats.longest_sparse_streak,
        diagnostics=diagnostics,
        mask_preset=mask_preset,
        fast_path="mode_passthrough",
        input_device="condition",
        work_device=str(mask.device),
        output_device=str(mask.device),
        lower_contact_refine=_disabled_lower_contact_stats(
            enabled=False,
            grow_pixels=0,
            band_ratio=0.0,
            area_cap_ratio=0.0,
        ),
    )


def build_replacement_denoise_mask(
    *,
    condition: Any,
    pose_video_mask: Any,
    mask_preset: Any = "custom",
    grow_pixels: Any = 0,
    blur_pixels: Any = 0,
    strict_replacement_mode: bool = False,
    invert: bool = False,
    lower_contact_refine: Any = DEFAULT_LOWER_CONTACT_REFINE,
    lower_contact_grow_pixels: Any = DEFAULT_LOWER_CONTACT_GROW_PIXELS,
    lower_contact_band_ratio: Any = DEFAULT_LOWER_CONTACT_BAND_RATIO,
    lower_contact_area_cap_ratio: Any = DEFAULT_LOWER_CONTACT_AREA_CAP_RATIO,
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

    preset, grow, blur = resolve_mask_preset(
        mask_preset,
        grow_pixels=grow_pixels,
        blur_pixels=blur_pixels,
    )
    refine_lower_contact = _coerce_bool(lower_contact_refine)
    contact_grow = _non_negative_int(
        "lower_contact_grow_pixels",
        lower_contact_grow_pixels,
    )
    contact_band_ratio = _float_in_range(
        "lower_contact_band_ratio",
        lower_contact_band_ratio,
        minimum=0.05,
        maximum=0.95,
    )
    contact_area_cap_ratio = _float_in_range(
        "lower_contact_area_cap_ratio",
        lower_contact_area_cap_ratio,
        minimum=0.0,
        maximum=10.0,
    )
    if valid_condition.mode != "replacement":
        return _build_mode_passthrough_mask(
            valid_condition,
            mask_preset=preset,
            grow_pixels=grow,
            blur_pixels=blur,
        )
    if _is_torch_tensor(pose_video_mask):
        try:
            return _build_tensor_subject_mask(
                pose_video_mask,
                condition=valid_condition,
                mask_preset=preset,
                grow_pixels=grow,
                blur_pixels=blur,
                invert=bool(invert),
                lower_contact_refine=refine_lower_contact,
                lower_contact_grow_pixels=contact_grow,
                lower_contact_band_ratio=contact_band_ratio,
                lower_contact_area_cap_ratio=contact_area_cap_ratio,
            )
        except RuntimeError as exc:
            if "cuda" not in str(exc).lower():
                raise
            return _build_tensor_subject_mask(
                pose_video_mask,
                condition=valid_condition,
                mask_preset=preset,
                grow_pixels=grow,
                blur_pixels=blur,
                invert=bool(invert),
                lower_contact_refine=refine_lower_contact,
                lower_contact_grow_pixels=contact_grow,
                lower_contact_band_ratio=contact_band_ratio,
                lower_contact_area_cap_ratio=contact_area_cap_ratio,
                work_device="cpu",
            )

    indices = _normalize_pose_mask_indices(pose_video_mask)
    _validate_shape(condition=valid_condition, indices=indices)

    mask = _indices_to_subject_tensor(indices)
    raw_subject_pixels = float(mask.sum().item())
    if raw_subject_pixels <= 0:
        raise ValueError("pose_video_mask contains no subject pixels")
    coverage_stats = _coverage_stats_from_ratios(_frame_subject_ratios(mask))

    mask = _grow_mask(mask, grow)
    mask, lower_contact_stats = _refine_lower_contact_mask(
        mask,
        enabled=refine_lower_contact,
        grow_pixels=contact_grow,
        band_ratio=contact_band_ratio,
        area_cap_ratio=contact_area_cap_ratio,
    )
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
    diagnostics = mask_diagnostics(mask)
    summary = _replacement_summary(
        condition=valid_condition,
        frame_count=frame_count,
        height=height,
        width=width,
        subject_ratio=subject_ratio,
        coverage_stats=coverage_stats,
        mask_preset=preset,
        diagnostics=diagnostics,
        grow_pixels=grow,
        blur_pixels=blur,
        invert=bool(invert),
        fast_path="semantic_indices",
        input_device="python",
        work_device=str(mask.device),
        output_device=str(mask.device),
        lower_contact_stats=lower_contact_stats,
    )
    return ReplacementDenoiseMaskResult(
        mask=mask,
        summary=summary,
        frame_count=frame_count,
        height=height,
        width=width,
        subject_ratio=subject_ratio,
        raw_subject_ratio=coverage_stats.raw_subject_ratio,
        coverage_warning=coverage_stats.coverage_warning,
        coverage_empty_frame_count=coverage_stats.empty_frame_count,
        coverage_sparse_frame_count=coverage_stats.sparse_frame_count,
        coverage_longest_sparse_streak=coverage_stats.longest_sparse_streak,
        diagnostics=diagnostics,
        mask_preset=preset,
        fast_path="semantic_indices",
        input_device="python",
        work_device=str(mask.device),
        output_device=str(mask.device),
        lower_contact_refine=lower_contact_stats,
    )
