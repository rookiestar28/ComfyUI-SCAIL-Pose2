"""Replacement condition-video subject suppression helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .masks import MASK_OFF_THRESHOLD
from .replacement_diagnostics import MaskDiagnostics, diagnostics_summary_fragment, mask_diagnostics
from .replacement_mask import _blur_mask, _grow_mask
from .replacement_presets import MASK_PRESETS, resolve_mask_preset


SUPPRESSION_MODES: tuple[str, ...] = (
    "blur_fill",
    "mean_fill",
    "black_fill",
    "white_fill",
    "noise_fill",
)


@dataclass(frozen=True)
class ReplacementConditionVideoResult:
    driving_video_condition: Any
    summary: str
    frame_count: int
    height: int
    width: int
    suppression_mode: str
    suppression_strength: float
    mask_preset: str
    grow_pixels: int
    blur_pixels: int
    diagnostics: MaskDiagnostics
    input_device: str
    work_device: str
    output_device: str


def _torch_required() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("torch is required to build replacement condition videos") from exc
    return torch


def _strength(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("suppression_strength must be between 0.0 and 1.0")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("suppression_strength must be between 0.0 and 1.0") from exc
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError("suppression_strength must be between 0.0 and 1.0")
    return parsed


def _image_frames(value: Any, *, name: str) -> Any:
    torch = _torch_required()
    tensor = value.detach() if hasattr(value, "detach") else torch.as_tensor(value)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[-1] < 3:
        raise ValueError(f"{name} must have shape [frames, height, width, channels]")
    if int(tensor.shape[0]) <= 0 or int(tensor.shape[1]) <= 0 or int(tensor.shape[2]) <= 0:
        raise ValueError(f"{name} must be non-empty")
    return tensor


def _validate_video_and_mask_shape(video: Any, mask: Any) -> tuple[int, int, int]:
    video_shape = tuple(int(part) for part in video.shape)
    mask_shape = tuple(int(part) for part in mask.shape)
    if video_shape[:3] != mask_shape[:3]:
        raise ValueError(
            "driving_video and pose_video_mask must share frame/spatial shape, "
            f"got {video_shape[:3]} and {mask_shape[:3]}"
        )
    return video_shape[0], video_shape[1], video_shape[2]


def _preferred_work_device(torch: Any, tensor: Any) -> Any:
    tensor_device = getattr(tensor, "device", torch.device("cpu"))
    if getattr(tensor_device, "type", None) == "cuda":
        return tensor_device
    try:
        from comfy import model_management

        device = torch.device(model_management.intermediate_device())
        if device.type == "cuda":
            return device
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _off_threshold(torch: Any, rgb: Any) -> float:
    normalized = bool(torch.logical_and(rgb >= 0.0, rgb <= 1.0).all().item())
    return MASK_OFF_THRESHOLD / 255.0 if normalized else float(MASK_OFF_THRESHOLD)


def _subject_mask_from_rgb(torch: Any, mask_rgb: Any) -> Any:
    rgb = mask_rgb[..., :3].to(dtype=torch.float32)
    threshold = _off_threshold(torch, rgb)
    return (rgb > threshold).any(dim=-1).to(dtype=torch.float32)


def _frame_background_mean(video: Any, subject_mask: Any) -> Any:
    torch = _torch_required()
    background = (1.0 - subject_mask).unsqueeze(-1)
    denom = background.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
    mean = (video * background).sum(dim=(1, 2), keepdim=True) / denom
    full_mean = video.mean(dim=(1, 2), keepdim=True)
    has_background = (background.sum(dim=(1, 2), keepdim=True) > 0).to(video.dtype)
    return torch.where(has_background.bool(), mean, full_mean).expand_as(video)


def _blur_fill(video: Any, radius: int) -> Any:
    import torch.nn.functional as F

    kernel_radius = max(1, int(radius))
    kernel_size = kernel_radius * 2 + 1
    frames = video.permute(0, 3, 1, 2)
    blurred = F.avg_pool2d(
        frames,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_radius,
    )
    return blurred.permute(0, 2, 3, 1).contiguous()


def _noise_fill(torch: Any, video: Any, seed: Any) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.rand(
        tuple(int(part) for part in video.shape),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(video.device, dtype=video.dtype)
    return noise


def _fill_video(
    torch: Any,
    video: Any,
    subject_mask: Any,
    *,
    suppression_mode: str,
    blur_radius: int,
    noise_seed: Any,
) -> Any:
    if suppression_mode == "blur_fill":
        return _blur_fill(video, blur_radius)
    if suppression_mode == "mean_fill":
        return _frame_background_mean(video, subject_mask)
    if suppression_mode == "black_fill":
        return video.new_zeros(video.shape)
    if suppression_mode == "white_fill":
        return video.new_ones(video.shape)
    if suppression_mode == "noise_fill":
        return _noise_fill(torch, video, noise_seed)
    raise ValueError("suppression_mode must be one of " + ", ".join(SUPPRESSION_MODES))


def _summary(
    *,
    result: ReplacementConditionVideoResult,
) -> str:
    return (
        "replacement_condition_video "
        f"frames={result.frame_count} "
        f"size={result.width}x{result.height} "
        f"mask_preset={result.mask_preset} "
        f"grow_pixels={result.grow_pixels} "
        f"blur_pixels={result.blur_pixels} "
        f"suppression_mode={result.suppression_mode} "
        f"suppression_strength={result.suppression_strength:.3f} "
        f"{diagnostics_summary_fragment(result.diagnostics)} "
        f"input_device={result.input_device} "
        f"work_device={result.work_device} "
        f"output_device={result.output_device}"
    )


def build_replacement_condition_video(
    *,
    driving_video: Any,
    pose_video_mask: Any,
    mask_preset: Any = "custom",
    grow_pixels: Any = 8,
    blur_pixels: Any = 0,
    suppression_mode: Any = "blur_fill",
    suppression_strength: Any = 1.0,
    noise_seed: Any = 0,
) -> ReplacementConditionVideoResult:
    """Suppress original subject pixels before SCAIL-2 replacement conditioning."""

    torch = _torch_required()
    mode = str(suppression_mode or "blur_fill")
    if mode not in SUPPRESSION_MODES:
        raise ValueError("suppression_mode must be one of " + ", ".join(SUPPRESSION_MODES))
    strength = _strength(suppression_strength)
    preset, grow, blur = resolve_mask_preset(
        mask_preset,
        grow_pixels=grow_pixels,
        blur_pixels=blur_pixels,
    )

    video = _image_frames(driving_video, name="driving_video")
    mask_image = _image_frames(pose_video_mask, name="pose_video_mask")
    frames, height, width = _validate_video_and_mask_shape(video, mask_image)
    input_device = str(getattr(video, "device", "unknown"))
    work_device = _preferred_work_device(torch, video)
    video_work = video[..., :3].to(device=work_device, dtype=torch.float32, non_blocking=True)
    mask_work = mask_image[..., :3].to(device=work_device, dtype=torch.float32, non_blocking=True)

    subject_mask = _subject_mask_from_rgb(torch, mask_work)
    raw_subject_pixels = float(subject_mask.sum().item())
    if raw_subject_pixels <= 0:
        raise ValueError("pose_video_mask contains no subject pixels")
    subject_mask = _grow_mask(subject_mask, grow)
    subject_mask = _blur_mask(subject_mask, blur).clamp(0.0, 1.0)
    diagnostics = mask_diagnostics(subject_mask)

    if strength == 0.0:
        output_work = video_work
    else:
        fill = _fill_video(
            torch,
            video_work,
            subject_mask,
            suppression_mode=mode,
            blur_radius=max(grow, blur, 1),
            noise_seed=noise_seed,
        )
        alpha = (subject_mask * strength).clamp(0.0, 1.0).unsqueeze(-1)
        output_work = video_work * (1.0 - alpha) + fill * alpha

    output = output_work.clamp(0.0, 1.0).to(device="cpu", dtype=torch.float32).contiguous()
    result = ReplacementConditionVideoResult(
        driving_video_condition=output,
        summary="",
        frame_count=frames,
        height=height,
        width=width,
        suppression_mode=mode,
        suppression_strength=strength,
        mask_preset=preset,
        grow_pixels=grow,
        blur_pixels=blur,
        diagnostics=diagnostics,
        input_device=input_device,
        work_device=str(work_device),
        output_device=str(output.device),
    )
    return ReplacementConditionVideoResult(
        **{
            **result.__dict__,
            "summary": _summary(result=result),
        }
    )
