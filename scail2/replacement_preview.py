"""Replacement preview diagnostics for SCAIL-Pose2 workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReplacementPreviewScheduleFrame:
    """One sampler-step view of the replacement preserve/replace split."""

    step_index: int
    threshold: float
    preserve_ratio: float
    replace_ratio: float


@dataclass(frozen=True)
class ReplacementPreviewPathDiagnostic:
    """Safe diagnostic for the replacement samples/noise-mask path."""

    status: str
    background_lock_expected: bool
    early_preview_original_background_reliable: bool
    reason: str


def _is_number(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


def _nested_shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        try:
            return tuple(int(part) for part in value.shape)
        except (TypeError, ValueError):
            pass
    if _is_number(value):
        return ()
    if hasattr(value, "tolist"):
        return _nested_shape(value.tolist())
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ValueError("noise_mask must contain numeric values")
    items = list(value)
    if not items:
        return (0,)
    first = _nested_shape(items[0])
    for item in items[1:]:
        if _nested_shape(item) != first:
            raise ValueError("noise_mask must be rectangular")
    return (len(items),) + first


def _flatten_values(value: Any) -> list[float]:
    if _is_number(value):
        return [float(value)]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        return [float(part) for part in value.reshape(-1).tolist()]
    if hasattr(value, "tolist"):
        return _flatten_values(value.tolist())
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ValueError("noise_mask must contain numeric values")
    flattened: list[float] = []
    for item in value:
        flattened.extend(_flatten_values(item))
    return flattened


def _mask_values_and_shape(noise_mask: Any) -> tuple[list[float], tuple[int, ...]]:
    shape = _nested_shape(noise_mask)
    values = _flatten_values(noise_mask)
    if not values:
        raise ValueError("noise_mask must not be empty")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError("noise_mask values must be normalized to [0, 1]")
    return values, shape


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def build_replacement_preview_schedule(
    noise_mask: Any,
    *,
    step_count: Any,
) -> list[ReplacementPreviewScheduleFrame]:
    """Model the wrapper differential-diffusion replacement preview schedule.

    The current wrapper uses `mask = (1 - noise_mask) > threshold` with
    `threshold = step_index / step_count`. In SCAIL-Pose2 replacement masks,
    subject/replace pixels are `1.0` and background/preserve pixels are `0.0`.
    """

    steps = _positive_int("step_count", step_count)
    values, _shape = _mask_values_and_shape(noise_mask)
    total = float(len(values))
    schedule: list[ReplacementPreviewScheduleFrame] = []
    for step_index in range(steps):
        threshold = step_index / steps
        preserve_pixels = sum(1 for value in values if (1.0 - value) > threshold)
        preserve_ratio = preserve_pixels / total
        schedule.append(
            ReplacementPreviewScheduleFrame(
                step_index=step_index,
                threshold=threshold,
                preserve_ratio=preserve_ratio,
                replace_ratio=1.0 - preserve_ratio,
            )
        )
    return schedule


def summarize_replacement_preview_schedule(
    noise_mask: Any,
    *,
    step_count: Any,
) -> str:
    """Return content-safe shape/ratio diagnostics for a preview schedule."""

    _values, shape = _mask_values_and_shape(noise_mask)
    schedule = build_replacement_preview_schedule(noise_mask, step_count=step_count)
    first = schedule[0]
    last = schedule[-1]
    return (
        "replacement_preview_schedule "
        f"mask_shape={shape} "
        f"steps={len(schedule)} "
        f"first_preserve_ratio={first.preserve_ratio:.6f} "
        f"first_replace_ratio={first.replace_ratio:.6f} "
        f"last_preserve_ratio={last.preserve_ratio:.6f} "
        f"last_replace_ratio={last.replace_ratio:.6f}"
    )


def classify_replacement_preview_path(
    *,
    samples_present: bool,
    noise_mask_present: bool,
    add_noise_to_samples: bool,
    samples_disabled: bool = False,
    condition_mode: str | None = None,
) -> ReplacementPreviewPathDiagnostic:
    """Classify the expected replacement preview path without media content."""

    if samples_disabled:
        return ReplacementPreviewPathDiagnostic(
            status="samples_disabled",
            background_lock_expected=False,
            early_preview_original_background_reliable=False,
            reason="samples path is disabled by upstream mask metadata",
        )
    if condition_mode is not None and condition_mode != "replacement":
        return ReplacementPreviewPathDiagnostic(
            status="samples_disabled",
            background_lock_expected=False,
            early_preview_original_background_reliable=False,
            reason=f"condition mode is {condition_mode}",
        )
    if not samples_present and not noise_mask_present:
        return ReplacementPreviewPathDiagnostic(
            status="not_wired",
            background_lock_expected=False,
            early_preview_original_background_reliable=False,
            reason="samples and noise_mask are both missing",
        )
    if not samples_present:
        return ReplacementPreviewPathDiagnostic(
            status="missing_samples",
            background_lock_expected=False,
            early_preview_original_background_reliable=False,
            reason="noise_mask is present but samples are missing",
        )
    if not noise_mask_present:
        return ReplacementPreviewPathDiagnostic(
            status="missing_noise_mask",
            background_lock_expected=False,
            early_preview_original_background_reliable=False,
            reason="samples are present but noise_mask is missing",
        )
    if add_noise_to_samples:
        return ReplacementPreviewPathDiagnostic(
            status="wired_noisy_preview_expected",
            background_lock_expected=True,
            early_preview_original_background_reliable=False,
            reason="background path is wired but early preview is noised",
        )
    return ReplacementPreviewPathDiagnostic(
        status="wired_clean_preview_expected",
        background_lock_expected=True,
        early_preview_original_background_reliable=True,
        reason="background path is wired without initial samples noise",
    )
