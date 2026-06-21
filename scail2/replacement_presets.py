"""Shared replacement mask preset helpers."""

from __future__ import annotations

from typing import Any


MASK_PRESETS: dict[str, tuple[int | None, int | None]] = {
    "custom": (None, None),
    "tight": (2, 0),
    "default": (8, 0),
    "loose": (16, 2),
    "soft": (12, 4),
}


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


def resolve_mask_preset(
    mask_preset: Any,
    *,
    grow_pixels: Any,
    blur_pixels: Any,
) -> tuple[str, int, int]:
    preset = str(mask_preset or "custom")
    if preset not in MASK_PRESETS:
        raise ValueError(
            "mask_preset must be one of " + ", ".join(sorted(MASK_PRESETS))
        )
    grow_default, blur_default = MASK_PRESETS[preset]
    grow = (
        _positive_or_zero_int("grow_pixels", grow_pixels)
        if grow_default is None
        else grow_default
    )
    blur = (
        _positive_or_zero_int("blur_pixels", blur_pixels)
        if blur_default is None
        else blur_default
    )
    return preset, grow, blur
