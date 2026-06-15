"""Explicit WanAnimate-compatible fallback adapter for SCAIL-2 conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .condition import SCAIL2Condition
from .masks import BACKGROUND_INDEX


LOSS_RGB_TO_GRAYSCALE = "rgb_semantic_masks_collapsed_to_binary_grayscale"
LOSS_28CH_MASK_LATENT = "scail2_28_channel_mask_latent_not_represented"
LOSS_REPLACEMENT_ROPE = "replacement_flag_rope_mode_not_represented"
LOSS_ADDITIONAL_REFS = "additional_reference_pairs_not_represented"
LOSS_TRACK_COLORS = "mask_palette_track_metadata_not_preserved_as_channels"


@dataclass(frozen=True)
class WanAnimateFallbackPayload:
    ref_images: Any
    pose_images: Any
    bg_images: Any | None
    mask: tuple[tuple[tuple[float, ...], ...], ...]
    metadata: dict[str, Any]


def _mask_indices_to_grayscale(
    indices: tuple[tuple[tuple[int, ...], ...], ...],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(0.0 if value == BACKGROUND_INDEX else 1.0 for value in row)
            for row in frame
        )
        for frame in indices
    )


def _semantic_losses(condition: SCAIL2Condition) -> tuple[str, ...]:
    losses = [
        LOSS_RGB_TO_GRAYSCALE,
        LOSS_28CH_MASK_LATENT,
        LOSS_TRACK_COLORS,
    ]
    if condition.replace_flag:
        losses.append(LOSS_REPLACEMENT_ROPE)
    if condition.additional_references:
        losses.append(LOSS_ADDITIONAL_REFS)
    return tuple(losses)


def convert_scail2_condition_to_wananimate(
    condition: SCAIL2Condition,
    *,
    allow_semantic_degradation: bool = False,
    bg_images: Any | None = None,
) -> WanAnimateFallbackPayload:
    """Convert a SCAIL-2 condition into a degraded WanAnimate-style payload."""

    losses = _semantic_losses(condition)
    if losses and not allow_semantic_degradation:
        raise ValueError(
            "WanAnimate fallback requires explicit semantic degradation approval"
        )

    grayscale_mask = _mask_indices_to_grayscale(condition.driving_mask_indices)
    metadata = {
        "target": "WanVideoAnimateEmbeds",
        "is_full_scail2_parity": False,
        "degradation_enabled": allow_semantic_degradation,
        "semantic_losses": losses,
        "source_condition_type": condition.type_name,
        "source_mode": condition.mode,
        "replace_flag": condition.replace_flag,
        "width": condition.width,
        "height": condition.height,
        "num_frames": condition.num_frames,
        "mask_palette": condition.mask_palette,
        "original_rgb_masks_preserved": True,
    }
    return WanAnimateFallbackPayload(
        ref_images=condition.ref_image,
        pose_images=condition.pose_video,
        bg_images=bg_images,
        mask=grayscale_mask,
        metadata=metadata,
    )
