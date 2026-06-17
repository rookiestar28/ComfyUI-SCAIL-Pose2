"""User-provided SCAIL-2 mask preprocessing helpers."""

from __future__ import annotations

from typing import Any, Sequence

from .condition import SCAIL2Condition, SCAIL2Mode, build_scail2_condition


def build_user_mask_condition(
    *,
    mode: SCAIL2Mode,
    ref_image: Any,
    ref_mask_frames: Sequence[Any],
    pose_video: Any,
    pose_frame_count: Any,
    driving_mask_frames: Sequence[Any],
    width: Any,
    height: Any,
    segment_len: Any = 81,
    segment_overlap: Any = 5,
    additional_ref_images: Sequence[Any] | None = None,
    additional_ref_masks: Sequence[Sequence[Any]] | None = None,
    source_kind: Any = "user_rgb_masks",
    previous_frame_count: Any = 0,
    video_frame_offset: Any = 0,
) -> SCAIL2Condition:
    """Build a validated SCAIL-2 condition from user-supplied RGB masks."""

    return build_scail2_condition(
        mode=mode,
        ref_image=ref_image,
        ref_mask_frames=ref_mask_frames,
        pose_video=pose_video,
        pose_frame_count=pose_frame_count,
        driving_mask_frames=driving_mask_frames,
        width=width,
        height=height,
        segment_len=segment_len,
        segment_overlap=segment_overlap,
        additional_ref_images=additional_ref_images,
        additional_ref_masks=additional_ref_masks,
        source_kind=source_kind,
        previous_frame_count=previous_frame_count,
        video_frame_offset=video_frame_offset,
    )
