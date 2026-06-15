"""Optional SAM3 preprocessing helpers with lazy dependency loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .condition import SCAIL2Condition, SCAIL2Mode, SCAIL2_MODES
from .masks import classify_rgb_semantic_color, semantic_mask_indices
from .preprocessing import build_user_mask_condition


BLACK_RGB = (0, 0, 0)
DEFAULT_SAM3_TRACK_RGB_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
)


class SAM3DependencyError(ImportError):
    """Raised when optional SAM3 runtime dependencies are unavailable."""


@dataclass(frozen=True)
class SAM3TrackMetadata:
    track_id: str
    color_rgb: tuple[int, int, int]
    pixels_per_frame: tuple[int, ...]


@dataclass(frozen=True)
class SAM3SemanticMaskBundle:
    mode: str
    frames: tuple[tuple[tuple[tuple[int, int, int], ...], ...], ...]
    track_metadata: tuple[SAM3TrackMetadata, ...]
    frame_count: int
    height: int
    width: int
    background_rgb: tuple[int, int, int]
    source: str = "sam3_mock"


@dataclass(frozen=True)
class SAM3ConditionPayload:
    condition: SCAIL2Condition
    ref_mask_bundle: SAM3SemanticMaskBundle
    driving_mask_bundle: SAM3SemanticMaskBundle
    mode: str
    source: str = "sam3_mock"


def require_sam3_predictors():
    """Import SAM3 predictor classes only when a SAM3 execution path is invoked."""

    try:
        from ultralytics.models.sam import (  # type: ignore[import-not-found]
            SAM3SemanticPredictor,
            SAM3VideoSemanticPredictor,
        )
    except Exception as exc:  # pragma: no cover - exact optional import failure varies
        raise SAM3DependencyError(
            "Optional SAM3 preprocessing requires an ultralytics build that provides "
            "SAM3SemanticPredictor and SAM3VideoSemanticPredictor. Install SAM3 "
            "dependencies manually in the active ComfyUI environment and provide "
            "local model weights; this node never auto-downloads gated weights."
        ) from exc
    return SAM3SemanticPredictor, SAM3VideoSemanticPredictor


def _normalize_bool_track_mask(
    mask: Sequence[Sequence[Sequence[Any]]],
    *,
    track_name: str,
) -> tuple[tuple[tuple[bool, ...], ...], ...]:
    if not mask:
        raise ValueError(f"{track_name} must contain at least one frame")
    frames = []
    expected_height: int | None = None
    expected_width: int | None = None
    for frame_index, frame in enumerate(mask):
        if not frame:
            raise ValueError(f"{track_name} frame {frame_index} is empty")
        frame_height = len(frame)
        frame_width = len(frame[0])
        if frame_width == 0:
            raise ValueError(f"{track_name} frame {frame_index} has empty rows")
        if expected_height is None:
            expected_height = frame_height
            expected_width = frame_width
        elif frame_height != expected_height or frame_width != expected_width:
            raise ValueError(f"{track_name} frames must have consistent dimensions")
        rows = []
        for row in frame:
            if len(row) != frame_width:
                raise ValueError(f"{track_name} rows must have consistent width")
            rows.append(tuple(bool(value) for value in row))
        frames.append(tuple(rows))
    return tuple(frames)


def _validate_color(color: Sequence[Any]) -> tuple[int, int, int]:
    rgb = tuple(int(part) for part in color[:3])
    classify_rgb_semantic_color(rgb)
    return rgb


def _normalize_colors(
    colors: Sequence[Sequence[Any]] | None,
    track_count: int,
) -> tuple[tuple[int, int, int], ...]:
    if colors is None:
        return tuple(
            DEFAULT_SAM3_TRACK_RGB_PALETTE[index % len(DEFAULT_SAM3_TRACK_RGB_PALETTE)]
            for index in range(track_count)
        )
    if len(colors) < track_count:
        raise ValueError("colors must contain at least one color per track")
    return tuple(_validate_color(color) for color in colors[:track_count])


def sam3_tracks_to_semantic_mask_frames(
    track_masks: Sequence[Sequence[Sequence[Sequence[Any]]]],
    *,
    mode: SCAIL2Mode,
    track_ids: Sequence[Any] | None = None,
    colors: Sequence[Sequence[Any]] | None = None,
    expected_track_count: int | None = None,
    background_rgb: Sequence[Any] = BLACK_RGB,
) -> SAM3SemanticMaskBundle:
    if mode not in SCAIL2_MODES:
        raise ValueError(f"mode must be one of {', '.join(SCAIL2_MODES)}")
    if not track_masks:
        raise ValueError("track_masks must contain at least one track")
    if expected_track_count is not None and len(track_masks) != expected_track_count:
        raise ValueError(
            f"track count mismatch: expected {expected_track_count}, got {len(track_masks)}"
        )

    normalized_masks = tuple(
        _normalize_bool_track_mask(mask, track_name=f"track_masks[{index}]")
        for index, mask in enumerate(track_masks)
    )
    frame_count = len(normalized_masks[0])
    height = len(normalized_masks[0][0])
    width = len(normalized_masks[0][0][0])
    for mask in normalized_masks:
        if len(mask) != frame_count:
            raise ValueError("all SAM3 tracks must have the same frame count")
        if len(mask[0]) != height or len(mask[0][0]) != width:
            raise ValueError("all SAM3 tracks must have the same spatial dimensions")

    ids = tuple(str(item) for item in (track_ids or range(len(normalized_masks))))
    if len(ids) != len(normalized_masks):
        raise ValueError("track_ids length must match track_masks length")
    track_colors = _normalize_colors(colors, len(normalized_masks))
    background = _validate_color(background_rgb)

    frames = [
        [[background for _col in range(width)] for _row in range(height)]
        for _frame in range(frame_count)
    ]
    metadata = []
    for track_index, mask in enumerate(normalized_masks):
        color = track_colors[track_index]
        pixels_per_frame = []
        for frame_index, frame in enumerate(mask):
            count = 0
            for row_index, row in enumerate(frame):
                for col_index, active in enumerate(row):
                    if active:
                        frames[frame_index][row_index][col_index] = color
                        count += 1
            pixels_per_frame.append(count)
        metadata.append(
            SAM3TrackMetadata(
                track_id=ids[track_index],
                color_rgb=color,
                pixels_per_frame=tuple(pixels_per_frame),
            )
        )

    frozen_frames = tuple(
        tuple(tuple(row) for row in frame)
        for frame in frames
    )
    semantic_mask_indices(frozen_frames)
    return SAM3SemanticMaskBundle(
        mode=mode,
        frames=frozen_frames,
        track_metadata=tuple(metadata),
        frame_count=frame_count,
        height=height,
        width=width,
        background_rgb=background,
    )


def build_condition_from_sam3_tracks(
    *,
    mode: SCAIL2Mode,
    ref_image: Any,
    ref_track_masks: Sequence[Sequence[Sequence[Sequence[Any]]]],
    pose_video: Any,
    driving_track_masks: Sequence[Sequence[Sequence[Sequence[Any]]]],
    width: Any,
    height: Any,
    pose_frame_count: Any | None = None,
    expected_ref_track_count: int | None = None,
    expected_driving_track_count: int | None = None,
    ref_track_ids: Sequence[Any] | None = None,
    driving_track_ids: Sequence[Any] | None = None,
    ref_background_rgb: Sequence[Any] = BLACK_RGB,
    driving_background_rgb: Sequence[Any] = BLACK_RGB,
    segment_len: Any = 81,
    segment_overlap: Any = 5,
) -> SAM3ConditionPayload:
    ref_bundle = sam3_tracks_to_semantic_mask_frames(
        ref_track_masks,
        mode=mode,
        track_ids=ref_track_ids,
        expected_track_count=expected_ref_track_count,
        background_rgb=ref_background_rgb,
    )
    driving_bundle = sam3_tracks_to_semantic_mask_frames(
        driving_track_masks,
        mode=mode,
        track_ids=driving_track_ids,
        expected_track_count=expected_driving_track_count,
        background_rgb=driving_background_rgb,
    )
    frames = driving_bundle.frame_count if pose_frame_count is None else pose_frame_count
    condition = build_user_mask_condition(
        mode=mode,
        ref_image=ref_image,
        ref_mask_frames=ref_bundle.frames,
        pose_video=pose_video,
        pose_frame_count=frames,
        driving_mask_frames=driving_bundle.frames,
        width=width,
        height=height,
        segment_len=segment_len,
        segment_overlap=segment_overlap,
    )
    return SAM3ConditionPayload(
        condition=condition,
        ref_mask_bundle=ref_bundle,
        driving_mask_bundle=driving_bundle,
        mode=mode,
    )
