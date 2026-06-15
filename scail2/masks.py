"""Pure Python SCAIL-2 RGB semantic mask utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


BACKGROUND_INDEX = -1
MASK_ON_THRESHOLD = 225
MASK_OFF_THRESHOLD = 30
TEMPORAL_COMPRESSION_STRIDE = 4


@dataclass(frozen=True)
class SemanticMaskColor:
    name: str
    rgb: tuple[int, int, int]
    index: int


SEMANTIC_MASK_COLORS: tuple[SemanticMaskColor, ...] = (
    SemanticMaskColor("white", (255, 255, 255), 0),
    SemanticMaskColor("red", (255, 0, 0), 1),
    SemanticMaskColor("green", (0, 255, 0), 2),
    SemanticMaskColor("blue", (0, 0, 255), 3),
    SemanticMaskColor("yellow", (255, 255, 0), 4),
    SemanticMaskColor("magenta", (255, 0, 255), 5),
    SemanticMaskColor("cyan", (0, 255, 255), 6),
)
SEMANTIC_MASK_COLOR_NAMES: tuple[str, ...] = tuple(
    color.name for color in SEMANTIC_MASK_COLORS
)

_ACTIVE_BITS_TO_INDEX = {
    (True, True, True): 0,
    (True, False, False): 1,
    (False, True, False): 2,
    (False, False, True): 3,
    (True, True, False): 4,
    (True, False, True): 5,
    (False, True, True): 6,
}


@dataclass(frozen=True)
class MaskIndexShape:
    frames: int
    height: int
    width: int


@dataclass(frozen=True)
class MaskLatent28:
    data: tuple[Any, ...]
    frame_count: int
    latent_frame_count: int
    height: int
    width: int
    temporal_stride: int = TEMPORAL_COMPRESSION_STRIDE
    color_order: tuple[str, ...] = SEMANTIC_MASK_COLOR_NAMES

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (28, self.latent_frame_count, self.height, self.width)

    def value(self, channel: int, latent_frame: int, row: int = 0, col: int = 0) -> int:
        return self.data[channel][latent_frame][row][col]


def _to_raw_rgb(rgb: Sequence[Any]) -> tuple[int, int, int]:
    if len(rgb) < 3:
        raise ValueError("RGB semantic mask pixels must have at least 3 channels")
    raw_values = []
    for value in rgb[:3]:
        if not isinstance(value, (int, float)):
            raise ValueError("RGB semantic mask channels must be numeric")
        raw_values.append(float(value))
    if all(0.0 <= value <= 1.0 for value in raw_values):
        raw_values = [value * 255.0 for value in raw_values]
    rounded = tuple(int(round(value)) for value in raw_values)
    if any(value < 0 or value > 255 for value in rounded):
        raise ValueError("RGB semantic mask channels must be in the 0..255 range")
    return rounded


def classify_rgb_semantic_color(rgb: Sequence[Any], *, strict: bool = True) -> int:
    raw = _to_raw_rgb(rgb)
    active = []
    for channel_value in raw:
        if channel_value >= MASK_ON_THRESHOLD:
            active.append(True)
        elif channel_value <= MASK_OFF_THRESHOLD:
            active.append(False)
        elif strict:
            raise ValueError(
                "RGB semantic mask channel is ambiguous; use solid palette colors"
            )
        else:
            active.append(False)
    active_bits = tuple(active)
    if active_bits == (False, False, False):
        return BACKGROUND_INDEX
    return _ACTIVE_BITS_TO_INDEX[active_bits]


def semantic_mask_indices(
    frames: Sequence[Sequence[Sequence[Sequence[Any]]]],
    *,
    strict: bool = True,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if not frames:
        raise ValueError("semantic mask must contain at least one frame")

    converted_frames = []
    expected_height: int | None = None
    expected_width: int | None = None

    for frame_index, frame in enumerate(frames):
        if not frame:
            raise ValueError(f"semantic mask frame {frame_index} is empty")
        converted_rows = []
        frame_height = len(frame)
        frame_width = len(frame[0])
        if frame_width == 0:
            raise ValueError(f"semantic mask frame {frame_index} has empty rows")
        if expected_height is None:
            expected_height = frame_height
            expected_width = frame_width
        elif frame_height != expected_height or frame_width != expected_width:
            raise ValueError("semantic mask frames must have consistent dimensions")
        for row in frame:
            if len(row) != frame_width:
                raise ValueError("semantic mask rows must have consistent width")
            converted_rows.append(
                tuple(
                    classify_rgb_semantic_color(pixel, strict=strict)
                    for pixel in row
                )
            )
        converted_frames.append(tuple(converted_rows))

    return tuple(converted_frames)


def mask_indices_shape(
    indices: Sequence[Sequence[Sequence[int]]],
) -> MaskIndexShape:
    if not indices:
        raise ValueError("mask indices must contain at least one frame")
    height = len(indices[0])
    if height == 0:
        raise ValueError("mask index frame must not be empty")
    width = len(indices[0][0])
    if width == 0:
        raise ValueError("mask index rows must not be empty")
    for frame in indices:
        if len(frame) != height:
            raise ValueError("mask index frames must have consistent height")
        for row in frame:
            if len(row) != width:
                raise ValueError("mask index rows must have consistent width")
            for value in row:
                if value != BACKGROUND_INDEX and not 0 <= int(value) <= 6:
                    raise ValueError("mask index values must be background or 0..6")
    return MaskIndexShape(frames=len(indices), height=height, width=width)


def _empty_latent_data(
    latent_frame_count: int,
    height: int,
    width: int,
) -> list[list[list[list[int]]]]:
    return [
        [
            [[0 for _col in range(width)] for _row in range(height)]
            for _latent in range(latent_frame_count)
        ]
        for _channel in range(28)
    ]


def _freeze_latent_data(data: list[list[list[list[int]]]]) -> tuple[Any, ...]:
    return tuple(
        tuple(tuple(tuple(row) for row in latent_frame) for latent_frame in channel)
        for channel in data
    )


def pack_semantic_mask_indices_to_28_channels(
    indices: Sequence[Sequence[Sequence[int]]],
    *,
    temporal_stride: int = TEMPORAL_COMPRESSION_STRIDE,
    require_vae_alignment: bool = True,
) -> MaskLatent28:
    if temporal_stride != TEMPORAL_COMPRESSION_STRIDE:
        raise ValueError("SCAIL-2 mask packing uses temporal stride 4")
    shape = mask_indices_shape(indices)
    if require_vae_alignment and (shape.frames - 1) % temporal_stride != 0:
        raise ValueError("SCAIL-2 mask frame count must be 4n+1 for strict packing")

    latent_frame_count = (shape.frames - 1) // temporal_stride + 1
    padded_frames = [indices[0]] * temporal_stride + list(indices[1:])
    target_frame_count = latent_frame_count * temporal_stride
    if len(padded_frames) < target_frame_count:
        padded_frames.extend([indices[-1]] * (target_frame_count - len(padded_frames)))
    if len(padded_frames) > target_frame_count:
        padded_frames = padded_frames[:target_frame_count]

    data = _empty_latent_data(latent_frame_count, shape.height, shape.width)
    for latent_frame in range(latent_frame_count):
        for slot in range(temporal_stride):
            frame = padded_frames[latent_frame * temporal_stride + slot]
            for row_index, row in enumerate(frame):
                for col_index, color_index in enumerate(row):
                    if color_index == BACKGROUND_INDEX:
                        continue
                    channel = slot * 7 + int(color_index)
                    data[channel][latent_frame][row_index][col_index] = 1

    return MaskLatent28(
        data=_freeze_latent_data(data),
        frame_count=shape.frames,
        latent_frame_count=latent_frame_count,
        height=shape.height,
        width=shape.width,
        temporal_stride=temporal_stride,
    )
