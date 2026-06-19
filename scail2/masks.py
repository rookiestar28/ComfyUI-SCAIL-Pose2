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


@dataclass(frozen=True)
class RuntimeMaskLatent28:
    data: tuple[Any, ...]
    frame_count: int
    latent_frame_count: int
    source_height: int
    source_width: int
    latent_height: int
    latent_width: int
    temporal_stride: int = TEMPORAL_COMPRESSION_STRIDE
    spatial_downsample: int = 8
    color_order: tuple[str, ...] = SEMANTIC_MASK_COLOR_NAMES

    @property
    def comfy_shape(self) -> tuple[int, int, int, int, int]:
        return (1, self.latent_frame_count, 28, self.latent_height, self.latent_width)

    @property
    def scail2_shape(self) -> tuple[int, int, int, int]:
        return (28, self.latent_frame_count, self.latent_height, self.latent_width)

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return self.comfy_shape

    def value(
        self,
        *,
        latent_frame: int,
        channel: int,
        row: int = 0,
        col: int = 0,
        batch: int = 0,
    ) -> float:
        return float(self.data[batch][latent_frame][channel][row][col])


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


def _freeze_index_frames(index_frames: Any) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple(tuple(int(item) for item in row) for row in frame)
        for frame in index_frames
    )


def semantic_mask_indices_tensor(
    frames: Any,
    *,
    strict: bool = True,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("torch is required for tensor semantic masks") from exc

    tensor = frames.detach() if hasattr(frames, "detach") else torch.as_tensor(frames)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[-1] < 3:
        raise ValueError("semantic mask tensor must have shape [frames, height, width, channels]")
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
        raise ValueError("semantic mask tensor must be non-empty")

    rgb = tensor[..., :3].to(dtype=torch.float32)
    if bool(torch.logical_and(rgb >= 0.0, rgb <= 1.0).all().item()):
        rgb = rgb * 255.0
    if bool(torch.logical_or(rgb < 0.0, rgb > 255.0).any().item()):
        raise ValueError("RGB semantic mask channels must be in the 0..255 range")

    active = rgb >= MASK_ON_THRESHOLD
    inactive = rgb <= MASK_OFF_THRESHOLD
    if strict and bool((~(active | inactive)).any().item()):
        raise ValueError(
            "RGB semantic mask channel is ambiguous; use solid palette colors"
        )
    active = torch.where(active, active, torch.zeros_like(active, dtype=torch.bool))
    r = active[..., 0]
    g = active[..., 1]
    b = active[..., 2]

    indices = torch.full(r.shape, BACKGROUND_INDEX, dtype=torch.int16, device=r.device)
    indices[r & g & b] = 0
    indices[r & ~g & ~b] = 1
    indices[~r & g & ~b] = 2
    indices[~r & ~g & b] = 3
    indices[r & g & ~b] = 4
    indices[r & ~g & b] = 5
    indices[~r & g & b] = 6

    return _freeze_index_frames(indices.cpu().tolist())


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


def _positive_size(name: str, value: int) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _latent_spatial_size(size: int, spatial_downsample: int) -> int:
    if spatial_downsample != 8:
        raise ValueError("SCAIL-2 runtime mask packing uses spatial downsample 8")
    latent_size = _positive_size("size", size)
    for _ in range(3):
        latent_size = (latent_size + 1) // 2
    return latent_size


def latent_spatial_size_for_pixels(
    *,
    height: int,
    width: int,
    spatial_downsample: int = 8,
) -> tuple[int, int]:
    return (
        _latent_spatial_size(height, spatial_downsample),
        _latent_spatial_size(width, spatial_downsample),
    )


def pose_control_latent_spatial_size(
    *,
    height: int,
    width: int,
    spatial_downsample: int = 8,
) -> tuple[int, int]:
    control_height = max(_positive_size("height", height) // 2, 1)
    control_width = max(_positive_size("width", width) // 2, 1)
    return latent_spatial_size_for_pixels(
        height=control_height,
        width=control_width,
        spatial_downsample=spatial_downsample,
    )


def _downsample_color_plane(
    frame: Sequence[Sequence[int]],
    *,
    color_index: int,
    latent_height: int,
    latent_width: int,
) -> tuple[tuple[float, ...], ...]:
    source_height = len(frame)
    source_width = len(frame[0])
    downsampled = []
    for latent_row in range(latent_height):
        source_row_start = (latent_row * source_height) // latent_height
        source_row_end = ((latent_row + 1) * source_height) // latent_height
        if latent_row == latent_height - 1:
            source_row_end = source_height
        source_row_end = max(source_row_end, source_row_start + 1)

        row_values = []
        for latent_col in range(latent_width):
            source_col_start = (latent_col * source_width) // latent_width
            source_col_end = ((latent_col + 1) * source_width) // latent_width
            if latent_col == latent_width - 1:
                source_col_end = source_width
            source_col_end = max(source_col_end, source_col_start + 1)

            area = (source_row_end - source_row_start) * (
                source_col_end - source_col_start
            )
            active = 0
            for source_row in range(source_row_start, source_row_end):
                for source_col in range(source_col_start, source_col_end):
                    if int(frame[source_row][source_col]) == color_index:
                        active += 1
            row_values.append(active / area)
        downsampled.append(tuple(row_values))
    return tuple(downsampled)


def _downsample_semantic_frame(
    frame: Sequence[Sequence[int]],
    *,
    latent_height: int,
    latent_width: int,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(
        _downsample_color_plane(
            frame,
            color_index=color_index,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        for color_index in range(7)
    )


def _empty_runtime_data(
    latent_frame_count: int,
    height: int,
    width: int,
) -> list[list[list[list[list[float]]]]]:
    return [
        [
            [
                [[0.0 for _col in range(width)] for _row in range(height)]
                for _channel in range(28)
            ]
            for _latent in range(latent_frame_count)
        ]
    ]


def _freeze_runtime_data(data: list[list[list[list[list[float]]]]]) -> tuple[Any, ...]:
    return tuple(
        tuple(
            tuple(tuple(tuple(row) for row in channel) for channel in latent_frame)
            for latent_frame in batch
        )
        for batch in data
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


def pack_semantic_mask_indices_to_runtime_28_channels(
    indices: Sequence[Sequence[Sequence[int]]],
    *,
    temporal_stride: int = TEMPORAL_COMPRESSION_STRIDE,
    spatial_downsample: int = 8,
    require_vae_alignment: bool = True,
) -> RuntimeMaskLatent28:
    if temporal_stride != TEMPORAL_COMPRESSION_STRIDE:
        raise ValueError("SCAIL-2 mask packing uses temporal stride 4")
    shape = mask_indices_shape(indices)
    if require_vae_alignment and (shape.frames - 1) % temporal_stride != 0:
        raise ValueError("SCAIL-2 mask frame count must be 4n+1 for strict packing")

    latent_height = _latent_spatial_size(shape.height, spatial_downsample)
    latent_width = _latent_spatial_size(shape.width, spatial_downsample)
    latent_frame_count = (shape.frames - 1) // temporal_stride + 1

    downsampled_frames = [
        _downsample_semantic_frame(
            frame,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        for frame in indices
    ]
    padded_frames = [downsampled_frames[0]] * temporal_stride + downsampled_frames[1:]
    target_frame_count = latent_frame_count * temporal_stride
    if len(padded_frames) < target_frame_count:
        padded_frames.extend(
            [downsampled_frames[-1]] * (target_frame_count - len(padded_frames))
        )
    if len(padded_frames) > target_frame_count:
        padded_frames = padded_frames[:target_frame_count]

    data = _empty_runtime_data(latent_frame_count, latent_height, latent_width)
    for latent_frame in range(latent_frame_count):
        for slot in range(temporal_stride):
            frame_planes = padded_frames[latent_frame * temporal_stride + slot]
            for color_index, plane in enumerate(frame_planes):
                channel = slot * 7 + color_index
                for row_index, row in enumerate(plane):
                    for col_index, value in enumerate(row):
                        data[0][latent_frame][channel][row_index][col_index] = value

    return RuntimeMaskLatent28(
        data=_freeze_runtime_data(data),
        frame_count=shape.frames,
        latent_frame_count=latent_frame_count,
        source_height=shape.height,
        source_width=shape.width,
        latent_height=latent_height,
        latent_width=latent_width,
        temporal_stride=temporal_stride,
        spatial_downsample=spatial_downsample,
    )


def runtime_mask_to_torch(
    runtime_mask: RuntimeMaskLatent28,
    *,
    layout: str = "comfy",
):
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("torch is required to materialize runtime mask tensors") from exc

    tensor = torch.tensor(runtime_mask.data, dtype=torch.float32)
    if layout == "comfy":
        return tensor
    if layout == "scail2":
        return tensor[0].permute(1, 0, 2, 3)
    raise ValueError("layout must be 'comfy' or 'scail2'")
