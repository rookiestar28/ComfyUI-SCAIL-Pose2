"""Identity-slot helpers for multi-person SCAIL-Pose2 routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


SCAIL2_IDENTITY_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
)

SCAIL2_IDENTITY_COLOR_NAMES: tuple[str, ...] = (
    "blue",
    "red",
    "green",
    "magenta",
    "cyan",
    "yellow",
)


@dataclass(frozen=True)
class IdentitySlot:
    """One selected subject slot in deterministic workflow order."""

    identity_id: int
    source_object_index: int
    color_name: str
    color_rgb: tuple[float, float, float]
    first_frame: int
    centroid_x: float
    area_ratio: float
    frame_count: int
    source: str = "sam3_object_order"


@dataclass(frozen=True)
class IdentityContractDiagnostics:
    slots: tuple[IdentitySlot, ...]
    warnings: tuple[str, ...] = ()

    @property
    def identity_count(self) -> int:
        return len(self.slots)

    @property
    def selected_source_indices(self) -> tuple[int, ...]:
        return tuple(slot.source_object_index for slot in self.slots)

    def summary(self) -> str:
        warning_text = ",".join(self.warnings) if self.warnings else "none"
        selected = ",".join(str(index) for index in self.selected_source_indices)
        if not selected:
            selected = "none"
        return (
            "identity_contract "
            f"slots={self.identity_count} selected={selected} warnings={warning_text}"
        )


def _identity_color(identity_id: int) -> tuple[str, tuple[float, float, float]]:
    palette_index = int(identity_id) % len(SCAIL2_IDENTITY_COLORS)
    return (
        SCAIL2_IDENTITY_COLOR_NAMES[palette_index],
        SCAIL2_IDENTITY_COLORS[palette_index],
    )


def _normalized_rgb(pixel: Any) -> tuple[float, float, float]:
    rgb = tuple(float(channel) for channel in pixel[:3])
    if any(channel > 1.0 for channel in rgb):
        return tuple(channel / 255.0 for channel in rgb)
    return rgb


def build_identity_slots(
    *,
    object_order: Sequence[int],
    object_stats: Sequence[tuple[int, float, float]],
    frame_count: int,
    object_count: int,
    source: str = "sam3_object_order",
) -> IdentityContractDiagnostics:
    """Build selected identity slots from Colored Mask object ordering.

    ``object_order`` is already sorted and filtered. Its position becomes the
    semantic identity id and therefore the SCAIL-2 palette color.
    """

    warnings: list[str] = []
    slots: list[IdentitySlot] = []
    if int(frame_count) <= 0:
        warnings.append("invalid_frame_count")
    for identity_id, raw_object_index in enumerate(object_order):
        try:
            object_index = int(raw_object_index)
        except (TypeError, ValueError):
            warnings.append("invalid_object_index")
            continue
        if object_index < 0 or object_index >= int(object_count):
            warnings.append("object_index_out_of_range")
            continue
        if object_index >= len(object_stats):
            warnings.append("missing_object_stats")
            continue
        first_frame, centroid_x, area_ratio = object_stats[object_index]
        color_name, color_rgb = _identity_color(identity_id)
        slots.append(
            IdentitySlot(
                identity_id=identity_id,
                source_object_index=object_index,
                color_name=color_name,
                color_rgb=color_rgb,
                first_frame=int(first_frame),
                centroid_x=float(centroid_x),
                area_ratio=float(area_ratio),
                frame_count=max(int(frame_count), 0),
                source=source,
            )
        )
    if object_count > 1 and len(slots) == 1:
        warnings.append("single_identity_selected_from_multi_object_track")
    if object_count > 1 and not slots:
        warnings.append("no_identity_selected_from_multi_object_track")
    return IdentityContractDiagnostics(
        slots=tuple(slots),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def identity_count_from_semantic_mask(value: Any) -> int:
    """Count SCAIL-2 palette colors present in an RGB semantic mask."""

    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach()
        view = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
        if view.ndim != 4 or view.shape[-1] < 3:
            raise ValueError("semantic mask tensor must have shape [H, W, C] or [T, H, W, C]")
        count = 0
        rgb = view[..., :3].float()
        normalized = bool(((rgb >= 0.0) & (rgb <= 1.0)).all().item())
        scale = 1.0 if normalized else 255.0
        tolerance = 0.01 if normalized else 2.55
        for color in SCAIL2_IDENTITY_COLORS:
            target = torch.tensor(
                tuple(channel * scale for channel in color),
                device=rgb.device,
                dtype=rgb.dtype,
            )
            active = (rgb - target.view(1, 1, 1, 3)).abs().amax(dim=-1) <= tolerance
            if bool(active.any().item()):
                count += 1
        return count

    frames = value
    if not frames:
        return 0
    if frames and frames[0] and frames[0][0] and isinstance(frames[0][0][0], (int, float)):
        frames = [frames]
    present: set[tuple[float, float, float]] = set()
    for frame in frames:
        for row in frame:
            for pixel in row:
                raw = _normalized_rgb(pixel)
                for color in SCAIL2_IDENTITY_COLORS:
                    if all(abs(raw[index] - color[index]) <= 0.01 for index in range(3)):
                        present.add(color)
    return len(present)


def semantic_identity_rgb_mask(value: Any, *, identity_index: int) -> Any:
    """Return an RGB semantic mask containing only one identity color."""

    index = int(identity_index)
    if index < 0:
        raise ValueError("identity_index must be non-negative")
    color = SCAIL2_IDENTITY_COLORS[index % len(SCAIL2_IDENTITY_COLORS)]
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach()
        single = tensor.ndim == 3
        view = tensor.unsqueeze(0) if single else tensor
        if view.ndim != 4 or view.shape[-1] < 3:
            raise ValueError("semantic mask tensor must have shape [H, W, C] or [T, H, W, C]")
        rgb = view[..., :3].float()
        normalized = bool(((rgb >= 0.0) & (rgb <= 1.0)).all().item())
        scale = 1.0 if normalized else 255.0
        tolerance = 0.01 if normalized else 2.55
        target = torch.tensor(
            tuple(channel * scale for channel in color),
            device=rgb.device,
            dtype=rgb.dtype,
        )
        active = (rgb - target.view(1, 1, 1, 3)).abs().amax(dim=-1) <= tolerance
        output = torch.zeros_like(view)
        output[..., :3] = target.view(1, 1, 1, 3)
        output = torch.where(active.unsqueeze(-1), output, torch.zeros_like(output))
        return output.squeeze(0) if single else output

    frames = value
    single = False
    if frames and frames[0] and frames[0][0] and isinstance(frames[0][0][0], (int, float)):
        frames = [frames]
        single = True
    output_frames = []
    for frame in frames:
        rows = []
        for row in frame:
            pixels = []
            for pixel in row:
                raw = _normalized_rgb(pixel)
                if all(abs(raw[channel] - color[channel]) <= 0.01 for channel in range(3)):
                    pixels.append(color)
                else:
                    pixels.append((0.0, 0.0, 0.0))
            rows.append(tuple(pixels))
        output_frames.append(tuple(rows))
    result = tuple(output_frames)
    return result[0] if single else result
