"""Shared replacement mask diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MaskDiagnostics:
    subject_ratio: float
    edge_contact_frames: int
    edge_contact_ratio: float
    min_bbox_margin_ratio: float
    worst_frame: int


def _torch_required() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("torch is required to compute replacement diagnostics") from exc
    return torch


def mask_diagnostics(mask: Any) -> MaskDiagnostics:
    torch = _torch_required()
    tensor = mask.detach() if hasattr(mask, "detach") else torch.as_tensor(mask)
    if tensor.ndim != 3:
        raise ValueError("replacement mask diagnostics require shape [frames, height, width]")
    active = tensor > 0.5
    frames, height, width = (int(part) for part in active.shape)
    edge_contact_frames = 0
    min_margin = 1.0
    worst_frame = 0
    denom = float(max(height, width, 1))
    for frame_index in range(frames):
        coords = torch.nonzero(active[frame_index], as_tuple=False)
        if coords.numel() == 0:
            continue
        y0 = int(coords[:, 0].min().item())
        y1 = int(coords[:, 0].max().item())
        x0 = int(coords[:, 1].min().item())
        x1 = int(coords[:, 1].max().item())
        touches_edge = y0 == 0 or x0 == 0 or y1 == height - 1 or x1 == width - 1
        if touches_edge:
            edge_contact_frames += 1
        margin = min(y0, x0, height - 1 - y1, width - 1 - x1) / denom
        if margin < min_margin:
            min_margin = float(margin)
            worst_frame = frame_index
    return MaskDiagnostics(
        subject_ratio=float(tensor.to(dtype=torch.float32).mean().item()),
        edge_contact_frames=edge_contact_frames,
        edge_contact_ratio=edge_contact_frames / max(frames, 1),
        min_bbox_margin_ratio=float(min_margin),
        worst_frame=worst_frame,
    )


def diagnostics_summary_fragment(diagnostics: MaskDiagnostics) -> str:
    return (
        f"subject_ratio={diagnostics.subject_ratio:.6f} "
        f"edge_contact_frames={diagnostics.edge_contact_frames} "
        f"edge_contact_ratio={diagnostics.edge_contact_ratio:.6f} "
        f"min_bbox_margin_ratio={diagnostics.min_bbox_margin_ratio:.6f} "
        f"worst_frame={diagnostics.worst_frame}"
    )
