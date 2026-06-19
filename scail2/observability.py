"""Safe observability helpers for ComfyUI-facing SCAIL-2 nodes."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def terminal_info(message: str) -> None:
    print(f"[SCAIL-Pose2] {message}", flush=True)


def perf_counter_ms() -> float:
    return time.perf_counter() * 1000.0


def elapsed_ms(start_ms: float) -> float:
    return perf_counter_ms() - start_ms


def safe_shape(value: Any, *, max_depth: int = 4) -> tuple[Any, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(part) for part in shape)
        except (TypeError, ValueError):
            return tuple(shape)
    if isinstance(value, (str, bytes, Mapping)):
        return None
    if isinstance(value, Sequence):
        dims = []
        current = value
        depth = 0
        while (
            isinstance(current, Sequence)
            and not isinstance(current, (str, bytes))
            and depth < max_depth
        ):
            dims.append(len(current))
            if not current:
                break
            current = current[0]
            depth += 1
        return tuple(dims)
    return None


def safe_value_summary(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    shape = safe_shape(value)
    if shape is not None:
        summary["shape"] = list(shape)
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        summary["dtype"] = str(dtype)
    device = getattr(value, "device", None)
    if device is not None:
        summary["device"] = str(device)
    return summary


@dataclass
class ProgressReporter:
    total: int
    _bar: Any = None

    def __post_init__(self) -> None:
        try:
            from comfy.utils import ProgressBar
        except Exception:
            self._bar = None
            return
        self._bar = ProgressBar(max(int(self.total), 1))

    def update(self, amount: int = 1) -> None:
        if self._bar is None:
            return
        for _ in range(max(int(amount), 0)):
            self._bar.update(1)


def make_progress(total: int) -> ProgressReporter:
    return ProgressReporter(total=max(int(total), 1))
