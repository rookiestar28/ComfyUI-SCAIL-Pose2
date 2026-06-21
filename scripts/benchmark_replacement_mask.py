"""Synthetic benchmark for SCAIL-Pose2 replacement denoise mask generation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scail2.condition import SCAIL2Condition, TYPE_SCAIL2_CONDITION
from scail2.replacement_mask import (
    _build_tensor_subject_mask,
    _blur_mask,
    _grow_mask,
    _indices_to_subject_tensor,
    build_replacement_denoise_mask,
)
from scail2.masks import semantic_mask_indices_tensor_raw


def _solid_frame(rgb: tuple[int, int, int], *, height: int, width: int):
    return [[rgb for _col in range(width)] for _row in range(height)]


def _condition(*, frames: int, height: int, width: int):
    return SCAIL2Condition(
        type_name=TYPE_SCAIL2_CONDITION,
        mode="replacement",
        replace_flag=True,
        width=width,
        height=height,
        num_frames=frames,
        ref_image="ref",
        ref_mask_indices=torch.zeros((1, height, width), dtype=torch.int8),
        pose_video="pose",
        driving_mask_indices=torch.zeros((frames, height, width), dtype=torch.int8),
        additional_references=(),
    )


def _mask(*, frames: int, height: int, width: int):
    mask = torch.zeros((frames, height, width, 3), dtype=torch.float32)
    row0 = height // 4
    row1 = max(row0 + 1, (height * 3) // 4)
    col0 = width // 4
    col1 = max(col0 + 1, (width * 3) // 4)
    mask[:, row0:row1, col0:col1, 2] = 1.0
    return mask


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_call(label: str, func):
    _sync()
    started = time.perf_counter()
    result = func()
    _sync()
    elapsed = (time.perf_counter() - started) * 1000.0
    print(f"{label}: elapsed_ms={elapsed:.2f} summary={result.summary}")
    return elapsed, result


def _cpu_semantic_baseline(pose_mask, *, condition, grow_pixels: int, blur_pixels: int):
    indices = semantic_mask_indices_tensor_raw(pose_mask)
    mask = _indices_to_subject_tensor(indices).to(device="cpu")
    raw_subject_pixels = float(mask.sum().item())
    if raw_subject_pixels <= 0:
        raise ValueError("pose_video_mask contains no subject pixels")
    mask = _grow_mask(mask, grow_pixels)
    mask = _blur_mask(mask, blur_pixels)
    mask = mask.clamp(0.0, 1.0).detach().to(device="cpu", dtype=torch.float32).contiguous()
    subject_ratio = float(mask.mean().item())

    class Result:
        summary = (
            "replacement_denoise_mask "
            f"mode={condition.mode} "
            f"frames={mask.shape[0]} "
            f"size={mask.shape[2]}x{mask.shape[1]} "
            f"subject_ratio={subject_ratio:.6f} "
            f"grow_pixels={grow_pixels} "
            f"blur_pixels={blur_pixels} "
            "invert=False "
            "fast_path=semantic_indices "
            "input_device=cpu work_device=cpu output_device=cpu"
        )

    return Result()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--grow-pixels", type=int, default=8)
    parser.add_argument("--blur-pixels", type=int, default=0)
    parser.add_argument("--skip-cpu-baseline", action="store_true")
    parser.add_argument("--skip-cpu-fast-path", action="store_true")
    args = parser.parse_args()

    condition = _condition(frames=args.frames, height=args.height, width=args.width)
    pose_mask = _mask(frames=args.frames, height=args.height, width=args.width)

    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(
        "case="
        f"frames={args.frames} height={args.height} width={args.width} "
        f"grow_pixels={args.grow_pixels} blur_pixels={args.blur_pixels}"
    )

    if not args.skip_cpu_baseline:
        _time_call(
            "cpu_semantic_baseline",
            lambda: _cpu_semantic_baseline(
                pose_mask,
                condition=condition,
                grow_pixels=args.grow_pixels,
                blur_pixels=args.blur_pixels,
            ),
        )

    if not args.skip_cpu_fast_path:
        _time_call(
            "cpu_tensor_fast_path",
            lambda: _build_tensor_subject_mask(
                pose_mask,
                condition=condition,
                grow_pixels=args.grow_pixels,
                blur_pixels=args.blur_pixels,
                invert=False,
                work_device="cpu",
            ),
        )

    _time_call(
        "selected_fast_path",
        lambda: build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_mask,
            grow_pixels=args.grow_pixels,
            blur_pixels=args.blur_pixels,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
