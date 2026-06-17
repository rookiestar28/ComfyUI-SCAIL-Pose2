"""Versioned SCAIL-2 condition payloads for WanVideoWrapper-oriented workflows."""

from __future__ import annotations

from typing import Any

from .condition import SCAIL2Condition, TYPE_SCAIL2_CONDITION
from .masks import pack_semantic_mask_indices_to_runtime_28_channels
from .wanvideo_contracts import UNSUPPORTED_CURRENT_WAN_SCAIL2_FEATURES


SCAIL2_TO_WANVIDEO_V1_SEMANTIC_LOSSES: tuple[str, ...] = (
    *UNSUPPORTED_CURRENT_WAN_SCAIL2_FEATURES,
    "source_kind_metadata",
    "previous_frame_continuation",
    "video_frame_offset",
)


def _require_scail2_condition(condition: Any) -> SCAIL2Condition:
    if not isinstance(condition, SCAIL2Condition):
        type_name = getattr(condition, "type_name", None)
        if type_name != TYPE_SCAIL2_CONDITION:
            raise ValueError("WanVideo SCAIL-2 adapter requires a SCAIL2_CONDITION")
    return condition


def _runtime_masks_for_condition(condition: SCAIL2Condition) -> dict[str, Any]:
    additional = tuple(
        pack_semantic_mask_indices_to_runtime_28_channels(item.mask_indices)
        for item in condition.additional_references
    )
    return {
        "reference": pack_semantic_mask_indices_to_runtime_28_channels(
            condition.ref_mask_indices
        ),
        "driving": pack_semantic_mask_indices_to_runtime_28_channels(
            condition.driving_mask_indices
        ),
        "additional_references": additional,
    }


def _degraded_v1_summary(condition: SCAIL2Condition) -> dict[str, Any]:
    return {
        "kind": "wan_scail_v1_lossy_condition_summary",
        "full_scail2_parity": False,
        "current_wrapper_path": "v1_scail_embeds",
        "width": condition.width,
        "height": condition.height,
        "num_frames": condition.num_frames,
        "ref_image": condition.ref_image,
        "pose_images": condition.pose_video,
        "mode": condition.mode,
        "replace_flag": condition.replace_flag,
    }


def build_wanvideo_scail2_adapter_payload(
    condition: Any,
    *,
    degrade_to_v1: bool = False,
    allow_degradation: bool = False,
) -> dict[str, Any]:
    scail2_condition = _require_scail2_condition(condition)
    if degrade_to_v1 and not allow_degradation:
        raise ValueError(
            "Lossy WanVideoWrapper v1 degradation requires allow_degradation=True"
        )

    runtime_masks = _runtime_masks_for_condition(scail2_condition)
    semantic_losses = (
        SCAIL2_TO_WANVIDEO_V1_SEMANTIC_LOSSES if degrade_to_v1 else ()
    )
    degraded_payload = (
        _degraded_v1_summary(scail2_condition) if degrade_to_v1 else None
    )

    return {
        "kind": "wanvideo_scail2_condition_adapter",
        "version": 1,
        "condition": scail2_condition,
        "target": {
            "wrapper_family": "ComfyUI-WanVideoWrapper",
            "current_wrapper_path": "v1_scail_embeds",
            "requires_wrapper_scail2_support": True,
            "live_wrapper_supported": False,
        },
        "mode": scail2_condition.mode,
        "replace_flag": scail2_condition.replace_flag,
        "dimensions": {
            "width": scail2_condition.width,
            "height": scail2_condition.height,
            "num_frames": scail2_condition.num_frames,
        },
        "source": {
            "source_kind": scail2_condition.source_kind,
            "previous_frame_count": scail2_condition.previous_frame_count,
            "video_frame_offset": scail2_condition.video_frame_offset,
        },
        "segment": {
            "segment_len": scail2_condition.segment_len,
            "segment_overlap": scail2_condition.segment_overlap,
        },
        "rgb_masks": {
            "reference_indices": scail2_condition.ref_mask_indices,
            "driving_indices": scail2_condition.driving_mask_indices,
            "palette": scail2_condition.mask_palette,
        },
        "runtime_masks": runtime_masks,
        "additional_references": scail2_condition.additional_references,
        "unsupported_current_wrapper_features": (
            scail2_condition.unsupported_wrapper_features
        ),
        "degraded": bool(degrade_to_v1),
        "semantic_losses": semantic_losses,
        "degraded_payload": degraded_payload,
    }


def summarize_wanvideo_scail2_payload(payload: dict[str, Any]) -> str:
    target = payload["target"]
    return (
        f"{payload['kind']} v{payload['version']} "
        f"live_wrapper_supported={target['live_wrapper_supported']} "
        f"degraded={payload['degraded']}"
    )
