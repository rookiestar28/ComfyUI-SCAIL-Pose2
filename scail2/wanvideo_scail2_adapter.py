"""Versioned SCAIL-2 condition payloads for WanVideoWrapper-oriented workflows."""

from __future__ import annotations

from typing import Any

from .condition import SCAIL2Condition, TYPE_SCAIL2_CONDITION
from .masks import pack_semantic_mask_indices_to_runtime_28_channels
from .wanvideo_adapter import build_wan_scail_images_payload
from .wanvideo_contracts import UNSUPPORTED_CURRENT_WAN_SCAIL2_FEATURES


PAYLOAD_SCHEMA_NAME = "scail_pose2.wanvideo_scail2_payload"
PAYLOAD_SCHEMA_VERSION = 1
CONDITION_SCHEMA_VERSION = 1
NATIVE_WRAPPER_CONSUMER_NODE = "WanVideoAddSCAIL2ConditionEmbeds"
NATIVE_WRAPPER_EMBEDS_KEY = "scail2_embeds"
LEGACY_WRAPPER_EMBEDS_KEY = "scail_embeds"
NATIVE_WRAPPER_OUTPUT_TYPE = "WANVIDIMAGE_EMBEDS"
NATIVE_WRAPPER_PATH = "native_scail2_embeds"
LEGACY_WRAPPER_PATH = "v1_scail_embeds"

SCAIL2_TO_WANVIDEO_V1_SEMANTIC_LOSSES: tuple[str, ...] = (
    *UNSUPPORTED_CURRENT_WAN_SCAIL2_FEATURES,
    "source_kind_metadata",
)


def _require_scail2_condition(condition: Any) -> SCAIL2Condition:
    if not isinstance(condition, SCAIL2Condition):
        type_name = getattr(condition, "type_name", None)
        if type_name != TYPE_SCAIL2_CONDITION:
            raise ValueError("WanVideo SCAIL-2 adapter requires a SCAIL2_CONDITION")
    return condition


def _runtime_masks_for_condition(condition: SCAIL2Condition) -> dict[str, Any]:
    additional = tuple(
        pack_semantic_mask_indices_to_runtime_28_channels(
            item.mask_indices,
            layout_role="additional_reference",
        )
        for item in condition.additional_references
    )
    return {
        "reference": pack_semantic_mask_indices_to_runtime_28_channels(
            condition.ref_mask_indices,
            layout_role="reference",
        ),
        "driving": pack_semantic_mask_indices_to_runtime_28_channels(
            condition.driving_mask_indices,
            layout_role="driving",
        ),
        "additional_references": additional,
    }


def _runtime_mask_layout(mask: Any) -> dict[str, Any]:
    return {
        "object_type": "RuntimeMaskLatent28",
        "layout_role": mask.layout_role,
        "comfy_layout": {
            "axes": ["batch", "latent_frame", "channel", "height", "width"],
            "shape": list(mask.comfy_shape),
        },
        "scail2_layout": {
            "axes": ["channel", "latent_frame", "height", "width"],
            "shape": list(mask.scail2_shape),
        },
        "frame_count": mask.frame_count,
        "latent_frame_count": mask.latent_frame_count,
        "source_height": mask.source_height,
        "source_width": mask.source_width,
        "latent_height": mask.latent_height,
        "latent_width": mask.latent_width,
        "channel_count": 28,
        "color_channel_count": len(mask.color_order),
        "temporal_stride": mask.temporal_stride,
        "spatial_downsample": mask.spatial_downsample,
        "color_order": list(mask.color_order),
    }


def _payload_schema(
    runtime_masks: dict[str, Any],
    condition: SCAIL2Condition,
) -> dict[str, Any]:
    additional_layouts = [
        _runtime_mask_layout(mask)
        for mask in runtime_masks["additional_references"]
    ]
    return {
        "name": PAYLOAD_SCHEMA_NAME,
        "version": PAYLOAD_SCHEMA_VERSION,
        "condition": {
            "type_name": TYPE_SCAIL2_CONDITION,
            "version": CONDITION_SCHEMA_VERSION,
        },
        "native_wrapper": {
            "consumer_node": NATIVE_WRAPPER_CONSUMER_NODE,
            "consumer_input_type": "SCAIL2_WANVIDEO_PAYLOAD",
            "output_type": NATIVE_WRAPPER_OUTPUT_TYPE,
            "embeds_key": NATIVE_WRAPPER_EMBEDS_KEY,
            "legacy_embeds_key": LEGACY_WRAPPER_EMBEDS_KEY,
            "simultaneous_legacy_and_native": "reject",
        },
        "mask_packing": {
            "channel_count": 28,
            "color_channel_count": 7,
            "temporal_stride": 4,
            "spatial_downsample": 8,
            "color_order": list(condition.mask_palette),
            "requires_4n_plus_1_frames": True,
        },
        "mask_data_flow": {
            "native_runtime_masks_authoritative": True,
            "full_resolution_indices_in_native_payload": False,
            "raw_indices_available_on_condition_object": True,
        },
        "runtime_mask_layouts": {
            "reference": _runtime_mask_layout(runtime_masks["reference"]),
            "driving": _runtime_mask_layout(runtime_masks["driving"]),
            "additional_references": additional_layouts,
        },
        "additional_references": {
            "count": len(condition.additional_references),
            "requires_paired_image_and_mask": True,
            "mask_frame_count": 1,
        },
        "degradation": {
            "v1_requires_explicit_enable": True,
            "v1_full_scail2_parity": False,
            "v1_semantic_losses": list(SCAIL2_TO_WANVIDEO_V1_SEMANTIC_LOSSES),
        },
    }


def _degraded_v1_summary(condition: SCAIL2Condition) -> dict[str, Any]:
    return {
        "kind": "wan_scail_v1_lossy_condition_summary",
        "full_scail2_parity": False,
        "current_wrapper_path": LEGACY_WRAPPER_PATH,
        "width": condition.width,
        "height": condition.height,
        "num_frames": condition.num_frames,
        "ref_image": condition.ref_image,
        "pose_images": condition.pose_video,
        "mode": condition.mode,
        "replace_flag": condition.replace_flag,
    }


def _wan_scail_v1_images_for_condition(condition: SCAIL2Condition) -> dict[str, Any]:
    return build_wan_scail_images_payload(
        ref_image=condition.ref_image,
        pose_images=condition.pose_video,
        width=condition.width,
        height=condition.height,
        num_frames=condition.num_frames,
        clip_ref_image=condition.ref_image,
    )


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
    degraded_payload = _degraded_v1_summary(scail2_condition) if degrade_to_v1 else None
    wan_scail_v1_images = (
        _wan_scail_v1_images_for_condition(scail2_condition)
        if degrade_to_v1
        else None
    )

    return {
        "kind": "wanvideo_scail2_condition_adapter",
        "version": 1,
        "schema": _payload_schema(runtime_masks, scail2_condition),
        "condition": scail2_condition,
        "target": {
            "wrapper_family": "ComfyUI-WanVideoWrapper",
            "current_wrapper_path": NATIVE_WRAPPER_PATH,
            "fallback_wrapper_path": LEGACY_WRAPPER_PATH,
            "native_consumer_node": NATIVE_WRAPPER_CONSUMER_NODE,
            "native_embeds_key": NATIVE_WRAPPER_EMBEDS_KEY,
            "requires_wrapper_scail2_support": True,
            "live_wrapper_supported": True,
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
        },
        "rgb_masks": {
            "indices": "omitted_from_native_payload",
            "native_runtime_masks_authoritative": True,
            "palette": scail2_condition.mask_palette,
        },
        "runtime_masks": runtime_masks,
        "additional_references": scail2_condition.additional_references,
        "unsupported_current_wrapper_features": (),
        "legacy_v1_semantic_losses": scail2_condition.unsupported_wrapper_features,
        "degraded": bool(degrade_to_v1),
        "semantic_losses": semantic_losses,
        "degraded_payload": degraded_payload,
        "wan_scail_v1_images": wan_scail_v1_images,
    }


def summarize_wanvideo_scail2_payload(payload: dict[str, Any]) -> str:
    target = payload["target"]
    v1_outputs = "available" if payload.get("wan_scail_v1_images") else "unavailable"
    return (
        f"{payload['kind']} v{payload['version']} "
        f"live_wrapper_supported={target['live_wrapper_supported']} "
        f"degraded={payload['degraded']} "
        f"wan_scail_v1_outputs={v1_outputs}"
    )
