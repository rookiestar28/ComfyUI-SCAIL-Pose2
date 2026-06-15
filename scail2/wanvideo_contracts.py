"""Static WanVideoWrapper contract metadata used by SCAIL-Pose2 adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


TYPE_IMAGE = "IMAGE"
TYPE_INT = "INT"
TYPE_FLOAT = "FLOAT"
TYPE_LATENT = "LATENT"
TYPE_CLIP_VISION = "CLIP_VISION"
TYPE_WANVAE = "WANVAE"
TYPE_WANVIDEOMODEL = "WANVIDEOMODEL"
TYPE_WANVIDEO_SCHEDULER = "WANVIDEOSCHEDULER"
TYPE_WANVIDEO_TEXT_EMBEDS = "WANVIDEOTEXTEMBEDS"
TYPE_WANVIDIMAGE_EMBEDS = "WANVIDIMAGE_EMBEDS"
TYPE_WANVIDIMAGE_CLIP_EMBEDS = "WANVIDIMAGE_CLIPEMBEDS"

NODE_WAN_EMPTY_EMBEDS = "WanVideoEmptyEmbeds"
NODE_WAN_ADD_SCAIL_REFERENCE = "WanVideoAddSCAILReferenceEmbeds"
NODE_WAN_ADD_SCAIL_POSE = "WanVideoAddSCAILPoseEmbeds"
NODE_WAN_CLIP_VISION_ENCODE = "WanVideoClipVisionEncode"
NODE_WAN_SAMPLER_V2 = "WanVideoSamplerv2"


@dataclass(frozen=True)
class SocketContract:
    """ComfyUI socket name/type metadata."""

    name: str
    comfy_type: str
    required: bool = True


@dataclass(frozen=True)
class NodeContract:
    """Minimal node contract needed for static adapter compatibility checks."""

    class_name: str
    return_type: str
    required_inputs: tuple[SocketContract, ...]
    optional_inputs: tuple[SocketContract, ...] = ()

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(socket.name for socket in self.required_inputs + self.optional_inputs)

    @property
    def required_input_names(self) -> tuple[str, ...]:
        return tuple(socket.name for socket in self.required_inputs)

    @property
    def optional_input_names(self) -> tuple[str, ...]:
        return tuple(socket.name for socket in self.optional_inputs)


@dataclass(frozen=True)
class AdapterFieldContract:
    """Maps a SCAIL-Pose2 adapter field to its intended wrapper socket."""

    field_name: str
    comfy_type: str
    wrapper_node: str
    wrapper_socket: str
    required_for_adapter: bool
    ownership: str = "scail_pose2"


WAN_SCAIL_V1_NODE_CONTRACTS: dict[str, NodeContract] = {
    NODE_WAN_EMPTY_EMBEDS: NodeContract(
        class_name=NODE_WAN_EMPTY_EMBEDS,
        return_type=TYPE_WANVIDIMAGE_EMBEDS,
        required_inputs=(
            SocketContract("width", TYPE_INT),
            SocketContract("height", TYPE_INT),
            SocketContract("num_frames", TYPE_INT),
        ),
        optional_inputs=(
            SocketContract("control_embeds", TYPE_WANVIDIMAGE_EMBEDS, required=False),
            SocketContract("extra_latents", TYPE_LATENT, required=False),
        ),
    ),
    NODE_WAN_ADD_SCAIL_REFERENCE: NodeContract(
        class_name=NODE_WAN_ADD_SCAIL_REFERENCE,
        return_type=TYPE_WANVIDIMAGE_EMBEDS,
        required_inputs=(
            SocketContract("embeds", TYPE_WANVIDIMAGE_EMBEDS),
            SocketContract("vae", TYPE_WANVAE),
            SocketContract("ref_image", TYPE_IMAGE),
            SocketContract("strength", TYPE_FLOAT),
            SocketContract("start_percent", TYPE_FLOAT),
            SocketContract("end_percent", TYPE_FLOAT),
        ),
        optional_inputs=(
            SocketContract("clip_embeds", TYPE_WANVIDIMAGE_CLIP_EMBEDS, required=False),
        ),
    ),
    NODE_WAN_ADD_SCAIL_POSE: NodeContract(
        class_name=NODE_WAN_ADD_SCAIL_POSE,
        return_type=TYPE_WANVIDIMAGE_EMBEDS,
        required_inputs=(
            SocketContract("embeds", TYPE_WANVIDIMAGE_EMBEDS),
            SocketContract("vae", TYPE_WANVAE),
            SocketContract("pose_images", TYPE_IMAGE),
            SocketContract("strength", TYPE_FLOAT),
            SocketContract("start_percent", TYPE_FLOAT),
            SocketContract("end_percent", TYPE_FLOAT),
        ),
    ),
    NODE_WAN_CLIP_VISION_ENCODE: NodeContract(
        class_name=NODE_WAN_CLIP_VISION_ENCODE,
        return_type=TYPE_WANVIDIMAGE_CLIP_EMBEDS,
        required_inputs=(
            SocketContract("clip_vision", TYPE_CLIP_VISION),
            SocketContract("image_1", TYPE_IMAGE),
            SocketContract("strength_1", TYPE_FLOAT),
            SocketContract("strength_2", TYPE_FLOAT),
            SocketContract("crop", "BOOLEAN"),
            SocketContract("combine_embeds", "BOOLEAN"),
            SocketContract("force_offload", "BOOLEAN"),
        ),
        optional_inputs=(
            SocketContract("image_2", TYPE_IMAGE, required=False),
            SocketContract("negative_image", TYPE_IMAGE, required=False),
            SocketContract("tiles", TYPE_INT, required=False),
            SocketContract("ratio", TYPE_FLOAT, required=False),
        ),
    ),
    NODE_WAN_SAMPLER_V2: NodeContract(
        class_name=NODE_WAN_SAMPLER_V2,
        return_type=TYPE_LATENT,
        required_inputs=(
            SocketContract("model", TYPE_WANVIDEOMODEL),
            SocketContract("image_embeds", TYPE_WANVIDIMAGE_EMBEDS),
            SocketContract("scheduler", TYPE_WANVIDEO_SCHEDULER),
            SocketContract("text_embeds", TYPE_WANVIDEO_TEXT_EMBEDS),
            SocketContract("cfg", TYPE_FLOAT),
        ),
    ),
}


ADAPTER_FIELD_TO_WRAPPER_SOCKET: dict[str, AdapterFieldContract] = {
    "width": AdapterFieldContract(
        field_name="width",
        comfy_type=TYPE_INT,
        wrapper_node=NODE_WAN_EMPTY_EMBEDS,
        wrapper_socket="width",
        required_for_adapter=True,
    ),
    "height": AdapterFieldContract(
        field_name="height",
        comfy_type=TYPE_INT,
        wrapper_node=NODE_WAN_EMPTY_EMBEDS,
        wrapper_socket="height",
        required_for_adapter=True,
    ),
    "num_frames": AdapterFieldContract(
        field_name="num_frames",
        comfy_type=TYPE_INT,
        wrapper_node=NODE_WAN_EMPTY_EMBEDS,
        wrapper_socket="num_frames",
        required_for_adapter=True,
    ),
    "ref_image": AdapterFieldContract(
        field_name="ref_image",
        comfy_type=TYPE_IMAGE,
        wrapper_node=NODE_WAN_ADD_SCAIL_REFERENCE,
        wrapper_socket="ref_image",
        required_for_adapter=True,
    ),
    "pose_images": AdapterFieldContract(
        field_name="pose_images",
        comfy_type=TYPE_IMAGE,
        wrapper_node=NODE_WAN_ADD_SCAIL_POSE,
        wrapper_socket="pose_images",
        required_for_adapter=True,
    ),
    "clip_ref_image": AdapterFieldContract(
        field_name="clip_ref_image",
        comfy_type=TYPE_IMAGE,
        wrapper_node=NODE_WAN_CLIP_VISION_ENCODE,
        wrapper_socket="image_1",
        required_for_adapter=False,
    ),
    "vae": AdapterFieldContract(
        field_name="vae",
        comfy_type=TYPE_WANVAE,
        wrapper_node=NODE_WAN_ADD_SCAIL_REFERENCE,
        wrapper_socket="vae",
        required_for_adapter=False,
        ownership="wanvideo_wrapper",
    ),
    "image_embeds": AdapterFieldContract(
        field_name="image_embeds",
        comfy_type=TYPE_WANVIDIMAGE_EMBEDS,
        wrapper_node=NODE_WAN_SAMPLER_V2,
        wrapper_socket="image_embeds",
        required_for_adapter=False,
        ownership="wanvideo_wrapper",
    ),
}

REQUIRED_WAN_SCAIL_V1_ADAPTER_FIELDS: tuple[str, ...] = tuple(
    field_name
    for field_name, contract in ADAPTER_FIELD_TO_WRAPPER_SOCKET.items()
    if contract.required_for_adapter
)

UNSUPPORTED_CURRENT_WAN_SCAIL2_FEATURES: tuple[str, ...] = (
    "rgb_semantic_reference_masks",
    "rgb_semantic_driving_masks",
    "mask_latents_28_channel",
    "replacement_flag_rope_mode",
    "additional_reference_mask_pairs",
    "clean_history_segment_overlap",
    "mask_palette_track_metadata",
)


def missing_required_adapter_fields(payload_keys: Iterable[str]) -> tuple[str, ...]:
    """Return required adapter fields absent from the provided payload keys."""

    present = set(payload_keys)
    return tuple(
        field_name
        for field_name in REQUIRED_WAN_SCAIL_V1_ADAPTER_FIELDS
        if field_name not in present
    )


def validate_required_adapter_fields(payload_keys: Iterable[str]) -> None:
    """Raise ValueError when required SCAIL v1-style wrapper inputs are missing."""

    missing = missing_required_adapter_fields(payload_keys)
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(f"Missing required WanVideo SCAIL adapter fields: {missing_csv}")


def wrapper_nodes_for_adapter_fields() -> tuple[str, ...]:
    """Return wrapper nodes that receive SCAIL-Pose2 adapter-owned fields."""

    nodes = {
        contract.wrapper_node
        for contract in ADAPTER_FIELD_TO_WRAPPER_SOCKET.values()
        if contract.ownership == "scail_pose2"
    }
    return tuple(sorted(nodes))
