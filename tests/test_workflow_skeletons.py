from __future__ import annotations

import copy
import json
import re
import unittest
from pathlib import Path

from scail2 import wanvideo_contracts
from scail2.workflow_contract_validator import (
    inspect_workflow_contract,
    load_contract_manifest,
    validate_workflow_contract,
)
from scail2.workflow_static import diagnose_render_nlf_connections


ROOT = Path(__file__).resolve().parents[1]
SKELETON_DIR = ROOT / "workflow_skeletons"
RENDER_NLF_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "workflow_contracts"
    / "render_nlf_connection_regression.json"
)
NATIVE_ANIMATION_CONTRACT_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "workflow_contracts"
    / "native_animation_host_contract.json"
)
REPLACEMENT_CONTRACT_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "workflow_contracts"
    / "replacement_background_lock_host_contract.json"
)
LEGACY_V1_CONTRACT_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "workflow_contracts"
    / "legacy_v1_pose_control_host_contract.json"
)
SKELETON_EXPECTED_HOST_FAMILIES = {
    "scail2_condition_builder.json": "rookiestar28-scail2",
    "wan_scail_v1_pose_control.json": "kijai-vanilla",
    "wananimate_fallback.json": "kijai-vanilla",
    "wanvideo_native_scail2.json": "rookiestar28-scail2",
    "wanvideo_replacement_background_lock.json": "rookiestar28-scail2",
}
CONCEPTUAL_REQUIRED_EXTERNAL_INPUTS = {
    "scail2_condition_builder.json": {
        "driving_track_data",
        "reference_track_data_or_mask",
        "driving_or_pose_video",
        "reference_image",
    },
    "wananimate_fallback.json": {
        "scail2_condition",
        "vae",
        "width",
        "height",
        "num_frames",
        "force_offload",
        "frame_window_size",
        "colormatch",
        "pose_strength",
        "face_strength",
    },
}
SKELETON_EXPECTED_OPTIONAL_EXTERNAL_INPUTS = {
    "wan_scail_v1_pose_control.json": {"bboxes", "pose_video_mask"},
}
HOST_FIXTURE_EXPECTED_EXTERNAL_INPUTS = {
    "native_animation_host_contract.json": {
        "scail2_condition",
        "vae",
        "model",
        "text_embeds",
        "scheduler",
    },
    "replacement_background_lock_host_contract.json": {
        "driving_video",
        "denoise_mask",
        "scail2_condition",
        "vae",
        "model",
        "text_embeds",
        "scheduler",
    },
    "legacy_v1_pose_control_host_contract.json": {
        "reference_image",
        "pose_images",
        "clip_vision",
        "vae",
        "model",
        "text_embeds",
        "scheduler",
    },
}
NATIVE_ANIMATION_FORBIDDEN_CLASSES = {
    "RenderNLFPoses",
    "NLFPredictPoses",
    "SCAILPose2PoseMaskGeometryAlign",
    "SCAILPose2ReplacementDenoiseMask",
    "SCAILPose2ReplacementConditionVideo",
}
NATIVE_ANIMATION_REQUIRED_LINKS = {
    (("workflow_inputs", "driving_video"), ("sam3_video_track", "images"), "IMAGE"),
    (
        ("sam3_video_track", "track_data"),
        ("colored_masks", "driving_track_data"),
        "SAM3_TRACK_DATA",
    ),
    (("workflow_inputs", "driving_video"), ("scail2_condition", "pose_video"), "IMAGE"),
    (("workflow_inputs", "reference_image"), ("scail2_condition", "ref_image"), "IMAGE"),
    (("workflow_inputs", "reference_mask"), ("colored_masks", "ref_mask"), "MASK"),
    (("colored_masks", "pose_video_mask"), ("scail2_condition", "pose_video_mask"), "IMAGE"),
    (("colored_masks", "reference_image_mask"), ("scail2_condition", "ref_mask"), "IMAGE"),
    (("scail2_condition", "condition"), ("wanvideo_scail2_adapter", "condition"), "SCAIL2_CONDITION"),
    (("wanvideo_scail2_adapter", "condition"), ("wan_scail2_condition_embeds", "condition"), "SCAIL2_WANVIDEO_PAYLOAD"),
    (("wan_context_options", "context_options"), ("wan_sampler_extra_args", "context_options"), "WANVIDCONTEXT"),
    (("wan_sampler_extra_args", "extra_args"), ("wan_sampler", "extra_args"), "WANVIDSAMPLEREXTRAARGS"),
}
REPLACEMENT_FORBIDDEN_CLASSES = {
    "RenderNLFPoses",
    "NLFPredictPoses",
    "SCAILPose2PoseMaskGeometryAlign",
    "SCAILPose2ReferenceImageGeometryAlign",
    "SCAILPose2ReplacementConditionVideo",
}
REPLACEMENT_REQUIRED_LINKS = {
    (("workflow_inputs", "driving_video"), ("sam3_video_track", "images"), "IMAGE"),
    (
        ("sam3_video_track", "track_data"),
        ("colored_masks", "driving_track_data"),
        "SAM3_TRACK_DATA",
    ),
    (("workflow_inputs", "reference_mask"), ("colored_masks", "ref_mask"), "MASK"),
    (
        ("workflow_inputs", "driving_video"),
        ("scail2_condition", "driving_video"),
        "IMAGE",
    ),
    (
        ("colored_masks", "pose_video_mask"),
        ("scail2_condition", "pose_video_mask"),
        "IMAGE",
    ),
    (
        ("workflow_inputs", "reference_image"),
        ("scail2_condition", "ref_image"),
        "IMAGE",
    ),
    (
        ("colored_masks", "reference_image_mask"),
        ("scail2_condition", "ref_mask"),
        "IMAGE",
    ),
    (
        ("scail2_condition", "condition"),
        ("replacement_denoise_mask", "condition"),
        "SCAIL2_CONDITION",
    ),
    (
        ("colored_masks", "pose_video_mask"),
        ("replacement_denoise_mask", "pose_video_mask"),
        "IMAGE",
    ),
    (
        ("replacement_denoise_mask", "mask"),
        ("wanvideo_encode", "mask"),
        "MASK",
    ),
    (
        ("workflow_inputs", "driving_video"),
        ("wanvideo_encode", "driving_video"),
        "IMAGE",
    ),
    (("workflow_inputs", "vae"), ("wanvideo_encode", "vae"), "WANVAE"),
    (
        ("scail2_condition", "condition"),
        ("wanvideo_scail2_adapter", "condition"),
        "SCAIL2_CONDITION",
    ),
    (
        ("wanvideo_scail2_adapter", "condition"),
        ("wan_scail2_condition_embeds", "condition"),
        "SCAIL2_WANVIDEO_PAYLOAD",
    ),
    (
        ("wan_empty_embeds", "image_embeds"),
        ("wan_scail2_condition_embeds", "embeds"),
        "WANVIDIMAGE_EMBEDS",
    ),
    (
        ("workflow_inputs", "vae"),
        ("wan_scail2_condition_embeds", "vae"),
        "WANVAE",
    ),
    (
        ("wan_scail2_condition_embeds", "image_embeds"),
        ("wan_sampler", "image_embeds"),
        "WANVIDIMAGE_EMBEDS",
    ),
    (("wanvideo_encode", "samples"), ("wan_sampler", "samples"), "LATENT"),
    (("workflow_inputs", "model"), ("wan_sampler", "model"), "WANVIDEOMODEL"),
    (
        ("workflow_inputs", "text_embeds"),
        ("wan_sampler", "text_embeds"),
        "WANVIDEOTEXTEMBEDS",
    ),
    (
        ("workflow_inputs", "scheduler"),
        ("wan_sampler", "scheduler"),
        "WANVIDEOSCHEDULER",
    ),
    (("workflow_inputs", "width"), ("scail2_condition", "width"), "INT"),
    (("workflow_inputs", "height"), ("scail2_condition", "height"), "INT"),
    (("workflow_inputs", "num_frames"), ("scail2_condition", "num_frames"), "INT"),
    (("workflow_inputs", "width"), ("wan_empty_embeds", "width"), "INT"),
    (("workflow_inputs", "height"), ("wan_empty_embeds", "height"), "INT"),
    (("workflow_inputs", "num_frames"), ("wan_empty_embeds", "num_frames"), "INT"),
    (
        ("wan_context_options", "context_options"),
        ("wan_sampler_extra_args", "context_options"),
        "WANVIDCONTEXT",
    ),
    (
        ("wan_sampler_extra_args", "extra_args"),
        ("wan_sampler", "extra_args"),
        "WANVIDSAMPLEREXTRAARGS",
    ),
}
V1_HOST_NODE_CLASSES = {
    "wan_empty_embeds": "WanVideoEmptyEmbeds",
    "wan_clip_vision": "WanVideoClipVisionEncode",
    "wan_scail_reference": "WanVideoAddSCAILReferenceEmbeds",
    "wan_scail_pose": "WanVideoAddSCAILPoseEmbeds",
    "wan_sampler": "WanVideoSamplerv2",
}
V1_REQUIRED_LOCAL_LINKS = {
    (("workflow_inputs", "nlf_poses"), ("nlf_render", "nlf_poses"), "NLFPRED"),
    (("workflow_inputs", "width"), ("nlf_render", "render_width"), "INT"),
    (("workflow_inputs", "height"), ("nlf_render", "render_height"), "INT"),
    (("nlf_render", "image"), ("wan_scail_pose", "pose_images"), "IMAGE"),
    (
        ("wan_clip_vision", "image_embeds"),
        ("wan_scail_reference", "clip_embeds"),
        "WANVIDIMAGE_CLIPEMBEDS",
    ),
}
WANANIMATE_REQUIRED_INPUTS = {
    "vae",
    "width",
    "height",
    "num_frames",
    "force_offload",
    "frame_window_size",
    "colormatch",
    "pose_strength",
    "face_strength",
}
WANANIMATE_SEMANTIC_LOSSES = {
    "rgb_semantic_masks_collapsed_to_binary_grayscale",
    "scail2_28_channel_mask_latent_not_represented",
    "replacement_flag_rope_mode_not_represented",
    "additional_reference_pairs_not_represented",
    "mask_palette_track_metadata_not_preserved_as_channels",
}


def load_skeleton(name: str):
    return json.loads((SKELETON_DIR / name).read_text(encoding="utf-8"))


def load_legacy_v1_contract_fixture():
    return json.loads(LEGACY_V1_CONTRACT_FIXTURE.read_text(encoding="utf-8"))


def expected_skeleton_external_inputs(data, name):
    if name in CONCEPTUAL_REQUIRED_EXTERNAL_INPUTS:
        return CONCEPTUAL_REQUIRED_EXTERNAL_INPUTS[name]
    workflow_inputs = next(
        node for node in data.get("nodes", []) if node.get("id") == "workflow_inputs"
    )
    all_inputs = {output["name"] for output in workflow_inputs["outputs"]}
    return all_inputs - SKELETON_EXPECTED_OPTIONAL_EXTERNAL_INPUTS.get(name, set())


def artifact_metadata_diagnostics(
    data, *, family, external_inputs, optional_external_inputs=()
):
    diagnostics = []
    if data.get("execution") != "static_only":
        diagnostics.append("INVALID_EXECUTION_CLASSIFICATION")
    if data.get("executable") is not False:
        diagnostics.append("EXECUTABLE_ARTIFACT_CLAIM")

    host = data.get("host", {})
    if host.get("family") != family:
        diagnostics.append("INVALID_HOST_FAMILY")
    else:
        manifest = load_contract_manifest()
        if manifest.find_family(host.get("family"), host.get("revision")) is None:
            diagnostics.append("UNSUPPORTED_HOST_REVISION")

    declared_inputs = data.get("required_external_inputs")
    if not isinstance(declared_inputs, list) or set(declared_inputs) != set(external_inputs):
        diagnostics.append("INCOMPLETE_EXTERNAL_INPUTS")
    elif len(declared_inputs) != len(set(declared_inputs)):
        diagnostics.append("DUPLICATE_EXTERNAL_INPUTS")

    declared_optional = data.get("optional_external_inputs", [])
    if not isinstance(declared_optional, list) or set(declared_optional) != set(
        optional_external_inputs
    ):
        diagnostics.append("INCOMPLETE_OPTIONAL_EXTERNAL_INPUTS")
    elif len(declared_optional) != len(set(declared_optional)):
        diagnostics.append("DUPLICATE_OPTIONAL_EXTERNAL_INPUTS")
    if isinstance(declared_inputs, list) and isinstance(declared_optional, list):
        if set(declared_inputs) & set(declared_optional):
            diagnostics.append("OVERLAPPING_EXTERNAL_INPUTS")

    limitations = data.get("known_limitations")
    if not isinstance(limitations, list) or not limitations or not all(
        isinstance(item, str) and item.strip() for item in limitations
    ):
        diagnostics.append("MISSING_KNOWN_LIMITATIONS")
    return tuple(sorted(diagnostics))


def load_render_nlf_fixture():
    return json.loads(RENDER_NLF_FIXTURE.read_text(encoding="utf-8"))


def load_native_animation_contract_fixture():
    return json.loads(NATIVE_ANIMATION_CONTRACT_FIXTURE.read_text(encoding="utf-8"))


def load_replacement_contract_fixture():
    return json.loads(REPLACEMENT_CONTRACT_FIXTURE.read_text(encoding="utf-8"))


def native_animation_forbidden_classes(data):
    classes = {node.get("class_type") for node in data.get("nodes", [])}
    return tuple(sorted(classes & NATIVE_ANIMATION_FORBIDDEN_CLASSES))


def native_animation_missing_links(data):
    links = {
        (tuple(link["from"]), tuple(link["to"]), link["type"])
        for link in data.get("links", [])
    }
    return tuple(sorted(NATIVE_ANIMATION_REQUIRED_LINKS - links))


def replacement_missing_links(data):
    links = {
        (tuple(link["from"]), tuple(link["to"]), link["type"])
        for link in data.get("links", [])
    }
    return tuple(sorted(REPLACEMENT_REQUIRED_LINKS - links))


def replacement_forbidden_classes(data):
    classes = {node.get("class_type") for node in data.get("nodes", [])}
    return tuple(sorted(classes & REPLACEMENT_FORBIDDEN_CLASSES))


def replacement_configuration_diagnostics(data):
    diagnostics = []
    links = {
        (tuple(link["from"]), tuple(link["to"]), link["type"])
        for link in data.get("links", [])
    }
    if any(target == ("scail2_condition", "pose_video") for _, target, _ in links):
        diagnostics.append("POSE_VIDEO_LINK_PRESENT")
    if any(
        source == ("wan_context_options", "context_options")
        and target[0] == "wan_sampler"
        for source, target, _ in links
    ):
        diagnostics.append("DIRECT_CONTEXT_LINK_PRESENT")

    contract = data.get("background_lock_contract", {})
    polarity = contract.get("mask_polarity", {})
    if polarity.get("subject_replace_area") != 1.0:
        diagnostics.append("INVALID_SUBJECT_MASK_POLARITY")
    if polarity.get("background_preserve_area") != 0.0:
        diagnostics.append("INVALID_BACKGROUND_MASK_POLARITY")

    sampler = next(
        (node for node in data.get("nodes", []) if node.get("id") == "wan_sampler"),
        {},
    )
    if sampler.get("inputs", {}).get("add_noise_to_samples") is not True:
        diagnostics.append("INVALID_SAMPLER_NOISE_INPUT")
    if sampler.get("required_settings", {}).get("add_noise_to_samples") is not True:
        diagnostics.append("INVALID_SAMPLER_NOISE_CONTRACT")
    return tuple(sorted(diagnostics))


def v1_contract_diagnostics(data):
    diagnostics = []
    host = data.get("host")
    if not isinstance(host, dict):
        return ("INVALID_HOST_DECLARATION",)

    manifest = load_contract_manifest()
    try:
        family = manifest.family(host.get("family"), host.get("revision"))
    except Exception:
        return ("UNSUPPORTED_HOST_FAMILY",)

    nodes = {node.get("id"): node for node in data.get("nodes", [])}
    links = data.get("links", [])
    incoming = {(tuple(link["to"]), link["type"]) for link in links}
    link_set = {
        (tuple(link["from"]), tuple(link["to"]), link["type"]) for link in links
    }

    for node_id, class_type in V1_HOST_NODE_CLASSES.items():
        node = nodes.get(node_id)
        if node is None or node.get("class_type") != class_type:
            diagnostics.append(f"MISSING_HOST_NODE:{node_id}")
            continue
        contract = family.node(class_type)
        literal_inputs = set(node.get("inputs", {}))
        linked_inputs = {target[1] for target, _ in incoming if target[0] == node_id}
        for socket in contract.required_inputs:
            if socket.name not in literal_inputs | linked_inputs:
                diagnostics.append(f"MISSING_REQUIRED_INPUT:{node_id}:{socket.name}")

        outputs = {socket.name: socket.comfy_type for socket in contract.outputs}
        for link in links:
            if link["from"][0] != node_id:
                continue
            source_name = link["from"][1]
            if source_name not in outputs:
                diagnostics.append(f"INVALID_HOST_OUTPUT:{node_id}:{source_name}")
            elif link["type"] != outputs[source_name]:
                diagnostics.append(f"INVALID_OUTPUT_TYPE:{node_id}:{source_name}")

    for required in V1_REQUIRED_LOCAL_LINKS - link_set:
        diagnostics.append("MISSING_LOCAL_LINK:" + "->".join((required[0][1], required[1][1])))
    return tuple(sorted(diagnostics))


def render_nlf_skeletons(workflows):
    return tuple(
        sorted(
            name
            for name, data in workflows.items()
            if any(node.get("class_type") == "RenderNLFPoses" for node in data.get("nodes", []))
        )
    )


def wananimate_design_note_diagnostics(data):
    diagnostics = []
    classification = data.get("classification", {})
    availability = data.get("availability", {})
    degradation = data.get("degradation", {})
    nodes = data.get("nodes", [])

    if classification.get("decision") != "reclassified_non_executable_design_note":
        diagnostics.append("INVALID_NECESSITY_DECISION")
    if data.get("executable") is not False:
        diagnostics.append("EXECUTABLE_FALLBACK_CLAIM")
    if availability.get("registered_comfyui_adapter_node") is not False:
        diagnostics.append("REGISTERED_ADAPTER_CLAIM")
    if any(node.get("id") == "wananimate_fallback_adapter" for node in nodes):
        diagnostics.append("PHANTOM_ADAPTER_NODE")
    if any("helper" in node for node in nodes):
        diagnostics.append("PHANTOM_GRAPH_HELPER")
    if data.get("links"):
        diagnostics.append("PHANTOM_EXECUTABLE_LINKS")
    if set(data.get("target_required_inputs", ())) != WANANIMATE_REQUIRED_INPUTS:
        diagnostics.append("INCOMPLETE_TARGET_REQUIREMENTS")
    if degradation.get("allow_semantic_degradation_default") is not False:
        diagnostics.append("DEGRADATION_DEFAULT_ENABLED")
    if degradation.get("requires_explicit_enable") is not True:
        diagnostics.append("DEGRADATION_OPT_IN_DISABLED")
    if set(degradation.get("semantic_losses", ())) != WANANIMATE_SEMANTIC_LOSSES:
        diagnostics.append("INCOMPLETE_SEMANTIC_LOSSES")
    return tuple(sorted(diagnostics))


class WorkflowSkeletonTests(unittest.TestCase):
    def test_all_skeletons_parse_and_use_local_schema(self) -> None:
        for path in sorted(SKELETON_DIR.glob("*.json")):
            with self.subTest(path=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual("scail_pose2.workflow_skeleton.v1", data["schema"])
                self.assertEqual("static_only", data["execution"])
                self.assertEqual("deferred", data["live_generation"])
                self.assertTrue(data["nodes"])

    def test_distributable_workflow_artifact_metadata_is_complete(self) -> None:
        for name, family in SKELETON_EXPECTED_HOST_FAMILIES.items():
            with self.subTest(kind="skeleton", name=name):
                data = load_skeleton(name)
                self.assertFalse(
                    artifact_metadata_diagnostics(
                        data,
                        family=family,
                        external_inputs=expected_skeleton_external_inputs(data, name),
                        optional_external_inputs=SKELETON_EXPECTED_OPTIONAL_EXTERNAL_INPUTS.get(
                            name, set()
                        ),
                    )
                )

        fixture_paths = {
            NATIVE_ANIMATION_CONTRACT_FIXTURE.name: NATIVE_ANIMATION_CONTRACT_FIXTURE,
            REPLACEMENT_CONTRACT_FIXTURE.name: REPLACEMENT_CONTRACT_FIXTURE,
            LEGACY_V1_CONTRACT_FIXTURE.name: LEGACY_V1_CONTRACT_FIXTURE,
        }
        expected_families = {
            NATIVE_ANIMATION_CONTRACT_FIXTURE.name: "rookiestar28-scail2",
            REPLACEMENT_CONTRACT_FIXTURE.name: "rookiestar28-scail2",
            LEGACY_V1_CONTRACT_FIXTURE.name: "kijai-vanilla",
        }
        for name, path in fixture_paths.items():
            with self.subTest(kind="host_fixture", name=name):
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertFalse(
                    artifact_metadata_diagnostics(
                        data,
                        family=expected_families[name],
                        external_inputs=HOST_FIXTURE_EXPECTED_EXTERNAL_INPUTS[name],
                    )
                )

    def test_workflow_artifact_metadata_mutations_fail_closed(self) -> None:
        data = load_skeleton("wanvideo_native_scail2.json")
        expected_inputs = expected_skeleton_external_inputs(
            data, "wanvideo_native_scail2.json"
        )
        mutations = []

        executable = copy.deepcopy(data)
        executable["executable"] = True
        mutations.append(executable)
        live = copy.deepcopy(data)
        live["execution"] = "live"
        mutations.append(live)
        wrong_revision = copy.deepcopy(data)
        wrong_revision["host"]["revision"] = "0" * 40
        mutations.append(wrong_revision)
        missing_input = copy.deepcopy(data)
        missing_input["required_external_inputs"] = sorted(expected_inputs)[1:]
        mutations.append(missing_input)
        no_limitations = copy.deepcopy(data)
        no_limitations["known_limitations"] = []
        mutations.append(no_limitations)

        for index, mutated in enumerate(mutations):
            with self.subTest(index=index):
                self.assertTrue(
                    artifact_metadata_diagnostics(
                        mutated,
                        family="rookiestar28-scail2",
                        external_inputs=expected_inputs,
                    )
                )

        legacy = load_skeleton("wan_scail_v1_pose_control.json")
        legacy_required = expected_skeleton_external_inputs(
            legacy, "wan_scail_v1_pose_control.json"
        )
        legacy_optional = SKELETON_EXPECTED_OPTIONAL_EXTERNAL_INPUTS[
            "wan_scail_v1_pose_control.json"
        ]
        promoted = copy.deepcopy(legacy)
        promoted["required_external_inputs"].append("bboxes")
        promoted["optional_external_inputs"] = ["pose_video_mask"]
        self.assertTrue(
            artifact_metadata_diagnostics(
                promoted,
                family="kijai-vanilla",
                external_inputs=legacy_required,
                optional_external_inputs=legacy_optional,
            )
        )

    def test_workflow_test_data_paths_are_tracked_only_boundaries(self) -> None:
        paths = (
            SKELETON_DIR,
            RENDER_NLF_FIXTURE,
            NATIVE_ANIMATION_CONTRACT_FIXTURE,
            REPLACEMENT_CONTRACT_FIXTURE,
            LEGACY_V1_CONTRACT_FIXTURE,
            ROOT / "workflow_contracts" / "wanvideo_host_contracts.v1.json",
        )
        forbidden_roots = {".planning", ".sessions", "reference", ".reference"}
        for path in paths:
            with self.subTest(path=path.name):
                relative = path.relative_to(ROOT)
                self.assertNotIn(relative.parts[0], forbidden_roots)
                self.assertTrue(path.exists())

    def test_legacy_v1_host_contract_passes_source_validator(self) -> None:
        fixture = load_legacy_v1_contract_fixture()
        result = validate_workflow_contract(fixture)

        self.assertTrue(result.valid)
        self.assertEqual((), result.diagnostics)
        self.assertEqual("static_host_subgraph", fixture["classification"])
        self.assertEqual("legacy_v1_pose_control", fixture["mode"])
        self.assertNotIn(
            "RenderNLFPoses",
            {node["class_type"] for node in fixture["nodes"]},
        )
        serialized = json.dumps(fixture, sort_keys=True)
        for forbidden in (".planning", "reference/docs", "prompt"):
            self.assertNotIn(forbidden, serialized)
        self.assertNotRegex(serialized, r"[A-Za-z]:\\")
        self.assertNotRegex(serialized, r"\bS2W\d+\b")

    def test_legacy_v1_host_contract_mutations_fail_closed(self) -> None:
        fixture = load_legacy_v1_contract_fixture()
        mutations = []

        missing_reference_vae = copy.deepcopy(fixture)
        next(
            node for node in missing_reference_vae["nodes"] if node["id"] == "reference"
        )["inputs"].pop("vae")
        mutations.append(missing_reference_vae)
        missing_pose_vae = copy.deepcopy(fixture)
        next(node for node in missing_pose_vae["nodes"] if node["id"] == "pose")[
            "inputs"
        ].pop("vae")
        mutations.append(missing_pose_vae)
        missing_clip_model = copy.deepcopy(fixture)
        next(node for node in missing_clip_model["nodes"] if node["id"] == "clip")[
            "inputs"
        ].pop("clip_vision")
        mutations.append(missing_clip_model)
        invalid_clip_output = copy.deepcopy(fixture)
        next(
            link
            for link in invalid_clip_output["links"]
            if link["to"] == ["reference", "clip_embeds"]
        )["from"][1] = "WANVIDIMAGE_CLIPEMBEDS"
        mutations.append(invalid_clip_output)

        for index, mutated in enumerate(mutations):
            with self.subTest(index=index):
                self.assertTrue(inspect_workflow_contract(mutated))

    def test_v1_pose_control_skeleton_matches_wan_scail_contracts(self) -> None:
        data = load_skeleton("wan_scail_v1_pose_control.json")
        class_types = {node["class_type"] for node in data["nodes"]}
        links = {(tuple(link["to"]), link["type"]) for link in data["links"]}

        self.assertEqual("kijai-vanilla", data["host"]["family"])
        self.assertEqual("legacy_v1_pose_control", data["classification"]["mode"])
        self.assertFalse(v1_contract_diagnostics(data))

        self.assertTrue(
            {
                "RenderNLFPoses",
                "ExternalWorkflowInputs",
                wanvideo_contracts.NODE_WAN_EMPTY_EMBEDS,
                wanvideo_contracts.NODE_WAN_CLIP_VISION_ENCODE,
                wanvideo_contracts.NODE_WAN_ADD_SCAIL_REFERENCE,
                wanvideo_contracts.NODE_WAN_ADD_SCAIL_POSE,
                wanvideo_contracts.NODE_WAN_SAMPLER_V2,
            }.issubset(class_types)
        )
        self.assertNotIn("SCAILPose2WanSCAILImages", class_types)
        self.assertIn((("wan_scail_reference", "ref_image"), "IMAGE"), links)
        self.assertIn((("wan_scail_pose", "pose_images"), "IMAGE"), links)
        self.assertIn((("wan_empty_embeds", "num_frames"), "INT"), links)
        nlf_render = next(node for node in data["nodes"] if node["id"] == "nlf_render")
        self.assertEqual(
            {
                "nlf_poses": "NLFPRED",
                "render_width": "INT",
                "render_height": "INT",
            },
            nlf_render["required_inputs"],
        )
        self.assertEqual(
            {
                "bboxes": "BBOX",
                "pose_video_mask": "IMAGE",
            },
            nlf_render["optional_inputs"],
        )
        self.assertEqual(
            "NLFPredictPoses.bboxes",
            nlf_render["geometry_contract"]["bboxes_source"],
        )
        self.assertEqual(
            "render on render_width/render_height and emit half-size IMAGE/MASK",
            nlf_render["geometry_contract"]["render_width_height_policy"],
        )
        self.assertEqual(
            "valid pose_video_mask alignment is preferred; invalid masks can fall back to bbox repair",
            nlf_render["geometry_contract"]["pose_video_mask_priority"],
        )
        self.assertIn(
            "semantic identity colors",
            nlf_render["geometry_contract"]["multi_person_identity_composition"],
        )

    def test_v1_required_contract_mutations_are_detected(self) -> None:
        data = load_skeleton("wan_scail_v1_pose_control.json")
        critical_targets = {
            ("wan_clip_vision", "clip_vision"),
            ("wan_scail_reference", "vae"),
            ("wan_scail_pose", "vae"),
            ("wan_scail_reference", "clip_embeds"),
            ("wan_sampler", "model"),
            ("wan_sampler", "scheduler"),
        }

        for target in critical_targets:
            with self.subTest(target=target):
                mutated = copy.deepcopy(data)
                removed = next(link for link in mutated["links"] if tuple(link["to"]) == target)
                mutated["links"].remove(removed)
                self.assertTrue(v1_contract_diagnostics(mutated))

        bad_output = copy.deepcopy(data)
        clip_link = next(
            link
            for link in bad_output["links"]
            if link["to"] == ["wan_scail_reference", "clip_embeds"]
        )
        clip_link["from"][1] = "WANVIDIMAGE_CLIPEMBEDS"
        self.assertTrue(v1_contract_diagnostics(bad_output))

        bad_type = copy.deepcopy(data)
        next(
            link
            for link in bad_type["links"]
            if link["to"] == ["wan_scail_reference", "clip_embeds"]
        )["type"] = "IMAGE"
        self.assertTrue(v1_contract_diagnostics(bad_type))

    def test_render_nlf_is_scoped_to_legacy_v1_skeleton(self) -> None:
        workflows = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in SKELETON_DIR.glob("*.json")
        }
        self.assertEqual(
            ("wan_scail_v1_pose_control.json",),
            render_nlf_skeletons(workflows),
        )

        mutated = copy.deepcopy(workflows)
        mutated["wanvideo_native_scail2.json"]["nodes"].append(
            {"id": "forbidden", "class_type": "RenderNLFPoses"}
        )
        self.assertEqual(
            ("wan_scail_v1_pose_control.json", "wanvideo_native_scail2.json"),
            render_nlf_skeletons(mutated),
        )

    def test_render_nlf_fixture_render_and_condition_dimensions_are_split(self) -> None:
        workflow = load_render_nlf_fixture()
        node_by_id = {int(node["id"]): node for node in workflow["nodes"]}
        link_by_id = {int(link[0]): link for link in workflow["links"]}

        def source_node_for_input(node, input_name: str):
            input_spec = next(item for item in node["inputs"] if item["name"] == input_name)
            link = link_by_id[int(input_spec["link"])]
            return node_by_id[int(link[1])]

        render = next(node for node in workflow["nodes"] if node["type"] == "RenderNLFPoses")
        condition = next(
            node for node in workflow["nodes"] if node["type"] == "SCAILPose2SCAIL2Condition"
        )

        self.assertEqual(
            ["width"],
            source_node_for_input(render, "render_width")["widgets_values"],
        )
        self.assertEqual(
            ["height"],
            source_node_for_input(render, "render_height")["widgets_values"],
        )
        self.assertEqual(
            ["gen_width"],
            source_node_for_input(condition, "width")["widgets_values"],
        )
        self.assertEqual(
            ["gen_height"],
            source_node_for_input(condition, "height")["widgets_values"],
        )
        self.assertEqual(
            "SCAILPose2ColoredMask",
            source_node_for_input(render, "pose_video_mask")["type"],
        )

    def test_render_nlf_fixture_reports_bbox_and_mask_connections(self) -> None:
        diagnostics = diagnose_render_nlf_connections(load_render_nlf_fixture())

        self.assertEqual(1, diagnostics.render_node_count)
        self.assertEqual(("362",), diagnostics.render_node_ids)
        self.assertTrue(diagnostics.bboxes_connected)
        self.assertTrue(diagnostics.pose_video_mask_connected)
        self.assertIn("bboxes_connected=True", diagnostics.summary())
        self.assertIn("pose_video_mask_connected=True", diagnostics.summary())

    def test_render_nlf_regression_fixture_is_static_and_public_safe(self) -> None:
        fixture = load_render_nlf_fixture()
        text = RENDER_NLF_FIXTURE.read_text(encoding="utf-8")

        self.assertEqual("scail_pose2.static_regression_fixture.v1", fixture["schema"])
        self.assertEqual("never", fixture["execution"])
        for forbidden in (
            "prompt",
            "model_name",
            "reference/",
            ".planning",
            "api_key",
            "token=",
        ):
            self.assertNotIn(forbidden, text)
        self.assertIsNone(re.search(r"[A-Za-z]:\\\\", text))

    def test_render_nlf_static_diagnostics_reports_missing_mask_connection(self) -> None:
        workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "RenderNLFPoses",
                    "inputs": [
                        {"name": "nlf_poses", "link": 10},
                        {"name": "bboxes", "link": 11},
                        {"name": "pose_video_mask", "link": None},
                    ],
                },
                {"id": 2, "type": "NLFPredict", "inputs": []},
            ],
            "links": [
                [10, 2, 0, 1, 0, "NLFPRED"],
                [11, 2, 1, 1, 3, "BBOX"],
            ],
        }

        diagnostics = diagnose_render_nlf_connections(workflow)

        self.assertEqual(1, diagnostics.render_node_count)
        self.assertTrue(diagnostics.bboxes_connected)
        self.assertFalse(diagnostics.pose_video_mask_connected)
        self.assertIn("pose_video_mask_connected=False", diagnostics.summary())

    def test_scail2_condition_skeleton_lists_unsupported_wrapper_features(self) -> None:
        data = load_skeleton("scail2_condition_builder.json")
        class_types = {node.get("class_type") for node in data["nodes"]}
        output_types = {node.get("output_type") for node in data["nodes"]}
        fields = set(data["required_condition_fields"])

        self.assertTrue(
            {
                "SCAILPose2ColoredMask",
                "SCAILPose2SCAIL2Condition",
                "SCAILPose2WanVideoSCAIL2Adapter",
            }.issubset(class_types)
        )
        self.assertIn("SCAIL2_CONDITION", output_types)
        self.assertIn("SCAIL2_WANVIDEO_PAYLOAD", output_types)
        self.assertEqual(
            [
                "pose_video",
                "driving_video",
                "pose_video_mask",
                "ref_image",
                "ref_mask",
                "additional_ref_image",
                "additional_ref_mask",
            ],
            next(
                node
                for node in data["nodes"]
                if node["id"] == "scail2_condition"
            )["input_order"],
        )
        condition_node = next(
            node
            for node in data["nodes"]
            if node["id"] == "scail2_condition"
        )
        colored_masks = next(node for node in data["nodes"] if node["id"] == "colored_masks")
        self.assertEqual(
            ["blue", "red", "green", "magenta", "cyan", "yellow"],
            colored_masks["identity_palette"],
        )
        self.assertEqual(
            {
                "animation": "pose_video",
                "replacement": "driving_video",
            },
            condition_node["mode_video_sources"],
        )
        self.assertEqual(
            "warning_only",
            condition_node["identity_diagnostics"]["under_provisioned_references"],
        )
        self.assertTrue(
            {
                "mode",
                "replace_flag",
                "driving_mask_indices",
                "identity",
                "source_kind",
            }.issubset(fields)
        )
        self.assertNotIn("segment_len", fields)
        self.assertNotIn("segment_overlap", fields)
        self.assertNotIn("previous_frame_count", fields)
        self.assertNotIn("video_frame_offset", fields)
        self.assertEqual(
            "native_scail2_embeds",
            data["wanvideo_scail2_adapter"]["target"]["current_wrapper_path"],
        )
        self.assertEqual(
            "v1_scail_embeds",
            data["wanvideo_scail2_adapter"]["target"]["fallback_wrapper_path"],
        )
        schema = data["wanvideo_scail2_adapter"]["payload_schema"]
        self.assertEqual("scail_pose2.wanvideo_scail2_payload", schema["name"])
        self.assertEqual(
            "WanVideoAddSCAIL2ConditionEmbeds",
            schema["native_wrapper_consumer"]["class_type"],
        )
        self.assertEqual(
            "WANVIDIMAGE_EMBEDS",
            schema["native_wrapper_consumer"]["output_type"],
        )
        self.assertEqual(
            "scail2_embeds",
            schema["native_wrapper_consumer"]["embeds_key"],
        )
        self.assertEqual(
            "reject",
            schema["native_wrapper_consumer"]["simultaneous_legacy_and_native"],
        )
        self.assertEqual(28, schema["runtime_mask_layouts"]["channel_count"])
        self.assertEqual(4, schema["runtime_mask_layouts"]["temporal_stride"])
        self.assertEqual(8, schema["runtime_mask_layouts"]["spatial_downsample"])
        self.assertEqual(
            ["reference", "driving", "additional_reference"],
            schema["runtime_mask_layouts"]["layout_roles"],
        )
        self.assertTrue(
            schema["mask_data_flow"]["native_runtime_masks_authoritative"]
        )
        self.assertEqual(
            [
                "driving_identity_count",
                "reference_identity_count",
                "additional_reference_identity_counts",
                "reference_slot_count",
                "warnings",
            ],
            schema["identity"]["fields"],
        )
        self.assertFalse(
            schema["mask_data_flow"]["full_resolution_indices_in_native_payload"]
        )
        self.assertTrue(
            data["wanvideo_scail2_adapter"]["target"]["live_wrapper_supported"]
        )
        self.assertEqual(
            set(wanvideo_contracts.UNSUPPORTED_CURRENT_WAN_SCAIL2_FEATURES),
            set(data["legacy_v1_semantic_losses"]),
        )
        adapter_node = next(
            node
            for node in data["nodes"]
            if node["id"] == "wanvideo_scail2_adapter"
        )
        self.assertEqual(["condition"], adapter_node["output_names"])
        self.assertEqual(1, adapter_node["public_output_count"])
        self.assertNotIn("v1_compat_output_type", adapter_node)
        self.assertNotIn("v1_compat_outputs", adapter_node)
        self.assertEqual(
            {
                "ref_image": "IMAGE",
                "pose_images": "IMAGE",
                "width": "INT",
                "height": "INT",
                "num_frames": "INT",
            },
            {
                key: data["wanvideo_scail2_adapter"]["degradation"][
                    "v1_payload_fields_when_enabled"
                ][key]
                for key in ("ref_image", "pose_images", "width", "height", "num_frames")
            },
        )

    def test_native_scail2_wrapper_skeleton_wires_expected_path(self) -> None:
        data = load_skeleton("wanvideo_native_scail2.json")
        class_types = {node.get("class_type") for node in data["nodes"]}
        links = {(tuple(link["from"]), tuple(link["to"]), link["type"]) for link in data["links"]}
        manifest = load_contract_manifest()
        fork = next(
            family
            for family in manifest.families
            if family.family_id == "rookiestar28-scail2"
        )

        self.assertEqual("static_only", data["execution"])
        self.assertEqual("rookiestar28-scail2", data["host"]["family"])
        self.assertEqual(fork.revision, data["host"]["revision"])
        self.assertEqual((), native_animation_forbidden_classes(data))
        self.assertEqual((), native_animation_missing_links(data))

        self.assertTrue(
            {
                "SAM3_VideoTrack",
                "SCAILPose2ColoredMask",
                "SCAILPose2SCAIL2Condition",
                "SCAILPose2WanVideoSCAIL2Adapter",
                "WanVideoAddSCAIL2ConditionEmbeds",
                "WanVideoContextOptions",
                "WanVideoSamplerExtraArgs",
                wanvideo_contracts.NODE_WAN_EMPTY_EMBEDS,
                wanvideo_contracts.NODE_WAN_SAMPLER_V2,
            }.issubset(class_types)
        )
        self.assertIn(
            (
                ("workflow_inputs", "driving_video"),
                ("sam3_video_track", "images"),
                "IMAGE",
            ),
            links,
        )
        self.assertIn(
            (
                ("workflow_inputs", "driving_video"),
                ("scail2_condition", "pose_video"),
                "IMAGE",
            ),
            links,
        )
        self.assertIn(
            (
                ("workflow_inputs", "reference_image"),
                ("scail2_condition", "ref_image"),
                "IMAGE",
            ),
            links,
        )
        self.assertIn(
            (
                ("workflow_inputs", "reference_mask"),
                ("colored_masks", "ref_mask"),
                "MASK",
            ),
            links,
        )
        self.assertIn(
            (
                ("sam3_video_track", "track_data"),
                ("colored_masks", "driving_track_data"),
                "SAM3_TRACK_DATA",
            ),
            links,
        )
        self.assertIn(
            (
                ("colored_masks", "reference_image_mask"),
                ("scail2_condition", "ref_mask"),
                "IMAGE",
            ),
            links,
        )
        self.assertIn(
            (
                ("wanvideo_scail2_adapter", "condition"),
                ("wan_scail2_condition_embeds", "condition"),
                "SCAIL2_WANVIDEO_PAYLOAD",
            ),
            links,
        )
        self.assertIn(
            (
                ("wan_scail2_condition_embeds", "image_embeds"),
                ("wan_sampler", "image_embeds"),
                "WANVIDIMAGE_EMBEDS",
            ),
            links,
        )
        self.assertIn(
            (
                ("wan_context_options", "context_options"),
                ("wan_sampler_extra_args", "context_options"),
                "WANVIDCONTEXT",
            ),
            links,
        )
        self.assertIn(
            (
                ("wan_sampler_extra_args", "extra_args"),
                ("wan_sampler", "extra_args"),
                "WANVIDSAMPLEREXTRAARGS",
            ),
            links,
        )
        self.assertNotIn(
            (
                ("wan_context_options", "context_options"),
                ("wan_sampler", "context_options"),
                "WANVIDCONTEXT",
            ),
            links,
        )
        for source_name, target_id, target_name, link_type in (
            ("model", "wan_sampler", "model", "WANVIDEOMODEL"),
            ("text_embeds", "wan_sampler", "text_embeds", "WANVIDEOTEXTEMBEDS"),
            ("vae", "wan_scail2_condition_embeds", "vae", "WANVAE"),
            ("scheduler", "wan_sampler", "scheduler", "WANVIDEOSCHEDULER"),
        ):
            self.assertIn(
                (
                    ("workflow_inputs", source_name),
                    (target_id, target_name),
                    link_type,
                ),
                links,
            )
        defaults = data["contract_defaults"]
        self.assertEqual(512, defaults["width"])
        self.assertEqual(512, defaults["height"])
        self.assertEqual(81, defaults["num_frames"])
        self.assertEqual(1, defaults["num_frames"] % 4)
        self.assertGreaterEqual(defaults["context_frames"], 2)
        self.assertLess(defaults["context_overlap"], defaults["context_frames"])
        context = data["context"]
        self.assertEqual("ComfyUI-WanVideoWrapper", context["owner"])
        self.assertEqual("WanVideoContextOptions", context["owner_node"])
        self.assertEqual("context_options", context["adapter_input_socket"])
        self.assertEqual("extra_args", context["sampler_socket"])
        self.assertEqual(
            ["context_frames", "context_stride", "context_overlap"],
            context["controls"],
        )
        self.assertFalse(context["scail2_condition_segment_controls"])
        self.assertFalse(context["official_scail2_clean_history_claimed"])
        self.assertEqual(
            "scail2_embeds",
            data["native_wrapper_contract"]["embeds_key"],
        )
        self.assertTrue(
            data["native_wrapper_contract"][
                "strength_defaults_are_backward_compatible"
            ]
        )
        self.assertEqual(
            [
                "ref_image_strength",
                "ref_mask_strength",
                "condition_video_strength",
                "driving_mask_strength",
            ],
            data["native_wrapper_contract"]["strength_controls"],
        )
        self.assertEqual(
            "scail_embeds",
            data["native_wrapper_contract"]["legacy_embeds_key"],
        )
        self.assertEqual(
            "reject",
            data["native_wrapper_contract"]["simultaneous_legacy_and_native"],
        )
        self.assertEqual(
            ["blue", "red", "green", "magenta", "cyan", "yellow"],
            data["multi_person_identity_contract"]["identity_palette"],
        )
        self.assertTrue(
            data["multi_person_identity_contract"][
                "under_provisioned_references_are_warnings"
            ]
        )
        self.assertFalse(data["degradation"]["v1_fallback_is_full_scail2_parity"])

    def test_native_animation_host_contract_passes_source_validator(self) -> None:
        fixture = load_native_animation_contract_fixture()
        result = validate_workflow_contract(fixture)

        self.assertTrue(result.valid)
        self.assertEqual((), result.diagnostics)
        self.assertEqual("static_host_subgraph", fixture["classification"])
        serialized = json.dumps(fixture, sort_keys=True)
        for forbidden in (".planning", "reference/docs", "RenderNLFPoses", "prompt"):
            self.assertNotIn(forbidden, serialized)
        self.assertNotRegex(serialized, r"[A-Za-z]:\\")
        self.assertNotRegex(serialized, r"\bS2W\d+\b")

    def test_native_animation_contract_mutations_fail_closed(self) -> None:
        fixture = load_native_animation_contract_fixture()

        missing_model = copy.deepcopy(fixture)
        sampler = next(node for node in missing_model["nodes"] if node["id"] == "sampler")
        sampler["inputs"].pop("model")
        self.assertIn(
            "MISSING_REQUIRED_INPUT",
            {item.code for item in inspect_workflow_contract(missing_model)},
        )

        bad_output = copy.deepcopy(fixture)
        next(link for link in bad_output["links"] if link["to"] == ["sampler", "image_embeds"])[
            "from"
        ][1] = 1
        self.assertIn(
            "INVALID_SOURCE_OUTPUT",
            {item.code for item in inspect_workflow_contract(bad_output)},
        )

        bad_revision = copy.deepcopy(fixture)
        bad_revision["host"]["revision"] = "0" * 40
        self.assertIn(
            "UNSUPPORTED_HOST_FAMILY",
            {item.code for item in inspect_workflow_contract(bad_revision)},
        )

    def test_native_animation_renderer_insertion_is_detected(self) -> None:
        data = load_skeleton("wanvideo_native_scail2.json")
        mutated = copy.deepcopy(data)
        mutated["nodes"].append({"id": "forbidden", "class_type": "RenderNLFPoses"})

        self.assertEqual(("RenderNLFPoses",), native_animation_forbidden_classes(mutated))

    def test_native_animation_required_link_removal_is_detected(self) -> None:
        data = load_skeleton("wanvideo_native_scail2.json")
        required_targets = (
            ("sam3_video_track", "images"),
            ("colored_masks", "driving_track_data"),
            ("colored_masks", "ref_mask"),
            ("scail2_condition", "pose_video"),
            ("scail2_condition", "ref_image"),
            ("scail2_condition", "pose_video_mask"),
            ("scail2_condition", "ref_mask"),
            ("wanvideo_scail2_adapter", "condition"),
            ("wan_scail2_condition_embeds", "condition"),
            ("wan_sampler_extra_args", "context_options"),
            ("wan_sampler", "extra_args"),
        )

        for target in required_targets:
            with self.subTest(target=target):
                mutated = copy.deepcopy(data)
                removed = next(
                    link for link in mutated["links"] if tuple(link["to"]) == target
                )
                mutated["links"].remove(removed)
                self.assertIn(
                    (tuple(removed["from"]), target, removed["type"]),
                    native_animation_missing_links(mutated),
                )

    def test_replacement_background_lock_skeleton_wires_samples_mask_path(self) -> None:
        data = load_skeleton("wanvideo_replacement_background_lock.json")
        class_types = {node.get("class_type") for node in data["nodes"]}
        links = {
            (tuple(link["from"]), tuple(link["to"]), link["type"])
            for link in data["links"]
        }

        self.assertEqual("rookiestar28-scail2", data["host"]["family"])
        self.assertEqual(load_replacement_contract_fixture()["host"], data["host"])
        self.assertEqual("replacement", data["classification"]["mode"])
        self.assertFalse(replacement_forbidden_classes(data))
        self.assertFalse(replacement_missing_links(data))
        self.assertFalse(replacement_configuration_diagnostics(data))

        self.assertTrue(
            {
                "SCAILPose2ColoredMask",
                "SCAILPose2SCAIL2Condition",
                "SCAILPose2ReplacementDenoiseMask",
                "WanVideoEncode",
                "WanVideoAddSCAIL2ConditionEmbeds",
                wanvideo_contracts.NODE_WAN_SAMPLER_V2,
            }.issubset(class_types)
        )
        self.assertNotIn("SCAILPose2ReferenceImageGeometryAlign", class_types)
        self.assertNotIn("RenderNLFPoses", class_types)
        self.assertNotIn("SCAILPose2ReplacementConditionVideo", class_types)
        self.assertIn(
            (
                ("colored_masks", "pose_video_mask"),
                ("replacement_denoise_mask", "pose_video_mask"),
                "IMAGE",
            ),
            links,
        )
        self.assertIn(
            (
                ("workflow_inputs", "driving_video"),
                ("scail2_condition", "driving_video"),
                "IMAGE",
            ),
            links,
        )
        self.assertIn(
            (
                ("workflow_inputs", "reference_image"),
                ("scail2_condition", "ref_image"),
                "IMAGE",
            ),
            links,
        )
        self.assertIn(
            (
                ("colored_masks", "reference_image_mask"),
                ("scail2_condition", "ref_mask"),
                "IMAGE",
            ),
            links,
        )
        self.assertIn(
            (
                ("scail2_condition", "condition"),
                ("replacement_denoise_mask", "condition"),
                "SCAIL2_CONDITION",
            ),
            links,
        )
        self.assertIn(
            (
                ("replacement_denoise_mask", "mask"),
                ("wanvideo_encode", "mask"),
                "MASK",
            ),
            links,
        )
        self.assertIn(
            (
                ("workflow_inputs", "driving_video"),
                ("wanvideo_encode", "driving_video"),
                "IMAGE",
            ),
            links,
        )
        self.assertIn(
            (
                ("wanvideo_encode", "samples"),
                ("wan_sampler", "samples"),
                "LATENT",
            ),
            links,
        )
        self.assertIn(
            (
                ("wan_scail2_condition_embeds", "image_embeds"),
                ("wan_sampler", "image_embeds"),
                "WANVIDIMAGE_EMBEDS",
            ),
            links,
        )
        embeds_node = next(
            node for node in data["nodes"] if node["id"] == "wan_scail2_condition_embeds"
        )
        condition_node = next(node for node in data["nodes"] if node["id"] == "scail2_condition")
        self.assertEqual(
            {
                "mode": "replacement_only_auto",
                "multi_identity_reference_slots": "auto_base_plus_generated_additional_refs",
                "manual_additional_refs": "preserved_no_duplicate_auto_split",
                "fit_mode_default": "auto",
                "anchor_default": "auto",
                "target_frame_policy_default": "median_bbox",
                "control_region_default": "auto",
                "bbox_margin_default": 0,
                "max_scale_default": 2.0,
                "min_mask_area_ratio_default": 0.0005,
            },
            condition_node["reference_geometry_alignment"],
        )
        self.assertEqual(
            {
                "ref_image_strength": 1.0,
                "ref_mask_strength": 1.0,
                "condition_video_strength": 1.0,
                "driving_mask_strength": 1.0,
            },
            embeds_node["strength_defaults"],
        )
        sampler = next(node for node in data["nodes"] if node["id"] == "wan_sampler")
        self.assertTrue(sampler["required_settings"]["add_noise_to_samples"])
        self.assertEqual(True, sampler["inputs"]["add_noise_to_samples"])
        defaults = data["contract_defaults"]
        self.assertEqual(0, (defaults["num_frames"] - 1) % 4)
        self.assertGreaterEqual(defaults["context_frames"], 2)
        self.assertLess(defaults["context_overlap"], defaults["context_frames"])
        contract = data["background_lock_contract"]
        self.assertEqual("driving_video", contract["encode_driving_video_socket"])
        self.assertEqual(
            "SCAILPose2SCAIL2Condition.replacement_only_auto_alignment",
            contract["reference_geometry_source"],
        )
        self.assertTrue(contract["required"])
        self.assertFalse(contract["conditioning_alone_hard_preserves_background"])
        self.assertEqual(1.0, contract["mask_polarity"]["subject_replace_area"])
        self.assertEqual(0.0, contract["mask_polarity"]["background_preserve_area"])
        self.assertFalse(contract["pose_geometry_alignment_required"])
        self.assertEqual(
            "workflow_inputs.driving_video -> scail2_condition.driving_video",
            contract["condition_video_source"],
        )
        fallback = contract["replacement_condition_video_fallback"]
        self.assertEqual("legacy_experimental_manual", fallback["status"])
        self.assertTrue(fallback["may_weaken_pose_latents"])
        self.assertTrue(fallback["does_not_replace_samples_path"])
        strengths = contract["wrapper_strength_controls"]
        self.assertTrue(strengths["defaults_preserve_existing_behavior"])
        self.assertIn("reference image", strengths["ref_image_strength"])
        self.assertIn("condition video", strengths["condition_video_strength"])
        self.assertFalse(contract["render_nlf_poses_required"])
        preview = contract["preview_contract"]
        self.assertTrue(preview["early_preview_background_may_be_noisy"])
        self.assertFalse(preview["preview_is_final_preservation_evidence"])
        self.assertTrue(preview["final_preservation_requires_samples_noise_mask_path"])
        mask_contract = contract["wrapper_noise_mask_contract"]
        self.assertTrue(mask_contract["metadata_required"])
        self.assertEqual("nearest_binary", mask_contract["tagged_interpolation_policy"])
        self.assertEqual("downstream_default", mask_contract["untagged_policy"])
        context_contract = contract["context_frame_map_contract"]
        self.assertEqual("WanVideoContextOptions", context_contract["context_owner"])
        self.assertTrue(context_contract["core_does_not_schedule_context_windows"])
        self.assertTrue(
            context_contract["pose_latents_and_driving_masks_must_share_frame_count"]
        )
        self.assertTrue(
            context_contract["samples_and_noise_mask_must_share_latent_timeline"]
        )
        self.assertEqual(
            ["blue", "red", "green", "magenta", "cyan", "yellow"],
            data["multi_person_identity_contract"]["identity_palette"],
        )
        self.assertTrue(
            data["multi_person_identity_contract"][
                "under_provisioned_references_are_warnings"
            ]
        )

    def test_replacement_host_contract_passes_source_validator(self) -> None:
        fixture = load_replacement_contract_fixture()
        result = validate_workflow_contract(fixture)

        self.assertTrue(result.valid)
        self.assertEqual((), result.diagnostics)
        self.assertEqual("static_host_subgraph", fixture["classification"])
        serialized = json.dumps(fixture, sort_keys=True)
        for forbidden in (".planning", "reference/docs", "RenderNLFPoses", "prompt"):
            self.assertNotIn(forbidden, serialized)
        self.assertNotRegex(serialized, r"[A-Za-z]:\\")
        self.assertNotRegex(serialized, r"\bS2W\d+\b")

    def test_replacement_host_contract_mutations_fail_closed(self) -> None:
        fixture = load_replacement_contract_fixture()

        missing_vae = copy.deepcopy(fixture)
        encode = next(node for node in missing_vae["nodes"] if node["id"] == "encode")
        encode["inputs"].pop("vae")
        self.assertIn(
            "MISSING_REQUIRED_INPUT",
            {item.code for item in inspect_workflow_contract(missing_vae)},
        )

        bad_socket = copy.deepcopy(fixture)
        encode = next(node for node in bad_socket["nodes"] if node["id"] == "encode")
        encode["inputs"]["image"] = encode["inputs"].pop("driving_video")
        self.assertIn(
            "UNKNOWN_NODE_INPUT",
            {item.code for item in inspect_workflow_contract(bad_socket)},
        )

        bad_output = copy.deepcopy(fixture)
        next(link for link in bad_output["links"] if link["to"] == ["sampler", "samples"])[
            "from"
        ][1] = 1
        self.assertIn(
            "INVALID_SOURCE_OUTPUT",
            {item.code for item in inspect_workflow_contract(bad_output)},
        )

        bad_type = copy.deepcopy(fixture)
        next(link for link in bad_type["links"] if link["to"] == ["sampler", "samples"])[
            "type"
        ] = "IMAGE"
        self.assertIn(
            "DECLARED_LINK_TYPE_MISMATCH",
            {item.code for item in inspect_workflow_contract(bad_type)},
        )

        bad_revision = copy.deepcopy(fixture)
        bad_revision["host"]["revision"] = "0" * 40
        self.assertIn(
            "UNSUPPORTED_HOST_FAMILY",
            {item.code for item in inspect_workflow_contract(bad_revision)},
        )

    def test_replacement_required_link_removal_is_detected(self) -> None:
        data = load_skeleton("wanvideo_replacement_background_lock.json")

        for required in REPLACEMENT_REQUIRED_LINKS:
            with self.subTest(required=required):
                mutated = copy.deepcopy(data)
                removed = next(
                    link
                    for link in mutated["links"]
                    if (tuple(link["from"]), tuple(link["to"]), link["type"]) == required
                )
                mutated["links"].remove(removed)
                self.assertIn(required, replacement_missing_links(mutated))

    def test_replacement_forbidden_class_insertion_is_detected(self) -> None:
        data = load_skeleton("wanvideo_replacement_background_lock.json")

        for class_type in REPLACEMENT_FORBIDDEN_CLASSES:
            with self.subTest(class_type=class_type):
                mutated = copy.deepcopy(data)
                mutated["nodes"].append({"id": "forbidden", "class_type": class_type})
                self.assertEqual((class_type,), replacement_forbidden_classes(mutated))

    def test_replacement_configuration_mutations_are_detected(self) -> None:
        data = load_skeleton("wanvideo_replacement_background_lock.json")

        pose_video = copy.deepcopy(data)
        pose_video["links"].append(
            {
                "from": ["workflow_inputs", "driving_video"],
                "to": ["scail2_condition", "pose_video"],
                "type": "IMAGE",
            }
        )
        self.assertIn(
            "POSE_VIDEO_LINK_PRESENT",
            replacement_configuration_diagnostics(pose_video),
        )

        direct_context = copy.deepcopy(data)
        direct_context["links"].append(
            {
                "from": ["wan_context_options", "context_options"],
                "to": ["wan_sampler", "context_options"],
                "type": "WANVIDCONTEXT",
            }
        )
        self.assertIn(
            "DIRECT_CONTEXT_LINK_PRESENT",
            replacement_configuration_diagnostics(direct_context),
        )

        subject_polarity = copy.deepcopy(data)
        subject_polarity["background_lock_contract"]["mask_polarity"][
            "subject_replace_area"
        ] = 0.0
        self.assertIn(
            "INVALID_SUBJECT_MASK_POLARITY",
            replacement_configuration_diagnostics(subject_polarity),
        )

        background_polarity = copy.deepcopy(data)
        background_polarity["background_lock_contract"]["mask_polarity"][
            "background_preserve_area"
        ] = 1.0
        self.assertIn(
            "INVALID_BACKGROUND_MASK_POLARITY",
            replacement_configuration_diagnostics(background_polarity),
        )

        sampler_input = copy.deepcopy(data)
        next(node for node in sampler_input["nodes"] if node["id"] == "wan_sampler")[
            "inputs"
        ]["add_noise_to_samples"] = False
        self.assertIn(
            "INVALID_SAMPLER_NOISE_INPUT",
            replacement_configuration_diagnostics(sampler_input),
        )

        sampler_contract = copy.deepcopy(data)
        next(node for node in sampler_contract["nodes"] if node["id"] == "wan_sampler")[
            "required_settings"
        ]["add_noise_to_samples"] = False
        self.assertIn(
            "INVALID_SAMPLER_NOISE_CONTRACT",
            replacement_configuration_diagnostics(sampler_contract),
        )

    def test_wananimate_fallback_skeleton_requires_explicit_degradation(self) -> None:
        data = load_skeleton("wananimate_fallback.json")
        class_types = {node.get("class_type") for node in data["nodes"]}

        self.assertIn("WanVideoAnimateEmbeds", class_types)
        self.assertFalse(wananimate_design_note_diagnostics(data))
        self.assertFalse(data["executable"])
        self.assertFalse(data["availability"]["registered_comfyui_adapter_node"])
        self.assertFalse(data["degradation"]["is_full_scail2_parity"])
        self.assertTrue(data["degradation"]["requires_explicit_enable"])
        self.assertIn(
            "rgb_semantic_masks_collapsed_to_binary_grayscale",
            data["degradation"]["semantic_losses"],
        )

    def test_wananimate_design_note_mutations_are_detected(self) -> None:
        data = load_skeleton("wananimate_fallback.json")

        mutations = []
        executable = copy.deepcopy(data)
        executable["executable"] = True
        mutations.append(executable)
        registered = copy.deepcopy(data)
        registered["availability"]["registered_comfyui_adapter_node"] = True
        mutations.append(registered)
        phantom_node = copy.deepcopy(data)
        phantom_node["nodes"].append(
            {"id": "wananimate_fallback_adapter", "helper": "unavailable_graph_helper"}
        )
        mutations.append(phantom_node)
        phantom_link = copy.deepcopy(data)
        phantom_link["links"].append(
            {"from": ["source", "mask"], "to": ["wananimate_embeds", "mask"], "type": "MASK"}
        )
        mutations.append(phantom_link)
        default_enabled = copy.deepcopy(data)
        default_enabled["degradation"]["allow_semantic_degradation_default"] = True
        mutations.append(default_enabled)
        opt_in_disabled = copy.deepcopy(data)
        opt_in_disabled["degradation"]["requires_explicit_enable"] = False
        mutations.append(opt_in_disabled)
        for loss in WANANIMATE_SEMANTIC_LOSSES:
            missing_loss = copy.deepcopy(data)
            if loss in missing_loss["degradation"]["semantic_losses"]:
                missing_loss["degradation"]["semantic_losses"].remove(loss)
            mutations.append(missing_loss)

        for index, mutated in enumerate(mutations):
            with self.subTest(index=index):
                self.assertTrue(wananimate_design_note_diagnostics(mutated))

    def test_skeletons_are_public_safe(self) -> None:
        forbidden_tokens = [
            "ref" + "erence/",
            "." + "planning",
            "." + "sessions",
            "AG" + "ENTS.md",
            "ROAD" + "MAP.md",
            "api_key",
            "token=",
        ]
        absolute_path_patterns = [
            re.compile(r"[A-Za-z]:\\\\"),
            re.compile(r"/Users/"),
            re.compile(r"/home/"),
        ]

        for path in sorted(SKELETON_DIR.glob("*.json")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                for token in forbidden_tokens:
                    self.assertNotIn(token, text)
                for pattern in absolute_path_patterns:
                    self.assertIsNone(pattern.search(text))


if __name__ == "__main__":
    unittest.main()
