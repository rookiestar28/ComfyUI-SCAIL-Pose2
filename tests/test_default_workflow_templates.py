from __future__ import annotations

import copy
import json
import re
import unittest
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "example_workflows"
DEFAULT_PATH = TEMPLATE_DIR / "scail2_replacement_default.json"
COMPACT_PATH = TEMPLATE_DIR / "scail2_replacement_wanvideowrapper_compact.json"

EXPECTED_COUNTS = {
    "recommended": (57, 69),
    "wanvideowrapper_compact": (51, 58),
}
PREVIEW_ONLY_NODE_IDS = {318, 319, 328, 418, 452, 460}
FORBIDDEN_NODE_TYPES = {
    "DownloadAndLoadNLFModel",
    "NLFPredict",
    "OnnxDetectionModelLoader",
    "PoseDetectionVitPoseToDWPose",
    "RenderNLFPoses",
    "SCAILPose2ReplacementConditionVideo",
    "SCAILPose2ReplacementDenoiseMask",
    "WanVideoEncode",
}
EXPECTED_ASSETS = {
    "select_clip_vision.safetensors",
    "select_driving_video.mp4",
    "select_reference_image.png",
    "select_sam3_checkpoint.safetensors",
    "select_scail2_model.safetensors",
    "select_umt5_text_encoder.safetensors",
    "select_wan_vae.pth",
}
ASSET_SUFFIXES = (
    ".avi",
    ".ckpt",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp4",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".webp",
)


def load_workflow(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def node_by_type(workflow: dict[str, Any], node_type: str) -> dict[str, Any]:
    matches = [node for node in workflow["nodes"] if node["type"] == node_type]
    if len(matches) != 1:
        raise AssertionError(f"expected one {node_type}, found {len(matches)}")
    return matches[0]


def input_by_name(node: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [value for value in node.get("inputs", []) if value.get("name") == name]
    if len(matches) != 1:
        raise AssertionError(f"expected one {node['type']}.{name}, found {len(matches)}")
    return matches[0]


def iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from iter_strings(key)
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)


def validate_public_template(workflow: dict[str, Any], variant: str) -> None:
    expected_nodes, expected_links = EXPECTED_COUNTS[variant]
    if len(workflow.get("nodes", [])) != expected_nodes:
        raise AssertionError(f"{variant}: unexpected node count")
    if len(workflow.get("links", [])) != expected_links:
        raise AssertionError(f"{variant}: unexpected link count")
    if workflow.get("groups") != [] or workflow.get("config") != {}:
        raise AssertionError(f"{variant}: public template must not inherit private groups/config")
    if workflow.get("last_node_id") != 487 or workflow.get("last_link_id") != 861:
        raise AssertionError(f"{variant}: node/link monotonicity changed")

    metadata = workflow.get("extra", {}).get("scail_pose2_template")
    if not isinstance(metadata, dict):
        raise AssertionError(f"{variant}: template metadata missing")
    if metadata.get("schema") != "scail_pose2.workflow_template.v1":
        raise AssertionError(f"{variant}: template metadata schema drift")
    if metadata.get("variant") != variant:
        raise AssertionError(f"{variant}: template metadata variant drift")
    if metadata.get("generation_backend") != "ComfyUI-WanVideoWrapper":
        raise AssertionError(f"{variant}: generation backend drift")
    if metadata.get("claims_wrapper_only") is not False:
        raise AssertionError(f"{variant}: false wrapper-only claim")
    if metadata.get("template_live_smoke") is not False:
        raise AssertionError(f"{variant}: generated template was not independently live-smoked")

    nodes = workflow["nodes"]
    links = workflow["links"]
    node_ids = [int(node["id"]) for node in nodes]
    link_ids = [int(link[0]) for link in links]
    if len(node_ids) != len(set(node_ids)):
        raise AssertionError(f"{variant}: duplicate node ID")
    if len(link_ids) != len(set(link_ids)):
        raise AssertionError(f"{variant}: duplicate link ID")
    nodes_by_id = {int(node["id"]): node for node in nodes}
    links_by_id = {int(link[0]): link for link in links}

    for link in links:
        if len(link) < 6:
            raise AssertionError(f"{variant}: malformed link")
        link_id, source_id, source_slot, target_id, target_slot = map(int, link[:5])
        if source_id not in nodes_by_id or target_id not in nodes_by_id:
            raise AssertionError(f"{variant}: dangling link endpoint {link_id}")
        source = nodes_by_id[source_id]
        target = nodes_by_id[target_id]
        if not 0 <= source_slot < len(source.get("outputs", [])):
            raise AssertionError(f"{variant}: invalid source socket {link_id}")
        if not 0 <= target_slot < len(target.get("inputs", [])):
            raise AssertionError(f"{variant}: invalid target socket {link_id}")
        if target["inputs"][target_slot].get("link") != link_id:
            raise AssertionError(f"{variant}: target socket serialization drift {link_id}")
        if link_id not in (source["outputs"][source_slot].get("links") or []):
            raise AssertionError(f"{variant}: source socket serialization drift {link_id}")

    for node in nodes:
        node_id = int(node["id"])
        for input_slot, node_input in enumerate(node.get("inputs", [])):
            link_id = node_input.get("link")
            if link_id is None:
                continue
            link = links_by_id.get(int(link_id))
            if link is None or (int(link[3]), int(link[4])) != (node_id, input_slot):
                raise AssertionError(f"{variant}: stale input link {link_id}")
        for output_slot, node_output in enumerate(node.get("outputs", [])):
            for link_id in node_output.get("links") or []:
                link = links_by_id.get(int(link_id))
                if link is None or (int(link[1]), int(link[2])) != (node_id, output_slot):
                    raise AssertionError(f"{variant}: stale output link {link_id}")

    node_types = {node["type"] for node in nodes}
    forbidden = node_types & FORBIDDEN_NODE_TYPES
    if forbidden:
        raise AssertionError(f"{variant}: forbidden nodes present: {sorted(forbidden)}")

    setters = [node for node in nodes if node["type"] == "SetNode"]
    getters = [node for node in nodes if node["type"] == "GetNode"]
    set_names: list[str] = []
    get_names: list[str] = []
    for node in setters + getters:
        values = node.get("widgets_values")
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str):
            raise AssertionError(f"{variant}: invalid {node['type']} variable serialization")
        name = values[0]
        properties = node.get("properties", {})
        if properties.get("aux_id") != "kijai/ComfyUI-KJNodes":
            raise AssertionError(f"{variant}: {node['type']} {name} lacks KJNodes ownership")
        if properties.get("Node name for S&R") != node["type"]:
            raise AssertionError(f"{variant}: {node['type']} {name} search metadata drift")
        if "ver" in properties:
            raise AssertionError(f"{variant}: {node['type']} {name} retained revision fingerprint")
        if node["type"] == "SetNode":
            if properties.get("previousName") != name:
                raise AssertionError(f"{variant}: SetNode {name} previousName drift")
            set_names.append(name)
        else:
            if "previousName" in properties:
                raise AssertionError(f"{variant}: GetNode {name} must not carry previousName")
            get_names.append(name)
    set_counts = Counter(set_names)
    if set(set_names) != set(get_names) or any(count != 1 for count in set_counts.values()):
        raise AssertionError(f"{variant}: unmatched or duplicate Set/Get variable ownership")
    if set_counts["driving_video"] != 1 or get_names.count("driving_video") != 1:
        raise AssertionError(f"{variant}: driving_video Set/Get pair drift")
    if "pose_video" in set_names or "pose_video" in get_names:
        raise AssertionError(f"{variant}: obsolete pose_video virtual route restored")

    condition = node_by_type(workflow, "SCAILPose2SCAIL2Condition")
    if condition.get("widgets_values", [None])[0] != "replacement":
        raise AssertionError(f"{variant}: condition mode must be replacement")
    if input_by_name(condition, "pose_video").get("link") is not None:
        raise AssertionError(f"{variant}: replacement pose_video must stay disconnected")
    driving_link = links_by_id[input_by_name(condition, "driving_video")["link"]]
    driving_get = nodes_by_id[int(driving_link[1])]
    if driving_get["type"] != "GetNode" or driving_get.get("widgets_values") != ["driving_video"]:
        raise AssertionError(f"{variant}: raw driving-video route drift")

    adapter = node_by_type(workflow, "SCAILPose2WanVideoSCAIL2Adapter")
    adapter_link = links_by_id[input_by_name(adapter, "condition")["link"]]
    if int(adapter_link[1]) != int(condition["id"]):
        raise AssertionError(f"{variant}: condition-to-adapter route drift")
    embeds = node_by_type(workflow, "WanVideoAddSCAIL2ConditionEmbeds")
    embeds_link = links_by_id[input_by_name(embeds, "condition")["link"]]
    if int(embeds_link[1]) != int(adapter["id"]):
        raise AssertionError(f"{variant}: adapter-to-wrapper route drift")
    empty_embeds = node_by_type(workflow, "WanVideoEmptyEmbeds")
    base_link = links_by_id[input_by_name(embeds, "embeds")["link"]]
    if int(base_link[1]) != int(empty_embeds["id"]):
        raise AssertionError(f"{variant}: empty-embed shape authority drift")

    sampler = node_by_type(workflow, "WanVideoSamplerv2")
    if input_by_name(sampler, "samples").get("link") is not None:
        raise AssertionError(f"{variant}: sampler samples must stay disconnected")
    sampler_embeds_link = links_by_id[input_by_name(sampler, "image_embeds")["link"]]
    if int(sampler_embeds_link[1]) != int(embeds["id"]):
        raise AssertionError(f"{variant}: SCAIL-2 embeds-to-sampler route drift")

    context = node_by_type(workflow, "WanVideoContextOptions")
    context_link = links_by_id[input_by_name(context, "context_frames")["link"]]
    context_value = nodes_by_id[int(context_link[1])]
    if context_value["type"] != "PrimitiveInt" or context_value.get("widgets_values", [None])[0] != 81:
        raise AssertionError(f"{variant}: context_frames must remain 81 pixel frames")

    expected_node_assets = {
        ("WanVideoModelLoader", 0): "select_scail2_model.safetensors",
        ("WanVideoVAELoader", 0): "select_wan_vae.pth",
        ("LoadImage", 0): "select_reference_image.png",
        ("CLIPVisionLoader", 0): "select_clip_vision.safetensors",
        ("WanVideoTextEncodeCached", 0): "select_umt5_text_encoder.safetensors",
        ("CheckpointLoaderSimple", 0): "select_sam3_checkpoint.safetensors",
    }
    for (node_type, index), expected in expected_node_assets.items():
        values = node_by_type(workflow, node_type).get("widgets_values", [])
        if len(values) <= index or values[index] != expected:
            raise AssertionError(f"{variant}: non-placeholder asset in {node_type}")
    video_values = node_by_type(workflow, "VHS_LoadVideo").get("widgets_values", {})
    if video_values.get("video") != "select_driving_video.mp4" or video_values.get("videopreview") != {}:
        raise AssertionError(f"{variant}: driving-video placeholder/preview state drift")
    for video_node in (node for node in nodes if node["type"] == "VHS_VideoCombine"):
        video_values = video_node.get("widgets_values", {})
        if video_values.get("videopreview") != {}:
            raise AssertionError(f"{variant}: output preview state must be empty")
        if not str(video_values.get("filename_prefix", "")).startswith("SCAIL2/"):
            raise AssertionError(f"{variant}: output prefix must be generic")

    lora_values = node_by_type(workflow, "WanVideoLoraSelectMulti").get("widgets_values")
    if lora_values != ["none", 0.0, "none", 0.0, "none", 0.0, "none", 0.0, "none", 0.0, False, False]:
        raise AssertionError(f"{variant}: LoRA selections must be disabled")
    text_values = node_by_type(workflow, "WanVideoTextEncodeCached").get("widgets_values", [])
    if text_values[2] != "Describe the replacement subject and desired scene appearance." or text_values[3] != "low quality, artifacts, distorted anatomy":
        raise AssertionError(f"{variant}: prompt placeholders drift")

    all_strings = list(iter_strings(workflow))
    joined = "\n".join(all_strings)
    forbidden_patterns = {
        "absolute Windows path": r"(?i)(?:^|\s)[a-z]:[\\/]",
        "home/user path": r"(?i)(?:/home/|/users/|\\users\\|/mnt/[a-z]/)",
        "internal path": r"(?i)(?:\.planning|\.sessions|reference/docs|roadmap\.md)",
        "private URL": r"(?i)https?://",
        "secret material": r"(?i)(?:api[_-]?key|access[_-]?token|bearer\s+[a-z0-9])",
        "internal item code": r"\bS2W\d+\b",
    }
    for label, pattern in forbidden_patterns.items():
        if re.search(pattern, joined):
            raise AssertionError(f"{variant}: {label} found")
    selected_assets = {value for value in all_strings if value.lower().endswith(ASSET_SUFFIXES)}
    if selected_assets != EXPECTED_ASSETS:
        raise AssertionError(f"{variant}: unexpected media/model asset set")


def node_without_serialized_links(node: dict[str, Any]) -> dict[str, Any]:
    projection = copy.deepcopy(node)
    for node_input in projection.get("inputs", []):
        node_input.pop("link", None)
    for node_output in projection.get("outputs", []):
        node_output.pop("links", None)
    return projection


def validate_compact_subset(
    recommended: dict[str, Any], compact: dict[str, Any]
) -> None:
    recommended_nodes = {int(node["id"]): node for node in recommended["nodes"]}
    compact_nodes = {int(node["id"]): node for node in compact["nodes"]}
    recommended_ids = set(recommended_nodes)
    compact_ids = set(compact_nodes)
    if not compact_ids <= recommended_ids:
        raise AssertionError("compact: unexpected node outside recommended template")
    if recommended_ids - compact_ids != PREVIEW_ONLY_NODE_IDS:
        raise AssertionError("compact: preview-only node removal set drift")

    recommended_links = {int(link[0]): link for link in recommended["links"]}
    compact_links = {int(link[0]): link for link in compact["links"]}
    expected_link_ids = {
        link_id
        for link_id, link in recommended_links.items()
        if int(link[1]) in compact_ids and int(link[3]) in compact_ids
    }
    if set(compact_links) != expected_link_ids:
        raise AssertionError("compact: retained induced-link set drift")
    for link_id, link in compact_links.items():
        if link != recommended_links[link_id]:
            raise AssertionError(f"compact: retained link tuple drift {link_id}")

    for node_id in compact_ids - {487}:
        if node_without_serialized_links(compact_nodes[node_id]) != node_without_serialized_links(
            recommended_nodes[node_id]
        ):
            raise AssertionError(f"compact: retained node projection drift {node_id}")


class DefaultWorkflowTemplateTests(unittest.TestCase):
    def test_both_default_templates_exist(self) -> None:
        self.assertTrue(DEFAULT_PATH.is_file(), DEFAULT_PATH)
        self.assertTrue(COMPACT_PATH.is_file(), COMPACT_PATH)

    def test_recommended_template_contract(self) -> None:
        validate_public_template(load_workflow(DEFAULT_PATH), "recommended")

    def test_compact_template_contract(self) -> None:
        validate_public_template(load_workflow(COMPACT_PATH), "wanvideowrapper_compact")

    def test_compact_is_recommended_final_output_subset(self) -> None:
        recommended = load_workflow(DEFAULT_PATH)
        compact = load_workflow(COMPACT_PATH)
        validate_compact_subset(recommended, compact)
        compact_ids = {int(node["id"]) for node in compact["nodes"]}
        self.assertIn(139, compact_ids)
        self.assertEqual(
            {node["type"] for node in compact["nodes"] if node["type"].startswith("WanVideo")},
            {node["type"] for node in recommended["nodes"] if node["type"].startswith("WanVideo")},
        )

        rewired = copy.deepcopy(compact)
        rewired_nodes = {int(node["id"]): node for node in rewired["nodes"]}
        rewired_link = next(link for link in rewired["links"] if int(link[0]) == 611)
        self.assertEqual((int(rewired_link[1]), int(rewired_link[2])), (351, 0))
        rewired_nodes[351]["outputs"][0]["links"].remove(611)
        rewired_nodes[415]["outputs"][0].setdefault("links", []).append(611)
        rewired_nodes[415]["outputs"][0]["links"].sort()
        rewired_link[1] = 415
        validate_public_template(rewired, "wanvideowrapper_compact")
        with self.assertRaisesRegex(AssertionError, "retained link tuple drift 611"):
            validate_compact_subset(recommended, rewired)

    def test_validator_rejects_security_and_topology_mutations(self) -> None:
        base = load_workflow(DEFAULT_PATH)
        mutations = []

        samples = copy.deepcopy(base)
        input_by_name(node_by_type(samples, "WanVideoSamplerv2"), "samples")["link"] = 596
        mutations.append(samples)

        dangling = copy.deepcopy(base)
        dangling["links"].append([900, 9999, 0, 139, 0, "IMAGE"])
        mutations.append(dangling)

        private_path = copy.deepcopy(base)
        node_by_type(private_path, "LoadImage")["widgets_values"][0] = r"C:\Users\Example\private.png"
        mutations.append(private_path)

        model_selection = copy.deepcopy(base)
        node_by_type(model_selection, "WanVideoModelLoader")["widgets_values"][0] = "private-model.safetensors"
        mutations.append(model_selection)

        forbidden_node = copy.deepcopy(base)
        forbidden_node["nodes"].append({"id": 999, "type": "RenderNLFPoses", "inputs": [], "outputs": []})
        mutations.append(forbidden_node)

        wrapper_claim = copy.deepcopy(base)
        wrapper_claim["extra"]["scail_pose2_template"]["claims_wrapper_only"] = True
        mutations.append(wrapper_claim)

        unmatched_get = copy.deepcopy(base)
        condition = node_by_type(unmatched_get, "SCAILPose2SCAIL2Condition")
        links_by_id = {int(link[0]): link for link in unmatched_get["links"]}
        driving_link = links_by_id[input_by_name(condition, "driving_video")["link"]]
        driving_get = next(
            node for node in unmatched_get["nodes"] if int(node["id"]) == int(driving_link[1])
        )
        driving_get["widgets_values"] = ["pose_video"]
        mutations.append(unmatched_get)

        missing_kjnodes_owner = copy.deepcopy(base)
        next(node for node in missing_kjnodes_owner["nodes"] if node["type"] == "SetNode")[
            "properties"
        ].pop("aux_id")
        mutations.append(missing_kjnodes_owner)

        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaises(AssertionError):
                    validate_public_template(mutation, "recommended")

    def test_readme_documents_template_operator_contract(self) -> None:
        readme = (ROOT / "readme.md").read_text(encoding="utf-8")
        for required in (
            "## Default Workflow Templates",
            "example_workflows/scail2_replacement_default.json",
            "example_workflows/scail2_replacement_wanvideowrapper_compact.json",
            "ComfyUI-WanVideoWrapper",
            "ComfyUI-VideoHelperSuite",
            "ComfyUI-KJNodes",
            "ComfyUI_Text_Processor",
            "SAM3_VideoTrack",
            "Select every `select_*` placeholder",
            "sampler `samples` input is intentionally disconnected",
            "compact variant has not been independently live-smoke-tested",
            "hard splice",
            "Keep the paired KJNodes Set/Get variable name `driving_video`",
        ):
            with self.subTest(required=required):
                self.assertIn(required, readme)
        self.assertIn("no-samples route is the default", readme)
        self.assertNotIn("Replacement background lock also requires the downstream video encode", readme)

    def test_template_directory_is_not_excluded_from_runtime_archive(self) -> None:
        comfyignore = {
            line.strip()
            for line in (ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertFalse(any(value.rstrip("/") == "example_workflows" for value in comfyignore))


if __name__ == "__main__":
    unittest.main()
