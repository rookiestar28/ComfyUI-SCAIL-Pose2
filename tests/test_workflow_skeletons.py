from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from scail2 import wanvideo_contracts


ROOT = Path(__file__).resolve().parents[1]
SKELETON_DIR = ROOT / "workflow_skeletons"


def load_skeleton(name: str):
    return json.loads((SKELETON_DIR / name).read_text(encoding="utf-8"))


class WorkflowSkeletonTests(unittest.TestCase):
    def test_all_skeletons_parse_and_use_local_schema(self) -> None:
        for path in sorted(SKELETON_DIR.glob("*.json")):
            with self.subTest(path=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual("scail_pose2.workflow_skeleton.v1", data["schema"])
                self.assertEqual("static_only", data["execution"])
                self.assertEqual("deferred", data["live_generation"])
                self.assertTrue(data["nodes"])

    def test_v1_pose_control_skeleton_matches_wan_scail_contracts(self) -> None:
        data = load_skeleton("wan_scail_v1_pose_control.json")
        class_types = {node["class_type"] for node in data["nodes"]}
        links = {(tuple(link["to"]), link["type"]) for link in data["links"]}

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
        self.assertTrue(
            {
                "mode",
                "replace_flag",
                "driving_mask_indices",
                "source_kind",
                "previous_frame_count",
                "video_frame_offset",
            }.issubset(fields)
        )
        self.assertEqual(
            "v1_scail_embeds",
            data["wanvideo_scail2_adapter"]["target"]["current_wrapper_path"],
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
        self.assertFalse(
            data["wanvideo_scail2_adapter"]["target"]["live_wrapper_supported"]
        )
        self.assertEqual(
            set(wanvideo_contracts.UNSUPPORTED_CURRENT_WAN_SCAIL2_FEATURES),
            set(data["unsupported_current_wan_scail_features"]),
        )
        self.assertEqual(
            "SCAIL_WAN_SCAIL_IMAGES",
            next(
                node
                for node in data["nodes"]
                if node["id"] == "wanvideo_scail2_adapter"
            )["v1_compat_output_type"],
        )
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
                    "v1_outputs_when_enabled"
                ][key]
                for key in ("ref_image", "pose_images", "width", "height", "num_frames")
            },
        )

    def test_wananimate_fallback_skeleton_requires_explicit_degradation(self) -> None:
        data = load_skeleton("wananimate_fallback.json")
        class_types = {node.get("class_type") for node in data["nodes"]}
        adapter = next(
            node for node in data["nodes"] if node["id"] == "wananimate_fallback_adapter"
        )

        self.assertIn("WanVideoAnimateEmbeds", class_types)
        self.assertFalse(adapter["allow_semantic_degradation_default"])
        self.assertFalse(data["degradation"]["is_full_scail2_parity"])
        self.assertTrue(data["degradation"]["requires_explicit_enable"])
        self.assertIn(
            "rgb_semantic_masks_collapsed_to_binary_grayscale",
            data["degradation"]["semantic_losses"],
        )

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
