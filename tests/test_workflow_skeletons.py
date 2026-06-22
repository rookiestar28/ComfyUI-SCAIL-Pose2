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
        self.assertEqual(
            {
                "animation": "pose_video",
                "replacement": "driving_video",
            },
            condition_node["mode_video_sources"],
        )
        self.assertTrue(
            {
                "mode",
                "replace_flag",
                "driving_mask_indices",
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

        self.assertTrue(
            {
                "SAM3_VideoTrack",
                "SCAILPose2ColoredMask",
                "SCAILPose2SCAIL2Condition",
                "SCAILPose2WanVideoSCAIL2Adapter",
                "WanVideoAddSCAIL2ConditionEmbeds",
                "WanVideoContextOptions",
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
                ("wan_sampler", "context_options"),
                "WANVIDCONTEXT",
            ),
            links,
        )
        context = data["context"]
        self.assertEqual("ComfyUI-WanVideoWrapper", context["owner"])
        self.assertEqual("WanVideoContextOptions", context["owner_node"])
        self.assertEqual("context_options", context["sampler_socket"])
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
        self.assertFalse(data["degradation"]["v1_fallback_is_full_scail2_parity"])

    def test_replacement_background_lock_skeleton_wires_samples_mask_path(self) -> None:
        data = load_skeleton("wanvideo_replacement_background_lock.json")
        class_types = {node.get("class_type") for node in data["nodes"]}
        links = {
            (tuple(link["from"]), tuple(link["to"]), link["type"])
            for link in data["links"]
        }

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
        contract = data["background_lock_contract"]
        self.assertEqual("driving_video", contract["encode_driving_video_socket"])
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
