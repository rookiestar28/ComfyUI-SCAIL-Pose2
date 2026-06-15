from __future__ import annotations

import importlib
import pathlib
import sys
import unittest


class WanVideoContractsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contracts = importlib.import_module("scail2.wanvideo_contracts")

    def test_expected_wrapper_node_contracts_are_declared(self) -> None:
        contracts = self.contracts

        self.assertEqual(
            {
                contracts.NODE_WAN_EMPTY_EMBEDS,
                contracts.NODE_WAN_ADD_SCAIL_REFERENCE,
                contracts.NODE_WAN_ADD_SCAIL_POSE,
                contracts.NODE_WAN_CLIP_VISION_ENCODE,
                contracts.NODE_WAN_SAMPLER_V2,
            },
            set(contracts.WAN_SCAIL_V1_NODE_CONTRACTS),
        )
        self.assertEqual(
            contracts.TYPE_WANVIDIMAGE_EMBEDS,
            contracts.WAN_SCAIL_V1_NODE_CONTRACTS[
                contracts.NODE_WAN_ADD_SCAIL_POSE
            ].return_type,
        )

    def test_adapter_field_mapping_exposes_expected_types(self) -> None:
        contracts = self.contracts
        field_map = contracts.ADAPTER_FIELD_TO_WRAPPER_SOCKET

        expected = {
            "width": (contracts.TYPE_INT, contracts.NODE_WAN_EMPTY_EMBEDS, "width"),
            "height": (contracts.TYPE_INT, contracts.NODE_WAN_EMPTY_EMBEDS, "height"),
            "num_frames": (
                contracts.TYPE_INT,
                contracts.NODE_WAN_EMPTY_EMBEDS,
                "num_frames",
            ),
            "ref_image": (
                contracts.TYPE_IMAGE,
                contracts.NODE_WAN_ADD_SCAIL_REFERENCE,
                "ref_image",
            ),
            "pose_images": (
                contracts.TYPE_IMAGE,
                contracts.NODE_WAN_ADD_SCAIL_POSE,
                "pose_images",
            ),
            "clip_ref_image": (
                contracts.TYPE_IMAGE,
                contracts.NODE_WAN_CLIP_VISION_ENCODE,
                "image_1",
            ),
        }

        for field_name, (comfy_type, wrapper_node, wrapper_socket) in expected.items():
            with self.subTest(field_name=field_name):
                contract = field_map[field_name]
                self.assertEqual(comfy_type, contract.comfy_type)
                self.assertEqual(wrapper_node, contract.wrapper_node)
                self.assertEqual(wrapper_socket, contract.wrapper_socket)

    def test_required_adapter_field_validation(self) -> None:
        contracts = self.contracts
        required = set(contracts.REQUIRED_WAN_SCAIL_V1_ADAPTER_FIELDS)

        contracts.validate_required_adapter_fields(required)
        missing = contracts.missing_required_adapter_fields({"width", "height"})

        self.assertEqual(("num_frames", "ref_image", "pose_images"), missing)
        with self.assertRaisesRegex(ValueError, "num_frames, ref_image, pose_images"):
            contracts.validate_required_adapter_fields({"width", "height"})

    def test_unsupported_scail2_features_are_machine_readable(self) -> None:
        unsupported = set(self.contracts.UNSUPPORTED_CURRENT_WAN_SCAIL2_FEATURES)

        self.assertTrue(
            {
                "rgb_semantic_reference_masks",
                "rgb_semantic_driving_masks",
                "mask_latents_28_channel",
                "replacement_flag_rope_mode",
                "additional_reference_mask_pairs",
                "clean_history_segment_overlap",
                "mask_palette_track_metadata",
            }.issubset(unsupported)
        )

    def test_contract_module_has_no_runtime_wrapper_imports(self) -> None:
        contracts = self.contracts

        self.assertNotIn("torch", contracts.__dict__)
        self.assertFalse(
            any(name.startswith("WanVideoWrapper") for name in sys.modules),
            "contract import must not import WanVideoWrapper modules",
        )

    def test_public_contract_source_does_not_reference_internal_paths(self) -> None:
        source_path = pathlib.Path(self.contracts.__file__)
        source_text = source_path.read_text(encoding="utf-8")
        reference_dir_token = "ref" + "erence/"
        planning_dir_token = "." + "planning"

        self.assertNotIn(reference_dir_token, source_text)
        self.assertNotIn(planning_dir_token, source_text)


if __name__ == "__main__":
    unittest.main()
