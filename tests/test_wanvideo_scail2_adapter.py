from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

from scail2.condition import build_scail2_condition


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "pkg_wan_adapter_test"

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


@dataclass(frozen=True)
class FakeImageBatch:
    shape: tuple[int, int, int, int]


def solid_frame(rgb, *, height=8, width=8):
    return [[rgb for _col in range(width)] for _row in range(height)]


def frames_from_colors(colors, *, height=8, width=8):
    return [solid_frame(color, height=height, width=width) for color in colors]


def build_condition(*, with_wrapper_images: bool = False):
    ref_image = FakeImageBatch((1, 8, 8, 3)) if with_wrapper_images else "ref"
    pose_video = FakeImageBatch((5, 8, 8, 3)) if with_wrapper_images else "pose"
    return build_scail2_condition(
        mode="replacement",
        ref_image=ref_image,
        ref_mask_frames=frames_from_colors([WHITE]),
        pose_video=pose_video,
        pose_frame_count=5,
        driving_mask_frames=frames_from_colors([RED, GREEN, BLUE, RED, GREEN]),
        width=8,
        height=8,
        additional_ref_images=["extra"],
        additional_ref_masks=[frames_from_colors([BLUE])],
        source_kind="unit_test",
    )


def import_root_package():
    for name in list(sys.modules):
        if name == PACKAGE_NAME or name.startswith(f"{PACKAGE_NAME}."):
            del sys.modules[name]

    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WanVideoSCAIL2AdapterTests(unittest.TestCase):
    def test_full_payload_preserves_condition_and_runtime_masks(self) -> None:
        from scail2.wanvideo_scail2_adapter import (
            build_wanvideo_scail2_adapter_payload,
        )

        condition = build_condition()
        payload = build_wanvideo_scail2_adapter_payload(condition)

        self.assertEqual("wanvideo_scail2_condition_adapter", payload["kind"])
        self.assertEqual(1, payload["version"])
        self.assertEqual(
            "scail_pose2.wanvideo_scail2_payload",
            payload["schema"]["name"],
        )
        self.assertEqual(1, payload["schema"]["version"])
        self.assertEqual(
            "SCAIL2_CONDITION",
            payload["schema"]["condition"]["type_name"],
        )
        self.assertEqual(
            "WanVideoAddSCAIL2ConditionEmbeds",
            payload["schema"]["native_wrapper"]["consumer_node"],
        )
        self.assertEqual(
            "scail2_embeds",
            payload["schema"]["native_wrapper"]["embeds_key"],
        )
        self.assertEqual(
            "reject",
            payload["schema"]["native_wrapper"]["simultaneous_legacy_and_native"],
        )
        self.assertIs(condition, payload["condition"])
        self.assertTrue(payload["target"]["live_wrapper_supported"])
        self.assertEqual("native_scail2_embeds", payload["target"]["current_wrapper_path"])
        self.assertEqual("v1_scail_embeds", payload["target"]["fallback_wrapper_path"])
        self.assertEqual(
            "WanVideoAddSCAIL2ConditionEmbeds",
            payload["target"]["native_consumer_node"],
        )
        self.assertEqual("scail2_embeds", payload["target"]["native_embeds_key"])
        self.assertTrue(payload["target"]["requires_wrapper_scail2_support"])
        self.assertEqual((1, 1, 28, 1, 1), payload["runtime_masks"]["reference"].shape)
        self.assertEqual((1, 2, 28, 1, 1), payload["runtime_masks"]["driving"].shape)
        self.assertEqual(
            [1, 1, 28, 1, 1],
            payload["schema"]["runtime_mask_layouts"]["reference"]["comfy_layout"][
                "shape"
            ],
        )
        self.assertEqual(
            "reference",
            payload["schema"]["runtime_mask_layouts"]["reference"]["layout_role"],
        )
        self.assertEqual(
            "driving",
            payload["schema"]["runtime_mask_layouts"]["driving"]["layout_role"],
        )
        self.assertEqual(
            [28, 2, 1, 1],
            payload["schema"]["runtime_mask_layouts"]["driving"]["scail2_layout"][
                "shape"
            ],
        )
        self.assertTrue(
            payload["schema"]["mask_data_flow"]["native_runtime_masks_authoritative"]
        )
        self.assertFalse(
            payload["schema"]["mask_data_flow"][
                "full_resolution_indices_in_native_payload"
            ]
        )
        self.assertEqual(
            "omitted_from_native_payload",
            payload["rgb_masks"]["indices"],
        )
        self.assertNotIn("reference_indices", payload["rgb_masks"])
        self.assertNotIn("driving_indices", payload["rgb_masks"])
        self.assertEqual(
            ["white", "red", "green", "blue", "yellow", "magenta", "cyan"],
            payload["schema"]["mask_packing"]["color_order"],
        )
        self.assertEqual(1, len(payload["runtime_masks"]["additional_references"]))
        self.assertEqual(
            1,
            payload["schema"]["additional_references"]["count"],
        )
        self.assertEqual(payload["schema"]["identity"], payload["identity"])
        self.assertEqual(3, payload["identity"]["driving_identity_count"])
        self.assertEqual(0, payload["identity"]["reference_identity_count"])
        self.assertEqual([1], payload["identity"]["additional_reference_identity_counts"])
        self.assertEqual(2, payload["identity"]["reference_slot_count"])
        self.assertIn(
            "multi_identity_reference_slots_under_provisioned",
            payload["identity"]["warnings"],
        )
        self.assertEqual(
            1,
            len(
                payload["schema"]["runtime_mask_layouts"][
                    "additional_references"
                ]
            ),
        )
        self.assertEqual(0, condition.ref_mask_indices[0][0][0])
        self.assertEqual(
            1.0,
            payload["runtime_masks"]["reference"].value(
                latent_frame=0,
                channel=0,
            ),
        )
        self.assertEqual(1, condition.driving_mask_indices[0][0][0])
        self.assertEqual({"source_kind": "unit_test"}, payload["source"])
        self.assertNotIn("segment", payload)
        self.assertEqual((), payload["unsupported_current_wrapper_features"])
        self.assertIn("mask_latents_28_channel", payload["legacy_v1_semantic_losses"])
        self.assertNotIn(
            "previous_frame_continuation",
            payload["schema"]["degradation"]["v1_semantic_losses"],
        )
        self.assertEqual((), payload["semantic_losses"])

    def test_schema_metadata_is_json_safe(self) -> None:
        from scail2.wanvideo_scail2_adapter import (
            build_wanvideo_scail2_adapter_payload,
        )

        payload = build_wanvideo_scail2_adapter_payload(build_condition())

        encoded = json.dumps(payload["schema"], sort_keys=True)

        self.assertIn("WanVideoAddSCAIL2ConditionEmbeds", encoded)
        self.assertIn("RuntimeMaskLatent28", encoded)

    def test_native_driving_mask_uses_pose_control_latent_shape(self) -> None:
        from scail2.wanvideo_scail2_adapter import (
            build_wanvideo_scail2_adapter_payload,
        )

        condition = build_scail2_condition(
            mode="animation",
            ref_image="ref",
            ref_mask_frames=frames_from_colors([WHITE], height=16, width=16),
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=frames_from_colors([RED] * 5, height=16, width=16),
            width=16,
            height=16,
            source_kind="unit_test",
        )

        payload = build_wanvideo_scail2_adapter_payload(condition)

        self.assertEqual((16, 16, 5), (
            payload["dimensions"]["width"],
            payload["dimensions"]["height"],
            payload["dimensions"]["num_frames"],
        ))
        self.assertEqual((1, 1, 28, 2, 2), payload["runtime_masks"]["reference"].shape)
        self.assertEqual((1, 2, 28, 1, 1), payload["runtime_masks"]["driving"].shape)
        self.assertEqual(
            [1, 2, 28, 1, 1],
            payload["schema"]["runtime_mask_layouts"]["driving"]["comfy_layout"][
                "shape"
            ],
        )
        self.assertEqual(
            [1, 1, 28, 2, 2],
            payload["schema"]["runtime_mask_layouts"]["reference"]["comfy_layout"][
                "shape"
            ],
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_payload_preserves_tensor_runtime_masks_for_tensor_condition(self) -> None:
        import torch
        from scail2.wanvideo_scail2_adapter import (
            build_wanvideo_scail2_adapter_payload,
        )

        ref_mask = torch.ones((1, 8, 8, 3), dtype=torch.float32)
        driving_mask = torch.zeros((5, 8, 8, 3), dtype=torch.float32)
        driving_mask[0, ..., 0] = 1.0
        driving_mask[1, ..., 1] = 1.0
        driving_mask[2, ..., 2] = 1.0
        driving_mask[3, ..., 0] = 1.0
        driving_mask[4, ..., 1] = 1.0
        condition = build_scail2_condition(
            mode="animation",
            ref_image="ref",
            ref_mask_frames=ref_mask,
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=driving_mask,
            width=8,
            height=8,
            source_kind="unit_test",
        )

        payload = build_wanvideo_scail2_adapter_payload(condition)

        self.assertTrue(torch.is_tensor(condition.driving_mask_indices))
        self.assertTrue(torch.is_tensor(payload["runtime_masks"]["reference"].data))
        self.assertTrue(torch.is_tensor(payload["runtime_masks"]["driving"].data))
        self.assertEqual((1, 1, 28, 1, 1), payload["runtime_masks"]["reference"].shape)
        self.assertEqual((1, 2, 28, 1, 1), payload["runtime_masks"]["driving"].shape)
        self.assertEqual(
            [1, 2, 28, 1, 1],
            payload["schema"]["runtime_mask_layouts"]["driving"]["comfy_layout"][
                "shape"
            ],
        )
        encoded = json.dumps(payload["schema"], sort_keys=True)
        self.assertIn("RuntimeMaskLatent28", encoded)

    def test_lossy_v1_degradation_is_refused_by_default(self) -> None:
        from scail2.wanvideo_scail2_adapter import (
            build_wanvideo_scail2_adapter_payload,
        )

        with self.assertRaisesRegex(ValueError, "Lossy WanVideoWrapper v1 degradation"):
            build_wanvideo_scail2_adapter_payload(
                build_condition(),
                degrade_to_v1=True,
            )

    def test_allowed_v1_degradation_lists_every_semantic_loss(self) -> None:
        from scail2.wanvideo_scail2_adapter import (
            SCAIL2_TO_WANVIDEO_V1_SEMANTIC_LOSSES,
            build_wanvideo_scail2_adapter_payload,
        )

        payload = build_wanvideo_scail2_adapter_payload(
            build_condition(with_wrapper_images=True),
            degrade_to_v1=True,
            allow_degradation=True,
        )

        self.assertTrue(payload["degraded"])
        self.assertEqual(
            SCAIL2_TO_WANVIDEO_V1_SEMANTIC_LOSSES,
            payload["semantic_losses"],
        )
        self.assertEqual(
            "wan_scail_v1_lossy_condition_summary",
            payload["degraded_payload"]["kind"],
        )
        self.assertFalse(payload["schema"]["degradation"]["v1_full_scail2_parity"])
        self.assertEqual(
            list(SCAIL2_TO_WANVIDEO_V1_SEMANTIC_LOSSES),
            payload["schema"]["degradation"]["v1_semantic_losses"],
        )
        self.assertFalse(payload["degraded_payload"]["full_scail2_parity"])
        self.assertEqual(
            "wan_scail_v1_images",
            payload["wan_scail_v1_images"]["kind"],
        )
        self.assertEqual(
            {"ref_image", "pose_images", "clip_ref_image"},
            {
                field
                for field, socket in payload["wan_scail_v1_images"][
                    "socket_map"
                ].items()
                if socket["comfy_type"] == "IMAGE"
            },
        )

    def test_v1_degradation_validates_wrapper_image_shapes(self) -> None:
        from scail2.wanvideo_scail2_adapter import (
            build_wanvideo_scail2_adapter_payload,
        )

        with self.assertRaisesRegex(ValueError, "must expose a BHWC shape"):
            build_wanvideo_scail2_adapter_payload(
                build_condition(),
                degrade_to_v1=True,
                allow_degradation=True,
            )

    def test_adapter_node_is_registered_and_has_no_wrapper_runtime_import(self) -> None:
        package = import_root_package()

        self.assertIn("SCAILPose2WanVideoSCAIL2Adapter", package.NODE_CLASS_MAPPINGS)
        self.assertIn(
            "SCAILPose2WanVideoSCAIL2Adapter",
            package.NODE_DISPLAY_NAME_MAPPINGS,
        )
        node_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2WanVideoSCAIL2Adapter"]
        self.assertEqual(("SCAIL2_WANVIDEO_PAYLOAD",), node_cls.RETURN_TYPES)
        self.assertEqual(("condition",), node_cls.RETURN_NAMES)

        node = node_cls()
        outputs = node.build(build_condition())
        (payload,) = outputs

        self.assertEqual("wanvideo_scail2_condition_adapter", payload["kind"])
        self.assertEqual(1, len(outputs))
        self.assertFalse(any(name.startswith("WanVideoWrapper") for name in sys.modules))

    def test_adapter_node_keeps_degraded_payload_internal_to_condition_output(self) -> None:
        package = import_root_package()
        node_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2WanVideoSCAIL2Adapter"]
        condition = build_condition(with_wrapper_images=True)

        (payload,) = node_cls().build(
            condition,
            degrade_to_v1=True,
            allow_degradation=True,
        )

        self.assertTrue(payload["degraded"])
        wan_scail_images = payload["wan_scail_v1_images"]
        self.assertEqual("wan_scail_v1_images", wan_scail_images["kind"])
        self.assertIs(condition.ref_image, wan_scail_images["ref_image"])
        self.assertIs(condition.pose_video, wan_scail_images["pose_images"])
        self.assertIs(condition.ref_image, wan_scail_images["clip_ref_image"])
        self.assertEqual((8, 8, 5), (
            wan_scail_images["width"],
            wan_scail_images["height"],
            wan_scail_images["num_frames"],
        ))
        self.assertTrue(payload["target"]["live_wrapper_supported"])
        self.assertEqual("v1_scail_embeds", payload["target"]["fallback_wrapper_path"])

    def test_adapter_rejects_non_condition_payload(self) -> None:
        from scail2.wanvideo_scail2_adapter import (
            build_wanvideo_scail2_adapter_payload,
        )

        with self.assertRaisesRegex(ValueError, "SCAIL2_CONDITION"):
            build_wanvideo_scail2_adapter_payload({"type_name": "wrong"})


if __name__ == "__main__":
    unittest.main()
