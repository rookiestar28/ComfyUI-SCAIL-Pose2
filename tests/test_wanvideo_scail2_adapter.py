from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from scail2.condition import build_scail2_condition


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "pkg_wan_adapter_test"

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def solid_frame(rgb, *, height=8, width=8):
    return [[rgb for _col in range(width)] for _row in range(height)]


def frames_from_colors(colors, *, height=8, width=8):
    return [solid_frame(color, height=height, width=width) for color in colors]


def build_condition():
    return build_scail2_condition(
        mode="replacement",
        ref_image="ref",
        ref_mask_frames=frames_from_colors([WHITE]),
        pose_video="pose",
        pose_frame_count=5,
        driving_mask_frames=frames_from_colors([RED, GREEN, BLUE, RED, GREEN]),
        width=8,
        height=8,
        segment_len=81,
        segment_overlap=5,
        additional_ref_images=["extra"],
        additional_ref_masks=[frames_from_colors([BLUE])],
        source_kind="unit_test",
        previous_frame_count=2,
        video_frame_offset=4,
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
        self.assertIs(condition, payload["condition"])
        self.assertFalse(payload["target"]["live_wrapper_supported"])
        self.assertEqual("v1_scail_embeds", payload["target"]["current_wrapper_path"])
        self.assertTrue(payload["target"]["requires_wrapper_scail2_support"])
        self.assertEqual((1, 1, 28, 1, 1), payload["runtime_masks"]["reference"].shape)
        self.assertEqual((1, 2, 28, 1, 1), payload["runtime_masks"]["driving"].shape)
        self.assertEqual(1, len(payload["runtime_masks"]["additional_references"]))
        self.assertEqual(1, condition.driving_mask_indices[0][0][0])
        self.assertIn(
            "mask_latents_28_channel",
            payload["unsupported_current_wrapper_features"],
        )
        self.assertEqual((), payload["semantic_losses"])

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
            build_condition(),
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
        self.assertFalse(payload["degraded_payload"]["full_scail2_parity"])

    def test_adapter_node_is_registered_and_has_no_wrapper_runtime_import(self) -> None:
        package = import_root_package()

        self.assertIn("SCAILPose2WanVideoSCAIL2Adapter", package.NODE_CLASS_MAPPINGS)
        self.assertIn(
            "SCAILPose2WanVideoSCAIL2Adapter",
            package.NODE_DISPLAY_NAME_MAPPINGS,
        )
        node_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2WanVideoSCAIL2Adapter"]
        self.assertEqual(("SCAIL2_WANVIDEO_PAYLOAD", "STRING"), node_cls.RETURN_TYPES)

        node = node_cls()
        payload, summary = node.build(build_condition())

        self.assertEqual("wanvideo_scail2_condition_adapter", payload["kind"])
        self.assertIn("live_wrapper_supported=False", summary)
        self.assertFalse(any(name.startswith("WanVideoWrapper") for name in sys.modules))

    def test_adapter_rejects_non_condition_payload(self) -> None:
        from scail2.wanvideo_scail2_adapter import (
            build_wanvideo_scail2_adapter_payload,
        )

        with self.assertRaisesRegex(ValueError, "SCAIL2_CONDITION"):
            build_wanvideo_scail2_adapter_payload({"type_name": "wrong"})


if __name__ == "__main__":
    unittest.main()
