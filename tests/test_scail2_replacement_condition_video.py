from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from scail2.replacement_condition_video import build_replacement_condition_video


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "scail_pose2_replacement_condition_video_test_pkg"


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


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
class Scail2ReplacementConditionVideoTests(unittest.TestCase):
    def test_mean_fill_suppresses_subject_and_preserves_background(self) -> None:
        import torch

        driving = torch.full((5, 2, 2, 3), 0.2, dtype=torch.float32)
        driving[:, 0, 0, :] = 0.9
        pose_mask = torch.zeros((5, 2, 2, 3), dtype=torch.float32)
        pose_mask[:, 0, 0, 2] = 1.0

        result = build_replacement_condition_video(
            driving_video=driving,
            pose_video_mask=pose_mask,
            mask_preset="custom",
            grow_pixels=0,
            blur_pixels=0,
            suppression_mode="mean_fill",
            suppression_strength=1.0,
        )

        self.assertEqual((5, 2, 2, 3), tuple(result.driving_video_condition.shape))
        self.assertAlmostEqual(
            0.2,
            float(result.driving_video_condition[0, 0, 0, 0].item()),
            places=6,
        )
        self.assertAlmostEqual(
            0.2,
            float(result.driving_video_condition[0, 1, 1, 0].item()),
            places=6,
        )
        self.assertIn("suppression_mode=mean_fill", result.summary)
        self.assertIn("subject_ratio=0.250000", result.summary)

    def test_zero_strength_passthrough_keeps_video(self) -> None:
        import torch

        driving = torch.rand((5, 3, 3, 3), dtype=torch.float32)
        pose_mask = torch.zeros((5, 3, 3, 3), dtype=torch.float32)
        pose_mask[:, 1, 1, 2] = 1.0

        result = build_replacement_condition_video(
            driving_video=driving,
            pose_video_mask=pose_mask,
            grow_pixels=1,
            blur_pixels=0,
            suppression_mode="black_fill",
            suppression_strength=0.0,
        )

        self.assertTrue(torch.allclose(driving, result.driving_video_condition))
        self.assertIn("suppression_strength=0.000", result.summary)

    def test_noise_fill_is_deterministic_for_seed(self) -> None:
        import torch

        driving = torch.zeros((5, 2, 2, 3), dtype=torch.float32)
        pose_mask = torch.zeros((5, 2, 2, 3), dtype=torch.float32)
        pose_mask[:, 0, 0, 2] = 1.0

        first = build_replacement_condition_video(
            driving_video=driving,
            pose_video_mask=pose_mask,
            grow_pixels=0,
            suppression_mode="noise_fill",
            noise_seed=123,
        )
        second = build_replacement_condition_video(
            driving_video=driving,
            pose_video_mask=pose_mask,
            grow_pixels=0,
            suppression_mode="noise_fill",
            noise_seed=123,
        )

        self.assertTrue(
            torch.allclose(first.driving_video_condition, second.driving_video_condition)
        )
        self.assertGreater(float(first.driving_video_condition[0, 0, 0, 0].item()), 0.0)

    def test_mask_presets_override_numeric_mask_controls(self) -> None:
        import torch

        driving = torch.zeros((5, 65, 65, 3), dtype=torch.float32)
        pose_mask = torch.zeros((5, 65, 65, 3), dtype=torch.float32)
        pose_mask[:, 32, 32, 2] = 1.0

        tight = build_replacement_condition_video(
            driving_video=driving,
            pose_video_mask=pose_mask,
            mask_preset="tight",
            grow_pixels=0,
            blur_pixels=0,
            suppression_mode="white_fill",
        )
        loose = build_replacement_condition_video(
            driving_video=driving,
            pose_video_mask=pose_mask,
            mask_preset="loose",
            grow_pixels=0,
            blur_pixels=0,
            suppression_mode="white_fill",
        )

        self.assertEqual(2, tight.grow_pixels)
        self.assertEqual(16, loose.grow_pixels)
        self.assertGreater(
            loose.diagnostics.subject_ratio,
            tight.diagnostics.subject_ratio,
        )
        self.assertIn("mask_preset=loose", loose.summary)

    def test_shape_mismatch_fails_clearly(self) -> None:
        import torch

        with self.assertRaisesRegex(ValueError, "share frame/spatial shape"):
            build_replacement_condition_video(
                driving_video=torch.zeros((5, 2, 2, 3), dtype=torch.float32),
                pose_video_mask=torch.zeros((4, 2, 2, 3), dtype=torch.float32),
            )

    def test_empty_subject_mask_fails_clearly(self) -> None:
        import torch

        with self.assertRaisesRegex(ValueError, "no subject pixels"):
            build_replacement_condition_video(
                driving_video=torch.zeros((5, 2, 2, 3), dtype=torch.float32),
                pose_video_mask=torch.zeros((5, 2, 2, 3), dtype=torch.float32),
            )

    def test_node_registration_and_contract(self) -> None:
        package = import_root_package()

        self.assertIn(
            "SCAILPose2ReplacementConditionVideo",
            package.NODE_CLASS_MAPPINGS,
        )
        node_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2ReplacementConditionVideo"]
        self.assertEqual(("IMAGE", "STRING"), node_cls.RETURN_TYPES)
        self.assertEqual(("driving_video_condition", "summary"), node_cls.RETURN_NAMES)
        input_types = node_cls.INPUT_TYPES()
        self.assertEqual(
            (
                "driving_video",
                "pose_video_mask",
                "mask_preset",
                "grow_pixels",
                "blur_pixels",
                "suppression_mode",
                "suppression_strength",
                "noise_seed",
            ),
            tuple(input_types["required"]),
        )


if __name__ == "__main__":
    unittest.main()
