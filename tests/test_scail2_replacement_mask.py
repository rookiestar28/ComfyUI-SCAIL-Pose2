from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import scail2.replacement_mask as replacement_mask_module
from scail2.condition import build_scail2_condition
from scail2.replacement_mask import build_replacement_denoise_mask
from scail2.replacement_mask import (
    SCAIL_POSE2_CONDITION_MODE_ATTR,
    SCAIL_POSE2_DISABLE_SAMPLES_ATTR,
    SCAIL_POSE2_DISABLE_SAMPLES_REASON_ATTR,
    SCAIL_POSE2_MASK_ROLE_ATTR,
    SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "scail_pose2_test_pkg"

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


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


def frames_from_pixels(pixels, *, frames=5):
    return [[pixels] for _frame in range(frames)]


def solid_frame(rgb, *, height, width):
    return [[rgb for _col in range(width)] for _row in range(height)]


def single_pixel_subject_frames(*, frames=5, height=10, width=10):
    pose_mask = []
    for _frame in range(frames):
        frame = solid_frame(BLACK, height=height, width=width)
        frame[0][0] = BLUE
        pose_mask.append(frame)
    return pose_mask


def replacement_condition(*, pose_mask, width=2, height=1, frames=5):
    return build_scail2_condition(
        mode="replacement",
        ref_image="ref",
        ref_mask_frames=[solid_frame(BLUE, height=height, width=width)],
        pose_video="pose",
        pose_frame_count=frames,
        driving_mask_frames=pose_mask,
        width=width,
        height=height,
    )


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
class Scail2ReplacementMaskTests(unittest.TestCase):
    def test_tuple_pose_mask_outputs_subject_one_background_zero(self) -> None:
        pose_mask = frames_from_pixels([BLUE, BLACK])
        condition = replacement_condition(pose_mask=pose_mask)

        result = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_mask,
            grow_pixels=0,
        )

        self.assertEqual((5, 1, 2), tuple(result.mask.shape))
        self.assertEqual(1.0, float(result.mask[0, 0, 0].item()))
        self.assertEqual(0.0, float(result.mask[0, 0, 1].item()))
        self.assertFalse(getattr(result.mask, SCAIL_POSE2_DISABLE_SAMPLES_ATTR, False))
        self.assertEqual(
            "replacement",
            getattr(result.mask, SCAIL_POSE2_CONDITION_MODE_ATTR, None),
        )
        self.assertEqual(
            SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE,
            getattr(result.mask, SCAIL_POSE2_MASK_ROLE_ATTR, None),
        )
        self.assertIn("subject_ratio=0.500000", result.summary)
        self.assertAlmostEqual(0.5, result.raw_subject_ratio)
        self.assertEqual("none", result.coverage_warning)
        self.assertIn("raw_subject_ratio=0.500000", result.summary)
        self.assertIn("final_subject_ratio=0.500000", result.summary)

    def test_low_coverage_subject_mask_reports_warning_without_blocking(self) -> None:
        pose_mask = single_pixel_subject_frames()
        condition = replacement_condition(
            pose_mask=pose_mask,
            width=10,
            height=10,
            frames=5,
        )

        result = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_mask,
            mask_preset="custom",
            grow_pixels=0,
            blur_pixels=0,
            lower_contact_refine=False,
        )

        self.assertEqual((5, 10, 10), tuple(result.mask.shape))
        self.assertAlmostEqual(0.01, result.raw_subject_ratio)
        self.assertAlmostEqual(0.01, result.subject_ratio)
        self.assertEqual("low_subject_coverage", result.coverage_warning)
        self.assertEqual(0, result.coverage_empty_frame_count)
        self.assertEqual(5, result.coverage_sparse_frame_count)
        self.assertEqual(5, result.coverage_longest_sparse_streak)
        self.assertIn("raw_subject_ratio=0.010000", result.summary)
        self.assertIn("final_subject_ratio=0.010000", result.summary)
        self.assertIn("coverage_warning=low_subject_coverage", result.summary)
        self.assertIn(
            "coverage_limitation=missing_regions_not_recovered_by_grow_blur",
            result.summary,
        )

    def test_animation_condition_is_allowed_by_default(self) -> None:
        pose_mask = frames_from_pixels([BLUE, BLACK])
        condition = build_scail2_condition(
            mode="animation",
            ref_image="ref",
            ref_mask_frames=[[[WHITE, WHITE]]],
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=pose_mask,
            width=2,
            height=1,
        )

        result = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_mask,
            grow_pixels=0,
        )

        self.assertEqual((5, 1, 2), tuple(result.mask.shape))
        self.assertEqual(1.0, float(result.mask[0, 0, 0].item()))
        self.assertEqual(1.0, float(result.mask[0, 0, 1].item()))
        self.assertEqual("mode_passthrough", result.fast_path)
        self.assertTrue(getattr(result.mask, SCAIL_POSE2_DISABLE_SAMPLES_ATTR, False))
        self.assertEqual(
            "non_replacement_mode",
            getattr(result.mask, SCAIL_POSE2_DISABLE_SAMPLES_REASON_ATTR, None),
        )
        self.assertEqual(
            "animation",
            getattr(result.mask, SCAIL_POSE2_CONDITION_MODE_ATTR, None),
        )
        self.assertEqual(
            SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE,
            getattr(result.mask, SCAIL_POSE2_MASK_ROLE_ATTR, None),
        )
        self.assertIn("mode=animation", result.summary)
        self.assertIn("background_lock=disabled", result.summary)
        self.assertIn("samples_path=disabled", result.summary)

    def test_explicit_strict_mode_rejects_animation_condition(self) -> None:
        pose_mask = frames_from_pixels([BLUE, BLACK])
        condition = build_scail2_condition(
            mode="animation",
            ref_image="ref",
            ref_mask_frames=[[[WHITE, WHITE]]],
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=pose_mask,
            width=2,
            height=1,
        )

        with self.assertRaisesRegex(ValueError, "replacement mode"):
            build_replacement_denoise_mask(
                condition=condition,
                pose_video_mask=pose_mask,
                strict_replacement_mode=True,
            )

    def test_empty_subject_mask_fails_clearly(self) -> None:
        pose_mask = frames_from_pixels([BLACK, BLACK])
        condition = replacement_condition(pose_mask=pose_mask)

        with self.assertRaisesRegex(ValueError, "no subject pixels"):
            build_replacement_denoise_mask(
                condition=condition,
                pose_video_mask=pose_mask,
            )

    def test_shape_mismatch_fails_clearly(self) -> None:
        pose_mask = frames_from_pixels([BLUE, BLACK])
        condition = replacement_condition(pose_mask=pose_mask)

        with self.assertRaisesRegex(ValueError, "width"):
            build_replacement_denoise_mask(
                condition=condition,
                pose_video_mask=frames_from_pixels([BLUE, BLACK, BLACK]),
            )

    def test_tensor_pose_mask_preserves_frame_and_spatial_shape(self) -> None:
        import torch

        pose_mask = torch.zeros((5, 2, 2, 3), dtype=torch.float32)
        pose_mask[:, 0, 0, :] = torch.tensor((0.0, 0.0, 1.0))
        condition = replacement_condition(
            pose_mask=pose_mask,
            width=2,
            height=2,
            frames=5,
        )

        result = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_mask,
            grow_pixels=0,
        )

        self.assertEqual((5, 2, 2), tuple(result.mask.shape))
        self.assertEqual(torch.float32, result.mask.dtype)
        self.assertEqual("cpu", result.mask.device.type)
        self.assertEqual(1.0, float(result.mask[0, 0, 0].item()))
        self.assertEqual(0.0, float(result.mask[0, 1, 1].item()))
        self.assertEqual("tensor_subject_mask", result.fast_path)
        self.assertIn("fast_path=tensor_subject_mask", result.summary)
        self.assertIn("input_device=cpu", result.summary)
        self.assertIn("output_device=cpu", result.summary)

    def test_tensor_fast_path_does_not_reparse_semantic_indices(self) -> None:
        import torch

        pose_mask = torch.zeros((5, 2, 2, 3), dtype=torch.float32)
        pose_mask[:, 0, 0, :] = torch.tensor((0.0, 0.0, 1.0))
        condition = replacement_condition(
            pose_mask=pose_mask,
            width=2,
            height=2,
            frames=5,
        )
        original = replacement_mask_module.semantic_mask_indices_tensor_raw

        def fail_if_called(_value):
            raise AssertionError("tensor fast path must not call semantic index parser")

        replacement_mask_module.semantic_mask_indices_tensor_raw = fail_if_called
        try:
            result = replacement_mask_module.build_replacement_denoise_mask(
                condition=condition,
                pose_video_mask=pose_mask,
                grow_pixels=0,
            )
        finally:
            replacement_mask_module.semantic_mask_indices_tensor_raw = original

        self.assertEqual("tensor_subject_mask", result.fast_path)
        self.assertEqual((5, 2, 2), tuple(result.mask.shape))
        self.assertEqual(torch.float32, result.mask.dtype)
        self.assertEqual("cpu", result.mask.device.type)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and __import__("torch").cuda.is_available(),
        "CUDA is unavailable",
    )
    def test_tensor_fast_path_uses_cuda_when_available(self) -> None:
        import torch

        pose_mask = torch.zeros((5, 2, 2, 3), dtype=torch.float32)
        pose_mask[:, 0, 0, :] = torch.tensor((0.0, 0.0, 1.0))
        condition = replacement_condition(
            pose_mask=pose_mask,
            width=2,
            height=2,
            frames=5,
        )

        result = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_mask,
            grow_pixels=0,
        )

        self.assertEqual("tensor_subject_mask", result.fast_path)
        self.assertTrue(result.work_device.startswith("cuda"))
        self.assertEqual("cpu", result.output_device)

    def test_grow_and_blur_controls_are_deterministic(self) -> None:
        pose_mask = []
        for _frame in range(5):
            pose_mask.append(
                [
                    [BLACK, BLACK, BLACK],
                    [BLACK, BLUE, BLACK],
                    [BLACK, BLACK, BLACK],
                ]
            )
        condition = replacement_condition(
            pose_mask=pose_mask,
            width=3,
            height=3,
            frames=5,
        )

        grown = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_mask,
            grow_pixels=1,
            blur_pixels=0,
        )
        blurred = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_mask,
            grow_pixels=1,
            blur_pixels=1,
        )

        self.assertEqual(1.0, float(grown.mask[0, 0, 0].item()))
        self.assertEqual(1.0, float(grown.mask[0, 2, 2].item()))
        self.assertEqual(1.0, float(blurred.mask[0, 1, 1].item()))
        self.assertAlmostEqual(4.0 / 9.0, float(blurred.mask[0, 0, 0].item()))

    def test_lower_contact_refine_covers_foot_contact_without_global_overgrow(self) -> None:
        pose_mask = []
        for _frame in range(5):
            frame = [[BLACK for _col in range(12)] for _row in range(12)]
            for row in range(1, 8):
                frame[row][5] = BLUE
                frame[row][6] = BLUE
            for row in range(8, 11):
                frame[row][6] = BLUE
            frame[10][7] = BLUE
            pose_mask.append(frame)
        condition = replacement_condition(
            pose_mask=pose_mask,
            width=12,
            height=12,
            frames=5,
        )

        baseline = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_mask,
            grow_pixels=0,
            blur_pixels=0,
            lower_contact_refine=False,
        )
        refined = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_mask,
            grow_pixels=0,
            blur_pixels=0,
            lower_contact_refine=True,
            lower_contact_grow_pixels=1,
            lower_contact_band_ratio=0.35,
            lower_contact_area_cap_ratio=1.0,
        )

        self.assertEqual(0.0, float(baseline.mask[0, 10, 8].item()))
        self.assertEqual(1.0, float(refined.mask[0, 10, 8].item()))
        self.assertEqual(0.0, float(refined.mask[0, 4, 4].item()))
        self.assertEqual(0.0, float(refined.mask[0, 0, 0].item()))
        self.assertIn("lower_contact_refine=True", refined.summary)
        self.assertIn("lower_contact_refined_frames=5", refined.summary)
        self.assertIn("lower_contact_area_delta_ratio=", refined.summary)

    def test_mask_preset_overrides_numeric_controls(self) -> None:
        pose_mask = []
        for _frame in range(5):
            pose_mask.append(
                [
                    [BLACK, BLACK, BLACK, BLACK, BLACK],
                    [BLACK, BLACK, BLACK, BLACK, BLACK],
                    [BLACK, BLACK, BLUE, BLACK, BLACK],
                    [BLACK, BLACK, BLACK, BLACK, BLACK],
                    [BLACK, BLACK, BLACK, BLACK, BLACK],
                ]
            )
        condition = replacement_condition(
            pose_mask=pose_mask,
            width=5,
            height=5,
            frames=5,
        )

        custom = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_mask,
            mask_preset="custom",
            grow_pixels=0,
            blur_pixels=0,
        )
        tight = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_mask,
            mask_preset="tight",
            grow_pixels=0,
            blur_pixels=0,
        )

        self.assertAlmostEqual(1.0 / 25.0, custom.subject_ratio)
        self.assertGreater(tight.subject_ratio, custom.subject_ratio)
        self.assertEqual("tight", tight.mask_preset)
        self.assertIn("mask_preset=tight", tight.summary)
        self.assertIn("edge_contact_ratio=", tight.summary)
        self.assertIsNotNone(tight.diagnostics)

    def test_background_lock_polarity_preserves_background_in_samples_path(self) -> None:
        import torch

        pose_mask = frames_from_pixels([BLUE, BLACK])
        result = build_replacement_denoise_mask(
            condition=replacement_condition(pose_mask=pose_mask),
            pose_video_mask=pose_mask,
            grow_pixels=0,
        )

        original_latent = torch.full((3, 5, 1, 2), 10.0)
        generated_latent = torch.full((3, 5, 1, 2), -5.0)
        preserve_mask = (1.0 - result.mask).unsqueeze(0).expand_as(original_latent) > 0
        merged = torch.where(preserve_mask, original_latent, generated_latent)

        self.assertEqual(-5.0, float(merged[0, 0, 0, 0].item()))
        self.assertEqual(10.0, float(merged[0, 0, 0, 1].item()))
        self.assertEqual(-5.0, float(merged[2, 4, 0, 0].item()))
        self.assertEqual(10.0, float(merged[2, 4, 0, 1].item()))

    def test_animation_mode_disables_samples_background_lock(self) -> None:
        import torch

        pose_mask = frames_from_pixels([BLUE, BLACK])
        condition = build_scail2_condition(
            mode="animation",
            ref_image="ref",
            ref_mask_frames=[[[WHITE, WHITE]]],
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=pose_mask,
            width=2,
            height=1,
        )

        result = build_replacement_denoise_mask(
            condition=condition,
            pose_video_mask=pose_mask,
            grow_pixels=0,
        )

        original_latent = torch.full((3, 5, 1, 2), 10.0)
        generated_latent = torch.full((3, 5, 1, 2), -5.0)
        preserve_mask = (1.0 - result.mask).unsqueeze(0).expand_as(original_latent) > 0
        merged = torch.where(preserve_mask, original_latent, generated_latent)

        self.assertFalse(bool(preserve_mask.any().item()))
        self.assertEqual(-5.0, float(merged[0, 0, 0, 0].item()))
        self.assertEqual(-5.0, float(merged[0, 0, 0, 1].item()))
        self.assertEqual("mode_passthrough", result.fast_path)
        self.assertTrue(getattr(result.mask, SCAIL_POSE2_DISABLE_SAMPLES_ATTR, False))
        self.assertIn("background_lock=disabled", result.summary)
        self.assertIn("samples_path=disabled", result.summary)

    def test_node_is_registered_with_mask_output_contract(self) -> None:
        package = import_root_package()

        self.assertIn("SCAILPose2ReplacementDenoiseMask", package.NODE_CLASS_MAPPINGS)
        node_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2ReplacementDenoiseMask"]
        self.assertEqual(("MASK", "STRING"), node_cls.RETURN_TYPES)
        self.assertEqual(("mask", "summary"), node_cls.RETURN_NAMES)
        input_types = node_cls.INPUT_TYPES()
        self.assertEqual(
            (
                "condition",
                "pose_video_mask",
                "mask_preset",
                "grow_pixels",
                "blur_pixels",
            ),
            tuple(input_types["required"]),
        )

    def test_node_allows_animation_condition_without_strict_widget(self) -> None:
        package = import_root_package()
        node_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2ReplacementDenoiseMask"]
        package_condition = __import__(
            f"{PACKAGE_NAME}.scail2.condition",
            fromlist=["build_scail2_condition"],
        )
        pose_mask = frames_from_pixels([BLUE, BLACK])
        condition = package_condition.build_scail2_condition(
            mode="animation",
            ref_image="ref",
            ref_mask_frames=[[[WHITE, WHITE]]],
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=pose_mask,
            width=2,
            height=1,
        )

        mask, summary = node_cls().build(
            condition=condition,
            pose_video_mask=pose_mask,
            grow_pixels=0,
            blur_pixels=0,
        )

        self.assertEqual((5, 1, 2), tuple(mask.shape))
        self.assertEqual(1.0, float(mask[0, 0, 0].item()))
        self.assertEqual(1.0, float(mask[0, 0, 1].item()))
        self.assertTrue(getattr(mask, SCAIL_POSE2_DISABLE_SAMPLES_ATTR, False))
        self.assertIn("mode=animation", summary)
        self.assertIn("background_lock=disabled", summary)
        self.assertIn("samples_path=disabled", summary)

    def test_core_path_does_not_import_wanvideowrapper(self) -> None:
        build_replacement_denoise_mask(
            condition=replacement_condition(pose_mask=frames_from_pixels([BLUE, BLACK])),
            pose_video_mask=frames_from_pixels([BLUE, BLACK]),
            grow_pixels=0,
        )

        self.assertFalse(any(name.startswith("WanVideoWrapper") for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
