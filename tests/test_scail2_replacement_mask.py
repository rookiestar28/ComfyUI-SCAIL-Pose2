from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from scail2.condition import build_scail2_condition
from scail2.replacement_mask import build_replacement_denoise_mask


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
        self.assertIn("subject_ratio=0.500000", result.summary)

    def test_strict_mode_rejects_animation_condition(self) -> None:
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
                "grow_pixels",
                "blur_pixels",
                "strict_replacement_mode",
                "invert",
            ),
            tuple(input_types["required"]),
        )

    def test_core_path_does_not_import_wanvideowrapper(self) -> None:
        build_replacement_denoise_mask(
            condition=replacement_condition(pose_mask=frames_from_pixels([BLUE, BLACK])),
            pose_video_mask=frames_from_pixels([BLUE, BLACK]),
            grow_pixels=0,
        )

        self.assertFalse(any(name.startswith("WanVideoWrapper") for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
