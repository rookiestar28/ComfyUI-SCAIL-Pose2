from __future__ import annotations

import importlib.util
import unittest

from scail2.replacement_mask import (
    SCAIL_POSE2_CONDITION_MODE_ATTR,
    SCAIL_POSE2_DISABLE_SAMPLES_ATTR,
)
from scail2.replacement_preview import build_replacement_preview_schedule
from scail2.replacement_preview import classify_replacement_preview_path
from scail2.replacement_preview import summarize_replacement_preview_schedule


class ReplacementPreviewContractTests(unittest.TestCase):
    def test_binary_replacement_mask_preserves_background_across_preview_steps(self) -> None:
        schedule = build_replacement_preview_schedule(
            [[1.0, 0.0], [1.0, 0.0]],
            step_count=4,
        )

        self.assertEqual(4, len(schedule))
        self.assertEqual(0, schedule[0].step_index)
        self.assertEqual(0.0, schedule[0].threshold)
        for frame in schedule:
            self.assertAlmostEqual(0.5, frame.preserve_ratio)
            self.assertAlmostEqual(0.5, frame.replace_ratio)

    def test_all_subject_replacement_mask_has_no_background_preserve_area(self) -> None:
        schedule = build_replacement_preview_schedule(
            [[1.0, 1.0], [1.0, 1.0]],
            step_count=3,
        )

        for frame in schedule:
            self.assertEqual(0.0, frame.preserve_ratio)
            self.assertEqual(1.0, frame.replace_ratio)

    def test_preview_schedule_summary_is_shape_and_ratio_only(self) -> None:
        summary = summarize_replacement_preview_schedule(
            [[1.0, 0.0], [1.0, 0.0]],
            step_count=2,
        )

        self.assertIn("mask_shape=(2, 2)", summary)
        self.assertIn("steps=2", summary)
        self.assertIn("first_preserve_ratio=0.500000", summary)
        self.assertIn("last_replace_ratio=0.500000", summary)

    def test_wired_path_with_samples_noise_explains_early_preview(self) -> None:
        diagnostic = classify_replacement_preview_path(
            samples_present=True,
            noise_mask_present=True,
            add_noise_to_samples=True,
            condition_mode="replacement",
        )

        self.assertEqual("wired_noisy_preview_expected", diagnostic.status)
        self.assertTrue(diagnostic.background_lock_expected)
        self.assertFalse(diagnostic.early_preview_original_background_reliable)
        self.assertIn("early preview is noised", diagnostic.reason)

    def test_missing_samples_or_noise_mask_are_not_background_lock(self) -> None:
        missing_samples = classify_replacement_preview_path(
            samples_present=False,
            noise_mask_present=True,
            add_noise_to_samples=True,
            condition_mode="replacement",
        )
        missing_mask = classify_replacement_preview_path(
            samples_present=True,
            noise_mask_present=False,
            add_noise_to_samples=True,
            condition_mode="replacement",
        )

        self.assertEqual("missing_samples", missing_samples.status)
        self.assertFalse(missing_samples.background_lock_expected)
        self.assertEqual("missing_noise_mask", missing_mask.status)
        self.assertFalse(missing_mask.background_lock_expected)

    def test_animation_metadata_classifies_samples_path_as_disabled(self) -> None:
        class Mask:
            pass

        mask = Mask()
        setattr(mask, SCAIL_POSE2_DISABLE_SAMPLES_ATTR, True)
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "animation")

        diagnostic = classify_replacement_preview_path(
            samples_present=True,
            noise_mask_present=True,
            add_noise_to_samples=True,
            samples_disabled=getattr(mask, SCAIL_POSE2_DISABLE_SAMPLES_ATTR, False),
            condition_mode=getattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, None),
        )

        self.assertEqual("samples_disabled", diagnostic.status)
        self.assertFalse(diagnostic.background_lock_expected)

    def test_rejects_empty_or_denormalized_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            build_replacement_preview_schedule([], step_count=2)
        with self.assertRaisesRegex(ValueError, "normalized"):
            build_replacement_preview_schedule([[255]], step_count=2)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            build_replacement_preview_schedule([[1.0]], step_count=0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_torch_noise_mask_matches_list_contract(self) -> None:
        import torch

        schedule = build_replacement_preview_schedule(
            torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            step_count=2,
        )

        self.assertEqual(2, len(schedule))
        self.assertAlmostEqual(0.5, schedule[0].preserve_ratio)
        self.assertAlmostEqual(0.5, schedule[1].replace_ratio)


if __name__ == "__main__":
    unittest.main()
