from __future__ import annotations

import importlib
import sys
import unittest

from scail2.condition import TYPE_SCAIL2_CONDITION, build_scail2_condition
from scail2.masks import (
    BACKGROUND_INDEX,
    SEMANTIC_MASK_COLORS,
    classify_rgb_semantic_color,
    pack_semantic_mask_indices_to_28_channels,
    semantic_mask_indices,
)


BLACK = (0, 0, 0)


def solid_frame(rgb, *, height=1, width=1):
    return [[rgb for _col in range(width)] for _row in range(height)]


def frames_from_colors(colors, *, height=1, width=1):
    return [solid_frame(color, height=height, width=width) for color in colors]


class Scail2ConditionMaskCoreTests(unittest.TestCase):
    def test_palette_classifies_all_semantic_colors_and_background(self) -> None:
        colors = [color.rgb for color in SEMANTIC_MASK_COLORS] + [BLACK]
        indices = semantic_mask_indices([[colors]])

        self.assertEqual(
            tuple(range(7)) + (BACKGROUND_INDEX,),
            indices[0][0],
        )
        self.assertEqual(BACKGROUND_INDEX, classify_rgb_semantic_color(BLACK))

    def test_strict_palette_rejects_ambiguous_non_semantic_colors(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            classify_rgb_semantic_color((128, 0, 0))

    def test_temporal_mask_packing_accepts_one_five_and_eighty_one_frames(self) -> None:
        one_frame = semantic_mask_indices(frames_from_colors([(255, 255, 255)]))
        five_frames = semantic_mask_indices(
            frames_from_colors(
                [
                    (255, 255, 255),
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (0, 255, 255),
                ]
            )
        )
        eighty_one_frames = semantic_mask_indices(
            frames_from_colors([(0, 0, 0)] * 80 + [(255, 0, 255)])
        )

        latent_one = pack_semantic_mask_indices_to_28_channels(one_frame)
        latent_five = pack_semantic_mask_indices_to_28_channels(five_frames)
        latent_eighty_one = pack_semantic_mask_indices_to_28_channels(
            eighty_one_frames
        )

        self.assertEqual((28, 1, 1, 1), latent_one.shape)
        self.assertEqual((28, 2, 1, 1), latent_five.shape)
        self.assertEqual((28, 21, 1, 1), latent_eighty_one.shape)
        self.assertEqual(1, latent_five.value(0, 0))
        self.assertEqual(1, latent_five.value(7, 0))
        self.assertEqual(1, latent_five.value(14, 0))
        self.assertEqual(1, latent_five.value(21, 0))
        self.assertEqual(1, latent_five.value(1, 1))
        self.assertEqual(1, latent_five.value(9, 1))
        self.assertEqual(1, latent_five.value(17, 1))
        self.assertEqual(1, latent_five.value(27, 1))

    def test_temporal_mask_packing_rejects_four_frame_strict_input(self) -> None:
        four_frames = semantic_mask_indices(frames_from_colors([(255, 0, 0)] * 4))

        with self.assertRaisesRegex(ValueError, "4n\\+1"):
            pack_semantic_mask_indices_to_28_channels(four_frames)

    def test_condition_builder_accepts_animation_and_replacement_modes(self) -> None:
        ref_mask = frames_from_colors([(255, 255, 255)])
        driving_mask = frames_from_colors([(255, 0, 0)] * 5)

        animation = build_scail2_condition(
            mode="animation",
            ref_image="ref",
            ref_mask_frames=ref_mask,
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=driving_mask,
            width=1,
            height=1,
        )
        replacement = build_scail2_condition(
            mode="replacement",
            ref_image="ref",
            ref_mask_frames=ref_mask,
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=driving_mask,
            width=1,
            height=1,
        )

        self.assertEqual(TYPE_SCAIL2_CONDITION, animation.type_name)
        self.assertFalse(animation.replace_flag)
        self.assertTrue(replacement.replace_flag)
        self.assertEqual(5, animation.num_frames)

    def test_condition_rejects_mismatched_pose_and_mask_frame_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame counts must match"):
            build_scail2_condition(
                mode="pose_driven",
                ref_image="ref",
                ref_mask_frames=frames_from_colors([(255, 255, 255)]),
                pose_video="pose",
                pose_frame_count=4,
                driving_mask_frames=frames_from_colors([(255, 0, 0)] * 5),
                width=1,
                height=1,
            )

    def test_condition_rejects_unpaired_additional_references(self) -> None:
        common_kwargs = {
            "mode": "animation",
            "ref_image": "ref",
            "ref_mask_frames": frames_from_colors([(255, 255, 255)]),
            "pose_video": "pose",
            "pose_frame_count": 5,
            "driving_mask_frames": frames_from_colors([(255, 0, 0)] * 5),
            "width": 1,
            "height": 1,
        }

        with self.assertRaisesRegex(ValueError, "additional_ref_masks"):
            build_scail2_condition(
                **common_kwargs,
                additional_ref_images=["extra"],
            )
        with self.assertRaisesRegex(ValueError, "additional_ref_images"):
            build_scail2_condition(
                **common_kwargs,
                additional_ref_masks=[frames_from_colors([(255, 0, 0)])],
            )
        with self.assertRaisesRegex(ValueError, "same length"):
            build_scail2_condition(
                **common_kwargs,
                additional_ref_images=["extra_a", "extra_b"],
                additional_ref_masks=[frames_from_colors([(255, 0, 0)])],
            )

    def test_condition_rejects_invalid_segment_settings(self) -> None:
        common_kwargs = {
            "mode": "animation",
            "ref_image": "ref",
            "ref_mask_frames": frames_from_colors([(255, 255, 255)]),
            "pose_video": "pose",
            "pose_frame_count": 5,
            "driving_mask_frames": frames_from_colors([(255, 0, 0)] * 5),
            "width": 1,
            "height": 1,
        }

        with self.assertRaisesRegex(ValueError, "segment_overlap"):
            build_scail2_condition(**common_kwargs, segment_len=5, segment_overlap=5)
        with self.assertRaisesRegex(ValueError, "segment_len"):
            build_scail2_condition(**common_kwargs, segment_len=0, segment_overlap=1)

    def test_core_modules_do_not_import_heavy_runtime_modules(self) -> None:
        masks = importlib.import_module("scail2.masks")
        condition = importlib.import_module("scail2.condition")

        self.assertNotIn("torch", masks.__dict__)
        self.assertNotIn("torch", condition.__dict__)
        self.assertFalse(any(name.startswith("WanVideoWrapper") for name in sys.modules))
        self.assertFalse(any(name.startswith("SCAIL2.wan") for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
