from __future__ import annotations

import importlib
import sys
import unittest

from scail2.condition import TYPE_SCAIL2_CONDITION, build_scail2_condition
from scail2.masks import (
    BACKGROUND_INDEX,
    SEMANTIC_MASK_COLORS,
    classify_rgb_semantic_color,
    latent_spatial_size_for_pixels,
    pack_semantic_mask_indices_to_28_channels,
    pose_control_latent_spatial_size,
    semantic_mask_indices,
    semantic_mask_indices_tensor,
)


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)


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

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_tensor_semantic_mask_classification_matches_palette_contract(self) -> None:
        import torch

        frames = torch.tensor(
            [
                [
                    [
                        (1.0, 1.0, 1.0),
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 0.0),
                    ]
                ]
            ],
            dtype=torch.float32,
        )

        indices = semantic_mask_indices_tensor(frames)

        self.assertEqual((0, 1, 2, BACKGROUND_INDEX), indices[0][0])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_condition_tensor_masks_do_not_freeze_full_resolution_indices(self) -> None:
        import torch
        from scail2 import masks as scail_masks

        ref_mask = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
        ref_mask[..., :] = torch.tensor((1.0, 1.0, 1.0))
        driving_mask = torch.zeros((5, 2, 2, 3), dtype=torch.float32)
        driving_mask[0, ..., :] = torch.tensor((1.0, 0.0, 0.0))
        driving_mask[1, ..., :] = torch.tensor((0.0, 1.0, 0.0))
        driving_mask[2, ..., :] = torch.tensor((0.0, 0.0, 1.0))
        driving_mask[3, ..., :] = torch.tensor((1.0, 0.0, 0.0))
        driving_mask[4, ..., :] = torch.tensor((0.0, 1.0, 0.0))
        original_freeze = scail_masks._freeze_index_frames

        def fail_on_freeze(_value):
            raise AssertionError(
                "Condition tensor path must not freeze full-resolution indices"
            )

        scail_masks._freeze_index_frames = fail_on_freeze
        try:
            condition = build_scail2_condition(
                mode="animation",
                ref_image="ref",
                ref_mask_frames=ref_mask,
                pose_video="pose",
                pose_frame_count=5,
                driving_mask_frames=driving_mask,
                width=2,
                height=2,
            )
        finally:
            scail_masks._freeze_index_frames = original_freeze

        self.assertEqual((1, 2, 2), tuple(condition.ref_mask_indices.shape))
        self.assertEqual((5, 2, 2), tuple(condition.driving_mask_indices.shape))
        self.assertEqual(torch.int8, condition.driving_mask_indices.dtype)
        self.assertEqual(0, int(condition.ref_mask_indices[0, 0, 0].item()))
        self.assertEqual(1, int(condition.driving_mask_indices[0, 0, 0].item()))
        self.assertEqual(2, int(condition.driving_mask_indices[1, 0, 0].item()))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_tensor_runtime_mask_packing_matches_tuple_contract(self) -> None:
        import torch
        from scail2 import masks as scail_masks

        colors = [
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 255),
        ]
        tuple_indices = semantic_mask_indices(
            frames_from_colors(colors, height=9, width=9)
        )
        tensor_frames = torch.tensor(
            frames_from_colors(colors, height=9, width=9),
            dtype=torch.float32,
        )
        tensor_indices = scail_masks.semantic_mask_indices_tensor_raw(tensor_frames)

        tuple_runtime = scail_masks.pack_semantic_mask_indices_to_runtime_28_channels(
            tuple_indices
        )
        tensor_runtime = scail_masks.pack_semantic_mask_indices_to_runtime_28_channels(
            tensor_indices
        )

        self.assertTrue(torch.is_tensor(tensor_runtime.data))
        self.assertEqual(tuple_runtime.shape, tensor_runtime.shape)
        self.assertTrue(
            torch.allclose(
                scail_masks.runtime_mask_to_torch(tuple_runtime),
                scail_masks.runtime_mask_to_torch(tensor_runtime),
            )
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_tensor_runtime_mask_packing_downsamples_per_color_chunks(self) -> None:
        import torch
        import torch.nn.functional as F
        from scail2 import masks as scail_masks

        tensor_indices = scail_masks.semantic_mask_indices_tensor_raw(
            torch.tensor(
                frames_from_colors([(255, 0, 0)] * 5, height=8, width=8),
                dtype=torch.float32,
            )
        )
        calls = []
        original_pool = F.adaptive_avg_pool2d

        def recording_pool(input_tensor, output_size):
            calls.append(tuple(input_tensor.shape))
            return original_pool(input_tensor, output_size)

        F.adaptive_avg_pool2d = recording_pool
        try:
            scail_masks.pack_semantic_mask_indices_to_runtime_28_channels(
                tensor_indices
            )
        finally:
            F.adaptive_avg_pool2d = original_pool

        self.assertTrue(calls)
        self.assertTrue(all(shape[1] == 1 for shape in calls))
        self.assertFalse(any(shape[1] == 7 for shape in calls))

    def test_strict_palette_rejects_ambiguous_non_semantic_colors(self) -> None:
        self.assertEqual(1, classify_rgb_semantic_color((225, 0, 0)))
        self.assertEqual(BACKGROUND_INDEX, classify_rgb_semantic_color((30, 0, 0)))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            classify_rgb_semantic_color((224, 0, 0))
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

    def test_runtime_mask_packing_exposes_comfy_and_scail2_shapes(self) -> None:
        from scail2 import masks as scail_masks

        five_frames = semantic_mask_indices(
            frames_from_colors(
                [
                    (255, 255, 255),
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (0, 255, 255),
                ],
                height=8,
                width=8,
            )
        )

        runtime = scail_masks.pack_semantic_mask_indices_to_runtime_28_channels(
            five_frames
        )

        self.assertEqual((1, 2, 28, 1, 1), runtime.shape)
        self.assertEqual((1, 2, 28, 1, 1), runtime.comfy_shape)
        self.assertEqual((28, 2, 1, 1), runtime.scail2_shape)
        self.assertEqual(1.0, runtime.value(latent_frame=0, channel=0))
        self.assertEqual(1.0, runtime.value(latent_frame=0, channel=7))
        self.assertEqual(1.0, runtime.value(latent_frame=0, channel=14))
        self.assertEqual(1.0, runtime.value(latent_frame=0, channel=21))
        self.assertEqual(1.0, runtime.value(latent_frame=1, channel=1))
        self.assertEqual(1.0, runtime.value(latent_frame=1, channel=9))
        self.assertEqual(1.0, runtime.value(latent_frame=1, channel=17))
        self.assertEqual(1.0, runtime.value(latent_frame=1, channel=27))

    def test_runtime_mask_packing_rejects_four_frames_and_accepts_eighty_one(
        self,
    ) -> None:
        from scail2 import masks as scail_masks

        four_frames = semantic_mask_indices(frames_from_colors([(255, 0, 0)] * 4))
        eighty_one_frames = semantic_mask_indices(
            frames_from_colors([(0, 0, 0)] * 80 + [(255, 0, 255)])
        )

        with self.assertRaisesRegex(ValueError, "4n\\+1"):
            scail_masks.pack_semantic_mask_indices_to_runtime_28_channels(four_frames)

        runtime = scail_masks.pack_semantic_mask_indices_to_runtime_28_channels(
            eighty_one_frames
        )
        self.assertEqual((1, 21, 28, 1, 1), runtime.shape)
        self.assertEqual(1.0, runtime.value(latent_frame=20, channel=26))

    def test_runtime_mask_spatial_downsample_handles_even_and_odd_dimensions(
        self,
    ) -> None:
        from scail2 import masks as scail_masks

        even_frame = []
        for row_index in range(16):
            row = []
            for col_index in range(16):
                if row_index < 8 and col_index < 8:
                    row.append((255, 0, 0))
                else:
                    row.append(BLACK)
            even_frame.append(row)
        even_runtime = scail_masks.pack_semantic_mask_indices_to_runtime_28_channels(
            semantic_mask_indices([even_frame])
        )

        self.assertEqual((1, 1, 28, 2, 2), even_runtime.shape)
        self.assertEqual(1.0, even_runtime.value(latent_frame=0, channel=1, row=0, col=0))
        self.assertEqual(0.0, even_runtime.value(latent_frame=0, channel=1, row=0, col=1))
        self.assertEqual(0.0, even_runtime.value(latent_frame=0, channel=1, row=1, col=0))
        self.assertEqual(0.0, even_runtime.value(latent_frame=0, channel=1, row=1, col=1))

        odd_runtime = scail_masks.pack_semantic_mask_indices_to_runtime_28_channels(
            semantic_mask_indices(frames_from_colors([(0, 255, 0)], height=9, width=9))
        )

        self.assertEqual((1, 1, 28, 2, 2), odd_runtime.shape)
        for row in range(2):
            for col in range(2):
                self.assertEqual(
                    1.0,
                    odd_runtime.value(latent_frame=0, channel=2, row=row, col=col),
                )

    def test_runtime_mask_packing_accepts_explicit_target_latent_shape(
        self,
    ) -> None:
        from scail2 import masks as scail_masks

        runtime = scail_masks.pack_semantic_mask_indices_to_runtime_28_channels(
            semantic_mask_indices(frames_from_colors([(255, 0, 0)] * 5, height=16, width=16)),
            target_latent_height=1,
            target_latent_width=1,
            layout_role="driving",
        )

        self.assertEqual((1, 2, 28, 1, 1), runtime.shape)
        self.assertEqual(16, runtime.source_height)
        self.assertEqual(16, runtime.source_width)
        self.assertEqual("driving", runtime.layout_role)

    def test_latent_shape_contract_separates_full_and_pose_control_layouts(
        self,
    ) -> None:
        self.assertEqual(
            (96, 140),
            latent_spatial_size_for_pixels(height=768, width=1120),
        )
        self.assertEqual(
            (48, 70),
            pose_control_latent_spatial_size(height=768, width=1120),
        )
        self.assertEqual(
            (2, 2),
            latent_spatial_size_for_pixels(height=16, width=16),
        )
        self.assertEqual(
            (1, 1),
            pose_control_latent_spatial_size(height=16, width=16),
        )

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

    def test_condition_mode_applies_colored_mask_replacement_polarity(self) -> None:
        ref_mask = [[[BLUE, WHITE]]]
        driving_mask = [[[BLUE, BLACK]] for _frame in range(5)]

        animation = build_scail2_condition(
            mode="animation",
            ref_image="ref",
            ref_mask_frames=ref_mask,
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=driving_mask,
            width=2,
            height=1,
        )
        replacement = build_scail2_condition(
            mode="replacement",
            ref_image="ref",
            ref_mask_frames=ref_mask,
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=driving_mask,
            width=2,
            height=1,
            additional_ref_images=["extra"],
            additional_ref_masks=[ref_mask],
        )

        self.assertEqual(3, animation.driving_mask_indices[0][0][0])
        self.assertEqual(BACKGROUND_INDEX, animation.driving_mask_indices[0][0][1])
        self.assertEqual(3, animation.ref_mask_indices[0][0][0])
        self.assertEqual(0, animation.ref_mask_indices[0][0][1])

        self.assertEqual(3, replacement.driving_mask_indices[0][0][0])
        self.assertEqual(0, replacement.driving_mask_indices[0][0][1])
        self.assertEqual(3, replacement.ref_mask_indices[0][0][0])
        self.assertEqual(BACKGROUND_INDEX, replacement.ref_mask_indices[0][0][1])
        extra = replacement.additional_references[0].mask_indices
        self.assertEqual(3, extra[0][0][0])
        self.assertEqual(BACKGROUND_INDEX, extra[0][0][1])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_condition_tensor_replacement_polarity_stays_tensor_native(self) -> None:
        import torch

        ref_mask = torch.tensor([[[[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]])
        driving_mask = torch.tensor(
            [[[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]] * 5,
            dtype=torch.float32,
        )

        condition = build_scail2_condition(
            mode="replacement",
            ref_image="ref",
            ref_mask_frames=ref_mask,
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=driving_mask,
            width=2,
            height=1,
        )

        self.assertTrue(torch.is_tensor(condition.ref_mask_indices))
        self.assertTrue(torch.is_tensor(condition.driving_mask_indices))
        self.assertEqual(torch.int8, condition.ref_mask_indices.dtype)
        self.assertEqual(3, int(condition.driving_mask_indices[0, 0, 0].item()))
        self.assertEqual(0, int(condition.driving_mask_indices[0, 0, 1].item()))
        self.assertEqual(3, int(condition.ref_mask_indices[0, 0, 0].item()))
        self.assertEqual(
            BACKGROUND_INDEX,
            int(condition.ref_mask_indices[0, 0, 1].item()),
        )

    def test_condition_replacement_polarity_does_not_mutate_index_masks(self) -> None:
        ref_mask_indices = (((3, 0),),)
        driving_mask_indices = tuple(((3, BACKGROUND_INDEX),) for _frame in range(5))

        condition = build_scail2_condition(
            mode="replacement",
            ref_image="ref",
            ref_mask_frames=ref_mask_indices,
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=driving_mask_indices,
            width=2,
            height=1,
        )

        self.assertEqual(ref_mask_indices, condition.ref_mask_indices)
        self.assertEqual(driving_mask_indices, condition.driving_mask_indices)

    def test_condition_rejects_mismatched_pose_and_mask_frame_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame counts must match"):
            build_scail2_condition(
                mode="animation",
                ref_image="ref",
                ref_mask_frames=frames_from_colors([(255, 255, 255)]),
                pose_video="pose",
                pose_frame_count=4,
                driving_mask_frames=frames_from_colors([(255, 0, 0)] * 5),
                width=1,
                height=1,
            )

    def test_condition_rejects_pose_driven_as_independent_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode must be one of animation, replacement"):
            build_scail2_condition(
                mode="pose_driven",
                ref_image="ref",
                ref_mask_frames=frames_from_colors([(255, 255, 255)]),
                pose_video="pose",
                pose_frame_count=5,
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

    def test_condition_rejects_empty_source_kind(self) -> None:
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

        with self.assertRaisesRegex(ValueError, "source_kind"):
            build_scail2_condition(**common_kwargs, source_kind=" ")

    def test_core_modules_do_not_import_heavy_runtime_modules(self) -> None:
        masks = importlib.import_module("scail2.masks")
        condition = importlib.import_module("scail2.condition")

        self.assertNotIn("torch", masks.__dict__)
        self.assertNotIn("torch", condition.__dict__)
        self.assertFalse(any(name.startswith("WanVideoWrapper") for name in sys.modules))
        self.assertFalse(any(name.startswith("SCAIL2.wan") for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
