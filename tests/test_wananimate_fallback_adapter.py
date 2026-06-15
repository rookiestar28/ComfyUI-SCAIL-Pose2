from __future__ import annotations

import sys
import unittest

from scail2.condition import build_scail2_condition
from scail2.wananimate_fallback import (
    LOSS_28CH_MASK_LATENT,
    LOSS_REPLACEMENT_ROPE,
    LOSS_RGB_TO_GRAYSCALE,
    convert_scail2_condition_to_wananimate,
)


WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


def solid_frame(rgb):
    return [[rgb]]


def frames_from_colors(colors):
    return [solid_frame(color) for color in colors]


def build_condition(mode="replacement"):
    return build_scail2_condition(
        mode=mode,
        ref_image="ref-image",
        ref_mask_frames=frames_from_colors([WHITE]),
        pose_video="pose-video",
        pose_frame_count=5,
        driving_mask_frames=frames_from_colors([RED, BLACK, GREEN, BLACK, RED]),
        width=1,
        height=1,
    )


class WanAnimateFallbackAdapterTests(unittest.TestCase):
    def test_default_conversion_refuses_semantic_degradation(self) -> None:
        condition = build_condition()

        with self.assertRaisesRegex(ValueError, "semantic degradation"):
            convert_scail2_condition_to_wananimate(condition)

    def test_explicit_degradation_returns_payload_and_loss_metadata(self) -> None:
        condition = build_condition()

        payload = convert_scail2_condition_to_wananimate(
            condition,
            allow_semantic_degradation=True,
            bg_images="background-images",
        )

        self.assertEqual("ref-image", payload.ref_images)
        self.assertEqual("pose-video", payload.pose_images)
        self.assertEqual("background-images", payload.bg_images)
        self.assertEqual(
            (
                ((1.0,),),
                ((0.0,),),
                ((1.0,),),
                ((0.0,),),
                ((1.0,),),
            ),
            payload.mask,
        )
        self.assertFalse(payload.metadata["is_full_scail2_parity"])
        self.assertEqual("WanVideoAnimateEmbeds", payload.metadata["target"])
        self.assertIn(LOSS_RGB_TO_GRAYSCALE, payload.metadata["semantic_losses"])
        self.assertIn(LOSS_28CH_MASK_LATENT, payload.metadata["semantic_losses"])
        self.assertIn(LOSS_REPLACEMENT_ROPE, payload.metadata["semantic_losses"])

    def test_original_rgb_mask_indices_remain_available(self) -> None:
        condition = build_condition()
        original_indices = condition.driving_mask_indices

        convert_scail2_condition_to_wananimate(
            condition,
            allow_semantic_degradation=True,
        )

        self.assertEqual(original_indices, condition.driving_mask_indices)
        self.assertEqual(1, condition.driving_mask_indices[0][0][0])
        self.assertEqual(2, condition.driving_mask_indices[2][0][0])

    def test_animation_mode_omits_replacement_loss(self) -> None:
        condition = build_condition(mode="animation")

        payload = convert_scail2_condition_to_wananimate(
            condition,
            allow_semantic_degradation=True,
        )

        self.assertNotIn(LOSS_REPLACEMENT_ROPE, payload.metadata["semantic_losses"])

    def test_no_wanvideo_wrapper_runtime_import_is_required(self) -> None:
        condition = build_condition()

        convert_scail2_condition_to_wananimate(
            condition,
            allow_semantic_degradation=True,
        )

        self.assertFalse(any(name.startswith("WanVideoWrapper") for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
