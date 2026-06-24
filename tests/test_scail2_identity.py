from __future__ import annotations

import unittest

from scail2.identity import (
    SCAIL2_IDENTITY_COLORS,
    build_identity_slots,
    identity_count_from_semantic_mask,
    semantic_identity_indices,
    semantic_identity_rgb_mask,
)


BLACK = (0.0, 0.0, 0.0)
WHITE = (1.0, 1.0, 1.0)


class Scail2IdentityTests(unittest.TestCase):
    def test_builds_deterministic_slots_from_selected_object_order(self) -> None:
        diagnostics = build_identity_slots(
            object_order=(1, 0),
            object_stats=((0, 0.2, 0.25), (3, 0.8, 0.5)),
            frame_count=9,
            object_count=2,
        )

        self.assertEqual(2, diagnostics.identity_count)
        self.assertEqual((1, 0), diagnostics.selected_source_indices)
        self.assertEqual("blue", diagnostics.slots[0].color_name)
        self.assertEqual(SCAIL2_IDENTITY_COLORS[0], diagnostics.slots[0].color_rgb)
        self.assertEqual(1, diagnostics.slots[0].source_object_index)
        self.assertEqual(3, diagnostics.slots[0].first_frame)
        self.assertEqual("red", diagnostics.slots[1].color_name)
        self.assertEqual((), diagnostics.warnings)

    def test_warns_when_one_identity_is_selected_from_multi_object_track(self) -> None:
        diagnostics = build_identity_slots(
            object_order=(1,),
            object_stats=((0, 0.2, 0.25), (3, 0.8, 0.5)),
            frame_count=9,
            object_count=2,
        )

        self.assertEqual(1, diagnostics.identity_count)
        self.assertEqual((1,), diagnostics.selected_source_indices)
        self.assertIn("single_identity_selected_from_multi_object_track", diagnostics.warnings)

    def test_counts_palette_colors_in_semantic_rgb_masks(self) -> None:
        frames = (
            (
                (BLACK, WHITE, SCAIL2_IDENTITY_COLORS[0]),
                (BLACK, SCAIL2_IDENTITY_COLORS[1], WHITE),
            ),
        )
        raw_frames = (
            (
                ((0, 0, 0), (255, 255, 255), (0, 0, 255)),
                ((0, 0, 0), (255, 0, 0), (255, 255, 255)),
            ),
        )

        self.assertEqual(2, identity_count_from_semantic_mask(frames))
        self.assertEqual(2, identity_count_from_semantic_mask(raw_frames))
        self.assertEqual((0, 1), semantic_identity_indices(frames))
        self.assertEqual((0, 1), semantic_identity_indices(raw_frames))

    def test_extracts_one_identity_semantic_mask(self) -> None:
        frames = (
            (
                (BLACK, SCAIL2_IDENTITY_COLORS[0], SCAIL2_IDENTITY_COLORS[1]),
                (WHITE, SCAIL2_IDENTITY_COLORS[1], BLACK),
            ),
        )

        extracted = semantic_identity_rgb_mask(frames, identity_index=1)

        self.assertEqual(BLACK, extracted[0][0][1])
        self.assertEqual(SCAIL2_IDENTITY_COLORS[1], extracted[0][0][2])
        self.assertEqual(SCAIL2_IDENTITY_COLORS[1], extracted[0][1][1])
        self.assertEqual(BLACK, extracted[0][1][0])


if __name__ == "__main__":
    unittest.main()
