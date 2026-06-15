from __future__ import annotations

import builtins
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scail2.preprocessing import build_user_mask_condition
from scail2.sam3_preprocessing import (
    DEFAULT_SAM3_TRACK_RGB_PALETTE,
    SAM3DependencyError,
    build_condition_from_sam3_tracks,
    require_sam3_predictors,
    sam3_tracks_to_semantic_mask_frames,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "ComfyUI_SCAIL_Pose2_SAM3TestPackage"

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


def solid_frame(rgb, *, height=1, width=1):
    return [[rgb for _col in range(width)] for _row in range(height)]


def frames_from_colors(colors, *, height=1, width=1):
    return [solid_frame(color, height=height, width=width) for color in colors]


def track(frames):
    return frames


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


def block_ultralytics_imports():
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("ultralytics"):
            raise ImportError("blocked optional SAM3 dependency")
        return original_import(name, *args, **kwargs)

    return patch("builtins.__import__", side_effect=fake_import)


class UserSam3MaskPreprocessingTests(unittest.TestCase):
    def test_user_masks_build_animation_and_replacement_conditions(self) -> None:
        ref_mask = frames_from_colors([WHITE])
        driving_mask = frames_from_colors([RED] * 5)

        animation = build_user_mask_condition(
            mode="animation",
            ref_image="ref",
            ref_mask_frames=ref_mask,
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=driving_mask,
            width=1,
            height=1,
        )
        replacement = build_user_mask_condition(
            mode="replacement",
            ref_image="ref",
            ref_mask_frames=ref_mask,
            pose_video="pose",
            pose_frame_count=5,
            driving_mask_frames=driving_mask,
            width=1,
            height=1,
        )

        self.assertEqual("SCAIL2_CONDITION", animation.type_name)
        self.assertFalse(animation.replace_flag)
        self.assertTrue(replacement.replace_flag)

    def test_mocked_sam3_tracks_produce_deterministic_masks_and_metadata(self) -> None:
        track_a = track(
            [
                [[True, False], [False, False]],
                [[False, True], [False, False]],
            ]
        )
        track_b = track(
            [
                [[False, False], [True, False]],
                [[False, False], [False, True]],
            ]
        )

        bundle = sam3_tracks_to_semantic_mask_frames(
            [track_a, track_b],
            mode="animation",
            track_ids=["a", "b"],
            expected_track_count=2,
        )

        self.assertEqual("sam3_mock", bundle.source)
        self.assertEqual("animation", bundle.mode)
        self.assertEqual(DEFAULT_SAM3_TRACK_RGB_PALETTE[0], bundle.frames[0][0][0])
        self.assertEqual(DEFAULT_SAM3_TRACK_RGB_PALETTE[1], bundle.frames[0][1][0])
        self.assertEqual(BLACK, bundle.frames[0][0][1])
        self.assertEqual(("a", "b"), tuple(item.track_id for item in bundle.track_metadata))
        self.assertEqual((1, 1), bundle.track_metadata[0].pixels_per_frame)

    def test_mocked_sam3_tracks_build_replacement_condition_payload(self) -> None:
        ref_track = track([[[True]]])
        driving_track = track([[[True]], [[True]], [[True]], [[True]], [[True]]])

        payload = build_condition_from_sam3_tracks(
            mode="replacement",
            ref_image="ref",
            ref_track_masks=[ref_track],
            pose_video="pose",
            driving_track_masks=[driving_track],
            width=1,
            height=1,
            expected_ref_track_count=1,
            expected_driving_track_count=1,
            ref_track_ids=["ref_subject"],
            driving_track_ids=["driver_subject"],
        )

        self.assertEqual("sam3_mock", payload.source)
        self.assertEqual("replacement", payload.mode)
        self.assertTrue(payload.condition.replace_flag)
        self.assertEqual("driver_subject", payload.driving_mask_bundle.track_metadata[0].track_id)
        self.assertEqual(5, payload.condition.num_frames)

    def test_track_count_mismatch_is_explicit(self) -> None:
        one_track = track([[[True]]])

        with self.assertRaisesRegex(ValueError, "track count mismatch"):
            sam3_tracks_to_semantic_mask_frames(
                [one_track],
                mode="animation",
                expected_track_count=2,
            )

    def test_missing_sam3_dependency_guard_is_actionable_and_lazy(self) -> None:
        with block_ultralytics_imports():
            with self.assertRaisesRegex(SAM3DependencyError, "Optional SAM3"):
                require_sam3_predictors()

    def test_root_import_does_not_load_sam3_and_node_errors_on_execution(self) -> None:
        package = import_root_package()

        self.assertIn("SCAIL2SAM3DependencyCheck", package.NODE_CLASS_MAPPINGS)
        self.assertFalse(any(name.startswith("ultralytics") for name in sys.modules))

        node = package.NODE_CLASS_MAPPINGS["SCAIL2SAM3DependencyCheck"]()
        with block_ultralytics_imports():
            with self.assertRaisesRegex(ImportError, "never auto-downloads"):
                node.check()


if __name__ == "__main__":
    unittest.main()
