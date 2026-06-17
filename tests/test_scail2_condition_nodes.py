from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from scail2.condition import TYPE_SCAIL2_CONDITION


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "ComfyUI_SCAIL_Pose2_ConditionNodeTestPackage"

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


def solid_frame(rgb, *, height=1, width=1):
    return [[rgb for _col in range(width)] for _row in range(height)]


def frames_from_colors(colors, *, height=1, width=1):
    return [solid_frame(color, height=height, width=width) for color in colors]


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


def condition_node():
    package = import_root_package()
    return package.NODE_CLASS_MAPPINGS["SCAILPose2SCAIL2Condition"]()


class Scail2ConditionNodeTests(unittest.TestCase):
    def test_condition_builder_node_is_registered(self) -> None:
        package = import_root_package()

        self.assertIn("SCAILPose2SCAIL2Condition", package.NODE_CLASS_MAPPINGS)
        self.assertIn("SCAILPose2SCAIL2Condition", package.NODE_DISPLAY_NAME_MAPPINGS)

        node_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2SCAIL2Condition"]
        self.assertEqual(("SCAIL2_CONDITION",), node_cls.RETURN_TYPES)
        self.assertEqual(("condition",), node_cls.RETURN_NAMES)

    def test_condition_node_builds_all_modes_and_preserves_mask_indices(self) -> None:
        node = condition_node()
        driving_mask = frames_from_colors([RED, GREEN, RED, GREEN, RED])

        for mode in ("animation", "replacement", "pose_driven"):
            condition, = node.build(
                ref_image="ref",
                ref_mask=frames_from_colors([WHITE]),
                pose_video="pose",
                driving_mask=driving_mask,
                mode=mode,
                width=1,
                height=1,
                num_frames=5,
                segment_len=81,
                segment_overlap=5,
                previous_frame_count=2,
                video_frame_offset=4,
            )

            self.assertEqual(TYPE_SCAIL2_CONDITION, condition.type_name)
            self.assertEqual(mode, condition.mode)
            self.assertEqual(mode == "replacement", condition.replace_flag)
            self.assertEqual(5, condition.num_frames)
            self.assertEqual(1, condition.driving_mask_indices[0][0][0])
            self.assertEqual(2, condition.driving_mask_indices[1][0][0])
            self.assertEqual("comfy_node:SCAILPose2SCAIL2Condition", condition.source_kind)
            self.assertEqual(2, condition.previous_frame_count)
            self.assertEqual(4, condition.video_frame_offset)
            with self.assertRaises(FrozenInstanceError):
                condition.mode = "animation"

    def test_condition_node_accepts_paired_additional_references(self) -> None:
        node = condition_node()

        condition, = node.build(
            ref_image="ref",
            ref_mask=frames_from_colors([WHITE]),
            pose_video="pose",
            driving_mask=frames_from_colors([RED] * 5),
            mode="animation",
            width=1,
            height=1,
            num_frames=5,
            segment_len=81,
            segment_overlap=5,
            additional_ref_images=["extra_ref"],
            additional_ref_masks=frames_from_colors([GREEN]),
        )

        self.assertEqual(1, len(condition.additional_references))
        self.assertEqual("extra_ref", condition.additional_references[0].image)
        self.assertEqual(2, condition.additional_references[0].mask_indices[0][0][0])

    def test_condition_node_rejects_invalid_user_inputs(self) -> None:
        node = condition_node()
        common = {
            "ref_image": "ref",
            "ref_mask": frames_from_colors([WHITE]),
            "pose_video": "pose",
            "driving_mask": frames_from_colors([RED] * 5),
            "mode": "animation",
            "width": 1,
            "height": 1,
            "num_frames": 5,
            "segment_len": 81,
            "segment_overlap": 5,
        }

        with self.assertRaisesRegex(ValueError, "frame counts must match"):
            node.build(**{**common, "num_frames": 4})
        with self.assertRaisesRegex(ValueError, "additional_ref_masks"):
            node.build(**{**common, "additional_ref_images": ["extra"]})
        with self.assertRaisesRegex(ValueError, "segment_overlap"):
            node.build(**{**common, "segment_len": 5, "segment_overlap": 5})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            node.build(**{**common, "driving_mask": frames_from_colors([(128, 0, 0)] * 5)})


if __name__ == "__main__":
    unittest.main()
