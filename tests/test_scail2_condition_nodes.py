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


class FakeTensorImage:
    shape = (5, 8, 8, 3)

    def detach(self):
        return self


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


def condition_node_module():
    import_root_package()
    return sys.modules[f"{PACKAGE_NAME}.nodes_scail2_condition"]


class Scail2ConditionNodeTests(unittest.TestCase):
    def test_normalize_image_frames_preserves_tensor_like_batch(self) -> None:
        tensor = FakeTensorImage()
        module = condition_node_module()

        self.assertIs(tensor, module._normalize_image_frames(tensor, name="pose_video_mask"))


    def test_condition_builder_node_is_registered(self) -> None:
        package = import_root_package()

        self.assertIn("SCAILPose2SCAIL2Condition", package.NODE_CLASS_MAPPINGS)
        self.assertIn("SCAILPose2SCAIL2Condition", package.NODE_DISPLAY_NAME_MAPPINGS)

        node_cls = package.NODE_CLASS_MAPPINGS["SCAILPose2SCAIL2Condition"]
        self.assertEqual(("SCAIL2_CONDITION",), node_cls.RETURN_TYPES)
        self.assertEqual(("condition",), node_cls.RETURN_NAMES)
        input_types = node_cls.INPUT_TYPES()
        self.assertEqual(
            (
                "pose_video",
                "pose_video_mask",
                "ref_image",
                "ref_mask",
                "mode",
                "width",
                "height",
                "num_frames",
            ),
            tuple(input_types["required"]),
        )
        self.assertEqual(
            ["animation", "replacement"],
            input_types["required"]["mode"][0],
        )
        self.assertEqual(
            ("additional_ref_image", "additional_ref_mask"),
            tuple(input_types["optional"]),
        )

    def test_condition_node_builds_all_modes_and_preserves_mask_indices(self) -> None:
        node = condition_node()
        driving_mask = frames_from_colors([RED, GREEN, RED, GREEN, RED])

        for mode in ("animation", "replacement"):
            condition, = node.build(
                pose_video="pose",
                pose_video_mask=driving_mask,
                ref_image="ref",
                ref_mask=frames_from_colors([WHITE]),
                mode=mode,
                width=1,
                height=1,
                num_frames=5,
            )

            self.assertEqual(TYPE_SCAIL2_CONDITION, condition.type_name)
            self.assertEqual(mode, condition.mode)
            self.assertEqual(mode == "replacement", condition.replace_flag)
            self.assertEqual(5, condition.num_frames)
            self.assertEqual(1, condition.driving_mask_indices[0][0][0])
            self.assertEqual(2, condition.driving_mask_indices[1][0][0])
            self.assertEqual("comfy_node:SCAILPose2SCAIL2Condition", condition.source_kind)
            with self.assertRaises(FrozenInstanceError):
                condition.mode = "animation"

    def test_condition_node_rejects_pose_driven_as_independent_mode(self) -> None:
        node = condition_node()

        with self.assertRaisesRegex(ValueError, "mode must be one of animation, replacement"):
            node.build(
                pose_video="pose",
                pose_video_mask=frames_from_colors([RED] * 5),
                ref_image="ref",
                ref_mask=frames_from_colors([WHITE]),
                mode="pose_driven",
                width=1,
                height=1,
                num_frames=5,
            )

    def test_condition_node_accepts_paired_additional_references(self) -> None:
        node = condition_node()

        condition, = node.build(
            pose_video="pose",
            pose_video_mask=frames_from_colors([RED] * 5),
            ref_image="ref",
            ref_mask=frames_from_colors([WHITE]),
            mode="animation",
            width=1,
            height=1,
            num_frames=5,
            additional_ref_image=["extra_ref"],
            additional_ref_mask=frames_from_colors([GREEN]),
        )

        self.assertEqual(1, len(condition.additional_references))
        self.assertEqual("extra_ref", condition.additional_references[0].image)
        self.assertEqual(2, condition.additional_references[0].mask_indices[0][0][0])

    def test_condition_node_rejects_invalid_user_inputs(self) -> None:
        node = condition_node()
        common = {
            "pose_video": "pose",
            "pose_video_mask": frames_from_colors([RED] * 5),
            "ref_image": "ref",
            "ref_mask": frames_from_colors([WHITE]),
            "mode": "animation",
            "width": 1,
            "height": 1,
            "num_frames": 5,
        }

        with self.assertRaisesRegex(ValueError, "frame counts must match"):
            node.build(**{**common, "num_frames": 4})
        with self.assertRaisesRegex(ValueError, "additional_ref_mask"):
            node.build(**{**common, "additional_ref_image": ["extra"]})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            node.build(**{**common, "pose_video_mask": frames_from_colors([(128, 0, 0)] * 5)})


if __name__ == "__main__":
    unittest.main()
