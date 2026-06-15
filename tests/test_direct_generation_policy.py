from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from scail2 import generation_policy


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "ComfyUI_SCAIL_Pose2_DirectGenerationTestPackage"


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


class DirectGenerationPolicyTests(unittest.TestCase):
    def test_decision_is_defer_to_wanvideo_wrapper(self) -> None:
        self.assertIn(
            generation_policy.DIRECT_GENERATION_DECISION,
            generation_policy.ALLOWED_DIRECT_GENERATION_DECISIONS,
        )
        self.assertEqual(
            generation_policy.DECISION_DEFER_TO_WANVIDEOWRAPPER,
            generation_policy.DIRECT_GENERATION_DECISION,
        )
        self.assertFalse(generation_policy.direct_generation_enabled())

    def test_policy_lists_required_scail2_generation_requirements(self) -> None:
        self.assertTrue(
            {
                "reference_mask",
                "driving_mask",
                "mask_latents_28_channel",
                "replacement_flag",
                "additional_reference_masks",
                "long_video_clean_history",
                "lora_support",
                "segment_overlap",
            }.issubset(generation_policy.SCAIL2_DIRECT_GENERATION_REQUIREMENTS)
        )

    def test_public_node_list_excludes_direct_generation_claims(self) -> None:
        package = import_root_package()

        for node_key in generation_policy.DIRECT_GENERATION_NODE_EXCLUSIONS:
            self.assertNotIn(node_key, package.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
