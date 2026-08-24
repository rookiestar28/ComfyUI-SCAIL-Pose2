from __future__ import annotations

import importlib
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class WorkflowContractSourceParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parity = importlib.import_module(
            "scripts.check_workflow_contract_source_parity"
        )

    def test_bounded_class_contract_check_accepts_matching_source_text(self) -> None:
        node = {
            "class": "ExampleNode",
            "required_inputs": [{"name": "image", "type": "IMAGE"}],
            "optional_inputs": [{"name": "mask", "type": "MASK"}],
            "outputs": [{"index": 0, "name": "samples", "type": "LATENT"}],
            "output_count": 1,
            "evidence": {"symbol": "ExampleNode"},
        }
        source = '''
class ExampleNode:
    def INPUT_TYPES(self):
        return {"required": {"image": ("IMAGE",)}, "optional": {"mask": ("MASK",)}}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)

class NextNode:
    pass
'''

        self.assertEqual((), self.parity.validate_node_source(node, source))

    def test_bounded_class_contract_check_reports_missing_socket(self) -> None:
        node = {
            "class": "ExampleNode",
            "required_inputs": [{"name": "image", "type": "IMAGE"}],
            "optional_inputs": [],
            "outputs": [{"index": 0, "name": "samples", "type": "LATENT"}],
            "output_count": 1,
            "evidence": {"symbol": "ExampleNode"},
        }
        source = 'class ExampleNode:\n    RETURN_TYPES = ("LATENT",)\n    RETURN_NAMES = ("samples",)\n'

        errors = self.parity.validate_node_source(node, source)

        self.assertTrue(any("image" in error for error in errors))

    def test_parity_module_has_no_reference_import_or_code_execution_path(self) -> None:
        source = inspect.getsource(self.parity)

        self.assertNotIn("importlib", source)
        self.assertNotIn("exec(", source)
        self.assertNotIn("eval(", source)
        self.assertNotIn("runpy", source)

    def test_reference_repository_rejects_symlink_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "ComfyUI"
            candidate.mkdir()
            original = Path.is_symlink

            def is_symlink(path: Path) -> bool:
                return path == candidate or original(path)

            with mock.patch.object(Path, "is_symlink", is_symlink):
                with self.assertRaisesRegex(ValueError, "Symlinked reference"):
                    self.parity._reference_repository(root, "ComfyUI")

    def test_nested_source_path_cannot_escape_top_level_reference_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()

            with self.assertRaisesRegex(ValueError, "escapes"):
                self.parity._contained_path(root, "ComfyUI/../../outside.py")


if __name__ == "__main__":
    unittest.main()
