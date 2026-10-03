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
        source = ('class ExampleNode:\n'
                  '    def INPUT_TYPES(self):\n        return {"required": {}, "optional": {}}\n'
                  '    RETURN_TYPES = ("LATENT",)\n    RETURN_NAMES = ("samples",)\n')

        errors = self.parity.validate_node_source(node, source)

        self.assertIn("input-contract-mismatch:ExampleNode:required_inputs", errors)

    def test_parity_module_has_no_reference_import_or_code_execution_path(self) -> None:
        source = inspect.getsource(self.parity)

        self.assertNotIn("importlib", source)
        self.assertNotIn("exec(", source)
        self.assertNotIn("eval(", source)
        self.assertNotIn("runpy", source)

    def test_v3_positional_output_names_and_optional_union_are_parsed_without_execution(self) -> None:
        node = {"class": "ExampleNode", "required_inputs": [{"name": "image", "type": "IMAGE"}],
                "optional_inputs": [{"name": "track", "type": "SAM3_TRACK_DATA/MASK"}],
                "schema_order": ["image", "track"],
                "outputs": [{"index": 0, "name": "colored", "type": "IMAGE"}], "output_count": 1}
        source = '''
class ExampleNode(io.ComfyNode):
    def define_schema(cls):
        return io.Schema(inputs=[io.Image.Input("image"),
            io.MultiType.Input("track", [SAM3TrackData, io.Mask], optional=True)],
            outputs=[io.Image.Output("colored")])
raise RuntimeError("do not execute")
'''
        self.assertEqual((), self.parity.validate_node_source(node, source))
        node["schema_order"] = ["track", "image"]
        self.assertIn("input-order-mismatch:ExampleNode", self.parity.validate_node_source(node, source))

    def test_tracked_manifest_preserves_historical_and_adds_candidate_core(self) -> None:
        import json
        raw = json.loads((Path(__file__).resolve().parents[1] /
                          "workflow_contracts/wanvideo_host_contracts.v1.json").read_text(encoding="utf-8"))
        rows = raw["families"]
        self.assertEqual(4, len(rows))
        self.assertEqual(19, sum(len(row["nodes"]) for row in rows))
        core = [row for row in rows if row["family"] == "comfyui-core"]
        self.assertEqual(2, len(core))
        self.assertEqual(core[0]["nodes"], core[1]["nodes"])
        self.assertEqual({self.parity.SOURCE_PROFILES[p]["ComfyUI"][1]
                          for p in self.parity.SOURCE_PROFILES}, {row["revision"] for row in core})

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


class SourceProfileIntegrationTests(unittest.TestCase):
    """Use disposable Git objects, not executed host code or mocked git show."""

    def setUp(self) -> None:
        self.parity = importlib.import_module("scripts.check_workflow_contract_source_parity")
        self.assertIn("candidate-20261003", getattr(self.parity, "SOURCE_PROFILES", {}),
                      "explicit candidate source profile is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repositories = {}
        self.origins = {}
        self.pins = {}
        for name in ("ComfyUI", "ComfyUI-WanVideoWrapper", "ComfyUI-WanVideoWrapper-SCAIL2"):
            repo = self.root / name
            repo.mkdir()
            self.repositories[name] = repo
            self.git(repo, "init", "--template=")
            self.git(repo, "config", "user.name", "Contract Test")
            self.git(repo, "config", "user.email", "test@example.invalid")
            self.origins[name] = f"https://github.com/example/{name}.git"
            self.git(repo, "remote", "add", "origin", self.origins[name])
            (repo / "nodes.py").write_text(self.source("image") + f"# {name}\n", encoding="utf-8")
            self.git(repo, "add", "nodes.py")
            self.git(repo, "commit", "-m", "test: synthetic source")
            self.pins[name] = self.git(repo, "rev-parse", "HEAD")
        core = self.repositories["ComfyUI"]
        (core / "nodes.py").write_text(self.source("picture"), encoding="utf-8")
        self.git(core, "commit", "-am", "test: synthetic candidate")
        self.candidate = self.git(core, "rev-parse", "HEAD")
        accepted = {name: (self.origins[name], sha, "") for name, sha in self.pins.items()}
        candidate = dict(accepted)
        candidate["ComfyUI"] = (self.origins["ComfyUI"], self.candidate, "")
        patch = mock.patch.object(self.parity, "SOURCE_PROFILES",
                                  {"accepted-20260824": accepted, "candidate-20261003": candidate})
        patch.start()
        self.addCleanup(patch.stop)
        self.manifest = self.root / "manifest.json"
        self.rows = []
        for family, repo, revision, name in (
            ("comfyui-core", "ComfyUI", self.pins["ComfyUI"], "image"),
            ("comfyui-core", "ComfyUI", self.candidate, "picture"),
            ("kijai-vanilla", "ComfyUI-WanVideoWrapper", self.pins["ComfyUI-WanVideoWrapper"], "image"),
            ("rookiestar28-scail2", "ComfyUI-WanVideoWrapper-SCAIL2", self.pins["ComfyUI-WanVideoWrapper-SCAIL2"], "image"),
        ):
            self.rows.append({"family": family, "origin": self.origins[repo], "revision": revision,
                              "nodes": [self.node(name)]})
        self.write_manifest()

    @staticmethod
    def git(repo: Path, *args: str) -> str:
        import subprocess
        return subprocess.check_output(["git", "-C", str(repo), *args],
                                       text=True, stderr=subprocess.STDOUT).strip()

    @staticmethod
    def source(name: str) -> str:
        return (f'class ExampleNode:\n'
                f'    def INPUT_TYPES(self):\n'
                f'        return {{"required": {{"{name}": ("IMAGE",)}}, "optional": {{}}}}\n'
                f'    RETURN_TYPES = ("LATENT",)\n'
                f'    RETURN_NAMES = ("samples",)\n'
                f'raise RuntimeError("host source must never execute")\n')

    @staticmethod
    def node(name: str) -> dict:
        return {"class": "ExampleNode", "required_inputs": [{"name": name, "type": "IMAGE"}],
                "optional_inputs": [], "schema_order": [name],
                "outputs": [{"index": 0, "name": "samples", "type": "LATENT"}], "output_count": 1,
                "evidence": {"path": "nodes.py", "symbol": "ExampleNode", "lines": "1-5"}}

    def write_manifest(self) -> None:
        import json
        self.manifest.write_text(json.dumps({"schema": "scail_pose2.host_compatibility.v1",
                                             "families": self.rows}), encoding="utf-8")

    def check(self, profile: str = "candidate-20261003") -> tuple[str, ...]:
        return self.parity.check_manifest_against_reference(self.manifest, self.root, profile=profile)

    def test_candidate_reads_each_rows_own_git_object_not_current_source(self) -> None:
        self.assertEqual((), self.check())

    def test_historical_profile_still_rejects_candidate_checkout(self) -> None:
        self.assertIn("revision-mismatch:ComfyUI", self.check("accepted-20260824"))

    def test_unknown_profile_and_manifest_revision_fail_closed(self) -> None:
        self.assertEqual(("profile-unknown",), self.check("unapproved"))
        self.rows[1]["revision"] = "0" * 40
        self.write_manifest()
        self.assertIn("family-provenance-mismatch:comfyui-core", self.check())

    def test_origin_and_tracked_drift_are_detected_but_untracked_not_read(self) -> None:
        core = self.repositories["ComfyUI"]
        (core / "untracked.py").write_text("raise RuntimeError('untrusted')", encoding="utf-8")
        self.assertEqual((), self.check())
        (core / "nodes.py").write_text(self.source("wrong"), encoding="utf-8")
        self.assertIn("status-mismatch:ComfyUI", self.check())
        self.git(core, "remote", "set-url", "origin", "https://github.com/example/wrong.git")
        self.assertIn("origin-mismatch:ComfyUI", self.check())

    def test_missing_old_object_is_not_replaced_with_head(self) -> None:
        original = self.parity._git
        def read(repo, *args):
            if args[0] == "show" and args[1].startswith(self.pins["ComfyUI"] + ":"):
                import subprocess
                raise subprocess.CalledProcessError(128, ["git", "show"])
            return original(repo, *args)
        with mock.patch.object(self.parity, "_git", side_effect=read):
            self.assertIn("source-object-unavailable:comfyui-core:ExampleNode", self.check())

    def test_containment_and_wrong_fork_provenance_are_rejected(self) -> None:
        self.rows[1]["nodes"][0]["evidence"]["path"] = "../outside.py"
        self.write_manifest()
        self.assertIn("source-containment-failed:comfyui-core:ExampleNode", self.check())
        self.rows[1]["nodes"][0]["evidence"]["path"] = "nodes.py"
        self.rows[-1]["revision"] = self.pins["ComfyUI-WanVideoWrapper"]
        self.write_manifest()
        self.assertIn("family-provenance-mismatch:rookiestar28-scail2", self.check())

    def test_invalid_evidence_and_socket_order_output_mutations_are_rejected(self) -> None:
        node = self.rows[1]["nodes"][0]
        node["evidence"]["lines"] = "1-99999"
        self.write_manifest()
        self.assertIn("evidence-lines-invalid:comfyui-core:ExampleNode", self.check())
        node["evidence"]["lines"] = "1-5"
        node["schema_order"] = ["unknown"]
        self.write_manifest()
        self.assertTrue(any("input-order-mismatch" in e for e in self.check()))
        node["schema_order"] = ["picture"]
        node["outputs"][0]["type"] = "IMAGE"
        self.write_manifest()
        self.assertTrue(any("output-contract-mismatch" in e for e in self.check()))


if __name__ == "__main__":
    unittest.main()
