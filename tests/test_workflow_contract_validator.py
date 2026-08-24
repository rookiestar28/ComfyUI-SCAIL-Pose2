from __future__ import annotations

import copy
import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from scail2.workflow_contract_validator import (
    WorkflowContractError,
    inspect_workflow_contract,
    load_contract_manifest,
    validate_workflow_contract,
)


FORK_FAMILY = "rookiestar28-scail2"
FORK_REVISION = bytes(
    (0xEF, 0x95, 0xCF, 0xA0, 0xEE, 0xF9, 0xB5, 0xBC, 0x87, 0x04,
     0xAE, 0xC6, 0xB7, 0xD5, 0x96, 0x3D, 0x23, 0xAD, 0x02, 0xA9)
).hex()
VANILLA_FAMILY = "kijai-vanilla"
VANILLA_REVISION = bytes(
    (0x08, 0x81, 0x28, 0xB2, 0x24, 0x24, 0x2E, 0x11, 0x0D, 0x39,
     0x06, 0xC6, 0x75, 0x0E, 0x9A, 0x3A, 0x34, 0x8A, 0x65, 0x9B)
).hex()


class WorkflowContractValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_contract_manifest()

    def _node(self, node_id: str, class_type: str) -> dict[str, object]:
        family = self.manifest.family(FORK_FAMILY, FORK_REVISION)
        contract = family.node(class_type)
        return {
            "id": node_id,
            "class_type": class_type,
            "inputs": {socket.name: "private-widget-value" for socket in contract.required_inputs},
        }

    def _valid_context_workflow(self) -> dict[str, object]:
        context = self._node("context", "WanVideoContextOptions")
        extra = self._node("extra", "WanVideoSamplerExtraArgs")
        sampler = self._node("sampler", "WanVideoSamplerv2")
        extra["inputs"].pop("context_options", None)  # satisfied by its incoming link
        sampler["inputs"].pop("extra_args", None)  # satisfied by its incoming link
        return {
            "schema": "scail_pose2.workflow_contract.v1",
            "workflow_id": "unit-context-route",
            "host": {"family": FORK_FAMILY, "revision": FORK_REVISION},
            "nodes": [context, extra, sampler],
            "links": [
                {
                    "from": ["context", "context_options"],
                    "to": ["extra", "context_options"],
                },
                {"from": ["extra", "extra_args"], "to": ["sampler", "extra_args"]},
            ],
        }

    def test_valid_extra_args_context_route_passes(self) -> None:
        result = validate_workflow_contract(
            self._valid_context_workflow(), manifest=self.manifest
        )

        self.assertTrue(result.valid)
        self.assertEqual((), result.diagnostics)

    def test_manifest_is_revisioned_complete_and_public_safe(self) -> None:
        self.assertEqual("scail_pose2.host_compatibility.v1", self.manifest.schema)
        self.assertEqual(
            {"comfyui-core", VANILLA_FAMILY, FORK_FAMILY},
            {family.family_id for family in self.manifest.families},
        )
        raw_path = (
            Path(__file__).resolve().parents[1]
            / "workflow_contracts"
            / "wanvideo_host_contracts.v1.json"
        )
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        for family in raw["families"]:
            self.assertRegex(family["revision"], r"^[0-9a-f]{40}$")
            self.assertTrue(family["origin"].startswith("https://github.com/"))
            for node in family["nodes"]:
                grouped_names = [
                    item["name"]
                    for group in ("required_inputs", "optional_inputs")
                    for item in node[group]
                ]
                self.assertEqual(set(grouped_names), set(node["schema_order"]))
                self.assertEqual(len(grouped_names), len(node["schema_order"]))
                self.assertEqual(node["output_count"], len(node["outputs"]))
                self.assertEqual(
                    list(range(node["output_count"])),
                    [output["index"] for output in node["outputs"]],
                )
                self.assertNotIn("reference/", node["evidence"]["path"])
                self.assertFalse(Path(node["evidence"]["path"]).is_absolute())

    def test_manifest_contracts_are_immutable(self) -> None:
        family = self.manifest.family(FORK_FAMILY, FORK_REVISION)
        with self.assertRaises(FrozenInstanceError):
            family.revision = "0" * 40

    def test_encode_socket_is_family_specific(self) -> None:
        fork = self.manifest.family(FORK_FAMILY, FORK_REVISION).node("WanVideoEncode")
        vanilla = self.manifest.family(VANILLA_FAMILY, VANILLA_REVISION).node(
            "WanVideoEncode"
        )

        self.assertIn("driving_video", fork.required_input_names)
        self.assertNotIn("image", fork.input_names)
        self.assertIn("image", vanilla.required_input_names)
        self.assertNotIn("driving_video", vanilla.input_names)

    def test_unknown_node_class_fails_closed(self) -> None:
        workflow = self._valid_context_workflow()
        workflow["nodes"].append(
            {"id": "unknown", "class_type": "MissingHostNode", "inputs": {}}
        )

        diagnostics = inspect_workflow_contract(workflow, manifest=self.manifest)

        self.assertIn("UNKNOWN_NODE_CLASS", {item.code for item in diagnostics})

    def test_missing_required_input_is_identified(self) -> None:
        workflow = self._valid_context_workflow()
        context = workflow["nodes"][0]
        missing_name = next(iter(context["inputs"]))
        context["inputs"].pop(missing_name)

        diagnostics = inspect_workflow_contract(workflow, manifest=self.manifest)

        missing = next(item for item in diagnostics if item.code == "MISSING_REQUIRED_INPUT")
        self.assertEqual("context", missing.node_id)
        self.assertEqual(missing_name, missing.endpoint)

    def test_unknown_target_socket_is_identified(self) -> None:
        workflow = self._valid_context_workflow()
        workflow["links"][0]["to"][1] = "stale_context_socket"

        diagnostics = inspect_workflow_contract(workflow, manifest=self.manifest)

        self.assertIn("UNKNOWN_TARGET_INPUT", {item.code for item in diagnostics})

    def test_out_of_range_source_output_matches_comfyui_failure_class(self) -> None:
        workflow = self._valid_context_workflow()
        workflow["links"][0]["from"][1] = 4

        diagnostics = inspect_workflow_contract(workflow, manifest=self.manifest)

        invalid = next(item for item in diagnostics if item.code == "INVALID_SOURCE_OUTPUT")
        self.assertEqual("context", invalid.node_id)
        self.assertIn("count=1", invalid.expected)

    def test_output_input_type_mismatch_is_identified(self) -> None:
        workflow = self._valid_context_workflow()
        empty = self._node("empty", "WanVideoEmptyEmbeds")
        workflow["nodes"].append(empty)
        workflow["links"][0]["from"] = ["empty", "image_embeds"]

        diagnostics = inspect_workflow_contract(workflow, manifest=self.manifest)

        self.assertIn("LINK_TYPE_MISMATCH", {item.code for item in diagnostics})

    def test_direct_context_link_to_sampler_v2_is_rejected(self) -> None:
        workflow = self._valid_context_workflow()
        workflow["links"] = [
            {
                "from": ["context", "context_options"],
                "to": ["sampler", "context_options"],
            }
        ]

        diagnostics = inspect_workflow_contract(workflow, manifest=self.manifest)

        self.assertIn("UNKNOWN_TARGET_INPUT", {item.code for item in diagnostics})

    def test_wrong_family_and_ambiguous_family_are_rejected(self) -> None:
        wrong = self._valid_context_workflow()
        wrong["host"] = {"family": VANILLA_FAMILY, "revision": VANILLA_REVISION}
        wrong["nodes"] = [
            self._node("native", "WanVideoAddSCAIL2ConditionEmbeds")
        ]
        ambiguous = self._valid_context_workflow()
        ambiguous.pop("host")

        wrong_codes = {
            item.code for item in inspect_workflow_contract(wrong, manifest=self.manifest)
        }
        ambiguous_codes = {
            item.code
            for item in inspect_workflow_contract(ambiguous, manifest=self.manifest)
        }

        self.assertIn("UNKNOWN_NODE_CLASS", wrong_codes)
        self.assertIn("INVALID_HOST_DECLARATION", ambiguous_codes)

    def test_unpinned_revision_is_rejected(self) -> None:
        workflow = self._valid_context_workflow()
        workflow["host"]["revision"] = "0" * 40

        diagnostics = inspect_workflow_contract(workflow, manifest=self.manifest)

        self.assertEqual(("UNSUPPORTED_HOST_FAMILY",), tuple(item.code for item in diagnostics))

    def test_valid_optional_input_is_accepted(self) -> None:
        family = self.manifest.family(FORK_FAMILY, FORK_REVISION)
        encode = self._node("encode", "WanVideoEncode")
        self.assertIn("mask", family.node("WanVideoEncode").optional_input_names)
        encode["inputs"]["mask"] = "private-mask-value"
        workflow = {
            "schema": "scail_pose2.workflow_contract.v1",
            "workflow_id": "optional-mask",
            "host": {"family": FORK_FAMILY, "revision": FORK_REVISION},
            "nodes": [encode],
            "links": [],
        }

        self.assertTrue(
            validate_workflow_contract(workflow, manifest=self.manifest).valid
        )

    def test_diagnostics_are_payload_safe_and_input_is_not_mutated(self) -> None:
        workflow = self._valid_context_workflow()
        workflow["nodes"][0]["inputs"].pop(next(iter(workflow["nodes"][0]["inputs"])))
        workflow["private_prompt"] = "DO-NOT-LEAK-PROMPT"
        before = copy.deepcopy(workflow)

        with self.assertRaises(WorkflowContractError) as raised:
            validate_workflow_contract(workflow, manifest=self.manifest)

        self.assertEqual(before, workflow)
        rendered = str(raised.exception)
        self.assertNotIn("DO-NOT-LEAK-PROMPT", rendered)
        self.assertNotIn("private-widget-value", rendered)
        self.assertIn("unit-context-route", rendered)
        self.assertIn(FORK_FAMILY, rendered)
        self.assertIn(FORK_REVISION, rendered)

    def test_path_like_diagnostic_identifiers_are_redacted(self) -> None:
        cases = []

        workflow_id = self._valid_context_workflow()
        workflow_id["workflow_id"] = "/Users/Ray/private-output"
        workflow_id["nodes"][0]["inputs"].pop(
            next(iter(workflow_id["nodes"][0]["inputs"]))
        )
        cases.append(workflow_id)

        node_id = self._valid_context_workflow()
        node_id["nodes"].append(
            {"id": "private/output", "class_type": "MissingHostNode", "inputs": {}}
        )
        cases.append(node_id)

        endpoint = self._valid_context_workflow()
        endpoint["links"][0]["to"][1] = "private/output"
        cases.append(endpoint)

        family = self._valid_context_workflow()
        family["host"]["family"] = "private/output"
        cases.append(family)

        for workflow in cases:
            with self.subTest(workflow=workflow):
                diagnostics = inspect_workflow_contract(workflow, manifest=self.manifest)
                rendered = "\n".join(item.render() for item in diagnostics)
                self.assertNotIn("/Users/", rendered)
                self.assertNotIn("private/output", rendered)
                self.assertIn("<invalid>", rendered)


if __name__ == "__main__":
    unittest.main()
