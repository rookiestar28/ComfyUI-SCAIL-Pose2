"""Offline, non-mutating validation for revision-pinned workflow contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "scail_pose2.host_compatibility.v1"
WORKFLOW_SCHEMA = "scail_pose2.workflow_contract.v1"
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "workflow_contracts"
    / "wanvideo_host_contracts.v1.json"
)
# SECURITY: diagnostic identifiers must never accept path separators from workflow data.
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.@+-]{1,96}$")
_SHA40 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class SocketContract:
    name: str
    comfy_type: str


@dataclass(frozen=True)
class OutputContract:
    index: int
    name: str
    comfy_type: str


@dataclass(frozen=True)
class SourceEvidence:
    path: str
    symbol: str
    lines: str


@dataclass(frozen=True)
class NodeContract:
    class_name: str
    required_inputs: tuple[SocketContract, ...]
    optional_inputs: tuple[SocketContract, ...]
    schema_order: tuple[str, ...]
    outputs: tuple[OutputContract, ...]
    evidence: SourceEvidence

    @property
    def required_input_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.required_inputs)

    @property
    def optional_input_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.optional_inputs)

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.required_input_names + self.optional_input_names

    def input(self, name: str) -> SocketContract | None:
        return next(
            (
                item
                for item in self.required_inputs + self.optional_inputs
                if item.name == name
            ),
            None,
        )

    def output(self, selector: str | int) -> OutputContract | None:
        if isinstance(selector, bool):
            return None
        if isinstance(selector, int):
            return next((item for item in self.outputs if item.index == selector), None)
        if isinstance(selector, str):
            return next((item for item in self.outputs if item.name == selector), None)
        return None


@dataclass(frozen=True)
class HostFamilyContract:
    family_id: str
    revision: str
    origin: str
    nodes: tuple[NodeContract, ...]

    def node(self, class_name: str) -> NodeContract:
        found = next((item for item in self.nodes if item.class_name == class_name), None)
        if found is None:
            raise KeyError(class_name)
        return found

    def find_node(self, class_name: str) -> NodeContract | None:
        return next((item for item in self.nodes if item.class_name == class_name), None)


@dataclass(frozen=True)
class ContractManifest:
    schema: str
    families: tuple[HostFamilyContract, ...]

    def family(self, family_id: str, revision: str) -> HostFamilyContract:
        found = next(
            (
                item
                for item in self.families
                if item.family_id == family_id and item.revision == revision
            ),
            None,
        )
        if found is None:
            raise KeyError((family_id, revision))
        return found

    def find_family(
        self, family_id: str, revision: str
    ) -> HostFamilyContract | None:
        return next(
            (
                item
                for item in self.families
                if item.family_id == family_id and item.revision == revision
            ),
            None,
        )


@dataclass(frozen=True)
class WorkflowDiagnostic:
    code: str
    message: str
    workflow_id: str
    family: str
    revision: str
    node_id: str = ""
    node_class: str = ""
    endpoint: str = ""
    expected: str = ""
    actual: str = ""

    def render(self) -> str:
        fields = [
            f"code={self.code}",
            f"workflow={self.workflow_id}",
            f"family={self.family}",
            f"revision={self.revision}",
        ]
        for name, value in (
            ("node", self.node_id),
            ("class", self.node_class),
            ("endpoint", self.endpoint),
            ("expected", self.expected),
            ("actual", self.actual),
        ):
            if value:
                fields.append(f"{name}={value}")
        return f"{self.message} ({', '.join(fields)})"


@dataclass(frozen=True)
class WorkflowValidationResult:
    workflow_id: str
    family: str
    revision: str
    diagnostics: tuple[WorkflowDiagnostic, ...]

    @property
    def valid(self) -> bool:
        return not self.diagnostics


class WorkflowContractError(ValueError):
    """Raised when a workflow does not satisfy its declared host contract."""

    def __init__(self, result: WorkflowValidationResult):
        self.result = result
        super().__init__("; ".join(item.render() for item in result.diagnostics))


def load_contract_manifest(path: str | Path | None = None) -> ContractManifest:
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _parse_manifest(raw)


def inspect_workflow_contract(
    workflow: Any,
    *,
    manifest: ContractManifest | None = None,
) -> tuple[WorkflowDiagnostic, ...]:
    return _validate(workflow, manifest or load_contract_manifest()).diagnostics


def validate_workflow_contract(
    workflow: Any,
    *,
    manifest: ContractManifest | None = None,
) -> WorkflowValidationResult:
    result = _validate(workflow, manifest or load_contract_manifest())
    if not result.valid:
        raise WorkflowContractError(result)
    return result


def _parse_manifest(raw: Any) -> ContractManifest:
    root = _require_mapping(raw, "manifest")
    if root.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported manifest schema: {_schema_token(root.get('schema'))}")
    family_rows = _require_sequence(root.get("families"), "families")
    families: list[HostFamilyContract] = []
    family_keys: set[tuple[str, str]] = set()
    for family_index, family_raw in enumerate(family_rows):
        row = _require_mapping(family_raw, f"families[{family_index}]")
        family_id = _required_string(row, "family")
        revision = _required_string(row, "revision")
        origin = _required_string(row, "origin")
        if not _SHA40.fullmatch(revision):
            raise ValueError(f"Invalid full revision for family {family_id}")
        if not origin.startswith("https://github.com/"):
            raise ValueError(f"Manifest origin must be public GitHub HTTPS: {family_id}")
        key = (family_id, revision)
        if key in family_keys:
            raise ValueError(f"Duplicate family/revision: {family_id}@{revision}")
        family_keys.add(key)
        nodes = _parse_nodes(row.get("nodes"), family_id)
        families.append(HostFamilyContract(family_id, revision, origin, nodes))
    if not families:
        raise ValueError("Manifest must declare at least one family")
    return ContractManifest(MANIFEST_SCHEMA, tuple(families))


def _parse_nodes(raw: Any, family_id: str) -> tuple[NodeContract, ...]:
    rows = _require_sequence(raw, f"{family_id}.nodes")
    nodes: list[NodeContract] = []
    seen: set[str] = set()
    for index, node_raw in enumerate(rows):
        row = _require_mapping(node_raw, f"{family_id}.nodes[{index}]")
        class_name = _required_string(row, "class")
        if class_name in seen:
            raise ValueError(f"Duplicate node class in {family_id}: {class_name}")
        seen.add(class_name)
        required = _parse_sockets(row.get("required_inputs"), class_name, "required")
        optional = _parse_sockets(row.get("optional_inputs"), class_name, "optional")
        all_names = tuple(item.name for item in required + optional)
        if len(set(all_names)) != len(all_names):
            raise ValueError(f"Duplicate input socket in {class_name}")
        schema_order = tuple(
            _required_scalar_string(item, f"{class_name}.schema_order")
            for item in _require_sequence(row.get("schema_order"), "schema_order")
        )
        if len(schema_order) != len(all_names) or set(schema_order) != set(all_names):
            raise ValueError(f"schema_order mismatch in {class_name}")
        outputs = _parse_outputs(row.get("outputs"), class_name)
        output_count = row.get("output_count")
        if isinstance(output_count, bool) or not isinstance(output_count, int):
            raise ValueError(f"Invalid output_count in {class_name}")
        if output_count != len(outputs):
            raise ValueError(f"Output count mismatch in {class_name}")
        evidence_raw = _require_mapping(row.get("evidence"), f"{class_name}.evidence")
        evidence = SourceEvidence(
            path=_required_string(evidence_raw, "path"),
            symbol=_required_string(evidence_raw, "symbol"),
            lines=_required_string(evidence_raw, "lines"),
        )
        nodes.append(
            NodeContract(
                class_name,
                required,
                optional,
                schema_order,
                outputs,
                evidence,
            )
        )
    return tuple(nodes)


def _parse_sockets(raw: Any, class_name: str, group: str) -> tuple[SocketContract, ...]:
    rows = _require_sequence(raw, f"{class_name}.{group}_inputs")
    return tuple(
        SocketContract(
            _required_string(_require_mapping(item, group), "name"),
            _required_string(_require_mapping(item, group), "type"),
        )
        for item in rows
    )


def _parse_outputs(raw: Any, class_name: str) -> tuple[OutputContract, ...]:
    rows = _require_sequence(raw, f"{class_name}.outputs")
    outputs: list[OutputContract] = []
    for expected_index, output_raw in enumerate(rows):
        row = _require_mapping(output_raw, f"{class_name}.outputs[{expected_index}]")
        index = row.get("index")
        if isinstance(index, bool) or index != expected_index:
            raise ValueError(f"Non-contiguous output index in {class_name}")
        outputs.append(
            OutputContract(
                expected_index,
                _required_string(row, "name"),
                _required_string(row, "type"),
            )
        )
    return tuple(outputs)


def _validate(workflow: Any, manifest: ContractManifest) -> WorkflowValidationResult:
    if not isinstance(workflow, Mapping):
        diagnostic = _diagnostic(
            "INVALID_WORKFLOW", "Workflow contract must be an object", "<invalid>", "", ""
        )
        return WorkflowValidationResult("<invalid>", "", "", (diagnostic,))

    workflow_id = _safe_value(workflow.get("workflow_id"), "<invalid>")
    diagnostics: list[WorkflowDiagnostic] = []
    if workflow.get("schema") != WORKFLOW_SCHEMA:
        diagnostics.append(
            _diagnostic(
                "INVALID_WORKFLOW_SCHEMA",
                "Unsupported workflow contract schema",
                workflow_id,
                "",
                "",
                expected=WORKFLOW_SCHEMA,
                actual=_schema_token(workflow.get("schema")),
            )
        )

    host = workflow.get("host")
    if not isinstance(host, Mapping):
        diagnostics.append(
            _diagnostic(
                "INVALID_HOST_DECLARATION",
                "Workflow must declare one host family and full revision",
                workflow_id,
                "<missing>",
                "<missing>",
                expected="family+40hex-revision",
            )
        )
        return WorkflowValidationResult(
            workflow_id, "<missing>", "<missing>", tuple(diagnostics)
        )
    family_id = _safe_value(host.get("family"), "<invalid>")
    revision = _safe_value(host.get("revision"), "<invalid>")
    if family_id == "<invalid>" or not _SHA40.fullmatch(revision):
        diagnostics.append(
            _diagnostic(
                "INVALID_HOST_DECLARATION",
                "Workflow host declaration is malformed",
                workflow_id,
                family_id,
                revision,
                expected="safe-family+40hex-revision",
            )
        )
        return WorkflowValidationResult(workflow_id, family_id, revision, tuple(diagnostics))
    family = manifest.find_family(family_id, revision)
    if family is None:
        diagnostics.append(
            _diagnostic(
                "UNSUPPORTED_HOST_FAMILY",
                "Declared host family/revision is unsupported",
                workflow_id,
                family_id,
                revision,
                expected="pinned-manifest-family",
            )
        )
        return WorkflowValidationResult(workflow_id, family_id, revision, tuple(diagnostics))

    node_rows = workflow.get("nodes")
    if not _is_sequence(node_rows):
        diagnostics.append(
            _diagnostic(
                "INVALID_NODE_LIST",
                "Workflow nodes must be an array",
                workflow_id,
                family_id,
                revision,
            )
        )
        return WorkflowValidationResult(workflow_id, family_id, revision, tuple(diagnostics))

    nodes: dict[str, Mapping[str, Any]] = {}
    contracts: dict[str, NodeContract] = {}
    literal_inputs: dict[str, set[str]] = {}
    for node_raw in node_rows:
        if not isinstance(node_raw, Mapping):
            diagnostics.append(
                _diagnostic(
                    "INVALID_NODE", "Node must be an object", workflow_id, family_id, revision
                )
            )
            continue
        node_id = _safe_value(node_raw.get("id"), "<invalid>")
        class_name = _safe_value(node_raw.get("class_type"), "<invalid>")
        if node_id == "<invalid>" or node_id in nodes:
            diagnostics.append(
                _diagnostic(
                    "INVALID_NODE_ID",
                    "Node ID is missing, unsafe, or duplicated",
                    workflow_id,
                    family_id,
                    revision,
                    node_id=node_id,
                    node_class=class_name,
                )
            )
            continue
        nodes[node_id] = node_raw
        contract = family.find_node(class_name)
        if contract is None:
            diagnostics.append(
                _diagnostic(
                    "UNKNOWN_NODE_CLASS",
                    "Node class is unavailable in the declared host family",
                    workflow_id,
                    family_id,
                    revision,
                    node_id=node_id,
                    node_class=class_name,
                    expected=f"family={family_id}@{revision}",
                )
            )
            continue
        contracts[node_id] = contract
        inputs = node_raw.get("inputs", {})
        if not isinstance(inputs, Mapping):
            diagnostics.append(
                _diagnostic(
                    "INVALID_NODE_INPUTS",
                    "Node inputs must be an object of socket names",
                    workflow_id,
                    family_id,
                    revision,
                    node_id=node_id,
                    node_class=class_name,
                )
            )
            literal_inputs[node_id] = set()
            continue
        names: set[str] = set()
        for raw_name in inputs:
            name = _safe_value(raw_name, "<invalid>")
            if name == "<invalid>" or contract.input(name) is None:
                diagnostics.append(
                    _diagnostic(
                        "UNKNOWN_NODE_INPUT",
                        "Declared node input is not in the host contract",
                        workflow_id,
                        family_id,
                        revision,
                        node_id=node_id,
                        node_class=class_name,
                        endpoint=name,
                        expected="/".join(contract.input_names),
                    )
                )
                continue
            names.add(name)
        literal_inputs[node_id] = names

    satisfied = {node_id: set(names) for node_id, names in literal_inputs.items()}
    links = workflow.get("links", [])
    if not _is_sequence(links):
        diagnostics.append(
            _diagnostic(
                "INVALID_LINK_LIST",
                "Workflow links must be an array",
                workflow_id,
                family_id,
                revision,
            )
        )
        links = []
    for link in links:
        if not isinstance(link, Mapping):
            diagnostics.append(
                _diagnostic(
                    "INVALID_LINK", "Link must be an object", workflow_id, family_id, revision
                )
            )
            continue
        source = link.get("from")
        target = link.get("to")
        if not (_endpoint_pair(source) and _endpoint_pair(target)):
            diagnostics.append(
                _diagnostic(
                    "INVALID_LINK",
                    "Link endpoints must be two-item arrays",
                    workflow_id,
                    family_id,
                    revision,
                )
            )
            continue
        source_id = _safe_value(source[0], "<invalid>")
        target_id = _safe_value(target[0], "<invalid>")
        target_name = _safe_value(target[1], "<invalid>")
        source_contract = contracts.get(source_id)
        target_contract = contracts.get(target_id)
        if source_contract is None or target_contract is None:
            diagnostics.append(
                _diagnostic(
                    "UNKNOWN_LINK_NODE",
                    "Link endpoint node is unavailable",
                    workflow_id,
                    family_id,
                    revision,
                    node_id=source_id if source_contract is None else target_id,
                )
            )
            continue
        target_socket = target_contract.input(target_name)
        if target_socket is None:
            diagnostics.append(
                _diagnostic(
                    "UNKNOWN_TARGET_INPUT",
                    "Target socket is unavailable in the host contract",
                    workflow_id,
                    family_id,
                    revision,
                    node_id=target_id,
                    node_class=target_contract.class_name,
                    endpoint=target_name,
                    expected="/".join(target_contract.input_names),
                )
            )
            continue
        selector = source[1]
        source_output = source_contract.output(selector)
        if source_output is None:
            diagnostics.append(
                _diagnostic(
                    "INVALID_SOURCE_OUTPUT",
                    "Source output is unavailable or outside declared cardinality",
                    workflow_id,
                    family_id,
                    revision,
                    node_id=source_id,
                    node_class=source_contract.class_name,
                    endpoint=_schema_token(selector),
                    expected=f"count={len(source_contract.outputs)} names={'/'.join(item.name for item in source_contract.outputs)}",
                )
            )
            continue
        if not _types_compatible(source_output.comfy_type, target_socket.comfy_type):
            diagnostics.append(
                _diagnostic(
                    "LINK_TYPE_MISMATCH",
                    "Source output and target input types are incompatible",
                    workflow_id,
                    family_id,
                    revision,
                    node_id=target_id,
                    node_class=target_contract.class_name,
                    endpoint=target_name,
                    expected=target_socket.comfy_type,
                    actual=source_output.comfy_type,
                )
            )
            continue
        declared_type = link.get("type")
        if declared_type is not None and not (
            isinstance(declared_type, str)
            and _types_compatible(source_output.comfy_type, declared_type)
            and _types_compatible(target_socket.comfy_type, declared_type)
        ):
            diagnostics.append(
                _diagnostic(
                    "DECLARED_LINK_TYPE_MISMATCH",
                    "Declared link type disagrees with its endpoints",
                    workflow_id,
                    family_id,
                    revision,
                    node_id=target_id,
                    node_class=target_contract.class_name,
                    endpoint=target_name,
                    expected=target_socket.comfy_type,
                    actual=_schema_token(declared_type),
                )
            )
            continue
        satisfied.setdefault(target_id, set()).add(target_name)

    for node_id, contract in contracts.items():
        present = satisfied.get(node_id, set())
        for socket in contract.required_inputs:
            if socket.name not in present:
                diagnostics.append(
                    _diagnostic(
                        "MISSING_REQUIRED_INPUT",
                        "Required host input is neither linked nor declared",
                        workflow_id,
                        family_id,
                        revision,
                        node_id=node_id,
                        node_class=contract.class_name,
                        endpoint=socket.name,
                        expected=socket.comfy_type,
                    )
                )
    return WorkflowValidationResult(workflow_id, family_id, revision, tuple(diagnostics))


def _diagnostic(
    code: str,
    message: str,
    workflow_id: str,
    family: str,
    revision: str,
    *,
    node_id: str = "",
    node_class: str = "",
    endpoint: str = "",
    expected: str = "",
    actual: str = "",
) -> WorkflowDiagnostic:
    return WorkflowDiagnostic(
        code=code,
        message=message,
        workflow_id=_safe_value(workflow_id, "<invalid>"),
        family=_safe_value(family, "<invalid>") if family else "",
        revision=_safe_value(revision, "<invalid>") if revision else "",
        node_id=_safe_value(node_id, "<invalid>") if node_id else "",
        node_class=_safe_value(node_class, "<invalid>") if node_class else "",
        endpoint=_safe_value(endpoint, "<invalid>") if endpoint else "",
        expected=_safe_expected(expected),
        actual=_safe_value(actual, "<invalid>") if actual else "",
    )


def _safe_expected(value: str) -> str:
    if len(value) > 240 or not re.fullmatch(r"[A-Za-z0-9_.@/+=: -]+", value):
        return "<contract>"
    return value


def _safe_value(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and _SAFE_TOKEN.fullmatch(value) else fallback


def _schema_token(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and _SAFE_TOKEN.fullmatch(value):
        return value
    return f"<{type(value).__name__}>"


def _types_compatible(left: str, right: str) -> bool:
    return bool(set(left.split("/")) & set(right.split("/")))


def _endpoint_pair(value: Any) -> bool:
    return _is_sequence(value) and len(value) == 2


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not _is_sequence(value):
        raise ValueError(f"{label} must be an array")
    return value


def _required_string(row: Mapping[str, Any], key: str) -> str:
    return _required_scalar_string(row.get(key), key)


def _required_scalar_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value
