"""Verify the tracked host manifest against pinned source text without loading host code."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any


FAMILY_REPOSITORIES = {
    "comfyui-core": "ComfyUI",
    "kijai-vanilla": "ComfyUI-WanVideoWrapper",
    "rookiestar28-scail2": "ComfyUI-WanVideoWrapper-SCAIL2",
}

PINNED_REPOSITORIES = {
    "ComfyUI": (
        "https://github.com/Comfy-Org/ComfyUI.git",
        bytes((0xB7, 0x8C, 0xEC, 0x87, 0x9B, 0x94, 0x60, 0xD5, 0xCB, 0x25,
               0x22, 0x8A, 0x83, 0xA9, 0x42, 0xFB, 0x78, 0xD2, 0xCD, 0x24)).hex(),
        "",
    ),
    "ComfyUI-WanVideoWrapper": (
        "https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
        bytes((0x08, 0x81, 0x28, 0xB2, 0x24, 0x24, 0x2E, 0x11, 0x0D, 0x39,
               0x06, 0xC6, 0x75, 0x0E, 0x9A, 0x3A, 0x34, 0x8A, 0x65, 0x9B)).hex(),
        "",
    ),
    "ComfyUI-WanVideoWrapper-SCAIL2": (
        "https://github.com/rookiestar28/ComfyUI-WanVideoWrapper.git",
        bytes((0xEF, 0x95, 0xCF, 0xA0, 0xEE, 0xF9, 0xB5, 0xBC, 0x87, 0x04,
               0xAE, 0xC6, 0xB7, 0xD5, 0x96, 0x3D, 0x23, 0xAD, 0x02, 0xA9)).hex(),
        "",
    ),
    "SCAIL-Pose": (
        "https://github.com/zai-org/SCAIL-Pose.git",
        bytes((0x51, 0x9C, 0x7F, 0x54, 0xCB, 0x97, 0x2E, 0x7F, 0x92, 0x68,
               0x42, 0x13, 0xB7, 0xEF, 0x6C, 0x3E, 0x05, 0xA8, 0xF3, 0xB2)).hex(),
        "",
    ),
    "SCAIL2": (
        "https://github.com/zai-org/SCAIL-2.git",
        bytes((0xF9, 0x98, 0xBC, 0xC2, 0x91, 0x27, 0xAE, 0x9B, 0x17, 0x77,
               0x11, 0xEE, 0x8F, 0x39, 0xD6, 0x5C, 0xCD, 0x73, 0xCC, 0xA1)).hex(),
        "UU README.md",
    ),
}

SOURCE_PROFILES = MappingProxyType({
    "accepted-20260824": MappingProxyType(dict(PINNED_REPOSITORIES)),
    "candidate-20261003": MappingProxyType({
        **PINNED_REPOSITORIES,
        "ComfyUI": (PINNED_REPOSITORIES["ComfyUI"][0],
                    bytes((0x30, 0xC4, 0xC3, 0xAA, 0x75, 0x6F, 0x46, 0x55, 0xCD, 0xE8,
                           0x9C, 0x28, 0x7C, 0x07, 0x45, 0x38, 0x1A, 0x6C, 0x2B, 0xBC)).hex(), ""),
    }),
})


def _literal_text(node: ast.AST) -> str:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise ValueError("Nonliteral schema name")
    return node.value


def _io_type(node: ast.AST) -> str:
    name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""
    types = {"Conditioning": "CONDITIONING", "Vae": "VAE", "Int": "INT", "Float": "FLOAT",
             "Image": "IMAGE", "Mask": "MASK", "Boolean": "BOOLEAN", "String": "STRING",
             "Combo": "COMBO", "Latent": "LATENT", "ClipVisionOutput": "CLIP_VISION_OUTPUT",
             "SAM3TrackData": "SAM3_TRACK_DATA"}
    if name not in types:
        raise ValueError("Unsupported IO type")
    return types[name]


def _source_schema(source: str, symbol: str) -> tuple[dict[str, list[tuple[str, str]]], list[str], list[tuple[str, str]]]:
    classes = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)}
    cls = classes[symbol]
    methods = {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    groups: dict[str, list[tuple[str, str]]] = {"required_inputs": [], "optional_inputs": []}
    order: list[str] = []
    if "define_schema" in methods:
        returns = [n.value for n in ast.walk(methods["define_schema"]) if isinstance(n, ast.Return)]
        schema = next(n for n in returns if isinstance(n, ast.Call) and
                      isinstance(n.func, ast.Attribute) and n.func.attr == "Schema")
        fields = {k.arg: k.value for k in schema.keywords}
        for spec in fields["inputs"].elts:
            if not isinstance(spec, ast.Call) or not isinstance(spec.func, ast.Attribute):
                raise ValueError("Unsupported input schema")
            name = _literal_text(spec.args[0])
            kind = spec.func.value
            if isinstance(kind, ast.Attribute) and kind.attr == "MultiType":
                comfy_type = "/".join(_io_type(t) for t in spec.args[1].elts)
            else:
                comfy_type = _io_type(kind)
            optional = next((k.value for k in spec.keywords if k.arg == "optional"), ast.Constant(False))
            if not isinstance(optional, ast.Constant) or type(optional.value) is not bool:
                raise ValueError("Nonliteral optional flag")
            groups["optional_inputs" if optional.value else "required_inputs"].append((name, comfy_type))
            order.append(name)
        outputs = []
        for spec in fields["outputs"].elts:
            comfy_type = _io_type(spec.func.value)
            positional_name = _literal_text(spec.args[0]) if spec.args else comfy_type
            name = next((_literal_text(k.value) for k in spec.keywords if k.arg == "display_name"),
                        positional_name)
            outputs.append((name, comfy_type))
        return groups, order, outputs
    returns = [n.value for n in ast.walk(methods["INPUT_TYPES"]) if isinstance(n, ast.Return)]
    schema = next(n for n in returns if isinstance(n, ast.Dict))
    for group, values in zip(schema.keys, schema.values):
        group_name = _literal_text(group) + "_inputs"
        if group_name not in groups:
            continue
        if not isinstance(values, ast.Dict):
            raise ValueError("Unsupported input dictionary")
        for key, spec in zip(values.keys, values.values):
            name = _literal_text(key)
            if not isinstance(spec, (ast.Tuple, ast.List)) or not spec.elts:
                raise ValueError("Unsupported input specification")
            kind = spec.elts[0]
            comfy_type = _literal_text(kind) if isinstance(kind, ast.Constant) else "COMBO"
            groups[group_name].append((name, comfy_type))
            order.append(name)
    def attribute(owner: ast.ClassDef, name: str, seen: frozenset[str] = frozenset()) -> ast.AST | None:
        if owner.name in seen:
            raise ValueError("Cyclic schema inheritance")
        for statement in owner.body:
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in statement.targets):
                return statement.value
        for base in owner.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                found = attribute(classes[base.id], name, seen | {owner.name})
                if found is not None:
                    return found
        return None
    types = attribute(cls, "RETURN_TYPES")
    if not isinstance(types, (ast.Tuple, ast.List)):
        raise ValueError("Unsupported output schema")
    names = attribute(cls, "RETURN_NAMES") or types
    if not isinstance(names, (ast.Tuple, ast.List)) or len(names.elts) != len(types.elts):
        raise ValueError("Unsupported output names")
    return groups, order, list(zip(map(_literal_text, names.elts), map(_literal_text, types.elts)))


def validate_node_source(
    node: Mapping[str, Any], source_text: str, evidence_text: str | None = None
) -> tuple[str, ...]:
    """Compare literal socket groups/order and outputs without evaluating host code."""
    symbol = _safe_id(node.get("evidence", {}).get("symbol", node.get("class", "")))
    try:
        groups, order, outputs = _source_schema(source_text, symbol)
    except (ValueError, KeyError, StopIteration, AttributeError, IndexError, SyntaxError, TypeError):
        return (f"source-schema-unreadable:{symbol}",)
    errors: list[str] = []
    for group in ("required_inputs", "optional_inputs"):
        expected = [(s.get("name"), s.get("type")) for s in node.get(group, [])]
        if expected != groups[group]:
            errors.append(f"input-contract-mismatch:{symbol}:{group}")
    if "schema_order" in node and node["schema_order"] != order:
        errors.append(f"input-order-mismatch:{symbol}")
    expected_outputs = [(s.get("name"), s.get("type")) for s in node.get("outputs", [])]
    if expected_outputs != outputs or node.get("output_count") != len(outputs):
        errors.append(f"output-contract-mismatch:{symbol}")
    if [s.get("index") for s in node.get("outputs", [])] != list(range(len(outputs))):
        errors.append(f"output-index-mismatch:{symbol}")
    if evidence_text is not None and any(not _quoted_name_present(evidence_text, str(name))
                                         for name in order + [o[0] for o in outputs]):
        errors.append(f"evidence-contract-missing:{symbol}")
    return tuple(errors)


def _safe_id(value: Any) -> str:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9-]{0,80}", value) else "invalid"


def check_manifest_against_reference(
    manifest_path: Path, reference_root: Path, *, profile: str = "accepted-20260824"
) -> tuple[str, ...]:
    if profile not in SOURCE_PROFILES:
        return ("profile-unknown",)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping) or manifest.get("schema") != "scail_pose2.host_compatibility.v1":
            return ("manifest-schema-invalid",)
        families = manifest.get("families")
        if not isinstance(families, list):
            return ("manifest-families-invalid",)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ("manifest-unreadable",)
    root = reference_root.resolve()
    pins = SOURCE_PROFILES[profile]
    errors: list[str] = []
    repositories: dict[str, Path] = {}
    for repo_name, (origin, revision, expected_status) in pins.items():
        try:
            repo = _reference_repository(root, repo_name)
        except (OSError, ValueError):
            errors.append(f"repository-containment-failed:{repo_name}")
            continue
        if not repo.is_dir():
            errors.append(f"repository-missing:{repo_name}")
            continue
        repositories[repo_name] = repo
        try:
            actual_origin = _git(repo, "remote", "get-url", "origin")
            actual_revision = _git(repo, "rev-parse", "HEAD")
            actual_status = _git(repo, "status", "--porcelain=v1", "-uno")
        except (OSError, subprocess.CalledProcessError, UnicodeError):
            errors.append(f"repository-metadata-read-failed:{repo_name}")
            continue
        if actual_origin != origin:
            errors.append(f"origin-mismatch:{repo_name}")
        if actual_revision != revision:
            errors.append(f"revision-mismatch:{repo_name}")
        if actual_status != expected_status:
            errors.append(f"status-mismatch:{repo_name}")
    allowed = {(repo, origin, revision) for values in SOURCE_PROFILES.values()
               for repo, (origin, revision, _) in values.items()}
    seen: set[tuple[str, str]] = set()
    for family in families:
        if not isinstance(family, Mapping):
            errors.append("manifest-family-invalid")
            continue
        family_id = _safe_id(family.get("family"))
        repo_name = FAMILY_REPOSITORIES.get(family_id)
        if repo_name is None:
            errors.append("family-repository-unknown")
            continue
        origin, revision = family.get("origin"), family.get("revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision) or not isinstance(origin, str):
            errors.append(f"family-provenance-mismatch:{family_id}")
            continue
        if (repo_name, origin, revision) not in allowed:
            errors.append(f"family-provenance-mismatch:{family_id}")
            continue
        key = (family_id, revision)
        if key in seen:
            errors.append(f"family-duplicate:{family_id}")
            continue
        seen.add(key)
        nodes = family.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            errors.append(f"family-nodes-invalid:{family_id}")
            continue
        repo = repositories.get(repo_name)
        if repo is None:
            errors.append(f"family-repository-unavailable:{family_id}")
            continue
        for node in nodes:
            if not isinstance(node, Mapping):
                errors.append(f"node-invalid:{family_id}")
                continue
            symbol = _safe_id(node.get("class"))
            evidence = node.get("evidence")
            if not isinstance(evidence, Mapping):
                errors.append(f"evidence-invalid:{family_id}:{symbol}")
                continue
            relative = evidence.get("path")
            try:
                # SECURITY: Git object paths must remain inside their own repository, not just reference/.
                if not isinstance(relative, str) or not re.fullmatch(r"[A-Za-z_0-9./-]+", relative):
                    raise ValueError("Invalid relative source path")
                source_path = _contained_path(root, str(Path(repo_name) / relative))
                source_path.relative_to(repo.resolve())
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError("Escaping source path")
            except (OSError, ValueError):
                errors.append(f"source-containment-failed:{family_id}:{symbol}")
                continue
            try:
                # CRITICAL: each row reads its pinned object; HEAD text would falsely rebaseline historical contracts.
                source_text = _git(repo, "show", f"{revision}:{relative}")
            except (OSError, subprocess.CalledProcessError, UnicodeError):
                errors.append(f"source-object-unavailable:{family_id}:{symbol}")
                continue
            try:
                evidence_text = _evidence_slice(source_text, str(evidence.get("lines", "")))
            except ValueError:
                errors.append(f"evidence-lines-invalid:{family_id}:{symbol}")
                continue
            try:
                errors.extend(validate_node_source(node, source_text, evidence_text))
            except (AttributeError, KeyError, TypeError):
                errors.append(f"node-invalid:{family_id}:{symbol}")
    expected_pairs = {(family, pins[repo][1]) for family, repo in FAMILY_REPOSITORIES.items() if repo in pins}
    if not expected_pairs <= seen:
        errors.append("manifest-profile-incomplete")
    return tuple(errors)


def _evidence_slice(source_text: str, specification: str) -> str:
    lines = source_text.splitlines()
    selected: list[str] = []
    for part in specification.split(","):
        token = part.strip()
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if match is None:
            raise ValueError("Invalid evidence line specification")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start or end > len(lines):
            raise ValueError("Evidence line specification is outside source")
        selected.extend(lines[start - 1 : end])
    if not selected:
        raise ValueError("Evidence line specification is empty")
    return "\n".join(selected)


def _quoted_name_present(text: str, name: str) -> bool:
    return f'"{name}"' in text or f"'{name}'" in text


def _contained_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Reference path escapes the supplied root")
    return candidate


def _reference_repository(reference_root: Path, repo_name: str) -> Path:
    root = reference_root.resolve()
    lexical_candidate = root / repo_name
    if lexical_candidate.is_symlink():
        raise ValueError("Symlinked reference repositories are not allowed")
    return _contained_path(root, repo_name)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8", stderr=subprocess.STDOUT
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(SOURCE_PROFILES), default="accepted-20260824")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "workflow_contracts"
        / "wanvideo_host_contracts.v1.json",
    )
    args = parser.parse_args()
    errors = check_manifest_against_reference(args.manifest, args.reference_root, profile=args.profile)
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    node_count = sum(len(family["nodes"]) for family in manifest["families"])
    print(
        f"PASS profile={args.profile} repositories={len(SOURCE_PROFILES[args.profile])} families="
        f"{len(manifest['families'])} nodes={node_count} mode=text-only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
