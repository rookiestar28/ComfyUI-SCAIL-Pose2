"""Verify the tracked host manifest against pinned source text without loading host code."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
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

TYPE_MARKERS = {
    "BOOLEAN": ("\"BOOLEAN\"", "io.Boolean"),
    "CLIP_VISION_OUTPUT": ("\"CLIP_VISION_OUTPUT\"", "io.ClipVisionOutput"),
    "COMBO": ("io.Combo", "([", "rope_functions"),
    "CONDITIONING": ("\"CONDITIONING\"", "io.Conditioning"),
    "FLOAT": ("\"FLOAT\"", "io.Float"),
    "IMAGE": ("\"IMAGE\"", "io.Image"),
    "INT": ("\"INT\"", "io.Int"),
    "LATENT": ("\"LATENT\"", "io.Latent"),
    "MASK": ("\"MASK\"", "io.Mask"),
    "SAM3_TRACK_DATA": ("SAM3_TRACK_DATA", "SAM3TrackData"),
    "STRING": ("\"STRING\"", "io.String"),
    "VAE": ("\"VAE\"", "io.Vae"),
}


def validate_node_source(
    node: Mapping[str, Any], source_text: str, evidence_text: str | None = None
) -> tuple[str, ...]:
    """Return content-free contract mismatches for one source class."""

    symbol = str(node.get("evidence", {}).get("symbol", node.get("class", "")))
    class_block = _class_block(source_text, symbol)
    if class_block is None:
        return (f"class-not-found:{symbol}",)
    bounded_text = evidence_text if evidence_text is not None else source_text
    errors: list[str] = []
    for group in ("required_inputs", "optional_inputs"):
        sockets = node.get(group, [])
        if not isinstance(sockets, Sequence) or isinstance(sockets, (str, bytes)):
            errors.append(f"invalid-{group}:{symbol}")
            continue
        for socket in sockets:
            if not isinstance(socket, Mapping):
                errors.append(f"invalid-socket:{symbol}")
                continue
            name = str(socket.get("name", ""))
            comfy_type = str(socket.get("type", ""))
            if not _quoted_name_present(class_block, name):
                errors.append(f"input-name-missing:{symbol}:{name}")
            if not _type_present(class_block, bounded_text, comfy_type):
                errors.append(f"input-type-missing:{symbol}:{name}:{comfy_type}")
    outputs = node.get("outputs", [])
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        return tuple(errors + [f"invalid-outputs:{symbol}"])
    if node.get("output_count") != len(outputs):
        errors.append(f"output-count-mismatch:{symbol}")
    for expected_index, output in enumerate(outputs):
        if not isinstance(output, Mapping):
            errors.append(f"invalid-output:{symbol}:{expected_index}")
            continue
        name = str(output.get("name", ""))
        comfy_type = str(output.get("type", ""))
        if output.get("index") != expected_index:
            errors.append(f"output-index-mismatch:{symbol}:{expected_index}")
        if not _quoted_name_present(bounded_text, name):
            errors.append(f"output-name-missing:{symbol}:{name}")
        if not _type_present(class_block, bounded_text, comfy_type):
            errors.append(f"output-type-missing:{symbol}:{name}:{comfy_type}")
    return tuple(errors)


def check_manifest_against_reference(
    manifest_path: Path, reference_root: Path
) -> tuple[str, ...]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = reference_root.resolve()
    errors: list[str] = []
    repositories: dict[str, Path] = {}
    for repo_name, (origin, revision, expected_status) in PINNED_REPOSITORIES.items():
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
        except (OSError, subprocess.CalledProcessError):
            errors.append(f"repository-metadata-read-failed:{repo_name}")
            continue
        if actual_origin != origin:
            errors.append(f"origin-mismatch:{repo_name}")
        if actual_revision != revision:
            errors.append(f"revision-mismatch:{repo_name}")
        if actual_status != expected_status:
            errors.append(f"status-mismatch:{repo_name}")

    families = manifest.get("families", [])
    if not isinstance(families, Sequence) or isinstance(families, (str, bytes)):
        return tuple(errors + ["manifest-families-invalid"])
    for family in families:
        if not isinstance(family, Mapping):
            errors.append("manifest-family-invalid")
            continue
        family_id = str(family.get("family", ""))
        repo_name = FAMILY_REPOSITORIES.get(family_id)
        if repo_name is None:
            errors.append(f"family-repository-unknown:{family_id}")
            continue
        expected = PINNED_REPOSITORIES[repo_name]
        if family.get("origin") != expected[0] or family.get("revision") != expected[1]:
            errors.append(f"family-provenance-mismatch:{family_id}")
        for node in family.get("nodes", []):
            if not isinstance(node, Mapping):
                errors.append(f"node-invalid:{family_id}")
                continue
            evidence = node.get("evidence", {})
            if not isinstance(evidence, Mapping):
                errors.append(f"evidence-invalid:{family_id}")
                continue
            repo = repositories.get(repo_name)
            if repo is None:
                errors.append(f"family-repository-unavailable:{family_id}")
                break
            relative_source = Path(repo_name) / str(evidence.get("path", ""))
            try:
                # SECURITY: keep every nested source anchored to the top-level reference root.
                source_path = _contained_path(root, str(relative_source))
            except (OSError, ValueError):
                errors.append(f"source-containment-failed:{family_id}:{node.get('class', '')}")
                continue
            if not source_path.is_file():
                errors.append(f"source-missing:{family_id}:{node.get('class', '')}")
                continue
            try:
                source_text = source_path.read_text(encoding="utf-8")
            except OSError:
                errors.append(f"source-read-failed:{family_id}:{node.get('class', '')}")
                continue
            try:
                evidence_text = _evidence_slice(
                    source_text, str(evidence.get("lines", ""))
                )
            except ValueError:
                errors.append(f"evidence-lines-invalid:{family_id}:{node.get('class', '')}")
                continue
            errors.extend(validate_node_source(node, source_text, evidence_text))
    return tuple(errors)


def _class_block(source_text: str, symbol: str) -> str | None:
    pattern = re.compile(rf"^class\s+{re.escape(symbol)}\b", re.MULTILINE)
    match = pattern.search(source_text)
    if match is None:
        return None
    next_match = re.compile(r"^class\s+[A-Za-z_]\w*\b", re.MULTILINE).search(
        source_text, match.end()
    )
    return source_text[match.start() : next_match.start() if next_match else len(source_text)]


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


def _type_present(class_block: str, source_text: str, comfy_type: str) -> bool:
    alternatives = comfy_type.split("/")
    for alternative in alternatives:
        markers = TYPE_MARKERS.get(alternative, (f'"{alternative}"', f"'{alternative}'"))
        if any(marker in class_block or marker in source_text for marker in markers):
            return True
    return False


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
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "workflow_contracts"
        / "wanvideo_host_contracts.v1.json",
    )
    args = parser.parse_args()
    errors = check_manifest_against_reference(args.manifest, args.reference_root)
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    node_count = sum(len(family["nodes"]) for family in manifest["families"])
    print(
        "PASS repositories=5 families="
        f"{len(manifest['families'])} nodes={node_count} mode=text-only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
