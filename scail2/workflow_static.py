"""Static workflow inspection helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


RENDER_NLF_CLASS_TYPE = "RenderNLFPoses"


@dataclass(frozen=True)
class RenderNLFNodeConnection:
    node_id: str
    bboxes_connected: bool
    pose_video_mask_connected: bool


@dataclass(frozen=True)
class RenderNLFConnectionDiagnostics:
    render_nodes: tuple[RenderNLFNodeConnection, ...]

    @property
    def render_node_count(self) -> int:
        return len(self.render_nodes)

    @property
    def render_node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.render_nodes)

    @property
    def bboxes_connected(self) -> bool:
        return bool(self.render_nodes) and all(
            node.bboxes_connected for node in self.render_nodes
        )

    @property
    def pose_video_mask_connected(self) -> bool:
        return bool(self.render_nodes) and all(
            node.pose_video_mask_connected for node in self.render_nodes
        )

    def summary(self) -> str:
        return (
            "render_nlf_workflow_connections "
            f"render_nodes={self.render_node_count} "
            f"render_node_ids={','.join(self.render_node_ids)} "
            f"bboxes_connected={self.bboxes_connected} "
            f"pose_video_mask_connected={self.pose_video_mask_connected}"
        )


def diagnose_render_nlf_connections(workflow: Any) -> RenderNLFConnectionDiagnostics:
    """Report whether exported workflow metadata wires RenderNLFPoses repair inputs."""

    if not isinstance(workflow, Mapping):
        return RenderNLFConnectionDiagnostics(render_nodes=())

    nodes = workflow.get("nodes", ())
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        return RenderNLFConnectionDiagnostics(render_nodes=())

    graph_links = _target_socket_links(workflow.get("links", ()))
    render_nodes: list[RenderNLFNodeConnection] = []
    for node in nodes:
        if not _is_render_nlf_node(node):
            continue
        node_id = _node_id(node)
        render_nodes.append(
            RenderNLFNodeConnection(
                node_id=node_id,
                bboxes_connected=_socket_connected(node, "bboxes", graph_links),
                pose_video_mask_connected=_socket_connected(
                    node,
                    "pose_video_mask",
                    graph_links,
                ),
            )
        )

    return RenderNLFConnectionDiagnostics(render_nodes=tuple(render_nodes))


def _is_render_nlf_node(node: Any) -> bool:
    if not isinstance(node, Mapping):
        return False
    return (
        node.get("type") == RENDER_NLF_CLASS_TYPE
        or node.get("class_type") == RENDER_NLF_CLASS_TYPE
    )


def _node_id(node: Mapping[str, Any]) -> str:
    value = node.get("id")
    if value is None:
        value = node.get("name", "")
    return str(value)


def _socket_connected(
    node: Mapping[str, Any],
    input_name: str,
    graph_links: frozenset[tuple[str, str]],
) -> bool:
    node_id = _node_id(node)
    if (node_id, input_name) in graph_links:
        return True

    inputs = node.get("inputs")
    if isinstance(inputs, Mapping):
        return _mapping_input_connected(inputs.get(input_name))
    if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
        for item in inputs:
            if isinstance(item, Mapping) and item.get("name") == input_name:
                return _input_link_is_connected(item.get("link"))
    return False


def _mapping_input_connected(input_spec: Any) -> bool:
    if isinstance(input_spec, Mapping):
        return _input_link_is_connected(input_spec.get("link"))
    return _input_link_is_connected(input_spec)


def _input_link_is_connected(value: Any) -> bool:
    return value is not None and value is not False and value != ""


def _target_socket_links(links: Any) -> frozenset[tuple[str, str]]:
    if not isinstance(links, Sequence) or isinstance(links, (str, bytes)):
        return frozenset()

    targets: set[tuple[str, str]] = set()
    for link in links:
        if not isinstance(link, Mapping):
            continue
        target = link.get("to")
        # IMPORTANT: support both this repo's skeleton links (`to`) and exported
        # workflow-style target fields; dropping either weakens static checks.
        if (
            isinstance(target, Sequence)
            and not isinstance(target, (str, bytes))
            and len(target) >= 2
        ):
            targets.add((str(target[0]), str(target[1])))
            continue
        target_node = link.get("target_node_id", link.get("to_node"))
        target_socket = link.get("target_input", link.get("to_input"))
        if target_node is not None and target_socket is not None:
            targets.add((str(target_node), str(target_socket)))
    return frozenset(targets)
