#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Union

import neo4j
from pydantic import BaseModel, Field, PositiveInt

if TYPE_CHECKING:
    from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem

GraphPathElement = Union["GraphNodeRef", "GraphRelationshipRef"]
GraphPath = list[GraphPathElement]


class ExplainConfig(BaseModel):
    """Configuration for GraphRAG explainability output."""

    cite_sources: bool = True


class GraphNodeRef(BaseModel):
    """A node referenced in explainability graph context."""

    id: str | None = None
    labels: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphRelationshipRef(BaseModel):
    """A relationship referenced in explainability graph context."""

    type: str
    start_id: str | None = None
    end_id: str | None = None


class GraphContext(BaseModel):
    """Graph neighborhood for a single retrieved source."""

    seed_node: GraphNodeRef | None = None
    related_nodes: list[GraphNodeRef] = Field(default_factory=list)
    relationships: list[GraphRelationshipRef] = Field(default_factory=list)
    paths: list[GraphPath] = Field(default_factory=list)


class SourceRef(BaseModel):
    """A single source item aligned with LLM context numbering."""

    index: PositiveInt
    content: str
    score: float | None = None
    node_id: str | None = None
    labels: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    """Minimal execution trace for a GraphRAG search."""

    retriever: str
    cypher: str | None = None


class ExplainResult(BaseModel):
    """Structured provenance for a GraphRAG answer."""

    sources: list[SourceRef]
    trace: TraceStep
    graph: list[GraphContext] | None = None


CITATION_SYSTEM_INSTRUCTIONS = (
    "Answer the user question using only the provided context. "
    "Cite sources inline using [1], [2], etc. matching the context numbering."
)


_SOURCE_METADATA_KEYS = frozenset(
    {"score", "id", "node_id", "element_id", "nodeLabels", "labels", "graph", "paths"}
)


def trace_from_retriever(retriever_result: RetrieverResult) -> TraceStep:
    from neo4j_graphrag.types import RetrieverResult as RetrieverResultType

    if not isinstance(retriever_result, RetrieverResultType):
        raise TypeError("retriever_result must be a RetrieverResult")
    metadata = retriever_result.metadata or {}
    retriever = metadata.get("__retriever", "unknown")
    cypher = metadata.get("cypher")
    return TraceStep(
        retriever=str(retriever),
        cypher=str(cypher) if cypher is not None else None,
    )


def sources_from_retriever(retriever_result: RetrieverResult) -> list[SourceRef]:
    from neo4j_graphrag.types import RetrieverResult as RetrieverResultType

    if not isinstance(retriever_result, RetrieverResultType):
        raise TypeError("retriever_result must be a RetrieverResult")

    sources: list[SourceRef] = []
    for index, item in enumerate(retriever_result.items, start=1):
        metadata = item.metadata or {}
        labels = metadata.get("labels") or metadata.get("nodeLabels") or []
        if not isinstance(labels, list):
            labels = [str(labels)]
        node_id = (
            metadata.get("node_id") or metadata.get("id") or metadata.get("element_id")
        )
        extra_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in _SOURCE_METADATA_KEYS
        }
        sources.append(
            SourceRef(
                index=index,
                content=str(item.content),
                score=_coerce_float(metadata.get("score")),
                node_id=str(node_id) if node_id is not None else None,
                labels=labels,
                metadata=extra_metadata,
            )
        )
    return sources


def graph_from_retriever(
    retriever_result: RetrieverResult,
) -> list[GraphContext] | None:
    from neo4j_graphrag.types import RetrieverResult as RetrieverResultType

    if not isinstance(retriever_result, RetrieverResultType):
        raise TypeError("retriever_result must be a RetrieverResult")

    contexts: list[GraphContext] = []
    for item in retriever_result.items:
        context = _graph_context_from_item_metadata(item.metadata)
        if context is not None:
            contexts.append(context)
    return contexts or None


def build_explain_result(retriever_result: RetrieverResult) -> ExplainResult:
    return ExplainResult(
        sources=sources_from_retriever(retriever_result),
        trace=trace_from_retriever(retriever_result),
        graph=graph_from_retriever(retriever_result),
    )


def format_retrieval_context(
    retriever_result: RetrieverResult,
    cite_sources: bool,
) -> str:
    from neo4j_graphrag.types import RetrieverResult as RetrieverResultType

    if not isinstance(retriever_result, RetrieverResultType):
        raise TypeError("retriever_result must be a RetrieverResult")
    if not cite_sources:
        return "\n".join(str(item.content) for item in retriever_result.items)
    return "\n".join(
        f"[{index}] {item.content}"
        for index, item in enumerate(retriever_result.items, start=1)
    )


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _graph_context_from_item_metadata(
    metadata: dict[str, Any] | None,
) -> GraphContext | None:
    if not metadata:
        return None

    graph_data = metadata.get("graph")
    if isinstance(graph_data, GraphContext):
        return graph_data
    if isinstance(graph_data, dict):
        return _graph_context_from_dict(graph_data, metadata.get("paths"))

    paths_data = metadata.get("paths")
    if paths_data:
        return GraphContext(paths=_parse_paths(paths_data))
    return None


def _graph_context_from_dict(
    data: dict[str, Any],
    extra_paths: Any = None,
) -> GraphContext:
    seed = data.get("seed_node")
    paths_data = data.get("paths", extra_paths)
    return GraphContext(
        seed_node=_parse_node_ref(seed) if seed else None,
        related_nodes=[
            _parse_node_ref(node)
            for node in data.get("related_nodes", [])
            if node is not None
        ],
        relationships=[
            _parse_relationship_ref(rel)
            for rel in data.get("relationships", [])
            if rel is not None
        ],
        paths=_parse_paths(paths_data),
    )


def _parse_paths(paths_data: Any) -> list[GraphPath]:
    if not paths_data:
        return []
    if not isinstance(paths_data, list):
        return []
    paths: list[GraphPath] = []
    for path in paths_data:
        if not isinstance(path, list):
            continue
        elements = [_parse_path_element(element) for element in path]
        paths.append(elements)
    return paths


def _parse_path_element(element: Any) -> GraphPathElement:
    if isinstance(element, GraphNodeRef):
        return element
    if isinstance(element, GraphRelationshipRef):
        return element
    if not isinstance(element, dict):
        raise ValueError("path element must be a mapping or graph ref model")
    if _is_relationship_mapping(element):
        return _parse_relationship_ref(element)
    return _parse_node_ref(element)


def _is_relationship_mapping(data: dict[str, Any]) -> bool:
    return "type" in data and "labels" not in data and "properties" not in data


def _parse_node_ref(data: Any) -> GraphNodeRef:
    if isinstance(data, GraphNodeRef):
        return data
    if not isinstance(data, dict):
        raise ValueError("node reference must be a mapping or GraphNodeRef")
    labels = data.get("labels", [])
    if not isinstance(labels, list):
        labels = [str(labels)]
    properties = data.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    node_id = data.get("id")
    return GraphNodeRef(
        id=str(node_id) if node_id is not None else None,
        labels=labels,
        properties=properties,
    )


def _parse_relationship_ref(data: Any) -> GraphRelationshipRef:
    if isinstance(data, GraphRelationshipRef):
        return data
    if not isinstance(data, dict):
        raise ValueError(
            "relationship reference must be a mapping or GraphRelationshipRef"
        )
    rel_type = data.get("type")
    if not rel_type:
        raise ValueError("relationship reference requires a type")
    start_id = data.get("start_id")
    end_id = data.get("end_id")
    return GraphRelationshipRef(
        type=str(rel_type),
        start_id=str(start_id) if start_id is not None else None,
        end_id=str(end_id) if end_id is not None else None,
    )


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    iso_format = getattr(value, "iso_format", None)
    if callable(iso_format):
        return iso_format()
    return str(value)


def _node_from_neo4j_graph_node(node: neo4j.graph.Node) -> GraphNodeRef:
    labels = list(node.labels)
    properties = _json_safe_value(dict(node.items()))
    if not isinstance(properties, dict):
        properties = {}
    return GraphNodeRef(
        id=node.element_id,
        labels=labels,
        properties=properties,
    )


def node_from_neo4j_graph_node(node: neo4j.graph.Node) -> GraphNodeRef:
    """Serialize a Neo4j driver node for explainability metadata."""
    return _node_from_neo4j_graph_node(node)


def _relationship_from_neo4j_graph_rel(
    relationship: neo4j.graph.Relationship,
) -> GraphRelationshipRef:
    start_node = relationship.start_node
    end_node = relationship.end_node
    return GraphRelationshipRef(
        type=relationship.type,
        start_id=start_node.element_id if start_node is not None else None,
        end_id=end_node.element_id if end_node is not None else None,
    )


def serialize_neo4j_path(path: neo4j.graph.Path) -> GraphPath:
    elements: GraphPath = []
    nodes = list(path.nodes)
    relationships = list(path.relationships)
    for index, node in enumerate(nodes):
        elements.append(_node_from_neo4j_graph_node(node))
        if index < len(relationships):
            elements.append(_relationship_from_neo4j_graph_rel(relationships[index]))
    return elements


def serialize_paths(paths_value: Any) -> list[GraphPath]:
    if not paths_value or not isinstance(paths_value, list):
        if isinstance(paths_value, neo4j.graph.Path):
            return [serialize_neo4j_path(paths_value)]
        return []
    paths: list[GraphPath] = []
    for path in paths_value:
        if isinstance(path, neo4j.graph.Path):
            paths.append(serialize_neo4j_path(path))
            continue
        if isinstance(path, list):
            parsed = _parse_paths([path])
            if parsed:
                paths.append(parsed[0])
    return paths


_GRAPH_RECORD_KEYS = frozenset({"path", "paths", "graph"})


def _collect_paths_from_record(record: neo4j.Record) -> list[GraphPath]:
    paths: list[GraphPath] = []
    for key in record.keys():
        value = record.get(key)
        if isinstance(value, neo4j.graph.Path):
            paths.append(serialize_neo4j_path(value))
            continue
        if key in _GRAPH_RECORD_KEYS or key.endswith("Path"):
            paths.extend(serialize_paths(value))
            continue
        if isinstance(value, list):
            paths.extend(serialize_paths(value))
    return paths


def _relationship_degree_in_paths(paths: list[GraphPath]) -> dict[str, int]:
    degree: dict[str, int] = {}
    for path in paths:
        for element in path:
            if not isinstance(element, GraphRelationshipRef):
                continue
            for node_id in (element.start_id, element.end_id):
                if node_id is not None:
                    degree[node_id] = degree.get(node_id, 0) + 1
    return degree


def _seed_node_from_record_scalars(record: neo4j.Record) -> GraphNodeRef | None:
    for value in record.values():
        if isinstance(value, neo4j.graph.Node):
            return node_from_neo4j_graph_node(value)

    properties: dict[str, Any] = {}
    for key in record.keys():
        if key in _GRAPH_RECORD_KEYS or key.endswith("Path"):
            continue
        value = record.get(key)
        if value is None:
            continue
        if isinstance(
            value, (neo4j.graph.Path, neo4j.graph.Node, neo4j.graph.Relationship)
        ):
            continue
        if (
            isinstance(value, list)
            and value
            and isinstance(
                value[0], (neo4j.graph.Path, neo4j.graph.Node, neo4j.graph.Relationship)
            )
        ):
            continue
        properties[key] = _json_safe_value(value)

    if not properties:
        return None
    return GraphNodeRef(properties=properties)


def _graph_elements_from_paths(
    paths: list[GraphPath],
) -> tuple[GraphNodeRef | None, list[GraphNodeRef], list[GraphRelationshipRef]]:
    node_order: list[GraphNodeRef] = []
    seen_node_keys: set[str] = set()
    relationships: list[GraphRelationshipRef] = []
    seen_rel_keys: set[tuple[str, str | None, str | None]] = set()

    for path in paths:
        for element in path:
            if isinstance(element, GraphNodeRef):
                node_key = element.id or str(element.properties)
                if node_key not in seen_node_keys:
                    node_order.append(element)
                    seen_node_keys.add(node_key)
                continue
            rel_key = (element.type, element.start_id, element.end_id)
            if rel_key not in seen_rel_keys:
                relationships.append(element)
                seen_rel_keys.add(rel_key)

    seed_node: GraphNodeRef | None = None
    if node_order:
        degree = _relationship_degree_in_paths(paths)
        seed_node = max(
            node_order,
            key=lambda node: (
                degree.get(node.id or "", 0),
                -node_order.index(node),
            ),
        )

    if seed_node is None:
        related_nodes = node_order
    else:
        seed_key = seed_node.id or str(seed_node.properties)
        related_nodes = [
            node for node in node_order if (node.id or str(node.properties)) != seed_key
        ]

    return seed_node, related_nodes, relationships


def graph_context_from_neo4j_record(record: neo4j.Record) -> dict[str, Any] | None:
    """Build metadata.graph from Neo4j paths or nodes returned by Text2Cypher."""
    paths = _collect_paths_from_record(record)
    if paths:
        seed_node, related_nodes, relationships = _graph_elements_from_paths(paths)
        if seed_node is None:
            seed_node = _seed_node_from_record_scalars(record)
        return GraphContext(
            seed_node=seed_node,
            related_nodes=related_nodes,
            relationships=relationships,
            paths=paths,
        ).model_dump(exclude_none=True)

    seed_node = _seed_node_from_record_scalars(record)
    if seed_node is None:
        return None
    return GraphContext(seed_node=seed_node).model_dump(exclude_none=True)


def vector_cypher_explain_result_formatter(
    record: neo4j.Record,
    *,
    content: str,
    score_key: str = "similarityScore",
    graph_builder: Callable[[neo4j.Record], dict[str, Any] | None] | None = None,
) -> RetrieverResultItem:
    from neo4j_graphrag.types import RetrieverResultItem

    metadata: dict[str, Any] = {"score": record.get(score_key)}
    if graph_builder is not None:
        graph = graph_builder(record)
        if graph:
            metadata["graph"] = graph
    return RetrieverResultItem(
        content=content,
        metadata=metadata,
    )


def text2cypher_explain_result_formatter(
    record: neo4j.Record,
    *,
    graph_builder: Callable[[neo4j.Record], dict[str, Any] | None] | None = None,
) -> RetrieverResultItem:
    from neo4j_graphrag.types import RetrieverResultItem

    parts: list[str] = []
    for key in record.keys():
        if key in _GRAPH_RECORD_KEYS:
            continue
        value = record.get(key)
        if value is None:
            continue
        if isinstance(
            value, (neo4j.graph.Path, neo4j.graph.Node, neo4j.graph.Relationship)
        ):
            continue
        if (
            isinstance(value, list)
            and value
            and isinstance(
                value[0], (neo4j.graph.Path, neo4j.graph.Node, neo4j.graph.Relationship)
            )
        ):
            continue
        parts.append(f"{key}: {value}")

    metadata: dict[str, Any] = {}
    graph = (graph_builder or graph_context_from_neo4j_record)(record)
    if graph:
        metadata["graph"] = graph

    return RetrieverResultItem(
        content=", ".join(parts) if parts else str(record),
        metadata=metadata,
    )
