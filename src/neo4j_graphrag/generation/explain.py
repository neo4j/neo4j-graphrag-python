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
    """Graph paths for a single retrieved source."""

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

    paths_data = metadata.get("paths")
    if paths_data:
        paths = serialize_paths(paths_data)
        return GraphContext(paths=paths) if paths else None
    return None


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
    if isinstance(paths_value, neo4j.graph.Path):
        return [serialize_neo4j_path(paths_value)]
    if not paths_value or not isinstance(paths_value, list):
        return []
    paths: list[GraphPath] = []
    for path in paths_value:
        if isinstance(path, neo4j.graph.Path):
            paths.append(serialize_neo4j_path(path))
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


def graph_context_from_neo4j_record(record: neo4j.Record) -> GraphContext | None:
    """Build graph context from Neo4j paths returned by Text2Cypher."""
    paths = _collect_paths_from_record(record)
    if not paths:
        return None
    return GraphContext(paths=paths)


def vector_cypher_explain_result_formatter(
    record: neo4j.Record,
    *,
    content: str,
    score_key: str = "similarityScore",
    graph_builder: Callable[[neo4j.Record], GraphContext | None] | None = None,
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
    graph_builder: Callable[[neo4j.Record], GraphContext | None] | None = None,
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
