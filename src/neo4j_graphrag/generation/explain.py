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

from typing import TYPE_CHECKING, Any, Union

from pydantic import BaseModel, Field, PositiveInt

if TYPE_CHECKING:
    from neo4j_graphrag.types import RetrieverResult

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
    return TraceStep(retriever=str(retriever))


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
