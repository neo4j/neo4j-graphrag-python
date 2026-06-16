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

from typing import Any, Union

from pydantic import BaseModel, Field, PositiveInt

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
