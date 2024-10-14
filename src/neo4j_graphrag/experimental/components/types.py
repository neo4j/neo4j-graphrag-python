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

from typing import Any, Optional

from pydantic import BaseModel, field_validator

from neo4j_graphrag.experimental.pipeline.component import DataModel


class TextChunk(BaseModel):
    """A chunk of text split from a document by a text splitter.

    Attributes:
        text (str): The raw chunk text.
        index (int): The position of this chunk in the original document.
        metadata (Optional[dict[str, Any]]): Metadata associated with this chunk such as the id of the next chunk in the original document.
    """

    text: str
    index: int
    metadata: Optional[dict[str, Any]] = None


class TextChunks(DataModel):
    """A collection of text chunks returned from a text splitter.

    Attributes:
        chunks (list[TextChunk]): A list of text chunks.
    """

    chunks: list[TextChunk]


class Neo4jNode(BaseModel):
    """Represents a Neo4j node.

    Attributes:
        id (str): The ID of the node.
        label (str): The label of the node.
        properties (dict[str, Any]): A dictionary of properties attached to the node.
        embedding_properties (Optional[dict[str, list[float]]]): A list of embedding properties attached to the node.
    """

    id: str
    label: str
    properties: dict[str, Any] = {}
    embedding_properties: Optional[dict[str, list[float]]] = None

    @field_validator("properties", "embedding_properties")
    @classmethod
    def check_for_id_properties(
        cls, v: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        if v and "id" in v.keys():
            raise TypeError("'id' as a property name is not allowed")
        return v


class Neo4jRelationship(BaseModel):
    """Represents a Neo4j relationship.

    Attributes:
        start_node_id (str): The ID of the start node.
        end_node_id (str): The ID of the end node.
        type (str): The relationship type.
        properties (dict[str, Any]): A dictionary of properties attached to the relationship.
        embedding_properties (Optional[dict[str, list[float]]]): A list of embedding properties attached to the relationship.
    """

    start_node_id: str
    end_node_id: str
    type: str
    properties: dict[str, Any] = {}
    embedding_properties: Optional[dict[str, list[float]]] = None


class Neo4jGraph(DataModel):
    """Represents a Neo4j graph.

    Attributes:
        nodes (list[Neo4jNode]): A list of nodes in the graph.
        relationships (list[Neo4jRelationship]): A list of relationships in the graph.
    """

    nodes: list[Neo4jNode] = []
    relationships: list[Neo4jRelationship] = []


class ResolutionStats(DataModel):
    number_of_nodes_to_resolve: int
    number_of_created_nodes: Optional[int] = None
