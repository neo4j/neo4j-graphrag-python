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

from pydantic import BaseModel

from neo4j_genai.pipeline.component import DataModel


class TextChunk(BaseModel):
    """A chunk of text split from a document by a text splitter.

    Attributes:
        text (str): The raw chunk text.
        metadata (Optional[dict[str, Any]]): Metadata associated with this chunk such as the id of the next chunk in the original document.
    """

    text: str
    metadata: Optional[dict[str, Any]] = None


class TextChunks(DataModel):
    """A collection of text chunks returned from a text splitter.

    Attributes:
        chunks (list[TextChunk]): A list of text chunks.
    """

    chunks: list[TextChunk]


class Neo4jProperty(BaseModel):
    """Represents a Neo4j property.

    Attributes:
        key (str): The property name.
        value (Any): The property value.
    """

    key: str
    value: Any


class Neo4jEmbeddingProperty(BaseModel):
    """Represents a Neo4j embedding property.

    Attributes:
        key (str): The property name.
        value (list[float]): The embedding vector.
    """

    key: str
    value: list[float]


class Neo4jNode(BaseModel):
    """Represents a Neo4j node.

    Attributes:
        id (str): The ID of the node.
        label (str): The label of the node.
        properties (Optional[list[Neo4jProperty]]): A list of properties associated with the node.
        embedding_properties (Optional[list[Neo4jEmbeddingProperty]]): A list of embedding properties associated with the node.
    """

    id: str
    label: str
    properties: Optional[list[Neo4jProperty]] = None
    embedding_properties: Optional[list[Neo4jEmbeddingProperty]] = None


class Neo4jRelationship(BaseModel):
    """Represents a Neo4j relationship.

    Attributes:
        start_node_id (str): The ID of the start node.
        end_node_id (str): The ID of the end node.
        type (str): The relationship type.
        properties (Optional[list[Neo4jProperty]]): A list of properties associated with the relationship.
        embedding_properties (Optional[list[Neo4jEmbeddingProperty]]): A list of embedding properties associated with the relationship.
    """

    start_node_id: str
    end_node_id: str
    type: str
    properties: Optional[list[Neo4jProperty]] = None
    embedding_properties: Optional[list[Neo4jEmbeddingProperty]] = None


class Neo4jGraph(BaseModel):
    """Represents a Neo4j graph.

    Attributes:
        nodes (list[Neo4jNode]): A list of nodes in the graph.
        relationships (list[Neo4jRelationship]): A list of relationships in the graph.
    """

    nodes: list[Neo4jNode]
    relationships: list[Neo4jRelationship]
