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
import asyncio
import logging
from typing import Any, Dict, Optional

from pydantic import validate_call

from neo4j_graphrag.experimental.components.pdf_loader import DocumentInfo
from neo4j_graphrag.experimental.components.types import (
    GraphResult,
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.experimental.pipeline import Component

logger = logging.getLogger(__name__)


class LexicalGraphBuilder(Component):
    """Builds the lexical graph to be inserted into neo4j.
    The lexical graph contains:
    - A node for each document
    - A node for each chunk
    - A relationship between each chunk and the document it was created from
    - A relationship between a chunk and the next one in the document
    """

    def __init__(
        self,
        config: LexicalGraphConfig = LexicalGraphConfig(),
    ):
        self.config = config

    @validate_call
    async def run(
        self,
        text_chunks: TextChunks,
        document_info: Optional[DocumentInfo] = None,
    ) -> GraphResult:
        if document_info is None:
            logger.info(
                "Document node not created in the lexical graph "
                "because no document metadata is provided"
            )
        graph = Neo4jGraph()
        document_id = None
        if document_info:
            document_node = self.create_document_node(document_info)
            graph.nodes.append(document_node)
            document_id = document_node.id
        tasks = [
            self.process_chunk(graph, chunk, document_id)
            for chunk in text_chunks.chunks
        ]
        await asyncio.gather(*tasks)
        return GraphResult(
            config=self.config,
            graph=graph,
        )

    def chunk_id(self, chunk_index: int) -> str:
        return f"{self.config.id_prefix}:{chunk_index}"

    async def process_chunk(
        self,
        graph: Neo4jGraph,
        chunk: TextChunk,
        document_id: Optional[str] = None,
    ) -> None:
        """Add chunks and relationships between them (NEXT_CHUNK)

        Updates `graph` in place.
        """
        if document_id:
            chunk_to_doc_rel = self.create_chunk_to_document_rel(chunk, document_id)
            graph.relationships.append(chunk_to_doc_rel)
        chunk_node = self.create_chunk_node(chunk)
        graph.nodes.append(chunk_node)
        if chunk.index > 0:
            next_chunk_rel = self.create_next_chunk_relationship(chunk)
            graph.relationships.append(next_chunk_rel)

    def create_document_node(self, document_info: DocumentInfo) -> Neo4jNode:
        """Create a Document node with 'path' property. Any document metadata is also
        added as a node property.
        """
        document_metadata = document_info.metadata or {}
        return Neo4jNode(
            id=document_info.path,
            label=self.config.document_node_label,
            properties={
                "path": document_info.path,
                **document_metadata,
            },
        )

    def create_chunk_node(self, chunk: TextChunk) -> Neo4jNode:
        """Create chunk node with properties 'text', 'index' and any 'metadata'
        added during the process. Special case for the potential chunk embedding
        property that gets added as an embedding_property"""
        chunk_id = self.chunk_id(chunk.index)
        chunk_properties: Dict[str, Any] = {
            self.config.chunk_text_property: chunk.text,
            self.config.chunk_index_property: chunk.index,
        }
        embedding_properties = {}
        if chunk.metadata:
            if "embedding" in chunk.metadata:
                embedding_properties[self.config.chunk_embedding_property] = (
                    chunk.metadata.pop("embedding")
                )
            chunk_properties.update(chunk.metadata)
        return Neo4jNode(
            id=chunk_id,
            label=self.config.chunk_node_label,
            properties=chunk_properties,
            embedding_properties=embedding_properties,
        )

    def create_chunk_to_document_rel(
        self, chunk: TextChunk, document_id: str
    ) -> Neo4jRelationship:
        """Create the relationship between a chunk and the document it belongs to."""
        chunk_id = self.chunk_id(chunk.index)
        return Neo4jRelationship(
            start_node_id=chunk_id,
            end_node_id=document_id,
            type=self.config.chunk_to_document_relationship_type,
        )

    def create_next_chunk_relationship(
        self,
        chunk: TextChunk,
    ) -> Neo4jRelationship:
        """Create relationship between a chunk and the next one"""
        chunk_id = self.chunk_id(chunk.index)
        previous_chunk_id = self.chunk_id(chunk.index - 1)
        return Neo4jRelationship(
            type=self.config.next_chunk_relationship_type,
            start_node_id=previous_chunk_id,
            end_node_id=chunk_id,
        )

    def create_node_to_chunk_rel(
        self, node: Neo4jNode, chunk_id: str
    ) -> Neo4jRelationship:
        """Create relationship between a chunk and entities found in that chunk"""
        return Neo4jRelationship(
            start_node_id=node.id,
            end_node_id=chunk_id,
            type=self.config.node_to_chunk_relationship_type,
        )

    async def process_chunk_extracted_entities(
        self,
        chunk_graph: Neo4jGraph,
        chunk: TextChunk,
    ) -> None:
        """Create relationship between Chunk and each entity
        extracted from it.

        Updates `chunk_graph` in place.
        """
        for node in chunk_graph.nodes:
            if node.label in (
                self.config.chunk_node_label,
                self.config.document_node_label,
            ):
                continue
            if chunk.metadata and (id := chunk.metadata.get("id")):
                chunk_id = id
            else:
                chunk_id = self.chunk_id(chunk.index)
            node_to_chunk_rel = self.create_node_to_chunk_rel(node, chunk_id)
            chunk_graph.relationships.append(node_to_chunk_rel)
