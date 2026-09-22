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
import datetime
import logging
from typing import Any, Dict, Optional

from pydantic import validate_call

from neo4j_graphrag.components.types import (
    DocumentInfo,
    GraphResult,
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
    PropertyValue,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.components.base import Component

logger = logging.getLogger(__name__)


class LexicalGraphBuilder(Component):
    """Builds the lexical graph to be inserted into neo4j.
    The lexical graph contains:
    - A node for each document
    - A node for each chunk
    - A relationship between each chunk and the document it was created from
    - A relationship between a chunk and the next one in the document
    """

    @validate_call
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
        if document_info:
            document_node = self._create_document_node(document_info)
            graph.nodes.append(document_node)
        if len(text_chunks.chunks) > 0:
            # Ensure chunks are linked so that NEXT_CHUNK relationships can be
            # created from each chunk's prev_chunk_id alone.
            prev_chunk_id: Optional[str] = None
            for chunk in text_chunks.chunks:
                if chunk.prev_chunk_id is None:
                    chunk.prev_chunk_id = prev_chunk_id
                prev_chunk_id = chunk.chunk_id
            for chunk in text_chunks.chunks:
                self._add_chunk_to_graph(graph, chunk, document_info)
        return GraphResult(
            config=self.config,
            graph=graph,
        )

    def run_for_chunk(
        self, chunk: TextChunk, document_info: Optional[DocumentInfo] = None
    ) -> Neo4jGraph:
        """Run the lexical graph builder for a single chunk.

        The returned graph contains the chunk node, the document node (when
        document metadata is provided) and the chunk's relationships
        (FROM_DOCUMENT and NEXT_CHUNK from the previous chunk when
        ``chunk.prev_chunk_id`` is set).
        """
        chunk_node = self._create_chunk_node(chunk)
        document_node = (
            self._create_document_node(document_info) if document_info else None
        )
        chunk_to_document_rel = (
            self._create_chunk_to_document_rel(chunk, document_info)
            if document_info
            else None
        )
        next_chunk_rel = (
            self._create_next_chunk_relationship(chunk.prev_chunk_id, chunk.chunk_id)
            if chunk.prev_chunk_id is not None
            else None
        )
        return Neo4jGraph(
            nodes=[node for node in (document_node, chunk_node) if node is not None],
            relationships=[
                rel
                for rel in (chunk_to_document_rel, next_chunk_rel)
                if rel is not None
            ],
        )

    def combine_graphs(self, graph1: Neo4jGraph, graph2: Neo4jGraph) -> Neo4jGraph:
        """Combine two graphs into a new one, deduplicating nodes by id and
        relationships by (start_node_id, end_node_id, type).

        Order is preserved and, on duplicates, the node or relationship from
        `graph1` wins. Inputs are left unchanged.
        """
        node_ids: set[str] = set()
        nodes: list[Neo4jNode] = []
        for node in graph1.nodes + graph2.nodes:
            if node.id not in node_ids:
                node_ids.add(node.id)
                nodes.append(node)
        rel_ids: set[tuple[str, str, str]] = set()
        relationships: list[Neo4jRelationship] = []
        for rel in graph1.relationships + graph2.relationships:
            rel_id = (rel.start_node_id, rel.end_node_id, rel.type)
            if rel_id not in rel_ids:
                rel_ids.add(rel_id)
                relationships.append(rel)
        return Neo4jGraph(nodes=nodes, relationships=relationships)

    def _add_chunk_to_graph(
        self,
        graph: Neo4jGraph,
        chunk: TextChunk,
        document_info: Optional[DocumentInfo] = None,
    ) -> None:
        """Add the chunk node and its relationships (FROM_DOCUMENT and
        NEXT_CHUNK from the previous chunk) to the graph.

        Only requires the current chunk: the NEXT_CHUNK relationship is
        created from ``chunk.prev_chunk_id`` when set.

        Updates `graph` in place.
        """
        chunk_graph = self.run_for_chunk(chunk, document_info)
        if document_info:
            chunk_graph.nodes = [
                node
                for node in chunk_graph.nodes
                if node.id != document_info.document_id
            ]
        combined = self.combine_graphs(graph, chunk_graph)
        graph.nodes[:] = combined.nodes
        graph.relationships[:] = combined.relationships

    def _create_document_node(self, document_info: DocumentInfo) -> Neo4jNode:
        """Create a Document node with 'path' property. Any document metadata is also
        added as a node property.
        """
        document_metadata = document_info.metadata or {}
        properties: dict[str, PropertyValue] = {
            "path": document_info.path,
            "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **document_metadata,
        }
        # Only add document_type if it's not None
        if document_info.document_type is not None:
            properties["document_type"] = document_info.document_type.value

        return Neo4jNode(
            id=document_info.document_id,
            label=self.config.document_node_label,
            properties=properties,
        )

    def _create_chunk_node(
        self,
        chunk: TextChunk,
    ) -> Neo4jNode:
        """Create chunk node with properties 'text', 'index' and any 'metadata'
        added during the process. Special case for the potential chunk embedding
        property that gets added as an embedding_property"""
        chunk_id = chunk.chunk_id
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

    def _create_chunk_to_document_rel(
        self,
        chunk: TextChunk,
        document_info: DocumentInfo,
    ) -> Neo4jRelationship:
        """Create the relationship between a chunk and the document it belongs to."""
        return Neo4jRelationship(
            start_node_id=chunk.chunk_id,
            end_node_id=document_info.document_id,
            type=self.config.chunk_to_document_relationship_type,
        )

    def _create_next_chunk_relationship(
        self,
        prev_chunk_id: str,
        chunk_id: str,
    ) -> Neo4jRelationship:
        """Create relationship between a chunk and the previous one"""
        return Neo4jRelationship(
            type=self.config.next_chunk_relationship_type,
            start_node_id=prev_chunk_id,
            end_node_id=chunk_id,
        )

    def _create_node_to_chunk_rel(
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
        """
        Create relationships between `TextChunk` and each entity
        extracted from it.

        Updates `chunk_graph` in place.
        """
        for node in chunk_graph.nodes:
            if node.label in (
                self.config.chunk_node_label,
                self.config.document_node_label,
            ):
                continue
            node_to_chunk_rel = self._create_node_to_chunk_rel(node, chunk.chunk_id)
            chunk_graph.relationships.append(node_to_chunk_rel)
