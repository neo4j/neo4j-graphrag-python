import asyncio
from typing import Any, Dict, Optional

from pydantic import validate_call

from neo4j_genai.experimental.components.constants import (
    CHUNK_NODE_LABEL,
    CHUNK_TO_DOCUMENT_RELATIONSHIP_TYPE,
    DOCUMENT_NODE_LABEL,
    NEXT_CHUNK_RELATIONSHIP_TYPE,
)
from neo4j_genai.experimental.components.pdf_loader import DocumentInfo
from neo4j_genai.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
    TextChunk,
    TextChunks,
)
from neo4j_genai.experimental.pipeline import Component


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
        id_prefix: str,
        document_node_label: str = DOCUMENT_NODE_LABEL,
        chunk_node_label: str = CHUNK_NODE_LABEL,
        chunk_to_document_relationship_type: str = CHUNK_TO_DOCUMENT_RELATIONSHIP_TYPE,
        next_chunk_relationship_type: str = NEXT_CHUNK_RELATIONSHIP_TYPE,
    ):
        self.id_prefix = id_prefix
        self.document_node_label = document_node_label
        self.chunk_node_label = chunk_node_label
        self.document_to_chunk_relationship_type = chunk_to_document_relationship_type
        self.next_chunk_relationship_type = next_chunk_relationship_type

    @validate_call
    async def run(
        self, text_chunks: TextChunks, document_info: Optional[DocumentInfo] = None
    ) -> Neo4jGraph:
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
        print(graph)
        return graph

    async def process_chunk(
        self,
        graph: Neo4jGraph,
        chunk: TextChunk,
        document_id: Optional[str] = None,
    ) -> None:
        """Add chunks and relationships between them (NEXT_CHUNK)"""
        chunk_id = f"{self.id_prefix}:{chunk.index}"
        if document_id:
            chunk_to_doc_rel = self.create_chunk_to_document_rel(chunk_id, document_id)
            graph.relationships.append(chunk_to_doc_rel)
        chunk_node = self.create_chunk_node(chunk, chunk_id)
        graph.nodes.append(chunk_node)
        if chunk.index > 0:
            previous_chunk_id = f"{self.id_prefix}:{chunk.index - 1}"
            next_chunk_rel = self.create_next_chunk_relationship(
                previous_chunk_id, chunk_id
            )
            graph.relationships.append(next_chunk_rel)

    def create_document_node(self, document_info: DocumentInfo) -> Neo4jNode:
        """Create a Document node with 'path' property. Any document metadata is also
        added as a node property.
        """
        document_metadata = document_info.metadata or {}
        return Neo4jNode(
            id=document_info.path,
            label=self.document_node_label,
            properties={
                "path": document_info.path,
                **document_metadata,
            },
        )

    def create_chunk_node(self, chunk: TextChunk, chunk_id: str) -> Neo4jNode:
        """Create chunk node with properties 'text', 'index' and any 'metadata' added during
        the process. Special case for the potential chunk embedding property that
        gets added as an embedding_property"""
        chunk_properties: Dict[str, Any] = {
            "text": chunk.text,
            "index": chunk.index,
        }
        embedding_properties = {}
        if chunk.metadata:
            if "embedding" in chunk.metadata:
                embedding_properties["embedding"] = chunk.metadata.pop("embedding")
            chunk_properties.update(chunk.metadata)
        return Neo4jNode(
            id=chunk_id,
            label=self.chunk_node_label,
            properties=chunk_properties,
            embedding_properties=embedding_properties,
        )

    def create_chunk_to_document_rel(
        self, chunk_id: str, document_id: str
    ) -> Neo4jRelationship:
        """Create the relationship between a chunk and the document it belongs to."""
        return Neo4jRelationship(
            start_node_id=chunk_id,
            end_node_id=document_id,
            type=self.document_to_chunk_relationship_type,
        )

    def create_next_chunk_relationship(
        self, previous_chunk_id: str, chunk_id: str
    ) -> Neo4jRelationship:
        """Create relationship between a chunk and the next one"""
        return Neo4jRelationship(
            type=self.next_chunk_relationship_type,
            start_node_id=previous_chunk_id,
            end_node_id=chunk_id,
        )
