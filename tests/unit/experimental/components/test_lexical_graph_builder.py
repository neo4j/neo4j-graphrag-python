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

import pytest
from neo4j_graphrag.experimental.components.lexical_graph import (
    DEFAULT_CHUNK_NODE_LABEL,
    DEFAULT_CHUNK_TO_DOCUMENT_RELATIONSHIP_TYPE,
    DEFAULT_DOCUMENT_NODE_LABEL,
    DEFAULT_NEXT_CHUNK_RELATIONSHIP_TYPE,
    LexicalGraphBuilder,
    LexicalGraphConfig,
    LexicalGraphResult,
)
from neo4j_graphrag.experimental.components.pdf_loader import DocumentInfo
from neo4j_graphrag.experimental.components.types import (
    Neo4jNode,
    TextChunk,
    TextChunks,
)


def test_lexical_graph_builder_create_chunk_node_no_metadata() -> None:
    builder = LexicalGraphBuilder(
        LexicalGraphConfig(id_prefix="test_create_chunk_node_no_metadata")
    )
    node = builder.create_chunk_node(chunk=TextChunk(text="text chunk", index=0))
    assert isinstance(node, Neo4jNode)
    assert node.id == "test_create_chunk_node_no_metadata:0"
    assert node.properties == {"index": 0, "text": "text chunk"}
    assert node.embedding_properties == {}


def test_lexical_graph_builder_create_chunk_node_metadata_no_embedding() -> None:
    builder = LexicalGraphBuilder(
        LexicalGraphConfig(id_prefix="test_create_chunk_node_metadata_no_embedding")
    )
    node = builder.create_chunk_node(
        chunk=TextChunk(text="text chunk", index=0, metadata={"status": "ok"})
    )
    assert isinstance(node, Neo4jNode)
    assert node.id == "test_create_chunk_node_metadata_no_embedding:0"
    assert node.properties == {"index": 0, "text": "text chunk", "status": "ok"}
    assert node.embedding_properties == {}


def test_lexical_graph_builder_create_chunk_node_metadata_embedding() -> None:
    builder = LexicalGraphBuilder(
        LexicalGraphConfig(id_prefix="test_create_chunk_node_metadata_embedding")
    )
    node = builder.create_chunk_node(
        chunk=TextChunk(
            text="text chunk",
            index=0,
            metadata={"status": "ok", "embedding": [1, 2, 3]},
        ),
    )
    assert isinstance(node, Neo4jNode)
    assert node.id == "test_create_chunk_node_metadata_embedding:0"
    assert node.properties == {"index": 0, "text": "text chunk", "status": "ok"}
    assert node.embedding_properties == {"embedding": [1, 2, 3]}


@pytest.mark.asyncio
async def test_lexical_graph_builder_run_with_document() -> None:
    lexical_graph_builder = LexicalGraphBuilder(
        config=LexicalGraphConfig(id_prefix="test")
    )
    result = await lexical_graph_builder.run(
        text_chunks=TextChunks(
            chunks=[
                TextChunk(text="text chunk 1", index=0),
                TextChunk(text="text chunk 1", index=1),
            ]
        ),
        document_info=DocumentInfo(path="test_lexical_graph"),
    )
    assert isinstance(result, LexicalGraphResult)
    graph = result.graph
    nodes = graph.nodes
    assert len(nodes) == 3
    document = nodes[0]
    assert document.id == "test_lexical_graph"
    assert document.label == DEFAULT_DOCUMENT_NODE_LABEL
    assert document.properties == {"path": "test_lexical_graph"}
    chunk1 = nodes[1]
    assert chunk1.label == DEFAULT_CHUNK_NODE_LABEL
    chunk2 = nodes[2]
    assert chunk2.label == DEFAULT_CHUNK_NODE_LABEL
    assert len(graph.relationships) == 3
    for relationship in graph.relationships[:2]:
        assert relationship.type == DEFAULT_CHUNK_TO_DOCUMENT_RELATIONSHIP_TYPE
    assert graph.relationships[2].type == DEFAULT_NEXT_CHUNK_RELATIONSHIP_TYPE


@pytest.mark.asyncio
async def test_lexical_graph_builder_run_no_document() -> None:
    lexical_graph_builder = LexicalGraphBuilder(
        config=LexicalGraphConfig(id_prefix="test")
    )
    result = await lexical_graph_builder.run(
        text_chunks=TextChunks(
            chunks=[
                TextChunk(text="text chunk 1", index=0),
                TextChunk(text="text chunk 1", index=1),
            ]
        ),
    )
    assert isinstance(result, LexicalGraphResult)
    graph = result.graph
    nodes = graph.nodes
    assert len(nodes) == 2
    chunk1 = nodes[0]
    assert chunk1.label == DEFAULT_CHUNK_NODE_LABEL
    chunk2 = nodes[1]
    assert chunk2.label == DEFAULT_CHUNK_NODE_LABEL
    assert len(graph.relationships) == 1
    assert graph.relationships[0].type == DEFAULT_NEXT_CHUNK_RELATIONSHIP_TYPE


@pytest.mark.asyncio
async def test_lexical_graph_builder_run_custom_labels() -> None:
    lexical_graph_builder = LexicalGraphBuilder(
        config=LexicalGraphConfig(
            id_prefix="test",
            document_node_label="Report",
            chunk_node_label="Page",
            chunk_to_document_relationship_type="IN_REPORT",
            next_chunk_relationship_type="NEXT_PAGE",
        ),
    )
    result = await lexical_graph_builder.run(
        text_chunks=TextChunks(
            chunks=[
                TextChunk(text="text chunk 1", index=0),
                TextChunk(text="text chunk 1", index=1),
            ]
        ),
        document_info=DocumentInfo(path="test_lexical_graph"),
    )
    assert isinstance(result, LexicalGraphResult)
    graph = result.graph
    nodes = graph.nodes
    assert len(nodes) == 3
    document = nodes[0]
    assert document.id == "test_lexical_graph"
    assert document.label == "Report"
    assert document.properties == {"path": "test_lexical_graph"}
    chunk1 = nodes[1]
    assert chunk1.label == "Page"
    chunk2 = nodes[2]
    assert chunk2.label == "Page"
    assert len(graph.relationships) == 3
    for relationship in graph.relationships[:2]:
        assert relationship.type == "IN_REPORT"
    assert graph.relationships[2].type == "NEXT_PAGE"
