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

import datetime
import functools
import uuid
from typing import Any
from unittest import mock

import pytest
from neo4j_graphrag.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.components.types import (
    DEFAULT_CHUNK_NODE_LABEL,
    DEFAULT_CHUNK_TO_DOCUMENT_RELATIONSHIP_TYPE,
    DEFAULT_DOCUMENT_NODE_LABEL,
    DEFAULT_NEXT_CHUNK_RELATIONSHIP_TYPE,
    DEFAULT_NODE_TO_CHUNK_RELATIONSHIP_TYPE,
    DocumentInfo,
    DocumentType,
    GraphResult,
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
    TextChunk,
    TextChunks,
)


@pytest.mark.parametrize(
    "chunk, expected_properties, expected_embedding_properties",
    [
        pytest.param(
            TextChunk(text="text chunk", index=0),
            {"index": 0, "text": "text chunk"},
            {},
            id="no_metadata",
        ),
        pytest.param(
            TextChunk(text="text chunk", index=0, metadata={"status": "ok"}),
            {"index": 0, "text": "text chunk", "status": "ok"},
            {},
            id="metadata_no_embedding",
        ),
        pytest.param(
            TextChunk(
                text="text chunk",
                index=0,
                metadata={"status": "ok", "embedding": [1, 2, 3]},
            ),
            {"index": 0, "text": "text chunk", "status": "ok"},
            {"embedding": [1, 2, 3]},
            id="metadata_embedding",
        ),
    ],
)
def test_lexical_graph_builder_create_chunk_node(
    chunk: TextChunk,
    expected_properties: dict[str, Any],
    expected_embedding_properties: dict[str, Any],
) -> None:
    builder = LexicalGraphBuilder()
    node = builder._create_chunk_node(chunk=chunk)
    assert isinstance(node, Neo4jNode)
    assert node.id is not None
    assert node.properties == expected_properties
    assert node.embedding_properties == expected_embedding_properties


FIXED_NOW = datetime.datetime(2026, 9, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)


@pytest.mark.parametrize(
    "document_info, expected_properties",
    [
        pytest.param(
            DocumentInfo(path="test_lexical_graph", uid="doc-1"),
            {"path": "test_lexical_graph", "createdAt": FIXED_NOW.isoformat()},
            id="no_metadata_no_document_type",
        ),
        pytest.param(
            DocumentInfo(
                path="test_lexical_graph", uid="doc-1", metadata={"author": "me"}
            ),
            {
                "path": "test_lexical_graph",
                "createdAt": FIXED_NOW.isoformat(),
                "author": "me",
            },
            id="metadata",
        ),
        pytest.param(
            DocumentInfo(
                path="test_lexical_graph", uid="doc-1", document_type=DocumentType.PDF
            ),
            {
                "path": "test_lexical_graph",
                "createdAt": FIXED_NOW.isoformat(),
                "document_type": "pdf",
            },
            id="document_type",
        ),
        pytest.param(
            DocumentInfo(
                path="test_lexical_graph",
                uid="doc-1",
                metadata={"author": "me"},
                document_type=DocumentType.MARKDOWN,
            ),
            {
                "path": "test_lexical_graph",
                "createdAt": FIXED_NOW.isoformat(),
                "author": "me",
                "document_type": "markdown",
            },
            id="metadata_and_document_type",
        ),
    ],
)
def test_lexical_graph_builder_create_document_node(
    document_info: DocumentInfo,
    expected_properties: dict[str, Any],
) -> None:
    builder = LexicalGraphBuilder()
    # Freeze the clock since createdAt is set to the current time
    with mock.patch(
        "neo4j_graphrag.components.lexical_graph.datetime"
    ) as mock_datetime:
        mock_datetime.datetime.now.return_value = FIXED_NOW
        node = builder._create_document_node(document_info)
    assert isinstance(node, Neo4jNode)
    assert node.id == document_info.document_id
    assert node.label == DEFAULT_DOCUMENT_NODE_LABEL
    assert node.properties == expected_properties
    assert node.embedding_properties == {}


@pytest.mark.parametrize(
    "config, expected_type",
    [
        pytest.param(
            LexicalGraphConfig(),
            DEFAULT_CHUNK_TO_DOCUMENT_RELATIONSHIP_TYPE,
            id="default_config",
        ),
        pytest.param(
            LexicalGraphConfig(chunk_to_document_relationship_type="IN_REPORT"),
            "IN_REPORT",
            id="custom_config",
        ),
    ],
)
def test_lexical_graph_builder_create_chunk_to_document_rel(
    config: LexicalGraphConfig,
    expected_type: str,
) -> None:
    builder = LexicalGraphBuilder(config=config)
    chunk = TextChunk(text="text chunk", index=0)
    document_info = DocumentInfo(path="test_lexical_graph", uid="doc-1")
    rel = builder._create_chunk_to_document_rel(chunk, document_info)
    assert isinstance(rel, Neo4jRelationship)
    assert rel.start_node_id == chunk.chunk_id
    assert rel.end_node_id == document_info.document_id
    assert rel.type == expected_type


@pytest.mark.parametrize(
    "config, expected_type",
    [
        pytest.param(
            LexicalGraphConfig(),
            DEFAULT_NEXT_CHUNK_RELATIONSHIP_TYPE,
            id="default_config",
        ),
        pytest.param(
            LexicalGraphConfig(next_chunk_relationship_type="NEXT_PAGE"),
            "NEXT_PAGE",
            id="custom_config",
        ),
    ],
)
def test_lexical_graph_builder_create_next_chunk_relationship(
    config: LexicalGraphConfig,
    expected_type: str,
) -> None:
    builder = LexicalGraphBuilder(config=config)
    rel = builder._create_next_chunk_relationship("prev-chunk-id", "chunk-id")
    assert isinstance(rel, Neo4jRelationship)
    assert rel.start_node_id == "prev-chunk-id"
    assert rel.end_node_id == "chunk-id"
    assert rel.type == expected_type


@pytest.mark.parametrize(
    "config, expected_type",
    [
        pytest.param(
            LexicalGraphConfig(),
            DEFAULT_NODE_TO_CHUNK_RELATIONSHIP_TYPE,
            id="default_config",
        ),
        pytest.param(
            LexicalGraphConfig(node_to_chunk_relationship_type="MENTIONED_IN"),
            "MENTIONED_IN",
            id="custom_config",
        ),
    ],
)
def test_lexical_graph_builder_create_node_to_chunk_rel(
    config: LexicalGraphConfig,
    expected_type: str,
) -> None:
    builder = LexicalGraphBuilder(config=config)
    node = Neo4jNode(id="entity-id", label="Person")
    rel = builder._create_node_to_chunk_rel(node, "chunk-id")
    assert isinstance(rel, Neo4jRelationship)
    assert rel.start_node_id == "entity-id"
    assert rel.end_node_id == "chunk-id"
    assert rel.type == expected_type


@pytest.mark.asyncio
async def test_lexical_graph_builder_run_with_document() -> None:
    lexical_graph_builder = LexicalGraphBuilder()
    doc_uid = str(uuid.uuid4())
    chunks = [
        TextChunk(text="text chunk 1", index=0),
        TextChunk(text="text chunk 1", index=1),
    ]
    result = await lexical_graph_builder.run(
        text_chunks=TextChunks(chunks=chunks),
        document_info=DocumentInfo(
            path="test_lexical_graph",
            uid=doc_uid,
            document_type=DocumentType.PDF,
        ),
    )
    assert isinstance(result, GraphResult)
    graph = result.graph
    nodes = graph.nodes
    assert len(nodes) == 3
    document = nodes[0]
    assert document.id == doc_uid
    assert document.label == DEFAULT_DOCUMENT_NODE_LABEL
    assert document.properties["path"] == "test_lexical_graph"
    assert document.properties["createdAt"] is not None
    assert document.properties["document_type"] == "pdf"
    chunk1 = nodes[1]
    assert chunk1.label == DEFAULT_CHUNK_NODE_LABEL
    chunk2 = nodes[2]
    assert chunk2.label == DEFAULT_CHUNK_NODE_LABEL
    assert len(graph.relationships) == 3
    rel_types = [rel.type for rel in graph.relationships]
    assert rel_types.count(DEFAULT_CHUNK_TO_DOCUMENT_RELATIONSHIP_TYPE) == 2
    assert rel_types.count(DEFAULT_NEXT_CHUNK_RELATIONSHIP_TYPE) == 1
    next_chunk_rel = next(
        rel
        for rel in graph.relationships
        if rel.type == DEFAULT_NEXT_CHUNK_RELATIONSHIP_TYPE
    )
    assert next_chunk_rel.start_node_id == chunks[0].chunk_id
    assert next_chunk_rel.end_node_id == chunks[1].chunk_id


@pytest.mark.asyncio
async def test_lexical_graph_builder_run_no_document() -> None:
    lexical_graph_builder = LexicalGraphBuilder()
    result = await lexical_graph_builder.run(
        text_chunks=TextChunks(
            chunks=[
                TextChunk(text="text chunk 1", index=0),
                TextChunk(text="text chunk 1", index=1),
            ]
        ),
    )
    assert isinstance(result, GraphResult)
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
            document_node_label="Report",
            chunk_node_label="Page",
            chunk_to_document_relationship_type="IN_REPORT",
            next_chunk_relationship_type="NEXT_PAGE",
        ),
    )
    doc_uid = str(uuid.uuid4())
    result = await lexical_graph_builder.run(
        text_chunks=TextChunks(
            chunks=[
                TextChunk(text="text chunk 1", index=0),
                TextChunk(text="text chunk 1", index=1),
            ]
        ),
        document_info=DocumentInfo(path="test_lexical_graph", uid=doc_uid),
    )
    assert isinstance(result, GraphResult)
    graph = result.graph
    nodes = graph.nodes
    assert len(nodes) == 3
    document = nodes[0]
    assert document.id == doc_uid
    assert document.label == "Report"
    assert document.properties["path"] == "test_lexical_graph"
    chunk1 = nodes[1]
    assert chunk1.label == "Page"
    chunk2 = nodes[2]
    assert chunk2.label == "Page"
    assert len(graph.relationships) == 3
    rel_types = [rel.type for rel in graph.relationships]
    assert rel_types.count("IN_REPORT") == 2
    assert rel_types.count("NEXT_PAGE") == 1


@pytest.mark.asyncio
async def test_lexical_graph_builder_run_equivalent_to_process_chunk_and_combine_graphs() -> (
    None
):
    lexical_graph_builder = LexicalGraphBuilder()
    doc_uid = str(uuid.uuid4())
    document_info = DocumentInfo(path="test_lexical_graph", uid=doc_uid)
    chunks = [
        TextChunk(text="text chunk 1", index=0),
        TextChunk(text="text chunk 1", index=1),
    ]

    # Freeze the clock so the Document node's createdAt timestamp is
    # identical whether created via run() or per-chunk via run_for_chunk().
    with mock.patch(
        "neo4j_graphrag.components.lexical_graph.datetime"
    ) as mock_datetime:
        mock_datetime.datetime.now.return_value = FIXED_NOW
        result = await lexical_graph_builder.run(
            text_chunks=TextChunks(chunks=chunks),
            document_info=document_info,
        )
        result_by_chunk: list[Neo4jGraph] = [
            lexical_graph_builder.run_for_chunk(chunk, document_info)
            for chunk in chunks
        ]
    combined = functools.reduce(lexical_graph_builder.combine_graphs, result_by_chunk)
    assert result.graph == combined
    # Sanity check that the frozen timestamp was actually used
    assert result.graph.nodes[0].properties["createdAt"] == FIXED_NOW.isoformat()


def test_lexical_graph_builder_combine_graphs_deduplicates() -> None:
    builder = LexicalGraphBuilder()
    node_a = Neo4jNode(id="a", label="Chunk")
    node_a_duplicate = Neo4jNode(id="a", label="Chunk", properties={"text": "other"})
    node_b = Neo4jNode(id="b", label="Chunk")
    rel_a_b = Neo4jRelationship(start_node_id="a", end_node_id="b", type="NEXT_CHUNK")
    rel_b_a = Neo4jRelationship(start_node_id="b", end_node_id="a", type="NEXT_CHUNK")
    graph1 = Neo4jGraph(nodes=[node_a], relationships=[rel_a_b])
    graph2 = Neo4jGraph(
        nodes=[node_a_duplicate, node_b], relationships=[rel_a_b, rel_b_a]
    )

    combined = builder.combine_graphs(graph1, graph2)

    # Duplicates dropped, first occurrence wins, order preserved
    assert combined.nodes == [node_a, node_b]
    assert combined.relationships == [rel_a_b, rel_b_a]
    # Inputs are unchanged
    assert graph1.nodes == [node_a]
    assert graph2.nodes == [node_a_duplicate, node_b]
