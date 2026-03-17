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
import logging
import tempfile
from pathlib import Path

import neo4j
import pytest

from neo4j_graphrag.experimental.components.filename_collision_handler import (
    FilenameCollisionHandler,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter, ParquetWriter
from neo4j_graphrag.experimental.components.types import (
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_kg_writer(driver: neo4j.Driver) -> None:
    start_node = Neo4jNode(
        id="1",
        label="MyLabel",
        properties={"id": "abc"},
        embedding_properties={"vectorProperty": [1.0, 2.0, 3.0]},
    )
    end_node = Neo4jNode(
        id="2",
        label="MyLabel",
        properties={"id": "def"},
    )
    relationship = Neo4jRelationship(
        start_node_id="1", end_node_id="2", type="MY_RELATIONSHIP"
    )
    node_with_two_embeddings = Neo4jNode(
        id="3",
        label="MyLabel",
        properties={"id": "ghi"},
        embedding_properties={
            "vectorProperty": [1.0, 2.0, 3.0],
            "otherVectorProperty": [10.0, 20.0, 30.0],
        },
    )
    graph = Neo4jGraph(
        nodes=[start_node, end_node, node_with_two_embeddings],
        relationships=[relationship],
    )

    neo4j_writer = Neo4jWriter(driver=driver)
    res = await neo4j_writer.run(graph=graph)
    assert res.status == "SUCCESS"

    query = """
    MATCH (a:MyLabel {id: 'abc'})-[r:MY_RELATIONSHIP]->(b:MyLabel {id: 'def'})
    RETURN a, r, b
    """
    record = driver.execute_query(query).records[0]
    assert "a" and "b" and "r" in record.keys()

    node_a = record["a"]
    assert start_node.label in list(node_a.labels)
    assert start_node.properties.get("id") == str(node_a.get("id"))
    for key, val in start_node.properties.items():
        assert key in node_a.keys()
        assert val == node_a.get(key)
    if start_node.embedding_properties:  # for mypy
        for emb_key, emb_val in start_node.embedding_properties.items():
            assert emb_key in node_a.keys()
            assert emb_val == node_a.get(emb_key)

    node_b = record["b"]
    assert end_node.label in list(node_b.labels)
    assert end_node.properties.get("id") == str(node_b.get("id"))
    for key, val in end_node.properties.items():
        assert key in node_b.keys()
        assert val == node_b.get(key)

    rel = record["r"]
    assert rel.type == relationship.type
    assert rel.start_node.get("id") == start_node.properties.get("id")
    assert rel.end_node.get("id") == end_node.properties.get("id")

    query = """
    MATCH (c:MyLabel {id: 'ghi'})
    RETURN c
    """
    records = driver.execute_query(query).records
    assert len(records) == 1
    node_c = records[0]["c"]
    if node_with_two_embeddings.embedding_properties:  # for mypy
        for emb_key, emb_val in node_with_two_embeddings.embedding_properties.items():
            assert emb_key in node_c.keys()
            assert emb_val == node_c.get(emb_key)


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_kg_writer_no_neo4j_deprecation_warning(
    driver: neo4j.Driver, caplog: pytest.LogCaptureFixture
) -> None:
    start_node = Neo4jNode(
        id="1",
        label="MyLabel",
        properties={"chunk": 1},
        embedding_properties={"vectorProperty": [1.0, 2.0, 3.0]},
    )
    end_node = Neo4jNode(
        id="2",
        label="MyLabel",
        properties={},
    )
    relationship = Neo4jRelationship(
        start_node_id="1", end_node_id="2", type="MY_RELATIONSHIP"
    )
    graph = Neo4jGraph(
        nodes=[start_node, end_node],
        relationships=[relationship],
    )

    neo4j_writer = Neo4jWriter(driver=driver)
    with caplog.at_level(logging.WARNING):
        res = await neo4j_writer.run(graph=graph)

    for record in caplog.records:
        if (
            "Neo.ClientNotification.Statement.FeatureDeprecationWarning"
            in record.message
        ):
            assert False, f"Deprecation warning found in logs: {record.message}"

    assert res.status == "SUCCESS"


class _LocalParquetDestination:
    """E2E test-only implementation of ParquetOutputDestination for a local directory."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    @property
    def output_path(self) -> str:
        return str(self._path.resolve())

    async def write(self, data: bytes, filename: str) -> None:
        (self._path / filename).write_bytes(data)


@pytest.mark.asyncio
async def test_parquet_writer_e2e() -> None:
    """E2E test for ParquetWriter: write graph to Parquet files and verify content."""
    pyarrow = pytest.importorskip("pyarrow")

    start_node = Neo4jNode(
        id="p1",
        label="Person",
        properties={"name": "Alice", "age": 30},
    )
    end_node = Neo4jNode(
        id="p2",
        label="Person",
        properties={"name": "Bob", "age": 25},
    )
    relationship = Neo4jRelationship(
        start_node_id="p1",
        end_node_id="p2",
        type="KNOWS",
    )
    graph = Neo4jGraph(
        nodes=[start_node, end_node],
        relationships=[relationship],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        dest = _LocalParquetDestination(output_path)
        collision_handler = FilenameCollisionHandler()
        writer = ParquetWriter(
            nodes_dest=dest,
            relationships_dest=dest,
            collision_handler=collision_handler,
            prefix="e2e_",
        )
        result = await writer.run(
            graph=graph,
            lexical_graph_config=LexicalGraphConfig(),
        )

        assert result.status == "SUCCESS"
        assert result.metadata is not None
        stats = result.metadata.get("statistics") or {}
        assert stats["node_count"] == 2
        assert stats["relationship_count"] == 1
        assert stats["nodes_per_label"]["Person"] == 2
        assert stats["rel_per_type"]["KNOWS"] == 1
        assert "input_files_count" in stats
        assert "input_files_total_size_bytes" in stats

        files_meta = result.metadata.get("files") or []
        assert len(files_meta) == 2, f"Expected 2 files, got {files_meta}"
        node_file = rel_file = None
        for f in files_meta:
            path = Path(f["file_path"])
            assert path.exists(), f"Expected file {path}"
            table = pyarrow.parquet.read_table(path)
            if "from" in table.column_names and "to" in table.column_names:
                rel_file = path
            elif "labels" in table.column_names:
                node_file = path
        assert node_file is not None, "No node Parquet file found"
        assert rel_file is not None, "No relationship Parquet file found"

        node_table = pyarrow.parquet.read_table(node_file)
        assert node_table.num_rows == 2
        assert "name" in node_table.column_names
        assert "age" in node_table.column_names
        name_values = node_table.column("name").to_pylist()
        assert "Alice" in name_values and "Bob" in name_values

        rel_table = pyarrow.parquet.read_table(rel_file)
        assert rel_table.num_rows == 1
        assert "from" in rel_table.column_names
        assert "to" in rel_table.column_names
        assert rel_table.column("type")[0].as_py() == "KNOWS"
        assert rel_table.column("from")[0].as_py() == "p1"
        assert rel_table.column("to")[0].as_py() == "p2"


@pytest.mark.asyncio
async def test_parquet_writer_preserves_lexical_graph_nodes_and_rels() -> None:
    """ParquetWriter must preserve lexical graph nodes and lexical relationship types in Parquet output."""
    pyarrow = pytest.importorskip("pyarrow")

    config = LexicalGraphConfig(
        document_node_label="__Document__",
        chunk_node_label="__Chunk__",
        chunk_to_document_relationship_type="__CHUNK_TO_DOCUMENT__",
        next_chunk_relationship_type="__NEXT_CHUNK__",
        node_to_chunk_relationship_type="__NODE_TO_CHUNK__",
    )

    # Lexical graph nodes (custom labels from config)
    doc_node = Neo4jNode(
        id="doc-1",
        label=config.document_node_label,
        properties={"id": "doc-1", "name": "MyDoc"},
    )
    chunk_a = Neo4jNode(
        id="chunk-1",
        label=config.chunk_node_label,
        properties={"id": "chunk-1", "index": 0, "text": "First chunk."},
    )
    chunk_b = Neo4jNode(
        id="chunk-2",
        label=config.chunk_node_label,
        properties={"id": "chunk-2", "index": 1, "text": "Second chunk."},
    )
    # Entity node (non-lexical)
    person_node = Neo4jNode(
        id="p1",
        label="Person",
        properties={"name": "Alice"},
    )

    # Lexical relationships: Chunk -> Document, Chunk -> Chunk, Person -> Chunk (node-to-chunk)
    rel_chunk_to_doc = Neo4jRelationship(
        start_node_id="chunk-1",
        end_node_id="doc-1",
        type=config.chunk_to_document_relationship_type,
    )
    rel_next_chunk = Neo4jRelationship(
        start_node_id="chunk-1",
        end_node_id="chunk-2",
        type=config.next_chunk_relationship_type,
    )
    rel_node_to_chunk = Neo4jRelationship(
        start_node_id="p1",
        end_node_id="chunk-1",
        type=config.node_to_chunk_relationship_type,
    )
    # Entity relationship
    rel_knows = Neo4jRelationship(
        start_node_id="p1",
        end_node_id="p1",
        type="KNOWS",
    )

    graph = Neo4jGraph(
        nodes=[doc_node, chunk_a, chunk_b, person_node],
        relationships=[rel_chunk_to_doc, rel_next_chunk, rel_node_to_chunk, rel_knows],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        dest = _LocalParquetDestination(output_path)
        collision_handler = FilenameCollisionHandler()
        writer = ParquetWriter(
            nodes_dest=dest,
            relationships_dest=dest,
            collision_handler=collision_handler,
            prefix="lex_",
        )
        result = await writer.run(
            graph=graph,
            lexical_graph_config=config,
        )

        assert result.status == "SUCCESS"
        assert result.metadata is not None
        statistics = result.metadata.get("statistics") or {}
        assert statistics["node_count"] == 4
        assert statistics["relationship_count"] == 4
        assert "input_files_count" in statistics
        assert "input_files_total_size_bytes" in statistics

        # Returned metadata must include lexical graph node and relationship entries in statistics
        nodes_per_label = statistics["nodes_per_label"]
        rel_per_type = statistics["rel_per_type"]

        # Custom lexical node labels must appear in nodes_per_label
        assert config.document_node_label in nodes_per_label, (
            f"FAIL: Custom '{config.document_node_label}' label not found in nodes_per_label. "
            f"Found: {list(nodes_per_label.keys())}. "
            f"Lexical graph config was not applied correctly."
        )
        assert config.chunk_node_label in nodes_per_label, (
            f"FAIL: Custom '{config.chunk_node_label}' label not found in nodes_per_label. "
            f"Found: {list(nodes_per_label.keys())}. "
            f"Lexical graph config was not applied correctly."
        )

        # Non-zero counts for lexical nodes
        assert nodes_per_label[config.document_node_label] > 0, (
            f"Expected at least 1 {config.document_node_label} node, "
            f"got {nodes_per_label[config.document_node_label]}"
        )
        assert nodes_per_label[config.chunk_node_label] > 0, (
            f"Expected at least 1 {config.chunk_node_label} node, "
            f"got {nodes_per_label[config.chunk_node_label]}"
        )

        # Default labels must NOT be present
        assert "Document" not in nodes_per_label, (
            f"FAIL: Found default 'Document' label (should be '{config.document_node_label}'). "
            f"Labels: {list(nodes_per_label.keys())}"
        )
        assert "Chunk" not in nodes_per_label, (
            f"FAIL: Found default 'Chunk' label (should be '{config.chunk_node_label}'). "
            f"Labels: {list(nodes_per_label.keys())}"
        )

        # Custom lexical relationship types must appear in rel_per_type (keyed by type name)
        assert config.chunk_to_document_relationship_type in rel_per_type, (
            f"FAIL: Custom '{config.chunk_to_document_relationship_type}' relationship not found. "
            f"Found: {list(rel_per_type.keys())}"
        )
        assert config.next_chunk_relationship_type in rel_per_type, (
            f"FAIL: Custom '{config.next_chunk_relationship_type}' relationship not found. "
            f"Found: {list(rel_per_type.keys())}"
        )
        assert config.node_to_chunk_relationship_type in rel_per_type, (
            f"FAIL: Custom '{config.node_to_chunk_relationship_type}' relationship not found. "
            f"Found: {list(rel_per_type.keys())}"
        )

        # Non-zero counts for lexical relationships
        assert (
            rel_per_type.get(config.chunk_to_document_relationship_type, 0) > 0
        ), f"Expected at least 1 {config.chunk_to_document_relationship_type} relationship"
        assert (
            rel_per_type.get(config.next_chunk_relationship_type, 0) > 0
        ), f"Expected at least 1 {config.next_chunk_relationship_type} relationship"
        assert (
            rel_per_type.get(config.node_to_chunk_relationship_type, 0) > 0
        ), f"Expected at least 1 {config.node_to_chunk_relationship_type} relationship"

        # Default relationship type names must NOT be present
        assert (
            "FROM_DOCUMENT" not in rel_per_type
        ), f"FAIL: Found default 'FROM_DOCUMENT' (should be '{config.chunk_to_document_relationship_type}')"
        assert (
            "FROM_CHUNK" not in rel_per_type
        ), f"FAIL: Found default 'FROM_CHUNK' (should be '{config.node_to_chunk_relationship_type}')"

        # metadata["files"] must include lexical graph node and relationship file entries
        files_meta = result.metadata.get("files") or []
        node_files_meta = [f for f in files_meta if f.get("is_node") is True]
        rel_files_meta = [f for f in files_meta if f.get("is_node") is False]
        assert any(
            f.get("name") == config.document_node_label
            or config.document_node_label in (f.get("labels") or [])
            for f in node_files_meta
        ), "metadata files should include lexical document node file"
        assert any(
            f.get("name") == config.chunk_node_label
            or config.chunk_node_label in (f.get("labels") or [])
            for f in node_files_meta
        ), "metadata files should include lexical chunk node file"
        assert any(
            f.get("relationship_type") == config.chunk_to_document_relationship_type
            for f in rel_files_meta
        ), "metadata files should include chunk-to-document relationship file"
        assert any(
            f.get("relationship_type") == config.next_chunk_relationship_type
            for f in rel_files_meta
        ), "metadata files should include next-chunk relationship file"
        assert any(
            f.get("relationship_type") == config.node_to_chunk_relationship_type
            for f in rel_files_meta
        ), "metadata files should include node-to-chunk relationship file"

        files_meta_paths = [
            Path(f["file_path"]) for f in (result.metadata.get("files") or [])
        ]
        node_files = {}
        rel_files = {}
        for path in files_meta_paths:
            assert path.exists(), f"Expected file {path}"
            table = pyarrow.parquet.read_table(path)
            if "from" in table.column_names and "to" in table.column_names:
                rel_type = table.column("type")[0].as_py() if table.num_rows else None
                key = (
                    table.column("from_label")[0].as_py(),
                    rel_type,
                    table.column("to_label")[0].as_py(),
                )
                rel_files[key] = path
            elif "labels" in table.column_names:
                labels_col = table.column("labels")
                first_labels = labels_col.slice(0, 1)
                label_set = set(first_labels[0].as_py()) if first_labels else set()
                if config.document_node_label in label_set:
                    node_files[config.document_node_label] = path
                elif (
                    config.chunk_node_label in label_set
                    and "__Entity__" not in label_set
                ):
                    node_files[config.chunk_node_label] = path
                elif "Person" in label_set:
                    node_files["Person"] = path

        assert (
            config.document_node_label in node_files
        ), "Document node Parquet file should exist"
        assert (
            config.chunk_node_label in node_files
        ), "Chunk node Parquet file should exist"
        assert "Person" in node_files, "Person node Parquet file should exist"

        # Lexical nodes: labels column must NOT contain __Entity__
        doc_table = pyarrow.parquet.read_table(node_files[config.document_node_label])
        assert doc_table.num_rows == 1
        doc_labels = doc_table.column("labels")[0].as_py()
        assert config.document_node_label in doc_labels
        assert (
            "__Entity__" not in doc_labels
        ), "Lexical document node must not have __Entity__ label"
        assert doc_table.column("__id__")[0].as_py() == "doc-1"
        assert doc_table.column("name")[0].as_py() == "MyDoc"

        chunk_table = pyarrow.parquet.read_table(node_files[config.chunk_node_label])
        assert chunk_table.num_rows == 2
        for i in range(2):
            chunk_labels = chunk_table.column("labels")[i].as_py()
            assert config.chunk_node_label in chunk_labels
            assert (
                "__Entity__" not in chunk_labels
            ), "Lexical chunk node must not have __Entity__ label"
        ids = chunk_table.column("__id__").to_pylist()
        assert "chunk-1" in ids and "chunk-2" in ids
        texts = chunk_table.column("text").to_pylist()
        assert "First chunk." in texts and "Second chunk." in texts

        # Entity node: must have __Entity__ in labels
        person_table = pyarrow.parquet.read_table(node_files["Person"])
        assert person_table.num_rows == 1
        person_labels = person_table.column("labels")[0].as_py()
        assert "Person" in person_labels
        assert "__Entity__" in person_labels

        # Lexical relationships preserved in Parquet files
        chunk_doc_key = (
            config.chunk_node_label,
            config.chunk_to_document_relationship_type,
            config.document_node_label,
        )
        chunk_chunk_key = (
            config.chunk_node_label,
            config.next_chunk_relationship_type,
            config.chunk_node_label,
        )
        node_to_chunk_key = (
            "Person",
            config.node_to_chunk_relationship_type,
            config.chunk_node_label,
        )
        assert (
            chunk_doc_key in rel_files
        ), "Chunk-to-document relationship file should exist"
        assert chunk_chunk_key in rel_files, "Next-chunk relationship file should exist"
        assert (
            node_to_chunk_key in rel_files
        ), "Node-to-chunk relationship file should exist"

        from_doc_table = pyarrow.parquet.read_table(rel_files[chunk_doc_key])
        assert from_doc_table.num_rows == 1
        assert (
            from_doc_table.column("type")[0].as_py()
            == config.chunk_to_document_relationship_type
        )
        assert from_doc_table.column("from")[0].as_py() == "chunk-1"
        assert from_doc_table.column("to")[0].as_py() == "doc-1"

        next_chunk_table = pyarrow.parquet.read_table(rel_files[chunk_chunk_key])
        assert next_chunk_table.num_rows == 1
        assert (
            next_chunk_table.column("type")[0].as_py()
            == config.next_chunk_relationship_type
        )
        assert next_chunk_table.column("from")[0].as_py() == "chunk-1"
        assert next_chunk_table.column("to")[0].as_py() == "chunk-2"

        node_to_chunk_table = pyarrow.parquet.read_table(rel_files[node_to_chunk_key])
        assert node_to_chunk_table.num_rows == 1
        assert (
            node_to_chunk_table.column("type")[0].as_py()
            == config.node_to_chunk_relationship_type
        )
        assert node_to_chunk_table.column("from")[0].as_py() == "p1"
        assert node_to_chunk_table.column("to")[0].as_py() == "chunk-1"
