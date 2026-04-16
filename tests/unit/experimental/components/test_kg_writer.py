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

import tempfile
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
from neo4j_graphrag.experimental.components.filename_collision_handler import (
    FilenameCollisionHandler,
)
from neo4j_graphrag.experimental.components.parquet_formatter import (
    Neo4jGraphParquetFormatter,
    get_unique_properties_for_node_type,
    sanitize_parquet_filestem,
)
from neo4j_graphrag.experimental.components.kg_writer import (
    Neo4jWriter,
    ParquetWriter,
    batched,
)
from neo4j_graphrag.experimental.components.types import (
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)
from neo4j_graphrag.neo4j_queries import (
    upsert_node_query,
    upsert_relationship_query,
)


def test_batched() -> None:
    assert list(batched([1, 2, 3, 4], batch_size=2)) == [
        [1, 2],
        [3, 4],
    ]
    assert list(batched([1, 2, 3], batch_size=2)) == [
        [1, 2],
        [3],
    ]
    assert list(batched([1, 2, 3], batch_size=4)) == [
        [1, 2, 3],
    ]


# --- sanitize_parquet_filestem tests ---


def test_sanitize_parquet_filestem_empty_returns_fallback() -> None:
    assert sanitize_parquet_filestem("") == "unnamed"


def test_sanitize_parquet_filestem_safe_chars_unchanged() -> None:
    assert sanitize_parquet_filestem("Person") == "Person"
    assert sanitize_parquet_filestem("Person_KNOWS_Person") == "Person_KNOWS_Person"
    assert sanitize_parquet_filestem("Label123") == "Label123"
    assert sanitize_parquet_filestem("a_z_9") == "a_z_9"


def test_sanitize_parquet_filestem_unicode_transliterated() -> None:
    assert sanitize_parquet_filestem("Zürich") == "Zurich"
    assert sanitize_parquet_filestem("café") == "cafe"
    assert sanitize_parquet_filestem("naïve") == "naive"


def test_sanitize_parquet_filestem_disallowed_replaced_with_underscore() -> None:
    assert sanitize_parquet_filestem("a b") == "a_b"
    assert sanitize_parquet_filestem("a-b") == "a_b"
    assert sanitize_parquet_filestem("a.b") == "a_b"


def test_sanitize_parquet_filestem_all_disallowed_replaced() -> None:
    # All disallowed chars become underscores (result non-empty, so no fallback)
    assert sanitize_parquet_filestem("...") == "___"
    assert sanitize_parquet_filestem("  ") == "__"


def test_get_unique_properties_for_node_type_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="get_unique_properties_for_node_type"):
        assert get_unique_properties_for_node_type(None, "Person") == ["__id__"]


# --- FilenameCollisionHandler tests ---


def test_filename_collision_handler_first_call_returns_unchanged() -> None:
    FilenameCollisionHandler.reset()
    handler = FilenameCollisionHandler()
    out = Path("/some/output")
    assert handler.get_unique_filename("Person.parquet", out) == "Person.parquet"


def test_filename_collision_handler_collisions_get_suffix() -> None:
    FilenameCollisionHandler.reset()
    handler = FilenameCollisionHandler()
    out = Path("/some/output")
    assert handler.get_unique_filename("Person.parquet", out) == "Person.parquet"
    assert handler.get_unique_filename("Person.parquet", out) == "Person_1.parquet"
    assert handler.get_unique_filename("Person.parquet", out) == "Person_2.parquet"


def test_filename_collision_handler_different_paths_no_collision() -> None:
    FilenameCollisionHandler.reset()
    handler = FilenameCollisionHandler()
    out1 = Path("/out/a")
    out2 = Path("/out/b")
    assert handler.get_unique_filename("Person.parquet", out1) == "Person.parquet"
    assert handler.get_unique_filename("Person.parquet", out2) == "Person.parquet"


def test_filename_collision_handler_reset_clears_state() -> None:
    FilenameCollisionHandler.reset()
    handler = FilenameCollisionHandler()
    out = Path("/out")
    handler.get_unique_filename("Person.parquet", out)
    handler.get_unique_filename("Person.parquet", out)
    FilenameCollisionHandler.reset()
    assert handler.get_unique_filename("Person.parquet", out) == "Person.parquet"


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_nodes(_: Mock, driver: MagicMock) -> None:
    driver.execute_query.return_value = (
        [{"element_id": "#1"}],
        None,
        None,
    )
    neo4j_writer = Neo4jWriter(driver=driver)
    node = Neo4jNode(id="1", label="Label", properties={"key": "value"})
    neo4j_writer._upsert_nodes(nodes=[node], lexical_graph_config=LexicalGraphConfig())
    driver.execute_query.assert_called_once_with(
        upsert_node_query(False),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {"key": "value"},
                    "embedding_properties": {},
                }
            ]
        },
        database_=None,
    )


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_nodes_with_embedding(
    _: Mock,
    driver: MagicMock,
) -> None:
    driver.execute_query.return_value = (
        [{"element_id": "#1"}],
        None,
        None,
    )
    neo4j_writer = Neo4jWriter(driver=driver)
    node = Neo4jNode(
        id="1",
        label="Label",
        properties={"key": "value"},
        embedding_properties={"embeddingProp": [1.0, 2.0, 3.0]},
    )
    neo4j_writer._upsert_nodes(nodes=[node], lexical_graph_config=LexicalGraphConfig())
    driver.execute_query.assert_any_call(
        upsert_node_query(False),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {"key": "value"},
                    "embedding_properties": {"embeddingProp": [1.0, 2.0, 3.0]},
                }
            ]
        },
        database_=None,
    )


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_relationship(_: Mock, driver: MagicMock) -> None:
    neo4j_writer = Neo4jWriter(driver=driver)
    rel = Neo4jRelationship(
        start_node_id="1",
        end_node_id="2",
        type="RELATIONSHIP",
        properties={"key": "value"},
    )
    neo4j_writer._upsert_relationships(
        rels=[rel],
    )
    parameters = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {"key": "value"},
                "embedding_properties": {},
            }
        ]
    }
    driver.execute_query.assert_called_once_with(
        upsert_relationship_query(False),
        parameters_=parameters,
        database_=None,
    )


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_relationship_with_embedding(_: Mock, driver: MagicMock) -> None:
    neo4j_writer = Neo4jWriter(driver=driver)
    rel = Neo4jRelationship(
        start_node_id="1",
        end_node_id="2",
        type="RELATIONSHIP",
        properties={"key": "value"},
        embedding_properties={"embeddingProp": [1.0, 2.0, 3.0]},
    )
    driver.execute_query.return_value.records = [{"elementId(r)": "rel_elem_id"}]
    neo4j_writer._upsert_relationships(
        rels=[rel],
    )
    parameters = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {"key": "value"},
                "embedding_properties": {"embeddingProp": [1.0, 2.0, 3.0]},
            }
        ]
    }
    driver.execute_query.assert_any_call(
        upsert_relationship_query(False),
        parameters_=parameters,
        database_=None,
    )


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@pytest.mark.asyncio
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
async def test_run(_: Mock, driver: MagicMock) -> None:
    driver.execute_query.return_value = (
        [
            {"element_id": "#1"},
            {"element_id": "#2"},
        ],
        None,
        None,
    )
    neo4j_writer = Neo4jWriter(driver=driver)
    node = Neo4jNode(id="1", label="Label")
    rel = Neo4jRelationship(start_node_id="1", end_node_id="2", type="RELATIONSHIP")
    graph = Neo4jGraph(nodes=[node], relationships=[rel])
    await neo4j_writer.run(graph=graph)
    driver.execute_query.assert_any_call(
        upsert_node_query(False),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {},
                    "embedding_properties": {},
                }
            ]
        },
        database_=None,
    )
    parameters_: dict[str, Any] = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {},
                "embedding_properties": {},
            }
        ]
    }
    driver.execute_query.assert_any_call(
        upsert_relationship_query(False),
        parameters_=parameters_,
        database_=None,
    )


@pytest.mark.asyncio
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
async def test_run_is_version_below_5_23(_: Mock) -> None:
    driver = MagicMock()
    driver.execute_query = Mock(
        side_effect=(
            # get_version
            ([{"versions": ["5.22.0"], "edition": "enterpise"}], None, None),
            # upsert nodes
            ([{"_internal_id": "1", "element_id": "#1"}], None, None),
            # upsert relationships
            (None, None, None),
        )
    )

    neo4j_writer = Neo4jWriter(driver=driver)

    node = Neo4jNode(id="1", label="Label")
    rel = Neo4jRelationship(start_node_id="1", end_node_id="2", type="RELATIONSHIP")
    graph = Neo4jGraph(nodes=[node], relationships=[rel])
    await neo4j_writer.run(graph=graph)

    driver.execute_query.assert_any_call(
        upsert_node_query(False),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {},
                    "embedding_properties": {},
                }
            ]
        },
        database_=None,
    )
    parameters_: dict[str, Any] = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {},
                "embedding_properties": {},
            }
        ]
    }
    driver.execute_query.assert_any_call(
        upsert_relationship_query(False),
        parameters_=parameters_,
        database_=None,
    )


@pytest.mark.asyncio
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
async def test_run_is_version_5_23_or_above(_: Mock) -> None:
    driver = MagicMock()
    driver.execute_query = Mock(
        side_effect=(
            # get_version
            ([{"versions": ["5.23.0"], "edition": "enterpise"}], None, None),
            # upsert nodes
            ([{"element_id": "#1"}], None, None),
            # upsert relationships
            (None, None, None),
        )
    )

    neo4j_writer = Neo4jWriter(driver=driver)
    neo4j_writer.is_version_5_23_or_above = True

    node = Neo4jNode(id="1", label="Label")
    rel = Neo4jRelationship(start_node_id="1", end_node_id="2", type="RELATIONSHIP")
    graph = Neo4jGraph(nodes=[node], relationships=[rel])
    await neo4j_writer.run(graph=graph)

    driver.execute_query.assert_any_call(
        upsert_node_query(True, False),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {},
                    "embedding_properties": {},
                }
            ]
        },
        database_=None,
    )
    parameters_: dict[str, Any] = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {},
                "embedding_properties": {},
            }
        ]
    }
    driver.execute_query.assert_any_call(
        upsert_relationship_query(True),
        parameters_=parameters_,
        database_=None,
    )


@pytest.mark.asyncio
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
async def test_run_is_version_5_24_or_above(_: Mock) -> None:
    driver = MagicMock()
    driver.execute_query = Mock(
        side_effect=(
            # get_version
            ([{"versions": ["5.24.0"], "edition": "enterprise"}], None, None),
            # upsert nodes
            ([{"element_id": "#1"}], None, None),
            # upsert relationships
            (None, None, None),
        )
    )

    neo4j_writer = Neo4jWriter(driver=driver)

    node = Neo4jNode(id="1", label="Label")
    rel = Neo4jRelationship(start_node_id="1", end_node_id="2", type="RELATIONSHIP")
    graph = Neo4jGraph(nodes=[node], relationships=[rel])
    await neo4j_writer.run(graph=graph)

    driver.execute_query.assert_any_call(
        upsert_node_query(True, True),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {},
                    "embedding_properties": {},
                }
            ]
        },
        database_=None,
    )
    parameters_: dict[str, Any] = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {},
                "embedding_properties": {},
            }
        ]
    }
    driver.execute_query.assert_any_call(
        upsert_relationship_query(True),
        parameters_=parameters_,
        database_=None,
    )


@pytest.mark.parametrize(
    "description, version, is_5_23_or_above, is_5_24_or_above",
    [
        ("SemVer, < 5.23", "5.22.0", False, False),
        ("SemVer, == 5.23", "5.23.0", True, False),
        ("SemVer, > 5.23", "5.24.0", True, True),
        ("SemVer, < 5.23, Aura", "5.22-aura", False, False),
        ("SemVer, > 5.23, Aura", "5.24-aura", True, True),
        ("CalVer", "2025.01.0", True, True),
        ("CalVer, Aura", "2025.01-aura", True, True),
    ],
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_get_version(
    _: Mock,
    driver: MagicMock,
    description: str,
    version: str,
    is_5_23_or_above: bool,
    is_5_24_or_above: bool,
) -> None:
    execute_query_mock = MagicMock(
        return_value=(
            [
                {"versions": [version], "edition": "enterprise"},
            ],
            None,
            None,
        )
    )
    driver.execute_query = execute_query_mock
    neo4j_writer = Neo4jWriter(driver=driver)
    assert (
        neo4j_writer.is_version_5_23_or_above is is_5_23_or_above
    ), f"Failed is_version_5_23_or_above test case: {description}"
    assert (
        neo4j_writer.is_version_5_24_or_above is is_5_24_or_above
    ), f"Failed is_version_5_24_or_above test case: {description}"


# --- ParquetWriter tests ---


class _LocalParquetDestination:
    """Test-only implementation of ParquetOutputDestination for a local directory."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    @property
    def output_path(self) -> str:
        return str(self._path.resolve())

    async def write(self, data: bytes, filename: str) -> None:
        (self._path / filename).write_bytes(data)


@pytest.mark.asyncio
async def test_parquet_writer_missing_pyarrow_raises() -> None:
    """When pyarrow is not installed, run() returns FAILURE with error mentioning pyarrow."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pyarrow":
            raise ImportError("No module named 'pyarrow'")
        return real_import(name, *args, **kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = _LocalParquetDestination(Path(tmpdir))
        writer = ParquetWriter(
            nodes_dest=dest,
            relationships_dest=dest,
            collision_handler=FilenameCollisionHandler(),
        )
        # Use non-empty graph so formatter calls format_parquet and triggers pyarrow import
        node = Neo4jNode(id="n1", label="Person", properties={})
        graph = Neo4jGraph(nodes=[node], relationships=[])
        with mock.patch("builtins.__import__", side_effect=fake_import):
            result = await writer.run(graph=graph)
    assert result.status == "FAILURE"
    assert result.metadata is not None and "error" in result.metadata
    assert "pyarrow" in result.metadata["error"].lower()


@pytest.mark.asyncio
async def test_parquet_writer_run_success() -> None:
    """ParquetWriter uses formatter and writes one file per node label and per (head, type, tail)."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        dest = _LocalParquetDestination(out)
        writer = ParquetWriter(
            nodes_dest=dest,
            relationships_dest=dest,
            collision_handler=FilenameCollisionHandler(),
        )

        node1 = Neo4jNode(id="n1", label="Person", properties={"name": "Alice"})
        node2 = Neo4jNode(id="n2", label="Person", properties={"name": "Bob"})
        rel = Neo4jRelationship(
            start_node_id="n1", end_node_id="n2", type="KNOWS", properties={}
        )
        graph = Neo4jGraph(nodes=[node1, node2], relationships=[rel])

        result = await writer.run(graph=graph)

        assert result.status == "SUCCESS"
        assert result.metadata is not None
        stats = result.metadata.get("statistics") or {}
        assert stats["node_count"] == 2
        assert stats["relationship_count"] == 1
        assert stats["nodes_per_label"] == {"Person": 2}
        assert "KNOWS" in stats["rel_per_type"]
        assert "input_files_count" in stats
        assert "input_files_total_size_bytes" in stats
        assert (out / "Person.parquet").exists()
        assert (out / "Person_KNOWS_Person.parquet").exists()

        # Check "files" metadata (file_path, columns, source mapping for rels)
        assert "files" in result.metadata
        assert len(result.metadata["files"]) == 2
        node_file_info = next(f for f in result.metadata["files"] if f["is_node"])
        assert node_file_info["name"] == "Person"
        assert (
            "file_path" in node_file_info
            and "Person.parquet" in node_file_info["file_path"]
        )
        assert "columns" in node_file_info
        assert any(
            c["name"] == "__id__" and c["is_primary_key"] and c["is_unique"] is False
            for c in node_file_info["columns"]
        )
        rel_file_info = next(f for f in result.metadata["files"] if not f["is_node"])
        assert rel_file_info["relationship_type"] == "KNOWS"
        assert rel_file_info["start_node_source"] == "Person"
        assert rel_file_info["end_node_source"] == "Person"
        assert rel_file_info["start_node_primary_keys"] == ["__id__"]
        assert rel_file_info["end_node_primary_keys"] == ["__id__"]

        # Read back and sanity-check (formatter uses __id__, labels, and flat properties)
        nodes_table = pq.read_table(out / "Person.parquet")
        assert nodes_table.num_rows == 2
        assert "__id__" in nodes_table.column_names
        assert "labels" in nodes_table.column_names
        assert "name" in nodes_table.column_names

        rels_table = pq.read_table(out / "Person_KNOWS_Person.parquet")
        assert rels_table.num_rows == 1
        assert "from" in rels_table.column_names
        assert "to" in rels_table.column_names
        assert rels_table.column("type")[0].as_py() == "KNOWS"

        rel_cols = {c["name"]: c for c in rel_file_info["columns"]}
        assert rel_cols["from"]["is_primary_key"] is True
        assert rel_cols["from"]["is_unique"] is False
        assert rel_cols["to"]["is_primary_key"] is True
        assert rel_cols["to"]["is_unique"] is False


@pytest.mark.asyncio
async def test_parquet_writer_columns_uniqueness_sets_is_unique() -> None:
    """UNIQUENESS maps to is_unique; __id__ remains synthetic primary key when no KEY."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        dest = _LocalParquetDestination(out)
        writer = ParquetWriter(
            nodes_dest=dest,
            relationships_dest=dest,
            collision_handler=FilenameCollisionHandler(),
        )
        schema_dict: dict[str, Any] = {
            "node_types": [
                {
                    "label": "Person",
                    "properties": [
                        {"name": "email", "type": "STRING"},
                        {"name": "name", "type": "STRING"},
                    ],
                }
            ],
            "constraints": [
                {
                    "type": "UNIQUENESS",
                    "node_type": "Person",
                    "property_name": "email",
                    "relationship_type": None,
                }
            ],
        }
        node = Neo4jNode(
            id="n1",
            label="Person",
            properties={"email": "a@b.c", "name": "Alice"},
        )
        graph = Neo4jGraph(nodes=[node], relationships=[])
        result = await writer.run(graph=graph, schema=schema_dict)
        assert result.status == "SUCCESS"
        assert result.metadata is not None
        node_file = next(f for f in result.metadata["files"] if f["is_node"])
        cols = {c["name"]: c for c in node_file["columns"]}
        assert cols["email"]["is_unique"] is True
        assert cols["email"]["is_primary_key"] is False
        assert cols["__id__"]["is_primary_key"] is True
        assert cols["__id__"]["is_unique"] is False


@pytest.mark.asyncio
async def test_parquet_writer_columns_key_sets_is_primary_key() -> None:
    """KEY maps to is_primary_key on that property; is_unique stays false."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        dest = _LocalParquetDestination(out)
        writer = ParquetWriter(
            nodes_dest=dest,
            relationships_dest=dest,
            collision_handler=FilenameCollisionHandler(),
        )
        schema_dict: dict[str, Any] = {
            "node_types": [
                {
                    "label": "Person",
                    "properties": [
                        {"name": "email", "type": "STRING"},
                        {"name": "name", "type": "STRING"},
                    ],
                }
            ],
            "constraints": [
                {
                    "type": "KEY",
                    "node_type": "Person",
                    "property_name": "email",
                    "relationship_type": None,
                }
            ],
        }
        node = Neo4jNode(
            id="n1",
            label="Person",
            properties={"email": "a@b.c", "name": "Alice"},
        )
        graph = Neo4jGraph(nodes=[node], relationships=[])
        result = await writer.run(graph=graph, schema=schema_dict)
        assert result.status == "SUCCESS"
        assert result.metadata is not None
        node_file = next(f for f in result.metadata["files"] if f["is_node"])
        cols = {c["name"]: c for c in node_file["columns"]}
        assert cols["email"]["is_primary_key"] is True
        assert cols["email"]["is_unique"] is False


@pytest.mark.asyncio
async def test_parquet_writer_run_empty_graph() -> None:
    """ParquetWriter accepts an empty graph and writes no files."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        dest = _LocalParquetDestination(out)
        writer = ParquetWriter(
            nodes_dest=dest,
            relationships_dest=dest,
            collision_handler=FilenameCollisionHandler(),
        )
        graph = Neo4jGraph(nodes=[], relationships=[])

        result = await writer.run(graph=graph)

    assert result.status == "SUCCESS"
    assert result.metadata is not None
    stats = result.metadata.get("statistics") or {}
    assert stats["node_count"] == 0
    assert stats["relationship_count"] == 0
    assert stats["nodes_per_label"] == {}
    assert stats["rel_per_type"] == {}
    assert result.metadata["files"] == []


# ---------------------------------------------------------------------------
# Neo4jGraphParquetFormatter._normalize_column_types
# ---------------------------------------------------------------------------


def test_normalize_column_types_single_row() -> None:
    rows = [{"age": 30, "name": "Alice"}]
    Neo4jGraphParquetFormatter._normalize_column_types(rows)
    assert rows == [{"age": 30, "name": "Alice"}]


def test_normalize_column_types_homogeneous() -> None:
    rows = [{"age": 30}, {"age": 25}]
    Neo4jGraphParquetFormatter._normalize_column_types(rows)
    assert rows == [{"age": 30}, {"age": 25}]


def test_normalize_column_types_mixed_str_int() -> None:
    rows: list[dict[str, Any]] = [{"age": "45"}, {"age": 30}]
    Neo4jGraphParquetFormatter._normalize_column_types(rows)
    assert rows == [{"age": "45"}, {"age": "30"}]


def test_normalize_column_types_mixed_int_float() -> None:
    rows: list[dict[str, Any]] = [{"score": 3}, {"score": 3.5}]
    Neo4jGraphParquetFormatter._normalize_column_types(rows)
    assert rows == [{"score": 3.0}, {"score": 3.5}]


def test_normalize_column_types_none_ignored() -> None:
    """None values should not influence type detection."""
    rows: list[dict[str, Any]] = [{"age": None}, {"age": 30}]
    Neo4jGraphParquetFormatter._normalize_column_types(rows)
    assert rows == [{"age": None}, {"age": 30}]


@pytest.mark.asyncio
async def test_parquet_writer_mixed_property_types() -> None:
    """ParquetWriter succeeds when nodes of the same label have mixed property types."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        dest = _LocalParquetDestination(out)
        writer = ParquetWriter(
            nodes_dest=dest,
            relationships_dest=dest,
            collision_handler=FilenameCollisionHandler(),
        )

        node1 = Neo4jNode(
            id="p1", label="Patient", properties={"name": "John", "age": "45"}
        )
        node2 = Neo4jNode(
            id="p2", label="Patient", properties={"name": "Jane", "age": 30}
        )
        graph = Neo4jGraph(nodes=[node1, node2], relationships=[])

        result = await writer.run(graph=graph)

        assert result.status == "SUCCESS"
        table = pq.read_table(out / "Patient.parquet")
        assert table.num_rows == 2
        # Both ages should have been coerced to str
        ages = {v.as_py() for v in table.column("age")}
        assert ages == {"45", "30"}
