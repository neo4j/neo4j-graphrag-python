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
        writer = ParquetWriter(
            output_path=Path(tmpdir),
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
        writer = ParquetWriter(
            output_path=out,
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
    assert result.metadata["node_count"] == 2
    assert result.metadata["relationship_count"] == 1
    assert result.metadata["nodes_per_label"] == {"Person": 2}
    assert "Person_KNOWS_Person" in result.metadata["rel_per_type"]
    assert (out / "Person.parquet").exists()
    assert (out / "Person_KNOWS_Person.parquet").exists()

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


@pytest.mark.asyncio
async def test_parquet_writer_run_empty_graph() -> None:
    """ParquetWriter accepts an empty graph and writes no files."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        writer = ParquetWriter(
            output_path=out,
            collision_handler=FilenameCollisionHandler(),
        )
        graph = Neo4jGraph(nodes=[], relationships=[])

        result = await writer.run(graph=graph)

    assert result.status == "SUCCESS"
    assert result.metadata["node_count"] == 0
    assert result.metadata["relationship_count"] == 0
    assert result.metadata["files_written"] == []
