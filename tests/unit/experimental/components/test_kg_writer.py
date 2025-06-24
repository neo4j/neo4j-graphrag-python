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

from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter, batched
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
    neo4j_writer._upsert_nodes(
        nodes=[node], lexical_graph_config=LexicalGraphConfig()
    )
    driver.execute_query.assert_called_once_with(
        upsert_node_query(False),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {"key": "value"},
                    "embedding_properties": None,
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
                "embedding_properties": None,
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
                    "embedding_properties": None,
                }
            ]
        },
        database_=None,
    )
    parameters_ = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {},
                "embedding_properties": None,
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
                    "embedding_properties": None,
                }
            ]
        },
        database_=None,
    )
    parameters_ = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {},
                "embedding_properties": None,
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
        upsert_node_query(True),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {},
                    "embedding_properties": None,
                }
            ]
        },
        database_=None,
    )
    parameters_ = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {},
                "embedding_properties": None,
            }
        ]
    }
    driver.execute_query.assert_any_call(
        upsert_relationship_query(True),
        parameters_=parameters_,
        database_=None,
    )


@pytest.mark.parametrize(
    "description, version, is_5_23_or_above",
    [
        ("SemVer, < 5.23", "5.22.0", False),
        ("SemVer, > 5.23", "5.24.0", True),
        ("SemVer, < 5.23, Aura", "5.22-aura", False),
        ("SemVer, > 5.23, Aura", "5.24-aura", True),
        ("CalVer", "2025.01.0", True),
        ("CalVer, Aura", "2025.01-aura", True),
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
