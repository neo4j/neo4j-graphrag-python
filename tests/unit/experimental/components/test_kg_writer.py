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
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)
from neo4j_graphrag.neo4j_queries import UPSERT_NODE_QUERY, UPSERT_RELATIONSHIP_QUERY


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_node(driver: MagicMock) -> None:
    neo4j_writer = Neo4jWriter(driver=driver)
    node = Neo4jNode(id="1", label="Label", properties={"key": "value"})
    neo4j_writer._upsert_node(node=node)
    driver.execute_query.assert_called_once_with(
        UPSERT_NODE_QUERY.format(label="Label"),
        parameters_={"id": "1", "properties": {"key": "value"}, "embeddings": None},
    )


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_node_with_embedding(
    driver: MagicMock,
) -> None:
    neo4j_writer = Neo4jWriter(driver=driver)
    node = Neo4jNode(
        id="1",
        label="Label",
        properties={"key": "value"},
        embedding_properties={"embeddingProp": [1.0, 2.0, 3.0]},
    )
    driver.execute_query.return_value.records = [{"elementID(n)": 1}]
    neo4j_writer._upsert_node(node=node)
    driver.execute_query.assert_any_call(
        UPSERT_NODE_QUERY.format(label="Label"),
        parameters_={
            "id": "1",
            "properties": {"key": "value"},
            "embeddings": {"embeddingProp": [1.0, 2.0, 3.0]},
        },
    )


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_relationship(driver: MagicMock) -> None:
    neo4j_writer = Neo4jWriter(driver=driver)
    rel = Neo4jRelationship(
        start_node_id="1",
        end_node_id="2",
        type="RELATIONSHIP",
        properties={"key": "value"},
    )
    neo4j_writer._upsert_relationship(rel=rel)
    parameters = {"start_node_id": "1", "end_node_id": "2", "key": "value"}
    driver.execute_query.assert_called_once_with(
        UPSERT_RELATIONSHIP_QUERY.format(
            type="RELATIONSHIP",
            properties="{key: $key}",
        ),
        parameters_=parameters,
    )


@mock.patch("neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup")
def test_upsert_relationship_with_embedding(driver: MagicMock) -> None:
    neo4j_writer = Neo4jWriter(driver=driver)
    rel = Neo4jRelationship(
        start_node_id="1",
        end_node_id="2",
        type="RELATIONSHIP",
        properties={"key": "value"},
        embedding_properties={"embeddingProp": [1.0, 2.0, 3.0]},
    )
    driver.execute_query.return_value.records = [{"elementID(r)": "rel_elem_id"}]
    neo4j_writer._upsert_relationship(rel=rel)
    parameters = {"start_node_id": "1", "end_node_id": "2", "key": "value"}
    driver.execute_query.assert_any_call(
        UPSERT_RELATIONSHIP_QUERY.format(
            type="RELATIONSHIP",
            properties="{key: $key}",
        ),
        parameters_=parameters,
    )
    query = (
        "MATCH ()-[r]->() "
        "WHERE elementId(r) = $id "
        "WITH r "
        "CALL db.create.setRelationshipVectorProperty(r, $embedding_property, $vector) "
        "RETURN r"
    )
    parameters_ = {
        "id": "rel_elem_id",
        "embedding_property": "embeddingProp",
        "vector": [1.0, 2.0, 3.0],
    }
    driver.execute_query.assert_any_call(query, parameters_, database_=None)


@pytest.mark.asyncio
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
async def test_run(_: Mock, driver: MagicMock) -> None:
    neo4j_writer = Neo4jWriter(driver=driver)
    node = Neo4jNode(id="1", label="Label")
    rel = Neo4jRelationship(start_node_id="1", end_node_id="2", type="RELATIONSHIP")
    graph = Neo4jGraph(nodes=[node], relationships=[rel])
    await neo4j_writer.run(graph=graph)
    driver.execute_query.assert_any_call(
        UPSERT_NODE_QUERY.format(label="Label"),
        parameters_={"id": "1", "properties": {}, "embeddings": None},
    )
    parameters_ = {"start_node_id": "1", "end_node_id": "2"}
    driver.execute_query.assert_any_call(
        UPSERT_RELATIONSHIP_QUERY.format(type="RELATIONSHIP", properties="{}"),
        parameters_=parameters_,
    )


@pytest.mark.asyncio
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._async_db_setup",
    return_value=None,
)
async def test_run_async_driver(_: Mock, async_driver: MagicMock) -> None:
    neo4j_writer = Neo4jWriter(driver=async_driver)
    node = Neo4jNode(id="1", label="Label")
    rel = Neo4jRelationship(start_node_id="1", end_node_id="2", type="RELATIONSHIP")
    graph = Neo4jGraph(nodes=[node], relationships=[rel])
    await neo4j_writer.run(graph=graph)
    async_driver.execute_query.assert_any_call(
        UPSERT_NODE_QUERY.format(label="Label"),
        parameters_={"id": "1", "properties": {}, "embeddings": None},
    )
    parameters_ = {"start_node_id": "1", "end_node_id": "2"}
    async_driver.execute_query.assert_any_call(
        UPSERT_RELATIONSHIP_QUERY.format(type="RELATIONSHIP", properties="{}"),
        parameters_=parameters_,
    )
