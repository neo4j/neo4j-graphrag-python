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

import neo4j
import pytest
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.types import (
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
        properties={"chunk": 1},
        embedding_properties={"vectorProperty": [1.0, 2.0, 3.0]},
    )
    end_node = Neo4jNode(
        id="2",
        label="MyLabel",
        properties={},
        embedding_properties=None,
    )
    relationship = Neo4jRelationship(
        start_node_id="1", end_node_id="2", type="MY_RELATIONSHIP"
    )
    node_with_two_embeddings = Neo4jNode(
        id="3",
        label="MyLabel",
        properties={"chunk": 1},
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
    MATCH (a:MyLabel {id: '1'})-[r:MY_RELATIONSHIP]->(b:MyLabel {id: '2'})
    RETURN a, r, b
    """
    record = driver.execute_query(query).records[0]
    assert "a" and "b" and "r" in record.keys()

    node_a = record["a"]
    assert start_node.label in list(node_a.labels)
    assert start_node.id == str(node_a.get("id"))
    for key, val in start_node.properties.items():
        assert key in node_a.keys()
        assert val == node_a.get(key)
    for key, val in start_node.embedding_properties.items():
        assert key in node_a.keys()
        assert node_a.get(key) == [1.0, 2.0, 3.0]

    node_b = record["b"]
    assert end_node.label in list(node_b.labels)
    assert end_node.id == str(node_b.get("id"))
    for key, val in end_node.properties.items():
        assert key in node_b.keys()
        assert val == node_b.get(key)

    rel = record["r"]
    assert rel.type == relationship.type
    assert relationship.start_node_id == rel.start_node.get("id")
    assert relationship.end_node_id == rel.end_node.get("id")

    query = """
    MATCH (c:MyLabel {id: '3'})
    RETURN c
    """
    records = driver.execute_query(query).records
    assert len(records) == 1
    node_c = records[0]["c"]
    for key, val in node_with_two_embeddings.embedding_properties.items():
        assert key in node_c.keys()
        assert val == node_c.get(key)
