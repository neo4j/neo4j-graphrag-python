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
from neo4j_genai.kg_construction.kg_writer import Neo4jWriter
from neo4j_genai.kg_construction.types import (
    Neo4jEmbeddingProperty,
    Neo4jGraph,
    Neo4jNode,
    Neo4jProperty,
    Neo4jRelationship,
)


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_kg_writer(driver: neo4j.Driver) -> None:
    start_node = Neo4jNode(
        id="1",
        label="Document",
        properties=[Neo4jProperty(key="chunk", value=1)],
        embedding_properties=[
            Neo4jEmbeddingProperty(key="vectorProperty", value=[1.0, 2.0, 3.0])
        ],
    )
    end_node = Neo4jNode(
        id="2",
        label="Document",
        properties=[Neo4jProperty(key="chunk", value=2)],
        embedding_properties=[
            Neo4jEmbeddingProperty(
                key="vectorProperty",
                value=[1.0, 2.0, 3.0],
            )
        ],
    )
    relationship = Neo4jRelationship(
        start_node_id="1", end_node_id="2", label="NEXT_CHUNK"
    )
    graph = Neo4jGraph(nodes=[start_node, end_node], relationships=[relationship])

    neo4j_writer = Neo4jWriter(driver=driver)
    await neo4j_writer.run(graph=graph)

    query = """
    MATCH (a:Document {id: 1})-[r:NEXT_CHUNK]-(b:Document {id: 2})
    RETURN a, r, b
    """
    record = driver.execute_query(query).records[0]
    assert "a" and "b" and "r" in record.keys()

    node_a = record["a"]
    assert start_node.label == list(node_a.labels)[0]
    assert start_node.id == str(node_a.get("id"))
    if start_node.properties:
        for prop in start_node.properties:
            assert prop.key in node_a.keys()
            assert prop.value == node_a.get(prop.key)
    if start_node.embedding_properties:
        for embedding_prop in start_node.embedding_properties:
            assert embedding_prop.key in node_a.keys()
            assert node_a.get(embedding_prop.key) == [1.0, 2.0, 3.0]

    node_b = record["b"]
    assert end_node.label == list(node_b.labels)[0]
    assert end_node.id == str(node_b.get("id"))
    if end_node.properties:
        for prop in end_node.properties:
            assert prop.key in node_b.keys()
            assert prop.value == node_b.get(prop.key)
    if end_node.embedding_properties:
        for embedding_prop in end_node.embedding_properties:
            assert embedding_prop.key in node_b.keys()
            assert node_b.get(embedding_prop.key) == [1.0, 2.0, 3.0]

    rel = record["r"]
    assert rel.type == relationship.label
    assert relationship.start_node_id and relationship.end_node_id in [
        str(node.get("id")) for node in rel.nodes
    ]
