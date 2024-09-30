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
from neo4j_graphrag.experimental.components.resolver import (
    SinglePropertyExactMatchResolver,
)


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_resolver_single_node(driver: neo4j.Driver) -> None:
    driver.execute_query("MATCH (n) DETACH DELETE n")
    driver.execute_query(
        """
        CREATE (d:Document:__Entity__ {id: "0", path: "path"})
        CREATE (c:Chunk:__Entity__ {id: "0:0"})
        CREATE (c)-[:FROM_DOCUMENT]->(d)
        CREATE (alice:__Entity__:Person {id: "0:0:1", name: "Alice"})
        CREATE (alice)-[:FROM_CHUNK]->(c)
        """
    )
    resolver = SinglePropertyExactMatchResolver(driver)
    res = await resolver.run("path")
    # __Entity__ nodes attached to a chunk
    assert res.number_of_affected_nodes == 1
    # Alice
    assert res.number_of_created_nodes == 1

    records, _, _ = driver.execute_query(
        "MATCH path=(:Person {name: 'Alice'}) RETURN path"
    )
    assert len(records) == 1
    path = records[0].get("path")
    assert path.start_node.get("name") == "Alice"
    assert path.start_node.labels == frozenset({"__Entity__", "Person"})
    assert path.start_node.get("id") == "0:0:1"


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_resolver_two_nodes_and_relationships(driver: neo4j.Driver) -> None:
    driver.execute_query("MATCH (n) DETACH DELETE n")
    driver.execute_query(
        """
        CREATE (d:Document:__Entity__ {id: "0", path: "path"})
        CREATE (c:Chunk:__Entity__ {id: "0:0"})
        CREATE (c)-[:FROM_DOCUMENT]->(d)
        CREATE (alice1:__Entity__:Person {id: "0:0:1", name: "Alice"})
        CREATE (alice2:__Entity__:Person {id: "0:0:2", name: "Alice"})
        CREATE (sweden:__Entity__:Country {id: "0:0:3", name: "Sweden"})
        CREATE (alice1)-[:LIVES_IN]->(sweden)
        CREATE (alice1)-[:FROM_CHUNK]->(c)
        CREATE (alice2)-[:FROM_CHUNK]->(c)
        CREATE (sweden)-[:FROM_CHUNK]->(c)
        """
    )
    resolver = SinglePropertyExactMatchResolver(driver)
    res = await resolver.run("path")
    # __Entity__ nodes attached to a chunk
    assert res.number_of_affected_nodes == 3
    # Alice and Sweden
    assert res.number_of_created_nodes == 2

    # check the domain graph
    records, _, _ = driver.execute_query(
        "MATCH path=(:Person {name: 'Alice'})-[:LIVES_IN]->(:Country {name: 'Sweden'}) RETURN path"
    )
    assert len(records) == 1
    path = records[0].get("path")
    assert path.start_node.get("name") == "Alice"
    assert path.start_node.labels == frozenset({"__Entity__", "Person"})
    assert path.end_node.get("name") == "Sweden"
    assert path.end_node.labels == frozenset({"__Entity__", "Country"})
    assert len(path.relationships) == 1
    assert path.relationships[0].type == "LIVES_IN"

    # check the lexical graph
    records, _, _ = driver.execute_query(
        "MATCH path=(:Person {name: 'Alice'})-[:FROM_CHUNK]->(:Chunk) RETURN path"
    )
    assert len(records) == 1
    path = records[0].get("path")
    assert path.start_node.get("name") == "Alice"
    assert path.end_node.labels == frozenset({"__Entity__", "Chunk"})
    assert path.end_node.get("id") == "0:0"


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_resolver_same_name_different_labels(driver: neo4j.Driver) -> None:
    driver.execute_query("MATCH (n) DETACH DELETE n")
    driver.execute_query(
        """
        CREATE (d:Document:__Entity__ {id: "0", path: "path"})
        CREATE (c:Chunk:__Entity__ {id: "0:0"})
        CREATE (c)-[:FROM_DOCUMENT]->(d)
        CREATE (alice1:__Entity__:Person {id: "0:0:1", name: "Alice"})
        CREATE (alice2:__Entity__:Human {id: "0:0:2", name: "Alice"})
        CREATE (alice1)-[:FROM_CHUNK]->(c)
        CREATE (alice2)-[:FROM_CHUNK]->(c)
        CREATE (sweden)-[:FROM_CHUNK]->(c)
        """
    )
    resolver = SinglePropertyExactMatchResolver(driver)
    res = await resolver.run("path")
    # __Entity__ nodes attached to a chunk
    assert res.number_of_affected_nodes == 2
    # Alice Person and Alice Human
    assert res.number_of_created_nodes == 2

    records, _, _ = driver.execute_query("MATCH (alice {name: 'Alice'}) RETURN alice")
    assert len(records) == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_resolver_custom_property(driver: neo4j.Driver) -> None:
    driver.execute_query("MATCH (n) DETACH DELETE n")
    driver.execute_query(
        """
        CREATE (d:Document:__Entity__ {id: "0", path: "path"})
        CREATE (c:Chunk:__Entity__ {id: "0:0"})
        CREATE (c)-[:FROM_DOCUMENT]->(d)
        CREATE (alice:__Entity__:Person {id: "0:0:1", name: "Alice"})
        CREATE (alicia:__Entity__:Person {id: "0:0:1", name: "Alicia"})
        CREATE (alice)-[:FROM_CHUNK]->(c)
        CREATE (alicia)-[:FROM_CHUNK]->(c)
        """
    )
    resolver = SinglePropertyExactMatchResolver(driver, resolve_property="id")
    res = await resolver.run("path")
    # __Entity__ nodes attached to a chunk
    assert res.number_of_affected_nodes == 2
    # Alice
    assert res.number_of_created_nodes == 1

    records, _, _ = driver.execute_query("MATCH (person:Person) RETURN person")
    assert len(records) == 1
    assert records[0].get("person").get("name") == "Alice"
