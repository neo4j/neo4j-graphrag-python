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
from neo4j import Driver
from neo4j_graphrag.schema import (
    BASE_ENTITY_LABEL,
    NODE_PROPERTIES_QUERY,
    REL_PROPERTIES_QUERY,
    REL_QUERY,
    get_structured_schema,
    query_database,
)


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_cypher_returns_correct_node_properties(driver: Driver) -> None:
    node_properties = query_database(
        driver, NODE_PROPERTIES_QUERY, params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL]}
    )

    expected_node_properties = [
        {
            "output": {
                "properties": [{"property": "property_a", "type": "STRING"}],
                "label": "LabelA",
            }
        }
    ]

    assert node_properties == expected_node_properties


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_cypher_returns_correct_relationship_properties(driver: Driver) -> None:
    relationships_properties = query_database(
        driver, REL_PROPERTIES_QUERY, params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL]}
    )

    expected_relationships_properties = [
        {
            "output": {
                "type": "REL_TYPE",
                "properties": [{"property": "rel_prop", "type": "STRING"}],
            }
        }
    ]

    assert relationships_properties == expected_relationships_properties


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_cypher_returns_correct_relationships(driver: Driver) -> None:
    relationships = query_database(
        driver, REL_QUERY, params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL]}
    )

    expected_relationships = [
        {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelB"}},
        {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelC"}},
    ]

    assert (
        sorted(relationships, key=lambda x: x["output"]["end"])
        == expected_relationships
    )


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_get_structured_schema_returns_correct_node_properties(driver: Driver) -> None:
    result = get_structured_schema(driver)
    assert result["node_props"]["LabelA"] == [
        {"property": "property_a", "type": "STRING"}
    ]


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_get_enhanced_structured_schema_returns_correct_node_properties(
    driver: Driver,
) -> None:
    result = get_structured_schema(driver, True)
    assert result["node_props"]["LabelA"] == [
        {
            "property": "property_a",
            "type": "STRING",
            "values": ["a"],
            "distinct_count": 1,
        }
    ]


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_get_structured_schema_returns_correct_relationship_properties(
    driver: Driver,
) -> None:
    result = get_structured_schema(driver)
    assert result["rel_props"]["REL_TYPE"] == [
        {"property": "rel_prop", "type": "STRING"}
    ]


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_get_enhanced_structured_schema_returns_correct_relationship_properties(
    driver: Driver,
) -> None:
    result = get_structured_schema(driver, True)
    assert result["rel_props"]["REL_TYPE"] == [
        {
            "property": "rel_prop",
            "type": "STRING",
            "values": ["abc"],
            "distinct_count": 1,
        }
    ]


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_get_structured_schema_returns_correct_relationships(driver: Driver) -> None:
    result = get_structured_schema(driver)
    assert sorted(result["relationships"], key=lambda x: x["end"]) == [
        {"end": "LabelB", "start": "LabelA", "type": "REL_TYPE"},
        {"end": "LabelC", "start": "LabelA", "type": "REL_TYPE"},
    ]


@pytest.mark.enterprise_only
@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_get_structured_schema_returns_correct_constraints(driver: Driver) -> None:
    query_database(driver, "DROP CONSTRAINT test_constraint IF EXISTS")
    query_database(
        driver,
        "CREATE CONSTRAINT test_constraint IF NOT EXISTS FOR (n:LabelA) REQUIRE n.property_a IS NOT NULL",
    )

    result = get_structured_schema(driver)
    assert result["metadata"]["constraint"][0].get("name") == "test_constraint"
    assert result["metadata"]["constraint"][0].get("type") == "NODE_PROPERTY_EXISTENCE"
    assert result["metadata"]["constraint"][0].get("entityType") == "NODE"
    assert result["metadata"]["constraint"][0].get("labelsOrTypes") == ["LabelA"]
    assert result["metadata"]["constraint"][0].get("properties") == ["property_a"]


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_get_structured_schema_returns_correct_indexes(driver: Driver) -> None:
    query_database(driver, "DROP INDEX node_range_index IF EXISTS")
    query_database(
        driver, "CREATE INDEX node_range_index FOR (n:LabelA) ON (n.property_a)"
    )

    result = get_structured_schema(driver)
    assert result["metadata"]["index"][0].get("label") == "LabelA"
    assert result["metadata"]["index"][0].get("properties") == ["property_a"]
    assert result["metadata"]["index"][0].get("type") == "RANGE"


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_neo4j_sanitize_values(driver: Driver) -> None:
    output = query_database(driver, "RETURN range(0,130,1) AS result", sanitize=True)
    assert output == [{}]


def test_enhanced_schema_exception(driver: Driver) -> None:
    query_database(driver, "MATCH (n) DETACH DELETE n")
    query_database(
        driver,
        "CREATE (:Node {foo: 'bar'}), (:Node {foo: 1}), (:Node {foo: [1,2]}), "
        "(: EmptyNode)",
    )
    query_database(
        driver,
        "MATCH (a:Node {foo: 'bar'}), (b:Node {foo: 1}), "
        "(c:Node {foo: [1,2]}), (d: EmptyNode) "
        "CREATE (a)-[:REL {foo: 'bar'}]->(b), (b)-[:REL {foo: 1}]->(c), "
        "(c)-[:REL {foo: [1,2]}]->(a), (d)-[:EMPTY_REL {}]->(d)",
    )
    result = get_structured_schema(driver, True)
    del result["metadata"]

    assert list(result.keys()) == ["node_props", "rel_props", "relationships"]
    node_props = result["node_props"]
    assert list(node_props.keys()) == ["Node"]
    assert len(node_props["Node"]) == 1
    assert list(node_props["Node"][0].keys()) == ["property", "type"]
    assert node_props["Node"][0]["property"] == "foo"
    assert node_props["Node"][0]["type"] in ["STRING", "INTEGER", "LIST"]

    rel_props = result["rel_props"]
    assert list(rel_props.keys()) == ["REL"]
    assert len(rel_props["REL"]) == 1
    assert list(rel_props["REL"][0].keys()) == ["property", "type"]
    assert rel_props["REL"][0]["property"] == "foo"
    assert rel_props["REL"][0]["type"] in ["STRING", "INTEGER", "LIST"]

    expected_rels = [
        {
            "end": "Node",
            "start": "Node",
            "type": "REL",
        },
        {"end": "EmptyNode", "start": "EmptyNode", "type": "EMPTY_REL"},
    ]
    rels = result["relationships"]
    assert rels == expected_rels
