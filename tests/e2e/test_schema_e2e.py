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

import pytest
from neo4j_genai.schema import (
    query_database,
    get_structured_schema,
    NODE_PROPERTIES_QUERY,
    BASE_ENTITY_LABEL,
    REL_PROPERTIES_QUERY,
    REL_QUERY,
)


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_cypher_returns_correct_node_properties(driver):
    node_properties = query_database(
        driver, NODE_PROPERTIES_QUERY, params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL]}
    )

    expected_node_properties = [
        {
            "output": {
                "properties": [{"property": "property_a", "type": "STRING"}],
                "labels": "LabelA",
            }
        }
    ]

    assert node_properties == expected_node_properties


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_cypher_returns_correct_relationship_properties(driver):
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
def test_cypher_returns_correct_relationships(driver):
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
def test_get_structured_schema_returns_correct_node_properties(driver):
    result = get_structured_schema(driver)
    assert result["node_props"]["LabelA"] == [
        {"property": "property_a", "type": "STRING"}
    ]


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_get_structured_schema_returns_correct_relationship_properties(driver):
    result = get_structured_schema(driver)
    assert result["rel_props"]["REL_TYPE"] == [
        {"property": "rel_prop", "type": "STRING"}
    ]


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_get_structured_schema_returns_correct_relationships(driver):
    result = get_structured_schema(driver)
    assert sorted(result["relationships"], key=lambda x: x["end"]) == [
        {"end": "LabelB", "start": "LabelA", "type": "REL_TYPE"},
        {"end": "LabelC", "start": "LabelA", "type": "REL_TYPE"},
    ]


@pytest.mark.enterprise_only
@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_get_structured_schema_returns_correct_constraints(driver):
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
def test_get_structured_schema_returns_correct_indexes(driver):
    query_database(driver, "DROP INDEX node_range_index IF EXISTS")
    query_database(
        driver, "CREATE INDEX node_range_index FOR (n:LabelA) ON (n.property_a)"
    )

    result = get_structured_schema(driver)
    assert result["metadata"]["index"][0].get("label") == "LabelA"
    assert result["metadata"]["index"][0].get("properties") == ["property_a"]
    assert result["metadata"]["index"][0].get("type") == "RANGE"
