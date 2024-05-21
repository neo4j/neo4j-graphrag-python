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
    get_schema,
    query_database,
    NODE_PROPERTIES_QUERY,
    BASE_ENTITY_LABEL,
    REL_PROPERTIES_QUERY,
    REL_QUERY,
)


@pytest.mark.usefixtures("setup_neo4j")
def test_get_schema_returns_correct_schema(driver):
    # Delete all nodes in the graph
    driver.execute_query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    driver.execute_query(
        """
        CREATE (la:LabelA {property_a: 'a'})
        CREATE (lb:LabelB)
        CREATE (lc:LabelC)
        MERGE (la)-[:REL_TYPE]-> (lb)
        MERGE (la)-[:REL_TYPE {rel_prop: 'abc'}]-> (lc)
        """
    )

    node_properties = query_database(
        driver, NODE_PROPERTIES_QUERY, params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL]}
    )
    relationships_properties = query_database(
        driver, REL_PROPERTIES_QUERY, params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL]}
    )
    relationships = query_database(
        driver, REL_QUERY, params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL]}
    )

    expected_node_properties = [
        {
            "output": {
                "properties": [{"property": "property_a", "type": "STRING"}],
                "labels": "LabelA",
            }
        }
    ]
    expected_relationships_properties = [
        {
            "output": {
                "type": "REL_TYPE",
                "properties": [{"property": "rel_prop", "type": "STRING"}],
            }
        }
    ]
    expected_relationships = [
        {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelB"}},
        {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelC"}},
    ]
    assert node_properties == expected_node_properties
    assert relationships_properties == expected_relationships_properties
    assert (
        sorted(relationships, key=lambda x: x["output"]["end"])
        == expected_relationships
    )

    # Retrieve the schema information
    result = get_schema(driver)
    assert (
        result
        == """Node properties:
LabelA {property_a: STRING}
Relationship properties:
REL_TYPE {rel_prop: STRING}
The relationships:
(:LabelA)-[:REL_TYPE]->(:LabelB)
(:LabelA)-[:REL_TYPE]->(:LabelC)"""
    )


@pytest.mark.usefixtures("setup_neo4j")
def test_get_schema_filtering_labels(driver):
    """Test that the excluded labels and relationships are correctly filtered."""
    # Delete all nodes in the graph
    driver.execute_query("MATCH (n) DETACH DELETE n")
    # Create two labels and a relationship
    driver.execute_query(
        "CREATE (:`_Bloom_Scene_`{property_b: 'b'})-[:_Bloom_HAS_SCENE_]->(:`_Bloom_Perspective_`)"
    )

    node_properties = query_database(
        driver, NODE_PROPERTIES_QUERY, params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL]}
    )
    relationships_properties = query_database(
        driver, REL_PROPERTIES_QUERY, params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL]}
    )

    assert node_properties == []
    assert relationships_properties == []
