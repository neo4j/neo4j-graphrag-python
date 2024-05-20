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
from typing import Any

import neo4j


BASE_ENTITY_LABEL = "__Entity__"
EXCLUDED_LABELS = ["_Bloom_Perspective_", "_Bloom_Scene_"]
EXCLUDED_RELS = ["_Bloom_HAS_SCENE_"]

NODE_PROPERTIES_QUERY = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "node" 
    AND NOT label IN $EXCLUDED_LABELS
WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
RETURN {labels: nodeLabels, properties: properties} AS output
"""

REL_PROPERTIES_QUERY = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
    AND NOT label in $EXCLUDED_LABELS
WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
RETURN {type: nodeLabels, properties: properties} AS output
"""

REL_QUERY = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE type = "RELATIONSHIP" AND elementType = "node"
UNWIND other AS other_node
WITH * WHERE NOT label IN $EXCLUDED_LABELS
    AND NOT other_node IN $EXCLUDED_LABELS
RETURN {start: label, type: property, end: toString(other_node)} AS output
"""


def _query_database(
    driver: neo4j.Driver, query: str, params: dict = {}
) -> list[dict[str, Any]]:
    """
    Queries the database.

    Args:
        driver (neo4j.Driver):  Neo4j Python driver instance.
        query (str): The cypher query.
        params (dict, optional): The query parameters. Defaults to {}.

    Returns:
        List[Dict[str, Any]]: the result of the query in json format.
    """
    data = driver.execute_query(query, params)
    return [r.data() for r in data.records]


def get_schema(
    driver: neo4j.Driver,
) -> str:
    """
    Returns the schema of the graph.

    Args:
        driver (neo4j.Driver): Neo4j Python driver instance.

    Returns:
        str: the graph schema information in a serialized format.
    """
    node_properties = [
        data["output"]
        for data in _query_database(
            driver,
            NODE_PROPERTIES_QUERY,
            params={"EXCLUDED_LABELS": EXCLUDED_LABELS + [BASE_ENTITY_LABEL]},
        )
    ]

    rel_properties = [
        data["output"]
        for data in _query_database(
            driver, REL_PROPERTIES_QUERY, params={"EXCLUDED_LABELS": EXCLUDED_RELS}
        )
    ]
    relationships = [
        data["output"]
        for data in _query_database(
            driver,
            REL_QUERY,
            params={"EXCLUDED_LABELS": EXCLUDED_LABELS + [BASE_ENTITY_LABEL]},
        )
    ]

    # Format node properties
    formatted_node_props = []
    for element in node_properties:
        props_str = ", ".join(
            [f"{prop['property']}: {prop['type']}" for prop in element["properties"]]
        )
        formatted_node_props.append(f"{element['labels']} {{{props_str}}}")

    # Format relationship properties
    formatted_rel_props = []
    for element in rel_properties:
        props_str = ", ".join(
            [f"{prop['property']}: {prop['type']}" for prop in element["properties"]]
        )
        formatted_rel_props.append(f"{element['type']} {{{props_str}}}")

    # Format relationships
    formatted_rels = [
        f"(:{element['start']})-[:{element['type']}]->(:{element['end']})"
        for element in relationships
    ]

    return "\n".join(
        [
            "Node properties:",
            "\n".join(formatted_node_props),
            "Relationship properties:",
            "\n".join(formatted_rel_props),
            "The relationships:",
            "\n".join(formatted_rels),
        ]
    )
