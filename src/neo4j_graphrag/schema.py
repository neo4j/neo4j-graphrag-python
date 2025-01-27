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

from typing import Any, Dict, Optional

import neo4j
from neo4j.exceptions import ClientError

BASE_KG_BUILDER_LABEL = "__KGBuilder__"
BASE_ENTITY_LABEL = "__Entity__"
EXCLUDED_LABELS = ["_Bloom_Perspective_", "_Bloom_Scene_"]
EXCLUDED_RELS = ["_Bloom_HAS_SCENE_"]
EXHAUSTIVE_SEARCH_LIMIT = 10000
LIST_LIMIT = 128
# Threshold for returning all available prop values in graph schema
DISTINCT_VALUE_LIMIT = 10

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

INDEX_QUERY = """
CALL apoc.schema.nodes() YIELD label, properties, type, size, valuesSelectivity
WHERE type = "RANGE" RETURN *,
size * valuesSelectivity as distinctValues
"""


def clean_string_values(text: str) -> str:
    """Clean string values for schema.

    Cleans the input text by replacing newline and carriage return characters.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    return text.replace("\n", " ").replace("\r", " ")


def query_database(
    driver: neo4j.Driver, query: str, params: Optional[dict[str, Any]] = None
) -> list[dict[str, Any]]:
    """
    Queries the database.

    Args:
        driver (neo4j.Driver):  Neo4j Python driver instance.
        query (str): The cypher query.
        params (dict, optional): The query parameters. Defaults to None.

    Returns:
        list[dict[str, Any]]: the result of the query in json format.
    """
    if params is None:
        params = {}
    data = driver.execute_query(query, params)
    return [r.data() for r in data.records]


def get_schema(driver: neo4j.Driver, is_enhanced: bool = False) -> str:
    """
    Returns the schema of the graph as a string with following format:

    .. code-block:: text

        Node properties:
        Person {id: INTEGER, name: STRING}
        Relationship properties:
        KNOWS {fromDate: DATE}
        The relationships:
        (:Person)-[:KNOWS]->(:Person)

    Args:
        driver (neo4j.Driver): Neo4j Python driver instance.

    Returns:
        str: the graph schema information in a serialized format.

    """
    structured_schema = get_structured_schema(driver)

    return format_schema(structured_schema, is_enhanced)


def get_structured_schema(driver: neo4j.Driver) -> dict[str, Any]:
    """
    Returns the structured schema of the graph.

    Returns a dict with following format:

    .. code:: python

        {
            'node_props': {
                'Person': [{'property': 'id', 'type': 'INTEGER'}, {'property': 'name', 'type': 'STRING'}]
            },
            'rel_props': {
                'KNOWS': [{'property': 'fromDate', 'type': 'DATE'}]
            },
            'relationships': [
                {'start': 'Person', 'type': 'KNOWS', 'end': 'Person'}
            ],
            'metadata': {
                'constraint': [
                    {'id': 7, 'name': 'person_id', 'type': 'UNIQUENESS', 'entityType': 'NODE', 'labelsOrTypes': ['Persno'], 'properties': ['id'], 'ownedIndex': 'person_id', 'propertyType': None},
                ],
                'index': [
                    {'label': 'Person', 'properties': ['name'], 'size': 2, 'type': 'RANGE', 'valuesSelectivity': 1.0, 'distinctValues': 2.0},
                ]
            }
        }

    Note:
        The internal structure of the returned dict depends on the apoc.meta.data
        and apoc.schema.nodes procedures.

    Warning:
        Some labels are excluded from the output schema:

        - The `__Entity__` and `__KGBuilder__` node labels which are created by the KG Builder pipeline within this package
        - Some labels related to Bloom internals.

    Args:
        driver (neo4j.Driver): Neo4j Python driver instance.

    Returns:
        dict[str, Any]: the graph schema information in a structured format.
    """
    node_properties = [
        data["output"]
        for data in query_database(
            driver,
            NODE_PROPERTIES_QUERY,
            params={
                "EXCLUDED_LABELS": EXCLUDED_LABELS
                + [BASE_ENTITY_LABEL, BASE_KG_BUILDER_LABEL]
            },
        )
    ]

    rel_properties = [
        data["output"]
        for data in query_database(
            driver, REL_PROPERTIES_QUERY, params={"EXCLUDED_LABELS": EXCLUDED_RELS}
        )
    ]

    relationships = [
        data["output"]
        for data in query_database(
            driver,
            REL_QUERY,
            params={
                "EXCLUDED_LABELS": EXCLUDED_LABELS
                + [BASE_ENTITY_LABEL, BASE_KG_BUILDER_LABEL]
            },
        )
    ]

    # Get constraints and indexes
    try:
        constraint = query_database(driver, "SHOW CONSTRAINTS")
        index = query_database(driver, INDEX_QUERY)
    except ClientError:
        constraint = []
        index = []

    return {
        "node_props": {el["labels"]: el["properties"] for el in node_properties},
        "rel_props": {el["type"]: el["properties"] for el in rel_properties},
        "relationships": relationships,
        "metadata": {"constraint": constraint, "index": index},
    }


def format_schema(schema: Dict[str, Any], is_enhanced: bool) -> str:
    formatted_node_props = []
    formatted_rel_props = []
    if is_enhanced:
        # Enhanced formatting for nodes
        for node_type, properties in schema["node_props"].items():
            formatted_node_props.append(f"- **{node_type}**")
            for prop in properties:
                example = ""
                if prop["type"] == "STRING" and prop.get("values"):
                    if prop.get("distinct_count", 11) > DISTINCT_VALUE_LIMIT:
                        example = (
                            f'Example: "{clean_string_values(prop["values"][0])}"'
                            if prop["values"]
                            else ""
                        )
                    else:  # If less than 10 possible values return all
                        example = (
                            (
                                "Available options: "
                                f'{[clean_string_values(el) for el in prop["values"]]}'
                            )
                            if prop["values"]
                            else ""
                        )

                elif prop["type"] in [
                    "INTEGER",
                    "FLOAT",
                    "DATE",
                    "DATE_TIME",
                    "LOCAL_DATE_TIME",
                ]:
                    if prop.get("min") and prop.get("max"):
                        example = f'Min: {prop["min"]}, Max: {prop["max"]}'
                    else:
                        example = (
                            f'Example: "{prop["values"][0]}"'
                            if prop.get("values")
                            else ""
                        )
                elif prop["type"] == "LIST":
                    # Skip embeddings
                    if not prop.get("min_size") or prop["min_size"] > LIST_LIMIT:
                        continue
                    example = (
                        f'Min Size: {prop["min_size"]}, Max Size: {prop["max_size"]}'
                    )
                formatted_node_props.append(
                    f"  - `{prop['property']}`: {prop['type']} {example}"
                )

        # Enhanced formatting for relationships
        for rel_type, properties in schema["rel_props"].items():
            formatted_rel_props.append(f"- **{rel_type}**")
            for prop in properties:
                example = ""
                if prop["type"] == "STRING" and prop.get("values"):
                    if prop.get("distinct_count", 11) > DISTINCT_VALUE_LIMIT:
                        example = (
                            f'Example: "{clean_string_values(prop["values"][0])}"'
                            if prop["values"]
                            else ""
                        )
                    else:  # If less than 10 possible values return all
                        example = (
                            (
                                "Available options: "
                                f'{[clean_string_values(el) for el in prop["values"]]}'
                            )
                            if prop["values"]
                            else ""
                        )
                elif prop["type"] in [
                    "INTEGER",
                    "FLOAT",
                    "DATE",
                    "DATE_TIME",
                    "LOCAL_DATE_TIME",
                ]:
                    if prop.get("min") and prop.get("max"):  # If we have min/max
                        example = f'Min: {prop["min"]}, Max: {prop["max"]}'
                    else:  # return a single value
                        example = (
                            f'Example: "{prop["values"][0]}"' if prop["values"] else ""
                        )
                elif prop["type"] == "LIST":
                    # Skip embeddings
                    if not prop.get("min_size") or prop["min_size"] > LIST_LIMIT:
                        continue
                    example = (
                        f'Min Size: {prop["min_size"]}, Max Size: {prop["max_size"]}'
                    )
                formatted_rel_props.append(
                    f"  - `{prop['property']}`: {prop['type']} {example}"
                )
    else:
        # Format node properties
        for label, props in schema["node_props"].items():
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in props]
            )
            formatted_node_props.append(f"{label} {{{props_str}}}")

        # Format relationship properties using structured_schema
        for type, props in schema["rel_props"].items():
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in props]
            )
            formatted_rel_props.append(f"{type} {{{props_str}}}")

    # Format relationships
    formatted_rels = [
        f"(:{el['start']})-[:{el['type']}]->(:{el['end']})"
        for el in schema["relationships"]
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
