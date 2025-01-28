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

from typing import Any, Dict, List, Optional

import neo4j
from neo4j.exceptions import ClientError, CypherTypeError

BASE_KG_BUILDER_LABEL = "__KGBuilder__"
BASE_ENTITY_LABEL = "__Entity__"
EXCLUDED_LABELS = ["_Bloom_Perspective_", "_Bloom_Scene_"]
EXCLUDED_RELS = ["_Bloom_HAS_SCENE_"]
EXHAUSTIVE_SEARCH_LIMIT = 10000
LIST_LIMIT = 128
DISTINCT_VALUE_LIMIT = 10

NODE_PROPERTIES_QUERY = (
    "CALL apoc.meta.data() "
    "YIELD label, other, elementType, type, property "
    "WHERE NOT type = 'RELATIONSHIP' AND elementType = 'node' "
    "AND NOT label IN $EXCLUDED_LABELS "
    "WITH label AS nodeLabels, collect({property:property, type:type}) AS properties "
    "RETURN {labels: nodeLabels, properties: properties} AS output"
)

REL_PROPERTIES_QUERY = (
    "CALL apoc.meta.data() "
    "YIELD label, other, elementType, type, property "
    "WHERE NOT type = 'RELATIONSHIP' AND elementType = 'relationship' "
    "AND NOT label in $EXCLUDED_LABELS "
    "WITH label AS nodeLabels, collect({property:property, type:type}) AS properties "
    "RETURN {type: nodeLabels, properties: properties} AS output"
)

REL_QUERY = (
    "CALL apoc.meta.data() "
    "YIELD label, other, elementType, type, property "
    "WHERE type = 'RELATIONSHIP' AND elementType = 'node' "
    "UNWIND other AS other_node "
    "WITH * WHERE NOT label IN $EXCLUDED_LABELS "
    "AND NOT other_node IN $EXCLUDED_LABELS "
    "RETURN {start: label, type: property, end: toString(other_node)} AS output"
)

INDEX_QUERY = (
    "CALL apoc.schema.nodes() YIELD label, properties, type, size, valuesSelectivity "
    "WHERE type = 'RANGE' RETURN *, "
    "size * valuesSelectivity as distinctValues"
)

SCHEMA_COUNTS_QUERY = (
    "CALL apoc.meta.graph({sample: 1000, maxRels: 100}) "
    "YIELD nodes, relationships "
    "RETURN nodes, [rel in relationships | {name:apoc.any.property"
    "(rel, 'type'), count: apoc.any.property(rel, 'count')}]"
    " AS relationships"
)


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
    if is_enhanced:
        get_enhanced_schema(driver, structured_schema)

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
    """
    Format the structured schema into a human-readable string.

    Depending on the `is_enhanced` flag, this function either creates a concise
    listing of node labels and relationship types alongside their properties or
    generates an enhanced, more verbose representation with additional details like
    example or available values and min/max statistics. It also includes a formatted
    list of existing relationships.

    Args:
        schema (Dict[str, Any]): The structured schema dictionary, containing
            properties for nodes and relationships as well as relationship definitions.
        is_enhanced (bool): Flag indicating whether to format the schema with
            detailed statistics (True) or in a simpler overview format (False).

    Returns:
        str: A formatted string representation of the graph schema, including
        node properties, relationship properties, and relationship patterns.
    """
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


def get_enhanced_schema_cypher(
    driver: neo4j.Driver,
    structured_schema: Dict[str, Any],
    label_or_type: str,
    properties: List[Dict[str, Any]],
    exhaustive: bool,
    is_relationship: bool = False,
) -> str:
    """
    Build a Cypher query for enhanced schema information.

    Constructs and returns a Cypher query string to gather detailed property
    statistics for either nodes or relationships. Depending on whether the target
    entities are below a certain threshold, it may collect exhaustive information
    or simply sample a few records. This query retrieves data such as minimum and
    maximum values, distinct value counts, and sample values.

    Args:
        driver (neo4j.Driver): Neo4j Python driver instance.
        structured_schema (Dict[str, Any]): The current schema information
            including metadata, indexes, and constraints.
        label_or_type (str): The node label or relationship type to query.
        properties (List[Dict[str, Any]]): A list of property definitions for
            the node label or relationship type.
        exhaustive (bool): Whether to perform an exhaustive search or a
            sampled query approach.
        is_relationship (bool, optional): Indicates if the query is for
            a relationship type (True) or a node label (False). Defaults to False.

    Returns:
        str: A Cypher query string that gathers enhanced property metadata.
    """
    if is_relationship:
        match_clause = f"MATCH ()-[n:`{label_or_type}`]->()"
    else:
        match_clause = f"MATCH (n:`{label_or_type}`)"

    with_clauses = []
    return_clauses = []
    output_dict = {}
    if exhaustive:
        for prop in properties:
            prop_name = prop["property"]
            prop_type = prop["type"]
            if prop_type == "STRING":
                with_clauses.append(
                    (
                        f"collect(distinct substring(toString(n.`{prop_name}`)"
                        f", 0, 50)) AS `{prop_name}_values`"
                    )
                )
                return_clauses.append(
                    (
                        f"values:`{prop_name}_values`[..{DISTINCT_VALUE_LIMIT}],"
                        f" distinct_count: size(`{prop_name}_values`)"
                    )
                )
            elif prop_type in [
                "INTEGER",
                "FLOAT",
                "DATE",
                "DATE_TIME",
                "LOCAL_DATE_TIME",
            ]:
                with_clauses.append(f"min(n.`{prop_name}`) AS `{prop_name}_min`")
                with_clauses.append(f"max(n.`{prop_name}`) AS `{prop_name}_max`")
                with_clauses.append(
                    f"count(distinct n.`{prop_name}`) AS `{prop_name}_distinct`"
                )
                return_clauses.append(
                    (
                        f"min: toString(`{prop_name}_min`), "
                        f"max: toString(`{prop_name}_max`), "
                        f"distinct_count: `{prop_name}_distinct`"
                    )
                )
            elif prop_type == "LIST":
                with_clauses.append(
                    (
                        f"min(size(n.`{prop_name}`)) AS `{prop_name}_size_min`, "
                        f"max(size(n.`{prop_name}`)) AS `{prop_name}_size_max`"
                    )
                )
                return_clauses.append(
                    f"min_size: `{prop_name}_size_min`, "
                    f"max_size: `{prop_name}_size_max`"
                )
            elif prop_type in ["BOOLEAN", "POINT", "DURATION"]:
                continue
            output_dict[prop_name] = "{" + return_clauses.pop() + "}"
    else:
        # Just sample 5 random nodes
        match_clause += " WITH n LIMIT 5"
        for prop in properties:
            prop_name = prop["property"]
            prop_type = prop["type"]

            # Check if indexed property, we can still do exhaustive
            prop_index = [
                el
                for el in structured_schema["metadata"]["index"]
                if el["label"] == label_or_type
                and el["properties"] == [prop_name]
                and el["type"] == "RANGE"
            ]
            if prop_type == "STRING":
                if (
                    prop_index
                    and prop_index[0].get("size") > 0
                    and prop_index[0].get("distinctValues") <= DISTINCT_VALUE_LIMIT
                ):
                    distinct_values = query_database(
                        driver,
                        f"CALL apoc.schema.properties.distinct("
                        f"'{label_or_type}', '{prop_name}') YIELD value",
                    )[0]["value"]
                    return_clauses.append(
                        (
                            f"values: {distinct_values},"
                            f" distinct_count: {len(distinct_values)}"
                        )
                    )
                else:
                    with_clauses.append(
                        (
                            f"collect(distinct substring(toString(n.`{prop_name}`)"
                            f", 0, 50)) AS `{prop_name}_values`"
                        )
                    )
                    return_clauses.append(f"values: `{prop_name}_values`")
            elif prop_type in [
                "INTEGER",
                "FLOAT",
                "DATE",
                "DATE_TIME",
                "LOCAL_DATE_TIME",
            ]:
                if not prop_index:
                    with_clauses.append(
                        f"collect(distinct toString(n.`{prop_name}`)) "
                        f"AS `{prop_name}_values`"
                    )
                    return_clauses.append(f"values: `{prop_name}_values`")
                else:
                    with_clauses.append(f"min(n.`{prop_name}`) AS `{prop_name}_min`")
                    with_clauses.append(f"max(n.`{prop_name}`) AS `{prop_name}_max`")
                    with_clauses.append(
                        f"count(distinct n.`{prop_name}`) AS `{prop_name}_distinct`"
                    )
                    return_clauses.append(
                        (
                            f"min: toString(`{prop_name}_min`), "
                            f"max: toString(`{prop_name}_max`), "
                            f"distinct_count: `{prop_name}_distinct`"
                        )
                    )

            elif prop_type == "LIST":
                with_clauses.append(
                    (
                        f"min(size(n.`{prop_name}`)) AS `{prop_name}_size_min`, "
                        f"max(size(n.`{prop_name}`)) AS `{prop_name}_size_max`"
                    )
                )
                return_clauses.append(
                    (
                        f"min_size: `{prop_name}_size_min`, "
                        f"max_size: `{prop_name}_size_max`"
                    )
                )
            elif prop_type in ["BOOLEAN", "POINT", "DURATION"]:
                continue

            output_dict[prop_name] = "{" + return_clauses.pop() + "}"

    with_clause = "WITH " + ",\n     ".join(with_clauses)
    return_clause = (
        "RETURN {"
        + ", ".join(f"`{k}`: {v}" for k, v in output_dict.items())
        + "} AS output"
    )

    # Combine all parts of the Cypher query
    cypher_query = "\n".join([match_clause, with_clause, return_clause])
    return cypher_query


def get_enhanced_schema(
    driver: neo4j.Driver, structured_schema: Dict[str, Any]
) -> None:
    """
    Enhance the structured schema with detailed property statistics.

    For each node label and relationship type in the structured schema, this
    function queries the database to gather additional property statistics such
    as minimum and maximum values, distinct value counts, and sample values.
    These statistics are then merged into the provided structured schema
    dictionary.

    Args:
        driver (neo4j.Driver): Neo4j Python driver instance.
        structured_schema (Dict[str, Any]): The initial structured schema
            containing node and relationship properties, which will be updated
            with enhanced statistics.

    Returns:
        None
    """
    schema_counts = query_database(driver, SCHEMA_COUNTS_QUERY)
    # Update node info
    for node in schema_counts[0]["nodes"]:
        # Skip bloom labels
        if node["name"] in EXCLUDED_LABELS:
            continue
        node_props = structured_schema["node_props"].get(node["name"])
        if not node_props:  # The node has no properties
            continue
        enhanced_cypher = get_enhanced_schema_cypher(
            driver,
            structured_schema,
            node["name"],
            node_props,
            node["count"] < EXHAUSTIVE_SEARCH_LIMIT,
        )
        # Due to schema-flexible nature of neo4j errors can happen
        try:
            enhanced_info = query_database(
                driver,
                enhanced_cypher,
                # Disable the
                # Neo.ClientNotification.Statement.AggregationSkippedNull
                # notifications raised by the use of collect in the enhanced
                # schema query
                params={"notifications_disabled_categories": ["UNRECOGNIZED"]},
            )[0]["output"]
            for prop in node_props:
                if prop["property"] in enhanced_info:
                    prop.update(enhanced_info[prop["property"]])
        except CypherTypeError:
            continue
    # Update rel info
    for rel in schema_counts[0]["relationships"]:
        # Skip bloom labels
        if rel["name"] in EXCLUDED_RELS:
            continue
        rel_props = structured_schema["rel_props"].get(rel["name"])
        if not rel_props:  # The rel has no properties
            continue
        enhanced_cypher = get_enhanced_schema_cypher(
            driver,
            structured_schema,
            rel["name"],
            rel_props,
            rel["count"] < EXHAUSTIVE_SEARCH_LIMIT,
            is_relationship=True,
        )
        try:
            enhanced_info = query_database(driver, enhanced_cypher)[0]["output"]
            for prop in rel_props:
                if prop["property"] in enhanced_info:
                    prop.update(enhanced_info[prop["property"]])
        # Due to schema-flexible nature of neo4j errors can happen
        except CypherTypeError:
            continue
