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

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from neo4j import Driver, Query
from neo4j_graphrag.schema import (
    BASE_ENTITY_LABEL,
    BASE_KG_BUILDER_LABEL,
    EXCLUDED_LABELS,
    EXCLUDED_RELS,
    INDEX_QUERY,
    LIST_LIMIT,
    NODE_PROPERTIES_QUERY,
    REL_PROPERTIES_QUERY,
    REL_QUERY,
    _value_sanitize,
    format_schema,
    get_enhanced_schema_cypher,
    get_schema,
    get_structured_schema,
)


def _query_return_value(*args: Any, **kwargs: Any) -> list[Any]:
    query = kwargs.get("query", args[1] if len(args) > 1 else "")
    if NODE_PROPERTIES_QUERY in query:
        return [
            {
                "output": {
                    "properties": [{"property": "property_a", "type": "STRING"}],
                    "label": "LabelA",
                }
            }
        ]
    if REL_PROPERTIES_QUERY in query:
        return [
            {
                "output": {
                    "type": "REL_TYPE",
                    "properties": [{"property": "rel_prop", "type": "STRING"}],
                }
            }
        ]
    if REL_QUERY in query:
        return [
            {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelB"}},
            {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelC"}},
        ]
    if "SHOW CONSTRAINTS" == query:
        return ["fake constraints"]
    if INDEX_QUERY == query:
        return ["fake indexes"]

    raise AssertionError("Unexpected query")


@patch("neo4j_graphrag.schema.query_database", side_effect=_query_return_value)
def test_get_schema_ensure_formatted_response(driver: Driver) -> None:
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


def test_get_structured_schema_happy_path(driver: MagicMock) -> None:
    get_structured_schema(driver)
    assert 5 == driver.execute_query.call_count
    calls = driver.execute_query.call_args_list

    args, kwargs = calls[0]
    query_obj = args[0]
    assert isinstance(query_obj, Query)
    assert query_obj.text == NODE_PROPERTIES_QUERY
    assert query_obj.timeout is None
    assert kwargs["database_"] is None
    assert kwargs["parameters_"] == {
        "EXCLUDED_LABELS": EXCLUDED_LABELS + [BASE_ENTITY_LABEL, BASE_KG_BUILDER_LABEL],
        "SAMPLE": 1000,
    }

    args, kwargs = calls[1]
    query_obj = args[0]
    assert isinstance(query_obj, Query)
    assert query_obj.text == REL_PROPERTIES_QUERY
    assert query_obj.timeout is None
    assert kwargs["database_"] is None
    assert kwargs["parameters_"] == {"EXCLUDED_LABELS": EXCLUDED_RELS, "SAMPLE": 1000}

    args, kwargs = calls[2]
    query_obj = args[0]
    assert isinstance(query_obj, Query)
    assert query_obj.text == REL_QUERY
    assert query_obj.timeout is None
    assert kwargs["database_"] is None
    assert kwargs["parameters_"] == {
        "EXCLUDED_LABELS": EXCLUDED_LABELS + [BASE_ENTITY_LABEL, BASE_KG_BUILDER_LABEL],
        "SAMPLE": 1000,
    }

    args, kwargs = calls[3]
    query_obj = args[0]
    assert isinstance(query_obj, Query)
    assert query_obj.text == "SHOW CONSTRAINTS"
    assert query_obj.timeout is None
    assert kwargs["database_"] is None
    assert kwargs["parameters_"] == {}

    args, kwargs = calls[4]
    query_obj = args[0]
    assert isinstance(query_obj, Query)
    assert query_obj.text == INDEX_QUERY
    assert query_obj.timeout is None
    assert kwargs["database_"] is None
    assert kwargs["parameters_"] == {}


@patch("neo4j_graphrag.schema.query_database", side_effect=_query_return_value)
def test_get_schema_ensure_structured_response(driver: MagicMock) -> None:
    result = get_structured_schema(driver)

    assert result["node_props"]["LabelA"] == [
        {"property": "property_a", "type": "STRING"}
    ]
    assert result["rel_props"]["REL_TYPE"] == [
        {"property": "rel_prop", "type": "STRING"}
    ]
    assert result["relationships"] == [
        {"end": "LabelB", "start": "LabelA", "type": "REL_TYPE"},
        {"end": "LabelC", "start": "LabelA", "type": "REL_TYPE"},
    ]
    assert result["metadata"]["constraint"] == ["fake constraints"]
    assert result["metadata"]["index"] == ["fake indexes"]


@pytest.mark.parametrize(
    "description, input_value, expected_output",
    [
        (
            "Small list",
            {"key1": "value1", "small_list": list(range(15))},
            {"key1": "value1", "small_list": list(range(15))},
        ),
        (
            "Oversized list",
            {"key1": "value1", "oversized_list": list(range(LIST_LIMIT + 1))},
            {"key1": "value1"},
        ),
        (
            "Nested oversized list",
            {"key1": "value1", "oversized_list": {"key": list(range(150))}},
            {"key1": "value1", "oversized_list": {}},
        ),
        (
            "Dict in list",
            {
                "key1": "value1",
                "oversized_list": [1, 2, {"key": list(range(LIST_LIMIT + 1))}],
            },
            {"key1": "value1", "oversized_list": [1, 2, {}]},
        ),
        (
            "Dict in nested list",
            {
                "key1": "value1",
                "deeply_nested_lists": [
                    [[[{"final_nested_key": list(range(LIST_LIMIT + 1))}]]]
                ],
            },
            {"key1": "value1", "deeply_nested_lists": [[[[{}]]]]},
        ),
        (
            "Bare oversized list",
            list(range(LIST_LIMIT + 1)),
            None,
        ),
        (
            "None value",
            None,
            None,
        ),
    ],
)
def test__value_sanitize(
    description: str, input_value: Dict[str, Any], expected_output: Any
) -> None:
    """Test the _value_sanitize function."""
    assert (
        _value_sanitize(input_value) == expected_output
    ), f"Failed test case: {description}"


@pytest.mark.parametrize(
    "description, schema, is_enhanced, expected_output",
    [
        (
            "Enhanced, string property with high distinct count",
            {
                "node_props": {
                    "Person": [
                        {
                            "property": "name",
                            "type": "STRING",
                            "values": ["Alice", "Bob", "Charlie"],
                            "distinct_count": 11,
                        }
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Person**\n"
                '  - `name`: STRING Example: "Alice"\n'
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, string property with low distinct count",
            {
                "node_props": {
                    "Animal": [
                        {
                            "property": "species",
                            "type": "STRING",
                            "values": ["Cat", "Dog"],
                            "distinct_count": 2,
                        }
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Animal**\n"
                "  - `species`: STRING Available options: ['Cat', 'Dog']\n"
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, numeric property with min and max",
            {
                "node_props": {
                    "Person": [
                        {"property": "age", "type": "INTEGER", "min": 20, "max": 70}
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Person**\n"
                "  - `age`: INTEGER Min: 20, Max: 70\n"
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, numeric property with values",
            {
                "node_props": {
                    "Event": [
                        {
                            "property": "date",
                            "type": "DATE",
                            "values": ["2021-01-01", "2021-01-02"],
                        }
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Event**\n"
                '  - `date`: DATE Example: "2021-01-01"\n'
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, list property that should be skipped",
            {
                "node_props": {
                    "Document": [
                        {
                            "property": "embedding",
                            "type": "LIST",
                            "min_size": 150,
                            "max_size": 200,
                        }
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Document**\n"
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, list property that should be included",
            {
                "node_props": {
                    "Document": [
                        {
                            "property": "keywords",
                            "type": "LIST",
                            "min_size": 2,
                            "max_size": 5,
                        }
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Document**\n"
                "  - `keywords`: LIST Min Size: 2, Max Size: 5\n"
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, relationship string property with high distinct count",
            {
                "node_props": {},
                "rel_props": {
                    "KNOWS": [
                        {
                            "property": "since",
                            "type": "STRING",
                            "values": ["2000", "2001", "2002"],
                            "distinct_count": 15,
                        }
                    ]
                },
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "\n"
                "Relationship properties:\n"
                "- **KNOWS**\n"
                '  - `since`: STRING Example: "2000"\n'
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, relationship string property with low distinct count",
            {
                "node_props": {},
                "rel_props": {
                    "LIKES": [
                        {
                            "property": "intensity",
                            "type": "STRING",
                            "values": ["High", "Medium", "Low"],
                            "distinct_count": 3,
                        }
                    ]
                },
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "\n"
                "Relationship properties:\n"
                "- **LIKES**\n"
                "  - `intensity`: STRING Available options: ['High', 'Medium', 'Low']\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, relationship numeric property with min and max",
            {
                "node_props": {},
                "rel_props": {
                    "WORKS_WITH": [
                        {
                            "property": "since",
                            "type": "INTEGER",
                            "min": 1995,
                            "max": 2020,
                        }
                    ]
                },
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "\n"
                "Relationship properties:\n"
                "- **WORKS_WITH**\n"
                "  - `since`: INTEGER Min: 1995, Max: 2020\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, relationship list property that should be skipped",
            {
                "node_props": {},
                "rel_props": {
                    "KNOWS": [
                        {
                            "property": "embedding",
                            "type": "LIST",
                            "min_size": 150,
                            "max_size": 200,
                        }
                    ]
                },
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "\n"
                "Relationship properties:\n"
                "- **KNOWS**\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, relationship list property that should be included",
            {
                "node_props": {},
                "rel_props": {
                    "KNOWS": [
                        {
                            "property": "messages",
                            "type": "LIST",
                            "min_size": 2,
                            "max_size": 5,
                        }
                    ]
                },
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "\n"
                "Relationship properties:\n"
                "- **KNOWS**\n"
                "  - `messages`: LIST Min Size: 2, Max Size: 5\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, relationship numeric property without min and max",
            {
                "node_props": {},
                "rel_props": {
                    "OWES": [
                        {
                            "property": "amount",
                            "type": "FLOAT",
                            "values": [3.14, 2.71],
                        }
                    ]
                },
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "\n"
                "Relationship properties:\n"
                "- **OWES**\n"
                '  - `amount`: FLOAT Example: "3.14"\n'
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, property with empty values list",
            {
                "node_props": {
                    "Person": [
                        {
                            "property": "name",
                            "type": "STRING",
                            "values": [],
                            "distinct_count": 15,
                        }
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Person**\n"
                "  - `name`: STRING \n"
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, property with missing values",
            {
                "node_props": {
                    "Person": [
                        {
                            "property": "name",
                            "type": "STRING",
                            "distinct_count": 15,
                        }
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Person**\n"
                "  - `name`: STRING \n"
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
    ],
)
def test_format_schema(
    description: str, schema: Dict[str, Any], is_enhanced: bool, expected_output: str
) -> None:
    result = format_schema(schema, is_enhanced)
    assert result == expected_output, f"Failed test case: {description}"


@pytest.mark.parametrize(
    "description, structured_schema, label_or_type, properties, exhaustive, sample_size, is_relationship, expected_query",
    [
        (
            "Exhaustive, string property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "name", "type": "STRING"}],
            True,
            5,
            False,
            (
                "MATCH (n:`Person`)\n"
                "WITH collect(distinct substring(toString(n.`name`), 0, 50)) AS `name_values`\n"
                "RETURN {`name`: {values: `name_values`[..10], distinct_count: size(`name_values`)}} AS output"
            ),
        ),
        (
            "Non-exhaustive, string property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "name", "type": "STRING"}],
            False,
            5,
            False,
            (
                "MATCH (n:`Person`) WITH n LIMIT 5\n"
                "WITH collect(distinct substring(toString(n.`name`), 0, 50)) AS `name_values`\n"
                "RETURN {`name`: {values: `name_values`}} AS output"
            ),
        ),
        (
            "Exhaustive, integer property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "age", "type": "INTEGER"}],
            True,
            5,
            False,
            (
                "MATCH (n:`Person`)\n"
                "WITH min(n.`age`) AS `age_min`,\n"
                "     max(n.`age`) AS `age_max`,\n"
                "     count(distinct n.`age`) AS `age_distinct`\n"
                "RETURN {`age`: {min: toString(`age_min`), max: toString(`age_max`), distinct_count: `age_distinct`}} AS output"
            ),
        ),
        (
            "Non-exhaustive, integer property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "age", "type": "INTEGER"}],
            False,
            5,
            False,
            (
                "MATCH (n:`Person`) WITH n LIMIT 5\n"
                "WITH collect(distinct toString(n.`age`)) AS `age_values`\n"
                "RETURN {`age`: {values: `age_values`}} AS output"
            ),
        ),
        (
            "Exhaustive, float property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "salary", "type": "FLOAT"}],
            True,
            5,
            False,
            (
                "MATCH (n:`Person`)\n"
                "WITH min(n.`salary`) AS `salary_min`,\n"
                "     max(n.`salary`) AS `salary_max`,\n"
                "     count(distinct n.`salary`) AS `salary_distinct`\n"
                "RETURN {`salary`: {min: toString(`salary_min`), max: toString(`salary_max`), distinct_count: `salary_distinct`}} AS output"
            ),
        ),
        (
            "Non-exhaustive, float property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "salary", "type": "FLOAT"}],
            False,
            5,
            False,
            (
                "MATCH (n:`Person`) WITH n LIMIT 5\n"
                "WITH collect(distinct toString(n.`salary`)) AS `salary_values`\n"
                "RETURN {`salary`: {values: `salary_values`}} AS output"
            ),
        ),
        (
            "Exhaustive, date property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "birth_date", "type": "DATE"}],
            True,
            5,
            False,
            (
                "MATCH (n:`Person`)\n"
                "WITH min(n.`birth_date`) AS `birth_date_min`,\n"
                "     max(n.`birth_date`) AS `birth_date_max`,\n"
                "     count(distinct n.`birth_date`) AS `birth_date_distinct`\n"
                "RETURN {`birth_date`: {min: toString(`birth_date_min`), max: toString(`birth_date_max`), distinct_count: `birth_date_distinct`}} AS output"
            ),
        ),
        (
            "Non-exhaustive, date property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "birth_date", "type": "DATE"}],
            False,
            5,
            False,
            (
                "MATCH (n:`Person`) WITH n LIMIT 5\n"
                "WITH collect(distinct toString(n.`birth_date`)) AS `birth_date_values`\n"
                "RETURN {`birth_date`: {values: `birth_date_values`}} AS output"
            ),
        ),
        (
            "Exhaustive, date time property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "birth_date", "type": "DATE_TIME"}],
            True,
            5,
            False,
            (
                "MATCH (n:`Person`)\n"
                "WITH min(n.`birth_date`) AS `birth_date_min`,\n"
                "     max(n.`birth_date`) AS `birth_date_max`,\n"
                "     count(distinct n.`birth_date`) AS `birth_date_distinct`\n"
                "RETURN {`birth_date`: {min: toString(`birth_date_min`), max: toString(`birth_date_max`), distinct_count: `birth_date_distinct`}} AS output"
            ),
        ),
        (
            "Non-exhaustive, date time property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "birth_date", "type": "DATE_TIME"}],
            False,
            5,
            False,
            (
                "MATCH (n:`Person`) WITH n LIMIT 5\n"
                "WITH collect(distinct toString(n.`birth_date`)) AS `birth_date_values`\n"
                "RETURN {`birth_date`: {values: `birth_date_values`}} AS output"
            ),
        ),
        (
            "Exhaustive, local date time property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "birth_date", "type": "LOCAL_DATE_TIME"}],
            True,
            5,
            False,
            (
                "MATCH (n:`Person`)\n"
                "WITH min(n.`birth_date`) AS `birth_date_min`,\n"
                "     max(n.`birth_date`) AS `birth_date_max`,\n"
                "     count(distinct n.`birth_date`) AS `birth_date_distinct`\n"
                "RETURN {`birth_date`: {min: toString(`birth_date_min`), max: toString(`birth_date_max`), distinct_count: `birth_date_distinct`}} AS output"
            ),
        ),
        (
            "Non-exhaustive, local date time property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "birth_date", "type": "LOCAL_DATE_TIME"}],
            False,
            5,
            False,
            (
                "MATCH (n:`Person`) WITH n LIMIT 5\n"
                "WITH collect(distinct toString(n.`birth_date`)) AS `birth_date_values`\n"
                "RETURN {`birth_date`: {values: `birth_date_values`}} AS output"
            ),
        ),
        (
            "Exhaustive, list property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "tags", "type": "LIST"}],
            True,
            5,
            False,
            (
                "MATCH (n:`Person`)\n"
                "WITH min(size(n.`tags`)) AS `tags_size_min`, max(size(n.`tags`)) AS `tags_size_max`\n"
                "RETURN {`tags`: {min_size: `tags_size_min`, max_size: `tags_size_max`}} AS output"
            ),
        ),
        (
            "Non-exhaustive, list property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "tags", "type": "LIST"}],
            False,
            5,
            False,
            (
                "MATCH (n:`Person`) WITH n LIMIT 5\n"
                "WITH min(size(n.`tags`)) AS `tags_size_min`, max(size(n.`tags`)) AS `tags_size_max`\n"
                "RETURN {`tags`: {min_size: `tags_size_min`, max_size: `tags_size_max`}} AS output"
            ),
        ),
        (
            "Exhaustive, boolean property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "active", "type": "BOOLEAN"}],
            True,
            5,
            False,
            "MATCH (n:`Person`)\nRETURN {} AS output",
        ),
        (
            "Non-exhaustive, boolean property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "active", "type": "BOOLEAN"}],
            False,
            5,
            False,
            "MATCH (n:`Person`) WITH n LIMIT 5\nRETURN {} AS output",
        ),
        (
            "Exhaustive, point property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "location", "type": "POINT"}],
            True,
            5,
            False,
            "MATCH (n:`Person`)\nRETURN {} AS output",
        ),
        (
            "Non-exhaustive, point property",
            {"metadata": {"index": []}},
            "Person",
            [{"property": "location", "type": "POINT"}],
            False,
            5,
            False,
            "MATCH (n:`Person`) WITH n LIMIT 5\nRETURN {} AS output",
        ),
        (
            "Exhaustive, duration property",
            {"metadata": {"index": []}},
            "Journey",
            [{"property": "duration", "type": "DURATION"}],
            True,
            5,
            False,
            ("MATCH (n:`Journey`)\nRETURN {} AS output"),
        ),
        (
            "Non-exhaustive, duration property",
            {"metadata": {"index": []}},
            "Journey",
            [{"property": "duration", "type": "DURATION"}],
            False,
            5,
            False,
            "MATCH (n:`Journey`) WITH n LIMIT 5\nRETURN {} AS output",
        ),
        (
            "Relationship property",
            {"metadata": {"index": []}},
            "FRIENDS_WITH",
            [{"property": "since", "type": "INTEGER"}],
            True,
            5,
            True,
            (
                "MATCH ()-[n:`FRIENDS_WITH`]->()\n"
                "WITH min(n.`since`) AS `since_min`,\n"
                "     max(n.`since`) AS `since_max`,\n"
                "     count(distinct n.`since`) AS `since_distinct`\n"
                "RETURN {`since`: {min: toString(`since_min`), max: toString(`since_max`), distinct_count: `since_distinct`}} AS output"
            ),
        ),
    ],
)
def test_get_enhanced_schema_cypher(
    driver: MagicMock,
    description: str,
    structured_schema: Dict[str, Any],
    label_or_type: str,
    properties: List[Dict[str, Any]],
    exhaustive: bool,
    sample_size: int,
    is_relationship: bool,
    expected_query: str,
) -> None:
    query = get_enhanced_schema_cypher(
        driver=driver,
        structured_schema=structured_schema,
        label_or_type=label_or_type,
        properties=properties,
        exhaustive=exhaustive,
        sample_size=sample_size,
        is_relationship=is_relationship,
    )
    assert query == expected_query, f"Failed test case: {description}"


def test_enhanced_schema_cypher_integer_exhaustive_false_with_index(
    driver: MagicMock,
) -> None:
    structured_schema = {
        "metadata": {
            "index": [
                {
                    "label": "Person",
                    "properties": ["age"],
                    "type": "RANGE",
                }
            ]
        }
    }
    properties = [{"property": "age", "type": "INTEGER"}]
    query = get_enhanced_schema_cypher(
        driver=driver,
        structured_schema=structured_schema,
        label_or_type="Person",
        properties=properties,
        exhaustive=False,
    )
    assert query == (
        "MATCH (n:`Person`) WITH n LIMIT 5\n"
        "WITH min(n.`age`) AS `age_min`,\n"
        "     max(n.`age`) AS `age_max`,\n"
        "     count(distinct n.`age`) AS `age_distinct`\n"
        "RETURN {`age`: {min: toString(`age_min`), max: toString(`age_max`), distinct_count: `age_distinct`}} AS output"
    )


@patch("neo4j_graphrag.schema.query_database")
def test_enhanced_schema_cypher_string_exhaustive_false_with_index(
    query_database_mock: MagicMock,
    driver: MagicMock,
) -> None:
    structured_schema = {
        "metadata": {
            "index": [
                {
                    "label": "Person",
                    "properties": ["status"],
                    "type": "RANGE",
                    "size": 5,
                    "distinctValues": 5,
                }
            ]
        }
    }
    query_database_mock.return_value = [{"value": ["Single", "Married", "Divorced"]}]
    properties = [{"property": "status", "type": "STRING"}]
    query = get_enhanced_schema_cypher(
        driver=driver,
        structured_schema=structured_schema,
        label_or_type="Person",
        properties=properties,
        exhaustive=False,
    )
    assert query == (
        "MATCH (n:`Person`) WITH n LIMIT 5\n\n"
        "RETURN {`status`: {values: ['Single', 'Married', 'Divorced'], distinct_count: 3}} AS output"
    )
