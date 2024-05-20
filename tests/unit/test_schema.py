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
from unittest.mock import patch
from neo4j_genai.schema import (
    get_schema,
    NODE_PROPERTIES_QUERY,
    REL_PROPERTIES_QUERY,
    REL_QUERY,
    EXCLUDED_LABELS,
    BASE_ENTITY_LABEL,
    EXCLUDED_RELS,
)


def _query_return_value(*args, **kwargs):
    if NODE_PROPERTIES_QUERY in args[1]:
        return [
            {
                "output": {
                    "properties": [{"property": "property_a", "type": "STRING"}],
                    "labels": "LabelA",
                }
            }
        ]
    if REL_PROPERTIES_QUERY in args[1]:
        return [
            {
                "output": {
                    "type": "REL_TYPE",
                    "properties": [{"property": "rel_prop", "type": "STRING"}],
                }
            }
        ]
    if REL_QUERY in args[1]:
        return [
            {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelB"}},
            {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelC"}},
        ]

    raise AssertionError("Unexpected query")


def test_get_schema_happy_path(driver):
    get_schema(driver)
    assert 3 == driver.execute_query.call_count
    driver.execute_query.assert_any_call(
        NODE_PROPERTIES_QUERY,
        {"EXCLUDED_LABELS": EXCLUDED_LABELS + [BASE_ENTITY_LABEL]},
    )
    driver.execute_query.assert_any_call(
        REL_PROPERTIES_QUERY,
        {"EXCLUDED_LABELS": EXCLUDED_RELS},
    )
    driver.execute_query.assert_any_call(
        REL_QUERY,
        {"EXCLUDED_LABELS": EXCLUDED_LABELS + [BASE_ENTITY_LABEL]},
    )


@patch("neo4j_genai.schema._query_database", side_effect=_query_return_value)
def test_get_schema_ensure_formatted_response(driver):
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
