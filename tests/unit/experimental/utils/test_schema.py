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
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from neo4j_viz import VisualizationGraph
from neo4j_graphrag.experimental.components.schema import GraphSchema
from neo4j_graphrag.experimental.utils.schema import schema_visualization


@pytest.fixture(scope="module")
def valid_schema_dict() -> dict[str, Any]:
    return {
        "node_types": [
            "Location",
            {
                "label": "Person",
                "properties": [
                    {"name": "name", "type": "STRING", "required": True},
                    {"name": "birthYear", "type": "INTEGER"},
                ],
            },
        ],
        "relationship_types": [
            "BORN_IN",
            {
                "label": "KNOWS",
                "properties": [
                    {"name": "since", "type": "LOCAL_DATETIME"},
                ],
            },
        ],
        "patterns": [
            ("Person", "BORN_IN", "Location"),
            ("Person", "KNOWS", "Person"),
        ],
    }


@pytest.fixture(scope="module")
def invalid_schema_dict() -> dict[str, Any]:
    return {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {"name": "name", "type": "STRING", "required": True},
                    {"name": "birthYear", "type": "INTEGER"},
                ],
            },
        ],
        "relationship_types": [
            "BORN_IN",
        ],
        "patterns": [
            (
                "Person",
                "BORN_IN",
                "Location",
            ),  # invalid pattern, "Location" node type not defined
        ],
    }


@patch("neo4j_graphrag.experimental.utils.schema.neo4j_viz", None)
def test_schema_visualization_import_error() -> None:
    with pytest.raises(ImportError):
        schema_visualization({})


def test_schema_visualization_invalid_schema_dict(
    invalid_schema_dict: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        schema_visualization(invalid_schema_dict)


def test_schema_visualization_valid_schema_dict(
    valid_schema_dict: dict[str, Any],
) -> None:
    g = schema_visualization(valid_schema_dict)
    assert isinstance(g, VisualizationGraph)
    assert len(g.nodes) == 2
    assert len(g.relationships) == 2


def test_schema_visualization_schema_object(valid_schema_dict: dict[str, Any]) -> None:
    schema = GraphSchema.model_validate(valid_schema_dict)
    g = schema_visualization(schema)
    assert isinstance(g, VisualizationGraph)
    assert len(g.nodes) == 2
    assert len(g.relationships) == 2
