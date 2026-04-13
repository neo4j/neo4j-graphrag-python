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

"""Explicit coverage for :class:`PropertyType` ``required`` deprecation (see EXISTENCE constraints)."""

from __future__ import annotations

import pytest

from neo4j_graphrag.experimental.components.schema import GraphSchema, PropertyType


def test_property_type_required_field_emits_deprecation_on_access() -> None:
    """Reading ``.required`` on a model instance should warn (Pydantic deprecated field)."""
    prop = PropertyType(name="x", type="STRING", required=True)
    with pytest.warns(DeprecationWarning, match="EXISTENCE"):
        assert prop.required is True


def test_legacy_required_true_in_json_becomes_existence_constraint() -> None:
    """Loading JSON with ``required: true`` migrates to EXISTENCE (avoid asserting ``.required``; any read warns)."""
    schema = GraphSchema.model_validate(
        {
            "node_types": [
                {
                    "label": "Person",
                    "properties": [
                        {"name": "name", "type": "STRING", "required": True},
                    ],
                }
            ],
        }
    )
    assert schema.existence_property_names_for_node("Person") == {"name"}
    assert any(
        c.type == "EXISTENCE" and c.node_type == "Person" and c.property_name == "name"
        for c in schema.constraints
    )
