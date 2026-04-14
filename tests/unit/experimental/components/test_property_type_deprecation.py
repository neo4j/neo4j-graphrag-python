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

from neo4j_graphrag.experimental.components.schema import (
    ConstraintType,
    GraphConstraintType,
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)


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


def test_programmatic_node_type_required_migrates_to_existence() -> None:
    """``GraphSchema(node_types=(NodeType(..., required=True),))`` migrates like dict input."""
    nt = NodeType(
        label="Person",
        properties=[PropertyType(name="name", type="STRING", required=True)],
    )
    schema = GraphSchema(node_types=(nt,))
    assert schema.existence_property_names_for_node("Person") == {"name"}
    assert len(schema.constraints) == 1
    assert schema.constraints[0].type == GraphConstraintType.EXISTENCE
    assert schema.constraints[0].node_type == "Person"
    assert schema.constraints[0].property_name == "name"


def test_programmatic_relationship_type_required_migrates_to_existence() -> None:
    """``RelationshipType`` with ``PropertyType(required=True)`` migrates to relationship-scoped EXISTENCE."""
    person = NodeType(
        label="Person",
        properties=[PropertyType(name="id", type="STRING")],
    )
    knows = RelationshipType(
        label="KNOWS",
        properties=[
            PropertyType(name="since", type="LOCAL_DATETIME", required=True),
        ],
    )
    schema = GraphSchema(node_types=(person,), relationship_types=(knows,))
    assert schema.existence_property_names_for_relationship("KNOWS") == {"since"}
    assert any(
        c.type == GraphConstraintType.EXISTENCE
        and c.relationship_type == "KNOWS"
        and c.property_name == "since"
        for c in schema.constraints
    )


def test_programmatic_required_deduped_when_existence_constraint_already_present() -> (
    None
):
    """Pre-existing EXISTENCE constraint does not duplicate when ``required=True`` on instances."""
    nt = NodeType(
        label="Person",
        properties=[PropertyType(name="name", type="STRING", required=True)],
    )
    existing = ConstraintType(
        type=GraphConstraintType.EXISTENCE,
        node_type="Person",
        property_name="name",
        relationship_type=None,
    )
    schema = GraphSchema(node_types=(nt,), constraints=(existing,))
    assert len(schema.constraints) == 1
    assert schema.existence_property_names_for_node("Person") == {"name"}
