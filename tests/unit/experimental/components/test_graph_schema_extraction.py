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

"""Behavioral tests for schema-from-text extraction wire types and conversion."""

from __future__ import annotations

import pytest

from neo4j_graphrag.exceptions import SchemaExtractionError
from neo4j_graphrag.experimental.components.graph_schema_extraction import (
    ExtractedConstraintType,
    ExtractedNodeType,
    ExtractedPropertyType,
    ExtractedRelationshipType,
    GraphSchemaExtractionOutput,
    wire_extraction_constraints_for_graph_schema,
)
from neo4j_graphrag.experimental.components.schema import GraphSchema, Pattern


def test_wire_extraction_constraints_empty_and_null_become_none() -> None:
    """Maps ``\"\"`` and legacy ``null`` to ``None`` for :class:`ConstraintType`."""
    out = wire_extraction_constraints_for_graph_schema(
        [
            {
                "type": "UNIQUENESS",
                "node_type": "Person",
                "property_names": ["id"],
                "relationship_type": "",
            },
            {
                "type": "UNIQUENESS",
                "node_type": "Org",
                "property_names": ["id"],
                "relationship_type": None,
            },
        ]
    )
    assert out[0]["relationship_type"] is None
    assert out[1]["relationship_type"] is None


def test_wire_extraction_constraints_preserves_relationship_scoped_existence() -> None:
    out = wire_extraction_constraints_for_graph_schema(
        [
            {
                "type": "EXISTENCE",
                "node_type": "",
                "property_names": ["since"],
                "relationship_type": "KNOWS",
            }
        ]
    )
    assert out[0]["relationship_type"] == "KNOWS"


def test_uniqueness_with_relationship_type_fails_at_graph_schema_validation() -> None:
    """If a bad constraint survives extraction filters, :class:`ConstraintType` rejects it."""
    dto = GraphSchemaExtractionOutput(
        node_types=[
            ExtractedNodeType(
                label="Person",
                properties=[ExtractedPropertyType(name="name", type="STRING")],
            )
        ],
        relationship_types=[],
        patterns=[],
        constraints=[
            ExtractedConstraintType(
                type="UNIQUENESS",
                node_type="Person",
                property_names=["name"],
                relationship_type="KNOWS",
            ),
        ],
    )
    with pytest.raises(SchemaExtractionError):
        GraphSchema.from_extraction_output(dto)


def test_invalid_constraints_dropped_by_extraction_filters_without_error() -> None:
    """Cross-reference filters drop semantically invalid constraints (see ``_extraction_filter_invalid_constraints``)."""
    dto = GraphSchemaExtractionOutput(
        node_types=[
            ExtractedNodeType(
                label="Person",
                properties=[ExtractedPropertyType(name="name", type="STRING")],
            )
        ],
        relationship_types=[],
        patterns=[],
        constraints=[
            ExtractedConstraintType(
                type="UNIQUENESS",
                node_type="",
                property_names=["name"],
                relationship_type="",
            ),
            ExtractedConstraintType(
                type="EXISTENCE",
                node_type="Person",
                property_names=["name"],
                relationship_type="KNOWS",
            ),
            ExtractedConstraintType(
                type="EXISTENCE",
                node_type="",
                property_names=["name"],
                relationship_type="",
            ),
        ],
    )
    gs = GraphSchema.from_extraction_output(dto)
    assert len(gs.constraints) == 0


def test_from_extraction_output_relationship_existence_constraint() -> None:
    """Relationship-scoped EXISTENCE uses empty ``node_type`` and non-empty ``relationship_type``."""
    dto = GraphSchemaExtractionOutput(
        node_types=[
            ExtractedNodeType(
                label="Person",
                properties=[ExtractedPropertyType(name="name", type="STRING")],
            )
        ],
        relationship_types=[
            ExtractedRelationshipType(
                label="KNOWS",
                properties=[ExtractedPropertyType(name="since", type="LOCAL_DATETIME")],
            )
        ],
        patterns=[
            Pattern(
                source="Person",
                relationship="KNOWS",
                target="Person",
            ),
        ],
        constraints=[
            ExtractedConstraintType(
                type="EXISTENCE",
                node_type="",
                property_names=["since"],
                relationship_type="KNOWS",
            ),
        ],
    )
    gs = GraphSchema.from_extraction_output(dto)
    assert gs.existence_property_names_for_relationship("KNOWS") == {"since"}
    assert gs.existence_property_names_for_node("Person") == set()


def test_from_extraction_output_two_constraints_same_property_distinct_kinds() -> None:
    """Sanity: UNIQUENESS + EXISTENCE on the same property name yields two runtime constraints."""
    dto = GraphSchemaExtractionOutput(
        node_types=[
            ExtractedNodeType(
                label="Person",
                properties=[ExtractedPropertyType(name="name", type="STRING")],
            )
        ],
        relationship_types=[],
        patterns=[],
        constraints=[
            ExtractedConstraintType(
                type="UNIQUENESS",
                node_type="Person",
                property_names=["name"],
            ),
            ExtractedConstraintType(
                type="EXISTENCE",
                node_type="Person",
                property_names=["name"],
                relationship_type="",
            ),
        ],
    )
    gs = GraphSchema.from_extraction_output(dto)
    assert len(gs.constraints) == 2


def test_from_extraction_output_key_constraint_node() -> None:
    dto = GraphSchemaExtractionOutput(
        node_types=[
            ExtractedNodeType(
                label="Person",
                properties=[ExtractedPropertyType(name="email", type="STRING")],
            )
        ],
        relationship_types=[],
        patterns=[],
        constraints=[
            ExtractedConstraintType(
                type="KEY",
                node_type="Person",
                property_names=["email"],
                relationship_type="",
            ),
        ],
    )
    gs = GraphSchema.from_extraction_output(dto)
    assert gs.key_property_names_for_node("Person") == {"email"}
    assert gs.mandatory_property_names_for_node("Person") == {"email"}


def test_from_extraction_output_composite_key_constraint() -> None:
    """Composite KEY constraint from LLM extraction preserves multi-property grouping."""
    dto = GraphSchemaExtractionOutput(
        node_types=[
            ExtractedNodeType(
                label="Actor",
                properties=[
                    ExtractedPropertyType(name="firstname", type="STRING"),
                    ExtractedPropertyType(name="surname", type="STRING"),
                ],
            )
        ],
        relationship_types=[],
        patterns=[],
        constraints=[
            ExtractedConstraintType(
                type="KEY",
                node_type="Actor",
                property_names=["firstname", "surname"],
            ),
        ],
    )
    gs = GraphSchema.from_extraction_output(dto)
    assert gs.key_property_names_for_node("Actor") == {"firstname", "surname"}
    assert gs.mandatory_property_names_for_node("Actor") == {"firstname", "surname"}
    assert len(gs.constraints) == 1
    assert gs.constraints[0].property_names == ("firstname", "surname")


def test_from_extraction_output_composite_uniqueness_constraint() -> None:
    """Composite UNIQUENESS constraint from LLM extraction."""
    dto = GraphSchemaExtractionOutput(
        node_types=[
            ExtractedNodeType(
                label="Book",
                properties=[
                    ExtractedPropertyType(name="title", type="STRING"),
                    ExtractedPropertyType(name="year", type="INTEGER"),
                ],
            )
        ],
        relationship_types=[],
        patterns=[],
        constraints=[
            ExtractedConstraintType(
                type="UNIQUENESS",
                node_type="Book",
                property_names=["title", "year"],
            ),
        ],
    )
    gs = GraphSchema.from_extraction_output(dto)
    assert gs.uniqueness_property_names_for_node("Book") == {"title", "year"}
    assert len(gs.constraints) == 1
    assert gs.constraints[0].property_names == ("title", "year")


def test_extraction_filter_rejects_composite_existence() -> None:
    """Composite EXISTENCE from LLM output is filtered out (Neo4j only supports single-property EXISTENCE)."""
    dto = GraphSchemaExtractionOutput(
        node_types=[
            ExtractedNodeType(
                label="Person",
                properties=[
                    ExtractedPropertyType(name="name", type="STRING"),
                    ExtractedPropertyType(name="email", type="STRING"),
                ],
            )
        ],
        relationship_types=[],
        patterns=[],
        constraints=[
            ExtractedConstraintType(
                type="EXISTENCE",
                node_type="Person",
                property_names=["name", "email"],
            ),
        ],
    )
    gs = GraphSchema.from_extraction_output(dto)
    assert len(gs.constraints) == 0


def test_extraction_filter_drops_composite_with_nonexistent_property() -> None:
    """Composite constraint where one property doesn't exist on the node type is filtered out."""
    dto = GraphSchemaExtractionOutput(
        node_types=[
            ExtractedNodeType(
                label="Actor",
                properties=[
                    ExtractedPropertyType(name="firstname", type="STRING"),
                ],
            )
        ],
        relationship_types=[],
        patterns=[],
        constraints=[
            ExtractedConstraintType(
                type="KEY",
                node_type="Actor",
                property_names=["firstname", "surname"],
            ),
        ],
    )
    gs = GraphSchema.from_extraction_output(dto)
    # "surname" doesn't exist on Actor, so the entire composite constraint is dropped
    assert len(gs.constraints) == 0
