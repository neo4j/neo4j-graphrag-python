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

from neo4j_graphrag.components.graph_schema_extraction import (
    ExtractedConstraintType,
    ExtractedNodeType,
    ExtractedPropertyType,
    ExtractedRelationshipType,
    GraphSchemaExtractionOutput,
    wire_extraction_constraints_for_graph_schema,
)
from neo4j_graphrag.components.schema import (
    GraphConstraintType,
    GraphSchema,
    Pattern,
)


def test_wire_extraction_constraints_keeps_relationship_type_for_uniqueness() -> None:
    """Empty/null relationship_type is removed for all constraint types; non-empty relationship_type is preserved for relationship-scoped constraints."""
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
            {
                "type": "UNIQUENESS",
                "node_type": "",
                "property_names": ["isbn"],
                "relationship_type": "KNOWS",
            },
        ]
    )
    assert "relationship_type" not in out[0]
    assert "relationship_type" not in out[1]
    assert out[2]["relationship_type"] == "KNOWS"


def test_wire_extraction_constraints_drops_empty_relationship_type_for_non_uniqueness() -> (
    None
):
    """For non-UNIQUENESS constraints, empty/null ``relationship_type`` is removed so the runtime default applies."""
    out = wire_extraction_constraints_for_graph_schema(
        [
            {
                "type": "EXISTENCE",
                "node_type": "Person",
                "property_names": ["name"],
                "relationship_type": "",
            },
            {
                "type": "KEY",
                "node_type": "Person",
                "property_names": ["id"],
                "relationship_type": None,
            },
        ]
    )
    assert "relationship_type" not in out[0]
    assert "relationship_type" not in out[1]


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


def test_uniqueness_with_relationship_type_preserved_by_wire_conversion() -> None:
    """Relationship-scoped UNIQUENESS flows through wire conversion and into GraphSchema correctly."""
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
                properties=[ExtractedPropertyType(name="since", type="INTEGER")],
            )
        ],
        patterns=[],
        constraints=[
            ExtractedConstraintType(
                type="UNIQUENESS",
                node_type="",
                property_names=["since"],
                relationship_type="KNOWS",
            ),
        ],
    )
    gs = GraphSchema.from_extraction_output(dto)
    assert len(gs.constraints) == 1
    assert gs.constraints[0].relationship_type == "KNOWS"
    assert gs.constraints[0].node_type is None
    assert gs.uniqueness_property_names_for_relationship("KNOWS") == {"since"}


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


def _subsumption_dto(
    constraints: list[ExtractedConstraintType],
) -> GraphSchemaExtractionOutput:
    """Extraction output with Person/Company nodes and a KNOWS relationship."""
    return GraphSchemaExtractionOutput(
        node_types=[
            ExtractedNodeType(
                label="Person",
                properties=[
                    ExtractedPropertyType(name="id", type="STRING"),
                    ExtractedPropertyType(name="firstname", type="STRING"),
                    ExtractedPropertyType(name="surname", type="STRING"),
                    ExtractedPropertyType(name="email", type="STRING"),
                ],
            ),
            ExtractedNodeType(
                label="Company",
                properties=[ExtractedPropertyType(name="id", type="STRING")],
            ),
        ],
        relationship_types=[
            ExtractedRelationshipType(
                label="KNOWS",
                properties=[
                    ExtractedPropertyType(name="id", type="STRING"),
                    ExtractedPropertyType(name="since", type="DATE"),
                ],
            )
        ],
        patterns=[Pattern(source="Person", relationship="KNOWS", target="Person")],
        constraints=constraints,
    )


def _constraint_tuples(gs: GraphSchema) -> set[tuple[str, str, str, tuple[str, ...]]]:
    return {
        (
            GraphConstraintType(c.type).value,
            c.node_type or "",
            c.relationship_type or "",
            c.property_names,
        )
        for c in gs.constraints
    }


def test_extraction_drops_existence_on_key_property() -> None:
    """EXISTENCE on a KEY property is redundant and dropped instead of failing validation."""
    gs = GraphSchema.from_extraction_output(
        _subsumption_dto(
            [
                ExtractedConstraintType(
                    type="KEY", node_type="Person", property_names=["id"]
                ),
                ExtractedConstraintType(
                    type="EXISTENCE", node_type="Person", property_names=["id"]
                ),
            ]
        )
    )
    assert _constraint_tuples(gs) == {("KEY", "Person", "", ("id",))}


def test_extraction_drops_existence_on_composite_key_member() -> None:
    gs = GraphSchema.from_extraction_output(
        _subsumption_dto(
            [
                ExtractedConstraintType(
                    type="KEY",
                    node_type="Person",
                    property_names=["firstname", "surname"],
                ),
                ExtractedConstraintType(
                    type="EXISTENCE", node_type="Person", property_names=["firstname"]
                ),
            ]
        )
    )
    assert _constraint_tuples(gs) == {("KEY", "Person", "", ("firstname", "surname"))}


def test_extraction_drops_uniqueness_on_same_properties_as_key() -> None:
    gs = GraphSchema.from_extraction_output(
        _subsumption_dto(
            [
                ExtractedConstraintType(
                    type="UNIQUENESS", node_type="Person", property_names=["id"]
                ),
                ExtractedConstraintType(
                    type="KEY", node_type="Person", property_names=["id"]
                ),
            ]
        )
    )
    assert _constraint_tuples(gs) == {("KEY", "Person", "", ("id",))}


def test_extraction_drops_uniqueness_on_key_properties_in_different_order() -> None:
    gs = GraphSchema.from_extraction_output(
        _subsumption_dto(
            [
                ExtractedConstraintType(
                    type="KEY",
                    node_type="Person",
                    property_names=["firstname", "surname"],
                ),
                ExtractedConstraintType(
                    type="UNIQUENESS",
                    node_type="Person",
                    property_names=["surname", "firstname"],
                ),
            ]
        )
    )
    assert _constraint_tuples(gs) == {("KEY", "Person", "", ("firstname", "surname"))}


def test_extraction_drops_existence_on_relationship_key_property() -> None:
    gs = GraphSchema.from_extraction_output(
        _subsumption_dto(
            [
                ExtractedConstraintType(
                    type="KEY",
                    node_type="",
                    property_names=["since"],
                    relationship_type="KNOWS",
                ),
                ExtractedConstraintType(
                    type="EXISTENCE",
                    node_type="",
                    property_names=["since"],
                    relationship_type="KNOWS",
                ),
            ]
        )
    )
    assert _constraint_tuples(gs) == {("KEY", "", "KNOWS", ("since",))}


def test_extraction_keeps_constraints_not_subsumed_by_key() -> None:
    """Only constraints made redundant by a KEY on the same type are dropped."""
    gs = GraphSchema.from_extraction_output(
        _subsumption_dto(
            [
                ExtractedConstraintType(
                    type="KEY", node_type="Company", property_names=["id"]
                ),
                ExtractedConstraintType(
                    type="KEY", node_type="Person", property_names=["firstname"]
                ),
                # EXISTENCE on a non-KEY property
                ExtractedConstraintType(
                    type="EXISTENCE", node_type="Person", property_names=["email"]
                ),
                # Same property name, but KEY is on another node type
                ExtractedConstraintType(
                    type="EXISTENCE", node_type="Person", property_names=["id"]
                ),
                # Composite UNIQUENESS only partially overlapping the KEY
                ExtractedConstraintType(
                    type="UNIQUENESS",
                    node_type="Person",
                    property_names=["firstname", "surname"],
                ),
                # Relationship property sharing a name with a node KEY property
                ExtractedConstraintType(
                    type="EXISTENCE",
                    node_type="",
                    property_names=["id"],
                    relationship_type="KNOWS",
                ),
            ]
        )
    )
    assert _constraint_tuples(gs) == {
        ("KEY", "Company", "", ("id",)),
        ("KEY", "Person", "", ("firstname",)),
        ("EXISTENCE", "Person", "", ("email",)),
        ("EXISTENCE", "Person", "", ("id",)),
        ("UNIQUENESS", "Person", "", ("firstname", "surname")),
        ("EXISTENCE", "", "KNOWS", ("id",)),
    }


def test_extraction_invalid_key_does_not_drop_valid_existence() -> None:
    """A KEY removed by the invalid-constraint filter must not subsume other constraints."""
    gs = GraphSchema.from_extraction_output(
        _subsumption_dto(
            [
                ExtractedConstraintType(
                    type="KEY",
                    node_type="Person",
                    property_names=["id", "nonexistent"],
                ),
                ExtractedConstraintType(
                    type="EXISTENCE", node_type="Person", property_names=["id"]
                ),
            ]
        )
    )
    assert _constraint_tuples(gs) == {("EXISTENCE", "Person", "", ("id",))}
