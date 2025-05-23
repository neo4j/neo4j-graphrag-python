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

import pytest

from neo4j_graphrag.experimental.components.graph_pruning import GraphPruning
from neo4j_graphrag.experimental.components.schema import (
    NodeType,
    PropertyType,
    RelationshipType,
)
from neo4j_graphrag.experimental.components.types import Neo4jNode, Neo4jRelationship


@pytest.mark.parametrize(
    "properties, valid_properties, additional_properties, expected_filtered_properties",
    [
        (
            # no required, additional allowed
            {
                "name": "John Does",
                "age": 25,
            },
            [
                PropertyType(
                    name="name",
                    type="STRING",
                )
            ],
            True,
            {
                "name": "John Does",
                "age": 25,
            },
        ),
        (
            # no required, additional not allowed
            {
                "name": "John Does",
                "age": 25,
            },
            [
                PropertyType(
                    name="name",
                    type="STRING",
                )
            ],
            False,
            {
                "name": "John Does",
            },
        ),
        (
            # required missing
            {
                "age": 25,
            },
            [
                PropertyType(
                    name="name",
                    type="STRING",
                    required=True,
                )
            ],
            True,
            {},
        ),
    ],
)
def test_graph_pruning_enforce_properties(
    properties: dict[str, Any],
    valid_properties: list[PropertyType],
    additional_properties: bool,
    expected_filtered_properties: dict[str, Any],
) -> None:
    prunner = GraphPruning()
    filtered_properties = prunner._enforce_properties(
        properties, valid_properties, additional_properties=additional_properties
    )
    assert filtered_properties == expected_filtered_properties


@pytest.fixture(scope="module")
def node_type_no_properties() -> NodeType:
    return NodeType(label="Person")


@pytest.fixture(scope="module")
def node_type_required_name() -> NodeType:
    return NodeType(
        label="Person",
        properties=[
            PropertyType(name="name", type="STRING", required=True),
            PropertyType(name="age", type="INTEGER"),
        ],
    )


@pytest.mark.parametrize(
    "node, entity, additional_node_types, expected_node",
    [
        # all good, with default values
        (
            Neo4jNode(id="1", label="Person", properties={"name": "John Doe"}),
            "node_type_no_properties",
            True,
            Neo4jNode(id="1", label="Person", properties={"name": "John Doe"}),
        ),
        # properties empty (missing default)
        (
            Neo4jNode(id="1", label="Person", properties={"age": 45}),
            "node_type_required_name",
            True,
            None,
        ),
        # node label not is schema, additional not allowed
        (
            Neo4jNode(id="1", label="Location", properties={"name": "New York"}),
            None,
            False,
            None,
        ),
        # node label not is schema, additional allowed
        (
            Neo4jNode(id="1", label="Location", properties={"name": "New York"}),
            None,
            True,
            Neo4jNode(id="1", label="Location", properties={"name": "New York"}),
        ),
    ],
)
def test_graph_pruning_validate_node(
    node: Neo4jNode,
    entity: str,
    additional_node_types: bool,
    expected_node: Neo4jNode,
    request: pytest.FixtureRequest,
) -> None:
    e = request.getfixturevalue(entity) if entity else None

    prunner = GraphPruning()
    result = prunner._validate_node(node, e, additional_node_types)
    if expected_node is not None:
        assert result == expected_node
    else:
        assert result is None


@pytest.fixture
def neo4j_relationship() -> Neo4jRelationship:
    return Neo4jRelationship(
        start_node_id="1",
        end_node_id="2",
        type="REL",
        properties={},
    )


@pytest.fixture
def neo4j_reversed_relationship(neo4j_relationship: Neo4jRelationship) -> Neo4jRelationship:
    return Neo4jRelationship(
        start_node_id=neo4j_relationship.end_node_id,
        end_node_id=neo4j_relationship.start_node_id,
        type=neo4j_relationship.type,
        properties=neo4j_relationship.properties,
    )


@pytest.mark.parametrize(
    "relationship, valid_nodes, relationship_type, additional_relationship_types, patterns, additional_patterns, expected_relationship",
    [
        # all good
        (
            "neo4j_relationship",
            {
                "1": "Person",
                "2": "Location",
            },
            RelationshipType(
                label="REL",
            ),
            True,
            (("Person", "REL", "Location"),),
            True,
            "neo4j_relationship",
        ),
        # reverse relationship
        (
            "neo4j_reversed_relationship",
            {
                "1": "Person",
                "2": "Location",
            },
            RelationshipType(
                label="REL",
            ),
            True,
            (("Person", "REL", "Location"),),
            True,
            "neo4j_relationship",
        ),
        # invalid type addition allowed
        (
            "neo4j_relationship",
            {
                "1": "Person",
                "2": "Location",
            },
            None,
            True,
            (("Person", "REL", "Location"),),
            True,
            "neo4j_relationship",
        ),
        # invalid_type_addition_not_allowed
        (
            "neo4j_relationship",
            {
                "1": "Person",
                "2": "Location",
            },
            None,
            False,
            (("Person", "REL", "Location"),),
            True,
            None,
        ),
        # invalid pattern, addition allowed
        (
            "neo4j_relationship",
            {
                "1": "Person",
                "2": "Location",
            },
            RelationshipType(
                label="REL",
            ),
            True,
            (("Person", "REL", "Person"),),
            True,
            "neo4j_relationship",
        ),
        # invalid pattern, addition not allowed
        (
            "neo4j_relationship",
            {
                "1": "Person",
                "2": "Location",
            },
            RelationshipType(
                label="REL",
            ),
            True,
            (("Person", "REL", "Person"),),
            False,
            None,
        ),
    ],
)
def test_graph_pruning_validate_relationship(
    relationship: str,
    valid_nodes: dict[str, str],
    relationship_type: RelationshipType,
    additional_relationship_types: bool,
    patterns: tuple[tuple[str, str, str], ...],
    additional_patterns: bool,
    expected_relationship: str | None,
    request: pytest.FixtureRequest,
) -> None:
    relationship_obj = request.getfixturevalue(relationship)
    expected_relationship_obj = (
        request.getfixturevalue(expected_relationship)
        if expected_relationship
        else None
    )

    pruner = GraphPruning()
    assert (
        pruner._validate_relationship(
            relationship_obj,
            valid_nodes,
            relationship_type,
            additional_relationship_types,
            patterns,
            additional_patterns,
        )
        == expected_relationship_obj
    )
