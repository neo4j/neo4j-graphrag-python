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
from typing import Any
from unittest.mock import patch, Mock, ANY

import pytest

from neo4j_graphrag.experimental.components.graph_pruning import (
    GraphPruning,
    GraphPruningResult,
    PruningStats,
)
from neo4j_graphrag.experimental.components.schema import (
    NodeType,
    PropertyType,
    RelationshipType,
    GraphSchema,
)
from neo4j_graphrag.experimental.components.types import (
    Neo4jNode,
    Neo4jRelationship,
    Neo4jGraph,
    LexicalGraphConfig,
)


@pytest.fixture(scope="module")
def lexical_graph_config() -> LexicalGraphConfig:
    return LexicalGraphConfig(
        chunk_node_label="Paragraph",
    )


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
    ],
)
def test_graph_pruning_filter_properties(
    properties: dict[str, Any],
    valid_properties: list[PropertyType],
    additional_properties: bool,
    expected_filtered_properties: dict[str, Any],
) -> None:
    pruner = GraphPruning()
    filtered_properties = pruner._filter_properties(
        properties,
        valid_properties,
        additional_properties=additional_properties,
        node_label="Label",
        pruning_stats=PruningStats(),
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
        # node label not valid
        (
            Neo4jNode(id="1", label="", properties={"name": "New York"}),
            "node_type_required_name",
            True,
            None,
        ),
        # node ID not valid
        (
            Neo4jNode(id="", label="Location", properties={"name": "New York"}),
            "node_type_required_name",
            True,
            None,
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

    pruner = GraphPruning()
    result = pruner._validate_node(node, PruningStats(), e, additional_node_types)
    if expected_node is not None:
        assert result == expected_node
    else:
        assert result is None


def test_graph_pruning_enforce_nodes_lexical_graph(
    lexical_graph_config: LexicalGraphConfig,
) -> None:
    pruner = GraphPruning()
    result = pruner._enforce_nodes(
        nodes=[
            Neo4jNode(id="1", label="Paragraph"),
        ],
        schema=GraphSchema(node_types=tuple(), additional_node_types=False),
        lexical_graph_config=lexical_graph_config,
        pruning_stats=PruningStats(),
    )
    assert len(result) == 1
    assert result[0].label == "Paragraph"


@pytest.fixture
def neo4j_relationship() -> Neo4jRelationship:
    return Neo4jRelationship(
        start_node_id="1",
        end_node_id="2",
        type="REL",
        properties={},
    )


@pytest.fixture
def neo4j_relationship_invalid_type() -> Neo4jRelationship:
    return Neo4jRelationship(
        start_node_id="1",
        end_node_id="2",
        type="",
        properties={},
    )


@pytest.fixture
def neo4j_reversed_relationship(
    neo4j_relationship: Neo4jRelationship,
) -> Neo4jRelationship:
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
            "neo4j_relationship",  # relationship,
            {  # valid_nodes
                "1": "Person",
                "2": "Location",
            },
            RelationshipType(  # relationship_type
                label="REL",
            ),
            True,  # additional_relationship_types
            (("Person", "REL", "Location"),),  # patterns
            True,  # additional_patterns
            "neo4j_relationship",  # expected_relationship
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
            True,  # additional_relationship_types
            (("Person", "REL", "Location"),),
            True,  # additional_patterns
            "neo4j_relationship",
        ),
        # invalid start node ID
        (
            "neo4j_reversed_relationship",
            {
                "10": "Person",
                "2": "Location",
            },
            RelationshipType(
                label="REL",
            ),
            True,  # additional_relationship_types
            (("Person", "REL", "Location"),),
            True,  # additional_patterns
            None,
        ),
        # invalid type, addition allowed
        (
            "neo4j_relationship",
            {
                "1": "Person",
                "2": "Location",
            },
            None,  # relationship_type
            True,  # additional_relationship_types
            (("Person", "REL", "Location"),),
            True,  # additional_patterns
            "neo4j_relationship",
        ),
        # invalid type, addition allowed but invalid node ID
        (
            "neo4j_relationship",
            {
                "1": "Person",
            },
            None,  # relationship_type
            True,  # additional_relationship_types
            (("Person", "REL", "Location"),),
            True,  # additional_patterns
            None,
        ),
        # invalid type, addition not allowed
        (
            "neo4j_relationship",
            {
                "1": "Person",
                "2": "Location",
            },
            None,  # relationship_type
            False,  # additional_relationship_types
            (("Person", "REL", "Location"),),
            True,  # additional_patterns
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
            True,  # additional_relationship_types
            (("Person", "REL", "Person"),),
            True,  # additional_patterns
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
            True,  # additional_relationship_types
            (("Person", "REL", "Person"),),
            False,  # additional_patterns
            None,
        ),
        # invalid extracted type
        (
            "neo4j_relationship_invalid_type",  # relationship,
            {  # valid_nodes
                "1": "Person",
                "2": "Location",
            },
            RelationshipType(  # relationship_type
                label="REL",
            ),
            True,  # additional_relationship_types
            (("Person", "REL", "Location"),),  # patterns
            True,  # additional_patterns
            None,  # expected_relationship
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
            PruningStats(),
            relationship_type,
            additional_relationship_types,
            patterns,
            additional_patterns,
        )
        == expected_relationship_obj
    )


@patch("neo4j_graphrag.experimental.components.graph_pruning.GraphPruning._clean_graph")
@pytest.mark.asyncio
async def test_graph_pruning_run_happy_path(
    mock_clean_graph: Mock,
    node_type_required_name: NodeType,
    lexical_graph_config: LexicalGraphConfig,
) -> None:
    initial_graph = Neo4jGraph(
        nodes=[Neo4jNode(id="1", label="Person"), Neo4jNode(id="2", label="Location")],
    )
    schema = GraphSchema(node_types=(node_type_required_name,))
    cleaned_graph = Neo4jGraph(nodes=[Neo4jNode(id="1", label="Person")])
    mock_clean_graph.return_value = (cleaned_graph, PruningStats())
    pruner = GraphPruning()
    pruner_result = await pruner.run(
        graph=initial_graph,
        schema=schema,
        lexical_graph_config=lexical_graph_config,
    )
    assert isinstance(pruner_result, GraphPruningResult)
    assert pruner_result.graph == cleaned_graph
    mock_clean_graph.assert_called_once_with(
        initial_graph, schema, lexical_graph_config
    )


@pytest.mark.asyncio
async def test_graph_pruning_run_no_schema() -> None:
    initial_graph = Neo4jGraph(nodes=[Neo4jNode(id="1", label="Person")])
    pruner = GraphPruning()
    pruner_result = await pruner.run(
        graph=initial_graph,
        schema=None,
    )
    assert isinstance(pruner_result, GraphPruningResult)
    assert pruner_result.graph == initial_graph


@patch(
    "neo4j_graphrag.experimental.components.graph_pruning.GraphPruning._enforce_nodes"
)
def test_graph_pruning_clean_graph(
    mock_enforce_nodes: Mock,
    lexical_graph_config: LexicalGraphConfig,
) -> None:
    mock_enforce_nodes.return_value = []
    initial_graph = Neo4jGraph(nodes=[Neo4jNode(id="1", label="Person")])
    schema = GraphSchema(node_types=())
    pruner = GraphPruning()
    cleaned_graph, pruning_stats = pruner._clean_graph(
        initial_graph, schema, lexical_graph_config
    )
    assert cleaned_graph == Neo4jGraph()
    assert isinstance(pruning_stats, PruningStats)
    mock_enforce_nodes.assert_called_once_with(
        [Neo4jNode(id="1", label="Person")],
        schema,
        lexical_graph_config,
        ANY,
    )
