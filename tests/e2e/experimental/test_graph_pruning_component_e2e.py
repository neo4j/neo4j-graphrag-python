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
from neo4j_graphrag.experimental.components.schema import GraphSchema
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)


@pytest.fixture
def extracted_graph() -> Neo4jGraph:
    """This is the graph to be pruned in all the below tests,
    using different schema configuration.
    """
    return Neo4jGraph(
        nodes=[
            Neo4jNode(
                id="1",
                label="Person",
                properties={
                    "name": "John Doe",
                },
            ),
            Neo4jNode(
                id="2",
                label="Person",
                properties={
                    "height": 180,
                },
            ),
            Neo4jNode(
                id="3",
                label="Person",
                properties={
                    "name": "Jane Doe",
                    "weight": 90,
                },
            ),
            Neo4jNode(
                id="10",
                label="Organization",
                properties={
                    "name": "Azerty Inc.",
                    "created": 1999,
                },
            ),
        ],
        relationships=[
            Neo4jRelationship(
                start_node_id="1",
                end_node_id="2",
                type="KNOWS",
                properties={"firstMetIn": 2025},
            ),
            Neo4jRelationship(
                start_node_id="1",
                end_node_id="3",
                type="KNOWS",
            ),
            Neo4jRelationship(
                start_node_id="1",
                end_node_id="2",
                type="MANAGES",
            ),
            Neo4jRelationship(
                start_node_id="1",
                end_node_id="10",
                type="MANAGES",
            ),
            Neo4jRelationship(
                start_node_id="1",
                end_node_id="10",
                type="WORKS_FOR",
            ),
        ],
    )


async def _test(
    extracted_graph: Neo4jGraph, schema_dict: dict[str, Any], expected_graph: Neo4jGraph
) -> None:
    schema = GraphSchema.model_validate(schema_dict)
    pruner = GraphPruning()
    res = await pruner.run(extracted_graph, schema)
    assert res.graph == expected_graph


@pytest.mark.asyncio
async def test_graph_pruning_loose(extracted_graph: Neo4jGraph) -> None:
    """Loose schema:
    - no required properties
    - all additional* allowed

    => we keep everything from the extracted graph
    """
    schema_dict = {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {"name": "name", "type": "STRING"},
                    {"name": "height", "type": "INTEGER"},
                ],
                "additional_properties": True,
            }
        ],
        "relationship_types": [
            {
                "label": "KNOWS",
            }
        ],
        "patterns": [
            ("Person", "KNOWS", "Person"),
        ],
        "additional_node_types": True,
        "additional_relationship_types": True,
        "additional_patterns": True,
    }
    await _test(extracted_graph, schema_dict, extracted_graph)


@pytest.mark.asyncio
async def test_graph_pruning_missing_required_property(
    extracted_graph: Neo4jGraph,
) -> None:
    """Person node type has a required 'name' property:
    - extracted nodes without this property are pruned
    - any relationship tied to this node is also pruned
    """
    schema_dict = {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {
                        "name": "name",
                        "type": "STRING",
                        "required": True,
                    },
                    {"name": "height", "type": "INTEGER"},
                ],
                "additional_properties": True,
            }
        ],
        "relationship_types": [
            {
                "label": "KNOWS",
            }
        ],
        "patterns": [
            ("Person", "KNOWS", "Person"),
        ],
        "additional_node_types": True,
        "additional_relationship_types": True,
        "additional_patterns": True,
    }
    filtered_graph = Neo4jGraph(
        nodes=[
            Neo4jNode(
                id="1",
                label="Person",
                properties={
                    "name": "John Doe",
                },
            ),
            # do not have the required "name" property
            # Neo4jNode(
            #     id="2",
            #     label="Person",
            #     properties={
            #         "height": 180,
            #     }
            # ),
            Neo4jNode(
                id="3",
                label="Person",
                properties={
                    "name": "Jane Doe",
                    "weight": 90,
                },
            ),
            Neo4jNode(
                id="10",
                label="Organization",
                properties={
                    "name": "Azerty Inc.",
                    "created": 1999,
                },
            ),
        ],
        relationships=[
            # node "2" was pruned
            # Neo4jRelationship(
            #     start_node_id="1",
            #     end_node_id="2",
            #     type="KNOWS",
            #     properties={"firstMetIn": 2025},
            # ),
            Neo4jRelationship(
                start_node_id="1",
                end_node_id="3",
                type="KNOWS",
            ),
            # node "2" was pruned
            # Neo4jRelationship(
            #     start_node_id="1",
            #     end_node_id="2",
            #     type="MANAGES",
            # ),
            Neo4jRelationship(
                start_node_id="1",
                end_node_id="10",
                type="MANAGES",
            ),
            Neo4jRelationship(
                start_node_id="1",
                end_node_id="10",
                type="WORKS_FOR",
            ),
        ],
    )
    await _test(extracted_graph, schema_dict, filtered_graph)


@pytest.mark.asyncio
async def test_graph_pruning_strict_properties_and_node_types(
    extracted_graph: Neo4jGraph,
) -> None:
    """Additional properties on Person nodes are not allowed.
    Additional node types are not allowed.

    => we prune "Organization" nodes (not in schema)
    and the "weight" property that was extracted for some persons.
    """
    schema_dict = {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {
                        "name": "name",
                        "type": "STRING",
                    },
                    {"name": "height", "type": "INTEGER"},
                ],
                # "additional_properties": False,  # default value
            }
        ],
        "relationship_types": [
            {
                "label": "KNOWS",
            }
        ],
        "patterns": [
            ("Person", "KNOWS", "Person"),
        ],
        # "additional_node_types": False,  # default value
        "additional_relationship_types": True,
        "additional_patterns": True,
    }
    filtered_graph = Neo4jGraph(
        nodes=[
            Neo4jNode(
                id="1",
                label="Person",
                properties={
                    "name": "John Doe",
                },
            ),
            Neo4jNode(
                id="2",
                label="Person",
                properties={
                    "height": 180,
                },
            ),
            Neo4jNode(
                id="3",
                label="Person",
                properties={
                    "name": "Jane Doe",
                    # weight not in listed properties
                    # "weight": 90,
                },
            ),
            # label "Organization" not in schema
            # Neo4jNode(
            #     id="10",
            #     label="Organization",
            #     properties={
            #         "name": "Azerty Inc.",
            #         "created": 1999,
            #     }
            # ),
        ],
        relationships=[
            Neo4jRelationship(
                start_node_id="1",
                end_node_id="2",
                type="KNOWS",
                properties={"firstMetIn": 2025},
            ),
            Neo4jRelationship(
                start_node_id="1",
                end_node_id="3",
                type="KNOWS",
            ),
            Neo4jRelationship(
                start_node_id="1",
                end_node_id="2",
                type="MANAGES",
            ),
            # node "10" was pruned (label not allowed)
            # Neo4jRelationship(
            #     start_node_id="1",
            #     end_node_id="10",
            #     type="MANAGES",
            # ),
            # Neo4jRelationship(
            #     start_node_id="1",
            #     end_node_id="10",
            #     type="WORKS_FOR",
            # )
        ],
    )
    await _test(extracted_graph, schema_dict, filtered_graph)


@pytest.mark.asyncio
async def test_graph_pruning_strict_patterns(extracted_graph: Neo4jGraph) -> None:
    """Additional patterns not allowed:

    - MANAGES: it's a known relationship type but without any pattern, it's pruned
    - WORKS_FOR: it's not a known relationship type, and additional_relationship_types is allowed
        so we keep it.
    """
    # - no additional patterns allowed
    schema_dict = {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {
                        "name": "name",
                        "type": "STRING",
                    },
                    {"name": "height", "type": "INTEGER"},
                ],
                "additional_properties": True,
            },
            {
                "label": "Organization",
            },
        ],
        "relationship_types": [
            {
                "label": "KNOWS",
            },
            {
                "label": "MANAGES",
            },
        ],
        "patterns": (
            ("Person", "KNOWS", "Person"),
            ("Person", "KNOWS", "Organization"),
        ),
        "additional_node_types": True,
        "additional_relationship_types": False,
        "additional_patterns": False,
    }
    filtered_graph = Neo4jGraph(
        nodes=[
            Neo4jNode(
                id="1",
                label="Person",
                properties={
                    "name": "John Doe",
                },
            ),
            Neo4jNode(
                id="2",
                label="Person",
                properties={
                    "height": 180,
                },
            ),
            Neo4jNode(
                id="3",
                label="Person",
                properties={
                    "name": "Jane Doe",
                    "weight": 90,
                },
            ),
            Neo4jNode(
                id="10",
                label="Organization",
                properties={
                    "name": "Azerty Inc.",
                    "created": 1999,
                },
            ),
        ],
        relationships=[
            Neo4jRelationship(
                start_node_id="1",
                end_node_id="2",
                type="KNOWS",
                properties={"firstMetIn": 2025},
            ),
            Neo4jRelationship(
                start_node_id="1",
                end_node_id="3",
                type="KNOWS",
            ),
            # invalid pattern (person, manages, person)
            # Neo4jRelationship(
            #     start_node_id="1",
            #     end_node_id="2",
            #     type="MANAGES",
            # ),
            # invalid pattern (person, works for, person)
            # Neo4jRelationship(
            #     start_node_id="1",
            #     end_node_id="10",
            #     type="WORKS_FOR",
            # ),
        ],
    )
    await _test(extracted_graph, schema_dict, filtered_graph)
