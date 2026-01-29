"""This example demonstrates how to use the GraphPruning component."""

import asyncio

from neo4j_graphrag.experimental.components.graph_pruning import GraphPruning
from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    NodeType,
    Pattern,
    PropertyType,
    RelationshipType,
)
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)

graph = Neo4jGraph(
    nodes=[
        Neo4jNode(
            id="Person/John",
            label="Person",
            properties={
                "firstName": "John",
                "lastName": "Doe",
                "occupation": "employee",
            },
        ),
        Neo4jNode(
            id="Person/Jane",
            label="Person",
            properties={
                "firstName": "Jane",
            },
        ),
        Neo4jNode(
            id="Person/Jack",
            label="Person",
            properties={"firstName": "Jack", "lastName": "Dae"},
        ),
        Neo4jNode(
            id="Organization/Corp1",
            label="Organization",
            properties={"name": "Corp1"},
        ),
    ],
    relationships=[
        Neo4jRelationship(
            start_node_id="Person/John",
            end_node_id="Person/Jack",
            type="KNOWS",
        ),
        Neo4jRelationship(
            start_node_id="Organization/Corp2",
            end_node_id="Person/Jack",
            type="WORKS_FOR",
        ),
        Neo4jRelationship(
            start_node_id="Person/John",
            end_node_id="Person/Jack",
            type="PARENT_OF",
        ),
    ],
)

schema = GraphSchema(
    node_types=(
        NodeType(
            label="Person",
            properties=[
                PropertyType(name="firstName", type="STRING", required=True),
                PropertyType(name="lastName", type="STRING", required=True),
                PropertyType(name="age", type="INTEGER"),
            ],
            additional_properties=False,
        ),
        NodeType(
            label="Organization",
            properties=[
                PropertyType(name="name", type="STRING", required=True),
                PropertyType(name="address", type="STRING"),
            ],
            additional_properties=True,
        ),
    ),
    relationship_types=(
        RelationshipType(
            label="WORKS_FOR",
            properties=[PropertyType(name="since", type="LOCAL_DATETIME")],
            additional_properties=True,
        ),
        RelationshipType(
            label="KNOWS",
        ),
    ),
    patterns=(
        Pattern(source="Person", relationship="KNOWS", target="Person"),
        Pattern(source="Person", relationship="WORKS_FOR", target="Organization"),
    ),
    additional_node_types=False,
    additional_relationship_types=False,
    additional_patterns=False,
)


async def main() -> None:
    pruner = GraphPruning()
    res = await pruner.run(graph, schema)
    print("=" * 20, "FINAL CLEANED GRAPH:", "=" * 20)
    print(res.graph)
    print("=" * 20, "PRUNED ITEM:", "=" * 20)
    print(res.pruning_stats)
    print("-" * 10, "PRUNED NODES:")
    for node in res.pruning_stats.pruned_nodes:
        print(
            node.item.label,
            "with properties",
            node.item.properties,
            "pruned because",
            node.pruned_reason,
            node.metadata,
        )
    print("-" * 10, "PRUNED RELATIONSHIPS:")
    for rel in res.pruning_stats.pruned_relationships:
        print(rel.item.type, "pruned because", rel.pruned_reason)
    print("-" * 10, "PRUNED PROPERTIES:")
    for prop in res.pruning_stats.pruned_properties:
        print(
            prop.item,
            "from node label",
            prop.label,
            "pruned because",
            prop.pruned_reason,
        )


if __name__ == "__main__":
    asyncio.run(main())
