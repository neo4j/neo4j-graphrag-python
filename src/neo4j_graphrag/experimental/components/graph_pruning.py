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
import logging
from typing import Optional, Any

from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    PropertyType,
    NodeType,
)
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)
from neo4j_graphrag.experimental.pipeline import Component

logger = logging.getLogger(__name__)


class GraphPruning(Component):
    async def run(
        self,
        graph: Neo4jGraph,
        schema: Optional[GraphSchema] = None,
    ) -> Neo4jGraph:
        if schema is None:
            return graph
        return self._clean_graph(graph, schema)

    def _clean_graph(
        self,
        graph: Neo4jGraph,
        schema: GraphSchema,
    ) -> Neo4jGraph:
        """
        Verify that the graph conforms to the provided schema.

        Remove invalid entities,relationships, and properties.
        If an entity is removed, all of its relationships are also removed.
        If no valid properties remain for an entity, remove that entity.
        """
        # enforce nodes (remove invalid labels, strip invalid properties)
        filtered_nodes = self._enforce_nodes(graph.nodes, schema)
        if not filtered_nodes:
            logger.warning(
                "PRUNING: all nodes were pruned, resulting graph will be empty. Check logs for details."
            )
            return Neo4jGraph()

        # enforce relationships (remove those referencing invalid nodes or with invalid
        # types or with start/end nodes not conforming to the schema, and strip invalid
        # properties)
        filtered_rels = self._enforce_relationships(
            graph.relationships, filtered_nodes, schema
        )

        return Neo4jGraph(nodes=filtered_nodes, relationships=filtered_rels)

    def _validate_node(
        self,
        node: Neo4jNode,
        schema_entity: Optional[NodeType] = None,
        additional_node_types: bool = True,
    ) -> Optional[Neo4jNode]:
        if not schema_entity:
            # node type not declared in the schema
            if additional_node_types:
                # keep node as it is as we do not have any additional info
                return node
            # it's not in schema
            return None
        allowed_props = schema_entity.properties
        filtered_props = self._enforce_properties(
            node.properties, allowed_props, schema_entity.additional_properties
        )
        if not filtered_props:
            return None
        return Neo4jNode(
            id=node.id,
            label=node.label,
            properties=filtered_props,
            embedding_properties=node.embedding_properties,
        )

    def _enforce_nodes(
        self, extracted_nodes: list[Neo4jNode], schema: GraphSchema
    ) -> list[Neo4jNode]:
        """
        Filter extracted nodes to be conformant to the schema.

        Keep only those whose label is in schema
        (unless schema has additional_node_types=True, default value)
        For each valid node, validate properties. If a node is left without
        properties, prune it.
        """
        valid_nodes = []
        for node in extracted_nodes:
            schema_entity = schema.node_type_from_label(node.label)
            new_node = self._validate_node(
                node,
                schema_entity,
                additional_node_types=schema.additional_node_types,
            )
            if new_node:
                valid_nodes.append(new_node)
        return valid_nodes

    def _enforce_relationships(
        self,
        extracted_relationships: list[Neo4jRelationship],
        filtered_nodes: list[Neo4jNode],
        schema: GraphSchema,
    ) -> list[Neo4jRelationship]:
        """
        Filter extracted nodes to be conformant to the schema.

        Keep only those whose types are in schema, start/end node conform to schema,
        and start/end nodes are in filtered nodes (i.e., kept after node enforcement).
        For each valid relationship, filter out properties not present in the schema.
        If a relationship direct is incorrect, invert it.
        """

        valid_rels = []
        valid_nodes = {node.id: node.label for node in filtered_nodes}

        patterns = schema.patterns

        for rel in extracted_relationships:
            schema_relation = schema.relationship_type_from_label(rel.type)
            if schema_relation is None:
                if schema.additional_relationship_types:
                    valid_rels.append(rel)
                else:
                    logger.debug(f"PRUNING:: {rel} as {rel.type} is not in the schema")
                continue

            if (
                rel.start_node_id not in valid_nodes
                or rel.end_node_id not in valid_nodes
            ):
                logger.debug(
                    f"PRUNING:: {rel} as one of {rel.start_node_id} or {rel.end_node_id} is not in the graph"
                )
                continue

            start_label = valid_nodes[rel.start_node_id]
            end_label = valid_nodes[rel.end_node_id]

            tuple_valid = True
            if patterns:
                tuple_valid = (start_label, rel.type, end_label) in patterns
                reverse_tuple_valid = (
                    end_label,
                    rel.type,
                    start_label,
                ) in patterns

                if (
                    not tuple_valid
                    and not reverse_tuple_valid
                    and not schema.additional_patterns
                ):
                    logger.debug(f"PRUNING:: {rel} not in the allowed patterns")
                    continue

            allowed_props = schema_relation.properties
            filtered_props = self._enforce_properties(
                rel.properties, allowed_props, schema_relation.additional_properties
            )

            valid_rels.append(
                Neo4jRelationship(
                    start_node_id=rel.start_node_id if tuple_valid else rel.end_node_id,
                    end_node_id=rel.end_node_id if tuple_valid else rel.start_node_id,
                    type=rel.type,
                    properties=filtered_props,
                    embedding_properties=rel.embedding_properties,
                )
            )

        return valid_rels

    def _enforce_properties(
        self,
        properties: dict[str, Any],
        valid_properties: list[PropertyType],
        additional_properties: bool,
    ) -> dict[str, Any]:
        """
        Filter properties.
        - Keep only those that exist in schema (i.e., valid properties).
        - Check that all required properties are present
        """
        valid_prop_names = {prop.name for prop in valid_properties}
        filtered_properties = {
            key: value
            for key, value in properties.items()
            if key in valid_prop_names or additional_properties
        }
        required_prop_names = {prop.name for prop in valid_properties if prop.required}
        for req_prop in required_prop_names:
            if filtered_properties.get(req_prop) is None:
                logger.info(
                    f"PRUNING:: {req_prop} is required but missing in {properties} - skipping node"
                )
                return {}  # node will be pruned
        return filtered_properties
