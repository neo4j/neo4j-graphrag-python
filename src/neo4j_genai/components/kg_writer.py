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

from abc import abstractmethod
from typing import Literal, Optional

import neo4j
from neo4j_genai.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship
from neo4j_genai.indexes import upsert_vector, upsert_vector_on_relationship
from neo4j_genai.neo4j_queries import UPSERT_NODE_QUERY, UPSERT_RELATIONSHIP_QUERY
from neo4j_genai.pipeline.component import Component, DataModel
from pydantic import validate_call


class KGWriterModel(DataModel):
    """Data model for the output of the Knowledge Graph writer.

    Attributes:
        status (Literal["SUCCESS", "FAILURE"]): Whether or not the write operation was successful.
    """

    status: Literal["SUCCESS", "FAILURE"]


class KGWriter(Component):
    """Abstract class used to write a knowledge graph to a data store."""

    @abstractmethod
    @validate_call
    async def run(self, graph: Neo4jGraph) -> KGWriterModel:
        """
        Writes the graph to a data store.

        Args:
            graph (Neo4jGraph): The knowledge graph to write to the data store.
        """
        pass


class Neo4jWriter(KGWriter):
    """Writes a knowledge graph to a Neo4j database.

    Args:
        driver (neo4j.driver): The Neo4j driver to connect to the database.
        neo4j_database (Optional[str]): The name of the Neo4j database to write to. Defaults to 'neo4j' if not provided.
    """

    def __init__(
        self,
        driver: neo4j.driver,
        neo4j_database: Optional[str] = None,
    ):
        self.driver = driver
        self.neo4j_database = neo4j_database

    def _upsert_node(self, node: Neo4jNode) -> None:
        """Upserts a single node into the Neo4j database."

        Args:
            node (Neo4jNode): The node to upsert into the database.
        """
        # Create the initial node
        parameters = {"id": node.id}
        if node.properties:
            parameters.update(node.properties)
        properties = (
            "{" + ", ".join(f"{key}: ${key}" for key in parameters.keys()) + "}"
        )
        query = UPSERT_NODE_QUERY.format(label=node.label, properties=properties)
        result = self.driver.execute_query(query, parameters_=parameters)
        node_id = result.records[0]["elementID(n)"]
        # Add the embedding properties to the node
        if node.embedding_properties:
            for prop, vector in node.embedding_properties.items():
                upsert_vector(
                    driver=self.driver,
                    node_id=node_id,
                    embedding_property=prop,
                    vector=vector,
                    neo4j_database=self.neo4j_database,
                )

    def _upsert_relationship(self, rel: Neo4jRelationship) -> None:
        """Upserts a single relationship into the Neo4j database.

        Args:
            rel (Neo4jRelationship): The relationship to upsert into the database.
        """
        # Create the initial relationship
        parameters = {
            "start_node_id": rel.start_node_id,
            "end_node_id": rel.end_node_id,
        }
        if rel.properties:
            properties = (
                "{" + ", ".join(f"{key}: ${key}" for key in rel.properties.keys()) + "}"
            )
            parameters.update(rel.properties)
        else:
            properties = "{}"
        query = UPSERT_RELATIONSHIP_QUERY.format(
            type=rel.type,
            properties=properties,
        )
        result = self.driver.execute_query(query, parameters_=parameters)
        rel_id = result.records[0]["elementID(r)"]
        # Add the embedding properties to the relationship
        if rel.embedding_properties:
            for prop, vector in rel.embedding_properties.items():
                upsert_vector_on_relationship(
                    driver=self.driver,
                    rel_id=rel_id,
                    embedding_property=prop,
                    vector=vector,
                    neo4j_database=self.neo4j_database,
                )

    @validate_call
    async def run(self, graph: Neo4jGraph) -> KGWriterModel:
        """Upserts a knowledge graph into a Neo4j database.

        Args:
            graph (Neo4jGraph): The knowledge graph to upsert into the database.
        """
        try:
            for node in graph.nodes:
                self._upsert_node(node)

            for rel in graph.relationships:
                self._upsert_relationship(rel)

            return KGWriterModel(status="SUCCESS")
        except neo4j.exceptions.ClientError:
            return KGWriterModel(status="FAILURE")
