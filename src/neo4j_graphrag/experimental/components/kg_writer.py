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

import logging
from abc import abstractmethod
from typing import Any, Generator, Literal, Optional

import neo4j
from pydantic import validate_call

from neo4j_graphrag.experimental.components.types import (
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)
from neo4j_graphrag.experimental.pipeline.component import Component, DataModel
from neo4j_graphrag.neo4j_queries import (
    UPSERT_NODE_QUERY,
    UPSERT_NODE_QUERY_VARIABLE_SCOPE_CLAUSE,
    UPSERT_RELATIONSHIP_QUERY,
    UPSERT_RELATIONSHIP_QUERY_VARIABLE_SCOPE_CLAUSE,
)

logger = logging.getLogger(__name__)


def batched(rows: list[Any], batch_size: int) -> Generator[list[Any], None, None]:
    index = 0
    for i in range(0, len(rows), batch_size):
        start = i
        end = min(start + batch_size, len(rows))
        batch = rows[start:end]
        yield batch
        index += 1


class KGWriterModel(DataModel):
    """Data model for the output of the Knowledge Graph writer.

    Attributes:
        status (Literal["SUCCESS", "FAILURE"]): Whether the write operation was successful.
    """

    status: Literal["SUCCESS", "FAILURE"]
    metadata: Optional[dict[str, Any]] = None


class KGWriter(Component):
    """Abstract class used to write a knowledge graph to a data store."""

    @abstractmethod
    @validate_call
    async def run(
        self,
        graph: Neo4jGraph,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> KGWriterModel:
        """
        Writes the graph to a data store.

        Args:
            graph (Neo4jGraph): The knowledge graph to write to the data store.
            lexical_graph_config (LexicalGraphConfig): Node labels and relationship types in the lexical graph.
        """
        pass


class Neo4jWriter(KGWriter):
    """Writes a knowledge graph to a Neo4j database.

    Args:
        driver (neo4j.driver): The Neo4j driver to connect to the database.
        neo4j_database (Optional[str]): The name of the Neo4j database to write to. Defaults to 'neo4j' if not provided.
        batch_size (int): The number of nodes or relationships to write to the database in a batch. Defaults to 1000.

    Example:

    .. code-block:: python

        from neo4j import GraphDatabase
        from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
        from neo4j_graphrag.experimental.pipeline import Pipeline

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")
        DATABASE = "neo4j"

        driver = GraphDatabase.driver(URI, auth=AUTH, database=DATABASE)
        writer = Neo4jWriter(driver=driver, neo4j_database=DATABASE)

        pipeline = Pipeline()
        pipeline.add_component(writer, "writer")

    """

    def __init__(
        self,
        driver: neo4j.Driver,
        neo4j_database: Optional[str] = None,
        batch_size: int = 1000,
    ):
        self.driver = driver
        self.neo4j_database = neo4j_database
        self.batch_size = batch_size
        self.is_version_5_23_or_above = self._check_if_version_5_23_or_above()

    def _db_setup(self) -> None:
        # create index on __Entity__.id
        # used when creating the relationships
        self.driver.execute_query(
            "CREATE INDEX __entity__id IF NOT EXISTS  FOR (n:__KGBuilder__) ON (n.id)"
        )

    @staticmethod
    def _nodes_to_rows(
        nodes: list[Neo4jNode], lexical_graph_config: LexicalGraphConfig
    ) -> list[dict[str, Any]]:
        rows = []
        for node in nodes:
            labels = [node.label]
            if node.label not in lexical_graph_config.lexical_graph_node_labels:
                labels.append("__Entity__")
            row = node.model_dump()
            row["labels"] = labels
            rows.append(row)
        return rows

    def _upsert_nodes(
        self, nodes: list[Neo4jNode], lexical_graph_config: LexicalGraphConfig
    ) -> None:
        """Upserts a single node into the Neo4j database."

        Args:
            nodes (list[Neo4jNode]): The nodes batch to upsert into the database.
        """
        parameters = {"rows": self._nodes_to_rows(nodes, lexical_graph_config)}
        if self.is_version_5_23_or_above:
            self.driver.execute_query(
                UPSERT_NODE_QUERY_VARIABLE_SCOPE_CLAUSE, parameters_=parameters
            )
        else:
            self.driver.execute_query(UPSERT_NODE_QUERY, parameters_=parameters)

    def _get_version(self) -> tuple[int, ...]:
        records, _, _ = self.driver.execute_query(
            "CALL dbms.components()", database_=self.neo4j_database
        )
        version = records[0]["versions"][0]
        # Drop everything after the '-' first
        version_main, *_ = version.split("-")
        # Convert each number between '.' into int
        version_tuple = tuple(map(int, version_main.split(".")))
        # If no patch version, consider it's 0
        if len(version_tuple) < 3:
            version_tuple = (*version_tuple, 0)
        return version_tuple

    def _check_if_version_5_23_or_above(self) -> bool:
        """
        Check if the connected Neo4j database version supports the required features.

        Sets a flag if the connected Neo4j version is 5.23 or above.
        """
        version_tuple = self._get_version()
        return version_tuple >= (5, 23, 0)

    def _upsert_relationships(self, rels: list[Neo4jRelationship]) -> None:
        """Upserts a single relationship into the Neo4j database.

        Args:
            rels (list[Neo4jRelationship]): The relationships batch to upsert into the database.
        """
        parameters = {"rows": [rel.model_dump() for rel in rels]}
        if self.is_version_5_23_or_above:
            self.driver.execute_query(
                UPSERT_RELATIONSHIP_QUERY_VARIABLE_SCOPE_CLAUSE, parameters_=parameters
            )
        else:
            self.driver.execute_query(UPSERT_RELATIONSHIP_QUERY, parameters_=parameters)

    @validate_call
    async def run(
        self,
        graph: Neo4jGraph,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> KGWriterModel:
        """Upserts a knowledge graph into a Neo4j database.

        Args:
            graph (Neo4jGraph): The knowledge graph to upsert into the database.
            lexical_graph_config (LexicalGraphConfig):
        """
        try:
            self._db_setup()

            for batch in batched(graph.nodes, self.batch_size):
                self._upsert_nodes(batch, lexical_graph_config)

            for batch in batched(graph.relationships, self.batch_size):
                self._upsert_relationships(batch)

            return KGWriterModel(
                status="SUCCESS",
                metadata={
                    "node_count": len(graph.nodes),
                    "relationship_count": len(graph.relationships),
                },
            )
        except neo4j.exceptions.ClientError as e:
            logger.exception(e)
            return KGWriterModel(status="FAILURE", metadata={"error": str(e)})
