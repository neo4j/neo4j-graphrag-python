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

import asyncio
import inspect
import logging
from abc import abstractmethod
from typing import Any, Generator, Literal, Optional

import neo4j
from pydantic import validate_call

from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    CHUNK_NODE_LABEL,
    DOCUMENT_NODE_LABEL,
)
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)
from neo4j_graphrag.experimental.pipeline.component import Component, DataModel
from neo4j_graphrag.neo4j_queries import UPSERT_NODE_QUERY, UPSERT_RELATIONSHIP_QUERY

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
        max_concurrency (int): The maximum number of concurrent tasks which can be used to make requests to the LLM.

    Example:

    .. code-block:: python

        from neo4j import AsyncGraphDatabase
        from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
        from neo4j_graphrag.experimental.pipeline import Pipeline

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")
        DATABASE = "neo4j"

        driver = AsyncGraphDatabase.driver(URI, auth=AUTH, database=DATABASE)
        writer = Neo4jWriter(driver=driver, neo4j_database=DATABASE)

        pipeline = Pipeline()
        pipeline.add_component(writer, "writer")

    """

    def __init__(
        self,
        driver: neo4j.driver,
        neo4j_database: Optional[str] = None,
        batch_size: int = 1000,
        max_concurrency: int = 5,
    ):
        self.driver = driver
        self.neo4j_database = neo4j_database
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency

    def _db_setup(self) -> None:
        # create index on __Entity__.id
        # used when creating the relationships
        self.driver.execute_query(
            "CREATE INDEX __entity__id IF NOT EXISTS  FOR (n:__KGBuilder__) ON (n.id)"
        )

    async def _async_db_setup(self) -> None:
        # create index on __Entity__.id
        # used when creating the relationships
        await self.driver.execute_query(
            "CREATE INDEX __entity__id IF NOT EXISTS  FOR (n:__KGBuilder__) ON (n.id)"
        )

    @staticmethod
    def _nodes_to_rows(nodes: list[Neo4jNode]) -> list[dict[str, Any]]:
        rows = []
        for node in nodes:
            labels = [node.label]
            if node.label not in (CHUNK_NODE_LABEL, DOCUMENT_NODE_LABEL):
                labels.append("__Entity__")
            row = node.model_dump()
            row["labels"] = labels
            rows.append(row)
        return rows

    def _upsert_nodes(self, nodes: list[Neo4jNode]) -> None:
        """Upserts a single node into the Neo4j database."

        Args:
            nodes (list[Neo4jNode]): The nodes batch to upsert into the database.
        """
        parameters = {"rows": self._nodes_to_rows(nodes)}
        self.driver.execute_query(UPSERT_NODE_QUERY, parameters_=parameters)

    async def _async_upsert_nodes(
        self,
        nodes: list[Neo4jNode],
        sem: asyncio.Semaphore,
    ) -> None:
        """Asynchronously upserts a single node into the Neo4j database."

        Args:
            nodes (list[Neo4jNode]): The nodes batch to upsert into the database.
        """
        async with sem:
            parameters = {"rows": self._nodes_to_rows(nodes)}
            await self.driver.execute_query(UPSERT_NODE_QUERY, parameters_=parameters)

    def _upsert_relationships(self, rels: list[Neo4jRelationship]) -> None:
        """Upserts a single relationship into the Neo4j database.

        Args:
            rels (list[Neo4jRelationship]): The relationships batch to upsert into the database.
        """
        parameters = {"rows": [rel.model_dump() for rel in rels]}
        self.driver.execute_query(UPSERT_RELATIONSHIP_QUERY, parameters_=parameters)

    async def _async_upsert_relationships(
        self, rels: list[Neo4jRelationship], sem: asyncio.Semaphore
    ) -> None:
        """Asynchronously upserts a single relationship into the Neo4j database.

        Args:
            rels (list[Neo4jRelationship]): The relationships batch to upsert into the database.
        """
        async with sem:
            parameters = {"rows": [rel.model_dump() for rel in rels]}
            await self.driver.execute_query(
                UPSERT_RELATIONSHIP_QUERY, parameters_=parameters
            )

    @validate_call
    async def run(self, graph: Neo4jGraph) -> KGWriterModel:
        """Upserts a knowledge graph into a Neo4j database.

        Args:
            graph (Neo4jGraph): The knowledge graph to upsert into the database.
        """
        try:
            if inspect.iscoroutinefunction(self.driver.execute_query):
                await self._async_db_setup()
                sem = asyncio.Semaphore(self.max_concurrency)
                node_tasks = [
                    self._async_upsert_nodes(batch, sem)
                    for batch in batched(graph.nodes, self.batch_size)
                ]
                await asyncio.gather(*node_tasks)

                rel_tasks = [
                    self._async_upsert_relationships(batch, sem)
                    for batch in batched(graph.relationships, self.batch_size)
                ]
                await asyncio.gather(*rel_tasks)
            else:
                self._db_setup()

                for batch in batched(graph.nodes, self.batch_size):
                    self._upsert_nodes(batch)

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
