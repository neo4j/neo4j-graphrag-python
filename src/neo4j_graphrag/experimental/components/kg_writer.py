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
from typing import Any, Dict, Literal, Optional, Tuple, List

import neo4j
from pydantic import validate_call

from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)
from neo4j_graphrag.experimental.pipeline.component import Component, DataModel
from neo4j_graphrag.neo4j_queries import UPSERT_NODE_QUERY, UPSERT_NODES_QUERY, \
    UPSERT_NODE_EMBEDDINGS_QUERY, UPSERT_RELATIONSHIP_QUERY, UPSERT_RELATIONSHIPS_QUERY

logger = logging.getLogger(__name__)


def chunks(xs, n=1_000):
    n = max(1, n)
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def get_node_embeddings(graph: Neo4jGraph) -> List:
    node_embeddings = []
    for node in graph.nodes:
        if node.embedding_properties:
            for k, v in node.embedding_properties.items():
                node_embeddings.append({'id': node.id, 'k': k, 'v': v})
    return node_embeddings


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
        max_concurrency: int = 5,
    ):
        self.driver = driver
        self.neo4j_database = neo4j_database
        self.max_concurrency = max_concurrency

    def _db_setup(self) -> None:
        # create index on __Entity__.id
        self.driver.execute_query(
            "CREATE INDEX __entity__id IF NOT EXISTS  FOR (n:__Entity__) ON (n.id)"
        )

    async def _async_db_setup(self) -> None:
        # create index on __Entity__.id
        await self.driver.execute_query(
            "CREATE INDEX __entity__id IF NOT EXISTS  FOR (n:__Entity__) ON (n.id)"
        )

    def _get_node_query(self, node: Neo4jNode) -> Tuple[str, Dict[str, Any]]:
        # Create the initial node
        parameters = {
            "id": node.id,
            "properties": node.properties or {},
            "embeddings": node.embedding_properties,
        }
        query = UPSERT_NODE_QUERY.format(label=node.label)
        return query, parameters

    def _upsert_node(self, node: Neo4jNode) -> None:
        """Upserts a single node into the Neo4j database."

        Args:
            node (Neo4jNode): The node to upsert into the database.
        """
        query, parameters = self._get_node_query(node)
        self.driver.execute_query(query, parameters_=parameters)

    async def _async_upsert_node(
        self,
        node: Neo4jNode,
        sem: asyncio.Semaphore,
    ) -> None:
        """Asynchronously upserts a single node into the Neo4j database."

        Args:
            node (Neo4jNode): The node to upsert into the database.
        """
        async with sem:
            query, parameters = self._get_node_query(node)
            await self.driver.execute_query(query, parameters_=parameters)

    def _upsert_nodes(self, nodes: List[Neo4jNode]) -> None:
        """Upserts a list of nodes into the Neo4j database."

        Args:
            nodes List[Neo4jNode]: The nodes to upsert into the database.
        """

        self.driver.execute_query(UPSERT_NODES_QUERY,
                                  parameters_={
                                      'records': [{
                                          "id": n.id,
                                          "properties": n.properties or {},
                                          "label": n.label
                                      } for n in nodes]})

    async def _async_upsert_nodes(
        self,
        nodes: List[Neo4jNode],
        sem: asyncio.Semaphore,
    ) -> None:
        """Asynchronously upserts a single node into the Neo4j database."

        Args:
            nodes List[Neo4jNode]: The nodes to upsert into the database.
        """
        async with sem:
            await self.driver.execute_query(UPSERT_NODES_QUERY,
                                            parameters_={
                                                'records': [{
                                                    "id": n.id,
                                                    "properties": n.properties or {},
                                                    "label": n.label
                                                } for n in nodes]})

    def _upsert_node_embeddings(self, node_embeddings: List) -> None:
        self.driver.execute_query(UPSERT_NODE_EMBEDDINGS_QUERY,
                                  parameters_={'records': node_embeddings})

    async def _async_upsert_node_embeddings(
        self,
        node_embeddings: List,
        sem: asyncio.Semaphore,
    ) -> None:
        async with sem:
            await self.driver.execute_query(UPSERT_NODE_EMBEDDINGS_QUERY,
                                            parameters_={'records': node_embeddings})

    def _get_rel_query(self, rel: Neo4jRelationship) -> Tuple[str, Dict[str, Any]]:
        # Create the initial relationship
        parameters = {
            "start_node_id": rel.start_node_id,
            "end_node_id": rel.end_node_id,
            "properties": rel.properties or {},
            "embeddings": rel.embedding_properties,
        }
        query = UPSERT_RELATIONSHIP_QUERY.format(
            type=rel.type,
        )
        return query, parameters

    def _upsert_relationship(self, rel: Neo4jRelationship) -> None:
        """Upserts a single relationship into the Neo4j database.

        Args:
            rel (Neo4jRelationship): The relationship to upsert into the database.
        """
        query, parameters = self._get_rel_query(rel)
        self.driver.execute_query(query, parameters_=parameters)

    async def _async_upsert_relationship(
        self, rel: Neo4jRelationship, sem: asyncio.Semaphore
    ) -> None:
        """Asynchronously upserts a single relationship into the Neo4j database.

        Args:
            rel (Neo4jRelationship): The relationship to upsert into the database.
        """
        async with sem:
            query, parameters = self._get_rel_query(rel)
            await self.driver.execute_query(query, parameters_=parameters)

    def _upsert_relationships(self, rels: List[Neo4jRelationship]) -> None:
        """Upserts a single relationship into the Neo4j database.

        Args:
            rels (List[Neo4jRelationship]): The relationships to upsert into the database.
        """
        self.driver.execute_query(UPSERT_RELATIONSHIPS_QUERY, parameters_={
            "records": [{
                "start_node_id": r.start_node_id,
                "end_node_id": r.end_node_id,
                "type": r.type,
                "properties": r.properties or {}} for r in rels]

        })

    async def _async_upsert_relationships(
        self, rels: List[Neo4jRelationship], sem: asyncio.Semaphore
    ) -> None:
        """Asynchronously upserts a single relationship into the Neo4j database.

        Args:
            rels (List[Neo4jRelationship]): The relationships to upsert into the database.
        """
        async with sem:
            await self.driver.execute_query(UPSERT_RELATIONSHIPS_QUERY,
                                            parameters_={
                                                "records": [{
                                                    "start_node_id": r.start_node_id,
                                                    "end_node_id": r.end_node_id,
                                                    "type": r.type,
                                                    "properties": r.properties or {}}
                                                    for r in rels]

                                            })

    @validate_call
    async def run(self, graph: Neo4jGraph, chunk_size: int = 1_000) -> KGWriterModel:
        """Upserts a knowledge graph into a Neo4j database.

        Args:
            chunk_size: The amount of nodes and rels to batch in each upsert query
            graph (Neo4jGraph): The knowledge graph to upsert into the database.
        """
        try:
            if inspect.iscoroutinefunction(self.driver.execute_query):
                await self._async_db_setup()
                sem = asyncio.Semaphore(self.max_concurrency)
                node_tasks = [
                    self._async_upsert_nodes(nodes, sem) for nodes in
                    chunks(graph.nodes, chunk_size)
                ]
                await asyncio.gather(*node_tasks)

                node_emb_tasks = [
                    self._async_upsert_node_embeddings(node_embs, sem) for node_embs in
                    chunks(get_node_embeddings(graph), chunk_size)
                ]
                await asyncio.gather(*node_emb_tasks)

                rel_tasks = [
                    self._async_upsert_relationships(rels, sem)
                    for rels in chunks(graph.relationships, chunk_size)
                ]
                await asyncio.gather(*rel_tasks)
            else:
                self._db_setup()

                for nodes in chunks(graph.nodes, chunk_size):
                    self._upsert_nodes(nodes)

                for node_embs in chunks(get_node_embeddings(graph), chunk_size):
                    self._upsert_node_embeddings(node_embs)

                for rels in chunks(graph.relationships, chunk_size):
                    self._upsert_relationships(rels)

            return KGWriterModel(status="SUCCESS")
        except neo4j.exceptions.ClientError as e:
            logger.exception(e)
            return KGWriterModel(status="FAILURE")
