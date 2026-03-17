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

import os
import logging
from abc import abstractmethod
from typing import Any, Generator, Literal, Optional

import neo4j
from pydantic import validate_call

from neo4j_graphrag.experimental.components.filename_collision_handler import (
    FilenameCollisionHandler,
)
from neo4j_graphrag.experimental.components.parquet_formatter import (
    Neo4jGraphParquetFormatter,
)
from neo4j_graphrag.experimental.components.parquet_output import (
    ParquetOutputDestination,
)
from neo4j_graphrag.experimental.components.types import (
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)
from neo4j_graphrag.experimental.pipeline.component import Component, DataModel
from neo4j_graphrag.neo4j_queries import (
    upsert_node_query,
    upsert_relationship_query,
    db_cleaning_query,
)
from neo4j_graphrag.utils.version_utils import (
    get_version,
    is_version_5_23_or_above,
    is_version_5_24_or_above,
)
from neo4j_graphrag.utils import driver_config

logger = logging.getLogger(__name__)


def _build_columns_from_schema(
    schema: Any, primary_key_names: list[str]
) -> list[dict[str, Any]]:
    """Build a list of column dicts (name, type, is_primary_key) from a PyArrow schema."""
    columns: list[dict[str, Any]] = []
    for i in range(len(schema)):
        field = schema.field(i)
        type_info = Neo4jGraphParquetFormatter.pyarrow_type_to_type_info(field.type)
        columns.append(
            {
                "name": field.name,
                "type": type_info.source_type,
                "is_primary_key": field.name in primary_key_names,
            }
        )
    return columns


def batched(rows: list[Any], batch_size: int) -> Generator[list[Any], None, None]:
    index = 0
    for i in range(0, len(rows), batch_size):
        start = i
        end = min(start + batch_size, len(rows))
        batch = rows[start:end]
        yield batch
        index += 1


def _graph_stats(
    graph: Neo4jGraph,
    nodes_per_label: Optional[dict[str, int]] = None,
    rel_per_type: Optional[dict[str, int]] = None,
    input_files_count: int = 0,
    input_files_total_size_bytes: int = 0,
) -> dict[str, Any]:
    """Build the statistics dict for writer metadata.

    Schema:
        node_count, relationship_count, nodes_per_label, rel_per_type,
        input_files_count, input_files_total_size_bytes.
    """
    if nodes_per_label is None:
        nodes_per_label = {}
        for node in graph.nodes:
            nodes_per_label[node.label] = nodes_per_label.get(node.label, 0) + 1
    if rel_per_type is None:
        rel_per_type = {}
        for rel in graph.relationships:
            rel_per_type[rel.type] = rel_per_type.get(rel.type, 0) + 1
    return {
        "node_count": len(graph.nodes),
        "relationship_count": len(graph.relationships),
        "nodes_per_label": nodes_per_label,
        "rel_per_type": rel_per_type,
        "input_files_count": input_files_count,
        "input_files_total_size_bytes": input_files_total_size_bytes,
    }


class KGWriterModel(DataModel):
    """Data model for the output of the Knowledge Graph writer.

    Attributes:
        status: Whether the write operation was successful ("SUCCESS" or "FAILURE").
        metadata: Optional dict. When status is SUCCESS, contains at least:
            - "statistics": dict with node_count, relationship_count, nodes_per_label,
              rel_per_type, input_files_count, input_files_total_size_bytes.
            - "files": list of file descriptors with file_path, etc. (ParquetWriter).
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
        neo4j_database (Optional[str]): The name of the Neo4j database. If not provided, this defaults to the server's default database ("neo4j" by default) (`see reference to documentation <https://neo4j.com/docs/operations-manual/current/database-administration/#manage-databases-default>`_).
        batch_size (int): The number of nodes or relationships to write to the database in a batch. Defaults to 1000.

    Example:

    .. code-block:: python

        from neo4j import GraphDatabase
        from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
        from neo4j_graphrag.experimental.pipeline import Pipeline

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")
        DATABASE = "neo4j"

        driver = GraphDatabase.driver(URI, auth=AUTH)
        writer = Neo4jWriter(driver=driver, neo4j_database=DATABASE)

        pipeline = Pipeline()
        pipeline.add_component(writer, "writer")

    """

    def __init__(
        self,
        driver: neo4j.Driver,
        neo4j_database: Optional[str] = None,
        batch_size: int = 1000,
        clean_db: bool = True,
    ):
        self.driver = driver_config.override_user_agent(driver)
        self.neo4j_database = neo4j_database
        self.batch_size = batch_size
        self._clean_db = clean_db
        version_tuple, _, _ = get_version(self.driver, self.neo4j_database)
        self.is_version_5_23_or_above = is_version_5_23_or_above(version_tuple)
        self.is_version_5_24_or_above = is_version_5_24_or_above(version_tuple)

    def _db_setup(self) -> None:
        self.driver.execute_query("""
        CREATE INDEX __entity__tmp_internal_id IF NOT EXISTS FOR (n:__KGBuilder__) ON (n.__tmp_internal_id)
        """)

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
        """Upserts a batch of nodes into the Neo4j database.

        Args:
            nodes (list[Neo4jNode]): The nodes batch to upsert into the database.
        """
        parameters = {"rows": self._nodes_to_rows(nodes, lexical_graph_config)}
        query = upsert_node_query(
            support_variable_scope_clause=self.is_version_5_23_or_above,
            support_dynamic_labels=self.is_version_5_24_or_above,
        )
        self.driver.execute_query(
            query,
            parameters_=parameters,
            database_=self.neo4j_database,
        )
        return None

    @staticmethod
    def _relationships_to_rows(
        relationships: list[Neo4jRelationship],
    ) -> list[dict[str, Any]]:
        return [relationship.model_dump() for relationship in relationships]

    def _upsert_relationships(self, rels: list[Neo4jRelationship]) -> None:
        """Upserts a batch of relationships into the Neo4j database.

        Args:
            rels (list[Neo4jRelationship]): The relationships batch to upsert into the database.
        """
        parameters = {"rows": self._relationships_to_rows(rels)}
        query = upsert_relationship_query(
            support_variable_scope_clause=self.is_version_5_23_or_above
        )
        self.driver.execute_query(
            query,
            parameters_=parameters,
            database_=self.neo4j_database,
        )

    def _db_cleaning(self) -> None:
        query = db_cleaning_query(
            support_variable_scope_clause=self.is_version_5_23_or_above,
            batch_size=self.batch_size,
        )
        with self.driver.session() as session:
            session.run(query)

    @validate_call
    async def run(
        self,
        graph: Neo4jGraph,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> KGWriterModel:
        """Upserts a knowledge graph into a Neo4j database.

        Args:
            graph (Neo4jGraph): The knowledge graph to upsert into the database.
            lexical_graph_config (LexicalGraphConfig): Node labels and relationship types for the lexical graph.
        """
        try:
            self._db_setup()

            for batch in batched(graph.nodes, self.batch_size):
                self._upsert_nodes(batch, lexical_graph_config)

            for batch in batched(graph.relationships, self.batch_size):
                self._upsert_relationships(batch)

            if self._clean_db:
                self._db_cleaning()

            return KGWriterModel(
                status="SUCCESS",
                metadata={
                    "statistics": _graph_stats(graph),
                    "files": [],
                },
            )
        except neo4j.exceptions.ClientError as e:
            logger.exception(e)
            return KGWriterModel(status="FAILURE", metadata={"error": str(e)})


class ParquetWriter(KGWriter):
    """Writes a knowledge graph to Parquet files using Neo4jGraphParquetFormatter.

    Writes one Parquet file per node label and one per (head_label, relationship_type, tail_label)
    to the given destinations, e.g. ``Person.parquet``, ``Person_KNOWS_Person.parquet``.

    Args:
        nodes_dest (ParquetOutputDestination): Destination for node Parquet files.
        relationships_dest (ParquetOutputDestination): Destination for relationship Parquet files.
        collision_handler (FilenameCollisionHandler): Handler for resolving filename collisions.
        prefix (str): Optional filename prefix for all written files. Defaults to "".

    Example:

    .. code-block:: python

        from neo4j_graphrag.experimental.components.filename_collision_handler import FilenameCollisionHandler
        from neo4j_graphrag.experimental.components.kg_writer import ParquetWriter
        from neo4j_graphrag.experimental.components.parquet_output import ParquetOutputDestination
        from neo4j_graphrag.experimental.pipeline import Pipeline

        # Provide your own implementation of ParquetOutputDestination (local, GCS, S3, etc.)
        nodes_dest: ParquetOutputDestination = ...
        relationships_dest: ParquetOutputDestination = ...

        writer = ParquetWriter(
            nodes_dest=nodes_dest,
            relationships_dest=relationships_dest,
            collision_handler=FilenameCollisionHandler(),
        )
        pipeline = Pipeline()
        pipeline.add_component(writer, "writer")
    """

    def __init__(
        self,
        nodes_dest: ParquetOutputDestination,
        relationships_dest: ParquetOutputDestination,
        collision_handler: FilenameCollisionHandler,
        prefix: str = "",
    ) -> None:
        self.nodes_dest = nodes_dest
        self.relationships_dest = relationships_dest
        self.collision_handler = collision_handler
        self.prefix = prefix

    @validate_call
    async def run(
        self,
        graph: Neo4jGraph,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
        schema: Optional[dict[str, Any]] = None,
    ) -> KGWriterModel:
        """Write the knowledge graph to Parquet files via Neo4jGraphParquetFormatter.

        Args:
            graph (Neo4jGraph): The knowledge graph to write.
            lexical_graph_config (LexicalGraphConfig): Used by the formatter for
                lexical graph labels (e.g. __Entity__) and key properties.
            schema (Optional[dict[str, Any]]): Optional GraphSchema as a dictionary for
                uniqueness constraints and key properties. If not provided, ``__id__`` is used.
        """
        try:
            formatter = Neo4jGraphParquetFormatter(schema=schema)
            data, file_metadata, stats = formatter.format_graph(
                graph, lexical_graph_config, prefix=self.prefix
            )

            meta_by_filename: dict[str, Any] = {m.filename: m for m in file_metadata}
            files: list[dict[str, Any]] = []
            node_label_to_source_name: dict[str, str] = {}

            base_nodes = self.nodes_dest.output_path.rstrip("/")
            for filename, content in data["nodes"].items():
                meta = meta_by_filename[filename]
                unique_filename = self.collision_handler.get_unique_filename(
                    filename, self.nodes_dest.output_path
                )
                await self.nodes_dest.write(content, unique_filename)
                file_path = os.path.join(base_nodes, unique_filename)

                resolved_stem = (
                    unique_filename[:-8]
                    if unique_filename.endswith(".parquet")
                    else unique_filename
                )
                if meta.node_label is not None:
                    node_label_to_source_name[meta.node_label] = resolved_stem

                columns = _build_columns_from_schema(
                    meta.schema,
                    meta.key_properties or [],
                )
                name = meta.node_label or (
                    meta.labels[0] if meta.labels else resolved_stem
                )
                files.append(
                    {
                        "name": name,
                        "file_path": file_path,
                        "columns": columns,
                        "is_node": True,
                        "labels": meta.labels or [],
                    }
                )

            base_rel = self.relationships_dest.output_path.rstrip("/")
            for filename, content in data["relationships"].items():
                meta = meta_by_filename[filename]
                unique_filename = self.collision_handler.get_unique_filename(
                    filename, self.relationships_dest.output_path
                )
                await self.relationships_dest.write(content, unique_filename)
                file_path = os.path.join(base_rel, unique_filename)

                start_node_source = node_label_to_source_name.get(
                    meta.relationship_head or "", meta.relationship_head or ""
                )
                end_node_source = node_label_to_source_name.get(
                    meta.relationship_tail or "", meta.relationship_tail or ""
                )
                columns = _build_columns_from_schema(
                    meta.schema,
                    ["from", "to"],
                )
                rel_name = (
                    f"{meta.relationship_head}_{meta.relationship_type}_{meta.relationship_tail}"
                    if meta.relationship_head
                    and meta.relationship_type
                    and meta.relationship_tail
                    else unique_filename[:-8]
                    if unique_filename.endswith(".parquet")
                    else unique_filename
                )
                files.append(
                    {
                        "name": rel_name,
                        "file_path": file_path,
                        "columns": columns,
                        "is_node": False,
                        "relationship_type": meta.relationship_type,
                        "start_node_source": start_node_source,
                        "start_node_primary_keys": meta.head_node_key_properties
                        or ["__id__"],
                        "end_node_source": end_node_source,
                        "end_node_primary_keys": meta.tail_node_key_properties
                        or ["__id__"],
                    }
                )

            logger.info(
                "Wrote %d node files and %d relationship files",
                len(data["nodes"]),
                len(data["relationships"]),
            )
            statistics = _graph_stats(
                graph,
                nodes_per_label=stats["nodes_per_label"],
                rel_per_type=stats["rel_per_type"],
                input_files_count=0,
                input_files_total_size_bytes=0,
            )
            return KGWriterModel(
                status="SUCCESS",
                metadata={
                    "statistics": statistics,
                    "files": files,
                },
            )
        except Exception as e:
            logger.exception(e)
            return KGWriterModel(status="FAILURE", metadata={"error": str(e)})
