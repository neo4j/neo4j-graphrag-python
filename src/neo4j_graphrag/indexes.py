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
from typing import Literal, Optional

import neo4j
from pydantic import ValidationError

from neo4j_graphrag.neo4j_queries import (
    UPSERT_VECTOR_ON_NODE_QUERY,
    UPSERT_VECTOR_ON_RELATIONSHIP_QUERY,
)

from .exceptions import Neo4jIndexError, Neo4jInsertionError
from .types import FulltextIndexModel, VectorIndexModel

logger = logging.getLogger(__name__)


def create_vector_index(
    driver: neo4j.Driver,
    name: str,
    label: str,
    embedding_property: str,
    dimensions: int,
    similarity_fn: Literal["euclidean", "cosine"],
    neo4j_database: Optional[str] = None,
) -> None:
    """
    This method constructs a Cypher query and executes it
    to create a new vector index in Neo4j.

    See Cypher manual on `creating vector indexes <https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/#create-vector-index>`_.

    Important: This operation will fail if an index with the same name already exists.
    Ensure that the index name provided is unique within the database context.

    Example:

    .. code-block:: python

        from neo4j import GraphDatabase
        from neo4j_graphrag.indexes import create_vector_index

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")

        INDEX_NAME = "vector-index-name"

        # Connect to Neo4j database
        driver = GraphDatabase.driver(URI, auth=AUTH)

        # Creating the index
        create_vector_index(
            driver,
            INDEX_NAME,
            label="Document",
            embedding_property="vectorProperty",
            dimensions=1536,
            similarity_fn="euclidean",
        )


    Args:
        driver (neo4j.Driver): Neo4j Python driver instance.
        name (str): The unique name of the index.
        label (str): The node label to be indexed.
        embedding_property (str): The property key of a node which contains embedding values.
        dimensions (int): Vector embedding dimension
        similarity_fn (str): case-insensitive values for the vector similarity function:
            ``euclidean`` or ``cosine``.
        neo4j_database (Optional[str]): The name of the Neo4j database. If not provided, this defaults to "neo4j" in the database (`see reference to documentation <https://neo4j.com/docs/operations-manual/current/database-administration/#manage-databases-default>`_).

    Raises:
        ValueError: If validation of the input arguments fail.
        neo4j.exceptions.ClientError: If creation of vector index fails.
    """
    try:
        VectorIndexModel(
            driver=driver,
            name=name,
            label=label,
            embedding_property=embedding_property,
            dimensions=dimensions,
            similarity_fn=similarity_fn,
        )
    except ValidationError as e:
        raise Neo4jIndexError(
            f"Error for inputs to create_vector_index {e.errors()}"
        ) from e

    try:
        query = (
            f"CREATE VECTOR INDEX $name FOR (n:{label}) ON n.{embedding_property} OPTIONS "
            "{ indexConfig: { `vector.dimensions`: toInteger($dimensions), `vector.similarity_function`: $similarity_fn } }"
        )
        logger.info(f"Creating vector index named '{name}'")
        driver.execute_query(
            query,
            {"name": name, "dimensions": dimensions, "similarity_fn": similarity_fn},
            database_=neo4j_database,
        )
    except neo4j.exceptions.ClientError as e:
        raise Neo4jIndexError(f"Neo4j vector index creation failed: {e.message}") from e


def create_fulltext_index(
    driver: neo4j.Driver,
    name: str,
    label: str,
    node_properties: list[str],
    neo4j_database: Optional[str] = None,
) -> None:
    """
    This method constructs a Cypher query and executes it
    to create a new fulltext index in Neo4j.

    See Cypher manual on `creating fulltext indexes <https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/#create-full-text-indexes>`_.

    Important: This operation will fail if an index with the same name already exists.
    Ensure that the index name provided is unique within the database context.

    Example:

    .. code-block:: python

        from neo4j import GraphDatabase
        from neo4j_graphrag.indexes import create_fulltext_index

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")

        INDEX_NAME = "fulltext-index-name"

        # Connect to Neo4j database
        driver = GraphDatabase.driver(URI, auth=AUTH)

        # Creating the index
        create_fulltext_index(
            driver,
            INDEX_NAME,
            label="Document",
            node_properties=["vectorProperty"],
        )


    Args:
        driver (neo4j.Driver): Neo4j Python driver instance.
        name (str): The unique name of the index.
        label (str): The node label to be indexed.
        node_properties (list[str]): The node properties to create the fulltext index on.
        neo4j_database (Optional[str]): The name of the Neo4j database. If not provided, this defaults to "neo4j" in the database (`see reference to documentation <https://neo4j.com/docs/operations-manual/current/database-administration/#manage-databases-default>`_).

    Raises:
        ValueError: If validation of the input arguments fail.
        neo4j.exceptions.ClientError: If creation of fulltext index fails.
    """
    try:
        FulltextIndexModel(
            driver=driver, name=name, label=label, node_properties=node_properties
        )
    except ValidationError as e:
        raise Neo4jIndexError(
            f"Error for inputs to create_fulltext_index: {e.errors()}"
        ) from e

    try:
        query = (
            "CREATE FULLTEXT INDEX $name "
            f"FOR (n:`{label}`) ON EACH "
            f"[{', '.join(['n.`' + prop + '`' for prop in node_properties])}]"
        )
        logger.info(f"Creating fulltext index named '{name}'")
        driver.execute_query(query, {"name": name}, database_=neo4j_database)
    except neo4j.exceptions.ClientError as e:
        raise Neo4jIndexError(
            f"Neo4j fulltext index creation failed {e.message}"
        ) from e


def drop_index_if_exists(
    driver: neo4j.Driver, name: str, neo4j_database: Optional[str] = None
) -> None:
    """
    This method constructs a Cypher query and executes it
    to drop an index in Neo4j, if the index exists.
    See Cypher manual on `dropping vector indexes <https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/#drop-vector-indexes>`_.

    Example:

    .. code-block:: python

        from neo4j import GraphDatabase
        from neo4j_graphrag.indexes import drop_index_if_exists

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")

        INDEX_NAME = "fulltext-index-name"

        # Connect to Neo4j database
        driver = GraphDatabase.driver(URI, auth=AUTH)

        # Dropping the index if it exists
        drop_index_if_exists(
            driver,
            INDEX_NAME,
        )


    Args:
        driver (neo4j.Driver): Neo4j Python driver instance.
        name (str): The name of the index to delete.
        neo4j_database (Optional[str]): The name of the Neo4j database. If not provided, this defaults to "neo4j" in the database (`see reference to documentation <https://neo4j.com/docs/operations-manual/current/database-administration/#manage-databases-default>`_).

    Raises:
        neo4j.exceptions.ClientError: If dropping of index fails.
    """
    try:
        query = "DROP INDEX $name IF EXISTS"
        parameters = {
            "name": name,
        }
        logger.info(f"Dropping index named '{name}'")
        driver.execute_query(query, parameters, database_=neo4j_database)
    except neo4j.exceptions.ClientError as e:
        raise Neo4jIndexError(f"Dropping Neo4j index failed: {e.message}") from e


def upsert_vector(
    driver: neo4j.Driver,
    node_id: int,
    embedding_property: str,
    vector: list[float],
    neo4j_database: Optional[str] = None,
) -> None:
    """
    This method constructs a Cypher query and executes it to upsert (insert or update) a vector property on a specific node.

    Example:

    .. code-block:: python

        from neo4j import GraphDatabase
        from neo4j_graphrag.indexes import upsert_vector

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")

        # Connect to Neo4j database
        driver = GraphDatabase.driver(URI, auth=AUTH)

        # Upsert the vector data
        upsert_vector(
            driver,
            node_id="nodeId",
            embedding_property="vectorProperty",
            vector=...,
        )

    Args:
        driver (neo4j.Driver): Neo4j Python driver instance.
        node_id (int): The id of the node.
        embedding_property (str): The name of the property to store the vector in.
        vector (list[float]): The vector to store.
        neo4j_database (Optional[str]): The name of the Neo4j database. If not provided, this defaults to "neo4j" in the database (`see reference to documentation <https://neo4j.com/docs/operations-manual/current/database-administration/#manage-databases-default>`_).

    Raises:
        Neo4jInsertionError: If upserting of the vector fails.
    """
    try:
        parameters = {
            "id": node_id,
            "embedding_property": embedding_property,
            "vector": vector,
        }
        driver.execute_query(
            UPSERT_VECTOR_ON_NODE_QUERY, parameters, database_=neo4j_database
        )
    except neo4j.exceptions.ClientError as e:
        raise Neo4jInsertionError(
            f"Upserting vector to Neo4j failed: {e.message}"
        ) from e


def upsert_vector_on_relationship(
    driver: neo4j.Driver,
    rel_id: int,
    embedding_property: str,
    vector: list[float],
    neo4j_database: Optional[str] = None,
) -> None:
    """
    This method constructs a Cypher query and executes it to upsert (insert or update) a vector property on a specific relationship.

    Example:

    .. code-block:: python

        from neo4j import GraphDatabase
        from neo4j_graphrag.indexes import upsert_vector_on_relationship

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")

        # Connect to Neo4j database
        driver = GraphDatabase.driver(URI, auth=AUTH)

        # Upsert the vector data
        upsert_vector_on_relationship(
            driver,
            node_id="nodeId",
            embedding_property="vectorProperty",
            vector=...,
        )

    Args:
        driver (neo4j.Driver): Neo4j Python driver instance.
        rel_id (int): The id of the relationship.
        embedding_property (str): The name of the property to store the vector in.
        vector (list[float]): The vector to store.
        neo4j_database (Optional[str]): The name of the Neo4j database. If not provided, this defaults to "neo4j" in the database (`see reference to documentation <https://neo4j.com/docs/operations-manual/current/database-administration/#manage-databases-default>`_).

    Raises:
        Neo4jInsertionError: If upserting of the vector fails.
    """
    try:
        parameters = {
            "id": rel_id,
            "embedding_property": embedding_property,
            "vector": vector,
        }
        driver.execute_query(
            UPSERT_VECTOR_ON_RELATIONSHIP_QUERY, parameters, database_=neo4j_database
        )
    except neo4j.exceptions.ClientError as e:
        raise Neo4jInsertionError(
            f"Upserting vector to Neo4j failed: {e.message}"
        ) from e


async def async_upsert_vector(
    driver: neo4j.AsyncDriver,
    node_id: int,
    embedding_property: str,
    vector: list[float],
    neo4j_database: Optional[str] = None,
) -> None:
    """
    This method constructs a Cypher query and asynchronously executes it
    to upsert (insert or update) a vector property on a specific node.

    Example:

    .. code-block:: python

        from neo4j import AsyncGraphDatabase
        from neo4j_graphrag.indexes import upsert_vector

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")

        # Connect to Neo4j database
        driver = AsyncGraphDatabase.driver(URI, auth=AUTH)

        # Upsert the vector data
        async_upsert_vector(
            driver,
            node_id="nodeId",
            embedding_property="vectorProperty",
            vector=...,
        )

    Args:
        driver (neo4j.AsyncDriver): Neo4j Python asynchronous driver instance.
        node_id (int): The id of the node.
        embedding_property (str): The name of the property to store the vector in.
        vector (list[float]): The vector to store.
        neo4j_database (Optional[str]): The name of the Neo4j database. If not provided, this defaults to "neo4j" in the database (`see reference to documentation <https://neo4j.com/docs/operations-manual/current/database-administration/#manage-databases-default>`_).

    Raises:
        Neo4jInsertionError: If upserting of the vector fails.
    """
    try:
        parameters = {
            "id": node_id,
            "embedding_property": embedding_property,
            "vector": vector,
        }
        await driver.execute_query(
            UPSERT_VECTOR_ON_NODE_QUERY, parameters, database_=neo4j_database
        )
    except neo4j.exceptions.ClientError as e:
        raise Neo4jInsertionError(
            f"Upserting vector to Neo4j failed: {e.message}"
        ) from e


async def async_upsert_vector_on_relationship(
    driver: neo4j.AsyncDriver,
    rel_id: int,
    embedding_property: str,
    vector: list[float],
    neo4j_database: Optional[str] = None,
) -> None:
    """
    This method constructs a Cypher query and asynchronously executes it
    to upsert (insert or update) a vector property on a specific relationship.

    Example:

    .. code-block:: python

        from neo4j import AsyncGraphDatabase
        from neo4j_graphrag.indexes import upsert_vector_on_relationship

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")

        # Connect to Neo4j database
        driver = AsyncGraphDatabase.driver(URI, auth=AUTH)

        # Upsert the vector data
        async_upsert_vector_on_relationship(
            driver,
            node_id="nodeId",
            embedding_property="vectorProperty",
            vector=...,
        )

    Args:
        driver (neo4j.AsyncDriver): Neo4j Python asynchronous driver instance.
        rel_id (int): The id of the relationship.
        embedding_property (str): The name of the property to store the vector in.
        vector (list[float]): The vector to store.
        neo4j_database (Optional[str]): The name of the Neo4j database. If not provided, this defaults to "neo4j" in the database (`see reference to documentation <https://neo4j.com/docs/operations-manual/current/database-administration/#manage-databases-default>`_).

    Raises:
        Neo4jInsertionError: If upserting of the vector fails.
    """
    try:
        parameters = {
            "id": rel_id,
            "embedding_property": embedding_property,
            "vector": vector,
        }
        await driver.execute_query(
            UPSERT_VECTOR_ON_RELATIONSHIP_QUERY, parameters, database_=neo4j_database
        )
    except neo4j.exceptions.ClientError as e:
        raise Neo4jInsertionError(
            f"Upserting vector to Neo4j failed: {e.message}"
        ) from e
