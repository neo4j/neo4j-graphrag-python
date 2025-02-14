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

from unittest.mock import MagicMock

import neo4j.exceptions
import pytest
from neo4j_graphrag.exceptions import Neo4jIndexError, Neo4jInsertionError
from neo4j_graphrag.indexes import (
    create_fulltext_index,
    create_vector_index,
    drop_index_if_exists,
    upsert_vector,
    upsert_vector_on_relationship,
    upsert_vectors,
)


def test_create_vector_index_happy_path(driver: MagicMock) -> None:
    create_query = (
        "CREATE VECTOR INDEX $name IF NOT EXISTS FOR (n:People) ON n.name OPTIONS "
        "{ indexConfig: { `vector.dimensions`: toInteger($dimensions), `vector.similarity_function`: $similarity_fn } }"
    )

    create_vector_index(driver, "my-index", "People", "name", 2048, "cosine")

    driver.execute_query.assert_called_once_with(
        create_query,
        {"name": "my-index", "dimensions": 2048, "similarity_fn": "cosine"},
        database_=None,
    )


def test_create_vector_index_fail_if_exists(driver: MagicMock) -> None:
    create_query = (
        "CREATE VECTOR INDEX $name  FOR (n:People) ON n.name OPTIONS "
        "{ indexConfig: { `vector.dimensions`: toInteger($dimensions), `vector.similarity_function`: $similarity_fn } }"
    )

    create_vector_index(
        driver, "my-index", "People", "name", 2048, "cosine", fail_if_exists=True
    )

    driver.execute_query.assert_called_once_with(
        create_query,
        {"name": "my-index", "dimensions": 2048, "similarity_fn": "cosine"},
        database_=None,
    )


def test_create_vector_index_ensure_escaping(driver: MagicMock) -> None:
    create_query = (
        "CREATE VECTOR INDEX $name IF NOT EXISTS FOR (n:People) ON n.name OPTIONS "
        "{ indexConfig: { `vector.dimensions`: toInteger($dimensions), `vector.similarity_function`: $similarity_fn } }"
    )

    create_vector_index(
        driver, "my-complicated-`-index", "People", "name", 2048, "cosine"
    )

    driver.execute_query.assert_called_once_with(
        create_query,
        {
            "name": "my-complicated-`-index",
            "dimensions": 2048,
            "similarity_fn": "cosine",
        },
        database_=None,
    )


def test_create_vector_index_negative_dimension(driver: MagicMock) -> None:
    with pytest.raises(Neo4jIndexError) as excinfo:
        create_vector_index(driver, "my-index", "People", "name", -5, "cosine")
    assert "Error for inputs to create_vector_index" in str(excinfo)


def test_create_vector_index_validation_error_dimensions(driver: MagicMock) -> None:
    with pytest.raises(Neo4jIndexError) as excinfo:
        create_vector_index(driver, "my-index", "People", "name", "no-dim", "cosine")  # type: ignore
    assert "Error for inputs to create_vector_index" in str(excinfo)


def test_create_vector_index_raises_error_with_neo4j_client_error(
    driver: MagicMock,
) -> None:
    driver.execute_query.side_effect = neo4j.exceptions.ClientError
    with pytest.raises(Neo4jIndexError):
        create_vector_index(driver, "my-index", "People", "name", 2048, "cosine")


def test_create_vector_index_validation_error_similarity_fn(driver: MagicMock) -> None:
    with pytest.raises(Neo4jIndexError) as excinfo:
        create_vector_index(driver, "my-index", "People", "name", 1536, "algebra")  # type: ignore
    assert "Error for inputs to create_vector_index" in str(excinfo)


def test_drop_index_if_exists(driver: MagicMock) -> None:
    drop_query = "DROP INDEX $name IF EXISTS"

    drop_index_if_exists(driver, "my-index")

    driver.execute_query.assert_called_once_with(
        drop_query,
        {"name": "my-index"},
        database_=None,
    )


def test_drop_index_if_exists_raises_error_with_neo4j_client_error(
    driver: MagicMock,
) -> None:
    drop_query = "DROP INDEX $name IF EXISTS"

    drop_index_if_exists(driver, "my-index")

    driver.execute_query.assert_called_once_with(
        drop_query,
        {"name": "my-index"},
        database_=None,
    )


def test_create_fulltext_index_happy_path(driver: MagicMock) -> None:
    label = "node-label"
    text_node_properties = ["property-1", "property-2"]
    create_query = (
        "CREATE FULLTEXT INDEX $name IF NOT EXISTS "
        f"FOR (n:`{label}`) ON EACH "
        f"[{', '.join(['n.`' + property + '`' for property in text_node_properties])}]"
    )

    create_fulltext_index(driver, "my-index", label, text_node_properties)

    driver.execute_query.assert_called_once_with(
        create_query,
        {"name": "my-index"},
        database_=None,
    )


def test_create_fulltext_index_fail_if_exists(driver: MagicMock) -> None:
    label = "node-label"
    text_node_properties = ["property-1", "property-2"]
    create_query = (
        "CREATE FULLTEXT INDEX $name  "
        f"FOR (n:`{label}`) ON EACH "
        f"[{', '.join(['n.`' + property + '`' for property in text_node_properties])}]"
    )

    create_fulltext_index(
        driver, "my-index", label, text_node_properties, fail_if_exists=True
    )

    driver.execute_query.assert_called_once_with(
        create_query,
        {"name": "my-index"},
        database_=None,
    )


def test_create_fulltext_index_raises_error_with_neo4j_client_error(
    driver: MagicMock,
) -> None:
    label = "node-label"
    text_node_properties = ["property-1", "property-2"]
    driver.execute_query.side_effect = neo4j.exceptions.ClientError

    with pytest.raises(Neo4jIndexError):
        create_fulltext_index(driver, "my-index", label, text_node_properties)


def test_create_fulltext_index_empty_node_properties(driver: MagicMock) -> None:
    label = "node-label"
    node_properties: list[str] = []

    with pytest.raises(Neo4jIndexError) as excinfo:
        create_fulltext_index(driver, "my-index", label, node_properties)

    assert "Error for inputs to create_fulltext_index" in str(excinfo)


def test_create_fulltext_index_ensure_escaping(driver: MagicMock) -> None:
    label = "node-label"
    text_node_properties = ["property-1", "property-2"]
    create_query = (
        "CREATE FULLTEXT INDEX $name IF NOT EXISTS "
        f"FOR (n:`{label}`) ON EACH "
        f"[{', '.join(['n.`' + property + '`' for property in text_node_properties])}]"
    )

    create_fulltext_index(driver, "my-complicated-`-index", label, text_node_properties)

    driver.execute_query.assert_called_once_with(
        create_query,
        {"name": "my-complicated-`-index"},
        database_=None,
    )


def test_upsert_vector_happy_path(driver: MagicMock) -> None:
    id = 1
    embedding_property = "embedding"
    vector = [1.0, 2.0, 3.0]

    upsert_vector(driver, id, embedding_property, vector)

    upsert_query = (
        "MATCH (n) "
        "WHERE elementId(n) = $node_element_id "
        "WITH n "
        "CALL db.create.setNodeVectorProperty(n, $embedding_property, $vector) "
        "RETURN n"
    )

    driver.execute_query.assert_called_once_with(
        upsert_query,
        {
            "node_element_id": id,
            "embedding_property": embedding_property,
            "vector": vector,
        },
        database_=None,
    )


def test_upsert_vector_on_relationship_happy_path(driver: MagicMock) -> None:
    id = 1
    embedding_property = "embedding"
    vector = [1.0, 2.0, 3.0]

    upsert_vector_on_relationship(driver, id, embedding_property, vector)

    upsert_query = (
        "MATCH ()-[r]->() "
        "WHERE elementId(r) = $rel_element_id "
        "WITH r "
        "CALL db.create.setRelationshipVectorProperty(r, $embedding_property, $vector) "
        "RETURN r"
    )

    driver.execute_query.assert_called_once_with(
        upsert_query,
        {
            "rel_element_id": id,
            "embedding_property": embedding_property,
            "vector": vector,
        },
        database_=None,
    )


def test_upsert_vector_on_relationship_raises_neo4j_insertion_error(
    driver: MagicMock,
) -> None:
    id = 1
    embedding_property = "embedding"
    vector = [1.0, 2.0, 3.0]
    driver.execute_query.side_effect = neo4j.exceptions.ClientError

    with pytest.raises(Neo4jInsertionError) as excinfo:
        upsert_vector_on_relationship(driver, id, embedding_property, vector)
    assert "Upserting vector to Neo4j failed" in str(excinfo)


def test_upsert_vector_raises_neo4j_insertion_error(
    driver: MagicMock,
) -> None:
    id = 1
    embedding_property = "embedding"
    vector = [1.0, 2.0, 3.0]
    driver.execute_query.side_effect = neo4j.exceptions.ClientError

    with pytest.raises(Neo4jInsertionError) as excinfo:
        upsert_vector(driver, id, embedding_property, vector)

    assert "Upserting vector to Neo4j failed" in str(excinfo)


def test_upsert_vectors_wrong_entity_type(driver: MagicMock) -> None:
    with pytest.raises(ValueError) as exc_info:
        upsert_vectors(
            driver=driver,
            ids=["1"],
            embedding_property="embedding",
            embeddings=[[1.0, 2.0, 3.0]],
            neo4j_database="neo4j",
            entity_type="WRONG_ENTITY_TYPE",  # type: ignore[arg-type]
        )
    assert str(exc_info.value) == "entity_type must be either 'NODE' or 'RELATIONSHIP'"


def test_upsert_vectors_mismatched_ids_and_embeddings(driver: MagicMock) -> None:
    with pytest.raises(ValueError) as exc_info:
        upsert_vectors(
            driver=driver,
            ids=["1"],
            embedding_property="embedding",
            embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            neo4j_database="neo4j",
        )
    assert str(exc_info.value) == "ids and embeddings must be the same length"


def test_upsert_vectors_inconsistent_embedding_sizes(driver: MagicMock) -> None:
    with pytest.raises(ValueError) as exc_info:
        upsert_vectors(
            driver=driver,
            ids=["1", "2"],
            embedding_property="embedding",
            embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0]],
            neo4j_database="neo4j",
        )
    assert str(exc_info.value) == "All embeddings must be of the same size"
