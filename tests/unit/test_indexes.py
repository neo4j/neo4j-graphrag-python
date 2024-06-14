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
from unittest.mock import MagicMock

import neo4j.exceptions
import pytest
from neo4j_genai.exceptions import Neo4jIndexError, Neo4jInsertionError
from neo4j_genai.indexes import (
    create_fulltext_index,
    create_vector_index,
    drop_index_if_exists,
    upsert_vector,
)


def test_create_vector_index_happy_path(driver: MagicMock) -> None:
    create_query = (
        "CREATE VECTOR INDEX $name FOR (n:People) ON n.name OPTIONS "
        "{ indexConfig: { `vector.dimensions`: toInteger($dimensions), `vector.similarity_function`: $similarity_fn } }"
    )

    create_vector_index(driver, "my-index", "People", "name", 2048, "cosine")

    driver.execute_query.assert_called_once_with(
        create_query,
        {"name": "my-index", "dimensions": 2048, "similarity_fn": "cosine"},
    )


def test_create_vector_index_ensure_escaping(driver: MagicMock) -> None:
    create_query = (
        "CREATE VECTOR INDEX $name FOR (n:People) ON n.name OPTIONS "
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
    )


def test_create_vector_index_negative_dimension(driver: MagicMock) -> None:
    with pytest.raises(Neo4jIndexError) as excinfo:
        create_vector_index(driver, "my-index", "People", "name", -5, "cosine")
    assert "Error for inputs to create_vector_index" in str(excinfo)


def test_create_vector_index_validation_error_dimensions(driver: MagicMock) -> None:
    with pytest.raises(Neo4jIndexError) as excinfo:
        create_vector_index(driver, "my-index", "People", "name", "no-dim", "cosine")
    assert "Error for inputs to create_vector_index" in str(excinfo)


def test_create_vector_index_raises_error_with_neo4j_client_error(
    driver: MagicMock,
) -> None:
    driver.execute_query.side_effect = neo4j.exceptions.ClientError
    with pytest.raises(Neo4jIndexError):
        create_vector_index(driver, "my-index", "People", "name", 2048, "cosine")


def test_create_vector_index_validation_error_similarity_fn(driver: MagicMock) -> None:
    with pytest.raises(Neo4jIndexError) as excinfo:
        create_vector_index(driver, "my-index", "People", "name", 1536, "algebra")
    assert "Error for inputs to create_vector_index" in str(excinfo)


def test_drop_index_if_exists(driver: MagicMock) -> None:
    drop_query = "DROP INDEX $name IF EXISTS"

    drop_index_if_exists(driver, "my-index")

    driver.execute_query.assert_called_once_with(
        drop_query,
        {"name": "my-index"},
    )


def test_drop_index_if_exists_raises_error_with_neo4j_client_error(
    driver: MagicMock,
) -> None:
    drop_query = "DROP INDEX $name IF EXISTS"

    drop_index_if_exists(driver, "my-index")

    driver.execute_query.assert_called_once_with(
        drop_query,
        {"name": "my-index"},
    )


def test_create_fulltext_index_happy_path(driver: MagicMock) -> None:
    label = "node-label"
    text_node_properties = ["property-1", "property-2"]
    create_query = (
        "CREATE FULLTEXT INDEX $name "
        f"FOR (n:`{label}`) ON EACH "
        f"[{', '.join(['n.`' + property + '`' for property in text_node_properties])}]"
    )

    create_fulltext_index(driver, "my-index", label, text_node_properties)

    driver.execute_query.assert_called_once_with(
        create_query,
        {"name": "my-index"},
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
        "CREATE FULLTEXT INDEX $name "
        f"FOR (n:`{label}`) ON EACH "
        f"[{', '.join(['n.`' + property + '`' for property in text_node_properties])}]"
    )

    create_fulltext_index(driver, "my-complicated-`-index", label, text_node_properties)

    driver.execute_query.assert_called_once_with(
        create_query,
        {"name": "my-complicated-`-index"},
    )


def test_upsert_vector_happy_path(driver: MagicMock) -> None:
    node_label = "node-label"
    id = 1
    vector_prop = "embedding"
    vector = [1.0, 2.0, 3.0]

    upsert_vector(driver, node_label, id, vector_prop, vector)

    upsert_query = f"""
        MATCH (n: {node_label})
        WHERE elementId(n) = $id
        WITH n
        CALL db.create.setNodeVectorProperty(n, $vector_prop, $vector)
        RETURN n
        """

    driver.execute_query.assert_called_once_with(
        upsert_query,
        {"id": id, "vector_prop": vector_prop, "vector": vector},
    )


def test_upsert_vector_raises_error_with_neo4j_insertion_error(
    driver: MagicMock,
) -> None:
    node_label = "node-label"
    id = 1
    vector_prop = "embedding"
    vector = [1.0, 2.0, 3.0]
    driver.execute_query.side_effect = neo4j.exceptions.ClientError

    with pytest.raises(Neo4jInsertionError) as excinfo:
        upsert_vector(driver, node_label, id, vector_prop, vector)

    assert "Upserting vector to Neo4j failed" in str(excinfo)
