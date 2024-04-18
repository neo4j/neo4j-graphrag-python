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

import pytest

from neo4j_genai.indexes import (
    create_vector_index,
    drop_index,
    create_fulltext_index,
)


def test_create_vector_index_happy_path(driver):
    create_query = (
        "CREATE VECTOR INDEX $name FOR (n:People) ON n.name OPTIONS "
        "{ indexConfig: { `vector.dimensions`: toInteger($dimensions), `vector.similarity_function`: $similarity_fn } }"
    )

    create_vector_index(driver, "my-index", "People", "name", 2048, "cosine")

    driver.execute_query.assert_called_once_with(
        create_query,
        {"name": "my-index", "dimensions": 2048, "similarity_fn": "cosine"},
    )


def test_create_vector_index_ensure_escaping(driver):
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


def test_create_vector_index_negative_dimension(driver):
    with pytest.raises(ValueError) as excinfo:
        create_vector_index(driver, "my-index", "People", "name", -5, "cosine")
    assert "Error for inputs to create_vector_index" in str(excinfo)


def test_create_vector_index_validation_error_dimensions(driver):
    with pytest.raises(ValueError) as excinfo:
        create_vector_index(driver, "my-index", "People", "name", "no-dim", "cosine")
    assert "Error for inputs to create_vector_index" in str(excinfo)


def test_create_vector_index_validation_error_similarity_fn(driver):
    with pytest.raises(ValueError) as excinfo:
        create_vector_index(driver, "my-index", "People", "name", 1536, "algebra")
    assert "Error for inputs to create_vector_index" in str(excinfo)


def test_drop_index(driver):
    drop_query = "DROP INDEX $name"

    drop_index(driver, "my-index")

    driver.execute_query.assert_called_once_with(
        drop_query,
        {"name": "my-index"},
    )


def test_create_fulltext_index_happy_path(driver):
    label = "node-label"
    text_node_properties = ["property-1", "property-2"]
    create_query = (
        "CREATE FULLTEXT INDEX $name"
        f"FOR (n:`{label}`) ON EACH "
        f"[{', '.join(['n.`' + property + '`' for property in text_node_properties])}]"
    )

    create_fulltext_index(driver, "my-index", label, text_node_properties)

    driver.execute_query.assert_called_once_with(
        create_query,
        {"name": "my-index"},
    )


def test_create_fulltext_index_empty_node_properties(driver):
    label = "node-label"
    node_properties = []

    with pytest.raises(ValueError) as excinfo:
        create_fulltext_index(driver, "my-index", label, node_properties)

    assert "Error for inputs to create_fulltext_index" in str(excinfo)


def test_create_fulltext_index_ensure_escaping(driver):
    label = "node-label"
    text_node_properties = ["property-1", "property-2"]
    create_query = (
        "CREATE FULLTEXT INDEX $name"
        f"FOR (n:`{label}`) ON EACH "
        f"[{', '.join(['n.`' + property + '`' for property in text_node_properties])}]"
    )

    create_fulltext_index(driver, "my-complicated-`-index", label, text_node_properties)

    driver.execute_query.assert_called_once_with(
        create_query,
        {"name": "my-complicated-`-index"},
    )
