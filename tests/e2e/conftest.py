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

import random
import string
import uuid
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest
from neo4j import Driver, GraphDatabase
from neo4j_genai.embedder import Embedder
from neo4j_genai.indexes import (
    create_fulltext_index,
    create_vector_index,
    drop_index_if_exists,
)
from neo4j_genai.llm import LLMInterface
from neo4j_genai.retrievers import VectorRetriever

from ..e2e.utils import EMBEDDING_BIOLOGY


@pytest.fixture(scope="module")
def driver() -> Generator[Any, Any, Any]:
    uri = "neo4j://localhost:7687"
    auth = ("neo4j", "password")
    driver = GraphDatabase.driver(uri, auth=auth)
    yield driver
    driver.close()


@pytest.fixture(scope="function")
def llm() -> MagicMock:
    return MagicMock(spec=LLMInterface)


class RandomEmbedder(Embedder):
    def embed_query(self, text: str) -> list[float]:
        return [random.random() for _ in range(1536)]


class BiologyEmbedder(Embedder):
    def embed_query(self, text: str) -> list[float]:
        if text == "biology":
            return EMBEDDING_BIOLOGY
        raise ValueError(f"Unknown embedding text: {text}")


@pytest.fixture(scope="module")
def random_embedder() -> RandomEmbedder:
    return RandomEmbedder()


@pytest.fixture(scope="module")
def biology_embedder() -> BiologyEmbedder:
    return BiologyEmbedder()


@pytest.fixture(scope="function")
def retriever_mock() -> MagicMock:
    return MagicMock(spec=VectorRetriever)


@pytest.fixture(scope="module")
def setup_neo4j_for_retrieval(driver: Driver) -> None:
    vector_index_name = "vector-index-name"
    fulltext_index_name = "fulltext-index-name"

    # Delete data and drop indexes to prevent data leakage
    driver.execute_query("MATCH (n) DETACH DELETE n")
    drop_index_if_exists(driver, vector_index_name)
    drop_index_if_exists(driver, fulltext_index_name)

    # Create a vector index
    create_vector_index(
        driver,
        vector_index_name,
        label="Document",
        embedding_property="vectorProperty",
        dimensions=1536,
        similarity_fn="euclidean",
    )

    # Create a fulltext index
    create_fulltext_index(
        driver,
        fulltext_index_name,
        label="Document",
        node_properties=["vectorProperty"],
    )

    # Insert 10 vectors and authors
    vector = [random.random() for _ in range(1536)]

    def random_str(n: int) -> str:
        return "".join([random.choice(string.ascii_letters) for _ in range(n)])

    for i in range(10):
        insert_query = (
            "MERGE (doc:Document {id: $id})"
            "ON CREATE SET  doc.int_property = $i, "
            "               doc.short_text_property = toString($i)"
            "WITH doc "
            "CALL db.create.setNodeVectorProperty(doc, 'vectorProperty', $vector)"
            "WITH doc "
            "MERGE (author:Author {name: $authorName})"
            "MERGE (doc)-[:AUTHORED_BY]->(author)"
            "RETURN doc, author"
        )

        parameters = {
            "id": str(uuid.uuid4()),
            "i": i,
            "vector": vector,
            "authorName": random_str(1536),
        }
        driver.execute_query(insert_query, parameters)


@pytest.fixture(scope="module")
def setup_neo4j_for_schema_query(driver: Driver) -> None:
    # Delete all nodes in the graph
    driver.execute_query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    driver.execute_query(
        """
        CREATE (la:LabelA {property_a: 'a'})
        CREATE (lb:LabelB)
        CREATE (lc:LabelC)
        MERGE (la)-[:REL_TYPE]-> (lb)
        MERGE (la)-[:REL_TYPE {rel_prop: 'abc'}]-> (lc)
        """
    )


@pytest.fixture(scope="module")
def setup_neo4j_for_schema_query_with_excluded_labels(driver: Driver) -> None:
    # Delete all nodes in the graph
    driver.execute_query("MATCH (n) DETACH DELETE n")
    # Create two labels and a relationship to be excluded
    driver.execute_query(
        "CREATE (:_Bloom_Scene_{property_a: 'a'})-[:_Bloom_HAS_SCENE_{property_b: 'b'}]->(:_Bloom_Perspective_)"
    )


@pytest.fixture(scope="function")
def setup_neo4j_for_kg_construction(driver: Driver) -> None:
    # Delete all nodes and indexes in the graph
    driver.execute_query("MATCH (n) DETACH DELETE n")
    vector_index_name = "vector-index-name"
    fulltext_index_name = "fulltext-index-name"
    drop_index_if_exists(driver, vector_index_name)
    drop_index_if_exists(driver, fulltext_index_name)

    # Create a vector index with the dimensions used by the Hugging Face all-MiniLM-L6-v2 model
    create_vector_index(
        driver,
        vector_index_name,
        label="Document",
        embedding_property="vectorProperty",
        dimensions=3,
        similarity_fn="euclidean",
    )
