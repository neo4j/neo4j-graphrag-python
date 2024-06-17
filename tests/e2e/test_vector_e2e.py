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
from neo4j import Driver
from neo4j_genai import VectorCypherRetriever, VectorRetriever
from neo4j_genai.embedder import Embedder
from neo4j_genai.types import RetrieverResult, RetrieverResultItem


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_vector_retriever_search_text(
    driver: Driver, random_embedder: Embedder
) -> None:
    retriever = VectorRetriever(driver, "vector-index-name", random_embedder)

    top_k = 5
    results = retriever.search(query_text="Find me a book about Fremen", top_k=top_k)

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == 5
    for result in results.items:
        assert f"'{retriever._embedding_node_property}': None" in result.content
        assert isinstance(result, RetrieverResultItem)


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_vector_cypher_retriever_search_text(
    driver: Driver, random_embedder: Embedder
) -> None:
    retrieval_query = (
        "MATCH (node)-[:AUTHORED_BY]->(author:Author) " "RETURN author.name"
    )
    retriever = VectorCypherRetriever(
        driver, "vector-index-name", retrieval_query, random_embedder
    )

    top_k = 5
    results = retriever.search(query_text="Find me a book about Fremen", top_k=top_k)

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == 5
    for record in results.items:
        assert isinstance(record, RetrieverResultItem)
        assert "author.name" in record.content


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_vector_retriever_search_vector(driver: Driver) -> None:
    retriever = VectorRetriever(driver, "vector-index-name")

    top_k = 5
    results = retriever.search(query_vector=[1.0 for _ in range(1536)], top_k=top_k)

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == 5
    for result in results.items:
        assert f"'{retriever._embedding_node_property}': None" in result.content
        assert isinstance(result, RetrieverResultItem)


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_vector_cypher_retriever_search_vector(driver: Driver) -> None:
    retrieval_query = (
        "MATCH (node)-[:AUTHORED_BY]->(author:Author) " "RETURN author.name"
    )
    retriever = VectorCypherRetriever(driver, "vector-index-name", retrieval_query)

    top_k = 5
    results = retriever.search(query_vector=[1.0 for _ in range(1536)], top_k=top_k)

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == 5
    for record in results.items:
        assert isinstance(record, RetrieverResultItem)
        assert "author.name" in record.content


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_vector_retriever_return_properties(driver: Driver) -> None:
    properties = ["name", "age"]
    retriever = VectorRetriever(
        driver,
        "vector-index-name",
        return_properties=properties,
    )

    top_k = 5
    results = retriever.search(
        query_vector=[1.0 for _ in range(1536)],
        top_k=top_k,
    )

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == 5
    for result in results.items:
        assert isinstance(result, RetrieverResultItem)


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_vector_retriever_filters(driver: Driver) -> None:
    retriever = VectorRetriever(
        driver,
        "vector-index-name",
    )

    top_k = 2
    results = retriever.search(
        query_vector=[1.0 for _ in range(1536)],
        filters={"int_property": {"$gt": 2}},
        top_k=top_k,
    )

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == 2
    for result in results.items:
        assert isinstance(result, RetrieverResultItem)
        # assert result.node["int_property"] > 2
