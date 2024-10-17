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
import logging

import pytest
from neo4j import Driver
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.retrievers import (
    HybridCypherRetriever,
    HybridRetriever,
)
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_hybrid_retriever_search_text(
    driver: Driver, random_embedder: Embedder
) -> None:
    retriever = HybridRetriever(
        driver, "vector-index-name", "fulltext-index-name", random_embedder
    )

    top_k = 5
    results = retriever.search(query_text="Find me a book about Fremen", top_k=top_k)

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == 5
    for result in results.items:
        assert isinstance(result, RetrieverResultItem)


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_hybrid_retriever_no_neo4j_deprecation_warning(
    driver: Driver, random_embedder: Embedder, caplog: pytest.LogCaptureFixture
) -> None:
    retriever = HybridRetriever(
        driver, "vector-index-name", "fulltext-index-name", random_embedder
    )

    top_k = 5
    with caplog.at_level(logging.WARNING):
        retriever.search(query_text="Find me a book about Fremen", top_k=top_k)

    for record in caplog.records:
        if (
            "Neo.ClientNotification.Statement.FeatureDeprecationWarning"
            in record.message
        ):
            assert False, f"Deprecation warning found in logs: {record.message}"


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_hybrid_cypher_retriever_search_text(
    driver: Driver, random_embedder: Embedder
) -> None:
    retrieval_query = (
        "MATCH (node)-[:AUTHORED_BY]->(author:Author) " "RETURN author.name"
    )
    retriever = HybridCypherRetriever(
        driver,
        "vector-index-name",
        "fulltext-index-name",
        retrieval_query,
        random_embedder,
    )

    top_k = 5
    results = retriever.search(query_text="Find me a book about Fremen", top_k=top_k)

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == 5
    for record in results.items:
        assert isinstance(record, RetrieverResultItem)
        assert "author.name" in record.content


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_hybrid_retriever_search_vector(driver: Driver) -> None:
    retriever = HybridRetriever(
        driver,
        "vector-index-name",
        "fulltext-index-name",
    )

    top_k = 5
    results = retriever.search(
        query_text="Find me a book about Fremen",
        query_vector=[1.0 for _ in range(1536)],
        top_k=top_k,
    )

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == 5
    for result in results.items:
        assert isinstance(result, RetrieverResultItem)


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_hybrid_cypher_retriever_search_vector(driver: Driver) -> None:
    retrieval_query = (
        "MATCH (node)-[:AUTHORED_BY]->(author:Author) " "RETURN author.name"
    )
    retriever = HybridCypherRetriever(
        driver,
        "vector-index-name",
        "fulltext-index-name",
        retrieval_query,
    )

    top_k = 5
    results = retriever.search(
        query_text="Find me a book about Fremen",
        query_vector=[1.0 for _ in range(1536)],
        top_k=top_k,
    )

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == 5
    for record in results.items:
        assert isinstance(record, RetrieverResultItem)
        assert "author.name" in record.content


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_hybrid_retriever_return_properties(driver: Driver) -> None:
    properties = ["name", "age"]
    retriever = HybridRetriever(
        driver,
        "vector-index-name",
        "fulltext-index-name",
        return_properties=properties,
    )

    top_k = 5
    results = retriever.search(
        query_text="Find me a book about Fremen",
        query_vector=[1.0 for _ in range(1536)],
        top_k=top_k,
    )

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == 5
    for result in results.items:
        assert isinstance(result, RetrieverResultItem)
