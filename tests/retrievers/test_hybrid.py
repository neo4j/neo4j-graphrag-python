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

from unittest.mock import patch, MagicMock

import pytest

from neo4j_genai import HybridRetriever, HybridCypherRetriever
from neo4j_genai.neo4j_queries import get_search_query
from neo4j_genai.types import SearchType


def test_vector_retriever_initialization(driver):
    with patch("neo4j_genai.retrievers.base.Retriever._verify_version") as mock_verify:
        HybridRetriever(
            driver=driver,
            vector_index_name="my-index",
            fulltext_index_name="fulltext-index",
        )
        mock_verify.assert_called_once()


def test_vector_cypher_retriever_initialization(driver):
    with patch("neo4j_genai.retrievers.base.Retriever._verify_version") as mock_verify:
        HybridCypherRetriever(
            driver=driver,
            vector_index_name="my-index",
            fulltext_index_name="fulltext-index",
            retrieval_query="",
        )
        mock_verify.assert_called_once()


@patch("neo4j_genai.HybridRetriever._verify_version")
def test_hybrid_search_text_happy_path(_verify_version_mock, driver):
    embed_query_vector = [1.0 for _ in range(1536)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector
    vector_index_name = "my-index"
    fulltext_index_name = "my-fulltext-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5
    retriever = HybridRetriever(
        driver, vector_index_name, fulltext_index_name, custom_embeddings
    )
    retriever.driver.execute_query.return_value = [
        [{"node": "dummy-node", "score": 1.0}],
        None,
        None,
    ]
    search_query = get_search_query(SearchType.HYBRID)

    records = retriever.search(query_text=query_text, top_k=top_k)

    retriever.driver.execute_query.assert_called_once_with(
        search_query,
        {
            "vector_index_name": vector_index_name,
            "top_k": top_k,
            "query_text": query_text,
            "fulltext_index_name": fulltext_index_name,
            "query_vector": embed_query_vector,
        },
    )
    custom_embeddings.embed_query.assert_called_once_with(query_text)
    assert records == [{"node": "dummy-node", "score": 1.0}]


@patch("neo4j_genai.HybridRetriever._verify_version")
def test_hybrid_search_favors_query_vector_over_embedding_vector(
    _verify_version_mock, driver
):
    embed_query_vector = [1.0 for _ in range(1536)]
    query_vector = [2.0 for _ in range(1536)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector
    vector_index_name = "my-index"
    fulltext_index_name = "my-fulltext-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5
    retriever = HybridRetriever(
        driver, vector_index_name, fulltext_index_name, custom_embeddings
    )
    retriever.driver.execute_query.return_value = [
        [{"node": "dummy-node", "score": 1.0}],
        None,
        None,
    ]
    search_query = get_search_query(SearchType.HYBRID)

    retriever.search(query_text=query_text, query_vector=query_vector, top_k=top_k)

    retriever.driver.execute_query.assert_called_once_with(
        search_query,
        {
            "vector_index_name": vector_index_name,
            "top_k": top_k,
            "query_text": query_text,
            "fulltext_index_name": fulltext_index_name,
            "query_vector": query_vector,
        },
    )
    custom_embeddings.embed_query.assert_not_called()


def test_error_when_hybrid_search_only_text_no_embedder(hybrid_retriever):
    query_text = "may thy knife chip and shatter"
    top_k = 5

    with pytest.raises(ValueError, match="Embedding method required for text query."):
        hybrid_retriever.search(
            query_text=query_text,
            top_k=top_k,
        )


def test_hybrid_search_retriever_search_missing_embedder_for_text(
    hybrid_retriever,
):
    query_text = "may thy knife chip and shatter"
    top_k = 5

    with pytest.raises(ValueError, match="Embedding method required for text query"):
        hybrid_retriever.search(
            query_text=query_text,
            top_k=top_k,
        )


@patch("neo4j_genai.HybridRetriever._verify_version")
def test_hybrid_retriever_return_properties(_verify_version_mock, driver):
    embed_query_vector = [1.0 for _ in range(1536)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector
    vector_index_name = "my-index"
    fulltext_index_name = "my-fulltext-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5
    return_properties = ["node-property-1", "node-property-2"]
    retriever = HybridRetriever(
        driver,
        vector_index_name,
        fulltext_index_name,
        custom_embeddings,
        return_properties,
    )
    driver.execute_query.return_value = [
        [{"node": "dummy-node", "score": 1.0}],
        None,
        None,
    ]
    search_query = get_search_query(SearchType.HYBRID, return_properties)

    records = retriever.search(query_text=query_text, top_k=top_k)

    custom_embeddings.embed_query.assert_called_once_with(query_text)
    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "vector_index_name": vector_index_name,
            "top_k": top_k,
            "query_text": query_text,
            "fulltext_index_name": fulltext_index_name,
            "query_vector": embed_query_vector,
        },
    )
    assert records == [{"node": "dummy-node", "score": 1.0}]


@patch("neo4j_genai.HybridCypherRetriever._verify_version")
def test_hybrid_cypher_retrieval_query_with_params(_verify_version_mock, driver):
    embed_query_vector = [1.0 for _ in range(1536)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector
    vector_index_name = "my-index"
    fulltext_index_name = "my-fulltext-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5
    retrieval_query = """
        RETURN node.id AS node_id, node.text AS text, score, {test: $param} AS metadata
        """
    query_params = {
        "param": "dummy-param",
    }
    retriever = HybridCypherRetriever(
        driver,
        vector_index_name,
        fulltext_index_name,
        retrieval_query,
        custom_embeddings,
    )
    driver.execute_query.return_value = [
        [{"node_id": 123, "text": "dummy-text", "score": 1.0}],
        None,
        None,
    ]
    search_query = get_search_query(SearchType.HYBRID, retrieval_query=retrieval_query)

    records = retriever.search(
        query_text=query_text,
        top_k=top_k,
        query_params=query_params,
    )

    custom_embeddings.embed_query.assert_called_once_with(query_text)

    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "vector_index_name": vector_index_name,
            "top_k": top_k,
            "query_text": query_text,
            "fulltext_index_name": fulltext_index_name,
            "query_vector": embed_query_vector,
            "param": "dummy-param",
        },
    )

    assert records == [{"node_id": 123, "text": "dummy-text", "score": 1.0}]
