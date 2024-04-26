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
from unittest.mock import patch, MagicMock

from neo4j.exceptions import CypherSyntaxError

from neo4j_genai import VectorRetriever
from neo4j_genai.retrievers import VectorCypherRetriever, HybridRetriever
from neo4j_genai.types import VectorSearchRecord


def test_vector_retriever_supported_aura_version(driver):
    driver.execute_query.return_value = [[{"versions": ["5.18-aura"]}], None, None]

    VectorRetriever(driver=driver, index_name="my-index")


def test_vector_retriever_no_supported_aura_version(driver):
    driver.execute_query.return_value = [[{"versions": ["5.3-aura"]}], None, None]

    with pytest.raises(ValueError) as excinfo:
        VectorRetriever(driver=driver, index_name="my-index")

    assert "This package only supports Neo4j version 5.18.1 or greater" in str(excinfo)


def test_vector_retriever_supported_version(driver):
    driver.execute_query.return_value = [[{"versions": ["5.19.0"]}], None, None]

    VectorRetriever(driver=driver, index_name="my-index")


def test_vector_retriever_no_supported_version(driver):
    driver.execute_query.return_value = [[{"versions": ["4.3.5"]}], None, None]

    with pytest.raises(ValueError) as excinfo:
        VectorRetriever(driver=driver, index_name="my-index")

    assert "This package only supports Neo4j version 5.18.1 or greater" in str(excinfo)


@patch("neo4j_genai.VectorRetriever._verify_version")
def test_similarity_search_vector_happy_path(_verify_version_mock, driver):
    index_name = "my-index"
    dimensions = 1536
    query_vector = [1.0 for _ in range(dimensions)]
    top_k = 5

    retriever = VectorRetriever(driver, index_name)

    retriever.driver.execute_query.return_value = [
        [{"node": "dummy-node", "score": 1.0}],
        None,
        None,
    ]
    search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """

    records = retriever.search(query_vector=query_vector, top_k=top_k)

    retriever.driver.execute_query.assert_called_once_with(
        search_query,
        {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": query_vector,
        },
    )

    assert records == [VectorSearchRecord(node="dummy-node", score=1.0)]


@patch("neo4j_genai.VectorRetriever._verify_version")
def test_similarity_search_text_happy_path(_verify_version_mock, driver):
    embed_query_vector = [1.0 for _ in range(1536)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector

    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5

    retriever = VectorRetriever(driver, index_name, custom_embeddings)

    driver.execute_query.return_value = [
        [{"node": "dummy-node", "score": 1.0}],
        None,
        None,
    ]

    search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """

    records = retriever.search(query_text=query_text, top_k=top_k)

    custom_embeddings.embed_query.assert_called_once_with(query_text)

    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": embed_query_vector,
        },
    )

    assert records == [VectorSearchRecord(node="dummy-node", score=1.0)]


@patch("neo4j_genai.VectorRetriever._verify_version")
def test_similarity_search_text_return_properties(_verify_version_mock, driver):
    embed_query_vector = [1.0 for _ in range(3)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector

    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5
    return_properties = ["node-property-1", "node-property-2"]

    retriever = VectorRetriever(
        driver, index_name, custom_embeddings, return_properties=return_properties
    )

    driver.execute_query.return_value = [
        [{"node": "dummy-node", "score": 1.0}],
        None,
        None,
    ]

    search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        RETURN node {.node-property-1, .node-property-2} as node, score
        """

    records = retriever.search(query_text=query_text, top_k=top_k)

    custom_embeddings.embed_query.assert_called_once_with(query_text)

    driver.execute_query.assert_called_once_with(
        search_query.rstrip(),
        {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": embed_query_vector,
        },
    )

    assert records == [VectorSearchRecord(node="dummy-node", score=1.0)]


def test_vector_retriever_search_missing_embedder_for_text(vector_retriever):
    query_text = "may thy knife chip and shatter"
    top_k = 5

    with pytest.raises(ValueError, match="Embedding method required for text query"):
        vector_retriever.search(query_text=query_text, top_k=top_k)


def test_vector_retriever_search_both_text_and_vector(vector_retriever):
    query_text = "may thy knife chip and shatter"
    query_vector = [1.1, 2.2, 3.3]
    top_k = 5

    with pytest.raises(
        ValueError, match="You must provide exactly one of query_vector or query_text."
    ):
        vector_retriever.search(
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k,
        )


def test_vector_cypher_retriever_search_missing_embedder_for_text(
    vector_cypher_retriever,
):
    query_text = "may thy knife chip and shatter"
    top_k = 5

    with pytest.raises(ValueError, match="Embedding method required for text query"):
        vector_cypher_retriever.search(query_text=query_text, top_k=top_k)


def test_vector_cypher_retriever_search_both_text_and_vector(vector_cypher_retriever):
    query_text = "may thy knife chip and shatter"
    query_vector = [1.1, 2.2, 3.3]
    top_k = 5

    with pytest.raises(
        ValueError, match="You must provide exactly one of query_vector or query_text."
    ):
        vector_cypher_retriever.search(
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k,
        )


@patch("neo4j_genai.VectorRetriever._verify_version")
def test_similarity_search_vector_bad_results(_verify_version_mock, driver):
    index_name = "my-index"
    dimensions = 1536
    query_vector = [1.0 for _ in range(dimensions)]
    top_k = 5

    retriever = VectorRetriever(driver, index_name)

    retriever.driver.execute_query.return_value = [
        [{"node": "dummy-node", "score": "adsa"}],
        None,
        None,
    ]
    search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """

    with pytest.raises(ValueError):
        retriever.search(query_vector=query_vector, top_k=top_k)

    retriever.driver.execute_query.assert_called_once_with(
        search_query,
        {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": query_vector,
        },
    )


@patch("neo4j_genai.VectorCypherRetriever._verify_version")
def test_retrieval_query_happy_path(_verify_version_mock, driver):
    embed_query_vector = [1.0 for _ in range(1536)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector
    index_name = "my-index"
    retrieval_query = """
        RETURN node.id AS node_id, node.text AS text, score
        """
    retriever = VectorCypherRetriever(
        driver, index_name, retrieval_query, embedder=custom_embeddings
    )
    query_text = "may thy knife chip and shatter"
    top_k = 5
    driver.execute_query.return_value = [
        [{"node_id": 123, "text": "dummy-text", "score": 1.0}],
        None,
        None,
    ]
    search_query = """
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
                YIELD node, score
                """

    records = retriever.search(
        query_text=query_text,
        top_k=top_k,
    )

    custom_embeddings.embed_query.assert_called_once_with(query_text)
    driver.execute_query.assert_called_once_with(
        search_query + retrieval_query,
        {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": embed_query_vector,
        },
    )
    assert records == [{"node_id": 123, "text": "dummy-text", "score": 1.0}]


@patch("neo4j_genai.VectorCypherRetriever._verify_version")
def test_retrieval_query_with_params(_verify_version_mock, driver):
    embed_query_vector = [1.0 for _ in range(1536)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector
    index_name = "my-index"
    retrieval_query = """
        RETURN node.id AS node_id, node.text AS text, score, {test: $param} AS metadata
        """
    query_params = {
        "param": "dummy-param",
    }
    retriever = VectorCypherRetriever(
        driver,
        index_name,
        retrieval_query,
        embedder=custom_embeddings,
    )
    query_text = "may thy knife chip and shatter"
    top_k = 5
    driver.execute_query.return_value = [
        [{"node_id": 123, "text": "dummy-text", "score": 1.0}],
        None,
        None,
    ]
    search_query = """
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
                YIELD node, score
                """

    records = retriever.search(
        query_text=query_text,
        top_k=top_k,
        query_params=query_params,
    )

    custom_embeddings.embed_query.assert_called_once_with(query_text)

    driver.execute_query.assert_called_once_with(
        search_query + retrieval_query,
        {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": embed_query_vector,
            "param": "dummy-param",
        },
    )

    assert records == [{"node_id": 123, "text": "dummy-text", "score": 1.0}]


@patch("neo4j_genai.VectorCypherRetriever._verify_version")
def test_retrieval_query_cypher_error(_verify_version_mock, driver):
    embed_query_vector = [1.0 for _ in range(1536)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector
    index_name = "my-index"
    retrieval_query = """
        this is not a cypher query
        """
    retriever = VectorCypherRetriever(
        driver, index_name, retrieval_query, embedder=custom_embeddings
    )
    query_text = "may thy knife chip and shatter"
    top_k = 5
    driver.execute_query.side_effect = CypherSyntaxError

    with pytest.raises(CypherSyntaxError):
        retriever.search(
            query_text=query_text,
            top_k=top_k,
        )


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
    search_query = (
        "CALL { "
        "CALL db.index.vector.queryNodes($vector_index_name, $top_k, $query_vector) "
        "YIELD node, score "
        "RETURN node, score UNION "
        "CALL db.index.fulltext.queryNodes($fulltext_index_name, $query_text, {limit: $top_k}) "
        "YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / max) AS score "
        "} "
        "WITH node, max(score) AS score ORDER BY score DESC LIMIT $top_k "
        "RETURN node, score"
    )

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
    search_query = (
        "CALL { "
        "CALL db.index.vector.queryNodes($vector_index_name, $top_k, $query_vector) "
        "YIELD node, score "
        "RETURN node, score UNION "
        "CALL db.index.fulltext.queryNodes($fulltext_index_name, $query_text, {limit: $top_k}) "
        "YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / max) AS score "
        "} "
        "WITH node, max(score) AS score ORDER BY score DESC LIMIT $top_k "
        "RETURN node, score"
    )

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
    search_query = (
        "CALL { "
        "CALL db.index.vector.queryNodes($vector_index_name, $top_k, $query_vector) "
        "YIELD node, score "
        "RETURN node, score UNION "
        "CALL db.index.fulltext.queryNodes($fulltext_index_name, $query_text, {limit: $top_k}) "
        "YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / max) AS score "
        "} "
        "WITH node, max(score) AS score ORDER BY score DESC LIMIT $top_k "
        "YIELD node, score "
        "RETURN node {.node-property-1, .node-property-2} as node, score"
    )

    records = retriever.search(query_text=query_text, top_k=top_k)

    custom_embeddings.embed_query.assert_called_once_with(query_text)

    driver.execute_query.assert_called_once_with(
        search_query.rstrip(),
        {
            "vector_index_name": vector_index_name,
            "top_k": top_k,
            "query_text": query_text,
            "fulltext_index_name": fulltext_index_name,
            "query_vector": embed_query_vector,
        },
    )

    assert records == [{"node": "dummy-node", "score": 1.0}]
