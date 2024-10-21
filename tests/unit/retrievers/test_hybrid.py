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

from unittest.mock import MagicMock, patch

import pytest
from neo4j_graphrag.exceptions import (
    EmbeddingRequiredError,
    RetrieverInitializationError,
)
from neo4j_graphrag.neo4j_queries import get_search_query
from neo4j_graphrag.retrievers import HybridCypherRetriever, HybridRetriever
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem, SearchType


def test_vector_retriever_initialization(driver: MagicMock) -> None:
    with patch(
        "neo4j_graphrag.retrievers.base.Retriever._verify_version"
    ) as mock_verify:
        HybridRetriever(
            driver=driver,
            vector_index_name="vector-index",
            fulltext_index_name="fulltext-index",
        )
        mock_verify.assert_called_once()


def test_vector_cypher_retriever_initialization(driver: MagicMock) -> None:
    with patch(
        "neo4j_graphrag.retrievers.base.Retriever._verify_version"
    ) as mock_verify:
        HybridCypherRetriever(
            driver=driver,
            vector_index_name="vector-index",
            fulltext_index_name="fulltext-index",
            retrieval_query="",
        )
        mock_verify.assert_called_once()


@patch("neo4j_graphrag.retrievers.HybridRetriever._verify_version")
def test_hybrid_retriever_invalid_fulltext_index_name(
    _verify_version_mock: MagicMock, driver: MagicMock
) -> None:
    with pytest.raises(RetrieverInitializationError) as exc_info:
        HybridRetriever(
            driver=driver,
            vector_index_name="vector-index",
            fulltext_index_name=42,  # type: ignore
        )

    assert "fulltext_index_name" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)


@patch("neo4j_graphrag.retrievers.HybridRetriever._verify_version")
def test_hybrid_retriever_with_result_format_function(
    _verify_version_mock: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
    neo4j_record: MagicMock,
    result_formatter: MagicMock,
) -> None:
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
    vector_index_name = "vector-index"
    fulltext_index_name = "fulltext-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5

    retriever = HybridRetriever(
        driver,
        vector_index_name,
        fulltext_index_name,
        embedder,
        result_formatter=result_formatter,
    )
    retriever.neo4j_version_is_5_23_or_above = True
    retriever.driver.execute_query.return_value = [  # type: ignore
        [neo4j_record],
        None,
        None,
    ]

    records = retriever.search(query_text=query_text, top_k=top_k)

    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="dummy-node", metadata={"score": 1.0, "node_id": 123}
            ),
        ],
        metadata={"__retriever": "HybridRetriever"},
    )


@patch("neo4j_graphrag.retrievers.HybridRetriever._verify_version")
def test_hybrid_retriever_invalid_database_name(
    _verify_version_mock: MagicMock, driver: MagicMock
) -> None:
    with pytest.raises(RetrieverInitializationError) as exc_info:
        HybridRetriever(
            driver=driver,
            vector_index_name="vector-index",
            fulltext_index_name="fulltext-index",
            neo4j_database=42,  # type: ignore
        )

    assert "database" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)


@patch("neo4j_graphrag.retrievers.HybridCypherRetriever._verify_version")
def test_hybrid_cypher_retriever_invalid_retrieval_query(
    _verify_version_mock: MagicMock, driver: MagicMock
) -> None:
    with pytest.raises(RetrieverInitializationError) as exc_info:
        HybridCypherRetriever(
            driver=driver,
            vector_index_name="vector-index",
            fulltext_index_name="fulltext-index",
            retrieval_query=42,  # type: ignore
        )

    assert "retrieval_query" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)


@patch("neo4j_graphrag.retrievers.HybridCypherRetriever._verify_version")
def test_hybrid_cypher_retriever_invalid_database_name(
    _verify_version_mock: MagicMock, driver: MagicMock
) -> None:
    retrieval_query = """
        RETURN node.id AS node_id, node.text AS text, score, {test: $param} AS metadata
        """
    with pytest.raises(RetrieverInitializationError) as exc_info:
        HybridCypherRetriever(
            driver=driver,
            vector_index_name="vector-index",
            fulltext_index_name="fulltext-index",
            retrieval_query=retrieval_query,
            neo4j_database=42,  # type: ignore
        )

    assert "database" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)


@patch("neo4j_graphrag.retrievers.HybridRetriever._fetch_index_infos")
@patch("neo4j_graphrag.retrievers.HybridRetriever._verify_version")
def test_hybrid_search_text_happy_path(
    _verify_version_mock: MagicMock,
    _fetch_index_infos_mock: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
    vector_index_name = "vector-index"
    fulltext_index_name = "fulltext-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5

    retriever = HybridRetriever(
        driver, vector_index_name, fulltext_index_name, embedder
    )
    retriever.neo4j_version_is_5_23_or_above = True
    retriever._embedding_node_property = (
        "embedding"  # variable normally filled by fetch_index_infos
    )
    retriever.driver.execute_query.return_value = [  # type: ignore
        [neo4j_record],
        None,
        None,
    ]
    search_query, _ = get_search_query(
        SearchType.HYBRID,
        embedding_node_property="embedding",
        neo4j_version_is_5_23_or_above=retriever.neo4j_version_is_5_23_or_above,
    )

    records = retriever.search(query_text=query_text, top_k=top_k)

    retriever.driver.execute_query.assert_called_once_with(  # type: ignore
        search_query,
        {
            "vector_index_name": vector_index_name,
            "top_k": top_k,
            "query_text": query_text,
            "fulltext_index_name": fulltext_index_name,
            "query_vector": embed_query_vector,
        },
        database_=None,
    )
    embedder.embed_query.assert_called_once_with(query_text)
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(content="dummy-node", metadata={"score": 1.0}),
        ],
        metadata={"__retriever": "HybridRetriever"},
    )


@patch("neo4j_graphrag.retrievers.HybridRetriever._fetch_index_infos")
@patch("neo4j_graphrag.retrievers.HybridRetriever._verify_version")
def test_hybrid_search_favors_query_vector_over_embedding_vector(
    _verify_version_mock: MagicMock,
    _fetch_index_infos_mock: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    embed_query_vector = [1.0 for _ in range(1536)]
    query_vector = [2.0 for _ in range(1536)]

    embedder.embed_query.return_value = embed_query_vector
    vector_index_name = "vector-index"
    fulltext_index_name = "fulltext-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5
    database = "neo4j"
    retriever = HybridRetriever(
        driver,
        vector_index_name,
        fulltext_index_name,
        embedder,
        neo4j_database=database,
    )
    retriever.neo4j_version_is_5_23_or_above = True
    retriever.driver.execute_query.return_value = [  # type: ignore
        [neo4j_record],
        None,
        None,
    ]
    search_query, _ = get_search_query(
        SearchType.HYBRID,
        neo4j_version_is_5_23_or_above=retriever.neo4j_version_is_5_23_or_above,
    )

    retriever.search(query_text=query_text, query_vector=query_vector, top_k=top_k)

    retriever.driver.execute_query.assert_called_once_with(  # type: ignore
        search_query,
        {
            "vector_index_name": vector_index_name,
            "top_k": top_k,
            "query_text": query_text,
            "fulltext_index_name": fulltext_index_name,
            "query_vector": query_vector,
        },
        database_=database,
    )
    embedder.embed_query.assert_not_called()


def test_error_when_hybrid_search_only_text_no_embedder(
    hybrid_retriever: HybridRetriever,
) -> None:
    query_text = "may thy knife chip and shatter"
    top_k = 5

    with pytest.raises(
        EmbeddingRequiredError, match="Embedding method required for text query."
    ):
        hybrid_retriever.search(
            query_text=query_text,
            top_k=top_k,
        )


def test_hybrid_search_retriever_search_missing_embedder_for_text(
    hybrid_retriever: HybridRetriever,
) -> None:
    query_text = "may thy knife chip and shatter"
    top_k = 5

    with pytest.raises(
        EmbeddingRequiredError, match="Embedding method required for text query"
    ):
        hybrid_retriever.search(
            query_text=query_text,
            top_k=top_k,
        )


@patch("neo4j_graphrag.retrievers.HybridRetriever._fetch_index_infos")
@patch("neo4j_graphrag.retrievers.HybridRetriever._verify_version")
def test_hybrid_retriever_return_properties(
    _verify_version_mock: MagicMock,
    _fetch_index_infos_mock: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
    vector_index_name = "vector-index"
    fulltext_index_name = "fulltext-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5
    return_properties = ["node-property-1", "node-property-2"]
    retriever = HybridRetriever(
        driver,
        vector_index_name,
        fulltext_index_name,
        embedder,
        return_properties,
    )
    retriever.neo4j_version_is_5_23_or_above = True
    driver.execute_query.return_value = [
        [neo4j_record],
        None,
        None,
    ]
    search_query, _ = get_search_query(
        SearchType.HYBRID,
        return_properties,
        neo4j_version_is_5_23_or_above=retriever.neo4j_version_is_5_23_or_above,
    )

    records = retriever.search(query_text=query_text, top_k=top_k)

    embedder.embed_query.assert_called_once_with(query_text)
    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "vector_index_name": vector_index_name,
            "top_k": top_k,
            "query_text": query_text,
            "fulltext_index_name": fulltext_index_name,
            "query_vector": embed_query_vector,
        },
        database_=None,
    )
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(content="dummy-node", metadata={"score": 1.0}),
        ],
        metadata={"__retriever": "HybridRetriever"},
    )


@patch("neo4j_graphrag.retrievers.HybridCypherRetriever._verify_version")
def test_hybrid_cypher_retrieval_query_with_params(
    _verify_version_mock: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
    vector_index_name = "vector-index"
    fulltext_index_name = "fulltext-index"
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
        embedder,
    )
    retriever.neo4j_version_is_5_23_or_above = True
    driver.execute_query.return_value = [
        [neo4j_record],
        None,
        None,
    ]
    search_query, _ = get_search_query(
        SearchType.HYBRID,
        retrieval_query=retrieval_query,
        neo4j_version_is_5_23_or_above=retriever.neo4j_version_is_5_23_or_above,
    )

    records = retriever.search(
        query_text=query_text,
        top_k=top_k,
        query_params=query_params,
    )

    embedder.embed_query.assert_called_once_with(query_text)

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
        database_=None,
    )

    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="<Record node='dummy-node' score=1.0 node_id=123>",
                metadata=None,
            ),
        ],
        metadata={"__retriever": "HybridCypherRetriever"},
    )


@patch("neo4j_graphrag.retrievers.HybridCypherRetriever._verify_version")
def test_hybrid_cypher_retriever_with_result_format_function(
    _verify_version_mock: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
    neo4j_record: MagicMock,
    result_formatter: MagicMock,
) -> None:
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
    vector_index_name = "vector-index"
    fulltext_index_name = "fulltext-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5

    retriever = HybridCypherRetriever(
        driver,
        vector_index_name,
        fulltext_index_name,
        "",
        embedder,
        result_formatter=result_formatter,
    )
    retriever.neo4j_version_is_5_23_or_above = True
    retriever.driver.execute_query.return_value = [  # type: ignore
        [neo4j_record],
        None,
        None,
    ]

    records = retriever.search(query_text=query_text, top_k=top_k)

    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="dummy-node", metadata={"score": 1.0, "node_id": 123}
            ),
        ],
        metadata={"__retriever": "HybridCypherRetriever"},
    )
