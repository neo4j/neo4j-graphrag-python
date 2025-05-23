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

from unittest.mock import MagicMock, patch

import neo4j
import pytest
from neo4j.exceptions import CypherSyntaxError
from neo4j_graphrag.exceptions import (
    EmbeddingRequiredError,
    RetrieverInitializationError,
    SearchValidationError,
)
from neo4j_graphrag.neo4j_queries import get_search_query
from neo4j_graphrag.retrievers import VectorCypherRetriever, VectorRetriever
from neo4j_graphrag.types import (
    RetrieverResult,
    RetrieverResultItem,
    SearchType,
)


def test_vector_retriever_initialization(driver: MagicMock) -> None:
    with patch("neo4j_graphrag.retrievers.base.get_version") as mock_get_version:
        mock_get_version.return_value = ((5, 23, 0), False, False)
        VectorRetriever(driver=driver, index_name="my-index")
        mock_get_version.assert_called_once()


@patch("neo4j_graphrag.retrievers.base.get_version")
def test_vector_retriever_invalid_index_name(
    mock_get_version: MagicMock, driver: MagicMock
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    with pytest.raises(RetrieverInitializationError) as exc_info:
        VectorRetriever(driver=driver, index_name=42)  # type: ignore

    assert "index_name" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)


@patch("neo4j_graphrag.retrievers.base.get_version")
def test_vector_retriever_invalid_database_name(
    mock_get_version: MagicMock, driver: MagicMock
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    with pytest.raises(RetrieverInitializationError) as exc_info:
        VectorRetriever(
            driver=driver,
            index_name="my-index",
            neo4j_database=42,  # type: ignore
        )

    assert "neo4j_database" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)


@patch("neo4j_graphrag.retrievers.base.get_version")
def test_vector_cypher_retriever_invalid_retrieval_query(
    mock_get_version: MagicMock, driver: MagicMock
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    with pytest.raises(RetrieverInitializationError) as exc_info:
        VectorCypherRetriever(driver=driver, index_name="my-index", retrieval_query=42)  # type: ignore

        assert "retrieval_query" in str(exc_info.value)
        assert "Input should be a valid string" in str(exc_info.value)


@patch("neo4j_graphrag.retrievers.base.get_version")
def test_vector_cypher_retriever_invalid_database_name(
    mock_get_version: MagicMock, driver: MagicMock
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    retrieval_query = """
        RETURN node.id AS node_id, node.text AS text, score
        """
    with pytest.raises(RetrieverInitializationError) as exc_info:
        VectorCypherRetriever(
            driver=driver,
            index_name="my-index",
            retrieval_query=retrieval_query,
            neo4j_database=42,  # type: ignore
        )

        assert "neo4j_database" in str(exc_info.value)
        assert "Input should be a valid string" in str(exc_info.value)


def test_vector_cypher_retriever_initialization(driver: MagicMock) -> None:
    with patch("neo4j_graphrag.retrievers.base.get_version") as mock_get_version:
        mock_get_version.return_value = ((5, 23, 0), False, False)
        VectorCypherRetriever(driver=driver, index_name="my-index", retrieval_query="")
        mock_get_version.assert_called_once()


@patch("neo4j_graphrag.retrievers.VectorRetriever._fetch_index_infos")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_similarity_search_vector_happy_path(
    mock_get_version: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    index_name = "my-index"
    dimensions = 1536
    query_vector = [1.0 for _ in range(dimensions)]
    top_k = 5
    effective_search_ratio = 2
    database = "neo4j"
    retriever = VectorRetriever(driver, index_name, neo4j_database=database)
    expected_records = [neo4j.Record({"node": {"text": "dummy-node"}, "score": 1.0})]
    retriever.driver.execute_query.return_value = [  # type: ignore
        expected_records,
        None,
        None,
    ]
    search_query, _ = get_search_query(SearchType.VECTOR)

    records = retriever.search(
        query_vector=query_vector,
        top_k=top_k,
        effective_search_ratio=effective_search_ratio,
    )

    retriever.driver.execute_query.assert_called_once_with(  # type: ignore
        search_query,
        {
            "vector_index_name": index_name,
            "top_k": top_k,
            "effective_search_ratio": effective_search_ratio,
            "query_vector": query_vector,
        },
        database_=database,
        routing_=neo4j.RoutingControl.READ,
    )
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="{'text': 'dummy-node'}",
                metadata={"score": 1.0, "nodeLabels": None, "id": None},
            ),
        ],
        metadata={"__retriever": "VectorRetriever", "query_vector": query_vector},
    )


@patch("neo4j_graphrag.retrievers.VectorRetriever._fetch_index_infos")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_similarity_search_text_happy_path(
    mock_get_version: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5
    effective_search_ratio = 2
    retriever = VectorRetriever(driver, index_name, embedder)
    driver.execute_query.return_value = [
        [neo4j_record],
        None,
        None,
    ]
    search_query, _ = get_search_query(SearchType.VECTOR)

    records = retriever.search(
        query_text=query_text,
        top_k=top_k,
        effective_search_ratio=effective_search_ratio,
    )

    embedder.embed_query.assert_called_once_with(query_text)
    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "vector_index_name": index_name,
            "top_k": top_k,
            "effective_search_ratio": effective_search_ratio,
            "query_vector": embed_query_vector,
        },
        database_=None,
        routing_=neo4j.RoutingControl.READ,
    )
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="dummy-node",
                metadata={"score": 1.0, "nodeLabels": None, "id": None},
            ),
        ],
        metadata={"__retriever": "VectorRetriever", "query_vector": embed_query_vector},
    )


@patch("neo4j_graphrag.retrievers.VectorRetriever._fetch_index_infos")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_similarity_search_text_return_properties(
    mock_get_version: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    embed_query_vector = [1.0 for _ in range(3)]
    embedder.embed_query.return_value = embed_query_vector
    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5
    effective_search_ratio = 2
    return_properties = ["node-property-1", "node-property-2"]

    retriever = VectorRetriever(
        driver, index_name, embedder, return_properties=return_properties
    )

    driver.execute_query.return_value = [
        [
            neo4j_record,
        ],
        None,
        None,
    ]
    search_query, _ = get_search_query(
        search_type=SearchType.VECTOR, return_properties=return_properties
    )

    records = retriever.search(
        query_text=query_text,
        top_k=top_k,
        effective_search_ratio=effective_search_ratio,
    )

    embedder.embed_query.assert_called_once_with(query_text)
    driver.execute_query.assert_called_once_with(
        search_query.rstrip(),
        {
            "vector_index_name": index_name,
            "top_k": top_k,
            "effective_search_ratio": effective_search_ratio,
            "query_vector": embed_query_vector,
        },
        database_=None,
        routing_=neo4j.RoutingControl.READ,
    )
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="dummy-node",
                metadata={"score": 1.0, "nodeLabels": None, "id": None},
            ),
        ],
        metadata={"__retriever": "VectorRetriever", "query_vector": embed_query_vector},
    )


def test_vector_retriever_search_missing_embedder_for_text(
    vector_retriever: VectorRetriever,
) -> None:
    query_text = "may thy knife chip and shatter"
    top_k = 5

    with pytest.raises(
        EmbeddingRequiredError, match="Embedding method required for text query"
    ):
        vector_retriever.search(query_text=query_text, top_k=top_k)


def test_vector_retriever_search_both_text_and_vector(
    vector_retriever: VectorRetriever,
) -> None:
    query_text = "may thy knife chip and shatter"
    query_vector = [1.1, 2.2, 3.3]
    top_k = 5

    with pytest.raises(
        SearchValidationError,
        match="You must provide exactly one of query_vector or query_text.",
    ):
        vector_retriever.search(
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k,
        )


@patch("neo4j_graphrag.retrievers.VectorRetriever._fetch_index_infos")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_vector_retriever_with_result_format_function(
    mock_get_version: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
    neo4j_record: MagicMock,
    result_formatter: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
    index_name = "my-index"

    retriever = VectorRetriever(
        driver,
        index_name,
        embedder=embedder,
        result_formatter=result_formatter,
    )
    query_text = "may thy knife chip and shatter"
    top_k = 5
    driver.execute_query.return_value = [
        [neo4j_record],
        None,
        None,
    ]

    records = retriever.search(
        query_text=query_text,
        top_k=top_k,
    )

    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="dummy-node", metadata={"score": 1.0, "node_id": 123}
            ),
        ],
        metadata={"__retriever": "VectorRetriever", "query_vector": embed_query_vector},
    )


def test_vector_cypher_retriever_search_missing_embedder_for_text(
    vector_cypher_retriever: VectorCypherRetriever,
) -> None:
    query_text = "may thy knife chip and shatter"
    top_k = 5

    with pytest.raises(
        EmbeddingRequiredError, match="Embedding method required for text query"
    ):
        vector_cypher_retriever.search(query_text=query_text, top_k=top_k)


def test_vector_cypher_retriever_search_both_text_and_vector(
    vector_cypher_retriever: VectorCypherRetriever,
) -> None:
    query_text = "may thy knife chip and shatter"
    query_vector = [1.1, 2.2, 3.3]
    top_k = 5

    with pytest.raises(
        SearchValidationError,
        match="You must provide exactly one of query_vector or query_text.",
    ):
        vector_cypher_retriever.search(
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k,
        )


@patch("neo4j_graphrag.retrievers.VectorCypherRetriever._fetch_index_infos")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_retrieval_query_happy_path(
    mock_get_version: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
    index_name = "my-index"
    retrieval_query = """
        RETURN node.id AS node_id, node.text AS text, score
        """
    database = "neo4j"
    retriever = VectorCypherRetriever(
        driver,
        index_name,
        retrieval_query,
        embedder=embedder,
        neo4j_database=database,
    )
    query_text = "may thy knife chip and shatter"
    top_k = 5
    effective_search_ratio = 2
    record = neo4j.Record({"node_id": 123, "text": "dummy-text", "score": 1.0})
    driver.execute_query.return_value = [
        [record],
        None,
        None,
    ]
    search_query, _ = get_search_query(
        SearchType.VECTOR, retrieval_query=retrieval_query
    )

    records = retriever.search(
        query_text=query_text,
        top_k=top_k,
        effective_search_ratio=effective_search_ratio,
    )

    embedder.embed_query.assert_called_once_with(query_text)
    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "vector_index_name": index_name,
            "top_k": top_k,
            "effective_search_ratio": effective_search_ratio,
            "query_vector": embed_query_vector,
        },
        database_=database,
        routing_=neo4j.RoutingControl.READ,
    )
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="<Record node_id=123 text='dummy-text' score=1.0>",
                metadata=None,
            ),
        ],
        metadata={
            "__retriever": "VectorCypherRetriever",
            "query_vector": embed_query_vector,
        },
    )


@patch("neo4j_graphrag.retrievers.VectorCypherRetriever._fetch_index_infos")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_retrieval_query_with_result_format_function(
    mock_get_version: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
    neo4j_record: MagicMock,
    result_formatter: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
    index_name = "my-index"
    retrieval_query = """
        RETURN node.id AS node_id, node.text AS text, score
        """

    retriever = VectorCypherRetriever(
        driver,
        index_name,
        retrieval_query,
        embedder=embedder,
        result_formatter=result_formatter,
    )
    query_text = "may thy knife chip and shatter"
    top_k = 5
    effective_search_ratio = 2
    driver.execute_query.return_value = [
        [neo4j_record],
        None,
        None,
    ]
    search_query, _ = get_search_query(
        SearchType.VECTOR, retrieval_query=retrieval_query
    )

    records = retriever.search(
        query_text=query_text,
        top_k=top_k,
        effective_search_ratio=effective_search_ratio,
    )

    embedder.embed_query.assert_called_once_with(query_text)
    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "vector_index_name": index_name,
            "top_k": top_k,
            "effective_search_ratio": effective_search_ratio,
            "query_vector": embed_query_vector,
        },
        database_=None,
        routing_=neo4j.RoutingControl.READ,
    )
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="dummy-node", metadata={"score": 1.0, "node_id": 123}
            ),
        ],
        metadata={
            "__retriever": "VectorCypherRetriever",
            "query_vector": embed_query_vector,
        },
    )


@patch("neo4j_graphrag.retrievers.VectorCypherRetriever._fetch_index_infos")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_retrieval_query_with_params(
    mock_get_version: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
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
        embedder=embedder,
    )
    query_text = "may thy knife chip and shatter"
    top_k = 5
    effective_search_ratio = 2
    driver.execute_query.return_value = [
        [neo4j.Record({"node_id": 123, "text": "dummy-text", "score": 1.0})],
        None,
        None,
    ]
    search_query, _ = get_search_query(
        SearchType.VECTOR, retrieval_query=retrieval_query
    )

    records = retriever.search(
        query_text=query_text,
        top_k=top_k,
        effective_search_ratio=effective_search_ratio,
        query_params=query_params,
    )

    embedder.embed_query.assert_called_once_with(query_text)

    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "vector_index_name": index_name,
            "top_k": top_k,
            "effective_search_ratio": effective_search_ratio,
            "query_vector": embed_query_vector,
            "param": "dummy-param",
        },
        database_=None,
        routing_=neo4j.RoutingControl.READ,
    )

    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="<Record node_id=123 text='dummy-text' score=1.0>",
                metadata=None,
            ),
        ],
        metadata={
            "__retriever": "VectorCypherRetriever",
            "query_vector": embed_query_vector,
        },
    )


@patch("neo4j_graphrag.retrievers.VectorCypherRetriever._fetch_index_infos")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_retrieval_query_cypher_error(
    mock_get_version: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
    index_name = "my-index"
    retrieval_query = """
        this is not a cypher query
        """
    retriever = VectorCypherRetriever(
        driver, index_name, retrieval_query, embedder=embedder
    )
    query_text = "may thy knife chip and shatter"
    top_k = 5
    driver.execute_query.side_effect = CypherSyntaxError

    with pytest.raises(CypherSyntaxError):
        retriever.search(
            query_text=query_text,
            top_k=top_k,
        )
