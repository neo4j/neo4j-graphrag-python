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
from unittest.mock import patch

import neo4j
import pytest
from typing import Any, Dict
from neo4j.exceptions import CypherSyntaxError
from unittest.mock import MagicMock
from neo4j_genai.retrievers import VectorRetriever, VectorCypherRetriever
from neo4j_genai.exceptions import (
    RetrieverInitializationError,
    EmbeddingRequiredError,
    SearchValidationError,
)
from neo4j_genai.neo4j_queries import get_search_query
from neo4j_genai.types import (
    SearchType,
    RetrieverResult,
    RetrieverResultItem,
)


def test_vector_retriever_initialization(driver: MagicMock) -> None:
    with patch("neo4j_genai.retrievers.base.Retriever._verify_version") as mock_verify:
        VectorRetriever(driver=driver, index_name="my-index")
        mock_verify.assert_called_once()


@patch("neo4j_genai.retrievers.VectorRetriever._verify_version")
def test_vector_retriever_invalid_index_name(
    _verify_version_mock: MagicMock, driver: MagicMock
) -> None:
    with pytest.raises(RetrieverInitializationError) as exc_info:
        VectorRetriever(driver=driver, index_name=42)  # type: ignore

    assert "index_name" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)


@patch("neo4j_genai.retrievers.VectorCypherRetriever._verify_version")
def test_vector_cypher_retriever_invalid_retrieval_query(
    _verify_version_mock: MagicMock, driver: MagicMock
) -> None:
    with pytest.raises(RetrieverInitializationError) as exc_info:
        VectorCypherRetriever(driver=driver, index_name="my-index", retrieval_query=42)  # type: ignore

        assert "retrieval_query" in str(exc_info.value)
        assert "Input should be a valid string" in str(exc_info.value)


def test_vector_cypher_retriever_initialization(driver: MagicMock) -> None:
    with patch("neo4j_genai.retrievers.base.Retriever._verify_version") as mock_verify:
        VectorCypherRetriever(driver=driver, index_name="my-index", retrieval_query="")
        mock_verify.assert_called_once()


@patch("neo4j_genai.retrievers.VectorRetriever._fetch_index_infos")
@patch("neo4j_genai.retrievers.VectorRetriever._verify_version")
def test_similarity_search_vector_happy_path(
    _verify_version_mock: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    index_name = "my-index"
    dimensions = 1536
    query_vector = [1.0 for _ in range(dimensions)]
    top_k = 5
    retriever = VectorRetriever(driver, index_name)
    expected_records = [neo4j.Record({"node": {"text": "dummy-node"}, "score": 1.0})]
    retriever.driver.execute_query.return_value = [  # type: ignore
        expected_records,
        None,
        None,
    ]
    search_query, _ = get_search_query(SearchType.VECTOR)

    records = retriever.search(query_vector=query_vector, top_k=top_k)

    retriever.driver.execute_query.assert_called_once_with(  # type: ignore
        search_query,
        {
            "vector_index_name": index_name,
            "top_k": top_k,
            "query_vector": query_vector,
        },
    )
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="{'text': 'dummy-node'}",
                metadata={"score": 1.0, "nodeLabels": None, "id": None},
            ),
        ],
        metadata={"__retriever": "VectorRetriever"},
    )


@patch("neo4j_genai.retrievers.VectorRetriever._fetch_index_infos")
@patch("neo4j_genai.retrievers.VectorRetriever._verify_version")
def test_similarity_search_text_happy_path(
    _verify_version_mock: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5
    retriever = VectorRetriever(driver, index_name, embedder)
    driver.execute_query.return_value = [
        [neo4j_record],
        None,
        None,
    ]
    search_query, _ = get_search_query(SearchType.VECTOR)

    records = retriever.search(query_text=query_text, top_k=top_k)

    embedder.embed_query.assert_called_once_with(query_text)
    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "vector_index_name": index_name,
            "top_k": top_k,
            "query_vector": embed_query_vector,
        },
    )
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="dummy-node",
                metadata={"score": 1.0, "nodeLabels": None, "id": None},
            ),
        ],
        metadata={"__retriever": "VectorRetriever"},
    )


@patch("neo4j_genai.retrievers.VectorRetriever._fetch_index_infos")
@patch("neo4j_genai.retrievers.VectorRetriever._verify_version")
def test_similarity_search_text_return_properties(
    _verify_version_mock: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    embed_query_vector = [1.0 for _ in range(3)]
    embedder.embed_query.return_value = embed_query_vector
    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5
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
    search_query, _ = get_search_query(SearchType.VECTOR, return_properties)

    records = retriever.search(query_text=query_text, top_k=top_k)

    embedder.embed_query.assert_called_once_with(query_text)
    driver.execute_query.assert_called_once_with(
        search_query.rstrip(),
        {
            "vector_index_name": index_name,
            "top_k": top_k,
            "query_vector": embed_query_vector,
        },
    )
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="dummy-node",
                metadata={"score": 1.0, "nodeLabels": None, "id": None},
            ),
        ],
        metadata={"__retriever": "VectorRetriever"},
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


@patch("neo4j_genai.retrievers.VectorCypherRetriever._fetch_index_infos")
@patch("neo4j_genai.retrievers.VectorCypherRetriever._verify_version")
def test_retrieval_query_happy_path(
    _verify_version_mock: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
) -> None:
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
    index_name = "my-index"
    retrieval_query = """
        RETURN node.id AS node_id, node.text AS text, score
        """
    retriever = VectorCypherRetriever(
        driver, index_name, retrieval_query, embedder=embedder
    )
    query_text = "may thy knife chip and shatter"
    top_k = 5
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
    )

    embedder.embed_query.assert_called_once_with(query_text)
    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "vector_index_name": index_name,
            "top_k": top_k,
            "query_vector": embed_query_vector,
        },
    )
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="<Record node_id=123 text='dummy-text' score=1.0>",
                metadata=None,
            ),
        ],
        metadata={"__retriever": "VectorCypherRetriever"},
    )


@patch("neo4j_genai.retrievers.VectorCypherRetriever._fetch_index_infos")
@patch("neo4j_genai.retrievers.VectorCypherRetriever._verify_version")
def test_retrieval_query_with_result_format_function(
    _verify_version_mock: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
) -> None:
    embed_query_vector = [1.0 for _ in range(1536)]
    embedder.embed_query.return_value = embed_query_vector
    index_name = "my-index"
    retrieval_query = """
        RETURN node.id AS node_id, node.text AS text, score
        """

    def format_function(record: Dict[str, Any]) -> RetrieverResultItem:
        return RetrieverResultItem(
            content=record.get("text"),
            metadata={"score": record.get("score"), "node_id": record.get("node_id")},
        )

    retriever = VectorCypherRetriever(
        driver,
        index_name,
        retrieval_query,
        embedder=embedder,
        result_formatter=format_function,
    )
    query_text = "may thy knife chip and shatter"
    top_k = 5
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
    )

    embedder.embed_query.assert_called_once_with(query_text)
    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "vector_index_name": index_name,
            "top_k": top_k,
            "query_vector": embed_query_vector,
        },
    )
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="dummy-text", metadata={"score": 1.0, "node_id": 123}
            ),
        ],
        metadata={"__retriever": "VectorCypherRetriever"},
    )


@patch("neo4j_genai.retrievers.VectorCypherRetriever._fetch_index_infos")
@patch("neo4j_genai.retrievers.VectorCypherRetriever._verify_version")
def test_retrieval_query_with_params(
    _verify_version_mock: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
) -> None:
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
        query_params=query_params,
    )

    embedder.embed_query.assert_called_once_with(query_text)

    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "vector_index_name": index_name,
            "top_k": top_k,
            "query_vector": embed_query_vector,
            "param": "dummy-param",
        },
    )

    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="<Record node_id=123 text='dummy-text' score=1.0>",
                metadata=None,
            ),
        ],
        metadata={"__retriever": "VectorCypherRetriever"},
    )


@patch("neo4j_genai.retrievers.VectorCypherRetriever._fetch_index_infos")
@patch("neo4j_genai.retrievers.VectorCypherRetriever._verify_version")
def test_retrieval_query_cypher_error(
    _verify_version_mock: MagicMock,
    _fetch_index_infos: MagicMock,
    driver: MagicMock,
    embedder: MagicMock,
) -> None:
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
