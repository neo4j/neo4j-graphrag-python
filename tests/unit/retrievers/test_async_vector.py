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

from unittest.mock import AsyncMock, MagicMock, patch

import neo4j
import pytest

from neo4j_graphrag.exceptions import (
    EmbeddingRequiredError,
    RetrieverInitializationError,
    SearchValidationError,
)
from neo4j_graphrag.retrievers import AsyncVectorRetriever, AsyncVectorCypherRetriever
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem


@pytest.fixture
def async_driver() -> MagicMock:
    driver = MagicMock(spec=neo4j.AsyncDriver)
    driver.execute_query = AsyncMock()
    return driver


@pytest.fixture
def embedder() -> MagicMock:
    mock = MagicMock()
    mock.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    return mock


def _make_retriever(async_driver, index_name="my-index", embedder=None):
    """Create AsyncVectorRetriever with mocked index info already set."""
    retriever = AsyncVectorRetriever(
        driver=async_driver,
        index_name=index_name,
        embedder=embedder,
    )
    retriever._node_label = "Document"
    retriever._embedding_node_property = "embedding"
    retriever._embedding_dimension = 3
    retriever._filterable_properties = []
    return retriever


# ── Initialization ─────────────────────────────────────────────────────────────

def test_async_vector_retriever_invalid_driver() -> None:
    with pytest.raises(RetrieverInitializationError):
        AsyncVectorRetriever(driver="not-a-driver", index_name="my-index")  # type: ignore


def test_async_vector_retriever_invalid_index_name(async_driver: MagicMock) -> None:
    with pytest.raises(RetrieverInitializationError):
        AsyncVectorRetriever(driver=async_driver, index_name=42)  # type: ignore


def test_async_vector_retriever_init_ok(async_driver: MagicMock) -> None:
    retriever = AsyncVectorRetriever(driver=async_driver, index_name="my-index")
    assert retriever.index_name == "my-index"


# ── async_init ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_vector_retriever_async_init(async_driver: MagicMock) -> None:
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: {
        "labels": ["Document"],
        "properties": ["embedding"],
        "dimensions": 3,
        "filterable_properties": [],
    }[key]
    async_driver.execute_query.return_value = MagicMock(records=[mock_record])

    retriever = AsyncVectorRetriever(driver=async_driver, index_name="my-index")
    result = await retriever.async_init()
    assert result is retriever
    async_driver.execute_query.assert_called_once()


# ── search ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_vector_retriever_search_with_vector(async_driver: MagicMock) -> None:
    mock_record = MagicMock(spec=neo4j.Record)
    mock_record.get = MagicMock(return_value=None)
    async_driver.execute_query.return_value = MagicMock(records=[mock_record])

    retriever = _make_retriever(async_driver)
    with patch("neo4j_graphrag.retrievers.async_vector.supports_search_clause", return_value=False):
        result = await retriever.search(query_vector=[0.1, 0.2, 0.3], top_k=2)

    assert isinstance(result, RetrieverResult)
    async_driver.execute_query.assert_called_once()


@pytest.mark.asyncio
async def test_async_vector_retriever_search_with_text(async_driver: MagicMock, embedder: MagicMock) -> None:
    mock_record = MagicMock(spec=neo4j.Record)
    mock_record.get = MagicMock(return_value=None)
    async_driver.execute_query.return_value = MagicMock(records=[mock_record])

    retriever = _make_retriever(async_driver, embedder=embedder)
    with patch("neo4j_graphrag.retrievers.async_vector.supports_search_clause", return_value=False):
        result = await retriever.search(query_text="find something", top_k=3)

    assert isinstance(result, RetrieverResult)
    embedder.embed_query.assert_called_once_with("find something")


@pytest.mark.asyncio
async def test_async_vector_retriever_search_no_embedder_raises(async_driver: MagicMock) -> None:
    retriever = _make_retriever(async_driver)
    with patch("neo4j_graphrag.retrievers.async_vector.supports_search_clause", return_value=False):
        with pytest.raises(EmbeddingRequiredError):
            await retriever.search(query_text="find something")


@pytest.mark.asyncio
async def test_async_vector_retriever_search_no_query_raises(async_driver: MagicMock) -> None:
    retriever = _make_retriever(async_driver)
    with pytest.raises(SearchValidationError):
        await retriever.search()


# ── AsyncVectorCypherRetriever ─────────────────────────────────────────────────

def test_async_vector_cypher_retriever_init_ok(async_driver: MagicMock) -> None:
    retriever = AsyncVectorCypherRetriever(
        driver=async_driver,
        index_name="my-index",
        retrieval_query="RETURN node.id AS id",
    )
    assert retriever.index_name == "my-index"
    assert retriever.retrieval_query == "RETURN node.id AS id"


def test_async_vector_cypher_retriever_invalid_driver() -> None:
    with pytest.raises(RetrieverInitializationError):
        AsyncVectorCypherRetriever(
            driver="not-a-driver",  # type: ignore
            index_name="my-index",
            retrieval_query="RETURN node",
        )


@pytest.mark.asyncio
async def test_async_vector_cypher_retriever_search(async_driver: MagicMock, embedder: MagicMock) -> None:
    mock_record = MagicMock(spec=neo4j.Record)
    mock_record.get = MagicMock(return_value=None)
    async_driver.execute_query.return_value = MagicMock(records=[mock_record])

    retriever = AsyncVectorCypherRetriever(
        driver=async_driver,
        index_name="my-index",
        retrieval_query="RETURN node.id AS id",
        embedder=embedder,
    )
    retriever._node_label = "Document"
    retriever._node_embedding_property = "embedding"
    retriever._embedding_dimension = 3
    retriever._filterable_properties = []

    with patch("neo4j_graphrag.retrievers.async_vector.supports_search_clause", return_value=False):
        result = await retriever.search(query_vector=[0.1, 0.2, 0.3], top_k=2)

    assert isinstance(result, RetrieverResult)


# ── default_record_formatter ───────────────────────────────────────────────────

def test_async_vector_retriever_default_formatter(async_driver: MagicMock) -> None:
    retriever = _make_retriever(async_driver)
    record = MagicMock(spec=neo4j.Record)
    record.get = MagicMock(side_effect=lambda k: {"score": 0.9, "node": "test-node", "nodeLabels": ["Doc"], "id": "1"}.get(k))
    item = retriever.default_record_formatter(record)
    assert isinstance(item, RetrieverResultItem)
    assert "test-node" in item.content
    assert item.metadata["score"] == 0.9
