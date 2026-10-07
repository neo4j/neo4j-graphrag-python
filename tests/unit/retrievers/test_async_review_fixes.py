from unittest.mock import AsyncMock, MagicMock, patch

import neo4j
import pytest

from neo4j_graphrag.retrievers import (
    AsyncHybridCypherRetriever,
    AsyncHybridRetriever,
    AsyncText2CypherRetriever,
)
from neo4j_graphrag.retrievers.text2cypher import READ_ONLY_QUERY_TYPE
from neo4j_graphrag.utils.version_utils import supports_search_clause_async


@pytest.fixture
def async_driver() -> MagicMock:
    driver = MagicMock(spec=neo4j.AsyncDriver)
    driver.execute_query = AsyncMock()
    return driver


@pytest.mark.asyncio
async def test_supports_search_clause_async_uses_async_driver(
    async_driver: MagicMock,
) -> None:
    record = MagicMock()
    record.__getitem__ = lambda _, key: {
        "versions": ["2026.01.0"],
        "edition": "enterprise",
    }[key]
    async_driver.execute_query.return_value = ([record], None, None)

    assert await supports_search_clause_async(async_driver) is True
    async_driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_text2cypher_retriever_uses_async_schema_and_llm(
    async_driver: MagicMock,
) -> None:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="MATCH (n) RETURN n")
    retriever = AsyncText2CypherRetriever(
        driver=async_driver,
        llm=llm,
    )

    explain = MagicMock()
    explain.summary = MagicMock(query_type=READ_ONLY_QUERY_TYPE)
    result = MagicMock(records=[neo4j.Record({"n": 1})])
    async_driver.execute_query.side_effect = [explain, result]

    with patch(
        "neo4j_graphrag.retrievers.async_text2cypher.get_schema_async",
        new=AsyncMock(return_value="RETURN n"),
    ) as get_schema_async:
        await retriever.async_init()
        get_schema_async.assert_awaited_once_with(async_driver)

    raw = await retriever.get_search_results("find nodes")
    assert raw.records == result.records
    llm.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_async_hybrid_retriever_search(
    async_driver: MagicMock,
) -> None:
    retriever = AsyncHybridRetriever(
        driver=async_driver,
        vector_index_name="vector-index",
        fulltext_index_name="fulltext-index",
    )
    async_driver.execute_query.return_value = MagicMock(records=[])

    with patch(
        "neo4j_graphrag.retrievers.async_hybrid.supports_search_clause_async",
        new=AsyncMock(return_value=False),
    ):
        raw = await retriever.get_search_results(
            query_text="find nodes",
            query_vector=[0.1, 0.2, 0.3],
        )

    assert raw.records == []
    async_driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_hybrid_cypher_retriever_search(
    async_driver: MagicMock,
) -> None:
    retriever = AsyncHybridCypherRetriever(
        driver=async_driver,
        vector_index_name="vector-index",
        fulltext_index_name="fulltext-index",
        retrieval_query="RETURN node",
    )
    async_driver.execute_query.return_value = MagicMock(records=[])

    with patch(
        "neo4j_graphrag.retrievers.async_hybrid.supports_search_clause_async",
        new=AsyncMock(return_value=False),
    ):
        raw = await retriever.get_search_results(
            query_text="find nodes",
            query_vector=[0.1, 0.2, 0.3],
        )

    assert raw.records == []
    async_driver.execute_query.assert_awaited_once()
