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
from unittest import mock
from unittest.mock import MagicMock

import neo4j
import pytest

pytest.importorskip("qdrant_client", "Qdrant client is not installed")

from neo4j_graphrag.exceptions import RetrieverInitializationError
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from neo4j_graphrag.retrievers.external.utils import get_match_query
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import QueryResponse, ScoredPoint
except ImportError:
    pass


@pytest.fixture(scope="function")
def client() -> MagicMock:
    return MagicMock(spec=QdrantClient)


def test_qdrant_retriever_search_happy_path(
    driver: MagicMock, client: MagicMock
) -> None:
    retriever = QdrantNeo4jRetriever(
        driver=driver,
        client=client,
        collection_name="dummy-text",
        id_property_neo4j="sync_id",
        id_property_external="sync_id",
    )
    with mock.patch.object(retriever, "client") as mock_client:
        top_k = 5
        mock_client.query_points.return_value = QueryResponse(
            points=[
                ScoredPoint(
                    id=i,
                    version=0,
                    score=i / top_k,
                    payload={
                        "sync_id": f"node_{i}",
                    },
                )
                for i in range(top_k)
            ]
        )
        driver.execute_query.return_value = (
            [
                neo4j.Record({"node": {"sync_id": f"node_{i}"}, "score": i / top_k})
                for i in range(top_k)
            ],
            None,
            None,
        )
        query_vector = [1.0 for _ in range(1536)]
        search_query = get_match_query()
        records = retriever.search(query_vector=query_vector)

        driver.execute_query.assert_called_once_with(
            search_query,
            {
                "match_params": [[f"node_{i}", i / top_k] for i in range(top_k)],
                "id_property": "sync_id",
            },
            database_=None,
        )

    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="<Record node={'sync_id': "
                + f"'node_{i}'"
                + "} "
                + f"score={i / top_k}>",
                metadata=None,
            )
            for i in range(top_k)
        ],
        metadata={"__retriever": "QdrantNeo4jRetriever"},
    )


def test_invalid_neo4j_database_name(driver: MagicMock, client: MagicMock) -> None:
    with pytest.raises(RetrieverInitializationError) as exc_info:
        QdrantNeo4jRetriever(
            driver=driver,
            client=client,
            collection_name="dummy-text",
            id_property_neo4j="sync_id",
            neo4j_database=42,  # type: ignore
        )

    assert "neo4j_database" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)


def test_qdrant_retriever_search_return_properties(
    driver: MagicMock, client: MagicMock
) -> None:
    retriever = QdrantNeo4jRetriever(
        driver=driver,
        client=client,
        collection_name="dummy-text",
        id_property_neo4j="sync_id",
        id_property_external="sync_id",
        return_properties=["sync_id"],
    )
    with mock.patch.object(retriever, "client") as mock_client:
        top_k = 5
        mock_client.query_points.return_value = QueryResponse(
            points=[
                ScoredPoint(
                    id=i,
                    version=0,
                    score=i / top_k,
                    payload={
                        "sync_id": f"node_{i}",
                    },
                )
                for i in range(top_k)
            ]
        )
        driver.execute_query.return_value = (
            [
                neo4j.Record({"node": {"sync_id": f"node_{i}"}, "score": i / top_k})
                for i in range(top_k)
            ],
            None,
            None,
        )
        query_vector = [1.0 for _ in range(1536)]
        search_query = get_match_query(return_properties=["sync_id"])
        records = retriever.search(
            query_vector=query_vector,
        )

        driver.execute_query.assert_called_once_with(
            search_query,
            {
                "match_params": [[f"node_{i}", i / top_k] for i in range(top_k)],
                "id_property": "sync_id",
            },
            database_=None,
        )

    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="<Record node={'sync_id': "
                + f"'node_{i}'"
                + "} "
                + f"score={i / top_k}>",
                metadata=None,
            )
            for i in range(top_k)
        ],
        metadata={"__retriever": "QdrantNeo4jRetriever"},
    )


def test_qdrant_retriever_search_retrieval_query(
    driver: MagicMock, client: MagicMock
) -> None:
    retrieval_query = "WITH node MATCH (node)--(m) RETURN n, m LIMIT 10"
    retriever = QdrantNeo4jRetriever(
        driver=driver,
        client=client,
        collection_name="dummy-text",
        id_property_neo4j="sync_id",
        id_property_external="sync_id",
        retrieval_query=retrieval_query,
    )
    with mock.patch.object(retriever, "client") as mock_client:
        top_k = 5
        mock_client.query_points.return_value = QueryResponse(
            points=[
                ScoredPoint(
                    id=i,
                    version=0,
                    score=i / top_k,
                    payload={
                        "sync_id": f"node_{i}",
                    },
                )
                for i in range(top_k)
            ]
        )
        driver.execute_query.return_value = (
            [
                neo4j.Record({"node": {"sync_id": f"node_{i}"}, "score": i / top_k})
                for i in range(top_k)
            ],
            None,
            None,
        )
        query_vector = [1.0 for _ in range(1536)]
        search_query = get_match_query(retrieval_query=retrieval_query)
        records = retriever.search(
            query_vector=query_vector,
        )

        driver.execute_query.assert_called_once_with(
            search_query,
            {
                "match_params": [[f"node_{i}", i / top_k] for i in range(top_k)],
                "id_property": "sync_id",
            },
            database_=None,
        )

    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="<Record node={'sync_id': "
                + f"'node_{i}'"
                + "} "
                + f"score={i / top_k}>",
                metadata=None,
            )
            for i in range(top_k)
        ],
        metadata={"__retriever": "QdrantNeo4jRetriever"},
    )


def test_qdrant_retriever_invalid_return_properties(
    driver: MagicMock, client: MagicMock
) -> None:
    with pytest.raises(RetrieverInitializationError) as exc_info:
        QdrantNeo4jRetriever(
            driver=driver,
            client=client,
            collection_name="dummy-text",
            id_property_neo4j="dummy-text",
            return_properties=42,  # type: ignore
        )

    assert "return_properties" in str(exc_info.value)
    assert "Input should be a valid list" in str(exc_info.value)


def test_qdrant_retriever_invalid_retrieval_query(
    driver: MagicMock, client: MagicMock
) -> None:
    with pytest.raises(RetrieverInitializationError) as exc_info:
        QdrantNeo4jRetriever(
            driver=driver,
            client=client,
            collection_name="dummy-text",
            id_property_neo4j="dummy-text",
            retrieval_query=42,  # type: ignore
        )

    assert "retrieval_query" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)
