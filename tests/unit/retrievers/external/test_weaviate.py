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

from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import neo4j
import pytest
from neo4j_genai.exceptions import RetrieverInitializationError
from neo4j_genai.retrievers import WeaviateNeo4jRetriever
from neo4j_genai.retrievers.external.utils import get_match_query
from neo4j_genai.types import RetrieverResult, RetrieverResultItem
from weaviate.client import WeaviateClient


# Weaviate class with fake methods
class WClient(WeaviateClient):
    def __init__(
        self,
        node_id_value: Optional[str] = None,
        node_match_score: Optional[float] = None,
    ) -> None:
        self.collections = MagicMock()
        self.collections.get = MagicMock()
        query = MagicMock()
        self.collections.get.return_value = SimpleNamespace(query=query)
        query.near_text.return_value = SimpleNamespace(
            objects=[
                SimpleNamespace(
                    properties={"neo4j_id": node_id_value},
                    metadata=SimpleNamespace(certainty=node_match_score),
                )
            ]
        )


def test_text_search_remote_vector_store_happy_path(driver: MagicMock) -> None:
    query_text = "may thy knife chip and shatter"
    top_k = 5
    node_id_value = "node-test-id"
    node_match_score = 0.9

    wc = WClient(node_id_value=node_id_value, node_match_score=node_match_score)

    retriever = WeaviateNeo4jRetriever(
        driver=driver,
        client=wc,
        collection="dummy-collection",
        id_property_neo4j="sync_id",
        id_property_external="neo4j_id",
    )
    driver.execute_query.return_value = [
        [neo4j.Record({"node": {"sync_id": node_id_value}, "score": node_match_score})],
        None,
        None,
    ]
    search_query = get_match_query()

    records = retriever.search(query_text=query_text, top_k=top_k)

    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "match_params": [
                [node_id_value, node_match_score],
            ],
            "id_property": "sync_id",
        },
        database_=None,
    )
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="<Record node={'sync_id': 'node-test-id'} score=0.9>",
                metadata=None,
            ),
        ],
        metadata={"__retriever": "WeaviateNeo4jRetriever"},
    )


def test_invalid_neo4j_database_name(driver: MagicMock) -> None:
    node_id_value = "node-test-id"
    node_match_score = 0.9

    wc = WClient(node_id_value=node_id_value, node_match_score=node_match_score)

    with pytest.raises(RetrieverInitializationError) as exc_info:
        WeaviateNeo4jRetriever(
            driver=driver,
            client=wc,
            collection="dummy-collection",
            id_property_neo4j="sync_id",
            id_property_external="neo4j_id",
            neo4j_database=42,  # type: ignore
        )

    assert "neo4j_database" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)


def test_text_search_remote_vector_store_return_properties(driver: MagicMock) -> None:
    query_text = "may thy knife chip and shatter"
    top_k = 5
    node_id_value = "node-test-id"
    node_match_score = 0.9

    wc = WClient(node_id_value=node_id_value, node_match_score=node_match_score)

    retriever = WeaviateNeo4jRetriever(
        driver=driver,
        client=wc,
        collection="dummy-collection",
        id_property_neo4j="sync_id",
        id_property_external="neo4j_id",
        return_properties=["sync_id"],
    )
    driver.execute_query.return_value = [
        [neo4j.Record({"node": {"sync_id": node_id_value}, "score": node_match_score})],
        None,
        None,
    ]
    search_query = get_match_query(return_properties=["sync_id"])

    records = retriever.search(query_text=query_text, top_k=top_k)

    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "match_params": [
                [node_id_value, node_match_score],
            ],
            "id_property": "sync_id",
        },
        database_=None,
    )
    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="<Record node={'sync_id': 'node-test-id'} score=0.9>",
                metadata=None,
            ),
        ],
        metadata={"__retriever": "WeaviateNeo4jRetriever"},
    )


def test_text_search_remote_vector_store_retrieval_query(driver: MagicMock) -> None:
    query_text = "may thy knife chip and shatter"
    top_k = 5
    node_id_value = "node-test-id"
    node_match_score = 0.9
    retrieval_query = "WITH node MATCH (node)--(m) RETURN n, m LIMIT 10"

    wc = WClient(node_id_value=node_id_value, node_match_score=node_match_score)

    retriever = WeaviateNeo4jRetriever(
        driver=driver,
        client=wc,
        collection="dummy-collection",
        id_property_neo4j="sync_id",
        id_property_external="neo4j_id",
        retrieval_query=retrieval_query,
    )
    driver.execute_query.return_value = [
        [neo4j.Record({"node": {"sync_id": node_id_value}, "score": node_match_score})],
        None,
        None,
    ]
    search_query = get_match_query(retrieval_query=retrieval_query)

    records = retriever.search(query_text=query_text, top_k=top_k)

    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "match_params": [
                [node_id_value, node_match_score],
            ],
            "id_property": "sync_id",
        },
        database_=None,
    )

    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="<Record node={'sync_id': 'node-test-id'} score=0.9>",
                metadata=None,
            ),
        ],
        metadata={"__retriever": "WeaviateNeo4jRetriever"},
    )


def test_match_query() -> None:
    match_query = get_match_query()
    expected = (
        "UNWIND $match_params AS match_param "
        "WITH match_param[0] AS match_id_value, match_param[1] AS score "
        "MATCH (node) "
        "WHERE node[$id_property] = match_id_value "
        "RETURN node, score"
    )
    assert match_query.strip() == expected.strip()


def test_match_query_with_return_properties() -> None:
    match_query = get_match_query(return_properties=["name", "age"])
    expected = (
        "UNWIND $match_params AS match_param "
        "WITH match_param[0] AS match_id_value, match_param[1] AS score "
        "MATCH (node) "
        "WHERE node[$id_property] = match_id_value "
        "RETURN node {.name, .age} as node, score"
    )
    assert match_query.strip() == expected.strip()


def test_match_query_with_retrieval_query() -> None:
    retrieval_query = "WITH node MATCH (node)--(m) RETURN node, m LIMIT 10"
    match_query = get_match_query(retrieval_query=retrieval_query)
    expected = (
        "UNWIND $match_params AS match_param "
        "WITH match_param[0] AS match_id_value, match_param[1] AS score "
        "MATCH (node) "
        "WHERE node[$id_property] = match_id_value " + retrieval_query
    )
    assert match_query.strip() == expected.strip()


def test_match_query_with_both_return_properties_and_retrieval_query() -> None:
    # Should ignore return_properties
    retrieval_query = "WITH node MATCH (node)--(m) RETURN node, m LIMIT 10"
    match_query = get_match_query(
        return_properties=["name", "age"], retrieval_query=retrieval_query
    )
    expected = (
        "UNWIND $match_params AS match_param "
        "WITH match_param[0] AS match_id_value, match_param[1] AS score "
        "MATCH (node) "
        "WHERE node[$id_property] = match_id_value " + retrieval_query
    )
    assert match_query.strip() == expected.strip()
