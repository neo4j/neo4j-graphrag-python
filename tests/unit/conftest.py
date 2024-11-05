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

from typing import Callable
from unittest.mock import MagicMock, patch

import neo4j
import pytest
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.retrievers import (
    HybridRetriever,
    Text2CypherRetriever,
    VectorCypherRetriever,
    VectorRetriever,
)
from neo4j_graphrag.types import RetrieverResultItem


@pytest.fixture(scope="function")
def driver() -> MagicMock:
    return MagicMock(spec=neo4j.Driver)


@pytest.fixture(scope="function")
def embedder() -> MagicMock:
    return MagicMock(spec=Embedder)


@pytest.fixture(scope="function")
def llm() -> MagicMock:
    return MagicMock(spec=LLMInterface)


@pytest.fixture(scope="function")
def retriever_mock() -> MagicMock:
    return MagicMock(spec=VectorRetriever)


@pytest.fixture(scope="function")
@patch("neo4j_graphrag.retrievers.VectorRetriever._verify_version")
def vector_retriever(
    _verify_version_mock: MagicMock, driver: MagicMock
) -> VectorRetriever:
    return VectorRetriever(driver, "my-index")


@pytest.fixture(scope="function")
@patch("neo4j_graphrag.retrievers.VectorCypherRetriever._verify_version")
def vector_cypher_retriever(
    _verify_version_mock: MagicMock, driver: MagicMock
) -> VectorCypherRetriever:
    retrieval_query = """
        RETURN node.id AS node_id, node.text AS text, score
        """
    return VectorCypherRetriever(driver, "my-index", retrieval_query)


@pytest.fixture(scope="function")
@patch("neo4j_graphrag.retrievers.HybridRetriever._verify_version")
def hybrid_retriever(
    _verify_version_mock: MagicMock, driver: MagicMock
) -> HybridRetriever:
    return HybridRetriever(driver, "my-index", "my-fulltext-index")


@pytest.fixture(scope="function")
@patch("neo4j_graphrag.retrievers.Text2CypherRetriever._verify_version")
def t2c_retriever(
    _verify_version_mock: MagicMock, driver: MagicMock, llm: MagicMock
) -> Text2CypherRetriever:
    return Text2CypherRetriever(driver, llm)


@pytest.fixture(scope="function")
def neo4j_record() -> neo4j.Record:
    return neo4j.Record({"node": "dummy-node", "score": 1.0, "node_id": 123})


@pytest.fixture(scope="function")
def result_formatter() -> Callable[[neo4j.Record], RetrieverResultItem]:
    def format_function(record: neo4j.Record) -> RetrieverResultItem:
        return RetrieverResultItem(
            content=record.get("node"),
            metadata={"score": record.get("score"), "node_id": record.get("node_id")},
        )

    return format_function
