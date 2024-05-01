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
from neo4j_genai import VectorRetriever, VectorCypherRetriever, HybridRetriever
from neo4j import Driver
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="function")
def driver():
    return MagicMock(spec=Driver)


@pytest.fixture(scope="function")
@patch("neo4j_genai.VectorRetriever._verify_version")
def vector_retriever(_verify_version_mock, driver):
    return VectorRetriever(driver, "my-index")


@pytest.fixture(scope="function")
@patch("neo4j_genai.VectorCypherRetriever._verify_version")
def vector_cypher_retriever(_verify_version_mock, driver):
    retrieval_query = """
        RETURN node.id AS node_id, node.text AS text, score
        """
    return VectorCypherRetriever(driver, "my-index", retrieval_query)


@pytest.fixture(scope="function")
@patch("neo4j_genai.HybridRetriever._verify_version")
def hybrid_retriever(_verify_version_mock, driver):
    return HybridRetriever(driver, "my-index", "my-fulltext-index")
