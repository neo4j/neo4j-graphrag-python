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
"""E2E tests for SEARCH clause vector retrieval with in-index filtering.

Requires Neo4j 2026.02+ running via:
    docker compose -f tests/e2e/docker-compose.neo4j2026.yml up -d
"""

from __future__ import annotations

import time
from typing import Any, Generator

import pytest
from neo4j import Driver, GraphDatabase

from neo4j_graphrag.indexes import create_vector_index, drop_index_if_exists
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem
from neo4j_graphrag.utils.version_utils import clear_version_cache

DIMENSIONS = 5
INDEX_NAME = "search-clause-e2e-index"
LABEL = "SearchDoc"
EMBEDDING_PROP = "embedding"


@pytest.fixture(scope="module")
def driver() -> Generator[Any, Any, Any]:
    uri = "neo4j://localhost:7687"
    auth = ("neo4j", "password")
    driver = GraphDatabase.driver(uri, auth=auth)
    yield driver
    clear_version_cache()
    driver.close()


@pytest.fixture(scope="module")
def setup_search_clause_data(driver: Driver) -> None:
    """Create index with filterable_properties, insert test nodes, wait for index online."""
    # Clean slate
    driver.execute_query("MATCH (n) DETACH DELETE n")
    drop_index_if_exists(driver, INDEX_NAME)

    # Create vector index with filterable properties for in-index filtering
    create_vector_index(
        driver,
        INDEX_NAME,
        label=LABEL,
        embedding_property=EMBEDDING_PROP,
        dimensions=DIMENSIONS,
        similarity_fn="cosine",
        filterable_properties=["category", "year"],
    )

    # Insert test nodes with embeddings and filterable property values.
    # Vectors are designed so doc1 is closest to query [1,0,0,0,0], doc2 next, etc.
    nodes = [
        {
            "id": "doc1",
            "category": "science",
            "year": 2020,
            "embedding": [1.0, 0.0, 0.0, 0.0, 0.0],
        },
        {
            "id": "doc2",
            "category": "science",
            "year": 2023,
            "embedding": [0.9, 0.1, 0.0, 0.0, 0.0],
        },
        {
            "id": "doc3",
            "category": "history",
            "year": 2020,
            "embedding": [0.8, 0.2, 0.0, 0.0, 0.0],
        },
        {
            "id": "doc4",
            "category": "history",
            "year": 2023,
            "embedding": [0.7, 0.3, 0.0, 0.0, 0.0],
        },
        {
            "id": "doc5",
            "category": "science",
            "year": 2021,
            "embedding": [0.6, 0.4, 0.0, 0.0, 0.0],
        },
    ]

    for node in nodes:
        driver.execute_query(
            f"CREATE (n:{LABEL} {{id: $id, category: $category, year: $year}}) "
            f"WITH n CALL db.create.setNodeVectorProperty(n, '{EMBEDDING_PROP}', $embedding)",
            {
                "id": node["id"],
                "category": node["category"],
                "year": node["year"],
                "embedding": node["embedding"],
            },
        )

    # Wait for the index to come online
    for _ in range(60):
        result = driver.execute_query(
            "SHOW INDEXES YIELD name, state WHERE name = $name RETURN state",
            {"name": INDEX_NAME},
        )
        if result.records and result.records[0]["state"] == "ONLINE":
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"Index {INDEX_NAME} did not come online within 60s")


# -- Tests --


@pytest.mark.search_clause
@pytest.mark.usefixtures("setup_search_clause_data")
class TestSearchClauseFilteredVectorSearch:
    """Test SEARCH clause path with in-index filtering on Neo4j 2026.02."""

    def test_search_no_filters(self, driver: Driver) -> None:
        """Basic vector search without filters — uses SEARCH clause on 2026."""
        retriever = VectorRetriever(driver, INDEX_NAME)
        results = retriever.search(
            query_vector=[1.0, 0.0, 0.0, 0.0, 0.0],
            top_k=3,
        )
        assert isinstance(results, RetrieverResult)
        assert len(results.items) == 3
        for item in results.items:
            assert isinstance(item, RetrieverResultItem)

    def test_search_with_eq_filter(self, driver: Driver) -> None:
        """Equality filter — only 'science' docs returned."""
        retriever = VectorRetriever(driver, INDEX_NAME)
        results = retriever.search(
            query_vector=[1.0, 0.0, 0.0, 0.0, 0.0],
            top_k=5,
            filters={"category": {"$eq": "science"}},
        )
        assert isinstance(results, RetrieverResult)
        assert len(results.items) == 3  # doc1, doc2, doc5
        for item in results.items:
            assert "science" in item.content

    def test_search_with_gt_filter(self, driver: Driver) -> None:
        """Greater-than filter — only year > 2020 docs."""
        retriever = VectorRetriever(driver, INDEX_NAME)
        results = retriever.search(
            query_vector=[1.0, 0.0, 0.0, 0.0, 0.0],
            top_k=5,
            filters={"year": {"$gt": 2020}},
        )
        assert isinstance(results, RetrieverResult)
        assert len(results.items) == 3  # doc2(2023), doc4(2023), doc5(2021)

    def test_search_with_multiple_and_filters(self, driver: Driver) -> None:
        """Multiple AND conditions — category=science AND year>2020."""
        retriever = VectorRetriever(driver, INDEX_NAME)
        results = retriever.search(
            query_vector=[1.0, 0.0, 0.0, 0.0, 0.0],
            top_k=5,
            filters={
                "$and": [
                    {"category": {"$eq": "science"}},
                    {"year": {"$gt": 2020}},
                ],
            },
        )
        assert isinstance(results, RetrieverResult)
        assert len(results.items) == 2  # doc2(science,2023), doc5(science,2021)

    def test_search_with_incompatible_or_filter_fallback(self, driver: Driver) -> None:
        """$or filter is SEARCH-incompatible — should fall back gracefully
        to procedure path and still return results."""
        retriever = VectorRetriever(driver, INDEX_NAME)
        results = retriever.search(
            query_vector=[1.0, 0.0, 0.0, 0.0, 0.0],
            top_k=5,
            filters={
                "$or": [
                    {"category": {"$eq": "science"}},
                    {"year": {"$eq": 2020}},
                ],
            },
        )
        assert isinstance(results, RetrieverResult)
        # science docs (doc1,doc2,doc5) + history year=2020 (doc3) = 4
        assert len(results.items) == 4

    def test_search_filter_excludes_all(self, driver: Driver) -> None:
        """Filter that matches no nodes — should return empty results."""
        retriever = VectorRetriever(driver, INDEX_NAME)
        results = retriever.search(
            query_vector=[1.0, 0.0, 0.0, 0.0, 0.0],
            top_k=5,
            filters={"category": {"$eq": "nonexistent"}},
        )
        assert isinstance(results, RetrieverResult)
        assert len(results.items) == 0
