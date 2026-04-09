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
"""Tests for Cypher 25 SEARCH clause support with in-index filtering."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from neo4j_graphrag.filters import (
    get_search_filter,
    is_search_compatible_filter,
)
from neo4j_graphrag.neo4j_queries import (
    get_search_query,
    _get_search_vector_query,
    _get_search_vector_query_no_filter,
)
from neo4j_graphrag.types import SearchType
from neo4j_graphrag.utils.version_utils import has_search_clause_support


# ---- Version detection ----


class TestVersionDetection:
    def test_neo4j_5_x_no_search_support(self) -> None:
        assert has_search_clause_support((5, 26, 5)) is False

    def test_neo4j_2026_01_has_search_support(self) -> None:
        assert has_search_clause_support((2026, 1, 0)) is True

    def test_neo4j_2026_02_has_search_support(self) -> None:
        assert has_search_clause_support((2026, 2, 0)) is True

    def test_neo4j_2025_no_search_support(self) -> None:
        assert has_search_clause_support((2025, 12, 0)) is False


# ---- Filter compatibility ----


class TestFilterCompatibility:
    def test_simple_equality_compatible(self) -> None:
        assert is_search_compatible_filter({"name": "Alice"}) is True

    def test_simple_comparison_compatible(self) -> None:
        assert is_search_compatible_filter({"age": {"$gte": 18}}) is True

    def test_multiple_and_fields_compatible(self) -> None:
        assert is_search_compatible_filter({"age": {"$gte": 18}, "score": {"$lt": 100}}) is True

    def test_explicit_and_compatible(self) -> None:
        filt = {"$and": [{"age": {"$gte": 18}}, {"rating": {"$lte": 5}}]}
        assert is_search_compatible_filter(filt) is True

    def test_or_not_compatible(self) -> None:
        filt = {"$or": [{"age": {"$gte": 18}}, {"name": "Bob"}]}
        assert is_search_compatible_filter(filt) is False

    def test_in_not_compatible(self) -> None:
        assert is_search_compatible_filter({"status": {"$in": ["A", "B"]}}) is False

    def test_nin_not_compatible(self) -> None:
        assert is_search_compatible_filter({"status": {"$nin": ["X"]}}) is False

    def test_like_not_compatible(self) -> None:
        assert is_search_compatible_filter({"name": {"$like": "Ali%"}}) is False

    def test_ilike_not_compatible(self) -> None:
        assert is_search_compatible_filter({"name": {"$ilike": "ali%"}}) is False

    def test_between_not_compatible(self) -> None:
        assert is_search_compatible_filter({"age": {"$between": [10, 20]}}) is False

    def test_ne_not_compatible(self) -> None:
        assert is_search_compatible_filter({"status": {"$ne": "deleted"}}) is False

    def test_empty_dict_compatible(self) -> None:
        assert is_search_compatible_filter({}) is True

    def test_non_dict_not_compatible(self) -> None:
        assert is_search_compatible_filter("not a dict") is False  # type: ignore

    def test_all_simple_operators_compatible(self) -> None:
        for op in ["$eq", "$lt", "$gt", "$lte", "$gte"]:
            assert is_search_compatible_filter({"field": {op: 42}}) is True


# ---- Search filter generation ----


class TestSearchFilter:
    def test_simple_equality(self) -> None:
        query, params = get_search_filter({"name": "Alice"})
        assert query == "node.name = $param_0"
        assert params == {"param_0": "Alice"}

    def test_comparison_operator(self) -> None:
        query, params = get_search_filter({"age": {"$gte": 18}})
        assert query == "node.age >= $param_0"
        assert params == {"param_0": 18}

    def test_multiple_fields_and(self) -> None:
        query, params = get_search_filter({"age": {"$gte": 18}, "score": {"$lt": 100}})
        assert "node.age >= $param_0" in query
        assert "node.score < $param_1" in query
        assert " AND " in query
        assert params == {"param_0": 18, "param_1": 100}

    def test_explicit_and(self) -> None:
        filt = {"$and": [{"age": {"$gte": 18}}, {"rating": {"$lte": 5}}]}
        query, params = get_search_filter(filt)
        assert "node.age >= $param_0" in query
        assert "node.rating <= $param_1" in query
        assert " AND " in query

    def test_custom_node_alias(self) -> None:
        query, params = get_search_filter({"name": "Bob"}, node_alias="m")
        assert query == "m.name = $param_0"


# ---- SEARCH query generation ----


class TestSearchQueryGeneration:
    def test_search_vector_query_no_filter(self) -> None:
        query, params = _get_search_vector_query_no_filter("Movie")
        assert "MATCH (node:`Movie`)" in query
        assert "SEARCH node IN (" in query
        assert "VECTOR INDEX $vector_index_name" in query
        assert "FOR $query_vector" in query
        assert "LIMIT $top_k" in query
        assert "SCORE AS score" in query
        assert "WHERE" not in query
        assert params == {}

    def test_search_vector_query_with_filter(self) -> None:
        filters = {"rating": {"$gte": 7.0}}
        query, params = _get_search_vector_query(filters, "Movie", "embedding")
        assert "MATCH (node:`Movie`)" in query
        assert "SEARCH node IN (" in query
        assert "VECTOR INDEX $vector_index_name" in query
        assert "WHERE node.rating >= $param_0" in query
        assert "LIMIT $top_k" in query
        assert "SCORE AS score" in query
        assert params == {"param_0": 7.0}

    def test_search_vector_query_with_multi_filter(self) -> None:
        filters = {"rating": {"$gte": 7.0}, "year": {"$gt": 2000}}
        query, params = _get_search_vector_query(filters, "Movie", "embedding")
        assert "WHERE" in query
        assert " AND " in query
        assert params == {"param_0": 7.0, "param_1": 2000}


# ---- Integration with get_search_query ----


class TestGetSearchQueryWithSearchClause:
    def test_vector_search_uses_search_clause_when_supported(self) -> None:
        """When neo4j_version_supports_search_clause=True and no filters,
        should use SEARCH clause if node_label is available."""
        result, params = get_search_query(
            SearchType.VECTOR,
            node_label="Movie",
            embedding_node_property="embedding",
            neo4j_version_supports_search_clause=True,
        )
        assert "SEARCH node IN (" in result
        assert "VECTOR INDEX" in result

    def test_vector_search_uses_procedure_when_no_support(self) -> None:
        """When neo4j_version_supports_search_clause=False, should use procedure."""
        result, params = get_search_query(
            SearchType.VECTOR,
            neo4j_version_supports_search_clause=False,
        )
        assert "db.index.vector.queryNodes" in result

    def test_vector_search_with_compatible_filters_uses_search(self) -> None:
        """Compatible filters + SEARCH support = SEARCH ... WHERE."""
        result, params = get_search_query(
            SearchType.VECTOR,
            node_label="Movie",
            embedding_node_property="embedding",
            embedding_dimension=1536,
            filters={"rating": {"$gte": 7.0}},
            neo4j_version_supports_search_clause=True,
        )
        assert "SEARCH node IN (" in result
        assert "WHERE" in result
        assert "rating" in result
        assert params.get("param_0") == 7.0

    @patch("neo4j_graphrag.neo4j_queries.get_metadata_filter", return_value=["True", {}])
    def test_vector_search_with_incompatible_filters_falls_back(self, _mock: Any) -> None:
        """Incompatible filters + SEARCH support = fallback to exact match."""
        result, params = get_search_query(
            SearchType.VECTOR,
            node_label="Movie",
            embedding_node_property="embedding",
            embedding_dimension=1536,
            filters={"status": {"$in": ["A", "B"]}},
            neo4j_version_supports_search_clause=True,
        )
        # Should fallback to exact match (MATCH ... WHERE ... vector.similarity.cosine)
        assert "SEARCH" not in result
        assert "vector.similarity.cosine" in result

    @patch("neo4j_graphrag.neo4j_queries.get_metadata_filter", return_value=["True", {}])
    def test_vector_search_with_filters_no_search_support(self, _mock: Any) -> None:
        """Filters + no SEARCH support = exact match."""
        result, params = get_search_query(
            SearchType.VECTOR,
            node_label="Movie",
            embedding_node_property="embedding",
            embedding_dimension=1536,
            filters={"rating": {"$gte": 7.0}},
            neo4j_version_supports_search_clause=False,
        )
        assert "SEARCH" not in result
        assert "vector.similarity.cosine" in result

    def test_vector_search_no_label_falls_back_to_procedure(self) -> None:
        """Without node_label, even with SEARCH support, use procedure."""
        result, params = get_search_query(
            SearchType.VECTOR,
            neo4j_version_supports_search_clause=True,
        )
        assert "db.index.vector.queryNodes" in result

    def test_hybrid_search_unaffected(self) -> None:
        """SEARCH clause support should not affect hybrid search."""
        result, _ = get_search_query(
            SearchType.HYBRID,
            neo4j_version_supports_search_clause=True,
        )
        assert "db.index.vector.queryNodes" in result
        assert "db.index.fulltext.queryNodes" in result
