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
from typing import Any
from unittest.mock import patch

import pytest
from neo4j_graphrag.neo4j_queries import (
    _get_upsert_texts_and_embeddings_query,
    get_query_tail,
    get_search_query,
)
from neo4j_graphrag.types import IndexType, SearchType


def test_vector_search_basic() -> None:
    expected = (
        "CALL db.index.vector.queryNodes($vector_index_name, $top_k * $effective_search_ratio, $query_vector) "
        "YIELD node, score "
        "WITH node, score LIMIT $top_k "
        "RETURN node { .*, `None`: null } AS node, labels(node) AS nodeLabels, elementId(node) AS elementId, score"
    )
    result, params = get_search_query(SearchType.VECTOR)
    assert result.strip() == expected.strip()
    assert params == {}


def test_rel_vector_search_basic() -> None:
    expected = (
        "CALL db.index.vector.queryRelationships($vector_index_name, $top_k * $effective_search_ratio, $query_vector) "
        "YIELD relationship, score "
        "WITH relationship, score LIMIT $top_k "
        "RETURN relationship { .*, `None`: null } AS relationship, elementId(relationship) AS elementId, score"
    )
    result, params = get_search_query(SearchType.VECTOR, IndexType.RELATIONSHIP)
    assert result.strip() == expected.strip()
    assert params == {}


def test_rel_vector_search_filters_err() -> None:
    with pytest.raises(Exception) as exc_info:
        get_search_query(
            SearchType.VECTOR, IndexType.RELATIONSHIP, filters={"filter": None}
        )
    assert str(exc_info.value) == "Filters are not supported for relationship indexes"


def test_rel_vector_search_hybrid_err() -> None:
    with pytest.raises(Exception) as exc_info:
        get_search_query(SearchType.HYBRID, IndexType.RELATIONSHIP)
    assert (
        str(exc_info.value) == "Hybrid search is not supported for relationship indexes"
    )


def test_hybrid_search_basic() -> None:
    expected = (
        "CALL { "
        "CALL db.index.vector.queryNodes($vector_index_name, $top_k * $effective_search_ratio, $query_vector) "
        "YIELD node, score "
        "WITH node, score LIMIT $top_k "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS vector_index_max_score "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / vector_index_max_score) AS score UNION "
        "CALL db.index.fulltext.queryNodes($fulltext_index_name, $query_text, {limit: $top_k}) "
        "YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS ft_index_max_score "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / ft_index_max_score) AS score "
        "} "
        "WITH node, max(score) AS score ORDER BY score DESC LIMIT $top_k "
        "RETURN node { .*, `None`: null } AS node, labels(node) AS nodeLabels, elementId(node) AS elementId, score"
    )
    result, _ = get_search_query(SearchType.HYBRID)
    assert result.strip() == expected.strip()


def test_vector_search_with_properties() -> None:
    properties = ["name", "age"]
    expected = (
        "CALL db.index.vector.queryNodes($vector_index_name, $top_k * $effective_search_ratio, $query_vector) "
        "YIELD node, score "
        "WITH node, score LIMIT $top_k "
        "RETURN node {.name, .age} AS node, labels(node) AS nodeLabels, elementId(node) AS elementId, score"
    )
    result, _ = get_search_query(SearchType.VECTOR, return_properties=properties)
    assert result.strip() == expected.strip()


def test_vector_search_with_retrieval_query() -> None:
    retrieval_query = "MATCH (n) RETURN n LIMIT 10"
    expected = (
        "CALL db.index.vector.queryNodes($vector_index_name, $top_k * $effective_search_ratio, $query_vector) "
        "YIELD node, score "
        "WITH node, score LIMIT $top_k " + retrieval_query
    )
    result, _ = get_search_query(SearchType.VECTOR, retrieval_query=retrieval_query)
    assert result.strip() == expected.strip()


@patch("neo4j_graphrag.neo4j_queries.get_metadata_filter", return_value=["True", {}])
def test_vector_search_with_filters(_mock: Any) -> None:
    expected = (
        "MATCH (node:`Label`) "
        "WHERE node.`vector` IS NOT NULL "
        "AND size(node.`vector`) = toInteger($embedding_dimension)"
        " AND (True) "
        "WITH node, "
        "vector.similarity.cosine(node.`vector`, $query_vector) AS score "
        "ORDER BY score DESC LIMIT $top_k "
        "RETURN node { .*, `vector`: null } AS node, labels(node) AS nodeLabels, elementId(node) AS elementId, score"
    )
    result, params = get_search_query(
        SearchType.VECTOR,
        node_label="Label",
        embedding_node_property="vector",
        embedding_dimension=1,
        filters={"field": "value"},
    )
    assert result.strip() == expected.strip()
    assert params == {"embedding_dimension": 1}


@patch(
    "neo4j_graphrag.neo4j_queries.get_metadata_filter",
    return_value=["True", {"param": "value"}],
)
def test_vector_search_with_params_from_filters(_mock: Any) -> None:
    expected = (
        "MATCH (node:`Label`) "
        "WHERE node.`vector` IS NOT NULL "
        "AND size(node.`vector`) = toInteger($embedding_dimension)"
        " AND (True) "
        "WITH node, "
        "vector.similarity.cosine(node.`vector`, $query_vector) AS score "
        "ORDER BY score DESC LIMIT $top_k "
        "RETURN node { .*, `vector`: null } AS node, labels(node) AS nodeLabels, elementId(node) AS elementId, score"
    )
    result, params = get_search_query(
        SearchType.VECTOR,
        node_label="Label",
        embedding_node_property="vector",
        embedding_dimension=1,
        filters={"field": "value"},
    )
    assert result.strip() == expected.strip()
    assert params == {"embedding_dimension": 1, "param": "value"}


def test_hybrid_search_with_retrieval_query() -> None:
    retrieval_query = "MATCH (n) RETURN n LIMIT 10"
    expected = (
        "CALL { "
        "CALL db.index.vector.queryNodes($vector_index_name, $top_k * $effective_search_ratio, $query_vector) "
        "YIELD node, score "
        "WITH node, score LIMIT $top_k "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS vector_index_max_score "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / vector_index_max_score) AS score UNION "
        "CALL db.index.fulltext.queryNodes($fulltext_index_name, $query_text, {limit: $top_k}) "
        "YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS ft_index_max_score "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / ft_index_max_score) AS score "
        "} "
        "WITH node, max(score) AS score ORDER BY score DESC LIMIT $top_k "
        + retrieval_query
    )
    result, _ = get_search_query(SearchType.HYBRID, retrieval_query=retrieval_query)
    assert result.strip() == expected.strip()


def test_hybrid_search_with_properties() -> None:
    properties = ["name", "age"]
    expected = (
        "CALL { "
        "CALL db.index.vector.queryNodes($vector_index_name, $top_k * $effective_search_ratio, $query_vector) "
        "YIELD node, score "
        "WITH node, score LIMIT $top_k "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS vector_index_max_score "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / vector_index_max_score) AS score UNION "
        "CALL db.index.fulltext.queryNodes($fulltext_index_name, $query_text, {limit: $top_k}) "
        "YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS ft_index_max_score "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / ft_index_max_score) AS score "
        "} "
        "WITH node, max(score) AS score ORDER BY score DESC LIMIT $top_k "
        "RETURN node {.name, .age} AS node, labels(node) AS nodeLabels, elementId(node) AS elementId, score"
    )
    result, _ = get_search_query(SearchType.HYBRID, return_properties=properties)
    assert result.strip() == expected.strip()


def test_get_query_tail_with_retrieval_query() -> None:
    retrieval_query = "MATCH (n) RETURN n LIMIT 10"
    expected = retrieval_query
    result = get_query_tail(retrieval_query=retrieval_query)
    assert result.strip() == expected.strip()


def test_get_query_tail_with_properties() -> None:
    properties = ["name", "age"]
    expected = "RETURN node {.name, .age} AS node, labels(node) AS nodeLabels, elementId(node) AS elementId, score"
    result = get_query_tail(return_properties=properties)
    assert result.strip() == expected.strip()


def test_get_query_tail_with_fallback() -> None:
    fallback = "HELLO"
    expected = fallback
    result = get_query_tail(fallback_return=fallback)
    assert result.strip() == expected.strip()


def test_get_query_tail_ordering_all() -> None:
    retrieval_query = "MATCH (n) RETURN n LIMIT 10"
    properties = ["name", "age"]
    fallback = "HELLO"

    expected = retrieval_query
    result = get_query_tail(
        retrieval_query=retrieval_query,
        return_properties=properties,
        fallback_return=fallback,
    )
    assert result.strip() == expected.strip()


def test_get_query_tail_ordering_no_retrieval_query() -> None:
    properties = ["name", "age"]
    fallback = "HELLO"

    expected = "RETURN node {.name, .age} AS node, labels(node) AS nodeLabels, elementId(node) AS elementId, score"
    result = get_query_tail(
        return_properties=properties,
        fallback_return=fallback,
    )
    assert result.strip() == expected.strip()


def test_get_upsert_texts_and_embeddings_query_is_above_5_23() -> None:
    expected_query = (
        "UNWIND $data AS row "
        "CALL (row) { "
        "MERGE (c:`Chunk` {id: row.id}) "
        "WITH c, row "
        "CALL db.create.setNodeVectorProperty(c, "
        "'embedding', row.embedding) "
        "SET c.`text` = row.text "
        "SET c += row.metadata "
        "} IN TRANSACTIONS OF 1000 ROWS "
    )
    actual_query = _get_upsert_texts_and_embeddings_query(
        node_label="Chunk",
        text_property="text",
        embedding_property="embedding",
        neo4j_version_is_5_23_or_above=True,
    )
    assert actual_query == expected_query


def test_get_upsert_texts_and_embeddings_query_is_below_5_23() -> None:
    expected_query = (
        "UNWIND $data AS row "
        "CALL { WITH row "
        "MERGE (c:`Chunk` {id: row.id}) "
        "WITH c, row "
        "CALL db.create.setNodeVectorProperty(c, "
        "'embedding', row.embedding) "
        "SET c.`text` = row.text "
        "SET c += row.metadata "
        "} IN TRANSACTIONS OF 1000 ROWS "
    )
    actual_query = _get_upsert_texts_and_embeddings_query(
        node_label="Chunk",
        text_property="text",
        embedding_property="embedding",
        neo4j_version_is_5_23_or_above=False,
    )
    assert actual_query == expected_query
