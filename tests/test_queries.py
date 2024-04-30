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

from neo4j_genai.neo4j_queries import get_search_query
from neo4j_genai.types import SearchType


def test_vector_search_basic():
    expected = (
        "CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector) "
        "RETURN node, score"
    )
    result = get_search_query(SearchType.VECTOR)
    assert result == expected


def test_hybrid_search_basic():
    expected = (
        "CALL { "
        "CALL db.index.vector.queryNodes($vector_index_name, $top_k, $query_vector) "
        "YIELD node, score "
        "RETURN node, score UNION "
        "CALL db.index.fulltext.queryNodes($fulltext_index_name, $query_text, {limit: $top_k}) "
        "YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / max) AS score "
        "} "
        "WITH node, max(score) AS score ORDER BY score DESC LIMIT $top_k "
        "RETURN node, score"
    )
    result = get_search_query(SearchType.HYBRID)
    assert result == expected


def test_vector_search_with_properties():
    properties = ["name", "age"]
    expected = (
        "CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector) "
        "YIELD node, score "
        "RETURN node {.name, .age} as node, score"
    )
    result = get_search_query(SearchType.VECTOR, return_properties=properties)
    assert result == expected


def test_hybrid_search_with_retrieval_query():
    retrieval_query = "MATCH (n) RETURN n LIMIT 10"
    expected = (
        "CALL { "
        "CALL db.index.vector.queryNodes($vector_index_name, $top_k, $query_vector) "
        "YIELD node, score "
        "RETURN node, score UNION "
        "CALL db.index.fulltext.queryNodes($fulltext_index_name, $query_text, {limit: $top_k}) "
        "YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / max) AS score "
        "} "
        "WITH node, max(score) AS score ORDER BY score DESC LIMIT $top_k "
        + retrieval_query
    )
    result = get_search_query(SearchType.HYBRID, retrieval_query=retrieval_query)
    assert result == expected
