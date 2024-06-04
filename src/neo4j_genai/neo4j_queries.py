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
from typing import Optional, Any

from neo4j_genai.types import SearchType
from neo4j_genai.filters import get_metadata_filter


VECTOR_INDEX_QUERY = (
    "CALL db.index.vector.queryNodes($vector_index_name, $top_k, $query_vector) "
    "YIELD node, score"
)

VECTOR_EXACT_QUERY = (
    "WITH node, "
    "vector.similarity.cosine(node.`{embedding_node_property}`, $query_vector) AS score "
    "ORDER BY score DESC LIMIT $top_k"
)

BASE_VECTOR_EXACT_QUERY = (
    "MATCH (node:`{node_label}`) "
    "WHERE node.`{embedding_node_property}` IS NOT NULL "
    "AND size(node.`{embedding_node_property}`) = toInteger($embedding_dimension)"
)

FULL_TEXT_SEARCH_QUERY = (
    "CALL db.index.fulltext.queryNodes($fulltext_index_name, $query_text, {limit: $top_k}) "
    "YIELD node, score"
)


def _get_hybrid_query() -> str:
    return (
        f"CALL {{ {VECTOR_INDEX_QUERY} "
        f"RETURN node, score "
        f"UNION "
        f"{FULL_TEXT_SEARCH_QUERY} "
        f"WITH collect({{node:node, score:score}}) AS nodes, max(score) AS max "
        f"UNWIND nodes AS n "
        f"RETURN n.node AS node, (n.score / max) AS score }} "
        f"WITH node, max(score) AS score ORDER BY score DESC LIMIT $top_k"
    )


def _get_filtered_vector_query(
    filters: dict[str, Any],
    node_label: Optional[str],
    embedding_node_property: Optional[str],
    embedding_dimension: Optional[int],
) -> tuple[str, dict[str, Any]]:
    """Build Cypher query for vector search with filters
    Uses exact KNN.

    Args:
        filters (dict[str, Any]): filters used to pre-filter the nodes before vector search
        node_label (str): node label we want to search for
        embedding_node_property (str): the name of the property holding the embeddings
        embedding_dimension (int): the dimension of the embeddings

    Returns:
        tuple[str, dict[str, Any]]: query and parameters
    """
    where_filters, query_params = get_metadata_filter(filters, node_alias="node")
    base_query = BASE_VECTOR_EXACT_QUERY.format(
        node_label=node_label,
        embedding_node_property=embedding_node_property,
    )
    vector_query = VECTOR_EXACT_QUERY.format(
        embedding_node_property=embedding_node_property,
    )
    query_params["embedding_dimension"] = embedding_dimension
    return f"{base_query} AND ({where_filters}) {vector_query}", query_params


def _get_vector_query(
    filters: Optional[dict[str, Any]],
    node_label: Optional[str],
    embedding_node_property: Optional[str],
    embedding_dimension: Optional[int],
) -> tuple[str, dict[str, Any]]:
    """Build the vector query with or without filters

    Args:
        filters (dict[str, Any]): filters used to pre-filter the nodes before vector search
        node_label (str): node label we want to search for
        embedding_node_property (str): the name of the property holding the embeddings
        embedding_dimension (int): the dimension of the embeddings

    Returns:
        tuple[str, dict[str, Any]]: query and parameters

    """
    if filters:
        return _get_filtered_vector_query(
            filters, node_label, embedding_node_property, embedding_dimension
        )
    return VECTOR_INDEX_QUERY, {}


def get_search_query(
    search_type: SearchType,
    return_properties: Optional[list[str]] = None,
    retrieval_query: Optional[str] = None,
    node_label: Optional[str] = None,
    embedding_node_property: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    filters: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Build the search query, including pre-filtering if needed, and return clause.

    Args
        search_type: Search type we want to search for:
        return_properties (list[str]): list of property names to return.
            It can't be provided together with retrieval_query.
        retrieval_query (str): the query to use to retrieve the search results
            It can't be provided together with return_properties.
        node_label (str): node label we want to search for
        embedding_node_property (str): the name of the property holding the embeddings
        embedding_dimension (int): the dimension of the embeddings
        filters (dict[str, Any]): filters used to pre-filter the nodes before vector search

    Returns:
        tuple[str, dict[str, Any]]: query and parameters

    """
    if search_type == SearchType.HYBRID:
        if filters:
            raise Exception("Filters is not supported with Hybrid Search")
        query = _get_hybrid_query()
        params: dict[str, Any] = {}
    elif search_type == SearchType.VECTOR:
        query, params = _get_vector_query(
            filters, node_label, embedding_node_property, embedding_dimension
        )
    else:
        raise ValueError(f"Search type is not supported: {search_type}")
    query_tail = get_query_tail(
        retrieval_query, return_properties, fallback_return="RETURN node, score"
    )
    return f"{query} {query_tail}", params


def get_query_tail(
    retrieval_query: Optional[str] = None,
    return_properties: Optional[list[str]] = None,
    fallback_return: Optional[str] = None,
) -> str:
    """Build the RETURN statement after the search is performed

    Args
        return_properties (list[str]): list of property names to return.
            It can't be provided together with retrieval_query.
        retrieval_query (str): the query to use to retrieve the search results
            It can't be provided together with return_properties.
        fallback_return (str): the fallback return statement to use to retrieve the search results

    Returns:
       str: the RETURN statement
    """
    if retrieval_query:
        return retrieval_query
    if return_properties:
        return_properties_cypher = ", ".join([f".{prop}" for prop in return_properties])
        return f"RETURN node {{{return_properties_cypher}}} as node, score"
    return fallback_return if fallback_return else ""
