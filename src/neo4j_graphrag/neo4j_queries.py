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
from __future__ import annotations

from typing import Any, Optional

from neo4j_graphrag.filters import get_metadata_filter
from neo4j_graphrag.types import IndexType, SearchType

NODE_VECTOR_INDEX_QUERY = (
    "CALL db.index.vector.queryNodes"
    "($vector_index_name, $top_k * $effective_search_ratio, $query_vector) "
    "YIELD node, score "
    "WITH node, score LIMIT $top_k"
)

REL_VECTOR_INDEX_QUERY = (
    "CALL db.index.vector.queryRelationships"
    "($vector_index_name, $top_k * $effective_search_ratio, $query_vector) "
    "YIELD relationship, score "
    "WITH relationship, score LIMIT $top_k"
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

UPSERT_NODE_QUERY = (
    "UNWIND $rows AS row "
    "CREATE (n:__KGBuilder__ {id: row.id}) "
    "SET n += row.properties "
    "WITH n, row CALL apoc.create.addLabels(n, row.labels) YIELD node "
    "WITH node as n, row CALL { "
    "WITH n, row WITH n, row WHERE row.embedding_properties IS NOT NULL "
    "UNWIND keys(row.embedding_properties) as emb "
    "CALL db.create.setNodeVectorProperty(n, emb, row.embedding_properties[emb]) "
    "RETURN count(*) as nbEmb "
    "} "
    "RETURN elementId(n)"
)

UPSERT_NODE_QUERY_VARIABLE_SCOPE_CLAUSE = (
    "UNWIND $rows AS row "
    "CREATE (n:__KGBuilder__ {id: row.id}) "
    "SET n += row.properties "
    "WITH n, row CALL apoc.create.addLabels(n, row.labels) YIELD node "
    "WITH node as n, row CALL (n, row) { "
    "WITH n, row WITH n, row WHERE row.embedding_properties IS NOT NULL "
    "UNWIND keys(row.embedding_properties) as emb "
    "CALL db.create.setNodeVectorProperty(n, emb, row.embedding_properties[emb]) "
    "RETURN count(*) as nbEmb "
    "} "
    "RETURN elementId(n)"
)

UPSERT_RELATIONSHIP_QUERY = (
    "UNWIND $rows as row "
    "MATCH (start:__KGBuilder__ {id: row.start_node_id}) "
    "MATCH (end:__KGBuilder__ {id: row.end_node_id}) "
    "WITH start, end, row "
    "CALL apoc.merge.relationship(start, row.type, {}, row.properties, end, row.properties) YIELD rel  "
    "WITH rel, row CALL { "
    "WITH rel, row WITH rel, row WHERE row.embedding_properties IS NOT NULL "
    "UNWIND keys(row.embedding_properties) as emb "
    "CALL db.create.setRelationshipVectorProperty(rel, emb, row.embedding_properties[emb]) "
    "} "
    "RETURN elementId(rel)"
)

UPSERT_RELATIONSHIP_QUERY_VARIABLE_SCOPE_CLAUSE = (
    "UNWIND $rows as row "
    "MATCH (start:__KGBuilder__ {id: row.start_node_id}) "
    "MATCH (end:__KGBuilder__ {id: row.end_node_id}) "
    "WITH start, end, row "
    "CALL apoc.merge.relationship(start, row.type, {}, row.properties, end, row.properties) YIELD rel  "
    "WITH rel, row CALL (rel, row) { "
    "WITH rel, row WITH rel, row WHERE row.embedding_properties IS NOT NULL "
    "UNWIND keys(row.embedding_properties) as emb "
    "CALL db.create.setRelationshipVectorProperty(rel, emb, row.embedding_properties[emb]) "
    "} "
    "RETURN elementId(rel)"
)

UPSERT_VECTOR_ON_NODE_QUERY = (
    "MATCH (n) "
    "WHERE elementId(n) = $node_element_id "
    "WITH n "
    "CALL db.create.setNodeVectorProperty(n, $embedding_property, $vector) "
    "RETURN n"
)

UPSERT_VECTOR_ON_RELATIONSHIP_QUERY = (
    "MATCH ()-[r]->() "
    "WHERE elementId(r) = $rel_element_id "
    "WITH r "
    "CALL db.create.setRelationshipVectorProperty(r, $embedding_property, $vector) "
    "RETURN r"
)


def _get_hybrid_query(neo4j_version_is_5_23_or_above: bool) -> str:
    call_prefix = "CALL () { " if neo4j_version_is_5_23_or_above else "CALL { "
    query_body = (
        f"{NODE_VECTOR_INDEX_QUERY} "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS vector_index_max_score "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / vector_index_max_score) AS score "
        "UNION "
        f"{FULL_TEXT_SEARCH_QUERY} "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS ft_index_max_score "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / ft_index_max_score) AS score } "
        "WITH node, max(score) AS score ORDER BY score DESC LIMIT $top_k"
    )
    return call_prefix + query_body


def _get_filtered_vector_query(
    filters: dict[str, Any],
    node_label: str,
    embedding_node_property: str,
    embedding_dimension: int,
    use_parallel_runtime: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Build Cypher query for vector search with filters
    Uses exact KNN.

    Args:
        filters (dict[str, Any]): filters used to pre-filter the nodes before vector search
        node_label (str): node label we want to search for
        embedding_node_property (str): the name of the property holding the embeddings
        embedding_dimension (int): the dimension of the embeddings
        use_parallel_runtime (bool): Whether or not use the parallel runtime to run the query.
            Defaults to False.

    Returns:
        tuple[str, dict[str, Any]]: query and parameters
    """
    parallel_query = (
        "CYPHER runtime = parallel parallelRuntimeSupport=all "
        if use_parallel_runtime
        else ""
    )
    where_filters, query_params = get_metadata_filter(filters, node_alias="node")
    base_query = BASE_VECTOR_EXACT_QUERY.format(
        node_label=node_label,
        embedding_node_property=embedding_node_property,
    )
    vector_query = VECTOR_EXACT_QUERY.format(
        embedding_node_property=embedding_node_property,
    )
    query_params["embedding_dimension"] = embedding_dimension
    return (
        parallel_query + f"{base_query} AND ({where_filters}) {vector_query}",
        query_params,
    )


def get_search_query(
    search_type: SearchType,
    index_type: IndexType = IndexType.NODE,
    return_properties: Optional[list[str]] = None,
    retrieval_query: Optional[str] = None,
    node_label: Optional[str] = None,
    embedding_node_property: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    filters: Optional[dict[str, Any]] = None,
    neo4j_version_is_5_23_or_above: bool = False,
) -> tuple[str, dict[str, Any]]:
    """
    Constructs a search query for vector or hybrid search, including optional pre-filtering
    and return clause.

    Args:
        search_type (SearchType): Specifies whether to perform a vector or hybrid search.
        index_type (Optional[IndexType]): Specifies whether to search over node or
            relationship indexes. Defaults to 'node'.
        return_properties (Optional[list[str]]): List of property names to return.
            Cannot be provided alongside `retrieval_query`.
        retrieval_query (Optional[str]): Query used to retrieve search results.
            Cannot be provided alongside `return_properties`.
        node_label (Optional[str]): Label of the nodes to search.
        embedding_node_property (Optional[str])): Name of the property containing the embeddings.
        embedding_dimension (Optional[int]): Dimension of the embeddings.
        filters (Optional[dict[str, Any]]): Filters to pre-filter nodes before vector search.
        neo4j_version_is_5_23_or_above (Optional[bool]): Whether the Neo4j version is 5.23 or above.

    Returns:
        tuple[str, dict[str, Any]]: A tuple containing the constructed query string and
        a dictionary of query parameters.

     Raises:
        Exception: If filters are used with Hybrid Search.
        Exception: If Vector Search with filters is missing required parameters.
        ValueError: If an unsupported search type is provided.
    """
    if index_type == IndexType.NODE:
        if search_type == SearchType.HYBRID:
            if filters:
                raise Exception("Filters are not supported with hybrid search")
            query = _get_hybrid_query(neo4j_version_is_5_23_or_above)
            params: dict[str, Any] = {}
        elif search_type == SearchType.VECTOR:
            if filters:
                if (
                    node_label is not None
                    and embedding_node_property is not None
                    and embedding_dimension is not None
                ):
                    query, params = _get_filtered_vector_query(
                        filters,
                        node_label,
                        embedding_node_property,
                        embedding_dimension,
                    )
                else:
                    raise Exception(
                        "Vector Search with filters requires: node_label, embedding_node_property, embedding_dimension"
                    )
            else:
                query, params = NODE_VECTOR_INDEX_QUERY, {}
        else:
            raise ValueError(f"Search type is not supported: {search_type}")
        fallback_return = (
            f"RETURN node {{ .*, `{embedding_node_property}`: null }} AS node, "
            "labels(node) AS nodeLabels, elementId(node) AS elementId, score"
        )
    elif index_type == IndexType.RELATIONSHIP:
        if filters:
            raise Exception("Filters are not supported for relationship indexes")
        if search_type == SearchType.HYBRID:
            raise Exception("Hybrid search is not supported for relationship indexes")
        elif search_type == SearchType.VECTOR:
            query, params = REL_VECTOR_INDEX_QUERY, {}
            fallback_return = (
                f"RETURN relationship {{ .*, `{embedding_node_property}`: null }} AS relationship, "
                "elementId(relationship) AS elementId, score"
            )
        else:
            raise ValueError(f"Search type is not supported: {search_type}")
    else:
        raise ValueError(f"Index type is not supported: {index_type}")

    query_tail = get_query_tail(
        retrieval_query,
        return_properties,
        fallback_return=fallback_return,
        index_type=index_type,
    )

    return f"{query} {query_tail}", params


def get_query_tail(
    retrieval_query: Optional[str] = None,
    return_properties: Optional[list[str]] = None,
    fallback_return: Optional[str] = None,
    index_type: IndexType = IndexType.NODE,
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
        if index_type == IndexType.NODE:
            return f"RETURN node {{{return_properties_cypher}}} AS node, labels(node) AS nodeLabels, elementId(node) AS elementId, score"
        elif index_type == IndexType.RELATIONSHIP:
            return f"RETURN relationship {{{return_properties_cypher}}} AS relationship, elementId(relationship) AS elementId, score"
        else:
            raise ValueError(f"Index type is not supported: {index_type}")
    return fallback_return if fallback_return else ""
