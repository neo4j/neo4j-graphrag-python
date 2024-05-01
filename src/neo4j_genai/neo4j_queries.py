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
from typing import Optional

from neo4j_genai.types import SearchType


def get_search_query(
    search_type: SearchType,
    return_properties: Optional[list[str]] = None,
    retrieval_query: Optional[str] = None,
):
    query_map = {
        SearchType.VECTOR: (
            "CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector) "
            "YIELD node, score "
        ),
        SearchType.HYBRID: (
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
        ),
    }

    base_query = query_map[search_type]
    additional_query = ""

    if search_type == SearchType.VECTOR:
        if retrieval_query:
            additional_query += retrieval_query
        elif return_properties:
            return_properties_cypher = ", ".join(
                [f".{prop}" for prop in return_properties]
            )
            additional_query += (
                f"RETURN node {{{return_properties_cypher}}} as node, score"
            )
    elif search_type == SearchType.HYBRID:
        if retrieval_query:
            additional_query += retrieval_query
        elif return_properties:
            return_properties_cypher = ", ".join(
                [f".{prop}" for prop in return_properties]
            )
            additional_query += (
                f"RETURN node {{{return_properties_cypher}}} as node, score"
            )
        else:
            additional_query += "RETURN node, score"

    return base_query + additional_query
