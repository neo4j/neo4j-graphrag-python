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

from typing import Optional

from neo4j_graphrag.neo4j_queries import get_query_tail


def get_match_query(
    return_properties: Optional[list[str]] = None, retrieval_query: Optional[str] = None
) -> str:
    match_query = (
        "UNWIND $match_params AS match_param "
        "WITH match_param[0] AS match_id_value, match_param[1] AS score "
        "MATCH (node) "
        "WHERE node[$id_property] = match_id_value "
    )
    return match_query + get_query_tail(
        return_properties=return_properties,
        retrieval_query=retrieval_query,
        fallback_return="RETURN node, score",
    )
