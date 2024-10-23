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

import neo4j


def validate_search_query_input(
    query_text: Optional[str] = None, query_vector: Optional[list[float]] = None
) -> None:
    if not (bool(query_vector) ^ bool(query_text)):
        raise ValueError("You must provide exactly one of query_vector or query_text.")


async def execute_query(
    driver: neo4j.Driver | neo4j.AsyncDriver, query: str, **kwargs: Any
) -> Any:
    if isinstance(driver, neo4j.AsyncDriver):
        result = await driver.execute_query(query, **kwargs)
    else:
        result = driver.execute_query(query, **kwargs)
    return result
