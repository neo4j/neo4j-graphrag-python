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

from functools import wraps
from typing import Optional
import asyncio
import concurrent.futures


def validate_search_query_input(
    query_text: Optional[str] = None, query_vector: Optional[list[float]] = None
) -> None:
    if not (bool(query_vector) ^ bool(query_text)):
        raise ValueError("You must provide exactly one of query_vector or query_text.")


def run_sync(function, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(lambda: asyncio.run(function(*args, **kwargs)))
        return_value = future.result()
        return return_value


def async_to_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return run_sync(func, *args, **kwargs)
    return wrapper
