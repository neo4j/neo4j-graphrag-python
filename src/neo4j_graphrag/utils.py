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

import inspect
from typing import Any, Optional, Union
import asyncio
import concurrent.futures

import neo4j


def validate_search_query_input(
    query_text: Optional[str] = None, query_vector: Optional[list[float]] = None
) -> None:
    if not (bool(query_vector) ^ bool(query_text)):
        raise ValueError("You must provide exactly one of query_vector or query_text.")


async def execute_query(
    driver: Union[neo4j.Driver, neo4j.AsyncDriver], query: str, **kwargs: Any
) -> list[neo4j.Record]:
    if inspect.iscoroutinefunction(driver.execute_query):
        records, _, _ = await driver.execute_query(query, **kwargs)
        return records  # type: ignore[no-any-return]
    # ignoring type because mypy complains about coroutine
    # but we're sure at this stage we do not have a coroutine anymore
    records, _, _ = driver.execute_query(query, **kwargs)  # type: ignore[misc]
    return records  # type: ignore[no-any-return]


def run_sync(function, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(lambda: asyncio.run(function(*args, **kwargs)))
        return_value = future.result()
        return return_value



if __name__ == "__main__":
    async def async_run(char: str, repeat: int = 2) -> str:
        await asyncio.sleep(5)
        return char * repeat

    async def async_run_multiple(char, n=10):
        return await asyncio.gather(*[
            async_run(char)
            for _ in range(n)
        ])

    print(
        run_sync(async_run_multiple, "abc")
    )
