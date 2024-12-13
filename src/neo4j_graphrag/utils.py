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

from typing import Optional, Any

from pydantic import BaseModel


def validate_search_query_input(
    query_text: Optional[str] = None, query_vector: Optional[list[float]] = None
) -> None:
    if not (bool(query_vector) ^ bool(query_text)):
        raise ValueError("You must provide exactly one of query_vector or query_text.")



class Prettyfier:
    """Prettyfy object for logging.

    I.e.: truncate long lists.
     """
    def __init__(self, max_items_in_list: int = 5):
        self.max_items_in_list = max_items_in_list

    def _prettyfy_dict(self, value: dict[Any, Any]) -> dict[Any, Any]:
        return {
            k: self(v)  # prettyfy each value
            for k, v in value.items()
        }

    def _prettyfy_list(self, value: list[Any]) -> list[Any]:
        items = [
            self(v)  # prettify each item
            for v in value[:self.max_items_in_list]
        ]
        remaining_items = len(value) - len(items)
        if remaining_items > 0:
            items.append(f"...truncated {remaining_items} items...")
        return items

    def __call__(self, value: Any) -> Any:
        if isinstance(value, dict):
            return self._prettyfy_dict(value)
        if isinstance(value, BaseModel):
            return self(value.model_dump())
        if isinstance(value, list):
            return self._prettyfy_list(value)
        return value


prettyfier = Prettyfier()
