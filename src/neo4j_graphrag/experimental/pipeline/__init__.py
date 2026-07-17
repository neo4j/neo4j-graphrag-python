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
import warnings
from typing import Any

from .pipeline import Pipeline
from .stores import InMemoryStore, Store

__all__ = [
    "Store",
    "InMemoryStore",
    "Pipeline",
]


def __getattr__(name: str) -> Any:
    if name in ("Component", "DataModel"):
        warnings.warn(
            "neo4j_graphrag.experimental.pipeline.Component "
            "and neo4j_graphrag.experimental.pipeline.DataModel are deprecated"
            " and will be removed in version 2.0. "
            "Please use neo4j_graphrag.components.base.Component "
            "and neo4j_graphrag.components.base.DataModel instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from neo4j_graphrag.components.base import Component, DataModel  # noqa

        return locals()[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")
