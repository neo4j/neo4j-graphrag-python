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
from typing import Any


class ComponentMeta(type):
    def __new__(
        meta, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> type:
        run_method = attrs.get("run")
        if run_method is not None:
            sig = inspect.signature(run_method)
            attrs["component_inputs"] = {
                param.name: {
                    "has_default": param.default != inspect.Parameter.empty,
                    "annotation": param.annotation,
                }
                for param in sig.parameters.values()
                if param.name not in ("self",)
            }
        return type.__new__(meta, name, bases, attrs)


class Component(metaclass=ComponentMeta):
    """Interface that needs to be implemented
    by all components.
    """

    async def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}
