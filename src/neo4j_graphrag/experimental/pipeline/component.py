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

import abc
import inspect
from typing import Any, get_type_hints

from pydantic import BaseModel

from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError


class DataModel(BaseModel):
    """Input or Output data model for Components"""

    pass


class ComponentMeta(abc.ABCMeta):
    def __new__(
        meta, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> type:
        # extract required inputs and outputs from the run method signature
        run_method = attrs.get("run")
        if run_method is not None:
            sig = inspect.signature(run_method)
            attrs["component_inputs"] = {
                param.name: {
                    "has_default": param.default != inspect.Parameter.empty,
                    "annotation": param.annotation,
                }
                for param in sig.parameters.values()
                if param.name not in ("self", "kwargs")
            }
            # extract returned fields from the run method return type hint
            return_model = get_type_hints(run_method).get("return")
            if return_model is None:
                raise PipelineDefinitionError(
                    f"The run method return type must be annotated in {name}"
                )
            # the type hint must be a subclass of DataModel
            if not issubclass(return_model, DataModel):
                raise PipelineDefinitionError(
                    f"The run method must return a subclass of DataModel in {name}"
                )
            attrs["component_outputs"] = {
                f: {
                    "has_default": field.is_required(),
                    "annotation": field.annotation,
                }
                for f, field in return_model.model_fields.items()
            }
        return type.__new__(meta, name, bases, attrs)


class Component(abc.ABC, metaclass=ComponentMeta):
    """Interface that needs to be implemented
    by all components.
    """

    # these variables are filled by the metaclass
    # added here for the type checker
    # DO NOT CHANGE
    component_inputs: dict[str, dict[str, str | bool]]
    component_outputs: dict[str, dict[str, str | bool]]

    @abc.abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> DataModel:
        pass
