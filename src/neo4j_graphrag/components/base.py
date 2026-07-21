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
from collections.abc import Awaitable
from typing import Any, Optional, Protocol, Union, get_type_hints, runtime_checkable

from pydantic import BaseModel, ConfigDict

from neo4j_graphrag.utils.validation import issubclass_safe


@runtime_checkable
class TaskProgressNotifierProtocol(Protocol):
    """This protocol is used to send events from the component to the
    Pipeline callback protocol.
    The event sent to the callback will be of type :ref:`TaskEvent`,
    with `event_type=TASK_PROGRESS`.
    """

    def __call__(self, message: str, data: dict[str, Any]) -> Awaitable[None]: ...


class RunContext(BaseModel):
    """Context passed to the component"""

    run_id: str
    task_name: str
    notifier: Optional[TaskProgressNotifierProtocol] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def notify(self, message: str, data: dict[str, Any]) -> None:
        if self.notifier:
            await self.notifier(message=message, data=data)


class DataModel(BaseModel):
    """Input or Output data model for Components"""

    pass


class ComponentDefinitionError(Exception):
    pass


class ComponentMeta(type):
    def __new__(
        meta, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> type:
        # extract required inputs and outputs from the run method signature
        run_method = attrs.get("run")
        run_context_method = attrs.get("run_with_context")
        run = run_context_method if run_context_method is not None else run_method
        if run is None:
            raise RuntimeError(
                f"You must implement either `run` or `run_with_context` in Component '{name}'"
            )
        sig = inspect.signature(run)
        attrs["component_inputs"] = {
            param.name: {
                "has_default": param.default != inspect.Parameter.empty,
                "annotation": param.annotation,
            }
            for param in sig.parameters.values()
            if param.name not in ("self", "kwargs", "context_")
        }
        # extract returned fields from the run method return type hint
        return_model = get_type_hints(run).get("return")
        if return_model is None:
            raise ComponentDefinitionError(
                f"The run method return type must be annotated in {name}"
            )
        # the type hint must be a subclass of DataModel
        if not issubclass_safe(return_model, DataModel):
            raise ComponentDefinitionError(
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


class Component(metaclass=ComponentMeta):
    """Interface that needs to be implemented
    by all components.
    """

    # these variables are filled by the metaclass
    # added here for the type checker
    # DO NOT CHANGE
    component_inputs: dict[str, dict[str, Union[str, bool]]]
    component_outputs: dict[str, dict[str, Union[str, bool, type]]]

    async def run(self, *args: Any, **kwargs: Any) -> DataModel:
        """Run the component and return its result.

        Note: if `run_with_context` is implemented, this method will not be used.
        """
        raise NotImplementedError(
            "You must implement the `run` or `run_with_context` method. "
        )

    async def run_with_context(
        self, context_: RunContext, *args: Any, **kwargs: Any
    ) -> DataModel:
        """This method is called by the pipeline orchestrator.
        The `context_` parameter contains information about
        the pipeline run: the `run_id` and a `notify` function
        that can be used to send events from the component to
        the pipeline callback.

        This feature will be moved to the `run` method in a future
        release.

        It defaults to calling the `run` method to prevent any breaking change.
        """
        # default behavior to prevent a breaking change
        return await self.run(*args, **kwargs)
