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

import datetime
import enum
from collections import defaultdict
from collections.abc import Awaitable
from typing import Any, Optional, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field

from neo4j_graphrag.experimental.pipeline.component import Component, DataModel


class ComponentDefinition(BaseModel):
    name: str
    component: Component
    run_params: dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ConnectionDefinition(BaseModel):
    start: str
    end: str
    input_config: dict[str, str]


class PipelineDefinition(BaseModel):
    components: list[ComponentDefinition]
    connections: list[ConnectionDefinition]

    def get_run_params(self) -> defaultdict[str, dict[str, Any]]:
        return defaultdict(
            dict, {c.name: c.run_params for c in self.components if c.run_params}
        )


class RunStatus(enum.Enum):
    UNKNOWN = "UNKNOWN"
    RUNNING = "RUNNING"
    DONE = "DONE"

    def possible_next_status(self) -> list[RunStatus]:
        if self == RunStatus.UNKNOWN:
            return [RunStatus.RUNNING]
        if self == RunStatus.RUNNING:
            return [RunStatus.DONE]
        if self == RunStatus.DONE:
            return []
        return []


class RunResult(BaseModel):
    status: RunStatus = RunStatus.DONE
    result: Optional[DataModel] = None
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )


class EventType(enum.Enum):
    PIPELINE_STARTED = "PIPELINE_STARTED"
    TASK_STARTED = "TASK_STARTED"
    TASK_FINISHED = "TASK_FINISHED"
    PIPELINE_FINISHED = "PIPELINE_FINISHED"

    @property
    def is_pipeline_event(self) -> bool:
        return self in [EventType.PIPELINE_STARTED, EventType.PIPELINE_FINISHED]

    @property
    def is_task_event(self) -> bool:
        return self in [EventType.TASK_STARTED, EventType.TASK_FINISHED]


class Event(BaseModel):
    event_type: EventType
    run_id: str
    """Pipeline unique run_id, same as the one returned in PipelineResult after pipeline.run"""
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    message: Optional[str] = None
    """Optional information about the status"""
    payload: Optional[dict[str, Any]] = None
    """Input or output data depending on the type of event"""


class PipelineEvent(Event):
    pass


class TaskEvent(Event):
    task_name: str
    """Name of the task as defined in pipeline.add_component"""


class EventCallbackProtocol(Protocol):
    def __call__(self, event: Event) -> Awaitable[None]: ...


EntityInputType = Union[str, dict[str, Union[str, list[dict[str, str]]]]]
RelationInputType = Union[str, dict[str, Union[str, list[dict[str, str]]]]]
"""Types derived from the SchemaEntity and SchemaRelation types,
 so the possible types for dict values are:
- str (for label and description)
- list[dict[str, str]] (for properties)
"""
