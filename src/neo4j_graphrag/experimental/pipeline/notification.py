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
from collections.abc import Awaitable
from typing import Any, Optional, Protocol

from pydantic import BaseModel

from neo4j_graphrag.experimental.pipeline.types import RunResult


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
    timestamp: datetime.datetime
    message: Optional[str] = None
    payload: Optional[dict[str, Any]] = None


class PipelineEvent(Event):
    pass


class TaskEvent(Event):
    task_name: str


class EventCallbackProtocol(Protocol):
    def __call__(self, event: Event) -> Awaitable[None]: ...


class EventNotifier:
    def __init__(self, callback: EventCallbackProtocol | None) -> None:
        self.callback = callback

    async def notify(self, event: Event) -> None:
        if self.callback:
            await self.callback(event)

    async def notify_pipeline_started(
        self, run_id: str, input_data: Optional[dict[str, Any]] = None
    ) -> None:
        event = PipelineEvent(
            event_type=EventType.PIPELINE_STARTED,
            run_id=run_id,
            timestamp=datetime.datetime.utcnow(),
            message=None,
            payload=input_data,
        )
        await self.notify(event)

    async def notify_pipeline_finished(
        self, run_id: str, output_data: Optional[dict[str, Any]] = None
    ) -> None:
        event = PipelineEvent(
            event_type=EventType.PIPELINE_FINISHED,
            run_id=run_id,
            timestamp=datetime.datetime.utcnow(),
            message=None,
            payload=output_data,
        )
        await self.notify(event)

    async def notify_task_started(
        self,
        run_id: str,
        task_name: str,
        input_data: Optional[dict[str, Any]] = None,
    ) -> None:
        event = TaskEvent(
            event_type=EventType.TASK_STARTED,
            run_id=run_id,
            task_name=task_name,
            timestamp=datetime.datetime.utcnow(),
            message=None,
            payload=input_data,
        )
        await self.notify(event)

    async def notify_task_finished(
        self,
        run_id: str,
        task_name: str,
        output_data: Optional[RunResult] = None,
    ) -> None:
        event = TaskEvent(
            event_type=EventType.TASK_FINISHED,
            run_id=run_id,
            task_name=task_name,
            timestamp=datetime.datetime.utcnow(),
            message=None,
            payload=output_data.result.model_dump()
            if output_data and output_data.result
            else None,
        )
        await self.notify(event)
