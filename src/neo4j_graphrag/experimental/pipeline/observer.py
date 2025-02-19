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
import abc
import asyncio
import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    TASK_STARTED = "TASK_STARTED"
    TASK_PROGRESS = "TASK_PROGRESS"
    TASK_FAILED = "TASK_FAILED"
    TASK_FINISHED = "TASK_FINISHED"


class Event(BaseModel):
    event_type: EventType
    sender: "Observable"
    message: Optional[str]
    data: dict[str, Any] = {}
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ObserverInterface(abc.ABC):
    @abc.abstractmethod
    async def observe(self, event: Event) -> None: ...


class Observable:
    def __init__(self) -> None:
        self.observers: list[ObserverInterface] = []

    def subscribe(self, observer: ObserverInterface) -> None:
        self.observers.append(observer)

    def subscribe_all(self, observers: list[ObserverInterface]) -> None:
        for observer in observers:
            self.subscribe(observer)

    def unsubscribe(self, observer: ObserverInterface) -> None:
        self.observers.remove(observer)

    async def notify(self, event: Event) -> None:
        tasks = [observer.observe(event) for observer in self.observers]
        await asyncio.gather(*tasks)
