import datetime
import enum
from typing import Any, Optional

from pydantic import BaseModel


class EventType(enum.Enum):
    PIPELINE_STARTED = "PIPELINE_STARTED"
    TASK_STARTED = "TASK_STARTED"
    TASK_FINISHED = "TASK_FINISHED"
    PIPELINE_FINISHED = "PIPELINE_FINISHED"


class Event(BaseModel):
    event_type: EventType
    run_id: str
    timestamp: datetime.datetime
    message: Optional[str] = None
    payload: Optional[dict[str, Any]] = None


class PipelineEvent(Event):
    pass


class ComponentEvent(Event):
    component_name: str


class EventNotifier:
    def __init__(self, callback: Any) -> None:
        # TODO: define protocol for callback
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
        event = ComponentEvent(
            event_type=EventType.TASK_STARTED,
            run_id=run_id,
            component_name=task_name,
            timestamp=datetime.datetime.utcnow(),
            message=None,
            payload=input_data,
        )
        await self.notify(event)

    async def notify_task_finished(
        self,
        run_id: str,
        task_name: str,
        output_data: Optional[dict[str, Any]] = None,
    ) -> None:
        event = ComponentEvent(
            event_type=EventType.TASK_STARTED,
            run_id=run_id,
            component_name=task_name,
            timestamp=datetime.datetime.utcnow(),
            message=None,
            payload=output_data,
        )
        await self.notify(event)
