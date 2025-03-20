import asyncio

from neo4j_graphrag.experimental.pipeline.component import Component, DataModel
from neo4j_graphrag.experimental.pipeline.pipeline import Pipeline
from neo4j_graphrag.experimental.pipeline.notification import EventType, Event
from neo4j_graphrag.experimental.pipeline.types.context import RunContext


# Define some example components with progress notifications
class OutputModel(DataModel):
    result: int


class SlowAdder(Component):
    """A component that slowly adds numbers and reports progress"""

    def __init__(self, number: int) -> None:
        self.number = number

    async def run_with_context(self, context_: RunContext, value: int) -> OutputModel:
        # Simulate work with progress updates
        for i in range(value):
            await asyncio.sleep(0.5)  # Simulate work
            await context_.notify(
                message=f"Added {i+1}/{value}", data={"current": i + 1, "total": value}
            )
        return OutputModel(result=value + self.number)


class SlowMultiplier(Component):
    """A component that slowly multiplies numbers and reports progress"""

    def __init__(self, multiplier: int) -> None:
        self.multiplier = multiplier

    async def run_with_context(self, context_: RunContext, value: int) -> OutputModel:
        # Simulate work with progress updates
        for i in range(3):  # Always do 3 steps
            await asyncio.sleep(0.7)  # Simulate work
            await context_.notify(
                message=f"Multiplication step {i+1}/3", data={"step": i + 1, "total": 3}
            )
        return OutputModel(result=value * self.multiplier)


async def callback(event: Event) -> None:
    await asyncio.sleep(0.1)


async def main() -> None:
    # Create pipeline
    pipeline = Pipeline(callback=callback)

    # Add components
    pipeline.add_component(SlowAdder(number=3), "adder")
    pipeline.add_component(SlowMultiplier(multiplier=2), "multiplier")

    # Connect components
    pipeline.connect("adder", "multiplier", {"value": "adder.result"})

    print("\n=== Running pipeline with streaming ===")
    # Run pipeline with streaming - see events as they happen
    async for event in pipeline.stream({"adder": {"value": 2}}):
        if event.event_type == EventType.PIPELINE_STARTED:
            print("Stream: Pipeline started!")
        elif event.event_type == EventType.PIPELINE_FINISHED:
            print(f"Stream: Pipeline finished! Final results: {event.payload}")
        elif event.event_type == EventType.TASK_STARTED:
            print(
                f"Stream: Task {event.task_name} started with inputs: {event.payload}"  # type: ignore
            )
        elif event.event_type == EventType.TASK_PROGRESS:
            print(f"Stream: Task {event.task_name} progress - {event.message}")  # type: ignore
        elif event.event_type == EventType.TASK_FINISHED:
            print(
                f"Stream: Task {event.task_name} finished with result: {event.payload}"  # type: ignore
            )


if __name__ == "__main__":
    asyncio.run(main())
