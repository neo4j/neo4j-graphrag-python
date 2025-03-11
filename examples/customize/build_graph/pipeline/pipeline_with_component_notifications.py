"""This example demonstrates how to use event callback to receive notifications
about the component progress.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from neo4j_graphrag.experimental.pipeline import Pipeline, Component, DataModel
from neo4j_graphrag.experimental.pipeline.notification import Event, EventType
from neo4j_graphrag.experimental.pipeline.types.context import RunContext

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class BatchComponentResult(DataModel):
    result: list[int]


class MultiplicationComponent(Component):
    def __init__(self, f: int) -> None:
        self.f = f

    async def run(self, numbers: list[int]) -> BatchComponentResult:
        return BatchComponentResult(result=[])

    async def multiply_number(
        self, context_: RunContext, number: int,
    ) -> int:
        await context_.notify(
            message=f"Processing number {number}",
            data={"number_processed": number},
        )
        return self.f * number

    async def run_with_context(
        self,
        context_: RunContext,
        numbers: list[int],
        **kwargs: Any,
    ) -> BatchComponentResult:
        result = await asyncio.gather(
            *[
                self.multiply_number(
                    context_,
                    number,
                )
                for number in numbers
            ]
        )
        return BatchComponentResult(result=result)


async def event_handler(event: Event) -> None:
    """Function can do anything about the event,
    here we're just logging it if it's a pipeline-level event.
    """
    if event.event_type == EventType.TASK_PROGRESS:
        logger.warning(event)
    else:
        logger.info(event)


async def main() -> None:
    """ """
    pipe = Pipeline(
        callback=event_handler,
    )
    # define the components
    pipe.add_component(
        MultiplicationComponent(f=2),
        "multiply_by_2",
    )
    pipe.add_component(
        MultiplicationComponent(f=10),
        "multiply_by_10",
    )
    # define the execution order of component
    # and how the output of previous components must be used
    pipe.connect(
        "multiply_by_2",
        "multiply_by_10",
        input_config={"numbers": "multiply_by_2.result"},
    )
    # user input:
    pipe_inputs_1 = {
        "multiply_by_2": {
            "numbers": [1, 2, 5, 4],
        },
    }
    pipe_inputs_2 = {
        "multiply_by_2": {
            "numbers": [3, 10, 1],
        }
    }
    # run the pipeline
    await asyncio.gather(
        pipe.run(pipe_inputs_1),
        pipe.run(pipe_inputs_2),
    )


if __name__ == "__main__":
    asyncio.run(main())
