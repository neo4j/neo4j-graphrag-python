import asyncio
import itertools
import logging
from typing import Any

from neo4j_graphrag.experimental.pipeline import Component, DataModel, Pipeline
from neo4j_graphrag.experimental.pipeline.observer import (
    Event,
    EventType,
    ObserverInterface,
)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.WARNING)


class MyObserver(ObserverInterface):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    async def observe(self, event: Event) -> None:
        logger.warning(
            "Something is happening in session %s: %s", self.session_id, event
        )


class BatchComponentOutput(DataModel):
    result: list[Any]


class BatchComponent(Component):
    def __init__(self, batch_size: int, suffix_name: str = "suffix") -> None:
        super().__init__()
        self.batch_size = batch_size
        self.suffix_name = suffix_name

    async def process_batch(self, batch: list[Any], batch_number: int) -> list[Any]:
        await self.notify(
            Event(
                event_type=EventType.TASK_PROGRESS,
                sender=self,
                message="Component batch processing started for batch %s"
                % batch_number,
            )
        )
        for row in batch:
            row["processed"] = True
            row[self.suffix_name] = self.__class__.__name__
        return batch

    @staticmethod
    def _flatten_list(lst: list[list[Any]]) -> list[Any]:
        return list(itertools.chain.from_iterable(lst))

    async def run(self, array: list[Any]) -> BatchComponentOutput:
        num_iterations = (len(array) - 1) // self.batch_size + 1
        batch_tasks = []
        for iteration in range(num_iterations):
            prev_index = iteration * self.batch_size
            next_index = (iteration + 1) * self.batch_size
            batch_data = array[prev_index:next_index]
            batch_tasks.append(self.process_batch(batch_data, batch_number=iteration))
        results = await asyncio.gather(*batch_tasks)
        return BatchComponentOutput(
            result=self._flatten_list(results),
        )


async def main() -> None:
    observer = MyObserver(session_id="SESSION XYZ")
    pipeline = Pipeline(
        observers=[observer],
    )
    pipeline.add_component(BatchComponent(batch_size=3), "batch")
    pipeline.add_component(BatchComponent(batch_size=3, suffix_name="batch2"), "batch2")
    pipeline.connect("batch", "batch2", {"array": "batch.result"})
    await pipeline.run(
        {
            "batch": {
                "array": [
                    {
                        "id": k,
                    }
                    for k in range(10)
                ]
            }
        }
    )
    # print(res)


if __name__ == "__main__":
    asyncio.run(main())
