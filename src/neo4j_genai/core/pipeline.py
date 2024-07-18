"""
Pipeline implementation.

Features:
- Sync/Async exec
- Branching:
      |
    /   \
e.g. DocumentChunker => ERExtractor, Embedder

- Aggregation:
    \  /
     |
e.g. SchemaBuilder + Chunker => ERExtractor
"""

import enum
import logging
import asyncio
from typing import Any, Optional, Callable, Awaitable, Generator

from pydantic import BaseModel

from neo4j_genai.core.stores import Store, InMemoryStore
from neo4j_genai.core.graph import Graph, Node


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


class MissingDependencyError(Exception):
    pass


class RunStatus(enum.Enum):
    UNKNOWN = "UNKNOWN"
    SCHEDULED = "SCHEDULED"
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    SKIP = "SKIP"
    DONE = "DONE"


class RunResult(BaseModel):
    status: RunStatus
    result: Optional[dict[str, Any]] = None


class Component:
    async def process(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}


class TaskNode(Node):
    def __init__(self, name: str, component: Component, pipeline: "Pipeline"):
        super().__init__(name, {})
        self.component = component
        self.status = RunStatus.UNKNOWN
        self._pipeline = pipeline

    async def execute(self, **kwargs: Any) -> RunResult:
        logger.debug(f"Running component {self.name} with {kwargs}")
        self.status = RunStatus.RUNNING
        res = await self.component.process(**kwargs)
        self.status = RunStatus.DONE
        return RunResult(
            status=self.status,
            result=res,
        )

    def get_input_defs_from_parents(self) -> dict[str, str]:
        input_defs: dict[str, Any] = {}
        # make sure dependencies are satisfied
        # and save the inputs defs that needs to be propagated from parent components
        for prev_edge in self._pipeline.previous_edges(self):
            if prev_edge.start.status != RunStatus.DONE:  # type: ignore
                logger.critical(f"Missing dependency {prev_edge.start.name}")
                raise MissingDependencyError(f"{prev_edge.start.name} not ready")
            if prev_edge.data:
                prev_edge_data = prev_edge.data.get("input_defs") or {}
                input_defs.update(**prev_edge_data)
        return input_defs

    async def run(
        self, data: dict[str, Any], callback: Callable[[Any, Any, Any], Awaitable[Any]]
    ) -> None:
        logger.debug(f"TASK START {self.name=} {data=}")
        # prepare the inputs defs that needs to be
        # propagated from parent components
        input_defs = self.get_input_defs_from_parents()
        # create component's input based on initial input data,
        # already done component's outputs and inputs definition mapping
        inputs = self._pipeline.get_component_inputs(self.name, input_defs, data)
        # execute component task
        res = await self.execute(**inputs)
        logger.debug(f"TASK RESULT {self.name=} {res=}")
        # save its results
        await callback(self, data, res)


class Orchestrator:
    """Orchestrate a pipeline
    Once a TaskNode is done, it calls the `on_task_complete` callback
    that will save the results, find the next tasks to be executed
    (checking that all dependencies are met), and run them.
    """

    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def save_results(self, node: Node, res: RunResult) -> None:
        self.pipeline.add_result_for_component(
            node.name, res.result or {}, is_final=node.is_leaf()
        )

    async def run_node(self, node: TaskNode, data: dict[str, Any]) -> None:
        await node.run(data, callback=self.on_task_complete)

    async def on_task_complete(
        self, node: TaskNode, data: dict[str, Any], res: RunResult
    ) -> None:
        self.save_results(node, res)
        await asyncio.gather(*[self.run_node(n, data) for n in self.next(node)])

    def check_dependencies_complete(self, node: TaskNode) -> None:
        dependencies = self.pipeline.previous_edges(node)
        for d in dependencies:
            if d.start.status != RunStatus.DONE:  # type: ignore
                logger.warning(
                    f"Missing dependency {d.start.name} for {node.name} (status: {d.start.status})"  # type: ignore
                )
                raise MissingDependencyError()

    def next(self, node: TaskNode) -> Generator[TaskNode, None, None]:
        possible_nexts = self.pipeline.next_edges(node)
        for next_edge in possible_nexts:
            next_node = next_edge.end
            # check status
            if next_node.status in [RunStatus.RUNNING, RunStatus.DONE]:  # type: ignore
                # already running
                continue
            # check deps
            try:
                self.check_dependencies_complete(next_node)  # type: ignore
            except MissingDependencyError:
                continue
            yield next_node  # type: ignore

    async def run(self, data: dict[str, Any]) -> None:
        logger.debug(f"PIPELINE START {data=}")
        tasks = [
            root.run(data, callback=self.on_task_complete)  # type: ignore
            for root in self.pipeline.roots()
        ]
        await asyncio.gather(*tasks)


class Pipeline(Graph):
    """This is our pipeline, when we configure components
    and their execution order."""

    def __init__(self, store: Optional[Store] = None) -> None:
        super().__init__()
        self._store = store or InMemoryStore()
        self._final_results = InMemoryStore()
        self.orchestrator = Orchestrator(self)

    def add_component(self, name: str, component: Component) -> None:
        task = TaskNode(name, component, self)
        self.add_node(task)

    def connect(  # type: ignore
        self,
        start_component_name: str,
        end_component_name: str,
        input_defs: Optional[dict[str, str]] = None,
    ) -> None:
        start_node = self.get_node_by_name(start_component_name, raise_exception=True)
        end_node = self.get_node_by_name(end_component_name, raise_exception=True)
        super().connect(start_node, end_node, data={"input_defs": input_defs})

    def add_result_for_component(
        self, name: str, result: dict[str, Any], is_final: bool = False
    ) -> None:
        self._store.add(name, result)
        if is_final:
            self._final_results.add(name, result)

    def get_results_for_component(self, name: str) -> Any:
        return self._store.get(name)

    def get_component_inputs(
        self,
        component_name: str,
        input_defs: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Find the component inputs from:
        - data: the user input data
        - input_defs: the mapping between components results and inputs
        """
        component_inputs: dict[str, Any] = input_data.get(component_name, {})
        if input_defs:
            for input_def, mapping in input_defs.items():
                value = self._store.find_all(mapping)
                if value:  # TODO: how to deal with multiple matches? Is it relevant?
                    value = value[0]
                component_inputs[input_def] = value
        return component_inputs

    async def run(self, data: dict[str, Any]) -> dict[str, Any]:
        await self.orchestrator.run(data)
        return self._final_results.all()
