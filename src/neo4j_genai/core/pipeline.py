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

from __future__ import annotations

import asyncio
import enum
import logging
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

from pydantic import BaseModel

from neo4j_genai.core.component import Component
from neo4j_genai.core.graph import Graph, Node
from neo4j_genai.core.stores import InMemoryStore, Store
from neo4j_genai.core.types import ComponentDef, ConnectionDef, PipelineDef

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


class MissingDependencyError(Exception):
    pass


class StatusUpdateError(Exception):
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


class TaskNode(Node):
    """Runnable node. It must have:
    - a name (unique within the pipeline)
    - a component instance
    - a reference to the pipline it belongs to
        (to find dependent tasks)
    """

    def __init__(self, name: str, component: Component, pipeline: "Pipeline"):
        """TaskNode is a graph node with a run method.

        Args:
            name (str): node's name
            component (Component): component instance
            pipeline (Pipeline): pipeline the task belongs to
        """
        super().__init__(name, {})
        self.component = component
        self.status = RunStatus.UNKNOWN
        self._pipeline = pipeline
        self._lock = asyncio.Lock()
        """This lock is used to make sure we're not trying
        to update the status in //. This should prevent the task to
        be executed multiple times because the status was not known
        by the orchestrator.
        """

    async def set_status(self, status: RunStatus) -> None:
        """Set a new status

        Args:
            status (RunStatus): new status

        Raises:
            StatusUpdateError if the new status is not
                compatible with the current one.
        """
        async with self._lock:
            if status == self.status:
                raise StatusUpdateError()
            if status == RunStatus.RUNNING and self.status == RunStatus.DONE:
                # can't go back to RUNNING from DONE
                raise StatusUpdateError()
            self.status = status

    async def read_status(self) -> RunStatus:
        async with self._lock:
            return self.status

    async def execute(self, **kwargs: Any) -> RunResult | None:
        """Execute the task:
        1. Set status to RUNNING
        2. Calls the component.run method
        3. Set status to DONE

        Returns:
            RunResult | None: RunResult with status and result dict
            if the task run successfully, None if the status update
            was unsuccessful.
        """
        logger.debug(f"Running component {self.name} with {kwargs}")
        try:
            await self.set_status(RunStatus.RUNNING)
        except StatusUpdateError:
            logger.info(f"Component {self.name} already running or done {self.status}")
            return None
        res = await self.component.run(**kwargs)
        await self.set_status(RunStatus.DONE)
        return RunResult(
            status=self.status,
            result=res,
        )

    async def get_input_defs_from_parents(self) -> dict[str, str]:
        """Build input definition for this component. For this,
        the method needs the input defs defined in the edges
        between this task and its parents.

        Returns:
            dict: a dict of
                {input_parameter: path_to_value_in_the_pipeline_results_dict}

        Raises:
            MissingDependencyError if a parent dependency is not done yet
        """
        input_defs: dict[str, Any] = {}
        # make sure dependencies are satisfied
        # and save the inputs defs that needs to be propagated from parent components
        for prev_edge in self._pipeline.previous_edges(self):
            prev_status = await prev_edge.start.read_status()  # type: ignore
            if prev_status != RunStatus.DONE:
                logger.critical(f"Missing dependency {prev_edge.start.name}")
                raise MissingDependencyError(f"{prev_edge.start.name} not ready")
            if prev_edge.data:
                prev_edge_data = prev_edge.data.get("input_defs") or {}
                input_defs.update(**prev_edge_data)
        return input_defs

    def reinitialize(self) -> None:
        self.status = RunStatus.SCHEDULED

    async def run(
        self, data: dict[str, Any], callback: Callable[[Any, Any, Any], Awaitable[Any]]
    ) -> None:
        """Main method to execute the task.
        1. Get the input defs (path to all required inputs)
        2. Build the input dict from the previous input defs
        3. Call the execute method
        4. Call the pipeline callback method to deal with child dependencies
        """
        logger.debug(f"TASK START {self.name=} {data=}")
        # prepare the inputs defs that needs to be
        # propagated from parent components
        input_defs = await self.get_input_defs_from_parents()
        # create component's input based on initial input data,
        # already done component's outputs and inputs definition mapping
        inputs = self._pipeline.get_component_inputs(self.name, input_defs, data)
        # execute component task
        res = await self.execute(**inputs)
        if res is None:
            return
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

    async def run_task(self, task: TaskNode, data: dict[str, Any]) -> None:
        await task.run(data, callback=self.on_task_complete)

    async def on_task_complete(
        self, task: TaskNode, data: dict[str, Any], res: RunResult
    ) -> None:
        """When a given task is complete, it will call this method
        to find the next tasks to run.
        """
        # first call the method for the pipeline
        # this is where the results can be saved
        self.pipeline.on_task_complete(task, res)
        # then get the next tasks to be executed
        # and run them in //
        await asyncio.gather(*[self.run_task(n, data) async for n in self.next(task)])

    async def check_dependencies_complete(self, task: TaskNode) -> None:
        """Check that all parent tasks are complete.

        Raises:
            MissingDependencyError if a parent task's status is different from DONE.
        """
        dependencies = self.pipeline.previous_edges(task)
        for d in dependencies:
            d_status = await d.start.read_status()  # type: ignore
            if d_status != RunStatus.DONE:
                logger.warning(
                    f"Missing dependency {d.start.name} for {task.name} (status: {d_status})"
                )
                raise MissingDependencyError()

    async def next(self, task: TaskNode) -> AsyncGenerator[TaskNode, None]:
        """Find the next tasks to be excuted after `task` is complete.

        1. Find the task children
        2. Check each child's status:
            - if it's already running or done, do not need to run it again
            - otherwise, check that all its dependencies are met, if yes
                add this task to the list of next tasks to be executed
        """
        possible_nexts = self.pipeline.next_edges(task)
        for next_edge in possible_nexts:
            next_node = next_edge.end
            # check status
            next_node_status = await next_node.read_status()  # type: ignore
            if next_node_status in [RunStatus.RUNNING, RunStatus.DONE]:
                # already running
                continue
            # check deps
            try:
                await self.check_dependencies_complete(next_node)  # type: ignore
            except MissingDependencyError:
                continue
            yield next_node  # type: ignore
        return

    async def run(self, data: dict[str, Any]) -> None:
        """Run the pipline, starting from the root nodes
        (node without any parent). Then the callback on_task_complete
        will handle the task dependencies.
        """
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

    @classmethod
    def from_template(
        cls, pipeline_template: PipelineDef, store: Optional[Store] = None
    ) -> Pipeline:
        """Create a Pipeline from a pydantic model defining the components and their connections"""
        pipeline = Pipeline(store=store)
        for component in pipeline_template.components:
            pipeline.add_component(component.name, component.component)
        for edge in pipeline_template.connections:
            pipeline.connect(edge.start, edge.end, edge.input_defs)
        return pipeline

    def show_as_dict(self) -> dict[str, Any]:
        component_defs = []
        for name, task in self._nodes.items():
            component_defs.append(
                ComponentDef(name=name, component=task.component)  # type: ignore
            )
        connection_defs = []
        for edge in self._edges:
            connection_defs.append(
                ConnectionDef(
                    start=edge.start.name,
                    end=edge.end.name,
                    input_defs=edge.data["input_defs"] if edge.data else {},
                )
            )
        pipeline_def = PipelineDef(
            components=component_defs, connections=connection_defs
        )
        return pipeline_def.model_dump()

    def add_component(self, name: str, component: Component) -> None:
        task = TaskNode(name, component, self)
        self.add_node(task)

    def set_component(self, name: str, component: Component) -> None:
        task = TaskNode(name, component, self)
        self.set_node(task)

    def connect(  # type: ignore
        self,
        start_component_name: str,
        end_component_name: str,
        input_defs: Optional[dict[str, str]] = None,
    ) -> None:
        start_node = self.get_node_by_name(start_component_name, raise_exception=True)
        end_node = self.get_node_by_name(end_component_name, raise_exception=True)
        super().connect(start_node, end_node, data={"input_defs": input_defs})
        if self.is_cyclic():
            raise Exception("Cyclic graph")

    def on_task_complete(self, node: TaskNode, result: RunResult) -> None:
        self.add_result_for_component(node.name, result.result, is_final=node.is_leaf())

    def add_result_for_component(
        self, name: str, result: dict[str, Any] | None, is_final: bool = False
    ) -> None:
        self._store.add(name, result)
        if is_final:
            # The pipeline only returns the results
            # of the leaf nodes
            # TODO: make this configurable in the future.
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

    def reinitialize(self) -> None:
        """Reinitialize the result stores and component status
        if we want to rerun the same pipeline again
        (maybe with inputs)"""
        self._store.empty()
        self._final_results.empty()
        for task in self._nodes.values():
            task.reinitialize()  # type: ignore

    async def run(self, data: dict[str, Any]) -> dict[str, Any]:
        self.reinitialize()
        orchestrator = Orchestrator(self)
        await orchestrator.run(data)
        return self._final_results.all()
