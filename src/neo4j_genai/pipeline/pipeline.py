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
from datetime import datetime
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

from pydantic import BaseModel, Field

from neo4j_genai.core.pipeline_graph import PipelineGraph, PipelineNode
from neo4j_genai.core.stores import InMemoryStore, Store
from neo4j_genai.pipeline.component import Component, DataModel
from neo4j_genai.pipeline.types import ComponentDef, ConnectionDef, PipelineDef

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


class MissingDependencyError(Exception):
    pass


class StatusUpdateError(Exception):
    pass


class PipelineDefinitionError(Exception):
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
    result: Optional[DataModel] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TaskPipelineNode(PipelineNode):
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
        component_result = await self.component.run(**kwargs)
        await self.set_status(RunStatus.DONE)
        run_result = RunResult(
            status=self.status,
            result=component_result,
        )
        return run_result

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
        for prev_edge in self._pipeline.previous_edges(self.name):
            prev_node = self._pipeline.get_node_by_name(prev_edge.start)
            prev_status = await prev_node.read_status()  # type: ignore
            if prev_status != RunStatus.DONE:
                logger.critical(f"Missing dependency {prev_edge.start}")
                raise MissingDependencyError(f"{prev_edge.start} not ready")
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

    async def run_task(self, task: TaskPipelineNode, data: dict[str, Any]) -> None:
        await task.run(data, callback=self.on_task_complete)

    async def on_task_complete(
        self, task: TaskPipelineNode, data: dict[str, Any], res: RunResult
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

    async def check_dependencies_complete(self, task: TaskPipelineNode) -> None:
        """Check that all parent tasks are complete.

        Raises:
            MissingDependencyError if a parent task's status is different from DONE.
        """
        dependencies = self.pipeline.previous_edges(task.name)
        for d in dependencies:
            start_node = self.pipeline.get_node_by_name(d.start)
            d_status = await start_node.read_status()  # type: ignore
            if d_status != RunStatus.DONE:
                logger.warning(
                    f"Missing dependency {d.start} for {task.name} (status: {d_status})"
                )
                raise MissingDependencyError()

    async def next(
        self, task: TaskPipelineNode
    ) -> AsyncGenerator[TaskPipelineNode, None]:
        """Find the next tasks to be excuted after `task` is complete.

        1. Find the task children
        2. Check each child's status:
            - if it's already running or done, do not need to run it again
            - otherwise, check that all its dependencies are met, if yes
                add this task to the list of next tasks to be executed
        """
        possible_nexts = self.pipeline.next_edges(task.name)
        for next_edge in possible_nexts:
            next_node = self.pipeline.get_node_by_name(next_edge.end)
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


class Pipeline(PipelineGraph):
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
                    start=edge.start,
                    end=edge.end,
                    input_defs=edge.data["input_defs"] if edge.data else {},
                )
            )
        pipeline_def = PipelineDef(
            components=component_defs, connections=connection_defs
        )
        return pipeline_def.model_dump()

    def add_component(self, name: str, component: Component) -> None:
        task = TaskPipelineNode(name, component, self)
        self.add_node(task)

    def set_component(self, name: str, component: Component) -> None:
        task = TaskPipelineNode(name, component, self)
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
            raise PipelineDefinitionError("Cyclic graph are not allowed")

    def on_task_complete(self, node: TaskPipelineNode, result: RunResult) -> None:
        res_to_save = None
        if result.result:
            res_to_save = result.result.model_dump()
        self.add_result_for_component(node.name, res_to_save, is_final=node.is_leaf())

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

    def validate_inputs_definition(self, data: dict[str, Any]) -> None:
        """Go through the graph and make sure each component will not miss any input"""
        for task in self._nodes.values():
            component = task.component  # type: ignore
            expected_mandatory_inputs = [
                param_name
                for param_name, config in component.component_inputs.items()
                if config["has_default"] is False
            ]
            # build the input def from previous edges
            prev_edges = self.previous_edges(task.name)
            # start building the actual input list, starting
            # from the inputs provided in the pipeline.run method
            actual_inputs = list(data.get(task.name, {}).keys())
            # then, iterate over all parents to find the parameter propagation
            for edge in prev_edges:
                edge_data = edge.data or {}
                edge_inputs = edge_data.get("input_defs") or {}
                # check that the previous component is actually returning
                # the mapped parameter
                for param, path in edge_inputs.items():
                    source_component_name, param_name = path.split(".")
                    source_node = self.get_node_by_name(source_component_name)
                    source_component = source_node.component  # type: ignore
                    source_component_outputs = source_component.component_outputs
                    if param_name not in source_component_outputs:
                        raise PipelineDefinitionError(
                            f"Parameter {param_name} is not valid output for "
                            f"{source_component_name} (must be one of "
                            f"{list(source_component_outputs.keys())})"
                        )

                actual_inputs.extend(list(edge_inputs.keys()))
            if set(expected_mandatory_inputs) - set(actual_inputs):
                raise PipelineDefinitionError(
                    f"Missing input parameters for {task.name}: "
                    f"Expected parameters: {expected_mandatory_inputs}. "
                    f"Got: {actual_inputs}"
                )

    async def run(self, data: dict[str, Any]) -> dict[str, Any]:
        self.validate_inputs_definition(data)
        self.reinitialize()
        orchestrator = Orchestrator(self)
        await orchestrator.run(data)
        return self._final_results.all()
