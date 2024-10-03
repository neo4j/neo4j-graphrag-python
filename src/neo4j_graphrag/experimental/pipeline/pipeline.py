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

import asyncio
import enum
import logging
import uuid
import warnings
from collections import defaultdict
from datetime import datetime
from timeit import default_timer
from typing import Any, AsyncGenerator, Optional

try:
    import pygraphviz as pgv
except ImportError:
    pgv = None

from pydantic import BaseModel, Field

from neo4j_graphrag.experimental.pipeline.component import Component, DataModel
from neo4j_graphrag.experimental.pipeline.exceptions import (
    PipelineDefinitionError,
    PipelineMissingDependencyError,
    PipelineStatusUpdateError,
)
from neo4j_graphrag.experimental.pipeline.pipeline_graph import (
    PipelineEdge,
    PipelineGraph,
    PipelineNode,
)
from neo4j_graphrag.experimental.pipeline.stores import InMemoryStore, ResultStore
from neo4j_graphrag.experimental.pipeline.types import (
    ComponentConfig,
    ConnectionConfig,
    PipelineConfig,
)

logger = logging.getLogger(__name__)


class RunStatus(enum.Enum):
    UNKNOWN = "UNKNOWN"
    RUNNING = "RUNNING"
    DONE = "DONE"


class RunResult(BaseModel):
    status: RunStatus = RunStatus.DONE
    result: Optional[DataModel] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TaskPipelineNode(PipelineNode):
    """Runnable node. It must have:
    - a name (unique within the pipeline)
    - a component instance
    """

    def __init__(self, name: str, component: Component):
        """TaskPipelineNode is a graph node with a run method.

        Args:
            name (str): node's name
            component (Component): component instance
        """
        super().__init__(name, {})
        self.component = component

    async def execute(self, **kwargs: Any) -> RunResult | None:
        """Execute the task

        Returns:
            RunResult | None: RunResult with status and result dict
            if the task run successfully, None if the status update
            was unsuccessful.
        """
        logger.debug(f"Running component {self.name} with {kwargs}")
        start_time = default_timer()
        component_result = await self.component.run(**kwargs)
        run_result = RunResult(
            result=component_result,
        )
        end_time = default_timer()
        logger.debug(f"Component {self.name} finished in {end_time - start_time}s")
        return run_result

    async def run(self, inputs: dict[str, Any]) -> RunResult | None:
        """Main method to execute the task."""
        logger.debug(f"TASK START {self.name=} {inputs=}")
        res = await self.execute(**inputs)
        logger.debug(f"TASK RESULT {self.name=} {res=}")
        return res


class Orchestrator:
    """Orchestrate a pipeline.

    The orchestrator is responsible for:
    - finding the next tasks to execute
    - building the inputs for each task
    - calling the run method on each task

    Once a TaskNode is done, it calls the `on_task_complete` callback
    that will save the results, find the next tasks to be executed
    (checking that all dependencies are met), and run them.
    """

    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline
        self.run_id = str(uuid.uuid4())

    async def run_task(self, task: TaskPipelineNode, data: dict[str, Any]) -> None:
        """Get inputs and run a specific task. Once the task is done,
        calls the on_task_complete method.

        Args:
            task (TaskPipelineNode): The task to be run
            data (dict[str, Any]): The pipeline input data

        Returns:
            None
        """
        param_mapping = self.get_input_config_for_task(task)
        inputs = await self.get_component_inputs(task.name, param_mapping, data)
        try:
            await self.set_task_status(task.name, RunStatus.RUNNING)
        except PipelineStatusUpdateError:
            logger.info(f"Component {task.name} already running or done")
            return None
        res = await task.run(inputs)
        await self.set_task_status(task.name, RunStatus.DONE)
        if res:
            await self.on_task_complete(data=data, task=task, result=res)

    async def set_task_status(self, task_name: str, status: RunStatus) -> None:
        """Set a new status

        Args:
            task_name (str): Name of the component
            status (RunStatus): New status

        Raises:
            PipelineStatusUpdateError if the new status is not
                compatible with the current one.
        """
        # prevent the method from being called by two concurrent async calls
        async with asyncio.Lock():
            current_status = await self.get_status_for_component(task_name)
            if status == current_status:
                raise PipelineStatusUpdateError(f"Status is already '{status}'")
            if status == RunStatus.RUNNING and current_status == RunStatus.DONE:
                raise PipelineStatusUpdateError("Can't go from DONE to RUNNING")
            return await self.pipeline.store.add_status_for_component(
                self.run_id, task_name, status.value
            )

    async def on_task_complete(
        self, data: dict[str, Any], task: TaskPipelineNode, result: RunResult
    ) -> None:
        """When a given task is complete, it will call this method
        to find the next tasks to run.
        """
        # first save this component results
        res_to_save = None
        if result.result:
            res_to_save = result.result.model_dump()
        await self.add_result_for_component(
            task.name, res_to_save, is_final=task.is_leaf()
        )
        # then get the next tasks to be executed
        # and run them in //
        await asyncio.gather(*[self.run_task(n, data) async for n in self.next(task)])

    async def check_dependencies_complete(self, task: TaskPipelineNode) -> None:
        """Check that all parent tasks are complete.

        Raises:
            MissingDependencyError if a parent task's status is not DONE.
        """
        dependencies = self.pipeline.previous_edges(task.name)
        for d in dependencies:
            d_status = await self.get_status_for_component(d.start)
            if d_status != RunStatus.DONE:
                logger.debug(
                    f"Missing dependency {d.start} for {task.name} (status: {d_status}). "
                    "Will try again when dependency is complete."
                )
                raise PipelineMissingDependencyError()

    async def next(
        self, task: TaskPipelineNode
    ) -> AsyncGenerator[TaskPipelineNode, None]:
        """Find the next tasks to be executed after `task` is complete.

        1. Find the task children
        2. Check each child's status:
            - if it's already running or done, do not need to run it again
            - otherwise, check that all its dependencies are met, if yes
                add this task to the list of next tasks to be executed
        """
        possible_next = self.pipeline.next_edges(task.name)
        for next_edge in possible_next:
            next_node = self.pipeline.get_node_by_name(next_edge.end)
            # check status
            next_node_status = await self.get_status_for_component(next_node.name)
            if next_node_status in [RunStatus.RUNNING, RunStatus.DONE]:
                # already running
                continue
            # check deps
            try:
                await self.check_dependencies_complete(next_node)
            except PipelineMissingDependencyError:
                continue
            yield next_node
        return

    def get_input_config_for_task(
        self, task: TaskPipelineNode
    ) -> dict[str, dict[str, str]]:
        """Build input definition for a given task.,
        The method needs to access the input defs defined in the edges
        between this task and its parents.

        Args:
            task (TaskPipelineNode): the task to get the input config for

        Returns:
            dict: a dict of
                {input_parameter: {source_component_name: "", param_name: ""}}
        """
        if not self.pipeline.is_validated:
            raise PipelineDefinitionError(
                "You must validate the pipeline input config first. Call `pipeline.validate_parameter_mapping()`"
            )
        return self.pipeline.param_mapping.get(task.name) or {}

    async def get_component_inputs(
        self,
        component_name: str,
        param_mapping: dict[str, dict[str, str]],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Find the component inputs from:
        - input_config: the mapping between components results and inputs
            (results are stored in the pipeline result store)
        - input_data: the user input data

        Args:
            component_name (str): the component/task name
            param_mapping (dict[str, dict[str, str]]): the input config
            input_data (dict[str, Any]): the pipeline input data (user input)
        """
        component_inputs: dict[str, Any] = input_data.get(component_name, {})
        if param_mapping:
            for parameter, mapping in param_mapping.items():
                component = mapping["component"]
                output_param = mapping.get("param")
                component_result = await self.get_results_for_component(component)
                if output_param is not None:
                    value = component_result.get(output_param)
                else:
                    value = component_result
                if parameter in component_inputs:
                    m = f"{component}.{parameter}" if parameter else component
                    warnings.warn(
                        f"In component '{component_name}', parameter '{parameter}' from user input will be ignored and replaced by '{m}'"
                    )
                component_inputs[parameter] = value
        return component_inputs

    async def add_result_for_component(
        self, name: str, result: dict[str, Any] | None, is_final: bool = False
    ) -> None:
        """This is where we save the results in the result store and, optionally,
        in the final result store.
        """
        await self.pipeline.store.add_result_for_component(self.run_id, name, result)
        if is_final:
            # The pipeline only returns the results
            # of the leaf nodes
            # TODO: make this configurable in the future.
            existing_results = await self.pipeline.final_results.get(self.run_id) or {}
            existing_results[name] = result
            await self.pipeline.final_results.add(
                self.run_id, existing_results, overwrite=True
            )

    async def get_results_for_component(self, name: str) -> Any:
        return await self.pipeline.store.get_result_for_component(self.run_id, name)

    async def get_status_for_component(self, name: str) -> RunStatus:
        status = await self.pipeline.store.get_status_for_component(self.run_id, name)
        if status is None:
            return RunStatus.UNKNOWN
        return RunStatus(status)

    async def run(self, data: dict[str, Any]) -> None:
        """Run the pipline, starting from the root nodes
        (node without any parent). Then the callback on_task_complete
        will handle the task dependencies.
        """
        logger.debug(f"PIPELINE START {data=}")
        tasks = [self.run_task(root, data) for root in self.pipeline.roots()]
        await asyncio.gather(*tasks)


class PipelineResult(BaseModel):
    run_id: str
    result: Any


class Pipeline(PipelineGraph[TaskPipelineNode, PipelineEdge]):
    """This is the main pipeline, where components
    and their execution order are defined"""

    def __init__(self, store: Optional[ResultStore] = None) -> None:
        super().__init__()
        self.store = store or InMemoryStore()
        self.final_results = InMemoryStore()
        self.is_validated = False
        self.param_mapping: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        """
        Dict structure:
        { component_name : {
                param_name: {
                    component: "",  # source component name
                    param_name: "",
                }
            }
        }
        """
        self.missing_inputs: dict[str, list[str]] = defaultdict()

    @classmethod
    def from_template(
        cls, pipeline_template: PipelineConfig, store: Optional[ResultStore] = None
    ) -> Pipeline:
        """Create a Pipeline from a pydantic model defining the components and their connections"""
        pipeline = Pipeline(store=store)
        for component in pipeline_template.components:
            pipeline.add_component(
                component.component,
                component.name,
            )
        for edge in pipeline_template.connections:
            pipeline_edge = PipelineEdge(
                edge.start, edge.end, data={"input_config": edge.input_config}
            )
            pipeline.add_edge(pipeline_edge)
        return pipeline

    def show_as_dict(self) -> dict[str, Any]:
        component_config = []
        for name, task in self._nodes.items():
            component_config.append(
                ComponentConfig(name=name, component=task.component)
            )
        connection_config = []
        for edge in self._edges:
            connection_config.append(
                ConnectionConfig(
                    start=edge.start,
                    end=edge.end,
                    input_config=edge.data["input_config"] if edge.data else {},
                )
            )
        pipeline_config = PipelineConfig(
            components=component_config, connections=connection_config
        )
        return pipeline_config.model_dump()

    def draw(
        self, path: str, layout: str = "dot", hide_unused_outputs: bool = True
    ) -> Any:
        G = self.get_pygraphviz_graph(hide_unused_outputs)
        G.layout(layout)
        G.draw(path)

    def get_pygraphviz_graph(self, hide_unused_outputs: bool = True) -> pgv.AGraph:
        if pgv is None:
            raise ImportError(
                "Could not import pygraphviz. "
                "Follow installation instruction in pygraphviz documentation "
                "to get it up and running on your system."
            )
        self.validate_parameter_mapping()
        G = pgv.AGraph(strict=False, directed=True)
        # create a node for each component
        for n, node in self._nodes.items():
            comp_inputs = ",".join(
                f"{i}: {d['annotation']}"
                for i, d in node.component.component_inputs.items()
            )
            G.add_node(
                n,
                node_type="component",
                shape="rectangle",
                label=f"{node.component.__class__.__name__}: {n}({comp_inputs})",
            )
            # create a node for each output field and connect them it to its component
            for o in node.component.component_outputs:
                param_node_name = f"{n}.{o}"
                G.add_node(param_node_name, label=o, node_type="output")
                G.add_edge(n, param_node_name)
        # then we create the edges between a component output
        # and the component it gets added to
        for component_name, params in self.param_mapping.items():
            for param, mapping in params.items():
                source_component = mapping["component"]
                source_param_name = mapping.get("param")
                if source_param_name:
                    source_output_node = f"{source_component}.{source_param_name}"
                else:
                    source_output_node = source_component
                G.add_edge(source_output_node, component_name, label=param)
        # remove outputs that are not mapped
        if hide_unused_outputs:
            for n in G.nodes():
                if n.attr["node_type"] == "output" and G.out_degree(n) == 0:  # type: ignore
                    G.remove_node(n)
        return G

    def add_component(self, component: Component, name: str) -> None:
        """Add a new component. Components are uniquely identified
        by their name. If 'name' is already in the pipeline, a ValueError
        is raised."""
        task = TaskPipelineNode(name, component)
        self.add_node(task)
        # invalidate the pipeline if it was already validated
        self.invalidate()

    def set_component(self, name: str, component: Component) -> None:
        """Replace a component with another. If 'name' is not yet in the pipeline,
        raises ValueError.
        """
        task = TaskPipelineNode(name, component)
        self.set_node(task)
        # invalidate the pipeline if it was already validated
        self.invalidate()

    def connect(
        self,
        start_component_name: str,
        end_component_name: str,
        input_config: Optional[dict[str, str]] = None,
    ) -> None:
        """Connect one component to another.

        Args:
            start_component_name (str): name of the component as defined in
                the add_component method
            end_component_name (str): name of the component as defined in
                the add_component method
            input_config (Optional[dict[str, str]]): end component input configuration:
                propagate previous components outputs.

        Raises:
            PipelineDefinitionError: if the provided component are not in the Pipeline
                or if the graph that would be created by this connection is cyclic.
        """
        edge = PipelineEdge(
            start_component_name,
            end_component_name,
            data={"input_config": input_config},
        )
        try:
            self.add_edge(edge)
        except KeyError:
            raise PipelineDefinitionError(
                f"{start_component_name} or {end_component_name} is not in the Pipeline"
            )
        if self.is_cyclic():
            raise PipelineDefinitionError("Cyclic graph are not allowed")
        # invalidate the pipeline if it was already validated
        self.invalidate()

    def invalidate(self) -> None:
        self.is_validated = False
        self.param_mapping = defaultdict(dict)
        self.missing_inputs = defaultdict()

    def validate_parameter_mapping(self) -> None:
        """Go through the graph and make sure parameter mapping is valid
        (without considering user input yet)
        """
        if self.is_validated:
            return
        for task in self._nodes.values():
            self.validate_parameter_mapping_for_task(task)
        self.is_validated = True

    def validate_input_data(self, data: dict[str, Any]) -> bool:
        """Performs parameter and data validation before running the pipeline:
        - Check parameters defined in the connect method
        - Make sure the missing parameters are present in the input `data` dict.

        Args:
            data (dict[str, Any]): input data to use for validation
                (usually from Pipeline.run)

        Raises:
            PipelineDefinitionError if any parameter mapping is invalid or if a
                parameter is missing.
        """
        if not self.is_validated:
            self.validate_parameter_mapping()
        for task in self._nodes.values():
            if task.name not in self.param_mapping:
                self.validate_parameter_mapping_for_task(task)
            missing_params = self.missing_inputs[task.name]
            task_data = data.get(task.name) or {}
            for param in missing_params:
                if param not in task_data:
                    raise PipelineDefinitionError(
                        f"Parameter '{param}' not provided for component '{task.name}'"
                    )
        return True

    def validate_parameter_mapping_for_task(self, task: TaskPipelineNode) -> bool:
        """Make sure that all the parameter mapping for a given task are valid.
        Does not consider user input yet.

        Considering the naming {param => target (component, [output_parameter]) },
        the mapping is valid if:
         - 'param' is a valid input for task
         - 'param' has not already been mapped
         - The target component exists in the pipeline and, if specified, the
            target output parameter is a valid field in the target component's
            result model.

        This method builds the param_mapping and missing_inputs instance variables.
        """
        component = task.component
        expected_mandatory_inputs = [
            param_name
            for param_name, config in component.component_inputs.items()
            if config["has_default"] is False
        ]
        # start building the actual input list, starting
        # from the inputs provided in the pipeline.run method
        actual_inputs = []
        prev_edges = self.previous_edges(task.name)
        # then, iterate over all parents to find the parameter propagation
        for edge in prev_edges:
            edge_data = edge.data or {}
            edge_inputs = edge_data.get("input_config") or {}
            # check that the previous component is actually returning
            # the mapped parameter
            for param, path in edge_inputs.items():
                if param in self.param_mapping[task.name]:
                    raise PipelineDefinitionError(
                        f"Parameter '{param}' already mapped to {self.param_mapping[task.name][param]}"
                    )
                if param not in task.component.component_inputs:
                    raise PipelineDefinitionError(
                        f"Parameter '{param}' is not a valid input for component '{task.name}' of type '{task.component.__class__.__name__}'"
                    )
                try:
                    source_component_name, param_name = path.split(".")
                except ValueError:
                    # no specific output mapped
                    # the full source component result will be
                    # passed to the next component
                    self.param_mapping[task.name][param] = {
                        "component": path,
                    }
                    continue
                try:
                    source_node = self.get_node_by_name(source_component_name)
                except KeyError:
                    raise PipelineDefinitionError(
                        f"Component {source_component_name} does not exist in the pipeline,"
                        f" can not map {param} to {path} for {task.name}."
                    )
                source_component = source_node.component
                source_component_outputs = source_component.component_outputs
                if param_name and param_name not in source_component_outputs:
                    raise PipelineDefinitionError(
                        f"Parameter {param_name} is not valid output for "
                        f"{source_component_name} (must be one of "
                        f"{list(source_component_outputs.keys())})"
                    )
                self.param_mapping[task.name][param] = {
                    "component": source_component_name,
                    "param": param_name,
                }
            actual_inputs.extend(list(edge_inputs.keys()))
        missing_inputs = list(set(expected_mandatory_inputs) - set(actual_inputs))
        self.missing_inputs[task.name] = missing_inputs
        return True

    async def run(self, data: dict[str, Any]) -> PipelineResult:
        logger.debug("Starting pipeline")
        start_time = default_timer()
        self.invalidate()
        self.validate_input_data(data)
        orchestrator = Orchestrator(self)
        await orchestrator.run(data)
        end_time = default_timer()
        logger.debug(
            f"Pipeline {orchestrator.run_id} finished in {end_time - start_time}s"
        )
        return PipelineResult(
            run_id=orchestrator.run_id,
            result=await self.final_results.get(orchestrator.run_id),
        )
