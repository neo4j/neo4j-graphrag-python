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

import logging
import warnings
from collections import defaultdict
from timeit import default_timer
from typing import Any, Optional, AsyncGenerator, Callable, List
import asyncio

from neo4j_graphrag.utils.logging import prettify

try:
    import pygraphviz as pgv
except ImportError:
    pgv = None

from pydantic import BaseModel

from neo4j_graphrag.experimental.pipeline.component import Component
from neo4j_graphrag.experimental.pipeline.exceptions import (
    PipelineDefinitionError,
)
from neo4j_graphrag.experimental.pipeline.orchestrator import Orchestrator
from neo4j_graphrag.experimental.pipeline.pipeline_graph import (
    PipelineEdge,
    PipelineGraph,
    PipelineNode,
)
from neo4j_graphrag.experimental.pipeline.stores import InMemoryStore, ResultStore
from neo4j_graphrag.experimental.pipeline.types.definitions import (
    ComponentDefinition,
    ConnectionDefinition,
    PipelineDefinition,
)
from neo4j_graphrag.experimental.pipeline.types.orchestration import RunResult
from neo4j_graphrag.experimental.pipeline.types.context import RunContext
from neo4j_graphrag.experimental.pipeline.notification import EventCallbackProtocol, Event


logger = logging.getLogger(__name__)


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

    async def execute(
        self, context: RunContext, inputs: dict[str, Any]
    ) -> RunResult | None:
        """Execute the task

        Returns:
            RunResult | None: RunResult with status and result dict
            if the task run successfully, None if the status update
            was unsuccessful.
        """
        component_result = await self.component.run_with_context(
            context_=context, **inputs
        )
        run_result = RunResult(
            result=component_result,
        )
        return run_result

    async def run(
        self, context: RunContext, inputs: dict[str, Any]
    ) -> RunResult | None:
        """Main method to execute the task."""
        logger.debug(f"TASK START {self.name=} input={prettify(inputs)}")
        start_time = default_timer()
        res = await self.execute(context, inputs)
        end_time = default_timer()
        logger.debug(
            f"TASK FINISHED {self.name} in {end_time - start_time} res={prettify(res)}"
        )
        return res


class PipelineResult(BaseModel):
    run_id: str
    result: Any


class Pipeline(PipelineGraph[TaskPipelineNode, PipelineEdge]):
    """This is the main pipeline, where components
    and their execution order are defined"""

    def __init__(
        self,
        store: Optional[ResultStore] = None,
        callback: Optional[EventCallbackProtocol] = None,
    ) -> None:
        super().__init__()
        self.store = store or InMemoryStore()
        self.callback = callback
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
        cls, pipeline_template: PipelineDefinition, store: Optional[ResultStore] = None
    ) -> Pipeline:
        warnings.warn(
            "from_template is deprecated, use from_definition instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_definition(pipeline_template, store)

    @classmethod
    def from_definition(
        cls,
        pipeline_definition: PipelineDefinition,
        store: Optional[ResultStore] = None,
    ) -> Pipeline:
        """Create a Pipeline from a pydantic model defining the components and their connections

        Args:
            pipeline_definition (PipelineDefinition): An object defining components and how they are connected to each other.
            store (Optional[ResultStore]): Where the results are stored. By default, uses the InMemoryStore.
        """
        pipeline = Pipeline(store=store)
        for component in pipeline_definition.components:
            pipeline.add_component(
                component.component,
                component.name,
            )
        for edge in pipeline_definition.connections:
            pipeline_edge = PipelineEdge(
                edge.start, edge.end, data={"input_config": edge.input_config}
            )
            pipeline.add_edge(pipeline_edge)
        return pipeline

    def show_as_dict(self) -> dict[str, Any]:
        component_config = []
        for name, task in self._nodes.items():
            component_config.append(
                ComponentDefinition(name=name, component=task.component)
            )
        connection_config = []
        for edge in self._edges:
            connection_config.append(
                ConnectionDefinition(
                    start=edge.start,
                    end=edge.end,
                    input_config=edge.data["input_config"] if edge.data else {},
                )
            )
        pipeline_config = PipelineDefinition(
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

    async def get_final_results(self, run_id: str) -> dict[str, Any]:
        return await self.final_results.get(run_id)  # type: ignore[no-any-return]

    async def stream(self, data: dict[str, Any]) -> AsyncGenerator[Event, None]:
        """Run the pipeline and stream events for task progress.
        
        Args:
            data: Input data for the pipeline components
            
        Yields:
            Event: Pipeline and task events including start, progress, and completion
        """
        # Create queue for events
        event_queue: asyncio.Queue[Event] = asyncio.Queue()
        
        # Store original callback
        original_callback = self.callback
        
        async def callback_and_event_stream(event: Event) -> None:
            # Put event in queue for streaming
            await event_queue.put(event)
            # Call original callback if it exists
            if original_callback:
                await original_callback(event)
        
        # Set up event callback
        self.callback = callback_and_event_stream
        
        try:
            # Start pipeline execution in background task
            run_task = asyncio.create_task(self.run(data))
            
            while True:
                # Wait for next event or pipeline completion
                done, pending = await asyncio.wait(
                    [run_task, event_queue.get()],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Pipeline finished
                if run_task in done:
                    if run_task.exception():
                        raise run_task.exception()
                    # Drain any remaining events
                    while not event_queue.empty():
                        yield await event_queue.get()
                    break
                    
                # Got an event from queue
                event_future = next(f for f in done if f != run_task)
                try:
                    event = event_future.result()
                    yield event
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    raise
        
        finally:
            # Restore original callback
            self.callback = original_callback

    async def run(self, data: dict[str, Any]) -> PipelineResult:
        logger.debug("PIPELINE START")
        start_time = default_timer()
        self.invalidate()
        self.validate_input_data(data)
        orchestrator = Orchestrator(self)
        logger.debug(f"PIPELINE ORCHESTRATOR: {orchestrator.run_id}")
        await orchestrator.run(data)
        end_time = default_timer()
        logger.debug(
            f"PIPELINE FINISHED {orchestrator.run_id} in {end_time - start_time}s"
        )
        return PipelineResult(
            run_id=orchestrator.run_id,
            result=await self.get_final_results(orchestrator.run_id),
        )
