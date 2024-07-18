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
from typing import Any, Optional

from pydantic import BaseModel

from neo4j_genai.core.stores import Store, InMemoryStore
from neo4j_genai.core.graph import Graph, Node


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    result: Optional[dict] = None


class Component:

    def process(self, **kwargs: Any) -> dict:
        return {}

    async def aprocess(self, **kwargs: Any) -> dict:
        return self.process(**kwargs)


class Task(Node):
    def __init__(self, name: str, component: Component, pipeline: "Pipeline"):
        super().__init__(name, {})
        self.component = component
        self.status = RunStatus.UNKNOWN
        self._pipeline = pipeline

    def execute(self, **kwargs: Any) -> RunResult:
        return asyncio.run(self.aexecute(**kwargs))

    async def aexecute(self, **kwargs: Any) -> RunResult:
        if self.status in (RunStatus.RUNNING, RunStatus.DONE):
            # warning: race condition here?
            logger.info(f"Skipping {self.name}, already in progress or done")
            return RunResult(status=RunStatus.SKIP)
        logger.info(f"Running component {self.name} with {kwargs}")
        self.status = RunStatus.RUNNING
        res = await self.component.aprocess(**kwargs)
        self.status = RunStatus.DONE
        return RunResult(
            status=self.status,
            result=res,
        )

    def get_input_defs_from_parents(self):
        input_defs = {}
        # make sure dependencies are satisfied
        # and save the inputs defs that needs to be propagated from parent components
        for prev_edge in self._pipeline.previous_edges(self):
            if prev_edge.start.status != RunStatus.DONE:
                logger.warning(f"Waiting for {prev_edge.start.name}")
                # let's wait, the run should be triggered once the last required
                # parent is done
                raise MissingDependencyError(f"{prev_edge.start.name} not ready")
            prev_edge_data = prev_edge.data.get("input_defs") or {}
            input_defs.update(**prev_edge_data)
        return input_defs

    def run_all(self, data) -> dict:
        # make sure dependencies are satisfied
        # and save the inputs defs that needs to be
        # propagated from parent components
        try:
            input_defs = self.get_input_defs_from_parents()
        except MissingDependencyError:
            return {"status": RunStatus.WAITING}
        # create component's input based on initial input data,
        # already done component's outputs and inputs definition mapping
        inputs = self._pipeline.get_component_inputs(self.name, input_defs, data)
        # execute component task
        res = self.execute(**inputs)
        # save its results
        if res.status == RunStatus.DONE:
            self._pipeline.add_result_for_component(self.name, res.result, is_final=self.is_leaf())
            # trigger execution of children nodes
            for c in self._pipeline.next_edges(self):
                if c.end.status != RunStatus.DONE:
                    c.end.run_all(data)
        return {"status": self.status}

    async def arun_all(self, data):
        # make sure dependencies are satisfied
        # and save the inputs defs that needs to be
        # propagated from parent components
        try:
            input_defs = self.get_input_defs_from_parents()
        except MissingDependencyError:
            return {"status": "WAITING"}
        # create component's input based on initial input data,
        # already done component's outputs and inputs definition mapping
        inputs = self._pipeline.get_component_inputs(self.name, input_defs, data)
        # execute component task
        res = await self.aexecute(**inputs)
        # save its results
        if res.status == RunStatus.DONE:
            self._pipeline.add_result_for_component(self.name, res.result, is_final=self.is_leaf())
            # trigger execution of children nodes
            tasks = [
                c.end.arun_all(data) for c in self._pipeline.next_edges(self)
            ]
            await asyncio.gather(*tasks)
        return {"status": self.status}


class Pipeline(Graph):
    def __init__(self, store: Optional[Store] = None):
        super().__init__()
        self._store = store or InMemoryStore()
        self._final_results = InMemoryStore()
        self._task_status = InMemoryStore()

    def add_component(self, name, component):
        task = Task(name, component, self)
        self.add_node(task)

    def connect(self, start_component_name, end_component_name,
                input_defs: Optional[dict[str, str]] = None):
        start_node = self.get_node_by_name(start_component_name, raise_exception=True)
        end_node = self.get_node_by_name(end_component_name, raise_exception=True)
        super().connect(start_node, end_node, data={"input_defs": input_defs})

    def add_result_for_component(self, name, result, is_final: bool = False):
        self._store.add(name, result)
        if is_final:
            self._final_results.add(name, result)

    def get_results_for_component(self, name: str):
        return self._store.get(name)

    def get_component_inputs(self, component_name: str, input_defs: dict,
                             input_data: dict) -> dict:
        component_inputs = input_data.get(component_name, {})
        if input_defs:
            for input_def, mapping in input_defs.items():
                value = self._store.find_all(mapping)
                if value:  # TODO: how to deal with multiple matches? Is it relevant?
                    value = value[0]
                component_inputs[input_def] = value
        return component_inputs

    def run_all(self, data):
        roots = self.roots()
        for root in roots:
            root.run_all(data)
        return self._final_results.all()

    async def arun_all(self, data):
        roots = self.roots()
        tasks = [root.arun_all(data) for root in roots]
        await asyncio.gather(*tasks)
        return self._final_results.all()
