import pytest
from neo4j_genai.pipeline import Component
from neo4j_genai.pipeline.pipeline import Orchestrator, Pipeline, RunStatus


@pytest.fixture(scope="function")
def pipeline_branch() -> Pipeline:
    pipe = Pipeline()
    pipe.add_component("a", Component())  # type: ignore
    pipe.add_component("b", Component())  # type: ignore
    pipe.add_component("c", Component())  # type: ignore
    pipe.connect("a", "b")
    pipe.connect("a", "c")
    return pipe


@pytest.fixture(scope="function")
def pipeline_aggregation() -> Pipeline:
    pipe = Pipeline()
    pipe.add_component("a", Component())  # type: ignore
    pipe.add_component("b", Component())  # type: ignore
    pipe.add_component("c", Component())  # type: ignore
    pipe.connect("a", "b")
    pipe.connect("a", "c")
    return pipe


@pytest.mark.asyncio
async def test_orchestrator_branch(pipeline_branch: Pipeline) -> None:
    orchestrator = Orchestrator(pipeline=pipeline_branch)
    node_a = pipeline_branch.get_node_by_name("a")
    node_a.status = RunStatus.DONE  # type: ignore
    next_tasks = [n async for n in orchestrator.next(node_a)]  # type: ignore
    next_task_names = [n.name for n in next_tasks]
    assert next_task_names == ["b", "c"]


@pytest.mark.asyncio
async def test_orchestrator_aggregation(pipeline_aggregation: Pipeline) -> None:
    orchestrator = Orchestrator(pipeline=pipeline_aggregation)
    node_a = pipeline_aggregation.get_node_by_name("a")
    node_a.status = RunStatus.DONE  # type: ignore
    node_b = pipeline_aggregation.get_node_by_name("b")
    node_b.status = RunStatus.DONE  # type: ignore
    next_tasks = [n async for n in orchestrator.next(node_a)]  # type: ignore
    next_task_names = [n.name for n in next_tasks]
    assert next_task_names == ["c"]


@pytest.mark.asyncio
async def test_orchestrator_aggregation_waiting(pipeline_aggregation: Pipeline) -> None:
    orchestrator = Orchestrator(pipeline=pipeline_aggregation)
    node_a = pipeline_aggregation.get_node_by_name("a")
    node_a.status = RunStatus.DONE  # type: ignore
    node_b = pipeline_aggregation.get_node_by_name("a")
    node_b.status = RunStatus.UNKNOWN  # type: ignore
    next_tasks = [n async for n in orchestrator.next(node_a)]  # type: ignore
    assert next_tasks == []
