import pytest
from neo4j_genai.core.pipeline import Component, Orchestrator, Pipeline, RunStatus


@pytest.fixture(scope="function")
def pipeline_branch() -> Pipeline:
    pipe = Pipeline()
    pipe.add_component("a", Component())
    pipe.add_component("b", Component())
    pipe.add_component("c", Component())
    pipe.connect("a", "b")
    pipe.connect("a", "c")
    return pipe


@pytest.fixture(scope="function")
def pipeline_aggregation() -> Pipeline:
    pipe = Pipeline()
    pipe.add_component("a", Component())
    pipe.add_component("b", Component())
    pipe.add_component("c", Component())
    pipe.connect("a", "b")
    pipe.connect("a", "c")
    return pipe


async def test_orchestrator_branch(pipeline_branch: Pipeline) -> None:
    orchestrator = Orchestrator(pipeline=pipeline_branch)
    node_a = pipeline_branch.get_node_by_name("a")
    node_a.status = RunStatus.DONE  # type: ignore
    next_tasks = [n async for n in orchestrator.next(node_a)]  # type: ignore
    next_task_names = [n.name for n in next_tasks]
    assert next_task_names == ["b", "c"]


async def test_orchestrator_aggregation(pipeline_aggregation: Pipeline) -> None:
    orchestrator = Orchestrator(pipeline=pipeline_aggregation)
    node_a = pipeline_aggregation.get_node_by_name("a")
    node_a.status = RunStatus.DONE  # type: ignore
    node_b = pipeline_aggregation.get_node_by_name("b")
    node_b.status = RunStatus.DONE  # type: ignore
    next_tasks = [n async for n in orchestrator.next(node_a)]  # type: ignore
    next_task_names = [n.name for n in next_tasks]
    assert next_task_names == ["c"]


async def test_orchestrator_aggregation_waiting(pipeline_aggregation: Pipeline) -> None:
    orchestrator = Orchestrator(pipeline=pipeline_aggregation)
    node_a = pipeline_aggregation.get_node_by_name("a")
    node_a.status = RunStatus.DONE  # type: ignore
    node_b = pipeline_aggregation.get_node_by_name("a")
    node_b.status = RunStatus.UNKNOWN  # type: ignore
    next_tasks = [n async for n in orchestrator.next(node_a)]  # type: ignore
    next_task_names = [n.name for n in next_tasks]
    assert next_task_names == []
