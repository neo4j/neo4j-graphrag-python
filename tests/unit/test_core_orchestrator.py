import pytest

from neo4j_genai.core.graph import Node, Graph
from neo4j_genai.core.pipeline import Orchestrator, Component, Pipeline, RunStatus


@pytest.fixture(scope="function")
def component():
    return Component()


@pytest.fixture(scope="function")
def pipeline_branch(component):
    pipe = Pipeline()
    pipe.add_component("a", component)
    pipe.add_component("b", component)
    pipe.add_component("c", component)
    pipe.connect("a", "b")
    pipe.connect("a", "c")
    return pipe


@pytest.fixture(scope="function")
def pipeline_aggregation(component):
    pipe = Pipeline()
    pipe.add_component("a", component)
    pipe.add_component("b", component)
    pipe.add_component("c", component)
    pipe.connect("a", "b")
    pipe.connect("a", "c")
    return pipe


def test_orchestrator_branch(pipeline_branch):
    orchestrator = Orchestrator(pipeline=pipeline_branch)
    node_a = pipeline_branch.get_node_by_name("a")
    node_a.status = RunStatus.DONE
    next_tasks = orchestrator.next(node_a)
    next_task_names = [n.name for n in next_tasks]
    assert next_task_names == ["b", "c"]


def test_orchestrator_aggregation(pipeline_aggregation):
    orchestrator = Orchestrator(pipeline=pipeline_aggregation)
    node_a = pipeline_aggregation.get_node_by_name("a")
    node_a.status = RunStatus.DONE
    node_b = pipeline_aggregation.get_node_by_name("b")
    node_b.status = RunStatus.DONE
    next_tasks = orchestrator.next(node_a)
    next_task_names = [n.name for n in next_tasks]
    assert next_task_names == ["c"]


def test_orchestrator_aggregation_waiting(pipeline_aggregation):
    orchestrator = Orchestrator(pipeline=pipeline_aggregation)
    node_a = pipeline_aggregation.get_node_by_name("a")
    node_a.status = RunStatus.DONE
    node_b = pipeline_aggregation.get_node_by_name("a")
    node_b.status = RunStatus.UNKNOWN
    next_tasks = orchestrator.next(node_a)
    next_task_names = [n.name for n in next_tasks]
    assert next_task_names == []
