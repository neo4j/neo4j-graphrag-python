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
import pytest
from neo4j_genai.experimental.pipeline import Component
from neo4j_genai.experimental.pipeline.pipeline import Orchestrator, Pipeline, RunStatus

from tests.unit.experimental.pipeline.components import (
    ComponentNoParam,
    ComponentPassThrough,
)


def test_orchestrator_get_component_inputs_from_user_only() -> None:
    pipe = Pipeline()
    pipe.add_component(ComponentPassThrough(), "a")
    pipe.add_component(ComponentPassThrough(), "b")
    orchestrator = Orchestrator(pipe)
    input_data = {"a": {"value": "text"}, "b": {"value": "other text"}}
    data = orchestrator.get_component_inputs("a", {}, input_data)
    assert data == {"value": "text"}
    data = orchestrator.get_component_inputs("b", {}, input_data)
    assert data == {"value": "other text"}


def test_pipeline_get_component_inputs_from_parent_specific() -> None:
    pipe = Pipeline()
    pipe.add_component(ComponentPassThrough(), "a")
    pipe.add_component(ComponentPassThrough(), "b")
    pipe.connect("a", "b", input_config={"value": "a.result"})

    # component "a" already run and results stored:
    pipe._store.add("a", {"result": "output from component a"})

    orchestrator = Orchestrator(pipe)
    data = orchestrator.get_component_inputs("b", {"value": "a.result"}, {})
    assert data == {"value": "output from component a"}


def test_orchestrator_get_component_inputs_from_parent_all() -> None:
    pipe = Pipeline()
    pipe.add_component(ComponentNoParam(), "a")
    pipe.add_component(ComponentPassThrough(), "b")
    pipe.connect("a", "b", input_config={"value": "a"})

    # component "a" already run and results stored:
    pipe._store.add("a", {"result": "output from component a"})

    orchestrator = Orchestrator(pipe)
    data = orchestrator.get_component_inputs("b", {"value": "a"}, {})
    assert data == {"value": {"result": "output from component a"}}


def test_orchestrator_get_component_inputs_from_parent_and_input() -> None:
    pipe = Pipeline()
    pipe.add_component(ComponentNoParam(), "a")
    pipe.add_component(ComponentPassThrough(), "b")
    pipe.connect("a", "b", input_config={"value": "a"})

    # component "a" already run and results stored:
    pipe._store.add("a", {"result": "output from component a"})

    orchestrator = Orchestrator(pipe)
    data = orchestrator.get_component_inputs(
        "b",
        {"value": "a"},
        {"b": {"other_value": "other_text"}},
    )
    assert data == {
        "value": {"result": "output from component a"},
        "other_value": "other_text",
    }


@pytest.fixture(scope="function")
def pipeline_branch() -> Pipeline:
    pipe = Pipeline()
    pipe.add_component(Component(), "a")  # type: ignore[abstract,unused-ignore]
    pipe.add_component(Component(), "b")  # type: ignore[abstract,unused-ignore]
    pipe.add_component(Component(), "c")  # type: ignore[abstract,unused-ignore]
    pipe.connect("a", "b")
    pipe.connect("a", "c")
    return pipe


@pytest.fixture(scope="function")
def pipeline_aggregation() -> Pipeline:
    pipe = Pipeline()
    pipe.add_component(Component(), "a")  # type: ignore[abstract,unused-ignore]
    pipe.add_component(Component(), "b")  # type: ignore[abstract,unused-ignore]
    pipe.add_component(Component(), "c")  # type: ignore[abstract,unused-ignore]
    pipe.connect("a", "c")
    pipe.connect("b", "c")
    return pipe


@pytest.mark.asyncio
async def test_orchestrator_branch(pipeline_branch: Pipeline) -> None:
    orchestrator = Orchestrator(pipeline=pipeline_branch)
    node_a = pipeline_branch.get_node_by_name("a")
    node_a.status = RunStatus.DONE
    next_tasks = [n async for n in orchestrator.next(node_a)]
    next_task_names = [n.name for n in next_tasks]
    assert next_task_names == ["b", "c"]


@pytest.mark.asyncio
async def test_orchestrator_aggregation(pipeline_aggregation: Pipeline) -> None:
    orchestrator = Orchestrator(pipeline=pipeline_aggregation)
    node_a = pipeline_aggregation.get_node_by_name("a")
    node_a.status = RunStatus.DONE
    next_tasks = [n async for n in orchestrator.next(node_a)]
    next_task_names = [n.name for n in next_tasks]
    # "c" not ready yet
    assert next_task_names == []
    # set "b" to DONE
    node_b = pipeline_aggregation.get_node_by_name("b")
    node_b.status = RunStatus.DONE
    # then "c" can start
    next_tasks = [n async for n in orchestrator.next(node_a)]
    next_task_names = [n.name for n in next_tasks]
    assert next_task_names == ["c"]


@pytest.mark.asyncio
async def test_orchestrator_aggregation_waiting(pipeline_aggregation: Pipeline) -> None:
    orchestrator = Orchestrator(pipeline=pipeline_aggregation)
    node_a = pipeline_aggregation.get_node_by_name("a")
    node_a.status = RunStatus.DONE
    node_b = pipeline_aggregation.get_node_by_name("a")
    node_b.status = RunStatus.UNKNOWN
    next_tasks = [n async for n in orchestrator.next(node_a)]
    assert next_tasks == []
