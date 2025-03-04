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
from unittest.mock import Mock, patch

import pytest
from neo4j_graphrag.experimental.pipeline import (
    Component,
    Pipeline,
)
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError, \
    PipelineMissingDependencyError, PipelineStatusUpdateError
from neo4j_graphrag.experimental.pipeline.orchestrator import Orchestrator
from neo4j_graphrag.experimental.pipeline.types import RunStatus

from tests.unit.experimental.pipeline.components import (
    ComponentNoParam,
    ComponentPassThrough,
)


def test_orchestrator_get_input_config_for_task_pipeline_not_validated() -> None:
    pipe = Pipeline()
    pipe.add_component(ComponentPassThrough(), "a")
    pipe.add_component(ComponentPassThrough(), "b")
    orchestrator = Orchestrator(pipe)
    with pytest.raises(PipelineDefinitionError) as exc:
        orchestrator.get_input_config_for_task(pipe.get_node_by_name("a"))
    assert "You must validate the pipeline input config first" in str(exc.value)


@pytest.mark.asyncio
async def test_orchestrator_get_component_inputs_from_user_only() -> None:
    """Components take all their inputs from user input."""
    pipe = Pipeline()
    pipe.add_component(ComponentPassThrough(), "a")
    pipe.add_component(ComponentPassThrough(), "b")
    orchestrator = Orchestrator(pipe)
    input_data = {
        "a": {"value": "user input for component a"},
        "b": {"value": "user input for component b"},
    }
    data = await orchestrator.get_component_inputs("a", {}, input_data)
    assert data == {"value": "user input for component a"}
    data = await orchestrator.get_component_inputs("b", {}, input_data)
    assert data == {"value": "user input for component b"}


@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.get_results_for_component"
)
@pytest.mark.asyncio
async def test_orchestrator_get_component_inputs_from_parent_specific(
    mock_result: Mock,
) -> None:
    """Propagate one specific output field from parent to a child component."""
    pipe = Pipeline()
    pipe.add_component(ComponentPassThrough(), "a")
    pipe.add_component(ComponentPassThrough(), "b")
    pipe.connect("a", "b", input_config={"value": "a.result"})

    # component "a" already run and results stored:
    mock_result.return_value = {"result": "output from component a"}

    orchestrator = Orchestrator(pipe)
    data = await orchestrator.get_component_inputs(
        "b", {"value": {"component": "a", "param": "result"}}, {}
    )
    assert data == {"value": "output from component a"}


@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.get_results_for_component"
)
@pytest.mark.asyncio
async def test_orchestrator_get_component_inputs_from_parent_all(
    mock_result: Mock,
) -> None:
    """Use the component name to get the full output
    (without extracting a specific field).
    """
    pipe = Pipeline()
    pipe.add_component(ComponentNoParam(), "a")
    pipe.add_component(ComponentPassThrough(), "b")
    pipe.connect("a", "b", input_config={"value": "a"})

    # component "a" already run and results stored:
    mock_result.return_value = {"result": "output from component a"}

    orchestrator = Orchestrator(pipe)
    data = await orchestrator.get_component_inputs(
        "b", {"value": {"component": "a"}}, {}
    )
    assert data == {"value": {"result": "output from component a"}}


@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.get_results_for_component"
)
@pytest.mark.asyncio
async def test_orchestrator_get_component_inputs_from_parent_and_input(
    mock_result: Mock,
) -> None:
    """Some parameters from user input, some other parameter from previous component."""
    pipe = Pipeline()
    pipe.add_component(ComponentNoParam(), "a")
    pipe.add_component(ComponentPassThrough(), "b")
    pipe.connect("a", "b", input_config={"value": "a"})

    # component "a" already run and results stored:
    mock_result.return_value = {"result": "output from component a"}

    orchestrator = Orchestrator(pipe)
    data = await orchestrator.get_component_inputs(
        "b",
        {"value": {"component": "a"}},
        {"b": {"other_value": "user input for component b 'other_value' param"}},
    )
    assert data == {
        "value": {"result": "output from component a"},
        "other_value": "user input for component b 'other_value' param",
    }


@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.get_results_for_component"
)
@pytest.mark.asyncio
async def test_orchestrator_get_component_inputs_ignore_user_input_if_input_def_provided(
    mock_result: Mock,
) -> None:
    """If a parameter is defined both in the user input and in an input definition
    (ie propagated from a previous component), the user input is ignored and a
    warning is raised.
    """
    pipe = Pipeline()
    pipe.add_component(ComponentNoParam(), "a")
    pipe.add_component(ComponentPassThrough(), "b")
    pipe.connect("a", "b", input_config={"value": "a"})

    # component "a" already run and results stored:
    mock_result.return_value = {"result": "output from component a"}

    orchestrator = Orchestrator(pipe)
    with pytest.warns(Warning) as w:
        data = await orchestrator.get_component_inputs(
            "b",
            {"value": {"component": "a"}},
            {"b": {"value": "user input for component a"}},
        )
        assert data == {"value": {"result": "output from component a"}}
        assert (
            w[0].message.args[0]  # type: ignore[union-attr]
            == "In component 'b', parameter 'value' from user input will be ignored and replaced by 'a.value'"
        )


@pytest.mark.asyncio
@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.get_status_for_component"
)
async def test_orchestrator_set_component_status(mock_status: Mock) -> None:
    pipe = Pipeline()
    orchestrator = Orchestrator(pipeline=pipe)
    # Normal status update: UNKNOWN -> RUNNING -> DONE
    # UNKNOWN -> RUNNING
    mock_status.side_effect = [
        RunStatus.UNKNOWN,
    ]
    assert await orchestrator.set_task_status("task_name", RunStatus.RUNNING) is None
    # RUNNING -> DONE
    mock_status.side_effect = [
        RunStatus.RUNNING,
    ]
    assert await orchestrator.set_task_status("task_name", RunStatus.DONE) is None
    # Error path, raising PipelineStatusUpdateError
    # Same status
    # RUNNING -> RUNNING
    mock_status.side_effect = [
        RunStatus.RUNNING,
    ]
    with pytest.raises(PipelineStatusUpdateError) as exc:
        await orchestrator.set_task_status("task_name", RunStatus.RUNNING)
    assert "Status is already" in str(exc)
    # Going back to RUNNING after the task is DONE
    # DONE -> RUNNING
    mock_status.side_effect = [
        RunStatus.DONE,
    ]
    with pytest.raises(PipelineStatusUpdateError) as exc:
        await orchestrator.set_task_status("task_name", RunStatus.RUNNING)
    assert "Can't go from DONE to RUNNING" in str(exc)


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
@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.get_status_for_component"
)
async def test_orchestrator_check_dependency_complete(mock_status: Mock, pipeline_branch: Pipeline) -> None:
    """a -> b, c"""
    orchestrator = Orchestrator(pipeline=pipeline_branch)
    node_a = pipeline_branch.get_node_by_name("a")
    assert await orchestrator.check_dependencies_complete(node_a) is None
    node_b = pipeline_branch.get_node_by_name("b")
    # dependency is DONE:
    mock_status.side_effect = [RunStatus.DONE]
    assert await orchestrator.check_dependencies_complete(node_b) is None
    # dependency is not DONE:
    mock_status.side_effect = [RunStatus.RUNNING]
    with pytest.raises(PipelineMissingDependencyError):
        await orchestrator.check_dependencies_complete(node_b)


@pytest.mark.asyncio
@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.get_status_for_component"
)
async def test_orchestrator_check_dependency_complete(mock_status: Mock, pipeline_branch: Pipeline) -> None:
    """a -> b, c"""
    orchestrator = Orchestrator(pipeline=pipeline_branch)
    node_a = pipeline_branch.get_node_by_name("a")
    assert await orchestrator.check_dependencies_complete(node_a) is None


@pytest.mark.asyncio
@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.get_status_for_component"
)
@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.check_dependencies_complete",
)
async def test_orchestrator_next_task_branch_no_missing_dependencies(
    mock_dep: Mock, mock_status: Mock, pipeline_branch: Pipeline
) -> None:
    """a -> b, c"""
    orchestrator = Orchestrator(pipeline=pipeline_branch)
    node_a = pipeline_branch.get_node_by_name("a")
    mock_status.side_effect = [
        # next "b"
        RunStatus.UNKNOWN,
        # next "c"
        RunStatus.UNKNOWN,
    ]
    mock_dep.side_effect = [
        None,  # "b" has no missing dependencies
        None,  # "c" has no missing dependencies
    ]
    next_tasks = [n async for n in orchestrator.next(node_a)]
    next_task_names = [n.name for n in next_tasks]
    assert next_task_names == ["b", "c"]


@pytest.mark.asyncio
@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.get_status_for_component"
)
@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.check_dependencies_complete",
)
async def test_orchestrator_next_task_branch_missing_dependencies(
    mock_dep: Mock, mock_status: Mock, pipeline_branch: Pipeline
) -> None:
    """a -> b, c"""
    orchestrator = Orchestrator(pipeline=pipeline_branch)
    node_a = pipeline_branch.get_node_by_name("a")
    mock_status.side_effect = [
        # next "b"
        RunStatus.UNKNOWN,
        # next "c"
        RunStatus.UNKNOWN,
    ]
    mock_dep.side_effect = [
        PipelineMissingDependencyError,  # "b" has missing dependencies
        None,  # "c" has no missing dependencies
    ]
    next_tasks = [n async for n in orchestrator.next(node_a)]
    next_task_names = [n.name for n in next_tasks]
    assert next_task_names == ["c"]


@pytest.mark.asyncio
@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.get_status_for_component"
)
@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.check_dependencies_complete",
)
async def test_orchestrator_next_task_aggregation_no_missing_dependencies(
    mock_dep: Mock, mock_status: Mock, pipeline_aggregation: Pipeline
) -> None:
    """a, b -> c"""
    orchestrator = Orchestrator(pipeline=pipeline_aggregation)
    node_a = pipeline_aggregation.get_node_by_name("a")
    mock_status.side_effect = [
        RunStatus.UNKNOWN,  # status for "c", not started
    ]
    mock_dep.side_effect = [
        None,  # no missing deps
    ]
    # then "c" can start
    next_tasks = [n async for n in orchestrator.next(node_a)]
    next_task_names = [n.name for n in next_tasks]
    assert next_task_names == ["c"]


@pytest.mark.asyncio
@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.get_status_for_component"
)
@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.check_dependencies_complete",
)
async def test_orchestrator_next_task_aggregation_missing_dependency(
    mock_dep: Mock, mock_status: Mock, pipeline_aggregation: Pipeline
) -> None:
    """a, b -> c"""
    orchestrator = Orchestrator(pipeline=pipeline_aggregation)
    node_a = pipeline_aggregation.get_node_by_name("a")
    mock_status.side_effect = [
        RunStatus.UNKNOWN,  # status for "c" is unknown, it's a possible next
    ]
    mock_dep.side_effect = [
        PipelineMissingDependencyError,  # some dependencies are not done yet
    ]
    next_task_names = [n.name async for n in orchestrator.next(node_a)]
    # "c" dependencies not ready yet
    assert next_task_names == []


@pytest.mark.asyncio
@patch(
    "neo4j_graphrag.experimental.pipeline.pipeline.Orchestrator.get_status_for_component"
)
async def test_orchestrator_next_task_aggregation_next_already_started(
    mock_status: Mock, pipeline_aggregation: Pipeline
) -> None:
    """a, b -> c"""
    orchestrator = Orchestrator(pipeline=pipeline_aggregation)
    node_a = pipeline_aggregation.get_node_by_name("a")
    mock_status.side_effect = [
        RunStatus.RUNNING,  # status for "c" is already running, do not start it again
    ]
    next_task_names = [n.name async for n in orchestrator.next(node_a)]
    assert next_task_names == []
