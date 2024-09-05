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

from unittest import mock
from unittest.mock import AsyncMock, call

import pytest
from neo4j_genai.experimental.pipeline import Component, Pipeline
from neo4j_genai.experimental.pipeline.exceptions import PipelineDefinitionError
from neo4j_genai.experimental.pipeline.pipeline import (
    RunResult,
    RunStatus,
    TaskPipelineNode,
)

from .components import (
    ComponentAdd,
    ComponentMultiply,
    ComponentNoParam,
    ComponentPassThrough,
    StringResultModel,
)


async def dummy_callback(task: TaskPipelineNode, res: RunResult) -> None:
    pass


@pytest.mark.asyncio
async def test_task_pipeline_node_status_done() -> None:
    task = TaskPipelineNode("task", ComponentNoParam())
    with mock.patch(
        "tests.unit.experimental.pipeline.test_pipeline.dummy_callback"
    ) as m:
        await task.run({}, m)
    args, kwargs = m.call_args
    assert len(kwargs) == 2
    assert kwargs["task"] == task
    assert isinstance(kwargs["res"], RunResult)
    assert task.status == RunStatus.DONE


@pytest.mark.asyncio
async def test_simple_pipeline_two_components() -> None:
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentNoParam()
    pipe.add_component(
        component_a,
        "a",
    )
    pipe.add_component(
        component_b,
        "b",
    )
    pipe.connect("a", "b", {})
    with mock.patch(
        "tests.unit.experimental.pipeline.test_pipeline.ComponentNoParam.run"
    ) as mock_run:
        mock_run.side_effect = [
            StringResultModel(result="1"),
            StringResultModel(result="2"),
        ]
        res = await pipe.run({})
        mock_run.assert_awaited_with(**{})
        mock_run.assert_awaited_with(**{})
    assert "b" in res
    assert res["b"] == {"result": "2"}


@pytest.mark.asyncio
async def test_pipeline_parameter_propagation() -> None:
    pipe = Pipeline()
    component_a = ComponentPassThrough()
    component_b = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    # first component output product goes to second component input number1
    pipe.connect("a", "b", {"value": "a.result"})
    with mock.patch(
        "tests.unit.experimental.pipeline.test_pipeline.ComponentPassThrough.run"
    ) as mock_run:
        mock_run.side_effect = [
            StringResultModel(result="1"),
            StringResultModel(result="2"),
        ]
        res = await pipe.run({"a": {"value": "text"}})
        mock_run.assert_has_awaits([call(**{"value": "text"}), call(**{"value": "1"})])
    assert res == {"b": {"result": "2"}}


def test_pipeline_validate_inputs_config_for_task_no_expected_params() -> None:
    pipe = Pipeline()
    component_a = ComponentNoParam()
    pipe.add_component(component_a, "a")
    is_valid = pipe.validate_connection_parameters_for_task(pipe.get_node_by_name("a"))
    assert is_valid is True


def test_pipeline_validate_inputs_config_for_task_one_component_all_good() -> None:
    pipe = Pipeline()
    component_a = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    is_valid = pipe.validate_connection_parameters_for_task(pipe.get_node_by_name("a"))
    assert is_valid is True


def test_pipeline_validate_inputs_config_for_task_one_component_input_param_missing() -> (
    None
):
    pipe = Pipeline()
    component_a = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.validate_connection_parameters_for_task(pipe.get_node_by_name("a"))
    assert pipe.missing_inputs["a"] == ["value"]


def test_pipeline_validate_inputs_config_for_task_one_component_full_input_missing() -> (
    None
):
    pipe = Pipeline()
    component_a = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.validate_connection_parameters_for_task(pipe.get_node_by_name("a"))
    assert pipe.missing_inputs["a"] == ["value"]


def test_pipeline_validate_inputs_config_for_task_connected_components_input() -> None:
    """Parameter for component 'b' comes from the pipeline inputs"""
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {})
    is_valid = pipe.validate_connection_parameters_for_task(pipe.get_node_by_name("b"))
    assert is_valid is True


def test_pipeline_validate_inputs_config_for_task_connected_components_result() -> None:
    """Parameter for component 'b' comes from the result of component 'a'"""
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {"value": "b.result"})
    is_valid = pipe.validate_connection_parameters_for_task(pipe.get_node_by_name("b"))
    assert is_valid is True


def test_pipeline_validate_inputs_config_for_task_connected_components_missing_input() -> (
    None
):
    """Parameter for component 'b' is missing"""
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {})
    pipe.validate_connection_parameters_for_task(pipe.get_node_by_name("b"))
    assert pipe.missing_inputs["b"] == ["value"]


@pytest.mark.asyncio
async def test_pipeline_branches() -> None:
    pipe = Pipeline()
    component_a = AsyncMock(spec=Component)
    component_a.run = AsyncMock(return_value={})
    component_b = AsyncMock(spec=Component)
    component_b.run = AsyncMock(return_value={})
    component_c = AsyncMock(spec=Component)
    component_c.run = AsyncMock(return_value={})

    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.add_component(component_c, "c")
    pipe.connect("a", "b")
    pipe.connect("a", "c")
    res = await pipe.run({})
    assert "b" in res
    assert "c" in res


@pytest.mark.asyncio
async def test_pipeline_aggregation() -> None:
    pipe = Pipeline()
    component_a = AsyncMock(spec=Component)
    component_a.run = AsyncMock(return_value={})
    component_b = AsyncMock(spec=Component)
    component_b.run = AsyncMock(return_value={})
    component_c = AsyncMock(spec=Component)
    component_c.run = AsyncMock(return_value={})

    pipe.add_component(
        component_a,
        "a",
    )
    pipe.add_component(
        component_b,
        "b",
    )
    pipe.add_component(component_c, "c")
    pipe.connect("a", "c")
    pipe.connect("b", "c")
    res = await pipe.run({})
    assert "c" in res


@pytest.mark.asyncio
async def test_pipeline_missing_param_on_init() -> None:
    pipe = Pipeline()
    component_a = ComponentAdd()
    component_b = ComponentAdd()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {"number1": "a.result"})
    with pytest.raises(PipelineDefinitionError) as excinfo:
        await pipe.run({"a": {"number1": 1}})
        assert (
            "Missing input parameters for a: Expected parameters: ['number1', 'number2']. Got: ['number1']"
            in str(excinfo.value)
        )


@pytest.mark.asyncio
async def test_pipeline_missing_param_on_connect() -> None:
    pipe = Pipeline()
    component_a = ComponentAdd()
    component_b = ComponentAdd()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {"number1": "a.result"})
    with pytest.raises(PipelineDefinitionError) as excinfo:
        await pipe.run({"a": {"number1": 1, "number2": 2}})
        assert (
            "Missing input parameters for b: Expected parameters: ['number1', 'number2']. Got: ['number1']"
            in str(excinfo.value)
        )


@pytest.mark.asyncio
async def test_pipeline_with_default_params() -> None:
    pipe = Pipeline()
    component_a = ComponentAdd()
    component_b = ComponentMultiply()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {"number1": "a.result"})
    res = await pipe.run({"a": {"number1": 1, "number2": 2}})
    assert res == {"b": {"result": 6}}  # (1+2)*2


@pytest.mark.asyncio
async def test_pipeline_cycle() -> None:
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentNoParam()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {})
    with pytest.raises(PipelineDefinitionError) as excinfo:
        pipe.connect("b", "a", {})
        assert "Cycles are not allowed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_pipeline_wrong_component_name() -> None:
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentNoParam()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    with pytest.raises(PipelineDefinitionError) as excinfo:
        pipe.connect("a", "c", {})
        assert "a or c not in the Pipeline" in str(excinfo.value)
