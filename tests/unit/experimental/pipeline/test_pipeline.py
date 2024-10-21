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
import tempfile
from unittest import mock
from unittest.mock import AsyncMock, call, patch

import pytest
from neo4j_graphrag.experimental.pipeline import Component, Pipeline
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError

from .components import (
    ComponentAdd,
    ComponentMultiply,
    ComponentNoParam,
    ComponentPassThrough,
    StringResultModel,
)


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
    assert "b" in res.result
    assert res.result["b"] == {"result": "2"}


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
    assert res.result == {"b": {"result": "2"}}


def test_pipeline_parameter_validation_no_expected_params() -> None:
    pipe = Pipeline()
    component_a = ComponentNoParam()
    pipe.add_component(component_a, "a")
    is_valid = pipe.validate_parameter_mapping_for_task(pipe.get_node_by_name("a"))
    assert is_valid is True


def test_pipeline_parameter_validation_one_component_all_good() -> None:
    pipe = Pipeline()
    component_a = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    is_valid = pipe.validate_parameter_mapping_for_task(pipe.get_node_by_name("a"))
    assert is_valid is True


def test_pipeline_invalidate() -> None:
    pipe = Pipeline()
    pipe.is_validated = True
    pipe.param_mapping = {"a": {"key": {"component": "component", "param": "param"}}}
    pipe.missing_inputs = {"a": ["other_key"]}
    pipe.invalidate()
    assert pipe.is_validated is False
    assert len(pipe.param_mapping) == 0
    assert len(pipe.missing_inputs) == 0


def test_pipeline_parameter_validation_called_twice() -> None:
    pipe = Pipeline()
    component_a = ComponentPassThrough()
    component_b = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {"value": "a.result"})
    is_valid = pipe.validate_parameter_mapping_for_task(pipe.get_node_by_name("b"))
    assert is_valid is True
    with pytest.raises(PipelineDefinitionError):
        pipe.validate_parameter_mapping_for_task(pipe.get_node_by_name("b"))
    pipe.invalidate()
    is_valid = pipe.validate_parameter_mapping_for_task(pipe.get_node_by_name("b"))
    assert is_valid is True


def test_pipeline_parameter_validation_one_component_input_param_missing() -> None:
    pipe = Pipeline()
    component_a = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.validate_parameter_mapping_for_task(pipe.get_node_by_name("a"))
    assert pipe.missing_inputs["a"] == ["value"]


def test_pipeline_parameter_validation_param_mapped_twice() -> None:
    pipe = Pipeline()
    component_a = ComponentPassThrough()
    component_b = ComponentPassThrough()
    component_c = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.add_component(component_c, "c")
    pipe.connect("a", "c", {"value": "a.result"})
    pipe.connect("b", "c", {"value": "b.result"})
    with pytest.raises(PipelineDefinitionError) as excinfo:
        pipe.validate_parameter_mapping_for_task(pipe.get_node_by_name("c"))
        assert (
            "Parameter 'value' already mapped to {'component': 'a', 'param': 'result'}"
            in str(excinfo)
        )


def test_pipeline_parameter_validation_unexpected_input() -> None:
    pipe = Pipeline()
    component_a = ComponentPassThrough()
    component_b = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {"unexpected_input_name": "a.result"})
    with pytest.raises(PipelineDefinitionError) as excinfo:
        pipe.validate_parameter_mapping_for_task(pipe.get_node_by_name("b"))
        assert (
            "Parameter 'unexpected_input_name' is not a valid input for component 'b' of type 'ComponentPassThrough'"
            in str(excinfo)
        )


def test_pipeline_parameter_validation_connected_components_input() -> None:
    """Parameter for component 'b' comes from the pipeline inputs"""
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {})
    is_valid = pipe.validate_parameter_mapping_for_task(pipe.get_node_by_name("b"))
    assert is_valid is True
    assert dict(pipe.missing_inputs) == {"b": ["value"]}


def test_pipeline_parameter_validation_connected_components_result() -> None:
    """Parameter for component 'b' comes from the result of component 'a'"""
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {"value": "b.result"})
    is_valid = pipe.validate_parameter_mapping_for_task(pipe.get_node_by_name("b"))
    assert is_valid is True
    assert pipe.missing_inputs == {"b": []}


def test_pipeline_parameter_validation_connected_components_missing_input() -> None:
    """Parameter for component 'b' is missing"""
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {})
    is_valid = pipe.validate_parameter_mapping_for_task(pipe.get_node_by_name("b"))
    assert is_valid is True
    assert pipe.missing_inputs["b"] == ["value"]


def test_pipeline_parameter_validation_full_missing_inputs_in_user_data() -> None:
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {})
    is_valid = pipe.validate_input_data(data={"b": {"value": "input for b"}})
    assert is_valid is True


def test_pipeline_parameter_validation_full_missing_inputs_in_component_name() -> None:
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {})
    with pytest.raises(PipelineDefinitionError):
        pipe.validate_input_data(data={"b": {}})


def test_pipeline_parameter_validation_full_missing_inputs() -> None:
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentPassThrough()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {})
    with pytest.raises(PipelineDefinitionError):
        pipe.validate_input_data(data={})


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
    pipeline_result = await pipe.run({})
    res = pipeline_result.result
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
    pipeline_result = await pipe.run({})
    res = pipeline_result.result
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
    pipeline_result = await pipe.run({"a": {"number1": 1, "number2": 2}})
    res = pipeline_result.result
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


@pytest.mark.asyncio
async def test_pipeline_async() -> None:
    pipe = Pipeline()
    pipe.add_component(ComponentAdd(), "add")
    run_params = [[1, 20], [10, 2]]
    runs = []
    for a, b in run_params:
        runs.append(pipe.run({"add": {"number1": a, "number2": b}}))
    pipeline_result = await asyncio.gather(*runs)
    assert len(pipeline_result) == 2
    assert pipeline_result[0].run_id != pipeline_result[1].run_id
    assert pipeline_result[0].result == {"add": {"result": 21}}
    assert pipeline_result[1].result == {"add": {"result": 12}}


def test_pipeline_to_pgv() -> None:
    pipe = Pipeline()
    component_a = ComponentAdd()
    component_b = ComponentMultiply()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {"number1": "a.result"})
    g = pipe.get_pygraphviz_graph()
    # 3 nodes:
    #   - 2 components 'a' and 'b'
    #   - 1 output 'a.result'
    assert len(g.nodes()) == 3
    g = pipe.get_pygraphviz_graph(hide_unused_outputs=False)
    # 4 nodes:
    #   - 2 components 'a' and 'b'
    #   - 2 output 'a.result' and 'b.result'
    assert len(g.nodes()) == 4


def test_pipeline_draw() -> None:
    pipe = Pipeline()
    pipe.add_component(ComponentAdd(), "add")
    t = tempfile.NamedTemporaryFile()
    pipe.draw(t.name)
    content = t.file.read()
    assert len(content) > 0


@patch("neo4j_graphrag.experimental.pipeline.pipeline.pgv", None)
def test_pipeline_draw_missing_pygraphviz_dep() -> None:
    pipe = Pipeline()
    pipe.add_component(ComponentAdd(), "add")
    t = tempfile.NamedTemporaryFile()
    with pytest.raises(ImportError):
        pipe.draw(t.name)
