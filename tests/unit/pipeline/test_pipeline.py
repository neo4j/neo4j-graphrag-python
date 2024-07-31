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
from unittest.mock import AsyncMock

import pytest
from neo4j_genai.pipeline import Component, Pipeline
from neo4j_genai.pipeline.exceptions import PipelineDefinitionError

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
    pipe.add_component("a", component_a)
    pipe.add_component("b", component_b)
    pipe.connect("a", "b", {})
    with mock.patch(
        "tests.unit.pipeline.test_pipeline.ComponentNoParam.run"
    ) as mock_run:
        mock_run.side_effect = [
            StringResultModel(result=""),
            StringResultModel(result=""),
        ]
        res = await pipe.run({})
        mock_run.assert_awaited_with(**{})
        mock_run.assert_awaited_with(**{})
    assert "b" in res
    assert res["b"] == {"result": ""}


@pytest.mark.asyncio
async def test_pipeline_parameter_propagation() -> None:
    pipe = Pipeline()
    component_a = ComponentPassThrough()
    component_b = ComponentPassThrough()
    pipe.add_component("a", component_a)
    pipe.add_component("b", component_b)
    # first component output product goes to second component input number1
    pipe.connect(
        "a",
        "b",
        {
            "value": "a.result",
        },
    )
    with mock.patch(
        "tests.unit.pipeline.test_pipeline.ComponentPassThrough.run"
    ) as mock_run:
        mock_run.side_effect = [
            StringResultModel(result="text"),
            StringResultModel(result="text"),
        ]
        res = await pipe.run({"a": {"value": "text"}})
        mock_run.assert_awaited_with(**{"value": "text"})
        mock_run.assert_awaited_with(**{"value": "text"})
    assert res == {"b": {"result": "text"}}


@pytest.mark.asyncio
async def test_pipeline_branches() -> None:
    pipe = Pipeline()
    component_a = AsyncMock(spec=Component)
    component_a.run = AsyncMock(return_value={})
    component_b = AsyncMock(spec=Component)
    component_b.run = AsyncMock(return_value={})
    component_c = AsyncMock(spec=Component)
    component_c.run = AsyncMock(return_value={})

    pipe.add_component("a", component_a)
    pipe.add_component("b", component_b)
    pipe.add_component("c", component_c)
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

    pipe.add_component("a", component_a)
    pipe.add_component("b", component_b)
    pipe.add_component("c", component_c)
    pipe.connect("a", "c")
    pipe.connect("b", "c")
    res = await pipe.run({})
    assert "c" in res


@pytest.mark.asyncio
async def test_pipeline_missing_param_on_init() -> None:
    pipe = Pipeline()
    component_a = ComponentAdd()
    component_b = ComponentAdd()
    pipe.add_component("a", component_a)
    pipe.add_component("b", component_b)
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
    pipe.add_component("a", component_a)
    pipe.add_component("b", component_b)
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
    pipe.add_component("a", component_a)
    pipe.add_component("b", component_b)
    pipe.connect("a", "b", {"number1": "a.result"})
    res = await pipe.run({"a": {"number1": 1, "number2": 2}})
    assert res == {"b": {"result": 6}}  # (1+2)*2


@pytest.mark.asyncio
async def test_pipeline_cycle() -> None:
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentNoParam()
    pipe.add_component("a", component_a)
    pipe.add_component("b", component_b)
    pipe.connect("a", "b", {})
    with pytest.raises(PipelineDefinitionError) as excinfo:
        pipe.connect("b", "a", {})
        assert "Cycles are not allowed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_pipeline_wrong_component_name() -> None:
    pipe = Pipeline()
    component_a = ComponentNoParam()
    component_b = ComponentNoParam()
    pipe.add_component("a", component_a)
    pipe.add_component("b", component_b)
    with pytest.raises(PipelineDefinitionError) as excinfo:
        pipe.connect("a", "c", {})
        assert "a or c not in the Pipeline" in str(excinfo.value)
