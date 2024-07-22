from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from neo4j_genai.core.pipeline import Component, Pipeline


@pytest.mark.asyncio
async def test_simple_pipeline_two_components() -> None:
    pipe = Pipeline()
    component_a = AsyncMock(spec=Component)
    component_a.run = AsyncMock(return_value={})
    component_b = AsyncMock(spec=Component)
    component_b.run = AsyncMock(return_value={"result": "result"})
    pipe.add_component("a", component_a)
    pipe.add_component("b", component_b)
    pipe.connect("a", "b", {})
    res = await pipe.run({})
    component_a.run.assert_awaited_once_with(**{})
    component_b.run.assert_awaited_once_with(**{})
    assert "b" in res
    assert res["b"] == {"result": "result"}


@pytest.mark.asyncio
async def test_simple_pipeline_two_components_parameter_propagation() -> None:
    pipe = Pipeline()
    component_a = AsyncMock(spec=Component)
    component_a.run = AsyncMock(return_value={"product": 20})
    component_b = AsyncMock(spec=Component)
    component_b.run = AsyncMock(return_value={"sum": 54})
    pipe.add_component("a", component_a)
    pipe.add_component("b", component_b)
    # first component output product goes to second component input number1
    pipe.connect(
        "a",
        "b",
        {
            "number1": "a.product",
        },
    )
    res = await pipe.run({"a": {}, "b": {"number2": 1}})
    component_a.run.assert_awaited_once_with(**{})
    component_b.run.assert_awaited_once_with(**{"number1": 20, "number2": 1})
    assert res == {"b": {"sum": 54}}


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
