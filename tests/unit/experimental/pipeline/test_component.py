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
from unittest.mock import AsyncMock

import pytest

from neo4j_graphrag.experimental.pipeline import Component
from neo4j_graphrag.experimental.pipeline.types.context import RunContext
from .components import ComponentMultiply, ComponentMultiplyWithContext, IntResultModel


def test_component_inputs() -> None:
    inputs = ComponentMultiply.component_inputs
    assert "number1" in inputs
    assert inputs["number1"]["has_default"] is False
    assert "number2" in inputs
    assert inputs["number2"]["has_default"] is True


def test_component_outputs() -> None:
    outputs = ComponentMultiply.component_outputs
    assert "result" in outputs
    assert outputs["result"]["has_default"] is True
    assert outputs["result"]["annotation"] == int


@pytest.mark.asyncio
async def test_component_run() -> None:
    c = ComponentMultiply()
    result = await c.run(number1=1, number2=2)
    assert isinstance(result, IntResultModel)
    assert isinstance(
        result.result,
        # we know this is a type and not a bool or str:
        ComponentMultiply.component_outputs["result"]["annotation"],  # type: ignore
    )


@pytest.mark.asyncio
async def test_component_run_with_context_default_implementation() -> None:
    c = ComponentMultiply()
    result = await c.run_with_context(
        # context can not be null in the function signature,
        # but it's ignored in this case
        None,  # type: ignore
        number1=1,
        number2=2,
    )
    # the type checker doesn't know about the type
    # because the method is not re-declared
    assert result.result == 2  # type: ignore


@pytest.mark.asyncio
async def test_component_run_with_context() -> None:
    c = ComponentMultiplyWithContext()
    notifier_mock = AsyncMock()
    result = await c.run_with_context(
        RunContext(run_id="run_id", task_name="task_name", _notifier=notifier_mock),
        number1=1,
        number2=2,
    )
    assert result.result == 2
    notifier_mock.assert_awaited_once()


def test_component_missing_method() -> None:
    with pytest.raises(RuntimeError) as e:

        class WrongComponent(Component):
            # we must have either run or run_with_context
            pass

    assert (
        "You must implement either `run` or `run_with_context` in Component 'WrongComponent'"
        in str(e)
    )
