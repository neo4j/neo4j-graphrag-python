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
import datetime
import tempfile
from typing import Sized
from unittest import mock
from unittest.mock import AsyncMock, call, patch

import pytest
from neo4j_graphrag.experimental.pipeline import Component, Pipeline
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
from neo4j_graphrag.experimental.pipeline.notification import (
    EventCallbackProtocol,
    EventType,
    PipelineEvent,
    TaskEvent,
    Event,
)
from neo4j_graphrag.experimental.pipeline.types.orchestration import RunResult

from .components import (
    ComponentAdd,
    ComponentMultiply,
    ComponentNoParam,
    ComponentPassThrough,
    StringResultModel,
    SlowComponentMultiply,
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
    component_a.run_with_context = AsyncMock(return_value={})
    component_b = AsyncMock(spec=Component)
    component_b.run_with_context = AsyncMock(return_value={})
    component_c = AsyncMock(spec=Component)
    component_c.run_with_context = AsyncMock(return_value={})

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
    component_a.run_with_context = AsyncMock(return_value={})
    component_b = AsyncMock(spec=Component)
    component_b.run_with_context = AsyncMock(return_value={})
    component_c = AsyncMock(spec=Component)
    component_c.run_with_context = AsyncMock(return_value={})

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


def test_run_result_no_warning(recwarn: Sized) -> None:
    RunResult()
    assert len(recwarn) == 0


@pytest.mark.asyncio
async def test_pipeline_event_notification() -> None:
    callback = AsyncMock(spec=EventCallbackProtocol)
    pipe = Pipeline(callback=callback)
    component_a = ComponentMultiply()
    pipe.add_component(
        component_a,
        "a",
    )
    a_input_data = {"number1": 2, "number2": 3}
    pipeline_result = await pipe.run({"a": a_input_data})

    await_calls = callback.await_args_list

    expected_event_list = [
        PipelineEvent(
            event_type=EventType.PIPELINE_STARTED,
            run_id=pipeline_result.run_id,
            timestamp=datetime.datetime.now(),
            message=None,
            payload={"a": a_input_data},
        ),
        TaskEvent(
            event_type=EventType.TASK_STARTED,
            run_id=pipeline_result.run_id,
            task_name="a",
            timestamp=datetime.datetime.now(),
            message=None,
            payload=a_input_data,
        ),
        TaskEvent(
            event_type=EventType.TASK_FINISHED,
            run_id=pipeline_result.run_id,
            task_name="a",
            timestamp=datetime.datetime.now(),
            message=None,
            payload={"result": 6},
        ),
        PipelineEvent(
            event_type=EventType.PIPELINE_FINISHED,
            run_id=pipeline_result.run_id,
            timestamp=datetime.datetime.now(),
            message=None,
            payload={"a": {"result": 6}},
        ),
    ]
    assert len(await_calls) == len(expected_event_list)

    previous_ts = None
    for await_call, expected_event in zip(await_calls, expected_event_list):
        actual_event = await_call[0][0]
        assert isinstance(actual_event, type(expected_event))
        assert actual_event.event_type == expected_event.event_type
        assert actual_event.run_id == expected_event.run_id
        assert actual_event.message == expected_event.message
        assert actual_event.payload == expected_event.payload
        if previous_ts:
            assert actual_event.timestamp > previous_ts
        previous_ts = actual_event.timestamp


def test_event_model_no_warning(recwarn: Sized) -> None:
    event = Event(
        event_type=EventType.PIPELINE_STARTED,
        run_id="run_id",
        message=None,
        payload=None,
    )
    assert event.timestamp is not None
    assert len(recwarn) == 0


@pytest.mark.asyncio
async def test_pipeline_streaming_no_user_callback_happy_path() -> None:
    pipe = Pipeline()
    events = []
    async for e in pipe.stream({}):
        events.append(e)
    assert len(events) == 2
    assert events[0].event_type == EventType.PIPELINE_STARTED
    assert events[1].event_type == EventType.PIPELINE_FINISHED
    assert len(pipe.callbacks) == 0


@pytest.mark.asyncio
async def test_pipeline_streaming_with_user_callback_happy_path() -> None:
    callback = AsyncMock()
    pipe = Pipeline(callback=callback)
    events = []
    async for e in pipe.stream({}):
        events.append(e)
    assert len(events) == 2
    assert len(callback.call_args_list) == 2
    assert len(pipe.callbacks) == 1


@pytest.mark.asyncio
async def test_pipeline_streaming_very_long_running_user_callback() -> None:
    async def callback(event: Event) -> None:
        await asyncio.sleep(2)

    pipe = Pipeline(callback=callback)
    events = []
    async for e in pipe.stream({}):
        events.append(e)
    assert len(events) == 2
    assert len(pipe.callbacks) == 1


@pytest.mark.asyncio
async def test_pipeline_streaming_very_long_running_pipeline() -> None:
    slow_component = SlowComponentMultiply()
    pipe = Pipeline()
    pipe.add_component(slow_component, "slow_component")
    events = []
    async for e in pipe.stream({"slow_component": {"number1": 1, "number2": 2}}):
        events.append(e)
    assert len(events) == 4
    last_event = events[-1]
    assert last_event.event_type == EventType.PIPELINE_FINISHED
    assert last_event.payload == {"slow_component": {"result": 2}}


@pytest.mark.asyncio
async def test_pipeline_streaming_error_in_pipeline_definition() -> None:
    pipe = Pipeline()
    component_a = ComponentAdd()
    component_b = ComponentAdd()
    pipe.add_component(component_a, "a")
    pipe.add_component(component_b, "b")
    pipe.connect("a", "b", {"number1": "a.result"})
    events = []
    with pytest.raises(PipelineDefinitionError):
        async for e in pipe.stream({"a": {"number1": 1, "number2": 2}}):
            events.append(e)
    # validation happens before pipeline run actually starts
    assert len(events) == 0


@pytest.mark.asyncio
async def test_pipeline_streaming_error_in_component() -> None:
    component = ComponentMultiply()
    pipe = Pipeline()
    pipe.add_component(component, "component")
    events = []
    with pytest.raises(TypeError):
        async for e in pipe.stream({"component": {"number1": None, "number2": 2}}):
            events.append(e)
    assert len(events) == 2
    assert events[0].event_type == EventType.PIPELINE_STARTED
    assert events[1].event_type == EventType.TASK_STARTED


@pytest.mark.asyncio
async def test_pipeline_streaming_error_in_user_callback() -> None:
    async def callback(event: Event) -> None:
        raise Exception("error in callback")

    pipe = Pipeline(callback=callback)
    events = []
    async for e in pipe.stream({}):
        events.append(e)
    assert len(events) == 2
    assert len(pipe.callbacks) == 1
