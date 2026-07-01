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

import sys
import warnings
from typing import Generator, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.litellm_llm import LiteLLMChat
from neo4j_graphrag.types import LLMMessage
from pydantic import BaseModel, ConfigDict


@pytest.fixture
def mock_litellm() -> Generator[MagicMock, None, None]:
    mock_litellm = MagicMock()
    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        yield mock_litellm


def _make_completion_response(
    text: str = "litellm response text",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    total_tokens: int = 30,
) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.choices[0].message.tool_calls = None
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = total_tokens
    return response


@patch("builtins.__import__", side_effect=ImportError)
def test_litellm_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        LiteLLMChat(model_name="something")


def test_litellm_happy_path(mock_litellm: MagicMock) -> None:
    mock_litellm.completion.return_value = _make_completion_response()
    llm = LiteLLMChat(model_name="gpt-4o")
    res = llm.invoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "litellm response text"
    assert res.usage is not None
    assert res.usage.request_tokens == 10
    assert res.usage.response_tokens == 20
    assert res.usage.total_tokens == 30


def test_litellm_invoke_with_message_history(mock_litellm: MagicMock) -> None:
    mock_litellm.completion.return_value = _make_completion_response()
    llm = LiteLLMChat(model_name="gpt-4o")
    message_history: List[LLMMessage] = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    system_instruction = "You are a helpful assistant."
    res = llm.invoke("What about next season?", message_history, system_instruction)
    assert isinstance(res, LLMResponse)
    assert res.content == "litellm response text"
    call_args = mock_litellm.completion.call_args
    messages = call_args[1]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_instruction
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "What about next season?"


def test_litellm_invoke_with_message_history_validation_error(
    mock_litellm: MagicMock,
) -> None:
    mock_litellm.completion.return_value = _make_completion_response()
    llm = LiteLLMChat(model_name="gpt-4o")
    message_history = [
        {"role": "robot", "content": "Invalid role"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    with pytest.raises(LLMGenerationError) as exc_info:
        llm.invoke("test", message_history)  # type: ignore
    assert "Input should be 'user', 'assistant' or 'system" in str(exc_info.value)


@pytest.mark.asyncio
async def test_litellm_happy_path_async(mock_litellm: MagicMock) -> None:
    mock_litellm.acompletion = AsyncMock(return_value=_make_completion_response())
    llm = LiteLLMChat(model_name="gpt-4o")
    res = await llm.ainvoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "litellm response text"


def test_litellm_invoke_failed(mock_litellm: MagicMock) -> None:
    mock_litellm.completion.side_effect = Exception("API Error")
    llm = LiteLLMChat(model_name="gpt-4o")
    with pytest.raises(LLMGenerationError):
        llm.invoke("my text")


@pytest.mark.asyncio
async def test_litellm_ainvoke_failed(mock_litellm: MagicMock) -> None:
    mock_litellm.acompletion = AsyncMock(side_effect=Exception("API Error"))
    llm = LiteLLMChat(model_name="gpt-4o")
    with pytest.raises(LLMGenerationError):
        await llm.ainvoke("my text")


# V2 Interface Tests


def test_litellm_invoke_v2_happy_path(mock_litellm: MagicMock) -> None:
    mock_litellm.completion.return_value = _make_completion_response(
        "litellm v2 response"
    )
    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    llm = LiteLLMChat(model_name="gpt-4o")
    response = llm.invoke(messages)
    assert isinstance(response, LLMResponse)
    assert response.content == "litellm v2 response"
    call_args = mock_litellm.completion.call_args
    assert call_args[1]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_litellm_ainvoke_v2_happy_path(mock_litellm: MagicMock) -> None:
    mock_litellm.acompletion = AsyncMock(
        return_value=_make_completion_response("litellm v2 async response")
    )
    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    llm = LiteLLMChat(model_name="gpt-4o")
    response = await llm.ainvoke(messages)
    assert isinstance(response, LLMResponse)
    assert response.content == "litellm v2 async response"


def test_litellm_invoke_v2_with_response_format(mock_litellm: MagicMock) -> None:
    mock_litellm.completion.return_value = _make_completion_response(
        '{"value": "test"}'
    )

    class TestModel(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str

    messages: List[LLMMessage] = [{"role": "user", "content": "Test"}]
    llm = LiteLLMChat(model_name="gpt-4o")
    response = llm.invoke(messages, response_format=TestModel)
    assert isinstance(response, LLMResponse)
    call_args = mock_litellm.completion.call_args
    rf = call_args[1]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "TestModel"


def test_litellm_invoke_v2_with_dict_response_format(
    mock_litellm: MagicMock,
) -> None:
    mock_litellm.completion.return_value = _make_completion_response("{}")
    messages: List[LLMMessage] = [{"role": "user", "content": "Test"}]
    llm = LiteLLMChat(model_name="gpt-4o")
    response = llm.invoke(messages, response_format={"type": "json_object"})
    assert isinstance(response, LLMResponse)
    call_args = mock_litellm.completion.call_args
    assert call_args[1]["response_format"] == {"type": "json_object"}


def test_litellm_kwargs_passed_through(mock_litellm: MagicMock) -> None:
    mock_litellm.completion.return_value = _make_completion_response()
    llm = LiteLLMChat(
        model_name="gpt-4o",
        api_key="test-key",
        drop_params=True,
    )
    llm.invoke("test")
    call_args = mock_litellm.completion.call_args
    assert call_args[1]["api_key"] == "test-key"
    assert call_args[1]["drop_params"] is True


def test_litellm_model_params_passed_through(mock_litellm: MagicMock) -> None:
    mock_litellm.completion.return_value = _make_completion_response()
    llm = LiteLLMChat(
        model_name="gpt-4o",
        model_params={"temperature": 0.5, "max_tokens": 100},
    )
    llm.invoke("test")
    call_args = mock_litellm.completion.call_args
    assert call_args[1]["temperature"] == 0.5
    assert call_args[1]["max_tokens"] == 100


def test_litellm_invoke_with_tools(mock_litellm: MagicMock) -> None:
    tool_call_mock = MagicMock()
    tool_call_mock.function.name = "get_weather"
    tool_call_mock.function.arguments = '{"city": "Paris"}'

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = None
    response.choices[0].message.tool_calls = [tool_call_mock]
    mock_litellm.completion.return_value = response

    tool = MagicMock()
    tool.get_name.return_value = "get_weather"
    tool.get_description.return_value = "Get weather for a city"
    tool.get_parameters.return_value = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }

    llm = LiteLLMChat(model_name="gpt-4o")
    res = llm.invoke_with_tools("What is the weather in Paris?", tools=[tool])
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "get_weather"
    assert res.tool_calls[0].arguments == {"city": "Paris"}


@pytest.mark.asyncio
async def test_litellm_ainvoke_with_tools(mock_litellm: MagicMock) -> None:
    tool_call_mock = MagicMock()
    tool_call_mock.function.name = "get_weather"
    tool_call_mock.function.arguments = '{"city": "Paris"}'

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = None
    response.choices[0].message.tool_calls = [tool_call_mock]
    mock_litellm.acompletion = AsyncMock(return_value=response)

    tool = MagicMock()
    tool.get_name.return_value = "get_weather"
    tool.get_description.return_value = "Get weather for a city"
    tool.get_parameters.return_value = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }

    llm = LiteLLMChat(model_name="gpt-4o")
    res = await llm.ainvoke_with_tools("What is the weather in Paris?", tools=[tool])
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "get_weather"
    assert res.tool_calls[0].arguments == {"city": "Paris"}


def test_litellm_close(mock_litellm: MagicMock) -> None:
    llm = LiteLLMChat(model_name="gpt-4o")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        llm.close()


@pytest.mark.asyncio
async def test_litellm_aclose(mock_litellm: MagicMock) -> None:
    llm = LiteLLMChat(model_name="gpt-4o")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await llm.aclose()
