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
import warnings
from typing import Any, List, cast
from unittest.mock import MagicMock, Mock, patch

import ollama
import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.ollama_llm import OllamaLLM
from neo4j_graphrag.llm.types import ToolCallResponse
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage
from pydantic import BaseModel, ConfigDict


def get_mock_ollama() -> MagicMock:
    mock = MagicMock()
    mock.ResponseError = ollama.ResponseError
    return mock


def _as_mock(value: Any) -> MagicMock:
    return cast(MagicMock, value)


@patch("builtins.__import__", side_effect=ImportError)
def test_ollama_llm_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        OllamaLLM(model_name="llama3.2")


@patch("builtins.__import__")
def test_ollama_llm_happy_path_deprecated_options(mock_import: Mock) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama
    mock_ollama.Client.return_value.chat.return_value = MagicMock(
        message=MagicMock(content="ollama chat response"),
    )
    mock_ollama.Message = MagicMock(side_effect=lambda **kw: kw)
    model = "gpt"
    model_params = {"temperature": 0.3}
    with pytest.warns(DeprecationWarning) as record:
        llm = OllamaLLM(
            model,
            model_params=model_params,
        )
    assert len(record) == 1
    assert isinstance(record[0].message, Warning)
    assert (
        'you must use model_params={"options": {"temperature": 0}}'
        in record[0].message.args[0]
    )

    question = "What is graph RAG?"
    res = llm.invoke([{"role": "user", "content": question}])
    assert isinstance(res, LLMResponse)
    assert res.content == "ollama chat response"
    messages = [
        {"role": "user", "content": question},
    ]
    _as_mock(llm.client.chat).assert_called_once_with(
        model=model, messages=messages, options={"temperature": 0.3}
    )


@patch("builtins.__import__")
def test_ollama_llm_unsupported_streaming(mock_import: Mock) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama
    mock_ollama.Client.return_value.chat.return_value = MagicMock(
        message=MagicMock(content="ollama chat response"),
    )
    model = "gpt"
    model_params = {"stream": True}
    with pytest.raises(ValueError):
        OllamaLLM(
            model,
            model_params=model_params,
        )


@patch("builtins.__import__")
def test_ollama_llm_happy_path(mock_import: Mock) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama
    mock_ollama.Client.return_value.chat.return_value = MagicMock(
        message=MagicMock(content="ollama chat response"),
    )
    model = "gpt"
    options = {"temperature": 0.3}
    model_params = {"options": options, "format": "json"}
    question = "What is graph RAG?"
    mock_ollama.Message = MagicMock(side_effect=lambda **kw: kw)
    llm = OllamaLLM(
        model_name=model,
        model_params=model_params,
    )
    res = llm.invoke([{"role": "user", "content": question}])
    assert isinstance(res, LLMResponse)
    assert res.content == "ollama chat response"
    messages = [
        {"role": "user", "content": question},
    ]
    _as_mock(llm.client.chat).assert_called_once_with(
        model=model,
        messages=messages,
        options=options,
        format="json",
    )


@patch("builtins.__import__")
def test_ollama_invoke_with_system_instruction_happy_path(mock_import: Mock) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama
    mock_ollama.Client.return_value.chat.return_value = MagicMock(
        message=MagicMock(content="ollama chat response"),
    )
    model = "gpt"
    options = {"temperature": 0.3}
    model_params = {"options": options, "format": "json"}
    llm = OllamaLLM(
        model,
        model_params=model_params,
    )
    system_instruction = "You are a helpful assistant."
    question = "What about next season?"
    mock_ollama.Message = MagicMock(side_effect=lambda **kw: kw)

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": question},
    ]
    response = llm.invoke(messages)  # type: ignore[arg-type]
    assert response.content == "ollama chat response"
    _as_mock(llm.client.chat).assert_called_once_with(
        model=model,
        messages=messages,
        options=options,
        format="json",
    )


@patch("builtins.__import__")
def test_ollama_invoke_with_message_history_happy_path(mock_import: Mock) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama
    mock_ollama.Client.return_value.chat.return_value = MagicMock(
        message=MagicMock(content="ollama chat response"),
    )
    model = "gpt"
    options = {"temperature": 0.3}
    model_params = {"options": options}
    llm = OllamaLLM(
        model,
        model_params=model_params,
    )
    message_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"
    mock_ollama.Message = MagicMock(side_effect=lambda **kw: kw)

    messages = [m for m in message_history]
    messages.append({"role": "user", "content": question})
    response = llm.invoke(messages)  # type: ignore[arg-type]
    assert response.content == "ollama chat response"
    _as_mock(llm.client.chat).assert_called_once_with(
        model=model, messages=messages, options=options
    )


@patch("builtins.__import__")
def test_ollama_invoke_with_message_history_and_system_instruction(
    mock_import: Mock,
) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama
    mock_ollama.Client.return_value.chat.return_value = MagicMock(
        message=MagicMock(content="ollama chat response"),
    )
    model = "gpt"
    options = {"temperature": 0.3}
    model_params = {"options": options}
    system_instruction = "You are a helpful assistant."
    llm = OllamaLLM(
        model,
        model_params=model_params,
    )
    message_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"
    mock_ollama.Message = MagicMock(side_effect=lambda **kw: kw)

    messages = [{"role": "system", "content": system_instruction}]
    messages.extend(message_history)
    messages.append({"role": "user", "content": question})
    response = llm.invoke(messages)  # type: ignore[arg-type]
    assert response.content == "ollama chat response"
    _as_mock(llm.client.chat).assert_called_once_with(
        model=model, messages=messages, options=options
    )
    assert _as_mock(llm.client.chat).call_count == 1


@pytest.mark.asyncio
@patch("builtins.__import__")
async def test_ollama_ainvoke_happy_path(mock_import: Mock) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama

    async def mock_chat_async(*_args: Any, **_kwargs: Any) -> MagicMock:
        return MagicMock(
            message=MagicMock(content="ollama chat response"),
        )

    mock_ollama.AsyncClient.return_value.chat = mock_chat_async
    model = "gpt"
    options = {"temperature": 0.3}
    model_params = {"options": options}
    question = "What is graph RAG?"
    llm = OllamaLLM(
        model,
        model_params=model_params,
    )

    res = await llm.ainvoke([{"role": "user", "content": question}])
    assert isinstance(res, LLMResponse)
    assert res.content == "ollama chat response"


@pytest.mark.asyncio
@patch("builtins.__import__")
async def test_ollama_ainvoke_spreads_model_params_like_sync(
    mock_import: Mock,
) -> None:
    """Regression test: async path must spread model_params the same way
    sync does, not pass it verbatim as the `options` value -- otherwise a
    model_params carrying a sibling key alongside "options" (e.g. "format")
    gets double-nested under `options=` instead of reaching the SDK call
    as its own top-level kwarg.
    """
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama

    captured_kwargs: dict[str, Any] = {}

    async def mock_chat_async(*_args: Any, **kwargs: Any) -> MagicMock:
        captured_kwargs.update(kwargs)
        return MagicMock(
            message=MagicMock(content="ollama chat response"),
        )

    mock_ollama.AsyncClient.return_value.chat = mock_chat_async
    options = {"temperature": 0.3}
    model_params = {"options": options, "format": "json"}
    llm = OllamaLLM(
        "gpt",
        model_params=model_params,
    )

    await llm.ainvoke([{"role": "user", "content": "What is graph RAG?"}])

    assert captured_kwargs["options"] == options
    assert captured_kwargs["format"] == "json"


@patch("builtins.__import__")
def test_ollama_llm_invoke_happy_path(mock_import: Mock) -> None:
    """Test invoke method with List[LLMMessage] input."""
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama
    mock_ollama.Client.return_value.chat.return_value = MagicMock(
        message=MagicMock(content="ollama v2 response"),
    )
    mock_ollama.Message = MagicMock()

    model = "llama2"
    options = {"temperature": 0.3}
    model_params = {"options": options}

    messages: list[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is graph RAG?"},
    ]

    llm = OllamaLLM(
        model_name=model,
        model_params=model_params,
    )
    res = llm.invoke(messages)

    assert isinstance(res, LLMResponse)
    assert res.content == "ollama v2 response"

    # Verify messages were built correctly
    assert mock_ollama.Message.call_count == 2
    mock_ollama.Message.assert_any_call(**messages[0])
    mock_ollama.Message.assert_any_call(**messages[1])

    # Verify the client was called with correct parameters
    _as_mock(llm.client.chat).assert_called_once_with(
        model=model,
        messages=[mock_ollama.Message.return_value, mock_ollama.Message.return_value],
        options=options,
    )


@patch("builtins.__import__")
def test_ollama_llm_get_messages_all_roles(mock_import: Mock) -> None:
    """Test build_llm_messages method handles all message roles correctly."""
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama
    mock_ollama.Message = MagicMock()

    messages: list[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    llm = OllamaLLM(model_name="llama2")
    result_messages = llm.build_llm_messages(messages)

    # Convert to list for easier testing
    result_list = list(result_messages)

    # Verify correct number of ollama.Message objects created
    assert len(result_list) == 4
    assert mock_ollama.Message.call_count == 4

    # Verify each message was converted properly
    for message in messages:
        mock_ollama.Message.assert_any_call(**message)


@patch("builtins.__import__")
def test_ollama_llm_invoke_with_tools_happy_path(
    mock_import: Mock,
    test_tool: Tool,
) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama

    # Mock the tool call response
    mock_function = MagicMock()
    mock_function.name = "test_tool"
    mock_function.arguments = {"param1": "value1"}

    mock_tool_call = MagicMock()
    mock_tool_call.function = mock_function

    mock_ollama.Client.return_value.chat.return_value = MagicMock(
        message=MagicMock(content="ollama tool response", tool_calls=[mock_tool_call])
    )

    llm = OllamaLLM(model_name="gpt", model_params={"options": {"temperature": 0}})
    tools = [test_tool]

    res = llm.invoke_with_tools("my text", tools)
    assert isinstance(res, ToolCallResponse)
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "test_tool"
    assert res.tool_calls[0].arguments == {"param1": "value1"}
    assert res.content == "ollama tool response"


@patch("builtins.__import__")
def test_ollama_llm_invoke_with_tools_with_message_history(
    mock_import: Mock,
    test_tool: Tool,
) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama

    # Mock the tool call response
    mock_function = MagicMock()
    mock_function.name = "test_tool"
    mock_function.arguments = {"param1": "value1"}

    mock_tool_call = MagicMock()
    mock_tool_call.function = mock_function
    mock_ollama.Client.return_value.chat.return_value = MagicMock(
        message=MagicMock(content="ollama tool response", tool_calls=[mock_tool_call])
    )
    llm = OllamaLLM(
        api_key="my key", model_name="gpt", model_params={"options": {"temperature": 0}}
    )
    tools = [test_tool]

    message_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    res = llm.invoke_with_tools(question, tools, message_history)  # type: ignore
    assert isinstance(res, ToolCallResponse)
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "test_tool"
    assert res.tool_calls[0].arguments == {"param1": "value1"}

    # Verify the correct messages were passed
    message_history.append({"role": "user", "content": question})
    # Use assert_called_once() instead of assert_called_once_with() to avoid issues with overloaded functions
    _as_mock(llm.client.chat).assert_called_once()
    # Check call arguments individually
    call_args = _as_mock(llm.client.chat).call_args[1]  # Get the keyword arguments
    assert call_args["messages"] == message_history
    assert call_args["model"] == "gpt"
    # Check tools content rather than direct equality
    assert len(call_args["tools"]) == 1
    assert call_args["tools"][0]["type"] == "function"
    assert call_args["tools"][0]["function"]["name"] == "test_tool"
    assert call_args["tools"][0]["function"]["description"] == "A test tool"


@patch("builtins.__import__")
def test_ollama_llm_invoke_with_tools_with_system_instruction(
    mock_import: Mock,
    test_tool: Mock,
) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama

    # Mock the tool call response
    mock_function = MagicMock()
    mock_function.name = "test_tool"
    mock_function.arguments = {"param1": "value1"}

    mock_tool_call = MagicMock()
    mock_tool_call.function = mock_function

    mock_ollama.Client.return_value.chat.return_value = MagicMock(
        message=MagicMock(content="ollama tool response", tool_calls=[mock_tool_call])
    )

    llm = OllamaLLM(
        api_key="my key", model_name="gpt", model_params={"options": {"temperature": 0}}
    )
    tools = [test_tool]

    system_instruction = "You are a helpful assistant."

    res = llm.invoke_with_tools("my text", tools, system_instruction=system_instruction)
    assert isinstance(res, ToolCallResponse)

    # Verify system instruction was included
    messages = [{"role": "system", "content": system_instruction}]
    messages.append({"role": "user", "content": "my text"})
    # Use assert_called_once() instead of assert_called_once_with() to avoid issues with overloaded functions
    _as_mock(llm.client.chat).assert_called_once()
    # Check call arguments individually
    call_args = _as_mock(llm.client.chat).call_args[1]  # Get the keyword arguments
    assert call_args["messages"] == messages
    assert call_args["model"] == "gpt"
    # Check tools content rather than direct equality
    assert len(call_args["tools"]) == 1
    assert call_args["tools"][0]["type"] == "function"
    assert call_args["tools"][0]["function"]["name"] == "test_tool"
    assert call_args["tools"][0]["function"]["description"] == "A test tool"


@patch("builtins.__import__")
def test_ollama_llm_invoke_with_tools_error(mock_import: Mock, test_tool: Tool) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama

    # Mock an Ollama response error
    mock_ollama.Client.return_value.chat.side_effect = ollama.ResponseError(
        "Test error"
    )

    llm = OllamaLLM(
        api_key="my key", model_name="gpt", model_params={"options": {"temperature": 0}}
    )
    tools = [test_tool]

    with pytest.raises(LLMGenerationError):
        llm.invoke_with_tools("my text", tools)


@pytest.mark.asyncio
@patch("builtins.__import__")
async def test_ollama_llm_ainvoke_with_tools_happy_path(
    mock_import: Mock, test_tool: Tool
) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama

    mock_function = MagicMock()
    mock_function.name = "test_tool"
    mock_function.arguments = {"param1": "value1"}

    mock_tool_call = MagicMock()
    mock_tool_call.function = mock_function

    async def mock_chat_async(*_args: Any, **_kwargs: Any) -> MagicMock:
        return MagicMock(
            message=MagicMock(
                content="ollama tool response", tool_calls=[mock_tool_call]
            )
        )

    mock_ollama.AsyncClient.return_value.chat = mock_chat_async

    llm = OllamaLLM(model_name="gpt", model_params={"options": {"temperature": 0}})
    res = await llm.ainvoke_with_tools("my text", [test_tool])

    assert isinstance(res, ToolCallResponse)
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "test_tool"
    assert res.tool_calls[0].arguments == {"param1": "value1"}
    assert res.content == "ollama tool response"


@pytest.mark.asyncio
@patch("builtins.__import__")
async def test_ollama_llm_ainvoke_with_tools_no_tool_calls(
    mock_import: Mock, test_tool: Tool
) -> None:
    """When the model returns no tool_calls, content is returned with empty tool_calls."""
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama

    async def mock_chat_async(*_args: Any, **_kwargs: Any) -> MagicMock:
        return MagicMock(message=MagicMock(content="plain content", tool_calls=[]))

    mock_ollama.AsyncClient.return_value.chat = mock_chat_async

    llm = OllamaLLM(model_name="gpt", model_params={"options": {"temperature": 0}})
    res = await llm.ainvoke_with_tools("my text", [test_tool])

    assert isinstance(res, ToolCallResponse)
    assert res.tool_calls == []
    assert res.content == "plain content"


@pytest.mark.asyncio
@patch("builtins.__import__")
async def test_ollama_llm_ainvoke_with_tools_error(
    mock_import: Mock, test_tool: Tool
) -> None:
    """ResponseError from the async client surfaces as LLMGenerationError."""
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama

    async def mock_chat_async_error(*_args: Any, **_kwargs: Any) -> None:
        raise ollama.ResponseError("async tool error")

    mock_ollama.AsyncClient.return_value.chat = mock_chat_async_error

    llm = OllamaLLM(model_name="gpt", model_params={"options": {"temperature": 0}})
    with pytest.raises(LLMGenerationError):
        await llm.ainvoke_with_tools("my text", [test_tool])


class _TestModelForOllama(BaseModel):
    """Test model for structured output tests."""

    model_config = ConfigDict(extra="forbid")
    value: str


@patch("builtins.__import__")
def test_ollama_invoke_with_response_format_raises_error(mock_import: Mock) -> None:
    """Test raises NotImplementedError when response_format is used."""
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama

    messages: List[LLMMessage] = [{"role": "user", "content": "Test"}]
    llm = OllamaLLM(model_name="llama2")

    with pytest.raises(NotImplementedError) as exc_info:
        llm.invoke(messages, response_format=_TestModelForOllama)

    assert "OllamaLLM does not currently support structured output" in str(
        exc_info.value
    )


@patch("builtins.__import__")
def test_ollama_llm_close(mock_import: Mock) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama

    llm = OllamaLLM(model_name="llama3.2", model_params={"options": {}})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        llm.close()


@pytest.mark.asyncio
@patch("builtins.__import__")
async def test_ollama_llm_aclose(mock_import: Mock) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama

    llm = OllamaLLM(model_name="llama3.2", model_params={"options": {}})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await llm.aclose()
