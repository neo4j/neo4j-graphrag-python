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
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import ollama
import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.ollama_llm import OllamaLLM
from neo4j_graphrag.llm.types import ToolCallResponse
from neo4j_graphrag.tool import Tool


def get_mock_ollama() -> MagicMock:
    mock = MagicMock()
    mock.ResponseError = ollama.ResponseError
    return mock


@patch("builtins.__import__", side_effect=ImportError)
def test_ollama_llm_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        OllamaLLM(model_name="gpt-4o")


@patch("builtins.__import__")
def test_ollama_llm_happy_path_deprecated_options(mock_import: Mock) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama
    mock_ollama.Client.return_value.chat.return_value = MagicMock(
        message=MagicMock(content="ollama chat response"),
    )
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
    res = llm.invoke(question)
    assert isinstance(res, LLMResponse)
    assert res.content == "ollama chat response"
    messages = [
        {"role": "user", "content": question},
    ]
    llm.client.chat.assert_called_once_with(  # type: ignore[attr-defined]
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
    llm = OllamaLLM(
        model_name=model,
        model_params=model_params,
    )
    res = llm.invoke(question)
    assert isinstance(res, LLMResponse)
    assert res.content == "ollama chat response"
    messages = [
        {"role": "user", "content": question},
    ]
    llm.client.chat.assert_called_once_with(  # type: ignore[attr-defined]
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

    response = llm.invoke(question, system_instruction=system_instruction)
    assert response.content == "ollama chat response"
    messages = [{"role": "system", "content": system_instruction}]
    messages.append({"role": "user", "content": question})
    llm.client.chat.assert_called_once_with(  # type: ignore[attr-defined]
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

    response = llm.invoke(question, message_history)  # type: ignore
    assert response.content == "ollama chat response"
    messages = [m for m in message_history]
    messages.append({"role": "user", "content": question})
    llm.client.chat.assert_called_once_with(  # type: ignore[attr-defined]
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

    response = llm.invoke(
        question,
        message_history,  # type: ignore
        system_instruction=system_instruction,
    )
    assert response.content == "ollama chat response"
    messages = [{"role": "system", "content": system_instruction}]
    messages.extend(message_history)
    messages.append({"role": "user", "content": question})
    llm.client.chat.assert_called_once_with(  # type: ignore[attr-defined]
        model=model, messages=messages, options=options
    )
    assert llm.client.chat.call_count == 1  # type: ignore


@patch("builtins.__import__")
def test_ollama_invoke_with_message_history_validation_error(mock_import: Mock) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama
    mock_ollama.ResponseError = ollama.ResponseError
    model = "gpt"
    options = {"temperature": 0.3}
    model_params = {"options": options}
    system_instruction = "You are a helpful assistant."
    llm = OllamaLLM(
        model,
        model_params=model_params,
        system_instruction=system_instruction,
    )
    message_history = [
        {"role": "human", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    with pytest.raises(LLMGenerationError) as exc_info:
        llm.invoke(question, message_history)  # type: ignore
    assert "Input should be 'user', 'assistant' or 'system" in str(exc_info.value)


@pytest.mark.asyncio
@patch("builtins.__import__")
async def test_ollama_ainvoke_happy_path(mock_import: Mock) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama

    async def mock_chat_async(*args: Any, **kwargs: Any) -> MagicMock:
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

    res = await llm.ainvoke(question)
    assert isinstance(res, LLMResponse)
    assert res.content == "ollama chat response"


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
    llm.client.chat.assert_called_once()  # type: ignore
    # Check call arguments individually
    call_args = llm.client.chat.call_args[  # type: ignore
        1
    ]  # Get the keyword arguments
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
    llm.client.chat.assert_called_once()  # type: ignore
    # Check call arguments individually
    call_args = llm.client.chat.call_args[  # type: ignore
        1
    ]  # Get the keyword arguments
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
