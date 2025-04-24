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
from unittest.mock import MagicMock, Mock, patch

import openai
import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.openai_llm import AzureOpenAILLM, OpenAILLM
from neo4j_graphrag.llm.types import ToolCallResponse
from neo4j_graphrag.tool import Tool


def get_mock_openai() -> MagicMock:
    mock = MagicMock()
    mock.OpenAIError = openai.OpenAIError
    return mock


@patch("builtins.__import__", side_effect=ImportError)
def test_openai_llm_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        OpenAILLM(model_name="gpt-4o")


@patch("builtins.__import__")
def test_openai_llm_happy_path(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="openai chat response"))],
    )
    llm = OpenAILLM(api_key="my key", model_name="gpt")

    res = llm.invoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "openai chat response"


@patch("builtins.__import__")
def test_openai_llm_with_message_history_happy_path(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="openai chat response"))],
    )
    llm = OpenAILLM(api_key="my key", model_name="gpt")
    message_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    res = llm.invoke(question, message_history)  # type: ignore
    assert isinstance(res, LLMResponse)
    assert res.content == "openai chat response"
    message_history.append({"role": "user", "content": question})
    # Use assert_called_once() instead of assert_called_once_with() to avoid issues with overloaded functions
    llm.client.chat.completions.create.assert_called_once()  # type: ignore
    # Check call arguments individually
    call_args = llm.client.chat.completions.create.call_args[  # type: ignore
        1
    ]  # Get the keyword arguments
    assert call_args["messages"] == message_history
    assert call_args["model"] == "gpt"


@patch("builtins.__import__")
def test_openai_llm_with_message_history_and_system_instruction(
    mock_import: Mock,
) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="openai chat response"))],
    )
    system_instruction = "You are a helpful assistent."
    llm = OpenAILLM(
        api_key="my key",
        model_name="gpt",
    )
    message_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    res = llm.invoke(question, message_history, system_instruction=system_instruction)  # type: ignore
    assert isinstance(res, LLMResponse)
    assert res.content == "openai chat response"
    messages = [{"role": "system", "content": system_instruction}]
    messages.extend(message_history)
    messages.append({"role": "user", "content": question})
    # Use assert_called_once() instead of assert_called_once_with() to avoid issues with overloaded functions
    llm.client.chat.completions.create.assert_called_once()  # type: ignore
    # Check call arguments individually
    call_args = llm.client.chat.completions.create.call_args[  # type: ignore
        1
    ]  # Get the keyword arguments
    assert call_args["messages"] == messages
    assert call_args["model"] == "gpt"

    assert llm.client.chat.completions.create.call_count == 1  # type: ignore


@patch("builtins.__import__")
def test_openai_llm_with_message_history_validation_error(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="openai chat response"))],
    )
    llm = OpenAILLM(api_key="my key", model_name="gpt")
    message_history = [
        {"role": "human", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    with pytest.raises(LLMGenerationError) as exc_info:
        llm.invoke(question, message_history)  # type: ignore
    assert "Input should be 'user', 'assistant' or 'system'" in str(exc_info.value)


@patch("builtins.__import__")
@patch("json.loads")
def test_openai_llm_invoke_with_tools_happy_path(
    mock_json_loads: Mock,
    mock_import: Mock,
    test_tool: Tool,
) -> None:
    # Set up json.loads to return a dictionary
    mock_json_loads.return_value = {"param1": "value1"}

    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai

    # Mock the tool call response
    mock_function = MagicMock()
    mock_function.name = "test_tool"
    mock_function.arguments = '{"param1": "value1"}'

    mock_tool_call = MagicMock()
    mock_tool_call.function = mock_function

    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="openai tool response", tool_calls=[mock_tool_call]
                )
            )
        ],
    )

    llm = OpenAILLM(api_key="my key", model_name="gpt")
    tools = [test_tool]

    res = llm.invoke_with_tools("my text", tools)
    assert isinstance(res, ToolCallResponse)
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "test_tool"
    assert res.tool_calls[0].arguments == {"param1": "value1"}
    assert res.content == "openai tool response"


@patch("builtins.__import__")
@patch("json.loads")
def test_openai_llm_invoke_with_tools_with_message_history(
    mock_json_loads: Mock,
    mock_import: Mock,
    test_tool: Tool,
) -> None:
    # Set up json.loads to return a dictionary
    mock_json_loads.return_value = {"param1": "value1"}

    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai

    # Mock the tool call response
    mock_function = MagicMock()
    mock_function.name = "test_tool"
    mock_function.arguments = '{"param1": "value1"}'

    mock_tool_call = MagicMock()
    mock_tool_call.function = mock_function

    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="openai tool response", tool_calls=[mock_tool_call]
                )
            )
        ],
    )

    llm = OpenAILLM(api_key="my key", model_name="gpt")
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
    llm.client.chat.completions.create.assert_called_once()  # type: ignore
    # Check call arguments individually
    call_args = llm.client.chat.completions.create.call_args[  # type: ignore
        1
    ]  # Get the keyword arguments
    assert call_args["messages"] == message_history
    assert call_args["model"] == "gpt"
    # Check tools content rather than direct equality
    assert len(call_args["tools"]) == 1
    assert call_args["tools"][0]["type"] == "function"
    assert call_args["tools"][0]["function"]["name"] == "test_tool"
    assert call_args["tools"][0]["function"]["description"] == "A test tool"
    assert call_args["tool_choice"] == "auto"
    assert call_args["temperature"] == 0.0


@patch("builtins.__import__")
@patch("json.loads")
def test_openai_llm_invoke_with_tools_with_system_instruction(
    mock_json_loads: Mock,
    mock_import: Mock,
    test_tool: Mock,
) -> None:
    # Set up json.loads to return a dictionary
    mock_json_loads.return_value = {"param1": "value1"}

    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai

    # Mock the tool call response
    mock_function = MagicMock()
    mock_function.name = "test_tool"
    mock_function.arguments = '{"param1": "value1"}'

    mock_tool_call = MagicMock()
    mock_tool_call.function = mock_function

    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="openai tool response", tool_calls=[mock_tool_call]
                )
            )
        ],
    )

    llm = OpenAILLM(api_key="my key", model_name="gpt")
    tools = [test_tool]

    system_instruction = "You are a helpful assistant."

    res = llm.invoke_with_tools("my text", tools, system_instruction=system_instruction)
    assert isinstance(res, ToolCallResponse)

    # Verify system instruction was included
    messages = [{"role": "system", "content": system_instruction}]
    messages.append({"role": "user", "content": "my text"})
    # Use assert_called_once() instead of assert_called_once_with() to avoid issues with overloaded functions
    llm.client.chat.completions.create.assert_called_once()  # type: ignore
    # Check call arguments individually
    call_args = llm.client.chat.completions.create.call_args[  # type: ignore
        1
    ]  # Get the keyword arguments
    assert call_args["messages"] == messages
    assert call_args["model"] == "gpt"
    # Check tools content rather than direct equality
    assert len(call_args["tools"]) == 1
    assert call_args["tools"][0]["type"] == "function"
    assert call_args["tools"][0]["function"]["name"] == "test_tool"
    assert call_args["tools"][0]["function"]["description"] == "A test tool"
    assert call_args["tool_choice"] == "auto"
    assert call_args["temperature"] == 0.0


@patch("builtins.__import__")
def test_openai_llm_invoke_with_tools_error(mock_import: Mock, test_tool: Tool) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai

    # Mock an OpenAI error
    mock_openai.OpenAI.return_value.chat.completions.create.side_effect = (
        openai.OpenAIError("Test error")
    )

    llm = OpenAILLM(api_key="my key", model_name="gpt")
    tools = [test_tool]

    with pytest.raises(LLMGenerationError):
        llm.invoke_with_tools("my text", tools)


@patch("builtins.__import__", side_effect=ImportError)
def test_azure_openai_llm_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        AzureOpenAILLM(model_name="gpt-4o")


@patch("builtins.__import__")
def test_azure_openai_llm_happy_path(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.AzureOpenAI.return_value.chat.completions.create.return_value = (
        MagicMock(
            choices=[MagicMock(message=MagicMock(content="openai chat response"))],
        )
    )
    llm = AzureOpenAILLM(
        model_name="gpt",
        azure_endpoint="https://test.openai.azure.com/",
        api_key="my key",
        api_version="version",
    )

    res = llm.invoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "openai chat response"


@patch("builtins.__import__")
def test_azure_openai_llm_with_message_history_happy_path(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.AzureOpenAI.return_value.chat.completions.create.return_value = (
        MagicMock(
            choices=[MagicMock(message=MagicMock(content="openai chat response"))],
        )
    )
    llm = AzureOpenAILLM(
        model_name="gpt",
        azure_endpoint="https://test.openai.azure.com/",
        api_key="my key",
        api_version="version",
    )

    message_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    res = llm.invoke(question, message_history)  # type: ignore
    assert isinstance(res, LLMResponse)
    assert res.content == "openai chat response"
    message_history.append({"role": "user", "content": question})
    # Use assert_called_once() instead of assert_called_once_with() to avoid issues with overloaded functions
    llm.client.chat.completions.create.assert_called_once()  # type: ignore
    # Check call arguments individually
    call_args = llm.client.chat.completions.create.call_args[  # type: ignore
        1
    ]  # Get the keyword arguments
    assert call_args["messages"] == message_history
    assert call_args["model"] == "gpt"


@patch("builtins.__import__")
def test_azure_openai_llm_with_message_history_validation_error(
    mock_import: Mock,
) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.AzureOpenAI.return_value.chat.completions.create.return_value = (
        MagicMock(
            choices=[MagicMock(message=MagicMock(content="openai chat response"))],
        )
    )
    llm = AzureOpenAILLM(
        model_name="gpt",
        azure_endpoint="https://test.openai.azure.com/",
        api_key="my key",
        api_version="version",
    )

    message_history = [
        {"role": "user", "content": 33},
    ]
    question = "What about next season?"

    with pytest.raises(LLMGenerationError) as exc_info:
        llm.invoke(question, message_history)  # type: ignore
    assert "Input should be a valid string" in str(exc_info.value)
