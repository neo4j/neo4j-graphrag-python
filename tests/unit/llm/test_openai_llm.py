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
from neo4j_graphrag.types import LLMMessage


def get_mock_openai() -> MagicMock:
    mock = MagicMock()
    mock.OpenAIError = openai.OpenAIError
    return mock


@patch("builtins.__import__", side_effect=ImportError)
def test_openai_llm_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        OpenAILLM(model_name="gpt-4o")


@patch("builtins.__import__")
def test_openai_llm_happy_path_e2e(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="openai chat response"))],
    )
    llm = OpenAILLM(api_key="my key", model_name="gpt")

    res = llm.invoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "openai chat response"


def test_openai_llm_get_messages() -> None:
    llm = OpenAILLM(api_key="my key", model_name="gpt")
    message_history = [
        LLMMessage(**{"role": "system", "content": "do something"}),
        LLMMessage(
            **{"role": "user", "content": "When does the sun come up in the summer?"}
        ),
        LLMMessage(**{"role": "assistant", "content": "Usually around 6am."}),
    ]

    messages = llm.get_messages(message_history)
    assert isinstance(messages, list)
    for actual, expected in zip(messages, message_history):
        assert isinstance(actual, dict)
        assert actual["role"] == expected["role"]
        assert actual["content"] == expected["content"]


def test_openai_llm_get_messages_unknown_role() -> None:
    llm = OpenAILLM(api_key="my key", model_name="gpt")
    message_history = [
        LLMMessage(**{"role": "unknown role", "content": "Usually around 6am."}),
    ]
    with pytest.raises(ValueError, match="Unknown role"):
        llm.get_messages(message_history)


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
