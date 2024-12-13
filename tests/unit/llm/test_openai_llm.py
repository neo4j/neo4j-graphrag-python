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
def test_openai_llm_with_chat_history_happy_path(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="openai chat response"))],
    )
    llm = OpenAILLM(api_key="my key", model_name="gpt")
    chat_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    res = llm.invoke(question, chat_history)
    assert isinstance(res, LLMResponse)
    assert res.content == "openai chat response"


@patch("builtins.__import__")
def test_openai_llm_with_chat_history_validation_error(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="openai chat response"))],
    )
    llm = OpenAILLM(api_key="my key", model_name="gpt")
    chat_history = [
        {"role": "human", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    with pytest.raises(LLMGenerationError) as exc_info:
        llm.invoke(question, chat_history)
    assert "Input should be 'user' or 'assistant'" in str(exc_info.value)


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
def test_azure_openai_llm_with_chat_history_happy_path(mock_import: Mock) -> None:
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

    chat_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    res = llm.invoke(question, chat_history)
    assert isinstance(res, LLMResponse)
    assert res.content == "openai chat response"


@patch("builtins.__import__")
def test_azure_openai_llm_with_chat_history_validation_error(mock_import: Mock) -> None:
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

    chat_history = [
        {"content": "When does the sun come up in the summer?"},
    ]
    question = "What about next season?"

    with pytest.raises(LLMGenerationError) as exc_info:
        llm.invoke(question, chat_history)
    assert (
        "{'type': 'missing', 'loc': ('messages', 0, 'role'), 'msg': 'Field required',"
        in str(exc_info.value)
    )
