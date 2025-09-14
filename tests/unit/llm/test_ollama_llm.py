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
    mock_ollama.Message.return_value = {"role": "user", "content": "test"}
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
    llm.client.chat.assert_called_once_with(  # type: ignore[attr-defined]
        model=model,
        messages=[
            {"role": "user", "content": "test"}
        ],
        options={"temperature": 0.3}
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
    mock_ollama.Message.return_value = {"role": "user", "content": "test"}
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
        {"role": "user", "content": "test"},
    ]
    llm.client.chat.assert_called_once_with(  # type: ignore[attr-defined]
        model=model,
        messages=messages,
        options=options,
        format="json",
    )


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

    with pytest.raises(LLMGenerationError, match="Input validation failed"):
        llm.invoke(question, message_history)  # type: ignore


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
