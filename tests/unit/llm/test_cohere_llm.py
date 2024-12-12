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
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import cohere.core
import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.cohere_llm import CohereLLM


@patch("neo4j_graphrag.llm.cohere_llm.cohere", None)
def test_cohere_llm_missing_dependency() -> None:
    with pytest.raises(ImportError):
        CohereLLM(model_name="something")


@patch("neo4j_graphrag.llm.cohere_llm.cohere.AsyncClientV2")
@patch("neo4j_graphrag.llm.cohere_llm.cohere.ClientV2")
def test_cohere_llm_invoke_happy_path(mock_cohere: Mock, mock_async_cohere:Mock) -> None:
    chat_response_mock = MagicMock()
    chat_response_mock.message.content = [
        MagicMock(text="cohere response text")
    ]
    mock_cohere.return_value.chat.return_value = chat_response_mock

    llm = CohereLLM(model_name="something")
    res = llm.invoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "cohere response text"


@patch("neo4j_graphrag.llm.cohere_llm.cohere.AsyncClientV2")
@patch("neo4j_graphrag.llm.cohere_llm.cohere.ClientV2")
def test_cohere_llm_invoke_with_chat_history_happy_path(mock_cohere: Mock, mock_async_cohere:Mock) -> None:
    chat_response_mock = MagicMock()
    chat_response_mock.message.content = [
        MagicMock(text="cohere response text")
    ]
    mock_cohere.return_value.chat.return_value = chat_response_mock

    system_instruction = "You are a helpful assistant."
    llm = CohereLLM(model_name="something", system_instruction=system_instruction)
    chat_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"
    
    res = llm.invoke(question, chat_history)
    assert isinstance(res, LLMResponse)
    assert res.content == "cohere response text"
    messages = [{"role": "system", "content": system_instruction}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": question})
    llm.client.chat.assert_called_once_with(
        messages=messages,
        model="something",
    )


@patch("neo4j_graphrag.llm.cohere_llm.cohere.AsyncClientV2")
@patch("neo4j_graphrag.llm.cohere_llm.cohere.ClientV2")
def test_cohere_llm_invoke_with_chat_history_validation_error(mock_cohere: Mock, mock_async_cohere:Mock) -> None:
    chat_response_mock = MagicMock()
    chat_response_mock.message.content = [
        MagicMock(text="cohere response text")
    ]
    mock_cohere.return_value.chat.return_value = chat_response_mock

    system_instruction = "You are a helpful assistant."
    llm = CohereLLM(model_name="something", system_instruction=system_instruction)
    chat_history = [
        {"role": "robot", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    with pytest.raises(LLMGenerationError) as exc_info:
        llm.invoke(question, chat_history)
    assert "Input should be 'user' or 'assistant'" in str(exc_info.value)


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.cohere_llm.cohere.AsyncClientV2")
@patch("neo4j_graphrag.llm.cohere_llm.cohere.ClientV2")
async def test_cohere_llm_ainvoke_happy_path_async(mock_cohere: Mock, mock_async_cohere:Mock) -> None:
    chat_response_mock = AsyncMock()
    chat_response_mock.message.content = [
        AsyncMock(text="cohere response text")
    ]
    mock_async_cohere.return_value.chat.return_value = chat_response_mock

    llm = CohereLLM(model_name="something")
    res = await llm.ainvoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "cohere response text"


@patch("neo4j_graphrag.llm.cohere_llm.cohere.AsyncClientV2")
@patch("neo4j_graphrag.llm.cohere_llm.cohere.ClientV2")
def test_cohere_llm_failed(mock_cohere: Mock, mock_async_cohere:Mock) -> None:
    mock_cohere.return_value.chat.side_effect = cohere.core.ApiError
    llm = CohereLLM(model_name="something")
    with pytest.raises(LLMGenerationError) as excinfo:
        llm.invoke("my text")
    assert "ApiError" in str(excinfo)


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.cohere_llm.cohere.AsyncClientV2")
@patch("neo4j_graphrag.llm.cohere_llm.cohere.ClientV2")
async def test_cohere_llm_ainvoke_failed_async(mock_cohere: Mock, mock_async_cohere:Mock) -> None:
    mock_async_cohere.return_value.chat.side_effect = cohere.core.ApiError
    llm = CohereLLM(model_name="something")

    with pytest.raises(LLMGenerationError) as excinfo:
        await llm.ainvoke("my text")
    assert "ApiError" in str(excinfo)
