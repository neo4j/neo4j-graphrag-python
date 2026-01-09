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
from typing import Generator, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import cohere.core
import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.cohere_llm import CohereLLM
from neo4j_graphrag.types import LLMMessage


@pytest.fixture
def mock_cohere() -> Generator[MagicMock, None, None]:
    mock_cohere = MagicMock()
    mock_cohere.core.api_error.ApiError = cohere.core.ApiError
    with patch.dict(sys.modules, {"cohere": mock_cohere}):
        yield mock_cohere


@patch("builtins.__import__", side_effect=ImportError)
def test_cohere_llm_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        CohereLLM(model_name="something")


def test_cohere_llm_happy_path(mock_cohere: Mock) -> None:
    chat_response_mock = MagicMock()
    chat_response_mock.message.content = [MagicMock(text="cohere response text")]
    mock_cohere.ClientV2.return_value.chat.return_value = chat_response_mock
    llm = CohereLLM(model_name="something")
    res = llm.invoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "cohere response text"


def test_cohere_llm_invoke_with_message_history_happy_path(mock_cohere: Mock) -> None:
    chat_response_mock = MagicMock()
    chat_response_mock.message.content = [MagicMock(text="cohere response text")]
    mock_cohere_client_chat = mock_cohere.ClientV2.return_value.chat
    mock_cohere_client_chat.return_value = chat_response_mock

    system_instruction = "You are a helpful assistant."
    llm = CohereLLM(model_name="something")
    message_history: List[LLMMessage] = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    res = llm.invoke(question, message_history, system_instruction=system_instruction)
    assert isinstance(res, LLMResponse)
    assert res.content == "cohere response text"
    messages: List[LLMMessage] = [{"role": "system", "content": system_instruction}]
    messages.extend(message_history)
    messages.append({"role": "user", "content": question})
    mock_cohere_client_chat.assert_called_once_with(
        messages=messages,
        model="something",
    )


def test_cohere_llm_invoke_with_message_history_and_system_instruction(
    mock_cohere: Mock,
) -> None:
    chat_response_mock = MagicMock()
    chat_response_mock.message.content = [MagicMock(text="cohere response text")]
    mock_cohere_client_chat = mock_cohere.ClientV2.return_value.chat
    mock_cohere_client_chat.return_value = chat_response_mock

    system_instruction = "You are a helpful assistant."
    llm = CohereLLM(model_name="gpt")
    message_history: List[LLMMessage] = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    res = llm.invoke(question, message_history, system_instruction=system_instruction)
    assert isinstance(res, LLMResponse)
    assert res.content == "cohere response text"
    messages: List[LLMMessage] = [{"role": "system", "content": system_instruction}]
    messages.extend(message_history)
    messages.append({"role": "user", "content": question})
    mock_cohere_client_chat.assert_called_once_with(
        messages=messages,
        model="gpt",
    )


def test_cohere_llm_invoke_with_message_history_validation_error(
    mock_cohere: Mock,
) -> None:
    chat_response_mock = MagicMock()
    chat_response_mock.message.content = [MagicMock(text="cohere response text")]
    mock_cohere.ClientV2.return_value.chat.return_value = chat_response_mock

    system_instruction = "You are a helpful assistant."
    llm = CohereLLM(model_name="something", system_instruction=system_instruction)
    message_history = [
        {"role": "robot", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    with pytest.raises(LLMGenerationError) as exc_info:
        llm.invoke(question, message_history)  # type: ignore
    assert "Input should be 'user', 'assistant' or 'system" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cohere_llm_happy_path_async(mock_cohere: Mock) -> None:
    chat_response_mock = MagicMock(
        message=MagicMock(content=[MagicMock(text="cohere response text")])
    )
    mock_cohere.AsyncClientV2.return_value.chat = AsyncMock(
        return_value=chat_response_mock
    )

    llm = CohereLLM(model_name="something")
    res = await llm.ainvoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "cohere response text"


def test_cohere_llm_failed(mock_cohere: Mock) -> None:
    mock_cohere.ClientV2.return_value.chat.side_effect = cohere.core.ApiError
    llm = CohereLLM(model_name="something")
    with pytest.raises(LLMGenerationError) as excinfo:
        llm.invoke("my text")
    assert "ApiError" in str(excinfo)


@pytest.mark.asyncio
async def test_cohere_llm_failed_async(mock_cohere: Mock) -> None:
    mock_cohere.AsyncClientV2.return_value.chat.side_effect = cohere.core.ApiError
    llm = CohereLLM(model_name="something")

    with pytest.raises(LLMGenerationError) as excinfo:
        await llm.ainvoke("my text")
    assert "ApiError" in str(excinfo)


# V2 Interface Tests


def test_cohere_llm_invoke_v2_happy_path(mock_cohere: Mock) -> None:
    """Test V2 interface invoke method with List[LLMMessage] input."""
    chat_response_mock = MagicMock()
    chat_response_mock.message.content = [MagicMock(text="cohere v2 response text")]
    mock_cohere.ClientV2.return_value.chat.return_value = chat_response_mock

    # Mock Cohere message types
    mock_cohere.SystemChatMessageV2 = MagicMock()
    mock_cohere.UserChatMessageV2 = MagicMock()
    mock_cohere.AssistantChatMessageV2 = MagicMock()

    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    llm = CohereLLM(model_name="something")
    response = llm.invoke(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "cohere v2 response text"

    # Verify the client was called correctly
    mock_cohere.ClientV2.return_value.chat.assert_called_once()
    call_args = mock_cohere.ClientV2.return_value.chat.call_args[1]
    assert call_args["model"] == "something"


@pytest.mark.asyncio
async def test_cohere_llm_ainvoke_v2_happy_path(mock_cohere: Mock) -> None:
    """Test V2 interface async invoke method with List[LLMMessage] input."""
    chat_response_mock = MagicMock()
    chat_response_mock.message.content = [
        MagicMock(text="cohere v2 async response text")
    ]
    mock_cohere.AsyncClientV2.return_value.chat = AsyncMock(
        return_value=chat_response_mock
    )

    # Mock Cohere message types
    mock_cohere.SystemChatMessageV2 = MagicMock()
    mock_cohere.UserChatMessageV2 = MagicMock()
    mock_cohere.AssistantChatMessageV2 = MagicMock()

    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    llm = CohereLLM(model_name="something")
    response = await llm.ainvoke(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "cohere v2 async response text"

    # Verify the async client was called correctly
    mock_cohere.AsyncClientV2.return_value.chat.assert_awaited_once()


def test_cohere_llm_invoke_v2_validation_error(mock_cohere: Mock) -> None:
    """Test V2 interface invoke with invalid message role raises error."""
    chat_response_mock = MagicMock()
    chat_response_mock.message.content = [MagicMock(text="should not get here")]
    mock_cohere.ClientV2.return_value.chat.return_value = chat_response_mock

    messages: List[LLMMessage] = [
        {"role": "invalid_role", "content": "This should fail."},  # type: ignore[typeddict-item]
    ]

    llm = CohereLLM(model_name="something")

    with pytest.raises(ValueError) as exc_info:
        llm.invoke(messages)
    assert "Unknown role: invalid_role" in str(exc_info.value)


def test_cohere_llm_get_messages_v2_all_roles(mock_cohere: Mock) -> None:
    """Test get_messages_v2 method handles all message roles correctly."""
    # Mock Cohere message types
    mock_system_msg = MagicMock()
    mock_user_msg = MagicMock()
    mock_assistant_msg = MagicMock()

    mock_cohere.SystemChatMessageV2.return_value = mock_system_msg
    mock_cohere.UserChatMessageV2.return_value = mock_user_msg
    mock_cohere.AssistantChatMessageV2.return_value = mock_assistant_msg

    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    llm = CohereLLM(model_name="something")
    result_messages = llm.get_messages_v2(messages)

    # Verify the correct number of messages are returned
    assert len(result_messages) == 4

    # Verify the correct Cohere message constructors were called
    mock_cohere.SystemChatMessageV2.assert_called_once_with(
        content="You are a helpful assistant."
    )
    assert mock_cohere.UserChatMessageV2.call_count == 2
    mock_cohere.AssistantChatMessageV2.assert_called_once_with(content="Hi there!")
