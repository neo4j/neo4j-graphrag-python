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
import subprocess
import sys
import warnings
from typing import Generator, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import cohere.core
import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.cohere_llm import CohereLLM
from neo4j_graphrag.types import LLMMessage
from pydantic import BaseModel, ConfigDict


@pytest.fixture
def mock_cohere() -> Generator[MagicMock, None, None]:
    mock_cohere = MagicMock()
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
    res = llm.invoke([{"role": "user", "content": "my text"}])
    assert isinstance(res, LLMResponse)
    assert res.content == "cohere response text"


def test_cohere_llm_invoke_with_message_history_happy_path(mock_cohere: Mock) -> None:
    chat_response_mock = MagicMock()
    chat_response_mock.message.content = [MagicMock(text="cohere response text")]
    mock_cohere_client_chat = mock_cohere.ClientV2.return_value.chat
    mock_cohere_client_chat.return_value = chat_response_mock

    system_instruction = "You are a helpful assistant."
    llm = CohereLLM(model_name="something")
    mock_cohere.SystemChatMessageV2 = MagicMock(side_effect=lambda **kw: kw)
    mock_cohere.UserChatMessageV2 = MagicMock(side_effect=lambda **kw: kw)
    mock_cohere.AssistantChatMessageV2 = MagicMock(side_effect=lambda **kw: kw)
    message_history: List[LLMMessage] = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    messages: List[LLMMessage] = [{"role": "system", "content": system_instruction}]
    messages.extend(message_history)
    messages.append({"role": "user", "content": question})
    res = llm.invoke(messages)
    assert isinstance(res, LLMResponse)
    assert res.content == "cohere response text"
    mock_cohere_client_chat.assert_called_once_with(
        messages=[{"content": m["content"]} for m in messages],
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
    mock_cohere.SystemChatMessageV2 = MagicMock(side_effect=lambda **kw: kw)
    mock_cohere.UserChatMessageV2 = MagicMock(side_effect=lambda **kw: kw)
    mock_cohere.AssistantChatMessageV2 = MagicMock(side_effect=lambda **kw: kw)
    message_history: List[LLMMessage] = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    messages: List[LLMMessage] = [{"role": "system", "content": system_instruction}]
    messages.extend(message_history)
    messages.append({"role": "user", "content": question})
    res = llm.invoke(messages)
    assert isinstance(res, LLMResponse)
    assert res.content == "cohere response text"
    mock_cohere_client_chat.assert_called_once_with(
        messages=[{"content": m["content"]} for m in messages],
        model="gpt",
    )


@pytest.mark.asyncio
async def test_cohere_llm_happy_path_async(mock_cohere: Mock) -> None:
    chat_response_mock = MagicMock(
        message=MagicMock(content=[MagicMock(text="cohere response text")])
    )
    mock_cohere.AsyncClientV2.return_value.chat = AsyncMock(
        return_value=chat_response_mock
    )

    llm = CohereLLM(model_name="something")
    res = await llm.ainvoke([{"role": "user", "content": "my text"}])
    assert isinstance(res, LLMResponse)
    assert res.content == "cohere response text"


def test_cohere_llm_failed(mock_cohere: Mock) -> None:
    original_error = cohere.core.ApiError(status_code=500, body="boom")
    mock_cohere.ClientV2.return_value.chat.side_effect = original_error
    llm = CohereLLM(model_name="something")
    with pytest.raises(LLMGenerationError) as excinfo:
        llm.invoke([{"role": "user", "content": "my text"}])
    assert excinfo.value.__cause__ is original_error


@pytest.mark.asyncio
async def test_cohere_llm_failed_async(mock_cohere: Mock) -> None:
    original_error = cohere.core.ApiError(status_code=500, body="boom")
    mock_cohere.AsyncClientV2.return_value.chat.side_effect = original_error
    llm = CohereLLM(model_name="something")

    with pytest.raises(LLMGenerationError) as excinfo:
        await llm.ainvoke([{"role": "user", "content": "my text"}])
    assert excinfo.value.__cause__ is original_error


def test_cohere_llm_parse_error_preserves_cause(mock_cohere: Mock) -> None:
    """_parse_response wraps any parsing failure into LLMGenerationError too,
    preserving the original exception as __cause__ -- not just the sync/async
    transport paths."""
    chat_response_mock = MagicMock()
    chat_response_mock.usage.tokens.input_tokens = "not-an-int"
    mock_cohere.ClientV2.return_value.chat.return_value = chat_response_mock

    llm = CohereLLM(model_name="something")
    with pytest.raises(LLMGenerationError) as excinfo:
        llm.invoke([{"role": "user", "content": "my text"}])
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_cohere_llm_invoke_happy_path(mock_cohere: Mock) -> None:
    """Test invoke method with List[LLMMessage] input."""
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
async def test_cohere_llm_ainvoke_happy_path(mock_cohere: Mock) -> None:
    """Test async invoke method with List[LLMMessage] input."""
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


def test_cohere_llm_invoke_validation_error(mock_cohere: Mock) -> None:
    """Test invoke with invalid message role raises error."""
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


def test_cohere_llm_get_messages_all_roles(mock_cohere: Mock) -> None:
    """Test get_messages method handles all message roles correctly."""
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
    result_messages = llm.get_messages(messages)

    # Verify the correct number of messages are returned
    assert len(result_messages) == 4

    # Verify the correct Cohere message constructors were called
    mock_cohere.SystemChatMessageV2.assert_called_once_with(
        content="You are a helpful assistant."
    )
    assert mock_cohere.UserChatMessageV2.call_count == 2
    mock_cohere.AssistantChatMessageV2.assert_called_once_with(content="Hi there!")


def test_cohere_invoke_with_response_format_raises_error(mock_cohere: Mock) -> None:
    """Test raises NotImplementedError when response_format is used."""

    class TestModel(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str

    messages: List[LLMMessage] = [{"role": "user", "content": "Test"}]
    llm = CohereLLM(api_key="test")

    with pytest.raises(NotImplementedError) as exc_info:
        llm.invoke(messages, response_format=TestModel)

    assert "CohereLLM does not currently support structured output" in str(
        exc_info.value
    )


def test_cohere_llm_close(mock_cohere: Mock) -> None:
    llm = CohereLLM(model_name="something")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        llm.close()


@pytest.mark.asyncio
async def test_cohere_llm_aclose(mock_cohere: Mock) -> None:
    llm = CohereLLM(model_name="something")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await llm.aclose()


def test_cohere_llm_constructs_against_the_real_sdk() -> None:
    """CohereLLM must be constructible with the real cohere package installed.

    Every other test here mocks `cohere`, and this module imports `cohere.core`
    at the top - which populates `core` as an attribute of the `cohere` package
    and hid a real defect: `cohere.core.api_error.ApiError` resolved in the test
    suite while failing for users with `AttributeError: No core found in
    _dynamic_imports`, because the top-level package resolves attributes lazily
    and does not list `core`.

    So this runs in a subprocess, where nothing has pre-imported the submodule.
    """
    source = (
        "from neo4j_graphrag.llm import CohereLLM\n"
        "llm = CohereLLM(model_name='command-a-03-2025', api_key='not-used')\n"
        "assert llm.cohere_api_error.__name__ == 'ApiError'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
