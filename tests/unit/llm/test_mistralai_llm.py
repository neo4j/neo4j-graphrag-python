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
from typing import List

import httpx
import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMResponse, MistralAILLM
from neo4j_graphrag.types import LLMMessage


# Mock SDKError for testing
class MockSDKError(Exception):
    """Mock SDKError for testing purposes."""

    ...


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral", None)
def test_mistralai_llm_missing_dependency() -> None:
    with pytest.raises(ImportError):
        MistralAILLM(model_name="mistral-model")


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistralai_llm_invoke(mock_mistral: Mock) -> None:
    mock_mistral_instance = mock_mistral.return_value

    chat_response_mock = MagicMock()
    chat_response_mock.choices = [
        MagicMock(message=MagicMock(content="mistral response"))
    ]

    mock_mistral_instance.chat.complete.return_value = chat_response_mock

    llm = MistralAILLM(model_name="mistral-model")

    res = llm.invoke("some input")

    assert isinstance(res, LLMResponse)
    assert res.content == "mistral response"


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistralai_llm_invoke_with_message_history(mock_mistral: Mock) -> None:
    mock_mistral_instance = mock_mistral.return_value
    chat_response_mock = MagicMock()
    chat_response_mock.choices = [
        MagicMock(message=MagicMock(content="mistral response"))
    ]
    mock_mistral_instance.chat.complete.return_value = chat_response_mock
    model = "mistral-model"
    system_instruction = "You are a helpful assistant."

    llm = MistralAILLM(model_name=model)

    message_history: List[LLMMessage] = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"
    res = llm.invoke(question, message_history, system_instruction=system_instruction)

    assert isinstance(res, LLMResponse)
    assert res.content == "mistral response"
    messages: List[LLMMessage] = [{"role": "system", "content": system_instruction}]
    messages.extend(message_history)
    messages.append({"role": "user", "content": question})
    llm.client.chat.complete.assert_called_once_with(  # type: ignore[attr-defined]
        messages=messages,
        model=model,
    )


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistralai_llm_invoke_with_message_history_and_system_instruction(
    mock_mistral: Mock,
) -> None:
    mock_mistral_instance = mock_mistral.return_value
    chat_response_mock = MagicMock()
    chat_response_mock.choices = [
        MagicMock(message=MagicMock(content="mistral response"))
    ]
    mock_mistral_instance.chat.complete.return_value = chat_response_mock
    model = "mistral-model"
    system_instruction = "You are a helpful assistant."
    llm = MistralAILLM(model_name=model)
    message_history: List[LLMMessage] = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    # first invocation - initial instructions
    res = llm.invoke(question, message_history, system_instruction=system_instruction)
    assert isinstance(res, LLMResponse)
    assert res.content == "mistral response"
    messages: List[LLMMessage] = [{"role": "system", "content": system_instruction}]
    messages.extend(message_history)
    messages.append({"role": "user", "content": question})
    llm.client.chat.complete.assert_called_once_with(  # type: ignore[attr-defined]
        messages=messages,
        model=model,
    )

    assert llm.client.chat.complete.call_count == 1  # type: ignore


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistralai_llm_invoke_with_message_history_validation_error(
    mock_mistral: Mock,
) -> None:
    mock_mistral_instance = mock_mistral.return_value
    chat_response_mock = MagicMock()
    chat_response_mock.choices = [
        MagicMock(message=MagicMock(content="mistral response"))
    ]
    mock_mistral_instance.chat.complete.return_value = chat_response_mock
    model = "mistral-model"
    system_instruction = "You are a helpful assistant."

    llm = MistralAILLM(model_name=model, system_instruction=system_instruction)

    message_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "monkey", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    with pytest.raises(LLMGenerationError) as exc_info:
        llm.invoke(question, message_history)  # type: ignore
    assert "Input should be 'user', 'assistant' or 'system" in str(exc_info.value)


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
async def test_mistralai_llm_ainvoke(mock_mistral: Mock) -> None:
    mock_mistral_instance = mock_mistral.return_value

    async def mock_complete_async(*_args: Any, **_kwargs: Any) -> MagicMock:
        chat_response_mock = MagicMock()
        chat_response_mock.choices = [
            MagicMock(message=MagicMock(content="async mistral response"))
        ]
        return chat_response_mock

    mock_mistral_instance.chat.complete_async = mock_complete_async

    llm = MistralAILLM(model_name="mistral-model")

    res = await llm.ainvoke("some input")

    assert isinstance(res, LLMResponse)
    assert res.content == "async mistral response"


@patch("neo4j_graphrag.llm.mistralai_llm.SDKError", MockSDKError)
@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistralai_llm_invoke_sdkerror(mock_mistral: Mock) -> None:
    mock_mistral_instance = mock_mistral.return_value
    raw_response = httpx.Response(status_code=500)
    mock_mistral_instance.chat.complete.side_effect = SDKError(
        "Some error", raw_response=raw_response
    )

    llm = MistralAILLM(model_name="mistral-model")

    with pytest.raises(LLMGenerationError):
        llm.invoke("some input")


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.mistralai_llm.SDKError", MockSDKError)
@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
async def test_mistralai_llm_ainvoke_sdkerror(mock_mistral: Mock) -> None:
    mock_mistral_instance = mock_mistral.return_value

    async def mock_complete_async(*args: Any, **kwargs: Any) -> None:
        raw_response = httpx.Response(status_code=500)
        raise SDKError("Some async error", raw_response=raw_response)

    mock_mistral_instance.chat.complete_async = mock_complete_async

    llm = MistralAILLM(model_name="mistral-model")

    with pytest.raises(LLMGenerationError):
        await llm.ainvoke("some input")


# V2 Interface Tests (List[LLMMessage] input)


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistralai_llm_invoke_v2_happy_path(mock_mistral: Mock) -> None:
    """Test V2 interface invoke method with List[LLMMessage] input."""
    mock_mistral_instance = mock_mistral.return_value
    chat_response_mock = MagicMock()
    chat_response_mock.choices = [
        MagicMock(message=MagicMock(content="mistral v2 response"))
    ]
    mock_mistral_instance.chat.complete.return_value = chat_response_mock

    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
    ]

    llm = MistralAILLM(model_name="mistral-model")
    response = llm.invoke(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "mistral v2 response"

    # Verify the correct method was called
    llm.client.chat.complete.assert_called_once()  # type: ignore[attr-defined]
    call_args = llm.client.chat.complete.call_args[1]  # type: ignore[attr-defined]
    assert call_args["model"] == "mistral-model"
    assert len(call_args["messages"]) == 2


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistralai_llm_invoke_v2_with_conversation_history(mock_mistral: Mock) -> None:
    """Test V2 interface invoke method with complex conversation history."""
    mock_mistral_instance = mock_mistral.return_value
    chat_response_mock = MagicMock()
    chat_response_mock.choices = [
        MagicMock(message=MagicMock(content="mistral conversation response"))
    ]
    mock_mistral_instance.chat.complete.return_value = chat_response_mock

    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python."},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "What about its history?"},
    ]

    llm = MistralAILLM(model_name="mistral-model")
    response = llm.invoke(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "mistral conversation response"

    # Verify the correct number of messages were passed
    llm.client.chat.complete.assert_called_once()  # type: ignore[attr-defined]
    call_args = llm.client.chat.complete.call_args[1]  # type: ignore[attr-defined]
    assert len(call_args["messages"]) == 4


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistralai_llm_invoke_v2_no_system_message(mock_mistral: Mock) -> None:
    """Test V2 interface invoke method without system message."""
    mock_mistral_instance = mock_mistral.return_value
    chat_response_mock = MagicMock()
    chat_response_mock.choices = [
        MagicMock(message=MagicMock(content="mistral no system response"))
    ]
    mock_mistral_instance.chat.complete.return_value = chat_response_mock

    messages: List[LLMMessage] = [
        {"role": "user", "content": "What is the capital of France?"},
    ]

    llm = MistralAILLM(model_name="mistral-model")
    response = llm.invoke(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "mistral no system response"

    # Verify only user message was passed
    llm.client.chat.complete.assert_called_once()  # type: ignore[attr-defined]
    call_args = llm.client.chat.complete.call_args[1]  # type: ignore[attr-defined]
    assert len(call_args["messages"]) == 1


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
async def test_mistralai_llm_ainvoke_v2_happy_path(mock_mistral: Mock) -> None:
    """Test V2 interface async invoke method with List[LLMMessage] input."""
    mock_mistral_instance = mock_mistral.return_value

    async def mock_complete_async(*_args: Any, **_kwargs: Any) -> MagicMock:
        chat_response_mock = MagicMock()
        chat_response_mock.choices = [
            MagicMock(message=MagicMock(content="async mistral v2 response"))
        ]
        return chat_response_mock

    mock_mistral_instance.chat.complete_async = mock_complete_async

    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is async programming?"},
    ]

    llm = MistralAILLM(model_name="mistral-model")
    response = await llm.ainvoke(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "async mistral v2 response"


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.mistralai_llm.SDKError", MockSDKError)
@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
async def test_mistralai_llm_ainvoke_v2_error_handling(mock_mistral: Mock) -> None:
    """Test V2 interface async invoke method error handling."""
    mock_mistral_instance = mock_mistral.return_value

    async def mock_complete_async(*args: Any, **kwargs: Any) -> None:
        raise MockSDKError("V2 async error")

    mock_mistral_instance.chat.complete_async = mock_complete_async

    messages: List[LLMMessage] = [
        {"role": "user", "content": "This should fail"},
    ]

    llm = MistralAILLM(model_name="mistral-model")

    with pytest.raises(LLMGenerationError):
        await llm.ainvoke(messages)


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistralai_llm_invoke_v2_validation_error(mock_mistral: Mock) -> None:
    """Test V2 interface invoke with invalid message role raises error."""
    mock_mistral_instance = mock_mistral.return_value
    chat_response_mock = MagicMock()
    chat_response_mock.choices = [
        MagicMock(message=MagicMock(content="should not reach here"))
    ]
    mock_mistral_instance.chat.complete.return_value = chat_response_mock

    messages: List[LLMMessage] = [
        {"role": "invalid_role", "content": "This should fail."},  # type: ignore[typeddict-item]
    ]

    llm = MistralAILLM(model_name="mistral-model")

    with pytest.raises(ValueError) as exc_info:
        llm.invoke(messages)
    assert "Unknown role: invalid_role" in str(exc_info.value)


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistralai_llm_invoke_invalid_input_type(_mock_mistral: Mock) -> None:
    """Test that invalid input type raises appropriate error."""
    llm = MistralAILLM(model_name="mistral-model")

    with pytest.raises(ValueError) as exc_info:
        llm.invoke(123)  # type: ignore
    assert "Invalid input type for invoke method" in str(exc_info.value)


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
async def test_mistralai_llm_ainvoke_invalid_input_type(_mock_mistral: Mock) -> None:
    """Test that invalid input type raises appropriate error for async invoke."""
    llm = MistralAILLM(model_name="mistral-model")

    with pytest.raises(ValueError) as exc_info:
        await llm.ainvoke(123)  # type: ignore
    assert "Invalid input type for ainvoke method" in str(exc_info.value)


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistralai_llm_get_brand_new_messages_all_roles(_mock_mistral: Mock) -> None:
    """Test get_brand_new_messages method handles all message roles correctly."""
    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    llm = MistralAILLM(model_name="mistral-model")
    result_messages = llm.get_brand_new_messages(messages)

    # Verify the correct number of messages are returned
    assert len(result_messages) == 4

    # Verify each message type is correctly converted
    assert result_messages[0].content == "You are a helpful assistant."
    assert result_messages[1].content == "Hello"
    assert result_messages[2].content == "Hi there!"
    assert result_messages[3].content == "How are you?"


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistralai_llm_get_brand_new_messages_unknown_role(_mock_mistral: Mock) -> None:
    """Test get_brand_new_messages method raises error for unknown role."""
    messages: List[LLMMessage] = [
        {"role": "unknown_role", "content": "This should fail."},  # type: ignore[typeddict-item]
    ]

    llm = MistralAILLM(model_name="mistral-model")

    with pytest.raises(ValueError) as exc_info:
        llm.get_brand_new_messages(messages)
    assert "Unknown role: unknown_role" in str(exc_info.value)
