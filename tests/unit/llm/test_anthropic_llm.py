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
from __future__ import annotations
from typing import List
import sys
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import anthropic
import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.anthropic_llm import AnthropicLLM
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.types import LLMMessage


@pytest.fixture
def mock_anthropic() -> Generator[MagicMock, None, None]:
    mock = MagicMock()
    mock.APIError = anthropic.APIError
    mock.NOT_GIVEN = anthropic.NOT_GIVEN

    with patch.dict(sys.modules, {"anthropic": mock}):
        yield mock


@patch("builtins.__import__", side_effect=ImportError)
def test_anthropic_llm_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        AnthropicLLM(model_name="claude-3-opus-20240229")


def test_anthropic_invoke_happy_path(mock_anthropic: Mock) -> None:
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content=[MagicMock(text="generated text")]
    )
    model_params = {"temperature": 0.3}
    llm = AnthropicLLM("claude-3-opus-20240229", model_params=model_params)
    input_text = "may thy knife chip and shatter"
    response = llm.invoke(input_text)
    assert response.content == "generated text"
    llm.client.messages.create.assert_called_once_with(  # type: ignore
        messages=[{"role": "user", "content": input_text}],
        model="claude-3-opus-20240229",
        system=anthropic.NOT_GIVEN,
        **model_params,
    )


def test_anthropic_invoke_with_message_history_happy_path(mock_anthropic: Mock) -> None:
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content=[MagicMock(text="generated text")]
    )
    model_params = {"temperature": 0.3}
    llm = AnthropicLLM(
        "claude-3-opus-20240229",
        model_params=model_params,
    )
    message_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    response = llm.invoke(question, message_history)  # type: ignore
    assert response.content == "generated text"
    message_history.append({"role": "user", "content": question})
    llm.client.messages.create.assert_called_once_with(  # type: ignore[attr-defined]
        messages=message_history,
        model="claude-3-opus-20240229",
        system=anthropic.NOT_GIVEN,
        **model_params,
    )


def test_anthropic_invoke_with_system_instruction(
    mock_anthropic: Mock,
) -> None:
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content=[MagicMock(text="generated text")]
    )
    model_params = {"temperature": 0.3}
    system_instruction = "You are a helpful assistant."
    llm = AnthropicLLM(
        "claude-3-opus-20240229",
        model_params=model_params,
    )

    question = "When does it come up in the winter?"
    response = llm.invoke(question, system_instruction=system_instruction)
    assert isinstance(response, LLMResponse)
    assert response.content == "generated text"
    messages: List[LLMMessage] = [{"role": "user", "content": question}]
    llm.client.messages.create.assert_called_with(  # type: ignore[attr-defined]
        model="claude-3-opus-20240229",
        system=system_instruction,
        messages=messages,
        **model_params,
    )

    assert llm.client.messages.create.call_count == 1  # type: ignore


def test_anthropic_invoke_with_message_history_and_system_instruction(
    mock_anthropic: Mock,
) -> None:
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content=[MagicMock(text="generated text")]
    )
    model_params = {"temperature": 0.3}
    system_instruction = "You are a helpful assistant."
    llm = AnthropicLLM(
        "claude-3-opus-20240229",
        model_params=model_params,
    )
    message_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]

    question = "When does it come up in the winter?"
    response = llm.invoke(question, message_history, system_instruction)  # type: ignore
    assert isinstance(response, LLMResponse)
    assert response.content == "generated text"
    message_history.append({"role": "user", "content": question})
    llm.client.messages.create.assert_called_with(  # type: ignore[attr-defined]
        model="claude-3-opus-20240229",
        system=system_instruction,
        messages=message_history,
        **model_params,
    )

    assert llm.client.messages.create.call_count == 1  # type: ignore


def test_anthropic_invoke_with_message_history_validation_error(
    mock_anthropic: Mock,
) -> None:
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content=[MagicMock(text="generated text")]
    )
    model_params = {"temperature": 0.3}
    system_instruction = "You are a helpful assistant."
    llm = AnthropicLLM(
        "claude-3-opus-20240229",
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
    assert "Input should be 'user', 'assistant' or 'system'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_anthropic_ainvoke_happy_path(mock_anthropic: Mock) -> None:
    mock_response = AsyncMock()
    mock_response.content = [MagicMock(text="Return text")]
    mock_model = mock_anthropic.AsyncAnthropic.return_value
    mock_model.messages.create = AsyncMock(return_value=mock_response)
    model_params = {"temperature": 0.3}
    llm = AnthropicLLM("claude-3-opus-20240229", model_params)
    input_text = "may thy knife chip and shatter"
    response = await llm.ainvoke(input_text)
    assert response.content == "Return text"
    llm.async_client.messages.create.assert_awaited_once_with(  # type: ignore
        model="claude-3-opus-20240229",
        system=anthropic.NOT_GIVEN,
        messages=[{"role": "user", "content": input_text}],
        **model_params,
    )


# V2 Interface Tests


def test_anthropic_llm_invoke_v2_happy_path(mock_anthropic: Mock) -> None:
    """Test V2 interface invoke method with List[LLMMessage] input."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content=[MagicMock(text="anthropic v2 response")]
    )
    mock_anthropic.types.MessageParam = MagicMock(side_effect=lambda **kwargs: kwargs)

    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
    ]

    model_params = {"temperature": 0.7}
    llm = AnthropicLLM(model_name="claude-3-opus-20240229", model_params=model_params)
    response = llm.invoke(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "anthropic v2 response"

    # Verify the correct method was called with system instruction and messages
    llm.client.messages.create.assert_called_once_with(  # type: ignore
        model="claude-3-opus-20240229",
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "What is machine learning?"}],
        **model_params,
    )


def test_anthropic_llm_invoke_v2_with_conversation_history(
    mock_anthropic: Mock,
) -> None:
    """Test V2 interface invoke method with complex conversation history."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content=[MagicMock(text="anthropic conversation response")]
    )
    mock_anthropic.types.MessageParam = MagicMock(side_effect=lambda **kwargs: kwargs)

    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python."},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "What about its history?"},
    ]

    llm = AnthropicLLM(model_name="claude-3-opus-20240229")
    response = llm.invoke(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "anthropic conversation response"

    # Verify the correct number of messages were passed (excluding system)
    llm.client.messages.create.assert_called_once()  # type: ignore
    call_args = llm.client.messages.create.call_args[1]  # type: ignore
    assert call_args["system"] == "You are a helpful assistant."
    assert len(call_args["messages"]) == 3


def test_anthropic_llm_invoke_v2_no_system_message(mock_anthropic: Mock) -> None:
    """Test V2 interface invoke method without system message."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content=[MagicMock(text="anthropic no system response")]
    )
    mock_anthropic.types.MessageParam = MagicMock(side_effect=lambda **kwargs: kwargs)

    messages: List[LLMMessage] = [
        {"role": "user", "content": "What is the capital of France?"},
    ]

    llm = AnthropicLLM(model_name="claude-3-opus-20240229")
    response = llm.invoke(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "anthropic no system response"

    # Verify only user message was passed and no system instruction
    llm.client.messages.create.assert_called_once()  # type: ignore
    call_args = llm.client.messages.create.call_args[1]  # type: ignore
    assert call_args["system"] == anthropic.NOT_GIVEN
    assert len(call_args["messages"]) == 1


@pytest.mark.asyncio
async def test_anthropic_llm_ainvoke_v2_happy_path(mock_anthropic: Mock) -> None:
    """Test V2 interface async invoke method with List[LLMMessage] input."""
    mock_response = AsyncMock()
    mock_response.content = [MagicMock(text="async anthropic v2 response")]
    mock_model = mock_anthropic.AsyncAnthropic.return_value
    mock_model.messages.create = AsyncMock(return_value=mock_response)
    mock_anthropic.types.MessageParam = MagicMock(side_effect=lambda **kwargs: kwargs)

    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is async programming?"},
    ]

    model_params = {"max_tokens": 100}
    llm = AnthropicLLM(model_name="claude-3-opus-20240229", model_params=model_params)
    response = await llm.ainvoke(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "async anthropic v2 response"

    # Verify the async client was called correctly
    llm.async_client.messages.create.assert_awaited_once_with(  # type: ignore
        model="claude-3-opus-20240229",
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "What is async programming?"}],
        **model_params,
    )


def test_anthropic_llm_invoke_v2_validation_error(mock_anthropic: Mock) -> None:
    """Test V2 interface invoke method with invalid role."""
    mock_anthropic.types.MessageParam = MagicMock(side_effect=lambda **kwargs: kwargs)

    messages: List[LLMMessage] = [
        {"role": "invalid_role", "content": "This should fail."},  # type: ignore[typeddict-item]
    ]

    llm = AnthropicLLM(model_name="claude-3-opus-20240229")

    with pytest.raises(ValueError) as exc_info:
        llm.invoke(messages)
    assert "Unknown role: invalid_role" in str(exc_info.value)


def test_anthropic_llm_invoke_invalid_input_type(
    mock_anthropic: Mock,
) -> None:  # noqa: ARG001
    """Test that invalid input type raises appropriate error."""
    llm = AnthropicLLM(model_name="claude-3-opus-20240229")

    with pytest.raises(ValueError) as exc_info:
        llm.invoke(123)  # type: ignore
    assert "Invalid input type for invoke method" in str(exc_info.value)


@pytest.mark.asyncio
async def test_anthropic_llm_ainvoke_invalid_input_type(
    mock_anthropic: Mock,
) -> None:  # noqa: ARG001
    """Test that invalid input type raises appropriate error for async invoke."""
    llm = AnthropicLLM(model_name="claude-3-opus-20240229")

    with pytest.raises(ValueError) as exc_info:
        await llm.ainvoke(123)  # type: ignore
    assert "Invalid input type for ainvoke method" in str(exc_info.value)


def test_anthropic_llm_get_brand_new_messages_all_roles(mock_anthropic: Mock) -> None:
    """Test get_brand_new_messages method handles all message roles correctly."""

    def create_message_param(**kwargs: str) -> MagicMock:
        return MagicMock(**kwargs)

    mock_anthropic.types.MessageParam = MagicMock(side_effect=create_message_param)

    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    llm = AnthropicLLM(model_name="claude-3-opus-20240229")
    system_instruction, result_messages = llm.get_messages_v2(messages)

    # Verify system instruction is extracted
    assert system_instruction == "You are a helpful assistant."

    result_messages = list(result_messages)

    # Verify the correct number of non-system messages are returned
    assert len(result_messages) == 3

    # Verify message content is preserved
    assert result_messages[0].content == "Hello"  # type: ignore[attr-defined]
    assert result_messages[1].content == "Hi there!"  # type: ignore[attr-defined]
    assert result_messages[2].content == "How are you?"  # type: ignore[attr-defined]


def test_anthropic_llm_get_brand_new_messages_unknown_role(
    mock_anthropic: Mock,
) -> None:  # noqa: ARG001
    """Test get_brand_new_messages method raises error for unknown role."""
    messages: List[LLMMessage] = [
        {"role": "unknown_role", "content": "This should fail."},  # type: ignore[typeddict-item]
    ]

    llm = AnthropicLLM(model_name="claude-3-opus-20240229")

    with pytest.raises(ValueError) as exc_info:
        llm.get_messages_v2(messages)
    assert "Unknown role: unknown_role" in str(exc_info.value)


def test_anthropic_llm_invoke_v2_empty_response_error(mock_anthropic: Mock) -> None:
    """Test V2 interface invoke method handles empty response."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content=[]  # Empty content should trigger error
    )
    mock_anthropic.types.MessageParam = MagicMock(side_effect=lambda **kwargs: kwargs)

    messages: List[LLMMessage] = [
        {"role": "user", "content": "This should return empty response."},
    ]

    llm = AnthropicLLM(model_name="claude-3-opus-20240229")

    with pytest.raises(LLMGenerationError) as exc_info:
        llm.invoke(messages)
    assert "LLM returned empty response" in str(exc_info.value)
