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
import json
import warnings
import sys
from typing import Any, Generator, List, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import anthropic
import httpx
import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.llm.anthropic_llm import (
    AnthropicLLM,
    BaseAnthropicLLM,
    _is_open_map,
    _resolve_ref,
    _restore_open_maps,
    _to_anthropic_schema,
)
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.types import LLMMessage
from pydantic import BaseModel, ConfigDict


@pytest.fixture
def mock_anthropic() -> Generator[MagicMock, None, None]:
    mock = MagicMock()
    mock.APIError = anthropic.APIError
    mock.omit = anthropic.omit

    with patch.dict(sys.modules, {"anthropic": mock}):
        yield mock


def _as_mock(value: Any) -> MagicMock:
    return cast(MagicMock, value)


def _as_async_mock(value: Any) -> AsyncMock:
    return cast(AsyncMock, value)


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
    _as_mock(llm.client.messages.create).assert_called_once_with(
        messages=[{"role": "user", "content": input_text}],
        model="claude-3-opus-20240229",
        system=anthropic.omit,
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
    _as_mock(llm.client.messages.create).assert_called_once_with(
        messages=message_history,
        model="claude-3-opus-20240229",
        system=anthropic.omit,
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
    _as_mock(llm.client.messages.create).assert_called_with(
        model="claude-3-opus-20240229",
        system=system_instruction,
        messages=messages,
        **model_params,
    )

    assert _as_mock(llm.client.messages.create).call_count == 1


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
    _as_mock(llm.client.messages.create).assert_called_with(
        model="claude-3-opus-20240229",
        system=system_instruction,
        messages=message_history,
        **model_params,
    )

    assert _as_mock(llm.client.messages.create).call_count == 1


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
    _as_async_mock(llm.async_client.messages.create).assert_awaited_once_with(
        model="claude-3-opus-20240229",
        system=anthropic.omit,
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
    _as_mock(llm.client.messages.create).assert_called_once_with(
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
    _as_mock(llm.client.messages.create).assert_called_once()
    call_args = _as_mock(llm.client.messages.create).call_args[1]
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
    _as_mock(llm.client.messages.create).assert_called_once()
    call_args = _as_mock(llm.client.messages.create).call_args[1]
    assert call_args["system"] == anthropic.omit
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
    _as_async_mock(llm.async_client.messages.create).assert_awaited_once_with(
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

    result_messages = cast(list[MagicMock], list(result_messages))

    # Verify the correct number of non-system messages are returned
    assert len(result_messages) == 3

    # Verify message content is preserved
    assert result_messages[0].content == "Hello"
    assert result_messages[1].content == "Hi there!"
    assert result_messages[2].content == "How are you?"


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


def test_anthropic_invoke_v2_with_pydantic_response_format(
    mock_anthropic: Mock,
) -> None:
    """Test V2 interface passes a Pydantic response_format as output_config."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"value": "structured result"}')]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 20
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response
    mock_anthropic.types.MessageParam = MagicMock(side_effect=lambda **kwargs: kwargs)

    class TestModel(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str

    messages: List[LLMMessage] = [{"role": "user", "content": "Test"}]
    llm = AnthropicLLM(api_key="test", model_name="claude-3-opus")
    response = llm.invoke(messages, response_format=TestModel)

    assert response.content == '{"value": "structured result"}'
    call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args[1]
    output_config = call_kwargs["output_config"]
    assert output_config["format"]["type"] == "json_schema"
    assert output_config["format"]["schema"] == TestModel.model_json_schema()


def test_anthropic_invoke_v2_with_dict_response_format(
    mock_anthropic: Mock,
) -> None:
    """Test V2 interface passes a dict response_format through as output_config."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"value": "dict result"}')]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 20
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response
    mock_anthropic.types.MessageParam = MagicMock(side_effect=lambda **kwargs: kwargs)

    output_config = {
        "format": {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {"value": {"type": "string"}}},
        }
    }
    messages: List[LLMMessage] = [{"role": "user", "content": "Test"}]
    llm = AnthropicLLM(api_key="test", model_name="claude-3-opus")
    response = llm.invoke(messages, response_format=output_config)

    assert response.content == '{"value": "dict result"}'
    call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args[1]
    assert call_kwargs["output_config"] == output_config


def test_anthropic_invoke_v2_without_response_format_omits_output_config(
    mock_anthropic: Mock,
) -> None:
    """Test V2 interface does not pass output_config when no response_format is given."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="plain response")]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 20
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response
    mock_anthropic.types.MessageParam = MagicMock(side_effect=lambda **kwargs: kwargs)

    messages: List[LLMMessage] = [{"role": "user", "content": "Test"}]
    llm = AnthropicLLM(api_key="test", model_name="claude-3-opus")
    response = llm.invoke(messages)

    assert response.content == "plain response"
    call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args[1]
    assert "output_config" not in call_kwargs


@pytest.mark.asyncio
async def test_anthropic_ainvoke_v2_with_pydantic_response_format(
    mock_anthropic: Mock,
) -> None:
    """Test async V2 interface passes a Pydantic response_format as output_config."""
    mock_response = AsyncMock()
    mock_response.content = [MagicMock(text='{"value": "async structured result"}')]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 20
    mock_model = mock_anthropic.AsyncAnthropic.return_value
    mock_model.messages.create = AsyncMock(return_value=mock_response)
    mock_anthropic.types.MessageParam = MagicMock(side_effect=lambda **kwargs: kwargs)

    class TestModel(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str

    messages: List[LLMMessage] = [{"role": "user", "content": "Test"}]
    llm = AnthropicLLM(api_key="test", model_name="claude-3-opus")
    response = await llm.ainvoke(messages, response_format=TestModel)

    assert response.content == '{"value": "async structured result"}'
    call_kwargs = mock_model.messages.create.call_args[1]
    assert call_kwargs["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_llm_supports_structured_output(mock_anthropic: Mock) -> None:  # noqa: ARG001
    """Test that AnthropicLLM advertises structured output support."""
    llm = AnthropicLLM(model_name="claude-3-opus")
    assert llm.supports_structured_output is True


def test_anthropic_llm_close(mock_anthropic: Mock) -> None:
    mock_anthropic.AsyncAnthropic.return_value.close = AsyncMock()

    llm = AnthropicLLM("claude-3-opus-20240229")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        llm.close()

    mock_anthropic.Anthropic.return_value.close.assert_called_once()
    mock_anthropic.AsyncAnthropic.return_value.close.assert_called_once()


# ---------------------------------------------------------------------------
# BaseAnthropicLLM / thin subclass contract tests
# ---------------------------------------------------------------------------


def test_minimal_base_anthropic_llm_subclass_exercises_invoke(
    mock_anthropic: Mock,
) -> None:
    """BaseAnthropicLLM does not construct SDK clients itself: a minimal
    subclass that only assigns client/async_client should exercise the shared
    invoke/schema logic correctly."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content=[MagicMock(text="minimal subclass response")]
    )

    class MinimalAnthropicLLM(BaseAnthropicLLM):
        def __init__(self, model_name: str) -> None:
            super().__init__(model_name=model_name)
            self.client = self.anthropic.Anthropic()
            self.async_client = self.anthropic.AsyncAnthropic()

    llm = MinimalAnthropicLLM(model_name="claude-3-opus-20240229")
    response = llm.invoke("hello")
    assert response.content == "minimal subclass response"


def test_anthropic_llm_is_subclass_of_base_anthropic_llm(mock_anthropic: Mock) -> None:
    """AnthropicLLM is the thin, concrete subclass of BaseAnthropicLLM."""
    llm = AnthropicLLM(model_name="claude-3-opus-20240229")
    assert isinstance(llm, BaseAnthropicLLM)


def test_anthropic_llm_base_url_reaches_both_clients(mock_anthropic: Mock) -> None:
    """base_url must be forwarded to both the sync and async SDK clients."""
    base_url = "https://custom-anthropic-endpoint.example.com"
    AnthropicLLM(model_name="claude-3-opus-20240229", base_url=base_url)

    _, sync_kwargs = mock_anthropic.Anthropic.call_args
    assert sync_kwargs.get("base_url") == base_url
    _, async_kwargs = mock_anthropic.AsyncAnthropic.call_args
    assert async_kwargs.get("base_url") == base_url


def test_anthropic_llm_no_base_url_not_passed_to_clients(mock_anthropic: Mock) -> None:
    """Omitting base_url should not pass it (or None) to either client."""
    AnthropicLLM(model_name="claude-3-opus-20240229")

    _, sync_kwargs = mock_anthropic.Anthropic.call_args
    assert "base_url" not in sync_kwargs
    _, async_kwargs = mock_anthropic.AsyncAnthropic.call_args
    assert "base_url" not in async_kwargs


def test_anthropic_llm_base_url_with_http_client(mock_anthropic: Mock) -> None:
    """base_url and http_client can be combined; both reach the expected client."""
    http_client = httpx.Client()
    base_url = "https://custom-anthropic-endpoint.example.com"
    try:
        AnthropicLLM(
            model_name="claude-3-opus-20240229",
            base_url=base_url,
            http_client=http_client,
        )
        _, sync_kwargs = mock_anthropic.Anthropic.call_args
        assert sync_kwargs.get("base_url") == base_url
        assert sync_kwargs.get("http_client") is http_client
        _, async_kwargs = mock_anthropic.AsyncAnthropic.call_args
        assert async_kwargs.get("base_url") == base_url
        assert "http_client" not in async_kwargs
    finally:
        http_client.close()


# ---------------------------------------------------------------------------
# http_client sync/async routing tests
# ---------------------------------------------------------------------------


def test_anthropic_llm_with_httpx_client(mock_anthropic: Mock) -> None:
    """Test that httpx.Client is forwarded only to the sync Anthropic client."""
    http_client = httpx.Client()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AnthropicLLM(model_name="claude-3-opus-20240229", http_client=http_client)

        assert not any("Invalid http_client" in str(w.message) for w in caught)
        _, sync_kwargs = mock_anthropic.Anthropic.call_args
        assert sync_kwargs.get("http_client") is http_client
        _, async_kwargs = mock_anthropic.AsyncAnthropic.call_args
        assert "http_client" not in async_kwargs
    finally:
        http_client.close()


@pytest.mark.asyncio
async def test_anthropic_llm_with_httpx_async_client(mock_anthropic: Mock) -> None:
    """Test that httpx.AsyncClient is forwarded only to the async Anthropic client."""
    async_http_client = httpx.AsyncClient()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AnthropicLLM(model_name="claude-3-opus-20240229", http_client=async_http_client)

    assert not any("Invalid http_client" in str(w.message) for w in caught)
    _, sync_kwargs = mock_anthropic.Anthropic.call_args
    assert "http_client" not in sync_kwargs
    _, async_kwargs = mock_anthropic.AsyncAnthropic.call_args
    assert async_kwargs.get("http_client") is async_http_client

    await async_http_client.aclose()


def test_anthropic_llm_no_http_client_no_warning(mock_anthropic: Mock) -> None:
    """Test that omitting http_client does not emit a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AnthropicLLM(model_name="claude-3-opus-20240229")

    assert not any("Invalid http_client" in str(w.message) for w in caught)


def test_anthropic_llm_with_invalid_http_client_warns(mock_anthropic: Mock) -> None:
    """Test that an invalid http_client type emits a warning and falls back to
    default construction for both clients."""
    with pytest.warns(UserWarning, match="Invalid http_client type"):
        AnthropicLLM(model_name="claude-3-opus-20240229", http_client="not-a-client")

    _, sync_kwargs = mock_anthropic.Anthropic.call_args
    _, async_kwargs = mock_anthropic.AsyncAnthropic.call_args
    assert "http_client" not in sync_kwargs
    assert "http_client" not in async_kwargs


@pytest.mark.asyncio
async def test_anthropic_llm_aclose(mock_anthropic: Mock) -> None:
    mock_anthropic.AsyncAnthropic.return_value.close = AsyncMock()

    llm = AnthropicLLM("claude-3-opus-20240229")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await llm.aclose()

    mock_anthropic.Anthropic.return_value.close.assert_called_once()
    mock_anthropic.AsyncAnthropic.return_value.close.assert_called_once()


# ---------------------------------------------------------------------------
# INTERMEDIATE FIX: open-map <-> key/value schema transform.
# Remove these tests together with the workaround in anthropic_llm.py once
# cross-provider strict JSON schema handling is added.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema,expected",
    [
        ({"type": "object", "additionalProperties": {"type": "string"}}, True),
        ({"type": "object", "properties": {"a": {"type": "string"}}}, False),
        ({"type": "object", "additionalProperties": False}, False),
        ({"type": "array", "items": {"type": "string"}}, False),
    ],
)
def test_is_open_map(schema: dict[str, Any], expected: bool) -> None:
    assert _is_open_map(schema) is expected


@pytest.mark.parametrize(
    "schema,defs,expected",
    [
        ({"$ref": "#/$defs/Foo"}, {"Foo": {"type": "object"}}, {"type": "object"}),
        ({"$ref": "#/$defs/Missing"}, {}, {}),
        ({"type": "string"}, {}, {"type": "string"}),
    ],
)
def test_resolve_ref(
    schema: dict[str, Any], defs: dict[str, Any], expected: dict[str, Any]
) -> None:
    assert _resolve_ref(schema, defs) == expected


def test_to_anthropic_schema_rewrites_open_map_to_key_value_array() -> None:
    result = _to_anthropic_schema(
        {"type": "object", "additionalProperties": {"type": "integer"}}
    )
    assert result["type"] == "array"
    item = result["items"]
    assert item["type"] == "object"
    assert item["properties"]["key"] == {"type": "string"}
    assert item["properties"]["value"] == {"type": "integer"}
    assert item["required"] == ["key", "value"]
    assert item["additionalProperties"] is False


def test_restore_open_maps_empty_list_becomes_empty_dict_for_map() -> None:
    schema = {"type": "object", "additionalProperties": {"type": "integer"}}
    assert _restore_open_maps([], schema, {}) == {}


def test_restore_open_maps_keeps_genuine_array() -> None:
    schema = {"type": "array", "items": {"type": "number"}}
    assert _restore_open_maps([1.0, 2.0], schema, {}) == [1.0, 2.0]


def test_neo4jgraph_transform_then_restore_round_trip() -> None:
    original = Neo4jGraph.model_json_schema()
    transformed = _to_anthropic_schema(original)
    node = transformed["$defs"]["Neo4jNode"]["properties"]
    assert node["properties"]["type"] == "array"
    assert node["embedding_properties"]["type"] == "array"

    payload = {
        "nodes": [
            {
                "id": "1",
                "label": "Person",
                "properties": [{"key": "name", "value": "Alice"}],
                "embedding_properties": [{"key": "vec", "value": [0.1, 0.2]}],
            }
        ],
        "relationships": [
            {
                "start_node_id": "1",
                "end_node_id": "1",
                "type": "KNOWS",
                "properties": [],
                "embedding_properties": [],
            }
        ],
    }
    restored = _restore_open_maps(payload, original, original.get("$defs", {}))
    graph = Neo4jGraph.model_validate(restored)
    assert graph.nodes[0].properties == {"name": "Alice"}
    assert graph.nodes[0].embedding_properties == {"vec": [0.1, 0.2]}
    assert graph.relationships[0].properties == {}


def test_anthropic_invoke_v2_restores_open_maps_in_response(
    mock_anthropic: Mock,
) -> None:
    raw = json.dumps(
        {
            "nodes": [
                {
                    "id": "1",
                    "label": "Person",
                    "properties": [{"key": "name", "value": "Alice"}],
                    "embedding_properties": [],
                }
            ],
            "relationships": [],
        }
    )
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=raw)]
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response
    mock_anthropic.types.MessageParam = MagicMock(side_effect=lambda **kwargs: kwargs)

    llm = AnthropicLLM(api_key="test", model_name="claude-3-opus")
    messages: List[LLMMessage] = [{"role": "user", "content": "x"}]
    response = llm.invoke(messages, response_format=Neo4jGraph)

    graph = Neo4jGraph.model_validate_json(response.content)
    assert graph.nodes[0].properties == {"name": "Alice"}
