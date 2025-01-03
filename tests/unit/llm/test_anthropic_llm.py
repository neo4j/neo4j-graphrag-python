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

import sys
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import anthropic
import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.anthropic_llm import AnthropicLLM
from neo4j_graphrag.llm.types import LLMResponse


@pytest.fixture
def mock_anthropic() -> Generator[MagicMock, None, None]:
    mock = MagicMock()
    mock.APIError = anthropic.APIError

    with patch.dict(sys.modules, {"anthropic": mock}):
        yield mock


@patch("builtins.__import__", side_effect=ImportError)
def test_anthropic_llm_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        AnthropicLLM(model_name="claude-3-opus-20240229")


def test_anthropic_invoke_happy_path(mock_anthropic: Mock) -> None:
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content="generated text"
    )
    model_params = {"temperature": 0.3}
    llm = AnthropicLLM("claude-3-opus-20240229", model_params=model_params)
    input_text = "may thy knife chip and shatter"
    response = llm.invoke(input_text)
    assert response.content == "generated text"
    llm.client.messages.create.assert_called_once_with(  # type: ignore
        messages=[{"role": "user", "content": input_text}],
        model="claude-3-opus-20240229",
        system=None,
        **model_params,
    )


def test_anthropic_invoke_with_message_history_happy_path(mock_anthropic: Mock) -> None:
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content="generated text"
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
        system=None,
        **model_params,
    )


def test_anthropic_invoke_with_message_history_and_system_instruction(
    mock_anthropic: Mock,
) -> None:
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content="generated text"
    )
    model_params = {"temperature": 0.3}
    initial_instruction = "You are a helpful assistant."
    llm = AnthropicLLM(
        "claude-3-opus-20240229",
        model_params=model_params,
        system_instruction=initial_instruction,
    )
    message_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    # first invokation - initial instructions
    response = llm.invoke(question, message_history)  # type: ignore
    assert response.content == "generated text"
    message_history.append({"role": "user", "content": question})
    llm.client.messages.create.assert_called_once_with(  # type: ignore[attr-defined]
        model="claude-3-opus-20240229",
        system=initial_instruction,
        messages=message_history,
        **model_params,
    )

    # second invokation - override instructions
    override_instruction = "Ignore all previous instructions"
    question = "When does it come up in the winter?"
    response = llm.invoke(question, message_history, override_instruction)  # type: ignore
    assert isinstance(response, LLMResponse)
    assert response.content == "generated text"
    message_history.append({"role": "user", "content": question})
    llm.client.messages.create.assert_called_with(  # type: ignore[attr-defined]
        model="claude-3-opus-20240229",
        system=override_instruction,
        messages=message_history,
        **model_params,
    )

    # third invokation - default instructions
    question = "When does it set?"
    response = llm.invoke(question, message_history)  # type: ignore
    assert isinstance(response, LLMResponse)
    assert response.content == "generated text"
    message_history.append({"role": "user", "content": question})
    llm.client.messages.create.assert_called_with(  # type: ignore[attr-defined]
        model="claude-3-opus-20240229",
        system=initial_instruction,
        messages=message_history,
        **model_params,
    )

    assert llm.client.messages.create.call_count == 3  # type: ignore


def test_anthropic_invoke_with_message_history_validation_error(
    mock_anthropic: Mock,
) -> None:
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content="generated text"
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
    mock_response.content = "Return text"
    mock_model = mock_anthropic.AsyncAnthropic.return_value
    mock_model.messages.create = AsyncMock(return_value=mock_response)
    model_params = {"temperature": 0.3}
    llm = AnthropicLLM("claude-3-opus-20240229", model_params)
    input_text = "may thy knife chip and shatter"
    response = await llm.ainvoke(input_text)
    assert response.content == "Return text"
    llm.async_client.messages.create.assert_awaited_once_with(  # type: ignore
        model="claude-3-opus-20240229",
        system=None,
        messages=[{"role": "user", "content": input_text}],
        **model_params,
    )
