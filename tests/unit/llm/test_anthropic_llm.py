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
from anthropic import NOT_GIVEN, NotGiven

from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.anthropic_llm import AnthropicLLM
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


def test_anthropic_llm_get_messages_with_system_instructions() -> None:
    llm = AnthropicLLM(api_key="my key", model_name="claude")
    message_history = [
        LLMMessage(**{"role": "system", "content": "do something"}),
        LLMMessage(
            **{"role": "user", "content": "When does the sun come up in the summer?"}
        ),
        LLMMessage(**{"role": "assistant", "content": "Usually around 6am."}),
    ]

    system_instruction, messages = llm.get_messages(message_history)
    assert isinstance(system_instruction, str)
    assert system_instruction == "do something"
    assert isinstance(messages, list)
    assert len(messages) == 2  # exclude system instruction
    for actual, expected in zip(messages, message_history[1:]):
        assert isinstance(actual, dict)
        assert actual["role"] == expected["role"]
        assert actual["content"] == expected["content"]


def test_anthropic_llm_get_messages_without_system_instructions() -> None:
    llm = AnthropicLLM(api_key="my key", model_name="claude")
    message_history = [
        LLMMessage(
            **{"role": "user", "content": "When does the sun come up in the summer?"}
        ),
        LLMMessage(**{"role": "assistant", "content": "Usually around 6am."}),
    ]

    system_instruction, messages = llm.get_messages(message_history)
    assert isinstance(system_instruction, NotGiven)
    assert system_instruction == NOT_GIVEN
    assert isinstance(messages, list)
    assert len(messages) == 2
    for actual, expected in zip(messages, message_history):
        assert isinstance(actual, dict)
        assert actual["role"] == expected["role"]
        assert actual["content"] == expected["content"]


def test_anthropic_invoke_happy_path(mock_anthropic: Mock) -> None:
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content=[MagicMock(text="generated text")]
    )
    mock_anthropic.types.MessageParam.return_value = {"role": "user", "content": "hi"}
    model_params = {"temperature": 0.3}
    llm = AnthropicLLM("claude-3-opus-20240229", model_params=model_params)
    input_text = "may thy knife chip and shatter"
    response = llm.invoke(input_text)
    assert isinstance(response, LLMResponse)
    assert response.content == "generated text"
    llm.client.messages.create.assert_called_once_with(  # type: ignore
        messages=[{"role": "user", "content": "hi"}],
        model="claude-3-opus-20240229",
        system=anthropic.NOT_GIVEN,
        **model_params,
    )


@pytest.mark.asyncio
async def test_anthropic_ainvoke_happy_path(mock_anthropic: Mock) -> None:
    mock_response = AsyncMock()
    mock_response.content = [MagicMock(text="Return text")]
    mock_model = mock_anthropic.AsyncAnthropic.return_value
    mock_model.messages.create = AsyncMock(return_value=mock_response)
    mock_anthropic.types.MessageParam.return_value = {"role": "user", "content": "hi"}
    model_params = {"temperature": 0.3}
    llm = AnthropicLLM("claude-3-opus-20240229", model_params)
    input_text = "may thy knife chip and shatter"
    response = await llm.ainvoke(input_text)
    assert isinstance(response, LLMResponse)
    assert response.content == "Return text"
    llm.async_client.messages.create.assert_awaited_once_with(  # type: ignore
        model="claude-3-opus-20240229",
        system=anthropic.NOT_GIVEN,
        messages=[{"role": "user", "content": "hi"}],
        **model_params,
    )
