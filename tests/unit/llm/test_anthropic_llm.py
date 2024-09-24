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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j_graphrag.llm.anthropic_llm import AnthropicLLM


@patch("neo4j_graphrag.llm.anthropic_llm.anthropic", None)
def test_vertexai_llm_missing_dependency() -> None:
    with pytest.raises(ImportError):
        AnthropicLLM(model_name="gemini-1.5-flash-001")


@patch("neo4j_graphrag.llm.anthropic_llm.anthropic")
def test_invoke_happy_path(mock_anthropic: MagicMock) -> None:
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content="generated text"
    )
    model_params = {"temperature": 0.3}
    llm = AnthropicLLM("claude", model_params=model_params)
    input_text = "may thy knife chip and shatter"
    response = llm.invoke(input_text)
    assert response.content == "generated text"
    llm.client.messages.create.assert_called_once_with(
        messages=[{"role": "user", "content": input_text}],
        model="claude",
        **model_params,
    )


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.anthropic_llm.anthropic")
async def test_ainvoke_happy_path(mock_anthropic: MagicMock) -> None:
    mock_response = AsyncMock()
    mock_response.content = "Return text"
    mock_model = mock_anthropic.AsyncAnthropic.return_value
    mock_model.messages.create = AsyncMock(return_value=mock_response)
    model_params = {"temperature": 0.3}
    llm = AnthropicLLM("claude", model_params)
    input_text = "may thy knife chip and shatter"
    response = await llm.ainvoke(input_text)
    assert response.content == "Return text"
    llm.async_client.messages.create.assert_called_once_with(
        model="claude",
        messages=[{"role": "user", "content": input_text}],
        **model_params,
    )
