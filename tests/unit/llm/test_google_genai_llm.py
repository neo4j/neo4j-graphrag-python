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
from __future__ import annotations

from typing import Generator, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import GeminiLLM
from neo4j_graphrag.types import LLMMessage


@pytest.fixture
def mock_genai() -> Generator[Tuple[MagicMock, MagicMock], None, None]:
    with patch("neo4j_graphrag.llm.google_genai_llm.genai") as mock_genai, patch(
        "neo4j_graphrag.llm.google_genai_llm.types"
    ) as mock_types:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_types.Content = MagicMock(side_effect=lambda **kwargs: kwargs)
        mock_types.Part.from_text = MagicMock(side_effect=lambda text: {"text": text})
        mock_types.GenerateContentConfig = MagicMock(
            side_effect=lambda **kwargs: kwargs
        )
        mock_types.Tool = MagicMock(side_effect=lambda **kwargs: kwargs)
        mock_types.FunctionDeclaration = MagicMock(side_effect=lambda **kwargs: kwargs)
        mock_types.ToolConfig = MagicMock(side_effect=lambda **kwargs: kwargs)
        mock_types.FunctionCallingConfig = MagicMock(
            side_effect=lambda **kwargs: kwargs
        )
        mock_types.FunctionCallingConfigMode.ANY = "ANY"

        yield mock_genai, mock_types


def test_gemini_llm_missing_dependency() -> None:
    with patch("neo4j_graphrag.llm.google_genai_llm.genai", None):
        with pytest.raises(ImportError) as exc:
            GeminiLLM(model_name="gemini-2.0-flash")
        assert "Could not import google-genai python client" in str(exc.value)


def test_gemini_invoke_happy_path(mock_genai: Tuple[MagicMock, MagicMock]) -> None:
    mock_gen, _ = mock_genai
    mock_client = mock_gen.Client.return_value
    mock_response = MagicMock()
    mock_response.text = "generated text"
    mock_client.models.generate_content.return_value = mock_response

    llm = GeminiLLM("gemini-2.0-flash")
    input_text = "hello"
    response = llm.invoke(input_text)

    assert response.content == "generated text"
    mock_client.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_gemini_ainvoke_happy_path(
    mock_genai: Tuple[MagicMock, MagicMock],
) -> None:
    mock_gen, _ = mock_genai
    mock_client = mock_gen.Client.return_value
    mock_response = MagicMock()
    mock_response.text = "async generated text"
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    llm = GeminiLLM("gemini-2.0-flash")
    input_text = "hello"
    response = await llm.ainvoke(input_text)

    assert response.content == "async generated text"
    mock_client.aio.models.generate_content.assert_awaited_once()


def test_gemini_invoke_v2_happy_path(mock_genai: Tuple[MagicMock, MagicMock]) -> None:
    mock_gen, _ = mock_genai
    mock_client = mock_gen.Client.return_value
    mock_response = MagicMock()
    mock_response.text = "v2 generated text"
    mock_client.models.generate_content.return_value = mock_response

    messages: list[LLMMessage] = [
        {"role": "system", "content": "You are a bot"},
        {"role": "user", "content": "hello"},
    ]

    llm = GeminiLLM("gemini-2.0-flash")
    response = llm.invoke(messages)

    assert response.content == "v2 generated text"
    mock_client.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_gemini_ainvoke_v2_happy_path(
    mock_genai: Tuple[MagicMock, MagicMock],
) -> None:
    mock_gen, _ = mock_genai
    mock_client = mock_gen.Client.return_value
    mock_response = MagicMock()
    mock_response.text = "async v2 generated text"
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    messages: list[LLMMessage] = [{"role": "user", "content": "hello"}]

    llm = GeminiLLM("gemini-2.0-flash")
    response = await llm.ainvoke(messages)

    assert response.content == "async v2 generated text"
    mock_client.aio.models.generate_content.assert_awaited_once()


def test_gemini_invoke_error(mock_genai: Tuple[MagicMock, MagicMock]) -> None:
    mock_gen, _ = mock_genai
    mock_client = mock_gen.Client.return_value
    mock_client.models.generate_content.side_effect = Exception("API error")

    llm = GeminiLLM("gemini-2.0-flash")
    with pytest.raises(LLMGenerationError):
        llm.invoke("hello")
