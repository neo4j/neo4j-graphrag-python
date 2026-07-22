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
from neo4j_graphrag.llm import BaseGeminiLLM, GeminiLLM
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


def test_gemini_llm_is_base_gemini_llm_subclass() -> None:
    assert issubclass(GeminiLLM, BaseGeminiLLM)


def test_gemini_llm_init_only_constructs_client(
    mock_genai: Tuple[MagicMock, MagicMock],
) -> None:
    """GeminiLLM's __init__ should only be responsible for client construction."""
    mock_gen, _ = mock_genai
    llm = GeminiLLM("gemini-2.0-flash", api_key="my-key")

    mock_gen.Client.assert_called_once_with(api_key="my-key")
    assert llm.client is mock_gen.Client.return_value


def test_minimal_base_gemini_llm_subclass_exercises_invoke(
    mock_genai: Tuple[MagicMock, MagicMock],
) -> None:
    """BaseGeminiLLM does not construct the SDK client itself: a minimal
    subclass that only assigns ``client`` should exercise the shared
    invoke/message-building logic correctly. This is the extension contract
    exported to custom subclasses."""
    mock_gen, _ = mock_genai
    custom_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "minimal subclass response"
    custom_client.models.generate_content.return_value = mock_response

    class MinimalGeminiLLM(BaseGeminiLLM):
        def __init__(self, model_name: str) -> None:
            super().__init__(model_name=model_name)
            self.client = custom_client

    llm = MinimalGeminiLLM("gemini-2.0-flash")
    response = llm.invoke("hello")

    assert response.content == "minimal subclass response"
    custom_client.models.generate_content.assert_called_once()
    # the default genai.Client was never constructed
    mock_gen.Client.assert_not_called()


def test_gemini_llm_base_url_builds_http_options(
    mock_genai: Tuple[MagicMock, MagicMock],
) -> None:
    """base_url alone must be applied through a new types.HttpOptions."""
    mock_gen, mock_types = mock_genai
    base_url = "https://custom-gemini-endpoint.example.com"

    GeminiLLM(model_name="gemini-2.0-flash", base_url=base_url)

    mock_types.HttpOptions.assert_called_once_with(base_url=base_url)
    _, client_kwargs = mock_gen.Client.call_args
    assert client_kwargs["http_options"] is mock_types.HttpOptions.return_value


def test_gemini_llm_base_url_merges_into_http_options_dict(
    mock_genai: Tuple[MagicMock, MagicMock],
) -> None:
    """base_url must override the base_url field of a dict http_options,
    preserving the other fields."""
    mock_gen, _ = mock_genai
    base_url = "https://custom-gemini-endpoint.example.com"

    GeminiLLM(
        model_name="gemini-2.0-flash",
        base_url=base_url,
        http_options={"api_version": "v1", "base_url": "https://overridden"},
    )

    _, client_kwargs = mock_gen.Client.call_args
    assert client_kwargs["http_options"] == {
        "api_version": "v1",
        "base_url": base_url,
    }


def test_gemini_llm_base_url_updates_http_options_object(
    mock_genai: Tuple[MagicMock, MagicMock],
) -> None:
    """base_url must override the base_url field of a types.HttpOptions object
    via model_copy, preserving the other fields."""
    mock_gen, _ = mock_genai
    base_url = "https://custom-gemini-endpoint.example.com"
    http_options = MagicMock()  # stands in for a types.HttpOptions instance

    GeminiLLM(
        model_name="gemini-2.0-flash", base_url=base_url, http_options=http_options
    )

    http_options.model_copy.assert_called_once_with(update={"base_url": base_url})
    _, client_kwargs = mock_gen.Client.call_args
    assert client_kwargs["http_options"] is http_options.model_copy.return_value


def test_gemini_llm_no_base_url_not_passed_to_client(
    mock_genai: Tuple[MagicMock, MagicMock],
) -> None:
    """Omitting base_url should not pass http_options (or None) to the client."""
    mock_gen, _ = mock_genai

    GeminiLLM(model_name="gemini-2.0-flash")

    _, client_kwargs = mock_gen.Client.call_args
    assert "http_options" not in client_kwargs
    assert "base_url" not in client_kwargs


@pytest.mark.asyncio
async def test_gemini_llm_aclose_closes_sync_and_async_clients(
    mock_genai: Tuple[MagicMock, MagicMock],
) -> None:
    """aclose must release both the sync client and its async (aio) surface."""
    mock_gen, _ = mock_genai
    mock_client = mock_gen.Client.return_value
    mock_client.aio.aclose = AsyncMock()

    llm = GeminiLLM(model_name="gemini-2.0-flash")
    await llm.aclose()

    mock_client.close.assert_called_once()
    mock_client.aio.aclose.assert_awaited_once()
